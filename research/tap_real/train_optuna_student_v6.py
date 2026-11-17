#!/usr/bin/env python3
"""Optuna 2-objective binary TAP+SWIPE student training from V5 silver labels.

Objectives: MAXIMIZE participant-held-out silver-label F1; MINIMIZE exact
export_plus generated static-model bytes. Exports ten size/quality tradeoffs.

No pointer, metronome, timestamps, category, gesture label, user ID, or
future-context teacher features enter model input. Scores reflect V5 SILVER
pseudo-label agreement and are NOT independently measured physical accuracy.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import export_plus

EMG = tuple(f"p_core->emg_psd[0][CH{ch}][{zone}]" for ch in (1,2,3) for zone in range(5))
WL = tuple(f"p_core->wl[CH{ch}]" for ch in (1,2,3))
ACC = tuple(f"p_core->acc_psd[0][{axis}][{zone}]" for axis in ("X","Y","Z") for zone in range(6))
FULL_FEATURES = EMG + WL + ACC
TARGETS = {"TAP","SWIPE"}
NEGATIVES = {"ARM","BAND","FINGER","WRIST","NOTHING"}
ALL_CLASSES = TARGETS | NEGATIVES
LABEL_COLUMNS = ("Timestamp","gesture_label","gesture_weight","gesture_kind","gesture_event_id")
FLOAT_COLUMNS = ("gesture_weight",)
LABEL_SUFFIX = ".labels.csv.gz"

def digest(path, decompress=False):
    h=hashlib.sha256()
    opener = gzip.open if decompress and path.suffix==".gz" else open
    with opener(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def save_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")

def person_group(user, file_stem):
    s=str(user).strip().lower()
    if not s or s=="nan":
        s=file_stem.split("_")[0].lower()
    s=re.sub(r"[^a-z0-9]","",s)
    s=re.sub(r"(lefthand|righthand|left|right|noise|psd\d+|psd|hand)$","",s)
    for prefix in ("alex","ohad","adi","gilad","benkiviti","elin","eli","michael",
                   "kirill","daniel","neomi","perry","shay","eitan"):
        if s.startswith(prefix):return prefix
    return s if s else "unknown"

def spaced(values, cap):
    """Deterministic spread, avoiding a dense run of neighboring frames."""
    if len(values)<=cap:return values
    indexes=np.linspace(0,len(values)-1,num=cap,dtype=int)
    return values[indexes]

def keep_samples(labels, max_per_class):
    y=labels["gesture_label"].to_numpy(dtype=np.int8)
    weight=labels["gesture_weight"].to_numpy(dtype=np.float32)
    if not np.all(np.isin(y,[-1,0,1])):
        raise ValueError("Unexpected gesture label")
    if (weight[y==-1]!=0).any() or (weight[y!=-1]<0).any():
        raise ValueError("Invalid weights or unknown-row leakage")
    pos=np.flatnonzero((y==1)&(weight>0))
    neg=np.flatnonzero((y==0)&(weight>0))
    if len(pos):
        # Preserve diverse temporal events, not just one long labeled window.
        event=labels["gesture_event_id"].to_numpy(dtype=np.int32)
        anchors=[]
        for event_id in np.unique(event[pos]):
            if event_id<0:continue
            subset=pos[event[pos]==event_id]
            anchors.append(int(subset[len(subset)//2]))
        anchors=spaced(np.asarray(anchors,dtype=np.int64),max_per_class)
        remaining=max_per_class-len(anchors)
        filler=spaced(pos,max(0,remaining)) if remaining>0 else np.zeros(0,np.int64)
        pos=np.unique(np.concatenate([anchors,filler]))
        if len(pos)>max_per_class:pos=spaced(pos,max_per_class)
    neg=spaced(neg,max_per_class)
    return np.sort(np.concatenate([pos,neg]))

def _manifest_checks(label_root):
    manifest=label_root/"manifest.jsonl"
    if not manifest.exists():raise FileNotFoundError(manifest)
    records=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    trusted={}
    for r in records:
        if r.get("status")!="OK":continue
        rel=str(r["source"]).replace("\\","/")
        if r.get("category") not in ALL_CLASSES or rel in trusted:
            raise ValueError("Invalid duplicate/source category in V5 manifest")
        expected=r.get("source_sha256")
        if not expected or not re.fullmatch("[0-9a-f]{64}",expected):
            raise ValueError("V5 export not source-hash-bound")
        trusted[rel]=expected
    if len(trusted)<2:raise ValueError("Empty source-bound V5 export")
    return trusted

def _source_for(root,category,stem):
    """Prefer the canonical compressed recording when a local .csv copy also exists.

    The chosen filename must still appear in the frozen source manifest,
    and load_dataset() validates its decompressed SHA-256 before using rows.
    Never combine, delete, or silently substitute the uncompressed copy.
    """
    compressed = root/category/(stem+".csv.gz")
    if compressed.is_file():
        return compressed
    plain = root/category/(stem+".csv")
    if plain.is_file():
        return plain
    raise FileNotFoundError(f"Missing source for {category}/{stem}; expected {compressed} or {plain}")

def _scan_sources(source_root,label_root,manifest,limit=None):
    label_files=sorted(label_root.glob("*/*.labels.csv.gz"))
    if limit:
        # Proportional category coverage for a small integration smoke run.
        by=defaultdict(list)
        for f in label_files:by[f.parent.name].append(f)
        label_files=sorted(f for pool in by.values() for f in spaced(np.array(pool,dtype=object),limit))
    matched=[];common=set(FULL_FEATURES)
    for f in label_files:
        cat=f.parent.name
        if cat not in ALL_CLASSES:raise ValueError(f"Unknown label category: {cat}")
        stem=f.name[:-len(LABEL_SUFFIX)]
        original=_source_for(source_root,cat,stem)
        rel=original.relative_to(source_root).as_posix()
        if rel not in manifest:raise ValueError(f"Untrusted exported label {rel}")
        opener=gzip.open if original.suffix==".gz" else open
        with opener(original,"rt",newline="") as handle:
            header=next(__import__("csv").reader(handle))
        common.intersection_update(header)
        matched.append((f,original,rel))
    if not matched:raise ValueError("No V5 label files")
    missing=set(EMG)-common
    if missing:raise ValueError(f"Mandatory firmware EMG features missing: {sorted(missing)}")
    available=tuple(x for x in FULL_FEATURES if x in common)
    return matched,available

def load_dataset(source_root, label_root, output, max_per_class, sample_files=None,verify_hash=True):
    if source_root.name!="DATA_TAP" or not source_root.is_dir():
        raise ValueError("Source must be the actual DATA/DATA_TAP directory")
    if not label_root.is_dir():raise FileNotFoundError(label_root)
    manifest=_manifest_checks(label_root)
    matched,features=_scan_sources(source_root,label_root,manifest,sample_files)
    sets={"emg":list(EMG)}
    if set(WL).issubset(features):sets["emg_wl"]=list(EMG+WL)
    if set(ACC).issubset(features):sets["emg_acc"]=list(EMG+ACC)
    if set(WL+ACC).issubset(features):sets["all"]=list(EMG+WL+ACC)
    # Only physically available, firmware-addressable features may enter.
    export_plus.validate_feature_schema_accessors(features)
    columns=list(features)
    parts=[];labels=[];weights=[];groups=[];cats=[];files=[];events=[];provenances=[]
    counts=Counter();start=time.monotonic()
    for i,(sidecar,source,rel) in enumerate(matched):
        if verify_hash and digest(source,decompress=True)!=manifest[rel]:
            raise ValueError("Input source content changed: "+rel)
        lab=pd.read_csv(sidecar,compression="gzip",usecols=list(LABEL_COLUMNS))
        chosen=keep_samples(lab,max_per_class)
        if not len(chosen):
            raise ValueError(f"No labeled rows in {rel}")
        raw=pd.read_csv(source,compression=("gzip" if source.suffix==".gz" else None),
                        usecols=lambda name:name in columns or name in {"Timestamp","User"},
                        low_memory=False)
        if len(raw)!=len(lab) or not np.array_equal(
                raw["Timestamp"].to_numpy(dtype=np.int64),
                lab["Timestamp"].to_numpy(dtype=np.int64)):
            raise ValueError("Timestamp/row alignment mismatch: "+rel)
        if not set(columns)<=set(raw.columns):
            raise ValueError("Feature mismatch within source "+rel)
        sub=raw.iloc[chosen]
        numeric=sub[columns].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite raw firmware features in "+rel)
        yy=lab.gesture_label.to_numpy(dtype=np.int8)[chosen]
        ww=lab.gesture_weight.to_numpy(dtype=np.float32)[chosen]
        if source.parent.name in NEGATIVES and np.any(yy==1):
            raise ValueError("Negative category contains positive label: "+rel)
        if source.parent.name in TARGETS and not np.any(yy==1):
            raise ValueError("Positive recording has no positive witness: "+rel)
        user=str(raw.User.iloc[0]) if "User" in raw.columns else ""
        subject=person_group(user,source.stem)
        parts.append(numeric);labels.append(yy);weights.append(ww)
        groups.extend([subject]*len(chosen))
        cats.extend([source.parent.name]*len(chosen))
        files.extend([rel]*len(chosen))
        events.extend(lab.gesture_event_id.to_numpy(dtype=np.int32)[chosen].tolist())
        counts[source.parent.name+"_files"]+=1
        counts["label_1"]+=int((yy==1).sum())
        counts["label_0"]+=int((yy==0).sum())
        if (i+1)%350==0:
            print("EXTRACT",i+1,len(matched),sum(len(p) for p in parts),
                  "seconds",round(time.monotonic()-start),flush=True)
    data={
        "X":np.concatenate(parts).astype(np.float32),
        "y":np.concatenate(labels).astype(np.int8),
        "w":np.concatenate(weights).astype(np.float32),
        "group":np.asarray(groups,dtype=object),
        "category":np.asarray(cats,dtype=object),
        "file":np.asarray(files,dtype=object),
        "event":np.asarray(events,dtype=np.int32),
        "columns":columns, "feature_sets":sets,
    }
    if not np.all((data["y"]==0)|(data["y"]==1)):
        raise ValueError("Unknown label included in training")
    out={"source":str(source_root),"labels":str(label_root),"sampled_rows":int(len(data["y"])),
         "source_files":int(len(matched)),"subject_strings":int(len(set(groups))),
         "classes":dict(sorted(counts.items())),"feature_sets":sets,
         "selected_input_columns":columns,"input_provenance":"V5 SILVER pseudo-labels",
         "label_manifest_sha256":digest(label_root/"manifest.jsonl"),
         "verification":"all timestamp joins and source hashes matched" if verify_hash else "hash verification DISABLED",
         "elapsed_s":round(time.monotonic()-start,2)}
    save_json(output/"dataset_manifest.json",out)
    return data,out

def split_subjects(data, seed, fraction):
    idx=np.arange(len(data["y"]))
    splitter=GroupShuffleSplit(n_splits=40,test_size=fraction,random_state=seed)
    for train,test in splitter.split(idx,data["y"],groups=data["group"]):
        ya=data["y"][train];yb=data["y"][test]
        ta=set(data["category"][test])
        if set(np.unique(ya))=={0,1} and set(np.unique(yb))=={0,1} and {"TAP","SWIPE","FINGER"}<=ta:
            assert not set(data["group"][train]).intersection(data["group"][test])
            return train,test
    raise ValueError("Could not find subject-disjoint outer split covering TAP/SWIPE/FINGER")

def weights_for(data, indices, pos_weight):
    y=data["y"][indices];w=data["w"][indices].copy()
    file_counts=Counter(data["file"][indices])
    w/=np.asarray([math.sqrt(file_counts[f]) for f in data["file"][indices]],dtype=np.float32)
    w[y==1]*=pos_weight
    for c in (0,1):
        s=float(w[y==c].sum())
        if s<=0:raise ValueError(f"Class {c} absent or has zero weight")
        w[y==c]*=len(w)/(2*s)
    return w.astype(np.float32)

def _auc(func,y,p,w=None):
    if len(np.unique(y))<2:return None
    try:return float(func(y,p,sample_weight=w))
    except ValueError:return None

def metrics(y, probability, threshold, weight=None):
    y=np.asarray(y,dtype=np.int8);p=np.asarray(probability,dtype=float)
    w=np.ones(len(y),dtype=float) if weight is None else np.asarray(weight,dtype=float)
    pred=(p>=threshold).astype(np.int8)
    tn,fp,fn,tp=[int(v) for v in confusion_matrix(y,pred,labels=[0,1]).ravel()]
    bins=np.linspace(0,1,11)
    ece=0.
    for lo,hi in zip(bins[:-1],bins[1:]):
        mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if mask.any():ece+=float(w[mask].sum()/max(1e-12,w.sum()))*abs(
            float(np.average(y[mask],weights=w[mask]))-float(np.average(p[mask],weights=w[mask])))
    return {
        "rows":int(len(y)),"positives":int(y.sum()),"negatives":int(len(y)-y.sum()),
        "threshold":float(threshold),
        "precision":float(precision_score(y,pred,sample_weight=w,zero_division=0)),
        "recall":float(recall_score(y,pred,sample_weight=w,zero_division=0)),
        "f1":float(f1_score(y,pred,sample_weight=w,zero_division=0)),
        "accuracy":float(accuracy_score(y,pred,sample_weight=w)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred,sample_weight=w)),
        "specificity":float(tn/(tn+fp)) if tn+fp else None,
        "false_positive_rate":float(fp/(tn+fp)) if tn+fp else None,
        "false_negative_rate":float(fn/(tp+fn)) if tp+fn else None,
        "mcc":float(matthews_corrcoef(y,pred,sample_weight=w)),
        "cohen_kappa":(lambda x: float(x) if np.isfinite(x) else None)(cohen_kappa_score(y,pred,sample_weight=w)),
        "roc_auc":_auc(roc_auc_score,y,p,w),
        "average_precision":_auc(average_precision_score,y,p,w),
        "brier":float(brier_score_loss(y,p,sample_weight=w)),
        "log_loss":float(log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1],sample_weight=w)),
        "ece_10bin":float(ece),
        "confusion_matrix":{"tn":tn,"fp":fp,"fn":fn,"tp":tp},
        "positive_prediction_rate":float(np.mean(pred)),
    }

def subgroup_metrics(data, test_idx, p, threshold):
    result={}
    cats=data["category"][test_idx]
    files=data["file"][test_idx]
    for cat in sorted(ALL_CLASSES):
        mask=cats==cat
        if mask.any():
            sub=metrics(data["y"][test_idx][mask],p[mask],threshold)
            result[cat]=sub
    # Intent-focused breakdowns of original recording instructions.
    for label,pred in (("SOFT_TAP",r"softtap"),("DOUBLE_TAP",r"doubletap"),("MID_TAP",r"midtap"),
                       ("STRONG_TAP",r"strongtap"),("TAP_ON_MOTION",r"taponmotion")):
        match=np.array([bool(re.search(pred,f,re.I)) for f in files],dtype=bool)&(cats=="TAP")
        if match.any():result[label]=metrics(data["y"][test_idx][match],p[match],threshold)
    # Pseudo-event recall: a detected row anywhere within each selected
    # SILVER event. This does not establish physical-onset recall.
    event=data["event"][test_idx];y=data["y"][test_idx]
    keys={(str(f),int(e)) for f,e,yy in zip(files,event,y) if yy==1 and e>=0}
    hits={(str(f),int(e)) for f,e,yy,pp in zip(files,event,y,p) if yy==1 and e>=0 and pp>=threshold}
    result["silver_event_coverage"]={"events_in_sample":len(keys),"events_detected":len(hits),
                                      "recall":len(hits)/len(keys) if keys else None}
    return result

def model_params(trial,seed):
    trees=trial.suggest_int("n_estimators",20,160,step=20)
    depth=trial.suggest_int("max_depth",2,6)
    leaves=trial.suggest_int("num_leaves",4,min(31,2**depth))
    return dict(objective="binary",n_estimators=trees,max_depth=depth,
        num_leaves=leaves,learning_rate=trial.suggest_float("learning_rate",.025,.16,log=True),
        min_child_samples=trial.suggest_int("min_child_samples",35,250,step=5),
        reg_lambda=trial.suggest_float("reg_lambda",.05,40,log=True),
        reg_alpha=trial.suggest_float("reg_alpha",1e-5,2.0,log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree",.70,1.0),
        subsample=trial.suggest_float("subsample",.70,1.0),
        subsample_freq=1,verbosity=-1,n_jobs=2,random_state=seed,
        max_bin=127)

def fit_one(X,y,w,params):
    model=lgb.LGBMClassifier(**params)
    model.fit(X,y,sample_weight=w)
    return model

def evaluate_trial(trial, data, train_indices, folds, seed):
    family=trial.suggest_categorical("feature_set",list(data["feature_sets"]))
    selected=[data["columns"].index(x) for x in data["feature_sets"][family]]
    threshold=trial.suggest_float("decision_threshold",.30,.75,step=.05)
    pos_weight=trial.suggest_float("positive_weight",.75,2.5)
    params=model_params(trial,seed)
    train_groups=data["group"][train_indices]
    cv=GroupKFold(n_splits=folds)
    oof_idx=[];oof_p=[];fold_reports=[];bytes_list=[]
    for fold,(tr,va) in enumerate(cv.split(train_indices, data["y"][train_indices],train_groups)):
        tid=train_indices[tr];vid=train_indices[va]
        if len(np.unique(data["y"][tid]))<2 or len(np.unique(data["y"][vid]))<2:
            raise optuna.TrialPruned("Insufficient labels in participant fold")
        mod=fit_one(data["X"][tid][:,selected],data["y"][tid],
                    weights_for(data,tid,pos_weight),params)
        predictions=mod.predict_proba(data["X"][vid][:,selected])[:,1]
        size=export_plus._estimate_compact_tree_payload_legacy(mod)
        bytes_list.append(int(size["total_static_data_bytes"]))
        fold_reports.append({"fold":fold,"size_bytes":int(size["total_static_data_bytes"]),
          "heldout_subjects":len(set(data["group"][vid])),
          "train_subjects":len(set(data["group"][tid])),
          "metrics":metrics(data["y"][vid],predictions,threshold,weights_for(data,vid,1.0))})
        oof_idx.extend(vid.tolist());oof_p.extend(predictions.tolist())
    vi=np.asarray(oof_idx,dtype=np.int64)
    m=metrics(data["y"][vi],np.asarray(oof_p),threshold,weights_for(data,vi,1.0))
    trial.set_user_attr("cv",m)
    trial.set_user_attr("folds",fold_reports)
    trial.set_user_attr("size_fold_max",max(bytes_list))
    trial.set_user_attr("selected_features",data["feature_sets"][family])
    trial.set_user_attr("training_params",params)
    trial.set_user_attr("evaluation","confidence-weighted V5 pseudo-label F1; user-block OOF")
    return float(m["f1"]), int(max(bytes_list))

def pareto_select(study,k):
    done=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE and t.values is not None]
    if len(done)<k:raise RuntimeError(f"Expected {k} successful trials, got {len(done)}")
    score=np.array([t.values[0] for t in done],dtype=float)
    size=np.array([t.values[1] for t in done],dtype=float)
    zsize=(np.log(size)-np.log(size).min())/(max(1e-9,np.ptp(np.log(size))))
    zscore=(score-score.min())/(max(1e-9,np.ptp(score)))
    frontier=sorted(study.best_trials,key=lambda t:t.values[1])
    selected=[]
    for candidate in (max(done,key=lambda t:t.values[0]),min(done,key=lambda t:t.values[1])):
        if candidate.number not in [v.number for v in selected]:selected.append(candidate)
    # Cover the Pareto frontier evenly before filling with quality-size utility.
    for q in np.linspace(0,len(frontier)-1,num=min(len(frontier),k),dtype=int):
        candidate=frontier[int(q)]
        if candidate.number not in [v.number for v in selected]:selected.append(candidate)
        if len(selected)>=k:break
    ranked=sorted(range(len(done)),key=lambda j:float(zscore[j]-.24*zsize[j]),reverse=True)
    for j in ranked:
        if len(selected)>=k:break
        if done[j].number not in [v.number for v in selected]:selected.append(done[j])
    return selected[:k]

def compile_header(header, name, output):
    cc=shutil.which("cc") or shutil.which("gcc")
    if not cc:return {"status":"not_available","reason":"No C compiler"}
    stub=output/"c_compiler_check"
    stub.mkdir(exist_ok=True)
    (stub/"core_type_def.h").write_text("""#ifndef CORE_TYPE_DEF_H
#define CORE_TYPE_DEF_H
#define CH1 0
#define CH2 1
#define CH3 2
#define X 0
#define Y 1
#define Z 2
typedef struct {
    float emg_psd[1][3][5];
    float wl[3];
    float acc_psd[1][3][6];
} t_core;
#endif
""")
    unit=stub/(name+".c")
    unit.write_text('#include "'+header.name+'"\nint model_compiles(void) { t_core data={0}; return '+name+'_predict(&data); }\n')
    cmd=[cc,"-std=c11","-O2","-c",str(unit),"-I",str(stub),"-I",str(header.parent),
         "-o",str(stub/(name+".o"))]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    if p.returncode:raise RuntimeError("Generated C header failed to compile: "+(p.stderr or p.stdout)[-2600:])
    return {"status":"passed","compiler":cc,"object_bytes":(stub/(name+".o")).stat().st_size,
            "scope":"C syntactic smoke with stub t_core; not actual nRF firmware integration"}

def export_selected(data, train_idx, test_idx, study, selected, output, seed):
    models_dir=output/"models";models_dir.mkdir(exist_ok=True)
    ranked=[]
    y=data["y"]
    for rank,trial in enumerate(selected,1):
        params=trial.user_attrs["training_params"]
        cols=trial.user_attrs["selected_features"]
        indices=[data["columns"].index(x) for x in cols]
        threshold=float(trial.params["decision_threshold"])
        pos_weight=float(trial.params["positive_weight"])
        # First compute a truly subject-held-out score for the candidate;
        # never use test subjects to tune hyperparameters or threshold.
        before=fit_one(data["X"][train_idx][:,indices],y[train_idx],
                       weights_for(data,train_idx,pos_weight),params)
        p=before.predict_proba(data["X"][test_idx][:,indices])[:,1]
        holdout=metrics(y[test_idx],p,threshold,weights_for(data,test_idx,1.0))
        holdout["unweighted"]=metrics(y[test_idx],p,threshold)
        holdout["per_subgroup"]=subgroup_metrics(data,test_idx,p,threshold)
        # Final export refits using all known labels, including the formerly
        # held-out evaluation groups. Holdout metrics above are PRE-refit.
        all_idx=np.arange(len(y))
        final=fit_one(data["X"][:,indices],y,weights_for(data,all_idx,pos_weight),params)
        tag=f"tap_swipe_v6_rank{rank:02d}_trial{trial.number:03d}"
        schema=models_dir/(tag+"_feature_schema.json")
        save_json(schema,{"feature_map":{str(j):col for j,col in enumerate(cols)},
                          "positive_class_id":1,
                          "target_semantics":"TAP and SWIPE combined vs other gestures",
                          "source":"raw causal per-frame p_core features"})
        info={"group_name":tag,"selected_indices":list(range(len(cols))),
              "decision_threshold":threshold,"target_class_id":1,
              "preprocessing":"none","params":params}
        header=models_dir/(tag+".h")
        report=export_plus.export_to_c_header(
            {"model":final,"preprocessing":"none","indices":list(range(len(cols)))},
            info,header,schema,validate=True)
        compile_result=compile_header(header,tag,output)
        estimated=export_plus._estimate_compact_tree_payload_legacy(final)
        assert int(estimated["total_static_data_bytes"])==int(report["total_static_data_bytes"])
        joblib.dump({"model":final,"preprocessing":"none","indices":list(range(len(cols))),
                     "feature_names":cols},models_dir/(tag+".pkl"),compress=3)
        full={
            "rank":rank,"name":tag,"optuna_trial":trial.number,
            "objective":{"f1_maximize":float(trial.values[0]),
                         "firmware_static_bytes_minimize":int(trial.values[1])},
            "cv_silver_metrics":trial.user_attrs["cv"],
            "cv_folds":trial.user_attrs["folds"],
            "heldout_subject_silver_metrics_before_full_refit":holdout,
            "final_model":{"num_trees":int(final.n_estimators_),
                           "num_features":len(cols),"features":cols,
                           "feature_importance_split":dict(zip(cols,[int(x) for x in final.feature_importances_])),
                           "params":params,"decision_threshold":threshold,
                           "trained_on_all_participants":True},
            "export_plus":report,"compiled_c_stub":compile_result,
            "files":{"header":header.name,"header_sha256":digest(header),
                     "header_text_bytes":header.stat().st_size,
                     "feature_schema":schema.name,"pickle":tag+".pkl"},
            "dataset":{"n_sampled":int(len(y)),"positive_sampled":int(y.sum()),
                       "negative_sampled":int((y==0).sum()),
                       "holdout_subjects":sorted(set(data["group"][test_idx])),
                       "training_subjects":len(set(data["group"][train_idx]))},
            "limitations":[
               "All row-level accuracy scores compare to V5 inferred SILVER pseudo-labels, not physical tap annotations.",
               "Held-out score was computed using train-only model before final full-data refit.",
               "Header compilation uses a t_core stub, not actual firmware headers or ARM toolchain.",
               "C parity is export_plus's Python representation validator; compiled numeric parity requires firmware integration.",
               "Binary target combines TAP and SWIPE and does not distinguish their gesture kind."
            ]
        }
        save_json(models_dir/(tag+".json"),full)
        ranked.append({"rank":rank,"trial":trial.number,"header":header.name,
            "json":tag+".json","cv_f1":float(trial.values[0]),
            "cv_static_bytes":int(trial.values[1]),
            "holdout_f1":holdout["f1"],
            "holdout_average_precision":holdout["average_precision"],
            "actual_static_bytes":int(report["total_static_data_bytes"]),
            "header_text_bytes":header.stat().st_size,
            "feature_count":len(cols),"c_compiles":compile_result["status"]})
        print("MODEL_EXPORT",json.dumps(ranked[-1]),flush=True)
    return ranked

def run(args):
    output=args.out_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty for an immutable experiment")
    output.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic()
    data,ds=load_dataset(args.input_dir.resolve(),args.labels_dir.resolve(),output,
                         args.max_per_class,args.sample_files,args.verify_hash)
    if len(set(data["group"]))<8:
        raise ValueError("Need at least eight independent subject groups")
    train,test=split_subjects(data,args.seed,args.test_fraction)
    if len(set(data["group"][train]))<args.cv_folds:
        raise ValueError("Too few independent training groups for CV")
    manifest={"model_target":"gesture_label: 1 TAP or SWIPE; 0 ARM/BAND/FINGER/WRIST/NOTHING",
        "dataset":ds,"train_subjects":sorted(set(data["group"][train])),
        "heldout_subjects":sorted(set(data["group"][test])),
        "train_rows":len(train),"holdout_rows":len(test),
        "cv_folds":args.cv_folds,"seed":args.seed,
        "objectives":[{"name":"confidence_weighted_silver_f1","direction":"maximize"},
                      {"name":"export_plus_static_model_bytes","direction":"minimize"}],
        "note":"No target event timestamps are independently verified physical ground truth."}
    save_json(output/"experiment_manifest.json",manifest)
    print("TRAIN_BEGIN",json.dumps({k:manifest[k] for k in
      ("train_rows","holdout_rows","cv_folds","objectives")}),flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(directions=["maximize","minimize"],
        sampler=optuna.samplers.TPESampler(seed=args.seed,multivariate=False),
        study_name="tap_swipe_student_v6",
        storage="sqlite:///"+str((output/"optuna_study.sqlite3").resolve()),
        load_if_exists=True)
    # Optional, backwards-compatible named baselines for comparative studies.
    # The original V6 CLI defines no initial_feature_sets and is unchanged.
    for feature_set in getattr(args, "initial_feature_sets", ()):
        study.enqueue_trial({"feature_set": feature_set}, skip_if_exists=True)
    def objective(trial):
        value=evaluate_trial(trial,data,train,args.cv_folds,args.seed)
        print("TRIAL",json.dumps({"trial":trial.number,"f1":value[0],
            "bytes":value[1],"features":trial.params["feature_set"]}),flush=True)
        return value
    study.optimize(objective,n_trials=args.trials,n_jobs=1,show_progress_bar=False,
                   gc_after_trial=True,
                   timeout=(args.timeout_minutes*60 if args.timeout_minutes else None))
    selection=pareto_select(study,args.top_k)
    ranks=export_selected(data,train,test,study,selection,output,args.seed)
    all_trials=[]
    for t in study.trials:
        all_trials.append({"id":t.number,"state":t.state.name,
          "objective_values":list(t.values) if t.values is not None else None,
          "params":t.params,"cv":t.user_attrs.get("cv"),
          "size_fold_max":t.user_attrs.get("size_fold_max")})
    save_json(output/"optuna_trials.json",all_trials)
    with (output/"optuna_trials.csv").open("w",encoding="utf-8") as f:
        pd.DataFrame([{"trial":x["id"],"state":x["state"],
            "f1":x["objective_values"][0] if x["objective_values"] else None,
            "static_bytes":x["objective_values"][1] if x["objective_values"] else None,
            **x["params"]} for x in all_trials]).to_csv(f,index=False)
    summary={"status":"success","selected_models":len(ranks),"top_k":args.top_k,
        "complete_trials":len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]),
        "pareto_front_trials":[t.number for t in study.best_trials],
        "ranking":ranks,
        "total_elapsed_s":round(time.monotonic()-t0,1),
        "dataset_manifest":ds,
        "note":"Two objectives are group-CV SILVER weighted F1 and exact export_plus static-array bytes. Holdout reports are subject-disjoint and pre-refit."}
    save_json(output/"summary.json",summary)
    print("TRAIN_END",json.dumps({k:summary[k] for k in
          ("status","selected_models","complete_trials","ranking","total_elapsed_s")}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,default=Path("DATA/DATA_TAP"))
    p.add_argument("--labels-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,default=Path("research/tap_real/results_v6"))
    p.add_argument("--trials",type=int,default=40)
    p.add_argument("--cv-folds",type=int,default=3)
    p.add_argument("--top-k",type=int,default=10)
    p.add_argument("--max-per-class",type=int,default=45)
    p.add_argument("--sample-files",type=int,default=None,
                   help="Research smoke only: max files per category")
    p.add_argument("--seed",type=int,default=20261116)
    p.add_argument("--test-fraction",type=float,default=.22)
    p.add_argument("--timeout-minutes",type=int,default=0)
    p.add_argument("--no-verify-hash",dest="verify_hash",action="store_false",
                   help="Research-only nonstrict mode; CI never uses this")
    args=p.parse_args()
    if args.trials<args.top_k or args.top_k!=10:
        p.error("This pipeline exports exactly 10 distinct Optuna trials; require trials >= 10 and top-k 10.")
    if args.max_per_class<3 or args.cv_folds<2:
        p.error("Too few samples or folds")
    run(args)

if __name__=="__main__":
    main()
