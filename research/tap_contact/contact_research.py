#!/usr/bin/env python3
"""Read-only, paired TAP/SWIPE EMG-connectivity ablation on actual CSV recordings.

Three person-held-out folds. Same 126 p_core inputs, model hyperparameters,
positive/negative evaluation samples, and fixed 0.85 threshold in every arm.
Only training-positive contact policy changes. No original CSV, label or
successful TAP_SWIPE model is modified. Quality is a proxy, NOT contact truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "tap_real"))
import train_claude_v2_optuna as tap
v6 = tap.v6

DEST = ROOT / "research" / "tap_contact" / "results"
CONTACT_ROOT = ROOT / "DATA" / "DATA_LONG_PRESS"
THRESHOLD = 0.85
CPU_THREADS = max(1, tap.CPU_THREADS)
FOLDS = 3
SEED = 204117
BIN_COUNTS = np.array([8., 10., 12., 46., 51.])
W = [f"p_core->wl[CH{i}]" for i in (1,2,3)]
H = [f"p_core->wl_head[CH{i}]" for i in (1,2,3)]
T = [f"p_core->wl_tail[CH{i}]" for i in (1,2,3)]
P = [[f"p_core->emg_psd_mean[CH{i}][{z}]" for z in range(5)] for i in (1,2,3)]
PI = [[f"p_core->emg_psd[0][CH{i}][{z}]" for z in range(5)] for i in (1,2,3)]
COLUMNS = set(W+H+T+sum(P,[])+sum(PI,[])+["psd_mean_applied","Timestamp","User"])

# No tuning on TAP labels or held-out results. Derived from previous LP control
# audit; these heuristics are a controlled hypothesis, not measured impedance.
def contact_fingerprint(path: Path):
    df = pd.read_csv(path, compression="gzip",
                     usecols=lambda c: c in COLUMNS, low_memory=False)
    needed = set(W+H+T+sum(P,[])+sum(PI,[]))
    if len(df) < 50 or not needed.issubset(df.columns):
        raise ValueError(f"Missing required EMG columns in {path}")
    take = np.linspace(0, len(df)-1, min(len(df), 600), dtype=int)
    sub = df.iloc[take]
    provenance = 0
    if "psd_mean_applied" in sub:
        flag = pd.to_numeric(sub["psd_mean_applied"], errors="coerce")
        if flag.eq(1).all(): provenance = 1
        elif not flag.eq(0).all(): raise ValueError(f"Mixed PSD normalization: {path}")
    out=[]
    for ch in range(3):
        wl = pd.to_numeric(sub[W[ch]],errors="coerce").to_numpy(float)
        head = pd.to_numeric(sub[H[ch]],errors="coerce").to_numpy(float)
        tail = pd.to_numeric(sub[T[ch]],errors="coerce").to_numpy(float)
        psd = sub[P[ch]].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        instant = sub[PI[ch]].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        if provenance:
            psd *= BIN_COUNTS[None,:]
            instant *= BIN_COUNTS[None,:]
        finite = (np.isfinite(wl)&np.isfinite(head)&np.isfinite(tail)&
                  np.isfinite(psd).all(axis=1)&np.isfinite(instant).all(axis=1))
        valid_fraction=float(finite.mean())
        wl = np.clip(np.nan_to_num(wl, nan=0., posinf=65535., neginf=0.),0,65535)
        psd = np.clip(np.nan_to_num(psd,nan=0.,posinf=65535.,neginf=0.),0,65535)
        instant=np.clip(np.nan_to_num(instant,nan=0.,posinf=65535.,neginf=0.),0,65535)
        head = np.clip(np.nan_to_num(head,nan=0.,posinf=65535.,neginf=0.),0,65535)
        tail = np.clip(np.nan_to_num(tail,nan=0.,posinf=65535.,neginf=0.),0,65535)
        p = np.median(psd,axis=1)/10000.
        ip = np.median(instant,axis=1)/10000.
        wlq=np.percentile(wl,[10,25,50,75,90])
        pq=np.percentile(p,[10,25,50,75,90])
        baseline_w=float(wlq[0]);baseline_p=float(pq[0])
        saturated=float(np.mean((wl>=65525)|(psd>=65525).any(axis=1)))
        flat=float(np.mean((wl<3)&(p<.0003)))
        difw=float(np.percentile(np.abs(np.diff(np.log1p(wl))),90))
        difp=float(np.percentile(np.abs(np.diff(p)),90))
        shape=np.median(psd,axis=0)/10000.
        shape=shape-float(np.median(shape))
        dyn_w=float(math.log1p(wlq[-1])-math.log1p(wlq[0]))
        dyn_p=float(pq[-1]-pq[0])
        # The exact V4.1 baseline screening conditions, plus clear failure
        # modes. A responsive good channel need NOT be quiet at tap onset.
        plausible=bool(65 <= baseline_w <= 3500 and baseline_p <= .60
                       and saturated < .01 and flat < .80 and valid_fraction > .98)
        f=np.r_[np.log1p(wlq),pq[[0,2,4]],shape,
                [dyn_w,dyn_p,float(np.median(ip-p)),float(np.percentile(np.abs(ip-p),90)),
                 float(np.median(np.log1p(head)-np.log1p(tail))),
                 saturated,flat,valid_fraction,difw,difp]]
        if not np.isfinite(f).all(): raise ValueError(f"Nonfinite contact descriptor {path} CH{ch+1}")
        out.append(dict(vector=f.astype(np.float32),plausible=plausible,
                        wl_p10=baseline_w,psd_p10=baseline_p,
                        saturation_fraction=saturated,flat_fraction=flat,
                        valid_fraction=valid_fraction,dynamic_wl=dyn_w))
    return out

def contact_reference():
    rows=[]; y=[]; people=[]; metadata=[]
    # The physically connected CH3 in nominal DISCONNECTED captures is GOOD.
    # CONNECTED all CH; DISCONNECTED only CH1/2 are nominal bad.
    for kind in ("CONNECTED","DISCONNECTED"):
        paths=sorted((CONTACT_ROOT/kind).glob("*.csv.gz"))
        for i,path in enumerate(paths):
            ch=contact_fingerprint(path)
            person=v6.person_group("",path.name)
            for k in range(3):
                rows.append(ch[k]["vector"])
                y.append(int(kind=="DISCONNECTED" and k<2))
                people.append(person)
                metadata.append(dict(folder=kind,ch=k+1,
                                     plausible=ch[k]["plausible"],
                                     weak_bad=int(kind=="DISCONNECTED" and k<2)))
        print("CONTACT_REFERENCE_LOADED",kind,len(paths),flush=True)
    X=np.stack(rows); yy=np.asarray(y,dtype=int);uid=np.array(people)
    summary={"weak_good":int((yy==0).sum()),"weak_bad":int((yy==1).sum()),
             "people":len(set(uid)),
             "disconnected_ch3_labeled_good":True,
             "disclaimer":"Control labels are weak acquisition-context proxies, not impedance truth"}
    good=np.array([m["plausible"] for m in metadata],dtype=bool)
    for label,mask in [("nominal_good",yy==0),("nominal_bad",yy==1)]:
        summary[label+"_physics_pass_rate"]=float(good[mask].mean())
    if len(set(uid))>=5:
        pred=np.full(len(yy),np.nan)
        splitter=GroupKFold(n_splits=5)
        for tr,te in splitter.split(X,yy,uid):
            model=make_pipeline(StandardScaler(),LogisticRegression(
                C=.1,class_weight="balanced",max_iter=400,random_state=SEED))
            model.fit(X[tr],yy[tr])
            pred[te]=model.predict_proba(X[te])[:,1]
        summary["weak_person_oof_auc"]=float(roc_auc_score(yy,pred))
        summary["weak_person_oof_bal_acc"]=float(balanced_accuracy_score(yy,pred>=.5))
        summary["weak_person_oof_confusion"]=confusion_matrix(yy,pred>=.5,labels=[0,1]).tolist()
    model=make_pipeline(StandardScaler(),LogisticRegression(
        C=.1,class_weight="balanced",max_iter=400,random_state=SEED))
    model.fit(X,yy)
    return model,summary

def fingerprints_for_dataset(data,contact_model):
    root=tap.SOURCE
    paths=sorted(set(data["file"]))
    quality={};bad_files=[]; counters=Counter()
    for i,rel in enumerate(paths,1):
        try:
            ch=contact_fingerprint(root/rel)
            bad=np.asarray(contact_model.predict_proba(np.stack([c["vector"] for c in ch]))[:,1])
            plausible=np.asarray([c["plausible"] for c in ch],bool)
            q=dict(phys_good=int(plausible.sum()),
                   physics_good=plausible.tolist(),
                   p_bad=bad.round(5).tolist(),
                   model_good20=int((bad<.20).sum()),
                   model_good40=int((bad<.40).sum()),
                   model_good50=int((bad<.50).sum()),
                   model_good80=int((bad<.80).sum()),
                   raw={f"CH{j+1}":{k:v for k,v in c.items() if k!="vector"}
                        for j,c in enumerate(ch)})
            quality[rel]=q
        except Exception as ex:
            bad_files.append({"file":rel,"error":str(ex)[:250]})
            # Fail-open for training. Never silently assume bad contact.
            quality[rel]=dict(phys_good=3,physics_good=[True]*3,
                p_bad=[0.,0.,0.],model_good20=3,model_good40=3,
                model_good50=3,model_good80=3,quality_unknown=True)
        counters[(rel.split("/")[0],quality[rel]["phys_good"])]+=1
        if i%350==0:
            print("CONTACT_SCAN",i,len(paths),len(bad_files),flush=True)
    return quality,dict(failed=len(bad_files),failure_sample=bad_files[:20],
                        by_folder_n_phys_good=[{"folder":k[0],"n_good":k[1],"files":n}
                                               for k,n in sorted(counters.items())])

def strategies():
    return ("baseline","physics_any","physics_two","proxy_any20","proxy_weighted")

def train_mask_weight(data,ids,qual,name):
    # Never alter negative/background rows. Only contact-positive supervision.
    y=data["y"];f=data["file"]
    keep=np.ones(len(ids),bool)
    factors=np.ones(len(ids),np.float32)
    for j,ix in enumerate(ids):
        if y[ix]!=1:continue
        q=qual[str(f[ix])]
        if name=="physics_any":keep[j]=q["phys_good"]>=1
        elif name=="physics_two":keep[j]=q["phys_good"]>=2
        elif name=="proxy_any20":keep[j]=q["model_good20"]>=1
        elif name=="proxy_weighted":
            factors[j]=max(.30, 1.-min(q["p_bad"]))
    train=ids[keep]
    weights=v6.weights_for(data,train,1.0)
    if name=="proxy_weighted":
        fkeep=factors[keep]
        positives=y[train]==1
        old=float(weights[positives].sum())
        weights[positives]*=fkeep[positives]
        weights[positives]*=old/max(1e-9,float(weights[positives].sum()))
    if len(np.unique(y[train]))<2:raise ValueError(f"No two classes with {name}")
    return train,weights

def make_model():
    # Predeclared single hyperparameter configuration; NO Optuna, NO threshold
    # change, NO model selection on the held-out labels.
    return lgb.LGBMClassifier(
        objective="binary",n_estimators=100,max_depth=4,num_leaves=13,
        learning_rate=.08,min_child_samples=65,reg_lambda=4.,reg_alpha=.1,
        colsample_bytree=.85,subsample=.90,subsample_freq=1,max_bin=127,
        random_state=SEED,verbosity=-1,n_jobs=CPU_THREADS)

def score_events(data,indices,probs,quality):
    files=data["file"][indices]; ids=data["event"][indices]
    cat=data["category"][indices]; yy=data["y"][indices]
    rows={}
    for f,e,c,y,p in zip(files,ids,cat,yy,probs):
        if y!=1 or e<0:continue
        key=(str(f),int(e))
        if key not in rows:rows[key]=dict(hit=False,category=str(c),
              subtype=("DOUBLE_TAP" if re.search("doubletap",str(f),re.I) else
                       "SOFT_TAP" if re.search("softtap",str(f),re.I) else
                       "SWIPE_SOFT" if re.search("swipesoft",str(f),re.I) else
                       "SWIPE_MID" if re.search("swipemid",str(f),re.I) else
                       "TAP" if c=="TAP" else "SWIPE"),
              person=None, phys_good=quality[str(f)]["phys_good"],
              model_good20=quality[str(f)]["model_good20"])
        if p>=THRESHOLD:rows[key]["hit"]=True
    return rows

def rate_from_rows(data,idx,pred):
    a=pred>=THRESHOLD;y=data["y"][idx]
    return {"positive_row_recall":float(a[y==1].mean()) if (y==1).any() else None,
        "negative_sampled_fp_rate":float(a[y==0].mean()) if (y==0).any() else None,
        "positive_rows":int((y==1).sum()),"negative_sampled_rows":int((y==0).sum())}

def replay_negatives(data,indices,model,columns):
    negatives=sorted(set(str(f) for f,y in zip(data["file"][indices],data["y"][indices]) if y==0))
    total_minutes=0.;events=0;percat=defaultdict(lambda:[0.,0,0])
    for f in negatives:
        path=tap.SOURCE/f
        raw=pd.read_csv(path,compression="gzip",
                        usecols=lambda c:c in set(columns)|{"Timestamp"},
                        low_memory=False)
        ts=raw["Timestamp"].to_numpy(dtype=np.float64)
        if len(ts)<3 or ts[-1]<=ts[0]:continue
        duration=(ts[-1]-ts[0])/60000.
        if not (0<duration<15):continue
        stride=max(1,int(round(12./max(1.,float(np.median(np.diff(ts)[np.diff(ts)>0]))))))
        sub=raw.iloc[::stride]
        xx=sub[columns].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=np.float32)
        xx=np.nan_to_num(xx,nan=0.,posinf=65535.,neginf=0.)
        p=model.predict_proba(xx)[:,1]
        t=sub.Timestamp.to_numpy(dtype=float)
        high=p>=THRESHOLD
        # Count threshold-crossing clusters with a 150ms refractory period.
        firing=np.flatnonzero(high&~np.r_[False,high[:-1]])
        count=0;last=-1e20
        for j in firing:
            if t[j]-last >=150.:
                count+=1;last=t[j]
        total_minutes+=duration;events+=count
        q=percat[f.split("/")[0]];q[0]+=duration;q[1]+=count;q[2]+=1
    return {"recordings":len(negatives),"minutes":total_minutes,
            "trigger_events":events,
            "trigger_events_per_min":events/total_minutes if total_minutes else None,
            "by_negative_folder":{k:dict(minutes=v[0],events=v[1],files=v[2],
                events_per_min=v[1]/v[0] if v[0] else None) for k,v in percat.items()},
            "warning":"Assumes no intended tap/swipe in negative recording folders; not physical FP truth"}

def bootstrap_event_delta(fold_results,baseline,other):
    # Pair by exact held-out event key; cluster-resample participants.
    a=fold_results[baseline];b=fold_results[other]
    keys=sorted(a)
    if set(keys)!=set(b):raise ValueError("Unpaired silver event keys")
    byperson=defaultdict(list)
    for key in keys:
        byperson[a[key]["person"]].append((int(a[key]["hit"]),int(b[key]["hit"])))
    persons=sorted(byperson)
    if len(persons)<3:return None
    rng=np.random.default_rng(SEED)
    changes=[]
    for _ in range(600):
        sampled=rng.choice(persons,size=len(persons),replace=True)
        vals=np.concatenate([np.asarray(byperson[person],dtype=float) for person in sampled],axis=0)
        changes.append(float(vals[:,1].mean()-vals[:,0].mean()))
    return {"delta":float(np.mean(changes)),
            "ci95":[float(x) for x in np.quantile(changes,[.025,.975])],
            "clusters":len(persons)}

def run():
    begin=time.monotonic()
    DEST.mkdir(parents=True,exist_ok=True)
    print("CONTACT_RESEARCH_BEGIN","threads",CPU_THREADS,flush=True)
    tap.prepare()
    tap.enable_full_feature_schema()
    contact_model,contact_audit=contact_reference()
    data,manifest=v6.load_dataset(tap.SOURCE,tap.ADAPTED,DEST,
                                  max_per_class=tap.MAX_ROWS_PER_CLASS_PER_FILE,
                                  verify_hash=True)
    assert tuple(data["columns"])==tap.FULL_126 and len(data["columns"])==126
    quality,quality_audit=fingerprints_for_dataset(data,contact_model)
    # Include all recordings in held-out folds, even low-contact positives.
    groups=data["group"];y=data["y"]
    splits=list(GroupKFold(n_splits=FOLDS).split(data["X"],y,groups))
    for i,(tr,te) in enumerate(splits):
        if len(set(y[tr]))<2 or len(set(y[te]))<2:
            raise ValueError(f"Missing a label class in fold {i}")
        if set(groups[tr])&set(groups[te]):
            raise ValueError("Participant overlap")
    results={x:dict(folds=[],events={},prediction=np.full(len(y),np.nan))
             for x in strategies()}
    for fold,(tr,te) in enumerate(splits):
        print("CONTACT_FOLD_BEGIN",fold,"train",len(tr),"test",len(te),flush=True)
        for variant in strategies():
            tid,w=train_mask_weight(data,tr,quality,variant)
            model=make_model()
            model.fit(data["X"][tid],y[tid],sample_weight=w)
            p=model.predict_proba(data["X"][te])[:,1]
            results[variant]["prediction"][te]=p
            event=score_events(data,te,p,quality)
            for key,d in event.items():
                # Derive participant membership ONLY for grouping, not model input.
                where=np.where((data["file"][te]==key[0])&(data["event"][te]==key[1]))[0]
                d["person"]=str(groups[te[where[0]]])
            results[variant]["events"].update(event)
            score=rate_from_rows(data,te,p)
            score.update(fold=fold,train_rows=len(tid),
                         dropped_positive_rows=int(((y[tr]==1).sum()-(y[tid]==1).sum())),
                         byte_estimate=int(v6.export_plus._estimate_compact_tree_payload_legacy(model)
                                           ["total_static_data_bytes"]))
            # Full-duration inference on held-out NEGATIVE categories only.
            score["negative_replay"]=replay_negatives(data,te,model,data["columns"])
            results[variant]["folds"].append(score)
            print("CONTACT_FOLD_MODEL",json.dumps({
                "fold":fold,"variant":variant,
                "event_recall":sum(d["hit"] for d in event.values())/max(1,len(event)),
                "negative_event_rate":score["negative_replay"]["trigger_events_per_min"],
                "dropped_positive_rows":score["dropped_positive_rows"],
                "byte_estimate":score["byte_estimate"]}),flush=True)
    report={"disclaimer":"All TAP positives are Claude V2 silver onsets, not independently verified physical taps. All contact labels are proxies, not measured impedance.",
            "protocol":{"folds":FOLDS,"grouped_by":"person","threshold_fixed":THRESHOLD,
               "hyperparameters":"identical 100-tree 13-leaf LightGBM for all arms; no Optuna",
               "training_variation":"positive label filtering / weighting ONLY",
               "evaluation":"unfiltered held-out people, identical sampled onset events and full negative recordings",
               "reference":"CONNECTED all good / DISCONNECTED CH1-2 nominal bad; CH3 good",
               "features":126,"source":"DATA/DATA_TAP / Claude V2 frozen labels"},
            "contact_control":contact_audit,"contact_dataset":quality_audit,
            "dataset":{"files":manifest["source_files"],"sampled_rows":len(y),
                       "silver_positives":int((y==1).sum()),"negatives":int((y==0).sum()),
                       "people":len(set(groups))},
            "contact_by_type":{},"models":{},"comparisons":{}}
    perfile=defaultdict(list)
    for f in sorted(quality):
        category=f.split("/")[0]
        group=category
        if category=="TAP":
            if re.search("softtap",f,re.I):group="SOFT_TAP"
            elif re.search("doubletap",f,re.I):group="DOUBLE_TAP"
        elif category=="SWIPE":
            if re.search("swipesoft",f,re.I):group="SOFT_SWIPE"
        perfile[group].append(quality[f])
    for g,q in perfile.items():
        report["contact_by_type"][g]={
            "files":len(q),
            "n_phys_good":dict(Counter(str(r["phys_good"]) for r in q)),
            "n_model_good20":dict(Counter(str(r["model_good20"]) for r in q)),
            "no_physically_plausible_ch_pct":float(np.mean([r["phys_good"]==0 for r in q])),
            "no_model_good20_ch_pct":float(np.mean([r["model_good20"]==0 for r in q]))}
    negidx=np.flatnonzero(y==0)
    for variant,o in results.items():
        pp=o["prediction"]
        assert np.isfinite(pp).all(),(variant,int(np.isnan(pp).sum()))
        events=o["events"]
        recall=float(np.mean([d["hit"] for d in events.values()])) if events else None
        score=rate_from_rows(data,np.arange(len(y)),pp)
        subtype={}
        for name in ("TAP","SWIPE","SOFT_TAP","DOUBLE_TAP","SWIPE_SOFT","SWIPE_MID"):
            ev=[d for d in events.values() if d["category"]==name or d["subtype"]==name]
            if ev:subtype[name]={"events":len(ev),"silver_event_recall":float(np.mean([d["hit"] for d in ev]))}
        contact={}
        for n in range(4):
            ev=[d for d in events.values() if d["phys_good"]==n]
            if ev:contact[str(n)]={"events":len(ev),"silver_event_recall":float(np.mean([d["hit"] for d in ev]))}
        mins=sum(f["negative_replay"]["minutes"] for f in o["folds"])
        firings=sum(f["negative_replay"]["trigger_events"] for f in o["folds"])
        report["models"][variant]={
            "silver_event_recall":recall,"silver_events":len(events),
            "sampled_positive_recall":score["positive_row_recall"],
            "sampled_negative_fp_rate":score["negative_sampled_fp_rate"],
            "negative_recording_minutes":mins,"negative_trigger_events":firings,
            "negative_events_per_min":firings/mins if mins else None,
            "by_subtype":subtype,"by_n_physics_good":contact,
            "folds":o["folds"],
            "model_bytes_range":[min(x["byte_estimate"] for x in o["folds"]),
                                 max(x["byte_estimate"] for x in o["folds"])]}
        if variant!="baseline":
            report["comparisons"][variant]={
                "delta_event_recall":recall-report["models"]["baseline"]["silver_event_recall"],
                "bootstrap_delta_event_recall":bootstrap_event_delta(
                    {k:v["events"] for k,v in results.items()},"baseline",variant),
                "delta_negative_events_per_min":report["models"][variant]["negative_events_per_min"]-
                     report["models"]["baseline"]["negative_events_per_min"]}
    report["elapsed_s"]=round(time.monotonic()-begin,1)
    # Model probabilities are not saved, to avoid large artifacts / accidental
    # use of this research model as a production replacement.
    path=DEST/"report.json"
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("CONTACT_RESEARCH_COMPLETE",json.dumps({
        "files":report["dataset"]["files"],"elapsed_s":report["elapsed_s"],
        "contact_control":contact_audit,
        "contact_by_type":report["contact_by_type"],
        "models":{k:{q:v[q] for q in ("silver_event_recall","silver_events",
                    "negative_events_per_min","sampled_negative_fp_rate")}
                  for k,v in report["models"].items()},
        "comparisons":report["comparisons"]},allow_nan=False),flush=True)
    return report

if __name__=="__main__":
    run()
