#!/usr/bin/env python3
"""V3 research: weakly supervised event-shape learning on real Mudra TAP corpus.

This script scores *recording-category discrimination*, not independently
verified physical tap-event precision or recall. All event-level target labels
are weak and are evaluated with participant-string-block holdout splits.

It deliberately never uses Oculus PointerState, Metronome, TriggerPressure,
filename, BPM, class_id or participant metadata as model features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from tap_event_shape_v3 import ACC, EMG, WL, extract_signals, extract_events

CATEGORIES = {"TAP","SWIPE","FINGER","WRIST","ARM","BAND","NOTHING"}
EXCLUDED = {"file","category","tap","person","peak_ms","double","shape_pass","session_seconds"}
NO_EARLY = {"duration_ms","fall_time_ms","fall_fraction","compactness",
            "shape_score","fall_slope","episode_area","local_quiet_after"}
SENSORS = set([n for group in EMG for n in group] +
              [n for group in ACC for n in group] + WL)
SENSORS.update({"Timestamp","User"})

def canon_person(name):
    v = re.sub("[^a-z]", "", str(name).lower())
    v = re.sub(r"(lefthand|righthand|left|right|noise|psd\d+|psd|new|default|hand)$", "", v)
    for special in ["ohad","alex","neomi","yossef","noam","michael",
                    "perry","nikol","shay","eitan","kirill","sarina",
                    "adi","samuel"]:
        if v.startswith(special):
            return special
    return v or "unknown"

def describe_event(e, S):
    ts = S["ts"]
    q = S["activity"]
    i = e.peak_idx
    lo = e.onset_idx
    z = S["zact"][i]
    ch = S["per_channel"][i]
    bands = np.mean(z,axis=0)

    def at(offset_ms):
        j = np.searchsorted(ts,e.peak_ms+offset_ms,side="left")
        j = int(np.clip(j,0,len(ts)-1))
        return float(q[j])/max(0.01,e.activity)

    keys = ["activity","rise","rise_time_ms","fall_time_ms",
            "duration_ms","fall_fraction","compactness","channel_count",
            "zone_count","max_channel_fraction","wl_reactivity",
            "acc_reactivity","shape_score"]
    row = {k:float(getattr(e,k)) for k in keys}
    for t in [-120,-75,-45,-25,-10,10,25,45,75,120]:
        row[f"curve_{t}"] = at(t)
    for k in range(5):
        row[f"zone_{k}"] = float(bands[k])
        row[f"zoneratio_{k}"] = float(bands[k]/max(.05,np.sum(bands)))
    channel = np.sort(ch)[::-1]
    for k in range(3):
        row[f"channel_{k}"] = float(channel[k]/max(.05,float(np.sum(channel))))
    zflat = np.sort(z.ravel())[::-1]
    row["onezone_diversity"] = float(zflat[0]/max(.05,float(np.sum(z))))
    row["top3_zone_diversity"] = float(zflat[:3].sum()/max(.05,float(np.sum(z))))
    row["initial_vs_peak"] = float(q[lo]/max(.01,e.activity))
    row["fall_slope"] = float(e.activity*e.fall_fraction/max(1.,e.fall_time_ms))
    row["attack_slope"] = float(e.activity/max(1.,e.rise_time_ms))
    row["episode_area"] = float(e.activity*e.duration_ms*e.compactness)
    row["clip_frac"] = float(np.mean(S["emg_raw"][i]>65000))
    row["emg_level_ratio"] = float(np.mean(S["emg_raw"][i])/max(1.,np.mean(S["emg_raw"][lo])))
    row["local_quiet_before"] = at(-150)
    row["local_quiet_after"] = at(150)
    return row

def collect(root, dest, sample_each=0):
    files = sorted(p for p in root.rglob("*.csv.gz") if p.parent.name in CATEGORIES)
    if sample_each:
        from collections import defaultdict
        by=defaultdict(list)
        for f in files:
            name=f.name.lower()
            k = ("DOUBLE" if "doubletap" in name else
                 "SOFT" if "softtap" in name else "OTHER_TAP") if f.parent.name=="TAP" else f.parent.name
            by[k].append(f)
        files=sorted({f for pool in by.values() for f in
                     pool[::max(1,len(pool)//sample_each)][:sample_each]})
    seen = set()
    rows=[]
    errors=[]
    counts={"duplicates":0,"parsed":0,"failed":0}
    start=time.monotonic()
    for idx,f in enumerate(files):
        try:
            digest=hashlib.sha256(f.read_bytes()).hexdigest()
            if digest in seen:
                counts["duplicates"]+=1
                continue
            seen.add(digest)
            df=pd.read_csv(f,compression="gzip",low_memory=False,
                           usecols=lambda n:n in SENSORS)
            if "User" not in df.columns:
                raise ValueError("CSV missing User column")
            S=extract_signals(df)
            es=extract_events(S)
            user=canon_person(df.User.iloc[0])
            path=str(f.relative_to(root))
            duration=(S["ts"][-1]-S["ts"][0])/1000.
            for e in es:
                row=describe_event(e,S)
                row.update(file=path,category=f.parent.name,
                           tap=int(f.parent.name=="TAP"),person=user,
                           double=int("doubletap" in f.name.lower()),
                           peak_ms=e.peak_ms,shape_pass=int(e.provisional=="TAP_SHAPE"),
                           session_seconds=duration)
                rows.append(row)
            counts["parsed"]+=1
        except Exception as err:
            counts["failed"]+=1
            errors.append({"path":str(f),"error":str(err)[:200]})
        if (idx+1)%100==0:
            print("EXTRACT_PROGRESS",idx+1,len(files),len(rows),
                  round(time.monotonic()-start,1),flush=True)
    events=pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError("No events extracted")
    dest.mkdir(parents=True,exist_ok=True)
    events.to_csv(dest/"v3_features.csv.gz",index=False)
    (dest/"extraction.json").write_text(json.dumps(
        {"files":len(files),"events":len(rows),"counts":counts,
         "failures":errors[:40],"elapsed_s":round(time.monotonic()-start,2)},
        indent=2))
    return events,counts

def fit_model(X,y,weights):
    mod=LGBMClassifier(n_estimators=160,learning_rate=.055,
        num_leaves=7,max_depth=5,min_child_samples=40,
        reg_lambda=16.,verbosity=-1,n_jobs=3)
    mod.fit(X,y,sample_weight=weights)
    return mod

def evaluate_variant(data,columns,variant):
    X=data[columns].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy(np.float32)
    y=data["tap"].to_numpy(np.int8)
    groups=data["person"].to_numpy()
    predictions=np.full(len(y),np.nan)
    fold_data=[]
    for fold,(train,test) in enumerate(GroupKFold(5).split(X,y,groups)):
        local=data.iloc[train].reset_index(drop=True)
        sizes=local.groupby("file").file.transform("size").to_numpy(float)
        weights=1./sizes
        weights*=len(weights)/weights.sum()
        if variant.endswith("two_stage"):
            bootstrap=fit_model(X[train],y[train],weights)
            prior=bootstrap.predict_proba(X[train])[:,1]
            select=np.zeros(len(train),bool)
            # select only high-priority events from TAP bags, NOT timestamps
            # indicated by the pointer or metronome. This is weak MIL-like
            # mining and must not be confused with verified physical labels.
            for f,part in local.groupby("file",sort=False):
                ix=part.index.to_numpy()
                if part.tap.iloc[0]==0:
                    select[ix]=True
                else:
                    k=max(2,int(.35*len(ix)))
                    select[ix[np.argsort(prior[ix])[::-1][:k]]]=True
            model=fit_model(X[train][select],y[train][select],weights[select])
        else:
            model=fit_model(X[train],y[train],weights)
        predictions[test]=model.predict_proba(X[test])[:,1]
        fold_data.append({"fold":fold,"train_people":len(set(groups[train])),
                          "held_out_people":len(set(groups[test])),
                          "held_out_events":len(test)})
    r=data[["file","category","tap","person","double","peak_ms","session_seconds"]].copy()
    r["probability"]=predictions
    bags=r.groupby("file",sort=False).agg(
        y=("tap","first"),category=("category","first"),
        person=("person","first"),time_s=("session_seconds","first"),
        mean=("probability","mean"),
        top5=("probability",lambda s:np.mean(np.sort(s)[-5:])),
        over70=("probability",lambda s:np.mean(s>=.7)),
        count=("probability","size"))
    bag_auc={k:float(roc_auc_score(bags.y,bags[k]))
             for k in ("mean","top5","over70")}
    bag_ap={k:float(average_precision_score(bags.y,bags[k]))
            for k in ("mean","top5","over70")}
    cats={}
    for category,tab in r.groupby("category"):
        duration=float(bags[bags.category==category].time_s.sum())
        cats[category]={
            "events":int(len(tab)),
            "high_p70_events":int(np.sum(tab.probability>=.7)),
            "high_p70_events_per_hr":round(float((tab.probability>=.7).sum()*3600/max(1,duration)),2),
            "mean_probability":float(np.mean(tab.probability))}
    return r,{"name":variant,"heldout_bag_auc":bag_auc,
               "heldout_bag_average_precision":bag_ap,"classes":cats,
               "folds":fold_data,
               "warning":"Weak recording-category label metrics; no physical event ground truth"}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir",default="DATA/DATA_TAP")
    ap.add_argument("--out-dir",default="research/tap_real/results_v3")
    ap.add_argument("--sample-each",default=0,type=int)
    ap.add_argument("--reuse-features",action="store_true")
    args=ap.parse_args()
    root=Path(args.input_dir).resolve()
    dest=Path(args.out_dir).resolve()
    if root.name!="DATA_TAP" or not root.is_dir():
        ap.error("input must point to existing DATA_TAP")
    if root in dest.parents or dest==root:
        ap.error("output must not be inside source data")
    if args.reuse_features:
        data=pd.read_csv(dest/"v3_features.csv.gz")
        counts={"reused":True}
    else:
        data,counts=collect(root,dest,args.sample_each)
    f=[c for c in data.columns if c not in EXCLUDED]
    variants={
       "full_all":f,
       "no_acc_two_stage":[c for c in f if c!="acc_reactivity"],
       "early_all":[c for c in f if c not in NO_EARLY and
                    not (c.startswith("curve_") and int(c.split("_")[1])>0)]
    }
    summary={"source":"DATA/DATA_TAP only","files":int(data.file.nunique()),
             "events":int(len(data)),"people_strings":int(data.person.nunique()),
             "dedup":counts,"scores":{}}
    for name,columns in variants.items():
        label=name if name!="no_acc_two_stage" else name
        predictions,result=evaluate_variant(data,columns,label)
        predictions.to_csv(dest/(name+"_oof_predictions.csv.gz"),index=False)
        summary["scores"][name]=result
        (dest/"summary.json").write_text(json.dumps(summary,indent=2))
        print("MODEL_RESULT",name,json.dumps(result["heldout_bag_auc"]),
              flush=True)
    print("V3_RESULT_BEGIN",flush=True)
    print(json.dumps(summary),flush=True)
    print("V3_RESULT_END",flush=True)

if __name__=="__main__":
    main()
