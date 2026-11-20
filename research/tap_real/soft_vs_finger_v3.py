#!/usr/bin/env python3
"""Specialist weak-label SoftTap-vs-FINGER event-shape ablation.

This is a research diagnostic, NOT a physical tap classifier. Uses the
V3 extracted event-feature CSV; no Oculus, metronome, class_id, participant
name or filename fields enter the model. Holds out participant-string groups.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score

META={"file","category","tap","person","double","peak_ms","shape_pass","session_seconds"}
FUTURE={"duration_ms","fall_time_ms","fall_fraction","compactness",
        "shape_score","fall_slope","episode_area","local_quiet_after"}

def fit(X,y,w):
    m=LGBMClassifier(n_estimators=145,learning_rate=.055,num_leaves=7,
         max_depth=5,min_child_samples=40,reg_lambda=16,verbosity=-1,n_jobs=3)
    m.fit(X,y,sample_weight=w)
    return m

def evaluate(df,columns,mode):
    X=df[columns].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    y=df.tap.to_numpy(np.int8)
    group=df.person.to_numpy()
    pred=np.full(len(y),np.nan)
    for fold,(train,test) in enumerate(GroupKFold(5).split(X,y,group)):
        loc=df.iloc[train].reset_index(drop=True)
        count=loc.groupby("file").file.transform("size").to_numpy(float)
        weights=1./count
        weights*=len(weights)/weights.sum()
        if mode=="two_stage":
            first=fit(X[train],y[train],weights)
            p0=first.predict_proba(X[train])[:,1]
            selected=np.zeros(len(train),bool)
            for name,group_rows in loc.groupby("file"):
                ix=group_rows.index.to_numpy()
                if group_rows.tap.iloc[0]==0:
                    selected[ix]=True
                else:
                    best=ix[np.argsort(p0[ix])[::-1][:max(2,int(.35*len(ix)))]]
                    selected[best]=True
            model=fit(X[train][selected],y[train][selected],weights[selected])
        else:
            model=fit(X[train],y[train],weights)
        pred[test]=model.predict_proba(X[test])[:,1]
    ev=df[["file","category","tap","person","session_seconds","peak_ms"]].copy()
    ev["p_soft"]=pred
    bags=ev.groupby("file").agg(label=("tap","first"),category=("category","first"),
                sec=("session_seconds","first"),mean=("p_soft","mean"),
                top5=("p_soft",lambda z:np.mean(np.sort(z)[-5:])))
    result={"file_auc_mean":round(float(roc_auc_score(bags.label,bags["mean"])),4),
            "file_auc_top5":round(float(roc_auc_score(bags.label,bags.top5)),4),
            "files":len(bags),"events":len(ev)}
    for category,part in ev.groupby("category"):
        seconds=float(bags[bags.category==category].sec.sum())
        result[category+"_high_p70_per_hr"]=round(float((part.p_soft>=.7).sum()*3600/seconds),1)
        result[category+"_high_p80_per_hr"]=round(float((part.p_soft>=.8).sum()*3600/seconds),1)
    return ev,result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",default="research/tap_real/results_v3/v3_features.csv.gz")
    ap.add_argument("--out-dir",default="research/tap_real/results_v3_soft_finger")
    args=ap.parse_args()
    data=pd.read_csv(args.features)
    data=data[(data.category=="FINGER")|((data.category=="TAP")&
         data.file.str.contains("SoftTap",case=False))].reset_index(drop=True)
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    allf=[c for c in data if c not in META]
    variants={
        "all_all":(allf,"all"),
        "all_two_stage":(allf,"two_stage"),
        "no_acc_two_stage":([x for x in allf if x!="acc_reactivity"],"two_stage"),
        "early_two_stage":([x for x in allf if x not in FUTURE and
            not (x.startswith("curve_") and int(x.split("_")[1])>0)],"two_stage")}
    summary={"warning":"Weak folder-category results, NOT physical tap event accuracy",
             "variants":{}}
    for name,(features,mode) in variants.items():
        events,metric=evaluate(data,features,mode)
        summary["variants"][name]=metric
        events.to_csv(out/(name+"_oof.csv.gz"),index=False)
        print("SOFT_FINGER_RESULT",name,json.dumps(metric),flush=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
if __name__=="__main__":
    main()
