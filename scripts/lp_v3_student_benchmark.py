#!/usr/bin/env python3
"""Held-out-user V3 student imitation benchmark, on ORIGINAL feature CSVs.

The offline V3 teacher may use future samples; student features use
only current/past samples except explicitly labelled non-deployable oracle.
These metrics are NOT real physiological/contact/press accuracy.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,time,json
import numpy as np
import pandas as pd

from lp_v3_teacher import label_recording,VERSION
from lp_v3_labeler import train_contact_quality,normalize_person,EXTRA_READ
from scripts.lp_v3_feature_probe import record
from scripts.lp_classifier_benchmark_v2 import (
  evaluate,get_features,GROUPS,TARGETS,KINDS
)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,default=Path("DATA/DATA_LONG_PRESS"))
    p.add_argument("--per-file",type=int,default=80)
    p.add_argument("--tap-cap",type=int,default=100)
    p.add_argument("--swipe-cap",type=int,default=100)
    p.add_argument("--folds",type=int,default=3)
    p.add_argument("--out",type=Path,default=Path("/tmp/lp_v3_student.json"))
    a=p.parse_args()
    root=a.input_dir
    started=time.monotonic()
    full,models,known,ref,quality=train_contact_quality(root,.7)
    xs=defaultdict(list)
    ys=[];wts=[];targets=[];classes=[];users=[];groups=[]
    count=Counter();fails=[]
    paths=[]
    for g in GROUPS:
        pp=sorted((root/g).glob("*.csv.gz"))
        cap=a.tap_cap if g=="TAP" else a.swipe_cap if g=="SWIPE" else 0
        if cap and len(pp)>cap:
            pp=[pp[i] for i in np.linspace(0,len(pp)-1,cap,dtype=int)]
        paths.extend((g,f) for f in pp)
    print("SELECTED_FILES",len(paths),flush=True)
    for i,(group,path) in enumerate(paths,1):
        try:
            df=pd.read_csv(path,compression="gzip",usecols=lambda c:c in EXTRA_READ)
            if len(df)<80:raise ValueError("too few rows for student sampling")
            if str(path) in known:
                probs=known[str(path)]
            else:
                rec=record(path,group)
                if rec is None:raise ValueError("feature profile unavailable")
                user=normalize_person(df["User"].iloc[0] if "User" in df else None,path.name)
                m=models.get(user,full)
                probs=m.predict_proba(np.array([c["feat"] for c in rec["ch"]]))[:,1]
            label,stats=label_recording(df,contact_prob=probs,
                good_reference_wl=ref,context=group)
            feats=get_features(df)
            idx=np.unique(np.linspace(0,len(df)-1,min(len(df),a.per_file),dtype=int))
            sub=label.iloc[idx]
            for k,x in feats.items():xs[k].append(x[idx])
            ys.append(sub["lp_state"].to_numpy(np.uint8))
            wts.append(sub["lp_state_weight"].to_numpy(np.float32))
            targets.append(sub[TARGETS].to_numpy(np.float32))
            classes.append(np.column_stack([
              np.argmax(sub[[f"lp_{v}_ch{ch}" for v in KINDS]].to_numpy(),axis=1)
              for ch in range(1,4)]).astype(np.uint8))
            user=normalize_person(df["User"].iloc[0] if "User" in df else None,path.name)
            users.extend([user]*len(idx));groups.extend([group]*len(idx));count[group]+=1
        except Exception as e:
            fails.append({"group":group,"file":path.name,"error":str(e)[:150]})
            print("ERROR",group,path.name,str(e)[:140],flush=True)
        if i%100==0 or i==len(paths):
            print("BENCHMARK_EXTRACT",i,len(paths),"ok",sum(count.values()),
                  "errors",len(fails),"elapsed",round(time.monotonic()-started,1),flush=True)
    dataset={"x":{k:np.concatenate(v) for k,v in xs.items()},
      "y":np.concatenate(ys),"weight":np.concatenate(wts),
      "targets":np.concatenate(targets),"classes":np.concatenate(classes),
      "users":np.asarray(users),"groups":np.asarray(groups)}
    meta={"total_files":len(paths),"successful_files":dict(count),
      "failed_files":fails[:60],"rows":len(dataset["y"]),
      "unique_person_ids":len(set(users)),"positive_rest_fraction":float(np.mean(dataset["y"])),
      "positive_rest_by_group":{g:float(np.mean(dataset["y"][dataset["groups"]==g]))
             for g in GROUPS if g in count},
      "mean_weight":float(np.mean(dataset["weight"])),
      "features":{k:v.shape[1] for k,v in dataset["x"].items()},
      "contact_weak_proxy":quality}
    print("DATASET",json.dumps(meta),flush=True)
    results=evaluate(dataset,a.folds)
    report={"warning":"Only V3 pseudo-label imitation, NOT actual muscle rest or contact correctness.",
      "teacher":VERSION,"offline_teacher":True,"split":"GroupKFold normalized User",
      "meta":meta,"results":results,"time_s":round(time.monotonic()-started,1)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2))
    print("V3_STUDENT_DONE",json.dumps({"output":str(a.out),
        "balanced_accuracy":results["models"]["lightgbm_causal"]["balanced_accuracy"],
        "rest_f1":results["models"]["lightgbm_causal"]["f1_rest"],
        "auc":results["models"]["lightgbm_causal"]["auc"],
        "elapsed":report["time_s"]}),flush=True)

if __name__=="__main__":main()
