#!/usr/bin/env python3
"""Participant-held-out classification of sEMG heuristic teacher pseudo-labels.

NOT measured truth. lp_state=0 includes uncertain samples. All comparisons
measure student imitation of lp_rest_labeler.py; no PointerState/class_id.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score, average_precision_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lp_rest_labeler import compute_labels, median_step_ms, WL_COLUMNS, PSD_COLUMNS

GROUPS = ("CONNECTED","DISCONNECTED","LOOSE","NOTHING","TABLE","MID_LONG_PRESS",
          "SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS","DRAG_AND_DROP","TAP","SWIPE")
TARGETS=[f"lp_{name}_ch{i}" for i in (1,2,3)
         for name in ("resting","high_activity","disconnection")]
KINDS=("resting","high_activity","disconnection","unknown")


def roll(a,n,kind="median",q=None):
    obj=pd.DataFrame(a).rolling(max(1,n),min_periods=1)
    return obj.median().to_numpy() if kind=="median" else obj.quantile(q).to_numpy()


def get_features(df):
    n=len(df)
    dt=median_step_ms(df)
    nf=max(4,min(250,round(180/dt)))
    ns=max(nf+1,min(800,round(2200/dt)))
    wl=np.maximum(np.nan_to_num(np.column_stack(
       [pd.to_numeric(df[x],errors="coerce").to_numpy(float) for x in WL_COLUMNS]),
       nan=0,posinf=0,neginf=0),0)
    spec=np.zeros((n,3,5))
    for c in range(3):
        for z,name in enumerate(PSD_COLUMNS[c]):
            if name in df:
                spec[:,c,z]=np.maximum(np.nan_to_num(
                  pd.to_numeric(df[name],errors="coerce").to_numpy(float),
                  nan=0,posinf=0,neginf=0),0)
    power=spec.sum(axis=2)
    fractions=(spec+1)/(power[:,:,None]+5)
    fwl=roll(wl,nf)
    fp=roll(power,nf)
    swl=roll(fwl,ns)
    sp=roll(fp,ns)
    wd=(roll(fwl,ns,"quantile",.9)+1)/(roll(fwl,ns,"quantile",.1)+1)
    raw=np.column_stack((np.log1p(wl),np.log1p(power),fractions.reshape(n,15)))
    causal=np.column_stack((raw,np.log1p(fwl),np.log1p(fp),
           np.log1p(swl),np.log1p(sp),
           np.log((fwl+25)/(swl+25)),np.log((fp+25)/(sp+25)),
           np.log(np.maximum(wd,1e-8)),
           np.log1p(np.arange(n)*dt/1000),np.full(n,dt)))
    basew=np.percentile(fwl,15,axis=0)
    basep=np.percentile(fp,15,axis=0)
    dyn=(np.percentile(fwl,90,axis=0)+1)/(np.percentile(fwl,10,axis=0)+1)
    # Demonstrates an UNDEPLOYABLE offline oracle feature mode.
    oracle=np.column_stack((causal,
      np.tile(np.log1p(basew),(n,1)),np.tile(np.log1p(basep),(n,1)),
      np.tile(np.log1p(dyn),(n,1)),
      np.log((fwl+25)/(basew+25)),np.log((fp+25)/(basep+25))))
    return {k:np.nan_to_num(v,nan=0,posinf=20,neginf=-20).astype("float32")
            for k,v in (("raw",raw),("causal",causal),("oracle",oracle))}


def subject(value,file):
    name=str(value) if value is not None and pd.notna(value) and str(value).strip() else file.split("_")[0]
    name=re.sub("[^a-z0-9]+","",name.lower())
    for stem in ("ohad","nikol","shahar"):
        if name.startswith(stem):return stem
    return name or "unknown"


def extract(root,per_file=80,tap_cap=0,swipe_cap=0):
    needed=set(WL_COLUMNS)|{"User","Timestamp","MonotonicTimeUs"}
    for group in PSD_COLUMNS:needed.update(group)
    collection=defaultdict(list)
    ys,weights,targets,classes,users,groups=[],[],[],[],[],[]
    count=Counter()
    failed=[]
    paths=[]
    for g in GROUPS:
        files=sorted((root/g).glob("*.csv.gz"))
        limit=tap_cap if g=="TAP" else swipe_cap if g=="SWIPE" else 0
        if limit and len(files)>limit:
            files=[files[i] for i in np.linspace(0,len(files)-1,limit,dtype=int)]
        paths.extend((g,p) for p in files)
    print("FILES_SELECTED",len(paths),flush=True)
    start=time.monotonic()
    for j,(group,path) in enumerate(paths,1):
        try:
            df=pd.read_csv(path,compression="gzip",
                usecols=lambda c:c in needed,low_memory=False)
            if len(df)<16:raise ValueError("fewer than 16 samples")
            label,_=compute_labels(df)
            feats=get_features(df)
            select=np.unique(np.linspace(0,len(df)-1,min(per_file,len(df)),dtype=int))
            y=label.iloc[select]
            for k in feats:collection[k].append(feats[k][select])
            ys.append(y["lp_state"].to_numpy(np.uint8))
            weights.append(y["lp_state_weight"].to_numpy(np.float32))
            targets.append(y[TARGETS].to_numpy(np.float32))
            classes.append(np.column_stack([
               np.argmax(y[[f"lp_{kind}_ch{ch}" for kind in KINDS]].to_numpy(),axis=1)
               for ch in (1,2,3)
            ]).astype(np.uint8))
            uid=subject(df["User"].iloc[0] if "User" in df else None,path.name)
            users.extend([uid]*len(select));groups.extend([group]*len(select))
            count[group]+=1
        except Exception as exc:
            failed.append([str(path),f"{type(exc).__name__}: {exc}"[:140]])
        if j%100==0 or j==len(paths):
            print("EXTRACT",j,len(paths),"ok",sum(count.values()),"failed",len(failed),
                "elapsed_s",round(time.monotonic()-start,1),flush=True)
    if not ys:raise RuntimeError("No files successfully processed")
    return {
      "x":{k:np.concatenate(v) for k,v in collection.items()},
      "y":np.concatenate(ys),"weight":np.concatenate(weights),
      "targets":np.concatenate(targets),"classes":np.concatenate(classes),
      "users":np.array(users),"groups":np.array(groups),
      "files":dict(count),"selected":len(paths),"failed":failed[:100],
    }


def clf(kind):
    if kind=="lr":
        return make_pipeline(StandardScaler(),
          LogisticRegression(solver="liblinear",max_iter=400))
    return lgb.LGBMClassifier(n_estimators=200,learning_rate=.055,
      num_leaves=31,max_depth=8,min_child_samples=50,
      reg_lambda=2,n_jobs=2,verbosity=-1,random_state=42)


def reg():
    return lgb.LGBMRegressor(n_estimators=200,learning_rate=.055,
      num_leaves=31,max_depth=8,min_child_samples=50,
      reg_lambda=2,n_jobs=2,verbosity=-1,random_state=42)


def stats(y,pr,proba,weight=None):
    y=y.astype(int);pr=pr.astype(int)
    d={"n":len(y),"rest_fraction":float(np.mean(y)),
      "accuracy":float(accuracy_score(y,pr)),
      "balanced_accuracy":float(balanced_accuracy_score(y,pr)),
      "f1_rest":float(f1_score(y,pr,zero_division=0)),
      "macro_f1":float(f1_score(y,pr,average="macro",zero_division=0)),
      "precision_rest":float(precision_score(y,pr,zero_division=0)),
      "recall_rest":float(recall_score(y,pr,zero_division=0)),
      "specificity_nonrest":float(np.mean(pr[y==0]==0)) if (y==0).any() else None,
      "confusion_0_then_1":confusion_matrix(y,pr,labels=[0,1]).tolist(),
      "auc":float(roc_auc_score(y,proba)) if len(set(y))==2 else None,
      "pr_auc_rest":float(average_precision_score(y,proba)) if len(set(y))==2 else None}
    if weight is not None:
        d["weighted_accuracy"]=float(accuracy_score(y,pr,sample_weight=weight))
        d["weighted_balanced_accuracy"]=float(balanced_accuracy_score(
            y,pr,sample_weight=weight))
    return d


def evaluate(d,nfolds):
    y=d["y"];weight=d["weight"];group=d["groups"];uid=d["users"]
    X=d["x"]
    splits=list(GroupKFold(n_splits=nfolds).split(X["causal"],y,uid))
    print("SPLITS",json.dumps([
      {"fold":i+1,"n_test":len(te),"subjects":len(set(uid[te])),
       "rest_fraction":round(float(np.mean(y[te])),3)}
      for i,(_,te) in enumerate(splits)]),flush=True)
    results={"models":{},"regressions":{},"four_class":{}}
    for model_name,view in (
      ("majority","raw"),("lightgbm_raw","raw"),
      ("logistic_causal","causal"),("lightgbm_causal","causal"),
      ("lightgbm_oracle","oracle")
    ):
        pred=np.zeros(len(y),np.uint8);p=np.zeros(len(y),np.float32)
        folds=[];start=time.monotonic()
        for i,(tr,te) in enumerate(splits,1):
            if model_name=="majority":
                pp=np.full(len(te),np.mean(y[tr]))
            else:
                model=clf("lr" if "logistic" in model_name else "gbm")
                sample_weight=np.maximum(weight[tr],.1)
                if model_name.startswith("logistic"):
                    model.fit(X[view][tr],y[tr],logisticregression__sample_weight=sample_weight)
                else:
                    model.fit(X[view][tr],y[tr],sample_weight=sample_weight)
                pp=model.predict_proba(X[view][te])[:,1]
            p[te]=pp;pred[te]=pp>=.5
            folds.append({"fold":i,"balanced_accuracy":float(balanced_accuracy_score(y[te],pred[te])),
              "f1_rest":float(f1_score(y[te],pred[te],zero_division=0))})
        out=stats(y,pred,p,weight)
        out["by_group"]={}
        for g in GROUPS:
            mask=group==g
            if np.any(mask):
                small=stats(y[mask],pred[mask],p[mask])
                out["by_group"][g]={k:small[k] for k in (
                  "n","rest_fraction","accuracy","balanced_accuracy",
                  "f1_rest","recall_rest","specificity_nonrest")}
        out["folds"]=folds;out["elapsed_s"]=round(time.monotonic()-start,1)
        out["features"]=X[view].shape[1]
        results["models"][model_name]=out
        print("CLASSIFIER",model_name,json.dumps({k:v for k,v in out.items()
             if k not in ("by_group","folds")}),flush=True)
        print("CLASSIFIER_GROUPS",model_name,json.dumps(out["by_group"]),flush=True)
    # First held-out fold only, 9 regressors.
    tr,te=splits[0];reg_results={}
    for j,name in enumerate(TARGETS):
        m=reg();m.fit(X["causal"][tr],d["targets"][tr,j],
          sample_weight=np.maximum(weight[tr],.1))
        prediction=np.clip(m.predict(X["causal"][te]),0,1)
        true=d["targets"][te,j]
        entry={"n":len(te),"subjects":len(set(uid[te])),
          "mae":float(mean_absolute_error(true,prediction)),
          "rmse":float(np.sqrt(mean_squared_error(true,prediction))),
          "r2":float(r2_score(true,prediction)) if np.var(true)>1e-10 else None,
          "target_positive_fraction":float(np.mean(true>=.5))}
        reg_results[name]=entry
        print("REGRESSION",name,json.dumps(entry),flush=True)
    results["regressions"]=reg_results
    # Per-channel 4-class label argmax, using same first held-out fold.
    for ch in range(3):
        m=lgb.LGBMClassifier(n_estimators=200,learning_rate=.055,
          num_leaves=31,max_depth=8,min_child_samples=50,
          reg_lambda=2,n_jobs=2,verbosity=-1,random_state=42)
        classes=d["classes"][:,ch]
        m.fit(X["causal"][tr],classes[tr],sample_weight=np.maximum(weight[tr],.1))
        prediction=m.predict(X["causal"][te])
        true=classes[te]
        entry={"n":len(te),
          "true_counts":dict(zip(KINDS,[int(np.sum(true==i)) for i in range(4)])),
          "accuracy":float(accuracy_score(true,prediction)),
          "macro_f1":float(f1_score(true,prediction,
               labels=[0,1,2,3],average="macro",zero_division=0)),
          "confusion_rest_active_disconnected_unknown":
              confusion_matrix(true,prediction,labels=[0,1,2,3]).tolist()}
        results["four_class"][f"CH{ch+1}"]=entry
        print("CHANNEL_4CLASS",f"CH{ch+1}",json.dumps(entry),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=Path("DATA/DATA_LONG_PRESS"))
    parser.add_argument("--per-file",type=int,default=80)
    parser.add_argument("--tap-cap",type=int,default=0)
    parser.add_argument("--swipe-cap",type=int,default=0)
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--out",type=Path,default=Path("/tmp/lp_classifier_benchmark.json"))
    a=parser.parse_args()
    start=time.monotonic()
    dataset=extract(a.input_dir,a.per_file,a.tap_cap,a.swipe_cap)
    overview={"selected_files":dataset["selected"],"successful_files":dataset["files"],
      "failed":dataset["failed"],"rows":len(dataset["y"]),
      "subjects":len(set(dataset["users"])),
      "rest_fraction":float(np.mean(dataset["y"])),
      "mean_weight":float(np.mean(dataset["weight"])),
      "features":{k:int(v.shape[1]) for k,v in dataset["x"].items()},
      "rest_fraction_by_group":{
         g:round(float(np.mean(dataset["y"][dataset["groups"]==g])),3)
         for g in GROUPS if np.any(dataset["groups"]==g)}}
    print("DATASET",json.dumps(overview),flush=True)
    results=evaluate(dataset,a.folds)
    report={"warning":"Only imitation of noncausal pseudo-labels; NOT physical rest/press/contact accuracy.",
       "split":"GroupKFold by normalized User, no pointer or class_id in model inputs",
       "offline_oracle":"oracle uses whole recording; not stream-deployable",
       "dataset":overview,"results":results,
       "elapsed_s":round(time.monotonic()-start,1)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2))
    print("COMPLETE",json.dumps({"output":str(a.out),"elapsed_s":report["elapsed_s"],
      "primary_ba":results["models"]["lightgbm_causal"]["balanced_accuracy"],
      "primary_f1":results["models"]["lightgbm_causal"]["f1_rest"]}),flush=True)


if __name__=="__main__":main()
