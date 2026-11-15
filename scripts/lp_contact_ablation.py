#!/usr/bin/env python3
"""Contact channel ablation, matched conditions, muscle-activation stress.

IMPORTANT: labels are operational recording contexts, not verified
channel impedance. In DISCONNECTED, CH3 is *not* automatically disconnected.
"""
from pathlib import Path
from collections import defaultdict
import json,re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score,precision_recall_fscore_support

ROOT=Path("DATA/DATA_LONG_PRESS")
GROUPS=["CONNECTED","DISCONNECTED","LOOSE","TABLE","MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS"]
W=[f"p_core->wl[CH{i}]" for i in (1,2,3)]
P=[[f"p_core->emg_psd_mean[CH{i}][{z}]" for z in range(5)] for i in (1,2,3)]
HEAD=[f"p_core->wl_head[CH{i}]" for i in (1,2,3)]
TAIL=[f"p_core->wl_tail[CH{i}]" for i in (1,2,3)]
COLS=set(W+sum(P,[])+HEAD+TAIL+["User","Timestamp"])
Z=np.array([8.,10.,12.,46.,51.])

def uid(row,stem):
    raw=str(row) if pd.notna(row) and str(row).strip() else stem.split("_")[0]
    return re.sub("[^a-z0-9]+","",raw.lower())
def orient(name):
    s=name.lower()
    for opt in ["horizontal","vertical","reverse"]:
        if opt in s:return opt
    return "unknown"
def pct(a,p):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    return float(np.percentile(a,p)) if len(a) else None
def med(a):return pct(a,50)
def finite(a):return np.nan_to_num(a,nan=0,posinf=0,neginf=0)
def record(path,group):
    df=pd.read_csv(path,usecols=lambda c:c in COLS,compression="gzip")
    if len(df)<100:return None
    w=np.column_stack([df[c].to_numpy(dtype=float) for c in W])
    ps=np.stack([np.column_stack([df[c].to_numpy(dtype=float) for c in P[ch]]) for ch in range(3)],axis=1)/Z.reshape(1,1,5)
    head=np.column_stack([df[c].to_numpy(dtype=float) for c in HEAD])
    tail=np.column_stack([df[c].to_numpy(dtype=float) for c in TAIL])
    w=np.maximum(finite(w),0);ps=np.maximum(finite(ps),0)
    power=np.sum(ps,axis=2)
    feats=[]
    shape=[]
    for i in range(3):
        a=w[:,i];p=power[:,i]
        q10=pct(a,10);q50=pct(a,50);q90=pct(a,90)
        pq10=pct(p,10);pq50=pct(p,50);pq90=pct(p,90)
        spec=np.median(ps[:,i,:],axis=0)+1;spec/=np.sum(spec)
        entropy=-float(np.sum(spec*np.log(spec)))/np.log(5)
        # robust temporal change of signal based on 10-frame median sampling
        n=len(a)//10
        ww=np.median(a[:n*10].reshape(n,10),axis=1) if n else a
        rr=np.abs(np.diff(np.log1p(ww))) if len(ww)>1 else np.array([0.])
        hratio=float(np.median((head[:,i]+1)/(tail[:,i]+1)))
        corr=float(np.corrcoef(np.log1p(a),np.log1p(p))[0,1]) if np.std(a)>1e-4 and np.std(p)>1e-4 else 0.
        vec=[
           np.log1p(q10),np.log1p(q50),np.log1p(q90),np.log1p(pq10),np.log1p(pq50),np.log1p(pq90),
           np.log((q90+1)/(q10+1)),np.log((pq90+1)/(pq10+1)),
           pct(rr,50),pct(rr,90),hratio,corr,entropy,
           *[float(v) for v in spec]]
        feats.append(vec)
        shape.append(spec.tolist())
    return {"group":group,"user":uid(df["User"].iloc[0],path.name),
      "orientation":orient(path.name),"file":path.name,"features":feats,
      "wl_median":[float(np.median(w[:,i])) for i in range(3)],
      "power_median":[float(np.median(power[:,i])) for i in range(3)],
      "shape":shape}

def model_factory(name):
    if name=="logistic":
        return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                             LogisticRegression(C=.25,class_weight="balanced",max_iter=2000))
    return make_pipeline(SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=180,max_depth=4,min_samples_leaf=3,
                               class_weight="balanced",random_state=42,n_jobs=1))

def evaluate(records,spec,method,store_preds=False):
    X=np.asarray([r["features"] for r in records],dtype=float)
    y=np.asarray([r["target"] for r in records],dtype=int)
    groups=np.asarray([r["user"] for r in records])
    xx=X[:,spec]
    preds=np.zeros(len(records),dtype=float)
    folds=[]
    for tr,te in GroupKFold(n_splits=5).split(xx,y,groups):
        mdl=model_factory(method);mdl.fit(xx[tr],y[tr])
        pp=mdl.predict_proba(xx[te])[:,1]
        preds[te]=pp
        folds.append({"test_people":len(set(groups[te])),"test_pos":int(sum(y[te])),
                      "test_neg":int(len(te)-sum(y[te])),"auc":float(roc_auc_score(y[te],pp)) if len(set(y[te]))==2 else None})
    classified=(preds>=.5).astype(int)
    cm=confusion_matrix(y,classified,labels=[0,1]).tolist()
    return {"n":len(y),"users":len(set(groups)),
      "auc":float(roc_auc_score(y,preds)),"balanced_acc":float(balanced_accuracy_score(y,classified)),
      "confusion_0_then_1":cm,"folds":folds,
      **({"preds":preds} if store_preds else {})}

def main():
    by=defaultdict(list)
    for group in GROUPS:
        paths=sorted((ROOT/group).glob("*.csv.gz"))
        for p in paths:
            try:
                r=record(p,group)
                if r:by[group].append(r)
            except Exception as e:
                print("ERROR",group,p.name,str(e)[:120],flush=True)
        print("GROUP",group,len(by[group]),flush=True)
    # Match same person+orientation, no pooling across unrelated users.
    connected={(r["user"],r["orientation"]):r for r in by["CONNECTED"]}
    disconnected={(r["user"],r["orientation"]):r for r in by["DISCONNECTED"]}
    common=sorted(set(connected)&set(disconnected))
    matched={}
    for prop in ["wl_median","power_median"]:
        gains=[[np.log((disconnected[key][prop][i]+1)/(connected[key][prop][i]+1))
                  for key in common] for i in range(3)]
        matched[prop]=[{
          "median_fold":float(np.exp(med(a))),"p10_fold":float(np.exp(pct(a,10))),
          "p90_fold":float(np.exp(pct(a,90))),
          "fraction_increased":float(np.mean(np.asarray(a)>0))}
          for a in gains]
    print("MATCHED",json.dumps({"pairs":len(common),"users":len({k[0] for k in common}),
        "ratios":matched}),flush=True)
    # Channel label assumption: connected folder channels 1/2/3 positive
    # for healthy contact, disconnected folder CH1/CH2 provisionally bad,
    # CH3 unknown status and excluded from labels.
    records=[]
    for group in ["CONNECTED","DISCONNECTED"]:
        for r in by[group]:
            for ch in range(3 if group=="CONNECTED" else 2):
                records.append({"features":r["features"][ch],"user":r["user"],
                     "target":int(group=="DISCONNECTED"),"group":group,"channel":ch})
    subsets={
      "wl_median_only":[1],
      "psd_median_only":[4],
      "all_amplitude":[0,1,2,3,4,5],
      "temporal_variability":[6,7,8,9],
      "spectral_shape":[12,13,14,15,16,17],
      "head_tail_and_wl_psd_corr":[10,11],
      "amplitude_plus_shape":[0,1,2,3,4,5,12,13,14,15,16,17],
      "all_features":list(range(18))
    }
    for spec_name,spec in subsets.items():
        for method in ["logistic","forest"]:
            result=evaluate(records,spec,method)
            compact={key:result[key] for key in ["n","users","auc","balanced_acc","confusion_0_then_1"]}
            print("ABLATION",spec_name,method,json.dumps(compact),flush=True)
    # In-sample stress test of full contact model on other conditions:
    # informative OOD behavior, NOT heldout accuracy.
    model=model_factory("forest")
    X=np.asarray([r["features"] for r in records],dtype=float)
    y=np.asarray([r["target"] for r in records],dtype=int)
    model.fit(X,y)
    for group in GROUPS:
        test=np.asarray([r["features"] for r in by[group]],dtype=float)
        if not len(test):continue
        output=np.stack([model.predict_proba(test[:,ch,:])[:,1] for ch in range(3)],axis=1)
        print("OOD_DISCONNECTION_FLAG",group,json.dumps({
          "n_files":len(output),"fraction_flag_each_ch":[float(np.mean(output[:,i]>=.5)) for i in range(3)],
          "fraction_any_flag":float(np.mean(np.any(output>=.5,axis=1))),
          "median_p_disconnected_each_ch":[float(np.median(output[:,i])) for i in range(3)],
          "median_max_p_disconnected":float(np.median(np.max(output,axis=1)))
        }),flush=True)
    print("CONTACT_ABLATION_DONE",flush=True)

if __name__=="__main__":main()
