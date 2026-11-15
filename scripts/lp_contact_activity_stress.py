#!/usr/bin/env python3
"""Stress-test contact rules against physiological activity; read-only.
Provisional label: LP recording channels treated as possibly physically connected
for false-alarm stress tests only, NOT truth. All folds hold out participants.
"""
from pathlib import Path
import json,re
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,confusion_matrix
ROOT=Path("DATA/DATA_LONG_PRESS")
CATS=["CONNECTED","DISCONNECTED","LOOSE","TABLE","SOFT_LONG_PRESS","MID_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS","DRAG_AND_DROP"]
WL=[f"p_core->wl[CH{i}]" for i in (1,2,3)]
PS=[[f"p_core->emg_psd_mean[CH{i}][{j}]" for j in range(5)] for i in (1,2,3)]
cols=WL+sum(PS,[])+["User","Timestamp"]
Z=np.asarray([8,10,12,46,51],float)

def id(x,filename):
    s=x if pd.notna(x) and str(x).strip() else filename.split("_")[0]
    return re.sub(r"[^a-z0-9]","",str(s).lower())
def med(a):return float(np.median(a))
def logp(x):return float(np.log1p(max(0.,float(x))))
def prof(path,group):
    df=pd.read_csv(path,compression="gzip",usecols=cols)
    if len(df)<100:return None
    w=np.column_stack([df[c].to_numpy(dtype=float) for c in WL])
    ps=np.stack([np.column_stack([df[c].to_numpy(dtype=float) for c in PS[i]]) for i in range(3)],axis=1)/Z.reshape(1,1,5)
    w=np.maximum(np.nan_to_num(w),0);ps=np.maximum(np.nan_to_num(ps),0)
    pw=ps.sum(axis=2)
    out=[]
    for ch in range(3):
        x=w[:,ch];p=pw[:,ch]
        spec=np.median(ps[:,ch,:],axis=0)+1;spec/=sum(spec)
        # summaries over all 10s/30s recording, no per-window truth
        q=[np.percentile(x,t) for t in [10,20,50,80,90]]
        pq=[np.percentile(p,t) for t in [10,20,50,80,90]]
        sq=[(q[4]+1)/(q[0]+1),(pq[4]+1)/(pq[0]+1)]
        vec=[logp(q[2]),logp(pq[2]),logp(q[0]),logp(pq[0]),
             np.log(max(sq[0],1)),np.log(max(sq[1],1)),
             *list(spec),
             float(-sum(spec*np.log(spec))/np.log(5))]
        out.append({"vec":vec,"wl":q[2],"psd":pq[2],"w_dyn":sq[0],
                    "p_dyn":sq[1],"low_zone":spec[0]})
    return {"group":group,"user":id(df["User"].iloc[0],path.name),
            "file":path.name,"ch":out}

def mk(name):
    if name=="forest":return make_pipeline(SimpleImputer(),RandomForestClassifier(n_estimators=200,
                          max_depth=5,min_samples_leaf=4,class_weight="balanced",random_state=21,n_jobs=1))
    return make_pipeline(SimpleImputer(),StandardScaler(),
                         LogisticRegression(C=.35,class_weight="balanced",max_iter=2000))

def eval_model(items,feature_indices,kind):
    x=np.asarray([i["vec"] for i in items])[:,feature_indices]
    y=np.asarray([i["y"] for i in items]);usr=np.asarray([i["user"] for i in items])
    pr=np.zeros(len(y))
    fold=[]
    for tr,te in GroupKFold(n_splits=5).split(x,y,usr):
        m=mk(kind);m.fit(x[tr],y[tr])
        pp=m.predict_proba(x[te])[:,1];pr[te]=pp
        fold.append({"n":len(te),"positives":int(sum(y[te])),
            "auc":float(roc_auc_score(y[te],pp)) if len(set(y[te]))>1 else None})
    pred=pr>=.5
    return {"n":len(y),"people":len(set(usr)), "auc":float(roc_auc_score(y,pr)),
     "ba":float(balanced_accuracy_score(y,pred)),"cm":confusion_matrix(y,pred,labels=[0,1]).tolist(),
     "folds":fold}

def main():
    data=defaultdict(list)
    for group in CATS:
        for p in sorted((ROOT/group).glob("*.csv.gz")):
            try:
                r=prof(p,group)
                if r:data[group].append(r)
            except Exception as e:print("ERROR",group,p.name,type(e).__name__,str(e)[:150])
        print("GROUP",group,len(data[group]),flush=True)
    # Heuristic contact rules, deliberately blind to the file group.
    conditions={
      "WLgt3000":lambda x:x["wl"]>3000,
      "WLgt5000":lambda x:x["wl"]>5000,
      "WLgt3000_and_steady":lambda x:x["wl"]>3000 and x["w_dyn"]<1.5,
      "WLgt3000_and_lowzone_high":lambda x:x["wl"]>3000 and x["low_zone"]>.34,
      "WLgt3000_and_steady_and_lowzone":lambda x:x["wl"]>3000 and x["w_dyn"]<1.5 and x["low_zone"]>.34,
      "PSDgt1500_and_steady":lambda x:x["psd"]>1500 and x["p_dyn"]<1.5,
      "WLgt5000_and_steady":lambda x:x["wl"]>5000 and x["w_dyn"]<1.5,
    }
    for name,rule in conditions.items():
        table={}
        for group,rs in data.items():
            count=np.array([[rule(r["ch"][c]) for c in range(3)] for r in rs])
            table[group]={"n":len(rs),"fraction_each":[float(np.mean(count[:,c])) for c in range(3)],
                "fraction_any":float(np.mean(np.any(count,axis=1))),
                "fraction_CH1_CH2_both":float(np.mean(count[:,0]&count[:,1]))}
        print("RULE",name,json.dumps(table),flush=True)
    # Labeled 0=connected-like, 1=disconnected-like; LP pseudo-0 is
    # absolutely NOT proven channel-level good contact.
    vals=[]
    for r in data["CONNECTED"]:
        for c in range(3): vals.append({"vec":r["ch"][c]["vec"],"user":r["user"],"y":0,"from":"connected"})
    for r in data["DISCONNECTED"]:
        for c in range(2): vals.append({"vec":r["ch"][c]["vec"],"user":r["user"],"y":1,"from":"disconnected"})
    LP=["SOFT_LONG_PRESS","MID_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS"]
    pseudo=[]
    for group in LP:
        for r in data[group]:
            for c in range(3):
                pseudo.append({"vec":r["ch"][c]["vec"],"user":r["user"],"y":0,"from":group})
    subsets={"amplitude":[0,1,2,3],"dynamics":[4,5],"spectral_shape":[6,7,8,9,10,11],
      "amplitude_and_dynamics":[0,1,2,3,4,5],"full":list(range(12))}
    for case,items in [("contact_only",vals),("with_weak_lp",vals+pseudo)]:
        for fname,fi in subsets.items():
            for model in ["logistic","forest"]:
                result=eval_model(items,fi,model)
                print("ROBUST_CV",case,fname,model,json.dumps({k:v for k,v in result.items() if k!="folds"}),flush=True)
    # OOD predictions compare training domain: contact only vs plus weak LP.
    for case,items in [("contact_only",vals),("with_weak_lp",vals+pseudo)]:
        m=mk("forest");m.fit(np.array([i["vec"] for i in items]),np.array([i["y"] for i in items]))
        for group,records in data.items():
            test=np.asarray([r["ch"][c]["vec"] for r in records for c in range(3)])
            pp=m.predict_proba(test)[:,1].reshape(-1,3)
            print("ROBUST_OOD",case,group,json.dumps({
                "n":len(records),"any_disconnected":float(np.mean(np.any(pp>=.5,axis=1))),
                "channel_disconnected_rates":[float(np.mean(pp[:,c]>=.5)) for c in range(3)],
                "median_max_p":med(np.max(pp,axis=1))}),flush=True)
    print("ROBUSTNESS_DONE")

if __name__=="__main__":main()
