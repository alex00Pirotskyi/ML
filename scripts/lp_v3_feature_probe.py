#!/usr/bin/env python3
"""Exploratory sEMG two-state/rest and contact-feature ablations.

Read-only. No pointer, TriggerPressure or class_id used for supervision.
Directory conditions are WEAK contextual proxies, not per-sample truth.
"""
import json,math,re,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT=Path("DATA/DATA_LONG_PRESS")
GROUPS=("CONNECTED","DISCONNECTED","LOOSE","TABLE","NOTHING","MID_LONG_PRESS",
        "SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS","DRAG_AND_DROP","TAP","SWIPE")
LP={"MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS"}
W=[f"p_core->wl[CH{i}]" for i in range(1,4)]
WH=[f"p_core->wl_head[CH{i}]" for i in range(1,4)]
WT=[f"p_core->wl_tail[CH{i}]" for i in range(1,4)]
P=[[f"p_core->emg_psd_mean[CH{i}][{z}]" for z in range(5)] for i in range(1,4)]
PI=[[f"p_core->emg_psd[0][CH{i}][{z}]" for z in range(5)] for i in range(1,4)]
ACC=[f"p_core->acc_rate[{ax}][{z}]" for ax in "XYZ" for z in range(6)]
COL=set(W+WH+WT+sum(P,[])+sum(PI,[])+ACC+["Timestamp","MonotonicTimeUs","User"])
def pct(a,p):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return float(np.percentile(a,p)) if len(a) else 0.
def med(a):return pct(a,50)
def uid(row,path):
    if row is None or pd.isna(row) or not str(row).strip():row=path.name.split("_")[0]
    return re.sub("[^a-z0-9]","",str(row).lower())
def X(df,cols):
    return np.column_stack([pd.to_numeric(df[c],errors="coerce").to_numpy(float) for c in cols])
def step(df):
    for name,scale in [("MonotonicTimeUs",.001),("Timestamp",1)]:
        if name not in df:continue
        a=pd.to_numeric(df[name],errors="coerce").to_numpy(float)[:200]
        d=np.diff(a)*scale;d=d[(d>.2)&(d<50)&np.isfinite(d)]
        if len(d)>8:return float(np.median(d))
    return 6.
def record(path,group):
    d=pd.read_csv(path,compression="gzip",usecols=lambda c:c in COL,low_memory=False)
    if len(d)<250:return None
    dt=step(d);win=max(12,min(120,round(180/dt)))
    nw=len(d)//win
    if nw<12:return None
    n=nw*win
    w=np.maximum(np.nan_to_num(X(d,W)[:n]),0)
    p=np.nan_to_num(np.stack([X(d,P[c])[:n] for c in range(3)],axis=1))
    p=np.maximum(p,0)
    ins=np.nan_to_num(np.stack([X(d,PI[c])[:n] for c in range(3)],axis=1))
    ins=np.maximum(ins,0)
    head=np.maximum(np.nan_to_num(X(d,WH)[:n]),0)
    tail=np.maximum(np.nan_to_num(X(d,WT)[:n]),0)
    acc=np.maximum(np.nan_to_num(X(d,ACC)[:n]),0)
    w0=np.median(w.reshape(nw,win,3),axis=1)
    p0=np.median(p.reshape(nw,win,3,5),axis=1)
    raw0=np.median(ins.reshape(nw,win,3,5),axis=1)
    pw=np.median(p0,axis=2)/10000.
    latest=np.median(raw0,axis=2)/10000.
    logw=np.log1p(w0)
    acc_idx=np.log1p(np.median(acc.reshape(nw,win,18),axis=1))
    acc_motion=np.mean(acc_idx[:,1::6],axis=1)
    out={"group":group,"user":uid(d["User"].iloc[0] if "User" in d else None,path),
         "file":path.name,"windows":nw,"duration_s":nw*win*dt/1000,
         "dt_ms":dt,"ch":[]}
    for c in range(3):
        a=logw[:,c];b=pw[:,c];t=latest[:,c]
        ar=np.column_stack((a,b))
        try:
            m1=GaussianMixture(n_components=1,covariance_type="diag",
                reg_covar=0.007,random_state=11,max_iter=100,n_init=1).fit(ar)
            m2=GaussianMixture(n_components=2,covariance_type="diag",
                reg_covar=0.007,random_state=11,max_iter=100,n_init=2).fit(ar)
            # State ranked by sum of WL and PSD-relative separation, not absolute level.
            z=m2.predict(ar)
            order=np.argsort(m2.means_[:,0])
            lo=int(order[0]); hi=int(order[1])
            lf=z==lo;hf=z==hi
            dw=float(m2.means_[hi,0]-m2.means_[lo,0])
            dp=float(m2.means_[hi,1]-m2.means_[lo,1])
            occupancy=float(np.mean(hf))
            bic=float(m1.bic(ar)-m2.bic(ar))
            credible=bool(bic>10 and .09<occupancy<.91 and dw>math.log(1.4)
                            and dp>.018)
            rest_quant=float(np.mean(lf))
        except Exception:
            dw=dp=occupancy=bic=rest_quant=0.
            credible=False
        spec=np.median(p0[:,c,:],axis=0)/10000.
        spec_shape=np.exp((spec-np.max(spec))*math.log(10))
        spec_shape/=np.sum(spec_shape)+1e-8
        pn=np.percentile(pw[:,c],[10,25,50,75,90])
        wn=np.percentile(w0[:,c],[10,25,50,75,90])
        h0=np.median(head.reshape(nw,win,3)[:, :, c],axis=1)
        t0=np.median(tail.reshape(nw,win,3)[:, :, c],axis=1)
        h_ratio=(h0+1)/(t0+1)
        a_diff=np.abs(np.diff(np.log1p(w0[:,c])))
        b_diff=np.abs(np.diff(pw[:,c]))
        corr_motion=np.corrcoef(np.log1p(w0[:,c]),acc_motion)[0,1] if np.std(acc_motion)>.01 and np.std(w0[:,c])>.1 else 0.
        corr_psd=np.corrcoef(a,pw[:,c])[0,1] if np.std(a)>.01 and np.std(pw[:,c])>.001 else 0.
        feats=[
          *np.log1p(wn),*pn,
          math.log((wn[-1]+1)/(wn[0]+1)),pn[-1]-pn[0],
          float(np.median(a_diff)),float(np.percentile(a_diff,90)),
          float(np.median(b_diff)),float(np.percentile(b_diff,90)),
          float(np.median(h_ratio)),float(np.percentile(h_ratio,90)),
          float(corr_motion),float(corr_psd),
          *list(spec_shape),float(np.mean(t-pw[:,c])),
        ]
        out["ch"].append({"feat":list(map(float,feats)),"wl_median":med(w0[:,c]),
           "wl_dynamic":float((wn[-1]+1)/(wn[0]+1)),"psd_dynamic":pn[-1]-pn[0],
           "gmm_bic":bic,"gmm_dw":dw,"gmm_dp":dp,
           "gmm_high_frac":occupancy,"gmm_credible":credible,
           "gmm_rest_frac":rest_quant})
    return out

def factory(model):
    if model=="extratrees":
        return ExtraTreesClassifier(n_estimators=250,max_depth=5,min_samples_leaf=3,
           class_weight="balanced",n_jobs=2,random_state=41)
    return make_pipeline(StandardScaler(),LogisticRegression(
        C=.1,class_weight="balanced",max_iter=1000))
def cv(train,features,model):
    x=np.array([r["feat"] for r in train],float)
    x=x[:,features]
    y=np.array([r["label"] for r in train],int)
    user=np.array([r["user"] for r in train])
    pred=np.zeros(len(y))
    folds=[]
    for tr,te in GroupKFold(n_splits=5).split(x,y,user):
        m=factory(model);m.fit(x[tr],y[tr])
        pred[te]=m.predict_proba(x[te])[:,1]
        folds.append({"subjects":len(set(user[te])),"neg":int(sum(y[te]==0)),
            "pos":int(sum(y[te]==1))})
    return {"n":len(y),"users":len(set(user)),
        "auc":float(roc_auc_score(y,pred)),
        "ba":float(balanced_accuracy_score(y,pred>=.5)),
        "confusion":confusion_matrix(y,pred>=.5,labels=[0,1]).tolist(),
        "folds":folds}

def main():
    by=defaultdict(list)
    start=time.monotonic()
    for group in GROUPS:
        paths=sorted((ROOT/group).glob("*.csv.gz"))
        if group=="TAP":
            paths=[paths[i] for i in np.linspace(0,len(paths)-1,100,dtype=int)]
        if group=="SWIPE":
            paths=[paths[i] for i in np.linspace(0,len(paths)-1,100,dtype=int)]
        for i,p in enumerate(paths,1):
            try:
                r=record(p,group)
                if r:by[group].append(r)
            except Exception as e:print("ERROR",group,p.name,type(e).__name__,str(e)[:140],flush=True)
            if i%100==0:
                print("SCAN",group,i,"/",len(paths),flush=True)
        print("GROUP_DONE",group,len(by[group]),"seconds",round(time.monotonic()-start,1),flush=True)
    for group,rec in by.items():
        out={"n":len(rec),
            "gmm_credible_each_ch":[[sum(r["ch"][j]["gmm_credible"] for r in rec),len(rec)] for j in range(3)],
            "gmm_credible_any":[sum(any(ch["gmm_credible"] for ch in r["ch"]) for r in rec),len(rec)],
            "wl_median_ch":[med([r["ch"][j]["wl_median"] for r in rec]) for j in range(3)],
            "gmm_dw_median":[med([r["ch"][j]["gmm_dw"] for r in rec]) for j in range(3)],
            "gmm_dp_median":[med([r["ch"][j]["gmm_dp"] for r in rec]) for j in range(3)]}
        print("GMM_CONTEXT",group,json.dumps(out),flush=True)
    # Training independent participant splits; no leakage via User as feature.
    train=[]
    for group in ("CONNECTED","DISCONNECTED"):
        for r in by[group]:
            for c in range(3 if group=="CONNECTED" else 2):
                train.append({"feat":r["ch"][c]["feat"],"label":int(group=="DISCONNECTED"),
                              "user":r["user"],"group":group})
    fs={"amplitude":list(range(10)),"dynamics":list(range(10,20)),
        "spectrum":list(range(20,25)),
        "all":list(range(len(train[0]["feat"])))}
    for kind,f in fs.items():
        for model in ("linear","extratrees"):
            result=cv(train,f,model)
            print("CONTACT_CV",kind,model,json.dumps(result),flush=True)
    # Model OOD on LP: flagged examples are not necessarily true FP.
    for kind,f in fs.items():
        if kind not in ("spectrum","all"):continue
        m=factory("extratrees")
        m.fit(np.asarray([r["feat"] for r in train])[:,f],
              np.asarray([r["label"] for r in train]))
        for group,recs in by.items():
            if not recs:continue
            x=np.asarray([r["ch"][c]["feat"] for r in recs for c in range(3)])[:,f]
            pr=m.predict_proba(x)[:,1].reshape(-1,3)
            print("OOD_FLAGS",kind,group,json.dumps({
                "n":len(recs),"ch_rate":[float(np.mean(pr[:,c]>.5)) for c in range(3)],
                "any_rate":float(np.mean(np.any(pr>.5,axis=1))),
                "median_ch_p":[float(np.median(pr[:,c])) for c in range(3)]
            }),flush=True)
    print("PROBE_DONE",round(time.monotonic()-start,1),flush=True)
if __name__=="__main__":main()
