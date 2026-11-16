#!/usr/bin/env python3
"""Causal multi-timescale student and selective coverage on V3 weak labels.

All evaluation by held-out person. Never use PointerState, User, class_id,
folders, future values or teacher confidence as model INPUT FEATURES.
Teacher confidence is used for training weights and selective evaluation only.
"""
import json,time,math
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (balanced_accuracy_score,f1_score,roc_auc_score,
    confusion_matrix,precision_score,recall_score,mean_absolute_error,r2_score)

from lp_v3_teacher import label_recording
from lp_v3_labeler import train_contact_quality,normalize_person,EXTRA_READ
from lp_causal_fw_teacher import step_ms
from scripts.lp_v3_feature_probe import record
from scripts.lp_classifier_benchmark_v2 import get_features,TARGETS,GROUPS

def matrix(d,cols):
    out=[]
    for name in cols:
        if name in d:
            out.append(np.maximum(pd.to_numeric(d[name],errors="coerce").fillna(0).to_numpy(float),0))
        else:
            out.append(np.zeros(len(d)))
    return np.column_stack(out)

def feature_bank(d,base):
    n=len(d);dt=step_ms(d)
    w=matrix(d,[f"p_core->wl[CH{c}]" for c in range(1,4)])
    logs=np.log1p(w)
    ps=np.stack([matrix(d,[f"p_core->emg_psd_mean[CH{c}][{z}]"
                      for z in range(5)]) for c in range(1,4)],axis=1)
    raw=np.stack([matrix(d,[f"p_core->emg_psd[0][CH{c}][{z}]"
                      for z in range(5)]) for c in range(1,4)],axis=1)
    p=np.median(ps,axis=2)/10000.
    p0=np.median(raw,axis=2)/10000.
    X=[base["causal"].astype(float)]
    nms=[]
    for tau in (.25,.8,2.5,8.,20.):
        alpha=1-math.exp(-dt/(tau*1000))
        ew_w=pd.DataFrame(logs).ewm(alpha=alpha,adjust=False).mean().to_numpy()
        ew_p=pd.DataFrame(p).ewm(alpha=alpha,adjust=False).mean().to_numpy()
        X.extend([ew_w,ew_p,logs-ew_w,p-ew_p])
        nms+=["emaWL"+str(tau),"emaP"+str(tau),"WL-diff"+str(tau),"P-diff"+str(tau)]
    # Running asymmetric low floor (fast decrease, almost-frozen increase).
    floors=np.zeros_like(logs)
    low_ps=np.zeros_like(p)
    floors[0]=logs[0];low_ps[0]=p[0]
    adown=1-math.exp(-dt/650.)
    aup=1-math.exp(-dt/80000.)
    for i in range(1,n):
        diff=logs[i]-floors[i-1]
        floors[i]=floors[i-1]+np.where(diff<0,adown,aup)*diff
        diffp=p[i]-low_ps[i-1]
        low_ps[i]=low_ps[i-1]+np.where(diffp<0,adown,aup)*diffp
    X.extend([floors,low_ps,logs-floors,p-low_ps,p0-p])
    # Half-window WL asymmetry and short-range EMG event transitions.
    head=matrix(d,[f"p_core->wl_head[CH{c}]" for c in range(1,4)])
    tail=matrix(d,[f"p_core->wl_tail[CH{c}]" for c in range(1,4)])
    X.extend([np.log((head+2)/(tail+2)),np.log1p(head),np.log1p(tail)])
    # Motion is diagnostic of uncertainty but not itself a muscle activity label.
    acc=matrix(d,[f"p_core->acc_rate[{axis}][{z}]"
                  for axis in "XYZ" for z in range(6)])
    acc=np.log1p(acc)
    X.append(acc)
    axis_mean=np.column_stack([np.mean(acc[:,a*6:(a+1)*6],axis=1)
                               for a in range(3)])
    X.append(axis_mean)
    X.append(axis_mean-pd.DataFrame(axis_mean).ewm(
        alpha=1-math.exp(-dt/1500.),adjust=False).mean().to_numpy())
    # Spectral-shape changes over time.
    zones=ps/10000.
    centered=zones-np.mean(zones,axis=2,keepdims=True)
    X.append(centered.reshape(n,15))
    X.append(np.max(logs,axis=1,keepdims=True)-np.min(logs,axis=1,keepdims=True))
    out=np.column_stack(X)
    return np.nan_to_num(out,nan=0,posinf=20,neginf=-20).astype(np.float32)

def main():
    root=Path("DATA/DATA_LONG_PRESS")
    start=time.monotonic()
    full,subject_models,known,ref,audit=train_contact_quality(root,.7)
    xs=[];bank=[];ys=[];wts=[];target=[];unknown=[];users=[];groups=[]
    count=Counter();errors=[]
    files=[]
    for group in GROUPS:
        pp=sorted((root/group).glob("*.csv.gz"))
        if group in ("TAP","SWIPE") and len(pp)>100:
            pp=[pp[i] for i in np.linspace(0,len(pp)-1,100,dtype=int)]
        files.extend((group,p) for p in pp)
    for i,(group,path) in enumerate(files,1):
        try:
            df=pd.read_csv(path,compression="gzip",usecols=lambda c:c in EXTRA_READ)
            if len(df)<250:continue
            if str(path) in known:prob=known[str(path)]
            else:
                rec=record(path,group)
                if rec is None:continue
                user=normalize_person(df["User"].iloc[0] if "User" in df else None,path.name)
                m=subject_models.get(user,full)
                prob=m.predict_proba(np.array([c["feat"] for c in rec["ch"]]))[:,1]
            labels,_=label_recording(df,contact_prob=prob,
                good_reference_wl=ref,context=group)
            current=get_features(df)
            rich=feature_bank(df,current)
            take=np.unique(np.linspace(0,len(df)-1,80,dtype=int))
            xs.append(current["causal"][take])
            bank.append(rich[take])
            lab=labels.iloc[take]
            ys.append(lab["lp_state"].to_numpy(np.uint8))
            wts.append(lab["lp_state_weight"].to_numpy(np.float32))
            target.append(lab[TARGETS].to_numpy(np.float32))
            unknown.append(lab[[f"lp_unknown_ch{c}" for c in range(1,4)]].to_numpy(np.float32))
            uid=normalize_person(df["User"].iloc[0] if "User" in df else None,path.name)
            users.extend([uid]*len(take));groups.extend([group]*len(take))
            count[group]+=1
        except Exception as e:
            errors.append({"file":path.name,"group":group,"error":str(e)[:180]})
            print("ERROR",group,path.name,str(e)[:150],flush=True)
        if i%100==0 or i==len(files):
            print("PROGRESS",i,len(files),"ok",sum(count.values()),
                  "errors",len(errors),"sec",round(time.monotonic()-start,1),flush=True)
    y=np.concatenate(ys);w=np.concatenate(wts);U=np.concatenate(unknown)
    yt=np.concatenate(target)
    Xbasic=np.concatenate(xs);Xrich=np.concatenate(bank)
    persons=np.asarray(users);groups=np.asarray(groups)
    folds=list(GroupKFold(n_splits=3).split(Xbasic,y,groups=persons))
    print("DATA",json.dumps({"rows":len(y),"files":dict(count),
        "people":len(set(persons)),"teacher_rest_fraction":float(np.mean(y)),
        "mean_weight":float(np.mean(w)),
        "basic_features":Xbasic.shape[1],"rich_features":Xrich.shape[1],
        "errors":errors[:30]}),flush=True)
    results={}
    for kind,xx,strict in (
        ("basic44",Xbasic,False),("multiscale",Xrich,False),
        ("multiscale_train_confident",Xrich,True)
    ):
        predictions=np.zeros(len(y),float)
        for fi,(tr,te) in enumerate(folds):
            train=tr[w[tr]>=.25] if strict else tr
            m=lgb.LGBMClassifier(n_estimators=250,num_leaves=31,max_depth=7,
                learning_rate=.05,min_child_samples=35,reg_lambda=3,
                verbosity=-1,n_jobs=2,random_state=42)
            if len(set(y[train]))!=2:
                raise RuntimeError("A confidence filter excluded a class")
            m.fit(xx[train],y[train],sample_weight=np.maximum(w[train],.1))
            predictions[te]=m.predict_proba(xx[te])[:,1]
        pr=predictions>=.5
        out={"n":len(y),"accuracy":float(np.mean(pr==y)),
            "ba":float(balanced_accuracy_score(y,pr)),
            "rest_f1":float(f1_score(y,pr,zero_division=0)),
            "recall_rest":float(recall_score(y,pr,zero_division=0)),
            "precision_rest":float(precision_score(y,pr,zero_division=0)),
            "auc":float(roc_auc_score(y,predictions)),
            "confusion":confusion_matrix(y,pr,labels=[0,1]).tolist(),
            "confidence_strata":{},"groups":{}}
        for thr in (.1,.25,.4,.55,.65,.8):
            sel=w>=thr
            s=y[sel];p=pr[sel]
            out["confidence_strata"][str(thr)]={
                "rows":int(sum(sel)),"coverage":float(np.mean(sel)),
                "rest_frac":float(np.mean(s)) if len(s) else None,
                "ba":float(balanced_accuracy_score(s,p)) if len(set(s))==2 else None,
                "f1":float(f1_score(s,p,zero_division=0)) if len(s) else None}
        for g in ("CONNECTED","DISCONNECTED","SOFT_LONG_PRESS","MID_LONG_PRESS",
                  "STRONG_LONG_PRESS","TAP","SWIPE"):
            mask=groups==g
            out["groups"][g]={"n":int(sum(mask)),
              "rest_frac":float(np.mean(y[mask])),
              "recall_rest":float(recall_score(y[mask],pr[mask],zero_division=0)),
              "ba":float(balanced_accuracy_score(y[mask],pr[mask]))
              if len(set(y[mask]))==2 else None}
        results[kind]=out
        print("SELECTIVE_MODEL",kind,json.dumps(out),flush=True)

    # Per-channel regression: compare global weights against
    # appropriate channel-specific quality scores.
    tr,te=folds[0]
    regs={}
    for c in range(3):
        for t in range(3):
            index=c*3+t
            for style in ("global_weight","channel_specific"):
                mw=np.maximum(w[tr],.1) if style=="global_weight" else np.maximum(
                    1-U[tr,c],.1)
                model=lgb.LGBMRegressor(n_estimators=240,learning_rate=.05,
                   num_leaves=31,max_depth=7,min_child_samples=35,
                   reg_lambda=3,verbosity=-1,n_jobs=2,random_state=42)
                model.fit(Xrich[tr],yt[tr,index],sample_weight=mw)
                p=np.clip(model.predict(Xrich[te]),0,1)
                actual=yt[te,index]
                regs[f"{TARGETS[index]}_{style}"]={
                  "mae":float(mean_absolute_error(actual,p)),
                  "r2":float(r2_score(actual,p)) if np.var(actual)>1e-9 else None,
                  "test_n":len(te)}
                print("REGRESSION_ABLATION",TARGETS[index],style,
                    json.dumps(regs[f"{TARGETS[index]}_{style}"]),flush=True)
    out={"warning":"V3 teacher imitation with weak labels; not physiological ground truth.",
       "rows":len(y),"contacts_weak_proxy":audit,"models":results,
       "regressions":regs,"elapsed_s":round(time.monotonic()-start,1)}
    Path("/tmp/lp_v3_multiscale.json").write_text(json.dumps(out,indent=2))
    print("MULTISCALE_DONE",out["elapsed_s"],flush=True)

if __name__=="__main__":main()
