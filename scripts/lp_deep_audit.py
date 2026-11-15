#!/usr/bin/env python3
"""Deep audit of EMG contact quality and cohort. No dataset mutations.

All supervised contact scores are FOLDER-LABEL proxies. DISCONNECTED CH1/2
only are provisional affected-channel labels; CH3 excluded from targets.
"""
from pathlib import Path
from collections import defaultdict, Counter
import json, math, re, os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT=Path("DATA/DATA_LONG_PRESS")
GROUPS=["MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS",
        "DRAG_AND_DROP","CONNECTED","DISCONNECTED","LOOSE","NOTHING","TABLE","TAP","SWIPE"]
Z=np.array([8.,10.,12.,46.,51.])
WC=[f"p_core->wl[CH{i}]" for i in (1,2,3)]
HC=[f"p_core->wl_head[CH{i}]" for i in (1,2,3)]
TC=[f"p_core->wl_tail[CH{i}]" for i in (1,2,3)]
PC=[[f"p_core->emg_psd_mean[CH{i}][{z}]" for z in range(5)] for i in (1,2,3)]
AC=[f"p_core->acc_rate[{a}][{z}]" for a in "XYZ" for z in range(6)]
COLS=set(WC+HC+TC+sum(PC,[])+AC+["Timestamp","PointerState","TriggerPressure","class_id","User","Hand","SampleIndex","MonotonicTimeUs","psd_mean_applied"])

def val(v):
    try: 
        f=float(v)
        return f if np.isfinite(f) else None
    except (ValueError,TypeError): return None
def quant(x,p):
    x=np.asarray(x,dtype=float)
    x=x[np.isfinite(x)]
    return float(np.percentile(x,p)) if x.size else None
def summary(x):
    a=[y for y in [val(v) for v in x] if y is not None]
    return {"n":len(a),"p10":quant(a,10),"p25":quant(a,25),"median":quant(a,50),"p75":quant(a,75),"p90":quant(a,90)}
def user_id(x,stem):
    s=str(x) if pd.notna(x) and str(x).strip() else stem.split("_")[0]
    return re.sub(r"[^a-z0-9]+","",s.lower()) or "unknown"
def div(a,b):
    return a/b if b>0 and np.isfinite(b) and np.isfinite(a) else 0.0
def logratio(a,b):
    return np.log((max(float(a),0)+1)/(max(float(b),0)+1))

def file_stats(path,group):
    df=pd.read_csv(path,compression="gzip",usecols=lambda c:c in COLS,low_memory=False)
    if any(x not in df.columns for x in WC+sum(PC,[])) or len(df)<100:return None
    n=len(df)
    uid=user_id(df["User"].iloc[0] if "User" in df else None,path.stem)
    t=pd.to_numeric(df["Timestamp"],errors="coerce").to_numpy(dtype=float)
    dif=np.diff(t)
    valid_dif=dif[np.isfinite(dif)&(dif>0)&(dif<200)]
    dt=quant(valid_dif,50) or 5
    win=int(np.clip(round(250/dt),20,150))
    nwin=n//win
    if nwin<5:return None
    limit=nwin*win
    wl=np.column_stack([pd.to_numeric(df[c],errors="coerce").to_numpy(dtype=float)[:limit] for c in WC])
    h=np.column_stack([pd.to_numeric(df[c],errors="coerce").to_numpy(dtype=float)[:limit] for c in HC])
    tail=np.column_stack([pd.to_numeric(df[c],errors="coerce").to_numpy(dtype=float)[:limit] for c in TC])
    ps=np.stack([np.column_stack([pd.to_numeric(df[c],errors="coerce").to_numpy(dtype=float)[:limit] for c in PC[ch]]) for ch in range(3)],axis=1) / Z.reshape(1,1,5)
    bad=float(np.mean(~np.isfinite(wl)))
    wl=np.nan_to_num(wl,nan=0,posinf=0,neginf=0)
    ps=np.nan_to_num(ps,nan=0,posinf=0,neginf=0)
    wl=np.maximum(wl,0)
    ps=np.maximum(ps,0)
    w=np.median(wl.reshape(nwin,win,3),axis=1)
    pw=np.median(ps.reshape(nwin,win,3,5),axis=1)
    power=np.sum(pw,axis=2)
    ps_frac=(pw+1)/(power[:,:,None]+5)
    acc=np.stack([pd.to_numeric(df[c],errors="coerce").to_numpy(dtype=float)[:limit] for c in AC if c in df],axis=1)
    acc=np.nan_to_num(acc)
    am=np.median(np.mean(np.abs(acc),axis=1).reshape(nwin,win),axis=1)
    pstate=np.array(pd.to_numeric(df["PointerState"],errors="coerce")[:limit],dtype=float)
    press=np.array(pd.to_numeric(df["TriggerPressure"],errors="coerce")[:limit],dtype=float)
    wm=[np.median(w[:,i]) for i in range(3)]
    wlow=[quant(w[:,i],10) for i in range(3)]
    whigh=[quant(w[:,i],90) for i in range(3)]
    powmed=[np.median(power[:,i]) for i in range(3)]
    powerlog=np.log1p(power)
    wl_log=np.log1p(w)
    features=[]
    per_channel=[]
    for i in range(3):
        a=w[:,i];p=power[:,i]
        ratio=div(quant(a,90)+1,quant(a,10)+1)
        p_ratio=div(quant(p,90)+1,quant(p,10)+1)
        median_spec=np.median(ps_frac[:,i,:],axis=0)
        entropy=-np.sum(median_spec*np.log(np.maximum(median_spec,1e-12)))/np.log(5)
        headratio=np.median((h[:,i]+1)/(tail[:,i]+1))
        stability=np.median(np.abs(np.diff(np.log1p(a)))) if len(a)>1 else 0
        corr=float(np.corrcoef(wl_log[:,i],powerlog[:,i])[0,1]) if np.std(wl_log[:,i])>1e-6 and np.std(powerlog[:,i])>1e-6 else 0.0
        vec=[np.log1p(max(wm[i],0)),np.log1p(max(wlow[i],0)),
             np.log1p(max(whigh[i],0)),np.log1p(max(powmed[i],0)),
             np.log1p(max(ratio,0)),np.log1p(max(p_ratio,0)),
             float(entropy),float(stability),float(corr),float(headratio),
             *[float(x) for x in median_spec]]
        features+=vec
        per_channel.append(vec)
    imbalances=[
      float(np.max(wm)/(min(wm)+1)),float(np.max(powmed)/(min(powmed)+1)),
      float(np.corrcoef(wl_log[:,0],wl_log[:,1])[0,1]) if np.std(wl_log[:,0])>.001 and np.std(wl_log[:,1])>.001 else 0.0,
      float(np.corrcoef(wl_log[:,1],wl_log[:,2])[0,1]) if np.std(wl_log[:,1])>.001 and np.std(wl_log[:,2])>.001 else 0.0,
      float(np.corrcoef(wl_log[:,0],wl_log[:,2])[0,1]) if np.std(wl_log[:,0])>.001 and np.std(wl_log[:,2])>.001 else 0.0]
    features += [np.log1p(max(v,0)) if j<2 else v for j,v in enumerate(imbalances)]
    class_counts=Counter(pd.to_numeric(df["class_id"],errors="coerce").dropna().astype(int).tolist())
    return {"group":group,"user":uid,"file":path.name,"n":n,"schema":len(df.columns),
      "dt":dt,"window_ms":win*dt,"duration_s":n*dt/1000,
      "timestamp_dup_fraction":float(np.mean(dif==0)) if len(dif) else 0,
      "timestamp_gap_fraction":float(np.mean(dif>100)) if len(dif) else 0,
      "invalid_wl_fraction":bad,"class_counts":class_counts,
      "wl_median":list(map(float,wm)),"wl_q90_q10":list(map(float,[div(whigh[i]+1,wlow[i]+1) for i in range(3)])),
      "power_median":list(map(float,powmed)),"power_q90_q10":[div(quant(power[:,i],90)+1,quant(power[:,i],10)+1) for i in range(3)],
      "quiet_shape":[np.median(ps_frac[:,i,:],axis=0).tolist() for i in range(3)],
      "acc_motion_p90":quant(am,90),"saturation_fraction":float(np.mean(wl>=65000)),
      "pointer_on_fraction":float(np.mean(pstate>0.5)),"pressure_on_fraction":float(np.mean(press>0.5)),
      "pointer_pressure_disagreement":float(np.mean((pstate>0.5)!=(press>0.5))),
      "features":features,"per_channel":per_channel}

def kfold_evaluation(records, labels, groupids, models):
    X=np.asarray([r for r in records],dtype=float)
    Y=np.asarray(labels,dtype=int)
    groups=np.asarray(groupids)
    if len(np.unique(groups))<5 or len(np.unique(Y))!=2:return {"error":"insufficient diversity"}
    result={}
    for name,make in models.items():
        predictions=np.zeros(len(Y),dtype=float)
        for tr,te in GroupKFold(n_splits=5).split(X,Y,groups):
            if len(np.unique(Y[tr]))<2:
                predictions[te]=np.mean(Y[tr]);continue
            model=make()
            model.fit(X[tr],Y[tr])
            predictions[te]=model.predict_proba(X[te])[:,1]
        pred=predictions>=0.5
        cm=confusion_matrix(Y,pred,labels=[0,1]).tolist()
        precision,recall,f1,_=precision_recall_fscore_support(Y,pred,average="binary",zero_division=0)
        result[name]={"n":len(Y),"users":len(np.unique(groups)),"positive":int(Y.sum()),
                      "roc_auc":float(roc_auc_score(Y,predictions)),
                      "balanced_accuracy":float(balanced_accuracy_score(Y,pred)),
                      "precision":float(precision),"recall":float(recall),"f1":float(f1),
                      "cm_0_then_1":cm}
    return result

def main():
    by=defaultdict(list);failed=[]
    for group in GROUPS:
        paths=sorted((ROOT/group).glob("*.csv.gz"))
        for ix,path in enumerate(paths):
            try:
                r=file_stats(path,group)
                if r:by[group].append(r)
                else:failed.append((group,path.name,"missing columns / too short"))
            except Exception as e:
                failed.append((group,path.name,type(e).__name__+": "+str(e)[:150]))
            if (ix+1)%150==0:print("PROGRESS",group,ix+1,"/",len(paths),flush=True)
        print("GROUP_COMPLETE",group,len(by[group]),"failures",sum(x[0]==group for x in failed),flush=True)
    print("TOTAL_ANALYZED",sum(len(v) for v in by.values()),"FAIL",len(failed),flush=True)
    print("FAIL_EXAMPLES",json.dumps(failed[:30]),flush=True)
    for g,r in by.items():
        cls=Counter()
        for x in r:cls.update(x["class_counts"])
        out={
            "files":len(r),"unique_users":len({x["user"] for x in r}),
            "class_ids":dict(cls),
            "schema":dict(Counter(x["schema"] for x in r)),
            "duration_s":summary(x["duration_s"] for x in r),
            "sample_step_ms":summary(x["dt"] for x in r),
            "missing_wl":summary(x["invalid_wl_fraction"] for x in r),
            "timestamp_dup":summary(x["timestamp_dup_fraction"] for x in r),
            "timestamp_gaps":summary(x["timestamp_gap_fraction"] for x in r),
            "wl_median_per_channel":[summary(x["wl_median"][i] for x in r) for i in range(3)],
            "power_median_per_channel":[summary(x["power_median"][i] for x in r) for i in range(3)],
            "wl_q90_q10_per_channel":[summary(x["wl_q90_q10"][i] for x in r) for i in range(3)],
            "saturation":summary(x["saturation_fraction"] for x in r),
            "pointer_pressure_disagree":summary(x["pointer_pressure_disagreement"] for x in r),
            "user_examples":list(sorted({x["user"] for x in r}))[:50]}
        print("GROUP_AUDIT",g,json.dumps(out),flush=True)
    print("USERS_CONTACT_OVERLAP",json.dumps({
       g:len({x["user"] for x in by[g]}) for g in ["CONNECTED","DISCONNECTED","LOOSE","TABLE"]
    }),flush=True)
    print("CONTACT_SHARED_USERS",json.dumps(sorted({x["user"] for x in by["CONNECTED"]}&{x["user"] for x in by["DISCONNECTED"]})),flush=True)
    models={
      "logistic":lambda:make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=.25,class_weight="balanced",max_iter=1000)),
      "forest":lambda:make_pipeline(SimpleImputer(strategy="median"),RandomForestClassifier(n_estimators=120,max_depth=4,min_samples_leaf=3,class_weight="balanced",n_jobs=1,random_state=0))
    }
    files=by["CONNECTED"]+by["DISCONNECTED"]
    y=[int(r["group"]=="DISCONNECTED") for r in files]
    contact=kfold_evaluation([r["features"] for r in files],y,[r["user"] for r in files],models)
    print("CONTACT_FILE_BINARY",json.dumps(contact),flush=True)
    # Per-channel proxy labels: all channels in connected are connected;
    # for disconnected sessions, only CH1/CH2 are provisionally disconnected.
    # CH3 was not disconnected in many such sessions: do NOT label it disconnected.
    per=[];label=[];userid=[]
    for r in files:
        for i in range(3 if r["group"]=="CONNECTED" else 2):
            per.append(r["per_channel"][i])
            label.append(int(r["group"]=="DISCONNECTED"))
            userid.append(r["user"])
    per_model=kfold_evaluation(per,label,userid,models)
    print("CONTACT_CHANNEL_PROXY",json.dumps(per_model),flush=True)
    # Distribution overlap under a single WL cutoff; not ML selection.
    for th in [750,1250,2000,3500,5000,8000,12000]:
        tp=sum(max(r["wl_median"][:2])>=th for r in by["DISCONNECTED"])
        fp=sum(max(r["wl_median"][:2])>=th for r in by["CONNECTED"])
        lp=sum(max(r["wl_median"][:2])>=th for group in ("MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS") for r in by[group])
        print("THRESHOLD_WL",json.dumps({"th":th,"disconnected_flagged":tp,
          "disconnected_total":len(by["DISCONNECTED"]),
          "connected_flagged":fp,"connected_total":len(by["CONNECTED"]),
          "longpress_flagged":lp,"longpress_total":101}),flush=True)
    # Strange files: high resting energy but low dynamic.
    rlist=[r for group in GROUPS for r in by[group]]
    items=sorted(rlist,key=lambda r:max(r["wl_median"]),reverse=True)
    print("HIGHEST_WL_EXAMPLES",json.dumps([{
       "group":r["group"],"file":r["file"],"wl_median":r["wl_median"],"wl_q90_q10":r["wl_q90_q10"],
       "power_q90_q10":r["power_q90_q10"]} for r in items[:15]]),flush=True)
    print("DATA_AUDIT_DONE",flush=True)

if __name__=="__main__": main()
