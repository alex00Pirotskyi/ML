#!/usr/bin/env python3
"""Unsupervised within-recording rest/activation state discovery.

Does NOT use PointerState or folder label in fitting. Two Gaussian components
are descriptive candidate states, not physical press/rest labels.
"""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
ROOT=Path("DATA/DATA_LONG_PRESS")
GROUPS=["MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS",
        "DRAG_AND_DROP","CONNECTED","DISCONNECTED","LOOSE","TABLE","NOTHING","TAP","SWIPE"]
W=[f"p_core->wl[CH{i}]" for i in (1,2,3)]
P=[[f"p_core->emg_psd_mean[CH{i}][{z}]" for z in range(5)] for i in (1,2,3)]
COLS=set(W+sum(P,[])+["Timestamp"])
Z=np.array([8.,10.,12.,46.,51.])

def q(vals,p):
    if not vals:return None
    return float(np.percentile(vals,p))

def filedata(path):
    df=pd.read_csv(path,compression="gzip",usecols=lambda c:c in COLS)
    if len(df)<160:return None
    dif=np.diff(df["Timestamp"].to_numpy(dtype=float)[:1000])
    dif=dif[(dif>0)&(dif<100)]
    dt=float(np.median(dif)) if len(dif) else 5.
    win=int(np.clip(round(250/dt),20,150))
    nw=len(df)//win
    if nw<20:return None
    n=nw*win
    wl=np.column_stack([df[c].to_numpy(dtype=float)[:n] for c in W])
    power=np.stack([np.column_stack([df[c].to_numpy(dtype=float)[:n] for c in P[i]]) for i in range(3)],axis=1)
    power=np.sum(np.maximum(power,0)/Z.reshape(1,1,5),axis=2)
    w=np.median(np.maximum(np.nan_to_num(wl),0).reshape(nw,win,3),axis=1)
    p=np.median(np.maximum(np.nan_to_num(power),0).reshape(nw,win,3),axis=1)
    wl_floor=np.maximum(np.percentile(w,20,axis=0),1)
    ps_floor=np.maximum(np.percentile(p,20,axis=0),1)
    rw=np.log((w+1)/(wl_floor+1));rp=np.log((p+1)/(ps_floor+1))
    # Combine related channels without requiring 2 of 3 to respond.
    features=np.stack([np.median(rw,axis=1),np.max(rw,axis=1),
        np.median(rp,axis=1),np.max(rp,axis=1)],axis=1)
    features=np.nan_to_num(features)
    if not np.all(np.isfinite(features)):return None
    return features, w, p, float(dt*win/1000)

def runs(mask):
    runs=[];last=0
    for active in mask:
        if active:last+=1
        elif last:runs.append(last);last=0
    if last:runs.append(last)
    return runs

def analyze(path):
    output=filedata(path)
    if output is None:return None
    X,w,p,secs=output
    model1=GaussianMixture(n_components=1,covariance_type="diag",
            reg_covar=1e-3,random_state=18,n_init=1,max_iter=100).fit(X)
    model2=GaussianMixture(n_components=2,covariance_type="diag",
            reg_covar=1e-3,random_state=18,n_init=2,max_iter=100).fit(X)
    delta=float(model1.bic(X)-model2.bic(X))
    labels=model2.predict(X)
    # High state means higher relative muscle energy, NOT verified press.
    high_id=int(np.argmax(model2.means_[:,0]+model2.means_[:,2]))
    hi=labels==high_id
    floor=labels!=high_id
    if not np.any(hi) or not np.any(floor):return None
    occupancy=float(np.mean(hi))
    # Before claiming two distinct states, demand both substantial
    # occupation and separation in relative WL/PSD.
    mean_diff=float(np.mean(X[hi, [0,2]])-np.mean(X[floor, [0,2]]))
    credible=bool(delta>10 and .1<=occupancy<=.9 and mean_diff>np.log(1.4))
    # If high fraction occupies almost all points, "rest" calibration unknown.
    hruns=runs(hi)
    return {"bic_delta":delta,"high_frac":occupancy,"separation_log":mean_diff,
        "credible":credible,"high_dwell_s":max(hruns)*secs if hruns else 0,
        "median_high_run_s":float(np.median(hruns)*secs) if hruns else 0,
        "period_s":secs,"n_windows":len(X)}

def main():
    by=defaultdict(list);failed=defaultdict(int)
    for group in GROUPS:
        paths=sorted((ROOT/group).glob("*.csv.gz"))
        if group in ("TAP","SWIPE"):
            paths=[paths[i] for i in np.linspace(0,len(paths)-1,120,dtype=int)]
        for p in paths:
            try:
                r=analyze(p)
                if r:by[group].append(r)
                else:failed[group]+=1
            except Exception as e:
                failed[group]+=1
                print("ERROR",group,p.name,type(e).__name__,str(e)[:150],flush=True)
        print("PROGRESS",group,len(by[group]),"fail",failed[group],flush=True)
    for g,rs in by.items():
        print("GMM_GROUP",g,json.dumps({
            "n":len(rs),"credible_bistate":sum(x["credible"] for x in rs),
            "bic_delta_med":q([x["bic_delta"] for x in rs],50),
            "bic_delta_p10":q([x["bic_delta"] for x in rs],10),
            "high_occupancy_med":q([x["high_frac"] for x in rs],50),
            "high_low_log_separation_med":q([x["separation_log"] for x in rs],50),
            "high_dwell_med_s":q([x["high_dwell_s"] for x in rs],50),
            "high_dwell_p90_s":q([x["high_dwell_s"] for x in rs],90),
            "period_ms_median":1000*q([x["period_s"] for x in rs],50)
        }),flush=True)
    print("GMM_DONE",flush=True)

if __name__=="__main__":main()
