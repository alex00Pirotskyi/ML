#!/usr/bin/env python3
"""Full recording-level sustained EMG activity stress test, read-only.

IMPORTANT: Folder labels are weak recording-context labels. This script
does not measure per-timestamp press/rest truth or LP precision/recall.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json, math, re, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT=Path("DATA/DATA_LONG_PRESS")
GROUPS=["MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS",
        "DRAG_AND_DROP","CONNECTED","DISCONNECTED","LOOSE","NOTHING","TABLE","TAP","SWIPE"]
LP={"MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS"}
WC=[f"p_core->wl[CH{i}]" for i in (1,2,3)]
PC=[[f"p_core->emg_psd_mean[CH{i}][{z}]" for z in range(5)] for i in (1,2,3)]
AC=[f"p_core->acc_rate[{a}][{z}]" for a in "XYZ" for z in range(6)]
WANTED=set(WC+sum(PC,[])+["Timestamp","PointerState","TriggerPressure","User"]+AC)
COUNTS=np.array([8,10,12,46,51],dtype=float)
MODES=["first2s","oracle20"]
CONFIGS=[
 {"name":"WL_1ch_1p6_1s","wl":1.6,"psd":0,"min_channels":1,"duration":1.0},
 {"name":"WL_1ch_2p0_1s","wl":2.0,"psd":0,"min_channels":1,"duration":1.0},
 {"name":"WL_2ch_1p4_1s","wl":1.4,"psd":0,"min_channels":2,"duration":1.0},
 {"name":"WL_2ch_1p8_1s","wl":1.8,"psd":0,"min_channels":2,"duration":1.0},
 {"name":"PSD_1ch_1p8_1s","wl":0,"psd":1.8,"min_channels":1,"duration":1.0},
 {"name":"PSD_2ch_1p4_1s","wl":0,"psd":1.4,"min_channels":2,"duration":1.0},
 {"name":"BOTH_1ch_1p6_1s","wl":1.6,"psd":1.6,"min_channels":1,"duration":1.0},
 {"name":"BOTH_2ch_1p4_1s","wl":1.4,"psd":1.4,"min_channels":2,"duration":1.0},
 {"name":"BOTH_1ch_1p8_2s","wl":1.8,"psd":1.6,"min_channels":1,"duration":2.0},
 {"name":"BOTH_2ch_1p8_2s","wl":1.8,"psd":1.6,"min_channels":2,"duration":2.0},
 {"name":"BOTH_1ch_2p2_1s","wl":2.2,"psd":1.6,"min_channels":1,"duration":1.0},
 {"name":"BOTH_2ch_2p2_1s","wl":2.2,"psd":1.6,"min_channels":2,"duration":1.0},
 {"name":"WL_1ch_1p6_2s","wl":1.6,"psd":0,"min_channels":1,"duration":2.0},
 {"name":"WL_2ch_1p4_2s","wl":1.4,"psd":0,"min_channels":2,"duration":2.0},
]
ABLATIONS=[
 ("CH1", [0]),("CH2",[1]),("CH3",[2]),("CH1_CH2",[0,1]),("ALL",[0,1,2])
]

def nameid(user,stem):
    s=str(user) if pd.notna(user) and str(user).strip() else stem.split("_")[0]
    return re.sub("[^a-z0-9]+","",s.lower()) or "unknown"

def median(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else None

def pct(a,p):
    a=np.asarray([v for v in a if v is not None],dtype=float);a=a[np.isfinite(a)]
    return float(np.percentile(a,p)) if len(a) else None

def profile(path,group):
    df=pd.read_csv(path,compression="gzip",usecols=lambda c:c in WANTED)
    if len(df)<150:return None
    tt=pd.to_numeric(df["Timestamp"],errors="coerce").to_numpy(dtype=float)
    dif=np.diff(tt[:min(1200,len(tt))]);dif=dif[(dif>0)&(dif<200)]
    dt=np.median(dif) if len(dif) else 5
    win=int(np.clip(round(250/dt),20,150))
    nwin=len(df)//win
    if nwin<12:return None
    n=nwin*win
    wl=np.stack([df[c].to_numpy(dtype=float)[:n] for c in WC],axis=1)
    pw=np.stack([np.column_stack([df[c].to_numpy(dtype=float)[:n] for c in PC[ch]])
                 for ch in range(3)],axis=1)
    psd_power=np.sum(np.maximum(pw,0)/COUNTS.reshape(1,1,5),axis=2)
    wl=np.maximum(np.nan_to_num(wl,nan=0,posinf=0,neginf=0),0)
    psd_power=np.maximum(np.nan_to_num(psd_power,nan=0,posinf=0,neginf=0),0)
    w=np.median(wl.reshape(nwin,win,3),axis=1)
    p=np.median(psd_power.reshape(nwin,win,3),axis=1)
    ptr=np.median(df["PointerState"].to_numpy(dtype=float)[:n].reshape(nwin,win),axis=1)>.5
    pressure=np.median(df["TriggerPressure"].to_numpy(dtype=float)[:n].reshape(nwin,win),axis=1)>.5
    ac=np.column_stack([df[c].to_numpy(dtype=float)[:n] for c in AC])
    acc=np.median(np.mean(np.abs(np.nan_to_num(ac)),axis=1).reshape(nwin,win),axis=1)
    uid=nameid(df["User"].iloc[0] if "User" in df else None,path.stem)
    return {"group":group,"user":uid,"file":path.name,"w":w,"p":p,"ptr":ptr,"pressure":pressure,
      "acc":acc,"n":nwin,"period_s":win*dt/1000,"duration_s":nwin*win*dt/1000}

def mask_runs(x,minimum):
    min_count=max(1,int(round(minimum)))
    run=0;events=0;durations=[];start=0;mask=np.zeros(len(x),dtype=bool)
    for i,v in enumerate(list(x)+[False]):
        if v:
            if run==0:start=i
            run+=1
        else:
            if run>=min_count:
                events+=1
                durations.append(run)
                mask[start:i]=True
            run=0
    return events, mask, durations

def assess(r,config,mode,channels=None):
    w,p=r["w"],r["p"]
    chan=channels or [0,1,2]
    if mode=="first2s":
        nb=max(1,int(round(2/r["period_s"])))
        basew=np.percentile(w[:nb],20,axis=0)
        basep=np.percentile(p[:nb],20,axis=0)
    elif mode=="oracle20":
        basew=np.percentile(w,20,axis=0)
        basep=np.percentile(p,20,axis=0)
    else:raise ValueError(mode)
    rw=(w+1)/(basew[None,:]+1)
    rp=(p+1)/(basep[None,:]+1)
    if config["wl"]:
        active=rw>=config["wl"]
    else:active=np.ones_like(rw,dtype=bool)
    if config["psd"]:
        active&=(rp>=config["psd"])
    active=active[:,chan]
    raw=(np.sum(active,axis=1)>=config["min_channels"])
    nevt,held,durs=mask_runs(raw,config["duration"]/r["period_s"])
    good=np.median(basew)>0
    # Detect any stable elevated (stationary) plateau to disprove
    # "low local variance == resting".
    return {"event":int(nevt>0),"events":int(nevt),
      "active_fraction":float(np.mean(held)),
      "events_per_min":nevt/max(r["duration_s"]/60,.01),
      "mean_event_s":float(np.mean(durs)*r["period_s"]) if durs else 0,
      "outside_pointer_fraction":float(np.mean(held&(~r["ptr"]))),
      "inside_pointer_fraction":float(np.mean(held&r["ptr"])),
      "pointer_on_but_low_fraction":float(np.mean(r["ptr"]&(~raw))),
      "base_wl":basew,"base_psd":basep}

def summ(a):
    a=[v for v in a if v is not None and np.isfinite(v)]
    return {"n":len(a),"median":median(a),"p90":pct(a,90)}

def main():
    all_records=defaultdict(list);fails=[]
    for group in GROUPS:
        paths=sorted((ROOT/group).glob("*.csv.gz"))
        for ix,path in enumerate(paths):
            try:
                r=profile(path,group)
                if r:all_records[group].append(r)
                else:fails.append((group,path.name,"too_short"))
            except Exception as e:
                fails.append((group,path.name,type(e).__name__+": "+str(e)[:150]))
            if (ix+1)%200==0:print("PROGRESS",group,ix+1,"/",len(paths),flush=True)
        print("GROUP_COMPLETE",group,len(all_records[group]),flush=True)
    print("SCOPE",json.dumps({"success":{g:len(v) for g,v in all_records.items()},
      "failed":len(fails),"examples":fails[:20]}),flush=True)
    # Detector characterization: coverage not event-level sensitivity.
    for mode in MODES:
        for config in CONFIGS:
            out={}
            for group,records in all_records.items():
                values=[assess(r,config,mode) for r in records]
                out[group]={"n":len(records),"flagged":sum(x["event"] for x in values),
                  "rate":sum(x["event"] for x in values)/len(records) if records else None,
                  "events_per_min":summ([x["events_per_min"] for x in values]),
                  "active_frac":summ([x["active_fraction"] for x in values])}
            print("ACTIVITY_COMPARISON",mode,config["name"],json.dumps(out),flush=True)
    # Per-channel ablation for soft-press + idle/noise contrasts.
    cfg=next(x for x in CONFIGS if x["name"]=="BOTH_1ch_1p6_1s")
    for name,channels in ABLATIONS:
        alt={**cfg,"min_channels":1}
        out={}
        for group in ("SOFT_LONG_PRESS","STRONG_LONG_PRESS","CONNECTED","DISCONNECTED","LOOSE","TABLE","TAP","SWIPE"):
            a=[assess(r,alt,"first2s",channels) for r in all_records[group]]
            out[group]=[sum(x["event"] for x in a),len(a)]
        print("CHANNEL_ABLATION",name,json.dumps(out),flush=True)
    # Minimum-duration sweep for repeated high activity (temporal discriminator)
    for duration in [.25,.5,1.,1.5,2.,3.,4.]:
        conf={"wl":1.6,"psd":1.4,"min_channels":1,"duration":duration}
        out={}
        for group in ("MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS","CONNECTED","NOTHING","TABLE","TAP","SWIPE"):
            a=[assess(r,conf,"first2s") for r in all_records[group]]
            out[group]=[sum(x["event"] for x in a),len(a)]
        print("DURATION_SWEEP",duration,json.dumps(out),flush=True)
    # Held-out-person selection of threshold on weak LP-vs-CONNECTED context.
    positive=sum((all_records[g] for g in LP),[])
    controls=all_records["CONNECTED"]
    samples=positive+controls
    labels=np.array([1]*len(positive)+[0]*len(controls))
    users=np.array([r["user"] for r in samples])
    models=[]
    for config in CONFIGS:
        if config["name"] in ["BOTH_1ch_1p6_1s","BOTH_2ch_1p4_1s","BOTH_1ch_1p8_2s",
                              "WL_1ch_1p6_1s","WL_2ch_1p4_1s","PSD_1ch_1p8_1s"]:
            models.append(config)
    probs={c["name"]:np.array([assess(r,c,"first2s")["event"] for r in samples]) for c in models}
    cvpred=np.zeros_like(labels);choice=[]
    for tr,te in GroupKFold(n_splits=5).split(np.zeros(len(samples)),labels,users):
        scored=[]
        for c in models:
            pred=probs[c["name"]][tr]
            if np.any(labels[tr]==1) and np.any(labels[tr]==0):
                bal=balanced_accuracy_score(labels[tr],pred)
                scored.append((bal,c["name"]))
        _,winner=max(scored)
        choice.append({"selected":winner,"train_balacc":float(max(scored)[0]),
             "test_users":len(set(users[te])),"test_positive":int(labels[te].sum()),
             "test_controls":int((1-labels[te]).sum())})
        cvpred[te]=probs[winner][te]
    print("HELDOUT_CONTEXT_MODEL",json.dumps({"n":len(samples),"people":len(set(users)),
      "follows_groupkfold":5,"balanced_accuracy":float(balanced_accuracy_score(labels,cvpred)),
      "confusion":confusion_matrix(labels,cvpred,labels=[0,1]).tolist(),"folds":choice}),flush=True)
    # Static elevated plateau: local stability is not rest.
    plateau={}
    for g,records in all_records.items():
        count=0;fraction=[]
        for r in records:
            w=r["w"]
            base=np.maximum(np.percentile(w,20,axis=0),1)
            ratio=w/base
            k=max(2,round(1/r["period_s"]))
            high=np.any(ratio>1.8,axis=1)
            stable=np.zeros(len(high),dtype=bool)
            for i in range(k-1,len(high)):
                window=w[i-k+1:i+1]
                cvar=(np.max(window,axis=0)+1)/(np.min(window,axis=0)+1)
                stable[i]=bool(np.min(cvar)<1.15) # at least one stable channel
            candidate=high&stable
            has=bool(np.any(candidate))
            count+=has
            fraction.append(float(np.mean(candidate)))
        plateau[g]={"n":len(records),"with_stable_high":count,"median_fraction":median(fraction)}
    print("STABLE_HIGH_PLATEAUS",json.dumps(plateau),flush=True)
    print("TEMPORAL_EXPERIMENT_DONE",flush=True)

if __name__=="__main__":main()
