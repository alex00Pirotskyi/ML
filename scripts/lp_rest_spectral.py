#!/usr/bin/env python3
"""Spectral and channel-level audit, no PointerState labels or model training."""
from pathlib import Path
import numpy as np
import pandas as pd
import json

ROOT=Path("DATA/DATA_LONG_PRESS")
GROUPS=["MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS",
        "CONNECTED","DISCONNECTED","LOOSE","TABLE","NOTHING","TAP","SWIPE"]
Z=np.array([8,10,12,46,51],dtype=float)
WL=[f"p_core->wl[CH{c}]" for c in (1,2,3)]
PS=[[f"p_core->emg_psd_mean[CH{c}][{z}]" for z in range(5)] for c in (1,2,3)]
COLS=WL+sum(PS,[])+["Timestamp"]

def percent(values,p):
    vals=[float(x) for x in values if x is not None and np.isfinite(x)]
    return round(float(np.percentile(vals,p)),4) if vals else None

def profile(path):
    df=pd.read_csv(path,compression="gzip",usecols=COLS)
    if len(df)<150:return None
    t=df["Timestamp"].to_numpy(dtype=float)
    dif=np.diff(t[:min(len(t),1200)]);dif=dif[(dif>0)&(dif<100)]
    dt=np.median(dif) if len(dif) else 5
    wlen=int(np.clip(round(500/dt),32,200))
    nw=len(df)//wlen
    if nw<5:return None
    n=nw*wlen
    wl=np.stack([np.median(df[c].to_numpy(dtype=float)[:n].reshape(nw,wlen),axis=1) for c in WL],axis=1)
    ps=np.zeros((nw,3,5))
    for ch in range(3):
        for z in range(5):
            ps[:,ch,z]=np.median(df[PS[ch][z]].to_numpy(dtype=float)[:n].reshape(nw,wlen),axis=1)/Z[z]
    ps=np.maximum(ps,0)
    wl=np.maximum(wl,0)
    baseline=np.maximum(np.quantile(wl,.2,axis=0),1)
    score=np.median(np.log1p(wl/baseline),axis=1)
    quiet=score<=np.percentile(score,20)
    active=score>=np.percentile(score,80)
    if min(np.sum(quiet),np.sum(active))<2:return None
    low=np.median(ps[quiet],axis=0)
    high=np.median(ps[active],axis=0)
    ratio=(high+1)/(low+1)
    # PSD power and spectral shape by channel and by zone.
    power=np.sum(ps,axis=2)
    qpow=np.median(power[quiet],axis=0)
    hpow=np.median(power[active],axis=0)
    power_gain=(hpow+1)/(qpow+1)
    zone_fraction_quiet=(low+1)/(np.sum(low,axis=1,keepdims=True)+5)
    zone_fraction_active=(high+1)/(np.sum(high,axis=1,keepdims=True)+5)
    spec_drift=np.sum(np.abs(zone_fraction_active-zone_fraction_quiet),axis=1)/2
    corrs=[]
    for ch in range(3):
        x=np.log1p(wl[:,ch]);y=np.log1p(power[:,ch])
        corrs.append(float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-8 and np.std(y)>1e-8 else 0)
    return {
        "ratio_per_zone":ratio.tolist(),
        "power_gain":power_gain.tolist(),
        "spec_drift":spec_drift.tolist(),
        "correlations":corrs,
        "quiet_zone_fractions":zone_fraction_quiet.tolist(),
        "baseline_wl":baseline.tolist(),
        "stable_psd_power":((np.quantile(power,.9,axis=0)+1)/(np.quantile(power,.1,axis=0)+1)).tolist()
    }

def main():
    all_results={}
    for group in GROUPS:
        paths=sorted((ROOT/group).rglob("*.csv.gz"))
        if group in {"TAP","SWIPE"} and len(paths)>80:
            paths=[paths[i] for i in np.linspace(0,len(paths)-1,80,dtype=int)]
        records=[]
        for p in paths:
            try:
                r=profile(p)
                if r:records.append(r)
            except Exception as e:print("ERROR",group,p.name,type(e).__name__,str(e)[:100],flush=True)
        all_results[group]=records
        print("PROGRESS",group,len(records),flush=True)

    for group,records in all_results.items():
        if not records:continue
        key=lambda k,ch: [r[k][ch] for r in records]
        gain_by_zone=[[percent([r["ratio_per_zone"][ch][z] for r in records],50)
                         for z in range(5)] for ch in range(3)]
        quiet_shape=[[percent([r["quiet_zone_fractions"][ch][z] for r in records],50)
                         for z in range(5)] for ch in range(3)]
        out={
            "n":len(records),
            "power_gain_channel_medians":[percent(key("power_gain",ch),50) for ch in range(3)],
            "psd_gain_each_zone_medians":gain_by_zone,
            "spectral_shape_drift_channel_medians":[percent(key("spec_drift",ch),50) for ch in range(3)],
            "log_wl_psd_corr_channel_medians":[percent(key("correlations",ch),50) for ch in range(3)],
            "quiet_zone_fractions":quiet_shape,
            "power_q90_q10_medians":[percent(key("stable_psd_power",ch),50) for ch in range(3)],
            "fraction_one_channel_2x_power":round(sum(sum(x>=2 for x in r["power_gain"])>=1 for r in records)/len(records),3),
            "fraction_two_channel_2x_power":round(sum(sum(x>=2 for x in r["power_gain"])>=2 for r in records)/len(records),3),
            "fraction_no_channel_1p5x_power":round(sum(max(r["power_gain"])<1.5 for r in records)/len(records),3)
        }
        print("SPECTRAL_"+group, json.dumps(out),flush=True)
    print("SPECTRAL_DONE")

if __name__=="__main__":main()
