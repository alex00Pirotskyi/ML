#!/usr/bin/env python3
"""Causal, firmware-aware EMG pseudo-label teacher v2.

IMPORTANT: The firmware records LOG10 power-zone sums multiplied by 10000
and stores a 15-SNC-frame temporal mean of those encoded values.
EMG_PSD_MEAN=0 disables bin-count averaging, NOT temporal averaging.
All baseline/state estimates below depend on the current and past rows only.
"""
import math
import numpy as np
import pandas as pd

VERSION="lp_causal_fw_v2"
BIN_COUNTS=np.asarray([8.,10.,12.,46.,51.])
CHANNELS=("CH1","CH2","CH3")
WL=[f"p_core->wl[{ch}]" for ch in CHANNELS]
PSD=[[f"p_core->emg_psd_mean[{ch}][{z}]" for z in range(5)]
     for ch in CHANNELS]

def ramp(x, lo, hi):
    v=np.clip((np.asarray(x,dtype=float)-lo)/(hi-lo),0.,1.)
    return v*v*(3.-2.*v)

def col(df,key):
    return pd.to_numeric(df[key],errors="coerce").to_numpy(float)

def step_ms(df):
    # First 32 samples only. Labels are UNKNOWN until after this bootstrap.
    for name,mult in (("MonotonicTimeUs",.001),("Timestamp",1.)):
        if name in df:
            dt=np.diff(col(df,name)[:32])*mult
            dt=dt[np.isfinite(dt)&(dt>.2)&(dt<=80)]
            if len(dt)>=5:return float(np.median(dt))
    return 6.0

def roll(x,n,quantile=None):
    o=pd.DataFrame(x).rolling(max(1,int(n)),min_periods=1)
    return (o.median() if quantile is None else o.quantile(quantile)).to_numpy()

def psd_log_index(df,mode="auto"):
    """Log-domain PSD index. Never sum logged codes as linear power."""
    if mode not in ("auto","raw","legacy-divided"):
        raise ValueError("psd_mode must be auto/raw/legacy-divided")
    n=len(df)
    matrix=np.zeros((n,3,5),dtype=float)
    present=np.zeros(3,dtype=bool)
    invalid=np.ones((n,3),dtype=bool)
    for ch in range(3):
        if all(k in df for k in PSD[ch]):
            source=np.column_stack([col(df,k) for k in PSD[ch]])
            matrix[:,ch,:]=source
            invalid[:,ch]=~np.all(np.isfinite(source)&(source>=0),axis=1)
            present[ch]=True
    marker=None
    if "psd_mean_applied" in df:
        flags=pd.to_numeric(df["psd_mean_applied"],errors="coerce")
        if flags.eq(1).all():marker=True
        elif flags.eq(0).all():marker=False
        else:raise ValueError("Invalid/mixed psd_mean_applied marker")
    undo=(mode=="legacy-divided" or (mode=="auto" and marker is True))
    if undo:
        # Original psd_mean.py divided *already logged* uint16 features.
        # This multiplication is approximate due to previous rounding.
        matrix*=BIN_COUNTS.reshape(1,1,5)
    clipped=np.any(matrix>=65530,axis=2)
    matrix=np.maximum(np.nan_to_num(matrix,nan=0.,posinf=65535.,neginf=0.),0.)
    matrix=np.minimum(matrix,65535.)
    # Median across five logged frequency zones is a robust band-index.
    # Temporal mean of 15 history frames already took place on the FW.
    index=np.median(matrix,axis=2)
    meta={"psd_mode":mode,"legacy_undo_applied":undo,
          "psd_flag_present":marker is not None,
          "unmarked_assumed_raw":marker is None and mode=="auto",
          "psd_semantics":"median_five_encoded_log10_zone_sums",
          "firmware_emg_bin_average":0,"firmware_emg_history_mean":15}
    return index,present,invalid,clipped,meta

def compute_causal_labels(df,psd_mode="auto"):
    if len(df)==0:raise ValueError("Empty file")
    for name in WL:
        if name not in df:raise ValueError("Missing "+name)
    n=len(df);dt=step_ms(df)
    fast_n=max(4,min(160,round(120./dt)))
    psd_n=max(4,min(160,round(90./dt)))
    slow_n=max(fast_n+2,min(600,round(1800./dt)))
    cal_n=max(20,round(650./dt))
    rest_n=max(2,round(180./dt))
    start_n=max(33,round(450./dt))
    source=np.column_stack([col(df,k) for k in WL])
    valid=np.isfinite(source)&(source>=0)
    wl=np.maximum(np.nan_to_num(source,nan=0,posinf=0,neginf=0),0.)
    p,present,invalid_psd,sat_psd,meta=psd_log_index(df,psd_mode)
    fast=roll(wl,fast_n)
    spec=roll(p,psd_n)
    slow=roll(fast,slow_n)
    lo=roll(fast,slow_n,.1)
    hi=roll(fast,slow_n,.9)
    dyn=(hi+1.)/(lo+1.)
    flo_wl=np.full(3,np.nan);flo_psd=np.full(3,np.nan)
    trial_wl=np.full(3,np.inf);trial_psd=np.full(3,np.inf)
    streak=np.zeros(3,dtype=np.int32)
    ready=np.zeros(3,dtype=bool)
    last_response=np.full(3,1e7)
    rest=np.zeros((n,3),np.float32)
    active=np.zeros((n,3),np.float32)
    bad=np.zeros((n,3),np.float32)
    unknown=np.zeros((n,3),np.float32)
    ready_tr=np.zeros((n,3),np.uint8)
    state=np.zeros(n,np.uint8)
    weight=np.zeros(n,np.float32)
    count_rest=0
    # Hypothesis values, not calibrated probabilities.
    low_wl=math.log(1.06);high_wl=math.log(1.90)
    low_psd=math.log10(1.07);high_psd=math.log10(1.90)
    alpha_down=1-math.exp(-dt/700.)
    alpha_up=1-math.exp(-dt/60000.)
    for t in range(n):
        for ch in range(3):
            w=float(fast[t,ch]); pv=float(spec[t,ch])
            history=t>=start_n
            good=bool(valid[t,ch] and present[ch] and not invalid_psd[t,ch])
            clipped=bool(w>=65530 or sat_psd[t,ch])
            last_response[ch]+=dt
            static=float(ramp(slow[t,ch],3500.,8000.))
            flat=float(1-ramp(dyn[t,ch],1.16,1.70))
            # No baseline and high stationary energy might be sustained
            # true muscle activity. Delay any strong contact-fault claim
            # until 5 s without usable modulation; until then UNKNOWN.
            enough_artifact_history=ready[ch] or (t*dt>=5000.)
            dis=float(.98*static*flat) if history and enough_artifact_history else 0.
            if ready[ch] and last_response[ch]<4000:
                dis*=.22
            if w<3 and pv<3 and history:
                dis=max(dis,.97)
            if clipped:dis=max(dis,.25)
            if not good:dis=0.
            dis=float(np.clip(dis,0,1))
            unclear=max(float(.85*(1-ramp(w,65.,230.))),
                        .40 if clipped else 0.,
                        1. if not good else 0.)
            activity=0.
            if not ready[ch]:
                # A start-during-contraction may remain uncalibrated.
                # Never label high stationary noise as reliable rest.
                candidate=(good and history and w>=50 and w<3800
                           and dyn[t,ch]<1.55 and dis<.35 and not clipped)
                if candidate:
                    streak[ch]+=1
                    trial_wl[ch]=min(trial_wl[ch],w)
                    trial_psd[ch]=min(trial_psd[ch],pv)
                else:
                    streak[ch]=0
                    trial_wl[ch]=np.inf
                    trial_psd[ch]=np.inf
                if streak[ch]>=cal_n:
                    ready[ch]=True
                    flo_wl[ch]=max(trial_wl[ch],1.)
                    flo_psd[ch]=max(trial_psd[ch],0.)
            if ready[ch]:
                wr=(w+25)/(float(flo_wl[ch])+25)
                # Difference of encoded log10-power values / 10000
                # approximates log10 power ratio, unlike value ratios.
                pdiff=(pv-float(flo_psd[ch]))/10000.
                aw=float(ramp(math.log(max(wr,1e-8)),low_wl,high_wl))
                ap=float(ramp(pdiff,low_psd,high_psd))
                activity=float(np.clip(.76*aw+.24*ap,0,1))
                if activity>.48:
                    last_response[ch]=0.
                # Freeze floor on elevated energy; upward drift is very slow.
                if (activity<.12 and good and dis<.2 and
                    unclear<.25 and wr<1.18):
                    alpha=alpha_down if w<float(flo_wl[ch]) else alpha_up
                    flo_wl[ch]+=(w-flo_wl[ch])*alpha
                    alpha=alpha_down if pv<float(flo_psd[ch]) else alpha_up
                    flo_psd[ch]+=(pv-flo_psd[ch])*alpha
            else:
                unclear=1.
            unclear=float(np.clip(unclear,0,1))*(1-dis)
            confidence=max(0.,1-dis-unclear)
            rest[t,ch]=confidence*(1-activity)
            active[t,ch]=confidence*activity
            bad[t,ch]=dis
            unknown[t,ch]=unclear
            ready_tr[t,ch]=int(ready[ch])
        min_rest=float(np.min(rest[t]))
        max_unknown=float(np.max(unknown[t]))
        max_bad=float(np.max(bad[t]))
        candidate=(min_rest>=.78 and max_unknown<.2
                   and max_bad<.3 and bool(np.all(ready)))
        count_rest=count_rest+1 if candidate else 0
        state[t]=int(count_rest>=rest_n)
        if state[t]:
            weight[t]=min_rest*(1-max_unknown)
        elif max_unknown>.40 or not np.all(ready):
            weight[t]=.02
        else:
            weight[t]=float(np.clip(
                (max(float(np.max(active[t])),max_bad)-.12)*1.05,.05,1.))
    labels={}
    for ch in range(3):
        for name,m in (("resting",rest),("high_activity",active),
                       ("disconnection",bad),("unknown",unknown)):
            labels[f"lp_{name}_ch{ch+1}"]=m[:,ch]
    labels["lp_state"]=state
    labels["lp_state_weight"]=weight
    out=pd.DataFrame(labels)
    out["lp_teacher_version"]=VERSION
    stats={"rows":n,"version":VERSION,"causal":True,
      "sample_ms":round(dt,3),
      "lp_state_rest_fraction":round(float(np.mean(state)),5),
      "uncertain_fraction":round(float(np.mean(np.max(unknown,axis=1)>.40)),5),
      "calibrated_fraction_by_channel":[round(float(v),4) for v in np.mean(ready_tr,axis=0)],
      "mean_disconnection_by_channel":[round(float(v),4) for v in np.mean(bad,axis=0)],
      "mean_high_activity_by_channel":[round(float(v),4) for v in np.mean(active,axis=0)],
      **meta}
    return out,stats
