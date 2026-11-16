#!/usr/bin/env python3
"""LP V3 offline physics-informed pseudo-label teacher.

Core principle: find low/high *within a recording* using multi-feature
bidirectional temporal evidence, then gate on independent contact evidence.
PointerState, TriggerPressure and class_id NEVER provide labels.

This is a WEAK-SUPERVISION system. There is NO physical ground-truth
press/release or electrode-level impedance in these CSVs.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.ndimage import median_filter, maximum_filter1d

from lp_causal_fw_teacher import (
    BIN_COUNTS, PSD, WL, col, psd_log_index, step_ms
)

VERSION="lp_offline_bistate_contact_v3"
RAW_PSD=[[f"p_core->emg_psd[0][CH{c}][{z}]" for z in range(5)]
         for c in (1,2,3)]
HEAD=[f"p_core->wl_head[CH{c}]" for c in (1,2,3)]
TAIL=[f"p_core->wl_tail[CH{c}]" for c in (1,2,3)]
ACC=[f"p_core->acc_rate[{ax}][{z}]" for ax in "XYZ" for z in range(6)]

def ramp(x,lo,hi):
    t=np.clip((np.asarray(x,dtype=float)-lo)/(hi-lo),0.,1.)
    return t*t*(3.-2.*t)

def values(df,names,shape=None):
    if not all(n in df for n in names):
        return None
    v=np.column_stack([col(df,n) for n in names])
    if shape is not None:v=v.reshape(len(df),*shape)
    return v

def compress(v,k,mode="median"):
    rows=(len(v)+k-1)//k
    out=[]
    for i in range(rows):
        a=v[i*k:min((i+1)*k,len(v))]
        out.append(np.nanmedian(a,axis=0) if mode=="median" else np.nanmean(a,axis=0))
    return np.asarray(out,float)

def expand(v,n,k):
    return np.repeat(v,k,axis=0)[:n]

def transitions(mask,width=1):
    x=np.asarray(mask,bool)
    change=np.zeros(len(x),dtype=bool)
    change[1:]=x[1:]!=x[:-1]
    return maximum_filter1d(change.astype(np.uint8),size=2*width+1,
                            mode="nearest")>0

def two_state(logwl,psdlog):
    """One-vs-two state model, with BIC and physical concordance gates."""
    arr=np.column_stack([logwl,psdlog])
    result={"credible":False,"weak":False,"bic":0.,"wl_gap":0.,
            "psd_gap":0.,"frac_high":0.,"floor_logwl":float(np.quantile(logwl,.15)),
            "floor_psd":float(np.quantile(psdlog,.15)),
            "p_high":np.full(len(arr),0.5)}
    if len(arr)<14 or not np.all(np.isfinite(arr)):return result
    try:
        one=GaussianMixture(n_components=1,covariance_type="diag",
            reg_covar=.007,random_state=17,n_init=1,max_iter=120).fit(arr)
        two=GaussianMixture(n_components=2,covariance_type="diag",
            reg_covar=.007,random_state=17,n_init=2,max_iter=120).fit(arr)
        hi=int(np.argmax(two.means_[:,0]));lo=1-hi
        dw=float(two.means_[hi,0]-two.means_[lo,0])
        dp=float(two.means_[hi,1]-two.means_[lo,1])
        p=two.predict_proba(arr)[:,hi]
        frac=float(np.mean(p>.5))
        bic=float(one.bic(arr)-two.bic(arr))
        weak=bool(bic>10 and .07<=frac<=.93 and dw>math.log(1.18) and dp>.008)
        credible=bool(bic>10 and .09<=frac<=.91 and dw>math.log(1.40) and dp>.018)
        result.update(credible=credible,weak=weak,bic=bic,wl_gap=dw,
            psd_gap=dp,frac_high=frac,p_high=p,
            floor_logwl=float(two.means_[lo,0]),
            floor_psd=float(two.means_[lo,1]))
    except (ValueError,ArithmeticError,np.linalg.LinAlgError):
        pass
    return result

def label_recording(df:pd.DataFrame,contact_prob=None,good_reference_wl=None,
                    psd_mode="auto",boundary_frames=1):
    """Label one complete recording using future context deliberately.

    Contact scores are supplied by a held-out-person/weakly supervised
    model; fallback artifact rules used if no contact model.
    """
    n=len(df)
    if n==0:raise ValueError("Empty recording")
    raw=values(df,WL)
    if raw is None:raise ValueError("Need all three WL channels")
    badraw=~np.isfinite(raw)|(raw<0)
    wl=np.maximum(np.nan_to_num(raw,nan=0.,posinf=0.,neginf=0.),0.)
    dt=step_ms(df)
    k=max(8,min(100,round(180/dt)))
    nframe=(n+k-1)//k
    W=compress(wl,k)
    Wlog=np.log1p(W)
    pindex,present,invalid_psd,saturated,psdmeta=psd_log_index(df,psd_mode)
    P=compress(pindex/10000.,k)
    # The current PSD frame may detect fast changes obscured by
    # the firmware's 15-frame temporal mean.
    pinstant=values(df,[x for group in RAW_PSD for x in group],(3,5))
    if pinstant is not None:
        marker=("psd_mean_applied" in df and
                pd.to_numeric(df["psd_mean_applied"],errors="coerce").eq(1).all())
        if psd_mode=="legacy-divided" or (psd_mode=="auto" and marker):
            pinstant=pinstant*BIN_COUNTS.reshape(1,1,5)
        pinstant=np.minimum(np.maximum(np.nan_to_num(pinstant,nan=0),0),65535)
        PI=compress(np.median(pinstant,axis=2)/10000.,k)
    else:
        PI=P.copy()
    head=values(df,HEAD)
    tail=values(df,TAIL)
    if head is not None and tail is not None:
        hr=compress((np.maximum(head,0)+1)/(np.maximum(tail,0)+1),k)
    else:hr=np.ones_like(W)
    accel=values(df,ACC)
    if accel is not None:
        acc=compress(np.mean(np.log1p(np.maximum(accel,0)),axis=1),k)
        delta_acc=np.r_[0.,np.abs(np.diff(acc))]
        high_motion=delta_acc>max(.08,float(np.percentile(delta_acc,90)))
    else:
        high_motion=np.zeros(nframe,dtype=bool)
    if contact_prob is None:contact_prob=[.35]*3
    if good_reference_wl is None:good_reference_wl=[1800.]*3
    contact_prob=np.clip(np.asarray(contact_prob,float),0,1)
    good_reference_wl=np.asarray(good_reference_wl,float)

    r=np.zeros((nframe,3),float)
    a=np.zeros_like(r);bad=np.zeros_like(r);unk=np.zeros_like(r)
    descriptions=[]
    for ch in range(3):
        lw=Wlog[:,ch]
        ps=P[:,ch]
        dyn=(np.percentile(W[:,ch],90)+1)/(np.percentile(W[:,ch],10)+1)
        wlmedian=float(np.median(W[:,ch]))
        spec_median=float(np.median(ps))
        desc=two_state(lw,ps)
        credible=desc["credible"];weak=desc["weak"]
        floor_w=math.expm1(desc["floor_logwl"])
        physical_wl_low=(wlmedian<=max(300.,float(good_reference_wl[ch])*1.35))
        # Representative noise indicators on the WHOLE recording (offline).
        severe_static=(wlmedian>5000 and dyn<1.5 and not credible)
        hard_flat=(wlmedian<3 and spec_median<.0003)
        # Excellent evidence of a physical contraction transition
        # cannot independently prove GOOD electrode contact. Use it to
        # abstain rather than assert disconnection if the contact proxy
        # is high on a responsive channel.
        responsive=credible or (weak and desc["wl_gap"]>math.log(1.28))
        cp=float(contact_prob[ch])
        if severe_static:
            bad_global=max(.95,float(.92*ramp(cp,.4,.9)))
        elif hard_flat:
            bad_global=.97
        elif responsive:
            bad_global=float(.20*ramp(cp,.65,.99))
        else:
            bad_global=float(.95*ramp(cp,.33,.90))
        # A nearly silent electrode may be relaxed or missing contact.
        # The context classifier cannot resolve that without impedance;
        # prefer UNKNOWN over a high-confidence disconnect claim.
        if wlmedian<40 and not hard_flat:
            bad_global*=.12
        if not present[ch]:
            bad_global=0.
        base_unknown=float(.05 + .60*ramp(cp,.24,.70))
        if responsive and cp>.4:
            base_unknown=max(base_unknown,.25+.30*ramp(cp,.45,.85))
        if not physical_wl_low and not responsive:
            base_unknown=max(base_unknown,.70)
        if wlmedian<40 and not hard_flat:
            base_unknown=max(base_unknown,.80)
        if not present[ch]:
            base_unknown=1.

        if credible or weak:
            high=desc["p_high"].copy()
            # Posterior evidence with amplitude baseline support;
            # posterior has temporal direction but no pointer input.
            log_ratio=lw-desc["floor_logwl"]
            pdiff=ps-desc["floor_psd"]
            physics=.72*ramp(log_ratio,math.log(1.06),math.log(1.9)) \
                    +.28*ramp(pdiff,.005,.25)
            agree=.72 if credible else .42
            high=agree*high+(1-agree)*physics
            high=median_filter(high,size=3,mode="nearest")
            near_boundaries=transitions(high>.5,boundary_frames)
            uncertainty=np.maximum(
                .07 if credible else .16,
                .90*ramp(.22-np.abs(high-.5),.02,.22)
            )
            uncertainty=np.maximum(uncertainty,
                np.where(near_boundaries,.64 if credible else .76,0))
        else:
            # Unimodal low can be rest only if its level resembles
            # healthy connected references. High static/unimodal signals
            # remain UNKNOWN / contact-fault, never automatically rest.
            if physical_wl_low and cp<.4 and present[ch]:
                high=.06+.10*ramp(lw-np.median(lw),.06,.4)
                uncertainty=np.full(nframe,.08)
            else:
                high=np.full(nframe,.15)
                uncertainty=np.full(nframe,.75)

        # Movement artifacts are not always motion; only suppress
        # abrupt EMG changes accompanied by high accelerometer change.
        derivative=np.r_[0.,np.abs(np.diff(lw))]
        motion_suspect=high_motion & (
            derivative>max(.15,float(np.percentile(derivative,85))))
        uncertainty=np.maximum(uncertainty,
            np.where(motion_suspect,.68,0))
        # Big within-frame half-window WL mismatch can arise from a
        # rapid transition or artifact; do not treat this alone as bad contact.
        asym=np.maximum(hr[:,ch],1/np.maximum(hr[:,ch],.01))
        uncertainty=np.maximum(uncertainty,
            .35*ramp(asym,2,6))
        wl_clipped=W[:,ch]>=65525
        invalid=compress((badraw[:,ch]|invalid_psd[:,ch]),k,mode="mean")>.25
        sat=compress(saturated[:,ch].astype(float),k,mode="mean")>.1
        uncertainty=np.maximum(uncertainty,np.where(invalid,1.,0))
        uncertainty=np.maximum(uncertainty,np.where(sat|wl_clipped,.6,0))
        # Contact quality is a whole-recording inference, but severe
        # glitch episodes are additionally downweighted locally.
        low_signal=(W[:,ch]<5)
        bad_local=np.where(low_signal & (P[:,ch]<.001),.97,0.)
        dis=np.maximum(bad_global,bad_local)
        unknown=np.clip(np.maximum(uncertainty,base_unknown),0,1)*(1-dis)
        trust=np.maximum(0.,1-dis-unknown)
        r[:,ch]=trust*(1-high)
        a[:,ch]=trust*high
        bad[:,ch]=dis
        unk[:,ch]=unknown
        desc.update(contact_probability=cp,
            severe_static=bool(severe_static),quiet_uncertain=bool(wlmedian<40),
            reference_wl_p95=float(good_reference_wl[ch]),
            mean_rest=float(np.mean(r[:,ch])),
            mean_active=float(np.mean(a[:,ch])),
            mean_disconnection=float(np.mean(bad[:,ch])),
            mean_unknown=float(np.mean(unk[:,ch])))
        descriptions.append(desc)

    # Time-border uncertainty prevents model learning false sharp onset
    # timestamps, especially when timestamped pointer state is not accurate.
    min_rest=np.min(r,axis=1)
    max_unknown=np.max(unk,axis=1)
    max_bad=np.max(bad,axis=1)
    rest_candidates=(min_rest>=.74)&(max_unknown<.27)&(max_bad<.24)
    # Require at least two consecutive ~180ms frames to be confident.
    keep=np.zeros(nframe,dtype=bool)
    cnt=0
    for t,v in enumerate(rest_candidates):
        cnt=cnt+1 if v else 0
        if cnt>=2:keep[t]=True
    state=expand(keep.astype(np.uint8),n,k)
    R=expand(r,n,k);A=expand(a,n,k);B=expand(bad,n,k);U=expand(unk,n,k)
    low_confi=np.max(U,axis=1)
    confident_active=np.max(A,axis=1)
    confident_bad=np.max(B,axis=1)
    weight=np.where(state==1,
        np.min(R,axis=1),
        np.maximum(confident_active,confident_bad))
    # Ambiguous zero is not a verified active muscle sample.
    weight=np.where((state==0)&(low_confi>.4),.02,weight)
    weight=np.clip(weight,0,1)
    labels={}
    for ch in range(3):
        for name,m in (("resting",R),("high_activity",A),
                        ("disconnection",B),("unknown",U)):
            labels[f"lp_{name}_ch{ch+1}"]=np.clip(m[:,ch],0,1)
    labels["lp_state"]=state
    labels["lp_state_weight"]=weight
    result=pd.DataFrame(labels)
    result["lp_teacher_version"]=VERSION
    stats={"version":VERSION,"rows":n,"frame_ms":k*dt,
       "causal":False,"offline_bidirectional":True,
       "lp_state_rest_fraction":float(np.mean(state)),
       "uncertain_fraction":float(np.mean(low_confi>.4)),
       "mean_label_weight":float(np.mean(weight)),
       "psd_raw_slot0_used":pinstant is not None,
       "acc_motion_features_used":accel is not None,
       "channel_details":descriptions,
       **psdmeta}
    return result,stats
