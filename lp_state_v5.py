#!/usr/bin/env python3
"""Conservative, per-channel LP pseudo-label teacher, V5 research candidate.

Key distinctions:
- One trustworthy electrode is sufficient; a bad CH1/2 does not veto CH3.
- PointerState, TriggerPressure, class_id, folder are NOT used to infer LP events.
- Rest and active evidence are local to each channel, not a three-channel vote.
- Require repeated, stable, same-channel episodes for high-confidence training.
- Inference is offline/noncausal; this teacher MUST NOT be run in firmware.
- Physical electrode connectivity / hold truth is NOT measured by these CSVs.

9 main outputs = 3 x (lp_resting, lp_high_activity, lp_signal_quality).
Scores are *evidence*, not calibrated clinical or physical probabilities.
-1 global state means UNKNOWN; never uint8-cast or map unknown to active.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.ndimage import maximum_filter1d

# Firmware encodes each spectral zone as uint16(round(10000*log10(1+sum))).
# This is not linear PSD or a per-FFT-bin mean.
BIN_COUNTS = np.array([8.,10.,12.,46.,51.])
WL = [f"p_core->wl[CH{c}]" for c in (1,2,3)]
PSD = [[f"p_core->emg_psd_mean[CH{c}][{z}]" for z in range(5)]
       for c in (1,2,3)]
REQUIRED = set(WL) | {p for cols in PSD for p in cols}


def to_numbers(df, names):
    return np.column_stack([pd.to_numeric(df[name],errors='coerce').to_numpy(float)
                            for name in names])


def sample_timing(df):
    """Detect bad timestamps or gaps. Never manufacture a 6 ms cadence."""
    for name,scale in [('MonotonicTimeUs',.001),('Timestamp',1.)]:
        if name not in df:continue
        t=pd.to_numeric(df[name],errors='coerce').to_numpy(float)
        diff=np.diff(t)*scale
        valid=diff[np.isfinite(diff)&(diff>=1.)&(diff<=60.)]
        if len(valid)<max(10,min(len(diff),100)//2):continue
        dt=float(np.median(valid))
        gaps=np.r_[False,~np.isfinite(diff)|(diff<=0)|(diff>max(3*dt,40.))]
        return dt,gaps
    return None,np.ones(len(df),bool)


def frames(ms,dt):return max(1,int(round(ms/dt)))


def dilate(mask,half):
    return maximum_filter1d(np.asarray(mask,np.uint8),size=2*half+1,
                            mode='constant')>0


def keep_long_runs(mask,minimum):
    arr=np.asarray(mask,bool)
    out=np.zeros(len(arr),bool)
    for a,b in _runs(arr):
        if b-a>=minimum:out[a:b]=True
    return out


def _runs(mask):
    arr=np.asarray(mask,bool)
    starts=np.flatnonzero(arr&~np.r_[False,arr[:-1]])
    ends=np.flatnonzero(arr&~np.r_[arr[1:],False])+1
    return list(zip(starts,ends))


def _sensor_evidence(df,dt,gaps):
    """Offline per-channel WL + logged PSD evidence, with no pointer inputs."""
    n=len(df)
    state=np.full((n,3),-1,np.int8)
    responsive=np.zeros(3,bool)
    usable=np.zeros(3,bool)
    w_raw=to_numbers(df,WL)
    p_raw=np.stack([to_numbers(df,cols) for cols in PSD],axis=1)
    if 'psd_mean_applied' in df:
        marker=pd.to_numeric(df['psd_mean_applied'],errors='coerce')
        if marker.eq(1).all():p_raw*=BIN_COUNTS.reshape(1,1,5)
        elif not marker.eq(0).all():
            return state,responsive,usable,np.zeros((n,3),bool),'mixed_psd_provenance'
    valid=np.isfinite(w_raw)&(w_raw>=0)&(w_raw<65525)&np.all(
        np.isfinite(p_raw)&(p_raw>=0)&(p_raw<65530),axis=2)
    flat=(w_raw<3)&(np.median(np.maximum(np.nan_to_num(p_raw,nan=0,posinf=0,neginf=0),0),axis=2)<3)
    valid&=~flat
    guard=dilate(gaps,frames(150.,dt))
    valid&=~guard[:,None]
    w=median_filter(np.maximum(np.nan_to_num(w_raw,nan=0,posinf=0,neginf=0),0),
                    size=(frames(75.,dt)|1,1),mode='nearest')
    s=np.median(np.maximum(np.nan_to_num(p_raw,nan=0,posinf=0,neginf=0),0),axis=2)/10000.
    s=median_filter(s,size=(frames(75.,dt)|1,1),mode='nearest')
    for ch in range(3):
        vv=valid[:,ch]
        if vv.sum()<max(100,round(n*.65)):continue
        floor_w=float(np.percentile(w[vv,ch],15))
        floor_p=float(np.percentile(s[vv,ch],15))
        high_w=float(np.percentile(w[vv,ch],90))
        high_p=float(np.percentile(s[vv,ch],90))
        if not (30<=floor_w<=3500 and floor_p<=.60):continue
        usable[ch]=True
        ratio=(w[:,ch]+25)/(floor_w+25)
        delta=s[:,ch]-floor_p
        active=keep_long_runs((ratio>=1.40)&(delta>=.012)&vv,frames(150.,dt))
        quiet=keep_long_runs((ratio<=1.23)&(delta<=.025)&vv,frames(150.,dt))
        anchor=(active.sum()>=frames(260,dt) and quiet.sum()>=frames(260,dt) and
                high_w>=1.30*max(floor_w,1) and high_p-floor_p>=.012)
        if not anchor:continue
        responsive[ch]=True
        transitions=np.r_[False,active[1:]!=active[:-1]]
        boundary=dilate(transitions,frames(125.,dt))
        state[active&~boundary,ch]=0
        state[quiet&~boundary,ch]=1
    return state,responsive,usable,valid,'ok'


VERSION = "lp_emg_repeated_channel_v5_candidate"
STRICT_BAD_PROB = 0.20
REVIEW_BAD_PROB = 0.40
MIN_EVENTS = 3
MIN_EVENT_MS = 350.
CONTROL_CONTEXTS = ("CONNECTED", "NOTHING")
CONTACT_CONTROL_CONTEXTS = ("DISCONNECTED", "LOOSE", "TABLE")


@dataclass(frozen=True)
class Settings:
    strict_bad_prob: float = STRICT_BAD_PROB
    review_bad_prob: float = REVIEW_BAD_PROB
    min_events: int = MIN_EVENTS
    min_event_ms: float = MIN_EVENT_MS


def stable_quiet_control(df, dt, good_sample):
    """Extract weak quiet-reference windows, not ground truth or an active label.

    Validity and the independently trained contact proxy gate these windows.
    The recording folder is allowed to supply low-weight rest context only.
    """
    n=len(df)
    wl=to_numbers(df,WL)
    p=np.stack([to_numbers(df,ch) for ch in PSD],axis=1)
    if "psd_mean_applied" in df:
        m=pd.to_numeric(df["psd_mean_applied"],errors="coerce")
        if m.eq(1).all():p=p*BIN_COUNTS.reshape(1,1,5)
    w=median_filter(np.maximum(np.nan_to_num(wl,nan=0,posinf=0,neginf=0),0),
                    size=(frames(91.,dt)|1,1),mode="nearest")
    s=median_filter(np.median(np.maximum(np.nan_to_num(p,nan=0,posinf=0,neginf=0),0),axis=2)/10000.,
                    size=(frames(91.,dt)|1,1),mode="nearest")
    out=np.zeros((n,3),bool)
    for ch in range(3):
        v=good_sample[:,ch]
        if v.sum()<frames(400.,dt):continue
        low_w=float(np.percentile(w[v,ch],25))
        low_p=float(np.percentile(s[v,ch],25))
        rest=v & (w[:,ch]<=1.25*low_w+30) & (s[:,ch]<=low_p+.02)
        out[:,ch]=keep_long_runs(rest,frames(200.,dt))
    return out


def label_recording(df:pd.DataFrame, bad_contact_prob,
                    group:str="", *, settings:Settings=Settings()):
    """Return a strict tri-state teacher and audit. No pointer supervision.

    bad_contact_prob[ch] must come from a *person-held-out* contact proxy
    model for training captures, or a model trained on OTHER participants for
    a new recording. Do not use in-sample quality training probabilities.
    """
    n=len(df)
    bp=np.asarray(bad_contact_prob,dtype=float)
    if bp.shape!=(3,) or not np.all(np.isfinite(bp)) or np.any((bp<0)|(bp>1)):
        raise ValueError("Provide three finite per-channel contact probabilities in [0,1]")
    if not 0. <= settings.strict_bad_prob < settings.review_bad_prob <= 1.:
        raise ValueError("Invalid quality thresholds")
    if settings.min_events<1 or settings.min_event_ms<=0:
        raise ValueError("Invalid event acceptance settings")

    labels={"lp_state":np.full(n,-1,np.int8),
            "lp_state_weight":np.zeros(n,np.float32)}
    for ch in (1,2,3):
        for name in ("resting","high_activity","signal_quality","unknown"):
            labels[f"lp_{name}_ch{ch}"]=(np.ones(n,np.float32) if name=="unknown"
                                            else np.zeros(n,np.float32))
        labels[f"lp_target_weight_ch{ch}"]=np.zeros(n,np.float32)
        labels[f"lp_ch{ch}_event_supported"]=np.zeros(n,np.int8)
    labels["lp_teacher_version"]=np.full(n,VERSION,dtype=object)
    audit={"version":VERSION,"n":n,"group":group,"contact_bad_proxy":bp.round(4).tolist(),
           "channel_events":[],"quality_model_note":
           "Weak participant-held-out connected/disconnected classifier, NOT physical impedance."}

    def finish(reason):
        y=labels["lp_state"]
        audit.update(reason=reason,active=int(np.sum(y==0)),rest=int(np.sum(y==1)),
                     unknown=int(np.sum(y==-1)),
                     labeled_fraction=float(np.mean(y>=0)) if n else 0.,
                     strict_accepted_channels=int(sum(x.get("strict_accepted",False) for x in audit["channel_events"])),
                     review_candidate_channels=int(sum(x.get("review_candidate",False) for x in audit["channel_events"])),
                     single_channel=bool(sum(x.get("strict_accepted",False) for x in audit["channel_events"])==1))
        return pd.DataFrame(labels),audit

    if n<100 or not REQUIRED.issubset(df.columns):
        return finish("missing_features_or_short")
    dt,gaps=sample_timing(df)
    if dt is None:
        return finish("bad_timestamp")
    audit["dt_ms"]=round(dt,4)

    state,responsive,usable,valid,sensor_status=_sensor_evidence(df,dt,gaps)
    if sensor_status!='ok':return finish(sensor_status)

    accepted=np.zeros(3,bool)
    event_masks=np.zeros((n,3),bool)
    quality=np.clip(1.-bp,0,1)[None,:] * valid.astype(float)
    # Signal-quality proxy is per-channel; it is NOT the inverse of every
    # transient disconnection or a verified impedance measurement.
    for ch in range(3):
        labels[f"lp_signal_quality_ch{ch+1}"]=quality[:,ch].astype(np.float32)
        row=state[:,ch]
        candidate=(row==0)&valid[:,ch]
        runs=[(a,b) for a,b in _runs(candidate)
              if (b-a)>=frames(settings.min_event_ms,dt)]
        high_n=len(runs)
        trust=bool(responsive[ch] and usable[ch] and bp[ch]<settings.strict_bad_prob)
        accepted[ch]=trust and high_n>=settings.min_events
        review=bool(not accepted[ch] and responsive[ch] and usable[ch] and
                    bp[ch]<settings.review_bad_prob and
                    high_n>=max(2,settings.min_events-1))
        for a,b in runs:
            if accepted[ch]:
                event_masks[a:b,ch]=True
        detail={"channel":ch+1,"quality_bad_proxy":round(float(bp[ch]),4),
                "baseline_usable":bool(usable[ch]),"responsive":bool(responsive[ch]),
                "event_count":int(high_n),"minimum_event_ms":settings.min_event_ms,
                "strict_accepted":bool(accepted[ch]),"review_candidate":review,
                "quality_note":"Confidence is a weak usable-signal proxy, not measured contact"}
        audit["channel_events"].append(detail)

    # Session context never manufactures an ACTIVE event. For quiet-reference
    # captures, a convincing EMG event conflicts with the weak file intent;
    # quarantine the whole file for review rather than label it as contraction.
    if group in CONTROL_CONTEXTS and accepted.any():
        audit["suspicious_control_active_events"]=True
        return finish("control_conflicts_with_sensor_event")
    # Contact-stress captures are NOT LP trials; their potentially genuine
    # CH3 activity or electrode artifacts should not train LP state as truth.
    # The contact quality scores remain available for diagnostic training.
    if group in CONTACT_CONTROL_CONTEXTS:
        return finish("contact_stress_context_unknown")

    # Candidate event shape is independent of PointerState. Do not let
    # untrusted channels vote to reject a valid CH1/CH2/CH3 contraction.
    if accepted.any():
        active=np.any(event_masks[:,accepted],axis=1)
        # Rest is a verified low-energy interval after event-family evidence
        # was observed in each accepted channel. Other CHs cannot veto it.
        rest=np.all(state[:,accepted]==1,axis=1)
        rest &= np.all(valid[:,accepted],axis=1)
        rest=keep_long_runs(rest,frames(200.,dt))
        # A concurrent accepted active event always overrides a quiet vote.
        rest &= ~active
        y=labels["lp_state"]
        y[active]=0
        y[rest]=1
        labels["lp_state_weight"][active|rest]=1.
    elif group in CONTROL_CONTEXTS:
        # Known quiet background files are WEAK context and must never become
        # a high-confidence contraction label. They can supply partial rest
        # targets with lower weight when at least one channel is plausible.
        prelim=(bp<settings.strict_bad_prob)&usable
        local_quiet=stable_quiet_control(df,dt,valid)
        for ch in range(3):
            if prelim[ch]:
                rest=local_quiet[:,ch]
                score=np.minimum(.65, quality[:,ch])
                labels[f"lp_resting_ch{ch+1}"][rest]=score[rest]
                labels[f"lp_target_weight_ch{ch+1}"][rest]=.55
                labels[f"lp_unknown_ch{ch+1}"][rest]=0.
        if np.any(prelim):
            rest=np.all(local_quiet[:,prelim],axis=1)
            rest=keep_long_runs(rest,frames(200.,dt))
            labels["lp_state"][rest]=1
            labels["lp_state_weight"][rest]=.55
        # We do not produce high-activity weak labels from a quiet folder.
        for ch in range(3):
            if not prelim[ch]:labels[f"lp_unknown_ch{ch+1}"][:]=1.
        audit["weak_quiet_reference"]=True
        return finish("weak_control_rest")

    for ch in range(3):
        ych=state[:,ch]
        valid_ch=valid[:,ch]
        if accepted[ch]:
            a=event_masks[:,ch] & valid_ch
            r=(ych==1) & valid_ch
            r=keep_long_runs(r,frames(200.,dt))
            s=np.minimum(.95,quality[:,ch])
            labels[f"lp_high_activity_ch{ch+1}"][a]=s[a]
            labels[f"lp_resting_ch{ch+1}"][r]=s[r]
            labels[f"lp_target_weight_ch{ch+1}"][a|r]=1.
            labels[f"lp_ch{ch+1}_event_supported"][a]=1
        # Unknown means no usable muscle-state label (not bad contact).
        known=(labels[f"lp_target_weight_ch{ch+1}"]>0)
        labels[f"lp_unknown_ch{ch+1}"][known]=0.

    return finish("strict_emg_events" if accepted.any() else "no_confirmed_event_family")
