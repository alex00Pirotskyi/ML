#!/usr/bin/env python3
"""Experimental LP V4.2 teacher: partial electrode contact, pointer-independent.

Purpose: compare against strict V4.1 without modifying it or the v20 baseline.
Per-channel evidence uses WL + *log-coded* PSD, recording-local low floor and
temporal duration. Accept a single responsive channel, abstain on ambiguity.

PointerState is used ONLY for a diagnostic, never to generate or veto labels:
random time offset, FP and FN are expected. No output is ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from lp_state_v4 import (BIN_COUNTS, PSD, WL, VERSION as BASE_VERSION,
                         dilate, frames, keep_long_runs, parse_pointer,
                         sample_timing, to_numbers)

VERSION = "lp_state_partial_emg_v4_2_candidate"
REQUIRED = set(WL) | {k for ch in PSD for k in ch}
MIN_SIGNAL_MS = 150.
TRANSITION_MS = 125.


def _runs(mask):
    arr = np.asarray(mask, bool)
    starts = np.flatnonzero(arr & ~np.r_[False, arr[:-1]])
    ends = np.flatnonzero(arr & ~np.r_[arr[1:], False]) + 1
    return list(zip(starts, ends))


def _pointer_diagnostic(df, activity, dt):
    """Report pointer agreement under possible shifts; do NOT use it as truth."""
    if "PointerState" not in df:
        return {"has_pointer": False}
    ptr = parse_pointer(df["PointerState"])
    known = np.isfinite(ptr)
    if np.sum(known) < 50:
        return {"has_pointer": False}
    win = max(1, frames(61, dt) | 1)
    smooth = median_filter(np.where(known, ptr, .5), size=win, mode="nearest")
    pointer = smooth > .5
    # Alignment is purely post hoc, robust to constant lag; zero guard leakage.
    scores = []
    for offset_ms in (-600, -400, -200, 0, 200, 400, 600):
        d = int(round(offset_ms/dt))
        if d >= 0:
            p = pointer[:len(ptr)-d] if d else pointer
            e = activity[d:]
        else:
            p = pointer[-d:]
            e = activity[:len(ptr)+d]
        if len(e) < 50:
            continue
        intersection = np.count_nonzero(p & e)
        union = np.count_nonzero(p | e)
        scores.append((float(intersection/max(union,1)), offset_ms))
    direct = next((v for v,lag in scores if lag == 0), 0.)
    best = max(scores, default=(0.,0))
    return {"has_pointer": True, "pointer_positive_fraction": float(np.mean(pointer)),
            "pointer_events": len(_runs(pointer)),
            "emg_iou_no_lag": round(direct,4),
            "emg_iou_best_approx_lag": round(best[0],4),
            "approx_lag_ms": best[1],
            "caution": "Pointer is a noisy reference. IoU is not label accuracy."}


def label_recording(df:pd.DataFrame, *, min_wl_ratio=1.40,
                    min_psd_delta=.012, min_signal_ms=MIN_SIGNAL_MS):
    n = len(df)
    y = np.full(n, -1, np.int8)
    chan = np.full((n,3), -1, np.int8)
    usable = np.zeros(3, np.int8)
    responsive = np.zeros(3, np.int8)
    audit={"n":n,"version":VERSION,"channel_details":[]}
    def finish(reason):
        audit["reason"]=reason
        audit.update(rest=int((y==1).sum()),active=int((y==0).sum()),
                     unknown=int((y<0).sum()),
                     labeled_fraction=float(np.mean(y>=0)) if n else 0.,
                     responsive_channels=int(responsive.sum()),
                     usable_channels=int(usable.sum()),
                     active_bouts=len(_runs(y==0)),
                     rest_after_first_active=bool(np.any(y[np.flatnonzero(y==0)[0]+1:]==1))
                         if np.any(y==0) else False)
        lab=pd.DataFrame({"lp_state":y,
                          "lp_state_weight":(y>=0).astype(np.float32)})
        for ch in range(3):
            lab[f"lp_ch{ch+1}_state"]=chan[:,ch]
            lab[f"lp_ch{ch+1}_usable"]=usable[ch]
            lab[f"lp_ch{ch+1}_responsive"]=responsive[ch]
        lab["lp_teacher_version"]=VERSION
        return lab,audit

    if n<100 or not REQUIRED.issubset(df.columns):
        return finish("missing_data_or_short")
    dt,gaps=sample_timing(df)
    if dt is None:
        return finish("invalid_timing")
    audit["dt_ms"]=dt
    w_raw=to_numbers(df,WL)
    p_raw=np.stack([to_numbers(df,ch) for ch in PSD],axis=1)
    if "psd_mean_applied" in df:
        mark=pd.to_numeric(df.psd_mean_applied,errors="coerce")
        if mark.eq(1).all():
            p_raw*=BIN_COUNTS.reshape(1,1,5)
        elif not mark.eq(0).all():
            return finish("mixed_psd_provenance")
    valid=(np.isfinite(w_raw)&(w_raw>=0)&(w_raw<65525)&
           np.all(np.isfinite(p_raw)&(p_raw>=0)&(p_raw<65530),axis=2))
    w=median_filter(np.maximum(np.nan_to_num(w_raw, nan=0, posinf=0, neginf=0),0),
                    size=(frames(75,dt)|1,1),mode="nearest")
    p=np.median(np.maximum(np.nan_to_num(p_raw,nan=0,posinf=0,neginf=0),0),axis=2)/10000.
    p=median_filter(p,size=(frames(75,dt)|1,1),mode="nearest")
    evidence_active=np.zeros((n,3),bool)
    evidence_quiet=np.zeros((n,3),bool)
    reject=dilate(gaps,frames(140.,dt))
    for ch in range(3):
        valid_ch=valid[:,ch] & ~reject
        good_values=np.flatnonzero(valid_ch)
        detail={"channel":ch+1,"usable":False,"responsive":False}
        if len(good_values)<max(100,round(n*.65)):
            detail["reason"]="insufficient_valid_samples"
            audit["channel_details"].append(detail)
            continue
        ww=w[valid_ch,ch]
        pp=p[valid_ch,ch]
        floor_w=float(np.percentile(ww,15))
        floor_p=float(np.percentile(pp,15))
        high_w=float(np.percentile(ww,90))
        high_p=float(np.percentile(pp,90))
        ratio=(w[:,ch]+25)/(floor_w+25)
        delta=p[:,ch]-floor_p
        detail.update(floor_w=round(floor_w,2),floor_psd=round(floor_p,5),
                      dynamic_ratio=round((high_w+25)/(floor_w+25),3),
                      dynamic_psd=round(high_p-floor_p,5))
        # Control recordings show quiet connected channels can have near-zero PSD;
        # a PSD MINIMUM would incorrectly reject good silent channels.
        # These validity checks only reject obvious artifacts, not true lead-off.
        good=(30<=floor_w<=3500) and (floor_p<=.60)
        if not good:
            detail["reason"]="implausible_baseline"
            audit["channel_details"].append(detail)
            continue
        usable[ch]=1
        detail["usable"]=True
        # A responsive channel must exhibit actual paired WL+PSD transitions.
        # Low dynamic is uninformative, not proof of physical disconnection.
        active=keep_long_runs(
            (ratio>=min_wl_ratio) & (delta>=min_psd_delta) & valid_ch,
            frames(min_signal_ms,dt))
        quiet=keep_long_runs(
            (ratio<=1.23) & (delta<=.025) & valid_ch,
            frames(min_signal_ms,dt))
        anchor=(active.sum()>=frames(260,dt) and
                quiet.sum()>=frames(260,dt) and
                high_w>=1.30*max(floor_w,1) and
                high_p-floor_p>=min_psd_delta)
        if not anchor:
            detail["reason"]="no_clear_low_high_anchor"
            audit["channel_details"].append(detail)
            continue
        responsive[ch]=1
        detail["responsive"]=True
        # Guard EMG-state transitions instead of unreliable pointer timestamps.
        signal_change=np.zeros(n,bool)
        signal_change[1:] = active[1:]!=active[:-1]
        border=dilate(signal_change,frames(TRANSITION_MS,dt))
        active &= ~border
        quiet &= ~border
        evidence_active[:,ch]=active
        evidence_quiet[:,ch]=quiet
        chan[active,ch]=0
        chan[quiet,ch]=1
        detail.update(active_rows=int(active.sum()),quiet_rows=int(quiet.sum()),
                      reason="responsive")
        audit["channel_details"].append(detail)
    # Any one independently responsive channel can be sufficient.
    # Others do NOT vote negative or sabotage a valid responsive channel.
    if responsive.any():
        rmask=responsive.astype(bool)
        at_least_one=np.any(evidence_active[:,rmask],axis=1)
        all_quiet=np.all(evidence_quiet[:,rmask],axis=1)
        y[at_least_one]=0
        y[all_quiet & ~at_least_one]=1
        # Unknown at discontinuities. Pointer data NEVER alters these states.
        y[reject]=-1
    audit["pointer_diagnostic"]=_pointer_diagnostic(df,y==0,dt)
    return finish("signal_supported" if responsive.any() else "no_responsive_channels")
