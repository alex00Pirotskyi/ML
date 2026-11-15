#!/usr/bin/env python3
"""Read-only exploratory audit: resting vs sustained activity, not a classifier.

No labels inferred from PointerState/TriggerPressure. Group names are recording
context, not per-sample state or channel-level ground truth.
"""
from collections import defaultdict, Counter
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path("DATA/DATA_LONG_PRESS")
GROUPS = ("MID_LONG_PRESS", "SOFT_LONG_PRESS", "STRONG_LONG_PRESS",
          "LONG_PRESS", "DRAG_AND_DROP", "CONNECTED", "DISCONNECTED",
          "LOOSE", "NOTHING", "TABLE", "TAP", "SWIPE")
LP_GROUPS = {"MID_LONG_PRESS", "SOFT_LONG_PRESS", "STRONG_LONG_PRESS", "LONG_PRESS"}
ZONES = np.array([8, 10, 12, 46, 51], dtype=np.float64)
WL = [f"p_core->wl[CH{i}]" for i in (1,2,3)]
PSD = [[f"p_core->emg_psd_mean[CH{i}][{z}]" for z in range(5)] for i in (1,2,3)]
ACC = [f"p_core->acc_rate[{axis}][{z}]" for axis in "XYZ" for z in range(6)]
KEEP = set(WL + sum(PSD, []) + ACC + ["Timestamp", "PointerState", "TriggerPressure", "User", "class_id"])
REPORT = {}


def q(a, p):
    a = np.asarray(a, dtype=np.float64).flatten()
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else None


def safe_ratio(a, b):
    return float(a/b) if b and np.isfinite(b) and b > 0 and np.isfinite(a) else None


def median_numbers(a):
    a = [float(x) for x in a if x is not None and np.isfinite(x)]
    return float(np.median(a)) if a else None


def summary(a):
    a = np.asarray([float(x) for x in a if x is not None and np.isfinite(x)])
    return {"n":len(a),"p10":q(a,10),"p25":q(a,25),"median":q(a,50),"p75":q(a,75),"p90":q(a,90)}


def runs_true(mask, minimum):
    run=0
    for val in mask:
        run = run+1 if val else 0
        if run>=minimum:return True
    return False


def analyze_one(path, group):
    try:
        data=pd.read_csv(path, compression="gzip", usecols=lambda c: c in KEEP,
                         low_memory=False)
        if len(data)<100 or any(c not in data.columns for c in WL):
            return None
        sample_ms = np.diff(data["Timestamp"].to_numpy(dtype=float)[:1000])
        sample_ms = sample_ms[(sample_ms>0) & (sample_ms<200)]
        dt = float(np.median(sample_ms)) if sample_ms.size else 5.0
        samples = int(np.clip(round(500.0 / dt), 32, 200))
        nwin = len(data)//samples
        if nwin<4:return None
        limit=nwin*samples
        w = np.column_stack([data[c].to_numpy(dtype=float)[:limit] for c in WL])
        w = np.nanmedian(w.reshape(nwin,samples,3),axis=1)
        w = np.maximum(w,0)
        baseline=np.maximum(np.nanpercentile(w,20,axis=0),1)
        ratio=w/baseline
        # Relative activation indicators, not ground-truth press labels.
        one=np.sum(ratio >= 1.8, axis=1)
        two=np.sum(ratio >= 1.4, axis=1)
        coactivation=two>=2
        score=np.nanmedian(np.log1p(ratio),axis=1)
        low_mask=score<=np.percentile(score,20)
        high_mask=score>=np.percentile(score,80)
        if np.sum(low_mask)<2 or np.sum(high_mask)<2:
            return None
        emg = np.empty((nwin,3,5), dtype=float)
        for ch in range(3):
            for z, name in enumerate(PSD[ch]):
                emg[:,ch,z]=np.nanmedian(
                    data[name].to_numpy(dtype=float)[:limit].reshape(nwin,samples),axis=1
                ) / ZONES[z]
        emg=np.maximum(emg,0)
        power=np.sum(emg,axis=2)
        late=(emg[:,:,3]+emg[:,:,4])/(power+1e-8)
        acc_vals=np.column_stack([data[c].to_numpy(dtype=float)[:limit] for c in ACC])
        motion=np.nanmedian(np.nanmean(np.abs(acc_vals),axis=1).reshape(nwin,samples),axis=1)
        pw=np.nanmedian(data["PointerState"].to_numpy(dtype=float)[:limit].reshape(nwin,samples),axis=1)
        p_on=pw>0.5
        spec_amp_lo=np.nanmedian(power[low_mask],axis=0)
        spec_amp_hi=np.nanmedian(power[high_mask],axis=0)
        spectral_ratios=(spec_amp_hi+1)/(spec_amp_lo+1)
        wl_dynamic=(np.nanpercentile(w,90,axis=0)+1)/(np.nanpercentile(w,10,axis=0)+1)
        psd_z4share_change=np.nanmedian(late[high_mask],axis=0)-np.nanmedian(late[low_mask],axis=0)
        active_motion=median_numbers(motion[high_mask])
        resting_motion=median_numbers(motion[low_mask])
        common={
          "group":group,"rows":len(data),"duration_s":round((len(data)*dt/1000),1),
          "window_count":nwin,"median_ms":round(dt,3),"class_id":str(data["class_id"].iloc[0]),
          "baseline_wl":[round(float(x),1) for x in baseline],
          "wl_q90_over_q10":[round(float(x),3) for x in wl_dynamic],
          "psd_high_low_by_channel":[round(float(x),3) for x in spectral_ratios],
          "psd_late_band_share_high_minus_low":[round(float(x),4) for x in psd_z4share_change],
          "one_ch_above_1p8_fraction":round(float(np.mean(one>=1)),3),
          "two_ch_above_1p4_fraction":round(float(np.mean(coactivation)),3),
          "sustained_two_ch_1s":runs_true(coactivation,2),
          "flat_envelopes":bool(np.max(wl_dynamic)<1.3),
          "motion_high_over_low":safe_ratio(active_motion,resting_motion),
          "pointer_one_fraction":round(float(np.mean(p_on)),3),
          "pointer_transitions":int(np.count_nonzero(p_on[1:]!=p_on[:-1])),
          "pointer_score_on_minus_off":(
              round(float(np.median(score[p_on])-np.median(score[~p_on])),3)
              if np.any(p_on) and np.any(~p_on) else None
          )
        }
        return common
    except Exception as e:
        print("ERROR_FILE",path.as_posix(),repr(e), flush=True)
        return None


def print_json(title,obj):
    print(title+" "+json.dumps(obj,sort_keys=True,allow_nan=False), flush=True)


def main():
    by=defaultdict(list)
    errors=0
    for group in GROUPS:
        paths=sorted((ROOT/group).rglob("*.csv.gz"))
        if group in {"TAP","SWIPE"} and len(paths)>130:
            sample_indices=np.linspace(0,len(paths)-1,130,dtype=int)
            paths=[paths[i] for i in sample_indices]
        for path in paths:
            record=analyze_one(path,group)
            if record is None:errors+=1
            else:by[group].append(record)
        print("PROGRESS", group, len(by[group]), "recordings analyzed", flush=True)
    print_json("COUNTS", {"valid":{g:len(r) for g,r in by.items()},"failed_or_too_short":errors})

    for group, records in by.items():
        group_out={
            "n":len(records),
            "baseline_wl_by_channel":[summary([r["baseline_wl"][i] for r in records]) for i in range(3)],
            "within_record_q90_q10_by_channel":[summary([r["wl_q90_over_q10"][i] for r in records]) for i in range(3)],
            "psd_power_ratio_high_vs_low_by_channel":[summary([r["psd_high_low_by_channel"][i] for r in records]) for i in range(3)],
            "late_psd_share_change_by_channel":[summary([r["psd_late_band_share_high_minus_low"][i] for r in records]) for i in range(3)],
            "two_channel_activation_fraction":summary([r["two_ch_above_1p4_fraction"] for r in records]),
            "one_channel_activation_fraction":summary([r["one_ch_above_1p8_fraction"] for r in records]),
            "sustained_1s_two_ch_count":sum(r["sustained_two_ch_1s"] for r in records),
            "flat_envelopes_count":sum(r["flat_envelopes"] for r in records),
            "motion_ratio_high_vs_low":summary([r["motion_high_over_low"] for r in records]),
            "pointer_on_fraction":summary([r["pointer_one_fraction"] for r in records]),
            "pointer_rel_score_on_vs_off":summary([r["pointer_score_on_minus_off"] for r in records]),
            "pointer_transition_count":summary([r["pointer_transitions"] for r in records]),
            "duration_s":summary([r["duration_s"] for r in records]),
        }
        print_json("GROUP_SUMMARY_"+group,group_out)

    lp=sum((by[g] for g in LP_GROUPS),[])
    print_json("LONGPRESS_COMBINED",{
       "n":len(lp),
       "with_sustained_2ch_1s":sum(x["sustained_two_ch_1s"] for x in lp),
       "with_flat_wl":sum(x["flat_envelopes"] for x in lp),
       "psd_late_share_changes":[summary([x["psd_late_band_share_high_minus_low"][i] for x in lp]) for i in range(3)],
       "psd_high_over_low":[summary([x["psd_high_low_by_channel"][i] for x in lp]) for i in range(3)],
       "pointer_on_fraction":summary([x["pointer_one_fraction"] for x in lp]),
       "pointer_rel_score_on_minus_off":summary([x["pointer_score_on_minus_off"] for x in lp]),
       "two_channel_activation_fraction":summary([x["two_ch_above_1p4_fraction"] for x in lp]),
       "motion_ratio":summary([x["motion_high_over_low"] for x in lp]),
    })
    print_json("LP_PER_GROUP_ANOMALIES",{
       g:{"flat":sum(x["flat_envelopes"] for x in by[g]),
          "no_sustained_two_ch":sum(not x["sustained_two_ch_1s"] for x in by[g]),
          "pointer_negative_assoc":sum(x["pointer_score_on_minus_off"] is not None
                                        and x["pointer_score_on_minus_off"]<0 for x in by[g]),
          "pointer_comparable":sum(x["pointer_score_on_minus_off"] is not None for x in by[g])}
       for g in LP_GROUPS
    })
    print("AUDIT_DONE")


if __name__=="__main__":
    main()
