#!/usr/bin/env python3
"""Build LP V3 offline per-channel rest/activity/contact pseudo-labels.

Default output is DATA/DATA_LONG_PRESS_LABELED_V3, outside input tree.
Source .csv.gz files are read only. See research report for limitations.
We train an approximate CONTACT-ARTIFACT model on weak context labels:
  good = all CH in CONNECTED; bad = CH1/CH2 in DISCONNECTED
with responsive LP sessions as weak good examples. Participants are
cross-fit for ALL records used in model training, avoiding self-label leakage.
"""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lp_rest_labeler import write_labeled, OUTPUT_COLUMNS
from lp_v3_teacher import label_recording,VERSION
from scripts.lp_v3_feature_probe import record, ROOT as DEFAULT_INPUT, COL as PROBE_COL

DEFAULT_OUTPUT=Path("DATA/DATA_LONG_PRESS_LABELED_V3")
LP={"MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS"}
TRAIN_DIRS=("CONNECTED","DISCONNECTED","SOFT_LONG_PRESS","MID_LONG_PRESS",
            "STRONG_LONG_PRESS","LONG_PRESS")
ALL_DIRS=("CONNECTED","DISCONNECTED","LOOSE","NOTHING","TABLE",
          "MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS",
          "DRAG_AND_DROP","TAP","SWIPE")
EXTRA_READ=set(PROBE_COL)|{"psd_mean_applied"}

def normalize_person(value,filename):
    if value is None or pd.isna(value) or not str(value).strip():
        value=filename.split("_")[0]
    value=re.sub("[^a-z0-9]","",str(value).lower())
    for stem in ("ohad","nikol","shahar"):
        if value.startswith(stem):return stem
    return value or "unknown"

def fit_model():
    return make_pipeline(StandardScaler(),LogisticRegression(
        C=.1,class_weight="balanced",max_iter=1000))

def train_contact_quality(root,weak_weight=.7):
    """Weak-signal quality calibration; OOF by person, no CH3 bad assumption."""
    records=[]
    errors=[]
    for group in TRAIN_DIRS:
        for path in sorted((root/group).glob("*.csv.gz")):
            try:
                r=record(path,group)
                if r is None:continue
                # Same user normalization for cross-recording identity.
                uid=normalize_person(r["user"],path.name)
                for ch in range(3):
                    labeled=(group=="CONNECTED" or (group=="DISCONNECTED" and ch<2))
                    y=(1 if group=="DISCONNECTED" and ch<2 else 0 if labeled else -1)
                    weak_good=(group in LP and r["ch"][ch]["gmm_credible"])
                    records.append({"path":str(path),"user":uid,"group":group,"ch":ch,
                        "x":r["ch"][ch]["feat"],"target":y,"weak":weak_good,
                        "wl_median":r["ch"][ch]["wl_median"]})
            except Exception as exc:
                errors.append([str(path),f"{type(exc).__name__}: {exc}"[:160]])
    X=np.asarray([r["x"] for r in records],float)
    y=np.asarray([r["target"] for r in records],int)
    user=np.asarray([r["user"] for r in records])
    weak=np.asarray([r["weak"] for r in records])
    groups=np.asarray([r["group"] for r in records])
    oof=np.zeros(len(records),float)
    model_by_user={}
    splits=list(GroupKFold(n_splits=5).split(X,groups=user))
    for tr,te in splits:
        tr0=tr[y[tr]>=0]
        trw=tr[weak[tr]]
        ids=np.concatenate([tr0,trw])
        Y=np.concatenate([y[tr0],np.zeros(len(trw),int)])
        wt=np.concatenate([np.ones(len(tr0)),np.full(len(trw),weak_weight)])
        if len(np.unique(Y))<2:
            raise RuntimeError("Not enough contact conditions in a heldout fold")
        m=fit_model()
        m.fit(X[ids],Y,logisticregression__sample_weight=wt)
        oof[te]=m.predict_proba(X[te])[:,1]
        for name in set(user[te]):
            model_by_user[str(name)]=m
    known=y>=0
    pred=oof[known]
    yy=y[known]
    contact_report={
       "n_provisional":int(sum(known)),"users":int(len(set(user[known]))),
       "auc":float(roc_auc_score(yy,pred)),
       "balanced_accuracy":float(balanced_accuracy_score(yy,pred>=.5)),
       "confusion_0_connected_1_disconnected":
         confusion_matrix(yy,pred>=.5,labels=[0,1]).tolist(),
       "weak_responsive_LP_count":int(sum(weak)),
       "weak_LP_weight":weak_weight,
       "calibration_errors":errors,
       "note":"Weak recording context labels, NOT measured electrode-level contact truth"
    }
    for g in TRAIN_DIRS:
        ids=np.where(groups==g)[0]
        if not len(ids):continue
        mat=oof[ids].reshape(-1,3)
        contact_report[g+"_ood_flags"]={
            "n_recordings":len(ids)//3,
            "fraction_any_score_ge_0p5":float(np.mean(np.any(mat>=.5,axis=1)))}
    all_training=np.flatnonzero((y>=0)|weak)
    training_y=np.asarray([
        int(y[i]) if y[i]>=0 else 0 for i in all_training],int)
    training_w=np.asarray([
        1. if y[i]>=0 else weak_weight for i in all_training],float)
    full=fit_model()
    full.fit(X[all_training],training_y,
             logisticregression__sample_weight=training_w)
    # Per-person cross-fitted model for any file belonging to that participant;
    # otherwise use the final model.
    oof_lookup=defaultdict(lambda:[None]*3)
    for i,rec in enumerate(records):
        oof_lookup[rec["path"]][rec["ch"]]=float(oof[i])
    connected=[r for r in records if r["group"]=="CONNECTED"]
    healthy=[float(np.percentile(
        [r["wl_median"] for r in connected if r["ch"]==ch],95))
        for ch in range(3)]
    return full,model_by_user,dict(oof_lookup),healthy,contact_report

def review_intervals(labels, df, minimum_ms=500., max_regions=4):
    """Prioritize uncertain time spans for independent human/force review."""
    from lp_causal_fw_teacher import step_ms
    dt=step_ms(df)
    names=[f"lp_unknown_ch{ch}" for ch in (1,2,3)]
    U=labels[names].to_numpy(dtype=float)
    mask=np.max(U,axis=1)>.55
    spans=[];start=None
    for i,flag in enumerate(np.append(mask,False)):
        if flag and start is None:
            start=i
        elif not flag and start is not None:
            duration=(i-start)*dt
            if duration>=minimum_ms:
                sub=U[start:i]
                ch=int(np.argmax(sub.mean(axis=0)))+1
                spans.append({
                    "start_s":round(start*dt/1000,3),
                    "end_s":round(i*dt/1000,3),
                    "duration_s":round(duration/1000,3),
                    "dominant_unknown_channel":ch,
                    "mean_unknown":round(float(np.mean(sub[:,ch-1])),3)
                })
            start=None
    spans.sort(key=lambda r:r["duration_s"],reverse=True)
    return spans[:max_regions]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,default=DEFAULT_INPUT)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--psd-mode",choices=("auto","raw","legacy-divided"),default="auto")
    p.add_argument("--limit-per-group",type=int,default=0)
    p.add_argument("--tap-cap",type=int,default=0)
    p.add_argument("--swipe-cap",type=int,default=0)
    p.add_argument("--contact-weak-weight",type=float,default=.7)
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--dry-run",action="store_true",
        help="Compute/evaluate labels, but do not write labeled CSV.gz files")
    a=p.parse_args()
    root=a.input_dir.resolve();target=a.output_dir.resolve()
    if not root.is_dir():p.error("Input directory missing")
    if target==root or root in target.parents:
        p.error("Output directory must not be in input tree")
    if not 0<=a.contact_weak_weight<=1:
        p.error("Contact weak weight must be 0..1")
    started=time.monotonic()
    print("TRAIN_QUALITY_MODEL",flush=True)
    full,subject_models,known_scores,ref,quality=train_contact_quality(
        root,weak_weight=a.contact_weak_weight)
    print("CONTACT_OOF_EVAL",json.dumps(quality),flush=True)
    print("CONNECTED_WL_REFERENCE_P95",json.dumps(ref),flush=True)
    paths=[]
    for group in ALL_DIRS:
        candidates=sorted((root/group).glob("*.csv.gz"))
        if a.limit_per_group:
            candidates=candidates[:a.limit_per_group]
        if group=="TAP" and a.tap_cap and len(candidates)>a.tap_cap:
            candidates=[candidates[i] for i in
                np.linspace(0,len(candidates)-1,a.tap_cap,dtype=int)]
        if group=="SWIPE" and a.swipe_cap and len(candidates)>a.swipe_cap:
            candidates=[candidates[i] for i in
                np.linspace(0,len(candidates)-1,a.swipe_cap,dtype=int)]
        paths.extend([(group,x) for x in candidates])
    result_by_group=defaultdict(list)
    review=[]
    errors=[]
    total=Counter()
    for idx,(group,src) in enumerate(paths,1):
        dest=target/src.relative_to(root)
        if dest.exists() and not a.overwrite and not a.dry_run:
            total["skipped"]+=1
            continue
        try:
            df=pd.read_csv(src,compression="gzip",usecols=lambda c:c in EXTRA_READ,
                low_memory=False)
            short_recording=len(df)<250
            if short_recording:
                # No defensible per-file calibration exists. Preserve every
                # source row and mark all short recordings as uncertain.
                probs=[.55,.55,.55]
                total["short_uncalibrated"]+=1
            elif str(src) in known_scores:
                probs=known_scores[str(src)]
            else:
                rec=record(src,group)
                if rec is None:
                    probs=[.55,.55,.55]
                    total["short_uncalibrated"]+=1
                    rec=None
                if rec is not None:
                    model=subject_models.get(normalize_person(
                        df["User"].iloc[0] if "User" in df else None,src.name),
                        full)
                    X=np.asarray([r["feat"] for r in rec["ch"]],float)
                    probs=model.predict_proba(X)[:,1]
            labels,stat=label_recording(df,contact_prob=probs,
                good_reference_wl=ref,psd_mode=a.psd_mode,context=group)
            assert len(labels)==len(df)
            for ch in (1,2,3):
                cols=[f"lp_{v}_ch{ch}" for v in
                      ("resting","high_activity","disconnection","unknown")]
                if not np.allclose(labels[cols].sum(axis=1),1,atol=3e-5):
                    raise ValueError("Per-channel target sums do not equal one")
            if not a.dry_run:
                write_labeled(src,dest,labels)
            total["ok"]+=1
            details=stat["channel_details"]
            record_stat={
                "group":group,"file":src.name,"n":len(df),
                "mean_rest":stat["lp_state_rest_fraction"],
                "uncertainty":stat["uncertain_fraction"],
                "weight":stat["mean_label_weight"],
                "credible_channels":sum(r["credible"] for r in details),
                "weak_channels":sum(r["weak"] for r in details),
                "quality_contact_prob":list(map(float,probs)),
                "mean_disconnection":[r["mean_disconnection"] for r in details],
                "mean_activity":[r["mean_active"] for r in details],
                "mean_unknown":[r["mean_unknown"] for r in details],
                "uncertain_intervals":review_intervals(labels,df)
            }
            result_by_group[group].append(record_stat)
            if (stat["uncertain_fraction"]>.6 or
                (group in LP and stat["lp_state_rest_fraction"]<.015) or
                (group=="CONNECTED" and stat["lp_state_rest_fraction"]<.15)):
                review.append(record_stat)
        except Exception as e:
            total["error"]+=1
            errors.append({"group":group,"file":src.name,
                           "error":f"{type(e).__name__}: {e}"})
            print("ERROR",group,src.name,str(e)[:200],flush=True)
        if idx%50==0 or idx==len(paths):
            print("LABEL_PROGRESS",idx,len(paths),dict(total),
                  "elapsed_s",round(time.monotonic()-started,1),flush=True)
    by={}
    for group,items in result_by_group.items():
        by[group]={
            "n":len(items),
            "rest_fraction":float(np.mean([x["mean_rest"] for x in items])),
            "uncertain_fraction":float(np.mean([x["uncertainty"] for x in items])),
            "mean_weight":float(np.mean([x["weight"] for x in items])),
            "files_with_any_rest":int(sum(x["mean_rest"]>0 for x in items)),
            "files_with_any_credible_state":int(sum(x["credible_channels"]>0 for x in items)),
            "mean_bad_contact_probability_per_ch":[float(np.mean(
                [x["quality_contact_prob"][ch] for x in items])) for ch in range(3)]}
        print("LABEL_GROUP",group,json.dumps(by[group]),flush=True)
    report={"version":VERSION,"input":str(root),"output":str(target),
       "mode":a.psd_mode,"dry_run":a.dry_run,"selected_files":len(paths),
       "counts":dict(total),"quality_model":quality,
       "healthy_reference_wl_p95":ref,"groups":by,
       "errors":errors[:150],
       "review_queue":sorted(review,key=lambda r:(
           r["uncertainty"]+(.4 if r["mean_rest"]==0 else 0)),
           reverse=True)[:120],
       "warning":"Pseudo-labels, not physical rest/release or impedance ground truth.",
       "elapsed_s":round(time.monotonic()-started,1)}
    print("LP_V3_REPORT",json.dumps({k:v for k,v in report.items()
            if k not in ("review_queue","errors")}),flush=True)
    if not a.dry_run:
        target.mkdir(parents=True,exist_ok=True)
        (target/"label_summary.json").write_text(json.dumps(report,indent=2))
        (target/"review_queue.json").write_text(json.dumps(report["review_queue"],indent=2))
    else:
        Path("/tmp/lp_v3_dry_run.json").write_text(json.dumps(report,indent=2))
    print("LP_V3_DONE",round(time.monotonic()-started,1),flush=True)
    return 1 if total["error"] else 0

if __name__=="__main__":
    raise SystemExit(main())
