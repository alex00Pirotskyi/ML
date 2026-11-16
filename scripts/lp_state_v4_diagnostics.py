#!/usr/bin/env python3
"""Inspect reasons for lost LP labels on real recordings. No thresholds changed."""
from __future__ import annotations
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lp_state_v4 import (LP_GROUPS, WL, PSD, READ, BIN_COUNTS, to_numbers,
                         sample_timing, frames, dilate, pointer_interiors,
                         keep_long_runs, label_recording)

ROOT=Path("DATA/DATA_LONG_PRESS")
VARIANTS=((250,180),(175,180),(250,120),(175,120),(120,120))


def inspect(df, group):
    n=len(df)
    dt,gaps=sample_timing(df)
    if dt is None:
        return {"reason":"bad_timestamp"}
    pointer, guard=pointer_interiors(df,dt,250.)
    guard|=dilate(gaps,frames(250,dt))
    for v, ms in ((1,450.),(0,250.)):
        run=pointer==v
        guard|=run & ~keep_long_runs(run,frames(ms,dt))
    off=(pointer==0)&~guard
    data={"dt_ms":round(dt,2), "off_fraction":round(float(np.mean(off)),3),
          "on_fraction":round(float(np.mean((pointer==1)&~guard)),3),
          "off_ms":round(float(off.sum()*dt)),
          "clock_gaps":int(gaps.sum())}
    if off.sum()<frames(700,dt):
        data["reason"]="no_stable_off_reference"
        return data
    w=to_numbers(df,WL)
    p=np.stack([to_numbers(df,names) for names in PSD],axis=1)
    marker=df.get("psd_mean_applied")
    if marker is not None:
        m=pd.to_numeric(marker,errors="coerce")
        if m.eq(1).all():p=p*BIN_COUNTS.reshape(1,1,5)
    from scipy.ndimage import median_filter
    w=median_filter(np.maximum(np.nan_to_num(w),0),
       size=(frames(91.,dt)|1,1),mode="nearest")
    p=median_filter(np.median(np.maximum(np.nan_to_num(p),0),axis=2)/10000.,
       size=(frames(91.,dt)|1,1),mode="nearest")
    fw=np.percentile(w[off],30,axis=0)
    fp=np.percentile(p[off],30,axis=0)
    pass_high=(fw<=3500)&(fp<=5.5)
    pass_low=(fp>=.02)
    data["floor_w"]=[round(float(x),1) for x in fw]
    data["floor_p"]=[round(float(x),3) for x in fp]
    data["usable_channels"]={}
    for low in (5,20,40,65,100):
        good=(fw>=low)&pass_high&pass_low
        data["usable_channels"][str(low)]=int(np.sum(good))
    data["reason"]="measured"
    return data


def main():
    all_files=[]
    for group in LP_GROUPS:
        for path in sorted((ROOT/group).glob("*.csv.gz")):
            df=pd.read_csv(path,compression="gzip",usecols=lambda col:col in READ,low_memory=False)
            diag=inspect(df,group)
            base={}
            for boundary,minactive in VARIANTS:
                tag=f"boundary_{boundary}_active_{minactive}"
                out,report=label_recording(df,group,boundary_ms=boundary,
                                            min_active_ms=minactive)
                base[tag]={"active":int((out.lp_state==0).sum()),
                           "rest":int((out.lp_state==1).sum()),
                           "events":report["verified_events"],
                           "reason":report["reason"]}
            all_files.append({"file":str(path.relative_to(ROOT)),"group":group,
                              "n":len(df),"signal":diag,"runs":base})
    summary={}
    for group in LP_GROUPS:
        rows=[x for x in all_files if x["group"]==group]
        total_n=sum(x["n"] for x in rows)
        out={"files":len(rows),"reasons":Counter(
            x["runs"]["boundary_250_active_180"]["reason"] for x in rows),
            "off_fraction_mean":round(float(np.mean([x["signal"].get("off_fraction",0) for x in rows])),4),
            "good_ch_ge2_per_floor_min":{low:sum(x["signal"].get("usable_channels",{}).get(str(low),0)>=2
                                              for x in rows) for low in (5,20,40,65,100)},
            "floor_w_q":{},
            "floor_p_q":{}}
        for kind,field in (("floor_w_q","floor_w"),("floor_p_q","floor_p")):
            vals=np.array([x["signal"][field] for x in rows if field in x["signal"]],float)
            if len(vals):
                out[kind]={str(p):round(float(np.percentile(vals,p)),4)
                           for p in (5,25,50,75,95)}
        out["variants"]={}
        for b,m in VARIANTS:
            tag=f"boundary_{b}_active_{m}"
            activ=sum(x["runs"][tag]["active"] for x in rows)
            rest=sum(x["runs"][tag]["rest"] for x in rows)
            events=sum(x["runs"][tag]["events"] for x in rows)
            out["variants"][tag]={"labeled_fraction":round((activ+rest)/total_n,5),
                                  "active_fraction":round(activ/total_n,5),
                                  "rest_fraction":round(rest/total_n,5),
                                  "files_any_active":sum(x["runs"][tag]["active"]>0 for x in rows),
                                  "accepted_events":events}
        summary[group]=out
        print("DIAGNOSTICS",group,json.dumps(out,default=lambda z:dict(z)),flush=True)
    import os
    target=Path(os.environ.get("LP_DIAGNOSTIC_OUT","/tmp/lp_v4_diagnostics.json"))
    target.write_text(json.dumps({"summary":summary,"files":all_files},indent=2,
                                 default=lambda z:dict(z)))
    print("DIAGNOSTICS_REPORT_PATH",target)

if __name__=="__main__":
    main()
