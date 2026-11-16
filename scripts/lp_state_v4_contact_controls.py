#!/usr/bin/env python3
"""Evaluate fixed PSD/WL validity thresholds against weak contact controls.

CONNECTED=nominal good CH1-3, DISCONNECTED=nominal bad CH1/2 only;
CH3 in disconnected captures is NOT a bad-contact label.
These are weak recording-level proxies, NOT electrode impedance ground truth.
"""
from __future__ import annotations
from collections import defaultdict
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lp_state_v4 import WL,PSD,READ,BIN_COUNTS,to_numbers

def sample(path):
    df=pd.read_csv(path,compression="gzip",usecols=lambda k:k in READ,low_memory=False)
    if len(df)<10:return []
    raw_w=to_numbers(df,WL)
    raw_p=np.stack([to_numbers(df,ch) for ch in PSD],axis=1)
    marker=df.get("psd_mean_applied")
    if marker is not None:
        m=pd.to_numeric(marker,errors="coerce")
        if m.eq(1).all():raw_p=raw_p*BIN_COUNTS.reshape(1,1,5)
    wl=np.nanmedian(raw_w,axis=0)
    psd=np.nanmedian(raw_p,axis=(0,2))/10000.
    return [{"ch":i+1,"wl":float(wl[i]),"psd":float(psd[i])} for i in range(3)]

def main():
    out=defaultdict(list)
    for group in ("CONNECTED","DISCONNECTED"):
        for path in sorted((Path("DATA/DATA_LONG_PRESS")/group).glob("*.csv.gz")):
            rows=sample(path)
            for row in rows:
                kind=("nominal_good" if group=="CONNECTED" or row["ch"]==3
                      else "nominal_bad")
                out[kind].append({"file":path.name,"group":group,**row})
    result={}
    for kind,items in out.items():
        p=np.array([r["psd"] for r in items])
        w=np.array([r["wl"] for r in items])
        result[kind]={
            "channels":len(items),
            "wl_percentiles":{str(q):round(float(np.percentile(w,q)),5) for q in (5,25,50,75,95)},
            "psd_percentiles":{str(q):round(float(np.percentile(p,q)),5) for q in (5,25,50,75,95)},
            "old_gate_pass":round(float(np.mean((w>=65)&(w<=3500)&(p>=.02)&(p<=5.5))),5),
            "current_gate_pass":round(float(np.mean((w>=65)&(w<=3500)&(p<=.60))),5),
            "alternates":{f"wl{max_w}_psd{max_p}":round(float(np.mean((w>=65)&(w<=max_w)&(p<=max_p))),5)
                          for max_w in (2500,3000,3500) for max_p in (.3,.45,.6,.8)},
            "psd_min_pass":{str(t):round(float(np.mean(p>=t)),5)
                            for t in (0.001,0.005,0.01,0.015,0.02,0.03,0.05)},
            "median_wl":round(float(np.median(w)),2),
            "median_psd":round(float(np.median(p)),5)}
        print("CONTACT_CONTROL",kind,json.dumps(result[kind]),flush=True)
    result["note"]="Weak folder-level assumptions only: CH3 nominally good in DISCONNECTED recordings."
    Path("/tmp/lp_v4_control_qc.json").write_text(json.dumps({"summary":result,
        "channels":out},indent=2))
    print("CONTACT_CONTROL_SUMMARY",json.dumps(result),flush=True)

if __name__=="__main__":main()
