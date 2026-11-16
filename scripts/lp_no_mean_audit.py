#!/usr/bin/env python3
"""Audit new 'DATA NO PSD MEAN' CSVs without writing to source files."""
from __future__ import annotations
import collections
import csv
import gzip
import io
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path("DATA/DATA_LONG_PRESS")
BEFORE="2e14d95252b2ef94fdd1c8c92c643220ff432855"
GROUPS=("CONNECTED","DISCONNECTED","LOOSE","NOTHING","TABLE","MID_LONG_PRESS",
        "SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS","DRAG_AND_DROP","TAP","SWIPE")
BIN=[8,10,12,46,51]
NAME=lambda c,z:f"p_core->emg_psd_mean[CH{c}][{z}]"
WC=lambda c:f"p_core->wl[CH{c}]"

def sample_data(path, revision=None):
    if revision is None:
        return pd.read_csv(path,compression="gzip",nrows=1000,low_memory=False)
    result=subprocess.run(["git","show",f"{revision}:{path.as_posix()}"],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    return pd.read_csv(io.BytesIO(result.stdout),compression="gzip",
                       nrows=1000,low_memory=False)

def matrix_stats(path):
    a=sample_data(path)
    b=sample_data(path,BEFORE)
    z={}
    for ch in range(1,4):
        ratios=[]
        wl_same=None
        if WC(ch) in a and WC(ch) in b:
            wl_same=bool(np.allclose(a[WC(ch)].values,b[WC(ch)].values,equal_nan=True))
        for k in range(5):
            col=NAME(ch,k)
            if col not in a or col not in b:
                ratios.append(None)
            else:
                x=pd.to_numeric(a[col],errors="coerce").to_numpy(float)
                y=pd.to_numeric(b[col],errors="coerce").to_numpy(float)
                usable=np.isfinite(x)&np.isfinite(y)&(y>0)
                ratios.append(round(float(np.median(x[usable]/y[usable])),3) if usable.any() else None)
        z[f"CH{ch}"]={"median_new_to_old_psd_zone_ratio":ratios,"WL_unchanged_first1000":wl_same}
    return {
        "file":path.as_posix(),
        "rows_sampled":len(a),"same_column_names":list(a.columns)==list(b.columns),
        "old_marker_present":"psd_mean_applied" in b,
        "old_marker_values":b["psd_mean_applied"].value_counts(dropna=False).to_dict() if "psd_mean_applied" in b else {},
        "new_marker_present":"psd_mean_applied" in a,
        "new_marker_values":a["psd_mean_applied"].value_counts(dropna=False).to_dict() if "psd_mean_applied" in a else {},
        "new_psd_mean_columns":[x for x in a if "emg_psd_mean[" in x][:5],
        "new_psd_history_columns":[x for x in a if "p_core->emg_psd[" in x][:5],
        "by_channel":z,
    }

def main():
    n=0
    schema=collections.Counter()
    marker=collections.Counter()
    per_group={}
    for group in GROUPS:
        paths=sorted((ROOT/group).glob("*.csv.gz"))
        per_group[group]={"files":len(paths)}
        for p in paths:
            with gzip.open(p,"rt",encoding="utf-8-sig",newline="") as f:
                cols=tuple(next(csv.reader(f)))
            schema[cols]+=1
            n+=1
            marker[("present" if "psd_mean_applied" in cols else "absent")]+=1
        first=paths[0]
        with gzip.open(first,"rt",encoding="utf-8-sig",newline="") as f:
            group_head=next(csv.reader(f))
        per_group[group].update({
            "has_wl":all(WC(ch) in group_head for ch in range(1,4)),
            "psd_mean_3x5":sum(NAME(ch,z) in group_head for ch in range(1,4) for z in range(5)),
            "history_psd_count":sum("p_core->emg_psd[" in c for c in group_head),
            "psd_mean_applied_present":"psd_mean_applied" in group_head,
            "schema_columns":len(group_head),
        })
        print("GROUP_AUDIT",group,json.dumps(per_group[group]),flush=True)
    examples=[]
    for group in GROUPS:
        ps=sorted((ROOT/group).glob("*.csv.gz"))
        ix=sorted(set([0,len(ps)//2]))
        for i in ix:
            p=ps[i]
            try: examples.append({"group":group,**matrix_stats(p)})
            except Exception as exc:examples.append({"group":group,"file":p.as_posix(),"error":f"{type(exc).__name__}: {exc}"})
    report={
        "parent":BEFORE,"source_head":"DATA NO PSD MEAN",
        "scanned_recordings":n,
        "schema_counts":[{"n":count,"columns":len(cols),"has_marker":"psd_mean_applied" in cols,
             "psd_mean_names":[x for x in cols if "emg_psd_mean" in x][:3]}
             for cols,count in schema.most_common()],
        "marker":dict(marker),
        "groups":per_group,
        "before_after":examples,
    }
    print("SUMMARY",json.dumps({k:v for k,v in report.items() if k!="before_after"}),flush=True)
    for x in examples:
        print("EXAMPLE_COMPARE",json.dumps(x),flush=True)
    out=Path("/tmp/lp_no_mean_schema.json")
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    if n!=2431:raise AssertionError(f"Expected 2431 LP files; found {n}")

if __name__=="__main__":main()
