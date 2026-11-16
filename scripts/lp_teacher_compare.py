#!/usr/bin/env python3
"""Read-only v1 vs firmware-aware causal v2 teacher audit.

Fractions compare pseudo-labels, NOT physiological ground truth.
"""
from collections import defaultdict
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from lp_rest_labeler import compute_labels as v2, read_csv_features

legacy=Path("research/legacy/lp_rest_labeler_v1.py")
spec=importlib.util.spec_from_file_location("legacy_v1",legacy)
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
root=Path("DATA/DATA_LONG_PRESS")
groups=("CONNECTED","DISCONNECTED","LOOSE","NOTHING","TABLE",
        "MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS",
        "DRAG_AND_DROP","TAP","SWIPE")


def summarize(key,values):
    return {key:round(float(np.mean(values)),4) if values else None}

def main():
    by=defaultdict(list)
    for group in groups:
        paths=sorted((root/group).glob("*.csv.gz"))
        if group in ("TAP","SWIPE") and len(paths)>110:
            paths=[paths[i] for i in np.linspace(0,len(paths)-1,110,dtype=int)]
        for p in paths:
            try:
                df=read_csv_features(p)
                y1,_=old.compute_labels(df)
                y2,st=v2(df)
                n=len(df)
                result={
                    "file":p.name,"n":n,
                    "rest_v1":float(y1["lp_state"].mean()),
                    "rest_v2":float(y2["lp_state"].mean()),
                    "zero_weight_v2":float(np.mean(y2["lp_state_weight"]<.05)),
                    "unknown_v2":[float(y2[f"lp_unknown_ch{i}"].mean()) for i in (1,2,3)],
                    "disconnection_v2":[float(y2[f"lp_disconnection_ch{i}"].mean()) for i in (1,2,3)],
                    "high_v2":[float(y2[f"lp_high_activity_ch{i}"].mean()) for i in (1,2,3)],
                    "ready":st["calibrated_fraction_by_channel"],
                    "first_v2_rest_s":float(np.argmax(y2["lp_state"].to_numpy()>0)*st["sample_ms"]/1000)
                                      if bool(y2["lp_state"].any()) else None
                }
                by[group].append(result)
            except Exception as e:
                print("ERROR",group,p.name,type(e).__name__,str(e)[:180],flush=True)
        print("PROGRESS",group,len(by[group]),flush=True)
    for group,items in by.items():
        out={
            "files":len(items),
            "rest_v1":round(float(np.mean([x["rest_v1"] for x in items])),4),
            "rest_v2":round(float(np.mean([x["rest_v2"] for x in items])),4),
            "frac_files_any_rest_v1":round(float(np.mean([x["rest_v1"]>0 for x in items])),4),
            "frac_files_any_rest_v2":round(float(np.mean([x["rest_v2"]>0 for x in items])),4),
            "mean_low_weight_rows_v2":round(float(np.mean([x["zero_weight_v2"] for x in items])),4),
            "unknown_by_ch_v2":[round(float(np.mean([x["unknown_v2"][i] for x in items])),4) for i in range(3)],
            "disc_by_ch_v2":[round(float(np.mean([x["disconnection_v2"][i] for x in items])),4) for i in range(3)],
            "activity_by_ch_v2":[round(float(np.mean([x["high_v2"][i] for x in items])),4) for i in range(3)],
            "baseline_ready_fraction_by_ch":[round(float(np.mean([x["ready"][i] for x in items])),4) for i in range(3)],
            "median_first_rest_s":float(np.median([x["first_v2_rest_s"] for x in items if x["first_v2_rest_s"] is not None]))
                if any(x["first_v2_rest_s"] is not None for x in items) else None
        }
        print("GROUP_COMPARE",group,json.dumps(out),flush=True)
        if group=="STRONG_LONG_PRESS":
            print("STRONG_FILE_DETAIL",json.dumps(items),flush=True)
    print("TEACHER_COMPARE_DONE",flush=True)

if __name__=="__main__":main()
