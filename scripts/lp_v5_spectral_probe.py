#!/usr/bin/env python3
"""Descriptive five-band PSD/change consistency study; no truth or tuning."""
from pathlib import Path
from collections import defaultdict
import json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lp_state_v4 import PSD, READ, BIN_COUNTS
from lp_state_v4_partial import label_recording

ROOT=Path("DATA/DATA_LONG_PRESS")
GROUPS=("SOFT_LONG_PRESS","MID_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS",
        "CONNECTED","NOTHING","DISCONNECTED","LOOSE","TABLE")


def num(df,names):
    return np.column_stack([pd.to_numeric(df[k],errors="coerce").fillna(0).to_numpy(float)
                            for k in names])

def study(df,labels):
    rows=[]
    marker=df.get("psd_mean_applied")
    fix=(marker is not None and pd.to_numeric(marker,errors="coerce").eq(1).all())
    for ci in range(3):
        state=labels[f"lp_ch{ci+1}_state"].to_numpy()
        active=state==0
        rest=state==1
        if active.sum()<20 or rest.sum()<20:
            continue
        p=num(df,PSD[ci])
        if fix:p=p*BIN_COUNTS[None,:]
        before=np.median(p[rest,:],axis=0)/10000.
        after=np.median(p[active,:],axis=0)/10000.
        delta=np.maximum(after-before,0)
        total=float(delta.sum())
        n_positive=int((delta>=.015).sum())
        shares=np.asarray(delta/max(total,1e-10))
        entropy=float(-np.sum(shares*np.log(shares+1e-10))/np.log(5))
        rows.append({"channel":ci+1,"active":int(active.sum()),
            "rest":int(rest.sum()),"delta_psd":delta.round(4).tolist(),
            "delta_sum":total,"delta_positive_zones":n_positive,
            "delta_max_share":float(shares.max()),
            "delta_entropy":entropy,
            "delta_max":float(delta.max())})
    return rows


def summarize(rows):
    if not rows:return {"channels":0}
    def stat(k):
        vals=np.array([x[k] for x in rows],float)
        return {str(p):round(float(np.percentile(vals,p)),4) for p in (10,25,50,75,90)}
    return {"channels":len(rows),
            "positive_zones":stat("delta_positive_zones"),
            "max_band_share":stat("delta_max_share"),
            "entropy":stat("delta_entropy"),
            "delta_sum":stat("delta_sum"),
            "at_least_3_bands":round(float(np.mean([x["delta_positive_zones"]>=3 for x in rows])),4),
            "at_least_4_bands":round(float(np.mean([x["delta_positive_zones"]>=4 for x in rows])),4)}


def main():
    by=defaultdict(list)
    for group in GROUPS:
        for src in sorted((ROOT/group).glob("*.csv.gz")):
            df=pd.read_csv(src,compression="gzip",usecols=lambda c:c in READ,low_memory=False)
            lab,_=label_recording(df)
            items=study(df,lab)
            for item in items:
                by[group].append({"file":src.name,**item})
        print("SPECTRAL_PROBE",group,json.dumps(summarize(by[group])),flush=True)
    Path("/tmp/lp_v5_spectral_study.json").write_text(json.dumps({
        "summary":{k:summarize(x) for k,x in by.items()},
        "per_channel":dict(by),
        "warning":"control labels are weak; this is not physical accuracy"
    },indent=2))
    print("SPECTRAL_PROBE_DONE",sum(len(x) for x in by.values()),flush=True)

if __name__=="__main__":
    main()
