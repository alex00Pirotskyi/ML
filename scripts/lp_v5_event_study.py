#!/usr/bin/env python3
"""Read-only V4.2 event-morphology study on LP/negative-control recordings.

No physical truth: controls are weak proxies and can contain real activity.
Report sample coverage vs event duration/repetition without tuning labels.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from lp_state_v4 import READ, frames, sample_timing
from lp_state_v4_partial import label_recording, _runs

ROOT=Path("DATA/DATA_LONG_PRESS")
GROUPS=("SOFT_LONG_PRESS","MID_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS",
        "CONNECTED","NOTHING","DISCONNECTED","LOOSE","TABLE")
CUTS=(150,250,350,500,700)


def summarize(items):
    n=sum(x["n"] for x in items)
    result={"files":len(items),"samples":n,
            "any_active_files":sum(x["active"]>0 for x in items),
            "active_fraction":round(sum(x["active"] for x in items)/max(n,1),5),
            "events_dur_250_per_file_median":float(np.median([x["events_dur_250"] for x in items])) if items else 0.}
    result["cuts"]={}
    for cut in CUTS:
        k=str(cut)
        result["cuts"][k]={
            "files_any_event":sum(x["counts"][k]>0 for x in items),
            "files_2plus_events":sum(x["counts"][k]>=2 for x in items),
            "files_3plus_events":sum(x["counts"][k]>=3 for x in items),
            "event_total":sum(x["counts"][k] for x in items),
            "bracketed_events":sum(x["bracketed"][k] for x in items),
            "retained_active_fraction":round(sum(x["kept_rows"][k] for x in items)/max(n,1),5),
            "2plus_files_retained_fraction":round(sum(x["kept_rows"][k] for x in items if x["counts"][k]>=2)/max(n,1),5)
        }
    return result


def main():
    all_items=[]
    for group in GROUPS:
        paths=sorted((ROOT/group).glob("*.csv.gz"))
        for src in paths:
            df=pd.read_csv(src,compression="gzip",usecols=lambda c:c in READ,low_memory=False)
            y,meta=label_recording(df)
            s=y.lp_state.to_numpy()
            dt,_=sample_timing(df)
            runs=_runs(s==0)
            stats={"group":group,"file":src.name,"n":len(df),"active":meta["active"],
                   "rest":meta["rest"],"unknown":meta["unknown"],
                   "responsive_channels":meta["responsive_channels"],
                   "counts":{},"kept_rows":{},"bracketed":{},
                   "events_dur_250":0}
            if dt is not None:
                for c in CUTS:
                    minsz=frames(c,dt)
                    ev=[(a,b) for a,b in runs if b-a>=minsz]
                    k=str(c)
                    stats["counts"][k]=len(ev)
                    stats["kept_rows"][k]=sum(b-a for a,b in ev)
                    bracket=0
                    for a,b in ev:
                        pad=frames(1500,dt)
                        before=s[max(0,a-pad):a]
                        after=s[b:min(len(s),b+pad)]
                        if int((before==1).sum())>=frames(150,dt) and int((after==1).sum())>=frames(150,dt):
                            bracket+=1
                    stats["bracketed"][k]=bracket
                stats["events_dur_250"]=stats["counts"]["250"]
            else:
                for c in CUTS:
                    k=str(c);stats["counts"][k]=0;stats["kept_rows"][k]=0;stats["bracketed"][k]=0
            all_items.append(stats)
        print("EVENT_SHAPE",group,json.dumps(summarize([x for x in all_items if x["group"]==group])),flush=True)
    report={"groups":{g:summarize([x for x in all_items if x["group"]==g]) for g in GROUPS},
            "files":all_items,
            "warning":"Controls weak; not a physical false-positive rate."}
    Path("/tmp/lp_v5_event_study.json").write_text(json.dumps(report,indent=2))
    print("EVENT_SHAPE_DONE",len(all_items),flush=True)


if __name__=="__main__":
    main()
