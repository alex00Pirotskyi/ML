#!/usr/bin/env python3
"""Read-only real-data V4.1 vs V4.2 partial-channel stress audit.

No physical held-button reference is available. Coverage is NOT accuracy.
Pointer IoU is a noisy context diagnostic and has no role in training labels.
"""
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lp_state_v4 import READ, label_recording as old_label
from lp_state_v4_partial import label_recording as partial_label, VERSION

ROOT=Path("DATA/DATA_LONG_PRESS")
LP=("SOFT_LONG_PRESS","MID_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS")
CONTROLS=("CONNECTED","NOTHING","DISCONNECTED","LOOSE","TABLE")


def summary(items):
    n=sum(x["n"] for x in items)
    old=sum(x["v41"]["active"]+x["v41"]["rest"] for x in items)
    new=sum(x["v42"]["active"]+x["v42"]["rest"] for x in items)
    return {
        "files":len(items),"rows":n,
        "old_labeled_fraction":round(old/max(1,n),5),
        "new_labeled_fraction":round(new/max(1,n),5),
        "new_active_fraction":round(sum(x["v42"]["active"] for x in items)/max(1,n),5),
        "new_rest_fraction":round(sum(x["v42"]["rest"] for x in items)/max(1,n),5),
        "new_unknown_fraction":round(sum(x["v42"]["unknown"] for x in items)/max(1,n),5),
        "new_files_any_active":sum(x["v42"]["active"]>0 for x in items),
        "new_files_any_rest":sum(x["v42"]["rest"]>0 for x in items),
        "new_responsive_ch_distribution":dict(Counter(str(x["v42"]["responsive_channels"]) for x in items)),
        "new_usable_ch_distribution":dict(Counter(str(x["v42"]["usable_channels"]) for x in items)),
        "pointer_iou_zero_lag":round(float(np.mean([x["v42"].get("pointer_diagnostic",{}).get("emg_iou_no_lag",0)
                      for x in items])),4) if items else None,
        "pointer_iou_best_shift":round(float(np.mean([x["v42"].get("pointer_diagnostic",{}).get("emg_iou_best_approx_lag",0)
                      for x in items])),4) if items else None
    }


def main():
    rows=[]
    failures=[]
    for group in LP+CONTROLS:
        for src in sorted((ROOT/group).glob("*.csv.gz")):
            try:
                df=pd.read_csv(src,compression="gzip",low_memory=False,
                               usecols=lambda c:c in READ)
                # Normal V4.1 only has a defined output for LP + CONNECTED/NOTHING;
                # use -1 for other folders, not a fabricated comparison.
                old=None
                if group in LP+("CONNECTED","NOTHING"):
                    _,old=old_label(df,group)
                labels,report=partial_label(df)
                a=labels.lp_state.to_numpy()
                assert set(np.unique(a)).issubset({-1,0,1})
                assert np.array_equal(labels.lp_state_weight.to_numpy(),(a>=0).astype(float))
                assert report["n"]==report["active"]+report["rest"]+report["unknown"]
                rows.append({"group":group,"file":src.name,"n":len(df),"v41":old or
                             {"active":0,"rest":0,"unknown":len(df)},
                             "v42":report})
            except Exception as exc:
                failures.append({"group":group,"file":src.name,
                                 "error":type(exc).__name__+": "+str(exc)[:220]})
        arr=[x for x in rows if x["group"]==group]
        print("PARTIAL_GROUP",group,json.dumps(summary(arr)),flush=True)
    by={group:summary([x for x in rows if x["group"]==group]) for group in LP+CONTROLS}
    result={"teacher":VERSION,
            "warning":"weak-label coverage, NOT independently verified LP accuracy",
            "lp":summary([x for x in rows if x["group"] in LP]),
            "quiet_controls":summary([x for x in rows if x["group"] in ("CONNECTED","NOTHING")]),
            "disconnected_proxy":summary([x for x in rows if x["group"]=="DISCONNECTED"]),
            "groups":by,"processed":len(rows),"errors":failures,"files":rows}
    Path("/tmp/lp_v4_partial_audit.json").write_text(json.dumps(result,indent=2))
    print("PARTIAL_TOTAL",json.dumps({k:v for k,v in result.items() if k!="files"}),flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} audit errors")
    if len(rows)<190:
        raise RuntimeError(f"Expected >=190 recordings, found {len(rows)}")

if __name__=="__main__":
    main()
