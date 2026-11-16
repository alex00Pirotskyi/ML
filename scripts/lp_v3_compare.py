#!/usr/bin/env python3
"""Read-only compare old causal V2 and new offline V3 pseudo-label outputs.

Do NOT call rest fractions accuracy. Tests unlabeled recording contexts.
"""
from collections import defaultdict
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from lp_rest_labeler import compute_labels as v2
from lp_v3_teacher import label_recording
from lp_v3_labeler import train_contact_quality,normalize_person,EXTRA_READ
from scripts.lp_v3_feature_probe import record

ROOT=Path("DATA/DATA_LONG_PRESS")
GROUPS=("CONNECTED","DISCONNECTED","LOOSE","TABLE","NOTHING","MID_LONG_PRESS",
        "SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS","DRAG_AND_DROP","TAP","SWIPE")
LP=("MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS")
def mean(a):return float(np.mean(a)) if len(a) else 0.
def med(a):return float(np.median(a)) if len(a) else 0.

def main():
    start=time.monotonic()
    full,person,known,refs,quality=train_contact_quality(ROOT,.7)
    print("CONTACT_MODEL",json.dumps(quality),flush=True)
    by=defaultdict(list);details=[]
    for group in GROUPS:
        paths=sorted((ROOT/group).glob("*.csv.gz"))
        if group in ("TAP","SWIPE") and len(paths)>100:
            paths=[paths[i] for i in np.linspace(0,len(paths)-1,100,dtype=int)]
        for i,p in enumerate(paths,1):
            try:
                df=pd.read_csv(p,compression="gzip",
                    usecols=lambda c:c in EXTRA_READ,low_memory=False)
                if len(df)<250:continue
                l2,_=v2(df)
                if str(p) in known:probs=known[str(p)]
                else:
                    rec=record(p,group)
                    if rec is None:continue
                    user=normalize_person(df["User"].iloc[0] if "User" in df else None,p.name)
                    model=person.get(user,full)
                    probs=model.predict_proba(np.array([c["feat"] for c in rec["ch"]]))[:,1]
                l3,st=label_recording(df,contact_prob=probs,good_reference_wl=refs,context=group)
                n=len(df)
                stat={"group":group,"file":p.name,"n":n,
                   "rest2":mean(l2["lp_state"]),"rest3":mean(l3["lp_state"]),
                   "uncertain2":mean(np.max(l2[[f"lp_unknown_ch{x}" for x in (1,2,3)]].to_numpy(),axis=1)>.4),
                   "uncertain3":mean(np.max(l3[[f"lp_unknown_ch{x}" for x in (1,2,3)]].to_numpy(),axis=1)>.4),
                   "weight2":mean(l2["lp_state_weight"]),"weight3":mean(l3["lp_state_weight"]),
                   "gmm_credible":int(sum(d["credible"] for d in st["channel_details"])),
                   "gmm_weak":int(sum(d["weak"] for d in st["channel_details"])),
                   "quality_probs":list(map(float,probs)),
                   "active3":mean(np.max(l3[[f"lp_high_activity_ch{x}" for x in (1,2,3)]].to_numpy(),axis=1)),
                   "dis3":mean(np.max(l3[[f"lp_disconnection_ch{x}" for x in (1,2,3)]].to_numpy(),axis=1)),
                   "channel_rest3":[mean(l3[f"lp_resting_ch{x}"]) for x in (1,2,3)]}
                by[group].append(stat)
            except Exception as e:
                print("ERROR",group,p.name,str(e)[:160],flush=True)
        print("PROGRESS",group,len(by[group]),"elapsed",round(time.monotonic()-start,1),flush=True)
    total={}
    for group,items in by.items():
        out={"n":len(items),"rest_v2_mean":mean([x["rest2"] for x in items]),
           "rest_v3_mean":mean([x["rest3"] for x in items]),
           "files_with_rest_v2":sum(x["rest2"]>.003 for x in items),
           "files_with_rest_v3":sum(x["rest3"]>.003 for x in items),
           "uncertain_v2":mean([x["uncertain2"] for x in items]),
           "uncertain_v3":mean([x["uncertain3"] for x in items]),
           "teacher_weight_v2":mean([x["weight2"] for x in items]),
           "teacher_weight_v3":mean([x["weight3"] for x in items]),
           "gmm_credible_any":sum(x["gmm_credible"]>=1 for x in items),
           "gmm_credible_all":sum(x["gmm_credible"]>=3 for x in items),
           "model_bad_contact_prob_any_gt_half":sum(max(x["quality_probs"])>.5 for x in items),
           "mean_activity_v3":mean([x["active3"] for x in items]),
           "mean_bad_contact_v3":mean([x["dis3"] for x in items]),
           "channel_rest_v3":[mean([x["channel_rest3"][c] for x in items])
                              for c in range(3)]}
        print("GROUP_V2_V3",group,json.dumps(out),flush=True)
        total[group]=out
    for g in ("STRONG_LONG_PRESS","SOFT_LONG_PRESS","MID_LONG_PRESS","CONNECTED","DISCONNECTED"):
        print("FILE_CASES",g,json.dumps(by[g][:30]),flush=True)
    print("V3_COMPARISON_DONE",round(time.monotonic()-start,1),flush=True)
    Path("/tmp/lp_v3_v2_comparison.json").write_text(json.dumps(
        {"teacher":"offline lp-v3 vs causal v2","contact_model":quality,
         "groups":total,"examples":{g:by[g][:30] for g in total}},indent=2))

if __name__=="__main__":main()
