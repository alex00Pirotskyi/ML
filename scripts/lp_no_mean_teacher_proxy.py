#!/usr/bin/env python3
"""Weak-context sanity checks for LP pseudo-labels, NOT physiological accuracy."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,confusion_matrix
from lp_rest_labeler import read_csv_features,compute_labels
from lp_causal_fw_teacher import step_ms

ROOT=Path("DATA/DATA_LONG_PRESS")
GROUPS=("CONNECTED","DISCONNECTED","LOOSE","TABLE","NOTHING",
        "MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS",
        "DRAG_AND_DROP")
LP_GROUPS={"MID_LONG_PRESS","SOFT_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS"}

def mean(a):
    a=np.asarray(a,dtype=float)
    return float(np.mean(a)) if len(a) else 0.
def q(a,k):
    return float(np.percentile(a,k)) if len(a) else 0.

def profile(path,group):
    df=read_csv_features(path)
    y,meta=compute_labels(df)
    dt=step_ms(df)
    idx=max(0,int(round(5000./dt)))
    tail=y.iloc[min(idx,len(y)-1):]
    result={"group":group,"file":path.name,"n":len(df),"duration_sec":len(df)*dt/1000,
      "lp_state_rest_frac":mean(y["lp_state"]),
      "lowweight_frac":mean(y["lp_state_weight"]<.05),
      "uncertain_frac":mean(np.max(y[[f"lp_unknown_ch{i}" for i in range(1,4)]].to_numpy(),axis=1)>.4)}
    rest=[];activity=[];disconn=[];unknown=[];calibrated=[]
    for ch in range(1,4):
        rest.append(mean(tail[f"lp_resting_ch{ch}"]))
        activity.append(q(tail[f"lp_high_activity_ch{ch}"],90))
        disconn.append(mean(tail[f"lp_disconnection_ch{ch}"]))
        unknown.append(mean(tail[f"lp_unknown_ch{ch}"]))
        calibrated.append(meta["calibrated_fraction_by_channel"][ch-1])
    result.update(rest=rest,activity_p90=activity,disconn_mean=disconn,
                  unknown_mean=unknown,calibrated=calibrated,
                  activity_p90_max=max(activity))
    return result

def category_summary(g,rows):
    return {
      "n":len(rows),
      "rest_frac_median":q([r["lp_state_rest_frac"] for r in rows],50),
      "rest_frac_mean":mean([r["lp_state_rest_frac"] for r in rows]),
      "high_activity_p90_by_channel_median":[q([r["activity_p90"][i] for r in rows],50) for i in range(3)],
      "disconn_mean_by_channel_median":[q([r["disconn_mean"][i] for r in rows],50) for i in range(3)],
      "unknown_mean_by_channel_median":[q([r["unknown_mean"][i] for r in rows],50) for i in range(3)],
      "calibrated_fraction_by_channel_median":[q([r["calibrated"][i] for r in rows],50) for i in range(3)],
      "low_weight_fraction_median":q([r["lowweight_frac"] for r in rows],50),
    }

def main():
    rows=[]
    for group in GROUPS:
        ps=sorted((ROOT/group).glob("*.csv.gz"))
        for path in ps:
            try:rows.append(profile(path,group))
            except Exception as exc:print("ERROR",group,path.name,f"{type(exc).__name__}: {exc}",flush=True)
        print("PROGRESS",group,len([r for r in rows if r["group"]==group]),flush=True)
    by={g:[r for r in rows if r["group"]==g] for g in GROUPS}
    for g,collection in by.items():
        print("CONTEXT_SUMMARY",g,json.dumps(category_summary(g,collection)),flush=True)
    # Contact proxy: all 3 of CONNECTED class are nominally good, only CH1/2
    # in DISCONNECTED folder provisionally disconnected. CH3 is excluded.
    y=[];probs=[];groups=[]
    for row in by["CONNECTED"]:
        for ch in range(3):
            y.append(0);probs.append(row["disconn_mean"][ch]);groups.append("connected")
    for row in by["DISCONNECTED"]:
        for ch in (0,1):
            y.append(1);probs.append(row["disconn_mean"][ch]);groups.append("disconnected")
    pred=(np.asarray(probs)>=.5).astype(int)
    contact={"n":len(y),"positive":int(sum(y)),
      "auc":float(roc_auc_score(y,probs)),
      "balanced_accuracy_fixed_0p5":float(balanced_accuracy_score(y,pred)),
      "cm_0_connected_1_disconnected":confusion_matrix(y,pred,labels=[0,1]).tolist(),
      "note":"weak folder/channel proxies, not verified impedance ground truth"}
    print("CONTACT_CONTEXT_PROXY",json.dumps(contact),flush=True)
    # LP vs relaxed connected category is session classification, not press events
    yy=[];score=[]
    for row in by["CONNECTED"]:
        yy.append(0);score.append(row["activity_p90_max"])
    for group in LP_GROUPS:
        for row in by[group]:
            yy.append(1);score.append(row["activity_p90_max"])
    pred=(np.asarray(score)>=.5).astype(int)
    task={"n":len(yy),"LP_sessions":sum(yy),
          "auc":float(roc_auc_score(yy,score)),
          "balanced_accuracy_fixed_0p5":float(balanced_accuracy_score(yy,pred)),
          "cm_0_CONNECTED_1_LP":confusion_matrix(yy,pred,labels=[0,1]).tolist(),
          "note":"session context, not per-window physiological high-activity truth"}
    print("ACTIVITY_CONTEXT_PROXY",json.dumps(task),flush=True)
    report={"dataset_head":"36f02ac","teacher":"lp_causal_fw_v2",
      "scope":"all 101 LP + 39 connected + 42 disconnected + 42 loose + 9 table + 5 nothing +37 drag",
      "by_group":{g:category_summary(g,by[g]) for g in GROUPS},
      "contact_proxy":contact,"lp_activity_proxy":task}
    Path("/tmp/lp_no_mean_teacher_proxy.json").write_text(json.dumps(report,indent=2))
    print("TEACHER_PROXY_DONE",flush=True)

if __name__=="__main__":main()
