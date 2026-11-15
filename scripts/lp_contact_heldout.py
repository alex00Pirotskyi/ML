#!/usr/bin/env python3
"""Held-out-person OOD quality model comparison: with and without weak LP context.

Cannot validate true per-channel contact on LP/loose/table, which lack verified
impedance labels. OOD flag rates describe model behavior, not true error rates.
"""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,confusion_matrix
from lp_contact_activity_stress import prof, mk, ROOT

GROUPS=["CONNECTED","DISCONNECTED","LOOSE","TABLE","SOFT_LONG_PRESS",
    "MID_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS","DRAG_AND_DROP"]

def main():
    by=defaultdict(list)
    for group in GROUPS:
        for p in sorted((ROOT/group).glob("*.csv.gz")):
            try:
                x=prof(p,group)
                if x:by[group].append(x)
            except Exception as e:print("ERROR",group,p.name,str(e)[:120])
        print("PROGRESS",group,len(by[group]),flush=True)
    items=[]
    for group,rows in by.items():
        for r in rows:
            for ch,x in enumerate(r["ch"]):
                is_labeled=(group=="CONNECTED") or (group=="DISCONNECTED" and ch<2)
                y=1 if group=="DISCONNECTED" and ch<2 else 0
                items.append({"user":r["user"],"group":group,"channel":ch,
                    "x":x["vec"],"y":y,"is_labeled":is_labeled})
    X=np.asarray([a["x"] for a in items],float)
    users=np.asarray([a["user"] for a in items])
    labels=np.asarray([a["y"] for a in items])
    lab=np.asarray([a["is_labeled"] for a in items])
    lp=np.asarray([a["group"] in {"SOFT_LONG_PRESS","MID_LONG_PRESS","STRONG_LONG_PRESS","LONG_PRESS"} for a in items])
    allp={}
    for mode in ["contact_only","with_weak_lp"]:
        predictions=np.zeros(len(items),float)
        folds=[]
        for tr,te in GroupKFold(n_splits=5).split(X,groups=users):
            if mode=="contact_only":
                training=tr[lab[tr]]
            else:
                training=tr[lab[tr] | lp[tr]]
            assert len(np.unique(labels[training]))==2
            model=mk("forest")
            model.fit(X[training],labels[training])
            predictions[te]=model.predict_proba(X[te])[:,1]
            folds.append({"test_users":len(set(users[te])),
              "n_train":len(training),"test_labeled":int(np.sum(lab[te])),
              "test_lp":int(np.sum(lp[te]))})
        allp[mode]=predictions
        y=labels[lab];p=predictions[lab]
        print("OOF_VALIDATION",mode,json.dumps({
          "n":len(y),"people":len(set(users[lab])),
          "auc":float(roc_auc_score(y,p)),
          "balacc":float(balanced_accuracy_score(y,p>=.5)),
          "cm":confusion_matrix(y,p>=.5,labels=[0,1]).tolist(),
          "folds":folds}),flush=True)
        for group in GROUPS:
            mask=np.asarray([r["group"]==group for r in items])
            vals=predictions[mask]
            select=[items[i] for i in np.flatnonzero(mask)]
            rates={}
            for ch in range(3):
                z=vals[np.asarray([x["channel"]==ch for x in select])]
                rates[f"CH{ch+1}"]={"n":len(z),"fraction_disconnected":float(np.mean(z>=.5)),
                  "median_prob":float(np.median(z))}
            print("OOF_OOD",mode,group,json.dumps(rates),flush=True)
    # Compare on identical recording/channel examples by OOF predictions
    compare={}
    for group in GROUPS:
        mask=np.asarray([r["group"]==group for r in items])
        a=allp["contact_only"][mask]
        b=allp["with_weak_lp"][mask]
        compare[group]={"n":len(a),"disconnected_rate_contact_only":float(np.mean(a>=.5)),
         "disconnected_rate_with_lp":float(np.mean(b>=.5)),
         "median_probability_change":float(np.median(b-a))}
    print("COMPARISON",json.dumps(compare),flush=True)
    print("OOF_ROBUST_DONE")

if __name__=="__main__":main()
