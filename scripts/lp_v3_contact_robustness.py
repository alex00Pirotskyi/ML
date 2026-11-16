#!/usr/bin/env python3
"""Subject held-out contact model with weak responsive-muscle negative examples.

Weak LP examples are NOT physical verified contact truth.
Only CONNECTED (all CH) vs DISCONNECTED (CH1/2) counted for proxy CV.
Held-out subjects also excluded from weak-LP augmentation training.
"""
from collections import defaultdict
from pathlib import Path
import json,time
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lp_v3_feature_probe import record, ROOT

GROUPS=("CONNECTED","DISCONNECTED","SOFT_LONG_PRESS","MID_LONG_PRESS",
        "STRONG_LONG_PRESS","LONG_PRESS","DRAG_AND_DROP","LOOSE","TABLE")
LP=set(GROUPS[2:6])
F_INDICES=list(range(26))

def make_model(kind):
    if kind=="tree":
        return ExtraTreesClassifier(n_estimators=220,max_depth=5,min_samples_leaf=3,
            class_weight="balanced",n_jobs=2,random_state=13)
    return make_pipeline(StandardScaler(),LogisticRegression(
        C=.1,class_weight="balanced",max_iter=1000))

def main():
    start=time.monotonic()
    by=defaultdict(list)
    for g in GROUPS:
        for path in sorted((ROOT/g).glob("*.csv.gz")):
            try:
                rec=record(path,g)
                if rec:by[g].append(rec)
            except Exception as e:print("ERROR",g,path.name,str(e)[:120],flush=True)
        print("SCANNED",g,len(by[g]),flush=True)
    items=[]
    for g,recs in by.items():
        for rec in recs:
            for ch in range(3):
                truth=(0 if g=="CONNECTED" else 1 if g=="DISCONNECTED" and ch<2 else -1)
                usable_weak=(g in LP and rec["ch"][ch]["gmm_credible"])
                items.append({"user":rec["user"],"group":g,"channel":ch,
                     "features":rec["ch"][ch]["feat"],"label":truth,
                     "weak_good":usable_weak})
    X=np.asarray([x["features"] for x in items],float)
    users=np.asarray([x["user"] for x in items])
    labels=np.asarray([x["label"] for x in items])
    weak=np.asarray([x["weak_good"] for x in items])
    groups=np.asarray([x["group"] for x in items])
    print("COUNTS",json.dumps({"items":len(items),"people":len(set(users)),
       "labeled_contact":int(sum(labels>=0)),"weak_LP":int(sum(weak))}),flush=True)
    split=list(GroupKFold(n_splits=5).split(X,groups=users))
    for modelname in ("linear","tree"):
        for w in (0.,.12,.30,.70,1.):
            pred=np.zeros(len(items),float)
            fold=[]
            for fi,(tr,te) in enumerate(split):
                tr_labeled=tr[labels[tr]>=0]
                if w>0:
                    tr_extra=tr[weak[tr]]
                    train=np.concatenate([tr_labeled,tr_extra])
                    y=np.concatenate([labels[tr_labeled],np.zeros(len(tr_extra),int)])
                    weight=np.concatenate([np.ones(len(tr_labeled)),np.full(len(tr_extra),w)])
                else:
                    train=tr_labeled
                    y=labels[train]
                    weight=np.ones(len(train))
                if len(np.unique(y))<2:raise ValueError("no contacts in fold")
                model=make_model(modelname)
                if modelname=="linear":
                    model.fit(X[train],y,logisticregression__sample_weight=weight)
                else:
                    model.fit(X[train],y,sample_weight=weight)
                pred[te]=model.predict_proba(X[te])[:,1]
                fold.append({"test_users":len(set(users[te])),"n_contact_test":int(sum(labels[te]>=0)),
                    "n_weak_train":int(sum(weak[tr]))})
            m=labels>=0;ty=labels[m];pr=pred[m]
            proxy={"n":int(sum(m)),"auc":float(roc_auc_score(ty,pr)),
                "ba":float(balanced_accuracy_score(ty,pr>=.5)),
                "confusion":confusion_matrix(ty,pr>=.5,labels=[0,1]).tolist()}
            out={}
            for g,recs in by.items():
                vals=pred[groups==g].reshape(-1,3)
                out[g]={"n_files":len(recs),"any_flag":float(np.mean(np.any(vals>=.5,axis=1))),
                    "fraction_each":[float(np.mean(vals[:,c]>=.5)) for c in range(3)],
                    "median_p_each":[float(np.median(vals[:,c])) for c in range(3)]}
            print("AUGMENTED_CV",modelname,w,json.dumps({"contact_proxy":proxy,
                  "ood_flags":out,"folds":fold}),flush=True)
    print("CONTACT_AUGMENT_DONE",round(time.monotonic()-start,1),flush=True)

if __name__=="__main__":main()
