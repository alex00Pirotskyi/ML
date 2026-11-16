#!/usr/bin/env python3
"""Reproducible synthetic integration test of the entire V6 trainer/export contract.

Exercises source SHA binding, timestamp joins, subject split, Optuna dual
objectives, ten unique model headers+JSON, export_plus parity and C stub compile.
Synthetic tests are NOT evidence of real-world tap classification accuracy.
"""
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from train_optuna_student_v6 import EMG,WL,ACC,ALL_CLASSES

def source_hash(path):
    h=hashlib.sha256()
    with gzip.open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def make_fake_corpus(root,labels,subjects=14):
    rng=np.random.default_rng(1984)
    manifest=[]
    for cat in sorted(ALL_CLASSES):
        for person in range(subjects):
            name=f"person{person:02d}_{cat}_"+("SoftTap" if cat=="TAP" and person%2==0 else "DoubleTaps" if cat=="TAP" else "Default")
            src=root/cat/(name+".csv.gz")
            dst=labels/cat/(name+".labels.csv.gz")
            src.parent.mkdir(parents=True,exist_ok=True)
            dst.parent.mkdir(parents=True,exist_ok=True)
            n=100
            t=1_700_000_000_000 + person*500_000 + np.arange(n)*10
            positive=cat in ("TAP","SWIPE")
            activation=(np.arange(n)>=20)&(np.arange(n)<60)
            features={}
            for i,k in enumerate(EMG):
                level=720+18*i+12*person
                features[k]=(level+rng.normal(0,60,n)+
                    (350 if positive else 65)*activation + (20 if cat=="FINGER" else 0)).clip(0).astype(int)
            for i,k in enumerate(WL):
                features[k]=(200+i*10+rng.normal(0,20,n)+
                    (70 if positive else 25)*activation).clip(0).astype(int)
            for i,k in enumerate(ACC):
                features[k]=(310+i*3+rng.normal(0,20,n)).clip(0).astype(int)
            raw=pd.DataFrame({"Timestamp":t,"User":[f"subject{person:02d}"]*n,**features})
            raw.to_csv(src,index=False,compression="gzip")
            yy=np.zeros(n,dtype=np.int8)
            if positive:yy[20:60]=1
            yy[:5]=-1
            weight=np.where(yy==-1,0.,np.where(yy==1,.85,.9))
            ev=np.where(yy==1,1,-1)
            lab=pd.DataFrame({"Timestamp":t,"gesture_label":yy,
                              "gesture_weight":weight,
                              "gesture_kind":np.where(yy==1,1 if cat=="TAP" else 2,0),
                              "gesture_event_id":ev})
            lab.to_csv(dst,index=False,compression="gzip")
            manifest.append({"status":"OK","source":f"{cat}/{src.name}","category":cat,
                             "source_sha256":source_hash(src)})
    (labels/"manifest.jsonl").write_text("".join(json.dumps(x)+"\n" for x in manifest))
    return manifest

def main():
    with tempfile.TemporaryDirectory(prefix="tap_v6_smoke_") as base:
        base=Path(base)
        root=base/"DATA"/"DATA_TAP"
        label=base/"v5_labels"
        root.mkdir(parents=True)
        label.mkdir(parents=True)
        manifest=make_fake_corpus(root,label)
        target=base/"results"
        cmd=[sys.executable,str(Path(__file__).with_name("train_optuna_student_v6.py")),
             "--input-dir",str(root),"--labels-dir",str(label),"--out-dir",str(target),
             "--trials","10","--cv-folds","2","--top-k","10","--max-per-class","15",
             "--seed","456","--test-fraction",".28"]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=450)
        if p.returncode:
            print("TRAIN_STDOUT",p.stdout[-12000:],flush=True)
            print("TRAIN_STDERR",p.stderr[-12000:],flush=True)
            raise RuntimeError("V6 synthetic end-to-end training/export failed")
        summary=json.loads((target/"summary.json").read_text())
        assert summary["selected_models"]==10,summary
        assert len(summary["ranking"])==10
        assert len(set(x["trial"] for x in summary["ranking"]))==10
        assert (target/"optuna_trials.json").is_file()
        assert (target/"optuna_study.sqlite3").is_file()
        assert len(list((target/"models").glob("*.h")))==10
        assert len(list((target/"models").glob("*.json")))==20
        for model in summary["ranking"]:
            assert model["actual_static_bytes"]>0
            assert model["c_compiles"]=="passed"
            out=json.loads((target/"models"/model["json"]).read_text())
            assert out["export_plus"]["task"]=="BINARY_CLASSIFIER"
            assert out["export_plus"]["total_static_data_bytes"]==model["actual_static_bytes"]
            assert out["heldout_subject_silver_metrics_before_full_refit"]["f1"]>=0
            assert out["export_plus"].get("validation") is not None
        experiment=json.loads((target/"experiment_manifest.json").read_text())
        assert not set(experiment["train_subjects"])&set(experiment["heldout_subjects"])
        print("ALL_V6_TESTS_PASSED",json.dumps({"generated_sources":len(manifest),
            "trials":summary["complete_trials"],"models":10,
            "bytes":[r["actual_static_bytes"] for r in summary["ranking"]],
            "full_end_to_end":True}),flush=True)

if __name__=="__main__":
    main()
