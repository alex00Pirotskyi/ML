"""Focused training/export invariants; not a replacement for real firmware A/B."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from export_plus import estimate_compact_tree_payload
from train_lp_v5_optuna import (
    FEATURES, SETS, compile_stub, emit_export, grouped_cv,
    objective_builder, rank_candidates, raw_features, score_predictions,
    weighted_fit_weights
)


def synthetic(n=450, seed=8):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 5000, (n, len(FEATURES))).astype(np.float32)
    y = (X[:, 0] + 0.25 * X[:, 1] < 2300).astype(np.int8)
    group = np.asarray([f"user_{i//30}" for i in range(n)], dtype=object)
    w = np.ones(n, dtype=np.float32)
    return X, y, w, group


class TestV5EmbeddedTrainer(unittest.TestCase):
    def test_exact_schema_is_unmodified_firmware_expressions(self):
        self.assertEqual([len(SETS[n]) for n in ("core18","instant33","full39")],[18,33,39])
        self.assertEqual(len(FEATURES),39)
        self.assertTrue(all(name.startswith("p_core->") for name in FEATURES))
        self.assertFalse(any("log" in name or "ema" in name.lower() for name in FEATURES))

    def test_encoding_transform_only_with_marker(self):
        n=20
        d={k:np.full(n,20,dtype=float) for k in FEATURES}
        base=pd.DataFrame(d)
        raw=raw_features(base)
        old=base.copy()
        for ch in (1,2,3):
            for z,cnt in enumerate((8,10,12,46,51)):
                for key in (f"p_core->emg_psd_mean[CH{ch}][{z}]",
                            f"p_core->emg_psd[0][CH{ch}][{z}]"):
                    old[key]=old[key]/cnt
        old["psd_mean_applied"]=1
        np.testing.assert_allclose(raw,raw_features(old),atol=.0002)
        old.loc[0,"psd_mean_applied"]=0
        with self.assertRaises(ValueError):raw_features(old)

    def test_unknown_is_not_valid_binary(self):
        y=np.array([0,1,0,1],dtype=np.int8)
        w=weighted_fit_weights(y,np.ones(4,np.float32))
        self.assertEqual(w.shape,(4,))
        self.assertAlmostEqual(float(w[0]),1.)
        metric=score_predictions(y,np.array([.1,.8,.6,.3]),.65)
        self.assertLess(metric["hold_recall"],1.)
        self.assertGreaterEqual(metric["score"],0.)

    def test_subject_split_does_not_mix_people(self):
        X,y,w,uid=synthetic()
        for tr,te in grouped_cv(X,y,w,uid,3):
            self.assertFalse(set(uid[tr]).intersection(set(uid[te])))

    def test_multiobjective_is_score_and_real_static_bytes(self):
        X,y,w,uid=synthetic()
        groups=np.array(["SOFT_LONG_PRESS"]*len(y))
        paths=np.array([f"/fake/{i//30}.csv.gz" for i in range(len(y))],dtype=object)
        splits=grouped_cv(X,y,w,uid,3)
        study=optuna.create_study(
            directions=["maximize","minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=8)
        )
        objective=objective_builder((X,y,w,uid,groups,paths),splits,workers=1,seed=4,max_trees=32)
        study.optimize(objective,n_trials=2)
        self.assertEqual(len(study.trials),2)
        self.assertTrue(all(np.isfinite(t.values).all() for t in study.trials))
        self.assertTrue(all(t.values[1]>100 for t in study.trials))
        self.assertTrue(len(rank_candidates(study,100000))>=1)

    def test_export_plus_header_parity_and_host_compilation(self):
        X,y,_,_=synthetic()
        model=lgb.LGBMClassifier(
            objective="binary", n_estimators=12,num_leaves=6,
            max_depth=4,min_child_samples=10,learning_rate=.08,
            verbosity=-1,n_jobs=1,random_state=12
        )
        idx=SETS["core18"]
        model.fit(X[:,idx],y)
        expected=estimate_compact_tree_payload(model)["total_static_data_bytes"]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            params={"trees":12,"leaves":6,"depth":4,"min_child":10,"lr":.08,
                    "colsample":1.,"subsample":1.,"l2":1.,"threshold":.65,
                    "feature_set":"core18"}
            report=emit_export(path,model,params,.65,idx,30000)
            self.assertEqual(report["total_static_data_bytes"],expected)
            self.assertTrue((path/"lp_v5_model.h").is_file())
            self.assertTrue((path/"lp_v5_model.pkl").is_file())
            self.assertEqual(report["validation"]["decision_disagreements"],0)
            compiler=compile_stub(path/"lp_v5_model.h")
            self.assertTrue(compiler["compiled"])


if __name__=="__main__":
    unittest.main(verbosity=2)
