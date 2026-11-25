#!/usr/bin/env python3
"""Does Lean-8 improve TAP / LP? Controlled feature-set ablation with the unified trainer's data and replay.

    python -u research/gesture/ablate_lean8.py tap
    python -u research/gesture/ablate_lean8.py lp

Same people folds, same model configurations, same 32 KB budget trimming, same replay scoring; only the
feature set changes. Calibration (positive weight) is held fixed, so besides the firmware threshold 0.85
each model is also scored across a threshold sweep: "best" = what the trainer's positive-weight search can
reach. Analysis only; nothing here changes the fixed 0.85 used on the device.
"""
from __future__ import annotations

import sys as _sys
_sys.exit(f"{__file__}: retired. It folds by person, which is not allowed; research and training split by recording file only (tap_generalization_research.py, train_unified.py).")

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_unified as g
import replay as R

TASK = sys.argv[1] if len(sys.argv) > 1 else "tap"
OUT = Path(os.environ.get("MUDRA_ABLATION_OUT", g.ROOT / "research" / "gesture" / "results_lean8_ablation"))
if os.environ.get("MUDRA_LABEL_ROOT"):
    g.LABEL_TAP = Path(os.environ["MUDRA_LABEL_ROOT"]) / "DATA_TAP"
    g.LABEL_LP = Path(os.environ["MUDRA_LABEL_ROOT"]) / "DATA_LONG_PRESS"
BUDGET = int(g.MODEL_BUDGET_KB * 1024)
FOLDS = 3
SETS = ["raw126", "lean8_24", "raw126_lean8_instant12", "all_available"]
if os.environ.get("MUDRA_ABLATION_SETS"):
    SETS = os.environ["MUDRA_ABLATION_SETS"].split(",")
BASE = dict(objective="binary", n_estimators=400, learning_rate=0.1, min_child_samples=100, reg_lambda=1.0,
            reg_alpha=1e-3, colsample_bytree=0.7, subsample=0.8, subsample_freq=1, max_bin=127, verbosity=-1,
            n_jobs=g.CPU_THREADS, random_state=g.SEED, deterministic=True)
CONFIGS = {"15leaf_d6": dict(BASE, num_leaves=15, max_depth=6), "31leaf_d8": dict(BASE, num_leaves=31, max_depth=8)}
POS_WEIGHT = dict(tap=1.0, lp=6.0)
if os.environ.get("MUDRA_ABLATION_PW"):
    POS_WEIGHT[TASK] = float(os.environ["MUDRA_ABLATION_PW"])
if os.environ.get("MUDRA_ABLATION_CONFIGS"):
    CONFIGS = {k: v for k, v in CONFIGS.items() if k in os.environ["MUDRA_ABLATION_CONFIGS"].split(",")}
TAP_GRID = np.round(np.r_[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999, 0.9995], 4)
LP_GRID = np.round(np.r_[np.arange(0.40, 0.951, 0.05), 1.0], 3)   # Q20 average: 1.0 = all ten samples at 20


def first(mask, a, b):
    j = np.flatnonzero(mask[a:b])
    return a + int(j[0]) if len(j) else None


def tap_eval(model, k, sel, rpl, people):
    recs = rpl["recs"]; pick = recs[recs.group.isin(people)]
    res = {v: {thr: R.TapCounts() for thr in TAP_GRID} for v in ("usable", "steady")}
    for i, r in pick.iterrows():
        a, b = int(r.start), int(r.end)
        a0 = first(rpl["ready"], a, b)
        if a0 is None:
            continue
        p = model.booster_.predict(np.ascontiguousarray(np.asarray(rpl["X"][a0:b])[:, sel]), num_iteration=k)
        ons = np.asarray(rpl["onsets"].get(int(i), []), np.float64)
        a1 = first(rpl["steady"], a0, b)
        for thr in TAP_GRID:
            d = p >= thr
            o = ons[ons >= rpl["t"][a0] + R.TAP_HIT_BEFORE_MS]
            R.tap_replay_recording(d, rpl["t"][a0:b], rpl["label"][a0:b], o, r.folder, res["usable"][thr])
            if a1 is not None:
                o = ons[ons >= rpl["t"][a1] + R.TAP_HIT_BEFORE_MS]
                R.tap_replay_recording(d[a1 - a0:], rpl["t"][a1:b], rpl["label"][a1:b], o, r.folder, res["steady"][thr])
    return res


def lp_eval(model, k, sel, rpl, people):
    w = rpl["windows"]; recs = rpl["recs"]
    keep = w[np.isin(recs.group.to_numpy(object)[w.rec.to_numpy(int)], list(people))]
    res = {v: {thr: R.LpCounts() for thr in LP_GRID} for v in ("usable", "steady")}
    for r in keep.itertuples():
        if not rpl["ready"][int(r.start)]:
            continue
        a, b = int(r.start), int(r.end)
        q = R.q20(model.booster_.predict(np.ascontiguousarray(np.asarray(rpl["X"][a:b])[:, sel]), num_iteration=k))
        folder = recs.folder.iloc[int(r.rec)]
        for thr in LP_GRID:
            R.FIXED_THRESHOLD = float(thr)
            R.lp_replay_press(q, rpl["t"][a:b], r.start_ms, r.release_ms, r.kind, folder, res["usable"][thr])
            if rpl["steady"][a]:
                R.lp_replay_press(q, rpl["t"][a:b], r.start_ms, r.release_ms, r.kind, folder, res["steady"][thr])
    R.FIXED_THRESHOLD = 0.85
    return res


def summarize(counts_by_thr):
    rows = {float(t): c.report() for t, c in counts_by_thr.items()}
    at = rows[0.85]
    best_t = max(rows, key=lambda t: rows[t]["objective_score"])
    out = dict(at_085={k: v for k, v in at.items() if k != "per_folder"},
               best_threshold=best_t, best={k: v for k, v in rows[best_t].items() if k != "per_folder"})
    if TASK == "tap":
        for target in (0.5, 1.0, 2.0):
            ok = [r["recall"] for r in rows.values() if r["false_per_min"] <= target]
            out[f"recall_at_{target}_false_per_min"] = max(ok) if ok else 0.0
    return out


def merge(dst, src):
    for v in src:
        for thr, c in src[v].items():
            dst.setdefault(v, {}).setdefault(thr, (R.TapCounts() if TASK == "tap" else R.LpCounts())).add(c)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files_t = g.canonical_files(g.RAW_SOURCE_TAP); files_l = g.canonical_files(g.RAW_SOURCE_LP)
    data, rpl, info = g.build_task_data(TASK, g.selected_sources(TASK, files_t, files_l), OUT / TASK)
    groups = data["group"]; ids = np.arange(len(data["y"]))
    folds = list(GroupKFold(n_splits=FOLDS).split(ids, data["y"], groups))
    results = {}
    for cfg_name, params in CONFIGS.items():
        for set_name in SETS:
            cols = data["feature_sets"][set_name]; sel = [data["columns"].index(c) for c in cols]
            pooled, per_fold, t0 = {}, [], time.monotonic()
            for f, (tr, va) in enumerate(folds):
                model = g.fit_model(data["X"][tr][:, sel], data["y"][tr],
                                    g.weights_for_task(data, tr, TASK, POS_WEIGHT[TASK]), params, BUDGET)
                k, size = g.trees_within(model, BUDGET)
                people = set(groups[va])
                res = (tap_eval if TASK == "tap" else lp_eval)(model, k, sel, rpl, people)
                merge(pooled, res)
                per_fold.append(dict(fold=f, trees=k, bytes=size, usable=summarize(res["usable"]),
                                     steady=summarize(res["steady"])))
            results[f"{cfg_name}/{set_name}"] = dict(
                features=len(cols), seconds=round(time.monotonic() - t0, 1),
                usable=summarize(pooled["usable"]), steady=summarize(pooled["steady"]), folds=per_fold)
            u = results[f"{cfg_name}/{set_name}"]["usable"]
            print("ABLATION", TASK, cfg_name, set_name, json.dumps(dict(
                at_085=round(u["at_085"]["objective_score"], 4), best=round(u["best"]["objective_score"], 4),
                best_thr=u["best_threshold"], fold_best=[round(x["usable"]["best"]["objective_score"], 4) for x in per_fold],
                seconds=results[f"{cfg_name}/{set_name}"]["seconds"])), flush=True)
            (OUT / f"{TASK}_ablation.json").write_text(json.dumps(dict(task=TASK, gate=info.get("lean8_gate"),
                                                                       budget=BUDGET, results=results), indent=1, default=float))


if __name__ == "__main__":
    main()
