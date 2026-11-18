#!/usr/bin/env python3
"""Evidence runs for FEATURE_SELECTION_RESEARCH_AND_EXECUTION_PLAN.md on the full data (development people only;
the 22% person holdout is never touched).

    python research/gesture/fs_research.py gate tap|lp     # train frame gate 50 vs 500, scored on all usable and steady rows
    python research/gesture/fs_research.py rank tap|lp     # fold-local nested rankings for the 3 trainer CV folds
    python research/gesture/fs_research.py kcurve tap|lp   # fixed-config models at K = 5..150 from each fold's ranking

Environment: MUDRA_BENCH_LABEL_ROOT / MUDRA_BENCH_OUT (as bench_unified.py), MUDRA_FS_OUT (results).
Analysis-only threshold sweeps are reported next to the firmware threshold; nothing here tunes a threshold.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# These experiments define their own gates (rows from frame 50 are usable; steady = frame 500 reported
# separately) and their own operating points: run the trainer module in its v2 data semantics.
os.environ.setdefault("MUDRA_EXPERIMENT", "v2")
import bench_unified as B          # noqa: E402
import train_unified as g          # noqa: E402
import replay as R                 # noqa: E402
import feature_selection as FS     # noqa: E402

OUT = Path(os.environ.get("MUDRA_FS_OUT", HERE / "results_fs_research"))
CONFIG = dict(objective="binary", n_estimators=400, learning_rate=0.1, num_leaves=31, max_depth=8,
              min_child_samples=100, reg_lambda=1.0, reg_alpha=1e-3, colsample_bytree=0.7, subsample=0.8,
              subsample_freq=1, max_bin=127, verbosity=-1, deterministic=True)
PW = dict(tap=1.0, lp=0.02)
K_GRID = tuple(int(k) for k in os.environ.get("MUDRA_FS_KGRID", "5,10,15,20,30,40,60,80,100,125,150").split(","))
QUICK = os.environ.get("MUDRA_FS_QUICK") == "1"


def log(*a):
    print(*a, flush=True)


def slim(r):
    return None if r is None else {k: v for k, v in r.items() if k != "per_folder" and not isinstance(v, (list, dict))}


def summarize(ev, p, task, rows=None):
    grid = FS.TAP_GRID if task == "tap" else FS.LP_GRID
    reps = {thr: ev.report(p, thr, rows) for thr in grid}
    dep = g.task_threshold(task)
    at = reps[dep] if dep in reps else ev.report(p, dep, rows)
    ok = {k: v for k, v in reps.items() if v}
    best = max(ok, key=lambda k: ok[k]["objective_score"]) if ok else None
    out = dict(at_firmware_threshold=slim(at), best_threshold=best, best=slim(ok.get(best)))
    if task == "tap":
        for tgt in (1, 2, 3, 5):
            hits = [v["recall"] for v in ok.values() if v["false_per_min"] <= tgt]
            out[f"recall_at_fpm_{tgt}"] = max(hits) if hits else 0.0
        if at:
            out["per_folder_at_firmware"] = {f: dict(fpm=round(v["false_per_min"], 3), recall=v["recall"])
                                             for f, v in (at.get("per_folder") or {}).items()}
    return out


def setup(task):
    data, rpl, tr, te, budgets, _ = B.setup(task)
    folds = list(GroupKFold(n_splits=g.CV_FOLDS).split(tr, data["y"][tr], data["group"][tr]))
    if QUICK:                                    # a few people only: a smoke test of the research code
        people = sorted(set(data["group"][tr]))[:9]
        tr = tr[np.isin(data["group"][tr], people)]
        folds = list(GroupKFold(n_splits=3).split(tr, data["y"][tr], data["group"][tr]))
    return data, rpl, tr, te, budgets, folds


def fit(data, rows, sel, task, budget, pw=None):
    params = dict(CONFIG, n_jobs=g.CPU_THREADS, random_state=g.SEED)
    if QUICK:
        params["n_estimators"] = 40
    X = data["X"][np.ix_(rows, np.asarray(sel, np.intp))]
    m = g.fit_model(X, data["y"][rows], g.weights_for_task(data, rows, task, PW[task] if pw is None else pw), params, budget)
    k, size = g.trees_within(m, budget)
    return m, k, size


def gate(task):
    data, rpl, tr, te, budgets, folds = setup(task)
    sets = {s: [data["columns"].index(c) for c in data["feature_sets"][s]] for s in ("raw126", "all_available")}
    res = {}
    for f, (a, b) in enumerate(folds):
        trn = tr[a]; val = set(data["group"][tr[b]])
        ev = FS.InnerReplay(g, R, rpl, val, task, R.TAP_FALSE_PER_MIN_TARGET)
        for name, sel in sets.items():
            for gmin in (g.LEAN8_MIN_FRAMES, g.LEAN8_STEADY_FRAMES):
                rows = trn[data["frames"][trn] >= gmin]
                t0 = time.perf_counter()
                m, k, size = fit(data, rows, sel, task, budgets[0])
                p = ev.probabilities(m, sel, k)
                r = dict(train_rows=int(len(rows)), positives=int((data["y"][rows] == 1).sum()), trees=k, bytes=size,
                         usable_rows=summarize(ev, p, task), steady_rows=summarize(ev, p, task, ev.steady),
                         seconds=round(time.perf_counter() - t0, 1))
                res.setdefault(name, {}).setdefault(str(gmin), []).append(r)
                log("FS_GATE", task, f, name, gmin, json.dumps(dict(rows=r["train_rows"], pos=r["positives"],
                    usable_best=round(r["usable_rows"]["best"]["objective_score"], 4),
                    steady_best=round(r["steady_rows"]["best"]["objective_score"], 4))))
        del ev
    write(f"gate_{task}.json", dict(task=task, config=CONFIG, positive_weight=PW[task], folds=len(folds), results=res))


def rank(task):
    data, rpl, tr, te, budgets, folds = setup(task)
    if QUICK:
        FS.SCREEN_PARAMS["n_estimators"] = 20
    rankings = []
    only = os.environ.get("MUDRA_FS_FOLDS")                      # helper process: compute (cache) these folds only
    for f, (a, b) in enumerate(folds):
        if only and str(f) not in only.split(","):
            continue
        val = set(data["group"][tr[b]])
        r = FS.fit_fold_ranker(g, R, data, rpl, tr[a], task, seed=f, cache_dir=OUT / "rank_cache",
                               tap_target=R.TAP_FALSE_PER_MIN_TARGET, log=log)
        FS.validate_fold_local(r, val)
        rankings.append(r)
        log("FS_RANK_DONE", task, f, r["seconds"], "s; top 10:", r["ordered_features"][:10])
    if only:
        return
    stab = FS.report_stability(rankings)
    write(f"rankings_{task}.json", dict(task=task, rankings=rankings, stability=stab))


def kcurve(task):
    data, rpl, tr, te, budgets, folds = setup(task)
    rankings = json.loads((OUT / f"rankings_{task}.json").read_text())["rankings"]
    res = {}
    rng = np.random.default_rng(7)
    for f, (a, b) in enumerate(folds):
        trn = tr[a]; val = set(data["group"][tr[b]])
        FS.validate_fold_local(rankings[f], val)
        ev = FS.InnerReplay(g, R, rpl, val, task, R.TAP_FALSE_PER_MIN_TARGET)
        plans = [(f"K{k}", FS.select_top_k(rankings[f], k)) for k in (K_GRID if not QUICK else (5, 20, 150))]
        plans += [(f"random{k}", sorted(rng.choice(len(data["columns"]), k, replace=False).tolist())) for k in (20, 40)]
        for name, sel in plans:
            m, k, size = fit(data, trn, sel, task, budgets[0])
            p = ev.probabilities(m, sel, k)
            r = dict(features=[data["columns"][j] for j in sel], trees=k, bytes=size,
                     usable_rows=summarize(ev, p, task), steady_rows=summarize(ev, p, task, ev.steady))
            res.setdefault(name, []).append(r)
            log("FS_K", task, f, name, json.dumps(dict(trees=k, best=round(r["usable_rows"]["best"]["objective_score"], 4),
                at_fw=round((r["usable_rows"]["at_firmware_threshold"] or {}).get("objective_score", 0), 4),
                steady=round(r["steady_rows"]["best"]["objective_score"], 4))))
        del ev
    write(f"kcurve_{task}.json", dict(task=task, config=CONFIG, positive_weight=PW[task], results=res))


def report(task):
    """Markdown tables from the JSON results (mean over the CV folds; each fold = unseen people)."""
    L = []
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    def cell(rs, view, key, sub="best"):
        vals = [((r[view].get(sub) or {}) if sub else r[view]).get(key) for r in rs]
        vals = [v for v in vals if v is not None and v == v]
        return mean(vals)
    gp = OUT / f"gate_{task}.json"
    if gp.is_file():
        G = json.loads(gp.read_text())["results"]
        L += [f"### {task}: training gate (rows from frame 50 vs 500), scored on all usable rows and on steady-state rows", "",
              "| features | train gate | train rows | positives | usable: best score | steady: best score | steady: score at firmware threshold |"
              + (" steady: recall @<=2 false/min |" if task == "tap" else ""), "|---|---|---|---|---|---|---|" + ("---|" if task == "tap" else "")]
        for fs, by in G.items():
            for gate_, rs in by.items():
                row = (f"| {fs} | {gate_} | {mean([r['train_rows'] for r in rs]):.0f} | {mean([r['positives'] for r in rs]):.0f} | "
                       f"{cell(rs, 'usable_rows', 'objective_score'):.3f} | {cell(rs, 'steady_rows', 'objective_score'):.3f} | "
                       f"{cell(rs, 'steady_rows', 'objective_score', 'at_firmware_threshold'):.3f} |")
                if task == "tap":
                    row += f" {mean([r['steady_rows']['recall_at_fpm_2'] for r in rs]):.3f} |"
                L.append(row)
        L.append("")
    kp = OUT / f"kcurve_{task}.json"
    if kp.is_file():
        K = json.loads(kp.read_text())["results"]
        base = K.get("K150")
        L += [f"### {task}: K-curve (fold-local ranking, fixed configuration, retrained per K)", "",
              "| subset | usable: best score | vs K=150 per fold | steady: best score | at firmware threshold |"
              + (" recall @<=1 / @<=2 false/min |" if task == "tap" else ""), "|---|---|---|---|---|" + ("---|" if task == "tap" else "")]
        for name, rs in K.items():
            d = ([round(r["usable_rows"]["best"]["objective_score"] - b["usable_rows"]["best"]["objective_score"], 3)
                  for r, b in zip(rs, base)] if base else [])
            row = (f"| {name} | {cell(rs, 'usable_rows', 'objective_score'):.3f} | {d} | "
                   f"{cell(rs, 'steady_rows', 'objective_score'):.3f} | "
                   f"{cell(rs, 'usable_rows', 'objective_score', 'at_firmware_threshold'):.3f} |")
            if task == "tap":
                row += (f" {mean([r['usable_rows']['recall_at_fpm_1'] for r in rs]):.3f} / "
                        f"{mean([r['usable_rows']['recall_at_fpm_2'] for r in rs]):.3f} |")
            L.append(row)
        L.append("")
    rp = OUT / f"rankings_{task}.json"
    if rp.is_file():
        Rk = json.loads(rp.read_text())
        st = Rk["stability"]
        L += [f"### {task}: ranking stability (Nogueira) of the top-K sets", "",
              "| K | across outer folds | across inner splits |", "|---|---|---|"]
        L += [f"| {k} | {v['outer_folds']:.2f} | {v['inner_splits']:.2f} |" for k, v in st.items() if k.isdigit()]
        L += ["", "Top 15 per outer fold (family: name):", ""]
        for i, r in enumerate(Rk["rankings"]):
            fam = lambda c: ("Lean-8" if "cload" in c else "ACC" if "acc_" in c else "EMG" if "emg" in c else "WL")
            L.append(f"- fold {i} (cut {r['cluster_cut']}, {len(r['clusters'])} clusters, {r['seconds']:.0f} s): "
                     + ", ".join(f"{fam(c)}:{c.replace('p_core->', '')}" for c in r["ordered_features"][:15]))
        L.append("")
    text = "\n".join(L)
    (OUT / f"report_{task}.md").write_text(text)
    print(text)


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))
    log("FS_WROTE", OUT / name)


if __name__ == "__main__":
    dict(gate=gate, rank=rank, kcurve=kcurve, report=report)[sys.argv[1]](sys.argv[2])
