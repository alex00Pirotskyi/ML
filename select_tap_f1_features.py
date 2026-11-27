#!/usr/bin/env python3
"""Fast recursive feature elimination (RFE) for the TAP F1 trainer's inputs (build_tap_f1_dataset.py calls it).

On a row sample of the TRAINING folds only (the trainer's test fold never takes part):

  1. screen   the ~5,000 diff / ratio pairs (tap_f1_features.candidates) are computed on the fly, a block at a time,
              never stored for all rows, and scored twice: mutual information with the tap label on 32 quantile bins
              (sees any shape) and the F statistic (the trainer's own ranking); the best N_SCREEN by the mean of the
              two ranks go on, with every base input (raw fields, Lean-lite: few, and a model judges them better than
              one score each). Constant inputs go.
  2. twins    of inputs that move together (|Spearman| >= 0.98 on a sub-sample) the better-screened one stays.
  3. RFE      LightGBM with the legacy model's tree settings (100 trees, 63 bins: fast), each training fold in turn
              held out for validation (4 fits a round); the mean gain importance ranks the inputs and the weakest go:
              a third a round above 200 inputs, a fifth above 120, a tenth below, down to MIN_KEEP.
  4. choice   the smallest round (<= MAX_KEEP inputs) whose validation average precision is within one standard error
              (at least TOL) of the best round: rounds differ by noise of that size.

Output: the kept inputs in candidate order and a report of every round (size, validation average precision +-
standard error, F1 at 0.85). Everything is deterministic for the same sample.

    python select_tap_f1_features.py SAMPLE_DIR OUT.json    # SAMPLE_DIR: base.npy y.npy fold.npy meta.json
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import tap_f1_features as F  # noqa: E402

N_SCREEN = 500             # pairs that go on to the RFE (with every base input)
TWIN_RHO = 0.98
MIN_KEEP = 40
MAX_KEEP = 250             # the trainer selects at most 254
TOL = 0.002                # at least this much average precision a smaller set may lose against the best round
BLOCK = 128
MI_BINS = 32
DECISION_THRESHOLD = 0.85
RFE_PARAMS = dict(objective="binary", n_estimators=100, learning_rate=0.15, num_leaves=28, max_depth=17,
                  min_child_samples=88, lambda_l1=3.454, lambda_l2=7.710, subsample=0.714, subsample_freq=2,
                  colsample_bytree=0.738, min_split_gain=0.0416, scale_pos_weight=7.781, max_bin=63,
                  deterministic=True, force_col_wise=True, verbosity=-1, random_state=42)
SETTINGS = dict(n_screen=N_SCREEN, twin_rho=TWIN_RHO, min_keep=MIN_KEEP, max_keep=MAX_KEEP, tol=TOL, mi_bins=MI_BINS,
                rfe_params=RFE_PARAMS, version="rfe-3")

_G = {}


def _ctx():
    """worker processes from a clean server process (no inherited OpenMP / BLAS threads)"""
    import multiprocessing as mp
    return mp.get_context("forkserver") if "forkserver" in mp.get_all_start_methods() else None


def _init(sample_dir):
    d = Path(sample_dir)
    _G["B"] = np.load(d / "base.npy", mmap_mode="r")
    _G["y"] = np.load(d / "y.npy").astype(np.int64)
    _G["cols"] = json.loads((d / "meta.json").read_text())["base_cols"]
    rng = np.random.default_rng(0)
    _G["sub"] = np.sort(rng.choice(len(_G["y"]), min(len(_G["y"]), 20000), replace=False))


def mutual_info(x, y, sub, bins=MI_BINS):
    """mutual information (nats) of x and the binary y, on quantile bins of x (edges from the rows `sub`)"""
    edges = np.unique(np.quantile(x[sub], np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    if not len(edges):
        return 0.0
    b = np.searchsorted(edges, x, side="right")
    joint = np.bincount(b * 2 + y, minlength=(len(edges) + 1) * 2).reshape(-1, 2).astype(np.float64)
    p = joint / joint.sum()
    q = p.sum(1, keepdims=True) @ p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / q[nz])).sum())


def _screen_block(names):
    from sklearn.feature_selection import f_classif
    X = F.FeatureMaker(_G["cols"], names)(_G["B"])
    y, sub = _G["y"], _G["sub"]
    sd = X.std(0)
    with np.errstate(all="ignore"):
        fs, _ = f_classif(X, y)
    mi = np.array([mutual_info(X[:, j], y, sub) if sd[j] > 0 else 0.0 for j in range(X.shape[1])])
    return mi, np.nan_to_num(fs, nan=0.0, posinf=0.0), sd


def _ranks(v):
    r = np.empty(len(v)); r[np.argsort(-v, kind="stable")] = np.arange(len(v)); return r


def screen(sample_dir, names, workers, log=print):
    """indices into names: every base input (no helper call) first, then the best N_SCREEN pairs"""
    base = [i for i, n in enumerate(names) if not n.startswith(("p_core->diff(", "p_core->ratio("))]
    pairs = [i for i, n in enumerate(names) if n.startswith(("p_core->diff(", "p_core->ratio("))]
    blocks = [[names[i] for i in pairs[k:k + BLOCK]] for k in range(0, len(pairs), BLOCK)]
    with ProcessPoolExecutor(workers, mp_context=_ctx(), initializer=_init, initargs=(str(sample_dir),)) as ex:
        out = list(ex.map(_screen_block, blocks))
    mi = np.concatenate([o[0] for o in out]) if out else np.zeros(0)
    fs = np.concatenate([o[1] for o in out]) if out else np.zeros(0)
    sd = np.concatenate([o[2] for o in out]) if out else np.zeros(0)
    score = -(_ranks(mi) + _ranks(fs)) / 2.0                  # higher is better
    score[sd <= 0] = -np.inf
    order = [pairs[i] for i in np.argsort(-score, kind="stable") if np.isfinite(score[i])][:N_SCREEN]
    log(f"[rfe] screened {len(pairs)} pairs ({int((sd <= 0).sum())} constant): the best {len(order)} go on with "
        f"the {len(base)} base inputs")
    return base + order, dict(mi=mi, f=fs)


def drop_twins(X, sub, rho=TWIN_RHO):
    """indices (into X's columns, in their order) that survive: a column moving with an earlier one goes"""
    R = np.argsort(np.argsort(X[sub], axis=0, kind="stable"), axis=0).astype(np.float32)
    R -= R.mean(0); n = np.linalg.norm(R, axis=0); n[n == 0] = 1.0; R /= n
    C = np.abs(R.T @ R)
    keep = []
    for j in range(X.shape[1]):
        if not keep or C[j, keep].max() < rho:
            keep.append(j)
    return keep


def _fit_score(X, y, train, val, threads):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, f1_score
    m = lgb.LGBMClassifier(**RFE_PARAMS, n_jobs=threads).fit(X[train], y[train])
    p = m.predict_proba(X[val])[:, 1]
    gain = m.booster_.feature_importance(importance_type="gain")
    return (float(average_precision_score(y[val], p)), float(f1_score(y[val], p >= DECISION_THRESHOLD, zero_division=0)),
            gain / max(gain.sum(), 1e-12))


def rfe(X, y, fold, names, threads, log=print):
    """every training fold validates once per round; returns (chosen round, all rounds, validation folds)"""
    vals = sorted(set(int(f) for f in np.unique(fold)))
    splits = ([(np.flatnonzero(fold != v), np.flatnonzero(fold == v)) for v in vals] if len(vals) > 1
              else [(np.arange(len(y)), np.arange(len(y)))])
    alive = list(range(X.shape[1]))
    rounds = []
    while True:
        t1 = time.time()
        res = [_fit_score(X[:, alive], y, tr, va, threads) for tr, va in splits]
        aps = np.array([r[0] for r in res])
        ap = float(aps.mean()); se = float(aps.std(ddof=1) / np.sqrt(len(aps))) if len(aps) > 1 else 0.0
        f1 = float(np.mean([r[1] for r in res]))
        gain = np.mean([r[2] for r in res], axis=0)
        rounds.append(dict(n=len(alive), ap=ap, se=se, f1=f1, ap_per_split=aps.tolist(), inputs=[names[i] for i in alive]))
        log(f"[rfe] {len(alive):4d} inputs: average precision {ap:.4f} +- {se:.4f}, F1 at 0.85 {f1:.4f} "
            f"({time.time() - t1:.0f} s)")
        if len(alive) <= MIN_KEEP:
            break
        drop = max(1, int(len(alive) * (0.33 if len(alive) > 200 else 0.20 if len(alive) > 120 else 0.10)))
        drop = min(drop, len(alive) - MIN_KEEP)
        order = np.argsort(gain, kind="stable")                # weakest first
        gone = set(order[:drop].tolist())
        alive = [a for k, a in enumerate(alive) if k not in gone]
    return choose(rounds), rounds, vals


def choose(rounds):
    """the smallest round (<= MAX_KEEP inputs) whose average precision is within one standard error (at least TOL)
    of the best round (<= MAX_KEEP): rounds differ by noise of that size, so a smaller set inside it is as good"""
    eligible = [r for r in rounds if r["n"] <= MAX_KEEP] or [min(rounds, key=lambda r: r["n"])]
    best = max(eligible, key=lambda r: r["ap"])
    floor = best["ap"] - max(TOL, best.get("se", 0.0))
    return min((r for r in eligible if r["ap"] >= floor), key=lambda r: r["n"])


def select(sample_dir, candidates, out_json=None, workers=None, threads=None, log=print):
    """the kept inputs (candidate order) and the report; sample_dir holds base.npy (rows x base fields, float32),
    y.npy (tap label of the trainer), fold.npy (roadmap fold of every row) and meta.json (base_cols)"""
    t0 = time.time()
    d = Path(sample_dir)
    workers = int(workers or os.environ.get("TAP_F1_RFE_WORKERS") or min(16, os.cpu_count() or 1))
    threads = int(threads or os.cpu_count() or 1)
    _init(d)
    y = _G["y"]; fold = np.load(d / "fold.npy")
    log(f"[rfe] sample: {len(y):,} rows of folds {sorted(set(fold.tolist()))}, {int(y.sum()):,} tap rows; "
        f"{len(candidates)} candidates")
    order, scores = screen(d, list(candidates), workers, log)
    names = [candidates[i] for i in order]
    X = F.FeatureMaker(_G["cols"], names)(_G["B"])
    keep = drop_twins(X, _G["sub"])
    names = [names[i] for i in keep]; X = X[:, keep]
    log(f"[rfe] {len(names)} inputs after dropping near-identical ones (|Spearman| >= {TWIN_RHO}), "
        f"{sum(n in F.LITE_COLUMNS for n in names)} of them Lean-lite")
    chosen, rounds, vals = rfe(X, y, fold, names, threads, log)
    pos = {c: i for i, c in enumerate(candidates)}
    selected = sorted(chosen["inputs"], key=pos.__getitem__)
    report = dict(settings=SETTINGS, sample=dict(rows=int(len(y)), tap_rows=int(y.sum()),
                                                 folds=sorted(set(int(f) for f in fold)), validation_folds=vals),
                  n_candidates=len(candidates), n_screened=len(order), n_after_twins=len(names),
                  rounds=[{k: v for k, v in r.items() if k != "inputs"} for r in rounds], chosen_n=chosen["n"],
                  chosen_ap=chosen["ap"], best_ap=max(r["ap"] for r in rounds if r["n"] <= MAX_KEEP),
                  seconds=round(time.time() - t0, 1),
                  selected=selected,
                  families={k: sum(1 for s in selected if _family(s) == k) for k in
                            ("raw", "lean-lite", "acc diff", "acc ratio", "emg diff", "emg ratio")})
    log(f"[rfe] kept {len(selected)} inputs (best round {report['best_ap']:.4f}, kept {chosen['ap']:.4f} +- {chosen['se']:.4f}) "
        f"{report['families']} in {report['seconds']:.0f} s")
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=1))
    return selected, report


def _family(name):
    if name in F.LITE_COLUMNS:
        return "lean-lite"
    if name.startswith("p_core->diff(") or name.startswith("p_core->ratio("):
        op = "diff" if name.startswith("p_core->diff(") else "ratio"
        return ("emg " if "emg_psd" in name else "acc ") + op
    return "raw"


def report_markdown(report):
    lines = [f"# Feature elimination: {report['chosen_n']} inputs kept of {report['n_candidates']}", "",
             f"Sample: {report['sample']['rows']:,} rows of the training folds {report['sample']['folds']} "
             f"({report['sample']['tap_rows']:,} tap rows); validation folds {report['sample']['validation_folds']}.",
             f"Screened to {report['n_screened']}, {report['n_after_twins']} after dropping near-identical inputs.",
             f"Kept by family: {report['families']}.", "",
             "| inputs | validation average precision (+- standard error) | F1 at 0.85 |", "|---|---|---|"]
    for r in report["rounds"]:
        mark = " **kept**" if r["n"] == report["chosen_n"] else ""
        lines.append(f"| {r['n']}{mark} | {r['ap']:.4f} +- {r.get('se', 0):.4f} | {r['f1']:.4f} |")
    lines += ["", "Kept inputs:", ""] + [f"- `{s}`" for s in report["selected"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("out_json", type=Path)
    a = ap.parse_args(argv)
    meta = json.loads((a.sample_dir / "meta.json").read_text())
    cands = meta.get("candidates") or F.candidates(meta["raw_cols"], wl=False, lite=meta.get("lite", True))
    _, report = select(a.sample_dir, cands, a.out_json)
    a.out_json.with_suffix(".md").write_text(report_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
