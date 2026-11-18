#!/usr/bin/env python3
"""Benchmark + exact-equivalence harness for the unified trainer (TRAINING_PERFORMANCE_OPTIMIZATION_PLAN.md).

Runs the trainer's own objective on frozen trial parameters and records fingerprints of everything the
plan requires to stay identical: per-fold training X/y/weights bytes, serialized LightGBM models, trimmed
tree counts/bytes, every TAP decision sequence / LP Q20 sequence handed to the replay, fold reports and
objective values. The same script works against an older checkout (copy it next to that train_unified.py).

    # frozen parameters: built-in set, or the completed trials of a real study
    python research/gesture/bench_unified.py run tap  /tmp/new_tap.json [--study path/to/optuna_study.sqlite3] [--limit N]
    python research/gesture/bench_unified.py compare /tmp/old_tap.json /tmp/new_tap.json
    python research/gesture/bench_unified.py profile tap          # cProfile of one frozen trial
    python research/gesture/bench_unified.py stages tap           # old vs optimized time of each changed stage
    python research/gesture/bench_unified.py threads tap 8,16,32,76   # fit/predict time + exactness per thread count

Environment: MUDRA_BENCH_LABEL_ROOT (contains DATA_TAP, DATA_LONG_PRESS label dirs),
MUDRA_BENCH_OUT (cache/output root; default research/gesture/_bench).
"""
from __future__ import annotations

import cProfile
import hashlib
import io
import json
import os
import platform
import pstats
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_unified as g          # noqa: E402
import replay as R                 # noqa: E402
import optuna                      # noqa: E402

BASE = dict(reg_lambda=1.0, reg_alpha=1e-3, colsample_bytree=0.7, subsample=0.8)
FROZEN = {
    "tap": [
        dict(BASE, feature_set="raw126", positive_weight=1.0, max_depth=6, num_leaves=15, n_estimators=160,
             learning_rate=0.1, min_child_samples=100),
        dict(BASE, feature_set="all_available", positive_weight=0.35, max_depth=10, num_leaves=127, n_estimators=400,
             learning_rate=0.05, min_child_samples=30, colsample_bytree=0.5),
        dict(BASE, feature_set="acc_current18", positive_weight=3.0, max_depth=8, num_leaves=31, n_estimators=300,
             learning_rate=0.2, min_child_samples=200, subsample=0.6),
        dict(BASE, feature_set="all_available", positive_weight=1.0, max_depth=6, num_leaves=15, n_estimators=160,
             learning_rate=0.1, min_child_samples=100),
    ],
    "lp": [
        dict(BASE, feature_set="raw126", positive_weight=6.0, max_depth=6, num_leaves=15, n_estimators=160,
             learning_rate=0.1, min_child_samples=100),
        dict(BASE, feature_set="all_available", positive_weight=0.5, max_depth=10, num_leaves=127, n_estimators=400,
             learning_rate=0.05, min_child_samples=30, colsample_bytree=0.5),
        dict(BASE, feature_set="raw126_lean8_instant12", positive_weight=2.0, max_depth=8, num_leaves=31,
             n_estimators=300, learning_rate=0.2, min_child_samples=200, subsample=0.6),
        dict(BASE, feature_set="all_available", positive_weight=0.02, max_depth=8, num_leaves=31, n_estimators=200,
             learning_rate=0.1, min_child_samples=100),
    ],
}


def sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if isinstance(a, np.ndarray):
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        else:
            h.update(json.dumps(a, sort_keys=True, default=str).encode())
    return h.hexdigest()[:24]


def setup(task):
    if os.environ.get("MUDRA_BENCH_LABEL_ROOT"):
        root = Path(os.environ["MUDRA_BENCH_LABEL_ROOT"])
        g.LABEL_TAP = root / "DATA_TAP"; g.LABEL_LP = root / "DATA_LONG_PRESS"
    out = Path(os.environ.get("MUDRA_BENCH_OUT", HERE / "_bench")) / task
    t0 = time.perf_counter()
    files_t = g.canonical_files(g.RAW_SOURCE_TAP); files_l = g.canonical_files(g.RAW_SOURCE_LP)
    data, rpl, info = g.build_task_data(task, g.selected_sources(task, files_t, files_l), out)
    build_s = time.perf_counter() - t0
    tr, te = g.split_heldout(data, task)
    budgets = [b for b in (g.budget_bytes(g.MODEL_BUDGET_KB), g.budget_bytes(g.ALT_BUDGET_KB)) if b]
    return data, rpl, tr, te, budgets, build_s


def frozen_params(task, study=None, limit=None):
    if not study:
        rows = FROZEN[task]
    else:
        storage = "sqlite:///" + str(Path(study).resolve())
        names = optuna.study.get_all_study_names(storage)
        name = next(n for n in names if n.startswith(f"unified_v2_{task}"))
        st = optuna.load_study(study_name=name, storage=storage)
        rows = [t.params for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return rows[:limit] if limit else rows


class Recorder:
    """Wraps trainer functions to fingerprint their inputs/outputs without changing them."""

    def __init__(self):
        self.fits, self.tap_calls, self.lp_calls = [], [], []

    def install(self):
        self._fit, self._tap, self._lp = g.fit_model, R.tap_replay_recording, R.lp_replay_press
        rec = self

        def fit_model(X, y, w, params, budget):
            m = rec._fit(X, y, w, params, budget)
            rec.fits.append(dict(X=sha(np.asarray(X)), y=sha(np.asarray(y)), w=sha(np.asarray(w)),
                                 shape=list(np.asarray(X).shape), contiguous=bool(np.asarray(X).flags.c_contiguous),
                                 model=hashlib.sha256(m.booster_.model_to_string().encode()).hexdigest()[:24],
                                 trees=m.booster_.num_trees()))
            return m

        def tap(decision, t_ms, label, onsets, folder, counts):
            rec.tap_calls.append(sha(np.asarray(decision, bool), np.asarray(t_ms, np.float64),
                                     np.asarray(label, np.int8), np.sort(np.asarray(onsets, np.float64)), folder))
            return rec._tap(decision, t_ms, label, onsets, folder, counts)

        def lp(q, t_ms, start_ms, release_ms, kind, folder, counts, **kw):
            rec.lp_calls.append(sha(np.asarray(q, np.int64), np.asarray(t_ms, np.float64),
                                    [float(start_ms), float(release_ms), kind, folder]))
            return rec._lp(q, t_ms, start_ms, release_ms, kind, folder, counts, **kw)

        g.fit_model, R.tap_replay_recording, R.lp_replay_press = fit_model, tap, lp

    def uninstall(self):
        g.fit_model, R.tap_replay_recording, R.lp_replay_press = self._fit, self._tap, self._lp


def run(task, out_json, study=None, limit=None):
    data, rpl, tr, te, budgets, build_s = setup(task)
    results = []
    rows = frozen_params(task, study, limit)
    if os.environ.get("MUDRA_BENCH_ONLY_ALL") == "1":
        rows = [p for p in rows if p.get("feature_set", "all_available") == "all_available"]
    if getattr(g, "FEATURE_SELECTION", False):
        # v3 has no named feature sets: the all-feature trials become K = every available field.
        rows = [dict({k: v for k, v in p.items() if k != "feature_set"}, n_features=len(data["columns"]))
                for p in rows if p.get("feature_set", "all_available") == "all_available" or "n_features" in p]
    for params in rows:
        rec = Recorder(); rec.install()
        trial = optuna.trial.FixedTrial(dict(params))
        t0 = time.perf_counter()
        try:
            value = g.objective(trial, data, rpl, tr, task, budgets)
        finally:
            rec.uninstall()
        seconds = time.perf_counter() - t0
        ua = trial.user_attrs
        results.append(dict(params=params, value=value, seconds=round(seconds, 2), cv=ua.get("cv"),
                            folds=ua.get("folds"), timing=ua.get("timing"), fits=rec.fits,
                            tap_calls=sorted(rec.tap_calls), lp_calls=sorted(rec.lp_calls),
                            n_tap_calls=len(rec.tap_calls), n_lp_calls=len(rec.lp_calls)))
        print("BENCH_TRIAL", task, json.dumps(dict(feature_set=params.get("feature_set", f"K{params.get('n_features')}"),
              leaves=params["num_leaves"],
              value=value, seconds=round(seconds, 2), timing=ua.get("timing", {}).get("total") if ua.get("timing") else None)),
              flush=True)
    env = dict(python=sys.version.split()[0], platform=platform.platform(), cpu_threads=g.CPU_THREADS,
               numpy=np.__version__, lightgbm=g.lgb.__version__, optuna=optuna.__version__,
               build_s=round(build_s, 2), train_rows=int(len(data["y"])), replay_rows=int(len(rpl["X"])))
    Path(out_json).write_text(json.dumps(dict(task=task, env=env, budgets=budgets, results=results), indent=1,
                                         default=str))


def _strip(rep):
    return json.loads(json.dumps(rep, default=str))


def compare(a_path, b_path):
    A, B = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    ok_all = True
    for ra, rb in zip(A["results"], B["results"]):
        checks = {
            "objective value": ra["value"] == rb["value"],
            "cv reports": _strip(ra["cv"]) == _strip(rb["cv"]),
            "fold reports": _strip(ra["folds"]) == _strip(rb["folds"]),
            "train X/y/w + models": [{k: f[k] for k in ("X", "y", "w", "shape", "model", "trees")} for f in ra["fits"]]
                                    == [{k: f[k] for k in ("X", "y", "w", "shape", "model", "trees")} for f in rb["fits"]],
            "TAP decisions": ra["tap_calls"] == rb["tap_calls"] and ra["n_tap_calls"] == rb["n_tap_calls"],
            "LP Q20 sequences": ra["lp_calls"] == rb["lp_calls"] and ra["n_lp_calls"] == rb["n_lp_calls"],
        }
        ok = all(checks.values()); ok_all &= ok
        fs = ra["params"].get("feature_set", f"K{ra['params'].get('n_features')}")
        print(f"{'PASS' if ok else 'FAIL'} {A['task']} {fs:24s} leaves={ra['params']['num_leaves']:3d} "
              f"value={ra['value']:.6f}/{rb['value']:.6f} time {ra['seconds']:.1f}s -> {rb['seconds']:.1f}s "
              f"({ra['seconds'] / max(rb['seconds'], 1e-9):.2f}x) fits={len(rb['fits'])} "
              f"replay_calls={rb['n_tap_calls'] + rb['n_lp_calls']}"
              + ("" if ok else "  mismatch: " + ", ".join(k for k, v in checks.items() if not v)))
    print("ALL_EQUIVALENT" if ok_all else "NOT_EQUIVALENT")
    return ok_all


def profile(task):
    data, rpl, tr, te, budgets, _ = setup(task)
    params = FROZEN[task][0]
    pr = cProfile.Profile(); trial = optuna.trial.FixedTrial(dict(params))
    pr.enable(); g.objective(trial, data, rpl, tr, task, budgets); pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(28)
    print(s.getvalue())
    if trial.user_attrs.get("timing"):
        print(json.dumps(trial.user_attrs["timing"], indent=1))


def threads(task, counts, param_index=0):
    """Fit fold 0 of one frozen trial with each thread count: time + exact model/prediction equality."""
    from sklearn.model_selection import GroupKFold
    data, rpl, tr, te, budgets, _ = setup(task)
    params = FROZEN[task][param_index]
    sel = [data["columns"].index(c) for c in data["feature_sets"][params["feature_set"]]]
    a, b = next(GroupKFold(n_splits=g.CV_FOLDS).split(tr, data["y"][tr], data["group"][tr]))
    fold = tr[a]; X = np.ascontiguousarray(data["X"][fold][:, sel]); y = data["y"][fold]
    w = g.weights_for_task(data, fold, task, params["positive_weight"])
    val = rpl["recs"][rpl["recs"].group.isin(set(data["group"][tr[b]]))]
    rows = [(int(r.start), int(r.end)) for r in val.itertuples() if r.start >= 0][:400]
    Xv = np.ascontiguousarray(np.concatenate([np.asarray(rpl["X"][s:e]) for s, e in rows])[:, sel]) if rows else None
    if Xv is None:
        w_ = rpl["windows"]; Xv = np.ascontiguousarray(np.concatenate([np.asarray(rpl["X"][int(r.start):int(r.end)])
                                                                       for r in w_.head(400).itertuples()])[:, sel])
    keys = ("max_depth", "num_leaves", "n_estimators", "learning_rate", "min_child_samples", "reg_lambda", "reg_alpha",
            "colsample_bytree", "subsample")
    base = dict(objective="binary", subsample_freq=1, max_bin=127, verbosity=-1, random_state=g.SEED, deterministic=True,
                **{k: params[k] for k in keys})
    ref = None
    print(f"threads | fit_s | predict_s ({len(Xv)} rows) | trees identical | predictions identical")
    for n in counts:
        p = dict(base, n_jobs=n)
        t0 = time.perf_counter(); m = g.fit_model(X, y, w, p, budgets[0]); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); pr = m.booster_.predict(Xv, raw_score=True, num_threads=n); pred_s = time.perf_counter() - t0
        ms = _trees_only(m.booster_.model_to_string())
        if ref is None:
            ref = (ms, pr)
        print(f"{n:7d} | {fit_s:6.1f} | {pred_s:9.2f} | {ms == ref[0]} | {np.array_equal(pr, ref[1])}", flush=True)


def _trees_only(model_string: str) -> str:
    """Model text without the trailing parameters block (it records [num_threads: N], not the model)."""
    return model_string.split("\nparameters:")[0]


def stages(task, repeats=3):
    """Old vs optimized implementation of each changed stage, same data, warm caches, fold 0 of each frozen trial."""
    data, rpl, tr, te, budgets, _ = setup(task)
    folds = g.prepared_folds(data, rpl, tr, task)
    pf = folds[0]
    best = lambda f: min(_timed(f) for _ in range(repeats))
    rows = []
    for params in FROZEN[task]:
        sel = [data["columns"].index(c) for c in data["feature_sets"][params["feature_set"]]]
        pw = params["positive_weight"]
        t_old = best(lambda: (data["X"][pf["tr"]][:, sel], g.weights_for_task(data, pf["tr"], task, pw)))
        t_new = best(lambda: (data["X"][np.ix_(pf["tr"], np.asarray(sel, np.intp))], g.fold_weights(pf, pw)))
        rows.append(("fold X + weights", params["feature_set"], t_old, t_new))
        keys = ("max_depth", "num_leaves", "n_estimators", "learning_rate", "min_child_samples", "reg_lambda",
                "reg_alpha", "colsample_bytree", "subsample")
        p = dict(objective="binary", subsample_freq=1, max_bin=127, verbosity=-1, random_state=g.SEED,
                 deterministic=True, n_jobs=g.CPU_THREADS, **{k: params[k] for k in keys})
        model = g.fit_model(data["X"][np.ix_(pf["tr"], np.asarray(sel, np.intp))], pf["y"], g.fold_weights(pf, pw), p,
                            budgets[0])
        def size_new():
            g._SIZING.pop(model, None)
            return [g.trees_within(model, b) for b in budgets]
        t_old = best(lambda: [g.trees_within_reference(model, b) for b in budgets]); t_new = best(size_new)
        rows.append(("exact model size", params["feature_set"], t_old, t_new))
        ranges = [it[:2] for it in pf["replay"]]
        ident = lambda X: {"n": X[:, 0]}
        t_old = best(lambda: sum(1 for _ in g._ranges_predict_reference([ident], rpl["X"], ranges, sel)))
        t_new = best(lambda: sum(1 for _ in g._ranges_predict([ident], rpl["X"], ranges, sel)))
        rows.append(("replay gather (all validation rows)", params["feature_set"], t_old, t_new))
        t_old = best(lambda: (g.prepare_tap_replay if task == "tap" else g.prepare_lp_replay)(rpl, pf["val_groups"]))
        rows.append(("replay plan (per trial before, once now)", params["feature_set"], t_old, 0.0))
    print(f"{'stage':42s} {'feature set':24s} {'old s':>8s} {'new s':>8s}  (fold 0, best of {repeats})")
    for r in rows:
        print(f"{r[0]:42s} {r[1]:24s} {r[2]:8.3f} {r[3]:8.3f}")


def _timed(f):
    t0 = time.perf_counter(); f(); return time.perf_counter() - t0


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        study = sys.argv[sys.argv.index("--study") + 1] if "--study" in sys.argv else None
        limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        run(sys.argv[2], sys.argv[3], study, limit)
    elif mode == "compare":
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
    elif mode == "profile":
        profile(sys.argv[2])
    elif mode == "stages":
        stages(sys.argv[2])
    elif mode == "threads":
        threads(sys.argv[2], [int(x) for x in sys.argv[3].split(",")])
