#!/usr/bin/env python3
"""Isolated TAP label research. Does not modify production labels, models or studies."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POS = {"TAP", "SWIPE"}
NEG = {"FINGER", "WRIST", "ARM", "BAND", "NOTHING"}
VARIANTS = {
    "baseline80": (0, 80), "trusted80": (0, 80),
    "trusted50": (0, 50), "trusted60": (0, 60),
    "trusted20_80": (20, 80), "trusted20_100": (20, 100),
    "adaptive50": (0, 50),
}
SEED = 20261125
THRESHOLD = 0.85
BUDGET = 14 * 1024
GATE_MS = 1100.0

def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "research/labeling_v3/results")
    p.add_argument("--threads", type=int, default=int(os.environ.get("MUDRA_CPU_THREADS", 0)) or
                   max(1, int(0.8 * (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1))))
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--weights", default="0.5,1,2", help="Identical positive loss weights tested for each label variant")
    p.add_argument("--finger-fpm", type=float, default=None, help="Optional hard FINGER false events/min cap")
    p.add_argument("--fist-fpm", type=float, default=None, help="Optional hard fist false events/min cap")
    p.add_argument("--holdout", action="store_true", help="Evaluate only the CV-selected candidate on reserved people")
    p.add_argument("--limit-per-folder", type=int, default=0, help="Smoke run only; use separate output directory")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()

def log(*a):
    print(*a, flush=True)

def sha(path, compressed=False):
    h = hashlib.sha256()
    with (gzip.open(path, "rb") if compressed else open(path, "rb")) as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def subtype(folder, name):
    s = name.lower()
    if folder == "FINGER":
        for key, pattern in (("fist", r"fist|stretch"), ("index", r"index"),
                             ("middle", r"middle|midfinger"), ("ring", r"ring"),
                             ("pinky", r"pinky|little"), ("thumb", r"thumb")):
            if re.search(pattern, s):
                return "FINGER/" + key
    if folder in POS:
        for key, pattern in (("double", r"double"), ("slowpinch", r"slow.?pinch"),
                             ("soft", r"soft"), ("strong", r"strong"), ("mid", r"mid"),
                             ("motion", r"motion"), ("contraction", r"contraction")):
            if re.search(pattern, s):
                return folder + "/" + key
    return folder + "/other"

def import_stack(threads):
    global np, pd, g, TF, lab, lgb, GroupKFold, GroupShuffleSplit
    os.environ.update(MUDRA_CPU_THREADS=str(threads), MUDRA_USE_LEAN8="0",
                      MUDRA_TAP_FEATURES="diff", MUDRA_TAP_AUG="mild",
                      MUDRA_TASKS="tap", OMP_NUM_THREADS=str(threads))
    sys.path.insert(0, str(ROOT / "research/gesture"))
    sys.path.insert(0, str(ROOT / "claude_labeling"))
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import train_unified as g
    import tap_features as TF
    import label_gestures as lab
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

def inventory(limit):
    source = ROOT / "DATA/DATA_TAP"
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = []
    for folder in sorted(POS | NEG):
        f = sorted((source / folder).glob("*.csv.gz"))
        if limit and len(f) > limit:
            f = [f[i] for i in np.unique(np.linspace(0, len(f)-1, limit).astype(int))]
        files.extend(f)
    if not files:
        raise RuntimeError("No DATA/DATA_TAP recordings")
    h = hashlib.sha256()
    for p in [Path(__file__), Path(lab.__file__), Path(TF.__file__), Path(g.__file__)] + sorted((ROOT / "claude_labeling/models").glob("*")):
        if p.is_file():
            h.update(sha(p).encode())
    seen = {}
    unique = []
    for i, p in enumerate(files):
        digest = sha(p, compressed=True)
        if digest in seen:
            if seen[digest] != p.parent.name:
                raise RuntimeError("Identical recording in conflicting folders: " + str(p))
            continue
        seen[digest] = p.parent.name
        h.update((str(p.relative_to(ROOT)) + "|" + digest).encode())
        unique.append((p, digest))
        if (i+1) % 500 == 0:
            log("LABELING_V3_HASH", i+1, "/", len(files))
    log("LABELING_V3_INVENTORY", len(files), "unique", len(unique))
    return unique, h.hexdigest()

def prepare(files, fingerprint, out):
    cache = out / ("cache_" + fingerprint[:20])
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "manifest.json"
    if manifest.is_file():
        recs = json.loads(manifest.read_text())
        if all((cache / f"{r['id']}.npy").is_file() and (cache / f"{r['id']}.npz").is_file() for r in recs):
            log("LABELING_V3_CACHE_REUSED", len(recs), cache)
            return cache, recs
    tm = lab.FoldModels("tap_teacher_fold", "tap_teacher.json", "person_group_fold")
    cm = lab.FoldModels("contact_fold", "contact_folds.json", "person_group_fold")
    cols = list(g.RAW_126)
    recs = []
    for i, (path, digest) in enumerate(files):
        # Per-record checkpoint; never re-use results from a different source/model fingerprint.
        mpath = cache / f"{i}.json"
        xpath = cache / f"{i}.npy"
        zpath = cache / f"{i}.npz"
        if mpath.is_file() and xpath.is_file() and zpath.is_file():
            recs.append(json.loads(mpath.read_text()))
            continue
        labels, ev, _ = lab.label_recording(str(path), tm, cm)
        df = pd.read_csv(path, usecols=lambda c: c in set(cols) | {"Timestamp", "User"}, low_memory=False)
        ts = df.Timestamp.to_numpy(np.int64)
        if not np.array_equal(ts, labels.Timestamp.to_numpy(np.int64)):
            raise ValueError("Timestamp mismatch: " + str(path))
        t = (ts - ts[0]).astype(np.float64)
        if np.any(np.diff(t) < 0):
            raise ValueError("Nonmonotonic timestamp: " + str(path))
        X = df[cols].to_numpy(np.float32)
        if not np.isfinite(X).all():
            raise ValueError("Nonfinite firmware input: " + str(path))
        user = str(df.User.iloc[0]) if "User" in df else ""
        group = g.v6.person_group(user, path.name[:-7])
        if not group:
            raise ValueError("Unknown participant group: " + str(path))
        folder = path.parent.name
        strong = np.zeros(0, float)
        press = np.zeros(0, float)
        weak = np.zeros(0, float)
        times = np.zeros(0, float)
        if folder in POS and len(ev):
            pp = ev[(ev.type == "tap_press") & (ev.tier == "PRESS")]
            strong = np.sort(pp.loc[pp.labeled == 1, "t_on"].dropna().to_numpy(float))
            # Preserve all reference presses (including unconfirmed ones) as a separate, noisy yardstick.
            press = np.sort(pp.t_ptr.dropna().to_numpy(float))
            weak = np.sort(pp.loc[pp.labeled == -1, "t_ptr"].dropna().to_numpy(float))
            times = np.sort(np.concatenate([ev[c].dropna().to_numpy(float) for c in ("t_on", "t_ptr", "t_rel")]))
        np.save(xpath, X)
        np.savez(zpath, t=t, y=labels.tap_label.to_numpy(np.int8),
                 active=labels.any_active.to_numpy(np.int8), off=labels.n_ch_off.to_numpy(np.int8),
                 strong=strong, press=press, weak=weak, times=times)
        r = dict(id=i, folder=folder, subtype=subtype(folder, path.name), group=str(group),
                 source=str(path.relative_to(ROOT)), sha256=digest, rows=len(t),
                 strong=len(strong), press=len(press))
        save_json(mpath, r)
        recs.append(r)
        if (i+1) % 50 == 0 or i+1 == len(files):
            log("LABELING_V3_PREPARE", i+1, "/", len(files))
    save_json(manifest, recs)
    return cache, recs

def load_meta(cache, recs):
    result = {}
    for r in recs:
        with np.load(cache / f"{r['id']}.npz") as z:
            result[r["id"]] = {k: z[k] for k in z.files}
    return result

def labels_for(m, folder, variant, adaptive=None):
    if variant == "baseline80":
        return m["y"].copy()
    if folder in NEG:
        return np.zeros(len(m["t"]), np.int8)
    t = m["t"]
    y = np.full(len(t), -1, np.int8)
    a, b = VARIANTS[variant]
    for j, onset in enumerate(m["strong"]):
        start = adaptive[j] if adaptive is not None else a
        end = start + 50 if adaptive is not None else b
        y[(t >= onset+start) & (t <= onset+end)] = 1
    # Preserve the existing explicit disconnected-device target.
    y[m["off"] >= 2] = 0
    return y

def stride(t, mask, ms):
    return g.stride_pick(t, mask, ms)

def sample(m, y):
    t = m["t"]
    valid = t >= GATE_MS
    neg = (y == 0) & valid
    rows = np.unique(np.concatenate([stride(t, neg, 40),
        stride(t, neg & (m["active"] == 1), 20),
        stride(t, neg & g.near(t, m["times"], 500), 20),
        stride(t, (y == 1) & valid, 10)]))
    return rows

def features(X):
    X = np.asarray(X, np.float32)
    return np.hstack([X, TF.derived(X, list(g.RAW_126))])

def gather_training(cache, records, meta, variant, draws, selected=None, adaptive=None):
    xx, yy, ww = [], [], []
    columns = list(g.RAW_126) + list(TF.DERIVED)
    for r in records:
        i = r["id"]; m = meta[i]
        offsets = None if adaptive is None else adaptive.get(i)
        y = labels_for(m, r["folder"], variant, offsets)
        rows = sample(m, y)
        if not len(rows):
            continue
        raw = np.load(cache / f"{i}.npy", mmap_mode="r")[rows]
        X = features(raw)
        # Deterministic mild placement draw per recording; same draw for all variants and folds.
        X = TF.apply_placement(X, columns, np.repeat(draws["gain"][i:i+1], len(rows), axis=0),
                               np.repeat(draws["tilt"][i:i+1], len(rows), axis=0),
                               np.repeat(draws["noise"][i:i+1], len(rows), axis=0))
        xx.append(X if selected is None else X[:, selected])
        yy.append(y[rows])
        ww.append(np.full(len(rows), 1.0 / np.sqrt(len(rows)), np.float32))
    if not xx:
        raise ValueError("No training samples")
    X = np.concatenate(xx); y = np.concatenate(yy); w = np.concatenate(ww)
    for cls in (0, 1):
        ix = y == cls
        if not ix.any():
            raise ValueError("Training fold has no class " + str(cls))
        w[ix] *= len(y) / (2 * w[ix].sum())
    return X, y, w

def fixed_params(threads):
    return dict(objective="binary", n_estimators=220, max_depth=6, num_leaves=21,
                learning_rate=0.10, min_child_samples=40, reg_lambda=1.0, reg_alpha=0.01,
                colsample_bytree=0.85, subsample=0.8, subsample_freq=1, max_bin=127,
                deterministic=True, force_col_wise=True, verbosity=-1,
                random_state=SEED, n_jobs=threads)

def fit_compact(X, y, w, weight, threads):
    w = w.copy(); w[y == 1] *= weight
    model = g.fit_model(X, y, w, fixed_params(threads), BUDGET)
    trees, size = g.trees_within(model, BUDGET)
    assert size <= BUDGET
    return g.truncated(model, trees), int(size)

def rank_features(cache, records, meta, draws, threads):
    # Hold raw126 fixed; screen the 144 derived fields using only outer training people.
    # Keep 128, for an exporter-compatible maximum of 254 features.
    X, y, w = gather_training(cache, records, meta, "baseline80", draws)
    p = fixed_params(threads); p.update(n_estimators=50, num_leaves=15, max_depth=5)
    screen = lgb.LGBMClassifier(**p).fit(X, y, sample_weight=w)
    gain = screen.booster_.feature_importance("gain")
    chosen = np.argsort(-gain[126:], kind="stable")[:128] + 126
    return np.sort(np.r_[np.arange(126), chosen])

def adaptive_offsets(cache, records, meta, draws, threads):
    """Inner group-cross-fitted causal scores; pick a 50 ms interval, never drop an event."""
    groups = [r["group"] for r in records]
    result = {}
    for tr, va in GroupKFold(n_splits=min(3, len(set(groups)))).split(records, groups=groups):
        train = [records[j] for j in tr]
        valid = [records[j] for j in va]
        # Use raw126 only: genuinely causal, no fold-global feature ranking leakage.
        sel = np.arange(126)
        X, y, w = gather_training(cache, train, meta, "trusted80", draws, sel)
        model, _ = fit_compact(X, y, w, 1.0, threads)
        del X, y, w
        for r in valid:
            i = r["id"]; m = meta[i]; t = m["t"]
            if r["folder"] not in POS or not len(m["strong"]):
                continue
            raw = np.load(cache / f"{i}.npy", mmap_mode="r")
            starts = []
            # Evaluate only relevant event neighborhoods; scores use current logged p_core fields.
            row = np.flatnonzero(g.near(t, m["strong"], 110) & (t >= GATE_MS))
            p = model.booster_.predict(raw[row], num_threads=1) if len(row) else np.zeros(0)
            tt = t[row]
            for onset in m["strong"]:
                candidates = []
                for a in (0, 10, 20, 30, 40, 50):
                    z = (tt >= onset+a) & (tt <= onset+a+50)
                    candidates.append(float(np.mean(p[z])) if z.any() else -1.0)
                starts.append(int(np.argmax(candidates)) * 10)
            result[i] = starts
    return result

def events(t, prob):
    d = prob >= THRESHOLD
    rise = np.flatnonzero(d & ~np.r_[False, d[:-1]])
    kept = []
    for i in rise:
        if not kept or t[i] - t[kept[-1]] >= g.R.TAP_REFRACTORY_MS:
            kept.append(i)
    return t[np.asarray(kept, dtype=int)]

def match(ref, pred, before, after):
    # Earliest-expiring intervals, fixed-width windows: maximum-cardinality ordered matching.
    hits = np.zeros(len(ref), bool)
    latency = []
    j = 0
    for i, t in enumerate(ref):
        while j < len(pred) and pred[j] < t-before:
            j += 1
        if j < len(pred) and pred[j] <= t+after:
            hits[i] = True
            latency.append(float(pred[j]-t))
            j += 1
    return hits, latency

def evaluate(model, selected, cache, records, meta, stress, threads):
    # Predict batches instead of one multithreaded call for each tiny recording.
    by = {}
    lat = []
    cols = list(g.RAW_126) + list(TF.DERIVED)
    # Caller supplies full-size stress draws, indexed by global recording ID.
    draws = stress if isinstance(stress, dict) else None
    def stat(name):
        return by.setdefault(name, dict(strong=0, strong_hits=0, press=0, press_hits=0,
                                        weak=0, weak_hits=0, pairs=0, pairs_both=0,
                                        false=0, neg_ms=0.0))
    def score(r, prob, t):
        m = meta[r["id"]]
        if not len(t):
            return
        det = events(t, prob)
        keys = list(dict.fromkeys(["ALL", r["folder"], r["subtype"]]))
        if r["folder"] in NEG:
            duration = float(np.clip(np.diff(t), 0, 50).sum())
            for k in keys:
                s = stat(k); s["false"] += len(det); s["neg_ms"] += duration
        else:
            strong = m["strong"][(m["strong"] >= t[0]+100) & (m["strong"] <= t[-1])]
            press = m["press"][(m["press"] >= t[0]+150) & (m["press"] <= t[-1])]
            weak = m["weak"][(m["weak"] >= t[0]+150) & (m["weak"] <= t[-1])]
            a, l = match(strong, det, 100, 250)
            b, _ = match(press, det, 150, 250)
            # Weak recall is the weak subset of the SAME all-press matching, not a second assignment.
            weak_mask = np.isin(press, weak)
            pairs = both = 0
            if r["subtype"] == "TAP/double":
                j = 0
                while j+1 < len(press):
                    if 100 <= press[j+1]-press[j] <= 450:
                        pairs += 1; both += int(b[j] and b[j+1]); j += 2
                    else:
                        j += 1
            lat.extend(l)
            for k in keys:
                s = stat(k)
                s["strong"] += len(a); s["strong_hits"] += int(a.sum())
                s["press"] += len(b); s["press_hits"] += int(b.sum())
                s["weak"] += int(weak_mask.sum()); s["weak_hits"] += int((b & weak_mask).sum())
                s["pairs"] += pairs; s["pairs_both"] += both
    batch, plan, n = [], [], 0
    def flush():
        if not batch:
            return
        X = np.concatenate(batch)
        p = model.booster_.predict(X, num_threads=threads)
        for r, t, a, b in plan:
            score(r, p[a:b], t)
        batch.clear(); plan.clear()
    for r in records:
        i = r["id"]; m = meta[i]
        a = int(np.searchsorted(m["t"], GATE_MS))
        raw = np.load(cache / f"{i}.npy", mmap_mode="r")[a:]
        if not len(raw):
            continue
        X = features(raw)
        if draws is not None:
            X = TF.apply_placement(X, cols, np.repeat(draws["gain"][i:i+1], len(X), axis=0),
                                   np.repeat(draws["tilt"][i:i+1], len(X), axis=0),
                                   np.repeat(draws["noise"][i:i+1], len(X), axis=0))
        batch.append(np.ascontiguousarray(X[:, selected]))
        plan.append((r, m["t"][a:], n, n+len(X))); n += len(X)
        if n >= 150000:
            flush(); n = 0
    flush()
    return dict(counts=by, latency_ms=lat)

def pool(reports):
    counts = {}; lat = []
    for rep in reports:
        lat.extend(rep["latency_ms"])
        for k, s in rep["counts"].items():
            c = counts.setdefault(k, {x: 0 for x in s})
            for x, v in s.items():
                c[x] += v
    for s in counts.values():
        s["strong_recall"] = s["strong_hits"]/s["strong"] if s["strong"] else None
        s["press_recall"] = s["press_hits"]/s["press"] if s["press"] else None
        s["weak_recall"] = s["weak_hits"]/s["weak"] if s["weak"] else None
        s["double_pair_recall"] = s["pairs_both"]/s["pairs"] if s["pairs"] else None
        s["false_per_min"] = s["false"]*60000/s["neg_ms"] if s["neg_ms"] else None
    return dict(by_category=counts, latency_p50_ms=float(np.median(lat)) if lat else None,
                latency_p90_ms=float(np.percentile(lat, 90)) if lat else None)

def split_records(recs, folds):
    groups = np.array([r["group"] for r in recs])
    cat = np.array([r["folder"] for r in recs])
    for tr, te in GroupShuffleSplit(n_splits=100, test_size=0.22, random_state=SEED).split(recs, groups=groups):
        if len(set(groups[tr])) < folds or not ({"TAP", "FINGER"} <= set(cat[te]) and {"TAP", "FINGER"} <= set(cat[tr])):
            continue
        dev = [recs[j] for j in tr]; test = [recs[j] for j in te]
        dg = groups[tr]
        cv = [([dev[j] for j in a], [dev[j] for j in b])
              for a, b in GroupKFold(n_splits=folds).split(dev, groups=dg)]
        if all({"TAP", "FINGER"} <= {r["folder"] for r in a} and
               {"TAP", "FINGER"} <= {r["folder"] for r in b} for a, b in cv):
            assert not set(groups[tr]) & set(groups[te])
            return dev, test, cv
    raise ValueError("Cannot find usable participant-separated splits; increase dataset coverage")

def key(v, weight):
    return f"{v}__w{weight:g}"

def metric(row, state, group, name):
    return row[state]["by_category"].get(group, {}).get(name)

def choose(rows, a):
    # A Pareto table remains visible. With no explicit product cap, require no regression
    # against the best-recall baseline80 operating point on normal and stress FINGER/FIST.
    baseline = [r for r in rows if r["variant"] == "baseline80"]
    if not baseline:
        raise ValueError("baseline80 required for automatic comparison")
    b = max(baseline, key=lambda r: metric(r, "normal", "ALL", "press_recall") or 0)
    limits = {}
    for state in ("normal", "stress"):
        for group, explicit in (("FINGER", a.finger_fpm), ("FINGER/fist", a.fist_fpm)):
            val = explicit if explicit is not None else metric(b, state, group, "false_per_min")
            if val is None:
                raise ValueError("Missing FINGER/fist exposure in development split")
            limits[(state, group)] = val
    def feasible(r):
        return all((metric(r, s, g_, "false_per_min") is not None and
                    metric(r, s, g_, "false_per_min") <= lim+1e-9)
                   for (s, g_), lim in limits.items())
    ok = [r for r in rows if feasible(r)]
    def ranking(r):
        return (min(metric(r, s, "ALL", "press_recall") or 0 for s in ("normal", "stress")),
                min(metric(r, s, "ALL", "strong_recall") or 0 for s in ("normal", "stress")),
                -max(metric(r, s, "FINGER", "false_per_min") or 0 for s in ("normal", "stress")))
    return (max(ok, key=ranking) if ok else None), {f"{s}/{g_}": l for (s, g_), l in limits.items()}

def main(a):
    variants = a.variants.split(",")
    weights = [float(w) for w in a.weights.split(",")]
    if not set(variants) <= set(VARIANTS) or "baseline80" not in variants or min(weights) <= 0:
        raise ValueError("Invalid variants/weights; include baseline80")
    files, fp = inventory(a.limit_per_folder)
    a.output.mkdir(parents=True, exist_ok=True)
    cache, recs = prepare(files, fp, a.output)
    meta = load_meta(cache, recs)
    dev, test, cv = split_records(recs, a.folds)
    settings = dict(fingerprint=fp, folds=a.folds, variants=variants, weights=weights,
                    threshold=THRESHOLD, budget=BUDGET, gate_ms=GATE_MS, seed=SEED,
                    refractory_ms=g.R.TAP_REFRACTORY_MS, limit_per_folder=a.limit_per_folder)
    config_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    run = a.output / ("run_" + config_hash); run.mkdir(exist_ok=True)
    save_json(run / "manifest.json", dict(settings=settings, recordings=len(recs),
              development_people=sorted({r["group"] for r in dev}),
              reserved_people=sorted({r["group"] for r in test}),
              splits=[dict(train=sorted({r["group"] for r in tr}), validation=sorted({r["group"] for r in va})) for tr, va in cv]))
    draws = TF.draws(len(recs), SEED)
    stress = TF.stress_draws(len(recs))
    results = {}
    for fold, (tr, va) in enumerate(cv):
        log("LABELING_V3_FOLD", fold+1, "/", len(cv), "train", len(tr), "validation", len(va))
        rankpath = run / f"features_fold{fold}.json"
        if rankpath.is_file():
            selected = np.array(json.loads(rankpath.read_text()), dtype=int)
        else:
            selected = rank_features(cache, tr, meta, draws, a.threads)
            save_json(rankpath, selected.tolist())
        adap = None
        for variant in variants:
            pending = [w for w in weights if not (run / f"{key(variant,w)}_fold{fold}.json").is_file()]
            if pending:
                if variant == "adaptive50":
                    apath = run / f"adaptive_fold{fold}.json"
                    if apath.is_file():
                        adap = {int(k): v for k,v in json.loads(apath.read_text()).items()}
                    else:
                        adap = adaptive_offsets(cache, tr, meta, draws, a.threads)
                        save_json(apath, adap)
                X, y, w = gather_training(cache, tr, meta, variant, draws, selected,
                                          adap if variant == "adaptive50" else None)
                for weight in pending:
                    model, size = fit_compact(X, y, w, weight, a.threads)
                    normal = evaluate(model, selected, cache, va, meta, None, a.threads)
                    shifted = evaluate(model, selected, cache, va, meta, stress, a.threads)
                    obj = dict(variant=variant, weight=weight, fold=fold, bytes=size,
                               train_rows=len(y), positive_rows=int((y == 1).sum()),
                               normal=normal, stress=shifted)
                    save_json(run / f"{key(variant,weight)}_fold{fold}.json", obj)
                    z = pool([normal]); log("LABELING_V3_RESULT", variant, weight, "fold", fold,
                        "bytes", size, "press_recall", z["by_category"]["ALL"]["press_recall"],
                        "finger_fpm", z["by_category"].get("FINGER", {}).get("false_per_min"))
                del X, y, w
            for weight in weights:
                obj = json.loads((run / f"{key(variant,weight)}_fold{fold}.json").read_text())
                results.setdefault(key(variant,weight), []).append(obj)
    rows = []
    for vals in results.values():
        rows.append(dict(variant=vals[0]["variant"], weight=vals[0]["weight"],
                         max_bytes=max(x["bytes"] for x in vals),
                         normal=pool([x["normal"] for x in vals]), stress=pool([x["stress"] for x in vals])))
    winner, limits = choose(rows, a)
    summary = dict(settings=settings, limits=limits, selected=winner, candidates=rows,
                   warning="Research only. Pointer/teacher references are not independent physical truth.")
    save_json(run / "summary.json", summary)
    def fmt(x):
        return "-" if x is None else f"{x:.4f}"
    lines = ["# TAP labeling V3: development CV", "", "All results: fixed 0.85, <=14 KiB.",
             "FPM is measured only in authoritative negative folders, not against variant-specific labels.",
             "Stress is simulated EMG placement, not physical testing.", "",
             "| Variant | Weight | Bytes max | Press recall | Soft subtype recall | Double pair recall | FINGER FPM | Fist FPM | Stress press recall | Stress FINGER FPM | Stress fist FPM |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        vals = [r["variant"], str(r["weight"]), str(r["max_bytes"])]
        vals += [fmt(metric(r,s,k,m)) for s,k,m in [
            ("normal","ALL","press_recall"), ("normal","TAP/soft","press_recall"),
            ("normal","TAP/double","double_pair_recall"), ("normal","FINGER","false_per_min"),
            ("normal","FINGER/fist","false_per_min"), ("stress","ALL","press_recall"),
            ("stress","FINGER","false_per_min"), ("stress","FINGER/fist","false_per_min")]]
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "Selection constraints: " + json.dumps(limits),
              "Selected: " + (key(winner["variant"],winner["weight"]) if winner else "NONE (no candidate meets constraints)"),
              "", "Reserved people have not been used for model/label selection.",
              "No production artifacts were changed. See summary.json for all subtype counts and denominators."]
    (run / "REPORT.md").write_text("\n".join(lines)+"\n")
    if a.holdout and winner:
        # This is a one-shot audit, not a new selection stage.
        v = winner["variant"]; wgt = winner["weight"]
        selected = rank_features(cache, dev, meta, draws, a.threads)
        adap = adaptive_offsets(cache, dev, meta, draws, a.threads) if v == "adaptive50" else None
        X,y,w = gather_training(cache, dev, meta, v, draws, selected, adap)
        model,size = fit_compact(X,y,w,wgt,a.threads)
        save_json(run / "holdout.json", dict(variant=v, weight=wgt, bytes=size,
                  normal=pool([evaluate(model,selected,cache,test,meta,None,a.threads)]),
                  stress=pool([evaluate(model,selected,cache,test,meta,stress,a.threads)])))
    log("LABELING_V3_COMPLETE", run / "REPORT.md", "selected", None if winner is None else key(winner["variant"],winner["weight"]))

def self_test():
    t = np.arange(0, 500, 10, dtype=float)
    m = dict(t=t, y=np.zeros(len(t), np.int8), off=np.zeros(len(t), np.int8),
             strong=np.array([100., 300.]))
    y = labels_for(m, "TAP", "trusted50")
    assert (y == 1).sum() == 12 and y[0] == -1
    assert (labels_for(m, "FINGER", "trusted50") == 0).all()
    hit, _ = match(np.array([100., 200.]), np.array([150.]), 100, 250)
    assert hit.sum() == 1
    pred = np.zeros(len(t)); pred[[10, 20, 30]] = 0.9
    assert np.array_equal(events(t,pred), np.array([100.,300.]))
    assert subtype("FINGER","user_Fist&Stretch.csv.gz") == "FINGER/fist"
    log("LABELING_V3_SELF_TEST_OK")

if __name__ == "__main__":
    a = args()
    import_stack(a.threads)
    if a.self_test:
        self_test()
    else:
        main(a)
