#!/usr/bin/env python3
"""Filename-supervised sensor-pattern research for Mudra gesture recordings.

One recording is one labelled bag. Rows/windows inside a positive recording are
NOT individually labelled positive. Ground truth comes only from the folder
class and descriptive filename subtype. The filename prefix is used only to
group participants for validation.

Never reads LABELS, PointerState, event tables, teacher outputs, tap_label,
hold_label, or the CSV User column. Model inputs are pooled p_core features only.

Run from repository root:
  python -u gesture_correlation/analyze.py
  python -u gesture_correlation/analyze.py --lean8 required
  python -u gesture_correlation/analyze.py --skip-models
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

THREADS = int(os.environ.get("MUDRA_CPU_THREADS", "0") or 0)
if THREADS <= 0:
    THREADS = max(1, int((os.cpu_count() or 1) * 0.8))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, str(THREADS))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, rankdata, wilcoxon
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    from lightgbm import LGBMClassifier
    BACKEND = "LightGBM"
except Exception:
    from sklearn.ensemble import ExtraTreesClassifier
    LGBMClassifier = None
    BACKEND = "ExtraTrees fallback"

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [ROOT / "DATA" / "DATA_TAP", ROOT / "DATA" / "DATA_LONG_PRESS"]
VERSION = "1.0.0"
GLOBAL_STATS = ("global_median", "global_std", "global_p10", "global_p90", "global_p99", "global_mad", "abs_d1_p50", "abs_d1_p90")
WINDOW_STATS = ("median_p10", "median_p50", "median_p90", "median_p99", "std_p50", "std_p90", "range_p90")
META_STOP = {"horizontal", "vertical", "reverse", "default", "none", "noise", "left", "right", "lefthand", "righthand", "hand"}


def log(*x):
    print(*x, flush=True)


def stem(p):
    return p.name[:-7] if p.name.endswith(".csv.gz") else (p.name[:-4] if p.name.endswith(".csv") else p.stem)


def participant(p):
    s = re.sub(r"[^a-z0-9]", "", stem(p).split("_", 1)[0].lower())
    s = re.sub(r"(lefthand|righthand|left|right|noise|psd\d+|psd|hand)$", "", s)
    for alias in ("alex", "ohad", "adi", "gilad", "benkiviti", "elin", "eli", "michael", "kirill", "daniel", "neomi", "perry", "shay", "eitan"):
        if s.startswith(alias):
            return alias
    return s or "unknown"


def subtype(p):
    parts = stem(p).split("_")[1:]
    out = []
    for token in parts:
        low = token.lower()
        if low in {"left", "right", "lefthand", "righthand", "hand", "noise"}:
            continue
        if low in {"horizontal", "vertical", "reverse", "default", "none"} or re.fullmatch(r"\d+bpm", low):
            break
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token) or re.fullmatch(r"\d{1,2}-\d{2}-\d{2}", token):
            break
        if re.fullmatch(r"\d{1,2}", token):
            break
        out.append(token)
    return "_".join(out) if out else "unspecified"


def sha256(path):
    """Hash the uncompressed CSV payload so .csv and .csv.gz copies deduplicate."""
    h = hashlib.sha256()
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan(roots):
    out = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for p in sorted(root.rglob("*")):
            if p.is_file() and (p.name.endswith(".csv.gz") or p.name.endswith(".csv")) and not p.name.startswith("."):
                out.append({"raw": p, "root": root, "rel": p.relative_to(root), "class": p.parent.name.upper(),
                            "subtype": subtype(p), "participant": participant(p)})
    return out


def deduplicate(rows):
    seen, kept, duplicates, conflicts = {}, [], [], []
    for r in rows:
        h = sha256(r["raw"])
        r["sha"] = h
        if h not in seen:
            seen[h] = r
            kept.append(r)
        elif seen[h]["class"] == r["class"]:
            duplicates.append({"kept": str(seen[h]["raw"]), "duplicate": str(r["raw"]), "class": r["class"], "sha256": h})
        else:
            first = seen[h]
            conflicts.append({"first": str(first["raw"]), "first_class": first["class"],
                              "duplicate": str(r["raw"]), "duplicate_class": r["class"], "sha256": h})
            first["_conflict"] = True
    kept = [r for r in kept if not r.get("_conflict")]
    return kept, duplicates, conflicts


def lean8_root(root):
    return root.parent / (root.name + "_LEAN8")


def prepare_features(rows, roots, mode, parity_files):
    if mode == "off":
        for r in rows:
            r["feature"] = r["raw"]
        return {"lean8_used": False, "mode": mode, "parity": "off"}

    def derived(r):
        return lean8_root(r["root"]) / r["rel"]

    missing = [r for r in rows if not derived(r).is_file()]
    generated, parity = False, "not run"
    if missing:
        gp = ROOT / "research" / "gesture" / "generate_lean8_features.py"
        supported_sources = all(
            r["raw"].name.endswith(".csv.gz") or r["raw"].name.endswith(".csv")
            for r in rows
        )
        can = gp.is_file() and supported_sources
        if can:
            spec = importlib.util.spec_from_file_location("mudra_lean8_generator", gp)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            pairs = [(str(r), str(lean8_root(r))) for r in roots]
            log("LEAN8_PREPARE", len(missing), "missing; running incremental feature generator")
            result = module.ensure_datasets(pairs=pairs, parity_files=parity_files)
            parity = result.get("firmware_parity", {}).get("status", "unknown")
            generated = True
        elif mode == "required":
            raise RuntimeError("Lean-8 required but generated files are missing and cannot be generated")
    remaining = [r for r in rows if not derived(r).is_file()]
    if remaining:
        if mode == "required":
            raise RuntimeError(f"Lean-8 required; {len(remaining)} files remain missing")
        log("LEAN8_PREPARE", "using raw firmware features because derived Lean-8 tree is incomplete")
        for r in rows:
            r["feature"] = r["raw"]
        return {"lean8_used": False, "mode": mode, "generated": generated, "missing": len(remaining), "parity": parity}
    for r in rows:
        r["feature"] = derived(r)
    return {"lean8_used": True, "mode": mode, "generated": generated, "missing": 0, "parity": parity}


def common_columns(rows):
    order, counts = [], Counter()
    for i, r in enumerate(rows, 1):
        cols = [c for c in pd.read_csv(r["feature"], nrows=0).columns if c.startswith("p_core->")]
        for col in cols:
            if col not in counts:
                order.append(col)
            counts[col] += 1
        if i % 500 == 0:
            log("SCHEMA_SCAN", i, "/", len(rows), "union", len(order))
    availability = pd.DataFrame([
        {"feature": col, "files_present": counts[col], "files_total": len(rows),
         "presence_fraction": counts[col] / len(rows), "used_in_analysis": counts[col] == len(rows)}
        for col in order
    ])
    common = [col for col in order if counts[col] == len(rows)]
    if not common:
        raise RuntimeError("No p_core-> feature columns are present in every recording")
    return common, availability


def schema_for(features, windows):
    names, meta, index, groups = [], [], {}, [[] for _ in features]
    for j, feature in enumerate(features):
        stats = list(GLOBAL_STATS)
        for w in windows:
            stats += [f"w{w}_{s}" for s in WINDOW_STATS]
        for stat in stats:
            k = len(names)
            names.append(f"{feature}::{stat}")
            meta.append({"index": k, "feature_index": j, "feature": feature, "stat": stat})
            index[(j, stat)] = k
            groups[j].append(k)
    # The exploratory classifier receives every descriptor derived from every available p_core field.
    model_idx = np.arange(len(names), dtype=np.int32)
    model_base = np.asarray([x["feature_index"] for x in meta], dtype=np.int32)
    model_names = list(names)
    return {"names": names, "meta": meta, "index": index, "groups": groups,
            "model_indices": model_idx, "model_base": model_base, "model_names": model_names}


def q(x, quantiles, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(x, quantiles, axis=axis)


def starts_for(duration, width):
    if not np.isfinite(duration) or duration <= 0:
        return np.asarray([0.0])
    stride = max(1.0, width / 2.0)
    last = max(0.0, duration - width)
    s = np.arange(0.0, last + 1e-6, stride)
    if not len(s):
        s = np.asarray([0.0])
    if last - s[-1] > 1e-6:
        s = np.r_[s, last]
    return s


def describe(r, features, windows, schema):
    df = pd.read_csv(r["feature"], usecols=["Timestamp"] + features, low_memory=False)
    if len(df) < 2:
        raise ValueError("fewer than two rows")
    ts = pd.to_numeric(df["Timestamp"], errors="raise").to_numpy(np.float64)
    if not np.isfinite(ts).all():
        raise ValueError("non-finite Timestamp")
    sorted_ok = not np.any(np.diff(ts) < 0)
    if not sorted_ok:
        order = np.argsort(ts, kind="stable")
        ts, df = ts[order], df.iloc[order].reset_index(drop=True)
    X = np.array(df[features].apply(pd.to_numeric, errors="coerce").to_numpy(), dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan
    t = ts - ts[0]
    duration = float(t[-1])
    dt = np.diff(t)
    good_dt = dt[dt > 0]
    med_dt = float(np.median(good_dt)) if len(good_dt) else float("nan")
    rate = 1000.0 / med_dt if med_dt > 0 else float("nan")

    med = q(X, 0.50)
    std = np.nanstd(X, axis=0)
    p10, p90, p99 = q(X, [0.10, 0.90, 0.99])
    mad = q(np.abs(X - med), 0.50)
    valid = dt > 0
    if valid.any():
        dX = np.abs(np.diff(X, axis=0)[valid] / (dt[valid, None] / 1000.0))
        d50, d90 = q(dX, [0.50, 0.90])
    else:
        d50 = np.full(X.shape[1], np.nan)
        d90 = np.full(X.shape[1], np.nan)

    values = np.full(len(schema["names"]), np.nan, np.float32)
    ix = schema["index"]
    for j in range(X.shape[1]):
        vals = {"global_median": med[j], "global_std": std[j], "global_p10": p10[j],
                "global_p90": p90[j], "global_p99": p99[j], "global_mad": mad[j],
                "abs_d1_p50": d50[j], "abs_d1_p90": d90[j]}
        for stat, v in vals.items():
            values[ix[(j, stat)]] = v

    win_counts = {}
    for width in windows:
        wm, ws, wr = [], [], []
        for start in starts_for(duration, width):
            a = int(np.searchsorted(t, start, side="left"))
            b = int(np.searchsorted(t, start + width, side="left"))
            if b <= a:
                continue
            block = X[a:b]
            wmed = q(block, 0.50)
            wstd = np.nanstd(block, axis=0)
            lo, hi = q(block, [0.10, 0.90])
            wm.append(wmed)
            ws.append(wstd)
            wr.append(hi - lo)
        if not wm:
            wm, ws, wr = [med], [std], [p90 - p10]
        wm, ws, wr = np.asarray(wm), np.asarray(ws), np.asarray(wr)
        m10, m50, m90, m99 = q(wm, [0.10, 0.50, 0.90, 0.99])
        s50, s90 = q(ws, [0.50, 0.90])
        r90 = q(wr, 0.90)
        vals = {
            f"w{width}_median_p10": m10, f"w{width}_median_p50": m50,
            f"w{width}_median_p90": m90, f"w{width}_median_p99": m99,
            f"w{width}_std_p50": s50, f"w{width}_std_p90": s90,
            f"w{width}_range_p90": r90,
        }
        for j in range(X.shape[1]):
            for stat, vec in vals.items():
                values[ix[(j, stat)]] = vec[j]
        win_counts[str(width)] = len(wm)

    meta = {
        "file": str(r["raw"].relative_to(ROOT)) if r["raw"].is_relative_to(ROOT) else str(r["raw"]),
        "feature_file": str(r["feature"].relative_to(ROOT)) if r["feature"].is_relative_to(ROOT) else str(r["feature"]),
        "class": r["class"], "subtype": r["subtype"], "participant": r["participant"],
        "source_root": r["root"].name, "content_sha256": r["sha"],
        "rows": len(X), "duration_ms": duration, "median_dt_ms": med_dt,
        "rows_per_second": rate, "window_counts": json.dumps(win_counts, sort_keys=True),
        "finite_fraction": float(np.isfinite(X).mean()), "timestamp_sorted": sorted_ok,
    }
    return values, meta


def fingerprint(rows, features, windows, lean8_info):
    h = hashlib.sha256()
    h.update(VERSION.encode())
    h.update(json.dumps(features).encode())
    h.update(json.dumps(windows).encode())
    h.update(str(bool(lean8_info.get("lean8_used"))).encode())
    for r in rows:
        p = r["feature"]
        st = p.stat()
        h.update(f"{r['sha']}|{p}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def build_descriptors(rows, features, windows, out, lean8_info, rebuild=False):
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    npz = cache / "recording_descriptors.npz"
    csv = cache / "recordings.csv"
    stamp = cache / "fingerprint.json"
    schema = schema_for(features, windows)
    fp = fingerprint(rows, features, windows, lean8_info)
    if not rebuild and npz.is_file() and csv.is_file() and stamp.is_file():
        try:
            if json.loads(stamp.read_text()).get("fingerprint") == fp:
                with np.load(npz, allow_pickle=False) as z:
                    X = z["X"]
                meta = pd.read_csv(csv)
                log("CACHE_REUSED", X.shape)
                return X, meta, schema, fp
        except Exception as e:
            log("CACHE_INVALID", repr(e)[:200])

    arrays, metadata, failed = [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        try:
            a, m = describe(r, features, windows, schema)
            arrays.append(a)
            metadata.append(m)
        except Exception as e:
            failed.append({"file": str(r["raw"]), "error": repr(e)})
        if i % 100 == 0:
            log("DESCRIBE", i, "/", len(rows), "valid", len(arrays), "failed", len(failed),
                "elapsed_s", round(time.perf_counter() - t0, 1))
    if not arrays:
        raise RuntimeError("No recordings could be described")
    X = np.vstack(arrays).astype(np.float32, copy=False)
    meta = pd.DataFrame(metadata)
    tmp = cache / "recording_descriptors.tmp.npz"
    np.savez_compressed(tmp, X=X)
    os.replace(tmp, npz)
    meta.to_csv(csv, index=False)
    stamp.write_text(json.dumps({"fingerprint": fp, "version": VERSION}, indent=2))
    if failed:
        pd.DataFrame(failed).to_csv(out / "failed_recordings.csv", index=False)
    log("DESCRIPTORS_READY", X.shape, "seconds", round(time.perf_counter() - t0, 1), "failed", len(failed))
    return X, meta, schema, fp


def bh(p):
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    good = np.isfinite(p)
    if not good.any():
        return out
    vals = p[good]
    order = np.argsort(vals)
    ranked = vals[order]
    qv = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    tmp = np.empty(len(qv))
    tmp[order] = np.clip(qv, 0, 1)
    out[good] = tmp
    return out


def family(name):
    s = name.lower()
    if ".cload_lean_8." in s:
        return "LEAN8"
    if "acc_" in s:
        return "ACC"
    if "emg_" in s:
        return "EMG"
    if "wl" in s:
        return "WL"
    return "OTHER"


def model_stats(windows):
    names = list(GLOBAL_STATS)
    for w in windows:
        names += [f"w{w}_{stat}" for stat in WINDOW_STATS]
    return names


def feature_effects(X, meta, features, schema, windows):
    ix = schema["index"]
    stats = model_stats(windows)
    rows = []
    labels = meta["class"].to_numpy()
    for cls in sorted(meta["class"].unique()):
        pos, neg = np.flatnonzero(labels == cls), np.flatnonzero(labels != cls)
        if len(pos) < 2 or len(neg) < 2:
            continue
        local = []
        for j, f in enumerate(features):
            for stat in stats:
                a = X[pos, ix[(j, stat)]].astype(float)
                b = X[neg, ix[(j, stat)]].astype(float)
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                if len(a) < 2 or len(b) < 2:
                    continue
                try:
                    u, pv = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
                    delta = 2.0 * float(u) / (len(a) * len(b)) - 1.0
                except Exception:
                    pv, delta = np.nan, np.nan
                local.append({
                    "class": cls, "feature": f, "family": family(f), "stat": stat,
                    "n_class": len(a), "n_other": len(b),
                    "class_median": float(np.median(a)), "other_median": float(np.median(b)),
                    "median_difference": float(np.median(a) - np.median(b)),
                    "cliffs_delta": float(delta), "p_value": float(pv),
                })
        if local:
            for r, qv in zip(local, bh([x["p_value"] for x in local])):
                r["q_value"] = float(qv)
            rows.extend(local)
    return pd.DataFrame(rows)


def subtype_effects(X, meta, features, schema, windows):
    """Within-folder subtype contrasts (e.g. SoftTap vs MidTap), still recording-level only."""
    ix = schema["index"]
    stats = list(GLOBAL_STATS)
    for w in windows:
        stats += [f"w{w}_{stat}" for stat in WINDOW_STATS]
    rows = []
    m = meta.reset_index(drop=True)
    for cls in sorted(m["class"].unique()):
        in_class = np.flatnonzero(m["class"].to_numpy() == cls)
        subtypes = sorted(m.iloc[in_class]["subtype"].unique())
        if len(subtypes) < 2:
            continue
        for sub in subtypes:
            pos = in_class[m.iloc[in_class]["subtype"].to_numpy() == sub]
            neg = in_class[m.iloc[in_class]["subtype"].to_numpy() != sub]
            if len(pos) < 2 or len(neg) < 2:
                continue
            local = []
            for j, feature in enumerate(features):
                for stat in stats:
                    col = ix[(j, stat)]
                    a = X[pos, col].astype(float)
                    b = X[neg, col].astype(float)
                    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                    if len(a) < 2 or len(b) < 2:
                        continue
                    try:
                        u, pv = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
                        delta = 2.0 * float(u) / (len(a) * len(b)) - 1.0
                    except Exception:
                        pv, delta = np.nan, np.nan
                    local.append({
                        "class": cls, "subtype": sub, "feature": feature, "family": family(feature),
                        "stat": stat, "n_subtype_recordings": len(a), "n_other_subtype_recordings": len(b),
                        "subtype_median": float(np.median(a)), "other_subtype_median": float(np.median(b)),
                        "median_difference": float(np.median(a) - np.median(b)),
                        "cliffs_delta": float(delta), "p_value": float(pv),
                    })
            if local:
                for row, qv in zip(local, bh([x["p_value"] for x in local])):
                    row["q_value"] = float(qv)
                rows.extend(local)
    return pd.DataFrame(rows)


def paired_effects(X, meta, features, schema, windows, min_pairs=5):
    """Paired participant contrasts, with participant medians aggregated once per class."""
    ix = schema["index"]
    stats = list(GLOBAL_STATS)
    for w in windows:
        stats += [f"w{w}_{stat}" for stat in WINDOW_STATS]
    descriptors = [(j, stat, ix[(j, stat)]) for j in range(len(features)) for stat in stats]
    columns = [f"{j}:{stat}" for j, stat, _ in descriptors]
    m = meta.reset_index(drop=True)
    classes = sorted(m["class"].unique())

    # One groupby per class, not one groupby per feature and class-pair.
    participant_medians = {}
    for cls in classes:
        idx_rows = np.flatnonzero(m["class"].to_numpy() == cls)
        frame = pd.DataFrame(X[idx_rows][:, [d[2] for d in descriptors]], columns=columns)
        frame["participant"] = m.iloc[idx_rows]["participant"].to_numpy()
        participant_medians[cls] = frame.groupby("participant", sort=True).median(numeric_only=True)

    out = []
    for ca, cb in itertools.combinations(classes, 2):
        a_df, b_df = participant_medians[ca], participant_medians[cb]
        common = sorted(set(a_df.index) & set(b_df.index))
        if len(common) < min_pairs:
            continue
        A = a_df.loc[common].to_numpy(np.float64)
        B = b_df.loc[common].to_numpy(np.float64)
        D = A - B
        pvals = np.full(D.shape[1], np.nan, dtype=np.float64)
        complete = np.isfinite(A).all(axis=0) & np.isfinite(B).all(axis=0)
        zero = complete & np.all(np.isclose(D, 0.0), axis=0)
        pvals[zero] = 1.0
        active = complete & ~zero
        if active.any():
            try:
                pvals[active] = np.asarray(
                    wilcoxon(A[:, active], B[:, active], axis=0, zero_method="wilcox",
                             alternative="two-sided").pvalue
                ).reshape(-1)
            except Exception:
                for k in np.flatnonzero(active):
                    try:
                        pvals[k] = float(wilcoxon(D[:, k], zero_method="wilcox").pvalue)
                    except Exception:
                        pass
        # A few columns may contain missing values; test those independently.
        for k in np.flatnonzero(~complete):
            good = np.isfinite(A[:, k]) & np.isfinite(B[:, k])
            if good.sum() < min_pairs:
                continue
            d = D[good, k]
            if np.allclose(d, 0.0):
                pvals[k] = 1.0
            else:
                try:
                    pvals[k] = float(wilcoxon(d, zero_method="wilcox").pvalue)
                except Exception:
                    pass

        local = []
        for k, (j, stat, _) in enumerate(descriptors):
            good = np.isfinite(A[:, k]) & np.isfinite(B[:, k])
            if good.sum() < min_pairs:
                continue
            d = D[good, k]
            local.append({
                "class_a": ca, "class_b": cb, "feature": features[j],
                "family": family(features[j]), "stat": stat,
                "n_paired_participants": int(good.sum()),
                "median_a": float(np.median(A[good, k])),
                "median_b": float(np.median(B[good, k])),
                "median_paired_difference": float(np.median(d)),
                "fraction_a_gt_b": float(np.mean(d > 0)),
                "p_value": float(pvals[k]),
            })
        if local:
            for row, qv in zip(local, bh([x["p_value"] for x in local])):
                row["q_value"] = float(qv)
            out.extend(local)
    return pd.DataFrame(out)

def within_class_corr(X, meta, features, schema):
    cols = [schema["index"][(j, "global_median")] for j in range(len(features))]
    A = X[:, cols].astype(float)
    for j in range(A.shape[1]):
        good = np.isfinite(A[:, j])
        A[~good, j] = np.median(A[good, j]) if good.any() else 0.0
    residual_parts = []
    for cls in sorted(meta["class"].unique()):
        idx = np.flatnonzero(meta["class"].to_numpy() == cls)
        if len(idx) < 3:
            continue
        ranks = rankdata(A[idx], axis=0, method="average")
        residual_parts.append(ranks - ranks.mean(axis=0, keepdims=True))
    if not residual_parts:
        return np.eye(len(features), dtype=np.float64)
    residual = np.vstack(residual_parts)
    corr = np.corrcoef(residual, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def correlation_reports(corr, features, out):
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pairs.append({
                "feature_a": features[i], "feature_b": features[j],
                "family_a": family(features[i]), "family_b": family(features[j]),
                "within_class_spearman": float(corr[i, j]),
                "absolute_correlation": abs(float(corr[i, j])),
            })
    pair_df = pd.DataFrame(pairs).sort_values("absolute_correlation", ascending=False)
    pair_df.to_csv(out / "feature_correlation_pairs.csv", index=False)
    n = len(features)
    labels = np.ones(n, dtype=int)
    best_k, best_score = n, np.nan
    if n >= 3:
        dist = np.clip(1.0 - np.abs(corr), 0, 1)
        np.fill_diagonal(dist, 0)
        tree = linkage(squareform(dist, checks=False), method="average")
        try:
            from sklearn.metrics import silhouette_score
            best_score = -np.inf
            for k in range(2, n):
                candidate = fcluster(tree, t=k, criterion="maxclust")
                if 1 < len(np.unique(candidate)) < n:
                    score = silhouette_score(dist, candidate, metric="precomputed")
                    if score > best_score:
                        best_score, best_k, labels = float(score), len(np.unique(candidate)), candidate
            if not np.isfinite(best_score):
                labels = fcluster(tree, t=max(2, int(math.sqrt(n))), criterion="maxclust")
                best_k = len(np.unique(labels))
                best_score = np.nan
        except Exception:
            labels = fcluster(tree, t=max(2, int(math.sqrt(n))), criterion="maxclust")
            best_k = len(np.unique(labels))
    clusters = pd.DataFrame({"feature": features, "family": [family(x) for x in features],
                             "cluster": labels, "cluster_count": best_k, "silhouette": best_score})
    clusters.to_csv(out / "feature_correlation_clusters.csv", index=False)
    np.savetxt(out / "within_class_spearman.csv", corr, delimiter=",", fmt="%.6f")
    return pair_df, clusters


def impute(train, valid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0).astype(np.float32)
    return (
        np.where(np.isfinite(train), train, med[None, :]).astype(np.float32),
        np.where(np.isfinite(valid), valid, med[None, :]).astype(np.float32),
    )


def group_weights(y, groups):
    counts = Counter(groups.tolist())
    w = np.asarray([1.0 / counts[g] for g in groups], dtype=float)
    for cls in (0, 1):
        mask = y == cls
        if w[mask].sum() > 0:
            w[mask] *= 0.5 / w[mask].sum()
    return (w * len(w)).astype(np.float32)


def grouped_models(X, meta, features, schema, windows, out, folds=5, seed=20261118):
    idx = schema["model_indices"]
    base = schema["model_base"]
    groups = meta["participant"].astype(str).to_numpy()
    labels = meta["class"].astype(str).to_numpy()
    metric_rows, importance_rows, skipped = [], [], []
    for cls in sorted(meta["class"].unique()):
        y = (labels == cls).astype(np.int8)
        npos = len(set(groups[y == 1]))
        nneg = len(set(groups[y == 0]))
        nf = min(folds, npos, nneg)
        if nf < 3:
            skipped.append({"class": cls, "reason": "fewer than three independent groups in a class",
                            "positive_recordings": int(y.sum()), "positive_groups": npos, "negative_groups": nneg})
            continue
        splitter = StratifiedGroupKFold(n_splits=nf, shuffle=True, random_state=seed)
        imp_rows = []
        log("CLASS_MODEL_BEGIN", cls, "positive_recordings", int(y.sum()), "groups", npos, "folds", nf)
        for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                skipped.append({"class": cls, "fold": fold, "reason": "fold lacks both classes"})
                continue
            a, b = impute(X[tr][:, idx], X[va][:, idx])
            weights = group_weights(y[tr], groups[tr])
            if LGBMClassifier is not None:
                model = LGBMClassifier(
                    objective="binary", n_estimators=250, learning_rate=0.05,
                    num_leaves=31, max_depth=-1, min_child_samples=15,
                    colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                    reg_lambda=1.0, max_bin=127, n_jobs=THREADS,
                    random_state=seed + fold, deterministic=True,
                    force_col_wise=True, verbosity=-1,
                )
            else:
                model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                                             class_weight="balanced", n_jobs=THREADS, random_state=seed + fold)
            model.fit(a, y[tr], sample_weight=weights)
            prob = model.predict_proba(b)[:, 1]
            pred = (prob >= 0.5).astype(np.int8)
            row = {
                "class": cls, "fold": fold, "folds": nf, "train_recordings": len(tr),
                "valid_recordings": len(va), "valid_positive": int(y[va].sum()),
                "valid_participants": len(set(groups[va])),
                "roc_auc": float(roc_auc_score(y[va], prob)),
                "average_precision": float(average_precision_score(y[va], prob)),
                "balanced_accuracy_05": float(balanced_accuracy_score(y[va], pred)),
                "sensitivity_05": float(recall_score(y[va], pred, zero_division=0)),
                "backend": BACKEND,
            }
            metric_rows.append(row)
            if LGBMClassifier is not None:
                gain = model.booster_.feature_importance(importance_type="gain")
                split = model.booster_.feature_importance(importance_type="split")
            else:
                gain, split = model.feature_importances_, np.zeros(len(idx))
            gain_base = np.bincount(base, weights=gain, minlength=len(features))
            split_base = np.bincount(base, weights=split, minlength=len(features))
            for j, f in enumerate(features):
                imp_rows.append({"class": cls, "fold": fold, "feature": f, "family": family(f),
                                 "gain": float(gain_base[j]), "split": float(split_base[j])})
            log("CLASS_MODEL_FOLD", cls, fold, "auc", round(row["roc_auc"], 4),
                "ap", round(row["average_precision"], 4))
        if imp_rows:
            part = pd.DataFrame([r for r in imp_rows if r["class"] == cls])
            if not part.empty:
                agg = part.groupby(["class", "feature", "family"], as_index=False).agg(
                    gain_mean=("gain", "mean"), gain_sd=("gain", "std"),
                    split_mean=("split", "mean"), folds_present=("fold", "nunique"))
                total = agg["gain_mean"].sum()
                agg["gain_share"] = agg["gain_mean"] / total if total else 0.0
                agg["gain_rank"] = agg["gain_mean"].rank(ascending=False, method="min")
                importance_rows.extend(agg.to_dict("records"))
    metrics = pd.DataFrame(metric_rows)
    importance = pd.DataFrame(importance_rows)
    if not metrics.empty:
        metrics.to_csv(out / "recording_level_cv_metrics.csv", index=False)
    if not importance.empty:
        importance.to_csv(out / "recording_level_feature_importance.csv", index=False)
    pd.DataFrame(skipped).to_csv(out / "model_skips.csv", index=False)
    return metrics, importance, skipped


def summaries(meta):
    class_rows = []
    for cls, g in meta.groupby("class"):
        class_rows.append({
            "class": cls, "recordings": len(g), "participants": g["participant"].nunique(),
            "subtypes": g["subtype"].nunique(), "duration_hours": g["duration_ms"].sum() / 3_600_000,
            "median_duration_s": g["duration_ms"].median() / 1000,
            "median_rows": g["rows"].median(), "median_rows_per_second": g["rows_per_second"].median(),
            "median_finite_fraction": g["finite_fraction"].median(),
        })
    sub_rows = []
    for (cls, sub), g in meta.groupby(["class", "subtype"]):
        sub_rows.append({"class": cls, "subtype": sub, "recordings": len(g),
                         "participants": g["participant"].nunique(),
                         "duration_hours": g["duration_ms"].sum() / 3_600_000})
    return (pd.DataFrame(class_rows).sort_values("recordings", ascending=False),
            pd.DataFrame(sub_rows).sort_values(["class", "recordings"], ascending=[True, False]))


def top_effects(effects, cls, n=12):
    if effects.empty:
        return pd.DataFrame()
    d = effects[effects["class"] == cls].copy()
    if d.empty:
        return d
    d["abs_delta"] = d["cliffs_delta"].abs()
    return d.sort_values(["q_value", "abs_delta"], ascending=[True, False]).drop_duplicates("feature").head(n)


def markdown_table(df, cols, n=30):
    if df is None or df.empty or not cols:
        return "_No results available._"
    d = df[cols].head(n).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    head = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |"
            for row in d.itertuples(index=False, name=None)]
    return "\n".join([head, sep] + body)


def plots(meta, corr, features, effects, metrics, out):
    if plt is None:
        return
    d = out / "plots"
    d.mkdir(parents=True, exist_ok=True)
    counts = meta["class"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(counts))))
    counts.plot(kind="barh", ax=ax)
    ax.set_title("Recording counts by filename class")
    fig.tight_layout()
    fig.savefig(d / "class_counts.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Within-class feature correlation")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(d / "within_class_correlation.png", dpi=150)
    plt.close(fig)

    if not effects.empty:
        classes = [c for c in ("TAP", "SWIPE", "FINGER", "WRIST", "ARM", "BAND", "NOTHING") if c in set(effects["class"])]
        mat = np.zeros((len(classes), len(features)), dtype=np.float32)
        for i, cls in enumerate(classes):
            for j, f in enumerate(features):
                v = effects[(effects["class"] == cls) & (effects["feature"] == f)]["cliffs_delta"].to_numpy(float)
                if len(v) and np.isfinite(v).any():
                    mat[i, j] = v[np.nanargmax(np.abs(v))]
        if len(classes):
            fig, ax = plt.subplots(figsize=(16, max(3, len(classes) * 0.6)))
            im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_yticks(np.arange(len(classes)), labels=classes)
            ax.set_xlabel("Feature index; see feature schema for names")
            ax.set_title("Largest one-vs-rest effect per feature")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(d / "class_feature_effects.png", dpi=150)
            plt.close(fig)

    if not metrics.empty:
        auc = metrics.groupby("class")["roc_auc"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, max(4, len(auc) * 0.35)))
        auc.plot(kind="barh", ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title("Person-grouped recording-level ROC-AUC")
        fig.tight_layout()
        fig.savefig(d / "recording_level_auc.png", dpi=150)
        plt.close(fig)


def report(out, roots, lean8, raw_count, duplicates, conflicts, features, availability, windows, meta,
           class_df, subtype_df, effects, subtype_effect_df, paired, corr_pairs, clusters,
           metrics, importance, skipped, fp):
    classes = sorted(meta["class"].unique())
    lines = [
        "# Gesture correlation research",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Script version: {VERSION}",
        "",
        "## Ground-truth contract",
        "",
        "The class is the immediate recording-folder name. Filename tokens provide subtype descriptions. "
        "The filename prefix is used only to group participant recordings during validation. "
        "No LABELS, PointerState, event table, teacher output, tap_label, hold_label, or CSV User column is read.",
        "",
        "Each complete recording is one labelled bag. A TAP file can contain tap windows and rest windows; "
        "the script never pretends every row or window is a tap. Fixed-time window summaries are pooled into recording-level descriptors.",
        "",
        f"- Raw files discovered: {raw_count}",
        f"- Unique recordings analyzed: {len(meta)}",
        f"- Identical CSV-content duplicates removed: {len(duplicates)}",
        f"- Conflicting duplicate hashes excluded: {len(conflicts)}",
        f"- Filename classes: {len(classes)}",
        f"- Filename-derived participant groups: {meta['participant'].nunique()}",
        f"- Firmware features: {len(features)}",
        f"- Window scales (ms): {windows}",
        f"- Lean-8 status: {json.dumps(lean8, sort_keys=True)}",
        f"- Descriptor fingerprint: {fp}",
        f"- p_core fields not common to every file: {int((~availability['used_in_analysis']).sum())}",
        "",
        "## Feature schema availability",
        "",
        "Only fields present in every recording are included in the analysis matrix. The complete union and per-field coverage are in feature_availability.csv; no missing field is silently hidden.",
        "",
        markdown_table(availability[availability["used_in_analysis"] == False].head(30),
                       ["feature", "files_present", "files_total", "presence_fraction"], 30),
        "",
        "## Class inventory",
        "",
        markdown_table(class_df, ["class", "recordings", "participants", "subtypes", "duration_hours",
                                  "median_duration_s", "median_rows_per_second"], 60),
        "",
        "## Filename subtypes",
        "",
        markdown_table(subtype_df, ["class", "subtype", "recordings", "participants", "duration_hours"], 100),
        "",
        "## Person-grouped recording-level classification",
        "",
        "One-vs-rest models use pooled p_core descriptors only. Entire participants are held out per fold. "
        "This measures whether a recording's sensor profile identifies its filename class; it is not per-tap event recall.",
        "",
    ]
    if not metrics.empty:
        agg = metrics.groupby("class", as_index=False).agg(
            folds=("fold", "nunique"), valid_recordings=("valid_recordings", "sum"),
            roc_auc=("roc_auc", "mean"), average_precision=("average_precision", "mean"),
            balanced_accuracy=("balanced_accuracy_05", "mean"), sensitivity=("sensitivity_05", "mean"),
        ).sort_values("roc_auc", ascending=False)
        lines += [markdown_table(agg, ["class", "folds", "valid_recordings", "roc_auc",
                                       "average_precision", "balanced_accuracy", "sensitivity"], 60), ""]
    else:
        lines += ["_Models skipped or no class had enough independent participant groups._", ""]
    if skipped:
        lines += ["### Skipped model classes/folds", "", markdown_table(pd.DataFrame(skipped), list(pd.DataFrame(skipped).columns)[:6], 60), ""]

    lines += ["## Features distinguishing each class", "",
              "One-vs-rest recording-level distributions. Cliff's delta is positive when a descriptor tends to be larger in the target class. "
              "Unpaired p-values are descriptive because a person can contribute multiple recordings; use paired participant contrasts below for person-controlled comparisons.", ""]
    for cls in classes:
        top = top_effects(effects, cls, 12)
        lines += [f"### {cls} vs all other classes", "",
                  markdown_table(top, ["feature", "family", "stat", "class_median", "other_median", "cliffs_delta", "q_value"], 12), ""]
        if not importance.empty:
            imp = importance[importance["class"] == cls].sort_values("gain_share", ascending=False).head(10)
            if not imp.empty:
                lines += ["Model gain importance (diagnostic, not causal):", "",
                          markdown_table(imp, ["feature", "family", "gain_share", "gain_rank", "folds_present"], 10), ""]

    lines += ["## Differences between filename subtypes", "",
              "Within each broad folder class, these tests compare one filename subtype with the other subtypes in the same folder. "
              "For example, SoftTap vs MidTap is compared inside TAP recordings rather than against unrelated classes.", ""]
    if not subtype_effect_df.empty:
        for cls in ("TAP", "SWIPE"):
            subcls = subtype_effect_df[subtype_effect_df["class"] == cls]
            if subcls.empty:
                continue
            for subtype_name in sorted(subcls["subtype"].unique()):
                sub = subcls[subcls["subtype"] == subtype_name].copy()
                sub["abs_delta"] = sub["cliffs_delta"].abs()
                sub = sub.sort_values(["q_value", "abs_delta"], ascending=[True, False]).drop_duplicates("feature").head(8)
                lines += [f"### {cls} / {subtype_name}", "",
                          markdown_table(sub, ["feature", "stat", "n_subtype_recordings",
                                               "n_other_subtype_recordings", "median_difference",
                                               "cliffs_delta", "q_value"], 8), ""]
    else:
        lines += ["_No within-class subtype comparisons had enough recordings._", ""]

    lines += ["## Paired participant contrasts", "",
              "These compare participant-level medians for two classes only where the same filename-derived participant has both classes.", ""]
    if not paired.empty:
        for a, b in (("TAP", "FINGER"), ("TAP", "WRIST"), ("TAP", "ARM"), ("TAP", "BAND"),
                     ("TAP", "NOTHING"), ("SWIPE", "FINGER"), ("SWIPE", "WRIST"),
                     ("SWIPE", "ARM"), ("SWIPE", "BAND"), ("SWIPE", "NOTHING")):
            sub = paired[((paired["class_a"] == a) & (paired["class_b"] == b)) |
                         ((paired["class_a"] == b) & (paired["class_b"] == a))].copy()
            if sub.empty:
                continue
            sub["abs_diff"] = sub["median_paired_difference"].abs()
            sub = sub.sort_values(["q_value", "abs_diff"], ascending=[True, False]).drop_duplicates("feature").head(10)
            ca, cb = sub["class_a"].iloc[0], sub["class_b"].iloc[0]
            lines += [f"### {ca} vs {cb}", "",
                      markdown_table(sub, ["class_a", "class_b", "feature", "stat", "n_paired_participants",
                                           "median_paired_difference", "fraction_a_gt_b", "q_value"], 10), ""]
    else:
        lines += ["_No class pair had enough overlapping participants for paired inference._", ""]

    lines += [
        "## Correlated firmware features", "",
        "Correlation uses within-class rank residuals of recording-level global medians, reducing correlations caused only by class mean differences. "
        "Clusters are data-driven diagnostics; no feature family is forced or removed.", "",
        markdown_table(corr_pairs, ["feature_a", "feature_b", "within_class_spearman"], 30), "",
        f"Data-driven correlation clusters: {clusters['cluster'].nunique()}.", "",
        "## Important limitations", "",
        "1. Filename/folder labels describe a recording task, not exact gesture timestamps. This study cannot claim per-tap recall or latency.",
        "2. The recording is the independent labelled unit. Window quantiles describe within-recording patterns without frame-level labels.",
        "3. Participant grouping is inferred from the first filename token using the existing TAP research alias convention. Inspect recordings.csv for incorrect merges.",
        "4. Duration, row count, sampling cadence, filename tokens, participant and folder are diagnostics or grouping metadata, never model inputs.",
        "5. Recording sessions and collection protocols may still confound class differences. Compare acquisition diagnostics and participant-paired results.",
        "6. Feature gain is not causal importance. Correlated features can substitute for each other; use the paired effects and correlation reports together.",
        "7. This is a research classifier, not a firmware model. It does not establish thresholds, event timing, model size, or C parity.",
        "8. If Lean-8 C parity is not passed against the actual firmware header, the Lean-8 portion is not yet firmware-verified.",
        "",
        "## Output files", "",
        "- recordings.csv: one row per unique recording, filename-derived metadata and acquisition diagnostics.",
        "- cache/recording_descriptors.npz: cached recording-level descriptor matrix.",
        "- descriptor_schema.json: mapping from descriptors to original p_core features.",
        "- feature_effects.csv.gz: one-vs-rest recording-level effects.",
        "- paired_participant_effects.csv.gz: paired participant comparisons.",
        "- feature_correlation_pairs.csv and feature_correlation_clusters.csv: feature redundancy.",
        "- recording_level_cv_metrics.csv: grouped one-vs-rest classification.",
        "- recording_level_feature_importance.csv: model gain/split importance aggregated to original p_core fields.",
        "- plots/: class counts, feature correlations, class effects and CV results.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_windows(s):
    try:
        vals = sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    if not vals or min(vals) <= 0:
        raise argparse.ArgumentTypeError("window sizes must be positive milliseconds")
    return vals


def main():
    ap = argparse.ArgumentParser(description="Filename-supervised Mudra sensor-pattern research")
    ap.add_argument("--roots", nargs="+", type=Path, default=DEFAULT_ROOTS)
    ap.add_argument("--out", type=Path, default=ROOT / "gesture_correlation" / "results")
    ap.add_argument("--lean8", choices=("auto", "required", "off"), default="auto")
    ap.add_argument("--lean8-parity-files", type=int, default=24)
    ap.add_argument("--windows-ms", type=parse_windows, default=[100, 200, 400])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--min-paired-participants", type=int, default=5)
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    roots = [p.resolve() for p in args.roots]
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log("GESTURE_CORRELATION_START", VERSION, "threads", THREADS)
    log("GROUND_TRUTH", "folder class + filename subtype; participant prefix only for grouped CV")
    log("NOT_USED", "LABELS", "PointerState", "event tables", "teacher outputs", "CSV User")

    raw = scan(roots)
    log("FILES_DISCOVERED", len(raw))
    rows, duplicates, conflicts = deduplicate(raw)
    log("DEDUPLICATED", len(rows), "duplicates", len(duplicates), "conflicts", len(conflicts))
    pd.DataFrame(duplicates).to_csv(out / "duplicates_removed.csv", index=False)
    pd.DataFrame(conflicts).to_csv(out / "duplicate_class_conflicts.csv", index=False)

    rows, lean8 = prepare_features(rows, roots, args.lean8, args.lean8_parity_files)
    features, availability = common_columns(rows)
    availability.to_csv(out / "feature_availability.csv", index=False)
    missing_schema = availability[availability["used_in_analysis"] == False]
    if not missing_schema.empty:
        log("FEATURE_SCHEMA_MISSING", len(missing_schema), "fields are not present in every recording; see feature_availability.csv")
    log("FEATURE_SCHEMA", len(features), "lean8", lean8.get("lean8_used"))
    X, meta, schema, fp = build_descriptors(rows, features, args.windows_ms, out, lean8, args.rebuild_cache)
    meta.to_csv(out / "recordings.csv", index=False)
    (out / "descriptor_schema.json").write_text(json.dumps({
        "feature_columns": features, "descriptor_names": schema["names"],
        "descriptor_meta": schema["meta"], "model_descriptor_names": schema["model_names"],
        "windows_ms": args.windows_ms,
    }, indent=2), encoding="utf-8")

    class_df, subtype_df = summaries(meta)
    class_df.to_csv(out / "class_summary.csv", index=False)
    subtype_df.to_csv(out / "subtype_summary.csv", index=False)

    log("FEATURE_EFFECTS_BEGIN")
    effects = feature_effects(X, meta, features, schema, args.windows_ms)
    if not effects.empty:
        effects.to_csv(out / "feature_effects.csv.gz", index=False, compression="gzip")
    log("SUBTYPE_EFFECTS_BEGIN")
    subtype_effect_df = subtype_effects(X, meta, features, schema, args.windows_ms)
    if not subtype_effect_df.empty:
        subtype_effect_df.to_csv(out / "subtype_feature_effects.csv.gz", index=False, compression="gzip")
    log("PAIRED_EFFECTS_BEGIN")
    paired = paired_effects(X, meta, features, schema, args.windows_ms, args.min_paired_participants)
    if not paired.empty:
        paired.to_csv(out / "paired_participant_effects.csv.gz", index=False, compression="gzip")

    log("CORRELATION_BEGIN")
    corr = within_class_corr(X, meta, features, schema)
    corr_pairs, clusters = correlation_reports(corr, features, out)

    metrics, importance, skipped = pd.DataFrame(), pd.DataFrame(), []
    if not args.skip_models:
        log("GROUPED_MODELS_BEGIN", "classes", meta["class"].nunique(), "folds", args.folds, "backend", BACKEND)
        metrics, importance, skipped = grouped_models(X, meta, features, schema, args.windows_ms, out, args.folds, args.seed)

    plots(meta, corr, features, effects, metrics, out)
    report(out, roots, lean8, len(raw), duplicates, conflicts, features, availability, args.windows_ms,
           meta, class_df, subtype_df, effects, subtype_effect_df, paired, corr_pairs, clusters,
           metrics, importance, skipped, fp)

    manifest = {
        "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "roots": [str(x) for x in roots], "output": str(out),
        "raw_files": len(raw), "unique_recordings": len(meta),
        "duplicates": len(duplicates), "conflicting_duplicate_hashes": len(conflicts),
        "feature_count": len(features), "feature_columns": features,
        "feature_availability": availability.to_dict("records"),
        "windows_ms": args.windows_ms, "lean8": lean8,
        "classes": class_df.to_dict("records"), "participants": int(meta["participant"].nunique()),
        "descriptor_shape": list(X.shape), "fingerprint": fp,
        "model_backend": BACKEND, "threads": THREADS,
        "arguments": {"folds": args.folds, "seed": args.seed, "skip_models": args.skip_models,
                      "min_paired_participants": args.min_paired_participants},
        "ground_truth": "folder class + filename subtype; filename prefix only for participant grouping",
        "model_inputs": "p_core sensor features pooled within each recording",
        "excluded_inputs": ["PointerState", "LABELS", "tap_label", "hold_label", "event tables",
                            "teacher outputs", "CSV User", "filename tokens", "participant ID",
                            "folder name", "timestamps as features", "duration as features"],
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log("GESTURE_CORRELATION_DONE", out / "REPORT.md", "recordings", len(meta),
        "features", len(features), "elapsed_s", round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    main()
