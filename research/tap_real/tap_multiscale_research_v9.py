#!/usr/bin/env python3
"""V9 exhaustive, resumable Mudra TAP pattern experiment (research; NOT gold labeling).

Source: DATA/DATA_TAP/{TAP,SWIPE,FINGER,WRIST,ARM,BAND,NOTHING}/*.csv.gz.
Only recording folders are trusted labels; events are physiological hypotheses.

Commands:
    extract   audit/de-duplicate and cache event-centred full PSD trajectories
    search    cross-person/conditional cross-session multi-scale experiments
    all       extract then search; never changes a source file

Dependencies: numpy pandas scipy scikit-learn lightgbm.
Read research/TAP_V9_EXHAUSTIVE_README.md for procedures and interpretation.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

# This is a genuine signal-level baseline; it cannot turn a bag label into onset truth.
from tap_event_shape_v3 import EMG, WL, ACC, Config, extract_signals, extract_events

# Hardcoded CPU budget. Logical threads are affinity-aware (Linux/cpuset),
# falling back to the machine's logical CPU count. Always reserve ~20%.
CPU_THREAD_FRACTION = 0.80

def available_logical_threads() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)

LOGICAL_THREADS = available_logical_threads()
WORKER_THREADS = max(1, int(LOGICAL_THREADS * CPU_THREAD_FRACTION))

VERSION = "9.0.1-research"
SEED = 20261116
CATEGORIES = ("TAP", "SWIPE", "FINGER", "WRIST", "ARM", "BAND", "NOTHING")
NEGATIVE = frozenset(("FINGER", "WRIST", "ARM", "BAND", "NOTHING"))
EMG_COLS = [name for ch in EMG for name in ch]
WL_COLS = [name for ch in WL for name in ch]
ACC_COLS = [name for ch in ACC for name in ch]
REQUIRED = ["Timestamp"] + EMG_COLS
OPTIONAL = ["User", "PointerState", "Metronome"] + WL_COLS + ACC_COLS


def atom_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atom_csv(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    os.replace(tmp, path)


def sha_uncompressed(path: Path) -> str:
    d = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def canonical_person(filename: str, user: Any) -> str:
    """Conservative identity grouping; unknown aliases remain an explicit uncertainty."""
    x = re.sub(r"[^a-z]", "", str(user if pd.notna(user) and str(user).strip() else filename.split("_")[0]).lower())
    x = re.sub(r"(lefthand|righthand|left|right|noise|psd[0-9]*|hand)$", "", x)
    for prefix in ("ohad", "alex", "neomi", "yossef", "noam", "michael", "perry", "nikol", "shay", "eitan", "kirill", "sarina", "adi", "samuel"):
        if x.startswith(prefix):
            return prefix
    return x or "unknown"


def session_date(filename: str) -> str:
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", filename)
    return "-".join(m.groups()) if m else "unknown"


def subtype(filename: str, category: str) -> str:
    lo = filename.lower()
    for name in ("doubletap", "softtap", "strongtap", "midtap", "taponmotion", "slowpinch", "swipesoft", "swipemid"):
        if name in lo:
            return name.upper()
    return category


def collect_source(root: Path, max_files: int = 0) -> tuple[list[dict], dict]:
    if root.name != "DATA_TAP" or not root.is_dir():
        raise ValueError("Input must be the original DATA/DATA_TAP directory")
    files = sorted([p for p in root.rglob("*.csv*") if p.parent.name in CATEGORIES and (p.name.endswith(".csv.gz") or p.name.endswith(".csv"))])
    if max_files:
        # Stratified, deterministic --max-files: cap files PER category, not globally.
        by = Counter()
        selected = []
        for p in files:
            if by[p.parent.name] < max_files:
                selected.append(p)
                by[p.parent.name] += 1
        files = selected
    seen = {}
    records = []
    for i, p in enumerate(files, 1):
        digest = sha_uncompressed(p)
        rel = p.relative_to(root).as_posix()
        row = {"source": rel, "category": p.parent.name, "sha256_uncompressed": digest,
               "size_bytes": p.stat().st_size, "duplicate_of": seen.get(digest, "")}
        seen.setdefault(digest, rel)
        records.append(row)
        if i % 250 == 0:
            print("V9_INDEX", i, len(files), flush=True)
    categories_by_hash = defaultdict(set)
    for row in records:
        categories_by_hash[row["sha256_uncompressed"]].add(row["category"])
    conflicts = {h for h, cats in categories_by_hash.items() if len(cats) > 1}
    # Contradictory folder labels never become trusted training truth.
    unique = [r for r in records if not r["duplicate_of"]
              and r["sha256_uncompressed"] not in conflicts]
    # Include exact source list, compressed sizes and uncompressed digests in config fingerprint.
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    summary = {"source_archives": len(records), "unique_recordings": len(unique),
               "duplicates_excluded": sum(bool(r["duplicate_of"]) for r in records),
               "cross_category_conflict_groups_quarantined": len(conflicts),
               "cross_category_conflict_archives_quarantined": sum(r["sha256_uncompressed"] in conflicts for r in records),
               "per_category_unique": dict(Counter(r["category"] for r in unique)),
               "source_fingerprint_sha256": fingerprint}
    return unique, summary


def _strict_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer", low_memory=False,
                     usecols=lambda name: name in set(REQUIRED + OPTIONAL))
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError("Required original PSD columns missing: " + str(sorted(missing)[:5]))
    return df


def mine_candidates(df: pd.DataFrame, min_gap_ms: float = 40.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Union of V3 stateful episodes and lower-threshold rise peaks.

    Two candidate sources, independent of folder type, pointer and beat phase.
    Rise candidates provide a high-recall complement; neither is verified tap onset.
    """
    S = extract_signals(df, Config())
    events = extract_events(S)
    ts = S["ts"]
    dt = float(S["dt_med"])
    dist = max(1, int(round(min_gap_ms / max(1., dt))))
    source = []
    for e in events:
        source.append((e.peak_ms, float(e.activity), 1))
    for thr in (0.045, 0.075, 0.12):
        peaks, props = find_peaks(S["rise"], height=thr, prominence=thr * 0.35, distance=dist)
        for i in peaks:
            if S["activity"][i] >= 0.025:
                source.append((float(ts[i]), float(S["rise"][i]), 2))
    # Preserve strongest under a close-spacing collision, V3 preferred on exact ties.
    source.sort(key=lambda x: (-x[1], -int(x[2] == 1), x[0]))
    result = []
    for t, strength, kind in source:
        if ts[-1] - t < 650 or t - ts[0] < 350:
            continue
        if all(abs(t - x[0]) >= min_gap_ms for x in result):
            result.append((t, strength, kind))
    result.sort()
    return np.asarray([x[0] for x in result], dtype=np.float64), np.asarray([x[2] for x in result], dtype=np.int8), {
        "v3_events": len(events), "union_events": len(result), "sample_dt_ms": dt,
        "timestamp_gaps_gt_140ms": int(np.sum(np.diff(ts) > 140)),
    }


def extract_one(df: pd.DataFrame, item: dict, time_grid: np.ndarray) -> tuple[list[dict], np.ndarray, dict]:
    ts = pd.to_numeric(df["Timestamp"], errors="coerce").to_numpy(np.float64)
    if len(ts) < 40 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError("Insufficient, invalid or non-increasing Timestamp")
    peaks, kind, qc = mine_candidates(df)
    if not len(peaks):
        return [], np.empty((0, len(time_grid), 3, 5), np.float16), qc
    # Offline, noncausal smoothing deliberately isolated from firmware features.
    raw = df[EMG_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("NaN or infinity in source PSD")
    median_dt = float(np.median(np.diff(ts)))
    sigma = max(0.6, 10. / max(1., median_dt))
    smoothed = gaussian_filter1d(raw, sigma=sigma, axis=0)
    # Enforce fixed windows across candidate experiments; reject missing data spans.
    gaps = np.flatnonzero(np.diff(ts) > 140)
    starts = ts[gaps] if len(gaps) else np.empty(0)
    ends = ts[gaps+1] if len(gaps) else np.empty(0)
    person = canonical_person(item["source"], df["User"].iloc[0] if "User" in df else "")
    duration = float((ts[-1] - ts[0]) / 1000)
    collected, waves = [], []
    for idx, (peak, k) in enumerate(zip(peaks, kind)):
        left, right = peak + time_grid[0], peak + time_grid[-1]
        if left < ts[0] or right > ts[-1] or np.any((starts < right) & (ends > left)):
            continue
        quiet = (ts >= peak - 320.) & (ts <= peak - 200.)
        if int(quiet.sum()) < 3:
            continue
        base = np.median(smoothed[quiet], axis=0)
        curve = np.column_stack([np.interp(peak + time_grid, ts, smoothed[:, j]-base[j]) for j in range(15)]).astype(np.float32)
        # Difference of log-coded zone values, not addition of linear PSD powers.
        if not np.isfinite(curve).all():
            continue
        waves.append(np.clip(curve, -32000, 32000).reshape(len(time_grid), 3, 5).astype(np.float16))
        collected.append({"file": item["source"], "file_id": hashlib.sha256(item["source"].encode()).hexdigest()[:20],
                          "category": item["category"], "subtype": subtype(item["source"], item["category"]),
                          "person": person, "date": session_date(item["source"]), "peak_ms": peak-ts[0],
                          "candidate_source": int(k), "duration_s": duration,
                          "sample_dt_ms": median_dt, "peak_strength": float(np.max(np.abs(curve))),
                          "sha256_uncompressed": item["sha256_uncompressed"]})
    qc["kept_events"] = len(collected)
    qc["duration_s"] = duration
    return collected, np.stack(waves) if waves else np.empty((0, len(time_grid), 3, 5), np.float16), qc


def extract(args) -> dict:
    root = Path(args.input_dir).resolve()
    out = Path(args.out_dir).resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Output must be disjoint from the original DATA_TAP folder")
    out.mkdir(parents=True, exist_ok=True)
    chunks = out / "chunks"
    chunks.mkdir(exist_ok=True)
    manifest_file = out / "source_manifest.json"
    settings = {"version": VERSION, "sample_ms": args.sample_ms, "chunk_files": args.chunk_files,
                "max_files_per_category": args.max_files, "min_candidate_gap_ms": 40.0}
    # Rescan hashes on resume as a source integrity check; don't silently use stale results.
    items, info = collect_source(root, args.max_files)
    contract = {"settings": settings, "sources": info, "items": items}
    if manifest_file.exists():
        prior = json.loads(manifest_file.read_text())
        if prior != contract:
            raise ValueError("Source set or configuration changed. Use a NEW output dir, not --resume")
    else:
        atom_json(manifest_file, contract)
    time_grid = np.arange(-320., 641., args.sample_ms, dtype=np.float64)
    count = len(items)
    errors = []
    for part, start in enumerate(range(0, count, args.chunk_files)):
        target = chunks / f"part_{part:05d}.npz"
        meta_path = chunks / f"part_{part:05d}.csv.gz"
        qc_path = chunks / f"part_{part:05d}.qc.json"
        if target.exists() and meta_path.exists() and qc_path.exists():
            qc = json.loads(qc_path.read_text())
            if qc["source_fingerprint_sha256"] != info["source_fingerprint_sha256"]:
                raise ValueError("Fingerprint mismatch in cached chunk")
            print("V9_EXTRACT_REUSE", part, flush=True)
            continue
        if target.exists() or meta_path.exists() or qc_path.exists():
            raise ValueError("Incomplete cached chunk: remove ONLY its part files after inspection: " + str(part))
        entries, wave_list, audits = [], [], []
        for item in items[start:start + args.chunk_files]:
            path = root / item["source"]
            if sha_uncompressed(path) != item["sha256_uncompressed"]:
                raise ValueError("Source changed since index: " + item["source"])
            try:
                df = _strict_df(path)
                rec, wave, qc = extract_one(df, item, time_grid)
                entries.extend(rec)
                wave_list.extend(wave)
                audits.append({"source": item["source"], **qc})
            except (ValueError, KeyError, OverflowError) as e:
                errors.append({"source": item["source"], "error": str(e)})
                audits.append({"source": item["source"], "error": str(e)})
                print("V9_EXTRACT_WARNING", item["source"], str(e)[:150], flush=True)
        arr = np.stack(wave_list).astype(np.float16) if wave_list else np.empty((0, len(time_grid), 3, 5), np.float16)
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as f:
            np.savez_compressed(f, wave=arr)
        os.replace(temporary, target)
        atom_csv(meta_path, pd.DataFrame(entries, columns=["file", "file_id", "category", "subtype", "person", "date", "peak_ms", "candidate_source", "duration_s", "sample_dt_ms", "peak_strength", "sha256_uncompressed"]))
        atom_json(qc_path, {"source_fingerprint_sha256": info["source_fingerprint_sha256"],
                            "records": len(audits), "candidates": len(entries), "audits": audits})
        print("V9_EXTRACT_PART", part, "of", math.ceil(count/args.chunk_files), "files", len(audits),
              "candidates", len(entries), flush=True)
    all_errors = []
    for qc_file in sorted(chunks.glob("part_*.qc.json")):
        qc_data = json.loads(qc_file.read_text())
        all_errors.extend({"source": row["source"], "error": row["error"]}
                          for row in qc_data["audits"] if "error" in row)
    summary = {**info, "config": settings, "total_chunks": math.ceil(count / args.chunk_files),
               "extraction_errors": all_errors,
               "scope": "Unverified EMG event hypotheses, NO physical tap onset labels",
               "complete": not bool(all_errors)}
    atom_json(out / "extraction_summary.json", summary)
    if all_errors:
        raise RuntimeError(f"Source extraction incomplete: {len(all_errors)} invalid files; inspect extraction_summary.json")
    return summary


def load_cache(out: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    manifest = json.loads((out / "source_manifest.json").read_text())
    summary = json.loads((out / "extraction_summary.json").read_text())
    if not summary.get("complete"):
        raise ValueError("Extraction had source errors; review extraction_summary.json")
    paths = sorted((out / "chunks").glob("part_*.npz"))
    expected = math.ceil(len(manifest["items"]) / manifest["settings"]["chunk_files"])
    if len(paths) != expected or not (out / "extraction_summary.json").exists():
        raise ValueError(f"Extraction incomplete: found {len(paths)}/{expected} chunks")
    metadata = []
    matrix = []
    for p in paths:
        meta = pd.read_csv(p.with_suffix(".csv.gz"), keep_default_na=False)
        x = np.load(p)["wave"]
        if len(meta) != len(x):
            raise ValueError("Meta/matrix mismatch: " + str(p))
        metadata.append(meta)
        matrix.append(x)
    tab = pd.concat(metadata, ignore_index=True)
    waves = np.concatenate(matrix, axis=0)
    if not len(tab):
        raise ValueError("No eligible EMG candidates; check source and extraction warnings")
    if not set(tab.category).issubset(CATEGORIES):
        raise ValueError("Unexpected categories")
    return tab, waves, manifest


def config_grid(preset: str, tasks: tuple[str, ...]):
    if preset == "smoke":
        pre, post, representations, modes, orders, kinds, weights = [120], [160], ["moments"], ["relative"], ["ranked"], ["mil3"], [1.0]
    elif preset == "standard":
        pre, post = [80, 160, 320], [80, 160, 320, 640]
        representations = ["moments", "temporal", "spectrum", "wave"]
        modes, orders, kinds, weights = ["relative"], ["ranked", "native"], ["mil1", "mil3", "all"] , [1.0, 3.0]
    else:
        pre, post = [60, 100, 160, 240, 320], [60, 100, 160, 240, 360, 480, 640]
        representations = ["amplitude", "moments", "temporal", "spectrum", "wave", "hybrid"]
        modes, orders, kinds, weights = ["relative", "absolute"], ["ranked", "native"], ["mil1", "mil3", "mil8", "all"], [1.0, 3.0]
    for task in tasks:
        for p in pre:
            for q in post:
                for repr_name in representations:
                    for mode in modes:
                        for order in orders:
                            for kind in kinds:
                                for weight in weights:
                                    config = {"task": task, "pre_ms": p, "post_ms": q,
                                              "representation": repr_name, "mode": mode,
                                              "order": order, "positive_selection": kind,
                                              "finger_weight": weight}
                                    config["trial_id"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
                                    yield config


def waveform_features(wave: np.ndarray, cfg: dict, sample_ms: int) -> np.ndarray:
    """Extract multiscale representations from one frozen event-candidate population.

    NOTE: event-normalization and ranked-electrode ordering use post-event context;
    these are OFFLINE teacher representations, not firmware-causal features.
    """
    full_grid = np.arange(-320, 641, sample_ms)
    idx = np.flatnonzero((full_grid >= -cfg["pre_ms"]) & (full_grid <= cfg["post_ms"]))
    x = wave[:, idx, :, :].astype(np.float32)
    pos = np.maximum(x, 0)
    if cfg["order"] == "ranked":
        # All channel ranking is computed within this trial's available window.
        strength = pos.mean(axis=(1, 3))
        ch_order = np.argsort(-strength, axis=1)
        x = np.take_along_axis(x, ch_order[:, None, :, None], axis=2)
        pos = np.maximum(x, 0)
    peak_scale = np.maximum(10.0, np.quantile(pos.reshape(len(x), -1), .95, axis=1))
    if cfg["mode"] == "relative":
        z = np.clip(x / peak_scale[:, None, None, None], -3.0, 5.0)
    else:
        z = np.clip(x / 1500.0, -5.0, 8.0)
    pz = np.maximum(z, 0.0)
    # Shape geometry, frequency-zone change and cross-channel response.
    energy = pz.mean(axis=(2,3))
    ch = pz.mean(axis=3)
    zon = pz.mean(axis=2)
    early = pz[:, :max(1, len(idx)//3)]
    late = pz[:, max(0, len(idx)*2//3):]
    spectrum = np.concatenate([pz.mean(axis=1).reshape(len(z), -1),
                               early.mean(axis=1).reshape(len(z), -1),
                               late.mean(axis=1).reshape(len(z), -1),
                               np.max(pz, axis=1).reshape(len(z), -1)], axis=1)
    # 60 low-dimensional summary descriptors; avoids flatten-only overfitting.
    moments = np.concatenate([np.mean(pz, axis=1).reshape(len(z), -1),
                              np.std(z, axis=1).reshape(len(z), -1),
                              np.max(pz, axis=1).reshape(len(z), -1),
                              np.mean(np.abs(np.diff(z, axis=1)), axis=1).reshape(len(z), -1),
                              np.quantile(pz.reshape(len(z), len(idx), -1), [.25, .5, .9], axis=1).transpose(1,0,2).reshape(len(z), -1)], axis=1)
    amplitude = np.column_stack([np.log1p(peak_scale),
                                np.max(pz, axis=(1,2,3)),
                                np.mean(pz, axis=(1,2,3)),
                                np.std(z, axis=(1,2,3))])
    temporal = np.concatenate([energy, ch.reshape(len(z), -1), zon.reshape(len(z), -1)], axis=1)
    variants = {"amplitude": amplitude, "moments": moments, "temporal": temporal,
                "spectrum": spectrum, "wave": z.reshape(len(z), -1),
                "hybrid": np.concatenate([moments, temporal, amplitude], axis=1)}
    return variants[cfg["representation"]].astype(np.float32)


def _event_sampling_weights(file_id: np.ndarray, y: np.ndarray, cat: np.ndarray, finger_weight: float,
                            selected: np.ndarray | None = None) -> np.ndarray:
    counts = Counter(file_id)
    weights = np.array([1. / counts[f] for f in file_id], np.float64)
    if selected is not None:
        weights[y == 1] = 0.0
        for f in set(file_id[y == 1]):
            idx = np.flatnonzero((file_id == f) & (y == 1))
            subset = idx[selected[idx]]
            if len(subset) > 0:
                weights[subset] = 1 / len(subset)
    # Prevent sample-rich recording, or one category, from dominating.
    for c in set(cat):
        is_c = cat == c
        total = weights[is_c].sum()
        if total > 0:
            weights[is_c] /= total
    if finger_weight != 1.:
        weights[cat == "FINGER"] *= finger_weight
    return (weights / max(1e-9, weights[weights > 0].mean())).astype(np.float32)


def model_new(seed: int, trees: int):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(objective="binary", n_estimators=trees, learning_rate=0.045,
                          num_leaves=11, max_depth=5, min_child_samples=35,
                          reg_lambda=22., colsample_bytree=.85, subsample=.9,
                          verbosity=-1, n_jobs=WORKER_THREADS, random_state=seed,
                          deterministic=True, force_col_wise=True)


def _train_event_model(X, ids, cat, y, cfg, fold, trees):
    # Weak bag supervision: positive bags contain at least one gesture, NOT all events.
    # Initial all-positive assignment is a provisional ranking seed ONLY, never exported.
    weights = _event_sampling_weights(ids, y, cat, cfg["finger_weight"])
    initial = model_new(SEED + fold, trees)
    initial.fit(X, y, sample_weight=weights)
    if cfg["positive_selection"] == "all":
        return initial
    scores = initial.predict_proba(X)[:,1]
    k = int(cfg["positive_selection"].removeprefix("mil"))
    picked = np.zeros(len(y), dtype=bool)
    for f in set(ids[y == 1]):
        rows = np.flatnonzero((ids == f) & (y == 1))
        if len(rows):
            picked[rows[np.argsort(scores[rows])[-k:]]] = True
    # Second iteration is selective weak multiple-instance pseudo-witnessing.
    # It is still NOT gold per-event truth and may select a confounder.
    weighted = _event_sampling_weights(ids, y, cat, cfg["finger_weight"], selected=picked)
    refit = model_new(SEED + fold + 17000, trees)
    refit.fit(X, y, sample_weight=weighted)
    return refit


def evaluate_bags(meta: pd.DataFrame, proba: np.ndarray, task: str) -> dict:
    df = meta[["file", "category", "subtype", "person", "date", "duration_s"]].copy()
    df["score"] = proba
    bags = df.groupby("file", sort=False).agg(category=("category", "first"), subtype=("subtype", "first"),
                                                person=("person", "first"), date=("date", "first"),
                                                duration_s=("duration_s", "first"), max_score=("score", "max"),
                                                top3_score=("score", lambda v: float(np.mean(np.sort(v)[-3:]))))
    def auc_score(mask, pos):
        sub = bags.loc[mask]
        truth = pos.loc[mask].astype(int)
        if truth.nunique() < 2:
            return None
        return float(roc_auc_score(truth, sub.max_score))
    def ap_score(mask, pos):
        truth = pos.loc[mask].astype(int)
        return float(average_precision_score(truth, bags.loc[mask].max_score)) if truth.nunique() > 1 else None
    pos = bags.category.eq("TAP") if task == "tap" else bags.category.isin(("TAP", "SWIPE"))
    finger = bags.category.eq("FINGER")
    hard = bags.category.isin(NEGATIVE)
    res = {"bag_count": len(bags), "positive_bag_count": int(pos.sum()),
           "primary_tap_finger_auc": auc_score(bags.category.eq("TAP") | finger, bags.category.eq("TAP")),
           "primary_tap_finger_ap": ap_score(bags.category.eq("TAP") | finger, bags.category.eq("TAP")),
           "target_all_hardneg_auc": auc_score(pos | hard, pos),
           "target_all_hardneg_ap": ap_score(pos | hard, pos),
           "target_all_hardneg_top3_auc": None}
    if (pos | hard).sum() and pos.sum() and hard.sum():
        mask = pos | hard
        res["target_all_hardneg_top3_auc"] = float(roc_auc_score(pos.loc[mask].astype(int), bags.loc[mask].top3_score))
    for name in ("SOFTTAP", "MIDTAP", "STRONGTAP", "DOUBLETAP", "TAPONMOTION"):
        mask = bags.subtype.eq(name) | finger
        res[name.lower()+"_finger_auc"] = auc_score(mask, bags.subtype.eq(name))
    # These are rank diagnostics, never interpreted as event-level accuracy.
    res["max_recording_score_finger_p95_exploratory"] = float(bags.loc[finger].max_score.quantile(.95)) if finger.any() else None
    return res


def run_trial(X, tab, cfg: dict, splits, trees: int) -> tuple[dict, np.ndarray]:
    category = tab.category.to_numpy()
    is_positive = np.isin(category, ["TAP"] if cfg["task"] == "tap" else ["TAP", "SWIPE"])
    is_negative = np.isin(category, list(NEGATIVE))
    eligible = is_positive | is_negative
    if not np.any(is_positive) or not np.any(is_negative):
        raise ValueError("Training needs both positive and negative bags")
    file_id = tab.file_id.to_numpy()
    scores = np.full(len(tab), np.nan, np.float32)
    folds = []
    for fold, (tr, te) in enumerate(splits):
        tid = tr[eligible[tr]]
        if not np.any(is_positive[tid]) or not np.any(is_negative[tid]):
            raise ValueError(f"Missing classes in training fold {fold}")
        model = _train_event_model(X[tid], file_id[tid], category[tid], is_positive[tid].astype(np.int8),
                                   cfg, fold, trees)
        scores[te] = model.predict_proba(X[te])[:,1]
        folds.append({"fold": fold, "train_events": len(tid), "test_events": len(te),
                      "test_people": len(set(tab.person.iloc[te]))})
    if not np.isfinite(scores).all():
        raise AssertionError("Not every held-out event scored")
    metrics = evaluate_bags(tab, scores, cfg["task"])
    metrics["folds"] = folds
    return metrics, scores


def pattern_report(tab: pd.DataFrame, wave: np.ndarray, best_scores: np.ndarray,
                   sample_ms: int, out: Path):
    """Aggregate representative candidate shape in each folder; no physical-onset claim."""
    df = tab[["file", "category", "subtype", "peak_ms", "person", "date"]].copy()
    df["score"] = best_scores
    groups = defaultdict(list)
    for path, sub in df.groupby("file", sort=False):
        best = sub.score.idxmax()
        groups[tab.category.iloc[best]].append(best)
    rows = []
    trace = np.arange(-320, 641, sample_ms)
    for category, candidates in sorted(groups.items()):
        w = wave[candidates].astype(np.float32)
        # Normalize each candidate separately; preserve 3x5 spectral layout.
        den = np.maximum(10.0, np.quantile(np.maximum(w, 0).reshape(len(w),-1), .95, axis=1))
        normalized = np.clip(w / den[:, None, None, None], -3, 5)
        mean = np.mean(normalized, axis=0)
        for i, t in enumerate(trace):
            for ch in range(3):
                for zone in range(5):
                    rows.append({"category": category, "relative_time_ms": int(t),
                                 "channel": ch+1, "psd_zone": zone,
                                 "mean_normalized_delta": float(mean[i,ch,zone]),
                                 "recordings": len(candidates)})
    pd.DataFrame(rows).to_csv(out / "pattern_templates_by_category.csv", index=False)
    # Pairs are physiological candidate gaps, NOT verified double taps.
    intervals = []
    for f, sub in df.groupby("file", sort=False):
        gaps = np.diff(np.sort(sub.peak_ms.to_numpy(float)))
        for g in gaps:
            if 40 <= g <= 700:
                intervals.append({"category": sub.category.iloc[0], "subtype": sub.subtype.iloc[0], "candidate_gap_ms": float(g)})
    pd.DataFrame(intervals, columns=["category", "subtype", "candidate_gap_ms"]).to_csv(out / "candidate_intervals_not_ground_truth.csv", index=False)


def evaluate_date_shift(tab, proba) -> dict:
    """A descriptive cross-date sensitivity diagnostic, not a separate heldout CV."""
    df = tab[["file", "person", "date", "category"]].copy()
    df["p"] = proba
    bag = df.groupby("file").agg(person=("person", "first"), date=("date", "first"),
                                  category=("category", "first"), p=("p", "max"))
    out = []
    for person, records in bag.groupby("person"):
        for date, held in records.groupby("date"):
            if date == "unknown" or len(held) < 5:
                continue
            test = held[held.category.isin(("TAP", "FINGER"))]
            if test.category.nunique() < 2:
                continue
            out.append({"participant_hash": hashlib.sha256(person.encode()).hexdigest()[:14],
                        "date": date, "files": len(test),
                        "tap_finger_auc": float(roc_auc_score(test.category.eq("TAP"), test.p))})
    return {"per_date_auc": out, "minimum_date_auc": min((r["tap_finger_auc"] for r in out), default=None)}


def motif_cluster_scan(tab: pd.DataFrame, wave: np.ndarray, sample_ms: int,
                       folds, out: Path, max_events_per_category: int = 10000):
    """Cross-person OOF, positive-bag enrichment of UNSUPERVISED motifs.

    No V5 positives, no pointer alignment, and no forced per-event truth.
    Cluster enrichment is weighted per RECORDING rather than per candidate count.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cat = tab.category.to_numpy()
    fid = tab.file_id.to_numpy()
    results = []
    cfgs = [
        (80, 120, "spectrum"), (120, 240, "moments"),
        (160, 320, "hybrid"), (240, 480, "temporal"),
        (320, 640, "moments"),
    ]
    for before, after, rep in cfgs:
        base = {"pre_ms": before, "post_ms": after, "representation": rep,
                "mode": "relative", "order": "ranked"}
        X = waveform_features(wave, base, sample_ms)
        for clusters in (8, 16, 32, 64):
            oof = np.full(len(tab), np.nan, np.float32)
            for fold, (tr, te) in enumerate(folds):
                # Stratified event CAP for unbiased label-blind, unsupervised training.
                rng = np.random.default_rng(SEED + fold + clusters)
                selected = []
                for name in sorted(set(cat[tr])):
                    inds = tr[cat[tr] == name]
                    if len(inds) > max_events_per_category:
                        inds = rng.choice(inds, size=max_events_per_category, replace=False)
                    selected.extend(inds)
                inds = np.asarray(selected, dtype=int)
                scaler = StandardScaler().fit(X[inds]); W = np.clip(scaler.transform(X), -7, 7)
                n_comp = min(24, W.shape[1], len(inds) - 1)
                if n_comp < 2 or len(inds) < clusters:
                    raise ValueError("Not enough events to fit motif clusters")
                # PCA and clustering fit TRAIN ONLY; test fold is never seen during fitting.
                pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=SEED)
                pca.fit(W[inds]); Z = pca.transform(W).astype(np.float32)
                km = MiniBatchKMeans(n_clusters=clusters, batch_size=min(2048, len(inds)),
                                      max_iter=90, n_init=2, random_state=SEED + fold)
                km.fit(Z[inds]); labels = km.predict(Z)
                # Average per-record motif frequency, not event frequency.
                idx_train = tr[np.isin(cat[tr], ["TAP", "FINGER"])]
                class_hists = {"TAP": [], "FINGER": []}
                for file in np.unique(fid[idx_train]):
                    ix = idx_train[fid[idx_train] == file]
                    hist = np.bincount(labels[ix], minlength=clusters).astype(float)
                    if hist.sum():
                        class_hists[str(cat[ix[0]])].append(hist / hist.sum())
                if not class_hists["TAP"] or not class_hists["FINGER"]:
                    raise ValueError("Not enough TAP and FINGER bags in motif training fold")
                p = np.mean(class_hists["TAP"], axis=0)
                n = np.mean(class_hists["FINGER"], axis=0)
                enrichment = np.log((p + 0.005)/(n + 0.005))
                # Monotone transform to 0..1 so standard recording metrics apply.
                oof[te] = (1 / (1 + np.exp(-enrichment[labels[te]]))).astype(np.float32)
            if not np.isfinite(oof).all():
                raise AssertionError("Missing motif heldout predictions")
            m = evaluate_bags(tab, oof, "tap")
            record = {"pre_ms":before, "post_ms":after, "representation":rep,
                      "clusters":clusters, "tap_finger_recording_auc": m["primary_tap_finger_auc"],
                      "tap_allneg_recording_auc": m["target_all_hardneg_auc"],
                      "note":"Unsupervised waveform clusters; recording-bag enrichment; no event truth"}
            results.append(record)
            print("V9_MOTIF", before, after, rep, clusters,
                  "tap_finger_auc", round(m["primary_tap_finger_auc"] or 0.,4),flush=True)
            atom_json(out / "motif_summary.json", {"experiments": results,
                "limitations": "OOF recording-bag scores only, not verified per-event taps"})
    pd.DataFrame(results).to_csv(out / "motif_cluster_scan.csv", index=False)
    return results


def channel_dropout_stress(tab: pd.DataFrame, wave: np.ndarray, cfg: dict,
                           sample_ms: int, folds, trees: int,
                           out: Path):
    """Synthetic zeroed-channel stress, not actual poor-contact recordings."""
    category = tab.category.to_numpy()
    positive = np.isin(category, ["TAP"] if cfg["task"] == "tap" else ["TAP", "SWIPE"])
    negative = np.isin(category, list(NEGATIVE))
    eligible = positive | negative
    fid = tab.file_id.to_numpy()
    X = waveform_features(wave, cfg, sample_ms)
    masks = {"none": (), "CH1_zero": (0,), "CH2_zero": (1,), "CH3_zero": (2,),
             "CH1_CH2_zero": (0,1), "CH1_CH3_zero": (0,2), "CH2_CH3_zero": (1,2)}
    perturbed = {}
    for k, mask in masks.items():
        if not mask:
            perturbed[k] = X
        else:
            damaged = wave.copy()
            damaged[:,:,list(mask),:] = 0
            perturbed[k] = waveform_features(damaged, cfg, sample_ms)
    probs = {k: np.full(len(tab),np.nan,np.float32) for k in masks}
    for fold, (tr, te) in enumerate(folds):
        tid = tr[eligible[tr]]
        model = _train_event_model(X[tid], fid[tid], category[tid], positive[tid].astype(np.int8),
                                   cfg, fold, trees)
        for k in masks:
            probs[k][te] = model.predict_proba(perturbed[k][te])[:,1]
    summary = {}
    for k in masks:
        m = evaluate_bags(tab, probs[k], cfg["task"])
        summary[k] = {"tap_finger_recording_auc":m["primary_tap_finger_auc"],
                      "target_allneg_recording_auc":m["target_all_hardneg_auc"]}
    atom_json(out / "SIMULATED_CHANNEL_DROPOUT.json", {"best_configuration": cfg,
        "ablation": summary,
        "limitations": "Numerically setting PSD electrode channel deltas to zero is not a substitute for real partial-contact data."})
    return summary


def make_report(search_dir: Path, manifest: dict, winner: dict | None):
    trials_path = search_dir / "trials.jsonl"
    trials = [json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()] if trials_path.exists() else []
    successes = [r for r in trials if r.get("status") == "OK"]
    def rank(row):
        m = row["metrics"]
        return (float(m["primary_tap_finger_auc"] or 0), float(m["target_all_hardneg_auc"] or 0))
    sorted_success = sorted(successes, key=rank, reverse=True)
    cols = ("trial_id", "task", "pre_ms", "post_ms", "representation", "mode", "order", "positive_selection", "finger_weight")
    leaderboard = []
    for r in sorted_success:
        leaderboard.append({**{c:r["config"][c] for c in cols},
                            **{k:v for k,v in r["metrics"].items() if k != "folds"},
                            "duration_s": r["duration_s"]})
    pd.DataFrame(leaderboard).to_csv(search_dir / "leaderboard.csv", index=False)
    counts = Counter(r.get("status") for r in trials)
    summary = {"version": VERSION, "source": manifest["sources"],
               "completed_trials": len(trials), "passed_trials": counts.get("OK", 0),
               "failed_trials": counts.get("ERROR", 0),
               "best_trial": sorted_success[0]["config"] if sorted_success else None,
               "best_metrics": sorted_success[0]["metrics"] if sorted_success else None,
               "scope": "Recording-level evidence from weak labels only; event onset/FP and double-pair accuracy unknown",
               "research_restrictions": ["No V5 pseudo positives used as truth", "No pointer, metronome, filenames or BPM in feature matrix",
                 "Hyperparameter search on participant OOF results is exploratory; final heldout independent dataset required",
                 "No automatic production-positive or ground-truth labels emitted", "Potential wearer aliases and session confounding"]}
    atom_json(search_dir / "SUMMARY_FOR_CHATGPT.json", summary)
    lines = ["# V9 multiscale TAP research — automated aggregate summary", "",
             "**Research-only. Never interpret bag AUC as real event precision, recall, double-tap accuracy or firmware false positives.**", "",
             f"Sources: {manifest['sources']['unique_recordings']} unique recordings, {manifest['sources']['duplicates_excluded']} duplicates removed. ",
             f"Trials: {len(trials)} recorded, {counts.get('OK', 0)} successful, {counts.get('ERROR', 0)} failed.", "",
             "## Top models (recording-level, participant OOF)", "",
             "| Trial | Task | Window (before / after ms) | Representation | Selection | FINGER weight | TAP vs FINGER AUC | Target vs negatives AUC |",
             "|---|---|---|---|---|---:|---:|---:|"]
    for r in sorted_success[:25]:
        c, m = r["config"], r["metrics"]
        f = lambda x: f"{x:.3f}" if x is not None else "n/a"
        lines.append(f"| {c['trial_id']} | {c['task']} | {c['pre_ms']}/{c['post_ms']} | {c['representation']}/{c['mode']}/{c['order']} | {c['positive_selection']} | {c['finger_weight']} | {f(m['primary_tap_finger_auc'])} | {f(m['target_all_hardneg_auc'])} |")
    lines += ["", "## Important caveats", "", "No synchronized physical tap onset reference exists in these folders. SoftTap and DoubleTaps are recording instructions, NOT true event timestamps.",
              "TAP vs FINGER bag AUC is a **diagnostic for weak supervised pattern separation**, not a deployability score.",
              "Selection of best trial from this leaderboard is optimistic without an independent final heldout set.",
              "Event candidate population is V3 plus a weak-rise complement; real taps absent from both are unmeasurable.",
              "No signal-quality gold annotations for independently disconnected/partially connected electrodes were supplied.", "",
              "## Files for follow-up", "", "- `SUMMARY_FOR_CHATGPT.json` small aggregate findings", "- `leaderboard.csv` all successful configurations",
              "- `trials.jsonl` all successes and failures with fold details", "- `pattern_templates_by_category.csv` candidate shape averages (if a winning trial is revisited)",
              "- `candidate_intervals_not_ground_truth.csv` double-gap hypotheses (if winner revisited)",
              "- `out_of_fold_predictions.csv.gz` local detailed event predictions, **do not publish publicly**", ""]
    (search_dir / "REPORT_FOR_CHATGPT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def search(args) -> dict:
    out = Path(args.out_dir).resolve()
    tab, wave, manifest = load_cache(out)
    search_dir = out / "results_v9"
    search_dir.mkdir(parents=True, exist_ok=True)
    study_path = search_dir / "study_contract.json"
    tasks = tuple(x.strip() for x in args.tasks.split(",") if x.strip())
    if not tasks or not set(tasks).issubset({"tap", "combined"}):
        raise ValueError("Tasks must be tap,combined")
    settings = {"version": VERSION, "source_fingerprint_sha256": manifest["sources"]["source_fingerprint_sha256"],
                "preset": args.preset, "tasks": list(tasks), "folds": args.folds, "trees": args.trees,
                "sample_ms": manifest["settings"]["sample_ms"], "seed": SEED,
                "feature_contract": "max-window 320ms pre/640ms post, event-baseline, sorted/native channels"}
    if study_path.exists():
        if json.loads(study_path.read_text()) != settings:
            raise ValueError("Study config changed: use a NEW output dir")
    else:
        atom_json(study_path, settings)
    group = tab.person.to_numpy()
    people = sorted(set(group))
    if len(people) < args.folds:
        raise ValueError(f"Need at least {args.folds} people, got {len(people)}")
    folds = list(GroupKFold(n_splits=args.folds).split(np.arange(len(tab)), groups=group))
    for tr, te in folds:
        if set(group[tr]) & set(group[te]):
            raise AssertionError("Participant leakage")
    trials_path = search_dir / "trials.jsonl"
    existing = [json.loads(v) for v in trials_path.read_text().splitlines() if v.strip()] if trials_path.exists() else []
    completed = {t["config"]["trial_id"] for t in existing}
    configs = list(config_grid(args.preset, tasks))
    if args.max_trials:
        configs = configs[:args.max_trials]
    print("V9_SEARCH_START", "candidates", len(tab), "subjects", len(people), "planned", len(configs), "done", len(completed), flush=True)
    cache_key = None
    features = None
    start = time.time()
    budget_exhausted = False
    for number, cfg in enumerate(configs, 1):
        if cfg["trial_id"] in completed:
            continue
        t0 = time.time()
        try:
            key = (cfg["pre_ms"], cfg["post_ms"], cfg["representation"], cfg["mode"], cfg["order"])
            if key != cache_key:
                features = waveform_features(wave, cfg, settings["sample_ms"])
                cache_key = key
            metrics, scores = run_trial(features, tab, cfg, folds, args.trees)
            trial = {"status": "OK", "config": cfg, "metrics": metrics, "duration_s": round(time.time()-t0, 2)}
        except Exception as e:
            trial = {"status": "ERROR", "config": cfg, "error": str(e), "traceback": traceback.format_exc(limit=5),
                     "duration_s": round(time.time()-t0, 2)}
            print("V9_TRIAL_ERROR", cfg["trial_id"], str(e), flush=True)
        with trials_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trial, sort_keys=True, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        completed.add(cfg["trial_id"])
        if trial["status"] == "OK":
            print("V9_TRIAL", number, "of", len(configs), cfg["trial_id"],
                  "tap_finger_auc", round(trial["metrics"]["primary_tap_finger_auc"] or 0, 4),
                  "seconds", trial["duration_s"], flush=True)
        if number % 10 == 0:
            make_report(search_dir, manifest, None)
        # If a user specifies a time limit, stop only after a completed trial;
        # default 0 means exhaust grid without a time-based early stop.
        if args.hours_limit > 0 and (time.time()-start) >= args.hours_limit*3600:
            print("V9_TIME_BUDGET_REACHED_CAN_RESUME", flush=True)
            budget_exhausted = True
            break
    summary = make_report(search_dir, manifest, None)
    if args.motif_scan and summary["passed_trials"] and not budget_exhausted:
        # Scans are cached as a separate stage, and repeated invocation is safe.
        if not (search_dir / "motif_summary.json").exists() or not (search_dir / "motif_cluster_scan.csv").exists():
            motif_cluster_scan(tab, wave, settings["sample_ms"], folds, search_dir)
    # Refit/score the chosen best config with exactly the same OUTER splits;
    # save local detailed hypotheses only. Aggregate templates are not true TAPs.
    if summary["best_trial"] and args.winner_analysis and not budget_exhausted:
        cfg = summary["best_trial"]
        X = waveform_features(wave, cfg, settings["sample_ms"])
        metrics, scores = run_trial(X, tab, cfg, folds, args.trees)
        predictions = tab[["file", "file_id", "category", "subtype", "person", "date", "peak_ms", "duration_s"]].copy()
        predictions["oof_weak_score"] = scores
        atom_csv(search_dir / "out_of_fold_predictions.csv.gz", predictions)
        pattern_report(tab, wave, scores, settings["sample_ms"], search_dir)
        atom_json(search_dir / "DATE_SLICES_DESCRIPTIVE.json", evaluate_date_shift(tab, scores))
        if args.dropout_stress:
            channel_dropout_stress(tab, wave, cfg, settings["sample_ms"], folds,
                                   args.trees, search_dir)
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("all", "extract", "search"))
    ap.add_argument("--input-dir", type=Path, default=Path("DATA/DATA_TAP"))
    ap.add_argument("--out-dir", type=Path, default=Path("research/tap_real/results_v9"))
    ap.add_argument("--sample-ms", type=int, choices=(10, 20), default=20)
    ap.add_argument("--chunk-files", type=int, default=20)
    ap.add_argument("--max-files", type=int, default=0, help="Stratified max files per category, 0=all")
    ap.add_argument("--preset", choices=("smoke", "standard", "deep"), default="deep")
    ap.add_argument("--tasks", default="tap,combined")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=160)
    ap.add_argument("--max-trials", type=int, default=0, help="0=full sweep; for smoke tests only")
    ap.add_argument("--hours-limit", type=float, default=0., help="0=no early stopping; resume by rerunning same command")
    ap.add_argument("--winner-analysis", action="store_true", help="Create local detailed OOF and average patterns for best trial")
    ap.add_argument("--motif-scan", action="store_true", help="Train-independent clustering/enrichment OOF experiments")
    ap.add_argument("--dropout-stress", action="store_true", help="7 simulated electrode dropout tests (requires --winner-analysis)")
    return ap


def main():
    args = parser().parse_args()
    if args.chunk_files < 1 or args.folds < 2 or args.trees < 5:
        raise ValueError("Invalid CPU/extraction/CV configuration")
    print("V9_CPU_BUDGET", "logical_threads", LOGICAL_THREADS,
          "worker_threads", WORKER_THREADS,
          "fraction", CPU_THREAD_FRACTION, flush=True)
    t = time.time()
    if args.command in ("extract", "all"):
        e = extract(args)
        print("V9_EXTRACT_DONE", json.dumps({k:v for k,v in e.items() if k not in ("config", "extraction_errors")}), flush=True)
    if args.command in ("search", "all"):
        r = search(args)
        print("V9_SEARCH_DONE", json.dumps({"passed": r["passed_trials"], "failed": r["failed_trials"],
                                             "best": r["best_trial"]}), flush=True)
    print("V9_TOTAL_SECONDS", round(time.time()-t, 1), flush=True)


if __name__ == "__main__":
    main()
