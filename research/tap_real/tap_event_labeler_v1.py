#!/usr/bin/env python3
"""Conservative, auditable TAP event pseudo-labeler for Mudra feature CSVs.

Reads ONLY DATA/DATA_TAP/**/*.csv.gz (or .csv) and writes sidecar labels,
per-recording event audits and a manifest to a separate output folder.
Does not train a classifier or assume Oculus PointerState is truth.

Important: v1 thresholds are *candidate defaults*, not empirically validated
on the entire DATA_TAP corpus. Review candidates against independently
annotated tap onsets before using these pseudo-labels to train firmware.

Usage:
  python tap_event_labeler_v1.py
  python tap_event_labeler_v1.py --max-files 20 --full-rows
  python tap_event_labeler_v1.py --input-dir DATA/DATA_TAP --out-dir DATA/DATA_TAP_LABELED_V1
"""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

CATEGORIES = {"TAP", "SWIPE", "FINGER", "WRIST", "ARM", "BAND", "NOTHING"}
EMG_HEAD = [
    [f"p_core->emg_psd[0][CH{c}][{z}]" for z in range(5)]
    for c in (1, 2, 3)
]
ACC_HEAD = [
    [f"p_core->acc_psd[0][{a}][{z}]" for z in range(6)]
    for a in "XYZ"
]
WL = [f"p_core->wl[CH{c}]" for c in (1, 2, 3)]


@dataclass
class Candidate:
    onset_ms: float
    peak_ms: float
    peak_row: int
    strength: float
    wl_strength: float
    acc_strength: float
    pointer_support: bool = False
    metronome_support: bool = False
    decision: str = "UNKNOWN"
    confidence: float = 0.0
    event_id: int = -1


def sha256_file(path: Path) -> str:
    dig = hashlib.sha256()
    # Hash the decompressed CSV bytes, not gzip headers. This catches duplicate
    # recordings even when an archive has a different embedded file name.
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rb") as fd:
        for block in iter(lambda: fd.read(1 << 20), b""):
            dig.update(block)
    return dig.hexdigest()


def rising_times(values: np.ndarray, ts: np.ndarray) -> np.ndarray:
    b = np.asarray(values > 0, dtype=bool)
    return ts[b & np.r_[True, ~b[:-1]]]


def emg_matrix(df: pd.DataFrame) -> np.ndarray:
    missing = [name for names in EMG_HEAD for name in names if name not in df]
    if missing:
        raise ValueError(f"Missing EMG firmware features: {missing[:4]}")
    ch = []
    for names in EMG_HEAD:
        arr = df[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ch.append(np.maximum(0.0, np.where(np.isfinite(arr), arr, 0)).sum(axis=1))
    return np.stack(ch, axis=1)


def optional_matrix(df: pd.DataFrame, groups: list[list[str]]) -> np.ndarray | None:
    if any(name not in df for group in groups for name in group):
        return None
    ch = []
    for cols in groups:
        arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ch.append(np.maximum(0.0, np.where(np.isfinite(arr), arr, 0)).sum(axis=1))
    return np.stack(ch, axis=1)


def normalized_onset_signal(x: np.ndarray, ts_ms: np.ndarray,
                            fast_tau_ms: float = 22.0,
                            slow_up_tau_ms: float = 900.0,
                            slow_down_tau_ms: float = 230.0) -> np.ndarray:
    """Fast/slow causal updates with an offline whole-file scale floor.

    The 20th-percentile floor uses future samples: this function is for offline
    candidate labeling ONLY, not a firmware-inference feature implementation.
    The initialization uses first-row signal only. Baseline rises slowly during
    activity and falls faster after it. Computation can be ported to firmware.
    """
    if len(x) < 2:
        return np.zeros_like(x)
    x = np.asarray(x, dtype=np.float64)
    fast, base = x[0].copy(), x[0].copy()
    out = np.zeros_like(x)
    # Scale floor avoids division explosions from a nearly disconnected channel.
    # Per-recording low percentile is used ONLY to stabilize the offline labeler.
    floor = np.maximum(10.0, np.nanquantile(x, .20, axis=0) * 0.25)
    for i in range(1, len(x)):
        dt = float(np.clip(ts_ms[i] - ts_ms[i - 1], 1.0, 100.0))
        af = 1.0 - math.exp(-dt / fast_tau_ms)
        fast += af * (x[i] - fast)
        tau = np.where(x[i] > base, slow_up_tau_ms, slow_down_tau_ms)
        ab = 1.0 - np.exp(-dt / tau)
        base += ab * (x[i] - base)
        out[i] = np.maximum(0.0, (fast - base) / np.maximum(base, floor))
    return out


def nearest_abs(t: float, pts: np.ndarray) -> float:
    if pts.size == 0:
        return float("inf")
    return float(np.min(np.abs(pts - t)))


def pointer_alignment(pointer_edges: np.ndarray,
                      candidates: list[Candidate]) -> dict[str, Any]:
    """Estimate a recording-specific pointer-minus-EMG delay with abstention.

    Uses strong physiological candidates only. Does not presume each pointer
    press is correct. Periodic aliasing is still possible and must be reviewed.
    """
    pts = np.array([c.peak_ms for c in candidates if c.strength >= .28])
    if len(pointer_edges) < 4 or len(pts) < 4:
        return {"trusted": False, "offset_ms": None, "matches": 0}
    guesses = []
    for p in pointer_edges:
        d = p - pts[np.argmin(np.abs(p - pts))]
        if abs(d) <= 400.0:
            guesses.append(float(d))
    if len(guesses) < 4:
        return {"trusted": False, "offset_ms": None, "matches": len(guesses)}
    offset = float(np.median(guesses))
    corrected = pointer_edges - offset
    residuals = np.array([nearest_abs(p, pts) for p in corrected])
    match_fraction = float(np.mean(residuals <= 125.0))
    matched = residuals[residuals <= 125.0]
    jitter = float(np.median(matched)) if len(matched) else float("inf")
    trusted = match_fraction >= .65 and jitter <= 65.0 and len(matched) >= 4
    return {"trusted": bool(trusted), "offset_ms": offset if trusted else None,
            "matches": int(len(matched)), "match_fraction": match_fraction,
            "median_residual_ms": jitter if math.isfinite(jitter) else None}


def candidates_from_signal(ts: np.ndarray, emg: np.ndarray,
                           wl: np.ndarray | None,
                           acc: np.ndarray | None) -> tuple[list[Candidate], np.ndarray, np.ndarray]:
    en = normalized_onset_signal(emg, ts)
    emg_novelty = np.max(en, axis=1)
    wl_novelty = np.max(normalized_onset_signal(wl, ts), axis=1) if wl is not None else np.zeros(len(ts))
    acc_novelty = np.max(normalized_onset_signal(acc, ts), axis=1) if acc is not None else np.zeros(len(ts))
    # WL is supportive but correlated with EMG. ACC never gates an event.
    score = emg_novelty + 0.10 * np.minimum(wl_novelty, 1.0)
    if len(ts) < 15:
        return [], score, acc_novelty
    peaks, _ = find_peaks(score, height=.12, prominence=.06, distance=2)
    # Intervals in milliseconds, not number of rows. Preserve close double-taps.
    kept: list[int] = []
    for i in peaks:
        if kept and ts[i] - ts[kept[-1]] < 75.0:
            if score[i] > score[kept[-1]]:
                kept[-1] = int(i)
        else:
            kept.append(int(i))
    cs = []
    for i in kept:
        low = max(.045, float(score[i]) * .28)
        k = i
        while k > 0 and ts[i] - ts[k - 1] <= 170 and score[k - 1] >= low:
            k -= 1
        cs.append(Candidate(onset_ms=float(ts[k]), peak_ms=float(ts[i]),
                            peak_row=i, strength=float(emg_novelty[i]),
                            wl_strength=float(wl_novelty[i]),
                            acc_strength=float(acc_novelty[i])))
    return cs, score, acc_novelty


def label_recording(df: pd.DataFrame, category: str) -> tuple[pd.DataFrame, list[Candidate], dict[str, Any]]:
    if "Timestamp" not in df:
        raise ValueError("Missing Timestamp")
    ts = pd.to_numeric(df["Timestamp"], errors="coerce").to_numpy(float)
    if len(ts) < 15 or not np.all(np.isfinite(ts)):
        raise ValueError("Too few rows or invalid timestamps")
    diffs = np.diff(ts)
    if np.any(diffs <= 0):
        raise ValueError("Non-monotonic or duplicate timestamps; review file")
    emg = emg_matrix(df)
    wl = optional_matrix(df, [[c] for c in WL])
    acc = optional_matrix(df, ACC_HEAD)
    cs, score, acc_novelty = candidates_from_signal(ts, emg, wl, acc)
    pointer = pd.to_numeric(df.get("PointerState", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0).to_numpy(float)
    metro = pd.to_numeric(df.get("Metronome", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0).to_numpy(float)
    ptr_edges = rising_times(pointer, ts)
    met_edges = rising_times(metro, ts)
    ptr_duty = float(np.mean(pointer > 0))
    # Stuck-high Oculus state is NEVER a valid tap reference.
    ptr_valid = .01 <= ptr_duty <= .9 and len(ptr_edges) >= 4
    align = pointer_alignment(ptr_edges, cs) if ptr_valid else {
        "trusted": False, "offset_ms": None, "matches": 0,
        "reason": "missing/stuck or too few pointer edges",
    }
    corrected_ptr = ptr_edges - align["offset_ms"] if align["trusted"] else np.array([])
    median_dt = float(np.median(diffs))
    # At 240 BPM, nearest-beat evidence is nearly guaranteed by chance; do not
    # award beat-based confidence unless a future protocol-specific fit passes.
    median_met_period = float(np.median(np.diff(met_edges))) if len(met_edges) > 2 else float("inf")
    met_useful = 380.0 <= median_met_period <= 1600.0
    for candidate in cs:
        candidate.pointer_support = bool(align["trusted"] and nearest_abs(candidate.peak_ms, corrected_ptr) <= 125)
        candidate.metronome_support = bool(met_useful and nearest_abs(candidate.peak_ms, met_edges) <= 190)
        if category != "TAP":
            candidate.decision = "HARD_NEGATIVE_CANDIDATE"
            candidate.confidence = 0.0
            continue
        s = candidate.strength
        if s >= .85:
            candidate.decision, candidate.confidence = "POSITIVE", .90
        elif s >= .38 and (candidate.pointer_support or candidate.metronome_support):
            candidate.decision, candidate.confidence = "POSITIVE", .78
        elif s >= .20 and candidate.pointer_support and candidate.wl_strength >= .10:
            candidate.decision, candidate.confidence = "POSITIVE", .70
        else:
            candidate.decision, candidate.confidence = "UNKNOWN", 0.0
    labels = np.full(len(df), -1, dtype=np.int8)
    weights = np.zeros(len(df), dtype=np.float32)
    event_id = np.full(len(df), -1, dtype=np.int32)
    if category != "TAP":
        # Deliberate non-tap activity is valuable as a hard negative. Keep
        # moderate weight around very tap-like ambiguous negative gestures.
        labels[:] = 0
        weights[:] = 1.0 if category == "NOTHING" else .85
        weights[score > .75] *= .7
    else:
        # Confidently quiet, sufficiently separated from ALL plausible events.
        quiet = score < .09
        labels[quiet] = 0
        weights[quiet] = .85
        for c in cs:
            near = (ts >= c.onset_ms - 130) & (ts <= c.peak_ms + 170)
            labels[near] = -1
            weights[near] = 0
        accepted = [c for c in cs if c.decision == "POSITIVE"]
        for number, c in enumerate(accepted, 1):
            c.event_id = number
            # Short positive target starting with physiologically inferred onset.
            # Only causal EMG features available at this row are training inputs.
            core = (ts >= c.onset_ms) & (ts <= min(c.onset_ms + 75.0, c.peak_ms + 55.0))
            labels[core] = 1
            weights[core] = np.float32(c.confidence)
            event_id[core] = number
    # Do not learn from timing gaps or warm-up/cool-down.
    invalid = np.zeros(len(ts), bool)
    invalid[:max(1, min(len(ts), int(math.ceil(150.0 / max(1., median_dt)))))] = True
    for i in np.flatnonzero(diffs > 150):
        invalid[(ts >= ts[i] - 20) & (ts <= ts[i + 1] + 100)] = True
    labels[invalid] = -1
    weights[invalid] = 0
    event_id[invalid] = -1
    output = pd.DataFrame({"Timestamp": df["Timestamp"], "tap_label": labels,
                           "tap_label_weight": weights, "tap_event_id": event_id})
    qc = {"rows": int(len(df)), "columns": int(len(df.columns)),
          "duration_s": round(float((ts[-1] - ts[0]) / 1000.), 3),
          "median_dt_ms": median_dt, "p95_dt_ms": float(np.quantile(diffs, .95)),
          "timestamp_gaps_gt_150ms": int(np.sum(diffs > 150)),
          "pointer_edges": len(ptr_edges), "pointer_duty_cycle": ptr_duty,
          "metronome_edges": len(met_edges), "pointer_alignment": align,
          "emg_candidates": len(cs), "positive_events": int(sum(c.decision == "POSITIVE" for c in cs)),
          "labels_positive": int(np.sum(labels == 1)),
          "labels_negative": int(np.sum(labels == 0)),
          "labels_unknown": int(np.sum(labels == -1))}
    return output, cs, qc


def run(args: argparse.Namespace) -> None:
    root = Path(args.input_dir).expanduser().resolve()
    dest = Path(args.out_dir).expanduser().resolve()
    if not root.is_dir() or root.name != "DATA_TAP":
        raise SystemExit("Input must be an existing directory named DATA_TAP")
    if root == dest or root in dest.parents or dest in root.parents:
        raise SystemExit("Output must be separate from, and not contain, input DATA_TAP")
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit("Output directory is not empty. Use a fresh versioned --out-dir to avoid stale labels.")
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(set(root.rglob("*.csv.gz")) | set(root.rglob("*.csv")))
    if args.max_files is not None:
        files = files[:args.max_files]
    manifest, seen = [], {}
    for path in files:
        relative = path.relative_to(root)
        category = relative.parts[0] if relative.parts else ""
        if category not in CATEGORIES:
            manifest.append({"source": str(relative), "status": "SKIPPED_UNKNOWN_CATEGORY"})
            continue
        digest = sha256_file(path)
        if digest in seen:
            manifest.append({"source": str(relative), "status": "SKIPPED_EXACT_DUPLICATE",
                             "same_as": seen[digest], "sha256": digest})
            continue
        seen[digest] = str(relative)
        entry = {"source": str(relative), "category": category, "sha256": digest}
        try:
            df = pd.read_csv(path, compression="infer", low_memory=False)
            labels, events, qc = label_recording(df, category)
            folder = dest / category
            folder.mkdir(parents=True, exist_ok=True)
            stem = path.name.removesuffix(".gz").removesuffix(".csv")
            label_path = folder / f"{stem}.labels.csv.gz"
            if args.full_rows:
                labels = pd.concat([df.reset_index(drop=True), labels.drop(columns="Timestamp")], axis=1)
            labels.to_csv(label_path, index=False, compression={"method": "gzip", "mtime": 0})
            ev_path = folder / f"{stem}.events.csv"
            pd.DataFrame([asdict(e) for e in events], columns=list(Candidate.__dataclass_fields__)).to_csv(ev_path, index=False)
            entry.update({"status": "OK", "labels_file": str(label_path.relative_to(dest)),
                          "events_file": str(ev_path.relative_to(dest)), **qc})
            print(f"{relative} : {qc['positive_events']} positive events, {qc['emg_candidates']} EMG candidates")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            entry.update({"status": "FAILED_QC", "reason": str(exc)})
            print(f"[QC] {relative}: {exc}")
        manifest.append(entry)
    with (dest / "manifest.jsonl").open("w", encoding="utf-8") as fp:
        for item in manifest:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts = dict(Counter(x["status"] for x in manifest))
    (dest / "summary.json").write_text(json.dumps({"input": str(root), "output": str(dest),
        "files_seen": len(files), "status_counts": counts,
        "warning": "Pseudo-labels are NOT validated physical ground truth. Manually verify held-out events before training."}, indent=2), encoding="utf-8")
    print(f"Completed. Status: {counts}. Audit: {dest / 'manifest.jsonl'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="DATA/DATA_TAP")
    p.add_argument("--out-dir", default="DATA/DATA_TAP_LABELED_V1")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--full-rows", action="store_true", help="Append labels to complete input feature rows; otherwise produce compact sidecars")
    run(p.parse_args())


if __name__ == "__main__":
    main()