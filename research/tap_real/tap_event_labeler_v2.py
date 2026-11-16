#!/usr/bin/env python3
"""Conservative, double-tap-aware TAP event pseudo-labeler for Mudra CSVs.

Reads ONLY DATA/DATA_TAP/**/*.csv.gz (or .csv) and writes sidecar labels,
per-recording event audits and a manifest to a separate output folder.
Does not train a classifier or assume Oculus PointerState is truth.

The double-tap pathway identifies two independent rising muscle events.
It does NOT generate missing events from metronome beats or Oculus transitions,
and the filename alone is never enough to label a positive.

Important: v2 thresholds are candidate defaults, not validated ground truth.
Review independently annotated double taps (especially very short intervals)
before using the pseudo-labels to train production firmware.

Usage:
  python tap_event_labeler_v2.py
  python tap_event_labeler_v2.py --only-double-taps --max-files 20
  python tap_event_labeler_v2.py --input-dir DATA/DATA_TAP --out-dir DATA/DATA_TAP_LABELED_V2
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
    retrigger_strength: float = 0.0
    candidate_id: int = 0
    pair_id: int = 0
    pair_support: bool = False
    pair_gap_ms: float | None = None
    evidence: str = ""
    labeled_rows: int = 0
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


def short_rise_signal(x: np.ndarray, ts_ms: np.ndarray,
                      tau_ms: float = 12.0,
                      lag_ms: float = 24.0) -> np.ndarray:
    """Causal *local* rise, rearmed after each reduction in muscle activation.

    A slow baseline still contains the first activation when the second tap
    arrives. The short-lag comparison is designed to detect the second onset
    even when the signal has not returned to the pre-first-tap baseline.

    This is only a pseudo-label teacher feature. It is not added automatically
    to the embedded model's feature schema.
    """
    x = np.maximum(0.0, np.asarray(x, dtype=np.float64))
    if len(x) < 2:
        return np.zeros_like(x)
    fast = np.empty_like(x)
    fast[0] = x[0]
    for i in range(1, len(x)):
        dt = float(np.clip(ts_ms[i] - ts_ms[i - 1], 1.0, 100.0))
        fast[i] = fast[i - 1] + (1 - math.exp(-dt / tau_ms)) * (x[i] - fast[i - 1])
    past = np.searchsorted(ts_ms, ts_ms - lag_ms, side="left")
    # Use the earlier row, never a future sample. The first few values have
    # insufficient lag history and are excluded by the warm-up mask anyway.
    past = np.minimum(past, np.maximum(np.arange(len(ts_ms)) - 1, 0))
    floor = np.maximum(10.0, np.nanquantile(x, .20, axis=0) * .25)
    return np.maximum(0.0, (fast - fast[past]) / np.maximum(fast[past], floor))


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
                           acc: np.ndarray | None) -> tuple[list[Candidate], np.ndarray, np.ndarray, np.ndarray]:
    en = normalized_onset_signal(emg, ts)
    emg_novelty = np.max(en, axis=1)
    wl_novelty = np.max(normalized_onset_signal(wl, ts), axis=1) if wl is not None else np.zeros(len(ts))
    acc_novelty = np.max(normalized_onset_signal(acc, ts), axis=1) if acc is not None else np.zeros(len(ts))
    retrigger = np.max(short_rise_signal(emg, ts), axis=1)
    # WL is supportive but correlated with EMG. ACC never gates an event.
    score = emg_novelty + 0.10 * np.minimum(wl_novelty, 1.0)
    if len(ts) < 15:
        return [], score, acc_novelty, retrigger
    # Two *independent* routes into the same candidate set:
    #  (1) slow-baseline novelty for the first tap,
    #  (2) short-lag rising edge for the second (possibly softer) tap.
    # Neither route depends on a pointer/beat, and neither creates a positive.
    broad, broad_info = find_peaks(score, height=.12, prominence=.06, distance=2)
    # Real Mudra PSD traces can remain strongly activated for hundreds of
    # milliseconds. Fixed absolute prominence alone splits noisy plateaus
    # into many fake 'events'. A *relative* prominence gate is essential.
    broad = broad[broad_info["prominences"] >= np.maximum(.06, .14 * score[broad])]
    rises, rise_info = find_peaks(retrigger, height=.055, prominence=.038, distance=2)
    rises = rises[rise_info["prominences"] >= np.maximum(.038, .16 * retrigger[rises])]
    seed = sorted(set(int(i) for i in np.r_[broad, rises]))
    clusters: list[list[int]] = []
    for i in seed:
        if clusters and ts[i] - ts[clusters[-1][0]] <= 39.0:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    # The rise peak can precede the broad-energy peak by 10-35ms. Combine
    # these into *one* candidate but never enforce a 75ms global refractory.
    kept = [max(group, key=lambda i: score[i] + .5 * retrigger[i])
            for group in clusters]
    cs = []
    for n, i in enumerate(kept):
        # Estimate *this* rise's onset, not the beginning of the whole burst.
        # If a second peak arrives before the slow baseline decays, searching
        # backwards on that slow baseline can incorrectly locate tap #1.
        low_bound = int(np.searchsorted(ts, ts[i] - 135.0, side="left"))
        if n and ts[i] - ts[kept[n - 1]] < 210:
            low_bound = max(low_bound, kept[n - 1] + 1)
        early = max(low_bound, i - 1)
        lo = max(.022, float(np.max(retrigger[max(0, i - 3):i + 4])) * .20)
        for q in range(i - 1, low_bound - 1, -1):
            if retrigger[q] <= lo:
                early = q + 1
                break
            early = q
        # The rise can be too broad (or virtually absent); the local energy
        # trough is an alternative. It is *bounded* by the previous peak.
        if i > low_bound and (early == low_bound or retrigger[i] < .03):
            local = score[low_bound:i + 1]
            valley = low_bound + int(np.argmin(local))
            if score[valley] <= .75 * score[i]:
                early = max(early, valley)
        k = max(low_bound, min(early, i))
        lo_r = max(0, int(np.searchsorted(ts, ts[i] - 35.0)))
        hi_r = min(len(ts), int(np.searchsorted(ts, ts[i] + 36.0, side="right")))
        cs.append(Candidate(onset_ms=float(ts[k]), peak_ms=float(ts[i]),
                            peak_row=i, strength=float(emg_novelty[i]),
                            wl_strength=float(wl_novelty[i]),
                            acc_strength=float(acc_novelty[i]),
                            retrigger_strength=float(np.max(retrigger[lo_r:hi_r])),
                            candidate_id=n + 1))
    return cs, score, acc_novelty, retrigger


def mark_double_pairs(candidates: list[Candidate], ts: np.ndarray,
                      signal: np.ndarray) -> None:
    """Pair *already observed* independent events, without fabricating taps.

    A paired event must contain a physiologically visible second rise and a
    valley between event peaks. File metadata only enables this extra weak
    evidence; it cannot convert noise alone into a positive.
    """
    options = []
    for i in range(len(candidates) - 1):
        a, b = candidates[i:i + 2]
        gap = b.peak_ms - a.peak_ms
        if not 65.0 <= gap <= 500.0:
            continue
        if min(a.retrigger_strength, b.retrigger_strength) < .055:
            continue
        if max(a.strength, b.strength) < .38:
            continue
        if min(a.strength, b.strength) < .12:
            continue
        idx1, idx2 = a.peak_row, b.peak_row
        if idx2 - idx1 < 3:
            continue
        # A repeated positive-slope pulse requires renewed activation. A
        # plateau/shoulder is not a reliably separated second tap.
        lower = signal[idx1 + 1:idx2]
        valley = float(np.min(lower)) if lower.size else float("inf")
        peak_ref = min(float(signal[idx1]), float(signal[idx2]))
        if not np.isfinite(valley) or valley >= .72 * peak_ref:
            continue
        if b.onset_ms <= a.peak_ms:
            continue
        pair_score = min(a.retrigger_strength, b.retrigger_strength) + .10 * min(a.strength, b.strength)
        options.append((pair_score, i, gap))
    occupied: set[int] = set()
    chosen: list[tuple[int, float]] = []
    for _, i, gap in sorted(options, reverse=True):
        if i in occupied or i + 1 in occupied:
            continue
        occupied.update((i, i + 1))
        chosen.append((i, gap))
    for pair_id, (i, gap) in enumerate(sorted(chosen), start=1):
        for c in candidates[i:i + 2]:
            c.pair_id = pair_id
            c.pair_support = True
            c.pair_gap_ms = float(gap)


def label_recording(df: pd.DataFrame, category: str,
                    recording_name: str = "") -> tuple[pd.DataFrame, list[Candidate], dict[str, Any]]:
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
    cs, score, acc_novelty, retrigger = candidates_from_signal(ts, emg, wl, acc)
    double_protocol = category == "TAP" and "doubletap" in recording_name.casefold()
    if double_protocol:
        # Pairing uses a *separate*, quick positive-rising-edge trace, not
        # the slow baseline that can suppress the second activation.
        mark_double_pairs(cs, ts, retrigger)
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
        if s >= .85 and candidate.retrigger_strength >= .055:
            candidate.decision, candidate.confidence = "POSITIVE", .90
            candidate.evidence = "strong_emg"
        elif s >= .38 and (candidate.pointer_support or candidate.metronome_support):
            candidate.decision, candidate.confidence = "POSITIVE", .78
            candidate.evidence = "emg_plus_weak_reference"
        elif s >= .20 and candidate.pointer_support and candidate.wl_strength >= .10:
            candidate.decision, candidate.confidence = "POSITIVE", .70
            candidate.evidence = "emg_pointer_wl"
        else:
            candidate.decision, candidate.confidence = "UNKNOWN", 0.0
            candidate.evidence = "insufficient_independent_evidence"
    # Second-pass double-tap evidence: a clearly separated pair is itself
    # extra evidence. Require a strong physiological anchor even without
    # a usable Oculus pointer, rather than enforcing an arbitrary .85 cutoff
    # on the first event. Never promote two weak candidates from the filename.
    if double_protocol:
        pairs: dict[int, list[Candidate]] = {}
        for c in cs:
            if c.pair_support:
                pairs.setdefault(c.pair_id, []).append(c)
        for paired in pairs.values():
            if len(paired) != 2:
                continue
            anchor = any(c.decision == "POSITIVE" for c in paired) or (
                max(c.strength for c in paired) >= .68
                and max(c.retrigger_strength for c in paired) >= .14
                and min(c.strength for c in paired) >= .12
            )
            if not anchor:
                continue
            for c in paired:
                if c.decision == "UNKNOWN" and c.strength >= .12 and c.retrigger_strength >= .055:
                    c.decision, c.confidence = "POSITIVE", .65
                    c.evidence = "distinct_retrigger_plus_pair_anchor"
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
        # A weak/undetected tap must not accidentally become a ground-truth
        # negative. Pointer and metronome events can request abstention, but
        # NEVER create positive labels. Apply before event-positive cores.
        cue_holdout = np.zeros(len(ts), dtype=bool)
        if ptr_valid:
            # If the offset cannot be trusted, use a wider exclusion region.
            ptr_hint = corrected_ptr if align["trusted"] else ptr_edges
            ptr_radius = 140.0 if align["trusted"] else 280.0
            for t in ptr_hint:
                cue_holdout |= np.abs(ts - t) <= ptr_radius
        if len(met_edges):
            # DoubleTaps can have two muscle activations per beat/protocol
            # cycle. Do not assume one positive per beat; simply avoid hard
            # negatives near the expected activity interval.
            half_window = 400.0 if double_protocol else 220.0
            if math.isfinite(median_met_period):
                half_window = min(half_window, .48 * median_met_period)
            for t in met_edges:
                cue_holdout |= np.abs(ts - t) <= half_window
        labels[cue_holdout] = -1
        weights[cue_holdout] = 0
        for c in cs:
            near = (ts >= c.onset_ms - 130) & (ts <= c.peak_ms + 170)
            labels[near] = -1
            weights[near] = 0
        accepted = [c for c in cs if c.decision == "POSITIVE"]
        for number, c in enumerate(accepted, 1):
            c.event_id = number
            # Short positive target starting with physiologically inferred onset.
            # Only causal EMG features available at this row are training inputs.
            next_candidate_index = c.candidate_id  # candidate IDs start at 1
            next_onset = cs[next_candidate_index].onset_ms if next_candidate_index < len(cs) else float("inf")
            # A positive row must belong to exactly one event; no core may
            # overflow into the next tap when the two taps are very close.
            core_stop = min(c.onset_ms + 75.0, c.peak_ms + 55.0,
                            next_onset - max(1.0, median_dt * .5))
            core = (ts >= c.onset_ms) & (ts <= core_stop)
            labels[core] = 1
            weights[core] = np.float32(c.confidence)
            event_id[core] = number
        if not accepted:
            # This could be a genuine quiet tap session with bad placement.
            # Classifying its entire contents as negatives would teach the
            # student to reject unobserved soft taps.
            labels[:] = -1
            weights[:] = 0
    # Do not learn from timing gaps or warm-up/cool-down.
    invalid = np.zeros(len(ts), bool)
    invalid[ts <= ts[0] + 150.0] = True
    for i in np.flatnonzero(diffs > 150):
        invalid[(ts >= ts[i] - 20) & (ts <= ts[i + 1] + 100)] = True
    labels[invalid] = -1
    weights[invalid] = 0
    event_id[invalid] = -1
    for c in cs:
        if c.event_id > 0:
            c.labeled_rows = int(np.sum((labels == 1) & (event_id == c.event_id)))
    output = pd.DataFrame({"Timestamp": df["Timestamp"], "tap_label": labels,
                           "tap_label_weight": weights, "tap_event_id": event_id})
    qc = {"rows": int(len(df)), "columns": int(len(df.columns)),
          "duration_s": round(float((ts[-1] - ts[0]) / 1000.), 3),
          "median_dt_ms": median_dt, "p95_dt_ms": float(np.quantile(diffs, .95)),
          "timestamp_gaps_gt_150ms": int(np.sum(diffs > 150)),
          "pointer_edges": len(ptr_edges), "pointer_duty_cycle": ptr_duty,
          "metronome_edges": len(met_edges), "pointer_alignment": align,
          "emg_candidates": len(cs), "positive_events": int(sum(c.decision == "POSITIVE" for c in cs)),
          "doubletap_protocol": double_protocol,
          "distinct_double_pairs": len({c.pair_id for c in cs if c.pair_support}),
          "pair_supported_positive_events": sum(c.pair_support and c.decision == "POSITIVE" for c in cs),
          "unknown_events": sum(c.decision == "UNKNOWN" for c in cs),
          "trainable_positive_events": sum(c.labeled_rows > 0 for c in cs),
          "events_lost_to_gap_or_qc": sum(c.decision == "POSITIVE" and c.labeled_rows == 0 for c in cs),
          "no_confirmed_taps_abstained": bool(category == "TAP" and not any(c.decision == "POSITIVE" for c in cs)),
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
    if args.only_double_taps:
        files = [p for p in files if p.parent.name == "TAP"
                 and "doubletap" in p.name.casefold()]
    if args.max_files is not None:
        files = files[:args.max_files]
    if not files:
        raise SystemExit("No input files matched the requested filters")
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
            labels, events, qc = label_recording(df, category, recording_name=path.name)
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
    accepted = [m for m in manifest if m.get("status") == "OK"]
    double_success = [m for m in accepted if m.get("doubletap_protocol")]
    (dest / "summary.json").write_text(json.dumps({"input": str(root), "output": str(dest),
        "files_seen": len(files), "status_counts": counts,
        "doubletap_files_processed": len(double_success),
        "doubletap_paired_events": int(sum(x.get("distinct_double_pairs", 0) for x in double_success)),
        "doubletap_positive_events": int(sum(x.get("positive_events", 0) for x in double_success)),
        "doubletap_no_confirmed_taps": int(sum(x.get("no_confirmed_taps_abstained", False) for x in double_success)),
        "warning": "Pseudo-labels are NOT validated physical ground truth. Manually verify held-out events before training."}, indent=2), encoding="utf-8")
    print(f"Completed. Status: {counts}. Audit: {dest / 'manifest.jsonl'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="DATA/DATA_TAP")
    p.add_argument("--out-dir", default="DATA/DATA_TAP_LABELED_V2")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--only-double-taps", action="store_true", help="Audit TAP/*DoubleTaps* recordings only")
    p.add_argument("--full-rows", action="store_true", help="Append labels to complete input feature rows; otherwise produce compact sidecars")
    run(p.parse_args())


if __name__ == "__main__":
    main()