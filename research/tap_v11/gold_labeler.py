#!/usr/bin/env python3
"""Conservative GOLD TAP corrections to existing, file-fold ACC+EMG teacher labels.

No model is fitted or dispatched here. Input must be the CURRENT production
file-aware teacher export; do not generate it with the old person-fold script.
Original data and labels are read-only. The two changes are independent:
  recover: promote strongly supported, distinct fast/double partner events;
  background: restore only demonstrably quiet, unambiguous unknown background.
  both: combine them. The untouched input directory is the baseline.
See GOLD_README.md for the required evidence contract and replay protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "gold-v11.1"
ACC = [f"p_core->acc_psd[0][{a}][{z}]" for a in "XYZ" for z in range(1, 6)]
CH = [f"ch{i}_label" for i in (1, 2, 3)]
REQUIRED_LABELS = ["Timestamp", "gesture_kind", "tap_label", "tap_active",
                   "tap_event_id", "hold_label", "n_ch_off", *CH]
REQUIRED_EVENTS = ["recording", "type", "tier", "t_on", "t_ptr", "t_rel",
                   "score", "labeled", "k"]
NEGATIVE_FOLDERS = {"FINGER", "WRIST", "ARM", "BAND", "NOTHING",
                    "CONNECTED", "DISCONNECTED", "LOOSE", "TABLE"}
HOLD_FOLDERS = {"LONG_PRESS", "SOFT_LONG_PRESS", "MID_LONG_PRESS",
                "STRONG_LONG_PRESS", "DRAG_AND_DROP"}


@dataclass(frozen=True)
class Policy:
    strong_score: float
    mode: str = "recover"
    window_ms: float = 60.0
    min_partner_ms: float = 150.0
    max_partner_ms: float = 450.0
    fast_bpm: int = 240
    guard_ms: float = 250.0
    require_emg_active: bool = False
    motion_background: bool = False
    low_score: float | None = None

    def validate(self):
        if not 0 < self.strong_score <= 1:
            raise ValueError("strong_score must be the current calibrated teacher threshold in (0,1]")
        if self.mode not in ("recover", "background", "both"):
            raise ValueError("unknown mode")
        if not 0 < self.window_ms <= 100:
            raise ValueError("positive window must be short (<=100 ms)")
        if not 0 < self.window_ms < self.min_partner_ms <= self.max_partner_ms:
            raise ValueError("partner separation must exceed the positive window")
        if self.guard_ms < self.max_partner_ms / 2:
            raise ValueError("background guard is too short")
        if self.fast_bpm < 1:
            raise ValueError("fast_bpm must be positive")
        if self.motion_background and self.mode not in ("background", "both"):
            raise ValueError("motion-background needs background mode")
        if self.low_score is not None and not 0 <= self.low_score < self.strong_score:
            raise ValueError("low_score must be below strong_score")
        if self.motion_background and self.low_score is None:
            raise ValueError("motion-background requires a calibrated --low-score")


def need(df, cols, label):
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def numeric(df, col):
    return pd.to_numeric(df[col], errors="raise").to_numpy()


def times(raw, base):
    a = numeric(raw, "Timestamp")
    b = numeric(base, "Timestamp")
    if len(a) == 0 or len(a) != len(b) or not np.array_equal(a, b):
        raise ValueError("raw/label Timestamp mismatch; no positional or fuzzy join")
    if not np.isfinite(a).all() or (a[1:] < a[:-1]).any():
        raise ValueError("nonfinite or nonmonotonic Timestamp")
    return (a - a[0]).astype(np.float64)


def ternary(base, col, allowed=(-1, 0, 1)):
    x = numeric(base, col)
    if not np.isin(x, allowed).all():
        raise ValueError(f"invalid {col}; never coerce unknown into binary")
    return x.astype(np.int8)


def finite_number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def near(t, centers, radius):
    c = np.sort(np.asarray([x for x in centers if x is not None and np.isfinite(x)], dtype=float))
    if not len(c):
        return np.zeros(len(t), dtype=bool)
    j = np.searchsorted(c, t)
    d1 = np.abs(t - c[np.clip(j - 1, 0, len(c) - 1)])
    d2 = np.abs(t - c[np.clip(j, 0, len(c) - 1)])
    return np.minimum(d1, d2) <= radius


def spans(t, intervals):
    """Inclusive interval mask, in source-row time order."""
    diff = np.zeros(len(t) + 1, dtype=np.int64)
    for lo, hi in intervals:
        if lo is None or hi is None or hi < lo:
            continue
        a = np.searchsorted(t, lo, "left")
        b = np.searchsorted(t, hi, "right")
        diff[a] += 1
        diff[b] -= 1
    return np.cumsum(diff[:-1]) > 0


def pointer(raw, t):
    if "PointerState" not in raw:
        return None, np.array([], dtype=float)
    p = numeric(raw, "PointerState")
    if not np.isfinite(p).all() or not np.isin(p, (0, 1)).all():
        return None, np.array([], dtype=float)
    active = p == 1
    edges = t[1:][active[1:] != active[:-1]]
    return active, edges


def positive_kind(recording):
    parts = set(Path(recording).parts[:-1])
    parts = {x.upper() for x in parts}
    known = []
    if "TAP" in parts:
        known.append(1)
    if "SWIPE" in parts:
        known.append(2)
    if parts & HOLD_FOLDERS:
        known.append(3)
    if parts & NEGATIVE_FOLDERS:
        known.append(0)
    if len(known) != 1:
        raise ValueError(f"ambiguous/unknown gesture folder: {recording}")
    return known[0]


def is_fast(recording, min_bpm):
    name = Path(recording).name.lower()
    normalized = re.sub(r"[^a-z0-9]", "", name)
    return ("doubletap" in normalized or
            any(int(x) >= min_bpm for x in re.findall(r"(\d+)bpm", normalized)))


def good_contact(ch, bad):
    # A channel labeled REST or ACTIVE was confidently connected by the
    # existing contact teacher. Unknown is NOT proof of a good electrode.
    return (~bad) & ((np.isin(ch, (0, 1))).sum(axis=1) >= 2)


def tap_events_mask(events):
    return events["type"].astype(str).str.startswith("tap").to_numpy()


def repair_overrides(base, ev, t, kind, bad, audit):
    """Confirmed OFF and authoritative no-gesture states remain negatives."""
    y = base["tap_label"].to_numpy(copy=True)
    a = base["tap_active"].to_numpy(copy=True)
    ids = base["tap_event_id"].to_numpy(copy=True)
    locked = np.ones(len(t), bool) if kind == 0 else bad
    audit["overridden_positive_rows"] = int(np.count_nonzero(locked & (y == 1)))
    audit["locked_negative_rows"] = int(locked.sum())
    y[locked], a[locked], ids[locked] = 0, 0, -1
    base["tap_label"], base["tap_active"], base["tap_event_id"] = y, a, ids
    for i in ev.index[tap_events_mask(ev)]:
        on = finite_number(ev.at[i, "t_on"])
        if on is None:
            continue
        row = int(np.clip(np.searchsorted(t, on, "left"), 0, len(t)-1))
        if locked[row]:
            ev.at[i, "labeled"] = 0
            if bad[row]:
                ev.at[i, "rule_disconnect"] = True
            ev.at[i, "gold_reason"] = "authoritative_negative" if kind == 0 else "disconnected"


def recover_partners(base, ev, t, ch, bad, recording, policy, audit):
    """Only strong sensor-only partner events; no count/rhythm-forced labels.

    Use the existing sensor-fusion score, not pointer or EMG amplitude as
    truth. Original positives and zeros are immutable in this step.
    """
    if not is_fast(recording, policy.fast_bpm):
        return
    y = base["tap_label"].to_numpy(copy=True)
    a = base["tap_active"].to_numpy(copy=True)
    ids = base["tap_event_id"].to_numpy(copy=True)
    contact = good_contact(ch, bad)
    is_tap = tap_events_mask(ev)
    anchors = []
    for i in ev.index[is_tap]:
        on = finite_number(ev.at[i, "t_on"])
        score = finite_number(ev.at[i, "score"])
        if (on is not None and score is not None and
            ev.at[i, "labeled"] == 1 and score >= policy.strong_score and
            np.any((y == 1) & (np.abs(t-on) <= 100))):
            anchors.append(on)
    anchors = np.unique(np.asarray(anchors, dtype=float))
    audit["anchor_events"] = int(len(anchors))
    if not len(anchors):
        return

    # A promoted event cannot become an anchor for another promotion.
    original_positive_times = t[y == 1]
    chosen = list(anchors)
    candidates = []
    for i in ev.index[is_tap]:
        tier = str(ev.at[i, "tier"]).upper()
        if tier not in {"ORPHAN", "INPRESS"} or ev.at[i, "labeled"] != -1:
            continue
        on, score = finite_number(ev.at[i, "t_on"]), finite_number(ev.at[i, "score"])
        if on is not None and score is not None and score >= policy.strong_score:
            candidates.append((i, on, score))
    # Resolve competing peaks by confidence then timestamp; deterministic.
    candidates.sort(key=lambda z: (-z[2], z[1], z[0]))
    used_ids = [int(x) for x in ids if np.isfinite(x) and x >= 0]
    used_ids += [int(x) for x in pd.to_numeric(ev["k"], errors="coerce").dropna() if x >= 0]
    next_id = max(used_ids, default=-1) + 1
    rejected = {}
    def skip(i, reason):
        rejected[reason] = rejected.get(reason, 0) + 1
        ev.at[i, "gold_reason"] = reason

    for i, on, score in candidates:
        if not (0 <= on <= t[-1]):
            skip(i, "out_of_range")
            continue
        earlier = anchors[anchors < on]
        if not len(earlier) or not policy.min_partner_ms <= on-earlier[-1] <= policy.max_partner_ms:
            skip(i, "no_independent_previous_anchor")
            continue
        if min(abs(on-c) for c in chosen) < policy.min_partner_ms:
            skip(i, "too_close_to_confirmed_or_promoted")
            continue
        if len(original_positive_times) and np.min(np.abs(original_positive_times-on)) < policy.window_ms:
            skip(i, "near_existing_positive")
            continue
        # Respect a candidate-specific observable interval supplied by the
        # CURRENT pipeline. Otherwise use the fixed 60 ms comparison window.
        lo = finite_number(ev.at[i, "window_start_ms"]) if "window_start_ms" in ev else None
        hi = finite_number(ev.at[i, "window_end_ms"]) if "window_end_ms" in ev else None
        if lo is None and hi is None:
            lo, hi = on, on + policy.window_ms
        elif lo is None or hi is None or not (-100 <= lo-on <= 100 and 0 < hi-lo <= 100):
            skip(i, "invalid_observable_window")
            continue
        if lo < t[0] or hi > t[-1]:
            skip(i, "truncated_window")
            continue
        q = (t >= lo) & (t < hi)
        context = (t >= on-100) & (t <= on+100)
        # Reject truncated windows and sparse source intervals (do not label
        # an apparent event across a gap in acquisition).
        if q.sum() < 2 or not context.any():
            skip(i, "insufficient_rows")
            continue
        ts = np.unique(t[q])
        if len(ts) < 2 or np.max(np.diff(ts)) > policy.window_ms/2:
            skip(i, "timestamp_gap")
            continue
        if np.any(y[q] != -1):
            skip(i, "would_overwrite_known_label")
            continue
        if np.any(bad[q]) or not np.all(contact[q]):
            skip(i, "unconfirmed_contact")
            continue
        # The fused score already uses EMG+ACC+WL. Extra independent
        # ACTIVE evidence is an optional stricter ablation, not a hard
        # absolute amplitude requirement on all soft taps.
        emg = bool(np.any((ch[context] == 1) & (~bad[context, None])))
        if policy.require_emg_active and not emg:
            skip(i, "no_extra_emg_activity")
            continue
        if np.any(np.isin(ids[q], list(used_ids)) & (ids[q] >= 0)) or np.any(a[q] == 1):
            skip(i, "existing_event_extent")
            continue
        y[q], a[q], ids[q] = 1, 1, next_id
        # Do not invent a release or claim a whole hold. Only the short
        # positive onset is known; active outside that window is unchanged.
        ev.at[i, "labeled"] = 1
        ev.at[i, "k"] = next_id
        ev.at[i, "gold_reason"] = "recovered_fast_double"
        ev.at[i, "gold_emg_active"] = int(emg)
        ev.at[i, "gold_window_start_ms"] = lo
        ev.at[i, "gold_window_end_ms"] = hi
        ev.at[i, "rule_disconnect"] = False
        ev.at[i, "t_off"] = np.nan
        used_ids.append(next_id)
        next_id += 1
        chosen.append(on)
        audit["recovered_events"] += 1
        audit["recovered_rows"] += int(q.sum())
        audit["recovered_with_extra_emg_activity"] += int(emg)

    base["tap_label"], base["tap_active"], base["tap_event_id"] = y, a, ids
    audit["recovery_rejections"] = rejected


def quiet_acc(raw, reference):
    """Legacy encoded ACC-bin sums, robust reference; no physical-unit claim."""
    if not set(ACC).issubset(raw.columns) or reference.sum() < 20:
        return None
    x = raw[ACC].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1)
    ref = reference & valid
    if ref.sum() < 20:
        return None
    p = x.reshape(-1, 3, 5).sum(axis=2)
    p = np.column_stack([p, p.sum(axis=1)])
    v = p[ref]
    med = np.median(v, axis=0)
    q25, q75, q90 = np.percentile(v, (25, 75, 90), axis=0)
    mad = 1.4826 * np.median(np.abs(v-med), axis=0)
    scale = np.maximum.reduce([mad, (q75-q25)/1.349,
                               np.abs(med)*.05, np.abs(q90)*.01,
                               np.full(4, 1e-6)])
    # Same robust score idea as legacy support; this is an experimental
    # quiet exclusion gate, not a new tap identity classifier.
    return valid & np.all((p-med)/scale <= 3.0, axis=1)


def restore_background(raw, base, ev, t, ch, bad, policy, audit):
    """Only unknown, protected-from-all-event context, sensor-quiet rows.

    Optional movement-background arm also requires a continuous low fused
    teacher score, not just a missing peak in the sparse event export.
    """
    active, edges = pointer(raw, t)
    if active is None:
        audit["background_skipped"] = "missing_or_invalid_pointer"
        return
    y = base["tap_label"].to_numpy(copy=True)
    a = base["tap_active"].to_numpy(copy=True)
    ids = base["tap_event_id"].to_numpy(copy=True)
    is_tap = tap_events_mask(ev)
    e = ev.loc[is_tap]
    all_events = [finite_number(v) for v in e["t_on"]]
    # Guard every exported tap proposal, even weak/rejected candidates.
    # Guard pointer intervals separately, including missed/merged cycles.
    guards = near(t, all_events, policy.guard_ms) | near(t, edges, policy.guard_ms)
    for row in e.itertuples():
        u = finite_number(getattr(row, "t_ptr", None))
        d = finite_number(getattr(row, "t_rel", None))
        if u is not None and d is not None:
            guards |= spans(t, [(u-policy.guard_ms, d+policy.guard_ms)])
    guards |= near(t, t[y == 1], policy.guard_ms)
    guards |= (t < 1000) | (t > t[-1]-policy.guard_ms)
    contact = good_contact(ch, bad)
    allowed = contact & ~active & ~guards & (a != 1) & (ids < 0)
    allowed &= ternary(base, "hold_label") != 1
    rest = ((ch == 0).sum(axis=1) >= 2) & ~(ch == 1).any(axis=1)
    reference = allowed & rest & (y == 0)
    quiet = quiet_acc(raw, reference)
    if quiet is None:
        audit["background_skipped"] = "insufficient_trusted_quiet_reference_or_ACC"
        return
    # A large local impulse in neighboring frames also rules out quiet rest.
    quiet &= ~near(t, t[~quiet], 100)
    if policy.low_score is not None:
        if "tap_teacher_score" not in base:
            raise ValueError("calibrated low-score mode requires aligned tap_teacher_score")
        score = numeric(base, "tap_teacher_score")
        low = np.isfinite(score) & (score >= 0) & (score <= policy.low_score)
        low &= ~near(t, t[~low], 100)
    else:
        low = np.ones(len(t), bool)
    quiet_add = (y == -1) & allowed & quiet & rest & low
    movement_add = np.zeros(len(t), bool)
    if policy.motion_background:
        movement_add = (y == -1) & allowed & quiet & ~rest & low
        movement_add &= (ch == 1).any(axis=1)
    y[quiet_add | movement_add] = 0
    base["tap_label"] = y
    audit["restored_quiet_rows"] = int(quiet_add.sum())
    audit["restored_movement_rows"] = int(movement_add.sum())


def correct_recording(raw, base, events, recording, policy):
    """Pure in-memory API. Caller supplies already cross-fit teacher evidence.

    No file/participant split or prediction happens here. Returns corrected
    sidecar, corrected event table, and transparent per-recording counts.
    """
    policy.validate()
    need(raw, ["Timestamp"], "raw")
    need(base, REQUIRED_LABELS, "base sidecar")
    need(events, REQUIRED_EVENTS, "event report")
    t = times(raw, base)
    kind = positive_kind(recording)
    if not np.all(numeric(base, "gesture_kind") == kind):
        raise ValueError("sidecar gesture_kind differs from raw folder")
    base = base.copy(deep=True)
    ev = events.copy(deep=True).reset_index(drop=True)
    if len(ev) and not (ev["recording"] == recording).all():
        raise ValueError("event table belongs to a different recording")
    original = ternary(base, "tap_label")
    ternary(base, "tap_active")
    ternary(base, "hold_label")
    ch = np.column_stack([ternary(base, c, (-1, 0, 1, 2)) for c in CH])
    count = numeric(base, "n_ch_off")
    if not np.array_equal(count, (ch == 2).sum(axis=1)):
        raise ValueError("n_ch_off inconsistent with channel labels")
    bad = count >= 2
    base["tap_label"] = original
    base["tap_active"] = ternary(base, "tap_active")
    event_ids = numeric(base, "tap_event_id")
    if not np.isfinite(event_ids).all() or (event_ids < -1).any() or not np.all(event_ids == np.floor(event_ids)):
        raise ValueError("invalid tap_event_id")
    base["tap_event_id"] = event_ids.astype(np.int64)
    ev["gold_reason"] = ""
    ev["gold_emg_active"] = np.nan
    ev["gold_window_start_ms"] = np.nan
    ev["gold_window_end_ms"] = np.nan
    audit = dict(recording=recording, kind=kind, rows=len(t),
                 before=np.bincount(original+1, minlength=3).tolist(),
                 recovered_events=0, recovered_rows=0,
                 recovered_with_extra_emg_activity=0,
                 restored_quiet_rows=0, restored_movement_rows=0,
                 disconnected_rows=int(bad.sum()))
    repair_overrides(base, ev, t, kind, bad, audit)
    if kind in (1, 2):
        if policy.mode in ("recover", "both"):
            recover_partners(base, ev, t, ch, bad, recording, policy, audit)
        if policy.mode in ("background", "both"):
            restore_background(raw, base, ev, t, ch, bad, policy, audit)
    result = ternary(base, "tap_label")
    if np.any(result[bad] != 0):
        raise AssertionError("disconnection protection lost")
    if kind == 0 and np.any(result != 0):
        raise AssertionError("negative folder lost")
    if kind == 3 and np.any(result[~bad] != original[~bad]):
        raise AssertionError("hold/drag phase modified")
    if kind in (1, 2):
        if np.any((original == 1) & ~bad & (result != 1)):
            raise AssertionError("original confirmed positive changed")
        if np.any((original == 0) & (result != 0)):
            raise AssertionError("known negative changed")
    if not np.array_equal(base["Timestamp"], raw["Timestamp"]):
        raise AssertionError("row alignment lost")
    audit["after"] = np.bincount(result+1, minlength=3).tolist()
    return base, ev, audit


def sidecar_rel(recording):
    if not re.search(r"\.csv(?:\.gz)?$", recording, re.I):
        raise ValueError(f"not a recording CSV: {recording}")
    return re.sub(r"\.csv(?:\.gz)?$", ".labels.csv.gz", recording, flags=re.I)


def safe_rel(value):
    p = Path(str(value))
    if p.is_absolute() or ".." in p.parts or str(p) == "." or "\\" in str(value):
        raise ValueError(f"unsafe/noncanonical relative recording path: {value}")
    return p.as_posix()


def read_spec(path):
    s = json.loads(Path(path).read_text(encoding="utf8"))
    if s.get("split_policy") != "gesture_aware_file":
        raise ValueError("teacher spec must explicitly declare gesture_aware_file, never person folds")
    families = {str(x).upper() for x in s.get("feature_families", [])}
    if not {"ACC", "EMG"}.issubset(families):
        raise ValueError("teacher evidence must be fused ACC + EMG")
    if not s.get("run_id"):
        raise ValueError("teacher spec requires run_id from actual file-fold teacher run")
    return s


def run(raw_root, base_root, out, spec, policy):
    raw_root, base_root, out = (Path(p).resolve() for p in (raw_root, base_root, out))
    if not raw_root.is_dir() or not base_root.is_dir():
        raise FileNotFoundError("raw and base directories must exist")
    if out.exists() or out == raw_root or out == base_root or raw_root in out.parents or base_root in out.parents:
        raise ValueError("out must be a NEW directory outside the input trees")
    summary_path, event_path = base_root/"_summary.json", base_root/"_events.csv.gz"
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    if summary.get("totals", {}).get("failed") != 0:
        raise ValueError("base label export is incomplete or failed")
    records = summary.get("recordings")
    if not isinstance(records, dict) or not records:
        raise ValueError("base summary must enumerate exact recordings")
    keys = [safe_rel(k) for k in records]
    if len(set(keys)) != len(keys) or keys != list(records):
        raise ValueError("noncanonical or duplicate source recording keys")
    paths = [sidecar_rel(k) for k in keys]
    if len(set(paths)) != len(paths):
        raise ValueError("two source files map to one sidecar; deduplicate first")
    ev = pd.read_csv(event_path, low_memory=False)
    need(ev, REQUIRED_EVENTS, "event report")
    if not set(ev["recording"]).issubset(records):
        raise ValueError("event table has recordings absent from summary")
    by_file = {k: v for k, v in ev.groupby("recording", sort=False)}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name+".building.", dir=out.parent))
    reports, corrected_events = [], []
    try:
        for n, key in enumerate(keys, 1):
            raw_file = raw_root/key
            label_file = base_root/sidecar_rel(key)
            if not raw_file.is_file() or not label_file.is_file():
                raise FileNotFoundError(f"missing exact raw/sidecar: {key}")
            raw = pd.read_csv(raw_file, usecols=lambda c: c in ["Timestamp", "PointerState", *ACC], low_memory=False)
            base = pd.read_csv(label_file, low_memory=False)
            corrected, e, report = correct_recording(raw, base, by_file.get(key, ev.iloc[:0]), key, policy)
            dest = staging/sidecar_rel(key)
            dest.parent.mkdir(parents=True, exist_ok=True)
            corrected.to_csv(dest, index=False, compression="gzip")
            corrected_events.append(e)
            reports.append(report)
            st = summary["recordings"][key]
            st["tap"] = report["after"]
            st["taps_labeled"] = int(((e["labeled"] == 1) & e["type"].astype(str).str.startswith("tap")).sum())
            if n % 100 == 0 or n == len(keys):
                print(f"GOLD {n}/{len(keys)} recovered={sum(x['recovered_events'] for x in reports)}", flush=True)
        all_ev = pd.concat(corrected_events, ignore_index=True)
        all_ev.to_csv(staging/"_events.csv.gz", index=False, compression="gzip")
        # Never evaluate corrected predictions against these new hypotheses.
        # The ORIGINAL event table is the frozen comparison reference.
        shutil.copy2(event_path, staging/"_events_reference.csv.gz")
        totals = {k: sum(x[k] for x in reports) for k in
                  ("rows", "recovered_events", "recovered_rows",
                   "recovered_with_extra_emg_activity", "restored_quiet_rows",
                   "restored_movement_rows", "disconnected_rows",
                   "overridden_positive_rows", "locked_negative_rows")}
        totals["files"] = len(keys)
        totals["before"] = np.sum([x["before"] for x in reports], axis=0).tolist()
        totals["after"] = np.sum([x["after"] for x in reports], axis=0).tolist()
        meta = dict(version=VERSION, teacher_run=spec["run_id"],
                    split_policy=spec["split_policy"], policy=asdict(policy),
                    totals=totals, recordings=reports,
                    reference="_events_reference.csv.gz",
                    warning="Recovered events are hypotheses, not independent physical ground truth.")
        summary["totals"]["taps_labeled"] = sum(s["taps_labeled"] for s in summary["recordings"].values())
        summary["gold_correction"] = {k: v for k, v in meta.items() if k != "recordings"}
        (staging/"_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
        (staging/"_gold_report.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
        if out.exists():
            raise FileExistsError(out)
        staging.rename(out)
        print(json.dumps(totals, indent=2), flush=True)
        return totals
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--teacher-spec", required=True, help="actual gesture-aware-file fused teacher run metadata")
    p.add_argument("--mode", choices=["recover", "background", "both"], default="recover")
    p.add_argument("--window-ms", type=float, default=60)
    p.add_argument("--min-partner-ms", type=float, default=150)
    p.add_argument("--max-partner-ms", type=float, default=450)
    p.add_argument("--fast-bpm", type=int, default=240)
    p.add_argument("--guard-ms", type=float, default=250)
    p.add_argument("--require-emg-active", action="store_true")
    p.add_argument("--motion-background", action="store_true")
    a = p.parse_args()
    spec = read_spec(a.teacher_spec)
    policy = Policy(strong_score=float(spec["strong_score"]), mode=a.mode,
                    window_ms=a.window_ms, min_partner_ms=a.min_partner_ms,
                    max_partner_ms=a.max_partner_ms, fast_bpm=a.fast_bpm,
                    guard_ms=a.guard_ms, require_emg_active=a.require_emg_active,
                    motion_background=a.motion_background,
                    low_score=(float(spec["low_score"]) if "low_score" in spec else None))
    policy.validate()
    run(a.raw, a.base, a.out, spec, policy)


if __name__ == "__main__":
    main()
