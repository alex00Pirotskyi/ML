#!/usr/bin/env python3
"""Conservative LP state-labeling candidate (offline, not physical ground truth).

lp_state: 1 = EMG-supported rest/release, 0 = EMG-supported active/hold,
         -1 = unknown/ignore.  Never interpret -1 as an active negative.

PointerState provides *weak timing context*, not a force/EMG ground truth.
Only the stable interior of a pointer interval can receive a label, and an
independent, per-recording WL + PSD signal criterion must agree.  This script
has no learned teacher, no participant/global calibration, no label leakage.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, maximum_filter1d

VERSION = "lp_state_strict_v4_candidate"
SOURCE = Path("DATA/DATA_LONG_PRESS")
DEST = Path("DATA/DATA_LONG_PRESS_LP_STATE_V4")
LP_GROUPS = ("SOFT_LONG_PRESS", "MID_LONG_PRESS", "STRONG_LONG_PRESS", "LONG_PRESS")
CONTROL_GROUPS = ("CONNECTED", "NOTHING")
GROUPS = LP_GROUPS + CONTROL_GROUPS
WL = [f"p_core->wl[CH{ch}]" for ch in (1, 2, 3)]
PSD = [[f"p_core->emg_psd_mean[CH{ch}][{z}]" for z in range(5)]
       for ch in (1, 2, 3)]
BIN_COUNTS = np.asarray([8., 10., 12., 46., 51.])
READ = set(WL) | {x for ch in PSD for x in ch} | {
    "PointerState", "Timestamp", "MonotonicTimeUs", "psd_mean_applied"
}
ADDED = ("lp_state", "lp_state_weight", "lp_teacher_version")


def to_numbers(df, names):
    return np.column_stack([pd.to_numeric(df[k], errors="coerce").to_numpy(float)
                            for k in names])


def sample_timing(df):
    """Require real clock information; never silently assume 6 ms."""
    for name, scale in (("MonotonicTimeUs", .001), ("Timestamp", 1.)):
        if name not in df:
            continue
        t = pd.to_numeric(df[name], errors="coerce").to_numpy(float)
        d = np.diff(t) * scale
        normal = d[np.isfinite(d) & (d >= 1.) & (d <= 60.)]
        if len(normal) < max(10, min(len(d), 100)//2):
            continue
        dt = float(np.median(normal))
        gaps = np.r_[False, ~np.isfinite(d) | (d <= 0) | (d > max(3*dt, 40.))]
        return dt, gaps
    return None, np.ones(len(df), bool)


def frames(ms, dt):
    return max(1, int(round(ms / dt)))


def dilate(mask, size):
    return maximum_filter1d(np.asarray(mask, np.uint8),
                            size=2*size+1, mode="constant") > 0


def keep_long_runs(mask, minimum):
    """Never fill gaps; reject candidates shorter than minimum observations."""
    arr = np.asarray(mask, bool)
    out = np.zeros(len(arr), bool)
    starts = np.flatnonzero(arr & ~np.r_[False, arr[:-1]])
    ends = np.flatnonzero(arr & ~np.r_[arr[1:], False]) + 1
    for a, b in zip(starts, ends):
        if b-a >= minimum:
            out[a:b] = True
    return out


def parse_pointer(series):
    raw = series.astype("string").str.strip().str.lower()
    vals = raw.map({"true": 1, "false": 0, "1": 1, "0": 0,
                    "1.0": 1, "0.0": 0}).to_numpy(dtype=float, na_value=np.nan)
    return vals


def pointer_interiors(df, dt, boundary_ms):
    n = len(df)
    if "PointerState" not in df:
        return np.full(n, -1, np.int8), np.ones(n, bool)
    raw = parse_pointer(df["PointerState"])
    valid = np.isfinite(raw)
    if valid.sum() < max(30, n//2):
        return np.full(n, -1, np.int8), np.ones(n, bool)
    window = frames(61., dt) | 1
    smooth = median_filter(np.where(valid, raw, .5), size=window, mode="nearest")
    state = np.where(smooth > .5, 1, np.where(smooth < .5, 0, -1)).astype(np.int8)
    change = np.r_[False, state[1:] != state[:-1]]
    changed = valid & (state != raw)
    unknown = (~valid) | (state == -1) | changed
    guard = dilate(unknown | change, frames(boundary_ms, dt))
    return state, guard


def label_recording(df, group, boundary_ms=250., min_active_ms=180.):
    """Return labels and reasoned file audit; fail closed on incomplete data."""
    n = len(df)
    y = np.full(n, -1, np.int8)
    audit = {"n": n, "group": group, "reason": "", "good_channels": [],
             "verified_events": 0, "pointer_events": 0, "events": []}
    def done(reason):
        audit["reason"] = reason
        audit["rest"] = int(np.sum(y == 1))
        audit["active"] = int(np.sum(y == 0))
        audit["unknown"] = int(np.sum(y == -1))
        return pd.DataFrame({"lp_state": y, "lp_state_weight": (y >= 0).astype(float),
                             "lp_teacher_version": VERSION}), audit

    if group not in GROUPS or n < 100:
        return done("unsupported_group_or_short")
    if not all(k in df for k in WL) or not all(k in df for p in PSD for k in p):
        return done("missing_emg_features")
    dt, gaps = sample_timing(df)
    if dt is None:
        return done("invalid_or_missing_timestamps")
    audit["dt_ms"] = dt
    ptr, guard = pointer_interiors(df, dt, boundary_ms)
    guard |= dilate(gaps, frames(boundary_ms, dt))
    # Reject short pointer blips, not merely their edges.
    for state_value, minimum_ms in ((1, 450.), (0, 250.)):
        run = ptr == state_value
        short = run & ~keep_long_runs(run, frames(minimum_ms, dt))
        guard |= short
    off = (ptr == 0) & ~guard
    on = (ptr == 1) & ~guard
    audit["pointer_events"] = int(np.sum((ptr == 1) & ~np.r_[False, ptr[:-1] == 1]))
    if off.sum() < frames(700., dt):
        return done("no_stable_off_reference")

    w_raw = to_numbers(df, WL)
    p_raw = np.stack([to_numbers(df, names) for names in PSD], axis=1)
    marker = df.get("psd_mean_applied")
    if marker is not None:
        m = pd.to_numeric(marker, errors="coerce")
        if not (m.eq(1).all() or m.eq(0).all()):
            return done("mixed_or_invalid_psd_encoding")
        if m.eq(1).all():
            p_raw = p_raw * BIN_COUNTS.reshape(1, 1, 5)
    valid = (np.isfinite(w_raw) & (w_raw >= 0) & (w_raw < 65525) &
             np.all(np.isfinite(p_raw) & (p_raw >= 0) & (p_raw < 65530), axis=2))
    # Firmware PSD is log10(zone sum+1)*10000. Never treat it as linear power.
    w = median_filter(np.maximum(np.nan_to_num(w_raw), 0), size=(frames(91., dt) | 1, 1), mode="nearest")
    p = np.median(np.maximum(np.nan_to_num(p_raw), 0), axis=2) / 10000.
    p = median_filter(p, size=(frames(91., dt) | 1, 1), mode="nearest")

    floor_w = np.percentile(w[off], 30, axis=0)
    floor_p = np.percentile(p[off], 30, axis=0)
    # Recorded CONNECTED controls often have PSD <0.02, whereas disconnected
    # CH1/CH2 often show high PSD. A minimum PSD threshold falsely rejects
    # legitimate quiet channels. Reject implausibly HIGH resting PSD instead.
    good_ch = (floor_w >= 65.) & (floor_w <= 3500.) & (floor_p <= .60)
    audit["good_channels"] = [ch+1 for ch in range(3) if good_ch[ch]]
    if good_ch.sum() < 2:
        return done("fewer_than_two_usable_channels")
    # Ignore uncalibrated channels, but never let clipping/invalid data look quiet.
    ratio = (w+25.) / (floor_w+25.)
    dp = p - floor_p
    active_ch = (ratio >= 1.55) & (dp >= .018) & valid & good_ch
    quiet_ch = (ratio <= 1.28) & (dp <= .028) & valid & good_ch
    active = keep_long_runs(np.any(active_ch, axis=1), frames(min_active_ms, dt))
    quiet = keep_long_runs(np.all(~good_ch | quiet_ch, axis=1), frames(180., dt))
    # A large step-to-step jump or electrode saturation may be a movement artifact.
    jump = np.any((np.abs(np.diff(np.log1p(w), axis=0, prepend=np.log1p(w[:1]))) > 1.0) & good_ch, axis=1)
    artifact = dilate(jump | (np.sum(valid & good_ch, axis=1) < 2), frames(100., dt))
    on &= ~artifact
    off &= ~artifact
    # Long press folders must have at least one genuinely supported held segment.
    # Otherwise an empty/false camera recording could create fabricated rests.
    if group in LP_GROUPS:
        starts = np.flatnonzero((ptr == 1) & ~np.r_[False, ptr[:-1] == 1])
        ends = np.flatnonzero((ptr == 1) & ~np.r_[ptr[1:] == 1, False]) + 1
        valid_events = np.zeros(n, bool)
        for a, b in zip(starts, ends):
            inner = on[a:b]
            support = active[a:b] & inner
            accepted = bool(support.sum() >= frames(min_active_ms, dt) and
                            support.sum() >= .18*max(1, inner.sum()))
            audit["events"].append({"start_s": round(a*dt/1000., 3),
                                    "end_s": round(b*dt/1000., 3),
                                    "supported_ms": round(float(support.sum()*dt), 1),
                                    "accepted": accepted})
            if accepted:
                valid_events[a:b] = True
                audit["verified_events"] += 1
        if audit["verified_events"] == 0:
            return done("no_verified_lp_event")
        on &= valid_events
    elif np.any(keep_long_runs(ptr == 1, frames(450., dt))):
        return done("control_contains_pointer_hold")

    y[on & active] = 0
    y[off & quiet] = 1
    return done("labeled" if (y >= 0).any() else "no_agreeing_regions")


def open_csv(path, mode):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode, newline="", compresslevel=3,
                         encoding="utf-8-sig" if "r" in mode else "utf-8")
    return open(path, mode, newline="", encoding="utf-8-sig" if "r" in mode else "utf-8")


def append_labels(src, dest, labels):
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".lpstate-", suffix=".gz", dir=dest.parent)
    os.close(fd)
    temp = Path(temp_path)
    try:
        with open_csv(src, "rt") as infile, open_csv(temp, "wt") as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            original = next(reader)
            if set(ADDED) & set(original):
                raise ValueError("Source already contains lp_state columns")
            writer.writerow(original + list(ADDED))
            n = 0
            for row in reader:
                if n >= len(labels):
                    raise ValueError("Source row count changed")
                s = labels.iloc[n]
                writer.writerow(row + [str(int(s.lp_state)), str(int(s.lp_state_weight)), VERSION])
                n += 1
            if n != len(labels):
                raise ValueError("Source row count changed")
        os.replace(temp, dest)
    finally:
        if temp.exists():
            temp.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=SOURCE)
    p.add_argument("--output-dir", type=Path, default=DEST)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit-per-group", type=int, default=0)
    p.add_argument("--report", type=Path, help="Optional report path (also for dry run)")
    a = p.parse_args(argv)
    root = a.input_dir.resolve()
    out = a.output_dir.resolve()
    if not root.is_dir():
        p.error("Input directory missing")
    if root == out or root in out.parents or out in root.parents:
        p.error("Input and output directories must be completely separate")
    if a.limit_per_group < 0:
        p.error("limit must be >=0")
    if a.report and (a.report.resolve() == root or root in a.report.resolve().parents):
        p.error("Report must not be written inside the input tree")
    counts = {"ok": 0, "skipped": 0, "error": 0}
    details = []
    started = time.monotonic()
    for group in GROUPS:
        files = sorted((root/group).glob("*.csv.gz"))
        if a.limit_per_group:
            files = files[:a.limit_per_group]
        for src in files:
            try:
                if root not in src.resolve().parents:
                    raise ValueError("Input path escapes root")
                dst = out / src.relative_to(root)
                if dst.exists() and not a.overwrite and not a.dry_run:
                    raise FileExistsError("Output already exists; use --overwrite for a complete fresh run")
                df = pd.read_csv(src, compression="gzip",
                                 usecols=lambda c: c in READ, low_memory=False)
                lab, stat = label_recording(df, group)
                if not a.dry_run:
                    append_labels(src, dst, lab)
                counts["ok"] += 1
                details.append({"file": src.relative_to(root).as_posix(), **stat})
            except Exception as exc:
                counts["error"] += 1
                details.append({"file": src.relative_to(root).as_posix(),
                                "error": f"{type(exc).__name__}: {exc}"[:180]})
    report = {"version": VERSION, "input": str(root), "output": str(out),
              "label_contract": {"1": "verified resting", "0": "verified active",
                                 "-1": "unknown / EXCLUDE from training"},
              "groups": GROUPS, "counts": counts, "files": details,
              "warning": "Weak labels requiring independent physical/event validation; never use -1 as 0.",
              "elapsed_s": round(time.monotonic()-started, 1)}
    report["per_group"] = {
        g: {"files": sum(d.get("group") == g for d in details),
            "rest": sum(d.get("rest", 0) for d in details if d.get("group") == g),
            "active": sum(d.get("active", 0) for d in details if d.get("group") == g),
            "unknown": sum(d.get("unknown", 0) for d in details if d.get("group") == g)}
        for g in GROUPS
    }
    report["review_files"] = [d["file"] for d in details
                              if d.get("reason") not in ("labeled",) or
                              (d.get("group") in LP_GROUPS and
                               (d.get("active", 0) == 0 or d.get("rest", 0) == 0))]
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
    print("FILE_REASONS", json.dumps({r: sum(d.get("reason") == r for d in details)
                                       for r in sorted({d.get("reason") for d in details if d.get("reason")})}))
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
