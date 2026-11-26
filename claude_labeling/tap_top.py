"""TAP top labels: teach every tap at the TOP of its contact jolt (optionally from the middle of the rise), never at
its start.

    python claude_labeling/tap_top.py DATA/DATA_TAP DATA/LABELS/DATA_TAP DATA/LABELS_TOP/DATA_TAP --mode top
        (optional export for inspection; the trainer computes the same windows in memory)

A tap in the firmware ACC fields is a short impulse: the contact jolt. Per tap, on the dominant ACC axis (the sum of
zones 1-5 of the current spectrum acc_psd[0], log1p, 3-row centred mean, minus the rest level before the tap):

    start   the rise up to 50 % of the jolt's height           never a positive
    middle  the rise from 50 % to 80 %                         positive in mode "mid" only (measured worse)
    top     the peak: from 10 ms before it, while the jolt     positive (mode "top", the default)
            stays >= 80 % of its height (<= 45 ms after it)
    fall    after the jolt has fallen below 80 %               never a positive

Measured (research/gesture/TAP_TOP_LABELS.md, 3 seeds): the closer the taught rows are to the peak, the surer the model
is at real taps. Middle + top < top from the 80 % rise < top from 10 ms before the peak (the default).

Measured on 4,419 confirmed taps (relative to the labeller's onset t_on, p10 / p50 / p90): rise start -127 / -55 / -28
ms, middle (50 %) -27 / -13 / -2 ms, 80 % on the rise -3 / +8 / +22 ms, peak +23 / +40 / +63 ms, 80 % on the fall
+49 / +75 / +106 ms. The labeller's fixed [t_on, t_on + 80] window therefore starts on the rise for many taps and
misses the top of late ones; press-anchored windows (pointer time) land anywhere from the lead-in to the fall.

Which taps: exactly the taps of the training target (each tap = the onset its positive window was written from), so the
tap set (confirmed taps, press2 presses, soft taps) is unchanged; only WHERE each tap is taught moves. The rows of the old
window that are not in the new one become unknown (-1), never "not a tap". Rows the labeller set to 0 stay 0. When no
jolt is found (no rise of at least MIN_HEIGHT above rest within the search window), the tap keeps its window minus
the first FALLBACK_SKIP_MS (its start), and is counted.

Deterministic signal processing on the logged firmware fields only: no model, no folds, no person or file split.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np

VERSION = "tap_top_v2"
ACC_NOW = tuple(f"p_core->acc_psd[0][{a}][{z}]" for a in "XYZ" for z in range(6))
MODES = ("top", "mid")

# Anatomy (ms unless noted). Chosen from the measured anatomy above; every one is a named constant.
SEARCH_BEFORE_MS = 40        # the peak is searched from (anchor - 40) ...
SEARCH_AFTER_MS = 70         # ... to (old window end + 70): [t_on - 40, t_on + 150] for a labeller window
NEIGHBOR_GUARD_MS = 40       # search for a tap's peak only up to 40 ms before the next tap's anchor (onset)
PREVIOUS_GUARD_MS = 20       # windows end 20 ms before the next anchor and start 20 ms after the previous window
BASE_FROM_MS = 300           # rest level: 25th percentile of the envelope over [anchor - 300, anchor - 40]
BASE_TO_MS = 40
BASE_MIN_ROWS = 5
BASE_PCT = 25
SMOOTH_ROWS = 3              # centred moving average of the envelope (offline: 1 row of future)
MIN_HEIGHT = 1.0             # log1p units: the jolt rises >= e x the rest energy (confirmed taps: p10 4.3)
EDGE_GUARD_MS = 6            # a peak on the edge of the search window is not a peak
TOP_FRAC = 0.80              # top: envelope >= 80 % of the height
MID_FRAC = 0.50              # middle: the rise from 50 % of the height
WALK_LIMIT_MS = 200          # crossings are searched at most this far from the peak
CLAMP_BEFORE_MS = dict(top=10, mid=60)   # positives start at most this long before the peak (top: the rise is not
CLAMP_AFTER_MS = 45                      # taught, measured best) and end at most this long after it
MIN_WIDTH_MS = 25            # a top narrower than this is widened around the peak (inside the 50 % crossings)
FALLBACK_SKIP_MS = 10        # no jolt found: the run minus its first 10 ms
RUN_GAP_MS = 30              # positive rows closer than this belong to the same tap (10 ms sampling stride + jitter)
OWN_MS = (-10.0, 100.0)      # a positive row belongs to the tap anchor (onset) at most 10 ms after / 100 ms before it


def jolt_envelope(acc18) -> np.ndarray:
    """(n, 18) acc_psd[0] codes in X0..X5, Y0..Y5, Z0..Z5 order -> (n, 3) log1p(zones 1-5) per axis, smoothed."""
    a = np.clip(np.asarray(acc18, np.float64), 0, None).reshape(-1, 3, 6)[:, :, 1:].sum(2)
    e = np.log1p(a)
    if SMOOTH_ROWS > 1 and len(e) >= SMOOTH_ROWS:
        k = np.ones(SMOOTH_ROWS) / SMOOTH_ROWS
        pad = SMOOTH_ROWS // 2
        ep = np.pad(e, ((pad, pad), (0, 0)), mode="edge")
        e = np.stack([np.convolve(ep[:, c], k, mode="valid") for c in range(3)], 1)
    return e


def find_runs(t, pos, gap_ms: float = RUN_GAP_MS):
    """Index ranges [i0, i1] (inclusive) of positive rows; rows closer than gap_ms in time are one run."""
    idx = np.flatnonzero(np.asarray(pos, bool))
    if not len(idx):
        return []
    t = np.asarray(t, np.float64)
    cut = np.flatnonzero(np.diff(t[idx]) > gap_ms)
    starts = np.r_[0, cut + 1]; ends = np.r_[cut, len(idx) - 1]
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, ends)]


def tap_anchors(t, pos, anchors=None, own=OWN_MS):
    """Tap anchors (ms) of one recording and the anchor that owns every positive row.
    anchors: the onsets the positive windows were written from (labeller onsets, press2 / soft anchors). A positive row
    at r belongs to the latest anchor a with own[0] <= r - a <= own[1]. Positive rows that no given anchor owns form
    runs (gap > RUN_GAP_MS splits; a run starting within OWN_MS[1] of the previous run's start continues it, so a row
    gap in the recording does not split a tap); each such run's first row is an anchor too.
    Returns (anchors sorted, owner index per row or -1)."""
    t = np.asarray(t, np.float64); pos = np.asarray(pos, bool)
    a = np.unique(np.asarray([x for x in (anchors if anchors is not None else []) if x == x], np.float64))
    owner = np.full(len(t), -1, np.int64)
    pi = np.flatnonzero(pos)
    if len(a) and len(pi):
        k = np.searchsorted(a, t[pi] - own[0], side="right") - 1
        d = t[pi] - a[np.clip(k, 0, None)]
        ok = (k >= 0) & (d >= own[0]) & (d <= own[1])
        owner[pi[ok]] = k[ok]
    free = pos & (owner < 0)
    extra = []
    for i0, i1 in find_runs(t, free):
        if extra and t[i0] - extra[-1] <= own[1]:
            continue
        extra.append(float(t[i0]))
    if extra:
        a_all = np.unique(np.r_[a, np.asarray(extra)])
        return tap_anchors(t, pos, a_all, own)
    used = np.unique(owner[owner >= 0])            # anchors that own no positive row are not taps here
    if not len(used):
        return np.zeros(0, np.float64), np.full(len(t), -1, np.int64)
    remap = np.full(len(a), -1, np.int64); remap[used] = np.arange(len(used))
    owner = np.where(owner >= 0, remap[np.clip(owner, 0, None)], -1)
    return a[used], owner


def _walk(s, k, frac_h, step, t, limit_ms):
    """From index k, walk in direction step while s >= frac_h (within limit_ms); return the last index inside."""
    j = k
    while 0 <= j + step < len(s) and s[j + step] >= frac_h and abs(t[j + step] - t[k]) <= limit_ms:
        j += step
    return j


def anatomy(t, env, lo_ms, hi_ms, base_lo_ms, base_hi_ms, top_frac=TOP_FRAC):
    """The jolt inside [lo_ms, hi_ms] against the rest level in [base_lo_ms, base_hi_ms]. Returns a dict (ok False
    when there is no jolt of at least MIN_HEIGHT whose peak is inside the window). t_r80 / t_f80 are the crossings of
    top_frac (default 80 %) of the height on the rise / fall, t_r50 / t_f50 those of 50 %."""
    t = np.asarray(t, np.float64)
    sw = np.flatnonzero((t >= lo_ms) & (t <= hi_ms))
    out = dict(ok=False, reason="", height=0.0, axis=-1)
    if len(sw) < 3:
        out["reason"] = "short_window"; return out
    base = np.flatnonzero((t >= base_lo_ms) & (t <= base_hi_ms))
    if len(base) < BASE_MIN_ROWS:                          # recording start / dense taps: the quietest rows before
        base = np.flatnonzero((t >= base_lo_ms - 400) & (t < lo_ms))
    if len(base) < 2:
        out["reason"] = "no_rest_level"; return out
    b = np.percentile(env[base], BASE_PCT, axis=0)
    rise = env[sw].max(0) - b
    ax = int(np.argmax(rise)); h = float(rise[ax])
    s = env[:, ax] - b[ax]
    k = int(sw[np.argmax(s[sw])])
    out.update(axis=ax, height=h, t_top=float(t[k]), i_top=k)
    if h < MIN_HEIGHT:
        out["reason"] = "no_jolt"; return out
    if t[k] - lo_ms < EDGE_GUARD_MS or hi_ms - t[k] < EDGE_GUARD_MS:
        out["reason"] = "peak_on_edge"; return out
    i_r80 = _walk(s, k, top_frac * h, -1, t, WALK_LIMIT_MS); i_f80 = _walk(s, k, top_frac * h, +1, t, WALK_LIMIT_MS)
    i_r50 = _walk(s, k, MID_FRAC * h, -1, t, WALK_LIMIT_MS); i_f50 = _walk(s, k, MID_FRAC * h, +1, t, WALK_LIMIT_MS)
    i_r10 = _walk(s, k, 0.10 * h, -1, t, WALK_LIMIT_MS)
    out.update(ok=True, t_r10=float(t[i_r10]), t_r50=float(t[i_r50]), t_r80=float(t[i_r80]),
               t_f80=float(t[i_f80]), t_f50=float(t[i_f50]))
    return out


def params(mode, **over):
    """Window parameters of a mode (research overrides: frac, before, after, end50)."""
    p = dict(frac=TOP_FRAC, before=CLAMP_BEFORE_MS[mode], after=CLAMP_AFTER_MS, end50=False)
    unknown = set(over) - set(p)
    if unknown:
        raise ValueError(f"unknown window parameters {sorted(unknown)}")
    p.update(over)
    return p


def window(an, mode, prm=None):
    """Positive window (start_ms, end_ms) of one tap from its anatomy, or None. Never before the 50 % crossing of the
    rise (mode 'mid') or the top-fraction crossing (mode 'top')."""
    if not an.get("ok"):
        return None
    prm = prm or params(mode)
    top = an["t_top"]
    floor = an["t_r50"] if mode == "mid" else an["t_r80"]
    end = an["t_f50"] if prm["end50"] else an["t_f80"]
    a = max(floor, top - prm["before"]); b = min(end, top + prm["after"])
    if b - a < MIN_WIDTH_MS:                   # a very sharp jolt: widen around the peak inside the 50 % crossings
        a = max(an["t_r50"], min(a, top - MIN_WIDTH_MS / 2.0))
        b = min(an["t_f50"], max(b, top + MIN_WIDTH_MS / 2.0))
    return float(a), float(b)


def tap_windows(t, acc18, anchors, mode, spans=None, prm=None):
    """One window per tap anchor (ms, sorted). spans: optional (first, last) ms of each tap's old positive rows (the
    search covers them). prm: window parameters (params()). Returns a list of dicts: start_ms, end_ms, anchor_ms,
    fallback, anatomy fields."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    prm = prm or params(mode)
    t = np.asarray(t, np.float64)
    env = jolt_envelope(acc18)
    out = []; prev_end = -np.inf
    for n, a0 in enumerate(anchors):
        s0, s1 = spans[n] if spans is not None else (a0, a0 + 80.0)
        prev_a = anchors[n - 1] if n else -np.inf
        next_a = anchors[n + 1] if n + 1 < len(anchors) else np.inf
        lo = max(min(a0, s0) - SEARCH_BEFORE_MS, prev_a + NEIGHBOR_GUARD_MS)
        hi = min(max(a0 + 80.0, s1) + SEARCH_AFTER_MS, next_a - NEIGHBOR_GUARD_MS)
        an = anatomy(t, env, lo, hi, max(a0 - BASE_FROM_MS, prev_a + NEIGHBOR_GUARD_MS), a0 - BASE_TO_MS, prm["frac"])
        w = window(an, mode, prm)
        rec = dict(anchor_ms=float(a0), old_start_ms=float(s0), old_end_ms=float(s1), fallback=w is None, mode=mode,
                   **{k: v for k, v in an.items() if k != "i_top"})
        if w is None:                          # no jolt: the old window without its start
            w = (float(s0) + FALLBACK_SKIP_MS, float(s1))
        lo_w = max(w[0], prev_end + PREVIOUS_GUARD_MS)
        hi_w = min(w[1], next_a - PREVIOUS_GUARD_MS)
        rec.update(start_ms=float(lo_w), end_ms=float(hi_w))
        prev_end = max(prev_end, hi_w)
        out.append(rec)
    return out


def relabel(t, acc18, y, mode, anchors=None, soft=None, prm=None):
    """Training target of one recording with every tap moved to its top window.
    y: 1 tap rows, 0 not a tap, -1 unknown. anchors: the tap onsets its positive windows were written from (optional;
    see tap_anchors). soft: optional bool mask of positive rows of soft / added taps (a new window inherits it when any
    row of its old window had it). Returns (y2, soft2, windows, counts)."""
    t = np.asarray(t, np.float64); y = np.asarray(y)
    y2 = np.array(y, np.int8, copy=True)
    old_pos = y == 1
    anc, owner = tap_anchors(t, old_pos, anchors)
    spans = [(float(t[owner == k].min()), float(t[owner == k].max())) for k in range(len(anc))]
    wins = tap_windows(t, acc18, anc, mode, spans, prm)
    soft2 = None if soft is None else np.zeros(len(y), bool)
    info = Counter(taps=len(anc))
    y2[old_pos] = -1                           # every old positive row: unknown unless inside a new window
    for k, w in enumerate(wins):
        m = (t >= w["start_ms"]) & (t <= w["end_ms"]) & (y != 0)       # the labeller's 0 rows stay 0
        if not m.any():                        # a window between rows: keep the row nearest to its centre
            c = 0.5 * (w["start_ms"] + w["end_ms"]); j = int(np.argmin(np.abs(t - c)))
            m = np.zeros(len(t), bool)
            if y[j] != 0:
                m[j] = True
        y2[m] = 1
        if soft2 is not None and np.asarray(soft)[owner == k].any():
            soft2 |= m
        info["fallback" if w["fallback"] else "jolt"] += 1
        info["rows_new"] += int(m.sum())
    info["rows_old"] = int(old_pos.sum())
    info["rows_moved_to_unknown"] = int((old_pos & (y2 == -1)).sum())
    info["rows_unknown_to_tap"] = int((~old_pos & (y2 == 1)).sum())
    return y2, soft2, wins, info


# ------------------------------------------------------------------------------------------------ export (inspection)
def _export(raw_root, label_root, out_root, mode, limit=None):
    """Write <recording>.labels.csv.gz sidecars with tap_label replaced by the top windows (the labeller's confirmed
    taps; press2 and soft taps are trainer policies and are not part of this export) and _top_events.csv.gz."""
    import glob
    import pandas as pd
    files = sorted(glob.glob(os.path.join(label_root, "**", "*.labels.csv.gz"), recursive=True))
    files = [f for f in files if os.path.basename(os.path.dirname(f)) in ("TAP", "SWIPE")]
    if limit:
        files = files[:limit]
    os.makedirs(out_root, exist_ok=True)
    ev = pd.read_csv(os.path.join(label_root, "_events.csv.gz"), low_memory=False)
    ev = ev[(ev.type == "tap_press") & (ev.labeled == 1)]
    onsets = {r: g.t_on.to_numpy(float) for r, g in ev.groupby("recording")}
    rows, tot = [], Counter()
    for k, lf in enumerate(files):
        rel = os.path.relpath(lf, label_root)
        src = os.path.join(raw_root, re.sub(r"\.labels\.csv\.gz$", ".csv.gz", rel))
        lab = pd.read_csv(lf)
        df = pd.read_csv(src, usecols=["Timestamp", *ACC_NOW])
        if len(df) != len(lab) or not np.array_equal(df.Timestamp.to_numpy(), lab.Timestamp.to_numpy()):
            raise ValueError(f"Row mismatch: {rel}")
        t = (df.Timestamp - df.Timestamp.iloc[0]).to_numpy(np.float64)
        key = re.sub(r"\.labels\.csv\.gz$", ".csv.gz", rel)
        y2, _, wins, info = relabel(t, df[list(ACC_NOW)].to_numpy(), lab.tap_label.to_numpy(), mode,
                                    anchors=onsets.get(key))
        lab["tap_label_onset"] = lab.tap_label; lab["tap_label"] = y2
        dst = os.path.join(out_root, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        lab.to_csv(dst, index=False)
        for w in wins:
            rows.append(dict(recording=re.sub(r"\.labels\.csv\.gz$", ".csv.gz", rel), **w))
        tot.update(info)
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(files)}", flush=True)
    pd.DataFrame(rows).round(2).to_csv(os.path.join(out_root, "_top_events.csv.gz"), index=False)
    summary = dict(version=VERSION, mode=mode, recordings=len(files), **{k: int(v) for k, v in tot.items()})
    json.dump(summary, open(os.path.join(out_root, "_top_summary.json"), "w"), indent=1)
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw_root", help="recordings root, e.g. DATA/DATA_TAP")
    ap.add_argument("label_root", help="the labeller's output for it, e.g. DATA/LABELS/DATA_TAP")
    ap.add_argument("out_root", help="new folder for the top-label sidecars")
    ap.add_argument("--mode", default="top", choices=MODES)
    ap.add_argument("--limit", type=int, default=None, help="first N TAP/SWIPE recordings only")
    a = ap.parse_args()
    if os.path.abspath(a.out_root) in (os.path.abspath(a.label_root), os.path.abspath(a.raw_root)):
        ap.error("out_root must be a new folder")
    _export(a.raw_root, a.label_root, a.out_root, a.mode, a.limit)


if __name__ == "__main__":
    sys.exit(main())
