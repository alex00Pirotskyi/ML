#!/usr/bin/env python3
"""Check the firmware replay rules and a model against INDEPENDENT event logs.

The trainer scores models with replay.py: rising edges of TAP_SWIPE_predict with a refractory, an onset hit
window, and the LONG_PRESS ten-sample Q20 release. Those rules are assumptions until checked against what
the device does, and every training metric is graded against offline (teacher) labels. This tool closes
both gaps as soon as independent logs exist:

1. Firmware TAP log (--device-column): recordings in which the device logged its OWN tap decision (a 0/1
   column, rising edge = tap) while a known header ran. The header is replayed on the same rows; simulated
   and logged taps are matched. Reports agreement, timing offset, and the refractory that best explains the
   device. Agreement near 100% at the configured refractory = the replay rules are right.
2. Firmware LP log (--task lp --device-column HELD): a 0/1 column that is 1 while the device holds a press.
   For each logged press, the release time simulated by replay.lp_declared_release (Q20, 10-sample mean,
   release threshold) is compared with the logged release.
3. Gold annotations (--gold gold.csv): independently timed taps (synchronised video, touch surface), columns
   `recording,t_ms` (ms from the recording's first row) or `recording,timestamp` (absolute, same clock as the
   Timestamp column). Scores the header: recall, latency and false events per minute (all events that match
   no gold tap; use recordings that are fully annotated).

    python research/gesture/validate_events.py --header models/TAP_SWIPE.h \
        --recordings 'DATA/DATA_TAP_LEAN8/TAP/*.csv.gz' --device-column FW_TAP
    python research/gesture/validate_events.py --task lp --header models/LONG_PRESS.h \
        --recordings 'DATA/DATA_LONG_PRESS_LEAN8/MID_LONG_PRESS/*.csv.gz' --device-column FW_HELD
    python research/gesture/validate_events.py --header models/TAP_SWIPE.h --recordings '...' --gold gold.csv

The header's feature schema (<stem>_feature_schema.json next to it) gives the input columns. Recordings must
contain them (use the *_LEAN8 trees when the model uses Lean-8 fields).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay as R  # noqa: E402

REFRACTORY_GRID = tuple(range(0, 501, 10))


def schema_columns(header: Path) -> list[str]:
    stem = header.stem
    f = header.with_name(stem + "_feature_schema.json")
    fm = json.loads(f.read_text())["feature_map"]
    return [fm[k] for k in sorted(fm, key=int)]


def rising(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(bool)
    return np.flatnonzero(x & ~np.r_[False, x[:-1]])


def tap_events(decision: np.ndarray, t: np.ndarray, refractory: float) -> np.ndarray:
    kept, last = [], -1e18
    for r in rising(decision):
        if t[r] - last >= refractory:
            kept.append(t[r]); last = t[r]
    return np.asarray(kept, float)


def match(a: np.ndarray, b: np.ndarray, before: float, after: float):
    """Greedy one-to-one: each b (reference) takes the first unused a inside [b - before, b + after]."""
    used = np.zeros(len(a), bool); pairs = []
    for j, x in enumerate(b):
        cand = np.flatnonzero(~used & (a >= x - before) & (a <= x + after))
        if len(cand):
            used[cand[0]] = True; pairs.append((cand[0], j))
    return pairs


def load(path: Path, cols, extra):
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in set(cols) | set(extra) | {"Timestamp"})
    t = (df["Timestamp"].to_numpy(np.int64) - int(df["Timestamp"].iloc[0])).astype(float)
    return df, t


def firmware_tap(hm, cols, files, column, tol):
    sim_n = dev_n = matched = 0; offsets = []; per_ref = {r: [0, 0, 0] for r in REFRACTORY_GRID}
    min_gap = []
    for f in files:
        df, t = load(f, cols, [column])
        d = hm.decision(np.ascontiguousarray(df[cols].to_numpy(np.float32))).astype(bool)
        dev = t[rising(df[column].to_numpy() > 0)]
        if len(dev) > 1:
            min_gap.append(float(np.diff(dev).min()))
        sim = tap_events(d, t, R.TAP_REFRACTORY_MS)
        pairs = match(sim, dev, tol, tol)
        sim_n += len(sim); dev_n += len(dev); matched += len(pairs)
        offsets += [dev[j] - sim[i] for i, j in pairs]
        for r in REFRACTORY_GRID:
            s = tap_events(d, t, r); m = len(match(s, dev, tol, tol))
            per_ref[r][0] += len(s); per_ref[r][1] += len(dev); per_ref[r][2] += m
    f1 = {r: (2 * m / (s + d) if (s + d) else float("nan")) for r, (s, d, m) in per_ref.items()}
    best = max(f1, key=lambda r: (f1[r], -abs(r - R.TAP_REFRACTORY_MS)))
    return dict(mode="firmware_tap", recordings=len(files), device_taps=dev_n, simulated_taps=sim_n, matched=matched,
                agreement_f1=(2 * matched / (sim_n + dev_n) if (sim_n + dev_n) else None),
                offset_ms_median=float(np.median(offsets)) if offsets else None,
                offset_ms_p5_p95=[float(np.percentile(offsets, 5)), float(np.percentile(offsets, 95))] if offsets else None,
                replay_refractory_ms=R.TAP_REFRACTORY_MS, best_fit_refractory_ms=best, best_fit_f1=f1[best],
                device_min_interval_ms=min(min_gap) if min_gap else None,
                verdict=("replay rules agree with the device" if f1[R.TAP_REFRACTORY_MS] >= 0.98 else
                         f"replay rules disagree: set TAP_REFRACTORY_MS={best} and re-check the offset"))


def firmware_lp(hm, cols, files, column, threshold, tol):
    n = agree = 0; deltas = []
    for f in files:
        df, t = load(f, cols, [column])
        held = df[column].to_numpy() > 0
        q = hm.value_q20(np.ascontiguousarray(df[cols].to_numpy(np.float32)))
        starts = rising(held); ends = rising(~held)
        for s in starts:
            e = ends[ends > s]
            if not len(e):
                continue
            sim = R.lp_declared_release(q[s:], t[s:], t[s], threshold)
            if sim is None:
                continue
            n += 1; dlt = t[e[0]] - sim; deltas.append(dlt); agree += abs(dlt) <= tol
    return dict(mode="firmware_lp", presses=n, release_within_tol=(agree / n if n else None), tolerance_ms=tol,
                device_minus_simulated_ms_median=float(np.median(deltas)) if deltas else None,
                p5_p95=[float(np.percentile(deltas, 5)), float(np.percentile(deltas, 95))] if deltas else None,
                release_threshold=threshold, average_samples=R.LP_AVERAGE_SAMPLES,
                verdict=("LP replay agrees with the device" if n and agree / n >= 0.95 else
                         "LP replay disagrees: check the release threshold, Q20 averaging and sample rate"))


def gold_tap(hm, cols, files, gold: pd.DataFrame):
    by = {}
    for r in gold.itertuples():
        by.setdefault(Path(str(r.recording)).name.replace(".csv.gz", "").replace(".csv", ""), []).append(r)
    taps = hits = false = 0; minutes = 0.0; lat = []
    per = []
    for f in files:
        key = f.name.replace(".csv.gz", "").replace(".csv", "")
        if key not in by:
            continue
        df, t = load(f, cols, [])
        t0 = int(df["Timestamp"].iloc[0])
        ref = np.sort([float(r.t_ms) if hasattr(r, "t_ms") and pd.notna(getattr(r, "t_ms", np.nan))
                       else float(r.timestamp) - t0 for r in by[key]])
        d = hm.decision(np.ascontiguousarray(df[cols].to_numpy(np.float32))).astype(bool)
        sim = tap_events(d, t, R.TAP_REFRACTORY_MS)
        pairs = match(sim, ref, R.TAP_HIT_BEFORE_MS, R.TAP_HIT_AFTER_MS)
        taps += len(ref); hits += len(pairs); false += len(sim) - len(pairs); minutes += (t[-1] - t[0]) / 60000.0
        lat += [sim[i] - ref[j] for i, j in pairs]
        per.append(dict(recording=key, gold_taps=len(ref), hits=len(pairs), false_events=len(sim) - len(pairs)))
    return dict(mode="gold_tap", recordings=len(per), gold_taps=taps, recall=(hits / taps if taps else None),
                false_events=false, false_per_min=(false / minutes if minutes else None),
                latency_ms_median=float(np.median(lat)) if lat else None, per_recording=per)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--header", required=True, type=Path)
    ap.add_argument("--recordings", required=True, nargs="+", help="files or glob patterns")
    ap.add_argument("--task", choices=("tap", "lp"), default="tap")
    ap.add_argument("--device-column", help="device-logged decision column (TAP: tap flag; LP: held flag)")
    ap.add_argument("--gold", type=Path, help="gold annotation CSV (recording,t_ms | recording,timestamp)")
    ap.add_argument("--tolerance-ms", type=float, default=30.0)
    ap.add_argument("--lp-threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    files = sorted({Path(p) for pat in a.recordings for p in (glob.glob(pat) or [pat]) if Path(p).is_file()})
    if not files:
        ap.error("no recordings found")
    cols = schema_columns(a.header)
    hm = R.HeaderModel(a.header, cols, Path(tempfile.gettempdir()) / ("validate_" + a.header.stem.lower()))
    if a.gold is not None:
        out = gold_tap(hm, cols, files, pd.read_csv(a.gold))
    elif a.device_column and a.task == "tap":
        out = firmware_tap(hm, cols, files, a.device_column, a.tolerance_ms)
    elif a.device_column:
        out = firmware_lp(hm, cols, files, a.device_column, a.lp_threshold, max(a.tolerance_ms, 30.0))
    else:
        ap.error("give --device-column (firmware log) or --gold (annotations)")
    text = json.dumps(out, indent=1, default=float)
    if a.out:
        a.out.write_text(text)
    print(text)
    return out


if __name__ == "__main__":
    main()
