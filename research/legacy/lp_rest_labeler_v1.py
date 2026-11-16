#!/usr/bin/env python3
"""Generate conservative pseudo-labels for 3-channel sEMG activity and contact.

Nine regression targets: lp_resting_ch1..3, lp_high_activity_ch1..3,
lp_disconnection_ch1..3. Binary lp_state = 1 only when all channels appear
connected and confidently resting; 0 means activity, contact fault, OR
uncertainty. lp_unknown_ch1..3 and lp_state_weight encode that uncertainty.

No PointerState, TriggerPressure, class_id, subject, or directory labels are
ever used as truth. This is an OFFLINE heuristic teacher, not clinical ground
truth or a validated classifier. Output is a separate directory; input bytes
are never modified.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

VERSION = "lp_rest_teacher_v1"
DEFAULT_INPUT = Path("DATA/DATA_LONG_PRESS")
DEFAULT_OUTPUT = Path("DATA/DATA_LONG_PRESS_LABELED")
CHANNELS = ("CH1", "CH2", "CH3")
WL_COLUMNS = tuple(f"p_core->wl[{ch}]" for ch in CHANNELS)
PSD_COLUMNS = tuple(
    tuple(f"p_core->emg_psd_mean[{ch}][{z}]" for z in range(5))
    for ch in CHANNELS
)
OUTPUT_CHANNEL_COLUMNS = tuple(
    f"lp_{kind}_ch{ch}"
    for ch in (1, 2, 3)
    for kind in ("resting", "high_activity", "disconnection", "unknown")
)
OUTPUT_COLUMNS = OUTPUT_CHANNEL_COLUMNS + ("lp_state", "lp_state_weight", "lp_teacher_version")

# Exploratory, dataset-specific hypotheses; NOT calibrated probabilities.
ACTIVITY_LOW_WL = 1.08
ACTIVITY_HIGH_WL = 2.00
ACTIVITY_LOW_PSD = 1.08
ACTIVITY_HIGH_PSD = 2.10
STATIC_NOISE_LOW_WL = 3000.0
STATIC_NOISE_HIGH_WL = 6500.0
STATIC_MAX_DYN = 1.85
STATIC_MIN_DYN = 1.15
REST_SCORE_TO_ENTER = 0.78
REST_MIN_MS = 180.0


def smoothstep(x: np.ndarray | float, low: float, high: float) -> np.ndarray:
    """Smooth monotonic ramp [0, 1] without an expensive sigmoid."""
    z = np.clip((np.asarray(x, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
    return z * z * (3 - 2 * z)


def positive_float_column(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)


def median_step_ms(df: pd.DataFrame) -> float:
    """Use monotonic microseconds when present, otherwise timestamp ms."""
    for name, factor in (("MonotonicTimeUs", 0.001), ("Timestamp", 1.0)):
        if name not in df.columns:
            continue
        a = positive_float_column(df, name)
        d = np.diff(a[: min(len(a), 2000)]) * factor
        d = d[np.isfinite(d) & (d >= 0.25) & (d <= 100.0)]
        if len(d) >= 10:
            return float(np.median(d))
    return 6.0


def trailing_rolling_median(values: np.ndarray, n: int) -> np.ndarray:
    return pd.DataFrame(values).rolling(window=max(1, n), min_periods=1).median().to_numpy()


def trailing_quantile(values: np.ndarray, n: int, quantile: float) -> np.ndarray:
    return pd.DataFrame(values).rolling(window=max(1, n), min_periods=1).quantile(quantile).to_numpy()


def stable_prefix_run(mask: np.ndarray, minimum: int) -> np.ndarray:
    """True where mask holds continuously for >= minimum observations."""
    idx = np.arange(mask.size, dtype=np.int64)
    last_false = np.maximum.accumulate(np.where(~mask, idx, -1))
    return mask & ((idx - last_false) >= minimum)


def compute_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute OFFLINE pseudo-targets for one recording.

    Uses whole-recording 15th percentiles as estimated floors. These are
    NONCAUSAL training labels, not firmware-ready real-time decisions.
    """
    if not len(df):
        raise ValueError("Empty recording")
    for name in WL_COLUMNS:
        if name not in df.columns:
            raise ValueError(f"Missing required feature {name}")
    for name in OUTPUT_COLUMNS:
        if name in df.columns:
            raise ValueError(f"Input already has label column {name}; use raw input")

    nrows = len(df)
    step_ms = median_step_ms(df)
    fast_window = max(4, min(250, int(round(180.0 / step_ms))))
    slow_window = max(fast_window + 1, min(800, int(round(2200.0 / step_ms))))

    raw_wl = np.column_stack([positive_float_column(df, c) for c in WL_COLUMNS])
    valid_wl = np.isfinite(raw_wl) & (raw_wl >= 0)
    wl = np.maximum(np.nan_to_num(raw_wl, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    psd_ok = np.ones((nrows, 3), dtype=bool)
    psd = np.zeros((nrows, 3), dtype=np.float64)
    psd_available = np.ones(3, dtype=bool)
    for i, names in enumerate(PSD_COLUMNS):
        if not all(c in df for c in names):
            psd_available[i] = False
            psd_ok[:, i] = False
            continue
        psd_raw = np.column_stack([positive_float_column(df, c) for c in names])
        psd_ok[:, i] = np.all(np.isfinite(psd_raw) & (psd_raw >= 0), axis=1)
        # No zone division: old files lack a trustworthy normalization flag.
        # Ratios within one file work with either consistent PSD convention.
        psd[:, i] = np.sum(np.maximum(np.nan_to_num(psd_raw, nan=0.0), 0), axis=1)

    fast_wl = trailing_rolling_median(wl, fast_window)
    slow_wl = trailing_rolling_median(fast_wl, slow_window)
    pfast = trailing_rolling_median(psd, fast_window)

    floor_wl = np.percentile(fast_wl, 15, axis=0)
    floor_psd = np.percentile(pfast, 15, axis=0)
    global_med = np.median(fast_wl, axis=0)
    global_dyn = (np.percentile(fast_wl, 90, axis=0) + 1) / (
        np.percentile(fast_wl, 10, axis=0) + 1
    )

    ratio_wl = (fast_wl + 25) / (floor_wl[None, :] + 25)
    ratio_psd = (pfast + 25) / (floor_psd[None, :] + 25)
    a_wl = smoothstep(
        np.log(np.maximum(ratio_wl, 1e-8)),
        math.log(ACTIVITY_LOW_WL), math.log(ACTIVITY_HIGH_WL),
    )
    a_psd = smoothstep(
        np.log(np.maximum(ratio_psd, 1e-8)),
        math.log(ACTIVITY_LOW_PSD), math.log(ACTIVITY_HIGH_PSD),
    )
    # Correlated signal representations: no double-vote treatment.
    a = 0.75 * a_wl + 0.25 * a_psd
    a = np.where(psd_available[None, :], a, a_wl)

    # Only high AND stationary noise is strong contact-fault evidence.
    slow_lo = trailing_quantile(fast_wl, slow_window, 0.1)
    slow_hi = trailing_quantile(fast_wl, slow_window, 0.9)
    local_dyn = (slow_hi + 1) / (slow_lo + 1)
    high_level = smoothstep(slow_wl, STATIC_NOISE_LOW_WL, STATIC_NOISE_HIGH_WL)
    low_modulation = 1 - smoothstep(local_dyn, STATIC_MIN_DYN, STATIC_MAX_DYN)
    session_stability = 1 - smoothstep(global_dyn, 1.50, 2.40)
    # A strong contraction plateau has an elevated baseline but often
    # detectable activity elsewhere. Session-level modulation reduces the
    # chance of misclassifying that plateau as disconnection.
    disconnected = 0.98 * high_level * low_modulation * (
        0.20 + 0.80 * session_stability[None, :]
    )
    hard_flatline = (slow_wl < 3.0) & (
        trailing_rolling_median(psd, slow_window) < 3.0
    )
    disconnected = np.where(hard_flatline, np.maximum(disconnected, 0.95), disconnected)

    # A quiet but nonzero channel is UNKNOWN, not necessarily disconnected.
    # This specifically avoids assuming all channels share a file's contact
    # label (CH3 may remain physically attached in DISCONNECTED recordings).
    silence = 1 - smoothstep(slow_wl, 85.0, 250.0)
    no_reference = (1 - smoothstep(global_dyn, 1.15, 1.75)) * smoothstep(
        global_med, 1700.0, 5500.0
    )
    unknown = np.maximum(0.8 * silence, 0.90 * no_reference[None, :])
    unknown = np.where(valid_wl, unknown, 1.0)
    unknown = np.where(psd_available[None, :], unknown, np.maximum(unknown, 0.2))
    unknown = np.where(psd_ok | (~psd_available[None, :]), unknown, 1.0)

    # Outputs form a conservative four-way soft partition. The user-requested
    # three regression targets are the first three; unknown is an extra
    # abstention target to avoid inventing false rest/activity labels.
    disconnected = np.clip(disconnected, 0.0, 1.0)
    unknown = np.clip(unknown, 0.0, 1.0) * (1 - disconnected)
    trust = np.maximum(0.0, 1.0 - disconnected - unknown)
    resting = trust * (1.0 - a)
    activity = trust * a

    min_rest = resting.min(axis=1)
    max_disc = disconnected.max(axis=1)
    max_unknown = unknown.max(axis=1)
    rest_candidate = (
        (min_rest >= REST_SCORE_TO_ENTER)
        & (max_disc < 0.35)
        & (max_unknown < 0.22)
    )
    state = stable_prefix_run(
        rest_candidate, max(1, int(round(REST_MIN_MS / step_ms)))
    )
    # All UNCERTAIN situations are binary zero, by request. That does NOT
    # imply those rows are verified ACTIVE. Train with lp_state_weight.
    certainty = np.clip(1.0 - np.max(unknown, axis=1), 0.0, 1.0)
    ramp = np.clip(
        np.arange(nrows, dtype=np.float64) / max(1, slow_window - 1), 0.1, 1.0
    )
    clear_nonrest = np.maximum(np.max(activity, axis=1), max_disc)
    binary_margin = np.where(state, min_rest, clear_nonrest)
    weight = certainty * (
        0.3 + 0.7 * smoothstep(binary_margin, 0.25, 0.8)
    ) * ramp
    weight = np.where(
        (~state) & (max_unknown > 0.45), weight * 0.15, weight
    )

    labels: dict[str, np.ndarray] = {}
    for ch in range(3):
        ci = ch + 1
        labels[f"lp_resting_ch{ci}"] = resting[:, ch]
        labels[f"lp_high_activity_ch{ci}"] = activity[:, ch]
        labels[f"lp_disconnection_ch{ci}"] = disconnected[:, ch]
        labels[f"lp_unknown_ch{ci}"] = unknown[:, ch]
    labels["lp_state"] = state.astype(np.uint8)
    labels["lp_state_weight"] = weight
    result = pd.DataFrame(
        labels, columns=[x for x in OUTPUT_COLUMNS if x != "lp_teacher_version"]
    )
    result["lp_teacher_version"] = VERSION
    stats = {
        "rows": nrows,
        "sample_ms": round(step_ms, 3),
        "floor_wl": [round(float(x), 2) for x in floor_wl],
        "wl_global_dynamic": [round(float(x), 2) for x in global_dyn],
        "lp_state_rest_fraction": round(float(np.mean(state)), 4),
        "mean_disconnection_by_channel": [
            round(float(x), 3) for x in disconnected.mean(axis=0)
        ],
        "mean_high_activity_by_channel": [
            round(float(x), 3) for x in activity.mean(axis=0)
        ],
        "uncertain_fraction": round(
            float(np.mean(np.max(unknown, axis=1) > 0.45)), 4
        ),
        "psd_available": psd_available.tolist(),
    }
    return result, stats


def read_csv_features(path: Path) -> pd.DataFrame:
    needed = set(WL_COLUMNS) | {x for group in PSD_COLUMNS for x in group} | {
        "Timestamp", "MonotonicTimeUs"
    }
    return pd.read_csv(
        path, compression="infer", usecols=lambda name: name in needed,
        low_memory=False,
    )


def open_text(path: Path, mode: str):
    if path.name.lower().endswith(".gz"):
        return gzip.open(
            path, mode, newline="",
            encoding="utf-8-sig" if "r" in mode else "utf-8",
            compresslevel=3,
        )
    return open(
        path, mode, newline="",
        encoding="utf-8-sig" if "r" in mode else "utf-8",
    )


def write_labeled(src: Path, output: Path, labels: pd.DataFrame) -> None:
    """Append labels without pandas rewriting unrelated source CSV fields."""
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".gz" if output.name.lower().endswith(".gz") else ".csv"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".lp-label-", suffix=suffix, dir=output.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with open_text(src, "rt") as source, open_text(tmp, "wt") as destination:
            reader = csv.reader(source)
            writer = csv.writer(destination)
            header = next(reader)
            if set(header) & set(OUTPUT_COLUMNS):
                raise ValueError("Input CSV already contains LP teacher columns")
            writer.writerow(header + list(OUTPUT_COLUMNS))
            data = labels[list(OUTPUT_COLUMNS)].to_numpy()
            count = 0
            for count, row in enumerate(reader, 1):
                if count > len(data):
                    raise RuntimeError(
                        f"CSV row count changed between feature read and write: {src}"
                    )
                l = data[count - 1]
                writer.writerow(row + [
                    (str(int(value)) if name == "lp_state" else
                     str(value) if name == "lp_teacher_version" else
                     f"{float(value):.5f}")
                    for name, value in zip(OUTPUT_COLUMNS, l)
                ])
            if count != len(data):
                raise RuntimeError(
                    f"CSV row count mismatch: read {len(data)}, wrote {count}"
                )
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()


def label_one(task: tuple[str, str, str, bool]) -> dict[str, Any]:
    src_str, root_str, output_str, overwrite = task
    src, root, out_root = Path(src_str), Path(root_str), Path(output_str)
    rel = src.relative_to(root)
    target = out_root / rel
    if target.exists() and not overwrite:
        return {"status": "skipped", "file": rel.as_posix()}
    try:
        feature_df = read_csv_features(src)
        labels, stats = compute_labels(feature_df)
        write_labeled(src, target, labels)
        return {"status": "labeled", "file": rel.as_posix(), **stats}
    except Exception as e:
        return {
            "status": "error", "file": rel.as_posix(),
            "reason": f"{type(e).__name__}: {e}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers", type=int,
        default=min(12, max(1, round((os.cpu_count() or 1) * 0.8))),
        help="CSV processing processes (80%% of logical CPUs, capped at 12 for RAM)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recreate existing labeled outputs"
    )
    parser.add_argument(
        "--limit-per-group", type=int, default=0,
        help="Quick test: limit files per immediate folder",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Exit as soon as any file fails",
    )
    parser.add_argument(
        "--summary-file", type=Path, default=None,
        help="JSON summary path (default output-dir/label_summary.json)",
    )
    args = parser.parse_args()
    source = args.input_dir.resolve()
    destination = args.output_dir.resolve()
    if not source.is_dir():
        parser.error(f"Input directory does not exist: {source}")
    if destination == source or source in destination.parents:
        parser.error(
            "Output must be outside the input directory to avoid recursive re-labeling"
        )
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    paths = sorted(
        p for p in source.rglob("*")
        if p.is_file() and (
            p.name.lower().endswith(".csv") or
            p.name.lower().endswith(".csv.gz")
        )
    )
    if args.limit_per_group:
        from collections import defaultdict
        included: dict[str, int] = defaultdict(int)
        subset = []
        for p in paths:
            group = p.relative_to(source).parts[0]
            if included[group] < args.limit_per_group:
                subset.append(p)
                included[group] += 1
        paths = subset
    if not paths:
        parser.error("No CSV or CSV.gz recordings were found")
    if args.summary_file is None:
        args.summary_file = destination / "label_summary.json"
    print(f"[LP] Input: {source}\n[LP] Output: {destination}")
    print(
        f"[LP] Files: {len(paths):,}; workers: {args.workers}; teacher: {VERSION}",
        flush=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    jobs = (
        (str(p), str(source), str(destination), args.overwrite)
        for p in paths
    )
    counts = {"labeled": 0, "skipped": 0, "error": 0}
    group_summary: dict[str, dict[str, float]] = {}
    errors: list[dict[str, str]] = []

    def consume(result, i):
        status = result["status"]
        counts[status] += 1
        grp = result["file"].split("/")[0]
        s = group_summary.setdefault(
            grp, {"files": 0, "rows": 0, "rest_rows": 0, "uncertain_rows": 0}
        )
        if status == "labeled":
            s["files"] += 1
            s["rows"] += result["rows"]
            s["rest_rows"] += round(
                result["rows"] * result["lp_state_rest_fraction"]
            )
            s["uncertain_rows"] += round(
                result["rows"] * result["uncertain_fraction"]
            )
        if status == "error":
            errors.append({
                "file": result["file"], "reason": result["reason"]
            })
            print(
                f"[LP][ERROR] {result['file']}: {result['reason']}",
                flush=True,
            )
        if i % 25 == 0 or i == len(paths):
            print(
                f"[LP] {i:,}/{len(paths):,} files | "
                f"labeled={counts['labeled']}, "
                f"skipped={counts['skipped']}, errors={counts['error']}",
                flush=True,
            )

    if args.workers == 1:
        for i, task in enumerate(jobs, 1):
            result = label_one(task)
            consume(result, i)
            if args.fail_fast and result["status"] == "error":
                break
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i, result in enumerate(executor.map(label_one, jobs, chunksize=1), 1):
                consume(result, i)
                if args.fail_fast and result["status"] == "error":
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    summary = {
        "version": VERSION,
        "input": str(source), "output": str(destination),
        "files_selected": len(paths), "counts": counts,
        "groups": group_summary, "errors": errors[:200],
    }
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[LP] Summary: {args.summary_file}", flush=True)
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
