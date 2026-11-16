#!/usr/bin/env python3
"""Generate causal firmware-aware pseudo-labels for 3-channel sEMG activity.

Nine regression targets: lp_resting_ch1..3, lp_high_activity_ch1..3,
lp_disconnection_ch1..3. Binary lp_state = 1 only when all channels appear
connected and confidently resting; 0 means activity, contact fault, OR
uncertainty. lp_unknown_ch1..3 and lp_state_weight encode that uncertainty.

No PointerState, TriggerPressure, class_id, subject, or directory labels are
ever used as truth. This is a CAUSAL heuristic teacher, not clinical ground
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

VERSION = "lp_causal_fw_v2"
DEFAULT_INPUT = Path("DATA/DATA_LONG_PRESS")
DEFAULT_OUTPUT = Path("DATA/DATA_LONG_PRESS_LABELED_V2")
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


def compute_labels(df: pd.DataFrame, psd_mode: str = "auto") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Causal firmware-informed teacher V2. No full-recording lookahead."""
    from lp_causal_fw_teacher import compute_causal_labels
    return compute_causal_labels(df, psd_mode=psd_mode)


def read_csv_features(path: Path) -> pd.DataFrame:
    needed = set(WL_COLUMNS) | {x for group in PSD_COLUMNS for x in group} | {
        "Timestamp", "MonotonicTimeUs", "psd_mean_applied"
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


def label_one(task: tuple) -> dict[str, Any]:
    if len(task) == 4:
        src_str, root_str, output_str, overwrite = task
        psd_mode = "auto"
    else:
        src_str, root_str, output_str, overwrite, psd_mode = task
    src, root, out_root = Path(src_str), Path(root_str), Path(output_str)
    rel = src.relative_to(root)
    target = out_root / rel
    if target.exists() and not overwrite:
        return {"status": "skipped", "file": rel.as_posix()}
    try:
        feature_df = read_csv_features(src)
        labels, stats = compute_labels(feature_df, psd_mode=psd_mode)
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
    parser.add_argument("--psd-mode", choices=("auto", "raw", "legacy-divided"),
        default="auto", help="auto trusts psd_mean_applied flag, otherwise assumes raw FW log-sums")
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
    print("[LP] Firmware EMG_PSD_MEAN=0: sum log zone bins; mean field = 15-frame log history")
    print(f"[LP] PSD source mode: {args.psd_mode}")
    print(f"[LP] Input: {source}\n[LP] Output: {destination}")
    print(
        f"[LP] Files: {len(paths):,}; workers: {args.workers}; teacher: {VERSION}",
        flush=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    jobs = (
        (str(p), str(source), str(destination), args.overwrite, args.psd_mode)
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
