#!/usr/bin/env python3
"""Unified dominant-axis ACC-PSD tap labeler for embedded binary classification.

Edit only the GLOBAL SETTINGS section, then run:

    python unified_tap_labeler_deterministic.py

The script performs one deterministic labeling pass:

1. Removes all columns produced by previous labeling runs.
2. Initializes fresh binary ``state``, ``tap_id``, and ``regression`` columns.
3. Uses PointerState 0->1 and 1->0 transitions to define one press cycle.
4. Uses all PointerState==0 rows as the file-specific resting reference.
5. Builds X/Y/Z ACC PSD power from the configured frequency bins.
6. Inside each press cycle, finds the strongest baseline-normalized axis event.
7. Requires one axis to clearly dominate, exceed its own resting baseline, and
   form a short locally prominent impulse while total ACC power also rises.
8. Writes one short binary state block around every accepted impact.
9. Keeps every folder outside TAP/SWIPE as a hard negative with state=0.

``regression`` is only a diagnostic confidence shape and can be ignored by the
current classifier training. No ML model is trained or used by this labeler.

The script never deletes rows. It atomically rewrites CSV files in DATA_ROOT.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# ======================================================================================
# GLOBAL SETTINGS -- EDIT ONLY THIS SECTION
# ======================================================================================

# Printed at startup so you can confirm that the newest file is running.
SCRIPT_VERSION = "v10.0-deterministic-restored-2026-09-17"

# Input and output are intentionally identical. The script atomically rewrites the
# labeled CSVs. Keep untouched originals in your separate backup folder.
DATA_ROOT = Path("DATA/DATA_TAP")
DEFAULT_INPUT_ROOT = DATA_ROOT
DEFAULT_OUTPUT_ROOT = DATA_ROOT

# Before any calculation, drop every stale output created by this or old labelers.
# All fresh label_* diagnostics are regenerated from the original sensor columns.
COLUMNS_DROPPED_BEFORE_LABELING = (
    "state",
    "tap_id",
    "regression",
    "conseq_id",
)
COLUMN_PREFIXES_DROPPED_BEFORE_LABELING = ("label_",)

# True: calculate labels and reports without rewriting sensor CSV files.
# False: rewrite the CSV files in place.
ANALYZE_ONLY = False

# Reports are excluded from future sensor scans.
LABEL_REPORT_FOLDER_NAME = "_labeling_reports"

# Only these subfolders may receive state=1. Everything else is initialized as
# a hard negative with state=0, tap_id=0, regression=0.
POSITIVE_FOLDERS = ("TAP", "SWIPE")

# ACC PSD bins summed into X, Y, and Z power. Bin 0 is excluded by default because
# it is commonly dominated by slow/DC-like energy rather than the tap impulse.
POWER_BINS = (1, 2, 3, 4, 5)

# PointerState timing. PointerState is a phase supervisor, not an exact timestamp.
# Search around each 0->1 press edge, but use 1->0 release and the next press to
# close the cycle and prevent one impact leaking into another tap.
SEARCH_SAMPLES_BEFORE_PRESS = 20
SEARCH_SAMPLES_AFTER_PRESS = 30
RELEASE_GRACE_SAMPLES = 8
MINIMUM_SEARCH_AFTER_PRESS = 12
BETWEEN_CYCLE_GUARD_SAMPLES = 3
MINIMUM_PRESS_PHASE_SAMPLES = 8

# All valid PointerState==0 rows contribute to the file-specific rest reference.
MINIMUM_REST_ROWS = 20

# Small median smoothing suppresses isolated PSD glitches without strongly smearing
# a short impact. An odd value is enforced internally.
SMOOTHING_WINDOW = 3

# ----------------------------------------------------------------------
# MAIN EVENT ACCEPTANCE GATES
# ----------------------------------------------------------------------
# The strongest responding axis may be X, Y, or Z. It must exceed its own file
# baseline by both a ratio and a robust deviation score.
MINIMUM_DOMINANT_AXIS_REST_RATIO = 1.50
MINIMUM_DOMINANT_AXIS_REST_SCORE = 6.0

# The whole accelerometer must also rise; this prevents a single noisy feature bin
# from creating a positive label.
MINIMUM_TOTAL_POWER_REST_RATIO = 1.20
MINIMUM_TOTAL_POWER_REST_SCORE = 5.0

# One axis must clearly lead the other axes at the candidate peak.
# Example: 1.15 means the best normalized axis score must be at least 15% above
# the second-highest axis score.
MINIMUM_DOMINANCE_OVER_SECOND_AXIS = 1.15

# Fraction of the three baseline-normalized axis scores carried by the selected
# axis. This is more orientation-safe than raw-power share because X/Y/Z can have
# very different resting magnitudes. Together with the 1.15x lead rule, it still
# requires one clear primary axis instead of accepting general multi-axis motion.
MINIMUM_DOMINANT_SCORE_SHARE = 0.38

# Candidate must be a short local impulse rather than broad continuous motion.
MINIMUM_LOCAL_PROMINENCE_SIGMA = 3.0
MINIMUM_SUPPORT_SAMPLES = 2
SUPPORT_NEIGHBORHOOD_RADIUS = 2
SUPPORT_REST_SCORE = 3.0
SUPPORT_PEAK_EXCESS_FRACTION = 0.30

# Half-height width of the dominant-axis impulse. Broad events are rejected.
IMPACT_WIDTH_EXCESS_FRACTION = 0.50
MINIMUM_IMPACT_WIDTH_SAMPLES = 2
MAXIMUM_IMPACT_WIDTH_SAMPLES = 16

# During the impulse region, the selected axis should remain the strongest normalized
# axis for at least this fraction of rows. It need not dominate every ringing sample.
MINIMUM_DOMINANT_ROWS_FRACTION = 0.50

# Camera timing affects ranking only; it is not a hard acceptance rule.
EXPECTED_PEAK_OFFSET_FROM_PRESS = 6
TIMING_PREFERENCE_SCALE = 12.0

# ----------------------------------------------------------------------
# BINARY LABEL BLOCK
# ----------------------------------------------------------------------
# Expand from the accepted peak while the selected axis remains meaningfully above
# its own rest baseline. The final state block is clamped for embedded-friendly timing.
LABEL_AXIS_MINIMUM_REST_RATIO = 1.20
LABEL_AXIS_MINIMUM_REST_SCORE = 3.0
LABEL_PEAK_EXCESS_FRACTION = 0.40
MINIMUM_LABEL_SAMPLES = 3
MAXIMUM_LABEL_SAMPLES = 13

# Optional Gaussian confidence stored inside the binary state block.
REGRESSION_SIGMA = 2.0

# Diagnostic columns do not begin with p_core->, so train_2.py ignores them as model
# features. Keep them during label development; they are useful for auditing.
KEEP_DIAGNOSTIC_COLUMNS = True


# ======================================================================================
# INTERNAL TYPES / CONSTANTS
# ======================================================================================

ACC_AXES = ("X", "Y", "Z")
AXIS_CODE = {"X": 1, "Y": 2, "Z": 3}
POINTER_ALIASES = ("PointerState", "pointerState", "pointer_state", "pointer state")
GENERATED_LABEL_COLUMNS = frozenset(COLUMNS_DROPPED_BEFORE_LABELING)
GENERATED_LABEL_PREFIXES = tuple(COLUMN_PREFIXES_DROPPED_BEFORE_LABELING)
REPORT_CSV_FILENAMES = frozenset({"labeling_files.csv", "labeling_events.csv"})
EPSILON = 1.0e-6


@dataclass(frozen=True)
class RobustStats:
    center: float
    scale: float
    q10: float
    q90: float


@dataclass(frozen=True)
class PressCycle:
    press_index: int
    release_index: int
    observed_release: bool


@dataclass
class Candidate:
    index: int
    score: float
    accepted: bool
    reasons: list[str]
    offset_from_press: int
    offset_from_release: int
    dominant_axis: str
    dominant_axis_code: int
    x_power: float
    y_power: float
    z_power: float
    total_power: float
    x_rest_score: float
    y_rest_score: float
    z_rest_score: float
    dominant_rest_score: float
    second_axis_rest_score: float
    dominant_rest_ratio: float
    total_rest_score: float
    total_rest_ratio: float
    dominance_over_second_axis: float
    dominant_excess_share: float
    dominant_score_share: float
    local_prominence_sigma: float
    support_samples: int
    impact_width_samples: int
    dominant_rows_fraction: float
    impact_start: int
    impact_end: int
    label_threshold: float


@dataclass
class EventReport:
    file: str
    cycle_number: int
    press_index: int
    release_index: int
    observed_release: bool
    press_phase_samples: int
    search_start: int
    search_end: int
    accepted: bool
    reason: str
    tap_id: int
    peak_index: int | None
    offset_from_press: int | None
    offset_from_release: int | None
    label_start: int | None
    label_end: int | None
    rest_rows: int
    power_rest_mean_x: float | None
    power_rest_mean_y: float | None
    power_rest_mean_z: float | None
    power_rest_mean_total: float | None
    power_rest_median_x: float | None
    power_rest_median_y: float | None
    power_rest_median_z: float | None
    power_rest_scale_x: float | None
    power_rest_scale_y: float | None
    power_rest_scale_z: float | None
    power_rest_median_total: float | None
    power_rest_scale_total: float | None
    dominant_axis: str | None
    dominant_axis_code: int | None
    peak_x_power: float | None
    peak_y_power: float | None
    peak_z_power: float | None
    peak_total_power: float | None
    x_rest_score: float | None
    y_rest_score: float | None
    z_rest_score: float | None
    dominant_rest_score: float | None
    second_axis_rest_score: float | None
    dominant_rest_ratio: float | None
    total_rest_score: float | None
    total_rest_ratio: float | None
    dominance_over_second_axis: float | None
    dominant_excess_share: float | None
    dominant_score_share: float | None
    local_prominence_sigma: float | None
    support_samples: int | None
    impact_width_samples: int | None
    dominant_rows_fraction: float | None
    impact_start: int | None
    impact_end: int | None
    label_threshold: float | None
    candidate_score: float | None


@dataclass
class FileReport:
    file: str
    positive_folder: bool
    status: str
    rows: int
    pointer_column: str | None
    press_cycles: int
    accepted_events: int
    rejected_events: int
    positive_rows: int
    output_file: str | None
    error: str | None = None


# ======================================================================================
# BASIC HELPERS
# ======================================================================================


def resolve_column(columns: Iterable[Any], aliases: Sequence[str]) -> str | None:
    normalized: dict[str, list[str]] = {}
    for value in columns:
        if isinstance(value, str):
            normalized.setdefault(value.strip().lower(), []).append(value)

    for alias in aliases:
        matches = normalized.get(alias.strip().lower(), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous duplicate column for {alias!r}: {matches}")
    return None


def acc_psd_column(axis: str, bin_index: int) -> str:
    return f"p_core->acc_psd[0][{axis}][{bin_index}]"


def required_power_columns() -> dict[str, list[str]]:
    return {
        axis: [acc_psd_column(axis, bin_index) for bin_index in POWER_BINS]
        for axis in ACC_AXES
    }


def rolling_median(values: np.ndarray) -> np.ndarray:
    window = max(1, int(SMOOTHING_WINDOW))
    if window % 2 == 0:
        window += 1
    return (
        pd.Series(values, dtype="float64")
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )


def robust_stats(values: np.ndarray) -> RobustStats:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot calculate robust statistics from zero finite values.")

    center = float(np.median(finite))
    mad_scale = float(1.4826 * np.median(np.abs(finite - center)))
    q10, q25, q75, q90 = np.percentile(finite, [10, 25, 75, 90])
    iqr_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0

    # Prevent a nearly flat baseline from producing enormous scores for tiny noise.
    scale_floor = max(EPSILON, abs(center) * 0.05, abs(float(q90)) * 0.01)
    return RobustStats(
        center=center,
        scale=max(mad_scale, iqr_scale, scale_floor),
        q10=float(q10),
        q90=float(q90),
    )


def ratio_with_floor(value: float, baseline_mean: float, stats: RobustStats) -> float:
    floor = max(stats.scale * 0.25, abs(baseline_mean) * 0.02, EPSILON)
    denominator = baseline_mean + floor
    if denominator <= EPSILON:
        denominator = floor
    return float((value + floor) / denominator)


def pointer_values(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if np.isfinite(numeric).any():
        return np.nan_to_num(numeric, nan=0.0) > 0.0

    true_tokens = {"1", "true", "pressed", "press", "down", "hold", "active"}
    return np.fromiter(
        (str(value).strip().lower() in true_tokens for value in series.to_numpy()),
        dtype=bool,
        count=len(series),
    )


def pointer_cycles(active: np.ndarray) -> list[PressCycle]:
    if active.size == 0:
        return []

    previous = np.concatenate(([False], active[:-1]))
    following = np.concatenate((active[1:], [False]))
    presses = np.flatnonzero(active & ~previous)
    releases = np.flatnonzero(active & ~following)

    cycles: list[PressCycle] = []
    for press_index, release_index in zip(presses, releases):
        observed_release = not (
            int(release_index) == len(active) - 1 and bool(active[-1])
        )
        cycles.append(
            PressCycle(
                press_index=int(press_index),
                release_index=int(release_index),
                observed_release=observed_release,
            )
        )
    return cycles


def calculate_axis_powers(
    df: pd.DataFrame,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    columns_by_axis = required_power_columns()
    missing = [
        column
        for columns in columns_by_axis.values()
        for column in columns
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required ACC PSD columns. Example: {missing[:4]}"
        )

    raw: dict[str, np.ndarray] = {}
    smoothed: dict[str, np.ndarray] = {}
    valid = np.ones(len(df), dtype=bool)

    for axis, columns in columns_by_axis.items():
        matrix = (
            df.loc[:, columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64, copy=True)
        )
        valid &= np.isfinite(matrix).all(axis=1)
        values = np.nansum(matrix, axis=1)
        raw[axis] = values
        smoothed[axis] = rolling_median(values)

    total_raw = raw["X"] + raw["Y"] + raw["Z"]
    total_smoothed = smoothed["X"] + smoothed["Y"] + smoothed["Z"]
    valid &= np.isfinite(total_smoothed)
    return raw, smoothed, total_raw, total_smoothed, valid


def drop_previous_generated_columns(df: pd.DataFrame) -> list[str]:
    columns_to_drop = [
        column
        for column in df.columns
        if (
            column in GENERATED_LABEL_COLUMNS
            or (
                isinstance(column, str)
                and any(column.startswith(prefix) for prefix in GENERATED_LABEL_PREFIXES)
            )
        )
    ]
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
    return columns_to_drop


def initialize_labels(df: pd.DataFrame) -> None:
    df["state"] = np.zeros(len(df), dtype=np.int8)
    df["tap_id"] = np.zeros(len(df), dtype=np.int64)
    df["regression"] = np.zeros(len(df), dtype=np.float32)


def is_positive_path(path: Path, input_root: Path) -> bool:
    parts = {part.strip().upper() for part in path.relative_to(input_root).parts[:-1]}
    return any(folder.strip().upper() in parts for folder in POSITIVE_FOLDERS)


# ======================================================================================
# POINTER-STATE WINDOWS AND FILE REST BASELINE
# ======================================================================================


def cycle_search_window(
    cycles: Sequence[PressCycle],
    cycle_number: int,
    row_count: int,
) -> tuple[int, int, int, int]:
    cycle = cycles[cycle_number]
    previous_release = cycles[cycle_number - 1].release_index if cycle_number > 0 else -1
    next_press = (
        cycles[cycle_number + 1].press_index
        if cycle_number + 1 < len(cycles)
        else row_count
    )

    cycle_left = max(0, previous_release + 1 + BETWEEN_CYCLE_GUARD_SAMPLES)
    cycle_right = min(row_count - 1, next_press - 1 - BETWEEN_CYCLE_GUARD_SAMPLES)

    search_start = max(cycle_left, cycle.press_index - SEARCH_SAMPLES_BEFORE_PRESS)
    normal_end = cycle.press_index + SEARCH_SAMPLES_AFTER_PRESS
    release_end = cycle.release_index + RELEASE_GRACE_SAMPLES
    minimum_end = cycle.press_index + MINIMUM_SEARCH_AFTER_PRESS
    search_end = max(minimum_end, min(normal_end, release_end))
    search_end = min(cycle_right, row_count - 1, search_end)

    return int(search_start), int(search_end), int(cycle_left), int(cycle_right)


def file_rest_indices(active: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero((~active) & valid_mask).astype(np.int64)
    if indices.size < MINIMUM_REST_ROWS:
        raise ValueError(
            f"Only {indices.size} valid PointerState==0 rest rows; "
            f"need {MINIMUM_REST_ROWS}."
        )
    return indices


def rest_statistics(
    smoothed: dict[str, np.ndarray],
    total_smoothed: np.ndarray,
    rest_indices: np.ndarray,
) -> tuple[dict[str, RobustStats], RobustStats, dict[str, float], float]:
    axis_stats = {
        axis: robust_stats(values[rest_indices])
        for axis, values in smoothed.items()
    }
    total_stats = robust_stats(total_smoothed[rest_indices])
    axis_means = {
        axis: float(np.mean(values[rest_indices], dtype=np.float64))
        for axis, values in smoothed.items()
    }
    total_mean = float(np.mean(total_smoothed[rest_indices], dtype=np.float64))
    return axis_stats, total_stats, axis_means, total_mean


def normalized_axis_scores(
    smoothed: dict[str, np.ndarray],
    axis_stats: dict[str, RobustStats],
) -> dict[str, np.ndarray]:
    return {
        axis: np.maximum(
            0.0,
            (smoothed[axis] - axis_stats[axis].center) / axis_stats[axis].scale,
        )
        for axis in ACC_AXES
    }


# ======================================================================================
# CANDIDATE DETECTION
# ======================================================================================


def local_maxima(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    previous = np.concatenate(([-np.inf], values[:-1]))
    following = np.concatenate((values[1:], [-np.inf]))
    maxima = indices[
        (values[indices] >= previous[indices])
        & (values[indices] >= following[indices])
    ]
    return maxima if maxima.size else indices


def contiguous_region_above(
    values: np.ndarray,
    peak_index: int,
    threshold: float,
    left_limit: int,
    right_limit: int,
) -> tuple[int, int]:
    start = int(peak_index)
    end = int(peak_index)
    while start > left_limit and values[start - 1] >= threshold:
        start -= 1
    while end < right_limit and values[end + 1] >= threshold:
        end += 1
    return start, end


def candidate_metrics(
    index: int,
    press_index: int,
    release_index: int,
    search_start: int,
    search_end: int,
    smoothed: dict[str, np.ndarray],
    total_smoothed: np.ndarray,
    axis_scores: dict[str, np.ndarray],
    axis_stats: dict[str, RobustStats],
    total_stats: RobustStats,
    axis_rest_means: dict[str, float],
    total_rest_mean: float,
) -> Candidate:
    powers = {axis: float(smoothed[axis][index]) for axis in ACC_AXES}
    rest_scores = {axis: float(axis_scores[axis][index]) for axis in ACC_AXES}
    ordered_axes = sorted(ACC_AXES, key=lambda axis: rest_scores[axis], reverse=True)
    dominant_axis = ordered_axes[0]
    second_axis = ordered_axes[1]

    dominant_power = powers[dominant_axis]
    dominant_stats = axis_stats[dominant_axis]
    dominant_rest_mean = axis_rest_means[dominant_axis]
    dominant_rest_score = rest_scores[dominant_axis]
    second_axis_score = rest_scores[second_axis]
    dominance_over_second = dominant_rest_score / max(second_axis_score, EPSILON)
    total_axis_score = sum(rest_scores.values())
    dominant_score_share = dominant_rest_score / max(total_axis_score, EPSILON)

    excess = {
        axis: max(0.0, powers[axis] - axis_rest_means[axis])
        for axis in ACC_AXES
    }
    excess_total = sum(excess.values())
    dominant_excess_share = excess[dominant_axis] / max(excess_total, EPSILON)

    dominant_rest_ratio = ratio_with_floor(
        dominant_power, dominant_rest_mean, dominant_stats
    )
    total_power = float(total_smoothed[index])
    total_rest_score = float(
        (total_power - total_stats.center) / total_stats.scale
    )
    total_rest_ratio = ratio_with_floor(total_power, total_rest_mean, total_stats)

    # Local prominence is evaluated on the selected axis, not always Z.
    flank_start = max(search_start, index - 20)
    flank_end = min(search_end + 1, index + 21)
    left = np.arange(flank_start, max(flank_start, index - 5), dtype=np.int64)
    right = np.arange(min(flank_end, index + 6), flank_end, dtype=np.int64)
    flanks = np.concatenate((left, right))
    if flanks.size >= 6:
        local = robust_stats(smoothed[dominant_axis][flanks])
        local_center = local.center
        local_scale = max(local.scale, dominant_stats.scale * 0.50)
    else:
        local_center = dominant_stats.center
        local_scale = dominant_stats.scale
    local_prominence = (dominant_power - local_center) / local_scale

    # Supporting samples around the peak must also be meaningfully above rest.
    support_threshold = dominant_stats.center + max(
        SUPPORT_REST_SCORE * dominant_stats.scale,
        SUPPORT_PEAK_EXCESS_FRACTION
        * max(0.0, dominant_power - dominant_stats.center),
    )
    support_start = max(search_start, index - SUPPORT_NEIGHBORHOOD_RADIUS)
    support_end = min(search_end + 1, index + SUPPORT_NEIGHBORHOOD_RADIUS + 1)
    support_samples = int(
        np.count_nonzero(
            smoothed[dominant_axis][support_start:support_end] >= support_threshold
        )
    )

    # Half-height impulse width and axis consistency across that impulse.
    impact_threshold = dominant_stats.center + IMPACT_WIDTH_EXCESS_FRACTION * max(
        0.0, dominant_power - dominant_stats.center
    )
    impact_start, impact_end = contiguous_region_above(
        smoothed[dominant_axis],
        peak_index=index,
        threshold=impact_threshold,
        left_limit=search_start,
        right_limit=search_end,
    )
    impact_width = impact_end - impact_start + 1

    selected_scores = axis_scores[dominant_axis][impact_start : impact_end + 1]
    other_max = np.maximum.reduce(
        [
            axis_scores[axis][impact_start : impact_end + 1]
            for axis in ACC_AXES
            if axis != dominant_axis
        ]
    )
    dominant_rows_fraction = float(
        np.mean(selected_scores >= other_max)
    ) if selected_scores.size else 0.0

    label_threshold = max(
        dominant_rest_mean * LABEL_AXIS_MINIMUM_REST_RATIO,
        dominant_stats.center + LABEL_AXIS_MINIMUM_REST_SCORE * dominant_stats.scale,
        dominant_stats.center
        + LABEL_PEAK_EXCESS_FRACTION
        * max(0.0, dominant_power - dominant_stats.center),
    )

    expected_index = press_index + EXPECTED_PEAK_OFFSET_FROM_PRESS
    timing_bonus = math.exp(
        -abs(index - expected_index) / max(float(TIMING_PREFERENCE_SCALE), 1.0)
    )

    score = (
        1.5 * math.log1p(max(dominant_rest_score, 0.0))
        + 0.7 * math.log1p(max(total_rest_score, 0.0))
        + 1.2 * math.log1p(max(dominant_rest_ratio - 1.0, 0.0))
        + 0.5 * math.log1p(max(total_rest_ratio - 1.0, 0.0))
        + 1.2 * dominant_score_share
        + 0.6 * math.log1p(max(dominance_over_second, 0.0))
        + 0.8 * math.log1p(max(local_prominence, 0.0))
        + 0.5 * dominant_rows_fraction
        + 2.0 * timing_bonus
    )

    reasons: list[str] = []
    if dominant_rest_ratio < MINIMUM_DOMINANT_AXIS_REST_RATIO:
        reasons.append(
            f"dominant_rest_ratio<{MINIMUM_DOMINANT_AXIS_REST_RATIO:g}"
        )
    if dominant_rest_score < MINIMUM_DOMINANT_AXIS_REST_SCORE:
        reasons.append(
            f"dominant_rest_score<{MINIMUM_DOMINANT_AXIS_REST_SCORE:g}"
        )
    if total_rest_ratio < MINIMUM_TOTAL_POWER_REST_RATIO:
        reasons.append(f"total_rest_ratio<{MINIMUM_TOTAL_POWER_REST_RATIO:g}")
    if total_rest_score < MINIMUM_TOTAL_POWER_REST_SCORE:
        reasons.append(f"total_rest_score<{MINIMUM_TOTAL_POWER_REST_SCORE:g}")
    if dominance_over_second < MINIMUM_DOMINANCE_OVER_SECOND_AXIS:
        reasons.append(
            f"dominance<{MINIMUM_DOMINANCE_OVER_SECOND_AXIS:g}"
        )
    if dominant_score_share < MINIMUM_DOMINANT_SCORE_SHARE:
        reasons.append(
            f"dominant_score_share<{MINIMUM_DOMINANT_SCORE_SHARE:g}"
        )
    if local_prominence < MINIMUM_LOCAL_PROMINENCE_SIGMA:
        reasons.append(f"prominence<{MINIMUM_LOCAL_PROMINENCE_SIGMA:g}")
    if support_samples < MINIMUM_SUPPORT_SAMPLES:
        reasons.append(f"support<{MINIMUM_SUPPORT_SAMPLES}")
    if impact_width < MINIMUM_IMPACT_WIDTH_SAMPLES:
        reasons.append(f"impact_width<{MINIMUM_IMPACT_WIDTH_SAMPLES}")
    if impact_width > MAXIMUM_IMPACT_WIDTH_SAMPLES:
        reasons.append(f"impact_width>{MAXIMUM_IMPACT_WIDTH_SAMPLES}")
    if dominant_rows_fraction < MINIMUM_DOMINANT_ROWS_FRACTION:
        reasons.append(
            f"dominant_rows_fraction<{MINIMUM_DOMINANT_ROWS_FRACTION:g}"
        )

    return Candidate(
        index=int(index),
        score=float(score),
        accepted=not reasons,
        reasons=reasons,
        offset_from_press=int(index - press_index),
        offset_from_release=int(index - release_index),
        dominant_axis=dominant_axis,
        dominant_axis_code=AXIS_CODE[dominant_axis],
        x_power=powers["X"],
        y_power=powers["Y"],
        z_power=powers["Z"],
        total_power=total_power,
        x_rest_score=rest_scores["X"],
        y_rest_score=rest_scores["Y"],
        z_rest_score=rest_scores["Z"],
        dominant_rest_score=float(dominant_rest_score),
        second_axis_rest_score=float(second_axis_score),
        dominant_rest_ratio=float(dominant_rest_ratio),
        total_rest_score=float(total_rest_score),
        total_rest_ratio=float(total_rest_ratio),
        dominance_over_second_axis=float(dominance_over_second),
        dominant_excess_share=float(dominant_excess_share),
        dominant_score_share=float(dominant_score_share),
        local_prominence_sigma=float(local_prominence),
        support_samples=support_samples,
        impact_width_samples=int(impact_width),
        dominant_rows_fraction=float(dominant_rows_fraction),
        impact_start=int(impact_start),
        impact_end=int(impact_end),
        label_threshold=float(label_threshold),
    )


def choose_candidate(
    press_index: int,
    release_index: int,
    search_start: int,
    search_end: int,
    smoothed: dict[str, np.ndarray],
    total_smoothed: np.ndarray,
    axis_scores: dict[str, np.ndarray],
    axis_stats: dict[str, RobustStats],
    total_stats: RobustStats,
    axis_rest_means: dict[str, float],
    total_rest_mean: float,
    valid_mask: np.ndarray,
) -> tuple[Candidate | None, Candidate | None]:
    if search_end < search_start:
        return None, None

    indices = np.arange(search_start, search_end + 1, dtype=np.int64)
    indices = indices[valid_mask[indices]]
    if indices.size == 0:
        return None, None

    impact_series = np.maximum.reduce([axis_scores[axis] for axis in ACC_AXES])
    maxima = local_maxima(impact_series, indices)

    candidates = [
        candidate_metrics(
            index=int(index),
            press_index=press_index,
            release_index=release_index,
            search_start=search_start,
            search_end=search_end,
            smoothed=smoothed,
            total_smoothed=total_smoothed,
            axis_scores=axis_scores,
            axis_stats=axis_stats,
            total_stats=total_stats,
            axis_rest_means=axis_rest_means,
            total_rest_mean=total_rest_mean,
        )
        for index in maxima
    ]
    candidates.sort(
        key=lambda item: (
            -item.score,
            abs(item.offset_from_press - EXPECTED_PEAK_OFFSET_FROM_PRESS),
        )
    )

    accepted = [candidate for candidate in candidates if candidate.accepted]
    if accepted:
        return accepted[0], candidates[0]
    return None, candidates[0] if candidates else None


# ======================================================================================
# BINARY LABEL APPLICATION
# ======================================================================================


def pad_region_to_minimum(
    values: np.ndarray,
    start: int,
    end: int,
    peak_index: int,
    left_limit: int,
    right_limit: int,
    minimum_width: int,
) -> tuple[int, int]:
    while end - start + 1 < minimum_width:
        left_value = values[start - 1] if start > left_limit else -np.inf
        right_value = values[end + 1] if end < right_limit else -np.inf
        if not np.isfinite(left_value) and not np.isfinite(right_value):
            break
        if left_value >= right_value:
            start -= 1
        else:
            end += 1
    if peak_index < start or peak_index > end:
        raise AssertionError("Padded label region lost the peak index.")
    return start, end


def trim_region_to_maximum(
    start: int,
    end: int,
    peak_index: int,
    maximum_width: int,
) -> tuple[int, int]:
    if end - start + 1 <= maximum_width:
        return start, end

    half_before = maximum_width // 2
    trimmed_start = max(start, peak_index - half_before)
    trimmed_end = trimmed_start + maximum_width - 1
    if trimmed_end > end:
        trimmed_end = end
        trimmed_start = trimmed_end - maximum_width + 1
    return int(trimmed_start), int(trimmed_end)


def apply_label(
    df: pd.DataFrame,
    candidate: Candidate,
    tap_id: int,
    cycle_left: int,
    cycle_right: int,
    smoothed: dict[str, np.ndarray],
) -> tuple[int, int]:
    values = smoothed[candidate.dominant_axis]
    start, end = contiguous_region_above(
        values,
        peak_index=candidate.index,
        threshold=candidate.label_threshold,
        left_limit=cycle_left,
        right_limit=cycle_right,
    )
    start, end = pad_region_to_minimum(
        values,
        start=start,
        end=end,
        peak_index=candidate.index,
        left_limit=cycle_left,
        right_limit=cycle_right,
        minimum_width=MINIMUM_LABEL_SAMPLES,
    )
    start, end = trim_region_to_maximum(
        start=start,
        end=end,
        peak_index=candidate.index,
        maximum_width=MAXIMUM_LABEL_SAMPLES,
    )

    df.loc[start:end, "state"] = 1
    df.loc[start:end, "tap_id"] = int(tap_id)

    indices = np.arange(start, end + 1, dtype=np.float64)
    confidence = np.exp(
        -0.5
        * (
            (indices - float(candidate.index))
            / max(float(REGRESSION_SIGMA), EPSILON)
        )
        ** 2
    )
    df.loc[start:end, "regression"] = confidence.astype(np.float32)
    return int(start), int(end)


# ======================================================================================
# DIAGNOSTICS
# ======================================================================================


def add_diagnostics(
    df: pd.DataFrame,
    raw: dict[str, np.ndarray],
    smoothed: dict[str, np.ndarray],
    total_raw: np.ndarray,
    total_smoothed: np.ndarray,
    active: np.ndarray | None,
    cycles: Sequence[PressCycle],
    axis_stats: dict[str, RobustStats] | None = None,
    total_stats: RobustStats | None = None,
    axis_rest_means: dict[str, float] | None = None,
    total_rest_mean: float | None = None,
) -> None:
    if not KEEP_DIAGNOSTIC_COLUMNS:
        return

    for axis in ACC_AXES:
        lower = axis.lower()
        df[f"label_acc_power_{lower}"] = raw[axis].astype(np.float32)
        df[f"label_acc_power_{lower}_smooth"] = smoothed[axis].astype(np.float32)
    df["label_acc_power_total"] = total_raw.astype(np.float32)
    df["label_acc_power_total_smooth"] = total_smoothed.astype(np.float32)

    if active is not None:
        df["label_pointer_phase"] = active.astype(np.int8)
        cycle_ids = np.zeros(len(df), dtype=np.int32)
        for cycle_number, cycle in enumerate(cycles, start=1):
            cycle_ids[cycle.press_index : cycle.release_index + 1] = cycle_number
        df["label_pointer_cycle"] = cycle_ids

    if (
        axis_stats is None
        or total_stats is None
        or axis_rest_means is None
        or total_rest_mean is None
    ):
        return

    axis_score_arrays: dict[str, np.ndarray] = {}
    axis_ratio_arrays: dict[str, np.ndarray] = {}
    axis_excess_arrays: dict[str, np.ndarray] = {}

    for axis in ACC_AXES:
        lower = axis.lower()
        stats = axis_stats[axis]
        rest_mean = axis_rest_means[axis]
        score = np.maximum(
            0.0,
            (smoothed[axis] - stats.center) / stats.scale,
        )
        floor = max(stats.scale * 0.25, abs(rest_mean) * 0.02, EPSILON)
        denominator = rest_mean + floor
        if denominator <= EPSILON:
            denominator = floor
        ratio = (smoothed[axis] + floor) / denominator
        excess = np.maximum(0.0, smoothed[axis] - rest_mean)

        axis_score_arrays[axis] = score
        axis_ratio_arrays[axis] = ratio
        axis_excess_arrays[axis] = excess

        df[f"label_power_rest_mean_{lower}"] = np.float32(rest_mean)
        df[f"label_power_rest_median_{lower}"] = np.float32(stats.center)
        df[f"label_power_rest_scale_{lower}"] = np.float32(stats.scale)
        df[f"label_{lower}_power_rest_ratio"] = ratio.astype(np.float32)
        df[f"label_{lower}_rest_score"] = score.astype(np.float32)

    total_floor = max(
        total_stats.scale * 0.25,
        abs(total_rest_mean) * 0.02,
        EPSILON,
    )
    total_denominator = total_rest_mean + total_floor
    if total_denominator <= EPSILON:
        total_denominator = total_floor
    total_ratio = (total_smoothed + total_floor) / total_denominator
    total_score = np.maximum(
        0.0,
        (total_smoothed - total_stats.center) / total_stats.scale,
    )

    score_matrix = np.column_stack([axis_score_arrays[axis] for axis in ACC_AXES])
    dominant_indices = np.argmax(score_matrix, axis=1)
    sorted_scores = np.sort(score_matrix, axis=1)
    dominant_scores = sorted_scores[:, -1]
    second_scores = sorted_scores[:, -2]
    dominance = dominant_scores / np.maximum(second_scores, EPSILON)
    dominant_score_share = dominant_scores / np.maximum(
        np.sum(score_matrix, axis=1), EPSILON
    )

    excess_matrix = np.column_stack([axis_excess_arrays[axis] for axis in ACC_AXES])
    excess_total = np.sum(excess_matrix, axis=1)
    dominant_excess = excess_matrix[np.arange(len(df)), dominant_indices]
    dominant_share = dominant_excess / np.maximum(excess_total, EPSILON)

    df["label_power_rest_mean_total"] = np.float32(total_rest_mean)
    df["label_power_rest_median_total"] = np.float32(total_stats.center)
    df["label_power_rest_scale_total"] = np.float32(total_stats.scale)
    df["label_total_power_rest_ratio"] = total_ratio.astype(np.float32)
    df["label_total_rest_score"] = total_score.astype(np.float32)
    df["label_dominant_axis_code"] = (dominant_indices + 1).astype(np.int8)
    df["label_dominant_axis_score"] = dominant_scores.astype(np.float32)
    df["label_second_axis_score"] = second_scores.astype(np.float32)
    df["label_dominance_over_second_axis"] = dominance.astype(np.float32)
    df["label_dominant_score_share"] = dominant_score_share.astype(np.float32)
    df["label_dominant_excess_share"] = dominant_share.astype(np.float32)


# ======================================================================================
# FILE PROCESSING / REPORTS
# ======================================================================================


def atomic_write_csv(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        df.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def process_positive_file(
    path: Path,
    input_root: Path,
    output_path: Path,
    next_tap_id: int,
) -> tuple[FileReport, list[EventReport], int, pd.DataFrame]:
    df = pd.read_csv(path, low_memory=False)
    drop_previous_generated_columns(df)
    initialize_labels(df)

    raw, smoothed, total_raw, total_smoothed, valid_mask = calculate_axis_powers(df)
    pointer_column = resolve_column(df.columns, POINTER_ALIASES)
    if pointer_column is None:
        add_diagnostics(df, raw, smoothed, total_raw, total_smoothed, None, [])
        report = FileReport(
            file=str(path.relative_to(input_root)),
            positive_folder=True,
            status="rejected_missing_pointer_state",
            rows=len(df),
            pointer_column=None,
            press_cycles=0,
            accepted_events=0,
            rejected_events=0,
            positive_rows=0,
            output_file=str(output_path),
        )
        return report, [], next_tap_id, df

    active = pointer_values(df[pointer_column])
    cycles = pointer_cycles(active)

    axis_stats: dict[str, RobustStats] | None = None
    total_stats: RobustStats | None = None
    axis_rest_means: dict[str, float] | None = None
    total_rest_mean: float | None = None
    rest_indices = np.empty(0, dtype=np.int64)
    rest_error: str | None = None
    try:
        rest_indices = file_rest_indices(active, valid_mask)
        axis_stats, total_stats, axis_rest_means, total_rest_mean = rest_statistics(
            smoothed, total_smoothed, rest_indices
        )
    except Exception as exc:
        rest_error = f"{type(exc).__name__}:{exc}"

    add_diagnostics(
        df,
        raw,
        smoothed,
        total_raw,
        total_smoothed,
        active,
        cycles,
        axis_stats=axis_stats,
        total_stats=total_stats,
        axis_rest_means=axis_rest_means,
        total_rest_mean=total_rest_mean,
    )

    file_name = str(path.relative_to(input_root))
    events: list[EventReport] = []
    accepted_count = 0

    def make_event(
        *,
        cycle_number: int,
        cycle: PressCycle,
        search_start: int,
        search_end: int,
        accepted: bool,
        reason: str,
        tap_id: int = 0,
        chosen: Candidate | None = None,
        label_start: int | None = None,
        label_end: int | None = None,
    ) -> EventReport:
        return EventReport(
            file=file_name,
            cycle_number=cycle_number + 1,
            press_index=cycle.press_index,
            release_index=cycle.release_index,
            observed_release=cycle.observed_release,
            press_phase_samples=cycle.release_index - cycle.press_index + 1,
            search_start=search_start,
            search_end=search_end,
            accepted=accepted,
            reason=reason,
            tap_id=tap_id,
            peak_index=None if chosen is None else chosen.index,
            offset_from_press=None if chosen is None else chosen.offset_from_press,
            offset_from_release=None if chosen is None else chosen.offset_from_release,
            label_start=label_start,
            label_end=label_end,
            rest_rows=int(rest_indices.size),
            power_rest_mean_x=None if axis_rest_means is None else axis_rest_means["X"],
            power_rest_mean_y=None if axis_rest_means is None else axis_rest_means["Y"],
            power_rest_mean_z=None if axis_rest_means is None else axis_rest_means["Z"],
            power_rest_mean_total=total_rest_mean,
            power_rest_median_x=None if axis_stats is None else axis_stats["X"].center,
            power_rest_median_y=None if axis_stats is None else axis_stats["Y"].center,
            power_rest_median_z=None if axis_stats is None else axis_stats["Z"].center,
            power_rest_scale_x=None if axis_stats is None else axis_stats["X"].scale,
            power_rest_scale_y=None if axis_stats is None else axis_stats["Y"].scale,
            power_rest_scale_z=None if axis_stats is None else axis_stats["Z"].scale,
            power_rest_median_total=None if total_stats is None else total_stats.center,
            power_rest_scale_total=None if total_stats is None else total_stats.scale,
            dominant_axis=None if chosen is None else chosen.dominant_axis,
            dominant_axis_code=None if chosen is None else chosen.dominant_axis_code,
            peak_x_power=None if chosen is None else chosen.x_power,
            peak_y_power=None if chosen is None else chosen.y_power,
            peak_z_power=None if chosen is None else chosen.z_power,
            peak_total_power=None if chosen is None else chosen.total_power,
            x_rest_score=None if chosen is None else chosen.x_rest_score,
            y_rest_score=None if chosen is None else chosen.y_rest_score,
            z_rest_score=None if chosen is None else chosen.z_rest_score,
            dominant_rest_score=None if chosen is None else chosen.dominant_rest_score,
            second_axis_rest_score=None if chosen is None else chosen.second_axis_rest_score,
            dominant_rest_ratio=None if chosen is None else chosen.dominant_rest_ratio,
            total_rest_score=None if chosen is None else chosen.total_rest_score,
            total_rest_ratio=None if chosen is None else chosen.total_rest_ratio,
            dominance_over_second_axis=None
            if chosen is None
            else chosen.dominance_over_second_axis,
            dominant_excess_share=None
            if chosen is None
            else chosen.dominant_excess_share,
            dominant_score_share=None
            if chosen is None
            else chosen.dominant_score_share,
            local_prominence_sigma=None
            if chosen is None
            else chosen.local_prominence_sigma,
            support_samples=None if chosen is None else chosen.support_samples,
            impact_width_samples=None
            if chosen is None
            else chosen.impact_width_samples,
            dominant_rows_fraction=None
            if chosen is None
            else chosen.dominant_rows_fraction,
            impact_start=None if chosen is None else chosen.impact_start,
            impact_end=None if chosen is None else chosen.impact_end,
            label_threshold=None if chosen is None else chosen.label_threshold,
            candidate_score=None if chosen is None else chosen.score,
        )

    axis_scores: dict[str, np.ndarray] | None = None
    if axis_stats is not None:
        axis_scores = normalized_axis_scores(smoothed, axis_stats)

    for cycle_number, cycle in enumerate(cycles):
        search_start, search_end, cycle_left, cycle_right = cycle_search_window(
            cycles, cycle_number, len(df)
        )
        phase_samples = cycle.release_index - cycle.press_index + 1

        if phase_samples < MINIMUM_PRESS_PHASE_SAMPLES:
            events.append(
                make_event(
                    cycle_number=cycle_number,
                    cycle=cycle,
                    search_start=search_start,
                    search_end=search_end,
                    accepted=False,
                    reason=f"press_phase<{MINIMUM_PRESS_PHASE_SAMPLES}",
                )
            )
            continue

        if rest_error is not None:
            events.append(
                make_event(
                    cycle_number=cycle_number,
                    cycle=cycle,
                    search_start=search_start,
                    search_end=search_end,
                    accepted=False,
                    reason=f"rest_baseline_error:{rest_error}",
                )
            )
            continue

        assert axis_stats is not None
        assert total_stats is not None
        assert axis_rest_means is not None
        assert total_rest_mean is not None
        assert axis_scores is not None

        try:
            accepted, best_seen = choose_candidate(
                press_index=cycle.press_index,
                release_index=cycle.release_index,
                search_start=search_start,
                search_end=search_end,
                smoothed=smoothed,
                total_smoothed=total_smoothed,
                axis_scores=axis_scores,
                axis_stats=axis_stats,
                total_stats=total_stats,
                axis_rest_means=axis_rest_means,
                total_rest_mean=total_rest_mean,
                valid_mask=valid_mask,
            )

            if accepted is not None:
                tap_id = next_tap_id
                next_tap_id += 1
                accepted_count += 1
                label_start, label_end = apply_label(
                    df,
                    candidate=accepted,
                    tap_id=tap_id,
                    cycle_left=cycle_left,
                    cycle_right=cycle_right,
                    smoothed=smoothed,
                )
                events.append(
                    make_event(
                        cycle_number=cycle_number,
                        cycle=cycle,
                        search_start=search_start,
                        search_end=search_end,
                        accepted=True,
                        reason="accepted",
                        tap_id=tap_id,
                        chosen=accepted,
                        label_start=label_start,
                        label_end=label_end,
                    )
                )
            else:
                reason = (
                    "no_candidate"
                    if best_seen is None
                    else ";".join(best_seen.reasons) or "rejected"
                )
                events.append(
                    make_event(
                        cycle_number=cycle_number,
                        cycle=cycle,
                        search_start=search_start,
                        search_end=search_end,
                        accepted=False,
                        reason=reason,
                        chosen=best_seen,
                    )
                )
        except Exception as exc:
            events.append(
                make_event(
                    cycle_number=cycle_number,
                    cycle=cycle,
                    search_start=search_start,
                    search_end=search_end,
                    accepted=False,
                    reason=f"error:{type(exc).__name__}:{exc}",
                )
            )

    rejected_count = len(cycles) - accepted_count
    if not cycles:
        status = "rejected_no_pointer_cycles"
    elif accepted_count == len(cycles):
        status = "all_cycles_accepted"
    elif accepted_count > 0:
        status = "partially_accepted"
    else:
        status = "all_cycles_rejected"

    report = FileReport(
        file=file_name,
        positive_folder=True,
        status=status,
        rows=len(df),
        pointer_column=pointer_column,
        press_cycles=len(cycles),
        accepted_events=accepted_count,
        rejected_events=rejected_count,
        positive_rows=int((df["state"] == 1).sum()),
        output_file=str(output_path),
    )
    return report, events, next_tap_id, df


def process_hard_negative_file(
    path: Path,
    input_root: Path,
    output_path: Path,
) -> tuple[FileReport, pd.DataFrame]:
    df = pd.read_csv(path, low_memory=False)
    drop_previous_generated_columns(df)
    initialize_labels(df)
    report = FileReport(
        file=str(path.relative_to(input_root)),
        positive_folder=False,
        status="hard_negative_initialized",
        rows=len(df),
        pointer_column=resolve_column(df.columns, POINTER_ALIASES),
        press_cycles=0,
        accepted_events=0,
        rejected_events=0,
        positive_rows=0,
        output_file=str(output_path),
    )
    return report, df


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_reports(
    report_root: Path,
    file_reports: Sequence[FileReport],
    event_reports: Sequence[EventReport],
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(item) for item in file_reports]).to_csv(
        report_root / "labeling_files.csv", index=False
    )
    pd.DataFrame([asdict(item) for item in event_reports]).to_csv(
        report_root / "labeling_events.csv", index=False
    )

    summary = {
        "configuration": {
            "script_version": SCRIPT_VERSION,
            "input_root": str(DEFAULT_INPUT_ROOT),
            "output_root": str(DEFAULT_OUTPUT_ROOT),
            "analyze_only": ANALYZE_ONLY,
            "positive_folders": list(POSITIVE_FOLDERS),
            "power_bins": list(POWER_BINS),
            "search_samples_before_press": SEARCH_SAMPLES_BEFORE_PRESS,
            "search_samples_after_press": SEARCH_SAMPLES_AFTER_PRESS,
            "release_grace_samples": RELEASE_GRACE_SAMPLES,
            "minimum_search_after_press": MINIMUM_SEARCH_AFTER_PRESS,
            "minimum_press_phase_samples": MINIMUM_PRESS_PHASE_SAMPLES,
            "minimum_rest_rows": MINIMUM_REST_ROWS,
            "minimum_dominant_axis_rest_ratio": MINIMUM_DOMINANT_AXIS_REST_RATIO,
            "minimum_dominant_axis_rest_score": MINIMUM_DOMINANT_AXIS_REST_SCORE,
            "minimum_total_power_rest_ratio": MINIMUM_TOTAL_POWER_REST_RATIO,
            "minimum_total_power_rest_score": MINIMUM_TOTAL_POWER_REST_SCORE,
            "minimum_dominance_over_second_axis": MINIMUM_DOMINANCE_OVER_SECOND_AXIS,
            "minimum_dominant_score_share": MINIMUM_DOMINANT_SCORE_SHARE,
            "minimum_local_prominence_sigma": MINIMUM_LOCAL_PROMINENCE_SIGMA,
            "minimum_impact_width_samples": MINIMUM_IMPACT_WIDTH_SAMPLES,
            "maximum_impact_width_samples": MAXIMUM_IMPACT_WIDTH_SAMPLES,
            "minimum_dominant_rows_fraction": MINIMUM_DOMINANT_ROWS_FRACTION,
            "minimum_label_samples": MINIMUM_LABEL_SAMPLES,
            "maximum_label_samples": MAXIMUM_LABEL_SAMPLES,
        },
        "totals": {
            "files": len(file_reports),
            "positive_files": sum(item.positive_folder for item in file_reports),
            "hard_negative_files": sum(not item.positive_folder for item in file_reports),
            "press_cycles": sum(item.press_cycles for item in file_reports),
            "accepted_events": sum(item.accepted_events for item in file_reports),
            "rejected_events": sum(item.rejected_events for item in file_reports),
            "positive_rows": sum(item.positive_rows for item in file_reports),
            "errors": sum(item.error is not None for item in file_reports),
        },
        "dominant_axis_counts": {
            axis: sum(
                1
                for event in event_reports
                if event.accepted and event.dominant_axis == axis
            )
            for axis in ACC_AXES
        },
        "files": [
            {key: json_safe(value) for key, value in asdict(item).items()}
            for item in file_reports
        ],
        "events": [
            {key: json_safe(value) for key, value in asdict(item).items()}
            for item in event_reports
        ],
    }
    (report_root / "labeling_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def validate_settings() -> tuple[Path, Path]:
    input_root = DEFAULT_INPUT_ROOT.expanduser().resolve()
    output_root = DEFAULT_OUTPUT_ROOT.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"DEFAULT_INPUT_ROOT does not exist: {input_root}")
    if input_root != output_root:
        raise ValueError(
            "This version is configured for clean in-place rewriting. "
            "DEFAULT_INPUT_ROOT and DEFAULT_OUTPUT_ROOT must be identical."
        )
    if not POWER_BINS or any(index < 0 for index in POWER_BINS):
        raise ValueError("POWER_BINS must contain non-negative bin indices.")
    if SEARCH_SAMPLES_BEFORE_PRESS < 0 or SEARCH_SAMPLES_AFTER_PRESS < 0:
        raise ValueError("Search windows must be non-negative.")
    if MINIMUM_REST_ROWS < 1:
        raise ValueError("MINIMUM_REST_ROWS must be positive.")
    if MINIMUM_DOMINANT_AXIS_REST_RATIO <= 1.0:
        raise ValueError("MINIMUM_DOMINANT_AXIS_REST_RATIO must be greater than 1.0.")
    if MINIMUM_TOTAL_POWER_REST_RATIO <= 1.0:
        raise ValueError("MINIMUM_TOTAL_POWER_REST_RATIO must be greater than 1.0.")
    if MINIMUM_DOMINANCE_OVER_SECOND_AXIS <= 1.0:
        raise ValueError("MINIMUM_DOMINANCE_OVER_SECOND_AXIS must be greater than 1.0.")
    if not 0.0 < MINIMUM_DOMINANT_SCORE_SHARE <= 1.0:
        raise ValueError("MINIMUM_DOMINANT_SCORE_SHARE must be in (0, 1].")
    if MINIMUM_LABEL_SAMPLES < 1 or MAXIMUM_LABEL_SAMPLES < MINIMUM_LABEL_SAMPLES:
        raise ValueError("Label sample limits are invalid.")
    if MINIMUM_IMPACT_WIDTH_SAMPLES < 1:
        raise ValueError("MINIMUM_IMPACT_WIDTH_SAMPLES must be positive.")
    if MAXIMUM_IMPACT_WIDTH_SAMPLES < MINIMUM_IMPACT_WIDTH_SAMPLES:
        raise ValueError("Impact-width limits are invalid.")
    return input_root, output_root


def scan_sensor_csvs(input_root: Path, report_root: Path) -> list[Path]:
    csv_files: list[Path] = []
    for path in sorted(input_root.rglob("*.csv")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if path.name in REPORT_CSV_FILENAMES:
            continue
        if report_root == resolved or report_root in resolved.parents:
            continue
        csv_files.append(path)
    return csv_files


def main() -> int:
    input_root, output_root = validate_settings()
    report_root = output_root / LABEL_REPORT_FOLDER_NAME
    report_root.mkdir(parents=True, exist_ok=True)

    csv_files = scan_sensor_csvs(input_root, report_root)
    if not csv_files:
        raise FileNotFoundError(f"No sensor CSV files found below {input_root}")

    print("=" * 96)
    print("Unified dominant-axis deterministic ACC-PSD tap labeler")
    print("=" * 96)
    print(f"Version:               {SCRIPT_VERSION}")
    print(f"Data root:             {input_root}")
    print("In-place final output: True")
    print(f"Analyze only:          {ANALYZE_ONLY}")
    print(f"CSV files:             {len(csv_files)}")
    print(f"Positive folders:      {POSITIVE_FOLDERS}")
    print("Other folders:         hard locked state=0")
    print("-" * 96)

    file_reports: list[FileReport] = []
    event_reports: list[EventReport] = []
    next_tap_id = 1

    for file_number, path in enumerate(csv_files, start=1):
        relative = path.relative_to(input_root)
        destination = output_root / relative
        positive = is_positive_path(path, input_root)

        try:
            if positive:
                report, events, next_tap_id, frame = process_positive_file(
                    path, input_root, destination, next_tap_id
                )
                event_reports.extend(events)
            else:
                report, frame = process_hard_negative_file(
                    path, input_root, destination
                )
        except Exception as exc:
            # Preserve the original safety behavior: a failed file is rewritten with
            # fresh zero labels rather than keeping stale labels from an older run.
            frame = pd.read_csv(path, low_memory=False)
            drop_previous_generated_columns(frame)
            initialize_labels(frame)
            report = FileReport(
                file=str(relative),
                positive_folder=positive,
                status="error_safe_zero_labels",
                rows=len(frame),
                pointer_column=resolve_column(frame.columns, POINTER_ALIASES),
                press_cycles=0,
                accepted_events=0,
                rejected_events=0,
                positive_rows=0,
                output_file=str(destination),
                error=f"{type(exc).__name__}: {exc}",
            )

        file_reports.append(report)

        if not ANALYZE_ONLY:
            atomic_write_csv(frame, destination)

        print(
            f"[{file_number}/{len(csv_files)}] [{report.status}] {relative} | "
            f"presses={report.press_cycles}, accepted={report.accepted_events}, "
            f"rejected={report.rejected_events}, positive_rows={report.positive_rows}"
        )

    write_reports(report_root, file_reports, event_reports)

    total_positive_rows = sum(item.positive_rows for item in file_reports)
    total_events = sum(item.accepted_events for item in file_reports)
    total_rejected = sum(item.rejected_events for item in file_reports)
    total_errors = sum(item.error is not None for item in file_reports)

    print("\n" + "=" * 96)
    print("Done")
    print("=" * 96)
    print(f"Accepted events:       {total_events:,}")
    print(f"Rejected cycles:       {total_rejected:,}")
    print(f"Positive rows:         {total_positive_rows:,}")
    print(f"Files with errors:     {total_errors:,}")
    print(f"Reports:               {report_root}")
    if ANALYZE_ONLY:
        print("Source CSVs were not rewritten because ANALYZE_ONLY=True.")
    else:
        print("CSV files were atomically rewritten with deterministic labels.")
    print("Hard-negative contract: every folder outside TAP/SWIPE is state=0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
