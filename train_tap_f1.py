#!/usr/bin/env python3
"""High-throughput sequential Optuna training for embedded tree classifiers.

The fixed five-fold, FILE-level gesture-aware split is read exclusively from
DATA/fold_roadmap_files.json using the public fold_split module (python fold_split.py
build makes it: folds of recording files, never of people). One roadmap fold is
held out for final testing; the remaining folds rotate as Optuna validation.

Changes against the legacy trainer (everything else is the same code):
  * DATA_FOLDER = DATA/DATA_TAP_F1/ (build_tap_f1_dataset.py: signal labels as
    ``state`` plus the inputs), FOLD_ROADMAP_PATH = the file-level roadmap,
    TRAIN_FOLDER / study name of their own;
  * one line in _process_file_worker: rows with state < 0 ("unknown") are dropped
    before anything else, so they never train, tune or score. Without it the
    legacy rule (state > 0) would silently turn them into "not a tap".
Duplicates marked excluded by the roadmap never enter the data cache or model.

``TARGET_CLASS_ID`` is the only class allowed to produce binary target 1, and
only when the source ``state`` is active. Every other class remains a hard
negative. Rare detailed classes cannot always appear in all held-out groups;
missing classes are reported but never "fixed" by regrouping people or files.
Both binary target values must exist in every training/validation/test side.

The optimization target is binary F1 at ``DEFAULT_THRESHOLD`` and compact C
payload size. All supported preprocessing is non-log. Affine preprocessing is
folded into exported thresholds; row normalization is exported explicitly.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import warnings

# Limit native BLAS/OpenMP pools before importing NumPy/sklearn/LightGBM.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("SKLEARN_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import joblib
import numpy as np
import optuna
import pandas as pd
import psutil
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler
import lightgbm as lgb
import xgboost as xgb

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
except ImportError:  # Keep importable for model/export tests without parquet extras.
    pa = None  # type: ignore[assignment]
    ds = None  # type: ignore[assignment]

from export_plus import export_to_c_header, estimate_compact_tree_payload
from fold_split import load_roadmap, RoadmapError

try:
    import gesture_constants as _gesture_constants
except ImportError:
    _gesture_constants = None


# =================================================================================================
# USER/PROJECT CONFIGURATION — EDIT GLOBALS HERE, NO COMMAND-LINE ARGUMENTS
# =================================================================================================
FEATURES_VAR_TYPE = np.dtype(
    getattr(_gesture_constants, "FEATURES_VAR_TYPE", np.float32)
)
LABEL = "state"
#LABEL = "pointer_state_fixed"
GESTURE_GROUPS_ID = getattr(_gesture_constants, "GESTURE_GROUPS_ID", {"TAP": 1, "NONE": 0})
GESTURE_GROUPS_ID_TRAIN = getattr(_gesture_constants, "GESTURE_GROUPS_ID_TRAIN", {"TAP": 1})
MAX_MODEL_SIZE_BYTES = int(getattr(_gesture_constants, "MAX_MODEL_SIZE_BYTES", 102400))
MIN_REALISTIC_SIZE = int(getattr(_gesture_constants, "MIN_REALISTIC_SIZE", 6144))
N_FOLDS = int(getattr(_gesture_constants, "N_FOLDS", 5))
N_TRIALS = int(getattr(_gesture_constants, "N_TRIALS", 100))

FEATURE_PREFIXES = ("p_core->", "psd_features.")
NON_FEATURE_COLS = (LABEL, "class_id", "gesture_label", "timestamp", "source_file")

# Paths and run behavior.
TRAIN_FOLDER = "TAP_MODELS_F1_FILEFOLDS"
DATA_FOLDER = "DATA/DATA_TAP_F1/"
#DATA_FOLDER = "DATA/DATA_TAP_HERO_MIL/"
#DATA_FOLDER = "DATA/DATA_TAP_CLEANLAB/RFE"
#DATA_FOLDER = "DATA/DATA_LONG_PRESS_LABELED/labeled"
TO_RESUME = False
DEFAULT_THRESHOLD = 0.85

# ------------------- CANONICAL FOLD ROADMAP (NO RANDOM SPLITS) -------------------
# The source manifest is read-only. Do not generate/rebalance folds in training.
FOLD_ROADMAP_PATH = Path("DATA/fold_roadmap_files.json")
FOLD_ROADMAP_DATA_ROOT = Path("DATA")
FOLD_ROADMAP_DATASET = "DATA_TAP"  # or "DATA_LONG_PRESS" / "DATA_FIT_CHECK"
TEST_FOLD_ID = 0                    # choose 0..4; final test is NEVER used for tuning
# False is intentional: refined/RFE CSVs are often reserialized after organizing.
# Set True only for an unchanged, byte-identical snapshot of the organizer output.
FOLD_VERIFY_OUTPUT_SHA256 = False
# Optional stricter check if a participant alias audit has been completed.
FOLD_REQUIRE_VERIFIED_IDENTITIES = False
# Canonical directory scans use only the original manifest-listed CSV destinations;
# thus DATA/DATA_TAP/RFE, if present, will not be accidentally double-counted.
# Non-canonical directories (e.g., DATA/DATA_TAP/RFE) are resolved by original filename.
# Any unrecognized, missing, or multiply resolved source fails closed.


# Binary target selection. Change only this value to train another class.
# y_true = 1 exactly when: source_state > 0 AND class_id == TARGET_CLASS_ID.
# Every other class is forced to y_true = 0 even when its source state is active.
TARGET_CLASS_ID = 1

# Which studies to run, in order. Each family has its own independent multivariate-TPE study.
MODEL_FAMILY_ORDER: Tuple[str, ...] = ("LightGBM", "XGBoost")
MODEL_KEY_BY_TYPE = {"LightGBM": "lgbm", "XGBoost": "xgb"}
TRAIN_MODEL_FAMILIES: Tuple[str, ...] = ("LightGBM", "XGBoost")
LIGHTGBM_TRIALS = N_TRIALS
XGBOOST_TRIALS = N_TRIALS
# None means all groups from GESTURE_GROUPS_ID_TRAIN. Example: ("TAP",)
TRAIN_GROUPS: Tuple[str, ...] | None = None

# Rare class IDs have fewer than 5 independent people in this dataset.
# Validate binary {0,1} in every split; report missing detailed class IDs.
# Requiring all detailed class IDs in every fold would invalidate the fixed roadmap.
STRICT_ALL_CLASS_COVERAGE = False

# Fast tuning uses deterministic, class-preserving row caps for Optuna only.
# Final Pareto models are always trained and tested on the complete holdout split.
# Set either cap to 0 to use every row during Optuna CV.
CV_FOLDS = max(2, N_FOLDS - 1)  # all four non-test folds rotate as validation
OPTUNA_MAX_TRAIN_ROWS_PER_FOLD = 400_000
OPTUNA_MAX_VAL_ROWS_PER_FOLD = 160_000
OPTUNA_MIN_ROWS_PER_JOINT_STRATUM = 256

# Compact model constraints.
MIN_SELECTED_FEATURES = 2
MAX_SELECTED_FEATURES = 254
MAX_EXPORTED_PARETO_MODELS = 10
MAX_EXPORTED_PARETO_MODELS_PER_FAMILY = max(1, MAX_EXPORTED_PARETO_MODELS // 2)

# Diagnostics. Permutation importance is expensive and does not affect training/export.
PERMUTATION_MAX_ROWS = 25_000
PERMUTATION_TOP_RANKS = 3



PREPROCESSING_CHOICES: Tuple[str, ...] = (
    "none",
    "standard",
    "minmax_0_1",
    "minmax_m1_1",
    "maxabs",
    "robust_25_75",
    "robust_10_90",
    "normalize_l1",
    "normalize_l2",
    "normalize_max",
)

STUDY_SCHEMA_VERSION = 12
STUDY_NAME_SUFFIX = "fixed_file_roadmap_v12"
TPE_MULTIVARIATE = True
TPE_STARTUP_TRIALS = 8
TPE_N_EI_CANDIDATES = 24
SEARCH_SPACE_ENCODING = "separate_family_fast_cv_cap_xgb_lossguide_v1"

# Tree algorithms are usually limited by memory bandwidth, not logical thread count.
# Auto mode therefore budgets physical cores and uses fewer concurrent heavy trials.
LOGICAL_CPUS = max(1, psutil.cpu_count(logical=True) or multiprocessing.cpu_count())
PHYSICAL_CPUS = max(1, psutil.cpu_count(logical=False) or max(1, LOGICAL_CPUS // 2))
TRAINING_CPU_BUDGET = PHYSICAL_CPUS  # Set a positive integer to override auto physical-core budget.
XGBOOST_OPTUNA_JOBS = max(1, min(4, TRAINING_CPU_BUDGET))
XGBOOST_MODEL_THREADS = 0  # 0 = physical budget // concurrent Optuna trials
LIGHTGBM_OPTUNA_JOBS = max(1, min(4, TRAINING_CPU_BUDGET))
LIGHTGBM_MODEL_THREADS = 0
CSV_WORKERS = max(1, min(32, LOGICAL_CPUS))
FOLD_CACHE_MODE = "auto"  # "auto", "on", or "off"
FOLD_CACHE_MAX_RAM_FRACTION = 0.40
GC_EVERY_TRIALS = 25

# XGBoost grows histogram-based, leaf-limited trees under the shared flash budget.
PRUNE_ALL_ZERO_F1_AFTER_FOLDS = 2

MEMORY_RESERVE_BYTES = 4 * 1024**3
MEMORY_WAIT_TIMEOUT_SECONDS = 600
MEMORY_ALLOC_LOCK = threading.Lock()


# =================================================================================================
# SERIALIZATION / SMALL UTILITIES
# =================================================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


@dataclass(frozen=True)
class CsvProcessResult:
    file: str
    success: bool
    rows_written: int
    rows_dropped_nonfinite: int
    warning: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SplitSummary:
    train_rows: int
    test_rows: int
    train_groups: int
    test_groups: int
    classes: Tuple[int, ...]
    class_counts_train: Dict[int, int]
    class_counts_test: Dict[int, int]
    class_group_counts: Dict[int, int]
    class_distribution_train: Dict[int, float]
    class_distribution_test: Dict[int, float]


@dataclass(frozen=True)
class PreparedFold:
    train_indices: np.ndarray
    val_indices: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    class_ids_train: np.ndarray
    class_ids_val: np.ndarray
    feature_order: np.ndarray
    full_train_rows: int
    full_val_rows: int
    X_train_cached: np.ndarray | None = None
    X_val_cached: np.ndarray | None = None


def _require_pyarrow() -> None:
    if pa is None or ds is None:
        raise RuntimeError(
            "pyarrow is required for CSV-to-Parquet streaming. Install it with: pip install pyarrow"
        )


def _sqlite_wal_mode(db_path: Path) -> None:
    try:
        if db_path.exists():
            connection = sqlite3.connect(str(db_path), timeout=60)
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute("PRAGMA synchronous=NORMAL;")
            connection.execute("PRAGMA busy_timeout=60000;")
            connection.close()
    except Exception as exc:
        print(f"WARNING: Failed to enable SQLite WAL mode for {db_path}: {exc}")


def _normalized_distribution(labels: np.ndarray, classes: np.ndarray) -> Dict[int, float]:
    labels = np.asarray(labels)
    if labels.size == 0:
        return {int(value): 0.0 for value in classes}
    return {
        int(value): float(np.count_nonzero(labels == value) / labels.size)
        for value in classes
    }


def _class_counts(labels: np.ndarray, classes: np.ndarray) -> Dict[int, int]:
    labels = np.asarray(labels)
    return {int(value): int(np.count_nonzero(labels == value)) for value in classes}


def _class_group_counts(
    class_ids: np.ndarray, groups: np.ndarray, classes: np.ndarray
) -> Dict[int, int]:
    return {
        int(value): int(np.unique(groups[class_ids == value]).size)
        for value in classes
    }


def _joint_stratification_labels(class_ids: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Encode (detailed class, binary state) pairs for target-aware balancing."""

    pairs = list(zip(np.asarray(class_ids).tolist(), np.asarray(y).tolist()))
    unique_pairs = sorted(set((int(class_id), int(state)) for class_id, state in pairs))
    pair_to_code = {pair: index for index, pair in enumerate(unique_pairs)}
    return np.fromiter(
        (pair_to_code[(int(class_id), int(state))] for class_id, state in pairs),
        dtype=np.int32,
        count=len(pairs),
    )


def _missing_classes(
    class_ids: np.ndarray, indices: np.ndarray, required_classes: np.ndarray
) -> List[int]:
    present = set(int(value) for value in np.unique(class_ids[indices]))
    return [int(value) for value in required_classes if int(value) not in present]


def _require_all_class_coverage(
    class_ids: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    required_classes: np.ndarray,
    label: str,
) -> None:
    missing = _missing_classes(class_ids, indices, required_classes)
    if missing:
        raise ValueError(f"{label} is missing class IDs {missing}.")
    binary_values = np.unique(y[indices]).tolist()
    if binary_values != [0, 1]:
        raise ValueError(
            f"{label} binary target contains {binary_values}; expected both 0 and 1."
        )


def resolve_family_cpu(model_type: str) -> Tuple[int, int]:
    """Return (concurrent Optuna jobs, model threads) from global family settings.

    Tree training saturates memory bandwidth before SMT/hyperthreads. The automatic budget
    therefore uses physical cores and avoids the V8 ``16 x 6`` oversubscription pattern.
    """

    budget = max(1, int(TRAINING_CPU_BUDGET))
    if model_type == "XGBoost":
        jobs = max(1, int(XGBOOST_OPTUNA_JOBS))
        requested = int(XGBOOST_MODEL_THREADS)
    elif model_type == "LightGBM":
        jobs = max(1, int(LIGHTGBM_OPTUNA_JOBS))
        requested = int(LIGHTGBM_MODEL_THREADS)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    threads = requested if requested > 0 else max(1, budget // jobs)
    # Never allocate more inner workers than the physical-core budget.
    threads = max(1, min(budget, threads))
    return jobs, threads


def _periodic_gc(trial_number: int) -> None:
    if GC_EVERY_TRIALS > 0 and (trial_number + 1) % GC_EVERY_TRIALS == 0:
        gc.collect()


# =================================================================================================
# CANONICAL ROADMAP FILE SELECTION / DATA IDENTITY
# =================================================================================================
def _configured_roadmap():
    return load_roadmap(
        roadmap_path=FOLD_ROADMAP_PATH,
        data_root=FOLD_ROADMAP_DATA_ROOT,
        require_verified_identities=FOLD_REQUIRE_VERIFIED_IDENTITIES,
    )


def _roadmap_sha256() -> str:
    return hashlib.sha256(Path(FOLD_ROADMAP_PATH).read_bytes()).hexdigest()


def _roadmap_file_selection(data_folder: Path) -> list[tuple[Path, int, int, str]]:
    """Return only trainable files with (path, fold, expected_class, source_file_id).

    Do not silently train on another random set of files. The canonical dataset
    directory is an allow-list from the roadmap; other directories can use the
    original filenames, but every source must be uniquely recognized.
    """
    client = _configured_roadmap()
    if client.n_folds != int(N_FOLDS):
        raise ValueError(
            f"Roadmap contains {client.n_folds} folds but N_FOLDS={N_FOLDS}. "
            "Set N_FOLDS to match your organizer manifest."
        )
    if FOLD_ROADMAP_DATASET not in client.datasets:
        raise ValueError(
            f"Unknown FOLD_ROADMAP_DATASET={FOLD_ROADMAP_DATASET!r}; "
            f"available={client.datasets}"
        )
    official_root = (client.data_root / FOLD_ROADMAP_DATASET).resolve()
    actual_root = data_folder.resolve()
    records = []
    if actual_root == official_root:
        # This is the official organized dataset. Use only original category
        # paths, not outputs nested under its RFE/other processing directories.
        records = [
            (r.path, r.fold_id, r.class_id, r.file_id)
            for r in client.get_records(dataset=FOLD_ROADMAP_DATASET)
        ]
        found = {p.resolve() for p in data_folder.rglob("*.csv") if p.is_file()}
        allowed = {p.resolve() for p, _, _, _ in records}
        ignored = len(found - allowed)
        if ignored:
            print(
                f"[ROADMAP] Ignoring {ignored} CSVs not listed as trainable "
                f"{FOLD_ROADMAP_DATASET} destinations (including excluded "
                "duplicates or derived/RFE files)."
            )
    else:
        # Allow training on a transformed subtree while preserving original
        # file names (e.g. a feature-selected RFE directory).
        discovered = sorted(
            p for p in data_folder.rglob("*.csv")
            if p.is_file()
            and not any(
                part.startswith("_")
                for part in p.relative_to(data_folder).parts[:-1]
            )
        )
        for path in discovered:
            info = client.get_file_info(path, dataset=FOLD_ROADMAP_DATASET)
            if not info.exclude_from_training:
                records.append((path, info.fold_id, info.class_id, info.file_id))

    if not records:
        raise FileNotFoundError(
            f"No trainable roadmap-listed {FOLD_ROADMAP_DATASET} files in {data_folder}"
        )
    by_id: dict[str, Path] = {}
    for path, _, _, file_id in records:
        if file_id in by_id:
            raise RoadmapError(
                f"One source recording occurs twice in input: {by_id[file_id]} AND {path}. "
                "Use one CSV per source file, not combined original/RFE copies."
            )
        by_id[file_id] = path
    records.sort(key=lambda r: r[0].as_posix())
    absent = [str(r[0]) for r in records if not r[0].is_file()]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} roadmap-listed input files do not exist: {absent[:4]}"
        )
    # The public API explicitly verifies manifest-defined participant and
    # recording-content separation. Test membership is never reassigned.
    client.get_split(test_fold=int(TEST_FOLD_ID), dataset=FOLD_ROADMAP_DATASET)
    if FOLD_VERIFY_OUTPUT_SHA256:
        # The manifest's digest is only for the organizer's ORIGINAL CSV output.
        # Enforce exact byte identity for every selected source when requested.
        for path, _, _, _file_id in records:
            expected = client.get_file_info(
                path, dataset=FOLD_ROADMAP_DATASET
            ).organized_content_sha256
            if expected is None:
                raise ValueError(f"No organized CSV SHA256 for {path}")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(
                    f"CSV content changed since fold roadmap export: {path}. "
                    "Turn FOLD_VERIFY_OUTPUT_SHA256=False for intentionally processed CSVs."
                )
    fold_counts = {i: sum(x[1] == i for x in records) for i in range(client.n_folds)}
    if not all(fold_counts.values()):
        raise ValueError(
            f"At least one fixed roadmap fold is empty in {data_folder}: {fold_counts}"
        )
    print(
        f"[ROADMAP] dataset={FOLD_ROADMAP_DATASET}, "
        f"files={len(records):,}, excluded duplicates removed; "
        f"fold file counts={fold_counts}, test_fold={TEST_FOLD_ID}, "
        f"identity_reviews={len(client.identity_review_required)}"
    )
    return records


def _dataset_fingerprint(
    csv_files: Sequence[Path],
    feature_names: Sequence[str],
    target_class_id: int,
) -> str:
    """Fingerprint raw files plus every setting that changes generated labels."""

    digest = hashlib.sha256()
    digest.update(f"study_schema={STUDY_SCHEMA_VERSION}\n".encode())
    digest.update(f"dtype={FEATURES_VAR_TYPE.str}\n".encode())
    digest.update(f"target_class_id={int(target_class_id)}\n".encode())
    digest.update(f"roadmap_dataset={FOLD_ROADMAP_DATASET}\n".encode())
    digest.update(f"roadmap_sha256={_roadmap_sha256()}\n".encode())
    digest.update(b"target_rule=source_state_gt_zero_and_class_equals_target\n")
    digest.update("\n".join(feature_names).encode("utf-8"))
    for path in csv_files:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(f"|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def infer_feature_list_from_columns(columns: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for column in columns:
        if not isinstance(column, str) or column in NON_FEATURE_COLS:
            continue
        if column.startswith(FEATURE_PREFIXES):
            result.append(column)
    return result


def infer_feature_list_from_df(df: pd.DataFrame) -> List[str]:
    return infer_feature_list_from_columns(df.columns)


def resolve_target_class_id() -> int:
    """Validate and return the single globally configured binary target class."""

    value = TARGET_CLASS_ID
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"TARGET_CLASS_ID must be an integer class ID, got {value!r}.")
    target_class_id = int(value)
    limits = np.iinfo(np.int16)
    if not limits.min <= target_class_id <= limits.max:
        raise ValueError(
            f"TARGET_CLASS_ID={target_class_id} must fit int16 "
            f"[{limits.min}, {limits.max}]."
        )
    return target_class_id


def resolve_positive_class_id(group_name: str | None = None) -> int:
    """Backward-compatible alias; the group no longer chooses the target class."""

    del group_name
    return resolve_target_class_id()


def format_importances(names: Sequence[str], importances: Any) -> Dict[str, float]:
    if importances is None:
        return {}
    values = np.asarray(importances, dtype=np.float64).reshape(-1)
    if len(names) != values.size:
        raise ValueError(f"Importance width mismatch: {len(names)} names vs {values.size} values.")
    pairs = {
        name: round(float(value), 8)
        for name, value in zip(names, values)
        if math.isfinite(float(value))
    }
    return dict(sorted(pairs.items(), key=lambda item: (-item[1], item[0])))


# =================================================================================================
# DATA LOADING / VALIDATION / MEMMAP
# =================================================================================================
def _coerce_integer_series(series: pd.Series, column: str, file_name: str) -> np.ndarray:
    """Return validated integer labels as int64 without silent narrowing/wraparound."""

    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{file_name}: {column} contains NaN/non-numeric values.")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"{file_name}: {column} must contain integer values.")
    int64_limits = np.iinfo(np.int64)
    if np.any(rounded < int64_limits.min) or np.any(rounded > int64_limits.max):
        raise ValueError(f"{file_name}: {column} exceeds int64 range.")
    return rounded.astype(np.int64, copy=False)


def derive_binary_target(
    class_ids: np.ndarray,
    positive_class_id: int,
    source_state: np.ndarray | None = None,
) -> np.ndarray:
    """Build the binary target for one gesture without promoting other classes."""

    classes = np.asarray(class_ids)
    if source_state is None:
        return (classes == int(positive_class_id)).astype(np.int8, copy=False)
    state_values = np.asarray(source_state)
    if state_values.shape != classes.shape:
        raise ValueError(
            f"state/class_id shape mismatch: {state_values.shape} vs {classes.shape}."
        )
    return (
        (state_values > 0) & (classes == int(positive_class_id))
    ).astype(np.int8, copy=False)


def _process_file_worker(
    csv_file_tuple: Tuple[int, Path, int, int],
    expected_features: Sequence[str],
    dataset_dir: Path,
    data_root: Path,
    positive_class_id: int,
) -> CsvProcessResult:
    index, csv_file, roadmap_fold, expected_class_id = csv_file_tuple
    file_label = str(csv_file)
    try:
        header = pd.read_csv(csv_file, nrows=0)
        header_columns = list(header.columns)
        missing = [feature for feature in expected_features if feature not in header_columns]
        if missing:
            preview = ", ".join(missing[:8])
            return CsvProcessResult(
                file=file_label,
                success=False,
                rows_written=0,
                rows_dropped_nonfinite=0,
                error=f"Missing {len(missing)} schema features: {preview}",
            )

        available_non_features = [name for name in NON_FEATURE_COLS if name in header_columns]
        usecols = list(expected_features) + available_non_features
        frame = pd.read_csv(csv_file, usecols=usecols, low_memory=False)
        frame = frame.loc[pd.to_numeric(frame[LABEL], errors="coerce").fillna(0).to_numpy() >= 0].reset_index(drop=True) if LABEL in frame.columns else frame  # unknown rows (state -1) never train, tune or score
        if frame.empty:
            return CsvProcessResult(file_label, False, 0, 0, error="CSV has no data rows.")

        feature_frame = frame.loc[:, list(expected_features)].apply(pd.to_numeric, errors="coerce")
        feature_values = feature_frame.to_numpy(dtype=np.float32, copy=True)
        finite_rows = np.isfinite(feature_values).all(axis=1)
        dropped_nonfinite = int(np.count_nonzero(~finite_rows))
        if dropped_nonfinite:
            feature_values = feature_values[finite_rows]
            frame = frame.loc[finite_rows].reset_index(drop=True)
        if feature_values.shape[0] == 0:
            return CsvProcessResult(
                file_label,
                False,
                0,
                dropped_nonfinite,
                error="All rows contain NaN/Inf in at least one feature.",
            )

        warning_text = None
        if np.issubdtype(FEATURES_VAR_TYPE, np.integer):
            limits = np.iinfo(FEATURES_VAR_TYPE)
            observed_min = float(feature_values.min(initial=0.0))
            observed_max = float(feature_values.max(initial=0.0))
            if observed_min < limits.min or observed_max > limits.max:
                warning_text = (
                    f"feature range [{observed_min:.6g}, {observed_max:.6g}] exceeds "
                    f"{FEATURES_VAR_TYPE}; values were clipped"
                )
            np.clip(feature_values, limits.min, limits.max, out=feature_values)
            stored_features = feature_values.astype(FEATURES_VAR_TYPE, copy=False)
        else:
            limits = np.finfo(FEATURES_VAR_TYPE)
            np.clip(feature_values, limits.min, limits.max, out=feature_values)
            stored_features = feature_values.astype(FEATURES_VAR_TYPE, copy=False)

        output = pd.DataFrame(stored_features, columns=list(expected_features))

        if "class_id" not in frame.columns:
            return CsvProcessResult(
                file_label,
                False,
                0,
                dropped_nonfinite,
                error=(
                    "class_id is required for configurable TARGET_CLASS_ID training; "
                    "state alone cannot preserve dynamic multiclass split coverage."
                ),
            )
        class_ids_wide = _coerce_integer_series(frame["class_id"], "class_id", file_label)

        class_limits = np.iinfo(np.int16)
        if np.any(class_ids_wide < class_limits.min) or np.any(class_ids_wide > class_limits.max):
            return CsvProcessResult(
                file_label,
                False,
                0,
                dropped_nonfinite,
                error=(
                    f"class_id must fit int16 [{class_limits.min}, {class_limits.max}]; "
                    f"found range [{int(class_ids_wide.min())}, {int(class_ids_wide.max())}]"
                ),
            )
        class_ids = class_ids_wide.astype(np.int16, copy=False)
        # Files routed by org.py have exactly one class. Refuse silent
        # dataset-class drift in downstream training input.
        if np.any(class_ids != expected_class_id):
            observed = np.unique(class_ids).tolist()
            raise ValueError(
                f"{file_label}: CSV class_id {observed} disagrees with "
                f"roadmap class_id={expected_class_id}."
            )

        if LABEL not in frame.columns:
            raise ValueError(
                f"Refined training requires {LABEL!r} in every CSV; "
                "refusing to treat all target-class samples as active labels"
            )
        state_wide = _coerce_integer_series(frame[LABEL], LABEL, file_label)
        # Preserve the refined Cleanlab state as the only supervision source.
        state = derive_binary_target(class_ids, positive_class_id, state_wide)

        output[LABEL] = state
        output["class_id"] = class_ids
        output["roadmap_fold_id"] = np.int8(roadmap_fold)
        try:
            relative_source = csv_file.resolve().relative_to(data_root.resolve()).as_posix()
        except ValueError:
            relative_source = csv_file.resolve().as_posix()
        output["source_file"] = relative_source

        shard_path = dataset_dir / f"{index:06d}.parquet"
        output.to_parquet(shard_path, engine="pyarrow", compression="zstd", index=False)
        return CsvProcessResult(
            file=file_label,
            success=True,
            rows_written=len(output),
            rows_dropped_nonfinite=dropped_nonfinite,
            warning=warning_text,
        )
    except Exception as exc:
        return CsvProcessResult(
            file=file_label,
            success=False,
            rows_written=0,
            rows_dropped_nonfinite=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def load_and_validate_data(
    group_name: str,
    output_dir: Path,
    data_folder: Path,
    *,
    csv_workers: int = CSV_WORKERS,
) -> Tuple[Path, List[str], Path, str, List[Path]]:
    _require_pyarrow()
    print(f"--- Loading and validating data for gesture group: {group_name} ---")

    positive_class_id = resolve_target_class_id()
    roadmap_records = _roadmap_file_selection(data_folder)
    all_csv_files = [r[0] for r in roadmap_records]

    schema_path = output_dir / f"{group_name}_feature_schema.json"
    if schema_path.exists():
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        expected_features = [str(value) for value in schema["feature_names"]]
        if not expected_features:
            raise ValueError(f"Existing schema {schema_path} has no feature_names.")
        if len(set(expected_features)) != len(expected_features):
            raise ValueError(f"Existing schema {schema_path} contains duplicate feature names.")

        expected_map = {str(index): name for index, name in enumerate(expected_features)}
        raw_map = schema.get("feature_map")
        if raw_map is not None:
            normalized_map = {str(key): str(value) for key, value in raw_map.items()}
            if normalized_map != expected_map:
                raise ValueError(
                    f"Existing schema {schema_path} has a feature_map that disagrees with feature_names. "
                    "Delete/fix the schema before training to prevent wrong firmware accessors."
                )
        declared_count = schema.get("feature_count")
        if declared_count is not None and int(declared_count) != len(expected_features):
            raise ValueError(
                f"Existing schema {schema_path} declares feature_count={declared_count}, "
                f"but contains {len(expected_features)} names."
            )

        # Canonicalize older but otherwise valid schemas so downstream export
        # always receives a complete, consistent mapping.
        canonical_schema = dict(schema)
        # Target metadata is canonicalized rather than treated as part of the feature schema.
        # Changing TARGET_CLASS_ID therefore rebuilds label-dependent caches automatically.
        canonical_schema.update(
            {
                "feature_count": len(expected_features),
                "feature_names": expected_features,
                "feature_map": expected_map,
                "storage_dtype": FEATURES_VAR_TYPE.str,
                "target_class_id": positive_class_id,
                "positive_class_id": positive_class_id,
                "target_rule": "source_state_gt_zero_and_class_equals_target",
            }
        )
        if canonical_schema != schema:
            schema_path.write_text(json.dumps(canonical_schema, indent=2), encoding="utf-8")
        print(f"[SCHEMA] Loaded {len(expected_features)} features from {schema_path}")
    else:
        expected_features = []
        source_file_name = None
        for candidate in all_csv_files:
            try:
                header = pd.read_csv(candidate, nrows=0)
            except Exception:
                continue
            candidate_features = infer_feature_list_from_columns(header.columns)
            if candidate_features:
                expected_features = candidate_features
                source_file_name = str(candidate)
                break
        if not expected_features:
            raise FileNotFoundError(
                f"Could not create a schema: no columns started with {FEATURE_PREFIXES} in {data_folder}."
            )
        if len(set(expected_features)) != len(expected_features):
            raise ValueError("Inferred feature columns contain duplicate names.")

        schema = {
            "source_file": source_file_name,
            "feature_count": len(expected_features),
            "feature_names": expected_features,
            "feature_map": {str(index): name for index, name in enumerate(expected_features)},
            "storage_dtype": FEATURES_VAR_TYPE.str,
            "target_class_id": positive_class_id,
            "positive_class_id": positive_class_id,
            "target_rule": "source_state_gt_zero_and_class_equals_target",
        }
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        print(f"[SCHEMA] Created {schema_path} with {len(expected_features)} features")

    if len(expected_features) < MIN_SELECTED_FEATURES:
        raise ValueError(
            f"At least {MIN_SELECTED_FEATURES} feature columns are required; found {len(expected_features)}."
        )
    if len(expected_features) > 65535:
        raise ValueError("Feature schema exceeds uint16 original-index storage.")

    fingerprint = _dataset_fingerprint(all_csv_files, expected_features, positive_class_id)
    dataset_dir = output_dir / f"{group_name}_pq"
    dataset_manifest_path = output_dir / f"{group_name}_dataset_cache.json"
    expected_manifest = {
        "dataset_fingerprint": fingerprint,
        "feature_names": list(expected_features),
        "storage_dtype": FEATURES_VAR_TYPE.str,
        "target_class_id": int(positive_class_id),
        "positive_class_id": int(positive_class_id),
        "target_rule": "source_state_gt_zero_and_class_equals_target",
        "study_schema_version": STUDY_SCHEMA_VERSION,
        "roadmap_sha256": _roadmap_sha256(),
        "roadmap_dataset": FOLD_ROADMAP_DATASET,
        "roadmap_csv_count": len(all_csv_files),
    }
    if dataset_manifest_path.exists() and dataset_dir.exists():
        try:
            cached_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            cache_matches = all(
                cached_manifest.get(key) == value for key, value in expected_manifest.items()
            )
            parquet_files = list(dataset_dir.glob("*.parquet"))
            if cache_matches and parquet_files:
                cached_rows = int(ds.dataset(dataset_dir, format="parquet").count_rows())
                if cached_rows > 0:
                    print(
                        f"[DATA CACHE] Reusing {len(parquet_files)} Parquet shards "
                        f"with {cached_rows:,} rows from {dataset_dir}"
                    )
                    return dataset_dir, expected_features, schema_path, fingerprint, all_csv_files
        except Exception as exc:
            print(f"[DATA CACHE] Cache validation failed; rebuilding: {exc}")

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if dataset_manifest_path.exists():
        dataset_manifest_path.unlink()

    results: List[CsvProcessResult] = []
    worker_items = [
        (i, path, fold, expected_class)
        for i, (path, fold, expected_class, _file_id) in enumerate(roadmap_records)
    ]
    max_workers = max(1, min(int(csv_workers), len(all_csv_files)))
    if max_workers == 1:
        for item in worker_items:
            results.append(
                _process_file_worker(
                    item, expected_features, dataset_dir, data_folder, positive_class_id
                )
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _process_file_worker,
                    item,
                    expected_features,
                    dataset_dir,
                    data_folder,
                    positive_class_id,
                )
                for item in worker_items
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

    successes = [result for result in results if result.success]
    failures = [result for result in results if not result.success]
    warnings_found = [result for result in successes if result.warning or result.rows_dropped_nonfinite]

    if failures:
        report_path = output_dir / f"dropped_files_{group_name}.txt"
        report_path.write_text(
            "\n".join(f"{result.file}: {result.error}" for result in sorted(failures, key=lambda x: x.file)),
            encoding="utf-8",
        )
        print(f"[DATA] Dropped {len(failures)} files; details: {report_path}")
        # A missing roadmap-listed recording changes the meaning of the fixed
        # split and would make reproducibility claims false. Fail, don't skip.
        raise RuntimeError(
            f"Refusing to train with {len(failures)} invalid roadmap-listed files; "
            f"inspect {report_path}. First error: {failures[0].file}: {failures[0].error}"
        )
    if warnings_found:
        warning_path = output_dir / f"data_warnings_{group_name}.txt"
        warning_path.write_text(
            "\n".join(
                f"{result.file}: dropped_nonfinite={result.rows_dropped_nonfinite}; {result.warning or ''}"
                for result in sorted(warnings_found, key=lambda x: x.file)
            ),
            encoding="utf-8",
        )
        print(f"[DATA] Warnings for {len(warnings_found)} files; details: {warning_path}")

    if not successes:
        raise ValueError("No valid data files remain after schema and numeric validation.")

    total_rows = sum(result.rows_written for result in successes)
    completed_manifest = {
        **expected_manifest,
        "valid_files": len(successes),
        "dropped_files": len(failures),
        "usable_rows": int(total_rows),
        "parquet_shards": len(successes),
    }
    dataset_manifest_path.write_text(
        json.dumps(completed_manifest, indent=2, cls=NumpyEncoder),
        encoding="utf-8",
    )
    print(
        f"[DATA] Valid files: {len(successes)}, dropped: {len(failures)}, "
        f"usable rows: {total_rows:,}"
    )
    return dataset_dir, expected_features, schema_path, fingerprint, all_csv_files


def build_memmaps_from_parquet(
    dataset_dir: Path,
    feature_names: Sequence[str],
    x_mm_path: Path,
    *,
    batch_rows: int = 200_000,
) -> Tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require_pyarrow()
    dataset = ds.dataset(dataset_dir, format="parquet")
    total_rows = int(dataset.count_rows())
    feature_count = len(feature_names)
    if total_rows <= 0:
        raise ValueError(f"Parquet dataset {dataset_dir} contains no rows.")
    if feature_count <= 0:
        raise ValueError("Cannot create a memmap with zero features.")

    x_mm_path.parent.mkdir(parents=True, exist_ok=True)
    if x_mm_path.exists():
        x_mm_path.unlink()
    X_mm = np.memmap(
        x_mm_path,
        dtype=FEATURES_VAR_TYPE,
        mode="w+",
        shape=(total_rows, feature_count),
    )
    y = np.empty(total_rows, dtype=np.int8)
    class_ids = np.empty(total_rows, dtype=np.int16)
    groups = np.empty(total_rows, dtype=np.int32)
    roadmap_folds = np.empty(total_rows, dtype=np.int8)

    unique_sources: set[str] = set()
    for batch in dataset.scanner(columns=["source_file"], batch_size=batch_rows).to_batches():
        source_values = pa.Table.from_batches([batch])["source_file"].to_pylist()
        if any(value is None for value in source_values):
            raise ValueError("source_file contains null values.")
        unique_sources.update(str(value) for value in source_values)
    source_to_code = {source: code for code, source in enumerate(sorted(unique_sources))}
    if len(source_to_code) < 2:
        raise ValueError(
            "At least two distinct source files/groups are required for leakage-free train/test splitting."
        )

    scan_columns = list(feature_names) + [LABEL, "class_id", "source_file", "roadmap_fold_id"]
    scanner = dataset.scanner(columns=scan_columns, batch_size=batch_rows)
    write_position = 0

    for record_batch in scanner.to_batches():
        table = pa.Table.from_batches([record_batch])
        feature_blocks = [
            table[name].to_numpy(zero_copy_only=False).astype(FEATURES_VAR_TYPE, copy=False)
            for name in feature_names
        ]
        block = np.column_stack(feature_blocks)
        row_count = block.shape[0]
        if not np.all(np.isfinite(block)):
            raise ValueError("Non-finite feature reached memmap construction after validation.")
        X_mm[write_position : write_position + row_count] = block

        state_block = table[LABEL].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        class_block = table["class_id"].to_numpy(zero_copy_only=False).astype(np.int16, copy=False)
        if not np.all(np.isin(state_block, (0, 1))):
            raise ValueError("Parquet state column is not binary.")
        y[write_position : write_position + row_count] = state_block
        class_ids[write_position : write_position + row_count] = class_block
        fold_block = table["roadmap_fold_id"].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        if np.any((fold_block < 0) | (fold_block >= N_FOLDS)):
            raise ValueError("Parquet includes invalid fixed roadmap fold IDs.")
        roadmap_folds[write_position : write_position + row_count] = fold_block

        source_values = table["source_file"].to_pylist()
        groups[write_position : write_position + row_count] = np.fromiter(
            (source_to_code[str(source)] for source in source_values),
            dtype=np.int32,
            count=row_count,
        )
        write_position += row_count

    if write_position != total_rows:
        raise RuntimeError(f"Memmap write mismatch: expected {total_rows}, wrote {write_position}.")
    X_mm.flush()

    if int(groups.max()) <= np.iinfo(np.int16).max:
        groups = groups.astype(np.int16, copy=False)
    X_readonly = np.memmap(
        x_mm_path,
        dtype=FEATURES_VAR_TYPE,
        mode="r",
        shape=(total_rows, feature_count),
    )
    # A source file must have one consistent fold, including after row filtering.
    order = np.argsort(groups, kind="stable")
    if np.any((groups[order][1:] == groups[order][:-1]) &
              (roadmap_folds[order][1:] != roadmap_folds[order][:-1])):
        raise RuntimeError("A source file has rows in multiple roadmap folds!")
    return X_readonly, y, class_ids, groups, roadmap_folds


def load_or_build_memmaps(
    dataset_dir: Path,
    feature_names: Sequence[str],
    memmap_dir: Path,
    group_name: str,
    dataset_fingerprint: str,
) -> Tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reuse memmaps/labels when the validated dataset has not changed."""

    memmap_dir.mkdir(parents=True, exist_ok=True)
    x_path = memmap_dir / f"{group_name}_X_mm.dat"
    y_path = memmap_dir / f"{group_name}_y.npy"
    class_path = memmap_dir / f"{group_name}_class_ids.npy"
    groups_path = memmap_dir / f"{group_name}_groups.npy"
    fold_path = memmap_dir / f"{group_name}_roadmap_folds.npy"
    manifest_path = memmap_dir / f"{group_name}_memmap_cache.json"
    expected = {
        "dataset_fingerprint": dataset_fingerprint,
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "feature_dtype": FEATURES_VAR_TYPE.str,
        "study_schema_version": STUDY_SCHEMA_VERSION,
        "target_class_id": resolve_target_class_id(),
        "target_rule": "source_state_gt_zero_and_class_equals_target",
        "roadmap_sha256": _roadmap_sha256(),
        "roadmap_dataset": FOLD_ROADMAP_DATASET,
    }

    if manifest_path.exists() and all(
        path.exists() for path in (x_path, y_path, class_path, groups_path, fold_path)
    ):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if all(manifest.get(key) == value for key, value in expected.items()):
                rows = int(manifest["rows"])
                expected_bytes = rows * len(feature_names) * FEATURES_VAR_TYPE.itemsize
                if x_path.stat().st_size == expected_bytes:
                    X_cached = np.memmap(
                        x_path,
                        dtype=FEATURES_VAR_TYPE,
                        mode="r",
                        shape=(rows, len(feature_names)),
                    )
                    y_cached = np.load(y_path, mmap_mode="r")
                    class_cached = np.load(class_path, mmap_mode="r")
                    groups_cached = np.load(groups_path, mmap_mode="r")
                    folds_cached = np.load(fold_path, mmap_mode="r")
                    if len(y_cached) == len(class_cached) == len(groups_cached) == len(folds_cached) == rows:
                        print(
                            f"[MEMMAP CACHE] Reusing {rows:,} rows x "
                            f"{len(feature_names)} features from {memmap_dir}"
                        )
                        return X_cached, y_cached, class_cached, groups_cached, folds_cached
        except Exception as exc:
            print(f"[MEMMAP CACHE] Cache validation failed; rebuilding: {exc}")

    X_mm, y, class_ids, groups, roadmap_folds = build_memmaps_from_parquet(
        dataset_dir, feature_names, x_path
    )
    np.save(y_path, np.asarray(y))
    np.save(class_path, np.asarray(class_ids))
    np.save(groups_path, np.asarray(groups))
    np.save(fold_path, np.asarray(roadmap_folds))
    manifest = {
        **expected,
        "rows": int(len(y)),
        "class_dtype": np.asarray(class_ids).dtype.str,
        "group_dtype": np.asarray(groups).dtype.str,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return X_mm, y, class_ids, groups, roadmap_folds


# =================================================================================================
# FIXED PERSON-DISJOINT ROADMAP SPLITS (NO SHUFFLING / REASSIGNMENT)
# =================================================================================================
def make_roadmap_holdout_and_cv(
    class_ids: np.ndarray,
    y: np.ndarray,
    source_groups: np.ndarray,
    roadmap_folds: np.ndarray,
    *,
    test_fold: int,
    requested_folds: int,
) -> tuple[np.ndarray, np.ndarray, SplitSummary, list[tuple[np.ndarray, np.ndarray]], list[int]]:
    """Use assigned roadmap folds only. Returns outer holdout and inner CV."""
    client = _configured_roadmap()
    groups = np.asarray(source_groups)
    folds = np.asarray(roadmap_folds)
    class_ids = np.asarray(class_ids)
    y = np.asarray(y)
    if not (len(groups) == len(folds) == len(class_ids) == len(y)):
        raise ValueError("groups, roadmap_folds, class_ids and y must have equal length")
    expected_folds = set(range(client.n_folds))
    actual_folds = set(int(x) for x in np.unique(folds))
    if actual_folds != expected_folds:
        raise ValueError(f"Input missing/has extra roadmap folds: {actual_folds}, expected {expected_folds}")
    if not 0 <= test_fold < client.n_folds:
        raise ValueError(f"TEST_FOLD_ID must be in range 0..{client.n_folds - 1}")
    available_cv_folds = [i for i in range(client.n_folds) if i != test_fold]
    if not 2 <= int(requested_folds) <= len(available_cv_folds):
        raise ValueError(f"CV_FOLDS must be 2..{len(available_cv_folds)} for this roadmap")

    # Source-level group disjointness is checked independently of the manifest.
    # If a per-file fold ID was corrupted in CSV -> Parquet processing, stop.
    source_fold_pairs = np.unique(np.column_stack((groups, folds)), axis=0)
    if len(source_fold_pairs) != len(np.unique(groups)):
        raise AssertionError("Same source file has rows in more than one fold")

    classes = np.unique(class_ids)
    train_indices = np.flatnonzero(folds != test_fold).astype(np.int64)
    test_indices = np.flatnonzero(folds == test_fold).astype(np.int64)
    if not len(train_indices) or not len(test_indices):
        raise ValueError("Empty train or test side")
    if np.unique(y[train_indices]).tolist() != [0, 1]:
        raise ValueError("Roadmap train pool must contain both binary target values")
    if np.unique(y[test_indices]).tolist() != [0, 1]:
        raise ValueError(
            f"Roadmap test fold {test_fold} lacks binary 0 or 1. "
            "Choose another test fold or collect more positive samples; do not reshuffle people."
        )
    if np.intersect1d(groups[train_indices], groups[test_indices]).size:
        raise AssertionError("Holdout leaked a source recording")

    cv_validation_folds = available_cv_folds[:int(requested_folds)]
    cv_splits = []
    for validation_fold in cv_validation_folds:
        validation_indices = np.flatnonzero(folds == validation_fold).astype(np.int64)
        cv_train_indices = np.flatnonzero(
            (folds != test_fold) & (folds != validation_fold)
        ).astype(np.int64)
        if not len(validation_indices) or not len(cv_train_indices):
            raise ValueError(f"CV validation fold {validation_fold} is empty")
        if np.intersect1d(groups[cv_train_indices], groups[validation_indices]).size:
            raise AssertionError(f"CV fold {validation_fold} leaked a source recording")
        for side_name, side in (("train", cv_train_indices), ("validation", validation_indices)):
            if np.unique(y[side]).tolist() != [0, 1]:
                raise ValueError(
                    f"CV validation roadmap fold {validation_fold} {side_name} "
                    "does not contain binary target values [0, 1]."
                )
        cv_splits.append((cv_train_indices, validation_indices))

    split_summary = SplitSummary(
        train_rows=len(train_indices),
        test_rows=len(test_indices),
        train_groups=int(np.unique(groups[train_indices]).size),
        test_groups=int(np.unique(groups[test_indices]).size),
        classes=tuple(int(value) for value in classes),
        class_counts_train=_class_counts(class_ids[train_indices], classes),
        class_counts_test=_class_counts(class_ids[test_indices], classes),
        class_group_counts=_class_group_counts(class_ids, groups, classes),
        class_distribution_train=_normalized_distribution(class_ids[train_indices], classes),
        class_distribution_test=_normalized_distribution(class_ids[test_indices], classes),
    )
    all_sides = [("final train", train_indices), ("holdout test", test_indices)]
    for fid, (tr, va) in zip(cv_validation_folds, cv_splits):
        all_sides.extend(((f"CV fold {fid} train", tr), (f"CV fold {fid} validation", va)))
    missing_warnings = []
    for label, side in all_sides:
        missing = _missing_classes(class_ids, side, classes)
        if missing:
            missing_warnings.append((label, missing))
    if missing_warnings and STRICT_ALL_CLASS_COVERAGE:
        raise ValueError(
            f"Rare class IDs are absent in fixed roadmap splits: {missing_warnings}. "
            "Disable STRICT_ALL_CLASS_COVERAGE rather than randomizing participant folds."
        )
    for label, missing in missing_warnings:
        print(f"[ROADMAP COVERAGE] {label} lacks detailed class IDs {missing}; no regrouping performed")
    print(
        f"[ROADMAP SPLIT] final test=fold {test_fold}, "
        f"train folds={available_cv_folds}, "
        f"CV validation folds={cv_validation_folds}; "
        f"train={len(train_indices):,}, test={len(test_indices):,} rows; "
        "all participants fixed by JSON"
    )
    return train_indices, test_indices, split_summary, cv_splits, cv_validation_folds


def write_roadmap_split_report(
    path: Path,
    class_ids: np.ndarray,
    y: np.ndarray,
    source_groups: np.ndarray,
    roadmap_folds: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    cv_validation_folds: Sequence[int],
    *,
    positive_class_id: int,
    dataset_fingerprint: str,
) -> None:
    classes = np.unique(class_ids)
    groups = np.asarray(source_groups)
    fold_ids = np.asarray(roadmap_folds)

    def side(indices: np.ndarray) -> Dict[str, Any]:
        observed = set(int(i) for i in np.unique(class_ids[indices]))
        return {
            "rows": int(len(indices)),
            "source_files": int(np.unique(groups[indices]).size),
            "roadmap_folds": [int(i) for i in np.unique(fold_ids[indices])],
            "class_counts": _class_counts(class_ids[indices], classes),
            "binary_counts": _class_counts(y[indices], np.asarray([0, 1])),
            "classes_present": sorted(observed),
            "classes_missing": [int(i) for i in classes if int(i) not in observed],
        }

    report = {
        "split_source": f"{FOLD_ROADMAP_PATH.name} (fixed folds, plan={_configured_roadmap().plan})",
        "random_split_or_reassignment": False,
        "strict_all_class_coverage": bool(STRICT_ALL_CLASS_COVERAGE),
        "both_binary_targets_required": True,
        "roadmap_file_sha256": _roadmap_sha256(),
        "roadmap_dataset_fingerprint": _configured_roadmap().data_fingerprint,
        "roadmap_dataset": FOLD_ROADMAP_DATASET,
        "dataset_fingerprint": dataset_fingerprint,
        "target_class_id": int(positive_class_id),
        "target_rule": "source_state_gt_zero_and_class_equals_target",
        "required_classes": [int(value) for value in classes],
        "test_fold_id": int(TEST_FOLD_ID),
        "cv_validation_folds": list(cv_validation_folds),
        "holdout": {
            "train": side(train_indices),
            "test": side(test_indices),
            "source_file_overlap": int(np.intersect1d(groups[train_indices], groups[test_indices]).size),
        },
        "cv_folds": [
            {
                "validation_roadmap_fold": int(fid),
                "train": side(tr),
                "validation": side(va),
                "source_file_overlap": int(np.intersect1d(groups[tr], groups[va]).size),
            }
            for fid, (tr, va) in zip(cv_validation_folds, cv_splits)
        ],
        "identity_review_required": list(_configured_roadmap().identity_review_required),
    }
    path.write_text(json.dumps(report, indent=2, cls=NumpyEncoder), encoding="utf-8")


# =================================================================================================
# FEATURE SELECTION / PREPROCESSING / MODEL SIZE
# =================================================================================================
def rank_features_indices(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute one stable F-score ordering that can be reused by every trial."""

    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError(f"Expected a non-empty 2-D feature matrix, got {X.shape}.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scores, _ = f_classif(np.asarray(X, dtype=np.float32), np.asarray(y))
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(
        scores,
        nan=-np.inf,
        posinf=np.finfo(np.float64).max,
        neginf=-np.inf,
    )
    # Stable ordering makes ties reproducible across platforms.
    order = np.lexsort((np.arange(scores.size), -scores))
    return order.astype(np.uint16, copy=False)


def select_features_from_order(
    feature_order: np.ndarray, n_features: int, total_features: int
) -> np.ndarray:
    k = int(min(max(1, n_features), total_features, MAX_SELECTED_FEATURES))
    selected = np.sort(np.asarray(feature_order)[:k])
    return selected.astype(np.uint16, copy=False)


def select_features_indices(X: np.ndarray, y: np.ndarray, n_features: int) -> np.ndarray:
    return select_features_from_order(rank_features_indices(X, y), n_features, X.shape[1])


def make_preprocessor(name: str) -> Any | None:
    if name == "none":
        return None
    if name == "standard":
        return StandardScaler(copy=False)
    if name == "minmax_0_1":
        return MinMaxScaler(feature_range=(0.0, 1.0), copy=False, clip=False)
    if name == "minmax_m1_1":
        return MinMaxScaler(feature_range=(-1.0, 1.0), copy=False, clip=False)
    if name == "maxabs":
        return MaxAbsScaler(copy=False)
    if name == "robust_25_75":
        return RobustScaler(quantile_range=(25.0, 75.0), copy=False)
    if name == "robust_10_90":
        return RobustScaler(quantile_range=(10.0, 90.0), copy=False)
    if name == "normalize_l1":
        return Normalizer(norm="l1", copy=False)
    if name == "normalize_l2":
        return Normalizer(norm="l2", copy=False)
    if name == "normalize_max":
        return Normalizer(norm="max", copy=False)
    raise ValueError(f"Unknown preprocessing mode: {name!r}")


def _writable_float32_c(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32, order="C")
    if not result.flags.c_contiguous or not result.flags.writeable or not result.flags.owndata:
        result = np.array(result, dtype=np.float32, order="C", copy=True)
    return result


def fit_preprocessor(
    name: str,
    X_train: np.ndarray,
    X_other: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray | None, Any | None]:
    # Selected-column matrices are already private trial allocations. Reuse them
    # instead of immediately making a second full copy.
    train = _writable_float32_c(X_train)
    other = None if X_other is None else _writable_float32_c(X_other)
    if not np.all(np.isfinite(train)) or (other is not None and not np.all(np.isfinite(other))):
        raise ValueError("Preprocessing input contains NaN/Inf.")

    preprocessor = make_preprocessor(name)
    if preprocessor is None:
        return train, other, None

    train_transformed = np.asarray(preprocessor.fit_transform(train), dtype=np.float32)
    other_transformed = (
        None
        if other is None
        else np.asarray(preprocessor.transform(other), dtype=np.float32)
    )
    if not np.all(np.isfinite(train_transformed)) or (
        other_transformed is not None and not np.all(np.isfinite(other_transformed))
    ):
        raise ValueError(f"{name} produced NaN/Inf.")
    return train_transformed, other_transformed, preprocessor


def _stratified_cap_local_indices(
    class_ids: np.ndarray,
    y: np.ndarray,
    cap: int,
    *,
    random_state: int,
) -> np.ndarray:
    """Deterministically cap rows while preserving every observed (class_id, y) stratum."""

    class_ids = np.asarray(class_ids)
    y = np.asarray(y, dtype=np.int8)
    row_count = len(y)
    if cap <= 0 or cap >= row_count:
        return np.arange(row_count, dtype=np.int64)

    labels = _joint_stratification_labels(class_ids, y)
    unique_labels, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    stratum_count = len(unique_labels)
    if cap < stratum_count:
        raise ValueError(
            f"Optuna row cap {cap} is smaller than the {stratum_count} required joint strata."
        )

    preferred_min = np.minimum(counts, int(max(1, OPTUNA_MIN_ROWS_PER_JOINT_STRATUM)))
    if int(preferred_min.sum()) > cap:
        allocation = np.ones(stratum_count, dtype=np.int64)
    else:
        allocation = preferred_min.astype(np.int64, copy=True)

    remaining = int(cap - allocation.sum())
    capacity = counts.astype(np.int64) - allocation
    if remaining > 0 and int(capacity.sum()) > 0:
        exact = remaining * capacity.astype(np.float64) / float(capacity.sum())
        extra = np.minimum(capacity, np.floor(exact).astype(np.int64))
        allocation += extra
        remaining -= int(extra.sum())
        if remaining > 0:
            fractional = exact - np.floor(exact)
            order = np.lexsort((np.arange(stratum_count), -fractional))
            for stratum in order:
                if remaining <= 0:
                    break
                spare = int(counts[stratum] - allocation[stratum])
                if spare <= 0:
                    continue
                take = min(spare, remaining)
                allocation[stratum] += take
                remaining -= take

    rng = np.random.default_rng(random_state)
    selected_parts: List[np.ndarray] = []
    for stratum in range(stratum_count):
        candidates = np.flatnonzero(inverse == stratum)
        take = int(allocation[stratum])
        if take >= len(candidates):
            chosen = candidates
        else:
            chosen = rng.choice(candidates, size=take, replace=False)
        selected_parts.append(np.asarray(chosen, dtype=np.int64))
    selected = np.sort(np.concatenate(selected_parts))
    if len(selected) != cap:
        raise RuntimeError(f"Stratified cap selected {len(selected)} rows, expected {cap}.")
    required_classes = np.unique(class_ids)
    _require_all_class_coverage(class_ids, y, selected, required_classes, "Optuna capped fold side")
    return selected


def prepare_cv_folds(
    X_full_mm: np.ndarray,
    y_full: np.ndarray,
    class_ids_full: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    cache_mode: str = FOLD_CACHE_MODE,
) -> Tuple[List[PreparedFold], bool, int]:
    """Prepare feature rankings once and optionally cache raw fold matrices in RAM."""

    mode = str(cache_mode).strip().lower()
    if mode not in {"auto", "on", "off"}:
        raise ValueError("fold cache mode must be one of: auto, on, off.")

    feature_count = int(X_full_mm.shape[1])
    capped_splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_index, (train_idx, val_idx) in enumerate(cv_splits):
        train_local = _stratified_cap_local_indices(
            class_ids_full[train_idx],
            y_full[train_idx],
            int(OPTUNA_MAX_TRAIN_ROWS_PER_FOLD),
            random_state=10_000 + fold_index,
        )
        val_local = _stratified_cap_local_indices(
            class_ids_full[val_idx],
            y_full[val_idx],
            int(OPTUNA_MAX_VAL_ROWS_PER_FOLD),
            random_state=20_000 + fold_index,
        )
        capped_splits.append((np.asarray(train_idx)[train_local], np.asarray(val_idx)[val_local]))

    cache_bytes = int(
        sum(len(train_idx) + len(val_idx) for train_idx, val_idx in capped_splits)
        * feature_count
        * np.dtype(np.float32).itemsize
    )
    available = int(psutil.virtual_memory().available)
    safe_available = max(0, available - MEMORY_RESERVE_BYTES)
    auto_limit = int(safe_available * FOLD_CACHE_MAX_RAM_FRACTION)
    use_cache = mode == "on" or (mode == "auto" and cache_bytes <= auto_limit)
    if mode == "on" and cache_bytes > safe_available:
        raise MemoryError(
            f"Forced fold cache needs {cache_bytes / 1024**3:.2f} GiB, but only "
            f"{safe_available / 1024**3:.2f} GiB is safely available."
        )

    print(
        f"[CACHE] fold matrices need {cache_bytes / 1024**3:.2f} GiB; "
        f"mode={mode}, enabled={use_cache}, available={available / 1024**3:.2f} GiB"
    )

    prepared: List[PreparedFold] = []
    for fold_index, ((full_train_idx, full_val_idx), (train_idx, val_idx)) in enumerate(
        zip(cv_splits, capped_splits)
    ):
        y_train = np.asarray(y_full[train_idx], dtype=np.int8)
        y_val = np.asarray(y_full[val_idx], dtype=np.int8)
        class_ids_train = np.asarray(class_ids_full[train_idx], dtype=np.int16)
        class_ids_val = np.asarray(class_ids_full[val_idx], dtype=np.int16)

        if use_cache:
            X_train = np.ascontiguousarray(X_full_mm[train_idx], dtype=np.float32)
            X_val = np.ascontiguousarray(X_full_mm[val_idx], dtype=np.float32)
            feature_order = rank_features_indices(X_train, y_train)
            X_train.flags.writeable = False
            X_val.flags.writeable = False
            prepared.append(
                PreparedFold(
                    train_indices=np.asarray(train_idx, dtype=np.int64),
                    val_indices=np.asarray(val_idx, dtype=np.int64),
                    y_train=y_train,
                    y_val=y_val,
                    class_ids_train=class_ids_train,
                    class_ids_val=class_ids_val,
                    feature_order=feature_order,
                    full_train_rows=len(full_train_idx),
                    full_val_rows=len(full_val_idx),
                    X_train_cached=X_train,
                    X_val_cached=X_val,
                )
            )
        else:
            with MEMORY_ALLOC_LOCK:
                X_train_for_rank = np.ascontiguousarray(
                    X_full_mm[train_idx], dtype=np.float32
                )
            feature_order = rank_features_indices(X_train_for_rank, y_train)
            del X_train_for_rank
            prepared.append(
                PreparedFold(
                    train_indices=np.asarray(train_idx, dtype=np.int64),
                    val_indices=np.asarray(val_idx, dtype=np.int64),
                    y_train=y_train,
                    y_val=y_val,
                    class_ids_train=class_ids_train,
                    class_ids_val=class_ids_val,
                    feature_order=feature_order,
                    full_train_rows=len(full_train_idx),
                    full_val_rows=len(full_val_idx),
                )
            )
        print(
            f"[CACHE] prepared fold {fold_index + 1}/{len(cv_splits)}: "
            f"Optuna train={len(train_idx):,}/{len(full_train_idx):,}, "
            f"val={len(val_idx):,}/{len(full_val_idx):,} rows"
        )

    return prepared, use_cache, cache_bytes


def estimate_model_size_bytes(model: Any, selected_indices: Sequence[int] | None = None) -> int:
    """Return exact tree-data bytes emitted by the flash-optimized C exporter.

    The count includes the SoA threshold/child/feature/flag arrays and root-offset
    table. Compiler-dependent accessor/inference code is intentionally separate.
    """

    if model is None:
        return 0
    # Include Q20 probability lookup and optional early-exit tables in static budget.
    return int(estimate_compact_tree_payload(model)["total_static_data_bytes"])


def normalized_model_size(size_bytes: float) -> float:
    """Preserve size ordering across the entire valid firmware budget."""

    if MAX_MODEL_SIZE_BYTES <= 0:
        raise ValueError("MAX_MODEL_SIZE_BYTES must be positive.")
    # Below MIN_REALISTIC_SIZE, optimizing a few more bytes is not rewarded.
    # Small models are still allowed if their validation F1 makes them useful.
    return float(max(float(MIN_REALISTIC_SIZE), 0.0, size_bytes) / MAX_MODEL_SIZE_BYTES)


def wait_for_memory(
    trial_number: int,
    bytes_needed: int,
    allocation_name: str,
    *,
    timeout_seconds: int = MEMORY_WAIT_TIMEOUT_SECONDS,
) -> None:
    target = int(max(0, bytes_needed) + MEMORY_RESERVE_BYTES)
    deadline = time.monotonic() + timeout_seconds
    last_message = 0.0
    while psutil.virtual_memory().available < target:
        now = time.monotonic()
        if now >= deadline:
            raise optuna.TrialPruned(
                f"Trial {trial_number}: insufficient RAM for {allocation_name}; "
                f"needed available={target / 1024**3:.2f} GiB."
            )
        if now - last_message >= 30.0:
            available = psutil.virtual_memory().available
            print(
                f"[MEMORY] Trial {trial_number} waiting for {allocation_name}: "
                f"available={available / 1024**3:.2f} GiB, target={target / 1024**3:.2f} GiB"
            )
            last_message = now
        time.sleep(2.0)


def predict_positive_probability(model: Any, X: np.ndarray) -> np.ndarray:
    """Return positive-class probability without LightGBM feature-name warnings."""

    if isinstance(model, lgb.LGBMModel):
        return np.asarray(model.booster_.predict(X), dtype=np.float64)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


# =================================================================================================
# METRICS / IMPORTANCE
# =================================================================================================
def get_debug_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_ids: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = (np.asarray(y_proba, dtype=np.float64) >= threshold).astype(np.int8)
    class_ids = np.asarray(class_ids)

    def stats(yt: np.ndarray, yp: np.ndarray) -> Dict[str, Any]:
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_raw = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1_raw": float(f1_raw),
        }

    result = stats(y_true, y_pred)
    result["threshold"] = float(threshold)
    result["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
    result["by_class"] = {
        int(class_id): stats(y_true[class_ids == class_id], y_pred[class_ids == class_id])
        for class_id in np.unique(class_ids)
    }
    return result


def calculate_permutation_importance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    random_state: int = 42,
    max_rows: int = PERMUTATION_MAX_ROWS,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if len(X_val) > max_rows:
        sample_indices = rng.choice(len(X_val), size=max_rows, replace=False)
        X_work = np.array(X_val[sample_indices], dtype=np.float32, copy=True)
        y_work = y_val[sample_indices]
    else:
        X_work = np.array(X_val, dtype=np.float32, copy=True)
        y_work = np.asarray(y_val)

    base_probability = predict_positive_probability(model, X_work)
    base_score = f1_score(y_work, base_probability >= threshold, zero_division=0)
    importances = np.zeros(X_work.shape[1], dtype=np.float32)

    for feature_index in range(X_work.shape[1]):
        original = X_work[:, feature_index].copy()
        X_work[:, feature_index] = original[rng.permutation(len(original))]
        permuted_probability = predict_positive_probability(model, X_work)
        score = f1_score(y_work, permuted_probability >= threshold, zero_division=0)
        importances[feature_index] = np.float32(base_score - score)
        X_work[:, feature_index] = original
    return importances


# =================================================================================================
# MODEL PARAMETER SEARCH / TRAINING
# =================================================================================================
def worst_case_tree_payload_bytes(model_type: str, n_trees: int, leaves_per_tree: int) -> int:
    """Conservative static-data bound for capped tree ensembles in Cortex-M flash."""
    trees = max(1, int(n_trees))
    leaves = max(2, int(leaves_per_tree))
    nodes_per_tree = 2 * leaves - 1
    total_nodes = trees * nodes_per_tree
    root_bytes = 2 if total_nodes <= np.iinfo(np.uint16).max else 4
    if model_type in {"LightGBM", "XGBoost"}:
        if nodes_per_tree > 255:
            return math.inf
        # Constant/mixed routing flags may need bitsets. Add conservative 1 byte per node
        # beyond the 6-byte threshold+right-child+feature baseline.
        tree_bytes = total_nodes * 7 + trees * root_bytes
        q20_lut = 156
        # At most two float32 suffix-bound arrays, one entry per group of four trees.
        bounds = (math.ceil(trees / 4) + 1) * 8
        return int(tree_bytes + q20_lut + bounds)
    raise ValueError(f"Unknown model_type: {model_type!r}.")


def derive_budget_safe_leaves(
    model_type: str,
    n_trees: int,
    max_depth: int,
    budget_fraction: float,
    hard_cap: int,
) -> int:
    """Derive a leaf limit whose exact flash-optimized payload fits firmware."""

    n_trees = max(2, int(n_trees))
    max_depth = max(1, int(max_depth))
    if worst_case_tree_payload_bytes(model_type, n_trees, 2) > MAX_MODEL_SIZE_BYTES:
        raise ValueError(
            f"{n_trees} {model_type} trees cannot fit in {MAX_MODEL_SIZE_BYTES} bytes "
            "even at two leaves."
        )

    max_leaves_by_depth = 1 << min(max_depth, 16)
    upper = max(2, min(int(hard_cap), max_leaves_by_depth))
    if model_type in {"LightGBM", "XGBoost"}:
        upper = min(upper, 128)  # 2*leaves-1 must fit uint8 local indices.

    lo, hi = 2, upper
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2
        if worst_case_tree_payload_bytes(model_type, n_trees, mid) <= MAX_MODEL_SIZE_BYTES:
            lo = mid
        else:
            hi = mid - 1
    maximum = lo

    fraction = min(1.0, max(0.0, float(budget_fraction)))
    if maximum == 2:
        return 2
    return int(2 + math.floor(fraction * (maximum - 2)))


def suggest_trial_configuration(
    trial: optuna.Trial,
    model_type: str,
    feature_count: int,
    training_rows: int,
) -> Tuple[str, int]:
    """Suggest one fixed-family, static, budget-safe multivariate search space."""

    if model_type not in MODEL_FAMILY_ORDER:
        raise ValueError(f"Unsupported model_type: {model_type!r}.")
    if feature_count < MIN_SELECTED_FEATURES:
        raise ValueError(
            f"At least {MIN_SELECTED_FEATURES} features are required, got {feature_count}."
        )

    preprocessing = trial.suggest_categorical("preprocessing", list(PREPROCESSING_CHOICES))
    maximum_features = min(MAX_SELECTED_FEATURES, feature_count)
    n_features = trial.suggest_int("n_features", MIN_SELECTED_FEATURES, maximum_features)
    if model_type == "XGBoost":
        max_estimators = min(200, MAX_MODEL_SIZE_BYTES // worst_case_tree_payload_bytes("XGBoost", 1, 2))
        trial.suggest_int("xgb_n_estimators", 8, max_estimators)
        trial.suggest_int("xgb_max_depth", 2, 12)
        trial.suggest_float("xgb_leaf_budget_fraction", 0.0, 1.0)
        trial.suggest_float("xgb_learning_rate", 0.01, 0.20, log=True)
        trial.suggest_float("xgb_min_child_weight", 1.0, 100.0, log=True)
        trial.suggest_float("xgb_gamma", 0.0, 6.0)
        trial.suggest_float("xgb_reg_alpha", 0.0, 10.0)
        trial.suggest_float("xgb_reg_lambda", 1.0, 20.0)
        trial.suggest_float("xgb_subsample", 0.60, 1.0)
        trial.suggest_float("xgb_colsample_bytree", 0.60, 1.0)
        trial.suggest_float("xgb_scale_pos_weight", 1.0, 15.0)
        trial.suggest_categorical("xgb_max_bin", [128, 256])
    else:
        min_tree_bytes = worst_case_tree_payload_bytes("LightGBM", 1, 2)
        max_estimators = min(200, MAX_MODEL_SIZE_BYTES // min_tree_bytes)
        trial.suggest_int("lgb_n_estimators", 2, max_estimators)
        trial.suggest_int("lgb_max_depth", 1, 18)
        trial.suggest_float("lgb_leaf_budget_fraction", 0.0, 1.0)
        child_max = min(1000, max(10, training_rows // 10))
        trial.suggest_int("lgb_min_child_samples", 10, child_max)
        trial.suggest_float("lgb_lambda_l1", 0.0, 10.0)
        trial.suggest_float("lgb_lambda_l2", 0.0, 10.0)
        # Desktop-only Optuna proposal distribution; not a feature LOG transform.
        trial.suggest_float("lgb_learning_rate", 0.01, 0.15, log=True)
        trial.suggest_float("lgb_subsample", 0.6, 1.0)
        trial.suggest_int("lgb_subsample_freq", 1, 7)
        trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0)
        trial.suggest_float("lgb_scale_pos_weight", 1.0, 15.0)
        trial.suggest_float("lgb_min_split_gain", 0.0, 0.2)
    return preprocessing, n_features


def materialize_model_params(
    model_type: str,
    trial_params: Mapping[str, Any],
    *,
    n_jobs: int,
) -> Dict[str, Any]:
    """Convert static Optuna coordinates into concrete, firmware-safe parameters."""

    if model_type == "XGBoost":
        n_estimators = int(trial_params["xgb_n_estimators"])
        max_depth = int(trial_params["xgb_max_depth"])
        max_leaves = derive_budget_safe_leaves(
            "XGBoost", n_estimators, max_depth,
            float(trial_params["xgb_leaf_budget_fraction"]), 128,
        )
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "tree_method": "hist",
            "grow_policy": "lossguide",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_leaves": max_leaves,
            "learning_rate": float(trial_params["xgb_learning_rate"]),
            "min_child_weight": float(trial_params["xgb_min_child_weight"]),
            "gamma": float(trial_params["xgb_gamma"]),
            "reg_alpha": float(trial_params["xgb_reg_alpha"]),
            "reg_lambda": float(trial_params["xgb_reg_lambda"]),
            "subsample": float(trial_params["xgb_subsample"]),
            "colsample_bytree": float(trial_params["xgb_colsample_bytree"]),
            "scale_pos_weight": float(trial_params["xgb_scale_pos_weight"]),
            "max_bin": int(trial_params["xgb_max_bin"]),
            "random_state": 42,
            "n_jobs": int(n_jobs),
        }

    if model_type == "LightGBM":
        n_estimators = int(trial_params["lgb_n_estimators"])
        max_depth = int(trial_params["lgb_max_depth"])
        num_leaves = derive_budget_safe_leaves(
            "LightGBM",
            n_estimators,
            max_depth,
            float(trial_params["lgb_leaf_budget_fraction"]),
            100,
        )
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": int(trial_params["lgb_min_child_samples"]),
            "lambda_l1": float(trial_params["lgb_lambda_l1"]),
            "lambda_l2": float(trial_params["lgb_lambda_l2"]),
            "learning_rate": float(trial_params["lgb_learning_rate"]),
            "subsample": float(trial_params["lgb_subsample"]),
            "subsample_freq": int(trial_params["lgb_subsample_freq"]),
            "colsample_bytree": float(trial_params["lgb_colsample_bytree"]),
            "scale_pos_weight": float(trial_params["lgb_scale_pos_weight"]),
            "min_split_gain": float(trial_params["lgb_min_split_gain"]),
            "random_state": 42,
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": int(n_jobs),
        }
    raise ValueError(f"Unknown model_type: {model_type!r}")


def make_optuna_sampler(optuna_jobs: int, family_trials: int) -> optuna.samplers.TPESampler:
    """Create the static-family multivariate TPE sampler."""

    with warnings.catch_warnings():
        experimental_warning = getattr(optuna.exceptions, "ExperimentalWarning", Warning)
        warnings.simplefilter("ignore", category=experimental_warning)
        startup_trials = min(int(TPE_STARTUP_TRIALS), max(4, int(family_trials) // 3))
        return optuna.samplers.TPESampler(
            seed=42,
            multivariate=TPE_MULTIVARIATE,
            n_startup_trials=startup_trials,
            n_ei_candidates=TPE_N_EI_CANDIDATES,
            constant_liar=max(1, int(optuna_jobs)) > 1,
        )


def train_classifier(
    model_type: str,
    params: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    if np.unique(y_train).size != 2:
        raise ValueError("Training fold must contain both binary classes.")
    if model_type == "XGBoost":
        model = xgb.XGBClassifier(**dict(params))
        model.fit(X_train, y_train)
        return model
    if model_type == "LightGBM":
        model = lgb.LGBMClassifier(**dict(params))
        model.fit(X_train, y_train)
        return model
    raise ValueError(f"Unknown model_type: {model_type}")


# =================================================================================================
# OPTUNA OBJECTIVE
# =================================================================================================
def objective_simple(
    trial: optuna.Trial,
    model_type: str,
    X_full_mm: np.ndarray,
    prepared_folds: Sequence[PreparedFold],
    feature_names: Sequence[str],
    model_threads: int,
) -> Tuple[float, float]:
    trial_number = trial.number
    training_rows = int(
        sum(len(fold.train_indices) for fold in prepared_folds) / len(prepared_folds)
    )
    preprocessing, n_features = suggest_trial_configuration(
        trial, model_type, len(feature_names), training_rows
    )
    model_params = materialize_model_params(model_type, trial.params, n_jobs=model_threads)
    trial.set_user_attr("effective_model_params", dict(model_params))

    fold_scores: List[float] = []
    fold_sizes: List[int] = []
    fold_selected_indices: List[List[int]] = []

    for fold_index, fold in enumerate(prepared_folds):
        selected_indices = select_features_from_order(
            fold.feature_order, n_features, len(feature_names)
        )
        selected_bytes = (
            (len(fold.train_indices) + len(fold.val_indices))
            * len(selected_indices)
            * np.dtype(np.float32).itemsize
            * 2
        )

        if fold.X_train_cached is None or fold.X_val_cached is None:
            raw_bytes = (
                (len(fold.train_indices) + len(fold.val_indices))
                * X_full_mm.shape[1]
                * max(X_full_mm.dtype.itemsize, np.dtype(np.float32).itemsize)
            )
            wait_for_memory(
                trial_number,
                raw_bytes + selected_bytes,
                f"fold {fold_index} uncached matrices",
            )
            with MEMORY_ALLOC_LOCK:
                X_train_raw = np.ascontiguousarray(
                    X_full_mm[fold.train_indices], dtype=np.float32
                )
                X_val_raw = np.ascontiguousarray(
                    X_full_mm[fold.val_indices], dtype=np.float32
                )
        else:
            wait_for_memory(
                trial_number,
                selected_bytes,
                f"fold {fold_index} selected matrices",
            )
            X_train_raw = fold.X_train_cached
            X_val_raw = fold.X_val_cached

        # Advanced column selection creates private matrices that preprocessing
        # can safely transform in place without another full copy.
        X_train_selected = np.ascontiguousarray(
            X_train_raw[:, selected_indices], dtype=np.float32
        )
        X_val_selected = np.ascontiguousarray(
            X_val_raw[:, selected_indices], dtype=np.float32
        )
        if fold.X_train_cached is None:
            del X_train_raw, X_val_raw

        X_train_processed, X_val_processed, _ = fit_preprocessor(
            preprocessing, X_train_selected, X_val_selected
        )
        model = train_classifier(
            model_type=model_type,
            params=model_params,
            X_train=X_train_processed,
            y_train=fold.y_train,
        )
        probability = predict_positive_probability(model, X_val_processed)
        prediction = probability >= DEFAULT_THRESHOLD
        score = float(f1_score(fold.y_val, prediction, zero_division=0))
        size_bytes = estimate_model_size_bytes(model, selected_indices)

        fold_scores.append(score)
        fold_sizes.append(size_bytes)
        fold_selected_indices.append(selected_indices.astype(int).tolist())

        del X_train_processed, X_val_processed, model, probability, prediction

        if (
            PRUNE_ALL_ZERO_F1_AFTER_FOLDS > 0
            and len(fold_scores) >= PRUNE_ALL_ZERO_F1_AFTER_FOLDS
            and not any(score_value > 0.0 for score_value in fold_scores)
        ):
            trial.set_user_attr("pruned_all_zero_f1_after_folds", len(fold_scores))
            raise optuna.TrialPruned(
                f"F1 remained zero for the first {len(fold_scores)} folds at threshold "
                f"{DEFAULT_THRESHOLD:.2f}."
            )

        # This should be rare because leaf counts are now derived from the hard
        # payload budget, but keep the exact measured guard.
        if size_bytes > MAX_MODEL_SIZE_BYTES:
            trial.set_user_attr("oversize_fold", fold_index)
            trial.set_user_attr("oversize_bytes", size_bytes)
            raise optuna.TrialPruned(
                f"Compact payload {size_bytes} exceeds MAX_MODEL_SIZE_BYTES={MAX_MODEL_SIZE_BYTES}."
            )

    mean_score = float(np.mean(fold_scores))
    mean_size = float(np.mean(fold_sizes))
    trial.set_user_attr("fold_f1", fold_scores)
    trial.set_user_attr("fold_size_bytes", fold_sizes)
    trial.set_user_attr("mean_size_bytes", mean_size)
    trial.set_user_attr("selected_indices_by_fold", fold_selected_indices)
    trial.set_user_attr("decision_threshold", DEFAULT_THRESHOLD)
    trial.set_user_attr("preprocessing", preprocessing)
    trial.set_user_attr("model_type", model_type)
    _periodic_gc(trial_number)
    return mean_score, normalized_model_size(mean_size)


# =================================================================================================
# FINAL TRAINING / REPORTING
# =================================================================================================
def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_ids_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_ids_test: np.ndarray,
    best_params: Mapping[str, Any],
    feature_names: Sequence[str],
    n_original_features: int,
    *,
    model_threads: int,
    calculate_permutation: bool = True,
    permutation_max_rows: int = PERMUTATION_MAX_ROWS,
) -> Tuple[
    Dict[str, Any],
    float,
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    np.ndarray,
    np.ndarray | None,
]:
    model_type = str(best_params.get("model_type", "XGBoost"))
    preprocessing = str(best_params.get("preprocessing", "none"))
    n_features = int(best_params["n_features"])
    model_params = materialize_model_params(model_type, best_params, n_jobs=model_threads)

    print(f"\n--- Training final {model_type} model with preprocessing={preprocessing} ---")
    feature_order = rank_features_indices(np.asarray(X_train), y_train)
    selected_indices = select_features_from_order(
        feature_order, n_features, np.asarray(X_train).shape[1]
    )
    X_train_selected = np.ascontiguousarray(
        np.asarray(X_train)[:, selected_indices], dtype=np.float32
    )
    X_test_selected = np.ascontiguousarray(
        np.asarray(X_test)[:, selected_indices], dtype=np.float32
    )

    X_train_processed, X_test_processed, preprocessor = fit_preprocessor(
        preprocessing, X_train_selected, X_test_selected
    )
    model = train_classifier(model_type, model_params, X_train_processed, y_train)

    model_container: Dict[str, Any] = {
        "model": model,
        "preprocessor": preprocessor,
        # Keep the old key for consumers that still expect it.
        "scaler": preprocessor,
        "preprocessing": preprocessing,
        "indices": selected_indices,
        "n_original_features": int(n_original_features),
        "decision_threshold": float(DEFAULT_THRESHOLD),
        "effective_model_params": dict(model_params),
    }

    train_probability = predict_positive_probability(model, X_train_processed)
    test_probability = predict_positive_probability(model, X_test_processed)
    thresholds = np.round(np.arange(0.50, 1.00, 0.05), 2)

    train_metrics = [
        get_debug_metrics_at_threshold(y_train, train_probability, class_ids_train, float(threshold))
        for threshold in thresholds
    ]
    test_metrics = [
        get_debug_metrics_at_threshold(y_test, test_probability, class_ids_test, float(threshold))
        for threshold in thresholds
    ]
    default_test_score = float(
        next(
            metric["f1_score"]
            for metric in test_metrics
            if abs(metric["threshold"] - DEFAULT_THRESHOLD) < 1e-9
        )
    )
    print(f"FINAL TEST F1 at threshold {DEFAULT_THRESHOLD:.2f}: {default_test_score:.6f}")

    native_importances = np.asarray(
        getattr(model, "feature_importances_", np.zeros(len(selected_indices))),
        dtype=np.float64,
    )
    permutation_importances: np.ndarray | None = None
    if calculate_permutation:
        print(
            "Calculating deterministic permutation importance "
            f"on at most {permutation_max_rows:,} test rows..."
        )
        permutation_importances = calculate_permutation_importance(
            model,
            X_test_processed,
            y_test,
            threshold=DEFAULT_THRESHOLD,
            max_rows=max(1, int(permutation_max_rows)),
        )
    else:
        print("Skipping permutation importance for this lower Pareto rank.")

    return (
        model_container,
        default_test_score,
        test_metrics,
        train_metrics,
        native_importances,
        permutation_importances,
    )


def _make_storage(db_path: Path) -> optuna.storages.RDBStorage:
    absolute = db_path.resolve()
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{absolute.as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    _sqlite_wal_mode(db_path)
    return storage


def _validate_study_identity(
    study: optuna.Study,
    *,
    dataset_fingerprint: str,
    feature_names: Sequence[str],
    positive_class_id: int,
    model_type: str,
    observed_classes: Sequence[int],
    effective_folds: int,
) -> None:
    expected = {
        "study_schema_version": STUDY_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "feature_count": len(feature_names),
        "preprocessing_choices": list(PREPROCESSING_CHOICES),
        "decision_threshold": DEFAULT_THRESHOLD,
        "max_model_size_bytes": MAX_MODEL_SIZE_BYTES,
        "flash_storage_layout": "exact_float32_soa",
        "min_selected_features": MIN_SELECTED_FEATURES,
        "target_class_id": int(positive_class_id),
        "positive_class_id": int(positive_class_id),
        "model_type": model_type,
        "sampler": "TPESampler",
        "tpe_multivariate": bool(TPE_MULTIVARIATE),
        "tpe_startup_trials_max": int(TPE_STARTUP_TRIALS),
        "tpe_n_ei_candidates": int(TPE_N_EI_CANDIDATES),
        "cv_folds_requested": int(CV_FOLDS),
        "optuna_max_train_rows_per_fold": int(OPTUNA_MAX_TRAIN_ROWS_PER_FOLD),
        "optuna_max_val_rows_per_fold": int(OPTUNA_MAX_VAL_ROWS_PER_FOLD),
        "xgb_tree_method": "hist",
        "xgb_grow_policy": "lossguide",
        "xgb_max_bin_choices": [128, 256],
        "search_space_encoding": SEARCH_SPACE_ENCODING,
        "strict_all_class_coverage": bool(STRICT_ALL_CLASS_COVERAGE),
        "observed_classes": [int(value) for value in observed_classes],
        "effective_folds": int(effective_folds),
        "roadmap_test_fold": int(TEST_FOLD_ID),
        "roadmap_cv_validation_folds": [
            int(f) for f in range(_configured_roadmap().n_folds)
            if f != int(TEST_FOLD_ID)
        ][:int(CV_FOLDS)],
        "roadmap_file_sha256": _roadmap_sha256(),
    }
    existing = study.user_attrs
    for key, value in expected.items():
        if key in existing and existing[key] != value:
            raise RuntimeError(
                f"Refusing to resume incompatible Optuna study: attribute {key!r} is "
                f"{existing[key]!r}, expected {value!r}. Set TO_RESUME=False or use a new TRAIN_FOLDER."
            )
    for key, value in expected.items():
        study.set_user_attr(key, value)


def optimize_tree_simple(
    group_name: str,
    n_trials: int,
    output_dir: Path,
    *,
    data_folder: Path,
    model_types: Sequence[str] = MODEL_FAMILY_ORDER,
    trials_by_model: Mapping[str, int] | None = None,
    resume: bool = TO_RESUME,
    csv_workers: int = CSV_WORKERS,
    fold_cache: str = FOLD_CACHE_MODE,
    permutation_top_ranks: int = PERMUTATION_TOP_RANKS,
    permutation_max_rows: int = PERMUTATION_MAX_ROWS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path]:
    """Load once, then optimize XGBoost and LightGBM in independent sequential studies."""

    normalized_models = tuple(model_types)
    if not normalized_models:
        raise ValueError("At least one model family must be selected.")
    unknown = [name for name in normalized_models if name not in MODEL_FAMILY_ORDER]
    if unknown:
        raise ValueError(f"Unsupported model families: {unknown}.")

    dataset_dir, feature_names, schema_path, fingerprint, _ = load_and_validate_data(
        group_name,
        output_dir,
        data_folder,
        csv_workers=csv_workers,
    )
    memmap_dir = output_dir / "memmaps"
    memmap_dir.mkdir(parents=True, exist_ok=True)
    X_full_mm, y, class_ids, groups, roadmap_folds = load_or_build_memmaps(
        dataset_dir,
        feature_names,
        memmap_dir,
        group_name,
        fingerprint,
    )
    observed_classes = np.unique(class_ids)
    positive_class_id = resolve_target_class_id()
    if positive_class_id not in observed_classes:
        raise ValueError(
            f"TARGET_CLASS_ID={positive_class_id} is absent; observed classes are "
            f"{observed_classes.tolist()}."
        )
    if np.any((y == 1) & (class_ids != positive_class_id)):
        offending = np.unique(class_ids[(y == 1) & (class_ids != positive_class_id)]).tolist()
        raise AssertionError(
            f"Binary-positive rows escaped TARGET_CLASS_ID={positive_class_id}; "
            f"offending class IDs: {offending}."
        )
    if not np.any((y == 1) & (class_ids == positive_class_id)):
        raise ValueError(
            f"TARGET_CLASS_ID={positive_class_id} exists but has no active positive rows after "
            "applying source_state > 0 AND class_id == TARGET_CLASS_ID."
        )
    if np.unique(y).size != 2:
        raise ValueError(f"Binary target contains {np.unique(y).tolist()}, expected both 0 and 1.")
    print(
        f"[CLASSES] dynamically discovered {observed_classes.size} class IDs: "
        f"{observed_classes.tolist()}; target_class={positive_class_id}; "
        f"positive_rows={int(np.count_nonzero(y == 1)):,}"
    )

    train_indices, test_indices, split_summary, cv_splits, cv_validation_folds = (
        make_roadmap_holdout_and_cv(
            class_ids, y, groups, roadmap_folds,
            test_fold=int(TEST_FOLD_ID), requested_folds=int(CV_FOLDS),
        )
    )
    print(
        f"[SPLIT] train={split_summary.train_rows:,} rows/{split_summary.train_groups} files, "
        f"test={split_summary.test_rows:,} rows/{split_summary.test_groups} files"
    )
    print(
        f"[SPLIT] observed classes={list(split_summary.classes)}; "
        f"train counts={split_summary.class_counts_train}; "
        f"test counts={split_summary.class_counts_test}; "
        f"class-to-file counts={split_summary.class_group_counts}"
    )
    split_report_path = output_dir / f"{group_name}_roadmap_split_report.json"
    write_roadmap_split_report(
        split_report_path,
        class_ids, y, groups, roadmap_folds,
        train_indices, test_indices, cv_splits, cv_validation_folds,
        positive_class_id=positive_class_id, dataset_fingerprint=fingerprint,
    )
    print(f"[SPLIT] Auditable coverage report: {split_report_path}")
    # The content fingerprint identifies CSVs and the roadmap. This *additional*
    # split fingerprint separates outer test folds (and CV sets) across Optuna
    # resumes so trials are NEVER mixed between distinct evaluation protocols.
    split_fingerprint = hashlib.sha256(
        (fingerprint + "|" + str(TEST_FOLD_ID) + "|" +
         ",".join(map(str, cv_validation_folds))).encode("utf-8")
    ).hexdigest()

    prepared_folds, fold_cache_enabled, fold_cache_bytes = prepare_cv_folds(
        X_full_mm, y, class_ids, cv_splits, cache_mode=fold_cache
    )

    final_matrix_bytes = (
        (len(train_indices) + len(test_indices))
        * X_full_mm.shape[1]
        * max(X_full_mm.dtype.itemsize, np.dtype(np.float32).itemsize)
    )
    wait_for_memory(-1, final_matrix_bytes, "final train/test matrices")
    X_train_final = np.ascontiguousarray(X_full_mm[train_indices], dtype=np.float32)
    X_test = np.ascontiguousarray(X_full_mm[test_indices], dtype=np.float32)
    y_test = y[test_indices]
    class_ids_test = class_ids[test_indices]

    exported_models: List[Dict[str, Any]] = []
    model_infos: List[Dict[str, Any]] = []
    family_failures: List[str] = []
    family_failure_report = output_dir / f"{group_name}_family_study_failures.txt"
    if family_failure_report.exists():
        family_failure_report.unlink()

    for model_type in normalized_models:
        model_key = MODEL_KEY_BY_TYPE[model_type]
        family_trials = int((trials_by_model or {}).get(model_type, n_trials))
        family_optuna_jobs, runtime_model_threads = resolve_family_cpu(model_type)
        total_requested_threads = family_optuna_jobs * runtime_model_threads
        print(
            f"[CPU] {model_type}: physical={PHYSICAL_CPUS}, logical={LOGICAL_CPUS}, "
            f"budget={TRAINING_CPU_BUDGET}, Optuna jobs={family_optuna_jobs}, "
            f"model threads={runtime_model_threads}, aggregate={total_requested_threads}"
        )
        if family_trials <= 0:
            print(f"[SKIP] {model_type}: configured trials={family_trials}.")
            continue

        print(f"\n{'=' * 96}")
        print(f"SEQUENTIAL STUDY: {model_type} ({family_trials} trials)")
        print(f"{'=' * 96}")
        try:
            db_path = output_dir / (
                f"{group_name}_{model_key}_testfold{TEST_FOLD_ID}_optimization_v{STUDY_SCHEMA_VERSION}.db"
            )
            if not resume and db_path.exists():
                db_path.unlink()
            storage = _make_storage(db_path)
            study = optuna.create_study(
                study_name=f"{group_name}_{model_key}_fold{TEST_FOLD_ID}_{STUDY_NAME_SUFFIX}",
                directions=["maximize", "minimize"],
                sampler=make_optuna_sampler(family_optuna_jobs, family_trials),
                pruner=optuna.pruners.NopPruner(),
                storage=storage,
                load_if_exists=resume,
            )
            _validate_study_identity(
                study,
                dataset_fingerprint=split_fingerprint,
                feature_names=feature_names,
                positive_class_id=positive_class_id,
                model_type=model_type,
                observed_classes=split_summary.classes,
                effective_folds=len(prepared_folds),
            )
            print(
                f"Running {family_trials} {model_type} multivariate-TPE trials with "
                f"{family_optuna_jobs} concurrent trials x {runtime_model_threads} model threads "
                f"(startup<= {TPE_STARTUP_TRIALS}, EI candidates={TPE_N_EI_CANDIDATES})..."
            )
            study.optimize(
                lambda trial, fixed_model_type=model_type: objective_simple(
                    trial,
                    fixed_model_type,
                    X_full_mm,
                    prepared_folds,
                    feature_names,
                    runtime_model_threads,
                ),
                n_trials=family_trials,
                n_jobs=family_optuna_jobs,
                gc_after_trial=False,
                show_progress_bar=False,
            )

            pareto_trials = [
                trial
                for trial in study.best_trials
                if trial.state == optuna.trial.TrialState.COMPLETE
                and trial.values is not None
                and trial.user_attrs.get("mean_size_bytes", math.inf) <= MAX_MODEL_SIZE_BYTES
            ]
            pareto_trials.sort(
                key=lambda trial: (-float(trial.values[0]), float(trial.values[1]), trial.number)
            )
            pareto_trials = pareto_trials[:MAX_EXPORTED_PARETO_MODELS_PER_FAMILY]
            if not pareto_trials:
                raise RuntimeError(
                    f"{model_type} produced no complete Pareto trial inside the firmware budget."
                )

            for family_rank, best_trial in enumerate(pareto_trials, start=1):
                cv_f1 = float(best_trial.values[0])
                cv_size = float(best_trial.user_attrs.get("mean_size_bytes", 0.0))
                print(
                    f"\n--- {model_type} Pareto rank {family_rank}: CV F1={cv_f1:.6f}, "
                    f"mean payload={cv_size:.0f} bytes, trial={best_trial.number} ---"
                )
                best_params = {"model_type": model_type, **dict(best_trial.params)}
                (
                    final_model,
                    test_f1,
                    test_metrics,
                    train_metrics,
                    native_importances,
                    permutation_importances,
                ) = train_final_model(
                    X_train_final,
                    y[train_indices],
                    class_ids[train_indices],
                    X_test,
                    y_test,
                    class_ids_test,
                    best_params,
                    feature_names,
                    len(feature_names),
                    model_threads=runtime_model_threads,
                    calculate_permutation=family_rank <= max(0, int(permutation_top_ranks)),
                    permutation_max_rows=max(1, int(permutation_max_rows)),
                )
                final_model.update(
                    {
                        "target_class_id": int(positive_class_id),
                        "positive_class_id": int(positive_class_id),
                        "target_rule": "source_state_gt_zero_and_class_equals_target",
                    }
                )

                size_bytes = estimate_model_size_bytes(final_model, final_model["indices"])
                if size_bytes > MAX_MODEL_SIZE_BYTES:
                    print(
                        f"[SKIP] Final payload {size_bytes} exceeds "
                        f"MAX_MODEL_SIZE_BYTES={MAX_MODEL_SIZE_BYTES}."
                    )
                    continue

                selected_indices = np.asarray(final_model["indices"], dtype=np.int64)
                used_names = [feature_names[index] for index in selected_indices]
                info: Dict[str, Any] = {
                    "rank": family_rank,
                    "family_rank": family_rank,
                    "model_key": model_key,
                    "trial_number": best_trial.number,
                    "group_name": group_name,
                    "params": best_params,
                    "effective_model_params": dict(final_model["effective_model_params"]),
                    "model_type": model_type,
                    "preprocessing": best_params["preprocessing"],
                    "decision_threshold": float(DEFAULT_THRESHOLD),
                    "target_class_id": int(positive_class_id),
                    "positive_class_id": int(positive_class_id),
                    "target_rule": "source_state_gt_zero_and_class_equals_target",
                    "cv_f1_score": cv_f1,
                    "f1_score": test_f1,
                    "test_metrics_by_threshold": test_metrics,
                    "train_metrics_by_threshold": train_metrics,
                    "model_size_bytes": size_bytes,
                    "selected_indices": selected_indices.tolist(),
                    "selected_feature_names": used_names,
                    "feature_importances_native": format_importances(used_names, native_importances),
                    "feature_importances_gini": format_importances(used_names, native_importances),
                    "feature_importances_permutation": format_importances(
                        used_names, permutation_importances
                    ),
                    "permutation_importance_calculated": permutation_importances is not None,
                    "permutation_importance_max_rows": (
                        int(permutation_max_rows) if permutation_importances is not None else 0
                    ),
                    "schema_file": str(schema_path.resolve()),
                    "split_report": str(split_report_path.resolve()),
                    "dataset_fingerprint": fingerprint,
                    "split_fingerprint": split_fingerprint,
                    "roadmap_file_sha256": _roadmap_sha256(),
                    "roadmap_dataset": FOLD_ROADMAP_DATASET,
                    "roadmap_test_fold": int(TEST_FOLD_ID),
                    "roadmap_cv_validation_folds": list(cv_validation_folds),
                    "study_schema_version": STUDY_SCHEMA_VERSION,
                    "study_database": str(db_path.resolve()),
                    "split_summary": {
                        "train_rows": split_summary.train_rows,
                        "test_rows": split_summary.test_rows,
                        "train_groups": split_summary.train_groups,
                        "test_groups": split_summary.test_groups,
                        "classes": list(split_summary.classes),
                        "class_counts_train": split_summary.class_counts_train,
                        "class_counts_test": split_summary.class_counts_test,
                        "class_group_counts": split_summary.class_group_counts,
                        "class_distribution_train": split_summary.class_distribution_train,
                        "class_distribution_test": split_summary.class_distribution_test,
                        "effective_cv_folds": len(prepared_folds),
                        "observed_class_count": len(split_summary.classes),
                    },
                    "performance": {
                        "physical_cpus": PHYSICAL_CPUS,
                        "logical_cpus": LOGICAL_CPUS,
                        "training_cpu_budget": TRAINING_CPU_BUDGET,
                        "optuna_jobs": family_optuna_jobs,
                        "model_threads": runtime_model_threads,
                        "aggregate_workers": family_optuna_jobs * runtime_model_threads,
                        "fold_cache_enabled": fold_cache_enabled,
                        "fold_cache_bytes": fold_cache_bytes,
                        "optuna_train_row_cap_per_fold": OPTUNA_MAX_TRAIN_ROWS_PER_FOLD,
                        "optuna_val_row_cap_per_fold": OPTUNA_MAX_VAL_ROWS_PER_FOLD,
                        "final_training_uses_all_rows": True,
                        "static_data_includes_lut_and_bounds": True,
                    },
                }
                exported_models.append(final_model)
                model_infos.append(info)
        except Exception as exc:
            message = f"{group_name}/{model_type}: {type(exc).__name__}: {exc}"
            family_failures.append(message)
            error_path = output_dir / f"{group_name}_{model_key}_study_error.txt"
            error_path.write_text(message + "\n\n" + traceback.format_exc(), encoding="utf-8")
            print(f"❌ {message}")
            continue

    if family_failures:
        family_failure_report.write_text("\n".join(family_failures), encoding="utf-8")
        print(
            f"[WARNING] {len(family_failures)} family study failure(s): "
            f"{family_failure_report}"
        )
    if not exported_models:
        raise RuntimeError("No final XGBoost or LightGBM models were produced.")
    return exported_models, model_infos, schema_path


# =================================================================================================
# MAIN — ALL SETTINGS COME FROM THE GLOBAL CONFIGURATION ABOVE
# =================================================================================================
def _validate_global_configuration() -> None:
    client = _configured_roadmap()
    if not 0 <= int(TEST_FOLD_ID) < client.n_folds:
        raise ValueError(f"TEST_FOLD_ID must be in 0..{client.n_folds - 1}.")
    if int(N_FOLDS) != client.n_folds:
        raise ValueError(f"N_FOLDS={N_FOLDS} != roadmap n_folds={client.n_folds}.")
    if not 2 <= int(CV_FOLDS) <= client.n_folds - 1:
        raise ValueError(f"CV_FOLDS must be between 2 and {client.n_folds - 1}.")
    if not (0 < MIN_REALISTIC_SIZE <= MAX_MODEL_SIZE_BYTES):
        raise ValueError("Require 0 < MIN_REALISTIC_SIZE <= MAX_MODEL_SIZE_BYTES")
    if TRAINING_CPU_BUDGET <= 0:
        raise ValueError("TRAINING_CPU_BUDGET must be positive.")
    for name, value in {
        "XGBOOST_OPTUNA_JOBS": XGBOOST_OPTUNA_JOBS,
        "LIGHTGBM_OPTUNA_JOBS": LIGHTGBM_OPTUNA_JOBS,
    }.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")
    for name, value in {
        "XGBOOST_MODEL_THREADS": XGBOOST_MODEL_THREADS,
        "LIGHTGBM_MODEL_THREADS": LIGHTGBM_MODEL_THREADS,
    }.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be zero (auto) or positive.")
    if CV_FOLDS < 2:
        raise ValueError("CV_FOLDS must be at least 2.")
    for name, value in {
        "OPTUNA_MAX_TRAIN_ROWS_PER_FOLD": OPTUNA_MAX_TRAIN_ROWS_PER_FOLD,
        "OPTUNA_MAX_VAL_ROWS_PER_FOLD": OPTUNA_MAX_VAL_ROWS_PER_FOLD,
    }.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be zero (all rows) or positive.")
    if OPTUNA_MIN_ROWS_PER_JOINT_STRATUM <= 0:
        raise ValueError("OPTUNA_MIN_ROWS_PER_JOINT_STRATUM must be positive.")
    if CSV_WORKERS <= 0:
        raise ValueError("CSV_WORKERS must be positive.")
    if PERMUTATION_TOP_RANKS < 0:
        raise ValueError("PERMUTATION_TOP_RANKS must be non-negative.")
    if PERMUTATION_MAX_ROWS <= 0:
        raise ValueError("PERMUTATION_MAX_ROWS must be positive.")
    if FOLD_CACHE_MODE not in {"auto", "on", "off"}:
        raise ValueError("FOLD_CACHE_MODE must be 'auto', 'on', or 'off'.")
    if not TRAIN_MODEL_FAMILIES:
        raise ValueError("TRAIN_MODEL_FAMILIES cannot be empty.")
    unknown = [name for name in TRAIN_MODEL_FAMILIES if name not in MODEL_FAMILY_ORDER]
    if unknown:
        raise ValueError(f"Unsupported TRAIN_MODEL_FAMILIES entries: {unknown}.")
    if len(set(TRAIN_MODEL_FAMILIES)) != len(TRAIN_MODEL_FAMILIES):
        raise ValueError("TRAIN_MODEL_FAMILIES must not contain duplicates.")
    configured_trials = {
        "XGBoost": int(XGBOOST_TRIALS),
        "LightGBM": int(LIGHTGBM_TRIALS),
    }
    for model_type in TRAIN_MODEL_FAMILIES:
        if configured_trials[model_type] <= 0:
            raise ValueError(f"{model_type} trial count must be positive.")


def main() -> int:
    _validate_global_configuration()

    output_dir = Path(TRAIN_FOLDER)
    data_folder = Path(DATA_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    group_names = list(TRAIN_GROUPS) if TRAIN_GROUPS is not None else list(GESTURE_GROUPS_ID_TRAIN.keys())
    model_types = tuple(TRAIN_MODEL_FAMILIES)
    trials_by_model = {
        "XGBoost": int(XGBOOST_TRIALS),
        "LightGBM": int(LIGHTGBM_TRIALS),
    }
    total_trials = sum(trials_by_model[model_type] for model_type in model_types)
    failures: List[str] = []

    print(
        "[PLAN] Sequential studies: "
        + ", ".join(
            f"{model_type}={trials_by_model[model_type]} trials"
            for model_type in model_types
        )
    )
    print(
        f"[PLAN] CPU budget: physical={PHYSICAL_CPUS}, logical={LOGICAL_CPUS}, "
        f"training budget={TRAINING_CPU_BUDGET}; fold-cache={FOLD_CACHE_MODE}"
    )
    for model_type in model_types:
        jobs, threads = resolve_family_cpu(model_type)
        print(
            f"[PLAN] {model_type}: {jobs} Optuna jobs x {threads} model threads "
            f"= {jobs * threads} aggregate workers"
        )
    print(
        f"[PLAN] Optuna CV: requested_folds={CV_FOLDS}, "
        f"train_cap={OPTUNA_MAX_TRAIN_ROWS_PER_FOLD:,}, "
        f"val_cap={OPTUNA_MAX_VAL_ROWS_PER_FOLD:,}; final models use all rows."
    )
    print("[PLAN] Split source: canonical JSON; no random holdout or CV. "
          f"test_fold={TEST_FOLD_ID}; dataset={FOLD_ROADMAP_DATASET}")
    print("[PLAN] All detailed classes tracked; rare-class absences reported; "
          "binary [0,1] is required in every split side.")

    for group_name in group_names:
        if group_name == "NONE":
            continue
        try:
            models, infos, _ = optimize_tree_simple(
                group_name,
                n_trials=total_trials,
                output_dir=output_dir,
                data_folder=data_folder,
                model_types=model_types,
                trials_by_model=trials_by_model,
                resume=TO_RESUME,
                csv_workers=CSV_WORKERS,
                fold_cache=FOLD_CACHE_MODE,
                permutation_top_ranks=PERMUTATION_TOP_RANKS,
                permutation_max_rows=PERMUTATION_MAX_ROWS,
            )
            for model_container, info in zip(models, infos):
                tag = f"{group_name}_testfold{TEST_FOLD_ID}_{info['model_key']}_rank{info['family_rank']}_{timestamp}"
                model_path = output_dir / f"{tag}_model.pkl"
                info_path = output_dir / f"{tag}_info.json"
                header_path = output_dir / f"{tag}_model.h"

                joblib.dump(model_container, model_path, compress=3)
                info_path.write_text(
                    json.dumps(info, indent=2, cls=NumpyEncoder),
                    encoding="utf-8",
                )
                try:
                    export_report = export_to_c_header(
                        model=model_container,
                        trial_info=info,
                        output_path=header_path,
                        feature_schema_path=info["schema_file"],
                        validate=True,
                    )
                    report_path = output_dir / f"{tag}_export_report.json"
                    report_path.write_text(
                        json.dumps(export_report, indent=2, cls=NumpyEncoder),
                        encoding="utf-8",
                    )
                    print(
                        f"✅ Saved {info['model_type']} rank {info['family_rank']} for {group_name}: "
                        f"preprocessing={info['preprocessing']}, "
                        f"tree payload={export_report['tree_payload_bytes']} bytes"
                    )
                except Exception as export_exc:
                    message = (
                        f"{group_name} {info['model_type']} rank {info['family_rank']} export: "
                        f"{type(export_exc).__name__}: {export_exc}"
                    )
                    failures.append(message)
                    export_error_path = output_dir / f"{tag}_export_error.txt"
                    export_error_path.write_text(
                        message + "\n\n" + traceback.format_exc(),
                        encoding="utf-8",
                    )
                    print(f"⚠️  {message}")
                    continue
            family_failure_report = output_dir / f"{group_name}_family_study_failures.txt"
            if family_failure_report.exists():
                failures.extend(
                    line.strip()
                    for line in family_failure_report.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        except Exception as exc:
            failures.append(f"{group_name}: {type(exc).__name__}: {exc}")
            print(f"❌ Error for {group_name}: {exc}")
            traceback.print_exc()

    if failures:
        failure_path = output_dir / f"training_failures_{timestamp}.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        print(
            f"Training completed with {len(failures)} failure(s); details: {failure_path}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())