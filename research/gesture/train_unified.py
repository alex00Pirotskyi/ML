#!/usr/bin/env python3
"""One entrypoint to train two embedded Mudra models from Claude's unified y_true (version 2).

    python -u research/gesture/train_unified.py

TAP_SWIPE:  1 = TAP/SWIPE onset, 0 = no onset, unknown = excluded.
LONG_PRESS: 1 = RELEASE/REST,    0 = sustained HOLD, unknown = excluded.

What changed against the first unified trainer (see research/gesture/README.md):
  * Training sample: every labeled onset row, a time-strided background, and explicit
    hard negatives (rows near presses, rows with muscle activity), instead of 45 rows
    per class per recording.
  * Optuna scores what the device does: TAP by event replay at the fixed 0.85
    (onsets detected vs false events per minute), LP by press replay with the
    firmware's ten-sample average (holds kept, timely releases, taps released).
  * Wider LightGBM search (80-400 trees, depth 3-10, up to 127 leaves). Every model
    is trimmed to the model-data budget exactly, so the search trades trees for depth
    inside the budget instead of minimising size.
  * Performance-first selection: the best replay score (mean over person folds) inside
    MODEL_BUDGET_KB is the default header, unless it never fires / never releases on the
    held-out people, in which case the next candidate is promoted. The best inside
    ALT_BUDGET_KB is exported beside it.
  * The previous successful TAP model, LP v20 and the 2026-11-17 unified models are
    replayed READ-ONLY as baselines on the same held-out people.

Only directly logged p_core-> fields reach the models. PointerState, folder,
timestamps, contact scores, hold_ms and subject are labels, sampling aids or split
metadata, never model features. The threshold is fixed at 0.85 everywhere.
"""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "tap_real"))
sys.path.insert(0, str(ROOT / "research" / "gesture"))
sys.path.insert(0, str(ROOT))
# Import sets environment thread controls BEFORE loading the numerical stack.
import train_claude_v2_optuna as previous_tap

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import replay as R

v6 = previous_tap.v6
export_plus = v6.export_plus

# ============================================================================ settings
# Intentionally no CLI. Change these constants (or the MUDRA_* environment variables) if needed.
SOURCE_TAP = ROOT / "DATA" / "DATA_TAP"
SOURCE_LP = ROOT / "DATA" / "DATA_LONG_PRESS"
LABEL_TAP = ROOT / "DATA" / "LABELS" / "DATA_TAP"
LABEL_LP = ROOT / "DATA" / "LABELS" / "DATA_LONG_PRESS"
OUTPUT = ROOT / "research" / "gesture" / "results_unified_v2"
PREVIOUS_UNIFIED = ROOT / "research" / "gesture" / "results_unified_20261117"
CLAUDE_LABELER = ROOT / "claude_labeling" / "label_gestures.py"
TRIALS_PER_MODEL = 40
TOP_K_PER_MODEL = 5
CV_FOLDS = 3
TEST_FRACTION = 0.22
FIXED_THRESHOLD = 0.85
CPU_THREADS = previous_tap.CPU_THREADS
SEED = 20261118
VERIFIED_HASHES = True
FILE_LIMIT_PER_FOLDER = None        # testing aid: use at most this many recordings per folder

# Model-data budget: exact static C model bytes (export_plus), the same number the header reports.
MODEL_BUDGET_KB = float(os.environ.get("MUDRA_MODEL_BUDGET_KB", 32))
ALT_BUDGET_KB = (None if os.environ.get("MUDRA_ALT_BUDGET_KB", "24").lower() in ("", "none", "0")
                 else float(os.environ.get("MUDRA_ALT_BUDGET_KB", "24")))
NODE_BYTES = 6                      # export_plus packed LightGBM node: float threshold + 2 bytes

# LightGBM search space (agreed ranges). Leaves are capped at 2**depth and at 127 (uint8 child index).
N_ESTIMATORS = (80, 400)
MAX_DEPTH = (3, 10)
NUM_LEAVES = (8, 127)
LEARNING_RATE = (0.02, 0.3)
MIN_CHILD_SAMPLES = (10, 300)
POSITIVE_WEIGHT = dict(tap=(0.2, 10.0), lp=(0.2, 20.0))

# Training sample (times in ms, from the label files)
TAP_BACKGROUND_STRIDE_MS = 40       # ordinary negatives: one row per 40 ms
TAP_HARD_WINDOW_MS = 500            # negatives within 500 ms of any press or sensor event ...
TAP_HARD_STRIDE_MS = 20             # ... are taken every 20 ms
TAP_ACTIVE_STRIDE_MS = 20           # negatives with muscle activity (any_active == 1) every 20 ms
TAP_POSITIVE_STRIDE_MS = 10         # onset rows every 10 ms (about 8 rows per 80 ms onset window)
LP_HOLD_STRIDE_MS = 10              # held rows every 10 ms (all rows in the first 600 ms of a hold)
LP_EARLY_HOLD_MS = 600
LP_RELEASE_WINDOW_MS = 1000         # release rows within 1 s after a long-press release, every 10 ms
LP_TAP_RELEASE_WINDOW_MS = 600      # release rows within 600 ms after a tap/swipe release ...
LP_TAP_RELEASE_STRIDE_MS = 40       # ... every 40 ms (there are about 19,000 taps)
LP_HARD_STRIDE_MS = 10              # release rows after a long-press release: every 10 ms
LP_BACKGROUND_STRIDE_MS = 100       # all other release/rest rows: one per 100 ms

# Replay
REPLAY_LP_TAPS_PER_RECORDING = 3    # taps per TAP/SWIPE recording replayed for the LP "tap must release" check
REPLAY_LP_AFTER_RELEASE_MS = 2000   # replay this long after each pointer release
PARITY_ROWS = 200_000               # rows used to compare the exported C header with Python

# Baselines: replayed read-only, never written. Environment variables override the paths.
BASELINES = dict(
    tap=[("previous_tap_thr085", os.environ.get("MUDRA_TAP_BASELINE_HEADER",
          str(ROOT / "research" / "tap_real" / "results_claude_v2_optuna_126_thr085" / "models" / "TAP_SWIPE.h"))),
         ("unified_20261117", str(PREVIOUS_UNIFIED / "tap" / "models" / "TAP_SWIPE.h"))],
    lp=[("lp_v20", os.environ.get("MUDRA_LP_V20_HEADER", "")),
        ("unified_20261117", str(PREVIOUS_UNIFIED / "lp" / "models" / "LONG_PRESS.h"))],
)

ALL_RAW = tuple(previous_tap.FULL_126)
assert len(ALL_RAW) == 126 and len(set(ALL_RAW)) == 126
assert all(x.startswith("p_core->") for x in ALL_RAW)

TAP_FOLDERS = frozenset(("TAP", "SWIPE"))
HOLD_FOLDERS = frozenset((
    "SOFT_LONG_PRESS", "MID_LONG_PRESS", "STRONG_LONG_PRESS",
    "LONG_PRESS", "DRAG_AND_DROP"))
TAP_NEGATIVES = frozenset(("ARM", "BAND", "FINGER", "WRIST", "NOTHING"))
LP_NEGATIVES = frozenset(("CONNECTED", "DISCONNECTED", "LOOSE", "TABLE", "NOTHING"))
TAP_CLASSES = TAP_FOLDERS | TAP_NEGATIVES
LP_CLASSES = HOLD_FOLDERS | TAP_FOLDERS | TAP_NEGATIVES | LP_NEGATIVES

TASKS = (
    dict(name="tap", header="TAP_SWIPE", class_one="TAP/SWIPE onset", class_zero="not an onset",
         metric="event replay at 0.85: onset recall x min(1, target / false events per minute)"),
    dict(name="lp", header="LONG_PRESS", class_one="RELEASE/REST", class_zero="HELD",
         metric="press replay, ten-sample average at 0.85: weighted harmonic mean of holds kept, "
                "timely releases and taps released"),
)
SAMPLE_KINDS = {0: "background", 1: "hard_near_press", 2: "hard_muscle_activity", 3: "positive",
                4: "hold", 5: "release_after_press"}


def notice(*values):
    print(*values, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    v6.save_json(path, obj)


def budget_bytes(kb):
    return None if kb is None else int(round(float(kb) * 1024))


# ============================================================================ labels (unchanged logic)
def canonical_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing recording folder: {root}")
    files = sorted(root.glob("*/*.csv.gz"))
    if not files:
        raise ValueError(f"No compressed original recordings under {root}")
    seen = set()
    for file in files:
        rel = file.relative_to(root).as_posix()
        if rel in seen:
            raise ValueError(f"Duplicate recording: {rel}")
        seen.add(rel)
    return files


def sidecar(source: Path, source_root: Path, labels_root: Path) -> Path:
    return labels_root / source.relative_to(source_root).parent / (
        source.name[:-7] + ".labels.csv.gz")


def labels_complete(source: Path, output: Path, files: list[Path]) -> bool:
    summary = output / "_summary.json"
    if not summary.is_file():
        return False
    state = json.loads(summary.read_text(encoding="utf-8"))
    total = state.get("totals", {})
    args = total.get("args", {})
    if total.get("failed") != 0 or int(total.get("files", -1)) < len(files):
        return False
    if int(total["files"]) > len(files):
        if next(source.glob("*/*.csv"), None) is None:
            return False
    if args.get("kind", "auto") != "auto":
        return False
    if args.get("disconnect_rule", "all") != "all":
        return False
    if args.get("use_weak", False) or args.get("promote_double_partners", False):
        return False
    if args.get("trust_pointer", False):
        return False
    if list(args.get("tap_win", [0, 80])) != [0, 80]:
        return False
    if not (output / "_events.csv.gz").is_file():
        return False
    return all(sidecar(p, source, output).is_file() for p in files)


def ensure_labels(root: Path, label_dir: Path) -> list[Path]:
    """Reuse matching Claude outputs, otherwise generate without overwriting."""
    files = canonical_files(root)
    if labels_complete(root, label_dir, files):
        notice("UNIFIED_LABELS_REUSED", root.name, len(files), label_dir)
        return files
    if label_dir.exists():
        raise RuntimeError(
            f"Labels exist but are incomplete/incompatible: {label_dir}; "
            "move that directory out of the way, never overwrite it silently.")
    staging = label_dir.with_name(label_dir.name + ".building")
    if staging.exists():
        raise RuntimeError(f"Partial Claude output exists: {staging}; inspect it before retrying.")
    staging.parent.mkdir(parents=True, exist_ok=True)
    extra_csv = next(root.glob("*/*.csv"), None) is not None
    input_dir = root
    if extra_csv:
        symlink_root = OUTPUT / "_canonical_label_input" / root.name
        for p in files:
            dst = symlink_root / p.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                if dst.resolve() != p.resolve():
                    raise RuntimeError("Staging link points to different recording: " + str(dst))
            elif dst.exists():
                raise RuntimeError(f"Refuse to overwrite staged file {dst}")
            else:
                dst.symlink_to(p.resolve())
        input_dir = symlink_root
        notice("UNIFIED_CANONICAL_GZ_ONLY", root.name, len(files))
    notice("UNIFIED_LABELS_GENERATING", root.name, len(files))
    subprocess.run([sys.executable, str(CLAUDE_LABELER), str(input_dir), str(staging)],
                   check=True, cwd=ROOT)
    if not labels_complete(root, staging, files):
        raise ValueError(f"Claude unified label export failed audit: {staging}")
    staging.rename(label_dir)
    notice("UNIFIED_LABELS_READY", root.name, len(files), label_dir)
    return files


def selected_sources(task: str, tap_files: list[Path], lp_files: list[Path]):
    rows = []
    if task == "tap":
        rows.extend((p, SOURCE_TAP, LABEL_TAP) for p in tap_files if p.parent.name in TAP_CLASSES)
        # Long-press starts/releases are UNKNOWN in unified tap labels; the rest are negatives.
        rows.extend((p, SOURCE_LP, LABEL_LP) for p in lp_files if p.parent.name in HOLD_FOLDERS)
    else:
        rows.extend((p, SOURCE_LP, LABEL_LP) for p in lp_files if p.parent.name in LP_CLASSES)
    if FILE_LIMIT_PER_FOLDER:                      # testing aid: evenly spread, so many people are kept
        groups = {}
        for r in sorted(rows, key=lambda a: (a[1].name, str(a[0]))):
            groups.setdefault((r[1].name, r[0].parent.name), []).append(r)
        rows = [g[i] for g in groups.values()
                for i in np.unique(np.linspace(0, len(g) - 1, min(len(g), FILE_LIMIT_PER_FOLDER)).astype(int))]
    if not rows:
        raise ValueError(f"No usable {task} recordings")
    return sorted(rows, key=lambda a: (a[1].name, str(a[0])))


def common_p_core(records) -> list[str]:
    common = set(ALL_RAW)
    for p, _, _ in records:
        with gzip.open(p, "rt", newline="") as handle:
            names = set(next(csv.reader(handle)))
        common.intersection_update(names)
    cols = [f for f in ALL_RAW if f in common]
    if not set(previous_tap.EMG_CURRENT + previous_tap.WL_FULL).issubset(cols):
        raise ValueError("Essential instantaneous EMG/WL firmware columns unavailable")
    if not cols or any(not x.startswith("p_core->") for x in cols):
        raise ValueError("Only original p_core fields allowed")
    export_plus.validate_feature_schema_accessors(cols)
    notice("UNIFIED_FEATURES", len(cols), "/", len(ALL_RAW),
           "missing", [x for x in ALL_RAW if x not in common][:18])
    return cols


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    have = set(columns)
    sets = {"all_available": columns[:]}
    for k, fs in previous_tap.FEATURE_SETS.items():
        if k == "full126":
            continue
        if set(fs).issubset(have):
            sets[k] = list(fs)
    return sets


def load_events(label_root: Path) -> dict:
    ev = pd.read_csv(label_root / "_events.csv.gz", low_memory=False)
    return {k: g for k, g in ev.groupby("recording")}


# ============================================================================ sampling
def stride_pick(t_ms: np.ndarray, mask: np.ndarray, stride_ms: float) -> np.ndarray:
    """First row of every stride_ms time bin among rows where mask is true."""
    idx = np.flatnonzero(mask)
    if not len(idx) or stride_ms <= 0:
        return idx
    b = np.floor(t_ms[idx] / stride_ms).astype(np.int64)
    first = np.r_[True, b[1:] != b[:-1]]
    return idx[first]


def near(t_ms: np.ndarray, times, window_ms: float) -> np.ndarray:
    times = np.sort(np.asarray([x for x in times if x == x], np.float64))
    if not len(times):
        return np.zeros(len(t_ms), bool)
    j = np.searchsorted(times, t_ms)
    lo = np.abs(t_ms - times[np.clip(j - 1, 0, len(times) - 1)])
    hi = np.abs(t_ms - times[np.clip(j, 0, len(times) - 1)])
    return np.minimum(lo, hi) <= window_ms


def sample_tap(t, y, active, ev):
    """Return row indices and sample kind for the TAP training set."""
    times = []
    if ev is not None and len(ev):
        for c in ("t_on", "t_ptr", "t_rel"):
            if c in ev:
                times.extend(ev[c].dropna().tolist())
    neg = y == 0
    pos = stride_pick(t, y == 1, TAP_POSITIVE_STRIDE_MS)
    hard = stride_pick(t, neg & near(t, times, TAP_HARD_WINDOW_MS), TAP_HARD_STRIDE_MS)
    act = stride_pick(t, neg & (active == 1), TAP_ACTIVE_STRIDE_MS)
    bg = stride_pick(t, neg, TAP_BACKGROUND_STRIDE_MS)
    kind = {}
    for idx, k in ((bg, 0), (act, 2), (hard, 1), (pos, 3)):        # later kinds win
        for i in idx: kind[int(i)] = k
    rows = np.fromiter(kind.keys(), np.int64, len(kind)); order = np.argsort(rows)
    return rows[order], np.fromiter(kind.values(), np.int8, len(kind))[order]


def sample_lp(t, y, hold_ms, folder, ev):
    """y is the FIRMWARE class: 1 = RELEASE/REST, 0 = HELD."""
    held = y == 0
    rel = y == 1
    in_tap_press = np.zeros(len(t), bool)
    after = []; tap_after = np.zeros(len(t), bool)
    if ev is not None and len(ev):
        if folder in TAP_FOLDERS:
            for x in ev[ev.type == "tap_press"].itertuples():
                if x.t_rel == x.t_rel and x.t_on == x.t_on:
                    in_tap_press |= (t >= min(x.t_on, x.t_ptr)) & (t < x.t_rel)
                    tap_after |= (t >= x.t_rel) & (t <= x.t_rel + LP_TAP_RELEASE_WINDOW_MS)
        else:
            for x in ev[ev.type == "hold_press"].itertuples():
                if x.t_rel == x.t_rel:
                    after.append((x.t_rel, LP_RELEASE_WINDOW_MS))
    rel = rel & ~in_tap_press          # a finger still pressing a tap is not "released": leave it out
    hard_mask = np.zeros(len(t), bool)
    for r, w in after:
        hard_mask |= (t >= r) & (t <= r + w)
    h = stride_pick(t, held, LP_HOLD_STRIDE_MS)
    early = np.flatnonzero(held & (hold_ms >= 0) & (hold_ms <= LP_EARLY_HOLD_MS))
    hard = stride_pick(t, rel & hard_mask, LP_HARD_STRIDE_MS)
    tap_rel = stride_pick(t, rel & tap_after, LP_TAP_RELEASE_STRIDE_MS)
    bg = stride_pick(t, rel, LP_BACKGROUND_STRIDE_MS)
    kind = {}
    for idx, k in ((bg, 0), (tap_rel, 5), (hard, 5), (h, 4), (early, 4)):
        for i in idx: kind[int(i)] = k
    rows = np.fromiter(kind.keys(), np.int64, len(kind)); order = np.argsort(rows)
    return rows[order], np.fromiter(kind.values(), np.int8, len(kind))[order]


# ============================================================================ dataset + replay cache
def _fingerprint(task, records, columns):
    h = hashlib.sha256()
    h.update(json.dumps(dict(task=task, columns=columns, version=2,
        tap=[TAP_BACKGROUND_STRIDE_MS, TAP_HARD_WINDOW_MS, TAP_HARD_STRIDE_MS, TAP_ACTIVE_STRIDE_MS,
             TAP_POSITIVE_STRIDE_MS],
        lp=[LP_HOLD_STRIDE_MS, LP_EARLY_HOLD_MS, LP_RELEASE_WINDOW_MS, LP_TAP_RELEASE_WINDOW_MS,
            LP_TAP_RELEASE_STRIDE_MS, LP_HARD_STRIDE_MS, LP_BACKGROUND_STRIDE_MS],
        replay=[REPLAY_LP_TAPS_PER_RECORDING, REPLAY_LP_AFTER_RELEASE_MS]), sort_keys=True).encode())
    h.update(sha256(CLAUDE_LABELER).encode())
    for p, root, labels in records:
        st = p.stat(); lab = sidecar(p, root, labels).stat()
        h.update(f"{p.relative_to(ROOT)}|{st.st_size}|{lab.st_size}|{int(lab.st_mtime)}".encode())
    return h.hexdigest()


def build_task_data(task: str, records, out: Path):
    """Training sample in memory, full replay rows in an on-disk float32 matrix (cached)."""
    columns = common_p_core(records)
    cache = out / "cache"
    fp = _fingerprint(task, records, columns)
    stamp = cache / "fingerprint.json"
    if stamp.is_file() and json.loads(stamp.read_text()).get("fingerprint") == fp:
        notice("UNIFIED_CACHE_REUSED", task, cache)
        return load_cache(cache, columns)
    if cache.exists():
        shutil.rmtree(cache)                     # derived data only
    cache.mkdir(parents=True)
    events = {"tap": load_events(LABEL_TAP) if task == "tap" else None, "lp": load_events(LABEL_LP)}
    ncol = len(columns)
    raw = open(cache / "replay_X.f32", "wb"); nrep = 0
    T = dict(X=[], y=[], kind=[], rec=[], t=[])
    recs = []          # per recording metadata
    rp_t, rp_lab = [], []
    windows = []       # LP press windows
    onsets = []        # TAP confirmed onsets: (rec index, t_on)
    info = Counter()
    for i, (source, root, labels_root) in enumerate(records, 1):
        label_file = sidecar(source, root, labels_root)
        folder = source.parent.name
        rel = source.relative_to(root).as_posix()
        ev_all = (events["tap"] if root == SOURCE_TAP else events["lp"]) if task == "tap" else events["lp"]
        ev = ev_all.get(rel) if ev_all is not None else None
        lab = pd.read_csv(label_file, compression="gzip", low_memory=False,
                          usecols=["Timestamp", "tap_label", "hold_label", "hold_ms", "any_active"])
        original = lab["tap_label" if task == "tap" else "hold_label"].to_numpy(np.int8)
        if not np.isin(original, (-1, 0, 1)).all():
            raise ValueError(f"Invalid silver labels in {label_file}")
        y = np.where(original < 0, -1, original if task == "tap" else 1 - original).astype(np.int8)
        if task == "tap" and folder in HOLD_FOLDERS and (y == 1).any():
            raise ValueError(f"Unexpected positive tap onset in hold folder {source}")
        if task == "lp" and folder in TAP_FOLDERS and (y == 0).any():
            raise ValueError(f"Unexpected HOLD in TAP or SWIPE folder {source}")
        if task == "tap" and folder in TAP_FOLDERS and not (y == 1).any():
            info["positive_bags_without_confirmed_event"] += 1; continue
        if task == "lp" and folder in HOLD_FOLDERS and not (y == 0).any():
            info["hold_bags_without_confirmed_event"] += 1; continue
        ts = lab["Timestamp"].to_numpy(np.int64); t = (ts - ts[0]).astype(np.float64)
        if task == "tap":
            rows, kinds = sample_tap(t, y, lab["any_active"].to_numpy(np.int8), ev)
        else:
            rows, kinds = sample_lp(t, y, lab["hold_ms"].to_numpy(np.int32), folder, ev)
        df = pd.read_csv(source, compression="gzip", low_memory=False,
                         usecols=lambda c: c in set(columns) | {"Timestamp", "User"})
        if len(df) != len(lab):
            raise ValueError(f"Row count mismatch: {source}")
        if not np.array_equal(pd.to_numeric(df["Timestamp"], errors="raise").to_numpy(np.int64), ts):
            raise ValueError(f"Timestamp mismatch: {source}")
        X = df[columns].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
        if not np.isfinite(X).all():
            raise ValueError(f"Nonfinite firmware input in {source}")
        user = str(df["User"].iloc[0]) if "User" in df.columns else ""
        who = v6.person_group(user, source.name[:-7])
        ri = len(recs)
        # ---- replay rows
        if task == "tap":
            start = nrep; raw.write(X.tobytes()); nrep += len(X)
            rp_t.append(t.astype(np.float32)); rp_lab.append(y)
            if ev is not None and len(ev):
                for o in ev[(ev.type == "tap_press") & (ev.labeled == 1)].t_on.dropna():
                    onsets.append((ri, float(o)))
            recs.append(dict(file=source.relative_to(ROOT).as_posix(), folder=folder, group=who,
                             start=start, end=nrep))
        else:
            recs.append(dict(file=source.relative_to(ROOT).as_posix(), folder=folder, group=who, start=-1, end=-1))
            presses = []
            if ev is not None and len(ev):
                if folder in HOLD_FOLDERS:
                    for x in ev[(ev.type == "hold_press") & (ev.labeled == 1)].itertuples():
                        presses.append(("hold", float(x.t_on), float(x.t_rel)))
                elif folder in TAP_FOLDERS:
                    tp = ev[(ev.type == "tap_press") & (ev.labeled == 1)].sort_values("t_on")
                    if len(tp) > REPLAY_LP_TAPS_PER_RECORDING:
                        tp = tp.iloc[np.linspace(0, len(tp) - 1, REPLAY_LP_TAPS_PER_RECORDING).astype(int)]
                    for x in tp.itertuples():
                        presses.append(("tap", float(x.t_on), float(x.t_rel)))
            for kind_, s, r in presses:
                q = (t >= s) & (t <= r + REPLAY_LP_AFTER_RELEASE_MS)
                if q.sum() < R.LP_AVERAGE_SAMPLES:
                    continue
                a, b = np.flatnonzero(q)[[0, -1]]
                raw.write(X[a:b + 1].tobytes()); rp_t.append(t[a:b + 1].astype(np.float32))
                windows.append(dict(rec=ri, kind=kind_, start=nrep, end=nrep + (b + 1 - a), start_ms=s, release_ms=r))
                nrep += b + 1 - a
        # ---- training sample
        if len(rows):
            T["X"].append(X[rows]); T["y"].append(y[rows]); T["kind"].append(kinds)
            T["rec"].append(np.full(len(rows), ri, np.int32)); T["t"].append(t[rows].astype(np.float32))
        info["read_sources"] += 1; info[folder + "_files"] += 1
        if i % 300 == 0:
            notice("UNIFIED_DATASET_EXTRACT", task, i, len(records), sum(len(a) for a in T["y"]), "replay_rows", nrep)
    raw.close()
    if not T["X"]:
        raise ValueError(f"No supervised {task} recordings")
    rec_df = pd.DataFrame(recs)
    np.save(cache / "train_X.npy", np.concatenate(T["X"]).astype(np.float32))
    np.savez(cache / "train_meta.npz", y=np.concatenate(T["y"]), kind=np.concatenate(T["kind"]),
             rec=np.concatenate(T["rec"]), t=np.concatenate(T["t"]))
    np.savez(cache / "replay_meta.npz", t=np.concatenate(rp_t) if rp_t else np.zeros(0, np.float32),
             label=np.concatenate(rp_lab) if rp_lab else np.zeros(0, np.int8), n=nrep, ncol=ncol)
    rec_df.to_csv(cache / "recordings.csv", index=False)
    pd.DataFrame(windows, columns=["rec", "kind", "start", "end", "start_ms", "release_ms"]).to_csv(
        cache / "lp_windows.csv", index=False)
    pd.DataFrame(onsets, columns=["rec", "t_on"]).to_csv(cache / "tap_onsets.csv", index=False)
    write_json(cache / "info.json", dict(counts=dict(info)))
    write_json(stamp, dict(fingerprint=fp))
    return load_cache(cache, columns)


def load_cache(cache: Path, columns):
    X = np.load(cache / "train_X.npy")
    m = np.load(cache / "train_meta.npz")
    rp = np.load(cache / "replay_meta.npz")
    recs = pd.read_csv(cache / "recordings.csv")
    n = int(rp["n"]); ncol = int(rp["ncol"])
    replay_X = (np.memmap(cache / "replay_X.f32", dtype=np.float32, mode="r", shape=(n, ncol))
                if n else np.zeros((0, ncol), np.float32))
    windows = pd.read_csv(cache / "lp_windows.csv")
    onsets = pd.read_csv(cache / "tap_onsets.csv")
    rec = m["rec"]
    data = dict(X=X, y=m["y"].astype(np.int8), kind=m["kind"], t=m["t"],
                group=recs.group.to_numpy(object)[rec], file=recs.file.to_numpy(object)[rec],
                category=recs.folder.to_numpy(object)[rec], rec=rec,
                w=np.ones(len(rec), np.float32), columns=list(columns), feature_sets=feature_sets(list(columns)))
    rpl = dict(X=replay_X, t=rp["t"], label=rp["label"], recs=recs, windows=windows,
               onsets={int(k): g.t_on.to_numpy(float) for k, g in onsets.groupby("rec")} if len(onsets) else {})
    info = json.loads((cache / "info.json").read_text())["counts"]
    notice("UNIFIED_DATA", json.dumps(dict(train_rows=int(len(data["y"])), positives=int((data["y"] == 1).sum()),
           negatives=int((data["y"] == 0).sum()), replay_rows=n, recordings=len(recs),
           train_GB=round(X.nbytes / 1e9, 2), replay_GB=round(n * ncol * 4 / 1e9, 2),
           kinds={SAMPLE_KINDS[int(k)]: int(v) for k, v in zip(*np.unique(data["kind"], return_counts=True))})))
    return data, rpl, info


# ============================================================================ split, weights, params
def split_heldout(data, task):
    g = data["group"]; y = data["y"]; c = data["category"]
    splitter = GroupShuffleSplit(n_splits=200, test_size=TEST_FRACTION, random_state=SEED + int(task == "lp"))
    for tr, te in splitter.split(data["X"], y, g):
        if set(y[tr]) != {0, 1} or set(y[te]) != {0, 1}:
            continue
        if len(set(g[tr])) < CV_FOLDS + 1:
            continue
        if task == "tap" and not {"TAP", "SWIPE"}.issubset(set(c[te])):
            continue
        if task == "tap" and not (set(c[te]) & TAP_NEGATIVES):
            continue
        if task == "lp" and not (set(c[te]) & HOLD_FOLDERS):
            continue
        if set(g[tr]) & set(g[te]):
            raise AssertionError("Subject overlap in holdout")
        return tr, te
    raise RuntimeError(f"No usable person-disjoint {task} split, inspect category coverage")


def weights_for_task(data, indices, task, positive_weight):
    """File-normalized class balancing, then the tunable positive-class weight (see v1 notes)."""
    y = data["y"][indices]
    w = data["w"][indices].astype(np.float32).copy()
    file_counts = Counter(data["file"][indices])
    w /= np.asarray([math.sqrt(file_counts[f]) for f in data["file"][indices]], dtype=np.float32)
    for cls in (0, 1):
        take = y == cls
        total = float(w[take].sum())
        if total <= 0:
            raise ValueError(f"Missing class {cls} in train or evaluation weights")
        w[take] *= len(indices) / (2.0 * total)
    if task == "lp":
        is_hold_folder = np.isin(data["category"][indices], list(HOLD_FOLDERS))
        inside = (y == 1) & is_hold_folder
        outside = (y == 1) & ~is_hold_folder
        sw = float(w[inside].sum()); ow = float(w[outside].sum())
        if sw > 0 and ow > 0:
            total = sw + ow
            w[inside] *= (0.5 * total) / sw
            w[outside] *= (0.5 * total) / ow
    w[y == 1] *= float(positive_weight)
    return w.astype(np.float32)


def suggest_params(trial, task):
    depth = trial.suggest_int("max_depth", *MAX_DEPTH)
    leaves = min(trial.suggest_int("num_leaves", *NUM_LEAVES, log=True), 2 ** depth, 127)
    params = dict(objective="binary", n_estimators=trial.suggest_int("n_estimators", *N_ESTIMATORS, step=20),
                  max_depth=depth, num_leaves=leaves,
                  learning_rate=trial.suggest_float("learning_rate", *LEARNING_RATE, log=True),
                  min_child_samples=trial.suggest_int("min_child_samples", *MIN_CHILD_SAMPLES, log=True),
                  reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
                  reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
                  colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                  subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                  max_bin=127, verbosity=-1, n_jobs=CPU_THREADS, random_state=SEED, deterministic=True)
    return params


def fit_cap(params, budget):
    """Trees worth fitting: enough to fill the budget even if trees come out smaller than num_leaves."""
    full_tree = NODE_BYTES * (2 * params["num_leaves"] - 1)
    return int(max(20, min(params["n_estimators"], math.ceil(3 * budget / full_tree))))


def fit_model(X, y, w, params, budget):
    p = dict(params); p["n_estimators"] = fit_cap(params, budget)
    model = lgb.LGBMClassifier(**p)
    model.fit(X, y, sample_weight=w)
    return model


# ============================================================================ exact budget trimming
def static_bytes(model) -> int:
    return int(export_plus._estimate_compact_tree_payload_legacy(model)["total_static_data_bytes"])


def truncated(model, k: int):
    """The same model keeping its first k trees (boosting is additive, no refit)."""
    if k >= model.booster_.num_trees():
        return model
    m = copy.copy(model)
    m._Booster = lgb.Booster(model_str=model.booster_.model_to_string(num_iteration=k))
    return m


def trees_within(model, budget: int) -> tuple[int, int]:
    """Largest number of leading trees whose exported C model fits the budget, and its exact bytes.
    Exported size only grows with the tree count, so a binary search on the exact size is exact."""
    n = model.booster_.num_trees()
    full = static_bytes(model)
    if full <= budget:
        return n, full
    sizes = {}
    def size(k):
        if k not in sizes:
            sizes[k] = static_bytes(truncated(model, k))
        return sizes[k]
    lo, hi = 1, n - 1
    if size(lo) > budget:
        raise ValueError(f"Even one tree ({size(lo)} bytes) exceeds the budget of {budget} bytes")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if size(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo, size(lo)


# ============================================================================ replay scoring
def _ranges_predict(booster_fns, Xmm, ranges, selected, chunk=250_000):
    """Yield (range, outputs per fn) for contiguous replay ranges, batching small ranges."""
    batch, size = [], 0
    def flush(batch):
        X = np.concatenate([np.asarray(Xmm[a:b]) for a, b in batch])[:, selected]
        X = np.ascontiguousarray(X, dtype=np.float32)
        outs = [fn(X) for fn in booster_fns]
        pos = 0
        for a, b in batch:
            yield (a, b), [({k: v[pos:pos + b - a] for k, v in o.items()} if isinstance(o, dict)
                            else o[pos:pos + b - a]) for o in outs]
            pos += b - a
    for a, b in ranges:
        batch.append((a, b)); size += b - a
        if size >= chunk:
            yield from flush(batch); batch, size = [], 0
    if batch:
        yield from flush(batch)


def raw_fns(model, k_list):
    """Probability functions for each leading-tree count, sharing the tree evaluations."""
    bst = model.booster_; ks = sorted(set(k_list))
    def run(X):
        out, acc, prev = {}, None, 0
        for k in ks:
            part = bst.predict(X, raw_score=True, start_iteration=prev, num_iteration=k - prev)
            acc = part if acc is None else acc + part
            out[k] = R.sigmoid(acc); prev = k
        return out
    return run


def replay_tap(prob_fn, rpl, groups, selected):
    """prob_fn(X) -> dict name -> probability (or decision) arrays. Returns name -> TapCounts."""
    recs = rpl["recs"]; pick = recs[recs.group.isin(groups)]
    counts = {}
    ranges = list(zip(pick.start.astype(int), pick.end.astype(int)))
    by_start = {int(r.start): (int(i), r) for i, r in pick.iterrows()}
    for (a, b), (outs,) in _ranges_predict([prob_fn], rpl["X"], ranges, selected):
        ri, r = by_start[a]
        for name, p in outs.items():
            d = p if p.dtype == bool else (p >= FIXED_THRESHOLD)
            R.tap_replay_recording(d, rpl["t"][a:b], rpl["label"][a:b], rpl["onsets"].get(ri, []),
                                   r.folder, counts.setdefault(name, R.TapCounts()))
    return counts


def replay_lp(q_fn, rpl, groups, selected):
    """q_fn(X) -> dict name -> Q20 integer arrays. Returns name -> LpCounts."""
    w = rpl["windows"]
    if not len(w):
        return {}
    recs = rpl["recs"]
    keep = w[np.isin(recs.group.to_numpy(object)[w.rec.to_numpy(int)], list(groups))]
    counts = {}
    ranges = list(zip(keep.start.astype(int), keep.end.astype(int)))
    meta = {int(r.start): r for r in keep.itertuples()}
    for (a, b), (outs,) in _ranges_predict([q_fn], rpl["X"], ranges, selected):
        r = meta[a]; folder = recs.folder.iloc[int(r.rec)]
        for name, q in outs.items():
            R.lp_replay_press(q, rpl["t"][a:b], r.start_ms, r.release_ms, r.kind, folder,
                              counts.setdefault(name, R.LpCounts()))
    return counts


def model_prob_fn(model, ks, task):
    run = raw_fns(model, ks)
    if task == "tap":
        return run
    return lambda X: {k: R.q20(p) for k, p in run(X).items()}


# ============================================================================ Optuna objective
def objective(trial, data, rpl, train_ids, task, budgets):
    name = trial.suggest_categorical("feature_set", list(data["feature_sets"]))
    selected = [data["columns"].index(x) for x in data["feature_sets"][name]]
    lo, hi = POSITIVE_WEIGHT[task]
    positive_weight = trial.suggest_float("positive_weight", lo, hi, log=True)
    params = suggest_params(trial, task)
    primary = budgets[0]
    totals = {b: (R.TapCounts() if task == "tap" else R.LpCounts()) for b in budgets}
    folds = []
    t0 = time.monotonic()
    for fold, (a, b) in enumerate(GroupKFold(n_splits=CV_FOLDS).split(
            train_ids, data["y"][train_ids], data["group"][train_ids])):
        tr = train_ids[a]; val_groups = set(data["group"][train_ids[b]])
        if set(np.unique(data["y"][tr])) != {0, 1}:
            raise optuna.TrialPruned("Missing class in person-grouped fold")
        model = fit_model(data["X"][tr][:, selected], data["y"][tr],
                          weights_for_task(data, tr, task, positive_weight), params, primary)
        ks = {bud: trees_within(model, bud) for bud in budgets}
        fn = model_prob_fn(model, [k for k, _ in ks.values()], task)
        res = (replay_tap if task == "tap" else replay_lp)(fn, rpl, val_groups, selected)
        fold_rep = {}
        for bud, (k, size) in ks.items():
            c = res.get(k)
            if c is not None:
                totals[bud].add(c)
            rep_ = c.report() if c is not None else {}
            usable = bool(c is not None and (c.onsets if task == "tap" else (c.holds + c.taps)))
            fold_rep[str(bud)] = dict(trees=k, bytes=size, usable=usable,
                                      **{kk: vv for kk, vv in rep_.items() if kk != "per_folder"})
        folds.append(dict(fold=fold, people=len(val_groups), fitted_trees=model.booster_.num_trees(), budgets=fold_rep))
        running = [f["budgets"][str(primary)]["objective_score"] for f in folds if f["budgets"][str(primary)]["usable"]]
        trial.report(float(np.mean(running)) if running else 0.0, fold)
        if trial.should_prune():
            raise optuna.TrialPruned(f"fold {fold} below median")
    reports = {}
    for bud in budgets:
        pooled = totals[bud].report()
        per_fold = [f["budgets"][str(bud)]["objective_score"] for f in folds if f["budgets"][str(bud)]["usable"]]
        # Objective = mean of per-fold scores: a setting that only works for some people scores lower.
        pooled["pooled_objective_score"] = pooled["objective_score"]
        pooled["fold_scores"] = per_fold
        pooled["objective_score"] = float(np.mean(per_fold)) if per_fold else 0.0
        reports[str(bud)] = pooled
    trial.set_user_attr("training_params", params)
    trial.set_user_attr("selected_features", list(data["feature_sets"][name]))
    trial.set_user_attr("folds", folds)
    trial.set_user_attr("cv", reports)
    trial.set_user_attr("seconds", round(time.monotonic() - t0, 1))
    trial.set_user_attr("fixed_decision_threshold", FIXED_THRESHOLD)
    for bud in budgets:
        trial.set_user_attr(f"score_{bud}", reports[str(bud)]["objective_score"])
        trial.set_user_attr(f"max_bytes_{bud}", max(f["budgets"][str(bud)]["bytes"] for f in folds))
    return float(reports[str(primary)]["objective_score"])


# ============================================================================ header export and checks (unchanged API)
def uppercase_alias(header: Path, stem: str) -> None:
    source = header.read_text(encoding="utf-8")
    guard = f"{stem}_MODEL_H"
    final = f"#endif /* {guard} */"
    native = stem.lower() + "_predict"
    if final not in source or f"{native}(const t_core*" not in source:
        raise ValueError(f"Header {header} lacks native prediction function or expected guard")
    if "V6_RANK" in source or "v6_rank" in source:
        raise ValueError("Trial ID leaked into C firmware API")
    aliases = {
        f"{stem}_predict": native,
        f"{stem}_predict_value": stem.lower() + "_predict_value",
        f"{stem}_probability_q20": stem.lower() + "_probability_q20",
    }
    for public, private in aliases.items():
        if f"{private}(" not in source:
            raise ValueError(f"Missing native C prediction API {private} in {header}")
        if f"#define {public} " not in source:
            source = source.replace(final, f"#ifndef {public}\n#define {public} {private}\n#endif\n\n" + final)
    header.write_text(source, encoding="utf-8", newline="\n")


def check_header(header: Path, stem: str, output: Path) -> dict:
    uppercase_alias(header, stem)
    check = previous_tap.compile_header_126(header, stem.lower(), output)
    if check["status"] != "passed":
        raise RuntimeError(f"C compiler not available / syntax failed: {check}")
    stub = output / "c_compiler_check"
    src = stub / (stem.lower() + "_value_api.c")
    src.write_text('#include "' + header.name + '"\n'
                   'float model_value_api(const t_core *p_core) { return '
                   + stem + '_predict_value(p_core) + '
                   + stem + '_probability_q20(p_core)*0.05f; }\n')
    cc = shutil.which("cc") or shutil.which("gcc")
    command = [cc, "-std=c11", "-O2", "-c", str(src), "-I", str(stub),
               "-I", str(header.parent), "-o", str(stub / (stem.lower() + "_value_api.o"))]
    result = subprocess.run(command, text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError("Stable probability C API failed to compile: " + result.stderr[-2600:])
    check["stable_probability_api"] = "passed"
    return check


def write_header(model, subset, stem, folder: Path, task, outdir: Path):
    folder.mkdir(parents=True, exist_ok=True)
    schema = folder / (stem + "_feature_schema.json")
    write_json(schema, {"feature_map": {str(k): name for k, name in enumerate(subset)},
                        "positive_class_id": 1,
                        "positive_semantics": ("RELEASE/REST" if task == "lp" else "TAP/SWIPE ONSET"),
                        "source": "Claude unified labels, original p_core fields only"})
    info = dict(group_name=stem, selected_indices=list(range(len(subset))), decision_threshold=FIXED_THRESHOLD,
                target_class_id=1, preprocessing="none", params=model.get_params())
    header = folder / (stem + ".h")
    saved = export_plus.export_to_c_header(dict(model=model, preprocessing="none", indices=list(range(len(subset)))),
                                           info, header, schema, validate=True)
    checked = check_header(header, stem, outdir)
    size = int(saved["total_static_data_bytes"])
    if size != static_bytes(model):
        raise ValueError(f"Exported size differs from estimate: {size} != {static_bytes(model)}")
    return header, size, checked


# ============================================================================ evaluation on held-out people
def header_replay(hm: R.HeaderModel, rpl, groups, task):
    if task == "tap":
        fn = lambda X: {"h": hm.decision(X)}
        res = replay_tap(fn, rpl, groups, slice(None))
    else:
        fn = lambda X: {"h": hm.value_q20(X)}
        res = replay_lp(fn, rpl, groups, slice(None))
    c = res.get("h")
    return c.report() if c is not None else None


def parity(hm: R.HeaderModel, model, rpl, groups, selected, task, k):
    recs = rpl["recs"]
    if task == "tap":
        rr = recs[recs.group.isin(groups)]; ranges = list(zip(rr.start, rr.end))
    else:
        w = rpl["windows"]; keep = w[np.isin(recs.group.to_numpy(object)[w.rec.to_numpy(int)], list(groups))]
        ranges = list(zip(keep.start, keep.end))
    rows, n = [], 0
    for a, b in ranges:
        if n >= PARITY_ROWS: break
        rows.append((int(a), int(min(b, a + PARITY_ROWS - n)))); n += rows[-1][1] - rows[-1][0]
    if not rows:
        return None
    X = np.concatenate([np.asarray(rpl["X"][a:b]) for a, b in rows])
    p = R.sigmoid(model.booster_.predict(np.ascontiguousarray(X[:, selected]), raw_score=True, num_iteration=k))
    if task == "tap":
        c = hm.predict_bool(np.ascontiguousarray(X[:, selected])).astype(bool); py = p >= FIXED_THRESHOLD
        return dict(rows=len(X), decision_mismatch=float((c != py).mean()))
    c = hm.value_q20(np.ascontiguousarray(X[:, selected])); py = R.q20(p)
    return dict(rows=len(X), q20_mismatch=float((c != py).mean()), q20_max_step_diff=int(np.abs(c - py).max()))


# ============================================================================ selection, export, baselines
def pick_trials(study, budgets):
    """Performance first: candidates ordered by grouped-CV replay score inside the primary budget
    (rank_01 = best), plus the best inside the smaller budget."""
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not done:
        raise RuntimeError("No completed trials")
    ranked = sorted(done, key=lambda t: t.value, reverse=True)
    out = [("rank_01", budgets[0], ranked[0])]
    if len(budgets) > 1:
        alt = max(done, key=lambda t: t.user_attrs.get(f"score_{budgets[1]}", -1.0))
        out.append((f"budget_{int(round(budgets[1] / 1024))}kb", budgets[1], alt))
    for t in ranked[1:]:
        if len(out) >= TOP_K_PER_MODEL:
            break
        out.append((f"rank_{len([o for o in out if o[0].startswith('rank')]) + 1:02d}", budgets[0], t))
    return out


def promote(out: Path, stem: str, candidates, budgets):
    """Default header = best CV candidate inside the primary budget whose held-out C replay is not a failure
    (it detects taps / releases at all). A candidate that fails is skipped and the reason recorded."""
    model_dir = out / "models"
    primary = [c for c in candidates if c["role"].startswith("rank_")]
    chosen, skipped = None, []
    for c in primary:
        h = c.get("holdout_c_header") or {}
        if h.get("objective_score", 0) > 0:
            chosen = c; break
        skipped.append(dict(role=c["role"], trial=c["trial"], reason="held-out C replay score is 0 "
                            "(never fires / never releases on unseen people)"))
    if chosen is None:
        chosen = primary[0]
        skipped.append(dict(role=chosen["role"], reason="no candidate passed; using the CV best anyway"))
    def copy_to(c, dst: Path):
        dst.mkdir(parents=True, exist_ok=True)
        src = out / c["header"]
        for f in (src, src.parent / (stem + "_feature_schema.json")):
            shutil.copyfile(f, dst / f.name)
    copy_to(chosen, model_dir)
    alt = [c for c in candidates if c["role"].startswith("budget_")]
    for c in alt:
        copy_to(c, model_dir / c["role"])
    return chosen, skipped


def export_candidates(data, rpl, task, stem, tr, te, picks, out: Path):
    model_dir = out / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    y = data["y"]; cols = data["columns"]
    train_groups = set(data["group"][tr]); held = set(data["group"][te])
    every = np.arange(len(y))
    rows = []
    for role, budget, trial in picks:
        params = trial.user_attrs["training_params"]; subset = trial.user_attrs["selected_features"]
        ix = [cols.index(x) for x in subset]; pw = float(trial.params["positive_weight"])
        folder = model_dir / "candidates" / role
        # 1) honest held-out replay: trained without the held-out people, exported, replayed as C
        pre = fit_model(data["X"][tr][:, ix], y[tr], weights_for_task(data, tr, task, pw), params, budget)
        k_pre, _ = trees_within(pre, budget); pre = truncated(pre, k_pre)
        tmp = out / "_holdout_headers" / role
        h_pre, b_pre, _ = write_header(pre, subset, stem, tmp, task, out)
        hm = R.HeaderModel(h_pre, subset, out / "_replay_build", prefer=stem.lower())
        sel = [cols.index(c) for c in subset]
        rpl_sub = dict(rpl, X=_ColumnView(rpl["X"], sel))
        holdout_c = header_replay(hm, rpl_sub, held, task)
        par = parity(hm, pre, rpl, held, sel, task, k_pre)
        # 2) final model on every person, trimmed to the budget, exported with the stable API
        final = fit_model(data["X"][:, ix], y, weights_for_task(data, every, task, pw), params, budget)
        k_fin, _ = trees_within(final, budget); final = truncated(final, k_fin)
        header, size, checked = write_header(final, subset, stem, folder, task, out)
        if size > budget:
            raise AssertionError(f"{role}: {size} bytes exceeds budget {budget}")
        tag = f"{role}_trial{trial.number:03d}"
        joblib.dump(dict(model=final, preprocessing="none", indices=list(range(len(subset))), feature_names=subset),
                    model_dir / (tag + ".pkl"), compress=3)
        report = dict(role=role, trial=trial.number, budget_bytes=budget, static_bytes=size, trees=k_fin,
                      params=params, positive_weight=pw, features=subset, header=str(header.relative_to(out)),
                      cv=trial.user_attrs["cv"].get(str(budget)), holdout_c_header=holdout_c,
                      holdout_model_bytes=b_pre, c_vs_python_parity=par, c_stub=checked,
                      limitations=["Silver labels from an offline teacher, not physical event truth",
                                   "Replay timing rules are documented assumptions about the firmware",
                                   "C replay uses a synthetic all-float t_core, not the nRF52832 ABI"])
        write_json(model_dir / (tag + ".json"), report)
        rows.append(report)
        notice("UNIFIED_MODEL_EXPORT", json.dumps(dict(task=task, role=role, trial=trial.number, bytes=size,
               trees=k_fin, cv=(report["cv"] or {}).get("objective_score"),
               holdout=(holdout_c or {}).get("objective_score"), parity=par)))
    return rows


class _ColumnView:
    """Row slices of the replay matrix restricted to a feature subset (for header replay)."""
    def __init__(self, X, sel):
        self.X, self.sel = X, sel
    def __getitem__(self, s):
        return np.asarray(self.X[s])[:, self.sel]
    def __len__(self):
        return len(self.X)


def replay_baselines(task, data, rpl, te, out: Path):
    held = set(data["group"][te]); everyone = set(data["group"])
    rows = []
    for name, path in BASELINES[task]:
        if not path:
            rows.append(dict(name=name, status="not configured")); continue
        p = Path(path)
        if not p.is_file():
            rows.append(dict(name=name, status="not found", path=str(p))); continue
        if out.resolve() in p.resolve().parents:
            raise AssertionError("A baseline must not live inside the new output folder")
        digest = sha256(p)
        try:
            text = p.read_text(encoding="utf-8", errors="replace").replace(" ", "")
            missing = [c for c in ALL_RAW if c not in data["columns"] and c.replace(" ", "") in text]
            if missing:
                raise ValueError(f"baseline uses fields missing from this data: {missing[:5]}")
            hm = R.HeaderModel(p, data["columns"], out / "_replay_build",
                               prefer=("tap_swipe" if task == "tap" else "long_press"))
            rows.append(dict(name=name, status="replayed", path=str(p), sha256=digest,
                             api=dict(value=hm.value_prefix, decision=hm.bool_prefix),
                             holdout=header_replay(hm, rpl, held, task),
                             all_people=header_replay(hm, rpl, everyone, task),
                             note="all_people may include this baseline's own training data"))
        except Exception as exc:          # a baseline problem must never stop training
            rows.append(dict(name=name, status="not replayable", path=str(p), error=repr(exc)[:600]))
        if sha256(p) != digest:
            raise AssertionError(f"Baseline file changed during replay: {p}")
        notice("UNIFIED_BASELINE", task, name, rows[-1]["status"])
    return rows


def comparison_table(task, candidates, baselines):
    key = (["recall", "false_per_min", "latency_ms_median", "objective_score"] if task == "tap"
           else ["hold_kept", "timely_release", "tap_released", "release_latency_ms_median", "objective_score"])
    head = "| Model | Bytes | " + " | ".join(key) + " |\n|" + " --- |" * (len(key) + 2) + "\n"
    fmt = lambda v: "-" if v is None or (isinstance(v, float) and v != v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = []
    for c in candidates:
        h = c.get("holdout_c_header") or {}
        mark = " **default**" if c.get("default") else ""
        lines.append(f"| new {c['role']} (trial {c['trial']}){mark} | {c['static_bytes']} | " + " | ".join(fmt(h.get(k)) for k in key) + " |")
    for b in baselines:
        h = b.get("holdout") or {}
        lines.append(f"| baseline {b['name']} ({b['status']}) | - | " + " | ".join(fmt(h.get(k)) for k in key) + " |")
    return head + "\n".join(lines) + "\n"


# ============================================================================ one task
def train_task(task: str, tap_files, lp_files):
    definition = next(v for v in TASKS if v["name"] == task)
    stem = definition["header"]
    out = OUTPUT / task
    summary_file = out / "summary.json"
    budgets = [b for b in (budget_bytes(MODEL_BUDGET_KB), budget_bytes(ALT_BUDGET_KB)) if b]
    if len(budgets) == 2 and budgets[1] >= budgets[0]:
        budgets = budgets[:1]
    if summary_file.is_file():
        saved = json.loads(summary_file.read_text())
        if (saved.get("status") == "success" and saved.get("fixed_threshold") == FIXED_THRESHOLD
                and saved.get("budgets") == budgets and (out / "models" / (stem + ".h")).is_file()):
            notice("UNIFIED_COMPLETE_REUSED", task, str(out))
            return saved
        raise ValueError("Existing output does not match this configuration: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    data, rpl, info = build_task_data(task, selected_sources(task, tap_files, lp_files), out)
    tr, te = split_heldout(data, task)
    notice("UNIFIED_TRAIN_BEGIN", json.dumps({
        "task": task, "subjects": len(set(data["group"])), "train_rows": len(tr), "holdout_rows": len(te),
        "holdout_people": sorted(set(data["group"][te])), "budgets_bytes": budgets,
        "class_one": definition["class_one"], "fixed_threshold": FIXED_THRESHOLD}))
    write_json(out / "split.json", dict(train_groups=sorted(set(data["group"][tr])),
                                         heldout_groups=sorted(set(data["group"][te])), seed=SEED + int(task == "lp")))
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED + int(task == "lp"), multivariate=True, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
        storage="sqlite:///" + str((out / "optuna_study.sqlite3").resolve()),
        study_name="unified_v2_" + task, load_if_exists=True)
    if len(study.trials) == 0:
        # A known-good anchor (the research student) and a deep-tree anchor, inside the budget.
        # LP needs confident release probabilities for a 0.85 average, hence the larger positive weight.
        pw0 = 6.0 if task == "lp" else 1.0
        study.enqueue_trial(dict(feature_set="all_available", positive_weight=pw0, max_depth=6, num_leaves=15,
                                 n_estimators=160, learning_rate=0.1, min_child_samples=100, reg_lambda=1.0,
                                 reg_alpha=1e-3, colsample_bytree=0.8, subsample=0.8))
        study.enqueue_trial(dict(feature_set="all_available", positive_weight=pw0, max_depth=10, num_leaves=63,
                                 n_estimators=200, learning_rate=0.2, min_child_samples=50, reg_lambda=1.0,
                                 reg_alpha=1e-3, colsample_bytree=0.7, subsample=0.8))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    finished = lambda: len([t for t in study.trials if t.state != optuna.trial.TrialState.WAITING
                            and t.state != optuna.trial.TrialState.RUNNING])
    while finished() < TRIALS_PER_MODEL:
        def fn(trial):
            z = objective(trial, data, rpl, tr, task, budgets)
            c = trial.user_attrs["cv"][str(budgets[0])]
            notice("UNIFIED_TRIAL", json.dumps({"task": task, "trial": trial.number, "score": round(z, 4),
                   **({"recall": round(c["recall"], 4), "false_per_min": round(c["false_per_min"], 3)} if task == "tap" else
                      {"hold_kept": round(c["hold_kept"], 4), "timely_release": round(c["timely_release"], 4),
                       "tap_released": c["tap_released"]}),
                   "bytes": trial.user_attrs[f"max_bytes_{budgets[0]}"], "seconds": trial.user_attrs["seconds"],
                   "leaves": trial.user_attrs["training_params"]["num_leaves"],
                   "depth": trial.user_attrs["training_params"]["max_depth"]}))
            return z
        study.optimize(fn, n_trials=1, n_jobs=1, show_progress_bar=False, catch=())
    picks = pick_trials(study, budgets)
    candidates = export_candidates(data, rpl, task, stem, tr, te, picks, out)
    default, skipped = promote(out, stem, candidates, budgets)
    for c in candidates:
        c["default"] = c is default
    baselines = replay_baselines(task, data, rpl, te, out)
    table = comparison_table(task, candidates, baselines)
    (out / "COMPARISON.md").write_text(f"# {task}: held-out people, replayed through the C headers\n\n" + table)
    notice(table)
    write_json(out / "models" / (stem + "_model_info.json"),
               dict(selected=default, skipped=skipped, fixed_threshold=FIXED_THRESHOLD, stable_header=stem + ".h",
                    api=stem + "_predict(&m_feat_core)", budgets_bytes=budgets,
                    rationale="Best grouped-CV replay score inside the primary model-data budget, "
                              "skipping candidates that never fire / never release on held-out people"))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    summary = dict(status="success", task=task, model_header=stem + ".h", class_one=definition["class_one"],
                   class_zero=definition["class_zero"], fixed_threshold=FIXED_THRESHOLD, threshold_tuned=False,
                   budgets=budgets, selected=default, skipped=skipped, candidates=candidates, baselines=baselines,
                   trials=len(done), pruned=len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                   dataset=dict(train_rows=int(len(data["y"])), replay_rows=int(len(rpl["X"])),
                                recordings=int(len(rpl["recs"])), subjects=len(set(data["group"])), counts=info),
                   cv_objective=definition["metric"],
                   replay_rules={k: getattr(R, k) for k in dir(R) if k.startswith(("TAP_", "LP_"))},
                   note="No production firmware overwrite; silver labels; LP class 1 is RELEASE.")
    write_json(summary_file, summary)
    notice("UNIFIED_TRAIN_END", json.dumps({"task": task, "header": str(out / "models" / (stem + ".h")),
                                            "bytes": default["static_bytes"]}))
    return summary


def main():
    t0 = time.monotonic()
    notice("UNIFIED_TRAINER_START", "cpu_threads", CPU_THREADS, "fixed_threshold", FIXED_THRESHOLD,
           "budget_kb", MODEL_BUDGET_KB, "alt_budget_kb", ALT_BUDGET_KB, "trials", TRIALS_PER_MODEL)
    files_t = ensure_labels(SOURCE_TAP, LABEL_TAP)
    files_l = ensure_labels(SOURCE_LP, LABEL_LP)
    a = train_task("tap", files_t, files_l)
    b = train_task("lp", files_t, files_l)
    write_json(OUTPUT / "UNIFIED_SUMMARY.json",
               dict(status="success", tap=a, lp=b, elapsed_s=round(time.monotonic() - t0, 1), baseline_unchanged=True))
    comp = "".join((OUTPUT / t / "COMPARISON.md").read_text() + "\n" for t in ("tap", "lp"))
    (OUTPUT / "COMPARISON.md").write_text(comp)
    notice("UNIFIED_BOTH_READY", str(OUTPUT / "tap" / "models" / "TAP_SWIPE.h"), str(OUTPUT / "lp" / "models" / "LONG_PRESS.h"))


if __name__ == "__main__":
    main()
