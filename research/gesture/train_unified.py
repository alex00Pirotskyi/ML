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

Final TAP recipe (TAP_FOCUS_RESULTS.md, "Final recipe"; 14 KB device budget), one line:

    MUDRA_TAP_RECIPE=final14 python -u research/gesture/train_unified.py

Only directly logged p_core-> fields reach the models. PointerState, folder,
timestamps, contact scores, hold_ms and subject are labels, sampling aids or split
metadata, never model features. The threshold is fixed at 0.85 everywhere.
"""
from __future__ import annotations

import copy
import csv
import faulthandler
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

import replay as R
import generate_lean8_features as lean8
import generate_lean_lite_features as lite
import tap_features as TF
import feature_selection as FS

v6 = previous_tap.v6
export_plus = v6.export_plus

# ============================================================================ settings
# Intentionally no CLI. Change these constants (or the MUDRA_* environment variables) if needed.
# MUDRA_TAP_RECIPE=final14: the final TAP recipe (TAP_FOCUS_RESULTS.md, "Final recipe") as defaults; a MUDRA_*
# variable set explicitly still wins. TAP model only (LP is unchanged and not trained by it).
TAP_RECIPES = {
    "final14": dict(MUDRA_TASKS="tap", MUDRA_USE_LEAN8="0", MUDRA_TAP_FEATURES="diff", MUDRA_TAP_AUG="mild",
                    MUDRA_TAP_LITE="diff", MUDRA_TAP_LABELS="press2", MUDRA_MODEL_BUDGET_KB="14",
                    MUDRA_ALT_BUDGET_KB="0", MUDRA_NUM_LEAVES="4,15", MUDRA_TRAIN_MIN_FRAMES="50",
                    MUDRA_EVAL_MIN_FRAMES="50", MUDRA_TAP_SOFT_LABELS="off"),
}
# final14 + observable-pattern labels (TAP_OBSERVABLE_RESULTS.md): per tap only its most distinguishable 60 ms
TAP_RECIPES["final14obs"] = dict(TAP_RECIPES["final14"], MUDRA_TAP_OBS_MS="60")
# Maximum protection (TAP_FILESPLIT_RESULTS.md): 40 ms observable labels, distillation from a full-size teacher
# (half label, half teacher) and finger / fist negatives x 2. Same size; keeps the most taps when the false-tap
# target is strict, and at 0.85 fires far less on finger motion and fists for about 1 point of tap recall.
TAP_RECIPES["final14protect"] = dict(TAP_RECIPES["final14"], MUDRA_TAP_OBS_MS="40", MUDRA_TAP_DISTILL="0.5",
                                     MUDRA_TAP_FINGER_WEIGHT="2")
TAP_RECIPE = os.environ.get("MUDRA_TAP_RECIPE", "").strip().lower()
if TAP_RECIPE:
    if TAP_RECIPE not in TAP_RECIPES:
        raise ValueError(f"MUDRA_TAP_RECIPE must be one of {sorted(TAP_RECIPES)}")
    for _k, _v in TAP_RECIPES[TAP_RECIPE].items():
        os.environ.setdefault(_k, _v)
RAW_SOURCE_TAP = ROOT / "DATA" / "DATA_TAP"
RAW_SOURCE_LP = ROOT / "DATA" / "DATA_LONG_PRESS"
# MUDRA_USE_LEAN8 picks the EMG context inputs next to the 126 raw firmware fields:
#   1 (default)  the 24 float Lean-8 fields (core_type_def.h, NEW_CLOAD_FEATURES)
#   0            none (firmware built without NEW_CLOAD_FEATURES)
#   lite         the 15 integer Lean-lite fields (research/gesture/firmware/lean_lite.h, ~6x cheaper than Lean-8)
# Rows, labels, frame gates and the held-out split are identical in all three (the same frame counter is in
# both derived trees), so the three runs are clean A/B arms. Each writes its own results folder and study.
_LEAN8_ENV = os.environ.get("MUDRA_USE_LEAN8", "1").strip().lower()
LEAN8_MODE = "lite" if _LEAN8_ENV == "lite" else ("off" if _LEAN8_ENV in ("0", "false", "no", "off") else "full")
USE_LEAN8 = LEAN8_MODE == "full"
USE_LITE = LEAN8_MODE == "lite"
LEAN8_TAG = "lite" if USE_LITE else USE_LEAN8        # recorded in fingerprints and reports (full/off: as before)
# TAP generalization (TAP_GENERALIZATION_RESULTS.md), TAP model only, runs without Lean inputs:
#   MUDRA_TAP_FEATURES=diff  adds 135 within-channel EMG PSD diffs + 9 WL ratios (firmware helpers diff()/ratio()),
#                            computed from the raw columns (no new data files)
#   MUDRA_TAP_AUG=mild       placement augmentation of the TAP training rows (EMG gain, spectral tilt, contact noise)
# Each setting writes its own results folder; the default (raw, none) is unchanged.
#   MUDRA_TAP_LITE=diff      adds the 15 gain-invariant Lean-lite diff inputs p_core->cload_lite.{rise10,fall10,
#                            range10,dev3,xm}[CH1..3] (firmware/lean_lite.h), computed per recording from the logged WL
#                            (tap_features.lite_rows; no new data files) and recomputed under augmentation/stress
#   MUDRA_TAP_LABELS=press2  "trust the press": in DoubleTaps and 240 bpm TAP recordings every real pointer press is
#                            a tap (TAP_FOCUS_RESULTS.md). Unknown presses become positives, negatives right at a
#                            press are left out. Training labels only: replay references and scores are unchanged.
# Each setting writes its own results folder; the default (raw, none, off, confirmed) is unchanged.
TAP_FEATURES = os.environ.get("MUDRA_TAP_FEATURES", "raw").strip().lower()
TAP_AUG = os.environ.get("MUDRA_TAP_AUG", "none").strip().lower()
TAP_LITE = os.environ.get("MUDRA_TAP_LITE", "off").strip().lower()
TAP_LABELS = os.environ.get("MUDRA_TAP_LABELS", "confirmed").strip().lower()
#   MUDRA_TAP_OBS_MS=60      observable-pattern labels (TAP_OBSERVABLE_RESULTS.md): of every tap's positive rows only the
#                            60 ms that a teacher, cross-fitted by person, finds most distinguishable from everything
#                            else are trained; the tap's other positive rows are left out (never made negatives).
#                            Done separately inside every CV fold's training people, the development people and
#                            everyone, so no score sees a teacher fitted on its own people. Training labels only.
_obs = os.environ.get("MUDRA_TAP_OBS_MS", "off").strip().lower()
TAP_OBS_MS = None if _obs in ("off", "0", "none", "") else float(_obs)
if TAP_FEATURES not in ("raw", "diff") or TAP_AUG not in ("none", "mild"):
    raise ValueError("MUDRA_TAP_FEATURES must be raw or diff; MUDRA_TAP_AUG must be none or mild")
if TAP_LITE not in ("off", "diff") or TAP_LABELS not in ("confirmed", "press2"):
    raise ValueError("MUDRA_TAP_LITE must be off or diff; MUDRA_TAP_LABELS must be confirmed or press2")
TAP_GENERALIZATION = (TAP_FEATURES != "raw" or TAP_AUG != "none" or TAP_LITE != "off" or TAP_LABELS != "confirmed"
                      or TAP_OBS_MS is not None)
if TAP_GENERALIZATION and LEAN8_MODE != "off":
    raise ValueError("MUDRA_TAP_FEATURES / MUDRA_TAP_AUG / MUDRA_TAP_LITE / MUDRA_TAP_LABELS need MUDRA_USE_LEAN8=0 "
                     "(researched without Lean-8 inputs; MUDRA_TAP_LITE=diff brings the Lean-lite diff fields itself)")
TAP_TAG = (f"_tap-{TAP_FEATURES}-{TAP_AUG}" + ("-litediff" if TAP_LITE == "diff" else "")
           + ("-press2" if TAP_LABELS == "press2" else "")
           + (f"-obs{TAP_OBS_MS:g}" if TAP_OBS_MS is not None else "")) if TAP_GENERALIZATION else ""
# Observable-pattern teacher (as researched): a full-size LightGBM (no budget) on all model inputs, 3 person folds.
OBS_TEACHER = dict(objective="binary", n_estimators=300, learning_rate=0.08, num_leaves=31, max_depth=8,
                   min_child_samples=100, reg_lambda=1.0, reg_alpha=1e-3, colsample_bytree=0.7, subsample=0.8,
                   subsample_freq=1, max_bin=127, verbosity=-1, deterministic=True)
OBS_ANCHOR_MS = (-5.0, 95.0)          # a positive row belongs to the onset / press at most 5 ms after, 95 ms before it
# press2 constants (as researched): presses of these TAP recordings; soft = the labeller's unknown presses with a tap
# teacher score >= 0.5 (positive rows in its 0-80 ms onset window), the other unknown presses are anchored at their
# onset estimate t_on (rows t_on - 5 .. t_on + 90); negatives t_on - 30 .. t_on + 120 of any real press are left out.
PRESS2_NAME_KEYS = ("DoubleTap", "240bpm")
PRESS2_SOFT_SCORE = 0.5
PRESS2_ANCHOR_MS = (-5, 90)
PRESS2_NEGATIVE_GAP_MS = (-30, 120)
PRESS2_MATCH_MS = 60
# A budget or leaves range other than the defaults gets its own TAP results folder (and study).
_TAP_KB = float(os.environ.get("MUDRA_MODEL_BUDGET_KB", 32))
_TAP_LEAVES = os.environ.get("MUDRA_NUM_LEAVES", "8,127").replace(" ", "")
if TAP_GENERALIZATION and (_TAP_KB != 32 or _TAP_LEAVES != "8,127"):
    TAP_TAG += f"-kb{_TAP_KB:g}" + ("" if _TAP_LEAVES == "8,127" else "-leaves" + _TAP_LEAVES.replace(",", "-"))
# Distillation (MUDRA_TAP_DISTILL = weight a of the label, e.g. 0.5): the TAP model learns soft targets
# a*label + (1-a)*teacher, where the teacher is a cross-fitted full-size model (no size budget) on all inputs. The
# small model then copies how sure a big model is, frame by frame, instead of only 0/1 (TAP_FILESPLIT_RESULTS.md).
# Trained as the usual binary model (two weighted rows per frame), so the size accounting and C export are unchanged.
_distill = os.environ.get("MUDRA_TAP_DISTILL", "off").strip().lower()
TAP_DISTILL = None if _distill in ("off", "none", "") else float(_distill)
if TAP_DISTILL is not None and not 0.0 <= TAP_DISTILL < 1.0:
    raise ValueError("MUDRA_TAP_DISTILL is the label weight a in [0, 1) (0.5: half label, half teacher), or off")
if TAP_GENERALIZATION and TAP_DISTILL is not None:
    TAP_TAG += f"-distill{round(TAP_DISTILL * 100):d}"
# Finger-motion and fist negatives weighted x k in the TAP model's training (MUDRA_TAP_FINGER_WEIGHT, default 1):
# the FINGER folder's frames count k times as much as other "not a tap" frames before the classes are balanced.
# Teachers (observable labels, distillation) keep plain weights, as researched (TAP_FILESPLIT_RESULTS.md).
TAP_FINGER_WEIGHT = float(os.environ.get("MUDRA_TAP_FINGER_WEIGHT", "1"))
if TAP_FINGER_WEIGHT <= 0:
    raise ValueError("MUDRA_TAP_FINGER_WEIGHT must be > 0")
if TAP_GENERALIZATION and TAP_FINGER_WEIGHT != 1.0:
    TAP_TAG += f"-fingerw{TAP_FINGER_WEIGHT:g}"
# A fixed false-tap target (MUDRA_TAP_FALSE_PER_MIN_TARGET=<number>): own results folder and study, so runs with
# different targets never share trials and can be compared side by side.
_TARGET_ENV = os.environ.get("MUDRA_TAP_FALSE_PER_MIN_TARGET", "baseline").strip().lower()
if TAP_GENERALIZATION and _TARGET_ENV != "baseline":
    TAP_TAG += f"-target{float(_TARGET_ENV):g}"
# Soft labels on: own results folder too, so a run with and without them can be compared side by side.
_SOFT_ENV = os.environ.get("MUDRA_TAP_SOFT_LABELS", "off").strip().lower()
if TAP_GENERALIZATION and _SOFT_ENV not in ("off", "0", "none", ""):
    TAP_TAG += f"-soft{float(_SOFT_ENV):g}"
TASKS_TO_RUN = [t.strip() for t in os.environ.get("MUDRA_TASKS", "tap,lp").lower().split(",") if t.strip()]
if not TASKS_TO_RUN or not set(TASKS_TO_RUN) <= {"tap", "lp"}:
    raise ValueError("MUDRA_TASKS must list tap and/or lp")
# Derived trees: original lines + the generated columns (and the frame counter used by the gates).
SOURCE_TAP = ROOT / "DATA" / ("DATA_TAP_LITE" if USE_LITE else "DATA_TAP_LEAN8")
SOURCE_LP = ROOT / "DATA" / ("DATA_LONG_PRESS_LITE" if USE_LITE else "DATA_LONG_PRESS_LEAN8")
# Frame gates count Lean-8 handler calls since the recording (sensor stream) started (46.4 frames/s).
# EVAL: rows the firmware runs the models on. model_execution() returns until lean_8_features_ready()
#   (500 frames, ~10.8 s), so candidates are scored, ranked and chosen on those rows only.
# TRAIN: rows used as training examples. From frame 50 (~1.1 s) the warm-up rows still teach the model
#   (FEATURE_SELECTION_RESULTS.md, section 2); only the least reliable first second is skipped.
# Held-out metrics are also reported for steady rows (>= LEAN8_STEADY_FRAMES), whatever the gates are.
LEAN8_STEADY_FRAMES = lean8.HISTORY_10S
TRAIN_MIN_FRAMES = int(os.environ.get("MUDRA_TRAIN_MIN_FRAMES", os.environ.get("MUDRA_LEAN8_MIN_FRAMES", 50)))
EVAL_MIN_FRAMES = int(os.environ.get("MUDRA_EVAL_MIN_FRAMES", LEAN8_STEADY_FRAMES))
LEAN8_MIN_FRAMES = TRAIN_MIN_FRAMES                  # older name of the train gate (research scripts)
# One pipeline: fold-local learned feature ranking with an integer K over all available fields, firmware-gated
# evaluation, CV-only candidate choice, TAP false-event budget anchored on the previous TAP model
# (FEATURE_SELECTION_RESEARCH_AND_EXECUTION_PLAN.md). The earlier feature-set menu ("v2") was removed after its
# study finished; its code is in git history (commit 53e3e83d).
if os.environ.get("MUDRA_EXPERIMENT", "v3").strip().lower() != "v3":
    raise ValueError("MUDRA_EXPERIMENT=v2 was removed; check out commit 53e3e83d to run it")
EXPERIMENT = "v3"
# TAP score = recall * min(1, target / false events per minute). Default "baseline": the target is the
# false-event rate of the previous production TAP header replayed on the development people (a number fixes it
# explicitly). A fixed 1.0 made the search give up ~25% of taps (FEATURE_SELECTION_RESULTS.md, section 1).
TAP_TARGET_SETTING = os.environ.get("MUDRA_TAP_FALSE_PER_MIN_TARGET", "baseline")
TAP_TARGET_BASELINE = "previous_tap_thr085"
# TAP soft labels (optional, OFF by default). Presses in TAP/SWIPE recordings that the labeller left UNKNOWN although
# its tap teacher saw a sensor tap peak with score >= this value become training positives in their 0-80 ms onset
# window, weighted by that score; their recall then also counts in the objective. Measured on the full data
# (FEATURE_SELECTION_RESULTS.md, section 5): no gain in recall of ALL real pointer presses at a matched false rate,
# +0.9-1.3 false events/min at 0.85 -> off. The reference recalls (strong / soft / press) are always reported.
_soft = os.environ.get("MUDRA_TAP_SOFT_LABELS", "off").strip().lower()
TAP_SOFT_MIN_SCORE = None if _soft in ("off", "0", "none", "") else float(_soft)
TAP_ONSET_WINDOW_MS = (0, 80)        # the labeller's onset window (label_gestures.py P["TAP_WIN"])
PRESS_WINDOW_MS = (150, 250)         # pointer-press reference: [min(press, onset) - 150, max(press, onset) + 250]
TAP_TARGET_FALLBACK = 1.0
LABEL_TAP = ROOT / "DATA" / "LABELS" / "DATA_TAP"
LABEL_LP = ROOT / "DATA" / "LABELS" / "DATA_LONG_PRESS"
# What the held-out split and the CV folds group by. person (default): a person is never on both sides (scores =
# people the model never saw). file: every recording file is its own group, so the same person (same day, same band
# position) can be on both sides; all people take part in every fold. Scores then mostly describe people the model
# was trained on (TAP_OBSERVABLE_RESULTS.md: ~4x fewer finger false taps on trained-on people). The shipped header is
# refitted on everyone either way. Its own results folder (_splitfile) and study.
# The file split is gesture-aware: the held-out files and the CV folds are stratified by recording type (folder,
# gesture from the file name, and bpm for TAP / SWIPE), so every part gets its share of each gesture
# (recording_stratum, file_strata). Types with fewer than STRATUM_MIN_FILES files are pooled per folder.
SPLIT_BY = os.environ.get("MUDRA_SPLIT", "person").strip().lower()
if SPLIT_BY not in ("person", "file"):
    raise ValueError("MUDRA_SPLIT must be person or file")
SPLIT_TAG = "" if SPLIT_BY == "person" else "_splitfile"
# Study-name suffix: the file split is gesture-aware (stratified, v1); a study scored on the earlier plain file
# split is never resumed with these folds.
SPLIT_STUDY_TAG = "" if SPLIT_BY == "person" else "_splitfile_gesture1"
OUTPUT = ROOT / "research" / "gesture" / (dict(full="results_unified_v3_feature_selection",
                                               off="results_unified_v3_feature_selection_no_lean8",
                                               lite="results_unified_v3_feature_selection_lean_lite")[LEAN8_MODE]
                                          + TAP_TAG + SPLIT_TAG)
PREVIOUS_UNIFIED = ROOT / "research" / "gesture" / "results_unified_20261117"
CLAUDE_LABELER = ROOT / "claude_labeling" / "label_gestures.py"
TRIALS_PER_MODEL = 40
TOP_K_PER_MODEL = 5
CV_FOLDS = 3
EXPORT_MAX_FEATURES = 254             # export_plus compact node format: uint8 feature index, 0xFF marks a leaf
TEST_FRACTION = 0.22
FIXED_THRESHOLD = 0.85                # TAP_SWIPE_predict(): onset when p >= 0.85 (unchanged)
# LONG_PRESS: firmware releases when the mean of the last 10 Q20 values >= this. Changing it is a NEW
# experiment: the LP Optuna study name carries it, so an LP study run at another threshold is never resumed.
LP_RELEASE_THRESHOLD = float(os.environ.get("MUDRA_LP_RELEASE_THRESHOLD", 0.5))
# Thread count does not enter the models (LightGBM deterministic=True); benchmark with bench_unified.py threads.
CPU_THREADS = int(os.environ.get("MUDRA_CPU_THREADS", 0)) or previous_tap.CPU_THREADS
# The 3 CV folds of a trial run at the same time, each LightGBM with an equal
# share of CPU_THREADS. One LightGBM on ~0.5M rows does not scale to dozens of threads, and the replay scoring
# between fits is single-threaded, so one fold at a time left most of a large server idle. Trials stay
# sequential (Optuna's sampler depends on their order). MUDRA_FOLD_WORKERS=1 restores one fold at a time.
FOLD_WORKERS = max(1, min(CV_FOLDS, int(os.environ.get("MUDRA_FOLD_WORKERS", 0)) or CPU_THREADS // 8 or 1))
FOLD_THREADS = max(1, CPU_THREADS // FOLD_WORKERS)
SEED = 20261118
VERIFIED_HASHES = True
FILE_LIMIT_PER_FOLDER = None        # testing aid: use at most this many recordings per folder

# Model-data budget: exact static C model bytes (export_plus), the same number the header reports.
MODEL_BUDGET_KB = float(os.environ.get("MUDRA_MODEL_BUDGET_KB", 32))
ALT_BUDGET_KB = (None if os.environ.get("MUDRA_ALT_BUDGET_KB", "24").lower() in ("", "none", "0")
                 else float(os.environ.get("MUDRA_ALT_BUDGET_KB", "24")))
NODE_BYTES = 6                      # export_plus packed LightGBM node: float threshold + 2 bytes

# LightGBM search space (agreed ranges). Leaves are capped at 2**depth and at 127 (uint8 child index).
# Tree-count range of the search. MUDRA_N_ESTIMATORS=lo,hi widens it for bigger budgets: 400 trees of 4 leaves fill only
# about 18 KB (make train KB=32 passes 80,800).
N_ESTIMATORS = tuple(int(v) for v in os.environ.get("MUDRA_N_ESTIMATORS", "80,400").split(","))
if len(N_ESTIMATORS) != 2 or not 20 <= N_ESTIMATORS[0] < N_ESTIMATORS[1]:
    raise ValueError("MUDRA_N_ESTIMATORS must be lo,hi with 20 <= lo < hi")
MAX_DEPTH = (3, 10)
NUM_LEAVES = tuple(int(v) for v in os.environ.get("MUDRA_NUM_LEAVES", "8,127").split(","))
if len(NUM_LEAVES) != 2 or not 2 <= NUM_LEAVES[0] <= NUM_LEAVES[1] <= 127:
    raise ValueError("MUDRA_NUM_LEAVES must be 'low,high' with 2 <= low <= high <= 127")
# Small budgets need small trees (TAP_FOCUS_RESULTS.md, 14 KB: 7 leaves = 172 trees ~ the 32 KB quality, 31 leaves =
# 38 trees collapses): the final recipe searches 4..15 (the TAP results folder names a non-default range).
LEARNING_RATE = (0.02, 0.3)
MIN_CHILD_SAMPLES = (10, 300)
POSITIVE_WEIGHT = dict(tap=(0.2, 10.0), lp=(0.2, 20.0))
if LP_RELEASE_THRESHOLD != 0.85:
    # Measured: the LP replay optimum moves from threshold 1.0 (weight 6) to 0.85 (0.2) to 0.7 (0.05);
    # a 0.5 release threshold needs release weights around 0.01, so the LP range must reach that low.
    POSITIVE_WEIGHT["lp"] = (0.005, 20.0)
LP_ANCHOR_POSITIVE_WEIGHT = 6.0 if LP_RELEASE_THRESHOLD >= 0.85 else 0.02
# Earlier LP headers (v20, 2026-11-17) were made for a 0.85 firmware release; they are replayed at their own setting.
LP_BASELINE_RELEASE_THRESHOLD = float(os.environ.get("MUDRA_LP_BASELINE_THRESHOLD", 0.85))

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
         ("unified_20261117", str(PREVIOUS_UNIFIED / "tap" / "models" / "TAP_SWIPE.h")),
         # e.g. today's no-Lean TAP header, for an A/B on the same held-out people and rules
         ("extra_baseline", os.environ.get("MUDRA_TAP_BASELINE_EXTRA", ""))],
    lp=[("lp_v20", os.environ.get("MUDRA_LP_V20_HEADER", "")),
        ("unified_20261117", str(PREVIOUS_UNIFIED / "lp" / "models" / "LONG_PRESS.h"))],
)

RAW_126 = tuple(previous_tap.FULL_126)
LEAN8_24 = tuple(lean8.LEAN8_COLUMNS)
LITE_15 = tuple(lite.LITE_COLUMNS)
ALL_FEATURES = RAW_126 + dict(full=LEAN8_24, off=(), lite=LITE_15)[LEAN8_MODE]
assert len(RAW_126) == 126 and len(set(RAW_126)) == 126
assert len(LEAN8_24) == 24 and len(set(LEAN8_24)) == 24
assert len(LITE_15) == 15 and len(set(LITE_15)) == 15
assert len(ALL_FEATURES) == dict(full=150, off=126, lite=141)[LEAN8_MODE] and len(set(ALL_FEATURES)) == len(ALL_FEATURES)
assert all(x.startswith("p_core->") for x in ALL_FEATURES)

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
                4: "hold", 5: "release_after_press", 6: "positive_soft"}


def task_threshold(task: str) -> float:
    return FIXED_THRESHOLD if task == "tap" else LP_RELEASE_THRESHOLD


def study_name(task: str) -> str:
    """Every setting that changes the objective or the data is part of the name."""
    name = (f"unified_v3_{task}_fs{FS.RANKER_VERSION}_lean8{'lite' if USE_LITE else int(USE_LEAN8)}"
            f"_train{TRAIN_MIN_FRAMES}_eval{EVAL_MIN_FRAMES}")
    if task == "tap":
        name += f"_fpm{R.TAP_FALSE_PER_MIN_TARGET:.3f}"
        if TAP_SOFT_MIN_SCORE is not None:
            name += f"_soft{TAP_SOFT_MIN_SCORE:.2f}"
        name += TAP_TAG
    else:
        name += f"_rel{LP_RELEASE_THRESHOLD:.2f}"
    return name + SPLIT_STUDY_TAG


class StageClock:
    """Wall-clock seconds per named stage (time.perf_counter); never changes any result.
    seconds: summed over all threads; local(): this thread's seconds since local_reset() (per CV fold)."""

    def __init__(self):
        self.seconds = Counter()
        self._lock = threading.Lock()
        self._local = threading.local()

    def reset(self):
        with self._lock:
            self.seconds = Counter()

    def local_reset(self):
        self._local.seconds = Counter()

    def local(self):
        return dict(getattr(self._local, "seconds", Counter()))

    def add(self, name, dt):
        with self._lock:
            self.seconds[name] += dt
        mine = getattr(self._local, "seconds", None)
        if mine is not None:
            mine[name] += dt

    def __call__(self, name):
        clock = self

        class _Span:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                clock.add(name, time.perf_counter() - self.t0)
        return _Span()


CLOCK = StageClock()
TIMING_LOG = None                   # set per task: JSONL of per-trial stage timings
_FOLD_POOL = None


def fold_pool():
    """Thread pool for concurrent CV folds (None = one fold at a time). LightGBM fit and predict release the
    GIL; each fold uses its own model, gather buffer and counters."""
    global _FOLD_POOL
    if FOLD_WORKERS > 1 and _FOLD_POOL is None:
        _FOLD_POOL = ThreadPoolExecutor(max_workers=FOLD_WORKERS, thread_name_prefix="cv-fold")
    return _FOLD_POOL if FOLD_WORKERS > 1 else None


def notice(*values):
    print(*values, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(obj, where="", found=None):
    """NaN / inf (an undefined metric, e.g. a latency without any hit) -> null, with the places reported once."""
    if isinstance(obj, dict):
        return {k: _json_safe(v, f"{where}.{k}", found) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, f"{where}[{i}]", found) for i, v in enumerate(obj)]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        if found is not None:
            found.append(where)
        return None
    return obj


def write_json(path: Path, obj) -> None:
    found = []
    obj = _json_safe(obj, "", found)
    if found:
        notice("UNIFIED_JSON_UNDEFINED", Path(path).name, len(found), "undefined values written as null, e.g.", found[:6])
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


def _feature_root(raw_root: Path) -> Path:
    return SOURCE_TAP if raw_root == RAW_SOURCE_TAP else SOURCE_LP


def _feature_file(raw: Path, raw_root: Path, feature_root: Path) -> Path:
    return _feature_root(raw_root) / raw.relative_to(raw_root)


def selected_sources(task: str, tap_files: list[Path], lp_files: list[Path]):
    rows = []
    if task == "tap":
        rows.extend(
            (_feature_file(p, RAW_SOURCE_TAP, SOURCE_TAP), _feature_root(RAW_SOURCE_TAP), LABEL_TAP)
            for p in tap_files if p.parent.name in TAP_CLASSES
        )
        # Long-press starts/releases are UNKNOWN in unified tap labels; the rest are negatives.
        rows.extend(
            (_feature_file(p, RAW_SOURCE_LP, SOURCE_LP), _feature_root(RAW_SOURCE_LP), LABEL_LP)
            for p in lp_files if p.parent.name in HOLD_FOLDERS
        )
    else:
        rows.extend(
            (_feature_file(p, RAW_SOURCE_LP, SOURCE_LP), _feature_root(RAW_SOURCE_LP), LABEL_LP)
            for p in lp_files if p.parent.name in LP_CLASSES
        )
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
    common = set(ALL_FEATURES)
    for p, _, _ in records:
        with gzip.open(p, "rt", newline="") as handle:
            names = set(next(csv.reader(handle)))
        common.intersection_update(names)
    cols = [f for f in ALL_FEATURES if f in common]
    if USE_LITE and not set(LITE_15).issubset(cols):
        raise ValueError(f"Lean-lite columns missing in the derived files: {sorted(set(LITE_15) - set(cols))[:5]}")
    if not set(previous_tap.EMG_CURRENT + previous_tap.WL_FULL).issubset(cols):
        raise ValueError("Essential instantaneous EMG/WL firmware columns unavailable")
    if not cols or any(not x.startswith("p_core->") for x in cols):
        raise ValueError("Only p_core firmware fields are allowed")
    export_plus.validate_feature_schema_accessors(cols)
    notice("UNIFIED_FEATURES", len(cols), "/", len(ALL_FEATURES),
           "missing", [x for x in ALL_FEATURES if x not in common][:18])
    return cols


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    have = set(columns)
    sets = {"all_available": columns[:]}
    if set(RAW_126).issubset(have):
        sets["raw126"] = list(RAW_126)
    if set(LEAN8_24).issubset(have):
        sets["lean8_24"] = list(LEAN8_24)
        instant = [c for c in LEAN8_24 if any(f".{k}[" in c for k in ("lw", "Pm", "r01", "r32"))]
        if set(RAW_126).issubset(have):
            sets["raw126_lean8_instant12"] = list(RAW_126) + instant
    if set(LITE_15).issubset(have):
        sets["lean_lite_15"] = list(LITE_15)
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


def sample_tap(t, y, active, ev, soft=None):
    """Return row indices and sample kind for the TAP training set (soft: rows that are soft positives)."""
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
    if soft is not None:
        for i in pos:
            if soft[i]:
                kind[int(i)] = 6
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
_CONTENT_SHA = {}


def content_sha(path: Path) -> str:
    """SHA-256 of a file's content. For gzip files the header's modification time and stored file name are ignored,
    so a label file regenerated with the same content keeps its identity and a renamed copy of a recording is
    recognised, while any content change (also same-size replacements) changes it. Cached per (path, size,
    mtime_ns) within the process."""
    st = Path(path).stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _CONTENT_SHA:
        b = Path(path).read_bytes()
        if b[:3] == b"\x1f\x8b\x08" and len(b) >= 10:
            flg, pos = b[3], 10
            if flg & 0x04:                                   # FEXTRA
                pos += 2 + (b[10] | (b[11] << 8))
            if flg & 0x08:                                   # FNAME: zero-terminated original name
                end = b.index(0, pos)
                b = b[:pos] + b[end + 1:]
            b = b[:3] + bytes([flg & ~0x08]) + b"\0\0\0\0" + b[8:]
        _CONTENT_SHA[key] = hashlib.sha256(b).hexdigest()
    return _CONTENT_SHA[key]


def record_identity(records) -> dict:
    """Content identity of the training/replay inputs: every recording read (path, content) and its label sidecar,
    plus the event tables (_events.csv.gz) of the label roots. Paths are relative to the repository."""
    files = [[p.relative_to(ROOT).as_posix(), content_sha(p), content_sha(sidecar(p, root, labels))]
             for p, root, labels in records]
    events = {Path(labels).relative_to(ROOT).as_posix(): content_sha(Path(labels) / "_events.csv.gz")
              for labels in sorted({str(r[2]) for r in records})}
    h = hashlib.sha256(json.dumps(dict(files=files, events=events), sort_keys=True).encode()).hexdigest()
    return dict(sha256=h, recordings=len(files), events=events, files=files)


def identity_changes(old: dict | None, new: dict) -> list[str]:
    """Human-readable differences between two record_identity() results (for refusing a stale reuse)."""
    if not old:
        return ["no data identity stored (output from older code)"]
    if old.get("sha256") == new["sha256"]:
        return []
    out = []
    a = {f[0]: f[1:] for f in old.get("files", [])}; b = {f[0]: f[1:] for f in new["files"]}
    if set(b) - set(a):
        out.append(f"{len(set(b) - set(a))} new recordings, e.g. {sorted(set(b) - set(a))[:3]}")
    if set(a) - set(b):
        out.append(f"{len(set(a) - set(b))} recordings gone, e.g. {sorted(set(a) - set(b))[:3]}")
    src = sorted(k for k in set(a) & set(b) if a[k][0] != b[k][0])
    lab = sorted(k for k in set(a) & set(b) if a[k][1] != b[k][1])
    if src:
        out.append(f"{len(src)} recordings changed content, e.g. {src[:3]}")
    if lab:
        out.append(f"{len(lab)} label files changed, e.g. {lab[:3]}")
    for k in sorted(set(old.get("events", {})) | set(new["events"])):
        if old.get("events", {}).get(k) != new["events"].get(k):
            out.append(f"event table changed: {k}")
    return out or ["data identity differs"]


def dedupe_records(records):
    """Byte-identical copies of a recording are one recording (first path kept, by sorted path): duplicates would
    weigh twice and could otherwise end up on both sides of a split. Logged with their paths."""
    seen, keep, dup = {}, [], []
    for r in sorted(records, key=lambda a: a[0].relative_to(ROOT).as_posix()):
        k = content_sha(r[0])
        if k in seen:
            dup.append([r[0].relative_to(ROOT).as_posix(), seen[k]])
            continue
        seen[k] = r[0].relative_to(ROOT).as_posix(); keep.append(r)
    if dup:
        notice("UNIFIED_DUPLICATE_RECORDINGS", len(dup), "byte-identical copies left out, e.g.", json.dumps(dup[:3]))
    order = {id(r): i for i, r in enumerate(records)}
    return sorted(keep, key=lambda r: order[id(r)]), dup


def _fingerprint(task, records, columns):
    h = hashlib.sha256()
    h.update(json.dumps(dict(task=task, columns=columns, version=5, lean8=LEAN8_TAG,
        soft=TAP_SOFT_MIN_SCORE if task == "tap" else None, onset_win=TAP_ONSET_WINDOW_MS, press_win=PRESS_WINDOW_MS,
        tap=[TAP_BACKGROUND_STRIDE_MS, TAP_HARD_WINDOW_MS, TAP_HARD_STRIDE_MS, TAP_ACTIVE_STRIDE_MS,
             TAP_POSITIVE_STRIDE_MS],
        lp=[LP_HOLD_STRIDE_MS, LP_EARLY_HOLD_MS, LP_RELEASE_WINDOW_MS, LP_TAP_RELEASE_WINDOW_MS,
            LP_TAP_RELEASE_STRIDE_MS, LP_HARD_STRIDE_MS, LP_BACKGROUND_STRIDE_MS],
        replay=[REPLAY_LP_TAPS_PER_RECORDING, REPLAY_LP_AFTER_RELEASE_MS]), sort_keys=True).encode())
    h.update(sha256(CLAUDE_LABELER).encode())
    if True:                                     # frame counter (and Lean-8 inputs when used) come from here
        h.update(lean8.GENERATOR_SHA.encode())
        fw = lean8.firmware_header()
        h.update((sha256(fw) if fw else "no-firmware-header").encode())
    if USE_LITE:                                 # Lean-lite inputs (and their frame counter) come from here
        h.update(lite.GENERATOR_SHA.encode())
    h.update(("identity=" + record_identity(records)["sha256"]).encode())   # contents, not sizes/mtimes
    return h.hexdigest()


def press2_recording(source: Path) -> bool:
    """TAP recordings whose every real pointer press is a tap (MUDRA_TAP_LABELS=press2): double taps and 240 bpm."""
    return source.parent.name == "TAP" and any(k in source.name for k in PRESS2_NAME_KEYS)


def press2_labels(t, y, y_train, ev):
    """'Trust the press' training labels of one press2 recording (TAP_FOCUS_RESULTS.md). y: the labeller's tap_label,
    y_train: the current training target. Every real pointer press (tap_press, tier PRESS, labeled 1 or -1):
      * confirmed (labeled 1): unchanged;
      * unknown with a tap teacher peak (score >= PRESS2_SOFT_SCORE): its UNKNOWN rows in the labeller's 0-80 ms
        onset window become positives;
      * other unknown presses without a labelled onset within PRESS2_MATCH_MS: rows t_on - 5 .. t_on + 90 become
        positives (t_on = the labeller's onset estimate, else the pointer time);
      * negatives t_on - 30 .. t_on + 120 of any real press are left out (-1).
    Returns (target, added-positive mask, counts). Replay references and every score stay on the original labels."""
    y2 = np.array(y_train, np.int8, copy=True); added = np.zeros(len(y2), bool); info = Counter()
    if ev is None or not len(ev):
        return y2, added, info
    pr = ev[(ev.type == "tap_press") & (ev.tier == "PRESS") & (ev.labeled != 0)]
    if not len(pr):
        return y2, added, info
    t_on = pd.to_numeric(pr.t_on, errors="coerce").to_numpy(float)
    t_ptr = pd.to_numeric(pr.t_ptr, errors="coerce").to_numpy(float)
    a = np.where(np.isfinite(t_on), t_on, t_ptr)
    lab = pr.labeled.to_numpy(int)
    score = pd.to_numeric(pr.score, errors="coerce").to_numpy(float) if "score" in pr else np.full(len(pr), np.nan)
    disc = ((pr.rule_disconnect.astype(str).str.lower() == "true").to_numpy() if "rule_disconnect" in pr
            else np.zeros(len(pr), bool))
    ok = np.isfinite(a); a, lab, score, disc = a[ok], lab[ok], score[ok], disc[ok]
    soft = (lab == -1) & (score >= PRESS2_SOFT_SCORE) & ~disc
    for o in a[soft]:
        m = (t >= o + TAP_ONSET_WINDOW_MS[0]) & (t <= o + TAP_ONSET_WINDOW_MS[1]) & (y == -1)
        added |= m & (y2 != 1); y2[m] = 1
    labelled = np.r_[a[lab == 1], a[soft]]
    for o in a[(lab == -1) & ~soft]:
        if len(labelled) and np.min(np.abs(labelled - o)) < PRESS2_MATCH_MS:
            continue
        m = (t >= o + PRESS2_ANCHOR_MS[0]) & (t <= o + PRESS2_ANCHOR_MS[1])
        added |= m & (y2 != 1); y2[m] = 1; info["press2_anchored_presses"] += 1
    near_press = np.zeros(len(y2), bool)
    for o in a:
        near_press |= (t >= o + PRESS2_NEGATIVE_GAP_MS[0]) & (t <= o + PRESS2_NEGATIVE_GAP_MS[1])
    drop = near_press & (y2 == 0)
    y2[drop] = -1
    info["press2_recordings"] += 1; info["press2_presses"] += int(len(a))
    info["press2_soft_presses"] += int(soft.sum()); info["press2_positive_rows_added"] += int(added.sum())
    info["press2_negative_rows_left_out"] += int(drop.sum())
    return y2, added, info


def build_task_data(task: str, records, out: Path):
    """Training sample in memory, full replay rows in an on-disk float32 matrix (cached)."""
    records, _ = dedupe_records(records)          # idempotent (train_task already passes deduplicated records)
    raw_columns = common_p_core(records)
    derived = list(TF.DERIVED) if (task == "tap" and TAP_FEATURES == "diff") else []
    if derived:
        args = {a for pair in TF.DIFF_PAIRS + TF.RATIO_PAIRS for a in pair}
        if not args <= set(raw_columns):
            raise ValueError(f"Derived TAP inputs need raw fields that are missing: {sorted(args - set(raw_columns))[:5]}")
        export_plus.validate_feature_schema_accessors(derived)
    lite_cols = list(TF.LITE_COLUMNS) if (task == "tap" and TAP_LITE == "diff") else []
    if lite_cols:
        if not set(TF.WL_COLUMNS + TF.PLACEMENT_COLUMNS) <= set(raw_columns):
            raise ValueError("Lean-lite diff inputs need the raw WL and EMG PSD fields")
        export_plus.validate_feature_schema_accessors(lite_cols)
    press2 = task == "tap" and TAP_LABELS == "press2"
    columns = raw_columns + derived + lite_cols  # derived TAP inputs are stored like any other column
    cache = out / "cache"
    fp = _fingerprint(task, records, columns)
    if derived:                                  # the derived values depend on tap_features.py
        fp = hashlib.sha256((fp + "|tap_features=" + TF.VERSION).encode()).hexdigest()
    if lite_cols:                                # ... and the Lean-lite diff emulator
        fp = hashlib.sha256((fp + "|tap_lite=" + TF.LITE_VERSION).encode()).hexdigest()
    if press2:                                   # the training-label policy
        import inspect
        policy = json.dumps(dict(keys=PRESS2_NAME_KEYS, soft=PRESS2_SOFT_SCORE, anchor=PRESS2_ANCHOR_MS,
                                 gap=PRESS2_NEGATIVE_GAP_MS, match=PRESS2_MATCH_MS,
                                 code=hashlib.sha256((inspect.getsource(press2_labels) +
                                                      inspect.getsource(press2_recording)).encode()).hexdigest()))
        fp = hashlib.sha256((fp + "|tap_labels=press2|" + policy).encode()).hexdigest()
    stamp = cache / "fingerprint.json"
    if stamp.is_file() and json.loads(stamp.read_text()).get("fingerprint") == fp:
        notice("UNIFIED_CACHE_REUSED", task, cache)
        return load_cache(cache, columns)
    previous_sha = None
    if cache.exists():
        # A study may have run on this cache: remember its content so the study guard can verify the rebuild.
        try:
            if stamp.is_file():
                old = json.loads((cache / "info.json").read_text())
                previous_sha = old.get("content_sha256") or cache_content_sha(cache)
        except Exception as exc:                 # incomplete cache: nothing to compare against
            notice("UNIFIED_PREVIOUS_CACHE_UNREADABLE", repr(exc)[:200])
        shutil.rmtree(cache)                     # derived data only
    cache.mkdir(parents=True)
    events = {"tap": load_events(LABEL_TAP) if task == "tap" else None, "lp": load_events(LABEL_LP)}
    ncol = len(columns)
    raw = open(cache / "replay_X.f32", "wb"); nrep = 0
    T = dict(X=[], y=[], kind=[], rec=[], t=[], frames=[], w=[], row=[])
    recs = []          # per recording metadata
    rp_t, rp_lab, rp_frames = [], [], []
    windows = []       # LP press windows
    onsets = []        # TAP confirmed onsets: (rec index, t_on)
    info = Counter()
    for i, (source, root, labels_root) in enumerate(records, 1):
        label_file = sidecar(source, root, labels_root)
        folder = source.parent.name
        rel = source.relative_to(root).as_posix()
        ev_all = (events["tap"] if root in (SOURCE_TAP, RAW_SOURCE_TAP) else events["lp"]) if task == "tap" else events["lp"]
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
        y_train, wrow, soft_rows, soft_ev = y, None, None, None
        if (task == "tap" and TAP_SOFT_MIN_SCORE is not None and folder in TAP_FOLDERS
                and ev is not None and len(ev)):
            soft_ev = ev[(ev.type == "tap_press") & (ev.tier == "PRESS") & (ev.labeled == -1)
                         & (pd.to_numeric(ev.score, errors="coerce") >= TAP_SOFT_MIN_SCORE)]
            if "rule_disconnect" in soft_ev:
                soft_ev = soft_ev[soft_ev.rule_disconnect.astype(str).str.lower() != "true"]
            y_train = y.copy(); wrow = np.ones(len(y), np.float32); soft_rows = np.zeros(len(y), bool)
            for x in soft_ev.itertuples():
                m = (t >= x.t_on + TAP_ONSET_WINDOW_MS[0]) & (t <= x.t_on + TAP_ONSET_WINDOW_MS[1]) & (y == -1)
                y_train[m] = 1; wrow[m] = float(x.score); soft_rows[m] = True
            info["soft_label_presses"] += len(soft_ev)
        if press2 and press2_recording(source):
            y_train, added, p2 = press2_labels(t, y, y_train, ev)
            soft_rows = added if soft_rows is None else (soft_rows | added)
            info.update(p2)
        if task == "tap":
            rows, kinds = sample_tap(t, y_train, lab["any_active"].to_numpy(np.int8), ev, soft_rows)
        else:
            rows, kinds = sample_lp(t, y, lab["hold_ms"].to_numpy(np.int32), folder, ev)
        source_cols = set(raw_columns) | {"Timestamp", "User", lean8.FRAMES_COLUMN}
        df = pd.read_csv(source, compression="gzip", low_memory=False,
                         usecols=lambda c: c in source_cols)
        if len(df) != len(lab):
            raise ValueError(f"Row count mismatch: {source}")
        frames = df[lean8.FRAMES_COLUMN].to_numpy(np.int32)
        if not np.array_equal(pd.to_numeric(df["Timestamp"], errors="raise").to_numpy(np.int64), ts):
            raise ValueError(f"Timestamp mismatch: {source}")
        X = df[raw_columns].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
        if not np.isfinite(X).all():
            raise ValueError(f"Nonfinite firmware input in {source}")
        if derived:
            X = np.hstack([X, TF.derived(X, raw_columns)])
        if lite_cols:                            # from the recording's first row, as the firmware state after a reset
            X = np.hstack([X, TF.lite_rows(X, raw_columns + derived, frames)])
        user = str(df["User"].iloc[0]) if "User" in df.columns else ""
        who = v6.person_group(user, source.name[:-7])
        ri = len(recs)
        # ---- replay rows
        if task == "tap":
            start = nrep; raw.write(X.tobytes()); nrep += len(X)
            rp_t.append(t.astype(np.float32)); rp_lab.append(y); rp_frames.append(frames)
            if ev is not None and len(ev):
                lo_on, hi_on = R.TAP_HIT_BEFORE_MS, R.TAP_HIT_AFTER_MS
                for o in ev[(ev.type == "tap_press") & (ev.labeled == 1)].t_on.dropna():
                    onsets.append((ri, float(o), "strong", float(o) - lo_on, float(o) + hi_on))
                if soft_ev is not None:
                    for o in soft_ev.t_on.dropna():
                        onsets.append((ri, float(o), "soft", float(o) - lo_on, float(o) + hi_on))
                if folder in TAP_FOLDERS:          # every real pointer press of a TAP/SWIPE recording (folder truth)
                    pr = ev[(ev.type == "tap_press") & (ev.tier == "PRESS") & (ev.labeled != 0)]
                    for x in pr.itertuples():
                        a_ = np.nanmin([x.t_ptr, x.t_on]); b_ = np.nanmax([x.t_ptr, x.t_on])
                        onsets.append((ri, float(x.t_on), "press", float(a_) - PRESS_WINDOW_MS[0],
                                       float(b_) + PRESS_WINDOW_MS[1]))
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
                raw.write(X[a:b + 1].tobytes()); rp_t.append(t[a:b + 1].astype(np.float32)); rp_frames.append(frames[a:b + 1])
                windows.append(dict(rec=ri, kind=kind_, start=nrep, end=nrep + (b + 1 - a), start_ms=s, release_ms=r))
                nrep += b + 1 - a
        # ---- training sample
        if len(rows):
            T["X"].append(X[rows]); T["y"].append(y_train[rows]); T["kind"].append(kinds)
            T["w"].append(wrow[rows] if wrow is not None else np.ones(len(rows), np.float32))
            T["rec"].append(np.full(len(rows), ri, np.int32)); T["t"].append(t[rows].astype(np.float32))
            T["frames"].append(frames[rows]); T["row"].append(rows.astype(np.int32))
        info["read_sources"] += 1; info[folder + "_files"] += 1
        if i % 300 == 0:
            notice("UNIFIED_DATASET_EXTRACT", task, i, len(records), sum(len(a) for a in T["y"]), "replay_rows", nrep)
    raw.close()
    if not T["X"]:
        raise ValueError(f"No supervised {task} recordings")
    rec_df = pd.DataFrame(recs)
    np.save(cache / "train_X.npy", np.concatenate(T["X"]).astype(np.float32))
    meta = dict(y=np.concatenate(T["y"]), kind=np.concatenate(T["kind"]),
                rec=np.concatenate(T["rec"]), t=np.concatenate(T["t"]), frames=np.concatenate(T["frames"]),
                w=np.concatenate(T["w"]).astype(np.float32))
    if lite_cols:                                # row of the recording: the Lean-lite augmentation recomputes per recording
        meta["row"] = np.concatenate(T["row"])
    np.savez(cache / "train_meta.npz", **meta)
    np.savez(cache / "replay_meta.npz", t=np.concatenate(rp_t) if rp_t else np.zeros(0, np.float32),
             label=np.concatenate(rp_lab) if rp_lab else np.zeros(0, np.int8),
             frames=np.concatenate(rp_frames) if rp_frames else np.zeros(0, np.int32), n=nrep, ncol=ncol)
    rec_df.to_csv(cache / "recordings.csv", index=False)
    pd.DataFrame(windows, columns=["rec", "kind", "start", "end", "start_ms", "release_ms"]).to_csv(
        cache / "lp_windows.csv", index=False)
    pd.DataFrame(onsets, columns=["rec", "t_on", "kind", "lo", "hi"]).to_csv(cache / "tap_onsets.csv", index=False)
    write_json(cache / "info.json", dict(counts=dict(info), previous_content_sha256=previous_sha))
    write_json(stamp, dict(fingerprint=fp))
    return load_cache(cache, columns)


def _objective_onsets(onsets):
    """Onsets whose recall the objective counts: confirmed taps (+ rescued soft taps while soft labels are on)."""
    if not len(onsets):
        return {}
    if "kind" not in onsets:
        onsets = onsets.assign(kind="strong")
    kinds = ["strong"] + (["soft"] if TAP_SOFT_MIN_SCORE is not None else [])
    o = onsets[onsets.kind.isin(kinds)]
    return {int(k): np.sort(g.t_on.to_numpy(float)) for k, g in o.groupby("rec")}


def _reference_windows(onsets):
    """Reported-only recall references: name -> rec -> (lo, hi) arrays (strong, soft, press)."""
    if not len(onsets) or "kind" not in onsets:
        return {}
    out = {}
    for name, g in onsets.groupby("kind"):
        out[str(name)] = {int(r): (gg.lo.to_numpy(float), gg.hi.to_numpy(float))
                          for r, gg in g.sort_values("lo").groupby("rec")}
    return out


def cache_content_sha(cache: Path) -> str:
    """SHA-256 of the cache CONTENT that training and the objective see: array bytes (not .npz zip timestamps),
    replay matrix, plan tables. Report-only additions are left out so that a study keeps resuming when only
    they change: unit sample weights (soft labels off) and the reference windows of tap_onsets.csv. With soft
    labels off the hash equals the one of the cache format before those additions."""
    h = hashlib.sha256()
    for npz in ("train_meta.npz", "replay_meta.npz"):
        with np.load(cache / npz) as z:
            for k in sorted(z.files):
                a = np.ascontiguousarray(z[k])
                if npz == "train_meta.npz" and k == "w" and bool((a == 1).all()):
                    continue
                h.update(f"{npz}:{k}:{a.dtype}:{a.shape}".encode()); h.update(a.tobytes())
    for name in ("train_X.npy", "replay_X.f32", "recordings.csv", "lp_windows.csv", "tap_onsets.csv"):
        h.update(name.encode())
        if name == "tap_onsets.csv":
            h.update(_objective_onsets_bytes(cache / name))
            continue
        with open(cache / name, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()


def _objective_onsets_bytes(path: Path) -> bytes:
    """tap_onsets.csv as the objective sees it: confirmed onsets in the earlier two-column format, then the soft
    onsets (if soft labels were on). The pointer-press reference rows are report-only."""
    o = pd.read_csv(path)
    if "kind" not in o:
        return path.read_bytes()
    text = o[o.kind == "strong"][["rec", "t_on"]].to_csv(index=False).encode()
    soft = o[o.kind == "soft"]
    return text + (soft.to_csv(index=False).encode() if len(soft) else b"")


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
    gate = m["frames"] >= TRAIN_MIN_FRAMES
    gate_info = dict(min_frames=TRAIN_MIN_FRAMES, eval_min_frames=EVAL_MIN_FRAMES,
                     steady_frames=LEAN8_STEADY_FRAMES, lean8=LEAN8_TAG,
                     train_rows_built=int(len(gate)), train_rows_used=int(gate.sum()),
                     positives_built=int((m["y"] == 1).sum()), positives_used=int(((m["y"] == 1) & gate).sum()))
    if not gate.all():
        X = X[gate]
    rec = m["rec"][gate]
    data = dict(X=X, y=m["y"][gate].astype(np.int8), kind=m["kind"][gate], t=m["t"][gate], frames=m["frames"][gate],
                group=recs.group.to_numpy(object)[rec], file=recs.file.to_numpy(object)[rec],
                category=recs.folder.to_numpy(object)[rec], rec=rec,
                w=(m["w"][gate].astype(np.float32) if "w" in m.files else np.ones(len(rec), np.float32)),
                columns=list(columns), feature_sets=feature_sets(list(columns)))
    if "row" in m.files:
        data["row"] = m["row"][gate]
    fr = rp["frames"]
    rpl = dict(X=replay_X, t=rp["t"], label=rp["label"], frames=fr,
               ready=fr >= EVAL_MIN_FRAMES, steady=fr >= LEAN8_STEADY_FRAMES, recs=recs, windows=windows,
               onsets=_objective_onsets(onsets), extra=_reference_windows(onsets),
               anchors={int(r): np.unique(np.round(g_.t_on.to_numpy(float))) for r, g_ in
                        onsets[onsets.kind.isin(["strong", "soft", "press"])].groupby("rec")} if len(onsets) else {})
    info_all = json.loads((cache / "info.json").read_text())
    if "content_sha256" not in info_all:            # first use of this cache (or a cache from older code)
        t0 = time.perf_counter()
        info_all["content_sha256"] = cache_content_sha(cache)
        write_json(cache / "info.json", info_all)
        notice("UNIFIED_CACHE_CONTENT_SHA", info_all["content_sha256"][:16], round(time.perf_counter() - t0, 1), "s")
    info = info_all["counts"]
    data["content_sha256"] = info_all["content_sha256"]
    data["previous_content_sha256"] = info_all.get("previous_content_sha256")
    if data["previous_content_sha256"]:
        notice("UNIFIED_CACHE_REBUILD", "identical content" if data["previous_content_sha256"] == data["content_sha256"]
               else "CONTENT CHANGED", data["previous_content_sha256"][:16], "->", data["content_sha256"][:16])
    gate_info.update(usable_coverage(rpl))
    info["lean8_gate"] = gate_info
    notice("UNIFIED_LEAN8_GATE", json.dumps(gate_info))
    notice("UNIFIED_DATA", json.dumps(dict(train_rows=int(len(data["y"])), positives=int((data["y"] == 1).sum()),
           negatives=int((data["y"] == 0).sum()), replay_rows=n, recordings=len(recs),
           train_GB=round(X.nbytes / 1e9, 2), replay_GB=round(n * ncol * 4 / 1e9, 2),
           kinds={SAMPLE_KINDS[int(k)]: int(v) for k, v in zip(*np.unique(data["kind"], return_counts=True))})))
    return data, rpl, info


# ============================================================================ split, weights, params
def group_by_file(data, rpl):
    """MUDRA_SPLIT=file: every recording file becomes its own group (person|file) in the training rows and in the
    replay table, so the held-out split, the CV folds, the feature-ranking folds, the observable-label teachers and
    every replay selection group by file. In memory only (the cache keeps the person groups)."""
    recs = rpl["recs"]
    recs["person"] = recs.group.to_numpy(object)
    recs["group"] = (recs.group.astype(str) + "|" + recs.file.astype(str)).to_numpy(object)
    data["person"] = data["group"]
    data["group"] = recs.group.to_numpy(object)[data["rec"]]
    if "folder" in recs:                         # gesture-aware: the stratum of every recording and training row
        strata = np.array([recording_stratum(f, fo) for f, fo in zip(recs.file, recs.folder)], object)
        recs["stratum"] = strata
        data["stratum"] = strata[data["rec"]]
    data["content_sha256"] = hashlib.sha256((data["content_sha256"] + "|split=file-gesture1").encode()).hexdigest()


STRATUM_MIN_FILES = 5
GESTURE_ALIASES = {"stretch": "fiststretch", "fist": "fiststretch"}   # Fist_&_Stretch and Fist&Stretch
_GESTURE_KEYS = []


def recording_gesture(name: str) -> str:
    """Gesture of a recording from its file name, found with org.py's routing keywords (longest first, as org.py
    routes files): the underscore-separated part that holds the keyword, letters and digits only.
    SwipeMID -> swipemid, Ring_Finger -> ringfinger, TapOnArm -> taponarm, Fist_&_Stretch -> stretch."""
    if not _GESTURE_KEYS:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import org
        _GESTURE_KEYS.extend(sorted(org.GESTURE_MAP, key=lambda k: (-len(k), k)))
    s = Path(str(name)).name.lower()
    for k in _GESTURE_KEYS:
        i = s.find(k)
        if i >= 0:
            a = s.rfind("_", 0, i) + 1
            b = s.find("_", i + len(k))
            g_ = re.sub(r"[^a-z0-9]", "", s[a:len(s) if b < 0 else b])
            return GESTURE_ALIASES.get(g_, g_)
    squashed = s.replace("_", "")                # Tap_on_Hand, Soft_Tap: as org.find_mapping, only if nothing matched
    for k in _GESTURE_KEYS:
        k2 = k.replace("_", "")
        if k2 in squashed:
            return GESTURE_ALIASES.get(k2, k2)
    return "unknown"


def recording_stratum(name: str, folder: str) -> str:
    """folder|gesture, plus |<bpm> for TAP and SWIPE recordings (60 / 120 / 240 bpm taps are different tasks)."""
    st = f"{folder}|{recording_gesture(name)}"
    m = re.search(r"_(\d+)bpm", Path(str(name)).name.lower())
    return st + (f"|{m.group(1)}" if m and folder in TAP_FOLDERS else "")


def merge_rare(labels, min_count: int):
    """Pool strata with fewer than min_count members one level up (folder|gesture|bpm -> folder|gesture -> folder
    -> other); a stratum still below 2 members joins the largest one."""
    lab = np.asarray(labels, object).copy()
    for level in (2, 1, 0):
        counts = Counter(lab.tolist())
        rare = np.array([counts[x] < min_count for x in lab], bool)
        if not rare.any():
            break
        lab[rare] = ["|".join(str(x).split("|")[:level]) or "other" for x in lab[rare]]
    counts = Counter(lab.tolist())
    if counts and min(counts.values()) < 2:
        big = counts.most_common(1)[0][0]
        lab = np.array([big if counts[x] < 2 else x for x in lab], object)
    return lab


def file_strata(data, rows):
    """For training rows `rows`: (group code per row, groups, merged stratum per group)."""
    codes, groups = pd.factorize(pd.Series(data["group"][rows], dtype=object))
    _, first = np.unique(codes, return_index=True)
    return codes, np.asarray(groups, object), merge_rare(data["stratum"][rows][first], STRATUM_MIN_FILES)


def cv_splits(data, train_ids):
    """CV folds over train_ids (positions into train_ids). Gesture-aware file split: StratifiedKFold over the files
    by stratum; otherwise GroupKFold by group (person or file)."""
    if "stratum" not in data:
        return list(GroupKFold(n_splits=CV_FOLDS).split(train_ids, data["y"][train_ids], data["group"][train_ids]))
    codes, groups, strata = file_strata(data, train_ids)
    out = []
    for _, b_g in StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(groups, strata):
        val = np.zeros(len(groups), bool); val[b_g] = True
        out.append((np.flatnonzero(~val[codes]), np.flatnonzero(val[codes])))
    return out


def strata_balance(data, tr, te):
    """Files per stratum on each side of the held-out split (stratum -> [train files, held-out files])."""
    out = {}
    for side, rows in ((0, tr), (1, te)):
        g_, first = np.unique(data["group"][rows], return_index=True)
        for st in data["stratum"][rows][first]:
            out.setdefault(str(st), [0, 0])[side] += 1
    return dict(sorted(out.items()))


def _stratified_heldout(data, task):
    """Gesture-aware file split: TEST_FRACTION of the files of every stratum held out (StratifiedShuffleSplit)."""
    codes, groups, strata = file_strata(data, np.arange(len(data["y"])))
    for k in range(200):
        _, b_g = next(StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                                             random_state=SEED + int(task == "lp") + k).split(groups, strata))
        held = np.zeros(len(groups), bool); held[b_g] = True
        yield np.flatnonzero(~held[codes]), np.flatnonzero(held[codes])


def split_heldout(data, task):
    g = data["group"]; y = data["y"]; c = data["category"]
    splitter = (_stratified_heldout(data, task) if "stratum" in data else
                GroupShuffleSplit(n_splits=200, test_size=TEST_FRACTION,
                                  random_state=SEED + int(task == "lp")).split(data["X"], y, g))
    for tr, te in splitter:
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
    raise RuntimeError(f"No usable {SPLIT_BY}-grouped {task} split, inspect category coverage")


def weights_for_task(data, indices, task, positive_weight, finger=True):
    """File-normalized class balancing, then the tunable positive-class weight (see v1 notes). TAP: FINGER-folder
    negatives first weighted x MUDRA_TAP_FINGER_WEIGHT (finger=False: plain, for the teachers)."""
    y = data["y"][indices]
    w = data["w"][indices].astype(np.float32).copy()
    if task == "tap" and finger and TAP_FINGER_WEIGHT != 1.0:
        w[(y == 0) & (data["category"][indices] == "FINGER")] *= np.float32(TAP_FINGER_WEIGHT)
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


def soft_rows(X, target, w):
    """Soft targets as plain binary rows: each frame becomes (label 1, weight w*t) and (label 0, weight w*(1-t)).
    Their weighted binary log loss is exactly w*[t*log p + (1-t)*log(1-p)], the cross-entropy with target t."""
    t = np.clip(np.asarray(target, np.float64), 0.0, 1.0); w = np.asarray(w, np.float64)
    one, zero = t > 0, t < 1
    X2 = np.concatenate([X[one], X[zero]])
    y2 = np.r_[np.ones(int(one.sum()), np.int8), np.zeros(int(zero.sum()), np.int8)]
    return X2, y2, np.r_[w[one] * t[one], w[zero] * (1.0 - t[zero])].astype(np.float32)


def fit_model(X, y, w, params, budget, target=None):
    """target (distillation): soft targets per row; fitted as two weighted binary rows per frame, with
    min_child_samples doubled (it counts rows)."""
    p = dict(params); p["n_estimators"] = fit_cap(params, budget)
    if target is not None:
        X, y, w = soft_rows(X, target, w)
        p["min_child_samples"] = 2 * int(p.get("min_child_samples", 20))
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


def trees_within_reference(model, budget: int) -> tuple[int, int]:
    """Original exact search: every probe truncates the model and runs the exporter's size accounting."""
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


class _DumpPrefix:
    """Booster stand-in whose dump_model() is the first k trees of ONE dump of the fitted booster.
    export_plus's LightGBM size accounting reads only dump_model()["tree_info"]."""

    def __init__(self, tree_info, k):
        self._dump = {"tree_info": tree_info[:k]}

    def dump_model(self, *args, **kwargs):
        return self._dump


_SIZING = weakref.WeakKeyDictionary()       # per fitted model, never stored on the model (pickling, copies)
_SIZING_LOCK = threading.Lock()


def _sizing(model):
    """Per-model cache: full exact size and a prefix sizer that runs the unchanged exporter accounting."""
    with _SIZING_LOCK:
        cache = _SIZING.get(model)
    if cache is None:
        info = model.booster_.dump_model()["tree_info"]
        memo = {}

        def prefix(k):
            if k not in memo:
                m = copy.copy(model)
                m._Booster = _DumpPrefix(info, k)
                memo[k] = int(export_plus._estimate_compact_tree_payload_legacy(m)["total_static_data_bytes"])
            return memo[k]
        cache = dict(full=static_bytes(model), prefix=prefix, n=model.booster_.num_trees())
        with _SIZING_LOCK:
            _SIZING[model] = cache
    return cache


SIZING_FALLBACKS = Counter()


def trees_within(model, budget: int) -> tuple[int, int]:
    """Largest number of leading trees whose exported C model fits the budget, and its exact bytes.

    Same answer as trees_within_reference(): the binary search runs on the exporter's own accounting of a
    single dump, and the chosen k (fits) and k + 1 (does not fit) are re-checked with the original
    truncate-and-measure path. Any disagreement falls back to the original search."""
    sz = _sizing(model)
    n, full = sz["n"], sz["full"]
    if full <= budget:
        return n, full
    size = sz["prefix"]
    lo, hi = 1, n - 1
    if size(lo) > budget:
        return trees_within_reference(model, budget)       # raises the original error
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if size(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    exact_k = static_bytes(truncated(model, lo))
    exact_next = full if lo + 1 == n else static_bytes(truncated(model, lo + 1))
    if exact_k != size(lo) or exact_k > budget or exact_next <= budget:
        SIZING_FALLBACKS["prefix_disagreed"] += 1
        notice("UNIFIED_SIZING_FALLBACK", lo, exact_k, size(lo), exact_next, budget)
        return trees_within_reference(model, budget)
    return lo, exact_k


# ============================================================================ replay scoring
_GATHER_LOCAL = threading.local()          # one reusable gather buffer per thread (concurrent CV folds)


def _gather(Xmm, batch, selected):
    """Rows of the batch ranges, selected columns only, C-contiguous float32 (same bytes as
    np.concatenate([Xmm[a:b] ...])[:, selected]); one buffer per column count, reused across calls."""
    if isinstance(selected, slice):            # one-off header replay (column views): original gather
        return np.ascontiguousarray(np.concatenate([np.asarray(Xmm[a:b]) for a, b in batch])[:, selected],
                                    dtype=np.float32)
    n = sum(b - a for a, b in batch)
    ncol = len(selected)
    store = getattr(_GATHER_LOCAL, "buffers", None)
    if store is None:
        store = _GATHER_LOCAL.buffers = {}
    buf = store.get(ncol)
    if buf is None or len(buf) < n:
        store.clear()                               # keep at most one buffer per thread (bounded memory)
        buf = np.empty((max(n, 1), ncol), np.float32)
        store[ncol] = buf
    X = buf[:n]
    pos = 0
    for a, b in batch:
        np.take(Xmm[a:b], selected, axis=1, out=X[pos:pos + b - a], mode="clip")
        pos += b - a
    return X


def _ranges_predict_reference(booster_fns, Xmm, ranges, selected, chunk=250_000):
    """Original implementation (kept for the equivalence tests)."""
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


def _ranges_predict(booster_fns, Xmm, ranges, selected, chunk=250_000):
    """Yield (range, outputs per fn) for contiguous replay ranges, batching small ranges."""
    batch, size = [], 0
    def flush(batch):
        with CLOCK("replay_gather"):
            X = _gather(Xmm, batch, selected)
        with CLOCK("replay_predict"):
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


def _first_usable(rpl, a, b, mask_name="ready"):
    m = rpl.get(mask_name)
    if m is None:
        return a
    local = np.flatnonzero(np.asarray(m[a:b], dtype=bool))
    return a + int(local[0]) if len(local) else None


def usable_coverage(rpl):
    """How much labelled material the frame gate leaves (all people)."""
    out = {}
    for name in ("ready", "steady"):
        recs = rpl["recs"]
        if (recs.start >= 0).any():                                  # TAP replay: whole recordings
            total = used = 0
            for i, r in recs.iterrows():
                ons = rpl["onsets"].get(int(i), [])
                total += len(ons)
                a = _first_usable(rpl, int(r.start), int(r.end), name)
                if a is not None:
                    used += int((np.asarray(ons) >= float(rpl["t"][a]) + R.TAP_HIT_BEFORE_MS).sum())
            out[f"tap_onsets_{name}"] = f"{used}/{total}"
        w = rpl["windows"]
        if len(w):
            m = np.asarray(rpl[name])[w.start.to_numpy(int)]
            for kind in ("hold", "tap"):
                k = (w.kind == kind).to_numpy()
                out[f"lp_{kind}_windows_{name}"] = f"{int((m & k).sum())}/{int(k.sum())}"
    return out


def prepare_tap_replay(rpl, groups):
    """Per-recording TAP replay plan for these people (trial-invariant): usable row span + countable onsets."""
    recs = rpl["recs"]; pick = recs[recs.group.isin(groups)]
    ready = rpl.get("ready")
    items = []
    for i, r in pick.iterrows():
        a, b = int(r.start), int(r.end)
        if ready is not None:
            local = np.asarray(ready[a:b], dtype=bool)
            if not local.any():
                continue
            a += int(np.flatnonzero(local)[0])
        if a >= b:
            continue
        # Onsets before the gate opens cannot be detected by any model: not counted as misses.
        ons = np.asarray(rpl["onsets"].get(int(i), []), np.float64)
        t_a = float(rpl["t"][a])
        ons = ons[ons >= t_a + R.TAP_HIT_BEFORE_MS]
        extra = {}
        for name, per_rec in (rpl.get("extra") or {}).items():
            lo, hi = per_rec.get(int(i), (np.zeros(0), np.zeros(0)))
            keep = lo >= t_a
            extra[name] = (lo[keep], hi[keep])
        items.append((a, b, r.folder, ons, extra))
    return items


def prepare_lp_replay(rpl, groups):
    """Press windows of these people whose start row is usable (trial-invariant)."""
    w = rpl["windows"]
    if not len(w):
        return []
    recs = rpl["recs"]
    keep = w[np.isin(recs.group.to_numpy(object)[w.rec.to_numpy(int)], list(groups))]
    ready = rpl.get("ready")
    if ready is not None and len(keep):
        keep = keep[np.asarray(ready[keep.start.to_numpy(int)], dtype=bool)]
    return [(int(r.start), int(r.end), r.start_ms, r.release_ms, r.kind, recs.folder.iloc[int(r.rec)])
            for r in keep.itertuples()]


def replay_tap_items(prob_fn, rpl, items, selected):
    """prob_fn(X) -> dict name -> probability (or decision) arrays. Returns name -> TapCounts."""
    counts = {}
    preds = _ranges_predict([prob_fn], rpl["X"], [(it[0], it[1]) for it in items], selected)
    for (a, b, folder, ons, extra), ((a_, b_), (outs,)) in zip(items, preds):
        with CLOCK("replay_score"):
            for name, p in outs.items():
                d = p if p.dtype == bool else (p >= FIXED_THRESHOLD)
                R.tap_replay_recording(d, rpl["t"][a:b], rpl["label"][a:b], ons,
                                       folder, counts.setdefault(name, R.TapCounts()), extra=extra)
    return counts


def replay_lp_items(q_fn, rpl, items, selected, threshold=None):
    """q_fn(X) -> dict name -> Q20 integer arrays. Returns name -> LpCounts (release at LP_RELEASE_THRESHOLD)."""
    threshold = LP_RELEASE_THRESHOLD if threshold is None else threshold
    counts = {}
    preds = _ranges_predict([q_fn], rpl["X"], [(a, b) for a, b, *_ in items], selected)
    for (a, b, start_ms, release_ms, kind, folder), (_, (outs,)) in zip(items, preds):
        with CLOCK("replay_score"):
            for name, q in outs.items():
                R.lp_replay_press(q, rpl["t"][a:b], start_ms, release_ms, kind, folder,
                                  counts.setdefault(name, R.LpCounts()), threshold=threshold)
    return counts


def replay_tap(prob_fn, rpl, groups, selected):
    """Replay only from firmware-ready rows; warm-up can never create a false TAP event."""
    return replay_tap_items(prob_fn, rpl, prepare_tap_replay(rpl, groups), selected)


def replay_lp(q_fn, rpl, groups, selected, threshold=None):
    """q_fn(X) -> dict name -> Q20 integer arrays. Returns name -> LpCounts."""
    return replay_lp_items(q_fn, rpl, prepare_lp_replay(rpl, groups), selected, threshold)


# ============================================================================ trial-invariant fold preparation
_PREPARED = {}


def obs_select(tap, offset, score, dur_ms, stride_ms=None):
    """Observable-pattern window per tap. tap: owner id of each positive row (-1 = none), offset: ms after its
    onset, score: teacher score. For every tap the window of dur_ms (starting at one of its rows, holding at least
    dur_ms / stride rows when the tap has that many) with the highest mean score is kept; the tap's other rows are
    not. Rows without an owner and taps with one row are kept. Returns a keep mask over the rows."""
    stride_ms = TAP_POSITIVE_STRIDE_MS if stride_ms is None else stride_ms
    tap = np.asarray(tap); offset = np.asarray(offset, np.float64); score = np.nan_to_num(np.asarray(score, np.float64))
    keep = np.ones(len(tap), bool)
    order = np.lexsort((offset, tap))
    ts = tap[order]
    bounds = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1], True])
    need_full = max(1, int(round(dur_ms / stride_ms)))
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if tap[idx[0]] < 0 or len(idx) < 2:
            continue
        ot, sc = offset[idx], score[idx]
        inside = (ot[None, :] >= ot[:, None]) & (ot[None, :] < ot[:, None] + dur_ms)      # [start, row]
        cnt = inside.sum(1)
        mean = (inside * sc[None, :]).sum(1) / np.maximum(cnt, 1)
        mean[cnt < min(need_full, int(cnt.max()))] = -1.0
        j = int(np.argmax(mean))
        keep[idx[~inside[j]]] = False
    return keep


def _obs_owner(data, rpl, rows):
    """(tap owner id, ms after its onset) of each row: the latest onset / press of the recording at most 5 ms after
    the row and at most 95 ms before it."""
    anchors = rpl.get("anchors") or {}
    rec = data["rec"][rows]; t = data["t"][rows].astype(np.float64)
    owner = np.full(len(rows), -1, np.int64); off = np.full(len(rows), np.nan)
    ids = {}
    for r in np.unique(rec):
        an = anchors.get(int(r))
        if an is None or not len(an):
            continue
        m = np.flatnonzero(rec == r)
        k = np.searchsorted(an, t[m] - OBS_ANCHOR_MS[0], side="right") - 1
        ok = (k >= 0) & (t[m] - an[np.clip(k, 0, None)] <= OBS_ANCHOR_MS[1])
        for i, kk in zip(m[ok], k[ok]):
            owner[i] = ids.setdefault((int(r), float(an[kk])), len(ids)); off[i] = t[i] - an[kk]
    return owner, off


def obs_rows(data, rpl, rows, name, cache_dir=None):
    """The training rows `rows` with observable-pattern positives (MUDRA_TAP_OBS_MS): negatives and unowned rows
    unchanged, every tap reduced to its most distinguishable window. Teacher scores are cross-fitted by person
    inside `rows` (3 folds of its people): no row is scored by a teacher that saw its person. Cached by content."""
    if TAP_OBS_MS is None:
        return rows
    rows = np.asarray(rows)
    key = hashlib.sha256((hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest() + "|" +
                          str(data.get("obs_content_sha256")) + "|" + json.dumps(OBS_TEACHER, sort_keys=True) +
                          f"|{TAP_OBS_MS}|{OBS_ANCHOR_MS}|{SEED}|v1").encode()).hexdigest()[:20]
    path = (cache_dir / f"obs_{name}_{key}.npz") if cache_dir is not None else None
    if path is not None and path.is_file():
        with np.load(path) as z:
            return z["rows"]
    t0 = time.perf_counter()
    X = data.get("X_nominal", data["X"])
    y = data["y"][rows]; person = data["group"][rows]
    pos = np.flatnonzero(y == 1)
    score = np.full(len(rows), np.nan)
    people = np.unique(person)
    w = weights_for_task(data, rows, "tap", 1.0, finger=False)
    for k, (a_, b_) in enumerate(GroupKFold(n_splits=min(3, len(people))).split(people, groups=people)):
        fit = np.isin(person, people[a_]); lab = np.isin(person, people[b_]) & (y == 1)
        m = lgb.LGBMClassifier(**OBS_TEACHER, n_jobs=CPU_THREADS, random_state=SEED + 11 + k).fit(
            X[rows[fit]], y[fit], sample_weight=w[fit])
        score[lab] = m.predict_proba(X[rows[lab]])[:, 1]
    owner, off = _obs_owner(data, rpl, rows[pos])
    keep = np.ones(len(rows), bool)
    keep[pos] = obs_select(owner, off, score[pos], TAP_OBS_MS)
    out = rows[keep]
    notice("UNIFIED_TAP_OBS", name, json.dumps(dict(rows=int(len(rows)), positives=int(len(pos)),
           taps=int(len(np.unique(owner[owner >= 0]))), positive_rows_left_out=int((~keep).sum()),
           seconds=round(time.perf_counter() - t0, 1))))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, rows=out)
    return out


OBS_CACHE_DIR = None                # set per task (TAP with MUDRA_TAP_OBS_MS)
DISTILL_CACHE_DIR = None            # set per task (TAP with MUDRA_TAP_DISTILL)


def distill_targets(data, rows, name, cache_dir=None):
    """Soft targets of the training rows `rows` (MUDRA_TAP_DISTILL = a): a*label + (1-a)*teacher. The teacher is a
    full-size model (OBS_TEACHER, no budget) on all inputs of the real (nominal) rows, cross-fitted by group inside
    `rows` (3 folds of its groups: files with MUDRA_SPLIT=file): no row is scored by a teacher trained on its own
    group. None when distillation is off. Cached by content."""
    if TAP_DISTILL is None:
        return None
    rows = np.asarray(rows)
    key = hashlib.sha256((hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest() + "|" +
                          str(data.get("distill_content_sha256")) + "|" + json.dumps(OBS_TEACHER, sort_keys=True) +
                          f"|{TAP_DISTILL}|{SEED}|v1").encode()).hexdigest()[:20]
    path = (cache_dir / f"distill_{name}_{key}.npz") if cache_dir is not None else None
    if path is not None and path.is_file():
        with np.load(path) as z:
            return z["target"]
    t0 = time.perf_counter()
    X = data.get("X_nominal", data["X"])
    y = data["y"][rows].astype(np.float64); grp = data["group"][rows]
    groups = np.unique(grp)
    w = weights_for_task(data, rows, "tap", 1.0, finger=False)
    score = np.full(len(rows), np.nan)
    for k, (a_, b_) in enumerate(GroupKFold(n_splits=min(3, len(groups))).split(groups, groups=groups)):
        fit = np.isin(grp, groups[a_]); sc = np.isin(grp, groups[b_])
        m = lgb.LGBMClassifier(**OBS_TEACHER, n_jobs=CPU_THREADS, random_state=SEED + 21 + k).fit(
            X[rows[fit]], y[fit].astype(np.int8), sample_weight=w[fit])
        score[sc] = m.predict_proba(X[rows[sc]])[:, 1]
    if not np.isfinite(score).all():
        raise AssertionError("distillation teacher left rows unscored")
    target = (TAP_DISTILL * y + (1.0 - TAP_DISTILL) * score).astype(np.float32)
    notice("UNIFIED_TAP_DISTILL", name, json.dumps(dict(rows=int(len(rows)), label_weight=TAP_DISTILL,
           teacher_mean_positive=round(float(score[y == 1].mean()), 4) if (y == 1).any() else None,
           teacher_mean_negative=round(float(score[y == 0].mean()), 4) if (y == 0).any() else None,
           seconds=round(time.perf_counter() - t0, 1))))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, target=target)
    return target


def prepared_folds(data, rpl, train_ids, task):
    """The CV folds, their base sample weights and replay plans, computed once per dataset/split.

    Only the positive-class multiplier depends on the trial; it is applied exactly as weights_for_task()
    applies it (last step, in place, float32)."""
    key = (task, id(data), id(rpl), hashlib.sha256(np.ascontiguousarray(train_ids).tobytes()).hexdigest())
    if key in _PREPARED:
        return _PREPARED[key]
    folds = []
    t0 = time.perf_counter()
    for fold, (a, b) in enumerate(cv_splits(data, train_ids)):
        tr = train_ids[a]
        val_groups = set(data["group"][train_ids[b]])
        if task == "tap" and TAP_OBS_MS is not None:          # observable labels, inside this fold's training people
            tr = obs_rows(data, rpl, tr, f"cv{fold}", OBS_CACHE_DIR)
        target = distill_targets(data, tr, f"cv{fold}", DISTILL_CACHE_DIR) if task == "tap" else None
        y = data["y"][tr]
        folds.append(dict(tr=tr, y=y, val_groups=val_groups, classes=set(np.unique(y).tolist()), target=target,
                          base_w=weights_for_task(data, tr, task, 1.0), positive=(y == 1),
                          replay=(prepare_tap_replay if task == "tap" else prepare_lp_replay)(rpl, val_groups)))
    _PREPARED.clear()
    _PREPARED[key] = folds
    notice("UNIFIED_FOLDS_PREPARED", task, len(folds), round(time.perf_counter() - t0, 2), "s")
    return folds


_RANKINGS = {}
FS_CACHE_DIR = None                 # set per task: rankings are cached on disk by their full key


def fold_rankings(data, rpl, train_ids, task):
    """v3: one feature ranking per CV fold, fitted on that fold's TRAINING people only (feature_selection.py)."""
    key = (task, id(data), id(rpl), hashlib.sha256(np.ascontiguousarray(train_ids).tobytes()).hexdigest(),
           R.TAP_FALSE_PER_MIN_TARGET)
    if key in _RANKINGS:
        return _RANKINGS[key]
    me = sys.modules[__name__]
    folds = list(enumerate(prepared_folds(data, rpl, train_ids, task)))
    # One ranking at a time with all threads (once per study, cached). Three concurrent rankings crashed the
    # 96-CPU server with a segmentation fault (2026-11-18); concurrent CV folds in trials are not affected.
    ranked = [FS.fit_fold_ranker(me, R, data, rpl, pf["tr"], task, seed=fold, cache_dir=FS_CACHE_DIR,
                                 tap_target=R.TAP_FALSE_PER_MIN_TARGET, log=lambda *a: notice(*a), threads=CPU_THREADS)
              for fold, pf in folds]
    out = []
    for (fold, pf), r in zip(folds, ranked):
        FS.validate_fold_local(r, pf["val_groups"])
        out.append(r)
        notice("UNIFIED_FS_FOLD_RANKING", task, fold, r["seconds"], "s", r["ordered_features"][:8])
    _RANKINGS.clear()
    _RANKINGS[key] = out
    return out


def fold_weights(fold, positive_weight):
    w = fold["base_w"].copy()
    w[fold["positive"]] *= float(positive_weight)
    return w.astype(np.float32)


def model_prob_fn(model, ks, task):
    run = raw_fns(model, ks)
    if task == "tap":
        return run
    return lambda X: {k: R.q20(p) for k, p in run(X).items()}


# ============================================================================ Optuna objective
def objective(trial, data, rpl, train_ids, task, budgets):
    t_start = time.perf_counter()
    CLOCK.reset()
    n_all = len(data["columns"])
    # The compact C node format indexes at most 254 model features: above that, K is capped and the ranking
    # picks them (126/141/150-field runs are unchanged: their upper bound is n_all as before).
    n_features = trial.suggest_int("n_features", 1, min(n_all, EXPORT_MAX_FEATURES))
    selected = list(range(n_all))
    lo, hi = POSITIVE_WEIGHT[task]
    positive_weight = trial.suggest_float("positive_weight", lo, hi, log=True)
    params = suggest_params(trial, task)
    primary = budgets[0]
    totals = {b: (R.TapCounts() if task == "tap" else R.LpCounts()) for b in budgets}
    folds = []
    fold_timing = []
    t0 = time.monotonic()
    with CLOCK("fold_prepare"):
        prepared = prepared_folds(data, rpl, train_ids, task)
    rankings = None
    if n_features < n_all:
        with CLOCK("feature_ranking"):
            rankings = fold_rankings(data, rpl, train_ids, task)
    selected_by_fold, sels = [], []
    for fold, pf in enumerate(prepared):
        sel = FS.select_top_k(rankings[fold], n_features) if rankings is not None else selected
        sels.append(sel)
        selected_by_fold.append([data["columns"][j] for j in sel])
        if pf["classes"] != {0, 1}:
            raise optuna.TrialPruned("Missing class in person-grouped fold")
    pool = fold_pool()
    fold_params = dict(params, n_jobs=FOLD_THREADS) if pool is not None else params

    def run_fold(pf, sel):
        CLOCK.local_reset()
        with CLOCK("fold_prepare"):
            X = data["X"][np.ix_(pf["tr"], np.asarray(sel, dtype=np.intp))]   # == data["X"][tr][:, sel], one copy
            w = fold_weights(pf, positive_weight)
        with CLOCK("fit"):
            soft = {} if pf.get("target") is None else dict(target=pf["target"])
            model = fit_model(X, pf["y"], w, fold_params, primary, **soft)
        del X
        with CLOCK("size"):
            ks = {bud: trees_within(model, bud) for bud in budgets}
        fn = model_prob_fn(model, [k for k, _ in ks.values()], task)
        res = (replay_tap_items if task == "tap" else replay_lp_items)(fn, rpl, pf["replay"], sel)
        return ks, res, model.booster_.num_trees(), CLOCK.local()
    if pool is None:              # one fold at a time: a pruned trial stops before computing the next fold
        futures = None
        results = (run_fold(pf, sel) for pf, sel in zip(prepared, sels))
    else:                         # all folds at once; folds are reported (and pruned) in the same order
        futures = [pool.submit(run_fold, pf, sel) for pf, sel in zip(prepared, sels)]
        results = (fu.result() for fu in futures)
    try:
        for fold, (pf, (ks, res, fitted, timing)) in enumerate(zip(prepared, results)):
            fold_rep = {}
            for bud, (k, size) in ks.items():
                c = res.get(k)
                if c is not None:
                    totals[bud].add(c)
                rep_ = c.report() if c is not None else {}
                usable = bool(c is not None and (c.onsets if task == "tap" else (c.holds + c.taps)))
                fold_rep[str(bud)] = dict(trees=k, bytes=size, usable=usable,
                                          **{kk: vv for kk, vv in rep_.items() if kk != "per_folder"})
            folds.append(dict(fold=fold, people=len(pf["val_groups"]), fitted_trees=fitted, budgets=fold_rep))
            fold_timing.append({k: round(v, 3) for k, v in timing.items()})
            running = [f["budgets"][str(primary)]["objective_score"] for f in folds if f["budgets"][str(primary)]["usable"]]
            with CLOCK("study_io"):
                trial.report(float(np.mean(running)) if running else 0.0, fold)
                prune = trial.should_prune()
            if prune:
                _log_timing(trial, task, t_start, fold_timing, pruned=True)
                raise optuna.TrialPruned(f"fold {fold} below median")
    finally:
        if futures is not None:   # never leave a fold running into the next trial
            wait_futures(futures)
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
    trial.set_user_attr("n_features", n_features)
    trial.set_user_attr("selected_features_by_fold", selected_by_fold)
    trial.set_user_attr("feature_ranker", FS.RANKER_VERSION)
    trial.set_user_attr("folds", folds)
    trial.set_user_attr("cv", reports)
    trial.set_user_attr("seconds", round(time.monotonic() - t0, 1))
    trial.set_user_attr("fixed_decision_threshold", task_threshold(task))
    for bud in budgets:
        trial.set_user_attr(f"score_{bud}", reports[str(bud)]["objective_score"])
        trial.set_user_attr(f"max_bytes_{bud}", max(f["budgets"][str(bud)]["bytes"] for f in folds))
    _log_timing(trial, task, t_start, fold_timing, pruned=False)
    return float(reports[str(primary)]["objective_score"])


def _log_timing(trial, task, t_start, fold_timing, pruned):
    timing = dict(total=round(time.perf_counter() - t_start, 3),
                  stages={k: round(v, 3) for k, v in sorted(CLOCK.seconds.items())}, folds=fold_timing,
                  cpu_threads=CPU_THREADS, fold_workers=FOLD_WORKERS, fold_threads=FOLD_THREADS, pruned=pruned)
    trial.set_user_attr("timing", timing)
    if TIMING_LOG is not None:
        with open(TIMING_LOG, "a") as f:
            f.write(json.dumps(dict(task=task, trial=getattr(trial, "number", None), **timing)) + "\n")


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


def compile_header_150(header: Path, name: str, output: Path) -> dict:
    """C syntax smoke against a synthetic t_core with all 150 fields (126 raw + cload_lean_8)."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return {"status": "not_available", "reason": "No C compiler"}
    stub = output / "c_compiler_check"
    stub.mkdir(parents=True, exist_ok=True)
    (stub / "core_type_def.h").write_text(R.CORE_TYPE_DEF)
    probe = stub / "all150_feature_probe.c"
    probe.write_text('#include "core_type_def.h"\nfloat probe_all_150(const t_core *p_core) {\n  return '
                     + " +\n    ".join(f"(float)({v})" for v in ALL_FEATURES) + ";\n}\n")
    unit = stub / (name + ".c")
    unit.write_text('#include "' + header.name + '"\n'
                    "int model_compiles(void) { t_core data={0}; return " + name + "_predict(&data); }\n")
    for src in (probe, unit):
        cmd = [cc, "-std=c11", "-O2", "-c", str(src), "-I", str(stub), "-I", str(header.parent),
               "-o", str(stub / (src.stem + ".o"))]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if p.returncode:
            raise RuntimeError("150-field synthetic C compile failed: " + (p.stderr or p.stdout)[-3000:])
    return {"status": "passed", "compiler": cc,
            "scope": "synthetic 150-field t_core (126 raw + cload_lean_8); not the firmware ABI or ARM build",
            "object_bytes": (stub / (name + ".o")).stat().st_size}


def check_header(header: Path, stem: str, output: Path) -> dict:
    uppercase_alias(header, stem)
    check = compile_header_150(header, stem.lower(), output)
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
    info = dict(group_name=stem, selected_indices=list(range(len(subset))), decision_threshold=task_threshold(task),
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
def header_replay(hm: R.HeaderModel, rpl, groups, task, steady=False, lp_threshold=None):
    if steady:
        rpl = dict(rpl, ready=rpl["steady"])
    if task == "tap":
        fn = lambda X: {"h": hm.decision(X)}
        res = replay_tap(fn, rpl, groups, slice(None))
    else:
        fn = lambda X: {"h": hm.value_q20(X)}
        res = replay_lp(fn, rpl, groups, slice(None), lp_threshold)
    c = res.get("h")
    return c.report() if c is not None else None


def parity(hm: R.HeaderModel, model, rpl, groups, selected, task, k, header_cols=None):
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
    Xh = np.ascontiguousarray(X[:, selected if header_cols is None else header_cols])
    if task == "tap":
        c = hm.predict_bool(Xh).astype(bool); py = p >= FIXED_THRESHOLD
        return dict(rows=len(X), decision_mismatch=float((c != py).mean()))
    c = hm.value_q20(Xh); py = R.q20(p)
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
    """Default header = best CV candidate inside the primary budget that did not fail on any CV fold
    (fold score 0: never fires / never releases for those people). The held-out people never influence the
    choice; their C replay is an audit reported once (FEATURE_SELECTION_RESEARCH_AND_EXECUTION_PLAN.md, section 5.2)."""
    model_dir = out / "models"
    primary = [c for c in candidates if c["role"].startswith("rank_")]
    chosen, skipped = None, []
    for c in primary:
        folds = (c.get("cv") or {}).get("fold_scores") or []
        if folds and min(folds) > 0:
            chosen = c; break
        skipped.append(dict(role=c["role"], trial=c["trial"],
                            reason="a CV fold scored 0 (never fires / never releases for those people)"))
    if chosen is None:
        chosen = primary[0]
        skipped.append(dict(role=chosen["role"], reason="every candidate failed a CV fold; using the CV best"))
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
    """Phase 1: each candidate is trained WITHOUT the held-out people, exported and replayed as C on them;
    the held-out report is then frozen (written) before anything is fitted on the held-out people.
    Phase 2: final model on everyone, trimmed to the budget, exported with the stable API.
    The feature subset of phase 1 comes from a ranking fitted on the development people only; phase 2
    re-fits the ranking on everyone at the same (locked) K."""
    model_dir = out / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    y = data["y"]; cols = data["columns"]
    held = set(data["group"][te])
    every = np.arange(len(y))
    if task == "tap" and TAP_OBS_MS is not None:              # observable labels inside the people each model trains on
        tr = obs_rows(data, rpl, tr, "development", OBS_CACHE_DIR)
        every = obs_rows(data, rpl, every, "everyone", OBS_CACHE_DIR)
    soft_dev = soft_all = {}
    if task == "tap" and TAP_DISTILL is not None:            # teachers inside the rows each model trains on
        soft_dev = dict(target=distill_targets(data, tr, "development", DISTILL_CACHE_DIR))
        soft_all = dict(target=distill_targets(data, every, "everyone", DISTILL_CACHE_DIR))
    me = sys.modules[__name__]
    ranking_dev = ranking_all = None
    if any(t.params["n_features"] < len(cols) for _, _, t in picks):
        ranking_dev = FS.fit_fold_ranker(me, R, data, rpl, tr, task, seed=101, cache_dir=FS_CACHE_DIR,
                                         tap_target=R.TAP_FALSE_PER_MIN_TARGET, log=lambda *a: notice(*a))
        FS.validate_fold_local(ranking_dev, held)

    def subset_for(trial, ranking):
        k = int(trial.params["n_features"])
        return list(cols) if (k >= len(cols) or ranking is None) else [cols[j] for j in FS.select_top_k(ranking, k)]

    pre_reports = []
    for role, budget, trial in picks:                         # ---- phase 1: held-out audit
        params = trial.user_attrs["training_params"]; pw = float(trial.params["positive_weight"])
        subset = subset_for(trial, ranking_dev); ix = [cols.index(x) for x in subset]
        pre = fit_model(data["X"][np.ix_(tr, np.asarray(ix, np.intp))], y[tr], weights_for_task(data, tr, task, pw),
                        params, budget, **soft_dev)
        k_pre, _ = trees_within(pre, budget); pre = truncated(pre, k_pre)
        h_pre, b_pre, _ = write_header(pre, subset, stem, out / "_holdout_headers" / role, task, out)
        hm_cols = R.input_columns(subset); hm_ix = [cols.index(x) for x in hm_cols]   # + raw fields of helper calls
        hm = R.HeaderModel(h_pre, hm_cols, out / "_replay_build", prefer=stem.lower())
        rpl_sub = dict(rpl, X=_ColumnView(rpl["X"], hm_ix))
        pre_reports.append(dict(role=role, trial=trial.number, features_development=subset,
                                holdout_c_header=header_replay(hm, rpl_sub, held, task),
                                holdout_c_header_steady_state=header_replay(hm, rpl_sub, held, task, steady=True),
                                holdout_c_header_placement_stress=stress_replay(hm, rpl, cols, hm_ix, held, task),
                                holdout_model_bytes=b_pre,
                                c_vs_python_parity=parity(hm, pre, rpl, held, ix, task, k_pre, hm_ix)))
    write_json(out / "holdout_report.json", dict(
        note="Frozen before any model or feature ranking was fitted on the held-out people; never used to choose.",
        heldout_people=sorted(held), candidates=pre_reports))
    if ranking_dev is not None:
        ranking_all = FS.fit_fold_ranker(me, R, data, rpl, every, task, seed=202, cache_dir=FS_CACHE_DIR,
                                         tap_target=R.TAP_FALSE_PER_MIN_TARGET, log=lambda *a: notice(*a))
    rows = []
    for (role, budget, trial), pre_rep in zip(picks, pre_reports):    # ---- phase 2: production models
        params = trial.user_attrs["training_params"]; pw = float(trial.params["positive_weight"])
        subset = subset_for(trial, ranking_all); ix = [cols.index(x) for x in subset]
        folder = model_dir / "candidates" / role
        final = fit_model(data["X"][np.ix_(every, np.asarray(ix, np.intp))], y[every],
                          weights_for_task(data, every, task, pw), params, budget, **soft_all)
        k_fin, _ = trees_within(final, budget); final = truncated(final, k_fin)
        header, size, checked = write_header(final, subset, stem, folder, task, out)
        if size > budget:
            raise AssertionError(f"{role}: {size} bytes exceeds budget {budget}")
        tag = f"{role}_trial{trial.number:03d}"
        joblib.dump(dict(model=final, preprocessing="none", indices=list(range(len(subset))), feature_names=subset),
                    model_dir / (tag + ".pkl"), compress=3)
        report = dict(role=role, trial=trial.number, budget_bytes=budget, static_bytes=size, trees=k_fin,
                      params=params, positive_weight=pw, features=subset, n_features=len(subset),
                      header=str(header.relative_to(out)), cv=trial.user_attrs["cv"].get(str(budget)),
                      holdout_c_header=pre_rep["holdout_c_header"],
                      holdout_c_header_steady_state=pre_rep["holdout_c_header_steady_state"],
                      holdout_c_header_placement_stress=pre_rep["holdout_c_header_placement_stress"],
                      features_holdout_candidate=pre_rep["features_development"],
                      derived_features_used=[c for c in subset if R.helper_args(c)],
                      lean8_features_used=[c for c in subset if "cload_lean_8" in c],
                      holdout_model_bytes=pre_rep["holdout_model_bytes"],
                      c_vs_python_parity=pre_rep["c_vs_python_parity"], c_stub=checked,
                      limitations=["Silver labels from an offline teacher, not physical event truth",
                                   "Replay timing rules are documented assumptions about the firmware",
                                   "C replay uses a synthetic all-float t_core, not the nRF52832 ABI"])
        write_json(model_dir / (tag + ".json"), report)
        rows.append(report)
        notice("UNIFIED_MODEL_EXPORT", json.dumps(dict(task=task, role=role, trial=trial.number, bytes=size,
               trees=k_fin, features=len(subset), cv=(report["cv"] or {}).get("objective_score"),
               holdout=(report["holdout_c_header"] or {}).get("objective_score"),
               parity=report["c_vs_python_parity"])))
    export_candidates.rankings = (ranking_dev, ranking_all)
    return rows


def _lite_by_recording(data):
    """(recording, positions of its training rows) for every recording with training rows."""
    rec = data["rec"]; order = np.argsort(rec, kind="stable")
    u, first = np.unique(rec[order], return_index=True); last = np.r_[first[1:], len(order)]
    return [(int(r), order[a:b]) for r, a, b in zip(u, first, last)]


def augment_lite(data, rpl, nominal_check=None):
    """Lean-lite diff inputs of the augmented TAP training rows: per recording, the same placement draw as
    TF.augment applied to the recording's whole WL history from its first row, then the Lean-lite emulator
    (as researched; the inputs depend on ~10 s of history, so a per-row transform would be wrong)."""
    t0 = time.perf_counter()
    cols = data["columns"]; li = np.array([cols.index(c) for c in TF.LITE_COLUMNS])
    if "row" not in data:
        raise ValueError("Lean-lite augmentation needs the row index of each training row (rebuild the cache)")
    d = TF.draws(len(rpl["recs"]), SEED)
    starts = rpl["recs"].start.to_numpy(np.int64); ends = rpl["recs"].end.to_numpy(np.int64)
    check = nominal_check is not None
    for r, pos in _lite_by_recording(data):
        a, b = int(starts[r]), int(ends[r])
        Xr = np.asarray(rpl["X"][a:b]); fr = np.asarray(rpl["frames"][a:b]); rows = data["row"][pos]
        if check:                                # the cache's nominal values are the emulator's (row mapping right)
            first = pos[pos < len(nominal_check)]
            if len(first):
                want = TF.lite_rows(Xr, cols, fr)[data["row"][first]]
                if not np.array_equal(want, nominal_check[first]):
                    raise AssertionError(f"Lean-lite rows of recording {r} do not match the cache")
                check = False
        L = TF.lite_rows_placement(Xr, cols, fr, d["gain"][r], d["tilt"][r], d["noise"][r])
        data["X"][np.ix_(pos, li)] = L[rows]
    notice("UNIFIED_TAP_LITE_AUGMENTED", len(data["X"]), "rows", round(time.perf_counter() - t0, 1), "s")


def stress_enabled(task) -> bool:
    """The placement stress transforms raw EMG fields (and derived TAP inputs); Lean fields would not follow."""
    return task == "tap" and LEAN8_MODE == "off"


def _replay_rec_of_row(rpl):
    m = rpl.get("_rec_of_row")
    if m is None:
        m = np.full(len(rpl["X"]), -1, np.int32)
        for i, r in rpl["recs"].iterrows():
            if r.start >= 0:
                m[int(r.start):int(r.end)] = i
        rpl["_rec_of_row"] = m
    return m


class _StressView:
    """Replay rows after a simulated band move / contact change (TAP_GENERALIZATION_RESULTS.md 'placement': EMG gain
    +-1 octave, tilt +-0.5 octave, contact noise 0.1..1.0; ACC unchanged), restricted to `sel` columns.
    Lean-lite diff columns are recomputed from the changed WL history of the whole recording (slices are inside
    one recording: the replay reads per recording)."""
    def __init__(self, X, sel, columns, rec_of_row, draws, recs=None, frames=None):
        self.X, self.sel, self.columns, self.rec, self.d = X, sel, columns, rec_of_row, draws
        self.lite = ([columns.index(c) for c in TF.LITE_COLUMNS] if set(TF.LITE_COLUMNS) <= set(columns) else None)
        if self.lite is not None and (recs is None or frames is None):
            raise ValueError("Stressed Lean-lite inputs need the recordings table and frame counters")
        self.recs, self.frames, self._last = recs, frames, (None, None)
    def _lite(self, r):
        if self._last[0] != r:
            a, b = int(self.recs.start.iloc[r]), int(self.recs.end.iloc[r])
            self._last = (r, TF.lite_rows_placement(np.asarray(self.X[a:b]), self.columns, np.asarray(self.frames[a:b]),
                                                    self.d["gain"][r], self.d["tilt"][r], self.d["noise"][r]))
        return self._last[1]
    def __getitem__(self, s):
        rows = np.asarray(self.X[s]); r = self.rec[s]
        Y = TF.apply_placement(rows, self.columns, self.d["gain"][r], self.d["tilt"][r], self.d["noise"][r])
        if self.lite is not None and len(rows):
            rr = int(r[0])
            if not isinstance(s, slice) or (r != rr).any():
                raise ValueError("Stressed Lean-lite rows must be read one recording slice at a time")
            st = int(self.recs.start.iloc[rr]); a = s.start or 0
            Y[:, self.lite] = self._lite(rr)[a - st:a - st + len(rows)]
        return Y[:, self.sel]
    def __len__(self):
        return len(self.X)


def stress_replay(hm, rpl, cols, sel, groups, task):
    if not stress_enabled(task):
        return None
    view = _StressView(rpl["X"], sel, list(cols), _replay_rec_of_row(rpl), TF.stress_draws(len(rpl["recs"])),
                       recs=rpl["recs"], frames=rpl["frames"])
    return header_replay(hm, dict(rpl, X=view), groups, task)


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
    bthr = LP_BASELINE_RELEASE_THRESHOLD if task == "lp" else None
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
            missing = [c for c in ALL_FEATURES if c not in data["columns"] and c.replace(" ", "") in text]
            if missing:
                raise ValueError(f"baseline uses fields missing from this data: {missing[:5]}")
            hm = R.HeaderModel(p, data["columns"], out / "_replay_build",
                               prefer=("tap_swipe" if task == "tap" else "long_press"))
            rows.append(dict(name=name, status="replayed", path=str(p), sha256=digest,
                             api=dict(value=hm.value_prefix, decision=hm.bool_prefix),
                             release_threshold=LP_BASELINE_RELEASE_THRESHOLD if task == "lp" else None,
                             holdout=header_replay(hm, rpl, held, task, lp_threshold=bthr),
                             holdout_steady_state=header_replay(hm, rpl, held, task, steady=True, lp_threshold=bthr),
                             holdout_placement_stress=stress_replay(hm, rpl, data["columns"],
                                                                    list(range(len(data["columns"]))), held, task),
                             all_people=header_replay(hm, rpl, everyone, task, lp_threshold=bthr),
                             note="all_people may include this baseline's own training data"))
        except Exception as exc:          # a baseline problem must never stop training
            rows.append(dict(name=name, status="not replayable", path=str(p), error=repr(exc)[:600]))
        if sha256(p) != digest:
            raise AssertionError(f"Baseline file changed during replay: {p}")
        notice("UNIFIED_BASELINE", task, name, rows[-1]["status"])
    return rows


def comparison_table(task, candidates, baselines):
    text = _comparison_rows(task, candidates, baselines, "holdout_c_header", "holdout")
    if True:
        text += (f"\nSteady state only (rows after {LEAN8_STEADY_FRAMES} Lean-8 frames, ~10.8 s, "
                 "i.e. lean_8_features_ready()):\n\n"
                 + _comparison_rows(task, candidates, baselines, "holdout_c_header_steady_state", "holdout_steady_state"))
    if any(c.get("holdout_c_header_placement_stress") for c in candidates):
        text += ("\nSimulated band move / contact change on the same held-out people (TAP_GENERALIZATION_RESULTS.md: EMG "
                 "gain +-1 octave, spectral tilt +-0.5 octave, contact noise; ACC unchanged; the same draw for every "
                 "model). A model that generalizes keeps recall without more false events:\n\n"
                 + _comparison_rows(task, candidates, baselines, "holdout_c_header_placement_stress", "holdout_placement_stress"))
    rows = acceptance(task, candidates, baselines)
    if rows:
        text += ("\nAcceptance of the default header against each replayed baseline, held-out people "
                 "(PIPELINE.md; report only, the choice is made on CV):\n\n"
                 "| Baseline | Metric | New | Baseline | Rule | Result |\n| --- | --- | --- | --- | --- | --- |\n"
                 + "".join(f"| {r['baseline']} | {r['metric']} | {r['new']:.3f} | {r['baseline_value']:.3f} | "
                           f"{r['rule']} | {'pass' if r['pass'] else '**review**'} |\n" for r in rows))
    return text


# Go/no-go rules before a header goes into the firmware (PIPELINE.md): a new default should find the taps the
# previous header found (one point of slack), without more false events (10% + 0.1/min slack), and LP should
# keep holds, release on time and release taps at least as well.
ACCEPTANCE = dict(tap=(("recall_press", "min", 0.01), ("recall", "min", 0.01), ("false_per_min", "max", None)),
                  lp=(("hold_kept", "min", 0.0), ("timely_release", "min", 0.0), ("tap_released", "min", 0.0)))


def acceptance(task, candidates, baselines):
    """Report-only comparison of the default header with every replayed baseline on the held-out people."""
    d = next((c for c in candidates if c.get("default")), None)
    new = (d or {}).get("holdout_c_header") or {}
    rows = []
    for b in baselines:
        old = b.get("holdout") or {}
        if b.get("status") != "replayed" or not new or not old:
            continue
        for metric, kind, slack in ACCEPTANCE[task]:
            a, z = new.get(metric), old.get(metric)
            if a is None or z is None or a != a or z != z:
                continue
            if kind == "min":
                ok, rule = a >= z - slack, f">= baseline - {slack:g}"
            else:
                ok, rule = a <= z * 1.1 + 0.1, "<= baseline x 1.1 + 0.1"
            rows.append(dict(baseline=b["name"], metric=metric, new=float(a), baseline_value=float(z), rule=rule,
                             **{"pass": bool(ok)}))
    return rows


def _comparison_rows(task, candidates, baselines, ckey, bkey):
    # recall = confirmed taps (the objective); recall_press = every real pointer press of the TAP/SWIPE recordings,
    # a reference that does not depend on the labeller (FEATURE_SELECTION_RESULTS.md, section 5).
    key = (["recall", "recall_press", "false_per_min", "latency_ms_median", "objective_score"] if task == "tap"
           else ["hold_kept", "timely_release", "tap_released", "release_latency_ms_median", "objective_score"])
    head = "| Model | Bytes | " + " | ".join(key) + " |\n|" + " --- |" * (len(key) + 2) + "\n"
    fmt = lambda v: "-" if v is None or (isinstance(v, float) and v != v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = []
    for c in candidates:
        h = c.get(ckey) or {}
        mark = " **default**" if c.get("default") else ""
        kk = f", K={c['n_features']}" if c.get("n_features") else ""
        lines.append(f"| new {c['role']} (trial {c['trial']}{kk}){mark} | {c['static_bytes']} | " + " | ".join(fmt(h.get(k)) for k in key) + " |")
    for b in baselines:
        h = b.get(bkey) or {}
        at = f", release {b['release_threshold']}" if b.get("release_threshold") else ""
        lines.append(f"| baseline {b['name']} ({b['status']}{at}) | - | " + " | ".join(fmt(h.get(k)) for k in key) + " |")
    return head + "\n".join(lines) + "\n"


# ============================================================================ v3: TAP target + reports
def resolve_tap_target(data, rpl, tr, out: Path):
    """TAP false-event budget for the score recall * min(1, target / false per min).
    'baseline': the previous production TAP header replayed on the DEVELOPMENT people (same gate, same rules)."""
    setting = TAP_TARGET_SETTING.strip().lower()
    info = dict(setting=setting)
    if setting != "baseline":
        v = float(setting)
        info.update(value=v, source="configured")
        return v, info
    path = next((p for n, p in BASELINES["tap"] if n == TAP_TARGET_BASELINE), "")
    p = Path(path) if path else None
    if p is not None and p.is_file():
        try:
            digest = sha256(p)
            hm = R.HeaderModel(p, data["columns"], out / "_replay_build", prefer="tap_swipe")
            rep_ = header_replay(hm, rpl, set(data["group"][tr]), "tap")
            if sha256(p) != digest:
                raise AssertionError("baseline header changed during replay")
            fpm = float(rep_["false_per_min"])
            if np.isfinite(fpm) and fpm > 0:
                v = round(fpm, 3)
                info.update(value=v, source=f"{TAP_TARGET_BASELINE} replayed on the development people",
                            baseline=str(p), baseline_sha256=digest, baseline_development_recall=rep_["recall"],
                            baseline_development_false_per_min=fpm, per_folder=rep_.get("per_folder"))
                return v, info
            info["error"] = f"baseline false rate not usable: {fpm}"
        except Exception as exc:
            info["error"] = repr(exc)[:400]
    info.update(value=TAP_TARGET_FALLBACK, source="FALLBACK (previous TAP header not available): set "
                "MUDRA_TAP_BASELINE_HEADER, or MUDRA_TAP_FALSE_PER_MIN_TARGET to a number")
    notice("UNIFIED_TAP_TARGET_FALLBACK", info["source"], info.get("error", ""))
    return TAP_TARGET_FALLBACK, info


def _family(name: str) -> str:
    return ("EMG diff" if name.startswith("p_core->diff(") else "WL ratio" if name.startswith("p_core->ratio(") else
            "Lean-8" if "cload_lean_8" in name else "Lean-lite" if "cload_lite" in name else "ACC" if "acc_" in name else
            "EMG" if "emg_psd" in name else "WL" if ".wl" in name or "->wl" in name else "other")


def write_fs_reports(out: Path, task, data, study, candidates, default):
    """feature_selection/: fold rankings (all features, ranks, importances, clusters), stability, per-trial
    subsets, final features and a short report. Families are reporting labels only."""
    d = out / "feature_selection"; d.mkdir(parents=True, exist_ok=True)
    cols = list(data["columns"])
    folds = next(iter(_RANKINGS.values()), None)
    ranking_dev, ranking_all = getattr(export_candidates, "rankings", (None, None))

    def table(r):
        pos = {j: i + 1 for i, j in enumerate(r["order"])}
        gain = np.mean([x["gain"] for x in r["inner"]], axis=0) if r["inner"] else np.zeros(len(cols))
        return pd.DataFrame(dict(
            feature=cols, family=[_family(c) for c in cols], rank=[pos[j] for j in range(len(cols))],
            mean_importance=r["mean_importance"], cluster=r["cluster_of"],
            cluster_size=[len(r["clusters"][c]) for c in r["cluster_of"]],
            cluster_importance=[r["mean_group_importance"][c] for c in r["cluster_of"]],
            cluster_importance_se=[r["group_se"][c] for c in r["cluster_of"]], lightgbm_gain_share=gain,
        )).sort_values("rank")

    files = {}
    for name, r in ([(f"fold_{i}", x) for i, x in enumerate(folds or [])]
                    + [("development", ranking_dev), ("final", ranking_all)]):
        if r is not None:
            table(r).to_csv(d / f"{name}_ranking.csv", index=False)
            files[name] = dict(csv=f"{name}_ranking.csv", seconds=r["seconds"], cluster_cut=r["cluster_cut"],
                               silhouette=r["silhouette"], clusters=[[cols[j] for j in c] for c in r["clusters"]],
                               inner=[{k: v for k, v in x.items() if k in ("split", "people", "threshold", "baseline_score")}
                                      for x in r["inner"]])
    write_json(d / "correlation_clusters.json", files)
    stability = FS.report_stability(folds) if folds else None
    write_json(d / "selection_stability.json", stability)
    trials = {t.number: dict(n_features=t.user_attrs.get("n_features"), value=t.value,
                             by_fold=t.user_attrs.get("selected_features_by_fold"))
              for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE}
    write_json(d / "selected_features_by_trial.json", trials)
    final = dict(role=default["role"], trial=default["trial"], n_features=default["n_features"],
                 features=default["features"], features_holdout_candidate=default["features_holdout_candidate"],
                 families={f: sum(_family(c) == f for c in default["features"])
                           for f in ("ACC", "EMG", "WL", "EMG diff", "WL ratio", "Lean-8", "Lean-lite")})
    write_json(d / "final_selected_features.json", final)
    ks = sorted({v["n_features"] for v in trials.values() if v["n_features"] is not None})
    lines = [f"# {task}: learned feature selection ({FS.RANKER_VERSION})", "",
             f"- Candidate pool: {len(cols)} firmware fields (no family rules).",
             f"- Fold rankings: {len(folds or [])} (each from its fold's training people only); "
             f"method: {(folds or [ranking_all or {}])[0].get('method', 'n/a') if (folds or ranking_all) else 'n/a'}.",
             f"- K tried by Optuna: {ks}.",
             f"- Default model: trial {default['trial']}, K = {default['n_features']}, families {final['families']}.",
             f"- CV objective {(default.get('cv') or {}).get('objective_score')}; held-out (reported once, never used "
             f"to choose): {(default.get('holdout_c_header') or {}).get('objective_score')}.", ""]
    if stability:
        lines.append("Stability (Nogueira) of top-K across the outer-fold rankings: " + ", ".join(
            f"K={k}: {v['outer_folds']:.2f}" for k, v in stability.items() if k.isdigit()))
    (d / "report.md").write_text("\n".join(lines) + "\n")
    return dict(directory=str(d.relative_to(out)), default_k=default["n_features"], trial_ks=ks,
                stability=stability, families=final["families"])


# ============================================================================ study safety + reproducibility
def backup_study_db(db: Path, out: Path):
    """Consistent copy of an existing Optuna SQLite study (SQLite backup API) before it is resumed."""
    if not db.is_file():
        return None
    import sqlite3
    dst = out / "backups" / f"{db.stem}_{time.strftime('%Y%m%dT%H%M%S')}.sqlite3"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        with sqlite3.connect(dst) as copy_:
            src.backup(copy_)
    finally:
        src.close()
    notice("UNIFIED_STUDY_BACKUP", str(dst))
    return dst


def guard_study_data(study, data, out: Path, backup):
    """Fail closed if a resumed study was run on different data; snapshot the study before new trials."""
    trials = study.trials
    if trials:
        snap = [dict(number=t.number, state=t.state.name, params=t.params, value=t.value,
                     intermediate=t.intermediate_values) for t in trials]
        write_json(out / f"study_snapshot_{time.strftime('%Y%m%dT%H%M%S')}.json",
                   dict(study=study.study_name, backup=str(backup) if backup else None, trials=snap))
    want = data.get("content_sha256")
    have = study.user_attrs.get("data_content_sha256")
    if have is None:
        # Study from code without this guard: its trials ran on the cache that existed before any rebuild.
        have = (data.get("previous_content_sha256") if trials else None) or want
        study.set_user_attr("data_content_sha256", have)
        study.set_user_attr("data_content_adopted_at_trials", len(trials))
    if have != want:
        raise RuntimeError(f"Study {study.study_name} was run on different training/replay data "
                           f"(study {have[:16]}..., cache {want[:16]}...). Refusing to resume; restore the "
                           f"original data/labels or start a new study (move {out}).")


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], text=True, capture_output=True, timeout=20).stdout.strip()
    except Exception:
        return None


def write_run_manifest(out: Path, task, data, budgets, study):
    import platform
    def read(path, key):
        try:
            for line in Path(path).read_text().splitlines():
                if line.startswith(key):
                    return line.split(":", 1)[1].strip()
        except OSError:
            return None
    lscpu = None
    try:
        lscpu = subprocess.run(["lscpu"], text=True, capture_output=True, timeout=10).stdout
    except Exception:
        pass
    fw = lean8.firmware_header() if USE_LEAN8 else (lite.HEADER if (USE_LITE or TAP_LITE == "diff") else None)
    fp_file = out / "cache" / "fingerprint.json"
    write_json(out / "run_manifest.json", dict(
        written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), task=task, commit=_git("rev-parse", "HEAD"),
        dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
        versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                      lightgbm=lgb.__version__, optuna=optuna.__version__, platform=platform.platform()),
        cpu=dict(logical=os.cpu_count(), affinity=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                 model=read("/proc/cpuinfo", "model name"), lscpu=lscpu, lightgbm_threads=CPU_THREADS),
        memory_total=read("/proc/meminfo", "MemTotal"),
        disk_free_gb=round(shutil.disk_usage(out).free / 1e9, 1),
        hashes=dict(lean8_algorithm=lean8.GENERATOR_SHA if USE_LEAN8 else (lite.GENERATOR_SHA if USE_LITE else None),
                    lean_lite_diff=dict(emulator=lite.DIFF_SHA, inputs=TF.LITE_VERSION) if TAP_LITE == "diff" else None,
                    firmware_header=sha256(fw) if fw else None, labeler=sha256(CLAUDE_LABELER),
                    cache_fingerprint=json.loads(fp_file.read_text()).get("fingerprint") if fp_file.is_file() else None,
                    data_content=data.get("content_sha256")),
        study=dict(name=study.study_name, trials=len(study.trials), requested=TRIALS_PER_MODEL, folds=CV_FOLDS),
        budgets_bytes=budgets, threshold=task_threshold(task),
        experiment=EXPERIMENT, tap_soft_labels=TAP_SOFT_MIN_SCORE if task == "tap" else None, frame_gates=dict(train=TRAIN_MIN_FRAMES, eval=EVAL_MIN_FRAMES, steady=LEAN8_STEADY_FRAMES),
        tap_false_per_min_target=R.TAP_FALSE_PER_MIN_TARGET if task == "tap" else None,
        feature_ranker=dict(version=FS.RANKER_VERSION, inner_folds=FS.INNER_FOLDS, screen=FS.SCREEN_PARAMS,
                            repeats=[FS.REPEATS_FEATURE, FS.REPEATS_GROUP]),
        lean8=lean8_setting(), search=dict(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, num_leaves=NUM_LEAVES,
                                           learning_rate=LEARNING_RATE, min_child_samples=MIN_CHILD_SAMPLES,
                                           positive_weight=POSITIVE_WEIGHT[task]),
        replay_rules={k: getattr(R, k) for k in dir(R) if k.startswith(("TAP_", "LP_"))}))


# ============================================================================ one task
def same_lean8_setting(saved) -> bool:
    """Saved Lean-8 setting equals the current one; generator hashes of numerically identical earlier
    generator versions (lean8.LEGACY_EQUIVALENT_SHAS) count as equal."""
    if not isinstance(saved, dict):
        return False
    now = lean8_setting()
    a, b = dict(saved), dict(now)
    if a.get("generator_sha256") in getattr(lean8, "LEGACY_EQUIVALENT_SHAS", ()):
        a["generator_sha256"] = b["generator_sha256"]
    return a == b


def lean8_setting():
    out = dict(enabled=LEAN8_TAG, train_min_frames=TRAIN_MIN_FRAMES, eval_min_frames=EVAL_MIN_FRAMES,
               steady_frames=LEAN8_STEADY_FRAMES,
               generator_sha256=lite.GENERATOR_SHA if USE_LITE else lean8.GENERATOR_SHA)
    if TAP_LITE == "diff":                       # the TAP model reads Lean-lite diff fields (firmware/lean_lite.h)
        out["lean_lite_diff"] = dict(emulator_sha256=lite.DIFF_SHA, header_sha256=sha256(lite.HEADER))
    return out


def train_task(task: str, tap_files, lp_files):
    definition = next(v for v in TASKS if v["name"] == task)
    stem = definition["header"]
    out = OUTPUT / task
    summary_file = out / "summary.json"
    budgets = [b for b in (budget_bytes(MODEL_BUDGET_KB), budget_bytes(ALT_BUDGET_KB)) if b]
    if len(budgets) == 2 and budgets[1] >= budgets[0]:
        budgets = budgets[:1]
    records, duplicates = dedupe_records(selected_sources(task, tap_files, lp_files))
    identity = record_identity(records)
    notice("UNIFIED_DATA_IDENTITY", task, identity["sha256"][:16], identity["recordings"], "recordings",
           len(duplicates), "duplicates left out")
    if summary_file.is_file():
        saved = json.loads(summary_file.read_text())
        stored = saved.get("data_identity")
        if stored and (out / "data_identity.json").is_file():
            stored = json.loads((out / "data_identity.json").read_text())   # with the file list (names the changes)
        changed = identity_changes(stored, identity)
        if saved.get("status") == "success" and changed:
            raise ValueError("Existing output was trained on other data: " + "; ".join(changed) +
                             ". Move it away to train on the current recordings/labels: " + str(out))
        if (saved.get("status") == "success" and saved.get("fixed_threshold") == task_threshold(task)
                and saved.get("budgets") == budgets and same_lean8_setting(saved.get("lean8"))
                and saved.get("experiment") == EXPERIMENT
                and (task != "tap" or ((saved.get("tap_target") or {}).get("setting") == TAP_TARGET_SETTING.strip().lower()
                                       and saved.get("tap_soft_labels") == TAP_SOFT_MIN_SCORE))
                and (out / "models" / (stem + ".h")).is_file()):
            notice("UNIFIED_COMPLETE_REUSED", task, str(out))
            return saved
        raise ValueError("Existing output does not match this configuration (threshold, budgets, frame gates, "
                         "TAP target or soft labels changed); move it away to start a new experiment: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    data, rpl, info = build_task_data(task, records, out)
    info["duplicate_recordings_left_out"] = len(duplicates)
    if SPLIT_BY == "file":
        group_by_file(data, rpl)
        notice("UNIFIED_SPLIT_BY_FILE", task, len(set(data["group"])), "files as groups,",
               len(set(data["person"])), "people (the same person can be on both sides)")
    if task == "tap" and TAP_AUG == "mild":
        # Training rows only (CV, rankings and final models train on them); every replay / held-out score uses
        # the real recordings. Deterministic: the same draw per recording on every run and resume.
        t_aug = time.perf_counter()
        if TAP_OBS_MS is not None or TAP_DISTILL is not None:   # their teachers read the real (nominal) rows
            data["X_nominal"] = data["X"].copy()
        nominal_lite = None
        if TAP_LITE == "diff":                  # kept for the check below
            nominal_lite = data["X"][:2000, [data["columns"].index(c) for c in TF.LITE_COLUMNS]].copy()
        data["X"] = TF.augment(data["X"], data["columns"], data["rec"], len(rpl["recs"]), SEED)
        if TAP_LITE == "diff":
            augment_lite(data, rpl, nominal_lite)
        # The training data differs from the cache files: the study guard and the ranking cache must see that.
        data["content_sha256"] = hashlib.sha256((data["content_sha256"] + "|tap_aug=mild|" + TF.VERSION).encode()).hexdigest()
        notice("UNIFIED_TAP_AUGMENTED", "mild placement", len(data["X"]), "rows", round(time.perf_counter() - t_aug, 1), "s")
    if task == "tap" and TAP_OBS_MS is not None:
        import inspect
        data["obs_content_sha256"] = data["content_sha256"]
        obs_ver = hashlib.sha256((inspect.getsource(obs_select) + inspect.getsource(_obs_owner) +
                                  inspect.getsource(obs_rows) + json.dumps(OBS_TEACHER, sort_keys=True)).encode()).hexdigest()
        data["content_sha256"] = hashlib.sha256((data["content_sha256"] + f"|tap_obs={TAP_OBS_MS}|" + obs_ver).encode()).hexdigest()
        global OBS_CACHE_DIR
        OBS_CACHE_DIR = out / "cache" / "obs"
    if task == "tap" and TAP_FINGER_WEIGHT != 1.0:           # rankings and studies are fitted with these weights
        data["content_sha256"] = hashlib.sha256((data["content_sha256"] + f"|tap_finger_weight={TAP_FINGER_WEIGHT}").encode()).hexdigest()
    if task == "tap" and TAP_DISTILL is not None:
        import inspect
        data["distill_content_sha256"] = data["content_sha256"]
        d_ver = hashlib.sha256((inspect.getsource(distill_targets) + inspect.getsource(soft_rows) +
                                json.dumps(OBS_TEACHER, sort_keys=True)).encode()).hexdigest()
        data["content_sha256"] = hashlib.sha256((data["content_sha256"] + f"|tap_distill={TAP_DISTILL}|" + d_ver).encode()).hexdigest()
        global DISTILL_CACHE_DIR
        DISTILL_CACHE_DIR = out / "cache" / "distill"
    tr, te = split_heldout(data, task)
    notice("UNIFIED_TRAIN_BEGIN", json.dumps({
        "task": task, "split_by": SPLIT_BY, "subjects": len(set(data["group"])), "train_rows": len(tr), "holdout_rows": len(te),
        "holdout_people": sorted(set(data["group"][te])), "budgets_bytes": budgets,
        "class_one": definition["class_one"], "fixed_threshold": task_threshold(task)}))
    split_info = dict(train_groups=sorted(set(data["group"][tr])), heldout_groups=sorted(set(data["group"][te])),
                      seed=SEED + int(task == "lp"))
    if "stratum" in data:                        # gesture-aware file split: files per type on each side
        split_info["strata"] = strata_balance(data, tr, te)
        n_tr, n_te = len(split_info["train_groups"]), len(split_info["heldout_groups"])
        notice("UNIFIED_SPLIT_GESTURES", task, len(split_info["strata"]), "recording types,", n_tr, "training /",
               n_te, "held-out files; per folder:",
               json.dumps({f_: [sum(v[0] for k, v in split_info["strata"].items() if k.split("|")[0] == f_),
                                sum(v[1] for k, v in split_info["strata"].items() if k.split("|")[0] == f_)]
                           for f_ in sorted({k.split("|")[0] for k in split_info["strata"]})}))
    write_json(out / "split.json", split_info)
    global FS_CACHE_DIR
    FS_CACHE_DIR = out / "feature_selection" / "cache"
    target_info = None
    if task == "tap":
        R.TAP_FALSE_PER_MIN_TARGET, target_info = resolve_tap_target(data, rpl, tr, out)
        notice("UNIFIED_TAP_TARGET", json.dumps({k: v for k, v in target_info.items() if k != "per_folder"}))
    db = out / "optuna_study.sqlite3"
    backup = backup_study_db(db, out)
    global TIMING_LOG
    TIMING_LOG = out / "timing.jsonl"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED + int(task == "lp"), multivariate=True, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
        storage="sqlite:///" + str((out / "optuna_study.sqlite3").resolve()),
        study_name=study_name(task), load_if_exists=True)
    guard_study_data(study, data, out, backup)
    write_run_manifest(out, task, data, budgets, study)
    if len(study.trials) == 0:
        # A known-good anchor (the research student) and a deep-tree anchor, inside the budget.
        # LP anchors start at a release weight that puts the replay optimum near the release threshold.
        pw0 = LP_ANCHOR_POSITIVE_WEIGHT if task == "lp" else 1.0
        # Both anchors are the all-feature control (K = every available field).
        fsel = dict(n_features=min(len(data["columns"]), EXPORT_MAX_FEATURES))
        if NUM_LEAVES == (8, 127):
            study.enqueue_trial(dict(**fsel, positive_weight=pw0, max_depth=6, num_leaves=15,
                                     n_estimators=160, learning_rate=0.1, min_child_samples=100, reg_lambda=1.0,
                                     reg_alpha=1e-3, colsample_bytree=0.8, subsample=0.8))
            study.enqueue_trial(dict(**fsel, positive_weight=pw0, max_depth=10, num_leaves=63,
                                     n_estimators=200, learning_rate=0.2, min_child_samples=50, reg_lambda=1.0,
                                     reg_alpha=1e-3, colsample_bytree=0.7, subsample=0.8))
        else:                                    # small-tree range: anchors inside it
            clip = lambda v: int(min(max(v, NUM_LEAVES[0]), NUM_LEAVES[1]))
            # the researched reference (TAP_FOCUS_RESULTS.md, 14 KB / 7 leaves: 400 trees fitted, trimmed to the budget)
            study.enqueue_trial(dict(**fsel, positive_weight=pw0, max_depth=8, num_leaves=clip(7),
                                     n_estimators=400, learning_rate=0.1, min_child_samples=100, reg_lambda=1.0,
                                     reg_alpha=1e-3, colsample_bytree=0.7, subsample=0.8))
            study.enqueue_trial(dict(**fsel, positive_weight=pw0, max_depth=10, num_leaves=clip(15),
                                     n_estimators=200, learning_rate=0.2, min_child_samples=50, reg_lambda=1.0,
                                     reg_alpha=1e-3, colsample_bytree=0.7, subsample=0.8))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for t in study.trials:                      # trials run one at a time: a RUNNING trial here died with a crash
        if t.state == optuna.trial.TrialState.RUNNING:
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
            notice("UNIFIED_STALE_TRIAL_MARKED_FAILED", task, t.number)
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
                   "K": trial.user_attrs["n_features"],
                   "depth": trial.user_attrs["training_params"]["max_depth"]}))
            return z
        study.optimize(fn, n_trials=1, n_jobs=1, show_progress_bar=False, catch=())
    picks = pick_trials(study, budgets)
    candidates = export_candidates(data, rpl, task, stem, tr, te, picks, out)
    default, skipped = promote(out, stem, candidates, budgets)
    for c in candidates:
        c["default"] = c is default
    baselines = replay_baselines(task, data, rpl, te, out)
    fs_report = write_fs_reports(out, task, data, study, candidates, default)
    table = comparison_table(task, candidates, baselines)
    (out / "COMPARISON.md").write_text(f"# {task}: held-out people, replayed through the C headers\n\n" + table)
    notice(table)
    write_json(out / "models" / (stem + "_model_info.json"),
               dict(selected=default, skipped=skipped, fixed_threshold=task_threshold(task), stable_header=stem + ".h",
                    api=stem + "_predict(&m_feat_core)", budgets_bytes=budgets,
                    rationale="Best grouped-CV replay score inside the primary model-data budget, skipping "
                              "candidates that fail a CV fold; the held-out people are reported, never used to choose"))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    summary = dict(status="success", task=task, model_header=stem + ".h", class_one=definition["class_one"],
                   class_zero=definition["class_zero"], fixed_threshold=task_threshold(task), threshold_tuned=False,
                   budgets=budgets, lean8=lean8_setting(), selected=default, skipped=skipped, candidates=candidates,
                   baselines=baselines, experiment=EXPERIMENT, tap_target=target_info, feature_selection=fs_report,
                   tap_soft_labels=TAP_SOFT_MIN_SCORE if task == "tap" else None,
                   tap_generalization=(dict(features=TAP_FEATURES, augmentation=TAP_AUG, derived_inputs=len(TF.DERIVED)
                                            if TAP_FEATURES == "diff" else 0, lean_lite_diff=TAP_LITE,
                                            lean_lite_inputs=len(TF.LITE_COLUMNS) if TAP_LITE == "diff" else 0,
                                            labels=TAP_LABELS, observable_ms=TAP_OBS_MS, distill_label_weight=TAP_DISTILL,
                                            finger_weight=TAP_FINGER_WEIGHT, recipe=TAP_RECIPE or None,
                                            num_leaves=list(NUM_LEAVES)) if task == "tap" else None),
                   data_identity={k: identity[k] for k in ("sha256", "recordings", "events")},
                   acceptance=acceptance(task, candidates, baselines),
                   trials=len(done), pruned=len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                   dataset=dict(train_rows=int(len(data["y"])), replay_rows=int(len(rpl["X"])),
                                recordings=int(len(rpl["recs"])), subjects=len(set(data["group"])), counts=info),
                   cv_objective=definition["metric"],
                   replay_rules={k: getattr(R, k) for k in dir(R) if k.startswith(("TAP_", "LP_"))},
                   note="No production firmware overwrite; silver labels; LP class 1 is RELEASE.")
    write_json(out / "data_identity.json", identity)        # every recording and label file by content hash
    write_json(summary_file, summary)
    notice("UNIFIED_TRAIN_END", json.dumps({"task": task, "header": str(out / "models" / (stem + ".h")),
                                            "bytes": default["static_bytes"]}))
    return summary


def main():
    t0 = time.monotonic()
    faulthandler.enable(all_threads=True)       # a native crash prints every thread's Python stack
    notice("UNIFIED_TRAINER_START", "cpu_threads", CPU_THREADS, "of", previous_tap.cpu_count(),
           "fold_workers", FOLD_WORKERS, "x", FOLD_THREADS, "threads", "lean8_inputs", LEAN8_MODE, "tap_threshold", FIXED_THRESHOLD,
           "lp_release_threshold", LP_RELEASE_THRESHOLD,
           "budget_kb", MODEL_BUDGET_KB, "alt_budget_kb", ALT_BUDGET_KB, "trials", TRIALS_PER_MODEL,
           "tasks", ",".join(TASKS_TO_RUN), "tap_features", TAP_FEATURES, "tap_aug", TAP_AUG, "output", OUTPUT.name)
    generator = lite if USE_LITE else lean8      # also when no Lean inputs are used: the frame gates need it
    lean8_manifest = generator.ensure_datasets([(RAW_SOURCE_TAP, SOURCE_TAP), (RAW_SOURCE_LP, SOURCE_LP)],
                                               workers=CPU_THREADS)
    files_t = ensure_labels(RAW_SOURCE_TAP, LABEL_TAP)
    files_l = ensure_labels(RAW_SOURCE_LP, LABEL_LP)
    done_ = {task: train_task(task, files_t, files_l) for task in ("tap", "lp") if task in TASKS_TO_RUN}
    a, b = done_.get("tap"), done_.get("lp")
    write_json(OUTPUT / "UNIFIED_SUMMARY.json",
               dict(status="success", tap=a, lp=b, elapsed_s=round(time.monotonic() - t0, 1), baseline_unchanged=True,
                    tasks=TASKS_TO_RUN,
                    lean8=dict(enabled=LEAN8_TAG, train_min_frames=TRAIN_MIN_FRAMES, eval_min_frames=EVAL_MIN_FRAMES,
                               steady_frames=LEAN8_STEADY_FRAMES,
                               firmware_parity=(lean8_manifest or {}).get("firmware_parity"),
                               generator_sha256=(lean8_manifest or {}).get("generator_sha256"))))
    comp = "".join((OUTPUT / t / "COMPARISON.md").read_text() + "\n" for t in ("tap", "lp") if t in TASKS_TO_RUN)
    (OUTPUT / "COMPARISON.md").write_text(comp)
    notice("UNIFIED_BOTH_READY", *[str(OUTPUT / t / "models" / (h + ".h"))
                                   for t, h in (("tap", "TAP_SWIPE"), ("lp", "LONG_PRESS")) if t in TASKS_TO_RUN])


if __name__ == "__main__":
    main()
