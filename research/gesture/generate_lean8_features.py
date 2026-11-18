#!/usr/bin/env python3
"""Generate firmware-equivalent Lean-8 EMG features for unified gesture training.

Source:
  DATA/DATA_TAP
  DATA/DATA_LONG_PRESS

Output:
  DATA/DATA_TAP_LEAN8
  DATA/DATA_LONG_PRESS_LEAN8

The raw recordings are never modified. Each output CSV preserves the original
rows/columns and appends the 24 p_core->cload_lean_8 fields plus
lean8_features_ready.

Lean-8 runs on the 50 Hz EMG feature stream. The 3 s and 10 s histories are
therefore 150 and 500 feature frames respectively.

The feature formulas intentionally mirror the firmware reference:
  lw     = log10(1 + wl)
  Pm     = 10*log10(sum(10^(emg_psd/10000)-1))
  r01    = Pm_zone0 - Pm_zone1
  r32    = Pm_zone3 - Pm_zone2
  sd3    = trailing population std(lw, 150)
  min10  = trailing minimum(lw, 500)
  rise10 = lw - min10
  max10  = trailing maximum(lw, 500)

The firmware's history is causal: the current frame is included in the
trailing window. Warm-up is not used for model training/replay; ready becomes
true once the 500-frame history exists.

Run:
  python -u research/gesture/generate_lean8_features.py
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_TAP = ROOT / "DATA" / "DATA_TAP"
RAW_LP = ROOT / "DATA" / "DATA_LONG_PRESS"
OUT_TAP = ROOT / "DATA" / "DATA_TAP_LEAN8"
OUT_LP = ROOT / "DATA" / "DATA_LONG_PRESS_LEAN8"

EMG_HZ = 50
HISTORY_3S = 150
HISTORY_10S = 500
PSD_SCALE = 10000.0
READY_COLUMN = "lean8_features_ready"

FEATURES = ("lw", "Pm", "r01", "r32", "sd3", "min10", "rise10", "max10")
LEAN8_COLUMNS = tuple(
    f"p_core->cload_lean_8.{feature}[CH{ch}]"
    for feature in FEATURES for ch in (1, 2, 3)
)

# The firmware reference header is supplied by the firmware repository/build.
# The ML repo does not contain that private header, so the generated dataset
# records this as a provenance field instead of pretending the host is the ABI.
REFERENCE_HEADER = ROOT / "research" / "gesture" / "_lean8_reference" / "cload_lean_8.h"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emg_power_to_linear(x):
    # Same conversion used elsewhere in this repository for the firmware
    # 10000*log10(1+power) representation.
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(np.power(10.0, x / PSD_SCALE) - 1.0, 0.0)


def linear_to_db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-12))


def trailing_std(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    s = 0.0
    ss = 0.0
    for i, value in enumerate(x):
        s += value
        ss += value * value
        if i >= n:
            old = x[i - n]
            s -= old
            ss -= old * old
            count = n
        else:
            count = i + 1
        # Firmware warm-up values are irrelevant to the model. Keep the
        # mathematically causal partial-window value here for diagnostics.
        mean = s / count
        out[i] = math.sqrt(max(ss / count - mean * mean, 0.0))
    return out


def trailing_min(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    from collections import deque
    q = deque()
    for i, value in enumerate(x):
        while q and q[0] <= i - n:
            q.popleft()
        while q and x[q[-1]] >= value:
            q.pop()
        q.append(i)
        out[i] = x[q[0]]
    return out


def trailing_max(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    from collections import deque
    q = deque()
    for i, value in enumerate(x):
        while q and q[0] <= i - n:
            q.popleft()
        while q and x[q[-1]] <= value:
            q.pop()
        q.append(i)
        out[i] = x[q[0]]
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Timestamp"] + [
        f"p_core->wl[CH{ch}]" for ch in (1, 2, 3)
    ] + [
        f"p_core->emg_psd[0][CH{ch}][{z}]"
        for ch in (1, 2, 3) for z in range(5)
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Lean-8 source fields: {missing[:10]}")

    n = len(df)
    result = {}
    ready = np.arange(n) >= (HISTORY_10S - 1)

    for ch in (1, 2, 3):
        wl = pd.to_numeric(df[f"p_core->wl[CH{ch}]"], errors="raise").to_numpy(np.float64)
        lw = np.log10(np.maximum(wl, 0.0) + 1.0)

        psd = np.stack([
            pd.to_numeric(df[f"p_core->emg_psd[0][CH{ch}][{z}]"], errors="raise").to_numpy(np.float64)
            for z in range(5)
        ], axis=1)
        power = emg_power_to_linear(psd)
        zone_db = linear_to_db(power)

        # Pm is total EMG PSD power in dB.
        pm = linear_to_db(power.sum(axis=1))
        r01 = zone_db[:, 0] - zone_db[:, 1]
        r32 = zone_db[:, 3] - zone_db[:, 2]

        sd3 = trailing_std(lw, HISTORY_3S)
        min10 = trailing_min(lw, HISTORY_10S)
        max10 = trailing_max(lw, HISTORY_10S)

        result[f"p_core->cload_lean_8.lw[CH{ch}]"] = lw
        result[f"p_core->cload_lean_8.Pm[CH{ch}]"] = pm
        result[f"p_core->cload_lean_8.r01[CH{ch}]"] = r01
        result[f"p_core->cload_lean_8.r32[CH{ch}]"] = r32
        result[f"p_core->cload_lean_8.sd3[CH{ch}]"] = sd3
        result[f"p_core->cload_lean_8.min10[CH{ch}]"] = min10
        result[f"p_core->cload_lean_8.rise10[CH{ch}]"] = lw - min10
        result[f"p_core->cload_lean_8.max10[CH{ch}]"] = max10

    out = pd.DataFrame(result, index=df.index)
    out[READY_COLUMN] = ready.astype(np.uint8)
    return out


def generate_tree(raw_root: Path, out_root: Path) -> dict:
    files = sorted(raw_root.glob("*/*.csv.gz"))
    if not files:
        raise ValueError(f"No recordings under {raw_root}")

    count = 0
    for source in files:
        rel = source.relative_to(raw_root)
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(source, "rt", newline="") as f:
            df = pd.read_csv(f, low_memory=False)

        features = compute(df)
        if len(features) != len(df):
            raise AssertionError(f"Row mismatch for {source}")

        augmented = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
        if len(set(augmented.columns)) != len(augmented.columns):
            raise AssertionError(f"Duplicate columns after Lean-8 augmentation: {source}")
        if not np.isfinite(features[LEAN8_COLUMNS].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Non-finite Lean-8 feature generated for {source}")

        with gzip.open(dest, "wt", newline="") as f:
            augmented.to_csv(f, index=False)

        count += 1
        if count % 100 == 0:
            print(f"LEAN8_GENERATE {raw_root.name} {count}/{len(files)}", flush=True)

    return {
        "source": str(raw_root.relative_to(ROOT)),
        "output": str(out_root.relative_to(ROOT)),
        "files": count,
        "feature_count": len(LEAN8_COLUMNS),
        "emg_hz": EMG_HZ,
        "history_3s_frames": HISTORY_3S,
        "history_10s_frames": HISTORY_10S,
        "ready_after_frame": HISTORY_10S - 1,
        "ready_column": READY_COLUMN,
        "reference_header": str(REFERENCE_HEADER.relative_to(ROOT)),
    }


def ensure_datasets() -> dict:
    # Derived data is disposable. Never touch the raw directories.
    OUT_TAP.mkdir(parents=True, exist_ok=True)
    OUT_LP.mkdir(parents=True, exist_ok=True)

    summaries = [
        generate_tree(RAW_TAP, OUT_TAP),
        generate_tree(RAW_LP, OUT_LP),
    ]
    manifest = {
        "version": 1,
        "source_contract": "raw DATA/DATA_TAP and DATA/DATA_LONG_PRESS only",
        "features": list(LEAN8_COLUMNS),
        "summaries": summaries,
    }
    (OUT_TAP.parent / "lean8_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(ensure_datasets(), indent=2))
