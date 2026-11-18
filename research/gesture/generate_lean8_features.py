#!/usr/bin/env python3
"""Firmware-exact offline Lean-8 features for unified gesture training.

    python -u research/gesture/generate_lean8_features.py

Source (never modified):  DATA/DATA_TAP, DATA/DATA_LONG_PRESS
Derived (git-ignored):    DATA/DATA_TAP_LEAN8, DATA/DATA_LONG_PRESS_LEAN8

Each derived .csv.gz keeps every original line byte-for-byte and appends 27 columns:

    the 24 firmware fields  p_core->cload_lean_8.{lw,Pm,r01,r32,sd3,min10,rise10,max10}[CH1..CH3]
    lean8_frames            number of lean_8_features_handler() calls so far in this recording
    lean8_new_frame         1 on the row where a new EMG frame was handed to the handler
    lean8_features_ready    lean_8_features_ready() of the firmware (lean8_frames >= 500)

What "firmware-exact" means here (reference: the firmware core_type_def.h, Lean-8 section):

* The handler runs ONCE PER NEW EMG FEATURE FRAME, not once per logged row. Logged rows arrive
  at ~140-210 rows/s (every new p_core state, mostly ACC updates); EMG frames at ~46.4 per s.
  A new frame is a row where emg_psd changes. In ~11% of frames the logger captured the struct
  between the PSD and the WL update (PSD changes on row i, WL on row i+1); the frame is then
  complete on row i+1, which is where the handler sees both. Between frames the values hold.
* History is counted in handler calls: sd3 over 150 frames, min10/max10 over 500 frames
  (LEAN8_FEATURE_UPDATE_HZ = 50). At the measured 46.4 frames/s that is 3.2 s and 10.8 s.
* Q12 log-WL history (uint16), exact integer sd3 with nearest-edge padding during warm-up,
  partial-window min/max during warm-up, rise10 = max(lw - min10, 0).
* Pm/r01/r32 from emg_psd_MEAN, power from the 32-entry LUT + expm1 polynomial, zone power
  clamped at 1e-4, total power clamped at 1e-4, log10 by the firmware's IEEE-754 polynomial.
  All arithmetic in float32 in the firmware's operation order.
* One recording = one sensor stream after lean_8_features_reset().

If the firmware header is available locally (MUDRA_LEAN8_FIRMWARE_HEADER, or any .h in
research/gesture/_lean8_reference/ that defines lean_8_features_handler), the real C code is
compiled on the host and compared with this emulator on sample recordings; a mismatch aborts
generation. The header is kept out of Git on purpose (the ML repository is public).

Generation is incremental: a file is regenerated only if its source or this script changed.
"""
from __future__ import annotations

import ctypes
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_TAP = ROOT / "DATA" / "DATA_TAP"
RAW_LP = ROOT / "DATA" / "DATA_LONG_PRESS"
OUT_TAP = ROOT / "DATA" / "DATA_TAP_LEAN8"
OUT_LP = ROOT / "DATA" / "DATA_LONG_PRESS_LEAN8"
BUILD_DIR = ROOT / "research" / "gesture" / "_lean8_build"
REFERENCE_DIR = ROOT / "research" / "gesture" / "_lean8_reference"
REFERENCE_HEADER = Path(os.environ.get("MUDRA_LEAN8_FIRMWARE_HEADER", "") or REFERENCE_DIR / "core_type_def.h")

# ---- firmware constants (core_type_def.h, Lean-8 section)
UPDATE_HZ = 50                      # LEAN8_FEATURE_UPDATE_HZ default
HISTORY_3S = 3 * UPDATE_HZ          # 150 handler calls
HISTORY_10S = 10 * UPDATE_HZ        # 500 handler calls
EMG_HZ = UPDATE_HZ                  # kept for older callers; the measured frame rate is ~46.4 Hz
Q12 = np.float32(4096.0)
Q12_INV = np.float32(1.0 / 4096.0)
PSD_LUT = np.array([
    0.0000000000e+00, 6.0250723888e-01, 1.5680294507e+00, 3.1152857844e+00,
    5.5947752595e+00, 9.5681750921e+00, 1.5935577087e+01, 2.6139384876e+01,
    4.2491060723e+01, 6.8694739636e+01, 1.1068632478e+02, 1.7797814394e+02,
    2.8581377127e+02, 4.5862114467e+02, 7.3554621147e+02, 1.1793206357e+03,
    1.8904723628e+03, 3.0300981536e+03, 4.8563567329e+03, 7.7829493263e+03,
    1.2472835142e+04, 1.9988411112e+04, 3.2032176009e+04, 5.1332396438e+04,
    8.2261139388e+04, 1.3182467386e+05, 2.1125059662e+05, 3.3853121282e+05,
    5.4249932163e+05, 8.6935969251e+05, 1.3931558029e+06, 2.2325428616e+06], dtype=np.float32)

FEATURES = ("lw", "Pm", "r01", "r32", "sd3", "min10", "rise10", "max10")
LEAN8_COLUMNS = tuple(f"p_core->cload_lean_8.{feature}[CH{ch}]" for feature in FEATURES for ch in (1, 2, 3))
FRAMES_COLUMN = "lean8_frames"
NEW_FRAME_COLUMN = "lean8_new_frame"
READY_COLUMN = "lean8_features_ready"
EXTRA_COLUMNS = LEAN8_COLUMNS + (FRAMES_COLUMN, NEW_FRAME_COLUMN, READY_COLUMN)

WL = [f"p_core->wl[CH{c}]" for c in (1, 2, 3)]
PSD_NOW = [f"p_core->emg_psd[0][CH{c}][{z}]" for c in (1, 2, 3) for z in range(5)]
PSD_MEAN = [f"p_core->emg_psd_mean[CH{c}][{z}]" for c in (1, 2, 3) for z in range(5)]
SOURCE_COLUMNS = WL + PSD_NOW + PSD_MEAN
PARITY_FILES = int(os.environ.get("MUDRA_LEAN8_PARITY_FILES", 24))
PARITY_TOLERANCE = 1e-4             # absolute, any feature; host float32 has no FMA, so 0 is expected
WORKERS = max(1, int((os.cpu_count() or 1) * 0.8))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


GENERATOR_SHA = sha256(Path(__file__))


def firmware_header() -> Path | None:
    """The firmware header with lean_8_features_handler, if present locally."""
    cands = [REFERENCE_HEADER] + (sorted(REFERENCE_DIR.glob("*.h")) if REFERENCE_DIR.is_dir() else [])
    for p in cands:
        if p.is_file() and "lean_8_features_handler" in p.read_text(errors="replace"):
            return p
    return None


# ============================================================================ firmware math (float32)
F = np.float32


def fw_log10_positive(x: np.ndarray) -> np.ndarray:
    """lean8_log10_positive: exponent + atanh series on the mantissa, float32, C operation order."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    exponent = ((bits >> np.uint32(23)) & np.uint32(255)).astype(np.int32) - 127
    m = ((bits & np.uint32(0x007FFFFF)) | np.uint32(0x3F800000)).view(np.float32)
    y = (m - F(1.0)) / (m + F(1.0))
    y2 = y * y
    s = y * (F(1.0) + y2 * (F(0.3333333333333) + y2 * (F(0.2) + y2 * (F(0.1428571428571) + y2 * F(0.1111111111111)))))
    return exponent.astype(np.float32) * F(0.301029995664) + F(0.868588963807) * s


def fw_psd_linear(code: np.ndarray) -> np.ndarray:
    """lean8_psd_linear: 10^(code/10000) - 1 via LUT anchor and a 7th-order expm1 polynomial."""
    c = np.asarray(code).astype(np.int64).astype(np.uint16).astype(np.uint32)
    i = c >> np.uint32(11)
    x = (c & np.uint32(2047)).astype(np.float32) * F(0.00023025850929940458)
    em1 = x * (F(1.0) + x * (F(0.5) + x * (F(0.1666666666666667) + x * (F(0.0416666666666667) + x * (
        F(0.0083333333333333) + x * (F(0.0013888888888889) + x * F(0.0001984126984127)))))))
    anchor = PSD_LUT[i]
    return anchor + (F(1.0) + anchor) * em1


def _trailing(q: np.ndarray, n: int) -> np.ndarray:
    """(frames, n) windows ending at each frame, padded at the start with the first value."""
    pad = np.concatenate([np.full(n - 1, q[0], q.dtype), q])
    return np.lib.stride_tricks.sliding_window_view(pad, n)


def lean8_per_frame(wl: np.ndarray, psd_mean: np.ndarray) -> np.ndarray:
    """Emulate lean_8_features_handler for one recording after a reset.

    wl: (frames, 3) raw WL; psd_mean: (frames, 3, 5) emg_psd_mean codes.
    Returns (frames, 24) float32 in LEAN8_COLUMNS order (feature-major, CH1..CH3)."""
    n = len(wl)
    out = np.zeros((n, 8, 3), np.float32)
    if not n:
        return out.reshape(n, 24)
    inv_var = F(1.0) / (F(HISTORY_3S) * F(HISTORY_3S) * Q12 * Q12)
    for ch in range(3):
        lw = fw_log10_positive(wl[:, ch].astype(np.float32) + F(1.0))
        q = np.minimum((lw * Q12 + F(0.5)).astype(np.uint32), 65535).astype(np.int64)
        # min10 / max10: window of min(k+1, 500) frames == nearest padding with the first value
        w10 = _trailing(q, HISTORY_10S)
        min10 = w10.min(axis=1).astype(np.float32) * Q12_INV
        max10 = w10.max(axis=1).astype(np.float32) * Q12_INV
        # sd3: exact integer moments over 150 frames, nearest-edge padding with the first value
        w3 = _trailing(q, HISTORY_3S)
        s1 = w3.sum(axis=1)
        s2 = (w3 * w3).sum(axis=1)
        numerator = s2 * HISTORY_3S - s1 * s1
        variance = numerator.astype(np.float32) * inv_var
        sd3 = np.sqrt(np.maximum(variance, F(0.0))).astype(np.float32)
        rise10 = np.where(lw > min10, lw - min10, F(0.0)).astype(np.float32)
        # Pm, r01, r32 from the PSD mean spectrum
        x = fw_psd_linear(psd_mean[:, ch, :])                    # (frames, 5) float32
        power = np.where(x > F(1e-4), x, F(1e-4))
        total = np.zeros(n, np.float32)
        for z in range(5):                                       # firmware accumulation order
            total = total + x[:, z]
        total = np.where(total < F(1e-4), F(1e-4), total)
        pm = F(10.0) * fw_log10_positive(total)
        r01 = F(10.0) * fw_log10_positive(power[:, 0] / power[:, 1])
        r32 = F(10.0) * fw_log10_positive(power[:, 3] / power[:, 2])
        for k, v in enumerate((lw, pm, r01, r32, sd3, min10, rise10, max10)):
            out[:, k, ch] = v
    return out.reshape(n, 24)


# ============================================================================ frames in a logged recording
def frame_rows(psd_now: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Row index where each EMG frame is complete (handler call), strictly increasing.

    New frame = emg_psd changed. If WL did not change on that row but changes alone on the
    next row, the logger split the frame and it completes on the next row."""
    n = len(psd_now)
    if not n:
        return np.zeros(0, np.int64)
    cp = np.r_[True, (np.diff(psd_now, axis=0) != 0).any(axis=1)]
    cw = np.r_[True, (np.diff(wl, axis=0) != 0).any(axis=1)]
    idx = np.flatnonzero(cp)
    nxt = np.minimum(idx + 1, n - 1)
    split = (~cw[idx]) & (idx + 1 < n) & cw[nxt] & ~cp[nxt]
    return np.where(split, idx + 1, idx)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row Lean-8 values as the firmware's m_feat_core holds them, for one recording."""
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Lean-8 source fields: {missing[:6]}")
    wl = df[WL].to_numpy(np.float64)
    psd_now = df[PSD_NOW].to_numpy(np.int64)
    psd_mean = df[PSD_MEAN].to_numpy(np.int64).reshape(-1, 3, 5)
    rows = frame_rows(psd_now, wl)
    feats = lean8_per_frame(wl[rows], psd_mean[rows])
    n = len(df)
    frames = np.zeros(n, np.int64)
    frames[rows] = 1
    frames = np.cumsum(frames)
    per_row = np.zeros((n, 24), np.float32)
    has = frames > 0
    per_row[has] = feats[frames[has] - 1]
    out = pd.DataFrame(per_row, columns=list(LEAN8_COLUMNS), index=df.index)
    out[FRAMES_COLUMN] = frames
    out[NEW_FRAME_COLUMN] = np.isin(np.arange(n), rows).astype(np.uint8)
    out[READY_COLUMN] = (frames >= HISTORY_10S).astype(np.uint8)
    return out


# ============================================================================ derived files
def _text_block(features: pd.DataFrame) -> list[str]:
    cols = []
    for c in EXTRA_COLUMNS:
        v = features[c].to_numpy()
        cols.append(np.char.mod("%.9g", v) if v.dtype.kind == "f" else v.astype(str))
    return [",".join(r) for r in zip(*cols)]


def generate_file(source: Path, dest: Path) -> dict:
    """Append the Lean-8 columns to every original line (original bytes unchanged)."""
    with gzip.open(source, "rb") as f:
        raw = f.read()
    df = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in set(SOURCE_COLUMNS), low_memory=False)
    feats = compute(df)
    if not np.isfinite(feats[list(LEAN8_COLUMNS)].to_numpy(np.float64)).all():
        raise ValueError(f"Non-finite Lean-8 value: {source}")
    text = raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text[:4096] else "\n"
    lines = text.split(newline)
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != len(df) + 1:
        raise ValueError(f"{source}: {len(lines) - 1} text lines but {len(df)} CSV rows (quoted newlines?)")
    header = lines[0].split(",")
    if set(header) & set(EXTRA_COLUMNS):
        raise ValueError(f"{source}: already contains Lean-8 columns")
    block = _text_block(feats)
    body = [lines[0] + "," + ",".join(EXTRA_COLUMNS)] + [a + "," + b for a, b in zip(lines[1:], block)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        f.write((newline.join(body) + newline).encode("utf-8"))
    os.replace(tmp, dest)
    fr = feats[FRAMES_COLUMN].to_numpy()
    return dict(rows=len(df), frames=int(fr[-1]) if len(fr) else 0, ready_rows=int((fr >= HISTORY_10S).sum()))


def _job(args):
    source, dest = args
    return str(source), generate_file(Path(source), Path(dest))


def generate_tree(raw_root: Path, out_root: Path, workers: int = WORKERS) -> dict:
    files = sorted(raw_root.glob("*/*.csv.gz"))
    if not files:
        raise ValueError(f"No recordings under {raw_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_file = out_root / "_lean8_files.json"
    try:
        known = json.loads(manifest_file.read_text())
    except (OSError, ValueError):
        known = {}
    if known.get("generator_sha") != GENERATOR_SHA:
        known = {"generator_sha": GENERATOR_SHA, "files": {}}
    todo, entries = [], known["files"]
    for source in files:
        rel = str(source.relative_to(raw_root))
        st = source.stat()
        stamp = [st.st_size, st.st_mtime_ns]
        dest = out_root / rel
        if entries.get(rel, {}).get("source") == stamp and dest.is_file():
            continue
        todo.append((source, dest, rel, stamp))
    done = 0
    if todo:
        print(f"LEAN8_GENERATE {raw_root.name}: {len(todo)} of {len(files)} files to (re)generate", flush=True)
        with ProcessPoolExecutor(max(1, workers)) as ex:
            for (source, dest, rel, stamp), (_, stats) in zip(todo, ex.map(_job, [(str(a), str(b)) for a, b, _, _ in todo],
                                                                           chunksize=8)):
                entries[rel] = dict(source=stamp, **stats)
                done += 1
                if done % 250 == 0:
                    print(f"LEAN8_GENERATE {raw_root.name} {done}/{len(todo)}", flush=True)
                    manifest_file.write_text(json.dumps(known))
        manifest_file.write_text(json.dumps(known))
    live = [entries[str(p.relative_to(raw_root))] for p in files]
    return {
        "source": str(raw_root), "output": str(out_root), "files": len(files), "regenerated": len(todo),
        "rows": int(sum(e["rows"] for e in live)), "frames": int(sum(e["frames"] for e in live)),
        "rows_ready_500": int(sum(e["ready_rows"] for e in live)),
        "recordings_reaching_500_frames": int(sum(e["frames"] >= HISTORY_10S for e in live)),
    }


# ============================================================================ firmware C parity
ARM_MATH_STUB = """#pragma once
#include <math.h>
#include <stdint.h>
typedef float float32_t;
typedef enum { ARM_MATH_SUCCESS = 0, ARM_MATH_ARGUMENT_ERROR = -1 } arm_status;
static inline arm_status arm_sqrt_f32(float32_t in, float32_t *out)
{ if (in >= 0.0f) { *out = sqrtf(in); return ARM_MATH_SUCCESS; } *out = 0.0f; return ARM_MATH_ARGUMENT_ERROR; }
"""
PSD_TABLE_STUB = """#pragma once
#include <stdint.h>
#define NUM_ACC_ZONES 6
#define NUM_EMG_ZONES 5
#define ACC_PSD_HISTORY_SIZE 10
#define EMG_PSD_HISTORY_SIZE 15
typedef uint16_t acc_psd_type;
typedef uint16_t emg_psd_type;
typedef uint32_t emg_wl_type;
typedef uint32_t emg_rms_type;
"""
DRIVER = """#include "core_type_def.h"
#include <string.h>
static t_core core;
int lean8_run(int n, const uint32_t *wl, const uint16_t *psd_mean, float *out, uint8_t *ready)
{
    lean_8_features_reset();
    for (int f = 0; f < n; ++f) {
        for (int ch = 0; ch < 3; ++ch) {
            core.wl[ch] = (emg_wl_type)wl[f * 3 + ch];
            for (int z = 0; z < 5; ++z) core.emg_psd_mean[ch][z] = (emg_psd_type)psd_mean[(f * 3 + ch) * 5 + z];
        }
        lean_8_features_handler(&core);
        const t_cload_lean_8 *v = &core.cload_lean_8;
        const float *src[8] = { v->lw, v->Pm, v->r01, v->r32, v->sd3, v->min10, v->rise10, v->max10 };
        for (int k = 0; k < 8; ++k) for (int ch = 0; ch < 3; ++ch) out[f * 24 + k * 3 + ch] = src[k][ch];
        ready[f] = lean_8_features_ready();
    }
    return (int)sizeof(t_cload_lean_8);
}
"""


class FirmwareLean8:
    """The real firmware Lean-8 C code compiled for the host."""

    def __init__(self, header: Path, workdir: Path = BUILD_DIR):
        cc = shutil.which("cc") or shutil.which("gcc")
        if not cc:
            raise RuntimeError("No C compiler for the firmware parity build")
        workdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(header, workdir / "core_type_def.h")
        (workdir / "arm_math.h").write_text(ARM_MATH_STUB)
        (workdir / "psd_configuration_table.h").write_text(PSD_TABLE_STUB)
        (workdir / "lean8_driver.c").write_text(DRIVER)
        lib = workdir / "liblean8_fw.so"
        cmd = [cc, "-std=gnu11", "-O2", "-shared", "-fPIC", "-I", str(workdir), str(workdir / "lean8_driver.c"),
               "-o", str(lib), "-lm"]
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
        if p.returncode:
            raise RuntimeError("Firmware Lean-8 host build failed:\n" + p.stderr[-4000:])
        self.lib = ctypes.CDLL(str(lib))
        self.lib.lean8_run.restype = ctypes.c_int
        self.header_sha = sha256(header)

    def run(self, wl: np.ndarray, psd_mean: np.ndarray):
        n = len(wl)
        wl_ = np.ascontiguousarray(wl, dtype=np.uint32)
        psd_ = np.ascontiguousarray(psd_mean.reshape(n, 15), dtype=np.uint16)
        out = np.zeros((n, 24), np.float32); ready = np.zeros(n, np.uint8)
        size = self.lib.lean8_run(ctypes.c_int(n), wl_.ctypes.data_as(ctypes.c_void_p),
                                  psd_.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
                                  ready.ctypes.data_as(ctypes.c_void_p))
        if size != 96:
            raise AssertionError(f"t_cload_lean_8 is {size} bytes, expected 96")
        return out, ready.astype(bool)


def parity_check(files: list[Path], header: Path | None = None) -> dict:
    """Emulator vs. the firmware's own C code on real recordings (frame by frame)."""
    header = header or firmware_header()
    if header is None:
        return dict(status="not run", reason="firmware header not found (set MUDRA_LEAN8_FIRMWARE_HEADER)")
    fw = FirmwareLean8(header)
    worst = np.zeros(24); frames = 0; ready_mismatch = 0
    for f in files:
        df = pd.read_csv(f, usecols=lambda c: c in set(SOURCE_COLUMNS), low_memory=False)
        wl = df[WL].to_numpy(np.float64)
        rows = frame_rows(df[PSD_NOW].to_numpy(np.int64), wl)
        pm = df[PSD_MEAN].to_numpy(np.int64).reshape(-1, 3, 5)[rows]
        ours = lean8_per_frame(wl[rows], pm)
        theirs, ready = fw.run(wl[rows], pm)
        worst = np.maximum(worst, np.abs(ours.astype(np.float64) - theirs).max(axis=0))
        ready_mismatch += int(((np.arange(len(rows)) + 1 >= HISTORY_10S) != ready).sum())
        frames += len(rows)
    ok = bool(worst.max() <= PARITY_TOLERANCE and ready_mismatch == 0)
    return dict(status="passed" if ok else "FAILED", header=str(header), header_sha256=fw.header_sha,
                files=len(files), frames=frames, max_abs_diff=float(worst.max()),
                max_abs_diff_by_feature={c: float(v) for c, v in zip(LEAN8_COLUMNS, worst)},
                ready_mismatch=ready_mismatch, tolerance=PARITY_TOLERANCE)


def ensure_datasets(pairs=None, workers: int = WORKERS, parity_files: int = PARITY_FILES) -> dict:
    """Generate/refresh derived trees (raw roots are only read) and check firmware parity."""
    pairs = pairs or [(RAW_TAP, OUT_TAP), (RAW_LP, OUT_LP)]
    summaries = [generate_tree(Path(a), Path(b), workers) for a, b in pairs]
    sample = []
    for raw, _ in pairs:
        fs = sorted(Path(raw).glob("*/*.csv.gz"))
        if fs:
            step = max(1, len(fs) // max(1, parity_files // len(pairs)))
            sample += fs[::step][: max(1, parity_files // len(pairs))]
    parity = parity_check(sample) if parity_files else dict(status="not run", reason="disabled")
    print("LEAN8_FIRMWARE_PARITY", json.dumps({k: v for k, v in parity.items() if k != "max_abs_diff_by_feature"}),
          flush=True)
    if parity["status"] == "FAILED":
        raise RuntimeError("Offline Lean-8 does not match the firmware C code: " + json.dumps(parity, indent=1))
    manifest = {
        "version": 2, "generator_sha256": GENERATOR_SHA,
        "source_contract": "raw recordings are read only; derived files = original lines + Lean-8 columns",
        "features": list(LEAN8_COLUMNS), "extra_columns": [FRAMES_COLUMN, NEW_FRAME_COLUMN, READY_COLUMN],
        "handler_rate": "one call per new EMG frame (emg_psd change, split-frame aware)",
        "history_frames": dict(sd3=HISTORY_3S, min_max=HISTORY_10S, ready_after=HISTORY_10S),
        "summaries": summaries, "firmware_parity": parity,
    }
    (Path(pairs[0][1]).parent / "lean8_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = ensure_datasets()
    print(json.dumps({k: m[k] for k in ("summaries", "firmware_parity")}, indent=2))
    sys.exit(0)
