#!/usr/bin/env python3
"""Bit-exact offline Lean-lite features (research/gesture/firmware/lean_lite.h) for gesture training.

    python -u research/gesture/generate_lean_lite_features.py

Source (never modified):  DATA/DATA_TAP, DATA/DATA_LONG_PRESS
Derived (git-ignored):    DATA/DATA_TAP_LITE, DATA/DATA_LONG_PRESS_LITE

Each derived file keeps every original line byte-for-byte and appends 18 columns:

    the 15 firmware fields  p_core->cload_lite.{min10,max10,rise10,dev3,d01}[CH1..CH3]   (int16)
    lean8_frames            number of handler calls so far in this recording (same counter as Lean-8)
    lean8_new_frame         1 on the row where a new EMG frame was handed to the handler
    lean_lite_ready         lean_lite_ready() of the firmware (frames >= 500)

Lean-lite replaces the float Lean-8 handler with integer arithmetic (see the C header for the definitions).
Frames are detected exactly as for Lean-8 (generate_lean8_features.frame_rows), so frame counts, the
trainer's frame gates, rows, labels and the held-out split are identical: Lean-8, Lean-lite and no Lean
at all are clean A/B arms.

The C header is part of this repository, so the parity check always runs: the header is compiled for the
host (-DLEAN_LITE_HOST, plain C instead of the CMSIS intrinsics) and must equal this emulator exactly on
real recordings and on synthetic edge cases (zero/huge WL, saturating PSD differences). A mismatch aborts.

Generation is incremental: a file is regenerated only if its source or the algorithm changed.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_lean8_features as lean8                  # frame detection and source column names

ROOT = Path(__file__).resolve().parents[2]
RAW_TAP = ROOT / "DATA" / "DATA_TAP"
RAW_LP = ROOT / "DATA" / "DATA_LONG_PRESS"
OUT_TAP = ROOT / "DATA" / "DATA_TAP_LITE"
OUT_LP = ROOT / "DATA" / "DATA_LONG_PRESS_LITE"
HEADER = Path(__file__).resolve().parent / "firmware" / "lean_lite.h"
BUILD_DIR = ROOT / "research" / "gesture" / "_lean_lite_build"

# ---- constants of lean_lite.h
BLOCK_FRAMES = 46
BLOCKS = 10
RING = BLOCKS - 1
EMA_SHIFT = 7
DEV_SHIFT = 4
READY_FRAMES = 500

FEATURES = ("min10", "max10", "rise10", "dev3", "d01")                  # columns of the derived trees (unchanged)
LITE_COLUMNS = tuple(f"p_core->cload_lite.{feature}[CH{ch}]" for feature in FEATURES for ch in (1, 2, 3))
# Every t_cload_lite field, in struct order (lean_lite.h). range10 = max10 - min10, fall10 = max10 - L and
# xm = L - running mean of L (signed) were added for the gain-invariant "Lean-lite diff" TAP inputs
# (TAP_FOCUS_RESULTS.md, final recipe).
FIELDS = FEATURES + ("range10", "fall10", "xm")
# Lean-lite diff: the TAP model inputs. Only differences of log levels (gain invariant); the absolute levels
# min10 / max10 / d01 are left out (they did not transfer to the device).
DIFF_FEATURES = ("rise10", "fall10", "range10", "dev3", "xm")       # the researched set, in its order
DIFF_COLUMNS = tuple(f"p_core->cload_lite.{feature}[CH{ch}]" for feature in DIFF_FEATURES for ch in (1, 2, 3))
FRAMES_COLUMN = lean8.FRAMES_COLUMN
NEW_FRAME_COLUMN = lean8.NEW_FRAME_COLUMN
READY_COLUMN = "lean_lite_ready"
EXTRA_COLUMNS = LITE_COLUMNS + (FRAMES_COLUMN, NEW_FRAME_COLUMN, READY_COLUMN)
SOURCE_COLUMNS = lean8.SOURCE_COLUMNS
PARITY_FILES = int(os.environ.get("MUDRA_LEAN_LITE_PARITY_FILES", 24))
WORKERS = max(1, int((os.cpu_count() or 1) * 0.8))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================ the firmware arithmetic
def log2_q8(wl: np.ndarray) -> np.ndarray:
    """lean_lite_log2_q8: (floor(log2 v) << 8) | 8 mantissa bits after the leading one, v = wl + 1 (no wrap)."""
    w = np.asarray(wl).astype(np.uint64)
    v = w + (w != np.uint64(0xFFFFFFFF)).astype(np.uint64)
    _, e = np.frexp(v.astype(np.float64))                  # exact for v < 2**53: v = m * 2**e, m in [0.5, 1)
    n = (e - 1).astype(np.uint64)
    frac = ((v << (np.uint64(31) - n)) >> np.uint64(23)) & np.uint64(0xFF)
    return ((n << np.uint64(8)) | frac).astype(np.int64)


def lite_fields_per_frame(wl: np.ndarray, psd_mean: np.ndarray | None) -> np.ndarray:
    """Emulate lean_lite_update() for one recording after a reset: every t_cload_lite field.

    wl: (frames, 3) integer WL; psd_mean: (frames, 3, 5) emg_psd_mean codes (None: d01 = 0).
    Returns (frames, len(FIELDS), 3) int16 (field-major as FIELDS, CH1..CH3)."""
    n = len(wl)
    out = np.zeros((n, len(FIELDS), 3), np.int16)
    if not n:
        return out
    nb = -(-n // BLOCK_FRAMES)
    pad = nb * BLOCK_FRAMES - n
    for ch in range(3):
        x = log2_q8(wl[:, ch])
        # running min / max inside each block (the current block includes this frame)
        xb = np.concatenate([x, np.full(pad, x[-1])]).reshape(nb, BLOCK_FRAMES)
        cur_min = np.minimum.accumulate(xb, axis=1).reshape(-1)[:n]
        cur_max = np.maximum.accumulate(xb, axis=1).reshape(-1)[:n]
        # complete blocks: their min / max; frames of block b see blocks max(0, b-9) .. b-1
        full = n // BLOCK_FRAMES
        bmin = cur_min[BLOCK_FRAMES - 1::BLOCK_FRAMES][:full]
        bmax = cur_max[BLOCK_FRAMES - 1::BLOCK_FRAMES][:full]
        agg_min = np.empty(nb, np.int64); agg_max = np.empty(nb, np.int64)
        for b in range(nb):
            if b == 0:
                agg_min[b] = np.iinfo(np.int64).max; agg_max[b] = np.iinfo(np.int64).min
            else:
                agg_min[b] = bmin[max(0, b - RING):b].min(); agg_max[b] = bmax[max(0, b - RING):b].max()
        blk = np.arange(n) // BLOCK_FRAMES
        lo = np.minimum(cur_min, agg_min[blk])
        hi = np.maximum(cur_max, agg_max[blk])
        # running mean and mean absolute deviation (integer recursion, shifts of non-negative values)
        dev = np.empty(n, np.int64)
        xm = np.empty(n, np.int64)
        xs = x.tolist()
        m_acc = xs[0] << EMA_SHIFT
        d_acc = 0
        for i, xi in enumerate(xs):
            m_acc += xi - (m_acc >> EMA_SHIFT)
            d = xi - (m_acc >> EMA_SHIFT)
            d_acc += (d if d >= 0 else -d) - (d_acc >> EMA_SHIFT)
            dev[i] = d_acc
            xm[i] = d
        d01 = (np.zeros(n, np.int64) if psd_mean is None
               else psd_mean[:, ch, 0].astype(np.int64) - psd_mean[:, ch, 1].astype(np.int64))
        for k, v in enumerate((lo, hi, x - lo, dev >> DEV_SHIFT, d01, hi - lo, hi - x, xm)):
            out[:, k, ch] = np.clip(v, -32768, 32767).astype(np.int16)
    return out


def lite_per_frame(wl: np.ndarray, psd_mean: np.ndarray) -> np.ndarray:
    """(frames, 15) int16 in LITE_COLUMNS order (feature-major, CH1..CH3): the derived-tree columns."""
    return np.ascontiguousarray(lite_fields_per_frame(wl, psd_mean)[:, :len(FEATURES)]).reshape(len(wl), -1)


def lite_diff_per_frame(wl: np.ndarray) -> np.ndarray:
    """(frames, 15) int16 in DIFF_COLUMNS order: the Lean-lite diff TAP inputs (WL only)."""
    idx = [FIELDS.index(f) for f in DIFF_FEATURES]
    return np.ascontiguousarray(lite_fields_per_frame(wl, None)[:, idx]).reshape(len(wl), -1)


def frame_rows_from_counter(frames: np.ndarray) -> np.ndarray:
    """Rows where a new EMG frame reached the handler, from the per-row frame counter of the derived trees
    (FRAMES_COLUMN = cumulative count of lean8.frame_rows): the inverse of compute()'s cumsum."""
    f = np.asarray(frames, np.int64)
    step = np.diff(np.r_[0, f])
    if len(f) and ((step < 0).any() or (step > 1).any()):
        raise ValueError("Frame counter must start at 0 or 1 and grow by at most 1 per row")
    return np.flatnonzero(step == 1)


def lite_diff_rows(wl_rows: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Per-row Lean-lite diff values of one recording (from its first row) as m_feat_core.cload_lite holds them:
    wl_rows (rows, 3) WL of every row, frames the derived trees' frame counter. Rows before the first frame: 0."""
    rows = frame_rows_from_counter(frames)
    per = lite_diff_per_frame(np.asarray(wl_rows)[rows].astype(np.int64))
    out = np.zeros((len(wl_rows), len(DIFF_COLUMNS)), np.int16)
    f = np.asarray(frames, np.int64); has = f > 0
    out[has] = per[f[has] - 1]
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row Lean-lite values as m_feat_core.cload_lite holds them, for one recording."""
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Lean-lite source fields: {missing[:6]}")
    wl = df[lean8.WL].to_numpy(np.float64)
    psd_now = df[lean8.PSD_NOW].to_numpy(np.int64)
    psd_mean = df[lean8.PSD_MEAN].to_numpy(np.int64).reshape(-1, 3, 5)
    if (wl < 0).any() or (wl != np.round(wl)).any() or (psd_mean < 0).any() or (psd_mean > 65535).any():
        raise ValueError("WL must be non-negative integers and PSD codes uint16")
    rows = lean8.frame_rows(psd_now, wl)
    feats = lite_per_frame(wl[rows].astype(np.int64), psd_mean[rows])
    n = len(df)
    frames = np.zeros(n, np.int64)
    frames[rows] = 1
    frames = np.cumsum(frames)
    per_row = np.zeros((n, 15), np.int16)
    has = frames > 0
    per_row[has] = feats[frames[has] - 1]
    out = pd.DataFrame(per_row, columns=list(LITE_COLUMNS), index=df.index)
    out[FRAMES_COLUMN] = frames
    out[NEW_FRAME_COLUMN] = np.isin(np.arange(n), rows).astype(np.uint8)
    out[READY_COLUMN] = (frames >= READY_FRAMES).astype(np.uint8)
    return out


# ============================================================================ derived files
def _text_block(features: pd.DataFrame) -> list[str]:
    cols = [features[c].to_numpy().astype(np.int64).astype(str) for c in EXTRA_COLUMNS]
    return [",".join(r) for r in zip(*cols)]


def generate_file(source: Path, dest: Path) -> dict:
    """Append the Lean-lite columns to every original line (original bytes unchanged)."""
    if source.name.endswith(".gz"):
        with gzip.open(source, "rb") as f:
            raw = f.read()
    else:
        raw = source.read_bytes()
    df = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in set(SOURCE_COLUMNS), low_memory=False)
    feats = compute(df)
    text = raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text[:4096] else "\n"
    lines = text.split(newline)
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != len(df) + 1:
        raise ValueError(f"{source}: {len(lines) - 1} text lines but {len(df)} CSV rows (quoted newlines?)")
    if set(lines[0].split(",")) & set(EXTRA_COLUMNS):
        raise ValueError(f"{source}: already contains Lean-lite columns")
    block = _text_block(feats)
    body = [lines[0] + "," + ",".join(EXTRA_COLUMNS)] + [a + "," + b for a, b in zip(lines[1:], block)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    payload = (newline.join(body) + newline).encode("utf-8")
    if dest.name.endswith(".gz"):
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            f.write(payload)
    else:
        tmp.write_bytes(payload)
    os.replace(tmp, dest)
    fr = feats[FRAMES_COLUMN].to_numpy()
    return dict(rows=len(df), frames=int(fr[-1]) if len(fr) else 0, ready_rows=int((fr >= READY_FRAMES).sum()))


def _job(args):
    source, dest = args
    return str(source), generate_file(Path(source), Path(dest))


def _recordings(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*")
                  if p.is_file() and (p.name.endswith(".csv.gz") or p.name.endswith(".csv")))


def _write_manifest(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def generate_tree(raw_root: Path, out_root: Path, workers: int = WORKERS) -> dict:
    files = _recordings(raw_root)
    if not files:
        raise ValueError(f"No recordings under {raw_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_file = out_root / "_lean_lite_files.json"
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
    if todo:
        print(f"LEAN_LITE_GENERATE {raw_root.name}: {len(todo)} of {len(files)} files to (re)generate", flush=True)
        done = 0
        with ProcessPoolExecutor(max(1, workers)) as ex:
            for (source, dest, rel, stamp), (_, stats) in zip(
                    todo, ex.map(_job, [(str(a), str(b)) for a, b, _, _ in todo], chunksize=8)):
                entries[rel] = dict(source=stamp, **stats)
                done += 1
                if done % 250 == 0:
                    print(f"LEAN_LITE_GENERATE {raw_root.name} {done}/{len(todo)}", flush=True)
                    _write_manifest(manifest_file, known)
    _write_manifest(manifest_file, known)
    print(f"LEAN_LITE_FILES {raw_root.name}: {len(files) - len(todo)} reused, {len(todo)} generated", flush=True)
    live = [entries[str(p.relative_to(raw_root))] for p in files]
    return {
        "source": str(raw_root), "output": str(out_root), "files": len(files), "regenerated": len(todo),
        "rows": int(sum(e["rows"] for e in live)), "frames": int(sum(e["frames"] for e in live)),
        "rows_ready_500": int(sum(e["ready_rows"] for e in live)),
    }


# ============================================================================ algorithm version
def algorithm_sha() -> str:
    """Hash of everything that decides the derived bytes (numerical, frame and format code + constants)."""
    import inspect
    h = hashlib.sha256()
    for fn in (log2_q8, lite_fields_per_frame, lite_per_frame, compute, _text_block, generate_file, lean8.frame_rows):
        h.update(inspect.getsource(fn).encode())
    h.update(json.dumps(dict(block=BLOCK_FRAMES, blocks=BLOCKS, ema=EMA_SHIFT, dev=DEV_SHIFT, ready=READY_FRAMES,
                             columns=EXTRA_COLUMNS, source=SOURCE_COLUMNS)).encode())
    return h.hexdigest()


def diff_algorithm_sha() -> str:
    """Hash of everything that decides the Lean-lite diff TAP inputs (computed by the trainer, no derived files)."""
    import inspect
    h = hashlib.sha256()
    for fn in (log2_q8, lite_fields_per_frame, lite_diff_per_frame, frame_rows_from_counter, lite_diff_rows):
        h.update(inspect.getsource(fn).encode())
    h.update(json.dumps(dict(block=BLOCK_FRAMES, blocks=BLOCKS, ema=EMA_SHIFT, dev=DEV_SHIFT,
                             fields=FIELDS, columns=DIFF_COLUMNS)).encode())
    return h.hexdigest()


GENERATOR_SHA = algorithm_sha()
DIFF_SHA = diff_algorithm_sha()


# ============================================================================ C header parity
DRIVER = """#include "lean_lite.h"
static lean_lite_state_t state;
int lean_lite_run(int n, const uint32_t *wl, const uint16_t *psd_mean, int16_t *out, uint8_t *ready)
{
    lean_lite_reset(&state);
    for (int f = 0; f < n; ++f) {
        uint32_t w[3]; t_cload_lite v;
        for (int ch = 0; ch < 3; ++ch) w[ch] = wl[f * 3 + ch];
        lean_lite_update(&state, w, psd_mean + f * 15, &v);
        const int16_t *src[8] = { v.min10, v.max10, v.rise10, v.dev3, v.d01, v.range10, v.fall10, v.xm };
        for (int k = 0; k < 8; ++k) for (int ch = 0; ch < 3; ++ch) out[f * 24 + k * 3 + ch] = src[k][ch];
        ready[f] = lean_lite_ready(&state);
    }
    return (int)(sizeof(t_cload_lite) * 1000 + sizeof(lean_lite_state_t));
}
"""


class FirmwareLeanLite:
    """lean_lite.h compiled for the host (plain C instead of the CMSIS intrinsics)."""

    def __init__(self, header: Path = HEADER, workdir: Path = BUILD_DIR):
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if not cc:
            raise RuntimeError("No C compiler for the Lean-lite parity build")
        workdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(header, workdir / "lean_lite.h")
        (workdir / "lean_lite_driver.c").write_text(DRIVER)
        lib = workdir / f"liblean_lite_{sha256(header)[:12]}.so"
        cmd = [cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-DLEAN_LITE_HOST", "-shared", "-fPIC",
               "-I", str(workdir), str(workdir / "lean_lite_driver.c"), "-o", str(lib)]
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
        if p.returncode:
            raise RuntimeError("Lean-lite host build failed:\n" + p.stderr[-4000:])
        self.lib = ctypes.CDLL(str(lib))
        self.lib.lean_lite_run.restype = ctypes.c_int
        self.header_sha = sha256(header)

    def run(self, wl: np.ndarray, psd_mean: np.ndarray):
        n = len(wl)
        wl_ = np.ascontiguousarray(wl, dtype=np.uint32)
        psd_ = np.ascontiguousarray(np.asarray(psd_mean).reshape(n, 15), dtype=np.uint16)
        out = np.zeros((n, 3 * len(FIELDS)), np.int16); ready = np.zeros(n, np.uint8)
        sizes = self.lib.lean_lite_run(ctypes.c_int(n), wl_.ctypes.data_as(ctypes.c_void_p),
                                       psd_.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
                                       ready.ctypes.data_as(ctypes.c_void_p))
        if sizes // 1000 != 6 * len(FIELDS):
            raise AssertionError(f"t_cload_lite is {sizes // 1000} bytes, expected {6 * len(FIELDS)}")
        self.state_bytes = sizes % 1000
        return out, ready.astype(bool)


def synthetic_cases(seed: int = 7):
    """Edge cases the recordings never reach: WL 0 and near 2**32, every power of two, saturating PSD differences,
    long flat stretches (block extremes repeat), and a ramp (extremes expire every block)."""
    rng = np.random.default_rng(seed)
    cases = []
    n = 2000
    wl = rng.integers(0, 40000, size=(n, 3)).astype(np.uint64)
    wl[:40] = 0
    wl[40:72, 0] = 2 ** np.arange(32, dtype=np.uint64) - 1
    wl[40:72, 1] = 2 ** np.arange(32, dtype=np.uint64)[::-1] % (2 ** 32)
    wl[72:80] = 0xFFFFFFFF
    wl[300:900] = 1234                                       # flat
    wl[900:1500, 2] = np.arange(600, dtype=np.uint64) * 37   # ramp
    pm = rng.integers(0, 65536, size=(n, 3, 5)).astype(np.uint16)
    pm[:100, :, 0] = 65535; pm[:100, :, 1] = 0               # +65535 -> saturates
    pm[100:200, :, 0] = 0; pm[100:200, :, 1] = 65535         # -65535 -> saturates
    cases.append((wl, pm))
    cases.append((np.full((700, 3), 500, np.uint64), np.full((700, 3, 5), 1000, np.uint16)))
    cases.append((wl[:1], pm[:1]))
    return cases


def parity_check(files: list[Path]) -> dict:
    """Emulator vs. lean_lite.h on real recordings and synthetic edge cases: must be exactly equal."""
    fw = FirmwareLeanLite()
    frames = 0; mismatched = 0; ready_mismatch = 0
    runs = []
    for f in files:
        df = pd.read_csv(f, usecols=lambda c: c in set(SOURCE_COLUMNS), low_memory=False)
        wl = df[lean8.WL].to_numpy(np.float64)
        rows = lean8.frame_rows(df[lean8.PSD_NOW].to_numpy(np.int64), wl)
        runs.append((wl[rows].astype(np.uint64), df[lean8.PSD_MEAN].to_numpy(np.int64).reshape(-1, 3, 5)[rows]))
    runs += synthetic_cases()
    for wl, pm in runs:
        ours = lite_fields_per_frame(wl, pm).reshape(len(wl), -1)
        theirs, ready = fw.run(wl, pm)
        if len(wl):                                        # the trainer's WL-only diff path is the same arithmetic
            diff_idx = [3 * FIELDS.index(f) + c for f in DIFF_FEATURES for c in range(3)]
            mismatched += int((lite_diff_per_frame(wl) != theirs[:, diff_idx]).any(axis=1).sum())
        mismatched += int((ours != theirs).any(axis=1).sum())
        ready_mismatch += int(((np.arange(len(wl)) + 1 >= READY_FRAMES) != ready).sum())
        frames += len(wl)
    ok = mismatched == 0 and ready_mismatch == 0
    return dict(status="passed" if ok else "FAILED", header=str(HEADER.relative_to(ROOT)), header_sha256=fw.header_sha,
                recordings=len(files), synthetic_cases=len(runs) - len(files), frames=frames,
                frames_mismatched=mismatched, ready_mismatch=ready_mismatch, state_bytes=fw.state_bytes)


def ensure_datasets(pairs=None, workers: int = WORKERS, parity_files: int = PARITY_FILES) -> dict:
    """Generate/refresh the derived trees (raw roots are only read) and prove parity with lean_lite.h."""
    pairs = pairs or [(RAW_TAP, OUT_TAP), (RAW_LP, OUT_LP)]
    sample = []
    for raw, _ in pairs:
        fs = _recordings(raw)
        if fs and parity_files:
            k = max(1, parity_files // len(pairs))
            sample += fs[::max(1, len(fs) // k)][:k]
    parity = parity_check(sample)                          # before generating: a wrong emulator writes nothing
    print("LEAN_LITE_FIRMWARE_PARITY", json.dumps(parity), flush=True)
    if parity["status"] != "passed":
        raise RuntimeError("Offline Lean-lite does not match firmware/lean_lite.h: " + json.dumps(parity, indent=1))
    summaries = [generate_tree(Path(a), Path(b), workers) for a, b in pairs]
    manifest = {
        "version": 1, "generator_sha256": GENERATOR_SHA,
        "source_contract": "raw recordings are read only; derived files = original lines + Lean-lite columns",
        "features": list(LITE_COLUMNS), "extra_columns": [FRAMES_COLUMN, NEW_FRAME_COLUMN, READY_COLUMN],
        "handler_rate": "one call per new EMG frame (emg_psd change, split-frame aware; same as Lean-8)",
        "constants": dict(block_frames=BLOCK_FRAMES, blocks=BLOCKS, ema_shift=EMA_SHIFT, dev_shift=DEV_SHIFT,
                          ready_frames=READY_FRAMES),
        "summaries": summaries, "firmware_parity": parity,
    }
    _write_manifest(Path(pairs[0][1]).parent / "lean_lite_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    m = ensure_datasets()
    print(json.dumps({k: m[k] for k in ("summaries", "firmware_parity")}, indent=2))
    sys.exit(0)
