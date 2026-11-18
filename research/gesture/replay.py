"""Firmware-style replay for the unified gesture trainer.

Three pieces, all used both for Optuna scoring and for the final comparison:

1. HeaderModel   compiles ANY exported model header (new candidates or old
                 baselines) together with a 126-float t_core and calls its
                 *_predict (decision at the header's fixed threshold) and
                 *_predict_value (Q20 probability) on numpy rows. This is the
                 firmware arithmetic itself: float32 split thresholds, Q20 LUT.
2. tap_replay    rising edges of (probability >= 0.85) with a refractory time,
                 matched to confirmed tap/swipe onsets -> recall, false events
                 per minute of trusted-negative time, latency, per folder.
3. lp_replay     for every press, a ten-sample average of Q20 release values
                 that starts at the press; release when the average >= 0.85
                 -> holds kept, timely releases, taps that would become holds.

Every timing rule is a named constant so it can be matched to the firmware.
A "sample" is one logged row: in DATA_TAP nearly every row carries a new
p_core state (150-180 per second), so ten samples are about 55-65 ms.
"""
from __future__ import annotations

import ctypes
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

FIXED_THRESHOLD = 0.85

# TAP event replay
TAP_REFRACTORY_MS = 150      # a second rising edge within this time of the previous kept event is the same event
TAP_HIT_BEFORE_MS = 100      # an event this long before a labeled onset still counts as that tap
TAP_HIT_AFTER_MS = 250       # ... or this long after it
TAP_FALSE_PER_MIN_TARGET = 1.0   # score = recall * min(1, target / false events per minute)

# LP press replay
LP_AVERAGE_SAMPLES = 10      # firmware averages the last ten LONG_PRESS_predict_value() outputs
LP_MIN_HOLD_MS = 0           # firmware minimum hold timing: no release is declared before this
LP_PREMATURE_MARGIN_MS = 250 # a release declared more than this before the pointer release breaks the hold
LP_RELEASE_OK_MS = 500       # a release declared within this after the pointer release is timely
LP_LONG_PRESS_AFTER_MS = 500 # a tap still held this long after its onset would turn into a long press
LP_SCORE_WEIGHTS = dict(hold_kept=0.5, timely_release=0.3, tap_released=0.2)


# --------------------------------------------------------------------------- probabilities
def sigmoid(raw):
    raw = np.asarray(raw, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-raw))


def q20(probability):
    """Python twin of *_predict_value for LightGBM: nearest 0.05 step, as integer 0..20."""
    p = np.asarray(probability, dtype=np.float64)
    return np.clip(np.floor(p * 20.0 + 0.5), 0, 20).astype(np.int16)


# --------------------------------------------------------------------------- C header replay
CORE_TYPE_DEF = """#ifndef CORE_TYPE_DEF_H
#define CORE_TYPE_DEF_H
#include <stdint.h>
#define X 0
#define Y 1
#define Z 2
#define CH1 0
#define CH2 1
#define CH3 2
#define ACC_PSD_HISTORY_SIZE 10
#define EMG_PSD_HISTORY_SIZE 15
typedef struct {
    float emg_psd[EMG_PSD_HISTORY_SIZE][3][5];
    float emg_psd_mean[3][5];
    float acc_psd[ACC_PSD_HISTORY_SIZE][3][6];
    float acc_psd_mean[3][6];
    float acc_rate[3][6];
    float wl[3], wl_head[3], wl_tail[3];
} t_core;
#endif
"""


def _api_prefixes(text: str):
    value = re.findall(r"\b([A-Za-z_]\w*)_predict_value\s*\(\s*const\s+t_core\s*\*", text)
    boolean = re.findall(r"\b([A-Za-z_]\w*)_predict\s*\(\s*const\s+t_core\s*\*", text)
    return sorted(set(value)), sorted(set(boolean))


class HeaderModel:
    """An exported firmware header, callable from Python through a tiny C wrapper."""

    def __init__(self, header: Path, columns, workdir: Path, prefer: str | None = None):
        header = Path(header).resolve()
        if not header.is_file():
            raise FileNotFoundError(header)
        text = header.read_text(encoding="utf-8", errors="replace")
        values, bools = _api_prefixes(text)
        pick = lambda names: (prefer if prefer in names else (names[0] if len(names) == 1 else None))
        self.value_prefix = pick(values)
        self.bool_prefix = pick(bools)
        if self.value_prefix is None and self.bool_prefix is None:
            raise ValueError(f"No unique *_predict / *_predict_value(const t_core*) API in {header} "
                             f"(found value={values}, bool={bools})")
        self.columns = list(columns)
        for name in self.columns:
            if not name.startswith("p_core->"):
                raise ValueError(f"Not a p_core field: {name}")
        tag = hashlib.sha256((str(header) + text + "|".join(self.columns)).encode()).hexdigest()[:16]
        build = Path(workdir) / f"replay_{tag}"
        build.mkdir(parents=True, exist_ok=True)
        (build / "core_type_def.h").write_text(CORE_TYPE_DEF)
        fill = "\n".join(f"    c->{name[len('p_core->'):]} = x[{k}];" for k, name in enumerate(self.columns))
        src = ['#include <string.h>', '#include "core_type_def.h"', f'#include "{header.name}"',
               f"static void mudra_fill(t_core *c, const float *x) {{\n    memset(c, 0, sizeof(*c));\n{fill}\n}}"]
        n = len(self.columns)
        if self.value_prefix:
            src.append(f"void mudra_replay_value(const float *x, long rows, float *out) {{\n"
                       f"    t_core c; for (long i = 0; i < rows; ++i) {{ mudra_fill(&c, x + i * {n}); "
                       f"out[i] = (float){self.value_prefix}_predict_value(&c); }}\n}}")
        if self.bool_prefix:
            src.append(f"void mudra_replay_bool(const float *x, long rows, signed char *out) {{\n"
                       f"    t_core c; for (long i = 0; i < rows; ++i) {{ mudra_fill(&c, x + i * {n}); "
                       f"out[i] = (signed char)({self.bool_prefix}_predict(&c) ? 1 : 0); }}\n}}")
        (build / "replay.c").write_text("\n\n".join(src) + "\n")
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if not cc:
            raise RuntimeError("No C compiler for header replay")
        lib = build / "libreplay.so"
        cmd = [cc, "-std=c11", "-O2", "-shared", "-fPIC", str(build / "replay.c"),
               "-I", str(build), "-I", str(header.parent), "-o", str(lib)]
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise RuntimeError(f"Header replay build failed for {header}: {(result.stderr or result.stdout)[-2500:]}")
        self.header = header
        self._lib = ctypes.CDLL(str(lib))
        fp = ctypes.POINTER(ctypes.c_float)
        if self.value_prefix:
            self._lib.mudra_replay_value.argtypes = [fp, ctypes.c_long, fp]
        if self.bool_prefix:
            self._lib.mudra_replay_bool.argtypes = [fp, ctypes.c_long, ctypes.POINTER(ctypes.c_byte)]

    def _rows(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(self.columns):
            raise ValueError(f"Expected rows x {len(self.columns)} features, got {X.shape}")
        return X

    def predict_value(self, X):
        if not self.value_prefix:
            raise ValueError(f"{self.header.name} has no *_predict_value API")
        X = self._rows(X); out = np.empty(len(X), np.float32)
        self._lib.mudra_replay_value(X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(X),
                                     out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out

    def predict_bool(self, X):
        if not self.bool_prefix:
            raise ValueError(f"{self.header.name} has no *_predict API")
        X = self._rows(X); out = np.empty(len(X), np.int8)
        self._lib.mudra_replay_bool(X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(X),
                                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)))
        return out

    def value_q20(self, X):
        """Q20 integers 0..20 from *_predict_value (or 0/20 from *_predict if that is all there is)."""
        if self.value_prefix:
            return np.clip(np.floor(self.predict_value(X).astype(np.float64) * 20.0 + 0.5), 0, 20).astype(np.int16)
        return (self.predict_bool(X).astype(np.int16) * 20)

    def decision(self, X):
        if self.bool_prefix:
            return self.predict_bool(X).astype(bool)
        return self.value_q20(X) >= int(round(FIXED_THRESHOLD * 20))


# --------------------------------------------------------------------------- TAP event replay
class TapCounts:
    """Additive counts so folds can be combined before computing rates."""

    def __init__(self):
        self.onsets = 0; self.hits = 0; self.false = 0; self.neg_ms = 0.0; self.latency = []
        self.folder = {}

    def _f(self, name):
        return self.folder.setdefault(name, dict(onsets=0, hits=0, false=0, neg_ms=0.0))

    def add(self, other: "TapCounts"):
        self.onsets += other.onsets; self.hits += other.hits; self.false += other.false
        self.neg_ms += other.neg_ms; self.latency.extend(other.latency)
        for k, v in other.folder.items():
            f = self._f(k)
            for key in f: f[key] += v[key]
        return self

    def report(self, target=TAP_FALSE_PER_MIN_TARGET):
        rate = lambda false, ms: false / (ms / 60000.0) if ms > 0 else float("nan")
        recall = self.hits / self.onsets if self.onsets else float("nan")
        fpm = rate(self.false, self.neg_ms)
        score = (recall * min(1.0, target / fpm)) if (self.onsets and fpm > 0) else (recall if self.onsets else 0.0)
        lat = np.asarray(self.latency, float)
        folders = {k: dict(onsets=v["onsets"], recall=(v["hits"] / v["onsets"] if v["onsets"] else None),
                           false_events=v["false"], negative_minutes=round(v["neg_ms"] / 60000.0, 2),
                           false_per_min=(rate(v["false"], v["neg_ms"]) if v["neg_ms"] > 0 else None))
                   for k, v in sorted(self.folder.items())}
        return dict(onsets=self.onsets, hits=self.hits, recall=recall, false_events=self.false,
                    negative_minutes=self.neg_ms / 60000.0, false_per_min=fpm,
                    latency_ms_median=float(np.median(lat)) if len(lat) else None,
                    latency_ms_p90=float(np.percentile(lat, 90)) if len(lat) else None,
                    objective_score=float(score), false_per_min_target=target, per_folder=folders)


def tap_replay_recording(decision, t_ms, label, onsets, folder, counts: TapCounts):
    """One recording. decision: bool per row; label: tap_label per row; onsets: confirmed onset times (ms)."""
    d = np.asarray(decision, bool); t = np.asarray(t_ms, np.float64); lab = np.asarray(label)
    if not len(d):
        return counts
    rise = np.flatnonzero(d & ~np.r_[False, d[:-1]])
    kept = []; last = -1e18
    for r in rise:
        if t[r] - last >= TAP_REFRACTORY_MS:
            kept.append(r); last = t[r]
    kept = np.asarray(kept, dtype=np.int64)
    used = np.zeros(len(kept), bool)
    et = t[kept] if len(kept) else np.zeros(0)
    f = counts._f(folder)
    for o in np.sort(np.asarray(onsets, np.float64)):
        cand = np.flatnonzero(~used & (et >= o - TAP_HIT_BEFORE_MS) & (et <= o + TAP_HIT_AFTER_MS))
        counts.onsets += 1; f["onsets"] += 1
        if len(cand):
            used[cand[0]] = True; counts.hits += 1; f["hits"] += 1; counts.latency.append(float(et[cand[0]] - o))
    # unmatched events on trusted-negative or positive rows are errors; on unknown rows they are not counted
    spare = kept[~used]
    false = int((lab[spare] >= 0).sum()) if len(spare) else 0
    counts.false += false; f["false"] += false
    dt = np.diff(np.r_[t, t[-1] + (np.median(np.diff(t)) if len(t) > 1 else 0.0)])
    dt = np.clip(dt, 0, 50)
    neg = float(dt[lab == 0].sum())
    counts.neg_ms += neg; f["neg_ms"] += neg
    return counts


# --------------------------------------------------------------------------- LP press replay
class LpCounts:
    def __init__(self):
        self.holds = 0; self.kept = 0; self.timely = 0; self.stuck = 0; self.latency = []; self.hold_frac = []
        self.taps = 0; self.taps_released = 0; self.tap_latency = []; self.folder = {}

    def add(self, other: "LpCounts"):
        for k in ("holds", "kept", "timely", "stuck", "taps", "taps_released"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.latency += other.latency; self.hold_frac += other.hold_frac; self.tap_latency += other.tap_latency
        for k, v in other.folder.items():
            f = self.folder.setdefault(k, dict(n=0, ok=0))
            f["n"] += v["n"]; f["ok"] += v["ok"]
        return self

    def report(self, weights=None):
        w = dict(LP_SCORE_WEIGHTS if weights is None else weights)
        hk = self.kept / self.holds if self.holds else float("nan")
        tr = self.timely / self.holds if self.holds else float("nan")
        ta = self.taps_released / self.taps if self.taps else float("nan")
        parts = dict(hold_kept=hk, timely_release=tr, tap_released=ta)
        usable = {k: v for k, v in parts.items() if v == v}      # drop NaN parts (no presses of that kind)
        if usable and all(v > 0 for v in usable.values()):
            total = sum(w[k] for k in usable)
            score = total / sum(w[k] / v for k, v in usable.items())
        else:
            score = 0.0
        lat = np.asarray(self.latency, float); tl = np.asarray(self.tap_latency, float)
        return dict(holds=self.holds, hold_kept=hk, timely_release=tr, stuck=self.stuck,
                    release_latency_ms_median=float(np.median(lat)) if len(lat) else None,
                    release_latency_ms_p90=float(np.percentile(lat, 90)) if len(lat) else None,
                    held_fraction_median=float(np.median(self.hold_frac)) if self.hold_frac else None,
                    taps=self.taps, tap_released=ta,
                    tap_release_latency_ms_median=float(np.median(tl)) if len(tl) else None,
                    objective_score=float(score), weights=w,
                    per_folder={k: dict(presses=v["n"], ok=(v["ok"] / v["n"] if v["n"] else None))
                                for k, v in sorted(self.folder.items())})


def lp_declared_release(q, t_ms, start_ms):
    """First time the ten-sample average of Q20 values (from the press start) reaches the threshold, else None."""
    q = np.asarray(q, np.int64); t = np.asarray(t_ms, np.float64)
    n = LP_AVERAGE_SAMPLES; need = int(round(FIXED_THRESHOLD * 20 * n))
    if len(q) < n:
        return None
    s = np.convolve(q, np.ones(n, np.int64), "valid")              # s[j] = sum q[j .. j+n-1]
    end_t = t[n - 1:]
    ok = (s >= need) & (end_t - start_ms >= LP_MIN_HOLD_MS)
    j = np.flatnonzero(ok)
    return float(end_t[j[0]]) if len(j) else None


def lp_replay_press(q, t_ms, start_ms, release_ms, kind, folder, counts: LpCounts):
    """q: Q20 release values for rows from the press start onwards; kind 'hold' or 'tap'."""
    d = lp_declared_release(q, t_ms, start_ms)
    f = counts.folder.setdefault(folder, dict(n=0, ok=0)); f["n"] += 1
    if kind == "hold":
        counts.holds += 1
        premature = d is not None and d < release_ms - LP_PREMATURE_MARGIN_MS
        if not premature:
            counts.kept += 1
        if d is None:
            counts.stuck += 1
        else:
            counts.hold_frac.append(min(1.0, (d - start_ms) / max(1.0, release_ms - start_ms)))
            if not premature:
                counts.latency.append(d - release_ms)
        timely = (not premature) and d is not None and d - release_ms <= LP_RELEASE_OK_MS
        counts.timely += int(timely); f["ok"] += int(timely)
    else:
        counts.taps += 1
        ok = d is not None and d - start_ms < LP_LONG_PRESS_AFTER_MS
        counts.taps_released += int(ok); f["ok"] += int(ok)
        if d is not None:
            counts.tap_latency.append(d - release_ms)
    return counts
