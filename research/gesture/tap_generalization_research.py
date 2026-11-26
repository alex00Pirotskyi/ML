#!/usr/bin/env python3
"""TAP generalization research: which features, training tricks and labels make the TAP model robust to band
placement and electrode contact, without losing taps or adding false taps?

    python research/gesture/tap_generalization_research.py <arm> [<arm> ...]     # run arms (see ARMS)
    python research/gesture/tap_generalization_research.py report                # summary table (markdown)
    python research/gesture/tap_generalization_research.py focus <arm> [...]     # per-gesture press report
    python research/gesture/tap_generalization_research.py focus_table <arm> ... # compare (first arm = reference)

Focus arms (2026-11-20): press2 (every press of DoubleTaps / 240 bpm recordings is a positive), fingerw<k>
(FINGER-folder negatives, finger motion and fist, weighted x k), fmine (hard negatives mined from FINGER
recordings), litediff (gain-invariant Lean-lite context features in the log domain), pw<k> (positives of
DoubleTaps / 240 bpm recordings weighted x k), posw<c> (positive-class weight c/100, as Optuna sets it), trees<n>,
lr<k> (learning rate k/100), depth<k>, xgb (XGBoost, same byte accounting), lrscore (stacked linear
score input), accsum / emgsum (ACC / EMG total inputs). "rule <arm> min2 ..." re-scores saved fold models
with another firmware decision rule. See TAP_FOCUS_RESULTS.md.

File split (2026-11-24): MUDRA_GR_SPLIT=file uses the trainer's gesture-aware file split and folds (results in
results_tap_filesplit/); MUDRA_GR_STRESS=nominal,placement scores only those conditions. Arms softall (the
labeller's soft taps of every TAP / SWIPE recording as positives) and distill<a> (soft targets a*label +
(1-a)*cross-fitted full-size teacher, fitted as two weighted binary rows per frame). See TAP_FILESPLIT_RESULTS.md.

Data: an existing full TAP cache (MUDRA_GR_CACHE, default /tmp/v3soft/tap/cache: RAW_126 + Lean-8 24 columns,
built by train_unified.build_task_data). Development people only (the trainer's 22% person holdout is never
touched), 3 person folds (GroupKFold, as the trainer and fs_research). Soft-label rows are not used.

Evaluation never changes between arms: the firmware replay (replay.tap_replay_recording) on every steady row
of the validation people, with the ORIGINAL labels: confirmed TAP and SWIPE onsets (recall per folder), every
real pointer press (recall_press), false events per minute on trusted-negative rows (per folder).

Placement / contact stress: the validation EMG is rescaled per recording and per channel by a random amplitude
gain (log-uniform, +-0.5 or +-1 octave), exactly as a contact change rescales the signal: WL x g, PSD power x g^2
(codes are 10000*log10(1+power)), and the Lean-8 fields recomputed from those (min10/max10 exactly, sd3 kept).
ACC is unchanged. A model that generalizes keeps its numbers under the stress.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Rows from frame 50 are trained on and scored ("usable": 7x more confirmed onsets than the 500-frame steady gate,
# which only Lean-8 needs); the steady view (frame >= 500) is reported as well.
os.environ.setdefault("MUDRA_TRAIN_MIN_FRAMES", "50")
os.environ.setdefault("MUDRA_EVAL_MIN_FRAMES", "50")
import train_unified as g          # noqa: E402
import replay as R                 # noqa: E402
import feature_selection as FS     # noqa: E402
import generate_lean_lite_features as LL   # noqa: E402

CACHE = Path(os.environ.get("MUDRA_GR_CACHE", "/tmp/v3soft/tap/cache"))
# Recordings added after the full cache was built (tap_extra_cache.py): MUDRA_GR_EXTRA = comma-separated cache dirs.
# MUDRA_GR_EXTRA_MODE: "eval" (default) = replayed only, "train" = their rows join every training fold (never a
# validation fold). The person split and folds of the full cache stay exactly as they are.
EXTRA = [Path(p_) for p_ in os.environ.get("MUDRA_GR_EXTRA", "").split(",") if p_]
EXTRA_MODE = os.environ.get("MUDRA_GR_EXTRA_MODE", "eval")
# MUDRA_GR_SPLIT=file: the trainer's gesture-aware file split (train_unified.group_by_file): every recording is its
# own group, the held-out files and the 3 CV folds are stratified by recording type. Results then go to
# results_tap_filesplit unless MUDRA_GR_OUT says otherwise. File split only.
SPLIT = os.environ.get("MUDRA_GR_SPLIT", "file").strip().lower()
if SPLIT != "file":       # person splits are not allowed (as in the trainer); results_tap_generalization is history
    raise SystemExit("MUDRA_GR_SPLIT must be file: research splits and folds by recording file only")
OUT = Path(os.environ.get("MUDRA_GR_OUT", HERE / ("results_tap_generalization" if SPLIT == "person"
                                                  else "results_tap_filesplit")))
QUICK = os.environ.get("MUDRA_GR_QUICK") == "1"
THREADS = int(os.environ.get("MUDRA_GR_THREADS", os.cpu_count() or 2))
CONFIG = dict(objective="binary", n_estimators=400, learning_rate=0.1, num_leaves=31, max_depth=8,
              min_child_samples=100, reg_lambda=1.0, reg_alpha=1e-3, colsample_bytree=0.7, subsample=0.8,
              subsample_freq=1, max_bin=127, verbosity=-1, deterministic=True)
GRID = FS.TAP_GRID
STRESS = ("nominal", "gain", "tilt", "noise", "placement")    # see stress_tables()
# MUDRA_GR_STRESS=nominal,placement scores only those conditions (faster; their numbers are unchanged)
STRESS = tuple(c for c in os.environ.get("MUDRA_GR_STRESS", ",".join(STRESS)).split(",") if c) or STRESS
assert "nominal" in STRESS and set(STRESS) <= {"nominal", "gain", "tilt", "noise", "placement"}, STRESS
TEACHER_THRESHOLD = 0.5                 # fallback only (no movement recordings among the people)
NEG_FOLDERS = ("FINGER", "WRIST", "ARM", "BAND", "NOTHING")

COLS = list(g.RAW_126) + list(g.LEAN8_24)
CI = {c: i for i, c in enumerate(COLS)}
ACC = [CI[c] for c in g.RAW_126 if c.startswith(("p_core->acc_psd", "p_core->acc_rate"))]
ACC_NOW = [CI[f"p_core->acc_psd[0][{a}][{z}]"] for a in "XYZ" for z in range(6)]
ACC_NOW_ALL = ACC_NOW
ACC_MEAN_ALL = [CI[f"p_core->acc_psd_mean[{a}][{z}]"] for a in "XYZ" for z in range(6)]
ACC_RATE_ALL = [CI[f"p_core->acc_rate[{a}][{z}]"] for a in "XYZ" for z in range(6)]
SLOTS = ("p_core->emg_psd[0]", "p_core->emg_psd[EMG_PSD_HISTORY_SIZE - 1]", "p_core->emg_psd_mean")
EMG_PSD = [CI[f"{s}[CH{c}][{z}]"] for s in SLOTS for c in (1, 2, 3) for z in range(5)]
WL = {k: [CI[f"p_core->{k}[CH{c}]"] for c in (1, 2, 3)] for k in ("wl", "wl_head", "wl_tail")}
EMG_RAW = EMG_PSD + WL["wl"] + WL["wl_head"] + WL["wl_tail"]
LEAN = [CI[c] for c in g.LEAN8_24]
assert len(ACC) == 72 and len(EMG_RAW) == 54 and len(ACC) + len(EMG_RAW) == 126 and len(COLS) == 150


def _emg(slot, ch, z):
    return CI[f"{SLOTS[slot]}[CH{ch}][{z}]"]


# Within-channel EMG pairs: zone vs zone in each time slot (spectral shape) and slot vs slot per zone (change).
PAIRS = ([(_emg(s, c, a), _emg(s, c, b)) for c in (1, 2, 3) for s in range(3) for a in range(5) for b in range(a + 1, 5)] +
         [(_emg(sa, c, z), _emg(sb, c, z)) for c in (1, 2, 3) for z in range(5) for sa, sb in ((0, 1), (0, 2), (1, 2))])
WL_PAIRS = [(WL[a][c], WL[b][c]) for c in range(3) for a, b in (("wl_head", "wl_tail"), ("wl", "wl_head"), ("wl", "wl_tail"))]
assert len(PAIRS) == 135 and len(WL_PAIRS) == 9
PA = np.array([a for a, _ in PAIRS]); PB = np.array([b for _, b in PAIRS])
# ACC pairs (accdiff): zone vs zone of the current spectrum per axis (impact = broadband) and current vs oldest
# history / vs mean per axis and zone (onset sharpness). Codes as the EMG ones, p_core->diff in firmware.
_ACC_SLOT = ("p_core->acc_psd[0]", "p_core->acc_psd[ACC_PSD_HISTORY_SIZE - 1]", "p_core->acc_psd_mean")
_acc = lambda s_, a_, z_: CI[f"{_ACC_SLOT[s_]}[{a_}][{z_}]"]
ACC_PAIRS = ([(_acc(0, a_, z1), _acc(0, a_, z2)) for a_ in "XYZ" for z1 in range(6) for z2 in range(z1 + 1, 6)] +
             [(_acc(0, a_, z_), _acc(s_, a_, z_)) for a_ in "XYZ" for z_ in range(6) for s_ in (1, 2)])
assert len(ACC_PAIRS) == 81
AA = np.array([a for a, _ in ACC_PAIRS]); AB = np.array([b for _, b in ACC_PAIRS])
WA = np.array([a for a, _ in WL_PAIRS]); WB = np.array([b for _, b in WL_PAIRS])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================ features
def build(X, spec):
    """X: (n, 150) float32 rows in COLS order. spec: tuple of tokens. Returns (F, names, is_emg)."""
    parts, names, emg = [], [], []

    def add(block, nm, is_emg):
        parts.append(np.asarray(block, np.float32)); names.extend(nm); emg.extend([is_emg] * len(nm))

    if "acc" in spec:
        add(X[:, ACC], [COLS[i] for i in ACC], False)
    if "emgraw" in spec:
        add(X[:, EMG_RAW], [COLS[i] for i in EMG_RAW], True)
    if "lean" in spec:
        add(X[:, LEAN], [COLS[i] for i in LEAN], True)
    if "diff" in spec or "diffz" in spec or "difft" in spec:
        # diff = all 135; diffz = the 90 zone-vs-zone pairs (spectral shape); difft = the 45 slot-vs-slot pairs (change
        # of each zone: now vs oldest history vs mean, i.e. event-local)
        sel = np.arange(len(PAIRS)) if "diff" in spec else (np.arange(90) if "diffz" in spec else np.arange(90, 135))
        if "diffz" in spec and "difft" in spec:
            sel = np.arange(len(PAIRS))
        add(np.clip(X[:, PA[sel]] - X[:, PB[sel]], -32768, 32767),
            [f"p_core->diff({COLS[PAIRS[i][0]]}, {COLS[PAIRS[i][1]]})" for i in sel], True)
    if "ratio" in spec:
        add((X[:, PA] + 1.0) / (X[:, PB] + 1.0), [f"p_core->ratio({COLS[a]}, {COLS[b]})" for a, b in PAIRS], True)
    if "logratio" in spec:                      # exact log power ratio (would need a firmware LUT): upper bound
        lin = lambda c: np.power(10.0, c.astype(np.float64) / 1e4) - 1.0
        add(np.log10(lin(X[:, PA]) + 1e-4) - np.log10(lin(X[:, PB]) + 1e-4),
            [f"logratio({COLS[a]}, {COLS[b]})" for a, b in PAIRS], True)
    if "wlratio" in spec:
        add((X[:, WA] + 1.0) / (X[:, WB] + 1.0), [f"p_core->ratio({COLS[a]}, {COLS[b]})" for a, b in WL_PAIRS], True)
    if "accdiff" in spec:
        add(np.clip(X[:, AA] - X[:, AB], -32768, 32767), [f"p_core->diff({COLS[a]}, {COLS[b]})" for a, b in ACC_PAIRS],
            False)
    if "accsum" in spec:                        # ACC impact totals (sums of codes over axes and zones)
        now = X[:, ACC_NOW_ALL].astype(np.float32); mean = X[:, ACC_MEAN_ALL].astype(np.float32)
        hi = np.array([z_ >= 3 for _a in "XYZ" for z_ in range(6)])
        add(np.stack([now.sum(1), (now - mean).sum(1), (now - mean)[:, hi].sum(1),
                      X[:, ACC_RATE_ALL].astype(np.float32).sum(1)], 1),
            ["acc_now_total", "acc_rise_total", "acc_rise_hi_total", "acc_rate_total"], False)
    if "emgsum" in spec:                        # EMG across channels, log domain (Lean-lite rise10 / xm)
        assert X.shape[1] == len(COLS) + len(LITE_DIFF), "emgsum needs the Lean-lite columns appended (with_lite)"
        li = {n_: len(COLS) + i for i, n_ in enumerate(LITE_DIFF)}
        rise = X[:, [li[f"lite.rise10[CH{c}]"] for c in (1, 2, 3)]]; xm = X[:, [li[f"lite.xm[CH{c}]"] for c in (1, 2, 3)]]
        add(np.stack([rise.sum(1), rise.min(1), rise.max(1), xm.sum(1)], 1),
            ["lite.rise10_sum", "lite.rise10_min", "lite.rise10_max", "lite.xm_sum"], True)
    if "litediff" in spec or "litediff4" in spec:
        assert X.shape[1] == len(COLS) + len(LITE_DIFF), "litediff needs the Lean-lite columns appended (with_lite)"
        keep = [i for i, n in enumerate(LITE_DIFF) if "litediff" in spec or not n.startswith("lite.fall10")]
        add(X[:, len(COLS) + np.array(keep)], [LITE_DIFF[i] for i in keep], True)
    return np.hstack(parts), names, np.asarray(emg, bool)


WL_PER_SQRT_POWER = 850.0          # WL ~ 850 * sqrt(total EMG PSD power) in this data (log-log correlation 0.93)


def apply_emg(X, gain=None, tilt=None, noise=None):
    """Contact / placement change on the EMG fields of rows X (ACC untouched):
    gain (n, 3): amplitude x g per channel; tilt (n, 3, 5): amplitude x t per channel and PSD zone (a different
    muscle / filter coupling); noise (n, 3): broadband noise power added per channel (loose contact).
    PSD codes are 10000*log10(1+power): exact per slot (the history-mean slot is treated like a single spectrum).
    WL ~ sqrt(power): x g * (power ratio)^0.5, then the noise adds in quadrature. Lean-8 fields are recomputed
    (lw, min10, max10, rise10, Pm, r01, r32; sd3 kept)."""
    n = len(X)
    Y = X.copy()
    G = np.ones((n, 3)) if gain is None else gain.astype(np.float64)
    T = np.ones((n, 3, 5)) if tilt is None else tilt.astype(np.float64)
    N = np.zeros((n, 3)) if noise is None else noise.astype(np.float64)
    for c in range(3):
        lin_now = None
        for s_ in range(3):
            lins = []
            for z in range(5):
                j = _emg(s_, c + 1, z)
                lin = np.power(10.0, X[:, j].astype(np.float64) / 1e4) - 1.0
                new = (G[:, c] * T[:, c, z]) ** 2 * lin + N[:, c] / 5.0
                Y[:, j] = np.clip(np.round(1e4 * np.log10(1.0 + new)), 0, 65535)
                lins.append((lin, new))
            if s_ == 0:
                lin_now = lins
            if s_ == 2:
                lin_mean = lins
        p0 = sum(l for l, _ in lin_now); p1 = sum((G[:, c] * T[:, c, z]) ** 2 * l for z, (l, _) in enumerate(lin_now))
        tau = np.sqrt(np.where(p0 > 1e-9, p1 / np.maximum(p0, 1e-9), G[:, c] ** 2))      # amplitude factor of WL
        wl_map = lambda w: np.sqrt((w * tau) ** 2 + WL_PER_SQRT_POWER ** 2 * N[:, c])
        for k in ("wl", "wl_head", "wl_tail"):
            j = WL[k][c]; Y[:, j] = np.round(wl_map(X[:, j].astype(np.float64)))
        L = lambda name: CI[f"p_core->cload_lean_8.{name}[CH{c + 1}]"]
        f = lambda x: np.log10(wl_map(np.power(10.0, x.astype(np.float64)) - 1.0) + 1.0)
        # Lean-8 values are stored as the firmware computed them (handler timing), so they are moved by the same
        # change rather than recomputed from this row: identity stress leaves them exactly unchanged.
        lw, mn, mx = f(X[:, L("lw")]), f(X[:, L("min10")]), f(X[:, L("max10")])
        Y[:, L("lw")] = lw; Y[:, L("min10")] = mn; Y[:, L("max10")] = mx
        Y[:, L("rise10")] = np.maximum(lw - mn, 0.0)
        old_p = [np.maximum(o, 1e-4) for o, _ in lin_mean]; new_p = [np.maximum(n_, 1e-4) for _, n_ in lin_mean]
        old_t = np.maximum(sum(o for o, _ in lin_mean), 1e-4); new_t = np.maximum(sum(n_ for _, n_ in lin_mean), 1e-4)
        db = lambda a_, b_: 10.0 * np.log10(a_ / b_)
        Y[:, L("Pm")] = X[:, L("Pm")] + db(new_t, old_t)
        Y[:, L("r01")] = X[:, L("r01")] + db(new_p[0], old_p[0]) - db(new_p[1], old_p[1])
        Y[:, L("r32")] = X[:, L("r32")] + db(new_p[3], old_p[3]) - db(new_p[2], old_p[2])
    return Y


def apply_gain(X, G):
    return apply_emg(X, gain=G)


def gain_table(n_recs, octaves, seed):
    rng = np.random.default_rng(seed)
    return np.power(2.0, rng.uniform(-octaves, octaves, size=(n_recs, 3))).astype(np.float32)


def stress_tables(n_recs, seed=77, tilt_oct=0.5, noise_log10=(-1.0, 0.0)):
    """Per-recording draws (the same for every arm): gain +-1 octave per channel; tilt +-0.5 octave per channel
    and zone; noise power log-uniform 0.1..1.0 per channel (median total EMG power is ~0.7-1.1); placement = all.
    Recordings added with MUDRA_GR_EXTRA get their own draws, so the full cache's recordings keep theirs."""
    n_base = _D.get("n_base")
    if n_base is not None and n_recs > n_base:
        a = stress_tables(n_base, seed, tilt_oct, noise_log10)
        b = stress_tables.__wrapped__(n_recs - n_base, seed + 7919, tilt_oct, noise_log10)
        return {c: (None if a[c] is None else {k: np.concatenate([a[c][k], b[c][k]]) for k in a[c]}) for c in a}
    return stress_tables.__wrapped__(n_recs, seed, tilt_oct, noise_log10)


def _stress_tables(n_recs, seed=77, tilt_oct=0.5, noise_log10=(-1.0, 0.0)):
    rng = np.random.default_rng(seed)
    gain = np.power(2.0, rng.uniform(-1, 1, (n_recs, 3)))
    tilt = np.power(2.0, rng.uniform(-tilt_oct, tilt_oct, (n_recs, 3, 5)))
    noise = np.power(10.0, rng.uniform(noise_log10[0], noise_log10[1], (n_recs, 3)))
    return dict(nominal=None, gain=dict(gain=gain), tilt=dict(tilt=tilt), noise=dict(noise=noise),
                placement=dict(gain=gain, tilt=tilt, noise=noise))


stress_tables.__wrapped__ = _stress_tables


# ============================================================================ Lean-lite diff (context) features
# Log-domain differences of the Lean-lite state (firmware/lean_lite.h): x = log2(WL + 1) in Q8 per channel, lo / hi
# = its min / max over the current and the 9 previous 1 s blocks, the EMA mean (shift 7, ~2.8 s) and the mean
# absolute deviation from it. A contact gain multiplies WL, i.e. adds a constant to x: every value below is unchanged
# (up to rounding). rise10 / dev3 are in cload_lite today; fall10 / range10 / xm need one line each in lean_lite.h.
LITE_DIFF = tuple(f"lite.{n}[CH{c}]" for n in ("rise10", "fall10", "range10", "dev3", "xm") for c in (1, 2, 3))
LITE_FILE = "lite_diff_v1.npy"


def lite_frames(wl):
    """wl: (frames, 3) integer WL at the EMG frame rows -> (frames, 15) int64 in LITE_DIFF order (after a reset)."""
    n = len(wl)
    out = np.zeros((n, 5, 3), np.int64)
    if not n:
        return out.reshape(n, 15)
    B = LL.BLOCK_FRAMES; nb = -(-n // B); pad = nb * B - n; full = n // B
    for ch in range(3):
        x = LL.log2_q8(wl[:, ch]).astype(np.int64)
        xb = np.concatenate([x, np.full(pad, x[-1])]).reshape(nb, B)
        cur_min = np.minimum.accumulate(xb, axis=1).reshape(-1)[:n]
        cur_max = np.maximum.accumulate(xb, axis=1).reshape(-1)[:n]
        bmin = cur_min[B - 1::B][:full]; bmax = cur_max[B - 1::B][:full]
        agg_min = np.full(nb, np.iinfo(np.int64).max); agg_max = np.full(nb, np.iinfo(np.int64).min)
        for b in range(1, nb):
            agg_min[b] = bmin[max(0, b - LL.RING):b].min(); agg_max[b] = bmax[max(0, b - LL.RING):b].max()
        blk = np.arange(n) // B
        lo = np.minimum(cur_min, agg_min[blk]); hi = np.maximum(cur_max, agg_max[blk])
        dev = np.empty(n, np.int64); xm = np.empty(n, np.int64)
        m_acc = int(x[0]) << LL.EMA_SHIFT; d_acc = 0
        for i, xi in enumerate(x.tolist()):
            m_acc += xi - (m_acc >> LL.EMA_SHIFT)
            d = xi - (m_acc >> LL.EMA_SHIFT)
            d_acc += (d if d >= 0 else -d) - (d_acc >> LL.EMA_SHIFT)
            dev[i] = d_acc; xm[i] = d
        for k, v in enumerate((x - lo, hi - x, hi - lo, dev >> LL.DEV_SHIFT, xm)):
            out[:, k, ch] = v
    return out.reshape(n, 15)


def lite_recording(rec, stress=None):
    """Lean-lite diff values on every replay row of a recording (reset at its first row), optionally under a
    placement change (stress: per-recording tables as stress_tables() returns them)."""
    rpl = load()["rpl"]; r = rpl["recs"].iloc[int(rec)]; a, b = int(r.start), int(r.end)
    if b <= a:
        return np.zeros((0, 15), np.int64)
    fr = np.asarray(rpl["frames"][a:b])
    is_new = np.r_[fr[0] > 0, np.diff(fr) > 0]
    X = np.asarray(rpl["X"][a:b])[is_new]
    if stress:
        X = apply_emg(X, **{k: np.repeat(np.asarray(v[int(rec)])[None], len(X), 0) for k, v in stress.items()})
    feats = lite_frames(np.round(X[:, WL["wl"]]).astype(np.int64))
    idx = np.cumsum(is_new) - 1
    out = np.zeros((b - a, 15), np.int64); ok = idx >= 0; out[ok] = feats[idx[ok]]
    return out


_LITE = {}


def lite_nominal():
    """Lean-lite diff values of every replay row (cached next to the TAP cache)."""
    if "nom" in _LITE:
        return _LITE["nom"]
    D = load(); rpl = D["rpl"]; n = rpl["X"].shape[0]; path = CACHE / LITE_FILE
    nb = D.get("n_base_rows", n)
    if path.exists() and np.load(path, mmap_mode="r").shape == (nb, 15):
        arr = np.load(path)
        if n > nb:                                   # recordings added with MUDRA_GR_EXTRA
            ext = np.zeros((n - nb, 15), np.int16); recs = rpl["recs"]
            for rec in range(D["n_base"], len(recs)):
                a, b = int(recs.start.iloc[rec]), int(recs.end.iloc[rec])
                if b > a:
                    ext[a - nb:b - nb] = np.clip(lite_recording(rec), -32768, 32767)
            arr = np.concatenate([arr, ext])
    else:
        t0 = time.perf_counter(); arr = np.zeros((n, 15), np.int16); recs = rpl["recs"]
        for rec in range(len(recs)):
            a, b = int(recs.start.iloc[rec]), int(recs.end.iloc[rec])
            if b > a:
                arr[a:b] = np.clip(lite_recording(rec), -32768, 32767)
        try:
            if n == nb:
                np.save(path, arr)
        except OSError:
            pass
        log("GR_LITE_CACHE", round(time.perf_counter() - t0, 1), "s")
    _LITE["nom"] = arr
    return arr


def lite_rows(ridx, rec, stress=None):
    """Lean-lite diff values (float32) at replay rows ridx of recordings rec, nominal or under stress tables."""
    ridx = np.asarray(ridx, np.int64); rec = np.asarray(rec, np.int64)
    if not stress:
        return np.asarray(lite_nominal()[ridx], np.float32)
    starts = load()["rpl"]["recs"].start.to_numpy(np.int64)
    out = np.zeros((len(ridx), 15), np.float32)
    order = np.argsort(rec, kind="stable"); rs = rec[order]
    u, first = np.unique(rs, return_index=True); last = np.r_[first[1:], len(rs)]
    for r_, i0, i1 in zip(u, first, last):
        sel = order[i0:i1]
        out[sel] = lite_recording(int(r_), stress)[ridx[sel] - starts[r_]]
    return out


def data_ridx(rows):
    """Replay row of each training-cache row (same recording, same timestamp)."""
    D = load()
    if "ridx" not in D:
        data = D["data"]; rpl = D["rpl"]; recs = rpl["recs"]
        ridx = np.full(len(data["y"]), -1, np.int64); dts = np.zeros(len(data["y"]))
        order = np.argsort(data["rec"], kind="stable"); rr = data["rec"][order]
        bounds = np.searchsorted(rr, np.arange(len(recs) + 1))
        for rec in range(len(recs)):
            i0, i1 = bounds[rec], bounds[rec + 1]
            a, b = int(recs.start.iloc[rec]), int(recs.end.iloc[rec])
            if i1 <= i0 or b <= a:
                continue
            t = np.asarray(rpl["t"][a:b], np.float64); tt = data["t"][order[i0:i1]].astype(np.float64)
            j = np.clip(np.searchsorted(t, tt), 0, b - a - 1); j2 = np.clip(j - 1, 0, b - a - 1)
            pick = np.where(np.abs(t[j2] - tt) < np.abs(t[j] - tt), j2, j)
            ridx[order[i0:i1]] = a + pick; dts[order[i0:i1]] = np.abs(t[pick] - tt)
        D["ridx"] = ridx
        log("GR_RIDX", json.dumps(dict(mapped=int((ridx >= 0).sum()), exact=float(np.mean(dts == 0)),
                                       max_dt_ms=float(dts.max()))))
    return D["ridx"][rows]


# ============================================================================ press2 labels, finger negatives
def press2_recordings():
    recs = load()["rpl"]["recs"]; f = recs.file.str.split("/").str[-1]
    m = (recs.folder == "TAP") & (f.str.contains("DoubleTap") | f.str.contains("240bpm"))
    return set(np.flatnonzero(m.to_numpy()).tolist())


def press2_rows(rows):
    """press2: in DoubleTaps and 240 bpm TAP recordings every real pointer press is a tap. Returns (keep mask over
    rows, soft rows to add, anchored replay rows to add, info): negatives within [-30, +120] ms of a press are
    dropped, the labeller's soft rows of those presses are added with full weight, and presses with neither a
    confirmed nor a soft onset get positives at [press - 5, press + 90] ms (the confirmed window; onset = press)."""
    D = load(); data = D["data"]; rpl = D["rpl"]; recs = rpl["recs"]
    people = set(data["group"][rows]); target = press2_recordings()
    on = D["onsets"]; on = on[on.rec.isin(target)]
    press = {int(r): np.sort(v.t_on.to_numpy(float)) for r, v in on[on.kind == "press"].groupby("rec")}
    lab = {int(r): np.sort(v.t_on.to_numpy(float)) for r, v in on[on.kind.isin(["strong", "soft"])].groupby("rec")}
    rr = data["rec"][rows]; tt = data["t"][rows].astype(np.float64); yy = data["y"][rows]
    keep = np.ones(len(rows), bool)
    for n_ in np.flatnonzero(np.isin(rr, list(target)) & (yy == 0)):
        pr = press.get(int(rr[n_]))
        if pr is not None and len(pr):
            d = tt[n_] - pr
            keep[n_] = not np.any((d >= -30) & (d <= 120))
    soft = D["tr_soft"]
    soft = soft[np.isin(data["group"][soft], list(people)) & np.isin(data["rec"][soft], list(target))]
    new, n_left = [], 0
    for r, pr in press.items():
        if recs.group.iloc[r] not in people:
            continue
        lb = lab.get(r, np.zeros(0))
        for o in pr:
            if len(lb) and np.min(np.abs(lb - o)) < 60:
                continue
            idx = replay_window(r, o - 5, o + 90)
            if not len(idx):
                continue
            fr = np.asarray(rpl["frames"][idx]); t_ = np.asarray(rpl["t"][idx])
            pick = idx[g.stride_pick(t_, fr >= g.TRAIN_MIN_FRAMES, g.TAP_POSITIVE_STRIDE_MS)]
            if len(pick):
                new.append(pick); n_left += 1
    extra = np.concatenate(new) if new else np.zeros(0, np.int64)
    info = dict(press2_dropped_negatives=int((~keep).sum()), press2_soft_rows=int(len(soft)),
                press2_anchored_presses=n_left, press2_anchored_rows=int(len(extra)))
    return keep, soft, extra, info


def shift_positives(X, y, files, w0, rec, ridx, d_ms):
    """Move every TAP / SWIPE positive window d_ms later: [onset - 5, onset + 90] -> [onset + d - 5, onset + 90 + d]
    (onset = the labelled onset or press the positive row belongs to). Rows that fall out at the start are dropped,
    the new tail comes from the replay (10 ms stride, frame >= 50), and negatives inside the new tail are dropped.
    The model then decides with d ms more of the tap in its features (the contact, not only the finger motion)."""
    D = load(); rpl = D["rpl"]; recs = rpl["recs"]; on = D["onsets"]
    anchors = {int(r): np.unique(np.round(v.t_on.to_numpy(float))) for r, v in
               on[on.kind.isin(["strong", "soft", "press"])].groupby("rec")}
    t = np.asarray(rpl["t"][ridx], np.float64)
    keep = np.ones(len(y), bool); new = []; drop_neg = np.zeros(len(y), bool)
    order = np.argsort(rec, kind="stable"); rs = rec[order]
    u, first = np.unique(rs, return_index=True); last = np.r_[first[1:], len(rs)]
    for r_, i0, i1 in zip(u, first, last):
        an = anchors.get(int(r_))
        if an is None or not len(an):
            continue
        sel = order[i0:i1]; tt = t[sel]; pos = y[sel] == 1
        k = np.searchsorted(an, tt + 5.0, side="right") - 1
        a_t = np.where(k >= 0, an[np.clip(k, 0, None)], np.nan); off = tt - a_t
        own = pos & (off >= -5) & (off <= 95)
        keep[sel[own & (off < d_ms - 5)]] = False
        used = np.unique(a_t[own])
        for a0 in used:
            idx = replay_window(int(r_), a0 + 90.0 + 1e-6, a0 + 90.0 + d_ms)
            if len(idx):
                fr = np.asarray(rpl["frames"][idx]); ti = np.asarray(rpl["t"][idx])
                new.append(idx[g.stride_pick(ti, fr >= g.TRAIN_MIN_FRAMES, g.TAP_POSITIVE_STRIDE_MS)])
            neg = (~pos) & (tt > a0 + 85.0) & (tt <= a0 + 95.0 + d_ms)
            drop_neg[sel[neg]] = True
    keep &= ~drop_neg
    n_drop_pos = int(((~keep) & (y == 1)).sum())
    extra = np.unique(np.concatenate(new)) if new else np.zeros(0, np.int64)
    rec_e = np.searchsorted(recs.end.to_numpy(), extra, side="right")
    X = np.concatenate([X[keep], np.asarray(rpl["X"][extra])]); y = np.r_[y[keep], np.ones(len(extra), int)]
    files = np.r_[files[keep], recs.file.to_numpy(object)[rec_e]]
    w0 = np.r_[np.asarray(w0)[keep], np.ones(len(extra), np.float32)]
    rec = np.r_[rec[keep], rec_e]; ridx = np.r_[ridx[keep], extra]
    info = dict(shift_ms=d_ms, shift_dropped_positive_rows=n_drop_pos, shift_added_rows=int(len(extra)),
                shift_dropped_negatives=int(drop_neg.sum()))
    return X, y, files, w0, rec, ridx, info


def emg_rise_table():
    """EMG confirmation of every tap anchor (confirmed / soft onset or pointer press) in TAP and SWIPE recordings:
    the largest WL of any channel in [a - 20, a + 120] ms divided by that channel's resting floor, the floor being the
    lower of the recording's and the session's (same person and date) 5th-percentile WL over rows from frame 50.
    The session floor covers 240 bpm recordings, which never rest. Cached next to the TAP cache."""
    D = load(); rpl = D["rpl"]; recs = rpl["recs"].copy(); n = rpl["X"].shape[0]
    path = CACHE / f"emg_rise_v1_{n}.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    recs["session"] = recs.group.astype(str) + "|" + recs.file.str.split("/").str[-1].str.extract(
        r"(\d{4}-\d{2}-\d{2})")[0].fillna("?")
    floor = {}
    for r, x in recs.iterrows():
        a, b = int(x.start), int(x.end)
        if b - a < 100:
            continue
        fr = np.asarray(rpl["frames"][a:b]); wl = np.asarray(rpl["X"][a:b])[:, WL["wl"]]; ok = fr >= g.TRAIN_MIN_FRAMES
        floor[r] = np.percentile(wl[ok] if ok.sum() > 50 else wl, 5, axis=0)
    sess = {}
    for s_, gg in recs.groupby("session"):
        f_ = [floor[r] for r in gg.index if r in floor]
        if f_:
            sess[s_] = np.median(np.stack(f_), axis=0)
    on = D["onsets"]; on = on[on.kind.isin(["strong", "soft", "press"])]
    rows = []
    for r, gg in on.groupby("rec"):
        r = int(r)
        if recs.folder.iloc[r] not in ("TAP", "SWIPE") or r not in floor:
            continue
        a, b = int(recs.start.iloc[r]), int(recs.end.iloc[r])
        t = np.asarray(rpl["t"][a:b], np.float64); wl = np.asarray(rpl["X"][a:b])[:, WL["wl"]]
        base = np.maximum(np.minimum(floor[r], sess[recs.session.iloc[r]]), 1.0)
        for o in np.unique(np.round(gg.t_on.to_numpy(float))):
            w1 = (t >= o - 20) & (t <= o + 120)
            if w1.any():
                rows.append((r, o, float(np.max(wl[w1].max(axis=0) / base))))
    out = pd.DataFrame(rows, columns=["rec", "t", "rise"])
    try:
        out.to_csv(path, index=False, compression="gzip")
    except OSError:
        pass
    return out


def filter_positives(X, y, files, w0, rec, ridx, emg_min=None, win_ms=None, recq=None):
    """Tighter TAP positives (rows left out of training, never turned into negatives):
    emg_min: TAP-folder positives whose tap shows less than this EMG rise (emg_rise_table) are dropped;
    win_ms : positives later than win_ms after their onset are dropped (onset window [-5, win_ms] instead of 90);
    recq   : TAP recordings (not DoubleTaps / 240 bpm, whose second and fast taps are weak by nature) where the labeller
             confirmed less than this share of the presses (>= 5 presses) lose all their positives: a weak recording."""
    D = load(); rpl = D["rpl"]; recs = rpl["recs"]; on = D["onsets"]
    weak_recs = set()
    if recq is not None:
        n_p = on[on.kind == "press"].groupby("rec").size(); n_s = on[on.kind == "strong"].groupby("rec").size()
        share = (n_s.reindex(n_p.index).fillna(0) / n_p)[n_p >= 5]
        target = press2_recordings()
        weak_recs = {int(r) for r, v in share.items()
                     if v < recq and recs.folder.iloc[int(r)] == "TAP" and int(r) not in target}
    anchors = {int(r): np.unique(np.round(v.t_on.to_numpy(float))) for r, v in
               on[on.kind.isin(["strong", "soft", "press"])].groupby("rec")}
    rise = {}
    if emg_min is not None:
        tab = emg_rise_table()
        rise = {(int(r), float(t_)): v for r, t_, v in zip(tab.rec, tab.t, tab.rise)}
    t = np.asarray(rpl["t"][ridx], np.float64); folder = recs.folder.to_numpy(object)
    keep = np.ones(len(y), bool); n_emg = n_win = 0; dropped_taps = set()
    n_recq = 0
    for i in np.flatnonzero(y == 1):
        if int(rec[i]) in weak_recs:
            keep[i] = False; n_recq += 1; continue
        an = anchors.get(int(rec[i]))
        if an is None or not len(an):
            continue
        k = np.searchsorted(an, t[i] + 5.0, side="right") - 1
        if k < 0 or t[i] - an[k] > 95:
            continue
        a0 = float(an[k]); off = t[i] - a0
        if win_ms is not None and off > win_ms:
            keep[i] = False; n_win += 1; continue
        if emg_min is not None and folder[rec[i]] == "TAP" and rise.get((int(rec[i]), a0), np.inf) < emg_min:
            keep[i] = False; n_emg += 1; dropped_taps.add((int(rec[i]), a0))
    info = dict(emg_min=emg_min, win_ms=win_ms, emg_dropped_rows=n_emg, emg_dropped_taps=len(dropped_taps),
                win_dropped_rows=n_win, recq=recq, recq_recordings=len(weak_recs), recq_dropped_rows=n_recq)
    return X[keep], y[keep], files[keep], np.asarray(w0)[keep], rec[keep], ridx[keep], info


OBS_TEACHER = dict(objective="binary", n_estimators=300, learning_rate=0.08, num_leaves=31, max_depth=8,
                   min_child_samples=100, reg_lambda=1.0, reg_alpha=1e-3, colsample_bytree=0.7, subsample=0.8,
                   subsample_freq=1, max_bin=127, verbosity=-1, deterministic=True)
OBS_SPEC = ("acc", "emgraw", "diff", "wlratio", "litediff")


def observable_scores(X150, y, rec, ridx, fold, tag):
    """Cross-fitted teacher score of every positive training row of this fold: a full-size model (no budget) on the
    fold's training rows (same labels, nominal inputs), trained without the row's person (3 person folds inside the
    fold's training people). High = this frame looks like a tap and unlike everything else, to a model that never
    saw this person. Cached per fold and label set."""
    import hashlib
    import lightgbm as lgb
    D = load(); recs = D["rpl"]["recs"]
    key = hashlib.sha256(np.ascontiguousarray(ridx).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()[:12]
    path = OUT / "models" / f"obs_teacher_{tag}_fold{fold}_{key}.npz"
    if path.is_file():
        with np.load(path) as z:
            return z["score"]
    t0 = time.perf_counter()
    person = recs.group.to_numpy(object)[rec]
    people = np.unique(person)
    F, _, _ = build(np.hstack([X150, lite_rows(ridx, rec)]), OBS_SPEC)
    w = weights(y, recs.file.to_numpy(object)[rec], np.ones(len(y), np.float32))
    score = np.full(len(y), np.nan)
    for k, (a_, b_) in enumerate(GroupKFold(n_splits=3).split(people, groups=people)):
        fit_rows = np.isin(person, people[a_]); lab = np.isin(person, people[b_]) & (y == 1)
        m = lgb.LGBMClassifier(**OBS_TEACHER, n_jobs=THREADS, random_state=11 + k).fit(F[fit_rows], y[fit_rows],
                                                                                      sample_weight=w[fit_rows])
        score[lab] = m.predict_proba(F[lab])[:, 1]
        log("GR_OBS_TEACHER", fold, k, json.dumps(dict(fit_rows=int(fit_rows.sum()), scored=int(lab.sum()),
                                                       seconds=round(time.perf_counter() - t0, 1))))
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, score=score)
    return score


def observable_positives(X150, y, files, w0, rec, ridx, fold, mode, dur_ms, tag="press2"):
    """Observable-pattern labels: every tap keeps only the dur_ms of its positive rows where the cross-fitted teacher
    finds it most distinguishable (mode 'max': the window with the highest mean score; 'first': the earliest window
    whose mean reaches 90% of that maximum); its other positive rows are left out (never made negatives). Every tap
    keeps a window, so soft / second / fast taps stay in training."""
    D = load(); rpl = D["rpl"]; on = D["onsets"]
    score = observable_scores(X150, y, rec, ridx, fold, tag)
    anchors = {int(r): np.unique(np.round(v.t_on.to_numpy(float))) for r, v in
               on[on.kind.isin(["strong", "soft", "press"])].groupby("rec")}
    t = np.asarray(rpl["t"][ridx], np.float64)
    pos = np.flatnonzero(y == 1)
    tap = np.full(len(y), -1, np.int64); off = np.full(len(y), np.nan)
    owner = {}
    for i in pos:
        an = anchors.get(int(rec[i]))
        if an is None or not len(an):
            continue
        k = np.searchsorted(an, t[i] + 5.0, side="right") - 1
        if k < 0 or t[i] - an[k] > 95:
            continue
        key = (int(rec[i]), float(an[k])); tap[i] = owner.setdefault(key, len(owner)); off[i] = t[i] - an[k]
    keep = np.ones(len(y), bool); n_left = 0; starts = []
    order = np.lexsort((off[pos], tap[pos])); pp = pos[order]
    bounds = np.flatnonzero(np.diff(np.r_[-2, tap[pp], -3]) != 0)
    for a_, b_ in zip(bounds[:-1], bounds[1:]):
        idx = pp[a_:b_]
        if tap[idx[0]] < 0 or len(idx) < 2:
            continue
        ot = off[idx]; sc = np.nan_to_num(score[idx], nan=0.0)
        cnt = np.array([int(((ot >= o) & (ot < o + dur_ms)).sum()) for o in ot])
        full = cnt >= min(max(1, int(round(dur_ms / g.TAP_POSITIVE_STRIDE_MS))), int(cnt.max()))
        means = np.array([sc[(ot >= o) & (ot < o + dur_ms)].mean() if ok_ else -1.0 for o, ok_ in zip(ot, full)])
        if mode == "first":
            j = int(np.flatnonzero(means >= 0.9 * means.max())[0])
        else:
            j = int(np.argmax(means))
        inside = (ot >= ot[j]) & (ot < ot[j] + dur_ms)
        keep[idx[~inside]] = False; n_left += int((~inside).sum()); starts.append(ot[j])
    st = np.asarray(starts) if starts else np.zeros(1)
    info = dict(obs_mode=mode, obs_ms=dur_ms, obs_taps=len(starts), obs_rows_left_out=n_left,
                obs_start_ms_p10_p50_p90=[float(np.percentile(st, q)) for q in (10, 50, 90)])
    return X150[keep], y[keep], files[keep], np.asarray(w0)[keep], rec[keep], ridx[keep], info


def strict_taps(X150, y, files, w0, rec, ridx, fold, trn, q, exempt_press2=False):
    """Strict tap definition: taps that look like finger motion are not taught. A teacher, cross-fitted by person
    inside the fold's training people (3 folds), separates tap rows from muscle-activity rows of the finger-motion /
    fist recordings; every tap gets its best (max) score over its positive rows; the q% of taps with the lowest
    score (most finger-like) are left out of training (rows removed, never made negatives). exempt_press2 keeps
    every tap of the DoubleTaps / 240 bpm recordings (the firmware handles second / fast taps separately) and every
    swipe (a swipe is a finger motion by nature): only single taps are judged."""
    import hashlib
    import lightgbm as lgb
    D = load(); data = D["data"]; recs = D["rpl"]["recs"]; on = D["onsets"]
    act = data_ridx(trn[data["kind"][trn] == 2])
    folder = recs.folder.to_numpy(object)[rec]
    look = (y == 0) & (folder == "FINGER") & np.isin(ridx, act)
    pos = y == 1
    key = hashlib.sha256(np.ascontiguousarray(ridx).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()[:12]
    path = OUT / "models" / f"strict_teacher_fold{fold}_{key}.npz"
    if path.is_file():
        with np.load(path) as z:
            score = z["score"]
    else:
        t0 = time.perf_counter()
        person = recs.group.to_numpy(object)[rec]
        people = np.unique(person[pos | look])
        F, _, _ = build(np.hstack([X150, lite_rows(ridx, rec)]), OBS_SPEC)
        score = np.full(len(y), np.nan)
        for k, (a_, b_) in enumerate(GroupKFold(n_splits=3).split(people, groups=people)):
            fit = (pos | look) & np.isin(person, people[a_]); lab = pos & np.isin(person, people[b_])
            if (look & fit).sum() < 100:           # no finger-motion people in this inner fold: no judgement
                score[lab] = 1.0
                continue
            yy = y[fit]; w = np.where(yy == 1, 0.5 / max(1, yy.sum()), 0.5 / max(1, (yy == 0).sum())) * len(yy)
            m = lgb.LGBMClassifier(**OBS_TEACHER, n_jobs=THREADS, random_state=21 + k).fit(F[fit], yy, sample_weight=w)
            score[lab] = m.predict_proba(F[lab])[:, 1]
            log("GR_STRICT_TEACHER", fold, k, json.dumps(dict(fit_rows=int(fit.sum()), finger_rows=int((look & fit).sum()),
                                                            scored=int(lab.sum()), seconds=round(time.perf_counter() - t0, 1))))
        (OUT / "models").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, score=score)
    anchors = {int(r): np.unique(np.round(v.t_on.to_numpy(float))) for r, v in
               on[on.kind.isin(["strong", "soft", "press"])].groupby("rec")}
    t = np.asarray(D["rpl"]["t"][ridx], np.float64)
    owner = np.full(len(y), -1, np.int64); ids = {}
    for i in np.flatnonzero(pos):
        an = anchors.get(int(rec[i]))
        if an is None or not len(an):
            continue
        k = np.searchsorted(an, t[i] + 5.0, side="right") - 1
        if k >= 0 and t[i] - an[k] <= 95:
            owner[i] = ids.setdefault((int(rec[i]), float(an[k])), len(ids))
    o = owner[owner >= 0]; sc = np.nan_to_num(score[owner >= 0], nan=0.0)
    best = np.full(len(ids), -1.0); np.maximum.at(best, o, sc)
    tap_rec = np.zeros(len(ids), np.int64)
    for (r_, _), i_ in ids.items():
        tap_rec[i_] = r_
    eligible = np.ones(len(ids), bool)
    if exempt_press2:
        names = recs.file.to_numpy(object)
        eligible = np.array([recs.folder.iloc[r_] == "TAP" and not any(k_ in names[r_] for k_ in g.PRESS2_NAME_KEYS)
                             for r_ in tap_rec])
    thr = np.percentile(best[eligible], q) if eligible.any() else -1
    drop_tap = eligible & (best < thr)
    drop = (owner >= 0) & drop_tap[np.clip(owner, 0, None)]
    keep = ~drop
    kinds = [rec_type(recs.folder.iloc[r_], recs.file.iloc[r_]) for r_ in tap_rec[drop_tap]]
    info = dict(strict_q=q, strict_exempt_press2=exempt_press2, strict_taps=int(len(ids)), strict_taps_left_out=int(drop_tap.sum()),
                strict_rows_left_out=int(drop.sum()), strict_threshold=float(thr),
                strict_left_out_by_type=pd.Series(kinds).value_counts().to_dict() if kinds else {})
    return X150[keep], y[keep], files[keep], np.asarray(w0)[keep], rec[keep], ridx[keep], info


CONS_LOOKALIKE = ("FINGER", "WRIST", "ARM", "BAND")


def consistent_features(F, y, person, lookalike, k):
    """Inputs that separate taps from look-alikes in most people (fold-local: the fold's training rows only).
    For each training person with >= 30 tap rows and >= 30 look-alike rows: the univariate AUC of every input,
    taps vs that person's look-alike rows (muscle activity in finger / fist / wrist / arm / band recordings), oriented
    by the pooled direction. Score = the 25th percentile over people (works for most people, not on average);
    the k best inputs are kept. Returns (column indices, per-input scores, people used)."""
    from scipy.stats import rankdata
    pos_all = y == 1
    pooled = []
    def auc(a, b):                                     # Mann-Whitney AUC of every column, a = taps, b = look-alikes
        r = rankdata(np.vstack([a, b]), axis=0)
        return (r[:len(a)].sum(0) - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    rng = np.random.default_rng(1)
    def sub(m, n=4000):
        idx = np.flatnonzero(m)
        return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))
    direction = np.sign(auc(F[sub(pos_all, 20000)], F[sub(lookalike, 20000)]) - 0.5)
    direction[direction == 0] = 1
    per = []
    people = [p_ for p_ in np.unique(person) if (pos_all & (person == p_)).sum() >= 30
              and (lookalike & (person == p_)).sum() >= 30]
    for p_ in people:
        a_ = auc(F[sub(pos_all & (person == p_))], F[sub(lookalike & (person == p_))])
        per.append(0.5 + direction * (a_ - 0.5))
    per = np.asarray(per)
    score = np.percentile(per, 25, axis=0)
    return np.sort(np.argsort(-score)[:k]), score, people


MINE_FROM = HERE / "results_tap_generalization" / "models"


REL_AFTER_ONSET_MS = 150      # release part: from onset + 150 ms (past the firmware's refractory) ...
PRE_BEFORE_ONSET_MS = 40      # lead-in part: ... up to onset - 40 ms (margin for onset timing)
NEXT_EVENT_GUARD_MS = 100     # never within 100 ms before another press / tap, or 250 ms after one (lead-in)


def press_regions(rec, parts):
    """Replay row indices of the unknown press around each labelled tap of one TAP / SWIPE recording: "rel" = the
    unknown run after the tap's positive window, from onset + 150 ms; "pre" = the unknown run before the onset, up to
    onset - 40 ms. Guards keep other presses / taps (and the 250 ms after them) out."""
    D = load(); rpl = D["rpl"]; recs = rpl["recs"]; on = D["onsets"]
    a, b = int(recs.start.iloc[rec]), int(recs.end.iloc[rec])
    lab = np.asarray(rpl["label"][a:b]); tt = np.asarray(rpl["t"][a:b])
    o = on[on.rec == rec]; conf = np.sort(o[o.kind != "press"].t_on.to_numpy(float))
    ev = np.sort(o.t_on.to_numpy(float)); out = []
    for t0 in conf:
        i = np.searchsorted(tt, t0)
        if "rel" in parts:                       # walk past the positive window, then along the unknown run
            j = i
            while j < len(tt) and lab[j] == 1: j += 1
            k = j
            while k < len(tt) and lab[k] == -1: k += 1
            seg = np.arange(j, k)[tt[j:k] >= t0 + REL_AFTER_ONSET_MS]
            nxt = ev[ev > t0 + 1.0]
            if len(nxt): seg = seg[tt[seg] < nxt[0] - NEXT_EVENT_GUARD_MS]
            out.append(seg)
        if "pre" in parts:
            k = i
            while k > 0 and lab[k - 1] == -1: k -= 1
            seg = np.arange(k, i)[tt[k:i] <= t0 - PRE_BEFORE_ONSET_MS]
            prv = ev[ev < t0 - 1.0]
            if len(prv): seg = seg[tt[seg] > prv[-1] + 250]
            out.append(seg)
    return (np.unique(np.concatenate(out)) + a) if out else np.zeros(0, np.int64)


def press_rows(groups, parts, stride=6):
    """press_regions of every TAP / SWIPE recording of the training groups, every stride-th row, frame >= 50."""
    D = load(); rpl = D["rpl"]; recs = rpl["recs"]; fr = np.asarray(rpl["frames"]); out = []
    skip = press2_recordings() if "single" in parts else set()   # double / 240 bpm: second taps can be unlabelled
    for r in np.flatnonzero(recs.folder.isin(["TAP", "SWIPE"]).to_numpy() & recs.group.isin(groups).to_numpy()):
        if int(r) in skip:
            continue
        idx = press_regions(int(r), parts); idx = idx[fr[idx] >= 50]
        out.append(idx[::stride])
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)


def background_rows(groups, min_ms, stride=6):
    """Replay rows of TAP / SWIPE recordings (training groups) labelled unknown and > min_ms from every pointer press
    and labelled tap onset: rest time between taps. Every stride-th row (the training cache's density), frame >= 50."""
    D = load(); rpl = D["rpl"]; recs = rpl["recs"]; on = D["onsets"]
    ev = {int(r): np.sort(v.t_on.to_numpy(float)) for r, v in on.groupby("rec")}
    lab = np.asarray(rpl["label"]); tt = np.asarray(rpl["t"]); fr = np.asarray(rpl["frames"]); out = []
    for r in np.flatnonzero(recs.folder.isin(["TAP", "SWIPE"]).to_numpy() & recs.group.isin(groups).to_numpy()):
        a, b = int(recs.start.iloc[r]), int(recs.end.iloc[r])
        idx = np.arange(a, b)[(lab[a:b] == -1) & (fr[a:b] >= 50)]
        e = ev.get(int(r), np.zeros(0))
        if len(e) and len(idx):
            j = np.searchsorted(e, tt[idx]); d = np.minimum(np.abs(tt[idx] - e[np.clip(j - 1, 0, len(e) - 1)]),
                                                         np.abs(tt[idx] - e[np.clip(j, 0, len(e) - 1)]))
            idx = idx[d > min_ms]
        out.append(idx[::stride])
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)


def mine_finger(people, fold, thr=0.3, stride_ms=20, cap=40000, base="raw126_diff+augm"):
    """Hard negatives: FINGER-folder replay rows (finger motion and fist, frame >= 50) of these training people that
    the baseline fold model (trained on these people) still scores >= thr, one per stride_ms, at most cap."""
    import lightgbm as lgb
    D = load(); rpl = D["rpl"]; recs = rpl["recs"]
    m = lgb.Booster(model_file=str(MINE_FROM / f"{base}_fold{fold}.txt"))
    spec = ARMS["raw126_diff"]["spec"]; picks = []
    for i, r in recs[recs.group.isin(people) & (recs.folder == "FINGER")].iterrows():
        a, b = int(r.start), int(r.end)
        if b <= a:
            continue
        fr = np.asarray(rpl["frames"][a:b]); ok = fr >= g.TRAIN_MIN_FRAMES
        if not ok.any():
            continue
        p = m.predict(build(np.asarray(rpl["X"][a:b]), spec)[0], num_threads=THREADS)
        cand = ok & (p >= thr)
        if cand.any():
            picks.append(a + g.stride_pick(np.asarray(rpl["t"][a:b]), cand, stride_ms))
    idx = np.concatenate(picks) if picks else np.zeros(0, np.int64)
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(100 + fold).choice(idx, cap, replace=False))
    return idx


# ============================================================================ data
_D = {}


class _Rows:
    """Row view over the full cache's replay matrix and the added ones: slices and integer / boolean arrays."""

    def __init__(self, parts):
        self.parts = parts; self.off = np.cumsum([0] + [p_.shape[0] for p_ in parts])
        self.shape = (int(self.off[-1]), parts[0].shape[1])

    def __getitem__(self, k):
        if isinstance(k, slice):
            a, b, st = k.indices(self.shape[0])
            if st != 1:
                return self[np.arange(a, b, st)]
            out = [np.asarray(p_[max(a, lo) - lo:min(b, hi) - lo]) for p_, lo, hi in
                   zip(self.parts, self.off[:-1], self.off[1:]) if max(a, lo) < min(b, hi)]
            return out[0] if len(out) == 1 else (np.concatenate(out) if out else np.zeros((0, self.shape[1]), np.float32))
        idx = np.asarray(k)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        out = np.empty((len(idx), self.shape[1]), np.float32); part = np.searchsorted(self.off, idx, side="right") - 1
        for i, p_ in enumerate(self.parts):
            m = part == i
            if m.any():
                out[m] = np.asarray(p_[idx[m] - self.off[i]])
        return out


def _add_extra(data, rpl, onsets, cache):
    """Append an extra cache (same columns) to the loaded data: recording ids and replay rows are offset."""
    d2, r2, _ = g.load_cache(cache, COLS)
    n_rec, n_rows = len(rpl["recs"]), rpl["X"].shape[0]
    for k in ("X", "y", "kind", "t", "frames", "group", "file", "category", "w"):
        data[k] = np.concatenate([data[k], d2[k]])
    data["rec"] = np.concatenate([data["rec"], d2["rec"] + n_rec])
    for k in ("t", "label", "frames", "ready", "steady"):
        rpl[k] = np.concatenate([rpl[k], r2[k]])
    rpl["X"] = _Rows((rpl["X"].parts if isinstance(rpl["X"], _Rows) else [rpl["X"]]) + [r2["X"]])
    rr = r2["recs"].copy(); ok = rr.start >= 0
    rr.loc[ok, "start"] += n_rows; rr.loc[ok, "end"] += n_rows
    rpl["recs"] = pd.concat([rpl["recs"], rr], ignore_index=True)
    rpl["onsets"].update({k + n_rec: v for k, v in r2["onsets"].items()})
    for name, d_ in r2["extra"].items():
        rpl["extra"].setdefault(name, {}).update({k + n_rec: v for k, v in d_.items()})
    o2 = pd.read_csv(cache / "tap_onsets.csv"); o2["rec"] += n_rec
    return pd.concat([onsets, o2], ignore_index=True), len(d2["y"])


def load():
    if _D:
        return _D
    t0 = time.perf_counter()
    data, rpl, _ = g.load_cache(CACHE, COLS)
    assert data["X"].shape[1] == 150, data["X"].shape
    if SPLIT == "file":                                             # one group per recording, stratified by type
        g.group_by_file(data, rpl)
    tr, _te = g.split_heldout(data, "tap")
    tr_soft = tr[data["kind"][tr] == 6]                             # soft rows (only press2 uses them)
    tr = tr[data["kind"][tr] != 6]                                  # confirmed labels only (no soft rows)
    folds = (g.cv_splits(data, tr) if SPLIT == "file" else
             list(GroupKFold(n_splits=g.CV_FOLDS).split(tr, data["y"][tr], data["group"][tr])))
    onsets = pd.read_csv(CACHE / "tap_onsets.csv")
    _D.update(n_base=len(rpl["recs"]), n_base_rows=rpl["X"].shape[0])
    for ex in EXTRA:                                                # added recordings (person folds unchanged)
        n0 = len(data["y"])
        onsets, n_new = _add_extra(data, rpl, onsets, ex)
        new = np.arange(n0, n0 + n_new)
        new = new[data["kind"][new] != 6]
        if EXTRA_MODE == "train":                                   # in every training fold, never validated
            pos = np.arange(len(tr), len(tr) + len(new)); tr = np.r_[tr, new]
            folds = [(np.r_[a_, pos], b_) for a_, b_ in folds]
        log("GR_EXTRA", ex, EXTRA_MODE, json.dumps(dict(train_rows=int(len(new)),
                                                        people=sorted(set(data["group"][n0:n0 + n_new])))))
    if QUICK:
        people = sorted(set(data["group"][tr]))[:12]
        tr = tr[np.isin(data["group"][tr], people)]
        folds = list(GroupKFold(n_splits=3).split(tr, data["y"][tr], data["group"][tr]))
    _D.update(data=data, rpl=rpl, tr=tr, tr_soft=tr_soft, folds=folds, onsets=onsets, n_recs=len(rpl["recs"]))
    log("GR_DATA", json.dumps(dict(train_rows=int(len(tr)), people=len(set(data["group"][tr])),
                                   replay_rows=int(rpl["X"].shape[0]), recordings=len(rpl["recs"]),
                                   seconds=round(time.perf_counter() - t0, 1))))
    return _D


class Val:
    """Steady replay rows of the validation people (all 150 columns), firmware replay with the original labels."""

    def __init__(self, people):
        D = load(); rpl = D["rpl"]; recs = rpl["recs"]
        self.items, chunks, rec_ids = [], [], []
        for i, r in recs[recs.group.isin(people)].iterrows():
            a, b = int(r.start), int(r.end)
            if b <= a:
                continue
            local = np.asarray(rpl["ready"][a:b], bool)
            if not local.any():
                continue
            a += int(np.flatnonzero(local)[0])
            ons = np.asarray(rpl["onsets"].get(int(i), []), np.float64)
            t_a = float(rpl["t"][a]); ons = ons[ons >= t_a + R.TAP_HIT_BEFORE_MS]
            extra = {}
            for name, per_rec in (rpl.get("extra") or {}).items():
                lo, hi = per_rec.get(int(i), (np.zeros(0), np.zeros(0)))
                keep = lo >= t_a; extra[name] = (lo[keep], hi[keep])
            self.items.append((int(i), a, b, r.folder, ons, extra))
            chunks.append(np.asarray(rpl["X"][a:b])); rec_ids.append(np.full(b - a, int(i), np.int32))
        self.X = np.concatenate(chunks) if chunks else np.zeros((0, 150), np.float32)
        self.rec = np.concatenate(rec_ids) if rec_ids else np.zeros(0, np.int32)
        self.ridx = (np.concatenate([np.arange(a, b) for _, a, b, *_ in self.items]) if self.items
                     else np.zeros(0, np.int64))
        self.seg, pos = [], 0
        for _, a, b, *_ in self.items:
            self.seg.append((pos, pos + b - a)); pos += b - a
        self.t = [np.asarray(rpl["t"][a:b], np.float64) for _, a, b, *_ in self.items]
        self.label = [np.asarray(rpl["label"][a:b]) for _, a, b, *_ in self.items]
        self.steady_from = []                       # first steady row (frame >= 500) of each recording, or None
        for _, a, b, *_ in self.items:
            nz = np.flatnonzero(np.asarray(rpl["steady"][a:b], bool))
            self.steady_from.append(int(nz[0]) if len(nz) else None)

    def predict(self, model, k, spec, stress=None, chunk=250_000, cols=None, post=None):
        """model: LGBMClassifier or Booster (then k is ignored: the saved model is already cut)."""
        booster = getattr(model, "booster_", model)
        lite = lite_rows(self.ridx, self.rec, stress) if ("litediff" in spec or "litediff4" in spec) else None
        p = np.empty(len(self.X), np.float64)
        for s in range(0, len(self.X), chunk):
            Xb = self.X[s:s + chunk]
            if stress:
                r = self.rec[s:s + chunk]
                Xb = apply_emg(Xb, **{name: tab[r] for name, tab in stress.items()})
            if lite is not None:
                Xb = np.hstack([Xb, lite[s:s + chunk]])
            F, _, _ = build(Xb, spec)
            if cols is not None:
                F = F[:, cols]
            if post is not None:
                F = post(F)
            p[s:s + chunk] = (booster.predict(F, num_iteration=k, num_threads=THREADS) if hasattr(model, "booster_")
                              else booster.predict(F, num_threads=THREADS))
        return p

    def counts(self, p, threshold, steady=False):
        c = R.TapCounts()
        for (s, e), t, lab, (_, _, _, folder, ons, extra), a in zip(self.seg, self.t, self.label, self.items,
                                                                    self.steady_from):
            if steady:
                if a is None:
                    continue
                ons = ons[ons >= t[a] + R.TAP_HIT_BEFORE_MS]
                extra = {k: (lo[lo >= t[a]], hi[lo >= t[a]]) for k, (lo, hi) in extra.items()}
            else:
                a = 0
            R.tap_replay_recording(p[s + a:e] >= threshold, t[a:], lab[a:], ons, folder, c, extra=extra)
        return c


# ============================================================================ labels
def weights(y, files, w0):
    """train_unified.weights_for_task(task='tap', positive_weight=1): file-normalized, then class-balanced."""
    w = np.asarray(w0, np.float32).copy()
    fc = Counter(files)
    w /= np.asarray([np.sqrt(fc[f]) for f in files], np.float32)
    for cls in (0, 1):
        m = y == cls; w[m] *= len(y) / (2.0 * float(w[m].sum()))
    return w


def strong_onsets(recs_of_people):
    o = load()["onsets"]
    o = o[(o.kind == "strong") & o.rec.isin(recs_of_people)]
    return o[["rec", "t_on"]].to_numpy()


def assign_positives(rows):
    """Positive train row -> its confirmed onset (rec, t_on) index into strong onsets, or -1."""
    D = load(); data = D["data"]
    o = D["onsets"]; o = o[o.kind == "strong"]
    by_rec = {r: np.sort(v.t_on.to_numpy(float)) for r, v in o.groupby("rec")}
    keys = np.full(len(rows), -1, np.int64); key_t = np.full(len(rows), np.nan)
    for n, i in enumerate(rows):
        ts = by_rec.get(int(data["rec"][i]))
        if ts is None:
            continue
        d = data["t"][i] - ts
        ok = np.flatnonzero((d >= -5) & (d <= 90))
        if len(ok):
            key_t[n] = ts[ok[-1]]; keys[n] = int(data["rec"][i])
    return keys, key_t


def replay_window(rec, lo, hi):
    """Replay row indices of a recording between lo and hi ms."""
    rpl = load()["rpl"]; r = rpl["recs"].iloc[int(rec)]
    a, b = int(r.start), int(r.end)
    t = np.asarray(rpl["t"][a:b], np.float64)
    i0, i1 = np.searchsorted(t, lo, "left"), np.searchsorted(t, hi, "right")
    return a + np.arange(i0, i1)


def acc_evidence(rec, t_on):
    """ACC impulse in the label window: max of sum log1p(acc_psd[0]) over [t_on, t_on+80] minus its median over
    [t_on-400, t_on-200] (the pre-tap baseline)."""
    rpl = load()["rpl"]
    idx = replay_window(rec, t_on - 400, t_on + 80)
    if len(idx) < 8:
        return np.nan
    t = np.asarray(rpl["t"][idx[0]:idx[-1] + 1]); X = np.asarray(rpl["X"][idx[0]:idx[-1] + 1][:, ACC_NOW])
    s = np.log1p(np.clip(X, 0, None)).sum(1)
    base = s[(t >= t_on - 400) & (t < t_on - 200)]; win = s[(t >= t_on) & (t <= t_on + 80)]
    return float(win.max() - np.median(base)) if len(base) >= 5 and len(win) else np.nan


_ACC_THR = {}


def acc_threshold(people):
    """95th percentile of the same statistic at random instants of the negative recordings of these people."""
    key = tuple(sorted(people))
    if key in _ACC_THR:
        return _ACC_THR[key]
    rpl = load()["rpl"]; recs = rpl["recs"]; rng = np.random.default_rng(5)
    vals = []
    for i, r in recs[recs.group.isin(people) & recs.folder.isin(NEG_FOLDERS)].iterrows():
        t0, t1 = float(rpl["t"][int(r.start)]), float(rpl["t"][int(r.end) - 1])
        for t_on in rng.uniform(t0 + 1000, t1 - 500, 8) if t1 - t0 > 2000 else []:
            v = acc_evidence(i, t_on)
            if v == v:
                vals.append(v)
    _ACC_THR[key] = float(np.percentile(vals, 95))
    return _ACC_THR[key]


def teacher_labels(train_rows, fold):
    """ACC-only, pure-TAP teacher, cross-fitted by person inside the fold's training people. For every confirmed
    TAP/SWIPE onset of the training people: (keep, peak time) from a teacher that never saw that person."""
    D = load(); data = D["data"]; rpl = D["rpl"]; recs = rpl["recs"]
    people = np.unique(data["group"][train_rows])
    inner = list(GroupKFold(n_splits=3).split(people, groups=people))
    peaks_all, onset_peaks = [], {}
    for k, (a_, b_) in enumerate(inner):
        fit_people, lab_people = set(people[a_]), set(people[b_])
        rows = train_rows[np.isin(data["group"][train_rows], list(fit_people))]
        rows = rows[~((data["y"][rows] == 1) & (data["category"][rows] == "SWIPE"))]       # pure TAP teacher
        F, _, _ = build(data["X"][rows], ("acc",))
        y = data["y"][rows].astype(int)
        m = fit(F, y, weights(y, data["file"][rows], data["w"][rows]), budget=None, trees=200)
        # The teacher's peak over +-150 ms at random instants of these people's movement / no-gesture recordings:
        # pooled over the inner folds, its 95th percentile is the keep threshold (a learned version of acc_rule).
        rng = np.random.default_rng(11 + k); n_mov = 0
        for i, r in recs[recs.group.isin(lab_people) & recs.folder.isin(NEG_FOLDERS)].iterrows():
            t0_, t1_ = float(rpl["t"][int(r.start)]), float(rpl["t"][int(r.end) - 1])
            for tt in (rng.uniform(t0_ + 1000, t1_ - 500, 8) if t1_ - t0_ > 2000 else []):
                idx = replay_window(i, tt - 150, tt + 150)
                if len(idx):
                    Fw, _, _ = build(np.asarray(rpl["X"][idx[0]:idx[-1] + 1]), ("acc",))
                    peaks_all.append(float(m.booster_.predict(Fw, num_threads=THREADS).max())); n_mov += 1
        lab_recs = recs.index[recs.group.isin(lab_people) & recs.folder.isin(("TAP", "SWIPE"))]
        for rec, t_on in strong_onsets(lab_recs):
            idx = replay_window(rec, t_on - 150, t_on + 150)
            if not len(idx):
                onset_peaks[(int(rec), float(t_on))] = (np.nan, 0.0); continue
            Fw, _, _ = build(np.asarray(rpl["X"][idx[0]:idx[-1] + 1]), ("acc",))
            pr = m.booster_.predict(Fw, num_threads=THREADS)
            j = int(np.argmax(pr))
            onset_peaks[(int(rec), float(t_on))] = (float(rpl["t"][idx[0] + j]), float(pr[j]))
        log("GR_TEACHER", fold, k, json.dumps(dict(fit_rows=int(len(rows)), labelled=len(onset_peaks), movement_windows=n_mov)))
    thr = float(np.percentile(peaks_all, 95)) if len(peaks_all) >= 50 else TEACHER_THRESHOLD
    log("GR_TEACHER_THRESHOLD", fold, round(thr, 4), len(peaks_all))
    return {key: (bool(pk >= thr), t_pk, pk) for key, (t_pk, pk) in onset_peaks.items()}


def train_set(rows, labels):
    """(X150, y, files, w0, rec) for the fold's training rows under a label variant, plus a label report."""
    D = load(); data = D["data"]; rpl = D["rpl"]; recs = rpl["recs"]
    y = data["y"][rows].astype(int); cat = data["category"][rows]
    info = {}
    soft_add = np.zeros(0, np.int64)
    if labels == "default":
        keep = np.ones(len(rows), bool); extra = None
    elif labels == "press2":
        keep, soft_add, extra, inf = press2_rows(rows); info.update(inf)
    elif labels == "pure_tap":
        keep = ~((y == 1) & (cat == "SWIPE")); extra = None
    elif labels in ("acc_rule", "teacher", "teacher_anchor"):
        keys, key_t = assign_positives(rows)
        pos = y == 1
        info["positives_without_onset"] = int((pos & (keys < 0)).sum())
        if labels == "acc_rule":
            thr = acc_threshold(set(data["group"][rows]))
            decision = {}
            for r, t in set(zip(keys[pos & (keys >= 0)], key_t[pos & (keys >= 0)])):
                v = acc_evidence(r, t); decision[(int(r), float(t))] = (bool(v == v and v >= thr), t, v)
            info["acc_threshold"] = thr
        else:
            decision = D["teacher"][D["fold"]]
        ok = np.array([decision.get((int(r), float(t)), (False,))[0] if r >= 0 else False for r, t in zip(keys, key_t)])
        keep = ~pos | ok
        for folder in ("TAP", "SWIPE"):
            on = [(r, t) for (r, t), v in decision.items() if recs.folder.iloc[r] == folder]
            info[f"kept_{folder}"] = f"{sum(decision[q][0] for q in on)}/{len(on)}"
        extra = None
        if labels == "teacher_anchor":
            keep = ~pos                                               # all positives replaced by re-anchored ones
            people = set(data["group"][rows])
            new = []
            for (r, t), (ok_, t_pk, _) in decision.items():
                if not ok_ or recs.group.iloc[r] not in people:
                    continue
                idx = replay_window(r, t_pk, t_pk + 80)
                if not len(idx):
                    continue
                tt = np.asarray(rpl["t"][idx]); fr = np.asarray(rpl["frames"][idx])
                pick = idx[g.stride_pick(tt, fr >= g.TRAIN_MIN_FRAMES, g.TAP_POSITIVE_STRIDE_MS)]
                new.append(pick)
            extra = np.concatenate(new) if new else np.zeros(0, np.int64)
            info["anchored_positive_rows"] = int(len(extra))
    else:
        raise ValueError(labels)
    r = np.r_[rows[keep], soft_add]
    X = data["X"][r]; yy = data["y"][r].astype(int); files = data["file"][r]; w0 = data["w"][r].copy()
    rec = data["rec"][r]; ridx = data_ridx(r)
    w0[data["kind"][r] == 6] = 1.0                                  # press2: the press is trusted (full weight)
    if extra is not None and len(extra):
        extra = np.sort(extra)
        Xe = np.asarray(rpl["X"][extra])
        rec_e = np.searchsorted(recs.end.to_numpy(), extra, side="right")
        X = np.concatenate([X, Xe]); yy = np.r_[yy, np.ones(len(extra), int)]
        files = np.r_[files, recs.file.to_numpy(object)[rec_e]]; w0 = np.r_[w0, np.ones(len(extra), np.float32)]
        rec = np.r_[rec, rec_e]; ridx = np.r_[ridx, extra]
    info.update(train_rows=int(len(yy)), positives=int((yy == 1).sum()))
    return X, yy, files, w0, rec, ridx, info


# ============================================================================ distillation
def distill_scores(X150, y, rec, ridx, fold, spec):
    """Cross-fitted teacher probability for EVERY training row of this fold: a full-size model (OBS_TEACHER, no
    budget) on the student's inputs, 3 group folds inside the fold's training groups (a row is scored by a teacher
    that never saw its group). Cached per fold and row set."""
    import hashlib
    import lightgbm as lgb
    D = load(); recs = D["rpl"]["recs"]
    key = hashlib.sha256(np.ascontiguousarray(ridx).tobytes() + np.ascontiguousarray(y).tobytes()
                         + "|".join(spec).encode()).hexdigest()[:12]
    path = OUT / "models" / f"distill_teacher_fold{fold}_{key}.npz"
    if path.is_file():
        with np.load(path) as z:
            return z["score"]
    t0 = time.perf_counter()
    grp = recs.group.to_numpy(object)[rec]; groups = np.unique(grp)
    F, _, _ = build(np.hstack([X150, lite_rows(ridx, rec)]) if ("litediff" in spec or "litediff4" in spec) else X150,
                    spec)
    w = weights(y, recs.file.to_numpy(object)[rec], np.ones(len(y), np.float32))
    score = np.full(len(y), np.nan)
    for k, (a_, b_) in enumerate(GroupKFold(n_splits=3).split(groups, groups=groups)):
        fit_rows = np.isin(grp, groups[a_]); sc = np.isin(grp, groups[b_])
        m = lgb.LGBMClassifier(**OBS_TEACHER, n_jobs=THREADS, random_state=21 + k).fit(F[fit_rows], y[fit_rows],
                                                                                      sample_weight=w[fit_rows])
        score[sc] = m.predict_proba(F[sc])[:, 1]
        log("GR_DISTILL_TEACHER", fold, k, json.dumps(dict(fit_rows=int(fit_rows.sum()), scored=int(sc.sum()),
                                                           seconds=round(time.perf_counter() - t0, 1))))
    assert np.isfinite(score).all()
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, score=score)
    return score


def fit_distill(F, target, w, budget, extra_params=None):
    """Student on soft targets t in [0, 1], as a plain binary classifier: every frame becomes two weighted rows,
    (label 1, weight w*t) and (label 0, weight w*(1-t)). Their binary log loss is exactly the cross-entropy with the
    soft target, so the model type, its size accounting and the C export are those of the normal TAP model.
    min_child_samples is doubled (it counts rows, and most frames are now two rows)."""
    t = np.clip(np.asarray(target, np.float64), 0.0, 1.0); w = np.asarray(w, np.float64)
    one, zero = t > 0, t < 1
    F2 = np.concatenate([F[one], F[zero]])
    y2 = np.r_[np.ones(int(one.sum()), int), np.zeros(int(zero.sum()), int)]
    w2 = np.r_[w[one] * t[one], w[zero] * (1.0 - t[zero])].astype(np.float32)
    ep = dict(extra_params or {})
    ep["min_child_samples"] = 2 * int(ep.get("min_child_samples", CONFIG["min_child_samples"]))
    return fit(F2, y2, w2, budget, extra_params=ep)


# ============================================================================ training
def fit(F, y, w, budget, trees=None, extra_params=None):
    params = dict(CONFIG, n_jobs=THREADS, random_state=g.SEED)
    if trees:
        params["n_estimators"] = trees
    if QUICK:
        params["n_estimators"] = 40
    if extra_params:
        params.update(extra_params)
    if budget:
        return g.fit_model(F, y, w, params, budget)
    import lightgbm as lgb
    return lgb.LGBMClassifier(**params).fit(F, y, sample_weight=w)


class LinearScore:
    """Logistic regression on the standardized inputs (class-balanced weights as the trees); its logit is one more
    tree input. Firmware: sum_i a_i * x_i + b over the same inputs (a_i = coef_i / sd_i)."""

    def fit(self, F, y, w, C=0.1):
        from sklearn.linear_model import LogisticRegression
        t0 = time.perf_counter()
        self.mu = F.mean(0, dtype=np.float64).astype(np.float32); self.sd = F.std(0, dtype=np.float64).astype(np.float32)
        self.sd[self.sd < 1e-6] = 1.0
        Z = (F - self.mu) / self.sd
        lr = LogisticRegression(C=C, max_iter=300, tol=1e-4).fit(Z, y, sample_weight=w)
        self.a = (lr.coef_[0] / self.sd).astype(np.float64); self.b = float(lr.intercept_[0] - (lr.coef_[0] * self.mu / self.sd).sum())
        z = F.astype(np.float64) @ self.a + self.b
        self.info = dict(iterations=int(lr.n_iter_[0]), seconds=round(time.perf_counter() - t0, 1),
                         auc_train=round(float(_auc(z, y)), 4), nonzero=int((np.abs(lr.coef_[0]) > 1e-6).sum()))
        return self

    def append(self, F):
        return np.hstack([F, (F.astype(np.float64) @ self.a + self.b).astype(np.float32)[:, None]])


def _auc(score, y):
    from sklearn.metrics import roc_auc_score
    i = np.random.default_rng(0).choice(len(y), min(len(y), 200_000), replace=False)
    return roc_auc_score(y[i], score[i])


class XgbModel:
    """XGBoost model behind the LightGBM calls used here (Val.predict, importance, save). Size: the exporter's
    packed node (NODE_BYTES per split or leaf, as LightGBM) + 2 bytes per tree + 300 fixed: the same C format
    would hold these trees (a converter maps XGBoost's x < t to x <= nextafter(t, -inf))."""

    def __init__(self, clf):
        self.b = clf.get_booster(); self.booster_ = self
        nodes = self.b.trees_to_dataframe().groupby("Tree").size().sort_index().to_numpy()
        self.bytes = g.NODE_BYTES * np.cumsum(nodes) + 2 * np.arange(1, len(nodes) + 1) + 300

    def trees_within(self, budget):
        k = int(np.searchsorted(self.bytes, budget, side="right"))
        return k, int(self.bytes[k - 1])

    def predict(self, F, num_iteration=None, num_threads=None):
        return self.b.inplace_predict(np.asarray(F, np.float32), iteration_range=(0, int(num_iteration or 0)))

    def feature_importance(self, kind="gain", iteration=None):
        sc = self.b.get_score(importance_type="total_gain")
        return np.array([sc.get(f"f{i}", 0.0) for i in range(self.b.num_features())])

    def save_model(self, path, num_iteration=None):
        self.b[:int(num_iteration or 0) or None].save_model(str(path).replace(".txt", ".xgb.json"))


def fit_xgb(F, y, w, arm):
    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=arm.get("trees", 600), max_depth=arm.get("depth", 4),
                            learning_rate=arm.get("lr", 0.1), subsample=0.8, colsample_bytree=0.7,
                            min_child_weight=5.0, reg_lambda=1.0, reg_alpha=1e-3, tree_method="hist", max_bin=127,
                            n_jobs=THREADS, random_state=arm.get("seed", g.SEED), verbosity=0)
    clf.fit(np.asarray(F, np.float32), y, sample_weight=w)
    return XgbModel(clf)


# Arms: features (spec), labels, train-time EMG gain augmentation (octaves), EMG split-gain penalty (weight).
ARMS = {
    "raw126":           dict(spec=("acc", "emgraw")),
    "lean8":            dict(spec=("acc", "emgraw", "lean")),
    "acc_only":         dict(spec=("acc",)),
    "acc_diff":         dict(spec=("acc", "diff", "wlratio")),
    "acc_ratio":        dict(spec=("acc", "ratio", "wlratio")),
    "acc_logratio":     dict(spec=("acc", "logratio", "wlratio")),
    "raw126_diff":      dict(spec=("acc", "emgraw", "diff", "wlratio")),
    # no absolute EMG levels (codes, WL): EMG only as relative / event-local evidence (TAP_OBSERVABLE_RESULTS.md)
    "acc_emgrel":       dict(spec=("acc", "diff", "wlratio")),
    "acc_emgevent":     dict(spec=("acc", "difft", "wlratio")),
    "raw126_aug":       dict(spec=("acc", "emgraw"), aug=1.0),
    "raw126_augp":      dict(spec=("acc", "emgraw"), augp=True),
    "acc_diff_augp":    dict(spec=("acc", "diff", "wlratio"), augp=True),
    "raw126_penalty":   dict(spec=("acc", "emgraw"), penalty=0.3),
    "raw126_coarse":    dict(spec=("acc", "emgraw"), emg_bins=8),
    "pure_tap":         dict(spec=("acc", "emgraw"), labels="pure_tap"),
    "acc_rule":         dict(spec=("acc", "emgraw"), labels="acc_rule"),
    "teacher":          dict(spec=("acc", "emgraw"), labels="teacher"),
    "teacher_anchor":   dict(spec=("acc", "emgraw"), labels="teacher_anchor"),
    "press2":           dict(spec=("acc", "emgraw"), labels="press2"),
}


def arm_from_name(name):
    """Combinations: tokens joined by '+', e.g. 'acc_diff+aug1+teacher_anchor'."""
    if name in ARMS:
        return dict(ARMS[name])
    base = {}
    for tok in name.split("+"):
        if tok in ARMS:
            for k, v in ARMS[tok].items():
                base.setdefault(k, v) if k == "spec" else base.__setitem__(k, v)
            continue
        elif tok in ("augp", "augm", "augh"):
            base[tok] = True
        elif tok.startswith("aug"):
            base["aug"] = float(tok[3:] or 1)
        elif tok.startswith("pen"):
            base["penalty"] = float(tok[3:])
        elif tok.startswith("fingerw"):
            base["fingerw"] = float(tok[7:])
        elif tok.startswith("posw") and tok[4:].isdigit():
            base["posw"] = float(tok[4:]) / 100       # posw5: Optuna-style positive-class weight 0.05 (trainer default 1)
        elif tok.startswith("movew"):
            base["movew"] = float(tok[5:])
        elif tok.startswith("shift"):
            base["shift"] = float(tok[5:])
        elif tok.startswith("emg") and tok[3:4].isdigit():
            base["emg_min"] = float(tok[3:])          # emg1.2: TAP positives need an EMG rise of >= 1.2x
        elif tok == "notapneg":                       # GPT E2: no negatives from TAP / SWIPE recordings in training
            base["notapneg"] = True
        elif tok.startswith("kb") and tok[2:].replace(".", "", 1).isdigit():
            base["kb"] = float(tok[2:])                # kb14: model size budget 14 KB (exported static bytes)
        elif tok.startswith("leaves") and tok[6:].isdigit():
            base["leaves"] = int(tok[6:])             # leaves15: smaller trees (more of them fit the budget)
        elif tok == "xgb":                            # XGBoost (depth-wise trees, max depth: depth<k>, default 4)
            base["xgb"] = True
        elif tok.startswith("trees") and tok[5:].isdigit():
            base["trees"] = int(tok[5:])              # trees800: n_estimators cap (bigger budgets need more trees)
        elif tok.startswith("lr") and tok[2:].isdigit():
            base["lr"] = int(tok[2:]) / 100           # lr5: learning rate 0.05 (default 0.1)
        elif tok.startswith("depth") and tok[5:].isdigit():
            base["depth"] = int(tok[5:])              # depth10: max_depth (default 8)
        elif tok.startswith("mcs") and tok[3:].isdigit():
            base["mcs"] = int(tok[3:])                # mcs400: min_child_samples (leaf size)
        elif tok.startswith("seed") and tok[4:].isdigit():
            base["seed"] = int(tok[4:])               # seed<k>: another LightGBM seed (training noise)
        elif tok.startswith("recq") and tok[4:].isdigit():
            base["recq"] = float(tok[4:]) / 100       # recq50: weak recordings (< 50% presses confirmed) give no taps
        elif tok.startswith("strictx") and tok[7:].isdigit():
            base["strict"] = (int(tok[7:]), True)    # strictx20: 20% most finger-like single taps (doubles, 240 bpm, swipes exempt)
        elif tok.startswith("strict") and tok[6:].isdigit():
            base["strict"] = (int(tok[6:]), False)   # strict20: drop the 20% most finger-like taps
        elif tok.startswith("cons") and tok[4:].isdigit():
            base["cons"] = int(tok[4:])                # cons32: only the 32 inputs most consistent across people
        elif tok.startswith("obsfirst") and tok[8:].isdigit():
            base["obs"] = ("first", float(tok[8:]))  # obsfirst40: earliest 40 ms window near the teacher's best
        elif tok.startswith("obs") and tok[3:].isdigit():
            base["obs"] = ("max", float(tok[3:]))    # obs40: per tap the 40 ms the teacher finds most distinguishable
        elif tok.startswith("win") and tok[3:].isdigit():
            base["win_ms"] = float(tok[3:])           # win40: positive window [-5, 40] ms after the onset
        elif tok == "extra":                          # trained with the MUDRA_GR_EXTRA recordings (train mode)
            assert EXTRA and EXTRA_MODE == "train", "arm +extra needs MUDRA_GR_EXTRA and MUDRA_GR_EXTRA_MODE=train"
            base["extra"] = [str(p_) for p_ in EXTRA]
        elif tok.startswith("drop"):                 # drop<person>: none of that person's rows in training
            base.setdefault("drop", []).append(tok[4:])
        elif tok == "fmine":
            base["fmine"] = True
        elif tok in ("relneg", "preneg", "pressneg", "relnegs"):   # the tap's own press: release / lead-in / both;
            base["pressneg"] = {"relneg": ("rel",), "preneg": ("pre",), "pressneg": ("rel", "pre"),
                                "relnegs": ("rel", "single")}[tok]   # relnegs: release, single-tap recordings only
        elif tok.startswith("bgneg") and tok[5:].isdigit():   # bgneg500: TAP / SWIPE rest time > 500 ms from any
            base["bgneg"] = int(tok[5:])                       # press or tap (unknown today) trained as negative
        elif tok == "softall":                       # soft labels in every TAP / SWIPE recording (trainer soft 0.5)
            base["softall"] = True
        elif tok.startswith("distill") and tok[7:].isdigit():
            base["distill"] = int(tok[7:]) / 100     # distill50: target = 0.5 label + 0.5 cross-fitted teacher score
        elif tok.startswith("pw"):
            base["pw"] = float(tok[2:])
        elif tok == "lrscore":                       # one extra input: a linear tap score fitted per fold
            base["lrscore"] = True
        elif tok in ("litediff", "accdiff", "litediff4", "accsum", "emgsum"):
            base["spec"] = tuple(base.get("spec", ("acc", "emgraw"))) + (tok,)
        else:
            raise ValueError(f"Unknown arm token {tok}")
    return base


def run_arm(name):
    arm = arm_from_name(name); spec = arm["spec"]; labels = arm.get("labels", "default")
    D = load(); data = D["data"]; tr = D["tr"]
    budget = g.budget_bytes(arm.get("kb", g.MODEL_BUDGET_KB))
    sums = {v: {c: {thr: R.TapCounts() for thr in GRID} for c in STRESS} for v in ("usable", "steady")}
    per_fold = []
    focus_acc = FocusCounts()
    if labels.startswith("teacher"):
        D.setdefault("teacher", {})
    for f, (a, b) in enumerate(D["folds"]):
        t0 = time.perf_counter()
        trn = tr[a]; val_people = set(data["group"][tr[b]])
        D["fold"] = f
        if labels.startswith("teacher") and f not in D["teacher"]:
            D["teacher"][f] = teacher_labels(trn, f)
        if arm.get("drop"):
            trn = trn[~np.isin(data["group"][trn], arm["drop"])]
        X150, y, files, w0, rec, ridx, info = train_set(trn, labels)
        if arm.get("fmine"):                       # hard negatives from finger-motion / fist recordings
            mined = mine_finger(set(data["group"][trn]), f)
            rec_m = np.searchsorted(D["rpl"]["recs"].end.to_numpy(), mined, side="right")
            X150 = np.concatenate([X150, np.asarray(D["rpl"]["X"][mined])]); y = np.r_[y, np.zeros(len(mined), int)]
            files = np.r_[files, D["rpl"]["recs"].file.to_numpy(object)[rec_m]]
            w0 = np.r_[w0, np.ones(len(mined), np.float32)]; rec = np.r_[rec, rec_m]; ridx = np.r_[ridx, mined]
            info["mined_finger_rows"] = int(len(mined))
        if arm.get("bgneg"):                       # rest time far from every press / tap in TAP and SWIPE files
            mined = background_rows(set(data["group"][trn]), arm["bgneg"])
            rec_m = np.searchsorted(D["rpl"]["recs"].end.to_numpy(), mined, side="right")
            X150 = np.concatenate([X150, np.asarray(D["rpl"]["X"][mined])]); y = np.r_[y, np.zeros(len(mined), int)]
            files = np.r_[files, D["rpl"]["recs"].file.to_numpy(object)[rec_m]]
            w0 = np.r_[w0, np.ones(len(mined), np.float32)]; rec = np.r_[rec, rec_m]; ridx = np.r_[ridx, mined]
            info["background_negative_rows"] = int(len(mined))
        if arm.get("pressneg"):                    # the press around a tap (release / lead-in) trained as negative
            mined = press_rows(set(data["group"][trn]), arm["pressneg"])
            rec_m = np.searchsorted(D["rpl"]["recs"].end.to_numpy(), mined, side="right")
            X150 = np.concatenate([X150, np.asarray(D["rpl"]["X"][mined])]); y = np.r_[y, np.zeros(len(mined), int)]
            files = np.r_[files, D["rpl"]["recs"].file.to_numpy(object)[rec_m]]
            w0 = np.r_[w0, np.ones(len(mined), np.float32)]; rec = np.r_[rec, rec_m]; ridx = np.r_[ridx, mined]
            info["press_negative_rows"] = int(len(mined))
        if arm.get("notapneg"):                    # only authoritative negatives (movement, nothing, LP) remain
            fold_of = D["rpl"]["recs"].folder.to_numpy(object)[rec]
            keep = ~((y == 0) & np.isin(fold_of, ["TAP", "SWIPE"]))
            info["notapneg_dropped_rows"] = int((~keep).sum())
            X150, y, files, w0, rec, ridx = X150[keep], y[keep], files[keep], np.asarray(w0)[keep], rec[keep], ridx[keep]
        if arm.get("emg_min") or arm.get("win_ms") or arm.get("recq"):   # tighter positives
            X150, y, files, w0, rec, ridx, inf = filter_positives(X150, y, files, w0, rec, ridx,
                                                                  arm.get("emg_min"), arm.get("win_ms"), arm.get("recq"))
            info.update(inf)
        if arm.get("strict"):                      # strict tap definition: finger-like taps are not taught
            X150, y, files, w0, rec, ridx, inf = strict_taps(X150, y, files, w0, rec, ridx, f, trn, *arm["strict"])
            info.update(inf)
        if arm.get("obs"):                         # observable-pattern labels (cross-fitted teacher, per tap)
            X150, y, files, w0, rec, ridx, inf = observable_positives(X150, y, files, w0, rec, ridx, f, *arm["obs"],
                                                                      tag=labels)
            info.update(inf)
        if arm.get("shift"):                       # positive windows d ms later (decide after the contact)
            X150, y, files, w0, rec, ridx, inf = shift_positives(X150, y, files, w0, rec, ridx, arm["shift"])
            info.update(inf)
        if arm.get("fingerw"):                     # finger motion + fist negatives weighted x k
            fing = (D["rpl"]["recs"].folder.to_numpy(object)[rec] == "FINGER") & (y == 0)
            w0 = np.asarray(w0, np.float32).copy(); w0[fing] *= arm["fingerw"]
            info["finger_rows_weighted"] = int(fing.sum())
        if arm.get("movew"):                       # all movement negatives (finger, fist, wrist, arm, band) x k
            mv = np.isin(D["rpl"]["recs"].folder.to_numpy(object)[rec], ["FINGER", "WRIST", "ARM", "BAND"]) & (y == 0)
            w0 = np.asarray(w0, np.float32).copy(); w0[mv] *= arm["movew"]
            info["movement_rows_weighted"] = int(mv.sum())
        if arm.get("pw"):                          # positives of DoubleTaps / 240 bpm recordings weighted x k
            sel = np.isin(rec, list(press2_recordings())) & (y == 1)
            w0 = np.asarray(w0, np.float32).copy(); w0[sel] *= arm["pw"]
            info["press2_positive_rows_weighted"] = int(sel.sum())
        if arm.get("softall"):                     # the labeller's soft taps of every TAP / SWIPE recording
            soft = D["tr_soft"][np.isin(data["group"][D["tr_soft"]], list(set(data["group"][trn])))]
            if labels == "press2":                 # press2 already added those of its recordings
                soft = soft[~np.isin(data["rec"][soft], list(press2_recordings()))]
            X150 = np.concatenate([X150, data["X"][soft]]); y = np.r_[y, np.ones(len(soft), int)]
            files = np.r_[files, data["file"][soft]]; w0 = np.r_[np.asarray(w0, np.float32), data["w"][soft]]
            rec = np.r_[rec, data["rec"][soft]]; ridx = np.r_[ridx, data_ridx(soft)]
            info["softall_rows"] = int(len(soft))
        target = None
        if arm.get("distill") is not None:         # soft targets from a cross-fitted full-size teacher
            ts = distill_scores(X150, y, rec, ridx, f, spec)
            target = arm["distill"] * y + (1 - arm["distill"]) * ts
            info.update(distill_alpha=arm["distill"], teacher_mean_pos=round(float(ts[y == 1].mean()), 4),
                        teacher_mean_neg=round(float(ts[y == 0].mean()), 4))
        aug_tab = None
        if arm.get("aug"):
            X150 = apply_gain(X150, gain_table(D["n_recs"], arm["aug"], 1000 + f)[rec])
        if arm.get("augp"):                        # placement augmentation: gain + tilt + noise (other draws)
            tab = stress_tables(D["n_recs"], seed=2000 + f)["placement"]
            X150 = apply_emg(X150, **{k_: v[rec] for k_, v in tab.items()}); aug_tab = tab
        if arm.get("augm"):                        # milder placement augmentation: tilt +-0.25 oct, noise 0.03..0.3
            tab = stress_tables(D["n_recs"], seed=3000 + f, tilt_oct=0.25, noise_log10=(-1.5, -0.5))["placement"]
            X150 = apply_emg(X150, **{k_: v[rec] for k_, v in tab.items()}); aug_tab = tab
        if "litediff" in spec or "litediff4" in spec:   # recomputed from the (augmented) WL of each recording
            X150 = np.hstack([X150, lite_rows(ridx, rec, aug_tab)])
        if arm.get("augh"):                        # half the recordings augmented (placement), half untouched
            tab = stress_tables(D["n_recs"], seed=4000 + f)["placement"]
            pick = np.random.default_rng(5000 + f).random(D["n_recs"]) < 0.5
            mrow = pick[rec]
            X150[mrow] = apply_emg(X150[mrow], **{k_: v[rec[mrow]] for k_, v in tab.items()})
        F, names, is_emg = build(X150, spec)
        del X150
        cols = None
        if arm.get("cons"):                        # person-consistent minimal inputs (fold-local screen)
            act = data_ridx(trn[data["kind"][trn] == 2])
            folder_rec = D["rpl"]["recs"].folder.to_numpy(object)[rec]
            look = (y == 0) & np.isin(folder_rec, CONS_LOOKALIKE) & np.isin(ridx, act)
            person_rec = D["rpl"]["recs"].group.to_numpy(object)[rec]
            cols, cscore, cpeople = consistent_features(F, y, person_rec, look, arm["cons"])
            F = F[:, cols]; is_emg = is_emg[cols]
            info.update(cons_k=int(arm["cons"]), cons_people=len(cpeople), cons_lookalike_rows=int(look.sum()),
                        cons_inputs=[names[i] for i in cols],
                        cons_score_range=[float(cscore[cols].min()), float(cscore[cols].max())])
            names = [names[i] for i in cols]
        post = None
        if arm.get("lrscore"):                     # stacked linear tap score: one dot product in firmware
            ls = LinearScore().fit(F, y, weights(y, files, w0))
            F = ls.append(F); names = list(names) + ["lr_score"]; is_emg = np.r_[is_emg, True]; post = ls.append
            info["lrscore"] = ls.info
        extra = {}
        if arm.get("penalty") is not None:
            extra["feature_contri"] = [arm["penalty"] if e else 1.0 for e in is_emg]
        if arm.get("emg_bins"):
            extra["max_bin_by_feature"] = [arm["emg_bins"] if e else CONFIG["max_bin"] for e in is_emg]
        if arm.get("seed") is not None:
            extra["random_state"] = arm["seed"]
        if arm.get("leaves"):
            extra["num_leaves"] = arm["leaves"]
        if arm.get("mcs"):
            extra["min_child_samples"] = arm["mcs"]
        if arm.get("trees"):
            extra["n_estimators"] = arm["trees"]
        if arm.get("lr"):
            extra["learning_rate"] = arm["lr"]
        if arm.get("depth"):
            extra["max_depth"] = arm["depth"]
        w = weights(y, files, w0)
        if arm.get("posw"):                        # train_unified.weights_for_task: w[y == 1] *= positive_weight
            w[y == 1] *= np.float32(arm["posw"])
        if arm.get("xgb"):
            m = fit_xgb(F, y, w, arm)
            del F
            k, size = m.trees_within(budget)
        else:
            m = (fit_distill(F, target, w, budget, extra_params=extra) if target is not None
                 else fit(F, y, w, budget, extra_params=extra))
            del F
            k, size = g.trees_within(m, budget)
        imp = m.booster_.feature_importance("gain", iteration=k)
        emg_gain = float(imp[is_emg].sum() / max(imp.sum(), 1e-9))
        t_fit = time.perf_counter() - t0
        val = Val(val_people)
        fold_rep = dict(fold=f, trees=int(k), bytes=int(size), features=len(names), emg_gain_share=round(emg_gain, 4),
                        fit_s=round(t_fit, 1), labels=info)
        tables = stress_tables(D["n_recs"])
        (OUT / "models").mkdir(parents=True, exist_ok=True)
        m.booster_.save_model(str(OUT / "models" / f"{name}_fold{f}.txt"), num_iteration=k)
        for cond in STRESS:
            p = val.predict(m, k, spec, tables[cond], cols=cols, post=post)
            if cond in ("nominal", "placement"):
                focus_acc.add(val, p, cond)
            for thr in GRID:
                for view in ("usable", "steady"):
                    sums[view][cond][thr].add(val.counts(p, thr, steady=view == "steady"))
            fold_rep[cond] = summarize({thr: val.counts(p, thr).report() for thr in (0.85,)})
        fold_rep["seconds"] = round(time.perf_counter() - t0, 1)
        per_fold.append(fold_rep)
        log("GR_FOLD", name, f, json.dumps({k_: v for k_, v in fold_rep.items() if k_ not in STRESS}),
            json.dumps({c: fold_rep[c] for c in STRESS}))
        del val, m
    result = dict(arm=name, config=dict(arm, spec=list(spec)), lgbm=CONFIG, quick=QUICK, folds=per_fold,
                  pooled={c: {str(thr): slim(sums["usable"][c][thr].report()) for thr in GRID} for c in STRESS},
                  pooled_steady={c: {str(thr): slim(sums["steady"][c][thr].report()) for thr in GRID} for c in STRESS})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1, default=float))
    focus_acc.save(name)
    log("GR_ARM_DONE", name, json.dumps(headline(result)))
    return result


RULES = ("min2", "min3", "mean2", "mean3")


def apply_rule(p, rule):
    """Firmware decision rule on one recording's frame probabilities, as a transformed probability (threshold it
    as usual): min<k> = the last k frames all >= threshold; mean<k> = mean of the last k frames >= threshold."""
    k = int(rule[-1]); q = np.asarray(p, np.float64)
    lag = [np.r_[np.zeros(j), q[:len(q) - j]] for j in range(k)]     # frames before the start count as 0
    return np.minimum.reduce(lag) if rule.startswith("min") else np.mean(lag, axis=0)


def rule_arm(base, rule):
    """Re-score the saved fold models of `base` with another decision rule (no training): `base+rule<rule>`."""
    import lightgbm as lgb
    assert rule in RULES, rule
    name = f"{base}+rule{rule}"
    res0 = json.loads((OUT / f"{base}.json").read_text()); spec = tuple(res0["config"]["spec"])
    D = load(); data = D["data"]; tr = D["tr"]
    conds = tuple(c for c in STRESS if c in ("nominal", "placement"))
    tables = stress_tables(D["n_recs"])
    sums = {v: {c: {thr: R.TapCounts() for thr in GRID} for c in conds} for v in ("usable", "steady")}
    focus_acc = FocusCounts(); per_fold = []
    for f, (a, b) in enumerate(D["folds"]):
        t0 = time.perf_counter()
        val = Val(set(data["group"][tr[b]]))
        m = lgb.Booster(model_file=str(OUT / "models" / f"{base}_fold{f}.txt"))
        rep = dict(fold=f)
        for cond in conds:
            p = val.predict(m, None, spec, tables[cond])
            for s_, e_ in val.seg:
                p[s_:e_] = apply_rule(p[s_:e_], rule)
            focus_acc.add(val, p, cond)
            for thr in GRID:
                for view in ("usable", "steady"):
                    sums[view][cond][thr].add(val.counts(p, thr, steady=view == "steady"))
            rep[cond] = summarize({thr: val.counts(p, thr).report() for thr in (0.85,)})
        rep["seconds"] = round(time.perf_counter() - t0, 1)
        rep["emg_gain_share"] = res0["folds"][f].get("emg_gain_share", float("nan")); per_fold.append(rep)
        log("GR_RULE_FOLD", name, f, json.dumps(rep))
        del val, m
    result = dict(arm=name, config=dict(res0["config"], rule=rule), lgbm=res0.get("lgbm"), quick=QUICK, folds=per_fold,
                  pooled={c: {str(thr): slim(sums["usable"][c][thr].report()) for thr in GRID} for c in conds},
                  pooled_steady={c: {str(thr): slim(sums["steady"][c][thr].report()) for thr in GRID} for c in conds})
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1, default=float))
    focus_acc.save(name)
    log("GR_ARM_DONE", name, json.dumps(headline(result)))
    return result


def slim(r):
    if not r:
        return None
    pf = r.get("per_folder") or {}
    return dict(recall=r["recall"], recall_press=r.get("recall_press"), false_per_min=r["false_per_min"],
                onsets=r["onsets"], false_events=r["false_events"], negative_minutes=r["negative_minutes"],
                latency_ms_median=r.get("latency_ms_median"),
                recall_TAP=(pf.get("TAP") or {}).get("recall"), recall_SWIPE=(pf.get("SWIPE") or {}).get("recall"),
                fpm={k: v["false_per_min"] for k, v in pf.items() if v.get("false_per_min") is not None})


def summarize(reps):
    return {str(t): {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (slim(r) or {}).items() if k != "fpm"}
            for t, r in reps.items()}


def at_budget(pooled, fpm_max):
    """Highest-recall threshold whose false taps/min <= fpm_max (nominal), and that threshold's numbers."""
    ok = {float(t): r for t, r in pooled.items() if r and r["false_per_min"] <= fpm_max}
    if not ok:
        return None, None
    thr = max(ok, key=lambda t: (ok[t]["recall"], -ok[t]["false_per_min"]))
    return thr, ok[thr]


def headline(res, view="pooled"):
    out = {}
    nom = res[view]["nominal"]
    for fpm_max in (2.0, 5.0):
        thr, r = at_budget(nom, fpm_max)
        if thr is None:
            out[f"fpm<={fpm_max:g}"] = None; continue
        row = dict(threshold=thr, recall_TAP=r["recall_TAP"], recall_SWIPE=r["recall_SWIPE"], fpm=r["false_per_min"])
        for cond in STRESS[1:]:
            s = res[view][cond][str(thr)]
            row[cond] = dict(recall_TAP=s["recall_TAP"], recall_SWIPE=s["recall_SWIPE"], fpm=s["false_per_min"])
        out[f"fpm<={fpm_max:g}"] = row
    r85 = {c: res[view][c]["0.85"] for c in STRESS}
    out["thr0.85"] = {c: dict(recall_TAP=v["recall_TAP"], recall_SWIPE=v["recall_SWIPE"], fpm=v["false_per_min"])
                      for c, v in r85.items()}
    out["emg_gain_share"] = round(float(np.mean([f["emg_gain_share"] for f in res["folds"]])), 3)
    return out


def report(view="pooled"):
    rows = []
    for p in sorted(OUT.glob("*.json")):
        res = json.loads(p.read_text())
        if view not in res:
            continue
        h = headline(res, view)
        for budget in ("fpm<=2", "fpm<=5"):
            r = h.get(budget)
            if not r:
                continue
            row = dict(arm=res["arm"], budget=budget, thr=r["threshold"], TAP=r["recall_TAP"], SWIPE=r["recall_SWIPE"],
                       fpm=r["fpm"])
            for cond in STRESS[1:]:
                row[f"TAP_{cond}"] = r[cond]["recall_TAP"]; row[f"fpm_{cond}"] = r[cond]["fpm"]
            row["worst_TAP"] = min(r[c]["recall_TAP"] for c in STRESS[1:])
            row["worst_fpm"] = max(r[c]["fpm"] for c in STRESS[1:])
            row["emg_share"] = h["emg_gain_share"]
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        print(df.round(3).to_markdown(index=False))
    return df


# ============================================================================ focus: every real press, per gesture
FOCUS_CONDS = ("nominal", "placement", "warmup")   # warmup: nominal, rows before frame 500 (first ~10 s)
FOCUS_TH = 1.0 / (1.0 + np.exp(-np.linspace(0.0, np.log(0.9995 / 0.0005), 80)))   # 0.5 .. 0.9995, even in log-odds
POS_TYPES = ("normal", "soft", "double_1st", "double_2nd", "fast240", "slowpinch", "swipe")
NEG_TYPES = ("finger", "fist", "wrist", "arm", "band", "nothing", "longpress", "taprec")
PAIR_MS = 500                       # DoubleTaps: a press < 500 ms after the previous one is the 2nd tap


def rec_type(folder, file):
    f = str(file).split("/")[-1]
    if folder == "TAP":
        for key, typ in (("DoubleTap", "double"), ("240bpm", "fast240"), ("SlowPinch", "slowpinch"), ("SoftTap", "soft")):
            if key in f:
                return typ
        return "normal"
    if folder == "SWIPE":
        return "swipe"
    if folder == "FINGER":
        return "fist" if "Fist" in f else "finger"
    if folder in ("WRIST", "ARM", "BAND", "NOTHING"):
        return folder.lower()
    return "longpress"


def _events(t, rows):
    """Rising-edge rows -> event times with the firmware refractory (replay.TAP_REFRACTORY_MS)."""
    out, last = [], -1e18
    for r in rows:
        if t[r] - last >= R.TAP_REFRACTORY_MS:
            out.append(t[r]); last = t[r]
    return np.asarray(out)


def _match(ev, pr, which=False):
    used = np.zeros(len(ev), bool); hit = np.zeros(len(pr), bool); idx = np.full(len(pr), -1)
    for k, o in enumerate(pr):
        i = int(np.searchsorted(ev, o - R.TAP_HIT_BEFORE_MS))
        while i < len(ev) and used[i]:
            i += 1
        if i < len(ev) and ev[i] <= o + R.TAP_HIT_AFTER_MS:
            used[i] = True; hit[k] = True; idx[k] = i
    return (hit, used, idx) if which else (hit, used)


LAT_BINS = np.arange(-100, 255, 5)            # latency histogram (event - press, ms) of clicked taps


def _hold_events(t, starts, ends, hold_ms):
    """Runs of score >= threshold (start / end rows): an event once the run has lasted hold_ms (firmware:
    'the score stayed above the threshold for hold_ms'), then the refractory."""
    if hold_ms <= 0:
        return _events(t, starts)
    ok = t[ends] - t[starts] >= hold_ms
    fire = np.searchsorted(t, t[starts[ok]] + hold_ms, side="left")
    return _events(t, fire)


class FocusCounts:
    """Per-person counts on a log-odds threshold grid (FOCUS_TH): every real pointer press per gesture type (hit =
    an event within -100..+250 ms, as the replay), and events per minute per no-gesture recording type plus the
    unmatched events on trusted-negative rows of TAP / SWIPE recordings (taprec)."""

    def __init__(self, hold_ms=0):
        self.hold_ms = hold_ms
        recs = load()["rpl"]["recs"]
        self.people = sorted(recs.group.unique()); self.pi = {p_: i for i, p_ in enumerate(self.people)}
        nT, nP = len(FOCUS_TH), len(self.people)
        self.a = {}
        for c in FOCUS_CONDS:
            for t_ in POS_TYPES:
                self.a[f"hits|{c}|{t_}"] = np.zeros((nT, nP)); self.a[f"n|{c}|{t_}"] = np.zeros(nP)
            for t_ in NEG_TYPES:
                self.a[f"false|{c}|{t_}"] = np.zeros((nT, nP)); self.a[f"min|{c}|{t_}"] = np.zeros(nP)
            self.a[f"lat|{c}|taps"] = np.zeros((nT, len(LAT_BINS) - 1))
        on = load()["onsets"]
        self.press = {int(r): np.sort(v.t_on.to_numpy(float)) for r, v in on[on.kind == "press"].groupby("rec")}

    def add(self, val, p, cond):
        for (s, e), t, lab, item, st in zip(val.seg, val.t, val.label, val.items, val.steady_from):
            if e <= s:
                continue
            self._add(p[s:e], t, lab, item, cond)
            if cond == "nominal":
                stop = st if st is not None else e - s
                if stop > 1:
                    self._add(p[s:s + stop], t[:stop], lab[:stop], item, "warmup")

    def _add(self, pp, t, lab, item, cond):
        recs = load()["rpl"]["recs"]; nT = len(FOCUS_TH)
        rec, folder = item[0], item[3]
        if True:
            pi = self.pi[recs.group.iloc[rec]]; typ = rec_type(folder, recs.file.iloc[rec])
            d = pp[:, None] >= FOCUS_TH[None, :]
            rise = d & ~np.vstack([np.zeros((1, nT), bool), d[:-1]])
            ti, ri = np.nonzero(rise.T)                      # threshold-major: (threshold, row)
            bounds = np.searchsorted(ti, np.arange(nT + 1))
            fall = d & ~np.vstack([d[1:], np.zeros((1, nT), bool)])
            _, fi = np.nonzero(fall.T)                       # run ends, same order as the run starts
            evs = lambda j: _hold_events(t, ri[bounds[j]:bounds[j + 1]], fi[bounds[j]:bounds[j + 1]], self.hold_ms)
            dt = np.clip(np.diff(np.r_[t, t[-1]]), 0, 50)
            if folder in ("TAP", "SWIPE"):
                pr = self.press.get(int(rec), np.zeros(0))
                gap = np.diff(np.r_[-1e18, pr])
                ok = (pr >= t[0] + R.TAP_HIT_BEFORE_MS) & (pr <= t[-1] - R.TAP_HIT_AFTER_MS)
                kinds = np.array([(("double_2nd" if g_ < PAIR_MS else "double_1st") if typ == "double" else typ)
                                  for g_ in gap], object)
                pr, kinds = pr[ok], kinds[ok]
                for k_ in set(kinds):
                    self.a[f"n|{cond}|{k_}"][pi] += int((kinds == k_).sum())
                neg = lab == 0
                self.a[f"min|{cond}|taprec"][pi] += float(dt[neg].sum()) / 6e4
                tapk = kinds != "swipe"
                for j in range(nT):
                    ev = evs(j)
                    hit, used, mi = _match(ev, pr, which=True)
                    for k_ in set(kinds):
                        self.a[f"hits|{cond}|{k_}"][j, pi] += int(hit[kinds == k_].sum())
                    if (hit & tapk).any():
                        lat = ev[mi[hit & tapk]] - pr[hit & tapk]
                        self.a[f"lat|{cond}|taps"][j] += np.histogram(lat, LAT_BINS)[0]
                    if len(ev):
                        ix = np.clip(np.searchsorted(t, ev), 0, len(t) - 1)
                        self.a[f"false|{cond}|taprec"][j, pi] += int((~used & neg[ix]).sum())
            else:
                self.a[f"min|{cond}|{typ}"][pi] += float(dt.sum()) / 6e4
                for j in range(nT):
                    self.a[f"false|{cond}|{typ}"][j, pi] += len(evs(j))

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"focus_{name}.npz", people=np.array(self.people, object), th=FOCUS_TH, **self.a)
        (OUT / f"focus_{name}.json").write_text(json.dumps(focus_summary(self.a), indent=1))


def _load_focus(name):
    """Focus counts; people listed in MUDRA_FOCUS_EXCLUDE (comma separated) are left out of every sum."""
    z = np.load(OUT / f"focus_{name}.npz", allow_pickle=True)
    excl = [p_ for p_ in os.environ.get("MUDRA_FOCUS_EXCLUDE", "").split(",") if p_]
    keep = ~np.isin(np.asarray(z["people"], object), excl)
    return {k: (z[k] if k.startswith("lat|") else z[k][..., keep]) for k in z.files if "|" in k}


def _fpm_curve(a, cond, w=None):
    fal = sum(a[f"false|{cond}|{t_}"] for t_ in NEG_TYPES); mins = sum(a[f"min|{cond}|{t_}"] for t_ in NEG_TYPES)
    if w is None:
        return fal.sum(1) / mins.sum()
    return (fal @ w.T) / (mins @ w.T)


def _at(curve, target):
    """(j, alpha) so that curve interpolated at j + alpha equals target (curve falls with the threshold)."""
    for j in range(len(curve) - 1):
        if curve[j] >= target >= curve[j + 1]:
            return j, (0.0 if curve[j] == curve[j + 1] else (curve[j] - target) / (curve[j] - curve[j + 1]))
    return None, None


def focus_metrics(a, j, al, w=None):
    """Recall per gesture type and events/min per no-gesture type at grid position j + al (w: bootstrap weights)."""
    out = {}
    lerp = lambda v: v[j] * (1 - al) + v[j + 1] * al
    for c in FOCUS_CONDS:
        if f"n|{c}|normal" not in a:                # older focus files: no warm-up view
            continue
        for t_ in POS_TYPES:
            h, n = a[f"hits|{c}|{t_}"], a[f"n|{c}|{t_}"]
            out[f"{c}|{t_}"] = lerp(h.sum(1)) / max(n.sum(), 1) if w is None else lerp(h @ w.T) / np.maximum(n @ w.T, 1)
        for t_ in NEG_TYPES:
            f_, m_ = a[f"false|{c}|{t_}"], a[f"min|{c}|{t_}"]
            out[f"{c}|fpm_{t_}"] = (lerp(f_.sum(1)) / max(m_.sum(), 1e-9) if w is None
                                    else lerp(f_ @ w.T) / np.maximum(m_ @ w.T, 1e-9))
        fal = sum(a[f"false|{c}|{t_}"] for t_ in NEG_TYPES); mins = sum(a[f"min|{c}|{t_}"] for t_ in NEG_TYPES)
        out[f"{c}|fpm_all"] = lerp(fal.sum(1)) / mins.sum() if w is None else lerp(fal @ w.T) / (mins @ w.T)
        if f"lat|{c}|taps" in a:                     # median / 90th percentile latency of clicked taps (pooled)
            h = a[f"lat|{c}|taps"][min(j + int(al >= 0.5), len(FOCUS_TH) - 1)]
            cum = np.cumsum(h) / max(h.sum(), 1); mid = (LAT_BINS[:-1] + LAT_BINS[1:]) / 2
            for q in (50, 90):
                v = float(mid[min(np.searchsorted(cum, q / 100), len(mid) - 1)]) if h.sum() else np.nan
                out[f"{c}|lat_p{q}"] = v if w is None else np.full(w.shape[0], v)
    return out


def focus_summary(a, targets=(2.0, 2.8)):
    out = {"presses": {t_: int(a[f"n|nominal|{t_}"].sum()) for t_ in POS_TYPES},
           "minutes": {t_: round(float(a[f"min|nominal|{t_}"].sum()), 1) for t_ in NEG_TYPES}}
    curve = _fpm_curve(a, "nominal")
    for tg in targets:
        j, al = _at(curve, tg)
        if j is None:
            out[f"fpm{tg:g}"] = None; continue
        m = focus_metrics(a, j, al)
        out[f"fpm{tg:g}"] = dict(threshold=float(FOCUS_TH[j] * (1 - al) + FOCUS_TH[j + 1] * al),
                                 **{k: round(float(v), 4) for k, v in m.items()})
    return out


def focus(arms):
    """Focus counts of arms whose fold models are saved but have no focus file yet (e.g. older arms)."""
    import lightgbm as lgb
    D = load(); data = D["data"]; tr = D["tr"]; tables = stress_tables(D["n_recs"])
    todo = [a for a in arms if not (OUT / f"focus_{a}.npz").exists()]
    accs = {a: FocusCounts() for a in todo}
    for f, (_a, b) in enumerate(D["folds"]):
        if not todo:
            break
        val = Val(set(data["group"][tr[b]]))
        for arm in todo:
            spec = tuple(json.loads((OUT / f"{arm}.json").read_text())["config"]["spec"])
            m = lgb.Booster(model_file=str(OUT / "models" / f"{arm}_fold{f}.txt"))
            for cond in ("nominal", "placement"):
                accs[arm].add(val, val.predict(m, None, spec, tables[cond]), cond)
        log("GR_FOCUS_FOLD", f); del val
    for arm in todo:
        accs[arm].save(arm)
    return {a: json.loads((OUT / f"focus_{a}.json").read_text()) for a in arms}


def focus_hold(arms, holds=(0, 10, 20)):
    """Focus counts of saved fold models with the event rule 'score >= threshold held for h ms' (h = 0: the
    current rising edge). Written as focus_<arm>@hold<h>; predictions are made once per fold and condition."""
    import lightgbm as lgb
    D = load(); data = D["data"]; tr = D["tr"]; tables = stress_tables(D["n_recs"])
    accs = {(a, h): FocusCounts(hold_ms=h) for a in arms for h in holds}
    for f, (_a, b) in enumerate(D["folds"]):
        val = Val(set(data["group"][tr[b]]))
        for arm in arms:
            spec = tuple(json.loads((OUT / f"{arm}.json").read_text())["config"]["spec"])
            m = lgb.Booster(model_file=str(OUT / "models" / f"{arm}_fold{f}.txt"))
            for cond in ("nominal", "placement"):
                p = val.predict(m, None, spec, tables[cond])
                for h in holds:
                    accs[(arm, h)].add(val, p, cond)
        log("GR_FOCUS_HOLD_FOLD", f); del val
    for (a, h), acc in accs.items():
        acc.save(f"{a}@hold{h}")


FOCUS_SHOW = ("normal", "soft", "double_1st", "double_2nd", "fast240", "swipe", "fpm_finger", "fpm_fist",
              "fpm_wrist", "fpm_arm", "fpm_band", "fpm_all", "lat_p50", "lat_p90")


def focus_table(arms, target=2.8, n_boot=2000, seed=5, match=None):
    """Each arm at its own threshold for `target` overall false events/min (nominal), or (match="normal") at the
    threshold where its nominal recall of that gesture type equals the first arm's at `target`; the same threshold
    for placement and warm-up. Person bootstrap of the difference to the first arm (paired: same resampled people)."""
    A = {a: {k: v for k, v in _load_focus(a).items()} for a in arms}
    nP = len(next(iter(A.values()))["n|nominal|normal"])
    W = np.random.default_rng(seed).multinomial(nP, np.ones(nP) / nP, size=n_boot).astype(np.float64)
    pos = {}
    for a in arms:
        j, al = _at(_fpm_curve(A[a], "nominal"), target)
        pos[a] = (j, al)
    if match:
        j0, al0 = pos[arms[0]]
        rc = lambda a: A[a][f"hits|nominal|{match}"].sum(1) / A[a][f"n|nominal|{match}"].sum()
        goal = rc(arms[0])[j0] * (1 - al0) + rc(arms[0])[j0 + 1] * al0
        for a in arms[1:]:
            pos[a] = _at(rc(a), goal)                 # recall falls as the threshold rises
    rows, boots = [], {}
    for a in arms:
        j, al = pos[a]
        if j is None:
            continue
        m = focus_metrics(A[a], j, al); boots[a] = focus_metrics(A[a], j, al, W)
        for c in FOCUS_CONDS:
            if f"{c}|normal" not in m:
                continue
            row = dict(arm=a, cond=c, thr=round(float(FOCUS_TH[j] * (1 - al) + FOCUS_TH[j + 1] * al), 4))
            for k in FOCUS_SHOW:
                row[k] = round(float(m.get(f"{c}|{k}", np.nan)), 4 if not k.startswith(("fpm", "lat")) else 2)
            rows.append(row)
    df = pd.DataFrame(rows)
    print(f"## at {target:g} false events/min overall (nominal)" + (f", other arms at the same {match} recall" if match else "") + "\n")
    print(df.to_markdown(index=False))
    ref = arms[0]; delta = []
    for a in arms[1:]:
        if a not in boots or ref not in boots:
            continue
        for c in FOCUS_CONDS:
            if f"{c}|normal" not in boots[a] or f"{c}|normal" not in boots[ref]:
                continue
            row = dict(arm=a, cond=c)
            for k in FOCUS_SHOW:
                key = f"{c}|{k}"
                if key not in boots[a] or key not in boots[ref]:
                    row[k] = ""; continue
                d = boots[a][key] - boots[ref][key]
                lo, hi = np.nanpercentile(d, [2.5, 97.5]); sc = 100 if not k.startswith(("fpm", "lat")) else 1
                row[k] = f"{(np.nanmean(d)) * sc:+.2f} [{lo * sc:+.2f}, {hi * sc:+.2f}]"
            delta.append(row)
    if delta:
        print(f"\n## difference to {ref} (recall in points, fpm in events/min; 95% person bootstrap)\n")
        print(pd.DataFrame(delta).to_markdown(index=False))
    return df


# ============================================================================ re-scoring saved models
def rescore(arms, conditions=("nominal", "placement"), n_boot=2000, seed=3):
    """Per-person counts of saved fold models at each arm's fpm<=2 and fpm<=5 thresholds (chosen on nominal,
    pooled): person-bootstrap 95% intervals, paired differences vs the first arm, and a warm-up view (rows of
    frames 50..499, i.e. the first ~10 s after a (re)connect) next to the usable view."""
    import lightgbm as lgb
    D = load(); data = D["data"]; tr = D["tr"]; tables = stress_tables(D["n_recs"])
    res = {a: json.loads((OUT / f"{a}.json").read_text()) for a in arms}
    thr = {a: {b: at_budget(res[a]["pooled"]["nominal"], b)[0] for b in (2.0, 5.0)} for a in arms}
    rows = []                                       # (arm, budget, cond, view, person, TAP hits, TAP onsets, false, neg_ms)
    for f, (a_, b_) in enumerate(D["folds"]):
        val = Val(set(data["group"][tr[b_]]))
        person = load()["rpl"]["recs"].group.to_numpy(object)
        for arm in arms:
            spec = tuple(res[arm]["config"]["spec"])
            m = lgb.Booster(model_file=str(OUT / "models" / f"{arm}_fold{f}.txt"))
            for cond in conditions:
                p = np.empty(len(val.X))
                for s in range(0, len(val.X), 250_000):
                    Xb = val.X[s:s + 250_000]
                    if tables[cond]:
                        r = val.rec[s:s + 250_000]; Xb = apply_emg(Xb, **{k: v[r] for k, v in tables[cond].items()})
                    p[s:s + 250_000] = m.predict(build(Xb, spec)[0], num_threads=THREADS)
                for budget, t_ in thr[arm].items():
                    if t_ is None:
                        continue
                    for (s, e), t, lab, (rec, _, _, folder, ons, extra), st in zip(val.seg, val.t, val.label, val.items,
                                                                                  val.steady_from):
                        for view in ("usable", "warmup"):
                            if view == "warmup":
                                stop = st if st is not None else e - s
                                if stop <= 0:
                                    continue
                                pp, tt, ll = p[s:s + stop], t[:stop], lab[:stop]
                                oo = ons[ons <= tt[-1] - R.TAP_HIT_AFTER_MS] if len(tt) else ons[:0]
                            else:
                                pp, tt, ll, oo = p[s:e], t, lab, ons
                            c = R.TapCounts(); R.tap_replay_recording(pp >= t_, tt, ll, oo, folder, c)
                            fd = c.folder.get(folder, dict(onsets=0, hits=0, false=0, neg_ms=0.0))
                            rows.append((arm, budget, cond, view, person[rec], folder, fd["hits"], fd["onsets"],
                                         fd["false"], fd["neg_ms"]))
        log("GR_RESCORE_FOLD", f)
        del val
    df = pd.DataFrame(rows, columns=["arm", "budget", "cond", "view", "person", "folder", "hits", "onsets", "false", "neg_ms"])
    df.to_csv(OUT / "rescore_rows.csv.gz", index=False)
    return bootstrap(df, arms, n_boot, seed)


def bootstrap(df, arms, n_boot=2000, seed=3):
    rng = np.random.default_rng(seed)
    people = np.array(sorted(df.person.unique()))
    out = {}
    for (budget, cond, view), d in df.groupby(["budget", "cond", "view"]):
        tap = d[d.folder == "TAP"].groupby(["arm", "person"])[["hits", "onsets"]].sum()
        fal = d.groupby(["arm", "person"])[["false", "neg_ms"]].sum()
        def stats(idx):
            r = {}
            for a in arms:
                th = tap.loc[a].reindex(idx).fillna(0).sum() if a in tap.index.get_level_values(0) else None
                fa = fal.loc[a].reindex(idx).fillna(0).sum()
                r[a] = (th.hits / th.onsets if th is not None and th.onsets else np.nan,
                        fa["false"] / (fa.neg_ms / 60000.0) if fa.neg_ms else np.nan)
            return r
        point = stats(people)
        boots = [stats(rng.choice(people, len(people), replace=True)) for _ in range(n_boot)]
        key = f"budget{budget:g}|{cond}|{view}"
        out[key] = {}
        for a in arms:
            rec_b = np.array([b[a][0] for b in boots]); fpm_b = np.array([b[a][1] for b in boots])
            ent = dict(recall_TAP=round(point[a][0], 4), recall_ci=[round(x, 4) for x in np.nanpercentile(rec_b, [2.5, 97.5])],
                       fpm=round(point[a][1], 3), fpm_ci=[round(x, 3) for x in np.nanpercentile(fpm_b, [2.5, 97.5])])
            if a != arms[0]:
                d_rec = rec_b - np.array([b[arms[0]][0] for b in boots])
                ent["recall_minus_first"] = round(point[a][0] - point[arms[0]][0], 4)
                ent["recall_minus_first_ci"] = [round(x, 4) for x in np.nanpercentile(d_rec, [2.5, 97.5])]
            out[key][a] = ent
    (OUT / "rescore_bootstrap.json").write_text(json.dumps(out, indent=1))
    return out


def interp_budget(res, budget, view="pooled", key="recall_TAP"):
    """Recall at exactly `budget` false taps/min (nominal), linearly interpolated between the two grid thresholds
    around it; the same interpolation applied to every stress condition (the threshold is fixed by nominal)."""
    nom = res[view]["nominal"]
    ts = sorted(float(t) for t in nom if nom[t])
    fp = [nom[str(t)]["false_per_min"] for t in ts]
    for i in range(len(ts) - 1):
        if fp[i] >= budget >= fp[i + 1]:
            a = 0.0 if fp[i] == fp[i + 1] else (fp[i] - budget) / (fp[i] - fp[i + 1])
            out = {}
            for cond in STRESS:
                r0, r1 = res[view][cond][str(ts[i])], res[view][cond][str(ts[i + 1])]
                out[cond] = dict(recall=r0[key] + a * (r1[key] - r0[key]),
                                 fpm=r0["false_per_min"] + a * (r1["false_per_min"] - r0["false_per_min"]),
                                 SWIPE=r0["recall_SWIPE"] + a * (r1["recall_SWIPE"] - r0["recall_SWIPE"]))
            out["threshold"] = ts[i] + a * (ts[i + 1] - ts[i])
            return out
    return None


def table(view="pooled", budgets=(2.0, 5.0)):
    rows = []
    for p in sorted(OUT.glob("*.json")):
        res = json.loads(p.read_text())
        if view not in res or "pooled" not in res:
            continue
        for b in budgets:
            r = interp_budget(res, b, view)
            if not r:
                continue
            row = dict(arm=res["arm"], budget=b, TAP=r["nominal"]["recall"], SWIPE=r["nominal"]["SWIPE"])
            for cond in STRESS[1:]:
                row[f"TAP_{cond}"] = r[cond]["recall"]; row[f"fpm_{cond}"] = r[cond]["fpm"]
            row["worst_TAP"] = min(r[c]["recall"] for c in STRESS[1:])
            row["worst_fpm"] = max(r[c]["fpm"] for c in STRESS[1:])
            # recall x min(1, budget / false taps per minute): the trainer's objective, worst over the stresses
            row["worst_score"] = min(r[c]["recall"] * min(1.0, b / max(r[c]["fpm"], 1e-9)) for c in STRESS[1:])
            row["emg_share"] = round(float(np.mean([f["emg_gain_share"] for f in res["folds"]])), 3)
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["budget", "worst_score"], ascending=[True, False])
    print(df.round(3).to_markdown(index=False))
    return df


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else "pooled"); sys.exit(0)
    if sys.argv[1] == "table":
        table(sys.argv[2] if len(sys.argv) > 2 else "pooled"); sys.exit(0)
    if sys.argv[1] == "rule":                  # rule <base arm> <rule> [<rule> ...]: re-score saved models
        for r_ in sys.argv[3:]:
            rule_arm(sys.argv[2], r_)
        sys.exit(0)
    if sys.argv[1] == "rescore":
        print(json.dumps(rescore(sys.argv[2:]), indent=1)); sys.exit(0)
    if sys.argv[1] == "focus":
        print(json.dumps({a: v.get("fpm2.8") for a, v in focus(sys.argv[2:]).items()}, indent=1)); sys.exit(0)
    if sys.argv[1] == "focus_table":
        for tg in (2.8, 2.0):
            focus_table(sys.argv[2:], target=tg); print()
        sys.exit(0)
    if sys.argv[1] == "focus_hold":
        focus_hold(sys.argv[2:]); sys.exit(0)
    if sys.argv[1] == "focus_match":
        focus_table(sys.argv[2:], target=2.8, match="normal"); sys.exit(0)
    for arm_name in sys.argv[1:]:
        run_arm(arm_name)
