#!/usr/bin/env python3
"""TAP generalization research: which features, training tricks and labels make the TAP model robust to band
placement and electrode contact, without losing taps or adding false taps?

    python research/gesture/tap_generalization_research.py <arm> [<arm> ...]     # run arms (see ARMS)
    python research/gesture/tap_generalization_research.py report                # summary table (markdown)

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

CACHE = Path(os.environ.get("MUDRA_GR_CACHE", "/tmp/v3soft/tap/cache"))
OUT = Path(os.environ.get("MUDRA_GR_OUT", HERE / "results_tap_generalization"))
QUICK = os.environ.get("MUDRA_GR_QUICK") == "1"
THREADS = int(os.environ.get("MUDRA_GR_THREADS", os.cpu_count() or 2))
CONFIG = dict(objective="binary", n_estimators=400, learning_rate=0.1, num_leaves=31, max_depth=8,
              min_child_samples=100, reg_lambda=1.0, reg_alpha=1e-3, colsample_bytree=0.7, subsample=0.8,
              subsample_freq=1, max_bin=127, verbosity=-1, deterministic=True)
GRID = FS.TAP_GRID
STRESS = ("nominal", "gain", "tilt", "noise", "placement")    # see stress_tables()
TEACHER_THRESHOLD = 0.5                 # fallback only (no movement recordings among the people)
NEG_FOLDERS = ("FINGER", "WRIST", "ARM", "BAND", "NOTHING")

COLS = list(g.RAW_126) + list(g.LEAN8_24)
CI = {c: i for i, c in enumerate(COLS)}
ACC = [CI[c] for c in g.RAW_126 if c.startswith(("p_core->acc_psd", "p_core->acc_rate"))]
ACC_NOW = [CI[f"p_core->acc_psd[0][{a}][{z}]"] for a in "XYZ" for z in range(6)]
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
    if "diff" in spec:
        add(np.clip(X[:, PA] - X[:, PB], -32768, 32767), [f"p_core->diff({COLS[a]}, {COLS[b]})" for a, b in PAIRS], True)
    if "ratio" in spec:
        add((X[:, PA] + 1.0) / (X[:, PB] + 1.0), [f"p_core->ratio({COLS[a]}, {COLS[b]})" for a, b in PAIRS], True)
    if "logratio" in spec:                      # exact log power ratio (would need a firmware LUT): upper bound
        lin = lambda c: np.power(10.0, c.astype(np.float64) / 1e4) - 1.0
        add(np.log10(lin(X[:, PA]) + 1e-4) - np.log10(lin(X[:, PB]) + 1e-4),
            [f"logratio({COLS[a]}, {COLS[b]})" for a, b in PAIRS], True)
    if "wlratio" in spec:
        add((X[:, WA] + 1.0) / (X[:, WB] + 1.0), [f"p_core->ratio({COLS[a]}, {COLS[b]})" for a, b in WL_PAIRS], True)
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
    and zone; noise power log-uniform 0.1..1.0 per channel (median total EMG power is ~0.7-1.1); placement = all."""
    rng = np.random.default_rng(seed)
    gain = np.power(2.0, rng.uniform(-1, 1, (n_recs, 3)))
    tilt = np.power(2.0, rng.uniform(-tilt_oct, tilt_oct, (n_recs, 3, 5)))
    noise = np.power(10.0, rng.uniform(noise_log10[0], noise_log10[1], (n_recs, 3)))
    return dict(nominal=None, gain=dict(gain=gain), tilt=dict(tilt=tilt), noise=dict(noise=noise),
                placement=dict(gain=gain, tilt=tilt, noise=noise))


# ============================================================================ data
_D = {}


def load():
    if _D:
        return _D
    t0 = time.perf_counter()
    data, rpl, _ = g.load_cache(CACHE, COLS)
    assert data["X"].shape[1] == 150, data["X"].shape
    tr, _te = g.split_heldout(data, "tap")
    tr = tr[data["kind"][tr] != 6]                                  # confirmed labels only (no soft rows)
    folds = list(GroupKFold(n_splits=g.CV_FOLDS).split(tr, data["y"][tr], data["group"][tr]))
    if QUICK:
        people = sorted(set(data["group"][tr]))[:12]
        tr = tr[np.isin(data["group"][tr], people)]
        folds = list(GroupKFold(n_splits=3).split(tr, data["y"][tr], data["group"][tr]))
    onsets = pd.read_csv(CACHE / "tap_onsets.csv")
    _D.update(data=data, rpl=rpl, tr=tr, folds=folds, onsets=onsets, n_recs=len(rpl["recs"]))
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
        self.seg, pos = [], 0
        for _, a, b, *_ in self.items:
            self.seg.append((pos, pos + b - a)); pos += b - a
        self.t = [np.asarray(rpl["t"][a:b], np.float64) for _, a, b, *_ in self.items]
        self.label = [np.asarray(rpl["label"][a:b]) for _, a, b, *_ in self.items]
        self.steady_from = []                       # first steady row (frame >= 500) of each recording, or None
        for _, a, b, *_ in self.items:
            nz = np.flatnonzero(np.asarray(rpl["steady"][a:b], bool))
            self.steady_from.append(int(nz[0]) if len(nz) else None)

    def predict(self, model, k, spec, stress=None, chunk=250_000):
        p = np.empty(len(self.X), np.float64)
        for s in range(0, len(self.X), chunk):
            Xb = self.X[s:s + chunk]
            if stress:
                r = self.rec[s:s + chunk]
                Xb = apply_emg(Xb, **{name: tab[r] for name, tab in stress.items()})
            F, _, _ = build(Xb, spec)
            p[s:s + chunk] = model.booster_.predict(F, num_iteration=k, num_threads=THREADS)
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
    if labels == "default":
        keep = np.ones(len(rows), bool); extra = None
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
    r = rows[keep]
    X = data["X"][r]; yy = data["y"][r].astype(int); files = data["file"][r]; w0 = data["w"][r]; rec = data["rec"][r]
    if extra is not None and len(extra):
        Xe = np.asarray(rpl["X"][np.sort(extra)])
        rec_e = np.searchsorted(recs.end.to_numpy(), np.sort(extra), side="right")
        X = np.concatenate([X, Xe]); yy = np.r_[yy, np.ones(len(extra), int)]
        files = np.r_[files, recs.file.to_numpy(object)[rec_e]]; w0 = np.r_[w0, np.ones(len(extra), np.float32)]
        rec = np.r_[rec, rec_e]
    info.update(train_rows=int(len(yy)), positives=int((yy == 1).sum()))
    return X, yy, files, w0, rec, info


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


# Arms: features (spec), labels, train-time EMG gain augmentation (octaves), EMG split-gain penalty (weight).
ARMS = {
    "raw126":           dict(spec=("acc", "emgraw")),
    "lean8":            dict(spec=("acc", "emgraw", "lean")),
    "acc_only":         dict(spec=("acc",)),
    "acc_diff":         dict(spec=("acc", "diff", "wlratio")),
    "acc_ratio":        dict(spec=("acc", "ratio", "wlratio")),
    "acc_logratio":     dict(spec=("acc", "logratio", "wlratio")),
    "raw126_diff":      dict(spec=("acc", "emgraw", "diff", "wlratio")),
    "raw126_aug":       dict(spec=("acc", "emgraw"), aug=1.0),
    "raw126_augp":      dict(spec=("acc", "emgraw"), augp=True),
    "acc_diff_augp":    dict(spec=("acc", "diff", "wlratio"), augp=True),
    "raw126_penalty":   dict(spec=("acc", "emgraw"), penalty=0.3),
    "raw126_coarse":    dict(spec=("acc", "emgraw"), emg_bins=8),
    "pure_tap":         dict(spec=("acc", "emgraw"), labels="pure_tap"),
    "acc_rule":         dict(spec=("acc", "emgraw"), labels="acc_rule"),
    "teacher":          dict(spec=("acc", "emgraw"), labels="teacher"),
    "teacher_anchor":   dict(spec=("acc", "emgraw"), labels="teacher_anchor"),
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
        elif tok in ("augp", "augm", "augh"):
            base[tok] = True
        elif tok.startswith("aug"):
            base["aug"] = float(tok[3:] or 1)
        elif tok.startswith("pen"):
            base["penalty"] = float(tok[3:])
        else:
            raise ValueError(f"Unknown arm token {tok}")
    return base


def run_arm(name):
    arm = arm_from_name(name); spec = arm["spec"]; labels = arm.get("labels", "default")
    D = load(); data = D["data"]; tr = D["tr"]
    budget = g.budget_bytes(g.MODEL_BUDGET_KB)
    sums = {v: {c: {thr: R.TapCounts() for thr in GRID} for c in STRESS} for v in ("usable", "steady")}
    per_fold = []
    if labels.startswith("teacher"):
        D.setdefault("teacher", {})
    for f, (a, b) in enumerate(D["folds"]):
        t0 = time.perf_counter()
        trn = tr[a]; val_people = set(data["group"][tr[b]])
        D["fold"] = f
        if labels.startswith("teacher") and f not in D["teacher"]:
            D["teacher"][f] = teacher_labels(trn, f)
        X150, y, files, w0, rec, info = train_set(trn, labels)
        if arm.get("aug"):
            X150 = apply_gain(X150, gain_table(D["n_recs"], arm["aug"], 1000 + f)[rec])
        if arm.get("augp"):                        # placement augmentation: gain + tilt + noise (other draws)
            tab = stress_tables(D["n_recs"], seed=2000 + f)["placement"]
            X150 = apply_emg(X150, **{k_: v[rec] for k_, v in tab.items()})
        if arm.get("augm"):                        # milder placement augmentation: tilt +-0.25 oct, noise 0.03..0.3
            tab = stress_tables(D["n_recs"], seed=3000 + f, tilt_oct=0.25, noise_log10=(-1.5, -0.5))["placement"]
            X150 = apply_emg(X150, **{k_: v[rec] for k_, v in tab.items()})
        if arm.get("augh"):                        # half the recordings augmented (placement), half untouched
            tab = stress_tables(D["n_recs"], seed=4000 + f)["placement"]
            pick = np.random.default_rng(5000 + f).random(D["n_recs"]) < 0.5
            mrow = pick[rec]
            X150[mrow] = apply_emg(X150[mrow], **{k_: v[rec[mrow]] for k_, v in tab.items()})
        F, names, is_emg = build(X150, spec)
        del X150
        extra = {}
        if arm.get("penalty") is not None:
            extra["feature_contri"] = [arm["penalty"] if e else 1.0 for e in is_emg]
        if arm.get("emg_bins"):
            extra["max_bin_by_feature"] = [arm["emg_bins"] if e else CONFIG["max_bin"] for e in is_emg]
        m = fit(F, y, weights(y, files, w0), budget, extra_params=extra)
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
            p = val.predict(m, k, spec, tables[cond])
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
    if sys.argv[1] == "rescore":
        print(json.dumps(rescore(sys.argv[2:]), indent=1)); sys.exit(0)
    for arm_name in sys.argv[1:]:
        run_arm(arm_name)
