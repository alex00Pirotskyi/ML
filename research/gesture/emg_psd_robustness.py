#!/usr/bin/env python3
"""EMG PSD feature robustness (2026-11-27): which inputs built from the 45 firmware EMG PSD codes keep working when
the band sits differently, the skin contact changes, the band rotates or the power grid leaks into the electrodes?

Data: the TAP F1 proxy cache (legacy_f1_proxy.py `cache`: 2,881 DATA_TAP files, 5.65M rows), the labels of make
train (legacy signal labels, every 2nd TAP/SWIPE row left out: y_legacyhalf) and the file-level, gesture-aware folds
of DATA/fold_roadmap_files.json. Nothing is split by person. Sessions (user + date) appear only in `describe`, to
say where the variance of the EMG levels comes from.

The firmware codes: E = round(10000 * log10(1 + p)), p = the FFT power of a zone (bins 1-8, 9-18, 19-30, 31-76,
77-127 of a 256-point FFT), so p = 10^(E/10000) - 1 exactly (up to rounding). Slots: emg_psd[0] (now),
emg_psd[EMG_PSD_HISTORY_SIZE - 1] (oldest history entry, ~0.3 s ago), emg_psd_mean (mean of the last 15 codes).

Commands
  describe              encoding regime, per-recording variance, channel pattern, inputs one at a time
                        -> results_emg_psd_robustness/describe.json, univariate*.csv
  frames                Lean-lite frame counter per cached row (to recompute Lean-lite under stress) -> frames.npy
  stress ARM [ARM ...]  per arm: train on 4 folds, score the 5th nominal and under every stress, all 5 folds
                        (ARM e.g. acc+tz, acc+tz@aug, rfe81; see build()) -> stress_<arm>.json
  table                 the tables of EMG_PSD_ROBUST_FEATURES.md from the stress runs
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research" / "gesture"), str(ROOT / "claude_labeling")]
warnings.filterwarnings("ignore")

import legacy_f1_proxy as LP  # noqa: E402

OUT = Path(os.environ.get("EMG_ROBUST_DIR", "/tmp/work/emg"))
RES = ROOT / "research" / "gesture" / "results_emg_psd_robustness"
LABEL = os.environ.get("EMG_ROBUST_LABEL", "legacyhalf")
BINS = np.array([8, 10, 12, 46, 51])             # FFT bins per EMG PSD zone
SLOTS = ("p_core->emg_psd[0]", "p_core->emg_psd[EMG_PSD_HISTORY_SIZE - 1]", "p_core->emg_psd_mean")
NOW, OLD, MEAN = 0, 1, 2
WL_PER_SQRT_POWER = 850.0                        # WL ~ 850 * sqrt(total EMG PSD power) (tap_features.py)
FLOOR = float(os.environ.get("EMG_ROBUST_FLOOR", "0.1"))   # power floor of the re-floored log inputs (firmware: 1)
THS = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97])
RFE_JSON = Path(os.environ.get("EMG_ROBUST_RFE", "/tmp/work/rfe_selected_v3.json"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================================ cache and indices
def cache():
    files, cols, z = LP.load_cache()
    ci = {c: i for i, c in enumerate(cols["raw"])}
    emg = np.array([[[ci[f"{SLOTS[s]}[CH{c}][{k}]"] for k in range(5)] for c in (1, 2, 3)] for s in range(3)])
    wl = np.array([[ci[f"p_core->{w}[CH{c}]"] for w in ("wl", "wl_head", "wl_tail")] for c in (1, 2, 3)])
    return files, cols, z, emg, wl


def lin(E):
    """firmware codes -> zone power"""
    return np.power(10.0, np.asarray(E, np.float64) / 1e4) - 1.0


def code(p):
    """zone power -> firmware code (uint16)"""
    return np.clip(np.round(1e4 * np.log10(1.0 + np.maximum(p, 0.0))), 0, 65535)


def emg_name(s, c, k):
    return f"{SLOTS[s]}[CH{c + 1}][{k}]"


# ============================================================================================ perturbations
CHUNK = 150_000


def perturb(R, emg, wl, gain=None, tilt=None, add=None, mix=None):
    """perturb_rows over row chunks (the same result, bounded memory)"""
    out = np.empty((len(R), R.shape[1]), np.float32)
    for a in range(0, len(R), CHUNK):
        sl = slice(a, a + CHUNK)
        pick = lambda v: None if v is None else np.asarray(v)[sl]
        out[sl] = perturb_rows(R[sl], emg, wl, pick(gain), pick(tilt), pick(add), pick(mix))
    return out


def perturb_rows(R, emg, wl, gain=None, tilt=None, add=None, mix=None):
    """A copy of raw rows R (n, 126) after a contact / placement change of the EMG (ACC untouched). Per-row arrays:
    gain (n, 3) amplitude factor per channel (contact impedance); tilt (n, 3, 5) amplitude factor per channel and
    zone (tissue filter, a different muscle under the electrode); add (n, 3, 5) power added per channel and zone,
    stationary, so in every slot (white contact noise, power-line interference); mix (n,) share of the next
    channel's power mixed into each channel (band rotation). The codes are changed exactly per slot (the history
    mean is treated like one spectrum). WL ~ sqrt(power): it follows the mixing in quadrature, the gain / tilt by
    the amplitude factor of the newest spectrum and the added power in quadrature."""
    n = len(R)
    Y = np.array(R, np.float32, copy=True)
    P = lin(R[:, emg.reshape(-1)]).reshape(n, 3, 3, 5)
    W = np.asarray(R[:, wl.reshape(-1)], np.float64).reshape(n, 3, 3)
    if mix is not None:
        a = np.asarray(mix, np.float64)
        P = (1 - a)[:, None, None, None] * P + a[:, None, None, None] * np.roll(P, -1, axis=2)
        W = np.sqrt((1 - a)[:, None, None] * W ** 2 + a[:, None, None] * np.roll(W, -1, axis=1) ** 2)
    if gain is not None or tilt is not None:
        G = np.ones((n, 3, 5))
        if gain is not None:
            G = G * np.asarray(gain, np.float64)[:, :, None]
        if tilt is not None:
            G = G * np.asarray(tilt, np.float64)
        p0 = P[:, NOW].sum(-1)
        P = P * (G ** 2)[:, None]
        p1 = P[:, NOW].sum(-1)
        tau = np.sqrt(np.where(p0 > 1e-9, p1 / np.maximum(p0, 1e-9), (G ** 2).mean(-1)))
        W = W * tau[:, :, None]
    if add is not None:
        A = np.asarray(add, np.float64)
        P = P + A[:, None]
        W = np.sqrt(W ** 2 + WL_PER_SQRT_POWER ** 2 * A.sum(-1)[:, :, None])
    Y[:, emg.reshape(-1)] = code(P).reshape(n, -1)
    Y[:, wl.reshape(-1)] = np.round(W).reshape(n, -1)
    return Y


STRESSES = ("nominal", "gain", "tilt", "noise", "mains0", "mains1", "mains2", "mains3", "mains4", "rotation", "placement")
FOLDS = tuple(int(c) for c in os.environ.get("EMG_ROBUST_FOLDS", "01234"))
RUN_STRESSES = tuple(os.environ.get("EMG_ROBUST_STRESSES", ",".join(STRESSES)).split(","))


def stress_draws(n_recs, seed=77):
    """Per-recording draws, the same for every arm.
    gain: amplitude x 2^U(-1, 1) per channel (+-6 dB power; the observed SD of a channel's level between
      recordings of one session is ~2.5 dB, between recordings overall 3.3-4 dB).
    tilt: amplitude x 2^U(-0.5, 0.5) per channel and zone (+-3 dB; observed zone-shape SD 0.4-2.9 dB).
    noise: white contact noise, total power 10^U(-1, 0) per channel spread over the zones like their bin counts
      (median quiet power of a channel is ~0.4).
    mainsK: one power-grid line in zone K, power 10^U(-1.5, 0) per recording x coupling U(0.5, 1.5) per channel,
      stationary (the same in every slot).
    rotation: each channel takes U(0.2, 0.5) of its neighbour's power.
    placement: gain + tilt + noise (the 2026-11-19 research's combined stress)."""
    rng = np.random.default_rng(seed)
    gain = 2.0 ** rng.uniform(-1, 1, (n_recs, 3))
    tilt = 2.0 ** rng.uniform(-0.5, 0.5, (n_recs, 3, 5))
    noise = 10.0 ** rng.uniform(-1, 0, (n_recs, 3))
    white = noise[:, :, None] * (BINS / BINS.sum())[None, None]
    level = 10.0 ** rng.uniform(-1.5, 0, n_recs)
    coupling = rng.uniform(0.5, 1.5, (n_recs, 3))
    mix = rng.uniform(0.2, 0.5, n_recs)
    out = {"nominal": {}, "gain": dict(gain=gain), "tilt": dict(tilt=tilt), "noise": dict(add=white)}
    for k in range(5):
        a = np.zeros((n_recs, 3, 5)); a[:, :, k] = level[:, None] * coupling
        out[f"mains{k}"] = dict(add=a)
    out["rotation"] = dict(mix=mix)
    out["placement"] = dict(gain=gain, tilt=tilt, add=white)
    return out


def aug_draws(n_recs, seed=2026):
    """Training augmentation: the 2026-11-19 research's 'mild' placement draw (gain +-1 octave, tilt +-0.25 octave,
    white noise 10^U(-1.5, -0.5)). No power-line and no rotation term, so those stresses stay out of its family."""
    rng = np.random.default_rng(seed)
    gain = 2.0 ** rng.uniform(-1, 1, (n_recs, 3))
    tilt = 2.0 ** rng.uniform(-0.25, 0.25, (n_recs, 3, 5))
    noise = 10.0 ** rng.uniform(-1.5, -0.5, (n_recs, 3))
    return dict(gain=gain, tilt=tilt, add=noise[:, :, None] * (BINS / BINS.sum())[None, None])


def aug2_draws(n_recs, seed=2027):
    """aug_draws plus, on a random half of the recordings each, power-grid hum in one random zone (power
    10^U(-2, -0.5) x coupling U(0.5, 1.5) per channel) and band rotation (mixing U(0, 0.4))."""
    d = aug_draws(n_recs, seed)
    rng = np.random.default_rng(seed + 1)
    hum = np.zeros((n_recs, 3, 5))
    on = rng.random(n_recs) < 0.5
    zone = rng.integers(0, 5, n_recs)
    lvl = 10.0 ** rng.uniform(-2, -0.5, n_recs)[:, None] * rng.uniform(0.5, 1.5, (n_recs, 3))
    hum[np.flatnonzero(on), :, zone[on]] = lvl[on]
    d["add"] = d["add"] + hum
    d["mix"] = np.where(rng.random(n_recs) < 0.5, rng.uniform(0, 0.4, n_recs), 0.0)
    return d


def rows_of(draw, rec):
    """per-recording draw -> per-row arrays (rec: recording index of every row)"""
    return {k: v[rec] for k, v in draw.items()}


# ============================================================================================ inputs
def _pairs_t():
    """within channel and zone, slot vs slot: now - mean, now - oldest, oldest - mean (firmware diff)"""
    return [((sa, c, k), (sb, c, k)) for c in range(3) for k in range(5) for sa, sb in ((NOW, MEAN), (NOW, OLD), (OLD, MEAN))]


def _pairs_z():
    """within channel and slot, zone vs zone (spectral shape)"""
    return [((s, c, a), (s, c, b)) for s in range(3) for c in range(3) for a in range(5) for b in range(a + 1, 5)]


def _pairs_x():
    """across channels, same slot and zone"""
    return [((s, c1, k), (s, c2, k)) for s in range(3) for k in range(5) for c1 in range(3) for c2 in range(c1 + 1, 3)]


PAIRS = {"tdiff": _pairs_t(), "zdiff": _pairs_z(), "xdiff": _pairs_x()}
assert [len(v) for v in PAIRS.values()] == [45, 90, 45]


def _take(E, pairs):
    a = np.array([np.ravel_multi_index(p, (3, 3, 5)) for p, _ in pairs])
    b = np.array([np.ravel_multi_index(q, (3, 3, 5)) for _, q in pairs])
    F = E.reshape(len(E), -1)
    return F[:, a], F[:, b]


def emg_family(E, fam):
    """E (n, 3 slots, 3 channels, 5 zones) float codes -> (X, names) of one EMG input family"""
    n = len(E)
    if fam == "abs":
        return E.reshape(n, -1).astype(np.float32), [emg_name(s, c, k) for s in range(3) for c in range(3) for k in range(5)]
    if fam in ("tdiff", "zdiff", "xdiff"):
        A, B = _take(E, PAIRS[fam])
        return (np.clip(A - B, -32768, 32767).astype(np.float32),
                [f"p_core->diff({emg_name(*p)}, {emg_name(*q)})" for p, q in PAIRS[fam]])
    if fam in ("ratio", "tratio"):               # the firmware ratio on the tdiff + zdiff pairs (tratio: tdiff pairs only)
        pr = PAIRS["tdiff"] + (PAIRS["zdiff"] if fam == "ratio" else [])
        A, B = _take(E, pr)
        return ((A + 1.0) / (B + 1.0)).astype(np.float32), [f"p_core->ratio({emg_name(*p)}, {emg_name(*q)})" for p, q in pr]
    if fam in ("logp", "logp0"):                 # re-floored log power, 1000 per dB: 10000*log10(p + FLOOR); logp0: floor 1e-4
        fl = FLOOR if fam == "logp" else 1e-4
        L = (1e4 * np.log10(lin(E) + fl)).astype(np.float32)          # float32: the training matrix is large
        pr = PAIRS["tdiff"] + PAIRS["zdiff"]
        A, B = _take(L, pr)
        return (A - B).astype(np.float32), [f"{fam}({emg_name(*p)}, {emg_name(*q)})" for p, q in pr]
    if fam == "ddiff":                           # change of the spectral shape: (now_a - ref_a) - (now_b - ref_b), ref = mean / oldest
        parts, names = [], []
        for ref, rn in ((MEAN, "mean"), (OLD, "old")):
            d = E[:, NOW] - E[:, ref]                # (n, 3, 5)
            for c in range(3):
                for a in range(5):
                    for b in range(a + 1, 5):
                        parts.append(d[:, c, a] - d[:, c, b])
                        names.append(f"ddiff(now-{rn})[CH{c + 1}][{a}]-[{b}]")
        return np.clip(np.stack(parts, 1), -32768, 32767).astype(np.float32), names
    if fam == "chpool":                          # channel-symmetric: unchanged when the channels swap places
        d1 = E[:, NOW] - E[:, MEAN]; d2 = E[:, NOW] - E[:, OLD]
        parts, names = [], []
        for nm, d in (("now-mean", d1), ("now-old", d2)):
            for st, fn in (("max", np.max), ("mean", np.mean), ("min", np.min)):
                parts.append(fn(d, axis=1)); names += [f"chpool.{st}_ch({nm} code)[{k}]" for k in range(5)]
        P = lin(E).sum(axis=2).astype(np.float32)   # (n, slot, zone): power summed over the channels
        Lz = 1e4 * np.log10(P + FLOOR); Lt = 1e4 * np.log10(P.sum(-1) + FLOOR)
        parts += [Lz[:, NOW] - Lz[:, MEAN], Lz[:, NOW] - Lz[:, OLD]]
        names += [f"chpool.sum_ch(power) now-mean dB[{k}]" for k in range(5)] + [f"chpool.sum_ch(power) now-old dB[{k}]" for k in range(5)]
        parts += [(Lt[:, NOW] - Lt[:, MEAN])[:, None], (Lt[:, NOW] - Lt[:, OLD])[:, None]]
        names += ["chpool.total power now-mean dB", "chpool.total power now-old dB"]
        return np.hstack(parts).astype(np.float32), names
    raise KeyError(fam)


EMG_FAMILIES = ("abs", "tdiff", "zdiff", "xdiff", "ratio", "logp", "logp0", "chpool", "ddiff")
ARM_ONLY = ("tratio",)


class Arm:
    """An input set: 'acc' (72 raw ACC fields + 36 ACC ratios) plus EMG families joined by '+'
    (e.g. 'acc+tdiff+zdiff'; 'tz' = tdiff + zdiff), or 'rfe81' (make train's selected inputs on fold 0, with
    Lean-lite), or 'sel-NAME' (the inputs of OUT/selected_NAME.json, e.g. sel-robust: the feature elimination over
    make train's candidates with EMG=robust, on the same sample as rfe81). Suffix '@aug' trains on rows with the mild placement augmentation (gain, tilt, white noise),
    '@aug2' on rows with that plus power-grid hum and band rotation (aug2_draws)."""

    def __init__(self, spec, cols, emg, wl=None):
        self.spec = spec
        self._wl = wl
        base, _, aug = spec.partition("@")
        assert aug in ("", "aug", "aug2"), spec
        self.aug = aug
        self.raw_cols = cols["raw"]; self.base_cols = cols["raw"] + cols["lite"]
        self.emg = emg
        if base == "rfe81" or base.startswith("sel-"):      # sel-NAME: OUT/selected_NAME.json (select_tap_f1_features)
            src = RFE_JSON if base == "rfe81" else OUT / f"selected_{base[4:]}.json"
            names = json.loads(src.read_text())["selected"]
            self.fm = LP.FeatureMaker(self.base_cols, names); self.fams = []; self.uses_lite = True
            self.names = list(names)
            return
        toks = base.split("+")
        assert toks[0] == "acc", spec
        fams = []
        for t in toks[1:]:
            fams += ["tdiff", "zdiff"] if t == "tz" else [t]
        self.uses_lite = "lite" in fams                  # the 15 Lean-lite inputs (scale-free WL history)
        fams = [f for f in fams if f != "lite"]
        assert all(f in EMG_FAMILIES + ARM_ONLY for f in fams), spec
        acc = [c for c in self.raw_cols if c.startswith("p_core->acc_")] + LP.acc_ratio_names()
        direct = acc + (list(cols["lite"]) if self.uses_lite else [])
        self.fm = LP.FeatureMaker(self.base_cols, direct); self.fams = fams
        self.names = list(direct) + [nm for f in fams for nm in emg_family(np.zeros((1, 3, 3, 5)), f)[1]]

    def __call__(self, R, L, draw=None):
        """inputs of rows R (raw) / L (Lean-lite); draw: per-row perturbation applied to R first (training augmentation)"""
        X = np.empty((len(R), len(self.names)), np.float32)
        for a in range(0, len(R), CHUNK):
            sl = slice(a, a + CHUNK)
            Rc = np.asarray(R[sl], np.float32)
            if draw is not None:
                Rc = perturb_rows(Rc, self.emg, self._wl, **{k: v[sl] for k, v in draw.items()})
            B = np.hstack([Rc, np.asarray(L[sl], np.float32)])
            j = self.fm(B); k = j.shape[1]; X[sl, :k] = j
            if self.fams:
                E = Rc[:, self.emg.reshape(-1)].reshape(len(Rc), 3, 3, 5)
                for f in self.fams:
                    v = emg_family(E, f)[0]; X[sl, k:k + v.shape[1]] = v; k += v.shape[1]
        return X


# ============================================================================================ describe
def _auc(pos, neg):
    """Mann-Whitney AUC (ties averaged)"""
    from scipy.stats import rankdata
    x = np.r_[pos, neg]; r = rankdata(x)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def row_types(files, z, y):
    """1 tap row; 0 quiet row of a TAP / SWIPE recording (state 0, no pointer press within 300 ms); 2 a row of a
    no-tap recording (finger, fist, wrist, arm, band, nothing); -1 other"""
    t = np.asarray(z["t"]); ptr = np.asarray(z["ptr"]).astype(bool)
    rt = np.full(len(y), -1, np.int8)
    for _, r in files.iterrows():
        sl = slice(r.start, r.start + r.n)
        if r.folder not in ("TAP", "SWIPE"):
            rt[sl] = 2; continue
        tt = t[sl].astype(np.float64); pt = tt[ptr[sl]]; yy = y[sl]
        near = np.zeros(len(tt), bool)
        if len(pt):
            j = np.searchsorted(pt, tt)
            near = np.minimum(np.abs(tt - pt[np.clip(j - 1, 0, len(pt) - 1)]), np.abs(tt - pt[np.clip(j, 0, len(pt) - 1)])) <= 300
        rt[sl] = np.where(yy == 1, 1, np.where((yy == 0) & ~near, 0, -1))
    return rt


def cmd_describe():
    files, cols, z, emg, wl = cache()
    y = np.load(LP.OUT / "y_legacy.npy")                # every tap row (the descriptive view keeps all of them)
    rt = row_types(files, z, y)
    names = files.rel.str.split("/").str[-1]
    files["session"] = names.str.split("_").str[0] + "|" + names.str.extract(r"_(\d{4}-\d{2}-\d{2})_")[0]
    rec = np.repeat(np.arange(len(files)), files.n.to_numpy())
    raw = z["raw"]
    out = {}

    # ---- 1. encoding regime: where the codes sit on log10(1 + p)
    rng = np.random.default_rng(0)
    reg = {}
    for k, nm in ((0, "quiet"), (1, "tap"), (2, "no-tap recordings")):
        ix = np.sort(rng.choice(np.flatnonzero(rt == k), 150_000, replace=False))
        p = lin(np.asarray(raw[ix][:, emg[NOW].reshape(-1)])).reshape(-1, 3, 5)
        reg[nm] = {f"zone{q}": dict(p50=float(np.median(p[:, :, q])), p95=float(np.percentile(p[:, :, q], 95)),
                                    linear=float((p[:, :, q] < 0.25).mean()), log=float((p[:, :, q] >= 4).mean()))
                   for q in range(5)}
        reg[nm]["all"] = dict(linear=float((p < 0.25).mean()), log=float((p >= 4).mean()), p50=float(np.median(p)))
    out["regime"] = reg
    log("regime:", {k: v["all"] for k, v in reg.items()})

    # ---- 2. per-recording levels: quiet and tap medians of every TAP / SWIPE recording (newest spectrum, dB)
    recs = []
    for i, r in files[files.folder.isin(["TAP", "SWIPE"])].iterrows():
        sl = slice(r.start, r.start + r.n); k = rt[sl]
        if (k == 0).sum() < 100 or (k == 1).sum() < 20:
            continue
        p = lin(np.asarray(raw[sl][:, emg[NOW].reshape(-1)])).reshape(-1, 3, 5)
        q, tp = np.median(p[k == 0], 0), np.median(p[k == 1], 0)
        if (q <= 1e-4).any():                           # a dead channel (2 recordings)
            continue
        recs.append(dict(i=i, session=files.session[i], quiet=q, tap=tp))
    Q = 10 * np.log10(np.stack([d["quiet"] for d in recs])); T = 10 * np.log10(np.stack([d["tap"] for d in recs]))
    sess = np.array([d["session"] for d in recs])
    a = Q.mean((1, 2), keepdims=True); b = Q.mean(2, keepdims=True) - a; s = Q - a - b

    def split(v):
        df = pd.DataFrame(dict(v=v, g=sess))
        return dict(total=float(v.std()), within_session=float(df.groupby("g").v.transform(lambda x: x - x.mean()).std()),
                    between_session=float(df.groupby("g").v.mean().std()))
    eff = T - Q                                          # tap effect per channel and zone (dB)
    ech = eff.mean(2)                                    # per channel
    pat = ech - ech.mean(1, keepdims=True)               # which channel sees the tap (pattern), dB
    pool = 10 * np.log10(np.stack([d["tap"].sum((0, 1)) / d["quiet"].sum((0, 1)) for d in recs]))
    out["levels"] = dict(
        n_recordings=len(recs), n_sessions=int(len(set(sess))),
        quiet_sd_db=np.round(Q.std(0), 2).tolist(),
        overall=split(a[:, 0, 0]), channel=[split(b[:, c, 0]) for c in range(3)],
        shape_sd_db=np.round(s.std(0), 2).tolist(),
        shape_split=[[split(s[:, c, q]) for q in range(5)] for c in range(3)],
        tap_effect_db=dict(mean=np.round(eff.mean(0), 2).tolist(), sd=np.round(eff.std(0), 2).tolist()),
        channel_effect=dict(mean=np.round(ech.mean(0), 2).tolist(), sd=np.round(ech.std(0), 2).tolist(),
                            best_channel_share=np.bincount(ech.argmax(1), minlength=3).tolist(),
                            pattern=[split(pat[:, c]) for c in range(3)]),
        pooled_effect=dict(mean=float(pool.mean()), sd=float(pool.std()), below_3db=float((pool < 3).mean())),
        best_channel_effect=dict(mean=float(ech.max(1).mean()), sd=float(ech.max(1).std()), below_3db=float((ech.max(1) < 3).mean())),
        per_channel_below_3db=np.round((ech < 3).mean(0), 3).tolist(),
    )
    log("levels:", json.dumps(out["levels"]["channel_effect"]), "pooled", out["levels"]["pooled_effect"])

    # ---- 3. per-input view: univariate separation, recording dependence, stress sensitivity
    n_rec = len(files)
    draws = stress_draws(n_rec)
    pick = lambda k, m: np.sort(rng.choice(np.flatnonzero(rt == k), m, replace=False))
    ix = np.r_[pick(1, 40_000), pick(0, 40_000), pick(2, 80_000)]
    R = np.asarray(raw[ix], np.float32); kind = rt[ix]; rr = rec[ix]
    tap, mov, qui = kind == 1, kind == 2, kind == 0
    codes = lambda X: X[:, emg.reshape(-1)].reshape(-1, 3, 3, 5)
    rows, rule = {}, {}
    for f in EMG_FAMILIES:
        X, nm = emg_family(codes(R), f)
        for j, name in enumerate(nm):
            x = X[:, j].astype(np.float64)
            if not np.isfinite(x).all() or x.std() == 0:
                continue
            auc = _auc(x[tap], x[mov])
            sgn = 1.0 if auc >= 0.5 else -1.0
            th = np.percentile(sgn * x[mov], 95)        # a one-input rule with 5% false rows on no-tap recordings
            # recording dependence: share of the variance of quiet / no-tap rows that is a per-recording offset
            eta = []
            for m in (qui, mov):
                df = pd.DataFrame(dict(v=x[m], g=rr[m]))
                eta.append(float(df.groupby("g").v.transform("mean").var() / max(df.v.var(), 1e-12)))
            rows[(f, j)] = dict(family=f, name=name, auc=max(auc, 1 - auc), tpr_at5=float((sgn * x[tap] > th).mean()),
                                eta_quiet=eta[0], eta_move=eta[1])
            rule[(f, j)] = (sgn, th)
    for st in STRESSES[1:]:
        Es = codes(perturb(R, emg, wl, **rows_of(draws[st], rr)))
        for f in EMG_FAMILIES:
            Xs = emg_family(Es, f)[0]
            for j in range(Xs.shape[1]):
                if (f, j) not in rule:
                    continue
                sgn, th = rule[(f, j)]; xs = sgn * Xs[:, j].astype(np.float64)
                rows[(f, j)][f"fpr_{st}"] = float((xs[mov] > th).mean())
                rows[(f, j)][f"tpr_{st}"] = float((xs[tap] > th).mean())
        log(f"univariate stress {st}")
    U = pd.DataFrame(list(rows.values()))
    RES.mkdir(parents=True, exist_ok=True)
    U.to_csv(RES / "univariate.csv", index=False)
    univariate_groups(U).to_csv(RES / "univariate_groups.csv")
    tap_vs_fist(files, z, emg, rt, rec).to_csv(RES / "univariate_tap_vs_fist.csv", index=False)
    summ = []
    for f, g in U.groupby("family", sort=False):
        top = g.nlargest(10, "auc")
        d = dict(family=f, n=len(g), auc_top10=float(top.auc.mean()), tpr_top10=float(top.tpr_at5.mean()),
                 eta_quiet=float(top.eta_quiet.mean()), eta_move=float(top.eta_move.mean()))
        for st in STRESSES[1:]:
            d[f"fpr_{st}"] = float(top[f"fpr_{st}"].mean()); d[f"tpr_{st}"] = float(top[f"tpr_{st}"].mean())
        summ.append(d)
    out["univariate"] = summ
    (RES / "describe.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    S = pd.DataFrame(summ).set_index("family")
    print(S[["n", "auc_top10", "tpr_top10", "eta_quiet", "eta_move"] + [f"fpr_{s}" for s in STRESSES[1:]]].round(3))
    print(S[[f"tpr_{s}" for s in STRESSES[1:]]].round(3))


_SLOT_RX = r"(emg_psd\[0\]|emg_psd\[EMG_PSD_HISTORY_SIZE - 1\]|emg_psd_mean)\[CH(\d)\]\[(\d)\]"


def univariate_groups(U):
    """per group (ratio / log families split into over-time '_t' and zone-vs-zone '_z' pairs): the mean of its 10
    inputs with the best AUC (separation, per-recording constant, change of recall / false rows under each stress)"""
    import re

    def group(r):
        if r.family in ("ratio", "logp", "logp0"):
            (_, c1, z1), (_, c2, z2) = re.findall(_SLOT_RX, r["name"])
            return r.family + ("_t" if z1 == z2 else "_z")
        return r.family
    U = U.assign(group=U.apply(group, axis=1))
    st = [s for s in STRESSES[1:]]
    rows = []
    for g, d in U.groupby("group"):
        top = d.nlargest(10, "auc")
        r = dict(group=g, n=len(d), auc=top.auc.mean(), tpr=top.tpr_at5.mean(), eta_q=top.eta_quiet.mean(),
                 eta_m=top.eta_move.mean())
        for s_ in st:
            r["tpr_" + s_] = top["tpr_" + s_].mean() - top.tpr_at5.mean(); r["fpr_" + s_] = top["fpr_" + s_].mean()
        r["fpr_mains_worst"] = max(r[f"fpr_mains{k}"] for k in range(5))
        r["tpr_mains_worst"] = min(r[f"tpr_mains{k}"] for k in range(5))
        rows.append(r)
    return pd.DataFrame(rows).set_index("group").sort_values("eta_m")


def tap_vs_fist(files, z, emg, rt, rec, n=40_000, seed=1):
    """per input: AUC of tap rows against rows of fist recordings, and its per-recording constant on the fist rows"""
    from scipy.stats import rankdata
    kind = np.array([LP.rec_kind(f, r) for f, r in zip(files.folder, files.rel)])
    rng = np.random.default_rng(seed)
    fist = np.flatnonzero((rt == 2) & (kind[rec] == "fist"))
    ix = np.r_[np.sort(rng.choice(np.flatnonzero(rt == 1), n, replace=False)), np.sort(rng.choice(fist, n, replace=False))]
    E = np.asarray(z["raw"][ix], np.float32)[:, emg.reshape(-1)].reshape(-1, 3, 3, 5)
    is_tap = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    out = []
    for f in EMG_FAMILIES:
        X, names = emg_family(E, f)
        for j, nm in enumerate(names):
            x = X[:, j].astype(np.float64)
            if not np.isfinite(x).all() or x.std() == 0:
                continue
            r = rankdata(x); a = (r[is_tap].sum() - n * (n + 1) / 2) / n ** 2
            df = pd.DataFrame(dict(v=x[~is_tap], g=rec[ix][~is_tap]))
            out.append(dict(family=f, name=nm, auc_fist=max(a, 1 - a),
                            eta_fist=df.groupby("g").v.transform("mean").var() / max(df.v.var(), 1e-12)))
    return pd.DataFrame(out)


# ============================================================================================ frames (Lean-lite)
def _frames_file(rel):
    p = LP.DATA / rel.replace("DATA_TAP/", "DATA_TAP_LEAN8/", 1)
    p = p if p.exists() else p.with_name(p.name + ".gz")
    if not p.exists():
        return None
    return pd.read_csv(p, usecols=["lean8_frames"]).lean8_frames.to_numpy(np.int32)


def cmd_frames():
    from multiprocessing import Pool
    files, *_ = cache()
    out = np.zeros(int(files.n.sum()), np.int32); ok = np.zeros(len(files), bool)
    with Pool(2) as pool:
        for i, fr in enumerate(pool.imap(_frames_file, files.rel.tolist(), chunksize=16)):
            r = files.iloc[i]
            if fr is not None and len(fr) == r.n:
                out[r.start:r.start + r.n] = fr; ok[i] = True
    np.save(OUT / "frames.npy", out); np.save(OUT / "frames_ok.npy", ok)
    log(f"frames: {ok.mean():.1%} of the recordings")


def lite_under(Rs, te, cols, frames, frames_ok, lite_nominal):
    """Lean-lite of the test rows recomputed from their (changed) WL, recording by recording"""
    import tap_features as TF
    L = np.array(lite_nominal, np.float32, copy=True)
    for _, f in te.iterrows():
        if not frames_ok[f.i]:
            continue
        a = f.sstart
        L[a:a + f.n] = TF.lite_rows(Rs[a:a + f.n], cols["raw"], frames[f.start:f.start + f.n])
    return L


def lite_train_aug(tr, rows, draw, cols, z, emg, wl, frames, frames_ok):
    """Lean-lite of the training rows `rows` after the training augmentation `draw`: every training recording is
    changed as a whole and its Lean-lite recomputed from the changed WL (like lite_under for the test rows)."""
    import tap_features as TF
    full = np.zeros((int((tr.start + tr.n).max()), len(cols["lite"])), np.float32)
    for _, f in tr.iterrows():
        a, b = f.start, f.start + f.n
        if not frames_ok[f.i]:
            full[a:b] = np.asarray(z["lite"][a:b], np.float32); continue
        Rf = perturb(np.asarray(z["raw"][a:b], np.float32), emg, wl, **rows_of(draw, np.full(f.n, f.i)))
        full[a:b] = TF.lite_rows(Rf, cols["raw"], frames[a:b])
    return full[rows]


# ============================================================================================ stress runs
def cmd_stress(*arms):
    import lightgbm as lgb
    files, cols, z, emg, wl = cache()
    files["i"] = np.arange(len(files))
    y_all = np.load(LP.OUT / f"y_{LABEL}.npy")
    target = (y_all > 0) & (np.asarray(z["cls"]) == 1)
    keep = y_all >= 0
    rec = np.repeat(np.arange(len(files)), files.n.to_numpy())
    draws = stress_draws(len(files)); adraw = {"aug": aug_draws(len(files)), "aug2": aug2_draws(len(files))}
    A = [Arm(a, cols, emg, wl) for a in arms]
    need_lite = any(a.uses_lite for a in A)
    frames = np.load(OUT / "frames.npy", mmap_mode="r") if need_lite else None
    frames_ok = np.load(OUT / "frames_ok.npy") if need_lite else None
    acc = {a.spec: {s: None for s in RUN_STRESSES} for a in A}
    gain_share = {a.spec: [] for a in A}
    t0 = time.time()
    ck = OUT / ("ckpt_" + "_".join(a.replace("+", "-").replace("@", "_") for a in arms) + "_" + "".join(map(str, FOLDS))
                + "_" + str(len(RUN_STRESSES)) + ".json")
    done = json.loads(ck.read_text()) if ck.exists() else {"folds": [], "acc": {}, "gain_share": {}}
    if done["folds"]:
        log(f"resuming: folds {done['folds']} already scored ({ck.name})")
        for a in A:
            gain_share[a.spec] = done["gain_share"][a.spec]
            acc[a.spec] = {st: {q: (np.asarray(v) if isinstance(v, list) else v) for q, v in done["acc"][a.spec][st].items()}
                           for st in RUN_STRESSES}
    for k in FOLDS:
        if k in done["folds"]:
            continue
        tr = files[files.fold != k]
        rows = LP._rows(tr, keep)
        R = np.asarray(z["raw"][rows], np.float32); L = np.asarray(z["lite"][rows], np.float32); y = target[rows]
        models = {}
        lite_aug = {}
        for kind in {a.aug for a in A if a.aug and a.uses_lite}:
            lite_aug[kind] = lite_train_aug(tr, rows, adraw[kind], cols, z, emg, wl, frames, frames_ok)
        for a in A:
            X = a(R, lite_aug[a.aug] if (a.aug and a.uses_lite) else L,
                  draw=rows_of(adraw[a.aug], rec[rows]) if a.aug else None)
            params = dict(LP.LEGACY_MODEL_PARAMS, n_jobs=2)
            models[a.spec] = m = lgb.LGBMClassifier(**params).fit(X, y)
            g = m.booster_.feature_importance("gain"); g = g / max(g.sum(), 1e-12)
            is_emg = np.array([("emg" in nm) or nm.startswith(("logp", "chpool")) for nm in a.names])
            is_abs = np.array([nm.startswith("p_core->emg_psd") for nm in a.names])
            is_lite = np.array(["cload_lite" in nm for nm in a.names])
            gain_share[a.spec].append(dict(emg=float(g[is_emg].sum()), emg_abs=float(g[is_abs].sum()), lite=float(g[is_lite].sum())))
            del X
            log(f"fold {k} {a.spec}: trained ({len(a.names)} inputs, {int(y.sum()):,} positives)")
        del R, L
        te = files[files.fold == k].copy()
        te["sstart"] = np.r_[0, np.cumsum(te.n.to_numpy())[:-1]]
        ix = np.concatenate([np.arange(f.start, f.start + f.n) for _, f in te.iterrows()])
        Rt = np.asarray(z["raw"][ix], np.float32); Lt = np.asarray(z["lite"][ix], np.float32); rt_rec = rec[ix]
        for st in RUN_STRESSES:
            Rs = perturb(Rt, emg, wl, **rows_of(draws[st], rt_rec)) if st != "nominal" else Rt
            Ls = lite_under(Rs, te, cols, frames, frames_ok, Lt) if (need_lite and st != "nominal") else Lt
            for a in A:
                score = np.zeros(len(ix), np.float32)
                for c in range(0, len(ix), 250_000):
                    score[c:c + 250_000] = models[a.spec].predict_proba(a(Rs[c:c + 250_000], Ls[c:c + 250_000]))[:, 1]
                ev = LP.evaluate(te, z, score, THS)
                prev = acc[a.spec][st]
                acc[a.spec][st] = ev if prev is None else {q: prev[q] + ev[q] for q in ev}
            del Rs, Ls
            log(f"fold {k} stress {st} done ({time.time() - t0:.0f}s)")
        done["folds"].append(int(k))
        done["gain_share"] = gain_share
        done["acc"] = {a.spec: {st: {q: (v.tolist() if isinstance(v, np.ndarray) else v) for q, v in acc[a.spec][st].items()}
                                for st in RUN_STRESSES} for a in A}
        ck.write_text(json.dumps(done))
    RES.mkdir(parents=True, exist_ok=True)
    for a in A:
        res = dict(arm=a.spec, label=LABEL, n_inputs=len(a.names), seconds=round(time.time() - t0), folds=list(FOLDS),
                   gain_share=gain_share[a.spec],
                   at_085={s: LP.summarize(acc[a.spec][s], THS) for s in RUN_STRESSES},
                   acc={s: {q: (v.tolist() if isinstance(v, np.ndarray) else v) for q, v in acc[a.spec][s].items()} for s in RUN_STRESSES})
        tag = "" if FOLDS == (0, 1, 2, 3, 4) and RUN_STRESSES == STRESSES else "__smoke"
        (RES if not tag else OUT).joinpath(f"stress_{a.spec}{tag}.json").write_text(json.dumps(res))
        s0 = res["at_085"][RUN_STRESSES[0]]; last = RUN_STRESSES[-1]; sp = res["at_085"][last]
        log(f"{a.spec}: {RUN_STRESSES[0]} taps {s0['taps']:.3f} move {s0['fpm_move']:.2f}/min | {last} taps {sp['taps']:.3f} move {sp['fpm_move']:.2f}/min")


# ============================================================================================ tables
def cmd_table():
    """stress_table.csv (every number) and stress_table.md (the report's tables) from the stress runs"""
    rows = []
    for p in sorted(RES.glob("stress_*.json")):
        r = json.loads(p.read_text())
        d = dict(arm=r["arm"], n=r["n_inputs"], emg_gain=np.mean([g["emg"] for g in r["gain_share"]]),
                 abs_gain=np.mean([g["emg_abs"] for g in r["gain_share"]]),
                 lite_gain=np.mean([g.get("lite", 0.0) for g in r["gain_share"]]))
        for s in STRESSES:
            a = r["at_085"][s]
            d[f"taps|{s}"] = a["taps"]; d[f"move|{s}"] = a["fpm_move"]
            for k in LP.NEG_KINDS:
                d[f"{k}|{s}"] = a[f"fpm_{k}"]
        acc0 = {q: (np.asarray(v) if isinstance(v, list) else v) for q, v in r["acc"]["nominal"].items()}
        eq = LP.at_equal_false(acc0, 4.0, THS)
        d["taps_at4"] = np.nan if eq is None else eq["taps"]
        d["fist_at4"] = np.nan if eq is None else eq["fpm_fist"]
        mains = [s for s in STRESSES if s.startswith("mains")]
        d["taps|hum"] = min(d[f"taps|{s}"] for s in mains); d["move|hum"] = max(d[f"move|{s}"] for s in mains)
        st = [s for s in STRESSES if s != "nominal"]
        d["taps|worst"] = min(d[f"taps|{s}"] for s in st); d["move|worst"] = max(d[f"move|{s}"] for s in st)
        rows.append(d)
    T = pd.DataFrame(rows).set_index("arm")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    show = ["n", "emg_gain", "abs_gain", "lite_gain", "taps_at4", "fist_at4"] + [f"{m}|{s}" for s in ("nominal", "gain", "tilt", "noise", "rotation", "placement", "hum") for m in ("taps", "move")]
    print(T[show].round(3).to_string())
    print(T[[f"move|mains{k}" for k in range(5)] + [f"taps|mains{k}" for k in range(5)]].round(3).to_string())
    print(T[[f"{k}|{s}" for k in ("fist", "finger", "wrist", "arm", "band") for s in ("nominal", "placement")]].round(2).to_string())
    T.to_csv(RES / "stress_table.csv")
    f2 = lambda v: f"{v:.2f}"; p1 = lambda v: f"{100 * v:.1f}%"
    has = lambda arms: [a for a in arms if a in T.index]
    fam = has(["acc", "acc+abs", "acc+xdiff", "acc+zdiff", "acc+ratio", "acc+tdiff", "acc+chpool", "acc+tdiff+chpool"])
    pipe = has(["rfe81", "rfe81@aug2", "sel-robust", "sel-robust@aug2", "acc+tz+lite", "acc+tz+lite@aug2",
                "acc+tz+chpool+lite", "acc+tz+chpool+lite@aug2"])
    md = ["Table A, EMG families (ACC block in every row):", "",
          "| Inputs | n | Taps | False/min | Fist/min | Placement: taps | false/min | Contact noise: false/min | "
          "Rotation: taps | false/min | Hum, worst zone: taps | false/min |", "|---|" + "---|" * 11]
    for a in fam:
        d = T.loc[a]
        md.append(f"| {a} | {int(d.n)} | {p1(d['taps|nominal'])} | {f2(d['move|nominal'])} | {f2(d['fist|nominal'])} | "
                  f"{p1(d['taps|placement'])} | {f2(d['move|placement'])} | {f2(d['move|noise'])} | "
                  f"{p1(d['taps|rotation'])} | {f2(d['move|rotation'])} | {p1(d['taps|hum'])} | {f2(d['move|hum'])} |")
    md += ["", "Table B, pipeline options:", "",
           "| Inputs | n | Taps | False/min | Fist/min | At 4 false/min: taps | fist/min | Placement: taps | false/min | "
           "fist/min | Rotation: taps | false/min | Hum, worst zone: taps | false/min |", "|---|" + "---|" * 13]
    for a in pipe:
        d = T.loc[a]
        md.append(f"| {a} | {int(d.n)} | {p1(d['taps|nominal'])} | {f2(d['move|nominal'])} | {f2(d['fist|nominal'])} | "
                  f"{p1(d['taps_at4'])} | {f2(d['fist_at4'])} | {p1(d['taps|placement'])} | {f2(d['move|placement'])} | "
                  f"{f2(d['fist|placement'])} | {p1(d['taps|rotation'])} | {f2(d['move|rotation'])} | {p1(d['taps|hum'])} | "
                  f"{f2(d['move|hum'])} |")
    md += ["", "Table C, hum by zone (taps / false per min):", "",
           "| Inputs | " + " | ".join(f"zone {k}" for k in range(5)) + " |", "|---|" + "---|" * 5]
    for a in has(["rfe81", "sel-robust", "acc+abs", "acc+zdiff", "acc+ratio", "acc+tdiff", "acc+tz+lite@aug2",
                  "acc+tz+chpool+lite@aug2"]):
        d = T.loc[a]
        md.append(f"| {a} | " + " | ".join(f"{p1(d[f'taps|mains{k}'])} / {f2(d[f'move|mains{k}'])}" for k in range(5)) + " |")
    (RES / "stress_table.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd, *args = sys.argv[1:] or ["describe"]
    {"describe": cmd_describe, "frames": cmd_frames, "stress": cmd_stress, "table": cmd_table}[cmd](*args)
