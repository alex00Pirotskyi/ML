#!/usr/bin/env python3
"""What another band placement, skin contact, power-grid hum or a rotated band does to the EMG inputs.

Used two ways by the TAP F1 pipeline (research/gesture/EMG_PSD_ROBUST_FEATURES.md):

  draw(name)          the training augmentation ("aug2"): every recording of the TRAINING folds gets one random draw,
                      seeded by its file name (the same draw on every build); the held-out fold is never changed.
  stress(name, kind)  the back test's band changes (backtest_tap_f1.py), stronger, one kind at a time.

apply() changes the raw rows exactly as the electrodes would: the EMG PSD codes are E = round(10000*log10(1 + p)),
so every slot (now, oldest history entry, history mean) goes back to zone power p, is changed, and is re-encoded;
WL follows (WL ~ 850 * sqrt(power)); Lean-lite is then recomputed from the changed WL by the caller (it is a state
over the recording). ACC is not touched.

  gain    amplitude x g per channel (contact impedance)            -> power x g^2
  tilt    amplitude x t per channel and zone (tissue, other muscle) -> power x t^2
  add     power added per channel and zone, the same in every slot (white contact noise, a power-grid line)
  mix     each channel takes this share of its neighbour's power (the band rotated)
"""
from __future__ import annotations

import hashlib

import numpy as np

VERSION = "aug2-2026-11-28"
BINS = np.array([8, 10, 12, 46, 51], np.float64)       # FFT bins per EMG PSD zone (bins 1-8, 9-18, 19-30, 31-76, 77-127)
SLOTS = ("p_core->emg_psd[0]", "p_core->emg_psd[EMG_PSD_HISTORY_SIZE - 1]", "p_core->emg_psd_mean")
WL_KINDS = ("wl", "wl_head", "wl_tail")
WL_PER_SQRT_POWER = 850.0
STRESS_KINDS = ("placement", "hum0", "hum1", "hum2", "hum3", "hum4", "rotation")


def _rng(name, salt):
    h = hashlib.sha256(f"{salt}|{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _white(noise):
    return np.asarray(noise, np.float64)[:, None] * (BINS / BINS.sum())[None, :]


def draw(name):
    """The training augmentation of one recording (research arm 'aug2'): gain 2^U(-1, 1) per channel, tilt
    2^U(-0.25, 0.25) per channel and zone, white contact noise of power 10^U(-1.5, -0.5) per channel; on a random
    half of the recordings a power-grid line in one random zone (power 10^U(-2, -0.5) x U(0.5, 1.5) per channel),
    on another random half a band rotation (mix U(0, 0.4))."""
    r = _rng(name, "train")
    d = dict(gain=2.0 ** r.uniform(-1, 1, 3), tilt=2.0 ** r.uniform(-0.25, 0.25, (3, 5)),
             add=_white(10.0 ** r.uniform(-1.5, -0.5, 3)), mix=0.0)
    if r.random() < 0.5:
        d["add"][:, int(r.integers(0, 5))] += 10.0 ** r.uniform(-2, -0.5) * r.uniform(0.5, 1.5, 3)
    if r.random() < 0.5:
        d["mix"] = float(r.uniform(0, 0.4))
    return d


def stress(name, kind):
    """One back-test band change of one recording (the research's stress test, sized from the spread between our
    recordings): placement = gain 2^U(-1, 1) + tilt 2^U(-0.5, 0.5) + white noise 10^U(-1, 0); humK = one line in
    zone K, power 10^U(-1.5, 0) x U(0.5, 1.5) per channel; rotation = mix U(0.2, 0.5)."""
    r = _rng(name, f"stress|{kind}")
    if kind == "placement":
        return dict(gain=2.0 ** r.uniform(-1, 1, 3), tilt=2.0 ** r.uniform(-0.5, 0.5, (3, 5)),
                    add=_white(10.0 ** r.uniform(-1, 0, 3)), mix=0.0)
    if kind.startswith("hum"):
        a = np.zeros((3, 5)); a[:, int(kind[3:])] = 10.0 ** r.uniform(-1.5, 0) * r.uniform(0.5, 1.5, 3)
        return dict(gain=np.ones(3), tilt=np.ones((3, 5)), add=a, mix=0.0)
    if kind == "rotation":
        return dict(gain=np.ones(3), tilt=np.ones((3, 5)), add=np.zeros((3, 5)), mix=float(r.uniform(0.2, 0.5)))
    raise KeyError(kind)


def indices(cols):
    """(EMG code columns (slot, channel, zone), WL columns (channel, kind)) of a column list"""
    ci = {c: i for i, c in enumerate(cols)}
    try:
        emg = np.array([[[ci[f"{SLOTS[s]}[CH{c}][{k}]"] for k in range(5)] for c in (1, 2, 3)] for s in range(3)])
        wl = np.array([[ci[f"p_core->{w}[CH{c}]"] for w in WL_KINDS] for c in (1, 2, 3)])
    except KeyError as e:
        raise KeyError(f"augmentation needs every EMG PSD code and WL field, missing {e}") from None
    return emg, wl


def apply(B, cols, d, idx=None):
    """A float32 copy of the raw rows B (rows x cols) after the change d (draw() / stress())"""
    emg, wl = idx if idx is not None else indices(cols)
    n = len(B)
    Y = np.array(B, np.float32, copy=True)
    P = np.power(10.0, np.asarray(B[:, emg.reshape(-1)], np.float64) / 1e4).reshape(n, 3, 3, 5) - 1.0
    W = np.asarray(B[:, wl.reshape(-1)], np.float64).reshape(n, 3, 3)
    a = float(d.get("mix", 0.0))
    if a:
        P = (1 - a) * P + a * np.roll(P, -1, axis=2)
        W = np.sqrt((1 - a) * W ** 2 + a * np.roll(W, -1, axis=1) ** 2)
    G2 = (np.asarray(d["gain"], np.float64)[:, None] * np.asarray(d["tilt"], np.float64)) ** 2      # (3, 5)
    p0 = P[:, 0].sum(-1)
    P = P * G2[None, None]
    p1 = P[:, 0].sum(-1)
    tau = np.sqrt(np.where(p0 > 1e-9, p1 / np.maximum(p0, 1e-9), G2.mean(-1)[None, :]))
    W = W * tau[:, :, None]
    A = np.asarray(d["add"], np.float64)
    if A.any():
        P = P + A[None, None]
        W = np.sqrt(W ** 2 + WL_PER_SQRT_POWER ** 2 * A.sum(-1)[None, :, None])
    Y[:, emg.reshape(-1)] = np.clip(np.round(1e4 * np.log10(1.0 + np.maximum(P, 0.0))), 0, 65535).reshape(n, -1)
    Y[:, wl.reshape(-1)] = np.round(W).reshape(n, -1)
    return Y
