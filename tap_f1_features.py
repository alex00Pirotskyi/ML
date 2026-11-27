"""Inputs of the TAP F1 trainer (train_tap_f1.py), by name, and how to compute them from the firmware's raw fields.

Every input is either a raw p_core field of the recording CSVs, or a firmware helper call on two raw fields, named
exactly as the exporter (export_plus.py) writes it into the C header:

    p_core->diff(a, b)   = a - b, saturated to int16
    p_core->ratio(a, b)  = (a + 1) / (b + 1), float

Pools (the trainer ranks the pool with f_classif and picks the best n_features of it, as before):
    leg     the raw fields + the 64 diff / ratio inputs of the legacy model (its 95 inputs are all in here)
    newacc  leg + 36 ACC ratios: every (axis, zone) of the newest ACC PSD over its running mean and over its
            oldest history slot (the legacy model's strongest family, made complete)
    new     newacc + 144 EMG inputs: within-channel PSD zone vs zone and slot vs slot diffs, WL head / tail ratios
    lite    new + 15 Lean-lite diff inputs (p_core->cload_lite.*; need the Lean-8 frame counter of DATA_TAP_LEAN8
            and firmware/lean_lite.h on the device; built from WL, so only with WL on)

WL: by default no pool has a WL input (the 9 raw fields p_core->wl / wl_head / wl_tail and the 9 WL ratios of
`new`): tested on the device, the models whose top inputs were WL fired worse, and the legacy model used none.
126 raw fields - 9 WL = 117: leg 181 inputs, newacc 212, new 346 (with WL: 190 / 221 / 364, lite 379).
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research" / "gesture"))
import tap_features as TF  # noqa: E402  (EMG diff / WL ratio names, Lean-lite)

LEGACY_MODEL = ROOT / "research" / "gesture" / "legacy_tap_f1_features.json"
POOLS = ("leg", "newacc", "new", "lite")
DEFAULT_POOL = "newacc"
DEFAULT_WL = False         # WL inputs off: on the device, models that leaned on WL did worse (2026-11-27)


def legacy_model_inputs():
    """The 95 inputs of the legacy TAP model (31 raw fields + 64 diff / ratio)."""
    return list(json.loads(LEGACY_MODEL.read_text())["selected_feature_names"])


def acc_ratio_names():
    out = []
    for a in "XYZ":
        for z in range(6):
            cur = f"p_core->acc_psd[0][{a}][{z}]"
            out.append(f"p_core->ratio({cur}, p_core->acc_psd_mean[{a}][{z}])")
            out.append(f"p_core->ratio({cur}, p_core->acc_psd[ACC_PSD_HISTORY_SIZE - 1][{a}][{z}])")
    return out


def is_wl(name):
    """an EMG waveform-length input: p_core->wl / wl_head / wl_tail, or a helper call on one of them"""
    return "->wl[" in name or "->wl_head[" in name or "->wl_tail[" in name


def pool(name, raw_columns, wl=DEFAULT_WL):
    """the input names of a pool, in a fixed order, given the recording's raw p_core columns. wl=False (default)
    leaves out every WL input (the 9 raw WL fields and the WL ratios); Lean-lite is built from WL, so `lite` needs
    wl=True."""
    if name not in POOLS:
        raise ValueError(f"unknown input pool {name!r}; one of {POOLS}")
    if name == "lite" and not wl:
        raise ValueError("the lite inputs are computed from WL; use them with WL inputs on (--wl 1 / make train WL=1)")
    raw = [c for c in raw_columns if c.startswith("p_core->") and (wl or not is_wl(c))]
    dedupe = lambda xs: [x for x in dict.fromkeys(xs) if wl or not is_wl(x)]
    leg = dedupe(raw + [f for f in legacy_model_inputs() if f not in raw])
    if name == "leg":
        return leg
    newacc = dedupe(leg + acc_ratio_names())
    if name == "newacc":
        return newacc
    new = dedupe(newacc + TF.DERIVED)
    return new if name == "new" else dedupe(new + list(TF.LITE_COLUMNS))


def needs_lite(names):
    return any(n in TF.LITE_COLUMNS for n in names)


_CALL = re.compile(r"^p_core->(diff|ratio)\((p_core->.+?), (p_core->.+)\)$")


class FeatureMaker:
    """rows of base fields (raw, + Lean-lite) -> the named inputs (float32), with the firmware's helpers"""

    def __init__(self, base_cols, names):
        ci = {c: i for i, c in enumerate(base_cols)}
        self.names = list(names)
        self.direct, self.diff, self.ratio = [], [], []
        for j, n in enumerate(self.names):
            if n in ci:
                self.direct.append((j, ci[n])); continue
            m = _CALL.match(n)
            if not m or m.group(2) not in ci or m.group(3) not in ci:
                raise KeyError(f"cannot compute input {n!r} from the recording's fields")
            (self.diff if m.group(1) == "diff" else self.ratio).append((j, ci[m.group(2)], ci[m.group(3)]))
        self.float_columns = sorted(j for j, _, _ in self.ratio)

    def __call__(self, B):
        B = np.asarray(B, np.float32)
        X = np.empty((len(B), len(self.names)), np.float32)
        if self.direct:
            j, c = map(np.array, zip(*self.direct)); X[:, j] = B[:, c]
        if self.diff:
            j, a, b = map(np.array, zip(*self.diff)); X[:, j] = np.clip(B[:, a] - B[:, b], -32768, 32767)
        if self.ratio:
            j, a, b = map(np.array, zip(*self.ratio)); X[:, j] = (B[:, a] + np.float32(1)) / (B[:, b] + np.float32(1))
        return X


def lite_rows(raw, raw_cols, frames):
    """(rows, 15) Lean-lite diff inputs of one recording (from its first row), as the firmware holds them"""
    return TF.lite_rows(np.asarray(raw, np.float32), list(raw_cols), np.asarray(frames)).astype(np.float32)
