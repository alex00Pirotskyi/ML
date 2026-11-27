"""Inputs of the TAP F1 trainer (train_tap_f1.py), by name, and how to compute them from the firmware's raw fields.

Every input is either a raw p_core field of the recording CSVs, a Lean-lite field, or a firmware helper call on two
raw fields, named exactly as the exporter (export_plus.py) writes it into the C header:

    p_core->diff(a, b)   = a - b, saturated to int16
    p_core->ratio(a, b)  = (a + 1) / (b + 1), float

Candidates (`candidates`, the input of the feature elimination, select_tap_f1_features.py), like the legacy pool:
    the raw fields                                   117 (126 minus the 9 WL fields)
    ACC PSD with ACC PSD: diff and ratio             every pair of acc_psd[0], acc_psd[last], acc_psd_mean
                                                     (54 fields, 1431 pairs, any axis / zone): 2862
    EMG PSD with EMG PSD: diff and ratio             every pair of emg_psd[0], emg_psd[last], emg_psd_mean
                                                     (45 fields, 990 pairs): 1980
    ACC rate with ACC rate: diff only                acc_rate can be negative, so no ratio (153)
    Lean-lite                                        15 (p_core->cload_lite.*)
    = 5127. A pair is used once (diff(b, a) = -diff(a, b), ratio(b, a) = 1 / ratio(a, b): the same split for a tree),
    in the legacy model's orientation where it had one. Ratios only over PSD codes (never negative, so b + 1 >= 1).

Pools (the trainer ranks the pool with f_classif and picks the best n_features of it, as before):
    rfe     the candidates the feature elimination kept (default; DATA_TAP_F1/_build/selected_features.json)
    leg     the raw fields + the 64 diff / ratio inputs of the legacy model (its 95 inputs are all in here)
    newacc  leg + 36 ACC ratios: every (axis, zone) of the newest ACC PSD over its running mean and over its
            oldest history slot (the legacy model's strongest family, made complete)
    new     newacc + 144 EMG inputs: within-channel PSD zone vs zone and slot vs slot diffs (+ WL ratios with WL on)
    lite    new + 15 Lean-lite inputs

WL: no input uses the absolute WL level (the 9 raw fields p_core->wl / wl_head / wl_tail and the WL ratios): tested on
the device, the models whose top inputs were WL fired worse, and the legacy model used none. Lean-lite stays: it is
WL on a log scale relative to its own recent minimum, maximum and mean (rise10, fall10, range10, dev3, xm), so a
different strap position or skin contact (a gain on WL) cancels out. It needs firmware/lean_lite.h on the device and
the Lean-8 frame counter of DATA_TAP_LEAN8.
"""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research" / "gesture"))
import tap_features as TF  # noqa: E402  (EMG diff / WL ratio names, Lean-lite)

LEGACY_MODEL = ROOT / "research" / "gesture" / "legacy_tap_f1_features.json"
POOLS = ("rfe", "leg", "newacc", "new", "lite")
DEFAULT_POOL = "rfe"
DEFAULT_WL = False         # no absolute WL input: on the device, models that leaned on WL did worse (2026-11-27)
LITE_COLUMNS = list(TF.LITE_COLUMNS)

# raw field families; a pair is only formed inside one modality, ratios only over PSD codes (>= 0)
_FAMILIES = (
    ("acc_psd_now", r"^p_core->acc_psd\[0\]\["),
    ("acc_psd_old", r"^p_core->acc_psd\[ACC_PSD_HISTORY_SIZE - 1\]\["),
    ("acc_psd_mean", r"^p_core->acc_psd_mean\["),
    ("acc_rate", r"^p_core->acc_rate\["),
    ("emg_psd_now", r"^p_core->emg_psd\[0\]\["),
    ("emg_psd_old", r"^p_core->emg_psd\[EMG_PSD_HISTORY_SIZE - 1\]\["),
    ("emg_psd_mean", r"^p_core->emg_psd_mean\["),
)
_PAIR_GROUPS = ((("acc_psd_now", "acc_psd_old", "acc_psd_mean"), ("diff", "ratio")),
                (("emg_psd_now", "emg_psd_old", "emg_psd_mean"), ("diff", "ratio")),
                (("acc_rate",), ("diff",)))


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
    """an input on the absolute EMG waveform length: p_core->wl / wl_head / wl_tail, or a helper call on one of them
    (Lean-lite, p_core->cload_lite.*, is relative and not counted)"""
    return "->wl[" in name or "->wl_head[" in name or "->wl_tail[" in name


def families(raw_columns):
    """{family: [raw column, ...]} in the recording's column order"""
    out = {f: [] for f, _ in _FAMILIES}
    for c in raw_columns:
        for f, rx in _FAMILIES:
            if re.match(rx, c):
                out[f].append(c)
                break
    return out


def candidates(raw_columns, wl=DEFAULT_WL, lite=True):
    """every input the feature elimination may choose from (see the module text), in a fixed order"""
    raw = [c for c in raw_columns if c.startswith("p_core->") and (wl or not is_wl(c))]
    fam = families(raw_columns)
    legacy = set(legacy_model_inputs())
    out = list(raw)
    for group, ops in _PAIR_GROUPS:
        fields = [c for f in group for c in fam[f]]
        for a, b in combinations(fields, 2):
            for op in ops:
                fwd, rev = f"p_core->{op}({a}, {b})", f"p_core->{op}({b}, {a})"
                out.append(rev if (rev in legacy and fwd not in legacy) else fwd)
    if lite:
        out += LITE_COLUMNS
    return list(dict.fromkeys(out))


def pool(name, raw_columns, wl=DEFAULT_WL, selected=None):
    """the input names of a pool, in a fixed order, given the recording's raw p_core columns. No absolute-WL input
    unless wl=True. `rfe` is the list the feature elimination kept (selected)."""
    if name not in POOLS:
        raise ValueError(f"unknown input pool {name!r}; one of {POOLS}")
    if name == "rfe":
        if selected is None:
            raise ValueError("the rfe pool needs the selected inputs (select_tap_f1_features.py)")
        bad = [s for s in selected if is_wl(s) and not wl]
        if bad:
            raise ValueError(f"selected inputs use WL with WL off: {bad[:3]}")
        return list(selected)
    raw = [c for c in raw_columns if c.startswith("p_core->") and (wl or not is_wl(c))]
    dedupe = lambda xs: [x for x in dict.fromkeys(xs) if wl or not is_wl(x)]
    leg = dedupe(raw + [f for f in legacy_model_inputs() if f not in raw])
    if name == "leg":
        return leg
    newacc = dedupe(leg + acc_ratio_names())
    if name == "newacc":
        return newacc
    new = dedupe(newacc + TF.DERIVED)
    return new if name == "new" else dedupe(new + LITE_COLUMNS)


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
