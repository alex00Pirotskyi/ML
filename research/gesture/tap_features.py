"""TAP generalization features and placement augmentation (TAP_GENERALIZATION_RESULTS.md).

Derived inputs, computed from the 126 raw firmware fields (no new data files, nothing new in the firmware):

* 135 within-channel EMG PSD diffs, firmware helper ``diff(a, b) = a - b`` saturated to int16:
  zone vs zone inside each time slot (psd[0], psd[EMG_PSD_HISTORY_SIZE - 1], psd_mean) and slot vs slot per zone.
* 9 WL ratios, firmware helper ``ratio(a, b) = (a + 1) / (b + 1)`` in float: wl_head/wl_tail, wl/wl_head, wl/wl_tail.

They are named ``p_core->diff(<a>, <b>)`` / ``p_core->ratio(<a>, <b>)``; the exporter writes those calls into the
header (cached once per prediction) and the replay harness provides the same helpers.

Placement augmentation (training rows only): per recording and EMG channel, a random amplitude gain, a random
per-zone spectral tilt and added broadband noise power, applied exactly to the PSD codes (10000*log10(1+power))
and WL (~ sqrt(power)); the derived fields are recomputed afterwards. ACC is never changed.

Lean-lite diff inputs (MUDRA_TAP_LITE=diff, TAP_FOCUS_RESULTS.md): rise10, fall10, range10, dev3, xm per channel, the
integer Lean-lite fields of firmware/lean_lite.h that are differences of log WL levels. They depend on the WL
history of the recording, so they are computed per recording from its first row (lite_rows), and under a
placement change they are recomputed from the changed WL of the whole recording (lite_rows_placement), exactly
as the research did; apply_placement leaves them alone.
"""
from __future__ import annotations

import numpy as np

import generate_lean_lite_features as lite

SLOTS = ("p_core->emg_psd[0]", "p_core->emg_psd[EMG_PSD_HISTORY_SIZE - 1]", "p_core->emg_psd_mean")
WL_PER_SQRT_POWER = 850.0          # WL ~ 850 * sqrt(total EMG PSD power) in the training data (log-log r = 0.93)


def _emg(slot, ch, z):
    return f"{SLOTS[slot]}[CH{ch}][{z}]"


DIFF_PAIRS = ([(_emg(s, c, a), _emg(s, c, b)) for c in (1, 2, 3) for s in range(3) for a in range(5) for b in range(a + 1, 5)] +
              [(_emg(sa, c, z), _emg(sb, c, z)) for c in (1, 2, 3) for z in range(5) for sa, sb in ((0, 1), (0, 2), (1, 2))])
RATIO_PAIRS = [(f"p_core->{a}[CH{c}]", f"p_core->{b}[CH{c}]") for c in (1, 2, 3)
               for a, b in (("wl_head", "wl_tail"), ("wl", "wl_head"), ("wl", "wl_tail"))]
DIFF_NAMES = [f"p_core->diff({a}, {b})" for a, b in DIFF_PAIRS]
RATIO_NAMES = [f"p_core->ratio({a}, {b})" for a, b in RATIO_PAIRS]
DERIVED = DIFF_NAMES + RATIO_NAMES
assert len(DIFF_PAIRS) == 135 and len(RATIO_PAIRS) == 9 and len(set(DERIVED)) == 144


def derived(X, columns):
    """(rows, 144) float32 derived fields from rows X whose columns are `columns` (must contain the raw fields)."""
    ci = {c: i for i, c in enumerate(columns)}
    a = np.array([ci[x] for x, _ in DIFF_PAIRS]); b = np.array([ci[y] for _, y in DIFF_PAIRS])
    ra = np.array([ci[x] for x, _ in RATIO_PAIRS]); rb = np.array([ci[y] for _, y in RATIO_PAIRS])
    X = np.asarray(X, np.float32)
    d = np.clip(X[:, a] - X[:, b], -32768, 32767)
    r = (X[:, ra] + np.float32(1.0)) / (X[:, rb] + np.float32(1.0))
    return np.hstack([d, r]).astype(np.float32)


def draws(n_recs, seed, gain_oct=1.0, tilt_oct=0.25, noise_log10=(-1.5, -0.5)):
    """Per-recording placement draws. Defaults = the research's 'mild' augmentation."""
    rng = np.random.default_rng(seed)
    return dict(gain=np.power(2.0, rng.uniform(-gain_oct, gain_oct, (n_recs, 3))),
                tilt=np.power(2.0, rng.uniform(-tilt_oct, tilt_oct, (n_recs, 3, 5))),
                noise=np.power(10.0, rng.uniform(noise_log10[0], noise_log10[1], (n_recs, 3))))


def apply_placement(X, columns, gain, tilt, noise):
    """Rows X (columns `columns`) after a contact / placement change: per row gain (n,3), tilt (n,3,5), noise (n,3).
    Raw EMG fields are changed exactly; derived fields present in `columns` are recomputed; nothing else changes."""
    ci = {c: i for i, c in enumerate(columns)}
    Y = np.array(X, np.float32, copy=True)
    G = np.asarray(gain, np.float64); T = np.asarray(tilt, np.float64); N = np.asarray(noise, np.float64)
    for c in range(3):
        p0 = p1 = 0.0
        for s in range(3):
            for z in range(5):
                j = ci[_emg(s, c + 1, z)]
                lin = np.power(10.0, X[:, j].astype(np.float64) / 1e4) - 1.0
                amp2 = (G[:, c] * T[:, c, z]) ** 2
                Y[:, j] = np.clip(np.round(1e4 * np.log10(1.0 + amp2 * lin + N[:, c] / 5.0)), 0, 65535)
                if s == 0:
                    p0 = p0 + lin; p1 = p1 + amp2 * lin
        tau = np.sqrt(np.where(p0 > 1e-9, p1 / np.maximum(p0, 1e-9), G[:, c] ** 2))
        for k in ("wl", "wl_head", "wl_tail"):
            j = ci[f"p_core->{k}[CH{c + 1}]"]
            w = X[:, j].astype(np.float64)
            Y[:, j] = np.round(np.sqrt((w * tau) ** 2 + WL_PER_SQRT_POWER ** 2 * N[:, c]))
    have = [n for n in DERIVED if n in ci]
    if have:
        D = derived(Y, columns)
        for n in have:
            Y[:, ci[n]] = D[:, DERIVED.index(n)]
    return Y


def augment(X, columns, rec, n_recs, seed):
    """Mild placement augmentation of training rows (rec: recording index of each row)."""
    d = draws(n_recs, seed)
    out = np.empty_like(np.asarray(X, np.float32))
    for s in range(0, len(X), 200_000):
        r = rec[s:s + 200_000]
        out[s:s + 200_000] = apply_placement(X[s:s + 200_000], columns, d["gain"][r], d["tilt"][r], d["noise"][r])
    return out


def stress_draws(n_recs, seed=77):
    """The research's evaluation stress ('placement'): gain +-1 octave, tilt +-0.5 octave, noise 0.1..1.0."""
    return draws(n_recs, seed, gain_oct=1.0, tilt_oct=0.5, noise_log10=(-1.0, 0.0))


LITE_COLUMNS = list(lite.DIFF_COLUMNS)
WL_COLUMNS = [f"p_core->wl[CH{c}]" for c in (1, 2, 3)]
PLACEMENT_COLUMNS = ([_emg(s, c, z) for c in (1, 2, 3) for s in range(3) for z in range(5)] +
                     [f"p_core->{k}[CH{c}]" for c in (1, 2, 3) for k in ("wl", "wl_head", "wl_tail")])


def lite_rows(X, columns, frames):
    """(rows, 15) float32 Lean-lite diff values of ONE recording given from its first row (X: its rows, columns
    `columns`; frames: the derived trees' frame counter of those rows)."""
    ci = {c: i for i, c in enumerate(columns)}
    wl = np.asarray(X)[:, [ci[c] for c in WL_COLUMNS]]
    return lite.lite_diff_rows(np.round(wl).astype(np.int64), frames).astype(np.float32)


def lite_rows_placement(X, columns, frames, gain, tilt, noise):
    """lite_rows of ONE recording after a placement change: its draw gain (3,), tilt (3, 5), noise (3,) is applied
    to the EMG fields of its frame rows (apply_placement) and the WL history is recomputed from there."""
    ci = {c: i for i, c in enumerate(columns)}
    rows = lite.frame_rows_from_counter(frames)
    sub = np.asarray(X)[rows][:, [ci[c] for c in PLACEMENT_COLUMNS]]
    k = len(rows)
    Y = apply_placement(sub, PLACEMENT_COLUMNS, np.repeat(np.asarray(gain)[None], k, 0),
                        np.repeat(np.asarray(tilt)[None], k, 0), np.repeat(np.asarray(noise)[None], k, 0))
    wl_frames = np.round(Y[:, [PLACEMENT_COLUMNS.index(c) for c in WL_COLUMNS]]).astype(np.int64)
    per = lite.lite_diff_per_frame(wl_frames)
    f = np.asarray(frames, np.int64); out = np.zeros((len(f), len(LITE_COLUMNS)), np.float32); has = f > 0
    out[has] = per[f[has] - 1]
    return out


def _version():
    """Hash of the code and constants that decide the derived values and the augmentation."""
    import hashlib
    import inspect
    h = hashlib.sha256()
    for fn in (derived, draws, apply_placement, augment, stress_draws):
        h.update(inspect.getsource(fn).encode())
    h.update(repr((DIFF_PAIRS, RATIO_PAIRS, WL_PER_SQRT_POWER)).encode())
    return h.hexdigest()


VERSION = _version()


def _lite_version():
    import hashlib
    import inspect
    h = hashlib.sha256(lite.DIFF_SHA.encode())
    for fn in (lite_rows, lite_rows_placement, apply_placement, draws):
        h.update(inspect.getsource(fn).encode())
    h.update(repr((LITE_COLUMNS, WL_COLUMNS, PLACEMENT_COLUMNS)).encode())
    return h.hexdigest()


LITE_VERSION = _lite_version()
