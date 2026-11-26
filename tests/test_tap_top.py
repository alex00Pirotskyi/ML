"""Top labels (claude_labeling/tap_top.py): every tap is taught at the top of its contact jolt, never at its start."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_labeling"))
import tap_top as TT  # noqa: E402

DT = 5.5


def recording(peaks_ms, n_ms=4000, height=6.0, rise_ms=50.0, fall_ms=90.0, axis=0, seed=0, dt=DT):
    """Synthetic acc_psd[0] rows: rest noise plus one jolt per peak on one axis (zones 1-5), log-energy shaped as a
    linear rise and fall around each peak."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, n_ms, dt)
    log_e = np.full((len(t), 3), 1.0) + rng.normal(0, 0.05, (len(t), 3))
    for p in peaks_ms:
        up = (t >= p - rise_ms) & (t <= p)
        dn = (t > p) & (t <= p + fall_ms)
        log_e[up, axis] = np.maximum(log_e[up, axis], 1.0 + height * (t[up] - (p - rise_ms)) / rise_ms)
        log_e[dn, axis] = np.maximum(log_e[dn, axis], 1.0 + height * (1 - (t[dn] - p) / fall_ms))
    energy = np.expm1(log_e)                                # sum over zones 1-5 per axis
    acc = np.zeros((len(t), 3, 6))
    acc[:, :, 1:] = (energy / 5.0)[:, :, None]
    acc[:, :, 0] = 3.0                                     # zone 0 (motion / gravity) is not part of the jolt
    return t, acc.reshape(len(t), 18)


def onset_labels(t, onsets, lo=0.0, hi=80.0, mask=150.0):
    y = np.zeros(len(t), np.int8)
    for o in onsets:
        y[(t >= o - mask) & (t <= o + hi + mask)] = -1
    for o in onsets:
        y[(t >= o + lo) & (t <= o + hi)] = 1
    return y


def test_window_sits_on_the_top_never_on_the_start():
    peaks = [500.0, 1500.0, 2500.0]
    t, acc = recording(peaks)
    onsets = [p - 40.0 for p in peaks]                    # the labeller's onset is ~40 ms before the peak
    y = onset_labels(t, onsets)
    for mode in TT.MODES:
        y2, _, wins, info = TT.relabel(t, acc, y, mode, anchors=onsets)
        assert info["taps"] == 3 and info["jolt"] == 3 and info.get("fallback", 0) == 0
        for p, w in zip(peaks, wins):
            assert abs(w["t_top"] - p) <= DT + 1e-6
            assert w["start_ms"] <= p <= w["end_ms"]          # the peak is taught
            # never the start: the window begins at or after the 50 % (mid) / 80 % (top) crossing of the rise
            frac = 0.5 if mode == "mid" else 0.8
            assert w["start_ms"] >= p - (1 - frac) * 50.0 - DT - 1e-6
            assert w["end_ms"] <= p + TT.CLAMP_AFTER_MS + 1e-6
            if mode == "top":                                  # the default: the peak, not the rise before it
                assert w["start_ms"] >= p - TT.CLAMP_BEFORE_MS["top"] - DT - 1e-6
            pos = t[(y2 == 1) & (t > p - 200) & (t < p + 200)]
            assert pos.min() >= w["start_ms"] - 1e-6 and pos.max() <= w["end_ms"] + 1e-6
        # the rise below 50 % (the start) is never a positive
        for p in peaks:
            start = (t >= p - 50.0) & (t < p - 25.0 - DT)
            assert not (y2[start] == 1).any()


def test_mid_starts_earlier_than_top_and_rows_go_to_unknown_not_negative():
    t, acc = recording([800.0])
    y = onset_labels(t, [760.0])
    y_top, _, w_top, _ = TT.relabel(t, acc, y, "top", anchors=[760.0])
    y_mid, _, w_mid, _ = TT.relabel(t, acc, y, "mid", anchors=[760.0])
    assert w_mid[0]["start_ms"] < w_top[0]["start_ms"]
    moved = (y == 1) & (y_top != 1)
    assert moved.any() and (y_top[moved] == -1).all()     # left out, never "not a tap"
    assert ((y == 0) == (y_top == 0)).all()               # negatives unchanged
    assert ((y == 0) == (y_mid == 0)).all()


def test_labeller_zero_rows_stay_zero():
    t, acc = recording([800.0])
    y = onset_labels(t, [760.0])
    zero = (t >= 790.0) & (t <= 800.0)                     # e.g. the disconnection rule set these rows to 0
    y[zero] = 0
    y2, _, _, _ = TT.relabel(t, acc, y, "top", anchors=[760.0])
    assert (y2[zero] == 0).all()


def test_double_tap_keeps_two_windows():
    peaks = [1000.0, 1180.0]                               # second tap 180 ms later
    t, acc = recording(peaks, rise_ms=40.0, fall_ms=70.0)
    onsets = [p - 35.0 for p in peaks]
    y = onset_labels(t, onsets)
    y2, _, wins, info = TT.relabel(t, acc, y, "top", anchors=onsets)
    assert info["taps"] == 2 and info["jolt"] == 2
    assert wins[0]["end_ms"] < wins[1]["start_ms"]
    for p, w in zip(peaks, wins):
        assert abs(w["t_top"] - p) <= DT + 1e-6


def test_no_jolt_falls_back_without_the_start():
    t, acc = recording([], seed=3)                         # rest only: no jolt anywhere
    y = onset_labels(t, [1000.0])
    y2, _, wins, info = TT.relabel(t, acc, y, "top", anchors=[1000.0])
    assert info["fallback"] == 1 and wins[0]["fallback"]
    pos = t[y2 == 1]
    assert pos.min() >= 1000.0 + TT.FALLBACK_SKIP_MS - 1e-6 and pos.max() <= 1080.0 + 1e-6


def test_row_gap_does_not_split_a_tap():
    t, acc = recording([1000.0])
    keep = ~((t > 985.0) & (t < 1030.0))                   # a 45 ms hole in the logged rows inside the tap
    t, acc = t[keep], acc[keep]
    y = onset_labels(t, [960.0])
    y2, _, wins, info = TT.relabel(t, acc, y, "top", anchors=[960.0])
    assert info["taps"] == 1
    y3, _, wins3, info3 = TT.relabel(t, acc, y, "top")      # without anchors: runs are joined across the hole
    assert info3["taps"] == 1


def test_anchors_without_positive_rows_are_not_taps():
    t, acc = recording([1000.0, 2000.0])
    y = onset_labels(t, [960.0])                           # only the first press is a labelled tap
    y2, _, wins, info = TT.relabel(t, acc, y, "top", anchors=[960.0, 1960.0])
    assert info["taps"] == 1 and not (y2[(t > 1800) & (t < 2200)] == 1).any()


def test_soft_mask_follows_its_tap():
    t, acc = recording([1000.0, 2000.0])
    y = onset_labels(t, [960.0, 1960.0])
    soft = (y == 1) & (t > 1500)
    y2, soft2, _, _ = TT.relabel(t, acc, y, "top", anchors=[960.0, 1960.0], soft=soft)
    assert soft2[(y2 == 1) & (t > 1500)].all() and not soft2[(y2 == 1) & (t < 1500)].any()


def test_any_axis_and_noisy_rest():
    for axis in range(3):
        t, acc = recording([1200.0], axis=axis, seed=axis)
        acc = acc + np.random.default_rng(9).uniform(0, 2, acc.shape)
        y = onset_labels(t, [1160.0])
        _, _, wins, info = TT.relabel(t, acc, y, "top", anchors=[1160.0])
        assert info["jolt"] == 1 and wins[0]["axis"] == axis and abs(wins[0]["t_top"] - 1200.0) <= 2 * DT


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
