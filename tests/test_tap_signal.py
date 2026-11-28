"""Signal tap labels (claude_labeling/tap_signal.py): per-axis / per-zone conditions, the repaired pointer and the
no-unknown labels, on synthetic recordings with known taps."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_labeling"))
import tap_signal as TS  # noqa: E402


def jolt(t, t0, height, width_ms=25.0, zones=(1.0, 1.0, 1.0, 1.0, 1.0)):
    """linear ACC power of a contact jolt at t0: broadband over zones 1-5 (an impact), all three axes"""
    g = np.exp(-0.5 * ((t - t0) / width_ms) ** 2)
    return height * g[:, None] * np.asarray(zones)[None, :]


def recording(taps, presses, dur_ms=9000.0, dt=8.0, beat_ms=1000.0, seed=0, release=True):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, dur_ms, dt)
    n = len(t)
    acc = np.zeros((n, 3, 6)) + rng.uniform(2, 6, (n, 3, 6))          # rest power ~ a few codes
    for t0, h in taps:
        for a, scale in enumerate((1.0, 0.5, 0.3)):
            acc[:, a, 1:] += jolt(t, t0, h * scale)
    if release:                                                        # a weaker, slower release after each press
        for d, u in presses:
            for a in range(3):
                acc[:, a, 1:] += jolt(t, u + 30.0, 40.0, 60.0, zones=(1.0, 0.3, 0.1, 0.1, 1.0))
    ptr = np.zeros(n, np.int8)
    for d, u in presses:
        ptr[(t >= d) & (t <= u)] = 1
    met = np.zeros(n, np.int8)
    for b in np.arange(500.0, dur_ms, beat_ms):
        met[np.argmin(np.abs(t - b))] = 1
    emg = np.full((n, 3, 5), 500.0)
    return dict(ts=(t + 1e12).astype(np.int64), t=t, acc=acc, emg=emg, wl=np.zeros((n, 3)), ptr=ptr, met=met, n=n)


def taps_of(J):
    return np.sort(J.t[J.y == 1].to_numpy(float))


def test_single_taps_top_block_and_release_is_not_a_tap():
    tt = np.arange(1000.0, 8001.0, 1000.0)
    s = recording([(x, 5000.0) for x in tt], [(x - 20, x + 110) for x in tt])
    for preset in ("legacy", "hybrid"):
        y, J, q = TS.label(s, "TAP", "p_MidTap_Horizontal_Default_60bpm_x.csv.gz", preset)
        got = taps_of(J)
        assert len(got) == len(tt) and np.all(np.abs(got - tt) <= 10)
        for x in tt:                                                   # taught from <= 10 ms before the peak, never the start
            rows = np.flatnonzero((y == 1) & (np.abs(s["t"] - x) < 200))
            assert s["t"][rows].min() >= x - 10 - 8 - 1e-9 and s["t"][rows].max() <= x + 50 + 8 + 1e-9   # peak +-1 row
        assert q["tier"] == "clean" and q["n_contact"] == len(tt)
        rel = (s["t"] > tt[2] + 120) & (s["t"] < tt[2] + 260)          # the release jolt of a tap
        if preset == "legacy":                                         # no unknown rows; the release is "not a tap"
            assert set(np.unique(y)) <= {0, 1} and (y[rel] == 0).all()
        else:                                                          # hybrid: around the tap the rows stay out
            assert (y[rel] == -1).all() and (y[(s["t"] > tt[2] + 400) & (s["t"] < tt[2] + 600)] == 0).all()


def test_merged_double_tap_is_split():
    firsts = np.arange(1000.0, 8001.0, 1500.0)
    taps = [(x, 5000.0) for x in firsts] + [(x + 230, 4000.0) for x in firsts]
    s = recording(taps, [(x - 20, x + 330) for x in firsts], beat_ms=1500.0)          # one press per pair
    y, J, q = TS.label(s, "TAP", "p_DoubleTaps_Vertical_Default_40bpm_x.csv.gz")
    got = taps_of(J)
    want = np.sort([x for x, _ in taps])
    assert len(got) == len(want) and np.all(np.abs(got - want) <= 10)
    assert q["n_second"] == len(firsts)


def test_late_pointer_and_missed_press():
    tt = np.arange(1000.0, 8001.0, 1000.0)
    presses = [(x + 50, x + 170) for x in tt]                          # the pointer 70 ms late ...
    del presses[3]                                                      # ... and one tap not seen at all
    s = recording([(x, 5000.0) for x in tt], presses)
    y, J, q = TS.label(s, "TAP", "p_MidTap_Horizontal_Default_60bpm_x.csv.gz")
    got = taps_of(J)
    assert len(got) == len(tt) and np.all(np.abs(got - tt) <= 10)
    assert q["n_recovered"] == 1


def test_negative_folder_is_all_not_a_tap():
    tt = np.arange(1000.0, 8001.0, 1000.0)
    s = recording([(x, 5000.0) for x in tt], [])
    y, J, q = TS.label(s, "FINGER", "p_Fist&Stretch_Horizontal_Default_60bpm_x.csv.gz")
    assert (y == 0).all() and q["tier"] == "negative"


def test_recording_without_jolts_is_left_out():
    tt = np.arange(1000.0, 8001.0, 1000.0)
    s = recording([], [(x, x + 120) for x in tt], release=False)
    y, J, q = TS.label(s, "TAP", "p_MidTap_Horizontal_Default_60bpm_x.csv.gz")
    assert q["tier"] == "broken" and (y == -1).all()


def test_signal_conditions_are_per_axis():
    """the gate reads one axis' own zones: a jolt whose zones 3-4 are missing (movement) is not a crisp contact"""
    tt = np.arange(1000.0, 8001.0, 1000.0)
    s = recording([(x, 5000.0) for x in tt], [(x - 20, x + 110) for x in tt], release=False)
    for x in tt:                                                       # remove the impact band from every axis
        m = np.abs(s["t"] - x) < 80
        s["acc"][m, :, 3:5] = 3.0
    y, J, q = TS.label(s, "TAP", "p_MidTap_Horizontal_Default_60bpm_x.csv.gz")
    assert q["n_contact"] == 0 and q["n_weak"] == len(tt)
