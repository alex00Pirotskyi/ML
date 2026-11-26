"""The feature ranker's fast circular-shift scoring (feature_selection.FastTapShift) gives exactly the scores of the
original path (shift the columns, re-predict the whole model, replay every recording, restore), draws the same
random shifts and leaves the matrix unchanged."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "gesture"))


def _inner_replay(n_rec=40, seed=0):
    import feature_selection as FS
    import replay as R
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(seed)
    X, y, seg, t, lab, items = [], [], [], [], [], []
    pos = 0
    kinds = ["tap:mid", "tap:soft", "move:finger", "move:fist"]
    for r in range(n_rec):
        n = int(rng.integers(300, 900))
        x = rng.normal(size=(n, 12)).astype(np.float32)
        x[rng.random(x.shape) < 0.01] = np.nan
        tt = np.cumsum(rng.integers(18, 23, n)).astype(np.float64)
        lb = np.zeros(n, np.int8)
        kind = kinds[r % 4]
        ons = []
        if kind.startswith("tap"):
            for o in range(40, n - 20, 60):
                lb[o:o + 4] = 1; lb[o - 3:o] = -1; ons.append(tt[o])
                x[o:o + 4, :3] += 2.5
        else:
            x[rng.random(n) < 0.05, :3] += 2.0                       # tap-like movement
        X.append(x); y.append((lb == 1).astype(int)); seg.append((pos, pos + n)); t.append(tt); lab.append(lb)
        items.append((pos, pos + n, kind.split(":")[1].upper(), np.asarray(ons), {}, kind))
        pos += n
    X = np.ascontiguousarray(np.vstack(X)); y = np.concatenate(y)
    model = lgb.LGBMClassifier(**dict(FS.SCREEN_PARAMS, n_estimators=40, n_jobs=1, random_state=1)).fit(X, y)
    ev = FS.InnerReplay.__new__(FS.InnerReplay)
    ev.R, ev.task, ev.target = R, "tap", 2.0
    ev.X, ev.seg, ev.t, ev.label, ev.items, ev.steady = X, seg, t, lab, items, None
    return FS, ev, model


def test_fast_shift_scores_match_the_original_path():
    FS, ev, model = _inner_replay()
    p0 = ev.probabilities(model)
    thr, s0, _ = ev.operating_point(p0)
    fast = FS.FastTapShift.build(ev, model, thr, s0, p0, log=lambda *a: None)
    assert fast is not None, "self-checks failed"
    X0 = ev.X.copy()
    shifts = [[c] for c in range(ev.X.shape[1])] + [[0, 1], [0, 1, 2], [3, 4, 5, 6], [0, 11]] * 2
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    paths = set(); moved = 0
    for cols in shifts:
        slow = FS.shifted_score(ev, None, model, thr, cols, rng_a)
        before = dict(fast.stats)
        quick = FS.shifted_score(ev, fast, model, thr, cols, rng_b)
        paths |= {k for k in fast.stats if fast.stats[k] != before.get(k, 0)}
        assert quick == slow or (np.isnan(quick) and np.isnan(slow)), (cols, quick, slow)
        moved += quick != s0
        assert np.array_equal(ev.X, X0, equal_nan=True)                  # restored, bit for bit
    assert rng_a.integers(1 << 30) == rng_b.integers(1 << 30)          # the same random draws were consumed
    assert {"sub_model", "full_model"} <= paths, paths                 # both fast paths were exercised
    assert moved >= len(shifts) // 2                                   # and the shifts did change the scores


def test_fast_shift_full_model_fallback_matches(monkeypatch):
    FS, ev, model = _inner_replay(seed=3)
    monkeypatch.setattr(FS, "FAST_MAX_ROW_FRACTION", 0.0)               # every shift with rows in reach: full model
    p0 = ev.probabilities(model)
    thr, s0, _ = ev.operating_point(p0)
    fast = FS.FastTapShift.build(ev, model, thr, s0, p0, log=lambda *a: None)
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    for cols in ([0], [1, 2], [5]):
        assert FS.shifted_score(ev, fast, model, thr, cols, rng_b) == FS.shifted_score(ev, None, model, thr, cols, rng_a)
    assert fast.stats["full_model"] > 0
