"""make train FIST=k (MUDRA_TAP_FIST_WEIGHT): fist negatives weigh k times as much as other negatives; finger-motion
negatives, positives and the teachers' plain weights are untouched."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "gesture"))


def test_fist_negatives_weighted(monkeypatch):
    import train_unified as U
    files = ["a_Fist&Stretch_1.csv.gz", "b_Pinky_1.csv.gz", "c_MidTap_1.csv.gz", "d_Fist&Stretch_2.csv.gz"]
    cats = ["FINGER", "FINGER", "TAP", "FINGER"]
    n = 50
    data = dict(y=np.r_[np.zeros(n), np.zeros(n), np.ones(n // 2), np.zeros(n // 2), np.zeros(n)].astype(int),
                w=np.ones(4 * n, np.float32), file=np.repeat(np.array(files, object), n),
                category=np.repeat(np.array(cats, object), n))
    idx = np.arange(4 * n)
    monkeypatch.setattr(U, "TAP_FIST_WEIGHT", 3.0)
    w = U.weights_for_task(data, idx, "tap", 1.0)
    fist_a, pinky, fist_d = w[:n], w[n:2 * n], w[3 * n:]
    assert np.allclose(fist_a / pinky, 3.0) and np.allclose(fist_d / pinky, 3.0)
    assert data["is_fist"].sum() == 2 * n
    plain = U.weights_for_task(data, idx, "tap", 1.0, finger=False)              # teachers: no fist weighting
    assert np.allclose(plain[:n] / plain[n:2 * n], 1.0)
