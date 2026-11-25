"""Inputs for lp_flow_research.py that need each recording's full history (the LP cache keeps press windows only):

- lite_replay.npy / lite_train.npy: the 15 Lean-lite diff fields for every replay and training row;
- anchor_replay.npy: press-anchored fields for every replay row (each window starts at a press onset):
  rel_base[c] = log2(WL_c+1) - its mean 400..100 ms before the onset, drop_peak[c] = log2(WL_c+1) - its running
  maximum since the onset, dt = min(time since onset, 600 ms).

    python research/gesture/lp_flow_prepare.py [LP cache dir] [output dir]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tap_generalization_research as GR  # noqa: E402

ROOT = HERE.parent.parent
C = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/v3data/lp/cache")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/v3data/lp/lite")
OUT.mkdir(parents=True, exist_ok=True)
recs = pd.read_csv(C / "recordings.csv"); win = pd.read_csv(C / "lp_windows.csv")
rt = np.load(C / "replay_meta.npz")["t"].astype(np.float64)
m = np.load(C / "train_meta.npz"); mt = m["t"].astype(np.float64); mrec = m["rec"]
LR = np.full((len(rt), 15), np.nan, np.float32); LT = np.full((len(mt), 15), np.nan, np.float32)
A = np.full((len(rt), 7), np.nan, np.float32)
need = sorted(set(win.rec.tolist()) | set(np.unique(mrec).tolist()))
t0 = time.time(); bad = nobase = 0
for i, r in enumerate(need):
    df = pd.read_csv(ROOT / recs.file[r], low_memory=False,
                     usecols=lambda c: c in ("Timestamp", "lean8_frames") or c.startswith("p_core->wl["))
    fr = df["lean8_frames"].to_numpy(np.int64); new = np.r_[fr[0] > 0, np.diff(fr) > 0]
    wl = df[[f"p_core->wl[CH{c}]" for c in (1, 2, 3)]].to_numpy(np.float64)
    feats = GR.lite_frames(np.round(wl[new]).astype(np.int64)); idx = np.cumsum(new) - 1
    lite = np.zeros((len(df), 15), np.float32); ok = idx >= 0; lite[ok] = feats[idx[ok]]
    ts = df.Timestamp.to_numpy(np.int64); t = (ts - ts[0]).astype(np.float64); L = np.log2(wl + 1)
    for w in win[win.rec == r].itertuples():
        j = np.minimum(np.searchsorted(t, rt[w.start:w.end] - 0.01), len(t) - 1)
        if np.abs(t[j] - rt[w.start:w.end]).max() > 0.6:
            bad += 1; continue
        LR[w.start:w.end] = lite[j]
        pre = (t >= w.start_ms - 400) & (t <= w.start_ms - 100)
        base = L[pre].mean(0) if pre.sum() >= 5 else L[j[:3]].mean(0); nobase += int(pre.sum() < 5)
        Lw = L[j]
        A[w.start:w.end, 0:3] = Lw - base
        A[w.start:w.end, 3:6] = Lw - np.maximum.accumulate(Lw, axis=0)
        A[w.start:w.end, 6] = np.minimum(rt[w.start:w.end] - w.start_ms, 600.0)
    sel = np.flatnonzero(mrec == r)
    if len(sel):
        j = np.minimum(np.searchsorted(t, mt[sel] - 0.01), len(t) - 1)
        okm = np.abs(t[j] - mt[sel]) <= 0.6
        LT[sel[okm]] = lite[j[okm]]; bad += int((~okm).any())
    if i % 400 == 0:
        print(i, len(need), round(time.time() - t0), "s", flush=True)
np.save(OUT / "lite_replay.npy", LR); np.save(OUT / "lite_train.npy", LT); np.save(OUT / "anchor_replay.npy", A)
print("done; mismatched recordings", bad, "windows without a pre-press baseline", nobase,
      "rows without lite: replay", int(np.isnan(LR[:, 0]).sum()), "train", int(np.isnan(LT[:, 0]).sum()))
