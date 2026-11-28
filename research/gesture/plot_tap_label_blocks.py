#!/usr/bin/env python3
"""Old (v10) vs new (signal) tap labels on the same real taps, row by row -> figures/tap_f1_label_blocks.png

Needs the proxy cache and label sets of legacy_f1_proxy.py (cache, v10, sig legacy, derive half legacy).
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_labeling"))
import tap_signal as TS  # noqa: E402

LP = Path("/tmp/work/lp")
OLD, NEW, INK, MUTED, GRID = "#f47230", "#257de0", "#191919", "#666666", "#e6e6e6"
EXAMPLES = [
    ("A normal tap", "DATA_TAP/TAP/GonenYemini_RightHand_MidTap_Horizontal_Default_120bpm_2026-07-15_12-14-05.csv",
     543, 2, (-120, 160)),
    ("A double tap the pointer merged into one press", "DATA_TAP/TAP/OfekYehezkel_LeftHand_DoubleTaps_Horizontal_Default_60bpm_2026-07-15_11-59-44.csv",
     461, 1, (-260, 160)),
    ("A soft tap the pointer missed (no press for 2 s)", "DATA_TAP/TAP/yonatan_SoftTap_Reverse_NONE_120bpm_2026-01-29_11-26-52.csv",
     1207, 1, (-120, 160)),
]


def main(out=ROOT / "research" / "gesture" / "figures" / "tap_f1_label_blocks.png"):
    files = pd.read_csv(LP / "files.csv").set_index("rel")
    y_old, y_new, y_half = (np.load(LP / f"y_{n}.npy") for n in ("v10", "legacy", "legacyhalf"))
    fig = plt.figure(figsize=(15, 6.4), dpi=150)
    outer = fig.add_gridspec(1, len(EXAMPLES), wspace=0.12)
    for c, (title, rel, k, axis, (lo, hi)) in enumerate(EXAMPLES):
        r = files.loc[rel]
        sl = slice(int(r.start), int(r.start + r.n))
        s, _ = TS.read(ROOT / "DATA" / (rel + ".gz"))
        ex = TS.prepare(s)["ex"][:, axis]
        t = s["t"] - s["t"][k]
        w = np.flatnonzero((t >= lo) & (t <= hi))
        g = outer[c].subgridspec(5, 1, height_ratios=[4.2, 0.55, 0.55, 0.55, 0.55], hspace=0.12)
        ax = fig.add_subplot(g[0])
        ax.plot(t[w], ex[w], color=INK, lw=1.6, zorder=2)
        ax.scatter(t[w], ex[w], s=9, color=INK, zorder=3, linewidths=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(min(-0.1, ex[w].min() - 0.1), ex[w].max() * 1.12 + 0.1)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8, bottom=False, labelbottom=False)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        if c == 0:
            ax.set_ylabel("ACC power on the tap's axis\n(decades above rest)", fontsize=8.5, color=MUTED)
        n_old, n_new = int((y_old[sl][w] > 0).sum()), int((y_new[sl][w] > 0).sum())
        n_half = int((y_half[sl][w] > 0).sum())
        ax.text(0.99, 0.97, f"rows = 1 in this window\nold: {n_old}    new: {n_new}    new, trained: {n_half}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
        tw = t[w]                                            # rows are irregular (3-20 ms): each box spans
        mid = np.r_[tw[0] - (tw[1] - tw[0]) / 2, (tw[1:] + tw[:-1]) / 2, tw[-1] + (tw[-1] - tw[-2]) / 2]
        left, width = mid[:-1], np.diff(mid)                 # half-way to the previous / next row
        strips = [("pointer pressed", (np.asarray(s["ptr"])[w] > 0).astype(int), "#9c9c9c"),
                  ("old labels (v10)", (y_old[sl][w] > 0).astype(int), OLD),
                  ("new labels", (y_new[sl][w] > 0).astype(int), NEW),
                  ("new, rows trained (TAPROWS=0.5)", y_half[sl][w], NEW)]
        for i, (name, val, col) in enumerate(strips):
            sa = fig.add_subplot(g[i + 1], sharex=ax)
            for x0, wd, vv in zip(left, width, val):
                if vv < 0:                                   # left out of training: an outline only
                    sa.add_patch(plt.Rectangle((x0, 0.12), wd, 0.76, fill=False, ec="#c8c8c8", lw=0.5))
                else:
                    sa.add_patch(plt.Rectangle((x0, 0.12), wd, 0.76, fc=col if vv > 0 else "#f2f2f2",
                                               ec="white", lw=0.4))
            sa.set_ylim(0, 1); sa.set_yticks([])
            for sp in sa.spines.values():
                sp.set_visible(False)
            sa.text(lo, 1.02, name, fontsize=7.5, color=MUTED, va="bottom", ha="left", transform=sa.transData)
            sa.tick_params(colors=MUTED, labelsize=8, bottom=(i == 3), labelbottom=(i == 3), left=False)
            if i == 3:
                sa.set_xlabel("ms from the tap's ACC peak (one box = one row)", fontsize=8.5, color=MUTED)
    fig.suptitle("Old vs new tap labels on the same recordings: a block of rows per tap, the new one starts at the "
                 "peak instead of on the rise", fontsize=11.5, color=INK, x=0.01, ha="left", y=0.995)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
