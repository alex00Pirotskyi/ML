"""Look at the top labels: where every tap is taught, next to the labeller's onset window and the pointer.

    python claude_labeling/plot_tap_top.py out.png                      # one random recording per tap kind
    python claude_labeling/plot_tap_top.py out.png TAP/<name>.csv.gz ... # these recordings (paths under DATA/DATA_TAP)

Reads DATA/DATA_TAP and the labeller's DATA/LABELS/DATA_TAP (made by make train). Per recording: the contact-jolt
envelope of each ACC axis (what tap_top.py measures), the EMG level, the labeller's onset window, the top window
(green: what make train teaches), and the pointer.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tap_top as TT  # noqa: E402

ROOT = os.path.dirname(HERE)
KINDS = ("TAP/.*MidTap", "TAP/.*SoftTap", "TAP/.*StrongTap", "TAP/.*DoubleTaps", "TAP/.*240bpm", "SWIPE/")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = sys.argv[1]; wanted = sys.argv[2:]
    lab_root = os.path.join(ROOT, "DATA", "LABELS", "DATA_TAP")
    ev = pd.read_csv(os.path.join(lab_root, "_events.csv.gz"), low_memory=False)
    conf = ev[(ev.type == "tap_press") & (ev.labeled == 1)]
    recs = sorted(conf.recording.unique())
    rng = np.random.default_rng()
    if not wanted:
        wanted = [rng.choice([r for r in recs if re.search(k, r)]) for k in KINDS if any(re.search(k, r) for r in recs)]
    fig, axes = plt.subplots(len(wanted), 1, figsize=(15, 3.1 * len(wanted)), squeeze=False)
    for ax, rel in zip(axes[:, 0], wanted):
        lab = pd.read_csv(os.path.join(lab_root, re.sub(r"\.csv\.gz$", ".labels.csv.gz", rel)))
        wl = [f"p_core->wl[CH{c}]" for c in (1, 2, 3)]
        df = pd.read_csv(os.path.join(ROOT, "DATA", "DATA_TAP", rel), usecols=["Timestamp", "PointerState", *TT.ACC_NOW, *wl])
        t = (df.Timestamp - df.Timestamp.iloc[0]).to_numpy(float); y = lab.tap_label.to_numpy()
        on = conf[conf.recording == rel].t_on.to_numpy(float)
        acc = df[list(TT.ACC_NOW)].to_numpy()
        y_top, _, wins, _ = TT.relabel(t, acc, y, "top", anchors=on)
        env = TT.jolt_envelope(acc)
        o = on[min(2, len(on) - 1)] if len(on) else t[len(t) // 2]
        sel = (t >= o - 700) & (t <= o + 1800); tt = t[sel]
        for c, col in zip(range(3), "rgb"):
            ax.plot(tt, env[sel, c], col, lw=1, label=f"ACC {'XYZ'[c]} jolt (log)")
        e = np.log2(df[wl].to_numpy().mean(1) + 1)[sel]
        ax.plot(tt, e - np.percentile(e, 5), "k:", lw=1, label="EMG WL (log2)")
        ax.fill_between(tt, -1.6, -0.6, where=y[sel] == 1, color="orange", step="mid", label="labeller onset window")
        ax.fill_between(tt, -2.8, -1.8, where=y_top[sel] == 1, color="green", step="mid", label="TOP (taught)")
        ax.fill_between(tt, -4.0, -3.0, where=df.PointerState.to_numpy()[sel] > 0, color="gray", step="mid", label="pointer")
        for w in wins:
            if tt[0] <= w["t_top"] <= tt[-1] and not w["fallback"]:
                ax.axvline(w["t_top"], color="green", lw=0.5, ls="--")
        ax.set_title(rel, fontsize=9); ax.set_xlim(tt[0], tt[-1]); ax.legend(fontsize=6, ncol=7, loc="upper right")
    plt.tight_layout(); plt.savefig(out, dpi=80)
    print(out)


if __name__ == "__main__":
    main()
