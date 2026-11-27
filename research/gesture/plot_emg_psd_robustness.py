#!/usr/bin/env python3
"""Figures of EMG_PSD_ROBUST_FEATURES.md from results_emg_psd_robustness/ (emg_psd_robustness.py describe / stress).

figures/emg_psd_inputs_one_at_a_time.png  per input group: per-recording constant, worst recall change, worst false rows
figures/emg_psd_model_stress.png           per model arm: tap recall and false taps/min, nominal and under each stress
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results_emg_psd_robustness"
FIG = HERE / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#257de0", "#f47230", "#00b894", "#f2a600"
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"

GROUPS = [  # (group, label, over time?)
    ("tdiff", "now − mean, now − old (diff of codes)", True),
    ("chpool", "same, pooled over the 3 channels", True),
    ("logp_t", "same pairs, log power with floor 0.1", True),
    ("logp0_t", "same pairs, exact log power", True),
    ("ratio_t", "same pairs, ratio (a+1)/(b+1)", True),
    ("ddiff", "shape change over time (diff of diffs)", True),
    ("zdiff", "zone vs zone (diff of codes)", False),
    ("logp_z", "zone vs zone, log power floor 0.1", False),
    ("logp0_z", "zone vs zone, exact log power", False),
    ("ratio_z", "zone vs zone, ratio", False),
    ("abs", "absolute codes", False),
    ("xdiff", "channel vs channel (diff of codes)", False),
]
STRESS_NAME = {"gain": "gain", "tilt": "tilt", "noise": "noise", "rotation": "rotation", "placement": "placement",
               **{f"mains{k}": f"hum zone {k}" for k in range(5)}}


def _style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def one_at_a_time():
    S = pd.read_csv(RES / "univariate_groups.csv").set_index("group")
    stresses = [s for s in STRESS_NAME]
    y = np.arange(len(GROUPS))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    labels = [g[1] for g in GROUPS]
    cols = [BLUE if g[2] else ORANGE for g in GROUPS]
    eta = [S.loc[g, "eta_m"] * 100 for g, *_ in GROUPS]
    d_tpr, w_tpr, fpr, w_fpr = [], [], [], []
    for g, *_ in GROUPS:
        dt = {s: S.loc[g, f"tpr_{s}"] * 100 for s in stresses}
        df = {s: S.loc[g, f"fpr_{s}"] * 100 for s in stresses}
        s1 = min(dt, key=dt.get); s2 = max(df, key=df.get)
        d_tpr.append(dt[s1]); w_tpr.append(STRESS_NAME[s1]); fpr.append(df[s2]); w_fpr.append(STRESS_NAME[s2])
    ax = axes[0]
    ax.barh(y, eta, color=cols, height=0.62)
    for yy, v in zip(y, eta):
        ax.text(v + 1, yy, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 75); ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_title("Share that is a per-recording constant\n(variance on no-tap recordings)", fontsize=9.5, color=INK, loc="left")
    ax = axes[1]
    ax.barh(y, d_tpr, color=cols, height=0.62)
    for yy, v, w in zip(y, d_tpr, w_tpr):
        ax.text(min(v, 0) - 0.6, yy, f"{v:+.0f} ({w})", va="center", ha="right", fontsize=8, color=INK)
    ax.set_xlim(-30, 2); ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_title("Tap recall change under the worst stress\n(points; one-input rule at 5% false rows)", fontsize=9.5, color=INK, loc="left")
    ax = axes[2]
    ax.barh(y, fpr, color=cols, height=0.62)
    for yy, v, w in zip(y, fpr, w_fpr):
        ax.text(v + 0.4, yy, f"{v:.1f}% ({w})", va="center", fontsize=8, color=INK)
    ax.axvline(5, color=MUTED, lw=0.8, ls="--"); ax.text(5.3, y[-1] - 0.75, "nominal 5%", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 42)
    ax.set_title("False rows under the worst stress\n(no-tap recordings; nominal 5%)", fontsize=9.5, color=INK, loc="left")
    for a in axes:
        _style(a)
    h = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(h, ["change over time, same channel and zone", "spectral shape, level or channel vs channel"],
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.55, -0.01))
    fig.suptitle("EMG PSD inputs one at a time: what a band change does to them (top 10 inputs of each group)",
                 fontsize=11, color=INK, x=0.01, ha="left", y=0.995)
    fig.savefig(FIG / "emg_psd_inputs_one_at_a_time.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


ARMS = [  # (arm, label), top to bottom
    ("rfe81", "make train today (feature elimination, 81 inputs)"),
    ("rfe81@aug2", "  same, trained with placement/hum/rotation aug"),
    ("sel-robust", "make train EMG=robust (82 inputs)"),
    ("sel-robust@aug2", "  same, trained with placement/hum/rotation aug"),
    ("acc", "ACC only (108 inputs, in every arm below)"),
    ("acc+abs", "+ absolute EMG codes"),
    ("acc+xdiff", "+ channel vs channel"),
    ("acc+zdiff", "+ zone vs zone"),
    ("acc+ratio", "+ ratios over time and zone"),
    ("acc+tdiff", "+ diffs over time"),
    ("acc+chpool", "+ channel-pooled change"),
    ("acc+tdiff+chpool", "+ diffs over time + channel-pooled"),
    ("acc+tz+lite", "+ over time, zone vs zone (same channel), Lean-lite"),
    ("acc+tz+lite@aug2", "  same, trained with placement/hum/rotation aug"),
    ("acc+tz+chpool+lite", "+ over time, zone vs zone, channel-pooled, Lean-lite"),
    ("acc+tz+chpool+lite@aug2", "  same, trained with placement/hum/rotation aug"),
]


def model_stress():
    runs = {}
    for a, _ in ARMS:
        p = RES / f"stress_{a}.json"
        if p.exists():
            runs[a] = json.loads(p.read_text())["at_085"]
    arms = [(a, lab) for a, lab in ARMS if a in runs]
    y = np.arange(len(arms))[::-1]
    series = [("nominal", "nominal", INK, "o", True), ("placement", "placement (gain + tilt + noise)", BLUE, "o", False),
              ("rotation", "rotation", ORANGE, "D", False), ("hum", "hum, worst zone", AQUA, "s", False)]

    def val(r, s, key):
        if s == "hum":
            vals = [r[f"mains{k}"][key] for k in range(5)]
            return min(vals) if key == "taps" else max(vals)
        return r[s][key]
    panels = (("taps", "Tap recall at threshold 0.85", 100), ("fpm_move", "False taps / min, all no-tap recordings", 1),
              ("fpm_fist", "False taps / min, fist recordings", 1))
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 0.42 * len(arms) + 1.6), sharey=True,
                             gridspec_kw=dict(wspace=0.06, width_ratios=[1.15, 1, 1]))
    for ax, (key, title, scale) in zip(axes, panels):
        for yy, (a, _) in zip(y, arms):
            v = [val(runs[a], s, key) * scale for s, *_ in series]
            ax.plot([min(v), max(v)], [yy, yy], color=GRID, lw=2.2, zorder=1)
        for s, lab, col, mk, filled in series:
            xs = [val(runs[a], s, key) * scale for a, _ in arms]
            ax.scatter(xs, y, s=34 if mk != "D" else 28, marker=mk, color=col if filled else "white", edgecolor=col,
                       lw=1.6, zorder=3, label=lab)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        _style(ax)
    axes[0].set_yticks(y); axes[0].set_yticklabels([lab for _, lab in arms], fontsize=9, color=INK)
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.55, -0.04))
    fig.suptitle("Model test: the same LightGBM on different EMG inputs, file-level folds, every recording scored once",
                 fontsize=11, color=INK, x=0.01, ha="left", y=1.0)
    fig.savefig(FIG / "emg_psd_model_stress.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    one_at_a_time()
    if any(RES.glob("stress_*.json")):
        model_stress()
