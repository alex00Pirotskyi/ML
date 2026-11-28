#!/usr/bin/env python3
"""Signal research for the tap labeller: what does a tap look like per ACC axis, per ACC zone and per EMG channel,
compared with its own release and with fists, fingers, wrist, arm and band movements?

    python research/gesture/tap_signal_research.py extract [OUT]     # every jolt of every TAP-task recording
    python research/gesture/tap_signal_research.py tables  [OUT]     # the comparison tables (markdown)

extract writes OUT/jolts.csv.gz (one row per jolt, claude_labeling/tap_signal.jolts plus pointer / metronome
context), OUT/presses.csv.gz and OUT/files.csv.gz. Byte-identical copies of a recording are read once.
"""
from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_labeling"))
import tap_signal as TS        # noqa: E402

TAP_DIRS = ["TAP", "SWIPE", "FINGER", "WRIST", "ARM", "BAND", "NOTHING"]
LP_DIRS = ["SOFT_LONG_PRESS", "MID_LONG_PRESS", "STRONG_LONG_PRESS", "LONG_PRESS", "DRAG_AND_DROP"]
ORIENT = {"Horizontal", "Vertical", "Reverse"}


def recordings():
    out = []
    for d in TAP_DIRS:
        out += sorted((ROOT / "DATA" / "DATA_TAP" / d).glob("*.csv.gz"))
    for d in LP_DIRS:
        out += sorted((ROOT / "DATA" / "DATA_LONG_PRESS" / d).glob("*.csv.gz"))
    return out


def meta(path: Path):
    name = path.name.replace(".csv.gz", "")
    tok = name.split("_")
    oi = next((i for i, x in enumerate(tok) if x in ORIENT), None)
    kind = tok[oi - 1] if oi else ""
    m = re.search(r"_(\d+)bpm", name)
    return dict(file=f"{path.parent.parent.name}/{path.parent.name}/{path.name}", folder=path.parent.name,
                kind=kind, orient=tok[oi] if oi else "", bpm=int(m.group(1)) if m else -1,
                fist=("fist" in name.lower()))


def nearest(x, ref):
    """signed distance from each x to the nearest value of ref (nan if ref is empty), and that ref's index"""
    if not len(ref):
        return np.full(len(x), np.nan), np.full(len(x), -1)
    i = np.clip(np.searchsorted(ref, x), 1, len(ref) - 1) if len(ref) > 1 else np.zeros(len(x), int)
    if len(ref) > 1:
        left = ref[i - 1]; right = ref[i]
        j = np.where(np.abs(x - left) <= np.abs(x - right), i - 1, i)
    else:
        j = np.zeros(len(x), int)
    return x - ref[j], j


def one(path):
    path = Path(path)
    try:
        s, sha = TS.read(path)
        t = s["t"]
        up, dn, starts_down = TS.presses(t, s["ptr"])
        bt = TS.beats(t, s["met"])
        p = TS.prepare(s, guard_edges=np.r_[up, dn])
        J = TS.jolts(s, p)
        md = meta(path)
        if len(J):
            J["dt_down"], J["press"] = nearest(J.t.to_numpy(), up)
            J["dt_up"], _ = nearest(J.t.to_numpy(), dn)
            J["in_press"] = [bool(((up <= x) & (dn >= x)).any()) for x in J.t]
            J["dt_beat"], _ = nearest(J.t.to_numpy(), bt)
            J.insert(0, "file", md["file"])
        P = pd.DataFrame(dict(file=md["file"], down=up, up=dn))
        F = dict(md, sha=sha, rows=s["n"], dur_s=float(t[-1] / 1000) if len(t) else 0.0, presses=len(up),
                 starts_down=starts_down, beats=len(bt), beat_ms=float(np.median(np.diff(bt))) if len(bt) > 2 else np.nan,
                 quiet=float(p["quiet"].mean()), **{f"bS{a}": float(p["bS"][i]) for i, a in enumerate(TS.AXES)},
                 **{f"sS{a}": float(p["sS"][i]) for i, a in enumerate(TS.AXES)}, err="")
        return J, P, F
    except Exception as e:                                  # noqa: BLE001 - reported per file
        return None, None, dict(meta(path), err=repr(e)[:300])


def extract(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    files = recordings()
    t0 = time.time()
    Js, Ps, Fs = [], [], []
    with ProcessPoolExecutor(int(os.environ.get("SIG_WORKERS", os.cpu_count() or 2))) as ex:
        for i, (J, P, F) in enumerate(ex.map(one, map(str, files), chunksize=4), 1):
            Fs.append(F)
            if J is not None and len(J):
                Js.append(J)
            if P is not None and len(P):
                Ps.append(P)
            if i % 250 == 0:
                print(f"{i}/{len(files)} files, {time.time() - t0:.0f} s", flush=True)
    F = pd.DataFrame(Fs)
    dup = F.sha.duplicated() & F.sha.notna()
    F["duplicate"] = dup
    keep = set(F.file[~dup])
    J = pd.concat(Js, ignore_index=True); J = J[J.file.isin(keep)]
    P = pd.concat(Ps, ignore_index=True); P = P[P.file.isin(keep)]
    J.to_csv(out / "jolts.csv.gz", index=False); P.to_csv(out / "presses.csv.gz", index=False)
    F.to_csv(out / "files.csv.gz", index=False)
    print(f"{len(F)} files ({dup.sum()} duplicates, {(F.err != '').sum()} errors), {len(J)} jolts, "
          f"{len(P)} presses, {time.time() - t0:.0f} s")


# ================================================================================================ pointer repair test
def _set(ptr, t, a, b, v):
    ptr[TS.window(t, a, b)] = v


def corrupt(s, how, rng):
    """a copy of the recording's signals with its pointer (or metronome) broken in one known way"""
    s = dict(s); t = s["t"]; ptr = s["ptr"].copy(); met = s["met"].copy()
    down, up, _ = TS.presses(t, ptr)
    if how == "drop":                                     # the pointer misses 15% of the taps
        for i in rng.choice(len(down), max(1, int(0.15 * len(down))), replace=False):
            _set(ptr, t, down[i], up[i], 0)
    elif how == "merge":                                  # double taps: the pair becomes one long press
        for i in range(len(down) - 1):
            if down[i + 1] - down[i] < 500:
                _set(ptr, t, up[i], down[i + 1], 1)
    elif how == "shift":                                  # the whole pointer 70 ms late
        idx = np.clip(np.searchsorted(t, t - 70.0), 0, len(t) - 1); ptr = s["ptr"][idx]
    elif how == "bounce":                                 # every press split by a 20 ms gap
        for a, b in zip(down, up):
            if b - a > 60:
                m = (a + b) / 2; _set(ptr, t, m, m + 20, 0)
    elif how == "stuck":                                  # 10% of the releases come 800 ms late
        for i in rng.choice(len(down), max(1, int(0.10 * len(down))), replace=False):
            _set(ptr, t, up[i], up[i] + 800, 1)
    elif how == "startdown":                              # the pointer is down from the first row to the first release
        if len(up):
            _set(ptr, t, t[0], up[0], 1)
    elif how == "beatshift":                              # control: a metronome half a beat off (chance alignments)
        bt = TS.beats(t, met)
        if len(bt) > 2:
            met[:] = 0; P = np.median(np.diff(bt))
            for b in bt + P / 2:
                met[TS.window(t, b, b + 5)] = 1
    s["ptr"] = ptr; s["met"] = met
    return s


def _taps(J):
    return np.sort(J.t[J.y == 1].to_numpy(float)) if len(J) else np.zeros(0)


def _match(ref, got, tol=25.0):
    used = np.zeros(len(got), bool); hit = 0
    for x in ref:
        m = np.flatnonzero(~used & (np.abs(got - x) <= tol))
        if len(m):
            used[m[np.argmin(np.abs(got[m] - x))]] = True; hit += 1
    return hit, int((~used).sum())


def corrupt_one(args):
    path, hows, seed = args
    path = Path(path); s, _ = TS.read(path); rng = np.random.default_rng(seed)
    y0, J0, q0 = TS.label(s, path.parent.name, path.name)
    ref = _taps(J0); out = []
    for how in hows:
        y1, J1, q1 = TS.label(corrupt(s, how, rng), path.parent.name, path.name)
        got = _taps(J1); hit, extra = _match(ref, got)
        out.append(dict(file=path.name, how=how, ref=len(ref), got=len(got), hit=hit, extra=extra,
                        recovered=q1.get("n_recovered", 0), second=q1.get("n_second", 0), tier=q1.get("tier")))
    return out


def corrupt_test(lab: Path, out: Path, n_files=240, seed=7):
    """break clean pointer tracks in known ways and count how many taps the repair gets back (and adds)"""
    Q = pd.read_csv(lab / "quality.csv.gz")
    Q = Q[(Q.folder == "TAP") & (Q.tier == "clean") & (Q.agreement >= 0.9) & (Q.n_recovered == 0)]
    rng = np.random.default_rng(seed)
    single = Q[~Q.double & ~Q.fast]; dbl = Q[Q.double]
    pick = lambda d, k: d.sample(min(k, len(d)), random_state=seed).file.tolist()
    jobs = [(str(ROOT / "DATA" / f), ["drop", "shift", "bounce", "stuck", "startdown", "beatshift"], int(rng.integers(1e9)))
            for f in pick(single, n_files)]
    jobs += [(str(ROOT / "DATA" / f), ["merge", "drop", "shift", "beatshift"], int(rng.integers(1e9))) for f in pick(dbl, n_files // 3)]
    rows = []
    with ProcessPoolExecutor(int(os.environ.get("SIG_WORKERS", os.cpu_count() or 2))) as ex:
        for r in ex.map(corrupt_one, jobs, chunksize=2):
            rows += r
    R = pd.DataFrame(rows); R["double"] = R.file.str.lower().str.contains("doubletap")
    R.to_csv(out / "corrupt.csv", index=False)
    g = R.groupby(["double", "how"])
    T = pd.DataFrame({"files": g.size(), "taps": g.ref.sum(), "kept": (g.hit.sum() / g.ref.sum()).round(4),
                      "added_per_100": (100 * g.extra.sum() / g.ref.sum()).round(2)})
    print(T.to_string())
    return T


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "extract"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/work/sig")
    if cmd == "extract":
        extract(out)
    elif cmd == "corrupt":
        corrupt_test(Path(os.environ.get("SIG_LABELS", "/tmp/work/siglab")), out)
    else:
        raise SystemExit(__doc__)


# ================================================================================================ review page
ROLE_STYLE = {"contact": ("#1b9e77", "o"), "second": ("#7570b3", "s"), "recovered": ("#e6ab02", "D"),
              "weak": ("#999999", "x"), "release": ("#d95f02", "v"), "inpress": ("#e7298a", "^")}


def plot_file(path, out_png, t_range=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path); s, _ = TS.read(path)
    y, J, q = TS.label(s, path.parent.name, path.name)
    p = TS.prepare(s); t = s["t"] / 1000.0
    lo, hi = (t_range or (t[0], min(t[-1], t[0] + 8.0)))
    m = (t >= lo) & (t <= hi)
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    for a, c in zip(range(3), ("#1f77b4", "#2ca02c", "#9467bd")):
        ax[0].plot(t[m], p["ex"][m, a], lw=0.9, color=c, label=f"ACC {TS.AXES[a]} (zones 1-5)")
    ax[0].fill_between(t[m], -0.3, -0.05, where=s["ptr"][m] > 0, step="post", color="#bbbbbb", label="pointer down")
    ax[0].fill_between(t[m], 0, p["ex"][m].max() + 0.2, where=y[m] == 1, step="post", color="#1b9e77", alpha=0.25,
                       label="label = tap")
    for r, (c, mk) in ROLE_STYLE.items():
        e = J[(J.role == r) & (J.t / 1000 >= lo) & (J.t / 1000 <= hi)]
        if len(e):
            ax[0].scatter(e.t / 1000, e.h + 0.12, color=c, marker=mk, s=28, zorder=5, label=r)
    bt = TS.beats(s["t"], s["met"]) / 1000
    for b in bt[(bt >= lo) & (bt <= hi)]:
        ax[0].axvline(b, color="#cccccc", lw=0.6, ls=":")
    ax[0].set_ylabel("decades above rest"); ax[0].legend(loc="upper right", fontsize=7, ncol=4)
    ech = p["exE"].mean(2)
    for c, col in zip(range(3), ("#8c564b", "#e377c2", "#7f7f7f")):
        ax[1].plot(t[m], ech[m, c], lw=0.8, color=col, label=f"EMG CH{c + 1}")
    ax[1].set_ylabel("dB above rest"); ax[1].set_xlabel("s"); ax[1].legend(loc="upper right", fontsize=7, ncol=3)
    ax[0].set_title(f"{path.parent.name}/{path.name}  |  tier {q.get('tier')}, presses {q.get('presses')}, taps "
                    f"{q.get('taps')} (2nd {q.get('n_second', 0)}, recovered {q.get('n_recovered', 0)})", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=90); plt.close(fig)


def review(lab: Path, out: Path, n_each=3, seed=3):
    """out/index.html: label summary, pointer-repair test, and example recordings (clean, merged double taps split,
    recovered taps, weak / left-out recordings)"""
    import base64
    out.mkdir(parents=True, exist_ok=True)
    Q = pd.read_csv(lab / "quality.csv.gz"); Q = Q[Q.duplicate_of.isna() | (Q.duplicate_of == "")]
    T = Q[Q.folder.isin(["TAP", "SWIPE"])]
    picks = [("Clean single taps", T[(T.tier == "clean") & ~T.double & ~T.fast]),
             ("Merged double taps split into two", T[T.n_second > 0].sort_values("n_second", ascending=False).head(40)),
             ("Taps the pointer missed, recovered on the beat", T[T.n_recovered > 0].sort_values("n_recovered", ascending=False).head(40)),
             ("Weak recordings (few presses with a clear contact jolt)", T[T.tier == "weak"]),
             ("Left out (broken)", T[T.tier == "broken"])]
    parts = []
    for title, d in picks:
        if not len(d):
            continue
        parts.append(f"<h2>{title}</h2>")
        for f in d.sample(min(n_each, len(d)), random_state=seed).file:
            png = out / (Path(f).name.replace(".csv.gz", ".png"))
            plot_file(ROOT / "DATA" / f, png)
            b64 = base64.b64encode(png.read_bytes()).decode()
            parts.append(f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>')
    (out / "examples.html").write_text("\n".join(parts))
    return out / "examples.html"
