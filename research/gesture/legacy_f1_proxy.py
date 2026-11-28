#!/usr/bin/env python3
"""Which labels and which inputs for the legacy F1 trainer (train_tap_f1.py)? A fast, label-independent check.

The real trainer (Optuna, F1 at 0.85, 100+100 trials) needs a big machine. This proxy keeps what decides how strict
the model is and trains ONE LightGBM per variant with the legacy model's own hyper-parameters (the user's model:
122 trees x 28 leaves, depth 17, scale_pos_weight 7.78, lr 0.091, ...), unweighted rows like the trainer, the top-95
f_classif inputs like that model, on the file-level gesture-aware folds of DATA/fold_roadmap_files.json (test = fold 0,
train = folds 1-4, every STRIDE-th row so it fits in this machine's memory; base rates are unchanged).

What is measured does not depend on any label set (so the label sets can be compared):
  * taps: every pointer press of the test files, per kind (normal / soft / double 1st / double 2nd / 240 bpm /
    slow pinch / swipe); hit = an event -100..+250 ms around the press (replay.py's rule);
  * false taps per minute of every no-tap test recording, per kind (finger, fist, wrist, arm, band, nothing);
  events = rising edges of score >= threshold with the 150 ms refractory, as the firmware fires.

    python research/gesture/legacy_f1_proxy.py cache                # DATA_TAP raw fields + Lean-lite, once
    python research/gesture/legacy_f1_proxy.py v10                  # the old deterministic labeller's state
    python research/gesture/legacy_f1_proxy.py sig legacy <labels.npz>   # a claude_labeling/label_signal.py output
    python research/gesture/legacy_f1_proxy.py rescue               # the old labeller on a repaired pointer
    python research/gesture/legacy_f1_proxy.py derive half legacy   # more label sets: see cmd_derive
    python research/gesture/legacy_f1_proxy.py run v10 leg - 01234  # train + evaluate: label set, input pool,
                                                                    #   scale_pos_weight (- = 7.78, auto), test folds
    python research/gesture/legacy_f1_proxy.py table [4.83]         # at 0.85 [and at that movement false rate]

Env: LEGACY_PROXY_SEED (LightGBM seed), LEGACY_PROXY_TAPFRAC (share of TAP / SWIPE training recordings),
RESCUE_INSERT / RESCUE_SPLIT / RESCUE_NAME (the rescue command), LEGACY_PROXY_DIR, LEGACY_PROXY_STRIDE.
The numbers of research/gesture/TAP_F1_LEGACY.md: cache; v10; sig legacy|hybrid <label_signal.py output>; rescue;
RESCUE_INSERT=0 RESCUE_NAME=v10split rescue; derive half legacy; derive half v10; derive broken v10;
derive union v10 legacy v10r; derive peak3 legacy <events> legacy3; then run each label set / pool on folds 01234.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research" / "gesture"), str(ROOT / "claude_labeling")]
warnings.filterwarnings("ignore")

OUT = Path(os.environ.get("LEGACY_PROXY_DIR", "/tmp/work/lp"))
DATA = ROOT / "DATA"
ROADMAP = DATA / "fold_roadmap_files.json"
TEST_FOLD = 0
STRIDE = int(os.environ.get("LEGACY_PROXY_STRIDE", "3"))
LEGACY_MODEL_PARAMS = dict(objective="binary", verbosity=-1, boosting_type="gbdt", n_estimators=122, num_leaves=28,
                           max_depth=17, min_child_samples=88, lambda_l1=3.454105162818673,
                           lambda_l2=7.710286979829347, learning_rate=0.09100112249715728,
                           subsample=0.7143525040349332, subsample_freq=2, colsample_bytree=0.738023377461196,
                           scale_pos_weight=7.78138487144547, min_split_gain=0.04156687427531621, random_state=42,
                           deterministic=True, force_col_wise=True)
N_FEATURES = 95
THRESHOLD = 0.85                     # the trainer's DEFAULT_THRESHOLD, the firmware's decision threshold
REFRACTORY_MS, HIT_BEFORE_MS, HIT_AFTER_MS, BOUNCE_MS, PAIR_MS = 150.0, 100.0, 250.0, 40.0, 500.0
POS_KINDS = ("normal", "soft", "double_1st", "double_2nd", "fast240", "slowpinch", "swipe")
NEG_KINDS = ("finger", "fist", "wrist", "arm", "band", "nothing")
TH_GRID = np.round(np.r_[np.arange(0.30, 0.95, 0.05), 0.97, 0.98, 0.99], 4)


# ================================================================================================ input pools
def _legacy_model_features():
    """The 95 inputs of the user's legacy LightGBM (b8d6db72 info JSON): 31 raw fields + 64 diff / ratio."""
    p = ROOT / "research" / "gesture" / "legacy_tap_f1_features.json"
    return json.loads(p.read_text())["selected_feature_names"]


def acc_ratio_names():
    """36 ACC ratios: each (axis, zone) of the newest ACC PSD over its running mean and over the oldest slot."""
    out = []
    for a in "XYZ":
        for z in range(6):
            cur = f"p_core->acc_psd[0][{a}][{z}]"
            out.append(f"p_core->ratio({cur}, p_core->acc_psd_mean[{a}][{z}])")
            out.append(f"p_core->ratio({cur}, p_core->acc_psd[ACC_PSD_HISTORY_SIZE - 1][{a}][{z}])")
    return out


def pools(raw_cols):
    import tap_features as TF
    leg = [f for f in _legacy_model_features() if f not in raw_cols]
    dedupe = lambda xs: list(dict.fromkeys(xs))
    p = {"raw": list(raw_cols)}
    p["leg"] = dedupe(p["raw"] + leg)
    p["new"] = dedupe(p["leg"] + TF.DERIVED + acc_ratio_names())
    p["newacc"] = dedupe(p["leg"] + acc_ratio_names())
    p["lite"] = dedupe(p["new"] + list(TF.LITE_COLUMNS))
    nowl = lambda xs: [x for x in xs if not any(w in x for w in ("->wl[", "->wl_head[", "->wl_tail["))]
    for k in ("leg", "newacc", "new"):
        p[f"{k}_nowl"] = nowl(p[k])                 # no EMG WL input (the default of make train since 2026-11-27)
    wl_ratios = [x for x in TF.DERIVED if x.startswith("p_core->ratio(p_core->wl")]
    p["newacc_wlratio"] = dedupe(p["newacc_nowl"] + wl_ratios)   # WL only as scale-free head / tail ratios
    return p


_CALL = re.compile(r"^p_core->(diff|ratio)\((p_core->.+?), (p_core->.+)\)$")


class FeatureMaker:
    """Rows of raw fields (+ Lean-lite) -> any named input, with the firmware helpers diff (a - b, saturated int16)
    and ratio ((a + 1) / (b + 1), float)."""

    def __init__(self, base_cols, names):
        ci = {c: i for i, c in enumerate(base_cols)}
        self.names = list(names)
        self.direct, self.diff, self.ratio = [], [], []
        for j, n in enumerate(self.names):
            if n in ci:
                self.direct.append((j, ci[n])); continue
            m = _CALL.match(n)
            if not m or m.group(2) not in ci or m.group(3) not in ci:
                raise KeyError(f"cannot compute input {n!r}")
            (self.diff if m.group(1) == "diff" else self.ratio).append((j, ci[m.group(2)], ci[m.group(3)]))

    def __call__(self, B):
        B = np.asarray(B, np.float32)
        X = np.empty((len(B), len(self.names)), np.float32)
        if self.direct:
            j, c = map(np.array, zip(*self.direct)); X[:, j] = B[:, c]
        if self.diff:
            j, a, b = map(np.array, zip(*self.diff)); X[:, j] = np.clip(B[:, a] - B[:, b], -32768, 32767)
        if self.ratio:
            j, a, b = map(np.array, zip(*self.ratio)); X[:, j] = (B[:, a] + 1.0) / (B[:, b] + 1.0)
        return X


# ================================================================================================ cache
def _read_file(item):
    import tap_features as TF
    rel, fold, cls = item
    p = DATA / rel
    df = pd.read_csv(p if p.exists() else p.with_name(p.name + ".gz"))
    raw_cols = [c for c in df.columns if c.startswith("p_core->")]
    raw = df[raw_cols].to_numpy(np.float64)
    lp = DATA / rel.replace("DATA_TAP/", "DATA_TAP_LEAN8/", 1)
    lp = lp if lp.exists() else lp.with_name(lp.name + ".gz")
    lite = np.zeros((len(df), len(TF.LITE_COLUMNS)), np.int16)
    if lp.exists():
        fr = pd.read_csv(lp, usecols=["Timestamp", "lean8_frames"])
        if len(fr) == len(df) and np.array_equal(fr.Timestamp.to_numpy(), df.Timestamp.to_numpy()):
            lite = TF.lite_rows(raw.astype(np.float32), raw_cols, fr.lean8_frames.to_numpy()).astype(np.int16)
    t = (df.Timestamp.to_numpy(np.float64) - float(df.Timestamp.iloc[0])) if len(df) else np.zeros(0)
    return dict(rel=rel, cols=raw_cols, raw=np.clip(np.round(raw), -32768, 32767).astype(np.int16), lite=lite,
                t=t.astype(np.float32), ptr=(df.PointerState.to_numpy() > 0).astype(np.int8),
                cls=df.class_id.to_numpy().astype(np.int8), fold=fold, n=len(df))


def cmd_cache():
    from multiprocessing import Pool
    from fold_split import load_roadmap
    import tap_features as TF
    c = load_roadmap(ROADMAP, DATA)
    items = [(r.path.relative_to(DATA).as_posix().removesuffix(".gz"), r.fold_id, r.class_id)
             for r in c.get_records("DATA_TAP")]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = []
    with Pool(2) as pool:
        for k, r in enumerate(pool.imap(_read_file, items, chunksize=8)):
            res.append(r)
            if k % 500 == 0:
                print(f"  {k}/{len(items)} files {time.time() - t0:.0f}s", flush=True)
    cols = res[0]["cols"]
    assert all(r["cols"] == cols for r in res), "raw columns differ between files"
    n = np.array([r["n"] for r in res]); start = np.r_[0, np.cumsum(n)[:-1]]
    np.save(OUT / "raw.npy", np.concatenate([r["raw"] for r in res]))
    np.save(OUT / "lite.npy", np.concatenate([r["lite"] for r in res]))
    for k in ("t", "ptr", "cls"):
        np.save(OUT / f"{k}.npy", np.concatenate([r[k] for r in res]))
    files = pd.DataFrame(dict(rel=[r["rel"] for r in res], fold=[r["fold"] for r in res], start=start, n=n))
    files["folder"] = files.rel.str.split("/").str[1]
    files["lite_ok"] = [bool(np.any(r["lite"])) for r in res]
    files.to_csv(OUT / "files.csv", index=False)
    (OUT / "columns.json").write_text(json.dumps(dict(raw=cols, lite=list(TF.LITE_COLUMNS))))
    print(f"cache: {len(files)} files, {int(n.sum()):,} rows, lite ok {files.lite_ok.mean():.1%}, {time.time() - t0:.0f}s")


def load_cache():
    files = pd.read_csv(OUT / "files.csv")
    cols = json.loads((OUT / "columns.json").read_text())
    z = {k: np.load(OUT / f"{k}.npy", mmap_mode="r") for k in ("raw", "lite", "t", "ptr", "cls")}
    return files, cols, z


# ================================================================================================ label sets
def _v10_file(rel):
    import unified_tap_labeler_deterministic as v10
    warnings.filterwarnings("ignore")
    p = DATA / rel
    p = p if p.exists() else p.with_name(p.name + ".gz")
    try:
        _, _, _, df = v10.process_positive_file(p, DATA / "DATA_TAP", p, 1)
        return (df["state"].to_numpy() > 0).astype(np.int8)
    except Exception:                     # the old labeller rewrites a failed file with zero labels
        return None


def cmd_v10():
    from multiprocessing import Pool
    files, _, z = load_cache()
    y = np.zeros(int(files.n.sum()), np.int8)
    pos = files[files.folder.isin(["TAP", "SWIPE"])]
    with Pool(2) as pool:
        out = pool.map(_v10_file, list(pos.rel), chunksize=16)
    bad = 0
    for (_, f), v in zip(pos.iterrows(), out):
        if v is None or len(v) != f.n:
            bad += 1; continue
        y[f.start:f.start + f.n] = v
    np.save(OUT / "y_v10.npy", y)
    print(f"v10: {int((y == 1).sum()):,} tap rows, {bad} files failed (all 0, as the labeller writes them)")


def _rescue_file(rel):
    import tap_v10_rescue as RESCUE
    warnings.filterwarnings("ignore")
    p = DATA / rel
    p = p if p.exists() else p.with_name(p.name + ".gz")
    try:
        y, info = RESCUE.label_file(p, workdir="/dev/shm" if os.path.isdir("/dev/shm") else None,
                                    insert=os.environ.get("RESCUE_INSERT", "1") == "1",
                                    split=os.environ.get("RESCUE_SPLIT", "1") == "1")
        return y, info
    except Exception as exc:
        return None, dict(status=f"error {exc}")


def cmd_rescue():
    """the old labeller on a repaired pointer (claude_labeling/tap_v10_rescue.py) -> y_v10rescue.npy"""
    from multiprocessing import Pool
    files, _, _ = load_cache()
    y = np.zeros(int(files.n.sum()), np.int8)
    pos = files[files.folder.isin(["TAP", "SWIPE"])]
    with Pool(2) as pool:
        out = pool.map(_rescue_file, list(pos.rel), chunksize=16)
    infos = []
    for (_, f), (v, info) in zip(pos.iterrows(), out):
        infos.append(dict(rel=f.rel, **info))
        if v is None or len(v) != f.n:
            continue
        y[f.start:f.start + f.n] = v
    name = os.environ.get("RESCUE_NAME", "v10rescue")
    np.save(OUT / f"y_{name}.npy", y)
    q = pd.DataFrame(infos); q.to_csv(OUT / f"{name}_quality.csv", index=False)
    print(f"{name}: {int((y == 1).sum()):,} tap rows, {int((y < 0).sum()):,} left out; splits "
          f"{int(q.get('splits', pd.Series(dtype=float)).sum())}, inserted {int(q.get('inserted', pd.Series(dtype=float)).sum())}, "
          f"bounces {int(q.get('bounces', pd.Series(dtype=float)).sum())}; status {q.status.value_counts().to_dict()}")


def cmd_sig(name, npz):
    files, _, _ = load_cache()
    z = np.load(npz)
    y = np.zeros(int(files.n.sum()), np.int8)
    miss = 0
    for _, f in files.iterrows():
        key = f.rel + ".gz"
        if key in z.files:
            v = z[key]
            if len(v) != f.n:
                raise ValueError(f"{key}: {len(v)} labels for {f.n} rows")
            y[f.start:f.start + f.n] = v
        elif f.folder in ("TAP", "SWIPE"):
            miss += 1
    np.save(OUT / f"y_{name}.npy", y)
    print(f"{name}: tap {int((y == 1).sum()):,} / unknown {int((y < 0).sum()):,} rows, {miss} TAP/SWIPE files absent")


def _blocks(y):
    e = np.diff(np.r_[0, (y > 0).astype(np.int8), 0])
    return np.flatnonzero(e > 0), np.flatnonzero(e < 0)


def cmd_derive(kind, base, *args):
    """label sets made from others (TAP_F1_LEGACY.md):
      half BASE                 every 2nd TAP / SWIPE row -1 (the legacy data mix)       -> y_<BASE>half
      broken BASE               recordings the signal labels leave out (y_legacy all -1) -> y_<BASE>b
      union BASE ADD NAME       BASE + every ADD tap with no BASE tap within 150 ms       -> y_NAME
      unknown BASE ADD NAME     those ADD taps' rows -100..+150 ms unknown instead      -> y_NAME
      role BASE EVENTS ROLE NAME   union, only ADD taps of one tap_signal role (events.csv.gz of label_signal.py)
      peak3 BASE EVENTS NAME    BASE's taps cut to the peak row +-1                      -> y_NAME"""
    files, _, z = load_cache()
    t = np.asarray(z["t"], float)
    y = np.load(OUT / f"y_{base}.npy").copy()
    pos = files[files.folder.isin(["TAP", "SWIPE"])]
    if kind == "half":
        for _, f in pos.iterrows():
            seg = y[f.start:f.start + f.n]; seg[1::2] = -1
        name = f"{base}half"
    elif kind == "broken":
        sig = np.load(OUT / "y_legacy.npy")
        for _, f in pos.iterrows():
            if (sig[f.start:f.start + f.n] < 0).all():
                y[f.start:f.start + f.n] = -1
        name = f"{base}b"
    elif kind in ("union", "unknown", "role"):
        if kind == "role":
            events, role, name = args
            add = np.load(OUT / "y_legacy.npy")
            ev = pd.read_csv(events, usecols=["file", "k", "role", "y"])
            keep_rows = {}
            for fname, g in ev[(ev.y == 1) & (ev.role == role)].groupby("file"):
                keep_rows[fname.removesuffix(".gz")] = set(g.k.astype(int))
        else:
            addname, name = args
            add = np.load(OUT / f"y_{addname}.npy")
        for _, f in pos.iterrows():
            sl = slice(f.start, f.start + f.n); tt = t[sl]; bt = tt[y[sl] > 0]; out = y[sl]
            for a, b in zip(*_blocks(add[sl])):
                c = 0.5 * (tt[a] + tt[b - 1])
                if len(bt) and np.min(np.abs(bt - c)) <= 150:
                    continue
                if kind == "role" and not (keep_rows.get(f.rel, set()) & set(range(a, b))):
                    continue
                if kind == "unknown":
                    lo, hi = np.searchsorted(tt, c - 100), np.searchsorted(tt, c + 150)
                    seg = out[lo:hi]; seg[seg == 0] = -1
                else:
                    out[a:b] = 1
    elif kind == "peak3":
        events, name = args
        ev = pd.read_csv(events, usecols=["file", "k", "y"])
        ks = {fn.removesuffix(".gz"): g.k.astype(int).to_numpy() for fn, g in ev[ev.y == 1].groupby("file")}
        new = y.copy(); new[new > 0] = 0
        for _, f in pos.iterrows():
            src = y[f.start:f.start + f.n]; dst = new[f.start:f.start + f.n]
            for k in ks.get(f.rel, []):
                if 0 <= k < len(src) and src[k] == 1:
                    lo, hi = max(0, k - 1), min(len(src), k + 2)
                    dst[lo:hi] = np.where(src[lo:hi] == 1, 1, dst[lo:hi])
        y = new
    else:
        raise SystemExit(cmd_derive.__doc__)
    np.save(OUT / f"y_{name}.npy", y)
    print(f"{name}: tap rows {int((y > 0).sum()):,}, left out {int((y < 0).sum()):,}")


# ================================================================================================ evaluation
def rec_kind(folder, rel):
    f = rel.split("/")[-1]
    if folder == "TAP":
        for key, kind in (("DoubleTap", "double"), ("240bpm", "fast240"), ("SlowPinch", "slowpinch"), ("SoftTap", "soft")):
            if key in f:
                return kind
        return "normal"
    if folder == "SWIPE":
        return "swipe"
    if folder == "FINGER":
        return "fist" if "fist" in f.lower() else "finger"
    return folder.lower()


def events(t, s, th):
    up = np.flatnonzero((s >= th) & ~np.r_[False, s[:-1] >= th])
    out, last = [], -1e18
    for r in up:
        if t[r] - last >= REFRACTORY_MS:
            out.append(t[r]); last = t[r]
    return np.asarray(out)


def presses(t, ptr):
    e = np.diff(np.r_[0, ptr.astype(np.int8), 0])
    dn = t[np.flatnonzero(e > 0)].astype(float); up_ = t[np.minimum(np.flatnonzero(e < 0) - 1, len(t) - 1)].astype(float)
    if len(dn) > 1:
        keep = np.r_[True, (dn[1:] - up_[:-1]) >= BOUNCE_MS]; dn = dn[keep]
    if len(ptr) and ptr[0]:
        dn = dn[1:]                                    # began before the recording
    return dn


def match(ev, pr):
    used = np.zeros(len(ev), bool); hit = np.zeros(len(pr), bool)
    for k, o in enumerate(pr):
        i = int(np.searchsorted(ev, o - HIT_BEFORE_MS))
        while i < len(ev) and used[i]:
            i += 1
        if i < len(ev) and ev[i] <= o + HIT_AFTER_MS:
            used[i] = True; hit[k] = True
    return hit, used


def evaluate(files, z, score, ths=TH_GRID):
    """counts per threshold: hits / presses per tap kind, false events / minutes per no-tap kind, extra events on
    tap recordings (not matched to a press)"""
    acc = {f"hit|{k}": np.zeros(len(ths)) for k in POS_KINDS}
    acc.update({f"n|{k}": 0 for k in POS_KINDS})
    acc.update({f"false|{k}": np.zeros(len(ths)) for k in NEG_KINDS})
    acc.update({f"min|{k}": 0.0 for k in NEG_KINDS})
    acc["extra|taprec"] = np.zeros(len(ths)); acc["min|taprec"] = 0.0
    for _, f in files.iterrows():
        sl = slice(f.start, f.start + f.n)
        t = np.asarray(z["t"][sl], float); s = score[f["sstart"]:f["sstart"] + f.n]
        kind = rec_kind(f.folder, f.rel)
        minutes = float(t[-1] - t[0]) / 6e4 if f.n > 1 else 0.0
        if f.folder in ("TAP", "SWIPE"):
            pr = presses(t, np.asarray(z["ptr"][sl]))
            pr = pr[(pr >= t[0] + HIT_BEFORE_MS) & (pr <= t[-1] - HIT_AFTER_MS)]
            gap = np.diff(np.r_[-1e18, pr])
            kinds = np.array([(("double_2nd" if g < PAIR_MS else "double_1st") if kind == "double" else kind) for g in gap])
            for k in set(kinds):
                acc[f"n|{k}"] += int((kinds == k).sum())
            acc["min|taprec"] += minutes
            for j, th in enumerate(ths):
                ev = events(t, s, th)
                hit, used = match(ev, pr)
                for k in set(kinds):
                    acc[f"hit|{k}"][j] += hit[kinds == k].sum()
                acc["extra|taprec"][j] += int((~used).sum())
        elif kind in NEG_KINDS:
            acc[f"min|{kind}"] += minutes
            for j, th in enumerate(ths):
                acc[f"false|{kind}"][j] += len(events(t, s, th))
    return acc


def summarize(acc, ths=TH_GRID, at=THRESHOLD):
    j = int(np.argmin(np.abs(ths - at)))
    out = {"threshold": float(ths[j])}
    taps_h = sum(acc[f"hit|{k}"][j] for k in POS_KINDS if k != "swipe"); taps_n = sum(acc[f"n|{k}"] for k in POS_KINDS if k != "swipe")
    out["taps"] = taps_h / max(taps_n, 1)
    for k in POS_KINDS:
        out[k] = acc[f"hit|{k}"][j] / max(acc[f"n|{k}"], 1)
    fal = sum(acc[f"false|{k}"][j] for k in NEG_KINDS); mins = sum(acc[f"min|{k}"] for k in NEG_KINDS)
    out["fpm_move"] = fal / max(mins, 1e-9)
    for k in NEG_KINDS:
        out[f"fpm_{k}"] = acc[f"false|{k}"][j] / max(acc[f"min|{k}"], 1e-9)
    out["extra_taprec_pm"] = acc["extra|taprec"][j] / max(acc["min|taprec"], 1e-9)
    return out


def at_equal_false(acc, target, ths=TH_GRID):
    """metrics where the movement false-tap rate equals target (linear interpolation over the threshold grid)"""
    fal = sum(acc[f"false|{k}"] for k in NEG_KINDS); mins = sum(acc[f"min|{k}"] for k in NEG_KINDS)
    curve = fal / max(mins, 1e-9)
    for j in range(len(ths) - 1):
        if curve[j] >= target >= curve[j + 1]:
            a = 0.0 if curve[j] == curve[j + 1] else (curve[j] - target) / (curve[j] - curve[j + 1])
            lerp = lambda v: v[j] * (1 - a) + v[j + 1] * a
            o = {"threshold": float(lerp(ths))}
            tn = sum(acc[f"n|{k}"] for k in POS_KINDS if k != "swipe")
            o["taps"] = lerp(sum(acc[f"hit|{k}"] for k in POS_KINDS if k != "swipe")) / max(tn, 1)
            for k in POS_KINDS:
                o[k] = lerp(acc[f"hit|{k}"]) / max(acc[f"n|{k}"], 1)
            for k in NEG_KINDS:
                o[f"fpm_{k}"] = lerp(acc[f"false|{k}"]) / max(acc[f"min|{k}"], 1e-9)
            o["fpm_move"] = target
            return o
    return None


# ================================================================================================ train + evaluate
def _fit(base, base_cols, names, rows, target, spw, log_tag=""):
    """f_classif top-N_FEATURES of `names` on `rows`, then the legacy LightGBM; (model, FeatureMaker of the inputs)"""
    import lightgbm as lgb
    from sklearn.feature_selection import f_classif
    B = base(rows); y = target[rows].astype(np.int8)
    scores = np.zeros(len(names))
    for a in range(0, len(names), 64):                          # f_classif, a block of inputs at a time
        s, _ = f_classif(FeatureMaker(base_cols, names[a:a + 64])(B), y)
        scores[a:a + 64] = np.nan_to_num(s, nan=-np.inf)
    order = np.lexsort((np.arange(len(scores)), -scores))
    sel = [names[i] for i in np.sort(order[:N_FEATURES])]
    fsel = FeatureMaker(base_cols, sel)
    X = fsel(B); del B
    params = dict(LEGACY_MODEL_PARAMS, n_jobs=2, scale_pos_weight=float(spw),
                  random_state=int(os.environ.get("LEGACY_PROXY_SEED", "42")))
    return lgb.LGBMClassifier(**params).fit(X, y), fsel, int(y.sum())


def _score(model, fsel, base, files):
    ix_all = np.concatenate([np.arange(f.start, f.start + f.n) for _, f in files.iterrows()])
    score = np.zeros(len(ix_all), np.float32)
    for a in range(0, len(ix_all), 200_000):
        ix = ix_all[a:a + 200_000]
        score[a:a + len(ix)] = model.predict_proba(fsel(base(ix)))[:, 1]
    return ix_all, score


def _rows(files, keep):
    rows = np.concatenate([np.arange(f.start + (k % STRIDE), f.start + f.n, STRIDE) for k, (_, f) in enumerate(files.iterrows())])
    return rows[keep[rows]]


SPW_GRID = (1.0, 2.0, 4.0, 7.78138487144547, 15.0)


def cmd_run(label, pool_name, spw=None, folds=(TEST_FOLD,), tag=None):
    """spw None = the legacy model's 7.78; "auto" = the trainer's choice on F1 at 0.85: every SPW_GRID value is fitted
    on the training folds but one and scored on that one (validation), the best F1 is refitted on all training folds."""
    from sklearn.metrics import f1_score
    files, cols, z = load_cache()
    base_cols = cols["raw"] + cols["lite"]
    if pool_name.startswith("file:"):                 # a kept-input list (select_tap_f1_features.py report JSON)
        names = json.loads(Path(pool_name[5:]).read_text())["selected"]
        pool_name = "rfe" + Path(pool_name[5:]).stem.replace("selected", "")
    else:
        names = pools(cols["raw"])[pool_name]
    y_all = np.load(OUT / f"y_{label}.npy")
    target = (y_all > 0) & (np.asarray(z["cls"]) == 1)          # the trainer: state > 0 AND class_id == 1
    keep = y_all >= 0                                           # the one line: unknown rows (state -1) never train

    def base(ix):
        return np.hstack([np.asarray(z["raw"][ix], np.float32), np.asarray(z["lite"][ix], np.float32)])

    total, f1s, chosen, t0 = None, [], [], time.time()
    tapfrac = float(os.environ.get("LEGACY_PROXY_TAPFRAC", "1"))   # keep this share of TAP / SWIPE training files
    for k in folds:
        tr = files[files.fold != k]
        if tapfrac < 1:
            rng = np.random.default_rng(1000 + k)
            pos = tr.folder.isin(["TAP", "SWIPE"]).to_numpy()
            tr = tr[~pos | (rng.random(len(tr)) < tapfrac)]
        use_spw = LEGACY_MODEL_PARAMS["scale_pos_weight"] if spw is None else spw
        if spw == "auto":
            vf = (k + 1) % 5
            best = (-1.0, None)
            for w in SPW_GRID:
                m, fs, _ = _fit(base, base_cols, names, _rows(tr[tr.fold != vf], keep), target, w)
                ix, sc = _score(m, fs, base, tr[tr.fold == vf])
                f1v = float(f1_score(target[ix][keep[ix]], sc[keep[ix]] >= THRESHOLD))
                print(f"   fold {k} val {vf} spw {w:g}: F1 {f1v:.4f}", flush=True)
                best = max(best, (f1v, w))
            use_spw = best[1]
        model, fsel, npos = _fit(base, base_cols, names, _rows(tr, keep), target, use_spw)
        te = files[files.fold == k].copy()
        te["sstart"] = np.r_[0, np.cumsum(te.n.to_numpy())[:-1]]
        ix_all, score = _score(model, fsel, base, te)
        f1s.append(float(f1_score(target[ix_all][keep[ix_all]], score[keep[ix_all]] >= THRESHOLD)))
        chosen.append(dict(fold=int(k), spw=float(use_spw), selected=fsel.names))
        acc = evaluate(te, z, score)
        total = acc if total is None else {a: total[a] + acc[a] for a in acc}
        s = summarize(acc)
        print(f"  fold {k}: spw={use_spw:g} F1={f1s[-1]:.4f} taps={s['taps']:.3f} move={s['fpm_move']:.2f} "
              f"fist={s['fpm_fist']:.2f} finger={s['fpm_finger']:.2f} band={s['fpm_band']:.2f}", flush=True)
    name = tag or (f"{label}__{pool_name}" + (f"__spw{spw}" if spw is not None else "") + (f"__tap{tapfrac:g}" if tapfrac < 1 else "") + (f"__seed{os.environ['LEGACY_PROXY_SEED']}" if os.environ.get("LEGACY_PROXY_SEED") else "")
                   + ("" if tuple(folds) == (TEST_FOLD,) else "__f" + "".join(map(str, folds))))
    res = dict(name=name, label=label, pool=pool_name, n_pool=len(names), folds=list(map(int, folds)),
               f1_own_labels=float(np.mean(f1s)), f1_per_fold=f1s, chosen=chosen, seconds=round(time.time() - t0),
               at_085=summarize(total),
               acc={a: (v.tolist() if isinstance(v, np.ndarray) else v) for a, v in total.items()})
    (OUT / "runs").mkdir(exist_ok=True)
    (OUT / "runs" / f"{name}.json").write_text(json.dumps(res))
    s = res["at_085"]
    print(f"{name}: F1(own)={res['f1_own_labels']:.4f} taps={s['taps']:.3f} move={s['fpm_move']:.2f}/min "
          f"fist={s['fpm_fist']:.2f} finger={s['fpm_finger']:.2f} wrist={s['fpm_wrist']:.2f} arm={s['fpm_arm']:.2f} "
          f"band={s['fpm_band']:.2f} ({res['seconds']}s)", flush=True)
    return res


def cmd_table(match_target=None):
    runs = sorted((OUT / "runs").glob("*.json"))
    rows = []
    for p in runs:
        r = json.loads(p.read_text())
        acc = {k: (np.array(v) if isinstance(v, list) else v) for k, v in r["acc"].items()}
        row = dict(name=r["name"], f1=r["f1_own_labels"], **{k: v for k, v in r["at_085"].items()})
        rows.append(row)
        if match_target:
            m = at_equal_false(acc, float(match_target))
            if m:
                rows.append(dict(name=r["name"] + f" @{match_target}/min", f1=np.nan, **m))
    df = pd.DataFrame(rows)
    show = ["name", "f1", "threshold", "taps", "normal", "soft", "double_1st", "double_2nd", "fast240", "slowpinch",
            "swipe", "fpm_move", "fpm_fist", "fpm_finger", "fpm_wrist", "fpm_arm", "fpm_band", "fpm_nothing",
            "extra_taprec_pm"]
    df = df[[c for c in show if c in df.columns]]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["table"]
    if cmd == "cache":
        cmd_cache()
    elif cmd == "v10":
        cmd_v10()
    elif cmd == "rescue":
        cmd_rescue()
    elif cmd == "sig":
        cmd_sig(args[0], args[1])
    elif cmd == "derive":
        cmd_derive(*args)
    elif cmd == "run":                  # run LABEL POOL [SPW|auto|-] [FOLDS e.g. 01234]
        sp = args[2] if len(args) > 2 and args[2] != "-" else None
        cmd_run(args[0], args[1], (sp if sp in (None, "auto") else float(sp)),
                tuple(int(c) for c in args[3]) if len(args) > 3 else (TEST_FOLD,))
    elif cmd == "table":
        cmd_table(args[0] if args else None)
    else:
        raise SystemExit(__doc__)
