#!/usr/bin/env python3
"""Top labels vs onset labels: which positive window gives the TAP model that works best on the band?

    MUDRA_GR_SPLIT=file python research/gesture/tap_top_research.py compare <ref arm> <arm> [...]
    MUDRA_GR_SPLIT=file python research/gesture/tap_top_research.py margin  <ref arm> <arm> [...]

Arms are tap_generalization_research.py arms (file split, saved focus files and fold models), e.g.
    B=raw126_diff+augm+litediff+press2+kb14+leaves7      (final14: the labeller's onset windows)
    $B+top / $B+midtop                                  (claude_labeling/tap_top.py: top / middle + top of the jolt)
    $B+obs60                                            (final14obs: the teacher's most distinguishable 60 ms)

compare: from the focus files (every real pointer press per tap kind, false taps per movement kind, per file):
  * at the firmware threshold 0.85 as trained;
  * at the same overall false-tap rate: 6.2 / min (the old header's level, where make train puts the model),
    2.8 / min and 2.0 / min (strict);
  * the same thresholds under the simulated band move (EMG gain, tilt, contact noise; "placement");
  * margin: at the 6.2 / min point, scores lowered by 1 and 2 logits (live scores on the band sat below the
    recordings' scores: taps that only just cross the threshold are lost there).
  Differences to the first arm with a 95% bootstrap over files (files of one person are correlated: too narrow).

margin: re-scores the saved fold models (validation files only) and, per confirmed tap, measures the highest score in
  the hit window and how long the score stays above the operating threshold (the 6.2 / min point); counts release
  clicks (a second kept event 150-600 ms after a hit on the tap's own press) and early clicks (> 100 ms before).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUDRA_GR_SPLIT", "file")
import tap_generalization_research as G   # noqa: E402
import replay as R                         # noqa: E402

TAPS = ("normal", "soft", "double_1st", "double_2nd", "fast240", "swipe")
MOVES = ("finger", "fist", "wrist", "arm", "band")
TARGETS = (6.2, 2.8, 2.0)
DELTAS = (1.0, 2.0)
LOGIT = np.log(G.FOCUS_TH / (1 - G.FOCUS_TH))


def short(arm, ref):
    return arm if arm == ref else (arm[len(ref):] if arm.startswith(ref) else arm)


def _pos(th):
    """Grid position (j, alpha) of a threshold on the focus grid (even in log-odds)."""
    lg = np.log(th / (1 - th))
    if lg <= LOGIT[0]:
        return 0, 0.0
    if lg >= LOGIT[-1]:
        return len(LOGIT) - 2, 1.0
    j = int(np.searchsorted(LOGIT, lg) - 1)
    return j, float((lg - LOGIT[j]) / (LOGIT[j + 1] - LOGIT[j]))


def _shift(j, al, delta):
    lg = LOGIT[j] * (1 - al) + LOGIT[j + 1] * al + delta
    return _pos(1.0 / (1.0 + np.exp(-lg)))


def view(A, j, al, cond, w=None):
    m = G.focus_metrics(A, j, al, w)
    out = {t_: m[f"{cond}|{t_}"] for t_ in TAPS}
    out["taps_mean"] = np.mean([out[t_] for t_ in TAPS], 0)
    out.update({f"{t_}/min": m[f"{cond}|fpm_{t_}"] for t_ in MOVES})
    out["all/min"] = m[f"{cond}|fpm_all"]
    out["worst/min"] = np.max([out[f"{t_}/min"] for t_ in MOVES], 0)
    if f"{cond}|lat_p50" in m:
        out["lat50"] = m[f"{cond}|lat_p50"]; out["lat90"] = m[f"{cond}|lat_p90"]
    return out


def positions(A):
    """name -> (j, alpha, condition) of every view."""
    curve = G._fpm_curve(A, "nominal")
    pos = {"0.85": (*_pos(0.85), "nominal"), "0.85 band moved": (*_pos(0.85), "placement")}
    for tg in TARGETS:
        j, al = G._at(curve, tg)
        if j is None:
            continue
        pos[f"{tg:g}/min"] = (j, al, "nominal")
        pos[f"{tg:g}/min band moved"] = (j, al, "placement")
        if tg == TARGETS[0]:
            for d in DELTAS:
                pos[f"{tg:g}/min, scores -{d:g} logit"] = (*_shift(j, al, d), "nominal")
                pos[f"{tg:g}/min, scores -{d:g} logit, band moved"] = (*_shift(j, al, d), "placement")
    return pos


def compare(arms, n_boot=1000, seed=5, out_md=None):
    ref = arms[0]
    A = {a: G._load_focus(a) for a in arms}
    n = len(A[ref]["n|nominal|normal"])
    W = np.random.default_rng(seed).multinomial(n, np.ones(n) / n, size=n_boot).astype(np.float64)
    pos = {a: positions(A[a]) for a in arms}
    lines = []
    cols = list(TAPS) + ["taps_mean"] + [f"{t_}/min" for t_ in MOVES] + ["all/min", "worst/min", "lat50", "lat90"]
    for name in pos[ref]:
        rows, deltas = [], []
        for a in arms:
            if name not in pos[a]:
                continue
            j, al, cond = pos[a][name]
            v = view(A[a], j, al, cond)
            thr = float(1 / (1 + np.exp(-(LOGIT[j] * (1 - al) + LOGIT[j + 1] * al))))
            rows.append(dict(arm=short(a, ref), thr=round(thr, 4),
                             **{k: (round(100 * float(v[k]), 1) if k in TAPS or k == "taps_mean"
                                    else round(float(v.get(k, np.nan)), 2)) for k in cols}))
            if a != ref and name in pos[ref]:
                vb = view(A[a], j, al, cond, W); jr, alr, cr = pos[ref][name]; vr = view(A[ref], jr, alr, cr, W)
                d = {}
                for k in cols:
                    if k not in vb or k not in vr:
                        continue
                    x = (vb[k] - vr[k]) * (100 if (k in TAPS or k == "taps_mean") else 1)
                    lo, hi = np.nanpercentile(x, [2.5, 97.5])
                    d[k] = f"{np.nanmean(x):+.1f}{'*' if (lo > 0 or hi < 0) else ''}"
                deltas.append(dict(arm=short(a, ref), **d))
        lines.append(f"\n### {name}\n")
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
        if deltas:
            lines.append("\nDifference to the reference (points / events per min; * = 95% file bootstrap excludes 0):\n")
            lines.append(pd.DataFrame(deltas).to_markdown(index=False))
    text = "\n".join(lines)
    print(text)
    if out_md:
        Path(out_md).write_text(text + "\n")
    return text


# ---------------------------------------------------------------------------------------------------------- margin
def _kept(d, t):
    rise = np.flatnonzero(d & ~np.r_[False, d[:-1]]); kept, last = [], -1e18
    for r in rise:
        if t[r] - last >= R.TAP_REFRACTORY_MS:
            kept.append(t[r]); last = t[r]
    return np.asarray(kept, float)


def margin(arms, target=TARGETS[0], out_json=None):
    import lightgbm as lgb
    D = G.load(); data = D["data"]; tr = D["tr"]; on = D["onsets"]
    res = []
    for a in arms:
        A = G._load_focus(a); j, al = G._at(G._fpm_curve(A, "nominal"), target)
        lg_thr = LOGIT[j] * (1 - al) + LOGIT[j + 1] * al; thr = float(1 / (1 + np.exp(-lg_thr)))
        spec = tuple(json.loads((G.OUT / f"{a}.json").read_text())["config"]["spec"])
        peak, above, lat = [], [], []
        taps = rel = early = 0
        for f, (_x, b) in enumerate(D["folds"]):
            val = G.Val(set(data["group"][tr[b]]))
            p = val.predict(lgb.Booster(model_file=str(G.OUT / "models" / f"{a}_fold{f}.txt")), None, spec, None)
            lp = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
            for (s, e), t, it in zip(val.seg, val.t, val.items):
                rec, folder = int(it[0]), it[3]
                if folder not in ("TAP", "SWIPE"):
                    continue
                t = np.asarray(t, float); conf = np.sort(on[(on.rec == rec) & (on.kind == "strong")].t_on.to_numpy(float))
                conf = conf[conf >= t[0] + R.TAP_HIT_BEFORE_MS]
                if not len(conf):
                    continue
                q = lp[s:e]; ev = _kept(q >= lg_thr, t)
                for k, o in enumerate(conf):
                    m = (t >= o - R.TAP_HIT_BEFORE_MS) & (t <= o + R.TAP_HIT_AFTER_MS)
                    if not m.any():
                        continue
                    taps += 1
                    peak.append(float(q[m].max() - lg_thr))
                    dt = np.diff(np.r_[t[m], t[m][-1] + 5.5])
                    above.append(float(dt[q[m] >= lg_thr].sum()))
                    hit = ev[(ev >= o - R.TAP_HIT_BEFORE_MS) & (ev <= o + R.TAP_HIT_AFTER_MS)]
                    if len(hit):
                        lat.append(float(hit[0] - o))
                        nxt = conf[k + 1] if k + 1 < len(conf) else np.inf
                        rel += int(((ev > hit[0] + R.TAP_REFRACTORY_MS - 1e-6) & (ev <= hit[0] + 600)
                                    & (ev < nxt - R.TAP_HIT_BEFORE_MS)).sum())
                    prv = conf[k - 1] if k else -np.inf
                    early += int(((ev < o - R.TAP_HIT_BEFORE_MS) & (ev >= o - 400) & (ev > prv + R.TAP_HIT_AFTER_MS)).sum())
            del val
            G.log("TOP_MARGIN", a, "fold", f, "done")
        pk = np.asarray(peak); ab = np.asarray(above); lt = np.asarray(lat)
        res.append(dict(arm=a, threshold=round(thr, 4), taps=taps,
                        peak_logit_above_thr_p10=round(float(np.percentile(pk, 10)), 2),
                        peak_logit_above_thr_p25=round(float(np.percentile(pk, 25)), 2),
                        peak_logit_above_thr_p50=round(float(np.percentile(pk, 50)), 2),
                        taps_found=round(float((pk >= 0).mean()), 4),
                        taps_found_scores_minus_1=round(float((pk >= 1).mean()), 4),
                        taps_found_scores_minus_2=round(float((pk >= 2).mean()), 4),
                        ms_above_thr_p25=round(float(np.percentile(ab, 25)), 1),
                        ms_above_thr_p50=round(float(np.percentile(ab, 50)), 1),
                        latency_p10_p50_p90=[round(float(np.percentile(lt, q_)), 1) for q_ in (10, 50, 90)],
                        release_clicks_per_100_taps=round(100 * rel / max(taps, 1), 2),
                        early_clicks_per_100_taps=round(100 * early / max(taps, 1), 2)))
        G.log("TOP_MARGIN_DONE", json.dumps(res[-1]))
    df = pd.DataFrame(res); ref = arms[0]
    df["arm"] = [short(a, ref) for a in df.arm]
    print(df.to_markdown(index=False))
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------------------------------------------------- seeds
SEED_VIEWS = ("0.85", "6.2/min", "6.2/min band moved", "6.2/min, scores -1 logit", "6.2/min, scores -1 logit, band moved",
              "6.2/min, scores -2 logit", "6.2/min, scores -2 logit, band moved", "2.8/min", "2/min")
SEED_COLS = ("taps_mean", "normal", "soft", "double_2nd", "fast240", "swipe", "finger/min", "fist/min", "all/min")


def seeds(ref, arm, suffixes=("", "+seed1", "+seed2"), out_md=None):
    """arm - ref per training seed (paired: the same seed for both), mean [min, max] over the seeds."""
    rows = []
    for name in SEED_VIEWS:
        diffs = {k: [] for k in SEED_COLS}
        for sfx in suffixes:
            A, Bm = G._load_focus(ref + sfx), G._load_focus(arm + sfx)
            pa, pb = positions(A), positions(Bm)
            if name not in pa or name not in pb:
                continue
            va = view(A, *pa[name][:2], pa[name][2]); vb = view(Bm, *pb[name][:2], pb[name][2])
            for k in SEED_COLS:
                sc = 100 if (k in TAPS or k == "taps_mean") else 1
                diffs[k].append(float(vb[k] - va[k]) * sc)
        rows.append(dict(view=name, **{k: (f"{np.mean(v):+.1f} [{min(v):+.1f}, {max(v):+.1f}]" if v else "")
                                       for k, v in diffs.items()}))
    text = f"{short(arm, ref)} - reference, per seed ({len(suffixes)} seeds): mean [min, max]\n\n" + pd.DataFrame(rows).to_markdown(index=False)
    print(text)
    if out_md:
        Path(out_md).write_text(text + "\n")
    return rows


if __name__ == "__main__":
    cmd, arms = sys.argv[1], sys.argv[2:]
    if cmd == "seeds":
        seeds(arms[0], arms[1], out_md=os.environ.get("TOP_OUT_MD"))
    elif cmd == "compare":
        compare(arms, out_md=os.environ.get("TOP_OUT_MD"))
    elif cmd == "margin":
        margin(arms, out_json=os.environ.get("TOP_OUT_JSON"))
    else:
        raise SystemExit(__doc__)
