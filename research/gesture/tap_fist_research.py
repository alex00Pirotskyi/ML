#!/usr/bin/env python3
"""Why the TAP model fires on fists, and what removes it (TAP_FIST.md, 2026-11-26).

    python research/gesture/tap_fist_research.py devlogs <folder with model_test_*.csv>
        device logs with only the model outputs (tap_probability_raw, hold_probability_raw, Q20): false taps per
        tap_gestures.h setting, and the tap / long-press score pattern of every fist false tap
    MUDRA_GR_CACHE=... MUDRA_GR_OUT=... python research/gesture/tap_fist_research.py separability <arm>
        held-out fires of a research arm on fist recordings vs on real taps: can the model's own inputs tell them
        apart at the moment it fires (and 50-300 ms later)? LightGBM, file-grouped 5-fold CV
    MUDRA_GR_OUT=... python research/gesture/tap_fist_research.py match <reference arm> <arm> [...]
        false taps per movement kind of every arm at the reference's tap recall (its 6.2 / min point), nominal,
        band moved, and scores 1 logit lower

Arms are tap_generalization_research.py arms (file split): e.g. B=raw126_diff+augm+litediff+press2+kb14+leaves7+top+topb10
and $B+fistw3, $B+lpneg, raw126_diff+augm+litediff+press2+kb64+leaves15+top+topb10.
"""
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TAPK = ("normal", "soft", "double_1st", "double_2nd", "fast240", "swipe")
MOVK = ("finger", "fist", "wrist", "arm", "band")
ALLK = MOVK + ("nothing", "longpress", "taprec")


def devlogs(folder):
    import tap_gestures as TG
    logs = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        rows = list(csv.DictReader(open(f)))
        t = np.array([int(r["elapsed_us"]) for r in rows], np.float64) / 1000.0
        logs[Path(f).stem] = (t, np.array([int(r["tap_probability_raw"]) for r in rows]),
                              np.array([int(r["hold_probability_raw"]) for r in rows]))
    minutes = sum((t[-1] - t[0]) / 6e4 for t, _, _ in logs.values())
    print(f"{len(logs)} logs, {minutes:.2f} min")
    for fire, second in ((17, 12), (18, 15), (19, 15), (20, 15)):
        n = [sum(1 for g in TG.simulate(q, t, TG.Config(fire=fire, second=second))[0]) for t, q, _ in logs.values()]
        print(f"  tap_gestures.h {fire / 20:.2f} / {second / 20:.2f}: {sum(n)} gestures = {sum(n) / minutes:.1f}/min")
    rows = []
    for name, (t, q, h) in logs.items():
        last = -1e9
        for i in np.flatnonzero((q >= 17) & (np.r_[0, q[:-1]] < 17)):
            if t[i] - last < 150:
                continue
            last = t[i]
            w = (t >= t[i]) & (t <= t[i] + 100); after = (t >= t[i]) & (t <= t[i] + 400)
            rows.append(dict(log=name, s=round((t[i] - t[0]) / 1000, 2), tap_peak_100ms=int(q[w].max()),
                             hold_max_400ms=int(h[after].max()),
                             hold_rise_ms=float(t[np.flatnonzero(after & (h >= 18))[0]] - t[i]) if (after & (h >= 18)).any() else None))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("tap peak within 100 ms:", df.tap_peak_100ms.value_counts().sort_index().to_dict())
    print("long-press score >= 0.90 within 400 ms:", int((df.hold_max_400ms >= 18).sum()), "of", len(df),
          "| median rise", df.hold_rise_ms.median(), "ms")


def separability(arm, offsets=(0, 50, 100, 150, 200, 300)):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    import tap_generalization_research as G
    import tap_gestures_strict as S
    P = S.predictions(arm)
    fire = G.at_budget(P["pooled"]["nominal"], 4.4)[0]
    D = G.load(); rpl = D["rpl"]; recs = rpl["recs"]; press = S.presses()
    X, cls, grp, off = [], [], [], []
    for rec, folder, file, t, p, lab in P["items"]:
        is_fist = folder == "FINGER" and "fist" in file.lower()
        is_tap = folder == "TAP" and "DoubleTap" not in file and "240bpm" not in file
        if not (is_fist or is_tap):
            continue
        a = int(recs.start.iloc[rec]); tt = np.asarray(rpl["t"][a:int(recs.end.iloc[rec])], np.float64)
        fires, last = [], -1e9
        for i in np.flatnonzero((p >= fire) & (np.r_[0, p[:-1]] < fire)):
            if t[i] - last >= 150:
                fires.append(t[i]); last = t[i]
        if is_tap:
            pr = press.get(rec, np.zeros(0))
            fires = [f for f in fires if ((f >= pr - 100) & (f <= pr + 250)).any()][:6]
        for f in fires:
            for o in offsets:
                j = int(np.searchsorted(tt, f + o))
                if j < len(tt):
                    X.append(np.asarray(rpl["X"][a + j], np.float32)); cls.append(int(is_fist)); grp.append(rec); off.append(o)
    F, names, _ = G.build(np.vstack(X), ("acc", "emgraw", "diff", "wlratio"))
    cls, grp, off = np.array(cls), np.array(grp), np.array(off)
    print(f"fist fires {int(cls[off == 0].sum())}, tap fires {int((cls[off == 0] == 0).sum())}")
    for o in offsets:
        m = off == o; Xo, yo, go = F[m], cls[m], grp[m]; pred = np.zeros(len(yo))
        for tr, te in GroupKFold(5).split(Xo, yo, go):
            pred[te] = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05, colsample_bytree=0.8,
                                          subsample=0.8, subsample_freq=1, class_weight="balanced", verbosity=-1,
                                          n_jobs=2).fit(Xo[tr], yo[tr]).predict_proba(Xo[te])[:, 1]
        keep98 = np.quantile(pred[yo == 0], 0.98)
        print(f"+{o:3d} ms: AUC {roc_auc_score(yo, pred):.3f}, keeping 98% of taps removes "
              f"{100 * float((pred[yo == 1] > keep98).mean()):.0f}% of fist fires")


def _curves(out, arm, cond):
    z = np.load(out / f"focus_{arm}.npz", allow_pickle=True)
    rec = {k: z[f"hits|{cond}|{k}"].sum(1) / z[f"n|{cond}|{k}"].sum() for k in TAPK}
    fpm = {k: z[f"false|{cond}|{k}"].sum(1) / z[f"min|{cond}|{k}"].sum() for k in ALLK}
    allf = sum(z[f"false|{cond}|{k}"].sum(1) for k in ALLK) / sum(z[f"min|{cond}|{k}"].sum() for k in ALLK)
    return z["th"], rec, fpm, allf, np.mean([rec[k] for k in TAPK], 0)


def _row(out, arm, i, cond="nominal"):
    th, rec, fpm, allf, mean = _curves(out, arm, cond)
    return dict(thr=round(float(th[i]), 4), taps=round(100 * mean[i], 1), **{k: round(100 * rec[k][i], 1) for k in TAPK},
                **{f"{k}/min": round(float(fpm[k][i]), 2) for k in MOVK}, **{"all/min": round(float(allf[i]), 2)})


def match(ref, arms, target=6.2):
    out = Path(os.environ.get("MUDRA_GR_OUT", HERE / "results_tap_filesplit"))
    th, _, _, allf, mean = _curves(out, ref, "nominal")
    i_ref = int(np.flatnonzero(allf <= target)[0]); taps = mean[i_ref]
    rows = {}
    for arm in [ref] + list(arms):
        th, _, _, _, m = _curves(out, arm, "nominal")
        i = i_ref if arm == ref else int(np.flatnonzero(m >= taps)[-1])
        j = int(np.argmin(np.abs(th - 1 / (1 + np.exp(-(np.log(th[i] / (1 - th[i])) + 1))))))   # scores 1 logit lower
        rows[arm] = _row(out, arm, i)
        rows[arm + " | band moved"] = _row(out, arm, i, "placement")
        rows[arm + " | scores -1 logit"] = _row(out, arm, j)
    df = pd.DataFrame(rows).T
    print(f"every arm at the reference's tap recall ({100 * taps:.1f}%, its {target}/min point)")
    print(df.to_string())
    return df


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "devlogs":
        devlogs(sys.argv[2])
    elif cmd == "separability":
        separability(sys.argv[2])
    elif cmd == "match":
        match(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(__doc__)
