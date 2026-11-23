"""Single / double tap decision (firmware/tap_gestures.h) on held-out replay.

Scores: the research TAP models (tap_generalization_research.py, arm raw126_diff+augm, one model per person
fold), each predicting only its own held-out people. Their probability scale is not the exported header's (the
trainer weights positives by ~0.21, the research models by 1, which shifts the log-odds). Thresholds are given
on the DEVICE scale and mapped by one log-odds shift, anchored so that device 0.85 lands where the research model
has the deployed model's false rate (~2.8 events/min, nominal). Truth: every real pointer press.

Per configuration:
  pairs   (DoubleTaps recordings, consecutive presses < 500 ms apart): DOUBLE_TAP / two TAPs / one TAP / nothing
  singles (other TAP recordings, presses with no other press within 600 ms): TAP / became a DOUBLE_TAP / nothing
  swipes  (SWIPE recordings, isolated the same way; the onset model fires on swipes too): TAP / DOUBLE_TAP / nothing
  false   gestures per minute where there is no gesture: movement folders (FINGER, WRIST, ARM, BAND), all
          non-TAP/SWIPE folders, and unmatched gestures on trusted-negative rows of TAP/SWIPE recordings

    python research/gesture/tap_gestures_research.py            # writes results_tap_gestures/results.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tap_generalization_research as G   # noqa: E402
import tap_gestures as TG                 # noqa: E402

OUT = Path(os.environ.get("MUDRA_TG_OUT", HERE / "results_tap_gestures"))
ARM = os.environ.get("MUDRA_TG_ARM", "raw126_diff+augm")
PAIR_MS = 500                 # DoubleTaps recordings: presses closer than this form a pair (valley of the gaps)
HIT = (100, 250)              # a gesture onset this long before / after a press belongs to it (replay.py)
MOVEMENT = ("FINGER", "WRIST", "ARM", "BAND")
ISOLATED_MS = 600             # a single tap / swipe is scored only if no other press is this close (240 bpm files)
DEVICE_FIRE = 0.85


def logit(x):
    x = min(max(float(x), 1e-6), 1 - 1e-6)
    return float(np.log(x / (1 - x)))


def to_proxy(x, shift):
    """Device-scale threshold -> research-model threshold (one log-odds shift)."""
    return float(1.0 / (1.0 + np.exp(-(logit(x) + shift))))


def predictions():
    """[(rec, folder, file, t, p, label)] for every held-out recording of the development people."""
    import pickle
    import lightgbm as lgb
    cache = os.environ.get("MUDRA_TG_PRED_CACHE")
    if cache and Path(cache).exists():
        return pickle.load(open(cache, "rb"))
    D = G.load(); data = D["data"]; tr = D["tr"]; recs = D["rpl"]["recs"]
    res = json.loads((G.OUT / f"{ARM}.json").read_text()); spec = tuple(res["config"]["spec"])
    out = []
    for f, (_a, b) in enumerate(D["folds"]):
        val = G.Val(set(data["group"][tr[b]]))
        m = lgb.Booster(model_file=str(G.OUT / "models" / f"{ARM}_fold{f}.txt"))
        p = np.empty(len(val.X))
        for s in range(0, len(val.X), 250_000):
            p[s:s + 250_000] = m.predict(G.build(val.X[s:s + 250_000], spec)[0], num_threads=G.THREADS)
        for (s, e), t, lab, (rec, *_rest) in zip(val.seg, val.t, val.label, val.items):
            out.append((rec, recs.folder.iloc[rec], recs.file.iloc[rec], t, p[s:e].astype(np.float32), lab))
        G.log("TG_FOLD", f)
        del val
    result = (out, json.loads((G.OUT / f"{ARM}.json").read_text()))
    if cache:
        pickle.dump(result, open(cache, "wb"), protocol=4)
    return result


def presses():
    on = pd.read_csv(G.CACHE / "tap_onsets.csv")
    return {r: np.sort(g.t_on.values.astype(np.float64)) for r, g in on[on.kind == "press"].groupby("rec")}


def score(items, press, cfg):
    c = dict(pair_double=0, pair_two_taps=0, pair_one_tap=0, pair_missed=0,
             single_tap=0, single_double=0, single_missed=0,
             swipe_tap=0, swipe_double=0, swipe_missed=0,
             mov_tap=0, mov_double=0, mov_min=0.0, neg_tap=0, neg_double=0, neg_min=0.0)
    for rec, folder, file, t, p, lab in items:
        gest, _ = TG.simulate(p, t, cfg)
        kind = np.array([g[0] for g in gest], np.int8); t1 = np.array([g[1] for g in gest])
        t2 = np.array([g[2] for g in gest])
        used = np.zeros(len(gest), bool)
        dt = np.clip(np.diff(np.r_[t, t[-1]]), 0, 50)

        def take(o):
            i = np.flatnonzero(~used & (t1 >= o - HIT[0]) & (t1 <= o + HIT[1]))
            if len(i):
                used[i[0]] = True
                return int(i[0])
            return None

        if folder in ("TAP", "SWIPE"):
            pr = press.get(rec, np.zeros(0))
            pr = pr[(pr >= t[0] + HIT[0]) & (pr <= t[-1] - cfg.window_ms - HIT[1])]
            if folder == "TAP" and "DoubleTap" in file:
                k = 0
                while k < len(pr):
                    if k + 1 < len(pr) and pr[k + 1] - pr[k] < PAIR_MS:
                        o1, o2 = pr[k], pr[k + 1]; k += 2
                        i = take(o1)
                        if i is not None and kind[i] == TG.DOUBLE_TAP:
                            c["pair_double"] += 1
                        elif i is not None:
                            c["pair_two_taps" if take(o2) is not None else "pair_one_tap"] += 1
                        else:
                            j = take(o2)
                            if j is not None and kind[j] == TG.DOUBLE_TAP:
                                c["pair_double"] += 1          # first tap missed, double formed late: rare
                            else:
                                c["pair_one_tap" if j is not None else "pair_missed"] += 1
                    else:
                        take(pr[k]); k += 1                    # unpaired press: not scored
            else:
                name = "single" if folder == "TAP" else "swipe"
                allp = press.get(rec, np.zeros(0))
                for o in pr:
                    i = take(o)
                    near = allp[(np.abs(allp - o) < ISOLATED_MS) & (np.abs(allp - o) > 1e-6)]
                    if len(near):
                        continue                                # fast repeated taps: double-tap timing
                    c[f"{name}_missed" if i is None else (f"{name}_double" if kind[i] == TG.DOUBLE_TAP else f"{name}_tap")] += 1
            # unmatched gestures on trusted-negative rows
            if len(gest):
                ix = np.clip(np.searchsorted(t, t1), 0, len(t) - 1)
                bad = ~used & (lab[ix] == 0)
                c["neg_tap"] += int((bad & (kind == TG.TAP)).sum()); c["neg_double"] += int((bad & (kind == TG.DOUBLE_TAP)).sum())
            c["neg_min"] += float(dt[lab == 0].sum()) / 6e4
        else:
            n_t = int((kind == TG.TAP).sum()); n_d = int((kind == TG.DOUBLE_TAP).sum()); mins = float(dt.sum()) / 6e4
            c["neg_tap"] += n_t; c["neg_double"] += n_d; c["neg_min"] += mins
            if folder in MOVEMENT:
                c["mov_tap"] += n_t; c["mov_double"] += n_d; c["mov_min"] += mins
    return c


def summarize(c):
    P = c["pair_double"] + c["pair_two_taps"] + c["pair_one_tap"] + c["pair_missed"]
    S = c["single_tap"] + c["single_double"] + c["single_missed"]
    W = c["swipe_tap"] + c["swipe_double"] + c["swipe_missed"]
    pct = lambda a, b: round(100.0 * a / max(b, 1), 2)
    return dict(pairs=P, double_ok=pct(c["pair_double"], P), double_as_two_taps=pct(c["pair_two_taps"], P),
                double_as_one_tap=pct(c["pair_one_tap"], P), double_missed=pct(c["pair_missed"], P),
                singles=S, single_ok=pct(c["single_tap"], S), single_as_double=pct(c["single_double"], S),
                single_missed=pct(c["single_missed"], S),
                swipes=W, swipe_as_tap=pct(c["swipe_tap"], W), swipe_as_double=pct(c["swipe_double"], W),
                swipe_missed=pct(c["swipe_missed"], W),
                movement_tap_per_min=round(c["mov_tap"] / max(c["mov_min"], 1e-9), 2),
                movement_double_per_min=round(c["mov_double"] / max(c["mov_min"], 1e-9), 2),
                false_tap_per_min=round(c["neg_tap"] / max(c["neg_min"], 1e-9), 2),
                false_double_per_min=round(c["neg_double"] / max(c["neg_min"], 1e-9), 2))


def configs(shift):
    """(name, device-scale description, proxy Config)."""
    m = lambda x: to_proxy(x, shift)
    fire = m(DEVICE_FIRE)
    yield "two_plain_onsets", dict(second=DEVICE_FIRE, dip=DEVICE_FIRE), TG.Config(fire=fire, second=fire, dip=fire)
    for second in (0.7, 0.6, 0.5, 0.4, 0.3):
        for dip in (0.1, 0.2, 0.3):
            if dip < second:
                yield f"second{second}_dip{dip}", dict(second=second, dip=dip), TG.Config(fire=fire, second=m(second), dip=m(dip))
    for window in (350, 450):
        yield f"second0.5_dip0.2_win{window}", dict(second=0.5, dip=0.2, window_ms=window), \
            TG.Config(fire=fire, second=m(0.5), dip=m(0.2), window_ms=window)
    yield "second0.5_dip0.2_lock0", dict(second=0.5, dip=0.2, lockout_ms=0), \
        TG.Config(fire=fire, second=m(0.5), dip=m(0.2), lockout_ms=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items, res = predictions(); press = presses()
    fire = G.at_budget(res["pooled"]["nominal"], 2.8)[0]
    shifts = {"fpm_anchor": logit(fire) - logit(DEVICE_FIRE),          # main
              "positive_weight": float(np.log(1.0 / 0.2135))}          # sensitivity: ln(1 / trainer positive weight)
    rows = {}
    for sname, shift in shifts.items():
        for name, dev, cfg in configs(shift):
            key = name if sname == "fpm_anchor" else f"{name}@{sname}"
            if sname != "fpm_anchor" and name not in ("two_plain_onsets", "second0.5_dip0.2", "second0.6_dip0.2"):
                continue
            rows[key] = dict(device=dev, proxy=cfg.__dict__, **summarize(score(items, press, cfg)))
            G.log("TG_CONFIG", key, json.dumps(rows[key]))
    # how often the score does NOT fall below a level between the two taps of a pair (cost of the dip rule)
    dip_cost = {}
    for rec, folder, file, t, p, lab in items:
        if folder != "TAP" or "DoubleTap" not in file:
            continue
        pr = press.get(rec, np.zeros(0))
        for o1, o2 in zip(pr[:-1], pr[1:]):
            if o2 - o1 >= PAIR_MS or o1 < t[0] or o2 > t[-1]:
                continue
            w = (t >= o1) & (t <= o2)
            if w.any():
                m = float(p[w].min())
                for lv in (0.1, 0.2, 0.3):
                    a = dip_cost.setdefault(str(lv), [0, 0]); a[0] += int(m >= to_proxy(lv, shifts["fpm_anchor"])); a[1] += 1
    out = dict(arm=ARM, fire_at_2p8_fpm=fire, shifts=shifts, pair_ms=PAIR_MS, hit_ms=HIT, isolated_ms=ISOLATED_MS,
               configs=rows,
               pairs_without_dip_below={k: round(100.0 * a / b, 2) for k, (a, b) in dip_cost.items()})
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    print("shifts", shifts)
    print(pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk not in ("device", "proxy")} for k, v in rows.items()}).T.to_string())
    print("pairs whose score never fell below the device-scale level between the taps (%):", out["pairs_without_dip_below"])


if __name__ == "__main__":
    main()
