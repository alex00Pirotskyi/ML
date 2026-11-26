"""Stricter firmware/tap_gestures.h thresholds on held-out replay (2026-11-26): TG_FIRE_Q20 x TG_SECOND_Q20.

Scores: the TAP research fold models of tap_generalization_research.py (file folds, gesture-aware: each fold model
scores only its own validation files), e.g. the top-label arm "raw126_diff+augm+litediff+press2+kb14+leaves7+top+topb10"
(make train's final14top data recipe) and the onset arm "raw126_diff+augm+litediff+press2+kb14+leaves7" (final14).
Device scale: one log-odds shift, anchored so that device 0.85 is where the research model has a given pooled
false-tap rate (4.4/min = the band baseline header's held-out rate; 2.8 and 6.2 as sensitivity), then Q20 exactly
as the firmware (replay.q20). View "lower1": every score 1 logit lower (the band failure: live scores below the
recordings'). The rule is the firmware's, via its bit-exact host twin tap_gestures.py.

Truth: every real pointer press. Double pairs: consecutive presses < 500 ms apart in DoubleTaps recordings. Singles
and swipes: presses with no other press within 600 ms. Movement: FINGER (fist = Fist & Stretch), WRIST, ARM, BAND
recordings, every gesture is false. HID chain mode (immediate + chain): 240 bpm taps clicked, isolated single taps
with an extra click within 400 ms, movement clicks per minute.

    MUDRA_GR_CACHE=<research cache> MUDRA_GR_OUT=<research results with models/> \\
        python research/gesture/tap_gestures_strict.py <arm> [<arm> ...]     # results_tap_gestures/strict_2026-11-26.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay as R                         # noqa: E402
import tap_generalization_research as G   # noqa: E402
import tap_gestures as TG                 # noqa: E402

OUT = HERE / "results_tap_gestures"
PAIR_MS, HIT, ISOLATED_MS = 500, (100, 250), 600
MOVEMENT = ("finger", "fist", "wrist", "arm", "band")
LONG_PRESS = ("SOFT_LONG_PRESS", "MID_LONG_PRESS", "STRONG_LONG_PRESS", "LONG_PRESS", "DRAG_AND_DROP")


def logit(x):
    x = min(max(float(x), 1e-9), 1 - 1e-9)
    return float(np.log(x / (1 - x)))


def device_q20(p, shift, lower=0.0):
    p = np.clip(np.asarray(p, np.float64), 1e-9, 1 - 1e-9)
    return R.q20(1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) - shift - lower)))).astype(np.int16)


def predictions(arm):
    """[(rec, folder, file, t, p, label)] for every validation recording of every fold, and the arm's pooled curve."""
    import lightgbm as lgb
    cache = G.OUT / f"tg_pred_{arm}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    D = G.load(); data = D["data"]; tr = D["tr"]; recs = D["rpl"]["recs"]
    res = json.loads((G.OUT / f"{arm}.json").read_text()); spec = tuple(res["config"]["spec"])
    items = []
    for f, (_a, b) in enumerate(D["folds"]):
        val = G.Val(set(data["group"][tr[b]]))
        p = val.predict(lgb.Booster(model_file=str(G.OUT / "models" / f"{arm}_fold{f}.txt")), None, spec)
        for (s, e), t, lab, it in zip(val.seg, val.t, val.label, val.items):
            items.append((it[0], recs.folder.iloc[it[0]], recs.file.iloc[it[0]], t, p[s:e].astype(np.float32), lab))
        del val
    out = dict(items=items, pooled=res["pooled"])
    pickle.dump(out, open(cache, "wb"), protocol=4)
    return out


def presses():
    on = pd.read_csv(G.CACHE / "tap_onsets.csv")
    return {r: np.sort(g.t_on.values.astype(np.float64)) for r, g in on[on.kind == "press"].groupby("rec")}


def kind_of(folder, file):
    if folder == "FINGER":
        return "fist" if "fist" in file.lower() else "finger"
    return "longpress" if folder in LONG_PRESS else folder.lower()


def deferred(items, press, cfg):
    """Deferred mode (the default): outcomes of pairs, singles and swipes, false gestures per minute."""
    c = dict(pair_double=0, pair_two=0, pair_one=0, pair_miss=0, single_tap=0, single_double=0, single_miss=0,
             swipe_tap=0, swipe_double=0, swipe_miss=0)
    false, mins = {}, {}
    for rec, folder, file, t, q, lab in items:
        gest, _ = TG.simulate(q, t, cfg)
        kind = np.array([g[0] for g in gest], np.int8); t1 = np.array([g[1] for g in gest], np.float64)
        used = np.zeros(len(gest), bool)

        def take(o):
            i = np.flatnonzero(~used & (t1 >= o - HIT[0]) & (t1 <= o + HIT[1]))
            if len(i):
                used[i[0]] = True
                return int(i[0])
            return None

        dt = np.clip(np.diff(np.r_[t, t[-1]]), 0, 50)
        if folder in ("TAP", "SWIPE"):
            allp = press.get(rec, np.zeros(0))
            pr = allp[(allp >= t[0] + HIT[0]) & (allp <= t[-1] - cfg.window_ms - HIT[1])]
            if folder == "TAP" and "DoubleTap" in file:
                k = 0
                while k < len(pr):
                    if k + 1 < len(pr) and pr[k + 1] - pr[k] < PAIR_MS:
                        i = take(pr[k])
                        if i is not None and kind[i] == TG.DOUBLE_TAP:
                            c["pair_double"] += 1
                        elif i is not None:
                            c["pair_two" if take(pr[k + 1]) is not None else "pair_one"] += 1
                        else:
                            j = take(pr[k + 1])
                            if j is not None and kind[j] == TG.DOUBLE_TAP:
                                c["pair_double"] += 1
                            else:
                                c["pair_one" if j is not None else "pair_miss"] += 1
                        k += 2
                    else:
                        take(pr[k]); k += 1
            else:
                name = "single" if folder == "TAP" else "swipe"
                for o in pr:
                    i = take(o)
                    d = np.abs(allp - o)
                    if ((d < ISOLATED_MS) & (d > 1e-6)).any():
                        continue
                    c[f"{name}_miss" if i is None else (f"{name}_double" if kind[i] == TG.DOUBLE_TAP else f"{name}_tap")] += 1
        else:
            k = kind_of(folder, file)
            false[k] = false.get(k, 0) + len(gest); mins[k] = mins.get(k, 0.0) + float(dt.sum()) / 6e4
            false[k + "_double"] = false.get(k + "_double", 0) + int((kind == TG.DOUBLE_TAP).sum())
    P = c["pair_double"] + c["pair_two"] + c["pair_one"] + c["pair_miss"]
    S = c["single_tap"] + c["single_double"] + c["single_miss"]
    W = c["swipe_tap"] + c["swipe_double"] + c["swipe_miss"]
    mv_min = sum(mins.get(k, 0.0) for k in MOVEMENT)
    r = dict(single_tap=100 * c["single_tap"] / S, single_as_double=100 * c["single_double"] / S,
             double=100 * c["pair_double"] / P, double_as_one_tap=100 * c["pair_one"] / P,
             swipe_tap=100 * c["swipe_tap"] / W,
             movement_false_per_min=sum(false.get(k, 0) for k in MOVEMENT) / mv_min,
             movement_double_per_min=sum(false.get(k + "_double", 0) for k in MOVEMENT) / mv_min,
             longpress_recordings_per_min=false.get("longpress", 0) / max(mins.get("longpress", 0.0), 1e-9),
             **{f"{k}_per_min": false.get(k, 0) / mins[k] for k in MOVEMENT})
    return {k: round(v, 2) for k, v in r.items()}, dict(pairs=P, singles=S, swipes=W, movement_minutes=round(mv_min, 1))


def chain(items, press, cfg):
    """Immediate + chain mode (HID clicks)."""
    fast_ok = fast_n = extra = singles = 0; mov = mov_min = 0.0
    for rec, folder, file, t, q, lab in items:
        clicks = np.array([g[1] for g in TG.simulate(q, t, cfg)[0]], np.float64)
        if folder == "TAP":
            allp = press.get(rec, np.zeros(0))
            pr = allp[(allp >= t[0] + HIT[0]) & (allp <= t[-1] - 650)]
            if "_240bpm" in file.lower():
                used = np.zeros(len(clicks), bool)
                for o in pr:
                    i = np.flatnonzero(~used & (clicks >= o - HIT[0]) & (clicks <= o + HIT[1]))
                    fast_n += 1
                    if len(i):
                        used[i[0]] = True; fast_ok += 1
            elif "DoubleTap" not in file:
                for o in pr:
                    d = np.abs(allp - o)
                    if ((d < ISOLATED_MS) & (d > 1e-6)).any():
                        continue
                    singles += 1; extra += int(((clicks >= o - HIT[0]) & (clicks <= o + 400)).sum()) >= 2
        elif kind_of(folder, file) in MOVEMENT:
            mov += len(clicks); mov_min += float(np.clip(np.diff(np.r_[t, t[-1]]), 0, 50).sum()) / 6e4
    return dict(fast_taps_clicked=round(100 * fast_ok / fast_n, 2), single_with_extra_click=round(100 * extra / singles, 2),
                movement_clicks_per_min=round(mov / mov_min, 2), fast_taps=fast_n, singles=singles)


def run(arm, press):
    P = predictions(arm)
    out = {}
    for anchor in (4.4, 2.8, 6.2):
        fire_r = G.at_budget(P["pooled"]["nominal"], anchor)[0]; shift = logit(fire_r) - logit(0.85)
        views = {"nominal": 0.0, "lower1": 1.0} if anchor == 4.4 else {"nominal": 0.0}
        grid = [(f, s) for f in (17, 18, 19) for s in (12, 14, 15, 16, 17) if s <= f] if anchor == 4.4 else \
            [(17, 12), (18, 15), (19, 15)]
        for view, lower in views.items():
            items = [(rec, folder, file, t, device_q20(p, shift, lower), lab) for rec, folder, file, t, p, lab in P["items"]]
            for f, s in grid:
                r, n = deferred(items, press, TG.Config(fire=f, second=s))
                out[f"anchor{anchor}|{view}|deferred|fire{f}_second{s}"] = r
                G.log("TG_STRICT", arm, anchor, view, f, s, json.dumps(r))
            if anchor == 4.4:
                for f, s in ((17, 12), (18, 12), (18, 15), (19, 12)):
                    r = chain(items, press, TG.Config(fire=f, second=s, immediate=True, chain=True))
                    out[f"anchor{anchor}|{view}|chain|fire{f}_second{s}"] = r
                    G.log("TG_STRICT", arm, anchor, view, "chain", f, s, json.dumps(r))
        out[f"anchor{anchor}|shift"] = round(shift, 4)
    out["counts"] = n
    return out


def main():
    arms = sys.argv[1:] or ["raw126_diff+augm+litediff+press2+kb14+leaves7+top+topb10",
                            "raw126_diff+augm+litediff+press2+kb14+leaves7"]
    press = presses()
    results = {arm: run(arm, press) for arm in arms}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "strict_2026-11-26.json").write_text(json.dumps(dict(arms=results, pair_ms=PAIR_MS, hit_ms=HIT,
                                                                isolated_ms=ISOLATED_MS), indent=1))
    print("written", OUT / "strict_2026-11-26.json")


if __name__ == "__main__":
    main()
