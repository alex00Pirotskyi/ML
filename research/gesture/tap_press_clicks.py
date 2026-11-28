#!/usr/bin/env python3
"""Extra clicks on a tap's own press (release / lead-in) for saved research arms, at matched protection.

The trainer's measure ignores events on unknown rows, so a model that fires again when the finger is lifted is not
counted anywhere. This script replays each arm's saved fold models (file split) at the threshold where fist false
taps reach --fist (per minute, the focus metric), and counts per confirmed tap:
  release clicks: kept events (firmware refractory 150 ms) in the tap's release part (onset + 150 ms to the end of
                  its unknown run, tap_generalization_research.press_regions "rel") that are not the tap's own hit
                  (an earlier event of that tap exists, or it is later than the 250 ms hit window)
  early clicks:   kept events more than 100 ms before the onset in the lead-in part ("pre"): not a hit
Also prints the focus numbers at that threshold (taps per kind, false taps per movement kind).

Usage: MUDRA_GR_SPLIT=file python research/gesture/tap_press_clicks.py [--fist 7] <arm> [<arm> ...]
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay as R                         # noqa: E402
import tap_generalization_research as G   # noqa: E402

args = sys.argv[1:]
fist = 7.0
if args[:1] == ["--fist"]:
    fist = float(args[1]); args = args[2:]


def threshold_for_fist(arm, value):
    """Model threshold where the focus metric fist false taps / min equals value (fine grid, interpolated)."""
    A = G._load_focus(arm); th = np.unique(np.r_[np.linspace(0.3, 0.999, 400), 1 - np.geomspace(1e-3, 3e-2, 80)])
    sys.path.insert(0, "/home/claude/work/fs")
    from summ import at_threshold                          # noqa: E402  (the focus grid interpolation)
    rows = []
    for t in th:
        j, al = at_threshold(t); m = G.focus_metrics(A, j, al)
        rows.append({k.split("|")[1]: v for k, v in m.items() if k.startswith("nominal|")} | {"thr": t})
    c = pd.DataFrame(rows).sort_values("fpm_fist")
    return float(np.interp(value, c.fpm_fist, c.thr)), {k: float(np.interp(value, c.fpm_fist, c[k])) for k in c.columns}


def kept_events(d, t):
    rise = np.flatnonzero(d & ~np.r_[False, d[:-1]]); kept, last = [], -1e18
    for r in rise:
        if t[r] - last >= R.TAP_REFRACTORY_MS:
            kept.append(t[r]); last = t[r]
    return np.asarray(kept)


D = G.load(); data = D["data"]; tr = D["tr"]; rpl = D["rpl"]; recs = rpl["recs"]; on = D["onsets"]
T = np.asarray(rpl["t"])
out = []
for a in args:
    thr, focus = threshold_for_fist(a, fist)
    spec = tuple(json.loads((G.OUT / f"{a}.json").read_text())["config"]["spec"])
    taps = rel_clicks = early_clicks = 0; single = G.press2_recordings(); s_taps = s_rel = 0
    for f, (_x, b) in enumerate(D["folds"]):
        val = G.Val(set(data["group"][tr[b]]))
        p = val.predict(lgb.Booster(model_file=str(G.OUT / "models" / f"{a}_fold{f}.txt")), None, spec, None)
        for (s, e), t, it in zip(val.seg, val.t, val.items):
            rec, folder = int(it[0]), it[3]
            if folder not in ("TAP", "SWIPE"):
                continue
            ev = kept_events(p[s:e] >= thr, np.asarray(t, float))
            conf = on[(on.rec == rec) & (on.kind != "press")].t_on.to_numpy(float)
            taps += len(conf); is_single = rec not in single
            s_taps += len(conf) if is_single else 0
            if not len(ev):
                continue
            rel = T[G.press_regions(rec, ("rel",))]; pre = T[G.press_regions(rec, ("pre",))]
            for region, early in ((rel, False), (pre, True)):
                if not len(region):
                    continue
                # an event counts when it lands on a row of the region (rows are ~7 ms apart)
                j = np.searchsorted(region, ev); j = np.clip(j, 0, len(region) - 1)
                inside = np.abs(region[j] - ev) <= 8
                if early:
                    nearest = conf[np.clip(np.searchsorted(conf, ev), 0, len(conf) - 1)] if len(conf) else ev
                    inside &= (nearest - ev) > R.TAP_HIT_BEFORE_MS
                    early_clicks += int(inside.sum())
                else:                            # not the tap's own (late) hit: an earlier event, or past the window
                    for x in ev[inside]:
                        o = conf[conf < x]
                        if not len(o):
                            continue
                        o = o[-1]
                        if np.any((ev >= o - R.TAP_HIT_BEFORE_MS) & (ev < x)) or x - o > R.TAP_HIT_AFTER_MS:
                            rel_clicks += 1; s_rel += int(is_single)
        del val
        print(a, "fold", f, "done", flush=True)
    out.append(dict(arm=a.replace("raw126_diff+augm+litediff+press2+", ""), thr=round(thr, 4),
                    normal=round(100 * focus["normal"], 1), soft=round(100 * focus["soft"], 1),
                    double_2nd=round(100 * focus["double_2nd"], 1), fast=round(100 * focus["fast240"], 1),
                    swipe=round(100 * focus["swipe"], 1), finger=round(focus["fpm_finger"], 2),
                    fist=round(focus["fpm_fist"], 2), wrist=round(focus["fpm_wrist"], 2), band=round(focus["fpm_band"], 2),
                    release_clicks_per_100_taps=round(100 * rel_clicks / max(taps, 1), 2),
                    release_clicks_single_tap_recordings=round(100 * s_rel / max(s_taps, 1), 2),
                    early_clicks_per_100_taps=round(100 * early_clicks / max(taps, 1), 2)))
print(f"matched at fist {fist:g} false taps / min")
print(pd.DataFrame(out).to_markdown(index=False))
(G.OUT / f"press_clicks_fist{fist:g}.json").write_text(json.dumps(out, indent=1))
