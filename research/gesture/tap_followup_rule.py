#!/usr/bin/env python3
"""Follow-up tap rule (firmware/tap_gestures.h, immediate + chain) for a strict TAP model.

A model trained with a logit shift s (MUDRA_TAP_LOGIT_SHIFT) fires at 0.85 on the device scale, which is a
probability of sigmoid(logit(0.85) + s) of the fitted model. The rule's second-tap and dip thresholds are on the
same device scale, so the shift makes them stricter too (0.60 -> 0.968 of the fitted model at s = 3). This script
replays each arm's saved fold models at the threshold of a trainer false-tap target (default 1.5/min) and
compares device-scale second / dip settings.

Usage: python research/gesture/tap_followup_rule.py [--target 1.5] <arm> [<arm> ...]
Rows: 2nd tap of a double clicked, 240 bpm taps clicked, single tap + extra click, movement clicks per minute
(finger / wrist / arm / band recordings, every click counted).
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tap_generalization_research as G   # noqa: E402
import tap_gestures as TG                 # noqa: E402

args = sys.argv[1:]
target = 1.5
if args[:1] == ["--target"]:
    target = float(args[1]); args = args[2:]
logit = lambda x: float(np.log(x / (1 - x)))
sig = lambda z: float(1 / (1 + np.exp(-z)))
# device-scale (second, dip): firmware default first; Q20 steps of 0.05
SETTINGS = [("no follow-up rule", None), ("0.60 / 0.20 (now)", (0.60, 0.20)), ("0.40 / 0.10", (0.40, 0.10)),
            ("0.25 / 0.05", (0.25, 0.05)), ("0.15 / 0.05", (0.15, 0.05)), ("0.10 / 0.05", (0.10, 0.05))]


def threshold_at(arm, X):
    nom = json.loads((G.OUT / f"{arm}.json").read_text())["pooled"]["nominal"]
    th = sorted(nom, key=float); f = np.array([nom[t]["false_per_min"] for t in th]); t_ = np.array([float(t) for t in th])
    for j in range(len(th) - 1):
        if f[j] >= X >= f[j + 1]:
            al = (np.log(f[j]) - np.log(X)) / (np.log(f[j]) - np.log(f[j + 1]))
            return float(t_[j] + al * (t_[j + 1] - t_[j]))
    raise ValueError(f"{arm}: target {X} outside the grid")


D = G.load(); data = D["data"]; tr = D["tr"]; recs = D["rpl"]["recs"]
on = D["onsets"]; press = {int(r): np.sort(v.t_on.to_numpy(float)) for r, v in on[on.kind == "press"].groupby("rec")}
out = []
for a in args:
    fire = threshold_at(a, target); sh = logit(fire) - logit(0.85)
    m = lambda x: sig(logit(x) + sh)
    cfgs = {name: TG.Config(fire=fire, second=fire if v is None else m(v[0]), dip=fire if v is None else m(v[1]),
                            immediate=True, chain=True) for name, v in SETTINGS}
    cnt = {c: dict(d2=[0, 0], fast=[0, 0], single=[0, 0], mov=0, mov_min=0.0) for c in cfgs}
    spec = tuple(json.loads((G.OUT / f"{a}.json").read_text())["config"]["spec"])
    for f, (_x, b) in enumerate(D["folds"]):
        val = G.Val(set(data["group"][tr[b]]))
        p = val.predict(lgb.Booster(model_file=str(G.OUT / "models" / f"{a}_fold{f}.txt")), None, spec, None)
        for (s, e), t, lab, it in zip(val.seg, val.t, val.label, val.items):
            rec, folder = it[0], it[3]; typ = G.rec_type(folder, recs.file.iloc[rec])
            for cn, cfg in cfgs.items():
                g, _ = TG.simulate(p[s:e], t, cfg); ct = np.array([x[1] for x in g]); c = cnt[cn]
                if folder in ("TAP", "SWIPE"):
                    pr = press.get(int(rec), np.zeros(0)); pr = pr[(pr >= t[0] + 100) & (pr <= t[-1] - 600)]
                    hit, used = G._match(ct, pr); gap = np.diff(np.r_[-1e18, pr])
                    if typ == "double":
                        sel = gap < 500; c["d2"][0] += int(hit[sel].sum()); c["d2"][1] += int(sel.sum())
                    elif typ == "fast240":
                        c["fast"][0] += int(hit.sum()); c["fast"][1] += len(hit)
                    elif typ in ("normal", "soft"):
                        allp = press.get(int(rec), np.zeros(0))
                        for o, h in zip(pr, hit):
                            if not h or np.any((np.abs(allp - o) < 600) & (np.abs(allp - o) > 1e-6)):
                                continue
                            c["single"][1] += 1; c["single"][0] += int(np.any(~used & (ct > o + 100) & (ct <= o + 700)))
                elif folder in ("FINGER", "WRIST", "ARM", "BAND"):
                    c["mov"] += len(ct); c["mov_min"] += (t[-1] - t[0]) / 6e4
        del val
        print(a, "fold", f, "done", flush=True)
    for cn, c in cnt.items():
        out.append(dict(arm=a.replace("raw126_diff+augm+litediff+press2+", ""), rule=cn,
                        model_second=None if dict(SETTINGS)[cn] is None else round(m(dict(SETTINGS)[cn][0]), 4),
                        d2=round(100 * c["d2"][0] / c["d2"][1], 1), fast=round(100 * c["fast"][0] / c["fast"][1], 1),
                        single_extra=round(100 * c["single"][0] / max(c["single"][1], 1), 2),
                        movement_clicks_per_min=round(c["mov"] / c["mov_min"], 2)))
print(f"fire threshold at trainer {target}/min; device-scale second / dip")
print(pd.DataFrame(out).to_markdown(index=False))
(G.OUT / f"followup_rule_target{target:g}.json").write_text(json.dumps(out, indent=1))
