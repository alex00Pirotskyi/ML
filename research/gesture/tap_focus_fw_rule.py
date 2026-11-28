#!/usr/bin/env python3
"""Plain model vs firmware chain rule (tap_gestures.h, immediate + chain) on held-out replay, per arm, at each arm's
threshold for 2.8 false events/min (focus files must exist).
Usage: python research/gesture/tap_focus_fw_rule.py <arm> [<arm> ...]
Device-scale rule thresholds (0.60 / 0.20 / fire 0.85) are mapped to each arm's
research scale by one log-odds shift anchored at that arm's 2.8-fpm threshold."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tap_generalization_research as G, tap_gestures as TG
D = G.load(); data = D["data"]; tr = D["tr"]; recs = D["rpl"]["recs"]
on = D["onsets"]; press = {int(r): np.sort(v.t_on.to_numpy(float)) for r, v in on[on.kind == "press"].groupby("rec")}
arms = sys.argv[1:]
logit = lambda x: float(np.log(x / (1 - x)))
def thr28(a):
    A = G._load_focus(a); j, al = G._at(G._fpm_curve(A, "nominal"), 2.8); return float(G.FOCUS_TH[j] * (1 - al) + G.FOCUS_TH[j + 1] * al)
out = {}
for a in arms:
    fire = thr28(a); sh = logit(fire) - logit(0.85); m = lambda x: 1 / (1 + np.exp(-(logit(x) + sh)))
    cfgs = {"plain": TG.Config(fire=fire, second=fire, dip=fire, immediate=True, chain=True),
            "chain 0.60": TG.Config(fire=fire, second=m(0.6), dip=m(0.2), immediate=True, chain=True)}
    cnt = {c: dict(d2=[0, 0], fast=[0, 0], single=[0, 0], extra=0, mov=0, mov_min=0.0) for c in cfgs}
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
                            if not h or np.any((np.abs(allp - o) < 600) & (np.abs(allp - o) > 1e-6)): continue
                            c["single"][1] += 1; c["single"][0] += int(np.any(~used & (ct > o + 100) & (ct <= o + 700)))
                elif folder in ("FINGER", "WRIST", "ARM", "BAND"):
                    c["mov"] += len(ct); c["mov_min"] += (t[-1] - t[0]) / 6e4
        del val
    out[a] = {cn: {"2nd tap of double clicked %": round(100 * c["d2"][0] / c["d2"][1], 1),
                   "240 bpm taps clicked %": round(100 * c["fast"][0] / c["fast"][1], 1),
                   "single tap + extra click %": round(100 * c["single"][0] / max(c["single"][1], 1), 2),
                   "movement clicks/min": round(c["mov"] / c["mov_min"], 2)} for cn, c in cnt.items()}
    print(a, json.dumps(out[a]), flush=True)
rows = [dict(arm=a, rule=cn, **v) for a, d in out.items() for cn, v in d.items()]
print(pd.DataFrame(rows).to_markdown(index=False))
(G.OUT / "fw_rule_eval.json").write_text(json.dumps(out, indent=1))
