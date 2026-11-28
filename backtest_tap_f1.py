#!/usr/bin/env python3
"""Back test of the TAP F1 models: the held-out recordings replayed through what the device runs.

Every saved model of TRAIN_FOLDER (train_tap_f1.py: <tag>_model.pkl / _info.json / _model.h) scores every row of every
recording of the held-out fold (DATA/DATA_TAP_F1, fold 0, never trained on, never augmented). The scores then go
through the firmware's tap logic, exactly as tap_gestures.c runs it (host twin: research/gesture/tap_gestures.py,
deferred mode): a tap starts on a rising edge of score >= FIRE, 150 ms refractory; a 2nd tap inside 150..400 ms after
a dip below 0.20 and a score >= SECOND (= FIRE - 0.15) makes it a DOUBLE_TAP, else it is reported as a TAP.

Measured without any label (the pointer is the reference):
  taps      share of pointer presses of the TAP recordings matched by a tap onset -100..+250 ms (per kind: normal,
            soft, 240 bpm, slow pinch, 1st / 2nd tap of a double; swipes reported apart)
  doubles   double-tap pairs reported as DOUBLE_TAP; single taps wrongly reported as DOUBLE_TAP
  false     TAP / DOUBLE_TAP events per minute of the no-tap recordings, per kind (finger, fist, wrist, arm, band)

Choice (written to backtest_choice.json, read by export_tap_bundle.py):
  1. per model, FIRE in 0.60..0.95 (0.05 steps) maximizing  U = taps (%) - LAMBDA x false events per minute, the
     false rate being the mean over the movement kinds (fist, finger, wrist, arm, band, nothing), fist and finger
     counting twice (they are the top priority), so a kind counts the same however long it was recorded.
     LAMBDA = 2 points of taps per false event per minute (BACKTEST_LAMBDA), a little stricter than the trades the
     device settings were chosen with (0.90 over 0.85: ~1.5; 0.95 not taken at ~1.9, research/gesture/TAP_GESTURES.md).
     Then the one-standard-error rule: the highest FIRE whose U is within BACKTEST_SE (1) standard errors of the best
     U is taken, so thin held-out evidence (few fist minutes: one event = several per minute) leans to the cautious
     side, the top priority, while a large held-out fold takes the FIRE of the best U. The standard error counts the
     presses as binomial and every movement kind's false events as Poisson, a kind with no event as one event
     (it is not known to be zero). BACKTEST_SE=0: the FIRE of the best U.
  2. every model is replayed again under band changes (tap_f1_augment.stress: placement = gain + tilt + contact
     noise, power-grid hum in each zone, band rotation; BACKTEST_STRESS_TOP=N: only the N best clean models), and the
     FIRE of step 1 is taken on the mean of U over clean, placement, hum (averaged over the zones) and rotation.
  3. the model with the best of that U wins; the models within BACKTEST_SE standard errors of it are as good (the
     held-out fold cannot tell them apart), and among them XGBoost goes first (BACKTEST_PREFER: it did better on the
     device; empty = no preference), then the smallest model (flash, cycles).

    python backtest_tap_f1.py [TRAIN_FOLDER] [--data DATA/DATA_TAP_F1] [--workers N]
    -> TRAIN_FOLDER/backtest.md, backtest.json, backtest_choice.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "research" / "gesture")]
import evaluate_tap_f1 as EV  # noqa: E402
import tap_gestures as TG  # noqa: E402  (host twin of the firmware tap logic)

FIRES = tuple(range(12, 20))           # Q20: 0.60 .. 0.95
SECOND_BELOW_FIRE = 3                  # SECOND = FIRE - 0.15 (TAP_GESTURES.md: 0.90 / 0.75)
DIP, MIN_GAP_MS, WINDOW_MS, LOCKOUT_MS = 4, 150.0, 400.0, 300.0
LAMBDA = float(os.environ.get("BACKTEST_LAMBDA", "2.0"))
KIND_WEIGHT = dict(fist=2.0, finger=2.0, wrist=1.0, arm=1.0, band=1.0, nothing=1.0)
STRESS_TOP = os.environ.get("BACKTEST_STRESS_TOP", "all")    # models replayed under band changes: all, or the N best
SE_RULE = float(os.environ.get("BACKTEST_SE", "1.0"))         # FIRE and model: within this many SE of the best U
PREFER = os.environ.get("BACKTEST_PREFER", "XGBoost")          # model type first among the models as good as the best
TAP_KINDS = ("normal", "soft", "fast240", "slowpinch", "double_1st", "double_2nd")


def config(fire):
    return TG.Config(fire=fire, second=max(fire - SECOND_BELOW_FIRE, DIP + 1), dip=DIP, min_gap_ms=MIN_GAP_MS,
                     window_ms=WINDOW_MS, lockout_ms=LOCKOUT_MS, immediate=False, chain=False)


def levels(p):
    """Q20 level of a probability: level >= k exactly when p >= k / 20 (what the C asks the trees)"""
    return np.clip(np.floor(np.asarray(p, np.float64) * 20.0 + 1e-9), 0, 20).astype(np.int16)


# ================================================================================================ one recording
def replay(folder, rel, t, ptr, q, fires=FIRES):
    """counts of one recording for every FIRE: dict fire -> {name: number}"""
    t = np.asarray(t, np.float64); t = t - t[0]
    kind = EV.rec_kind(folder, rel)
    minutes = float(t[-1]) / 6e4 if len(t) > 1 else 0.0
    out = {}
    if folder in ("TAP", "SWIPE"):
        pr = EV.presses(t, ptr)
        pr = pr[(pr >= EV.HIT_BEFORE_MS) & (pr <= t[-1] - EV.HIT_AFTER_MS)]
        gap = np.diff(np.r_[-1e18, pr])
        kinds = np.array([(("double_2nd" if g < EV.PAIR_MS else "double_1st") if kind == "double" else kind) for g in gap])
    for f in fires:
        gestures, _ = TG.simulate(q, t, config(f))
        c = {}
        if folder in ("TAP", "SWIPE"):
            onsets = np.sort(np.array([g[1] for g in gestures] + [g[2] for g in gestures if g[0] == TG.DOUBLE_TAP]))
            hit, used = EV.match(onsets, pr)
            for k in set(kinds):
                c[f"n|{k}"] = int((kinds == k).sum()); c[f"hit|{k}"] = int(hit[kinds == k].sum())
            c["min|taprec"] = minutes; c["extra|taprec"] = int((~used).sum())
            dbl = [g for g in gestures if g[0] == TG.DOUBLE_TAP]
            if kind == "double":                       # pairs: 1st press + the next press within PAIR_MS
                i1 = np.flatnonzero(kinds == "double_1st")
                pairs = [(pr[i], pr[i + 1]) for i in i1 if i + 1 < len(pr) and kinds[i + 1] == "double_2nd"]
                ok = sum(any(-EV.HIT_BEFORE_MS <= g[1] - a <= EV.HIT_AFTER_MS and
                             -EV.HIT_BEFORE_MS <= g[2] - b <= EV.HIT_AFTER_MS for g in dbl) for a, b in pairs)
                c["n|pair"] = len(pairs); c["hit|pair"] = int(ok)
            elif kind in ("normal", "soft", "slowpinch"):          # single taps that came out as a double
                c["n|single"] = int(len(pr))
                c["dbl|single"] = int(sum(any(-EV.HIT_BEFORE_MS <= g[1] - a <= EV.HIT_AFTER_MS for g in dbl) for a in pr))
        elif kind in EV.NEG_KINDS:
            c[f"min|{kind}"] = minutes
            c[f"false|{kind}"] = len(gestures)
            c[f"falsedbl|{kind}"] = sum(g[0] == TG.DOUBLE_TAP for g in gestures)
        out[f] = c
    return out


def add(total, part):
    for f, c in part.items():
        d = total.setdefault(f, {})
        for k, v in c.items():
            d[k] = d.get(k, 0) + v
    return total


def summarize(c):
    """metrics of one model at one FIRE from summed counts"""
    g = lambda k: c.get(k, 0)
    tap_n = sum(g(f"n|{k}") for k in TAP_KINDS)
    m = dict(taps=sum(g(f"hit|{k}") for k in TAP_KINDS) / max(tap_n, 1), presses=int(tap_n))
    for k in TAP_KINDS + ("swipe",):
        m[k] = g(f"hit|{k}") / max(g(f"n|{k}"), 1)
    m["double_pairs"] = g("hit|pair") / max(g("n|pair"), 1)
    m["single_as_double"] = g("dbl|single") / max(g("n|single"), 1)
    mins = sum(g(f"min|{k}") for k in EV.NEG_KINDS)
    m["false_pm"] = sum(g(f"false|{k}") for k in EV.NEG_KINDS) / max(mins, 1e-9)
    m["false_double_pm"] = sum(g(f"falsedbl|{k}") for k in EV.NEG_KINDS) / max(mins, 1e-9)
    for k in EV.NEG_KINDS:
        m[f"false_{k}_pm"] = g(f"false|{k}") / max(g(f"min|{k}"), 1e-9)
    seen = [k for k in EV.NEG_KINDS if g(f"min|{k}") > 0]          # every movement kind counts by its weight, not
    m["false_weighted_pm"] = (sum(KIND_WEIGHT[k] * m[f"false_{k}_pm"] for k in seen) /   # by how long it was recorded
                              max(sum(KIND_WEIGHT[k] for k in seen), 1e-9))
    m["minutes_no_tap"] = mins
    m["double_pairs_n"] = int(g("n|pair")); m["singles_n"] = int(g("n|single"))
    for k in ("fist", "finger"):
        m[f"minutes_{k}"] = g(f"min|{k}")
    m["U"] = 100.0 * m["taps"] - LAMBDA * m["false_weighted_pm"]
    hits = sum(g(f"hit|{k}") for k in TAP_KINDS)
    pt = (hits + 1.0) / (tap_n + 2.0)                                  # binomial presses (add-one: never 0 or 1)
    var_taps = 1e4 * pt * (1.0 - pt) / max(tap_n, 1)
    w_all = max(sum(KIND_WEIGHT[k] for k in seen), 1e-9)               # Poisson false events, >= 1 event per kind
    var_false = sum((KIND_WEIGHT[k] / w_all) ** 2 * max(g(f"false|{k}"), 1) / g(f"min|{k}") ** 2 for k in seen)
    m["U_se"] = float(np.sqrt(var_taps + LAMBDA ** 2 * var_false))
    return m


def pick_fire(curve, se_rule=None):
    """(chosen FIRE, FIRE of the best U): the highest FIRE whose U is within se_rule standard errors of the best U"""
    se_rule = SE_RULE if se_rule is None else se_rule
    top = max(curve, key=lambda f: (round(curve[f]["U"], 9), f))
    bar = curve[top]["U"] - se_rule * curve[top]["U_se"] - 1e-9
    return max(f for f in curve if curve[f]["U"] >= bar), top


# ================================================================================================ models
def load_models(train_folder):
    import joblib
    models = []
    for p in sorted(Path(train_folder).glob("*_model.pkl")):
        info_p = p.with_name(p.name.replace("_model.pkl", "_info.json"))
        info = json.loads(info_p.read_text()) if info_p.exists() else {}
        h = p.with_name(p.name.replace("_model.pkl", "_model.h"))
        models.append(dict(tag=p.name.replace("_model.pkl", ""), container=joblib.load(p), info=info,
                           header=str(h) if h.exists() else None, bytes=info.get("model_size_bytes")))
    if not models:
        raise SystemExit(f"no *_model.pkl in {train_folder}")
    return models


def schema_names(train_folder, models):
    sp = Path(models[0]["info"].get("schema_file") or "")
    sp = sp if sp.is_file() else Path(train_folder) / "TAP_feature_schema.json"
    return list(json.loads(sp.read_text())["feature_names"])


def test_files(data, fold, roadmap):
    from fold_split import load_roadmap
    client = load_roadmap(roadmap, ROOT / "DATA")
    files = [p for p in sorted(Path(data).rglob("*.csv")) if not any(x.startswith("_") for x in p.relative_to(data).parts[:-1])]
    out = []
    for p in files:
        info = client.get_file_info(p, "DATA_TAP")
        if info.fold_id == int(fold) and not info.exclude_from_training:
            out.append(p)
    return out


# ------------------------------------------------------------------ workers (module level: picklable)
_W = {}


def _init(train_folder, names, fires):
    _W["models"] = load_models(train_folder)
    _W["names"] = names
    _W["fires"] = fires


def _nominal_job(path):
    p = Path(path)
    df = pd.read_csv(p, usecols=_W["names"] + ["timestamp", "PointerState"])
    X = df[_W["names"]].to_numpy(np.float32)
    t = df["timestamp"].to_numpy(np.float64); ptr = df["PointerState"].to_numpy() > 0
    return [replay(p.parent.name, p.name, t, ptr, levels(EV._predict(m["container"], X)), _W["fires"])
            for m in _W["models"]]


def _stress_job(job):
    """one held-out recording under every band change (read once), scored by the picked models at every FIRE:
    {kind: [counts per pick]}"""
    import build_tap_f1_dataset as BUILD
    import tap_f1_augment as AUG
    import tap_f1_features as F
    rel, picks = job                          # picks: [model index]
    src = BUILD._source(rel)
    df = pd.read_csv(src, low_memory=False)
    raw_cols = [c for c in df.columns if c.startswith("p_core->")]
    B0 = df[raw_cols].to_numpy(np.float32)
    idx = AUG.indices(raw_cols)
    fr = None
    if F.needs_lite(_W["names"]):
        lp = BUILD._source(rel.replace("DATA_TAP/", "DATA_TAP_LEAN8/", 1))
        fr = pd.read_csv(lp, usecols=["lean8_frames"]).lean8_frames.to_numpy()
    fm = F.FeatureMaker(raw_cols + (F.LITE_COLUMNS if fr is not None else []), _W["names"])
    t = df["Timestamp"].to_numpy(np.float64); ptr = df["PointerState"].to_numpy() > 0
    out = {}
    for kind in AUG.STRESS_KINDS:
        B = AUG.apply(B0, raw_cols, AUG.stress(Path(rel).name, kind), idx)
        if fr is not None:
            B = np.hstack([B, F.lite_rows(B, raw_cols, fr)])
        X = fm(B)
        out[kind] = [replay(src.parent.name, src.name, t, ptr, levels(EV._predict(_W["models"][i]["container"], X)),
                            _W["fires"]) for i in picks]
    return out


def _ctx():
    import multiprocessing as mp
    return mp.get_context("forkserver") if "forkserver" in mp.get_all_start_methods() else None


# ================================================================================================ the back test
def backtest(train_folder, data=ROOT / "DATA" / "DATA_TAP_F1", fold=0, roadmap=ROOT / "DATA" / "fold_roadmap_files.json",
             workers=None, stress_top=STRESS_TOP, log=print):
    t0 = time.time()
    train_folder, data = Path(train_folder), Path(data)
    workers = int(workers or os.environ.get("TAP_F1_WORKERS") or max(1, (os.cpu_count() or 2) - 1))
    models = load_models(train_folder)
    names = schema_names(train_folder, models)
    files = test_files(data, fold, roadmap)
    log(f"[backtest] {len(models)} models x {len(files)} held-out recordings (fold {fold}), FIRE "
        f"{FIRES[0] / 20:.2f}..{FIRES[-1] / 20:.2f}, LAMBDA {LAMBDA:g}, {SE_RULE:g} SE rule")
    totals = [dict() for _ in models]
    with ProcessPoolExecutor(workers, mp_context=_ctx(), initializer=_init, initargs=(str(train_folder), names, FIRES)) as ex:
        for i, parts in enumerate(ex.map(_nominal_job, [str(p) for p in files], chunksize=4), 1):
            for tot, part in zip(totals, parts):
                add(tot, part)
            if i % 200 == 0:
                log(f"  {i}/{len(files)} recordings, {time.time() - t0:.0f} s")
    res = []
    for m, tot in zip(models, totals):
        curve = {f: summarize(tot.get(f, {})) for f in FIRES}
        res.append(dict(tag=m["tag"], model_type=m["info"].get("model_type"), bytes=m["bytes"], header=m["header"],
                        curve=curve, counts=tot, stress={}, robust=None))
    order = sorted(range(len(res)), key=lambda i: (-max(c["U"] for c in res[i]["curve"].values()), res[i]["bytes"] or 1e12))
    # ---- band changes: every FIRE of the models with a header (BACKTEST_STRESS_TOP: only the N best clean ones)
    top = [i for i in order if res[i]["header"]][:len(res) if stress_top in (None, "all") else max(0, int(stress_top))]
    if top:
        import tap_f1_augment as AUG
        rels = [f"DATA_TAP/{p.parent.name}/{p.name}" for p in files]
        log(f"[backtest] band changes for {len(top)} models: {', '.join(AUG.STRESS_KINDS)}")
        tot = {kind: [dict() for _ in top] for kind in AUG.STRESS_KINDS}
        with ProcessPoolExecutor(workers, mp_context=_ctx(), initializer=_init,
                                 initargs=(str(train_folder), names, FIRES)) as ex:
            for j, parts in enumerate(ex.map(_stress_job, [(rel, top) for rel in rels], chunksize=4), 1):
                for kind, per_model in parts.items():
                    for k, part in enumerate(per_model):
                        add(tot[kind][k], part)
                if j % 200 == 0:
                    log(f"  {j}/{len(rels)} recordings, {time.time() - t0:.0f} s")
        for k, i in enumerate(top):
            r = res[i]
            sc = {kind: {f: summarize(tot[kind][k].get(f, {})) for f in FIRES} for kind in AUG.STRESS_KINDS}
            hum = [kind for kind in AUG.STRESS_KINDS if kind.startswith("hum")]
            mean = lambda f, key: float(np.mean([r["curve"][f][key], sc["placement"][f][key],
                                                 np.mean([sc[h][f][key] for h in hum]), sc["rotation"][f][key]]))
            r["robust"] = {f: dict(U=mean(f, "U"), U_se=mean(f, "U_se")) for f in FIRES}
            r["stress_curve"] = sc
    for r in res:                    # FIRE: the highest within SE_RULE standard errors of the best (robust) U
        r["fire"], r["fire_best_u"] = pick_fire(r["robust"] or r["curve"])
        r["second"] = config(r["fire"]).second
        r["nominal"] = r["curve"][r["fire"]]
        if r["robust"]:
            r["stress"] = {kind: c[r["fire"]] for kind, c in r.pop("stress_curve").items()}
            r["U_final"], r["U_best"] = r["robust"][r["fire"]]["U"], r["robust"][r["fire_best_u"]]["U"]
        else:
            r["U_final"], r["U_best"] = None, r["curve"][r["fire_best_u"]]["U"]
    # ---- model: the best U with band changes; the models within SE_RULE standard errors of it are as good (the
    # held-out fold cannot tell them apart): XGBoost first (BACKTEST_PREFER, it did better on the device), then the
    # smallest (flash, cycles)
    cands = [r for r in res if r["robust"]] or [r for r in res if r["header"]] or res
    lead = max(cands, key=lambda r: (r["U_best"], -(r["bytes"] or 1e12)))
    se = (lead["robust"] or lead["curve"])[lead["fire_best_u"]]["U_se"]
    near = [r for r in cands if r["U_best"] >= lead["U_best"] - SE_RULE * se - 1e-9]
    win = min(near, key=lambda r: (bool(PREFER) and r["model_type"] != PREFER, r["bytes"] or 1e12, -r["U_best"]))
    for r in res:
        r["tie"] = r in near
    choice = dict(model=win["tag"], header=win["header"], fire_q20=int(win["fire"]), second_q20=int(win["second"]),
                  dip_q20=DIP, min_gap_ms=MIN_GAP_MS, window_ms=WINDOW_MS, lockout_ms=LOCKOUT_MS, lambda_=LAMBDA,
                  se_rule=SE_RULE, prefer=PREFER, fire_best_u_q20=int(win["fire_best_u"]), U_best=win["U_best"],
                  nominal=win["nominal"], stress=win["stress"], U_final=win["U_final"], fold=int(fold),
                  recordings=len(files))
    (train_folder / "backtest_choice.json").write_text(json.dumps(choice, indent=1, default=float))
    (train_folder / "backtest.json").write_text(json.dumps(dict(choice=choice, models=res), indent=1, default=float))
    (train_folder / "backtest.md").write_text(report(res, win, len(files), fold))
    log(f"[backtest] chosen {win['tag']}: FIRE {win['fire'] / 20:.2f} / SECOND {win['second'] / 20:.2f}, taps "
        f"{100 * win['nominal']['taps']:.1f}%, false {win['nominal']['false_pm']:.2f}/min "
        f"(fist {win['nominal']['false_fist_pm']:.2f}, finger {win['nominal']['false_finger_pm']:.2f}) in "
        f"{time.time() - t0:.0f} s")
    return choice


def report(res, win, n_files, fold):
    p = lambda v: f"{100 * v:.1f}%"
    f2 = lambda v: f"{v:.2f}"
    n0 = win["nominal"]
    md = [f"# TAP back test: held-out fold {fold}, {n_files} recordings", "",
          f"Every row scored, then the firmware tap logic (tap_gestures.c, deferred mode): FIRE rising edge, 150 ms "
          f"refractory, DOUBLE_TAP = 2nd tap 150..400 ms after the 1st with a dip below 0.20 and a score >= SECOND "
          f"(FIRE - 0.15). taps = pointer presses matched by a tap onset -100..+250 ms; false = TAP / DOUBLE_TAP "
          f"events per minute of the no-tap recordings. U = taps (%) - {LAMBDA:g} x the mean false rate over the "
          f"movement kinds (fist and finger count twice); U with band changes = the mean of U over clean, placement, "
          f"hum (mean of the zones) and rotation. FIRE per model: the highest whose U with band changes is within "
          f"{SE_RULE:g} standard error(s) of its best (thin evidence leans to fewer false taps). Model: the best U with "
          f"band changes; the models within {SE_RULE:g} standard error(s) of it (*) are as good, "
          f"{PREFER + ' first, then ' if PREFER else ''}the smallest.", "",
          f"**Chosen: `{win['tag']}` ({win['model_type']}, {win['bytes']} bytes), FIRE {win['fire'] / 20:.2f}, SECOND "
          f"{win['second'] / 20:.2f}.** Taps {p(n0['taps'])}, false {f2(n0['false_pm'])}/min (fist "
          f"{f2(n0['false_fist_pm'])}, finger {f2(n0['false_finger_pm'])}).", "",
          f"Held-out exposure: {n0['presses']} tap presses, {n0['double_pairs_n']} double-tap pairs, {n0['singles_n']} "
          f"single taps, {n0['minutes_no_tap']:.1f} minutes of no-tap recordings (fist {n0['minutes_fist']:.1f}, finger "
          f"{n0['minutes_finger']:.1f}); one false event in the fist recordings is "
          f"{1 / max(n0['minutes_fist'], 1e-9):.2f} per minute.", "",
          "| Model | Type | Bytes | FIRE | U | Taps | Soft | 240 bpm | Doubles as DOUBLE_TAP | Singles as DOUBLE_TAP | "
          "False/min | Fist/min | Finger/min | Wrist/min | Arm/min | Band/min | U with band changes | Best of it (FIRE) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ranked = sorted(res, key=lambda r: (-r["U_best"], r["bytes"] or 1e12))
    for r in ranked:
        n = r["nominal"]
        name = f"**{r['tag']}**" if r is win else r["tag"]
        md.append(f"| {name}{' *' if r.get('tie') else ''} | {r['model_type']} | {r['bytes']} | {r['fire'] / 20:.2f} | "
                  f"{n['U']:.1f} | {p(n['taps'])} | {p(n['soft'])} | {p(n['fast240'])} | {p(n['double_pairs'])} | "
                  f"{p(n['single_as_double'])} | {f2(n['false_pm'])} | {f2(n['false_fist_pm'])} | "
                  f"{f2(n['false_finger_pm'])} | {f2(n['false_wrist_pm'])} | {f2(n['false_arm_pm'])} | "
                  f"{f2(n['false_band_pm'])} | {'' if r['U_final'] is None else f'{r_final(r):.1f}'} | "
                  f"{r['U_best']:.1f} ({r['fire_best_u'] / 20:.2f}) |")
    rb = win["robust"]
    md += ["", "Chosen model, every FIRE (SECOND = FIRE - 0.15; +- one standard error):", "",
           "| FIRE | U | Taps | Doubles as DOUBLE_TAP | Singles as DOUBLE_TAP | False/min | Fist/min | Finger/min | "
           "False DOUBLE_TAP/min | U with band changes |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f in FIRES:
        n = win["curve"][f]
        mark = " (chosen)" if f == win["fire"] else " (best)" if f == win["fire_best_u"] else ""
        md.append(f"| {f / 20:.2f}{mark} | {n['U']:.1f} +- {n['U_se']:.1f} | {p(n['taps'])} | {p(n['double_pairs'])} | "
                  f"{p(n['single_as_double'])} | {f2(n['false_pm'])} | {f2(n['false_fist_pm'])} | "
                  f"{f2(n['false_finger_pm'])} | {f2(n['false_double_pm'])} | "
                  f"{robust_cell(rb, f)} |")
    shown = [r for r in ranked if r["stress"]][:4]
    shown += [win] if win["stress"] and win not in shown else []
    if shown:
        md += ["", "Band changes (tap_f1_augment.stress, each held-out recording its own random draw), at each model's "
               "FIRE, the best models:", "",
               "| Model | Change | U | Taps | False/min | Fist/min | Finger/min |", "|---|---|---|---|---|---|---|"]
        for r in shown:
            for k, n in [("clean", r["nominal"])] + list(r["stress"].items()):
                md.append(f"| {r['tag']} | {k} | {n['U']:.1f} | {p(n['taps'])} | {f2(n['false_pm'])} | "
                          f"{f2(n['false_fist_pm'])} | {f2(n['false_finger_pm'])} |")
    return "\n".join(md) + "\n"


def r_final(r):
    return r["U_final"]


def robust_cell(rb, f):
    return "" if not rb else f"{rb[f]['U']:.1f} +- {rb[f]['U_se']:.1f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train_folder", nargs="?", default="TAP_MODELS_F1_FILEFOLDS")
    ap.add_argument("--data", type=Path, default=ROOT / "DATA" / "DATA_TAP_F1")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--roadmap", type=Path, default=ROOT / "DATA" / "fold_roadmap_files.json")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--stress-top", default=STRESS_TOP, help="models replayed under band changes: all (default), or "
                    "the N best clean ones (0: none)")
    a = ap.parse_args(argv)
    backtest(a.train_folder, a.data, a.fold, a.roadmap, a.workers, a.stress_top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
