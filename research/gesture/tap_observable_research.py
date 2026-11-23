#!/usr/bin/env python3
"""When is a tap distinguishable from its look-alikes, and from what? (TAP_OBSERVABLE_RESULTS.md)

    python research/gesture/tap_observable_research.py timeline      # phase A: separability vs time from onset
    python research/gesture/tap_observable_research.py screen        # phase C helper: per-person input screen

Data: the research cache of tap_generalization_research.py (development people only; the trainer's held-out
people are never touched).

Phase A: for every real pointer press of the TAP / SWIPE recordings (labelled or not) take the replay row at
onset + tau (tau = -66 .. +110 ms, i.e. -3 .. +5 EMG frames of 22 ms), and contrast it with look-alike rows: rows
with muscle activity in the finger-motion / fist / wrist / arm / band recordings. A LightGBM per tau, cross-fitted
by person (3 folds), scores the held-out people. Reported per tau: AUC and the share of taps found while only 1%
(0.2%) of look-alike rows fire, per tap type and per look-alike type, per input family (all / EMG only / ACC
only), and the per-person AUC (taps vs look-alikes of the same person).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tap_generalization_research as G          # noqa: E402

OUT = HERE / "results_tap_observable"
FRAME_MS = 22.0
TAUS = tuple(round(k * FRAME_MS) for k in range(-3, 6))          # -66 .. +110 ms
VIEWS = {"all": ("acc", "emgraw", "diff", "wlratio", "litediff"),
         "emg": ("emgraw", "diff", "wlratio", "litediff"),
         "acc": ("acc", "accdiff")}
LOOKALIKE_FOLDERS = ("FINGER", "WRIST", "ARM", "BAND")
PARAMS = dict(objective="binary", n_estimators=200, learning_rate=0.08, num_leaves=15, max_depth=6,
              min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1, max_bin=127,
              verbosity=-1, deterministic=True, random_state=7)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def presses():
    """Every real pointer press of the development people's TAP / SWIPE recordings: (rec, t_on, type, person)."""
    D = G.load(); recs = D["rpl"]["recs"]; on = D["onsets"]
    dev = set(D["data"]["group"][D["tr"]])
    out = []
    for r, v in on[on.kind == "press"].groupby("rec"):
        rr = recs.iloc[int(r)]
        if rr.group not in dev or rr.folder not in ("TAP", "SWIPE"):
            continue
        typ = G.rec_type(rr.folder, rr.file)
        ts = np.sort(v.t_on.to_numpy(float)); gap = np.diff(np.r_[-1e18, ts])
        for t_, g_ in zip(ts, gap):
            k = ("double_2nd" if g_ < G.PAIR_MS else "double_1st") if typ == "double" else typ
            out.append((int(r), float(t_), k, rr.group))
    return pd.DataFrame(out, columns=["rec", "t_on", "type", "person"])


def rows_at(pr, tau):
    """Replay row nearest to onset + tau for each press (-1 if outside the recording or before frame 50)."""
    rpl = G.load()["rpl"]; recs = rpl["recs"]
    out = np.full(len(pr), -1, np.int64)
    for i, (r, t_) in enumerate(zip(pr.rec.to_numpy(), pr.t_on.to_numpy())):
        a, b = int(recs.start.iloc[r]), int(recs.end.iloc[r])
        t = np.asarray(rpl["t"][a:b], np.float64)
        j = int(np.searchsorted(t, t_ + tau))
        if 0 < j < len(t) and abs(t[j] - (t_ + tau)) > abs(t[j - 1] - (t_ + tau)):
            j -= 1
        if 0 <= j < len(t) and abs(t[j] - (t_ + tau)) <= 6 and rpl["frames"][a + j] >= 50:
            out[i] = a + j
    return out


def lookalikes(cap=80000, seed=3):
    """Replay rows with muscle activity in the look-alike recordings of the development people."""
    D = G.load(); data = D["data"]; recs = D["rpl"]["recs"]
    tr = D["tr"]
    cat = data["category"][tr]; kind = data["kind"][tr]; y = data["y"][tr]
    sel = tr[np.isin(cat, LOOKALIKE_FOLDERS) & (kind == 2) & (y == 0)]
    if len(sel) > cap:
        sel = np.sort(np.random.default_rng(seed).choice(sel, cap, replace=False))
    ridx = G.data_ridx(sel)
    rec = data["rec"][sel]
    typ = np.array([G.rec_type(recs.folder.iloc[r], recs.file.iloc[r]) for r in rec], object)
    return ridx, rec, typ, data["group"][sel]


def features(ridx, rec, spec):
    rpl = G.load()["rpl"]
    X = np.asarray(rpl["X"][ridx])
    if "litediff" in spec:
        X = np.hstack([X, G.lite_rows(ridx, rec)])
    F, names, _ = G.build(X, spec)
    return F, names


def tpr_at(y, p, fpr):
    neg = np.sort(p[y == 0])
    thr = neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))]
    return float((p[y == 1] > thr).mean())


def timeline():
    import lightgbm as lgb
    t0 = time.perf_counter()
    pr = presses(); n_ridx, n_rec, n_typ, n_person = lookalikes()
    recs = G.load()["rpl"]["recs"]
    log("OBS_DATA", json.dumps(dict(presses=len(pr), types=pr.type.value_counts().to_dict(),
                                    lookalike_rows=int(len(n_ridx)),
                                    lookalike_types=pd.Series(n_typ).value_counts().to_dict())))
    neg_feats = {v: features(n_ridx, n_rec, spec)[0] for v, spec in VIEWS.items()}
    people = sorted(set(pr.person) | set(n_person))
    folds = list(GroupKFold(n_splits=3).split(people, groups=people))
    fold_of = {people[i]: k for k, (_, b) in enumerate(folds) for i in b}
    results = []
    for tau in TAUS:
        pos_ridx = rows_at(pr, tau); ok = pos_ridx >= 0
        p_rows = pos_ridx[ok]; p_rec = pr.rec.to_numpy()[ok]; p_typ = pr.type.to_numpy()[ok]
        p_person = pr.person.to_numpy()[ok]
        for view, spec in VIEWS.items():
            Fp, _ = features(p_rows, p_rec, spec); Fn = neg_feats[view]
            F = np.vstack([Fp, Fn]); y = np.r_[np.ones(len(Fp), int), np.zeros(len(Fn), int)]
            person = np.r_[p_person, n_person]; fo = np.array([fold_of[p_] for p_ in person])
            w = np.where(y == 1, 0.5 / max(1, y.sum()), 0.5 / max(1, (1 - y).sum())) * len(y)
            oof = np.zeros(len(y))
            for k in range(3):
                trn, val = fo != k, fo == k
                m = lgb.LGBMClassifier(**PARAMS, n_jobs=G.THREADS).fit(F[trn], y[trn], sample_weight=w[trn])
                oof[val] = m.predict_proba(F[val])[:, 1]
            ntyp = np.r_[np.full(len(Fp), "", object), n_typ]
            ptyp = np.r_[p_typ, np.full(len(Fn), "", object)]
            row = dict(tau=tau, view=view, n_pos=int(len(Fp)), auc=roc_auc_score(y, oof),
                       tpr1=tpr_at(y, oof, 0.01), tpr02=tpr_at(y, oof, 0.002))
            for t_ in ("normal", "soft", "double_1st", "double_2nd", "fast240", "swipe"):
                m_ = (ptyp == t_) | (y == 0)
                if (ptyp == t_).sum() >= 30:
                    row[f"auc_{t_}"] = roc_auc_score(y[m_], oof[m_]); row[f"tpr1_{t_}"] = tpr_at(y[m_], oof[m_], 0.01)
            for t_ in ("finger", "fist", "wrist", "arm", "band"):
                m_ = (y == 1) | (ntyp == t_)
                if (ntyp == t_).sum() >= 30:
                    row[f"auc_vs_{t_}"] = roc_auc_score(y[m_], oof[m_])
            # per person: that person's taps vs that person's look-alikes (only people with both)
            pa = []
            for p_ in people:
                mp = person == p_
                if (y[mp] == 1).sum() >= 20 and (y[mp] == 0).sum() >= 20:
                    pa.append(roc_auc_score(y[mp], oof[mp]))
            row.update(person_n=len(pa), person_auc_min=float(np.min(pa)) if pa else np.nan,
                       person_auc_median=float(np.median(pa)) if pa else np.nan,
                       person_auc_p25=float(np.percentile(pa, 25)) if pa else np.nan)
            results.append(row)
            log("OBS_TAU", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUT / "timeline.csv", index=False)
    log("OBS_TIMELINE_DONE", round(time.perf_counter() - t0, 1), "s")
    return df


def event_shapes(arm="raw126_diff+augm+litediff+press2+kb14+leaves7", thr=0.9923):
    """What do the model's events look like over time? For every event of the arm's fold models on their held-out
    people at threshold thr: real taps (an event matched to a press in TAP recordings) and false events in finger-motion
    and fist recordings. Per event: ACC energy (sum log1p acc_psd[0]) and EMG level (mean log2(WL+1) of the 3
    channels) from -110 to +550 ms around the event, minus the median over -600..-300 ms. Saved as
    results_tap_observable/event_shapes.csv (medians and quartiles per class and time)."""
    import lightgbm as lgb
    D = G.load(); rpl = D["rpl"]; recs = rpl["recs"]; data = D["data"]
    acc_ix = G.ACC_NOW; wl_ix = [G.CI[f"p_core->wl[CH{c}]"] for c in (1, 2, 3)]
    grid = np.arange(-110, 551, 22.0)
    out = []
    on = D["onsets"]; press = {int(r): np.sort(v.t_on.to_numpy(float)) for r, v in on[on.kind == "press"].groupby("rec")}
    for f, (a, b) in enumerate(D["folds"]):
        people = set(data["group"][D["tr"][b]])
        bst = lgb.Booster(model_file=str(G.OUT / "models" / f"{arm}_fold{f}.txt"))
        val = G.Val(people)
        p = val.predict(bst, None, G.arm_from_name(arm)["spec"])
        for (s_, e_), t, item in zip(val.seg, val.t, val.items):
            rec, folder = item[0], item[3]
            typ = G.rec_type(folder, recs.file.iloc[rec])
            if typ not in ("finger", "fist", "normal", "soft", "double", "fast240"):
                continue
            pp = p[s_:e_]; d = pp >= thr
            rise = np.flatnonzero(d & ~np.r_[False, d[:-1]])
            ev = G._events(t, rise)
            if not len(ev):
                continue
            if folder == "TAP":
                pr = press.get(int(rec), np.zeros(0))
                hit, _ = G._match(ev, pr)        # presses found; take the event nearest each found press
                evs = []
                for o in pr[hit]:
                    j = int(np.argmin(np.abs(ev - o))); evs.append(ev[j])
                cls = "tap_" + typ
            else:
                evs = list(ev); cls = "false_" + typ
            a_, b_ = int(recs.start.iloc[rec]), int(recs.end.iloc[rec])
            tt = np.asarray(rpl["t"][a_:b_], np.float64); X = np.asarray(rpl["X"][a_:b_])
            accE = np.log1p(np.clip(X[:, acc_ix], 0, None)).sum(1)
            emgL = np.log2(np.clip(X[:, wl_ix], 0, None) + 1.0).mean(1)
            for e0 in evs:
                base_m = (tt >= e0 - 600) & (tt < e0 - 300)
                if base_m.sum() < 10 or tt[-1] < e0 + 550 or tt[0] > e0 - 600:
                    continue
                j = np.clip(np.searchsorted(tt, e0 + grid), 0, len(tt) - 1)
                out.append(dict(cls=cls, person=recs.group.iloc[rec], fold=f,
                                **{f"acc_{int(g_)}": float(accE[j][k] - np.median(accE[base_m])) for k, g_ in enumerate(grid)},
                                **{f"emg_{int(g_)}": float(emgL[j][k] - np.median(emgL[base_m])) for k, g_ in enumerate(grid)}))
        log("OBS_SHAPES_FOLD", f, len(out))
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "event_shapes_events.csv.gz", index=False)
    print(df.cls.value_counts().to_string())
    summ = df.groupby("cls")[[c for c in df.columns if c.startswith(("acc_", "emg_"))]].median().T
    summ.to_csv(OUT / "event_shapes.csv")
    print(summ.round(2).to_string())
    return df


def tails(arms, targets=(2.8, 2.0)):
    """Per-person worst cases and pooled event rates for finger / fist false taps (unseen people), per arm, at the
    threshold that gives each overall false rate (same rule as focus_table)."""
    rows = []
    for arm in arms:
        z = np.load(G.OUT / f"focus_{arm}.npz", allow_pickle=True); ppl = z["people"]
        negs = [k.split("|")[2] for k in z.files if k.startswith("false|nominal|")]
        for cond in ("nominal", "placement"):
            allf = (sum(z[f"false|{cond}|{t}"].sum(1) for t in negs) /
                    sum(z[f"min|{cond}|{t}"].sum() for t in negs))
            base = (sum(z[f"false|nominal|{t}"].sum(1) for t in negs) /
                    sum(z[f"min|nominal|{t}"].sum() for t in negs))
            for target in targets:
                i = int(np.argmin(np.abs(base - target)))      # threshold fixed on the nominal curve
                row = dict(arm=arm, cond=cond, target=target)
                for t in ("finger", "fist"):
                    f = z[f"false|{cond}|{t}"][i]; m = z[f"min|{cond}|{t}"]; has = m > 0.2
                    r = f[has] / m[has]
                    row[f"{t}_pooled"] = float(f[has].sum() / m[has].sum())
                    row[f"{t}_people_gt1"] = int((r > 1).sum()); row[f"{t}_people"] = int(has.sum())
                    row[f"{t}_worst"] = float(r.max()); row[f"{t}_worst_who"] = str(ppl[has][int(np.argmax(r))])
                for t in ("normal", "soft", "double_2nd", "fast240", "swipe"):
                    row[t] = float(z[f"hits|{cond}|{t}"][i].sum() / max(1, z[f"n|{cond}|{t}"].sum()))
                rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(df.round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "timeline"
    if cmd == "timeline":
        timeline()
    elif cmd == "tails":
        tails(sys.argv[2:])
    elif cmd == "shapes":
        event_shapes()
    else:
        raise SystemExit(f"unknown command {cmd}")
