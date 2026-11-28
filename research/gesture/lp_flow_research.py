"""Long-press flow research: a RELEASE EVENT detector with a latch, against the current hold/release STATE model.

Why: the current LONG_PRESS model answers "is the finger released right now?" for every frame. A soft hold has almost
no sustained muscle signal (activity 1-2 s into the hold: 30% of soft, 69% of mid long presses;
LONG_PRESS_CHANNEL_STATE_RESEARCH_2026-11-17.md), so after the release transient fades, resting looks like holding
again: the probability rises at the release and falls back. What the sensors do show is the change: the press
(the TAP model fires on it) and the lift.

Flow B (event + latch), per press, from its onset s (here the labelled press onset; on the band the TAP onset):
  - release events: rising edges of the RELEASE model's score >= thr (150 ms refractory), ignoring the first 40 ms;
  - a release event within D ms of the onset -> it was a TAP;
  - none by s + D -> HOLD (long press starts; the pending tap is cancelled);
  - the hold ends at the first release event after s + D.
Flow A (current): the state model's ten-sample average of Q20 release values >= thr declares the release
(replay.lp_declared_release); a press is a hold if no release is declared within D ms.

Both models: same inputs (126 raw fields + diffs), same fixed LightGBM settings, 14 KB, trees of 7 leaves, 3 folds
by recording (stratified by folder), scored on the press windows of the held-out recordings of each fold.

    python research/gesture/lp_flow_research.py            # results_lp_flow/summary.json + tables on stdout
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tap_generalization_research as GR  # noqa: E402
import train_unified as g                  # noqa: E402
import replay as R                         # noqa: E402

CACHE = Path(os.environ.get("MUDRA_LPF_CACHE", "/tmp/v3data/lp/cache"))
# MUDRA_LPF_LITE=<dir with lite_replay.npy / lite_train.npy>: add the 15 Lean-lite diff fields (10 s context,
# computed from each recording's full history) to both models' inputs.
LITE = os.environ.get("MUDRA_LPF_LITE", "")
# MUDRA_LPF_NOBG=1: neither model sees the background rows (rest / other motion far from any press). The flow only
# asks either model something after a press onset, so rest never has to be told apart from a soft hold.
NOBG = os.environ.get("MUDRA_LPF_NOBG", "") == "1"
# MUDRA_LPF_ANCHOR=1 (needs MUDRA_LPF_LITE's dir with anchor_replay.npy): press-anchored fields, both models
# trained only on the press windows. A = "since this press started, is the finger still down?" (0 before the
# release, 1 from 20 ms after it); B = the release event. Anchored fields per channel: EMG level against its
# 400-100 ms pre-press mean, against its running maximum since the onset, and min(time since onset, 600 ms).
ANCHOR = os.environ.get("MUDRA_LPF_ANCHOR", "") == "1"
OUT = Path(os.environ.get("MUDRA_LPF_OUT", HERE / ("results_lp_flow" + ("_lite" if LITE else "") + ("_nobg" if NOBG else "")
                                                     + ("_anchor" if ANCHOR else ""))))
SPEC = GR.arm_from_name("raw126_diff+litediff")["spec"] if LITE else GR.ARMS["raw126_diff"]["spec"]
BUDGET = 14 * 1024
EXTRA = dict(num_leaves=7)
POS_MS = 80              # release positives: [release, release + 80 ms]
GAP_BEFORE_MS = 60       # unknown: (release - 60, release)
GAP_AFTER_MS = 350       # unknown: (release + 80, release + 350]
IGNORE_FIRST_MS = 40     # no release event in the first 40 ms after the onset
DECISION_MS = (300, 400, 600)
EARLY_MS = 100           # a release declared more than 100 ms before the pointer release breaks the hold
ON_TIME_MS = 150         # declared within 150 ms after the pointer release
EVENT_THR = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
STATE_THR = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
NEG_STRIDE = 3           # every 3rd negative replay row in the event model's training
MIN_FRAMES = 50          # the firmware runs the models from frame 50 (about 1.1 s)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load():
    X = np.load(CACHE / "train_X.npy", mmap_mode="r")
    m = dict(np.load(CACHE / "train_meta.npz"))
    rp = np.load(CACHE / "replay_meta.npz")
    n, ncol = int(rp["n"]), int(rp["ncol"])
    assert ncol == 150, ncol
    RX = np.memmap(CACHE / "replay_X.f32", dtype=np.float32, mode="r", shape=(n, ncol))
    recs = pd.read_csv(CACHE / "recordings.csv")
    win = pd.read_csv(CACHE / "lp_windows.csv")
    win["folder"] = recs.folder.to_numpy()[win.rec]
    wl = np.asarray(X[:20000][:, GR.WL["wl"]])
    assert (wl >= 0).all() and np.median(wl) > 50, "column order: WL fields not where expected"
    LT = np.load(Path(LITE) / "lite_train.npy", mmap_mode="r") if LITE else None
    LR = np.load(Path(LITE) / "lite_replay.npy", mmap_mode="r") if LITE else None
    AR = np.load(Path(LITE) / "anchor_replay.npy", mmap_mode="r") if ANCHOR else None
    return X, m, RX, rp["t"].astype(np.float64), rp["frames"], recs, win, LT, LR, AR


def features(X, L=None, A=None):
    X = np.asarray(X, np.float32)
    F, _, _ = GR.build(np.hstack([X, np.asarray(L, np.float32)]) if LITE else X, SPEC)
    return np.hstack([F, np.asarray(A, np.float32)]) if ANCHOR else F


def event_labels(t, s, rel):
    y = np.full(len(t), -1, np.int8)
    y[(t >= rel) & (t <= rel + POS_MS)] = 1
    y[(t < rel - GAP_BEFORE_MS) | (t > rel + GAP_AFTER_MS)] = 0
    return y


def events(p, t, s, thr):
    d = p >= thr
    rise = np.flatnonzero(d & ~np.r_[False, d[:-1]])
    out, last = [], -1e18
    for r in rise:
        if t[r] - s < IGNORE_FIRST_MS:
            continue
        if t[r] - last >= R.TAP_REFRACTORY_MS:
            out.append(t[r]); last = t[r]
    return np.asarray(out)


def outcome(kind, s, rel, decided_tap_at, release_at, D):
    """One press under a flow. decided_tap_at: time the flow called it a tap (<= s + D) or None;
    release_at: for a press the flow holds, the declared release time or None (stuck)."""
    if kind == "tap":
        return dict(tap_kept=decided_tap_at is not None)
    if decided_tap_at is not None:
        return dict(hold_recognised=False, hold_ok=False, early=True, on_time=False, late=False, stuck=False)
    early = release_at is not None and release_at < rel - EARLY_MS
    on_time = release_at is not None and not early and release_at <= rel + ON_TIME_MS
    return dict(hold_recognised=True, early=early, on_time=on_time, hold_ok=on_time,
                late=release_at is not None and not early and not on_time, stuck=release_at is None,
                latency=(release_at - rel) if (release_at is not None and not early) else None)


def summarize(rows):
    df = pd.DataFrame(rows)
    out = {}
    for folder, d in df.groupby("folder"):
        r = dict(presses=len(d))
        for k in ("tap_kept", "hold_recognised", "hold_ok", "early", "on_time", "late", "stuck"):
            if k in d and d[k].notna().any():
                r[k] = round(100 * float(d[k].dropna().astype(float).mean()), 1)
        if "latency" in d and d.latency.notna().any():
            r["latency_ms_median"] = float(d.latency.dropna().median()); r["latency_ms_p90"] = float(d.latency.dropna().quantile(.9))
        out[folder] = r
    holds = df[df.kind == "hold"]; taps = df[(df.kind == "tap") & (df.folder == "TAP")]
    h = float(holds.hold_ok.mean()) if len(holds) else np.nan
    tk = float(taps.tap_kept.mean()) if len(taps) else np.nan
    out["ALL"] = dict(hold_ok=round(100 * h, 1), tap_kept=round(100 * tk, 1),
                      score=round(100 * 2 * h * tk / (h + tk), 2) if h + tk > 0 else 0.0)
    return out


def fit_and_trace():
    """Stage 1: per fold, fit the state model (A) and the release-event model (B) on the training recordings and
    write both scores for every replay row of the held-out recordings' press windows (traces.npz)."""
    OUT.mkdir(parents=True, exist_ok=True)
    X, m, RX, rt, rframes, recs, win, LT, LR, AR = load()
    ar = (lambda a, b, keep=slice(None): np.asarray(AR[a:b])[keep]) if ANCHOR else (lambda a, b, keep=None: None)
    lt = (lambda idx: LT[idx]) if LITE else (lambda idx: None)
    lr = (lambda a, b, keep=slice(None): np.asarray(LR[a:b])[keep]) if LITE else (lambda a, b, keep=None: None)
    log("LPF_DATA", json.dumps(dict(train_rows=len(m["y"]), replay_rows=len(rt), windows=len(win),
                                    holds=int((win.kind == "hold").sum()), taps=int((win.kind == "tap").sum()))))
    rec_ids = np.unique(np.r_[m["rec"], win.rec.to_numpy()])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=g.SEED).split(rec_ids, recs.folder.to_numpy()[rec_ids]))
    pA = np.full(len(rt), np.nan, np.float32); pB = np.full(len(rt), np.nan, np.float32)
    wfold = np.full(len(win), -1, np.int8); info = []
    for f, (a_, b_) in enumerate(folds):
        tr_recs, va_recs = set(rec_ids[a_].tolist()), set(rec_ids[b_].tolist())
        tri = np.flatnonzero(np.isin(m["rec"], list(tr_recs)) & (m["frames"] >= MIN_FRAMES))
        if ANCHOR:                                   # anchored state: windows only
            Xs, ys, rs, Ls, As = [], [], [], [], []
            for w in win[win.rec.isin(tr_recs)].itertuples():
                t = rt[w.start:w.end]; y = np.full(len(t), -1, np.int8)
                y[t < w.release_ms - GAP_BEFORE_MS] = 0; y[t >= w.release_ms + 20] = 1
                y[rframes[w.start:w.end] < MIN_FRAMES] = -1
                keep = np.flatnonzero((y >= 0) & (np.arange(len(y)) % 2 == 0))
                Xs.append(np.asarray(RX[w.start:w.end][keep])); ys.append(y[keep]); rs.append(np.full(len(keep), w.rec))
                Ls.append(lr(w.start, w.end, keep)); As.append(ar(w.start, w.end, keep))
            FA = features(np.concatenate(Xs), np.concatenate(Ls) if LITE else None, np.concatenate(As))
            yA = np.concatenate(ys).astype(int); rA = np.concatenate(rs); del Xs, Ls, As
            mA = GR.fit(FA, yA, GR.weights(yA, rA, np.ones(len(yA), np.float32)), BUDGET, extra_params=EXTRA)
        else:
            triA = tri[m["kind"][tri] != 0] if NOBG else tri
            FA = features(X[triA], lt(triA)); yA = m["y"][triA].astype(int)
            mA = GR.fit(FA, yA, GR.weights(yA, m["rec"][triA], m["w"][triA]), BUDGET, extra_params=EXTRA)
        kA, bA = g.trees_within(mA, BUDGET); del FA
        Xs, ys, rs, Ls, As = [], [], [], [], []
        for w in win[win.rec.isin(tr_recs)].itertuples():
            t = rt[w.start:w.end]; y = event_labels(t, w.start_ms, w.release_ms)
            y[rframes[w.start:w.end] < MIN_FRAMES] = -1
            keep = np.flatnonzero((y == 1) | ((y == 0) & (np.arange(len(y)) % NEG_STRIDE == 0)))
            Xs.append(np.asarray(RX[w.start:w.end][keep])); ys.append(y[keep]); rs.append(np.full(len(keep), w.rec))
            Ls.append(lr(w.start, w.end, keep)); As.append(ar(w.start, w.end, keep))
        bg = tri[m["kind"][tri] == 0] if not (NOBG or ANCHOR) else tri[:0]
        Xs.append(np.asarray(X[bg])); ys.append(np.zeros(len(bg), np.int8)); rs.append(m["rec"][bg]); Ls.append(lt(bg))
        XB = np.concatenate(Xs); yB = np.concatenate(ys).astype(int); rB = np.concatenate(rs); del Xs
        FB = features(XB, np.concatenate(Ls) if LITE else None, np.concatenate([a for a in As if a is not None]) if ANCHOR else None); del XB, Ls, As
        mB = GR.fit(FB, yB, GR.weights(yB, rB, np.ones(len(yB), np.float32)), BUDGET, extra_params=EXTRA)
        kB, bB = g.trees_within(mB, BUDGET); del FB
        (OUT / "models").mkdir(exist_ok=True)
        mA.booster_.save_model(str(OUT / "models" / f"state_fold{f}.txt"), num_iteration=kA)
        mB.booster_.save_model(str(OUT / "models" / f"event_fold{f}.txt"), num_iteration=kB)
        info.append(dict(fold=f, state=dict(rows=len(yA), trees=kA, bytes=int(bA)),
                         event=dict(rows=len(yB), positives=int(yB.sum()), trees=kB, bytes=int(bB))))
        log("LPF_FOLD_MODELS", f, json.dumps(info[-1]))
        va = win[win.rec.isin(va_recs)]
        wfold[va.index.to_numpy()] = f
        for w in va.itertuples():
            F = features(RX[w.start:w.end], lr(w.start, w.end), ar(w.start, w.end))
            pA[w.start:w.end] = mA.booster_.predict(F, num_iteration=kA)
            pB[w.start:w.end] = mB.booster_.predict(F, num_iteration=kB)
        log("LPF_FOLD_DONE", f)
    np.savez_compressed(OUT / "traces.npz", pA=pA, pB=pB, t=rt.astype(np.float32), frames=rframes, wfold=wfold)
    (OUT / "models.json").write_text(json.dumps(info, indent=1))


def flow_times(win, Z, rule, D, **k):
    """Per press: (tap_at, release_at) for a flow on the stored traces. rule: 'state' (A), 'event' (B),
    'both' (C: a release event confirmed by the state model's ten-sample average within the next confirm_ms)."""
    out = []
    for w in win.itertuples():
        if w.fold < 0 or Z["frames"][w.start] < MIN_FRAMES:
            out.append((np.nan, np.nan, False)); continue
        t = Z["t"][w.start:w.end].astype(np.float64)
        if rule == "state":
            d = R.lp_declared_release(R.q20(Z["pA"][w.start:w.end]), t, w.start_ms, k["thr"])
            cand = [d] if d is not None else []
            first = d
        elif rule in ("state_sustain", "event_sustain"):     # one model: score held >= thr for sustain_ms
            p_ = Z["pA" if rule == "state_sustain" else "pB"][w.start:w.end]
            if rule == "state_sustain":                       # the firmware's ten-sample average first
                q = R.q20(p_).astype(np.float64); n = R.LP_AVERAGE_SAMPLES
                p_ = np.r_[np.full(n - 1, 0.0), np.convolve(q, np.ones(n), "valid") / (20.0 * n)]
            hi = (p_ >= k["thr"]) & (t - w.start_ms >= IGNORE_FIRST_MS)
            start = np.flatnonzero(hi & ~np.r_[False, hi[:-1]]); cand = []
            for a in start:
                b = a
                while b + 1 < len(hi) and hi[b + 1]:
                    b += 1
                if t[b] - t[a] >= k["sustain_ms"]:
                    j = np.searchsorted(t, t[a] + k["sustain_ms"])
                    cand.append(float(t[min(j, len(t) - 1)]))
            first = cand[0] if cand else None
        else:
            ev = events(Z["pB"][w.start:w.end], t, w.start_ms, k["thr"])
            if rule == "both":
                q = R.q20(Z["pA"][w.start:w.end]).astype(np.int64); n = R.LP_AVERAGE_SAMPLES
                s10 = np.convolve(q, np.ones(n, np.int64), "valid"); end_t = t[n - 1:]
                ok = s10 >= int(round(k["thr_state"] * 20 * n))
                conf = []
                for e in ev:
                    j = np.flatnonzero(ok & (end_t >= e) & (end_t <= e + k["confirm_ms"]))
                    if len(j):
                        conf.append(end_t[j[0]])
                ev = np.asarray(conf)
            cand = list(ev); first = ev[0] if len(ev) else None
        tap_at = first if (first is not None and first <= w.start_ms + D) else np.nan
        after = [c for c in cand if c > w.start_ms + D]
        out.append((tap_at, after[0] if after else np.nan, True))
    return out


def score_flow(win, times, D, early_ms=EARLY_MS, on_time_ms=ON_TIME_MS):
    rows = []
    for w, (tap_at, rel_at, ok) in zip(win.itertuples(), times):
        if not ok:
            continue
        tap_at = None if np.isnan(tap_at) else tap_at; rel_at = None if np.isnan(rel_at) else rel_at
        e, o = EARLY_MS, ON_TIME_MS
        globals()["EARLY_MS"], globals()["ON_TIME_MS"] = early_ms, on_time_ms
        r = outcome(w.kind, w.start_ms, w.release_ms, tap_at, rel_at, D)
        globals()["EARLY_MS"], globals()["ON_TIME_MS"] = e, o
        rows.append(dict(folder=w.folder, kind=w.kind, **r))
    return summarize(rows)


def evaluate():
    """Stage 2: every flow and setting on the stored traces."""
    win = pd.read_csv(CACHE / "lp_windows.csv"); recs = pd.read_csv(CACHE / "recordings.csv")
    win["folder"] = recs.folder.to_numpy()[win.rec]
    Z = dict(np.load(OUT / "traces.npz")); win["fold"] = Z["wfold"]
    res = {}
    for D in DECISION_MS:
        for thr in STATE_THR:
            res[f"A_state|D{D}|thr{thr}"] = score_flow(win, flow_times(win, Z, "state", D, thr=thr), D)
        for thr in EVENT_THR:
            res[f"B_event|D{D}|thr{thr}"] = score_flow(win, flow_times(win, Z, "event", D, thr=thr), D)
            for ts in (0.3, 0.5, 0.7):
                for cm in (150, 300):
                    res[f"C_both|D{D}|thr{thr}|state{ts}|confirm{cm}"] = score_flow(
                        win, flow_times(win, Z, "both", D, thr=thr, thr_state=ts, confirm_ms=cm), D)
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
            for su in (50, 100, 200):
                res[f"D_state_sustain|D{D}|thr{thr}|sustain{su}"] = score_flow(
                    win, flow_times(win, Z, "state_sustain", D, thr=thr, sustain_ms=su), D)
                res[f"E_event_sustain|D{D}|thr{thr}|sustain{su}"] = score_flow(
                    win, flow_times(win, Z, "event_sustain", D, thr=thr, sustain_ms=su), D)
        log("LPF_EVAL", D, len(res))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    for flow in ("A_state", "B_event", "C_both", "D_state_sustain", "E_event_sustain"):
        best = max(((k, v) for k, v in res.items() if k.startswith(flow)), key=lambda kv: kv[1]["ALL"]["score"])
        log("LPF_BEST", best[0], json.dumps(best[1]["ALL"]))


if __name__ == "__main__":
    if "--evaluate" not in sys.argv:
        fit_and_trace()
    evaluate()
