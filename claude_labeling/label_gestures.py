#!/usr/bin/env python3
"""Offline y_true for Mudra recordings: taps, swipes, long presses, and per-channel EMG state.  One script, no training step.

    python claude_labeling/label_gestures.py DATA/DATA_TAP         DATA/LABELS/DATA_TAP
    python claude_labeling/label_gestures.py DATA/DATA_LONG_PRESS  DATA/LABELS/DATA_LONG_PRESS
    python claude_labeling/label_gestures.py one_recording.csv.gz  out_dir  --kind tap

Needs numpy, pandas, scipy, lightgbm and the folder `models/` next to this script.
Full description, evidence and limits: claude_labeling/GESTURE_LABELING_2026-11-17.md

OUTPUT  one `<name>.labels.csv.gz` per recording (same sub-folders as the input), one row per source row, same order,
        plus `_events.csv.gz` (every tap event and long press with its decision) and `_summary.json`.

  Timestamp            copied from the source
  gesture_kind         recording type from its folder: 0 none, 1 tap, 2 swipe, 3 long press
  tap_label            1 inside [onset, onset+80 ms] of a confirmed tap or swipe, 0 not a tap, -1 unknown
  tap_active           1 from the sensor onset to the (shifted) pointer release of that tap or swipe, 0, -1
  tap_event_id         pointer press index of that tap or swipe, -1 elsewhere
  hold_label           1 long press held, 0 not held, -1 unknown
  hold_ms              ms since the labeled hold started (rows with hold_label 1), -1 elsewhere
  hold_press_id        pointer press index of that long press, -1 elsewhere
  ch1_label..ch3_label 0 REST, 1 ACTIVE (muscle activity), 2 OFF (electrode not in proper contact), -1 unknown
  ch1_contact..        contact-problem score 0..1
  n_ch_off             number of channels confidently disconnected (label 2)
  any_active           1 some channel ACTIVE, 0 every channel REST, -1 otherwise

  -1 means "leave this row out of the loss", never "negative".

RULES (each one is measured in the report)
  Recording type   from the folder: TAP, SWIPE -> taps; SOFT_/MID_/STRONG_LONG_PRESS, LONG_PRESS, DRAG_AND_DROP -> long presses;
                   FINGER, WRIST, ARM, BAND, NOTHING, CONNECTED, DISCONNECTED, LOOSE, TABLE -> no gestures. --kind overrides.
  Taps             a pointer press is a tap only where a tap teacher (LightGBM on ACC + EMG + WL, scored by the fold that
                   never saw that RECORDING: gesture-aware file folds, matched by content; never person folds) has a peak at
                   the 1-false-per-minute level within 150 ms (second pass 400 ms); the label starts at that sensor peak.
                   Weaker, unsupported and edge presses, and strong sensor events without a press, are unknown.
                   (Where each tap is TAUGHT is the trainer's choice: MUDRA_TAP_WINDOW=top moves it to the top of the
                   contact jolt, claude_labeling/tap_top.py.)
  Long presses     a press of >= 400 ms in a long-press recording is held from its muscle-activity onset (>= 30 ms of
                   activity in [-300, +500] ms around pointer-down) to 250 ms before release. Unconfirmed: unknown.
  Channel state    OFF = contact score >= 0.5 or dead-quiet WL, held >= 1 s. REST / ACTIVE = within 3 dB / >= 6 dB above
                   the recording's resting floor. The contact score is the mean of the five contact models (no person
                   dispatch).
  Disconnection    where 2 or more channels are confidently OFF there is no tap and no long press: tap_label,
                   tap_active and hold_label are 0 there (--disconnect-rule).
  Cross-gesture    in long-press recordings there are no taps: tap_label is 0, except from 300 ms before to 500 ms after each
                   pointer-down and each release, which are unknown (the tap teacher sees a tap at 81-100% of long-press starts
                   and 11-30% of releases). In tap recordings hold_label is 0.

This is an OFFLINE labeler (it looks at whole recordings). It is meant for y_true, not as the firmware model.
"""
import sys, os, re, io, gzip, json, glob, hashlib, argparse
import numpy as np, pandas as pd
from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

G = 10
HERE = os.path.dirname(os.path.abspath(__file__))
TAP_FOLDERS, SWIPE_FOLDERS = {'TAP'}, {'SWIPE'}
HOLD_FOLDERS = {'SOFT_LONG_PRESS', 'MID_LONG_PRESS', 'STRONG_LONG_PRESS', 'LONG_PRESS', 'DRAG_AND_DROP'}
NONE_FOLDERS = {'FINGER', 'WRIST', 'ARM', 'BAND', 'NOTHING', 'CONNECTED', 'DISCONNECTED', 'LOOSE', 'TABLE'}
KIND_CODE = {'none': 0, 'tap': 1, 'swipe': 2, 'hold': 3}

# ------------------------------------------------------------------------------------------------ parameters
P = dict(
    # tap events (claude_research_tap V2)
    TAP_TOL=150, TAP_TOL2=400, TAP_EDGE_MS=300, BOUNCE_MS=80, TAP_HMIN=0.03, TAP_WIN=(0, 80),
    # long press (claude_research_lp)
    MIN_HOLD_MS=400, CONF_BEFORE=300, CONF_AFTER=500, ONSET_AFTER=200, DEFAULT_LEAD=100, MIN_ACT_RUN=3,
    G_START=150, G_REL_BEFORE=250, G_REL_AFTER=150,
    # channel state
    OFF_T=0.5, ON_T=0.2, LW_DEAD=2.2, LW_OK=2.35, REST_DB=3.0, ACT_DB=6.0, MIN_GOOD_FRAMES=200, WARMUP=100, MIN_OFF_RUN=100,
    # disconnection rule and cross-gesture masks (set from the research)
    MIN_CH_OFF=2, HOLD_TAP_MASK=(-300, 500), HOLD_TAP_MASK_REL=(-300, 500),
)

# ------------------------------------------------------------------------------------------------ helpers
def lin(v): return np.power(10.0, np.asarray(v, np.float64) / 10000.0) - 1.0      # firmware code = 10000*log10(1+power)
def db(p): return 10 * np.log10(np.maximum(p, 1e-4))

def past(x, n, fn):
    o = (n - 1) // 2
    if fn == 'min': return minimum_filter1d(x, n, origin=o, mode='nearest')
    if fn == 'max': return maximum_filter1d(x, n, origin=o, mode='nearest')
    mu = uniform_filter1d(x, n, origin=o, mode='nearest')
    return mu if fn == 'mean' else np.sqrt(np.maximum(uniform_filter1d(x * x, n, origin=o, mode='nearest') - mu * mu, 0))

def drop_short_runs(a, k):
    a = np.asarray(a, bool).copy()
    if k <= 1 or not a.any(): return a
    e = np.diff(np.r_[0, a.astype(int), 0]); s0 = np.where(e > 0)[0]; s1 = np.where(e < 0)[0]
    for x, y in zip(s0, s1):
        if y - x < k: a[x:y] = False
    return a

def recording_kind(path, forced='auto'):
    if forced != 'auto': return forced
    parts = {p.upper() for p in os.path.normpath(os.path.abspath(path)).split(os.sep)}
    if parts & TAP_FOLDERS: return 'tap'
    if parts & SWIPE_FOLDERS: return 'swipe'
    if parts & HOLD_FOLDERS: return 'hold'
    if parts & NONE_FOLDERS: return 'none'
    return None

# ------------------------------------------------------------------------------------------------ models
class FoldModels:
    """Five cross-fitted models. key: the spec's map from a recording key to the fold whose model never saw it
    (file_fold: decompressed-content SHA-256); None: always the mean of the five (no dispatch)."""
    def __init__(self, prefix, spec_file, key):
        import lightgbm as lgb
        self.spec = json.load(open(os.path.join(HERE, 'models', spec_file)))
        self.fold_of = self.spec[key] if key else {}
        self.m = [lgb.Booster(model_file=os.path.join(HERE, 'models', f'{prefix}{k}.txt')) for k in range(5)]
    def predict(self, X, group):
        k = self.fold_of.get(group)
        return (self.m[k].predict(X), k) if k is not None else (np.mean([m.predict(X) for m in self.m], 0), -1)

TEACHER = 'file_fold_v1'     # recorded in _summary.json: the trainer rebuilds labels made by another teacher
def tap_teacher():
    return FoldModels('tap_teacher_file_fold', 'tap_teacher_file.json', 'file_fold')
def contact_models():
    return FoldModels('contact_fold', 'contact_folds.json', None)

def read_recording(path):
    """(dataframe, SHA-256 of the decompressed CSV bytes): the content key of the file folds."""
    raw = gzip.open(path, 'rb').read() if str(path).endswith('.gz') else open(path, 'rb').read()
    return pd.read_csv(io.BytesIO(raw), low_memory=False), hashlib.sha256(raw).hexdigest()

# ------------------------------------------------------------------------------------------------ tap teacher (exact port)
TAP_COLS = ([f'p_core->acc_psd[0][{a}][{z}]' for a in 'XYZ' for z in range(6)] +
            [f'p_core->emg_psd[0][CH{c}][{z}]' for c in (1, 2, 3) for z in range(5)] +
            [f'p_core->{k}[CH{c}]' for c in (1, 2, 3) for k in ('wl', 'wl_head', 'wl_tail')])
O_A = np.arange(-150, 251, 10) // G; O_R = np.arange(-100, 201, 20) // G; O_E = np.arange(-240, 241, 20) // G

def tap_traces(df):
    X = np.stack([df[c].to_numpy(np.float32) if c in df else np.full(len(df), np.nan, np.float32) for c in TAP_COLS], 1)
    t = df['Timestamp'].to_numpy(np.float64); t = t - t[0]
    keep = np.r_[True, np.diff(t) > 0]; X = X[keep]; t = t[keep]
    g = np.arange(0, t[-1], G); X = X[np.searchsorted(t, g, side='right') - 1]
    e = np.log1p(np.clip(X[:, 18:33], 0, None)).reshape(-1, 3, 5); e = e - np.percentile(e, 10, axis=0, keepdims=True)
    w = np.log1p(np.clip(X[:, 33:42:3], 0, None)); w = w - np.percentile(w, 10, axis=0, keepdims=True)
    a = np.clip(X[:, 0:18], 0, None).reshape(-1, 3, 6)
    TA = np.c_[np.log1p(a.sum(1)), np.log1p(a.sum(2)), np.log1p(a.sum((1, 2)))].astype(np.float32)
    TR = np.log1p(a).reshape(-1, 18).astype(np.float32)
    es = np.sort(e, axis=1)
    TE = np.c_[e.mean(1), es[:, 2], es[:, 0], w.mean(1), w.max(1)].astype(np.float32)
    return TA, TR, TE

def _win(T, c, O):
    return T[np.clip(c[:, None] + O[None, :], 0, len(T) - 1)].reshape(len(c), -1)

def tap_score(df, tm, group, chunk=20000):
    """tap-teacher probability on the 10 ms grid (frame k at k*10 ms), or None if the recording is too short"""
    TA, TR, TE = tap_traces(df); n = len(TA)
    if n < 80: return None, -1
    out = np.zeros(n); fold = -1
    for a in range(0, n, chunk):
        c = np.arange(a, min(n, a + chunk)); X = np.c_[_win(TA, c, O_A), _win(TR, c, O_R), _win(TE, c, O_E)]
        out[c], fold = tm.predict(X, group)
    return out, fold

# ------------------------------------------------------------------------------------------------ channel state (claude_research_lp)
CONTACT_FEATS = None
def channel_features(df):
    t = df['Timestamp'].to_numpy(np.float64); t = t - t[0]
    g = np.arange(0, t[-1] + G, G); idx = np.clip(np.searchsorted(t, g, side='right') - 1, 0, len(t) - 1)
    col = lambda name: np.stack([df[name.format(c=c)].to_numpy(np.float64)[idx] for c in (1, 2, 3)], 1)
    lw = np.log10(col('p_core->wl[CH{c}]') + 1.0); wh, wt = col('p_core->wl_head[CH{c}]'), col('p_core->wl_tail[CH{c}]')
    pi = np.stack([lin(df[[f'p_core->emg_psd[0][CH{c}][{z}]' for z in range(5)]].to_numpy()[idx]) for c in (1, 2, 3)], 1)
    pm = np.stack([lin(df[[f'p_core->emg_psd_mean[CH{c}][{z}]' for z in range(5)]].to_numpy()[idx]) for c in (1, 2, 3)], 1)
    Pi, Pm = db(pi.sum(2)), db(pm.sum(2)); f = pm / np.maximum(pm.sum(2, keepdims=True), 1e-9); lz = db(pm)
    F = dict(lw=lw, ht=np.log10((wh + 1.0) / (wt + 1.0)), wlq=lw - Pm / 20.0, Pi=Pi, Pm=Pm, dPim=Pi - Pm)
    for z in range(5): F[f'f{z}'] = f[:, :, z]
    F['r01'] = lz[:, :, 0] - lz[:, :, 1]; F['r21'] = lz[:, :, 2] - lz[:, :, 1]; F['r32'] = lz[:, :, 3] - lz[:, :, 2]; F['r43'] = lz[:, :, 4] - lz[:, :, 3]
    roll = lambda fn, x, n: np.stack([past(x[:, c], n, fn) for c in range(3)], 1)
    F['sd1'] = roll('std', lw, 100); F['rng1'] = roll('max', lw, 100) - roll('min', lw, 100)
    F['min3'] = roll('min', lw, 300); F['max3'] = roll('max', lw, 300); F['sd3'] = roll('std', lw, 300)
    F['min6'] = roll('min', lw, 600); F['min10'] = roll('min', lw, 1000); F['max10'] = roll('max', lw, 1000); F['sd10'] = roll('std', lw, 1000)
    F['rise10'] = lw - F['min10']; F['Pmin10'] = roll('min', Pm, 1000); F['Prise10'] = Pm - F['Pmin10']
    return F, g, t

def channel_labels(F, cm, group):
    n = len(F['lw']); lab = np.full((n, 3), -1, np.int8); sc = np.zeros((n, 3), np.float32)
    for c in range(3):
        X = np.stack([F[k][:, c] for k in cm.spec['features']], 1); s, _ = cm.predict(X, group); lw = F['lw'][:, c]; A = 20 * lw
        off = (s >= P['OFF_T']) | (lw < P['LW_DEAD']); on = (s < P['ON_T']) & (lw >= P['LW_OK']); on[:P['WARMUP']] = False
        if on.sum() >= P['MIN_GOOD_FRAMES']:
            fl = np.percentile(A[on], 5); lab[on & (A <= fl + P['REST_DB']), c] = 0; lab[on & (A >= fl + P['ACT_DB']), c] = 1
        lab[off, c] = 2; lab[off & ~drop_short_runs(off, P['MIN_OFF_RUN']), c] = -1; lab[:P['WARMUP'], c] = -1; sc[:, c] = s
    return lab, sc

# ------------------------------------------------------------------------------------------------ pointer presses
def raw_presses(df, t):
    """pointer presses on source rows exactly as the tap research: (down_ms, release_ms)"""
    if 'PointerState' not in df: return np.zeros(0), np.zeros(0)
    p = np.nan_to_num(df['PointerState'].to_numpy(np.float64)).astype(int); d = np.diff(p)
    up = np.where(d > 0)[0] + 1; dn = np.where(d < 0)[0] + 1
    rel = np.array([t[dn[dn > u][0]] if (dn > u).any() else t[-1] for u in up])
    return t[up].astype(float), rel.astype(float)

def hold_presses(df, t):
    """pointer presses exactly as the long-press research: release = last high row, bounces < 80 ms merged"""
    if 'PointerState' not in df: return np.zeros(0), np.zeros(0)
    p = (np.nan_to_num(df['PointerState'].to_numpy(np.float64)) > 0).astype(int)
    if not p.any(): return np.zeros(0), np.zeros(0)
    e = np.diff(np.r_[0, p, 0]); up = np.where(e > 0)[0]; dn = np.where(e < 0)[0] - 1; out = []
    for u, d in zip(up, dn):
        if out and t[u] - out[-1][1] < P['BOUNCE_MS']: out[-1][1] = t[d]
        else: out.append([t[u], t[d]])
    a = np.array(out, float); return a[:, 0], a[:, 1]

def merge_bounces(up, dn):
    if not len(up): return up, dn
    keep = np.r_[True, (up[1:] - dn[:-1]) >= P['BOUNCE_MS']]; g_ = np.cumsum(keep) - 1
    return up[keep], np.array([dn[g_ == k].max() for k in range(g_.max() + 1)])

# ------------------------------------------------------------------------------------------------ tap events (claude_research_tap label3)
def _assign(up, t, h, tol):
    out = np.full(len(up), -1)
    if not len(up) or not len(t): return out
    D = np.abs(t[None, :] - up[:, None]); ok = D <= tol
    r, c = linear_sum_assignment(np.where(ok, -(h[None, :] - D / 1500.0) - 1.0, 1e3))
    for a, b in zip(r, c):
        if ok[a, b]: out[a] = b
    return out

def tap_events(score, up, dn, spec):
    thr = {k: float(v) for k, v in spec['thr_at_neg_false_per_min'].items()}
    s = uniform_filter1d(score, 5); pk, _ = find_peaks(np.r_[0, s, 0], distance=10, height=P['TAP_HMIN']); pk = pk - 1
    t = (pk * G).astype(float); h = s[pk]; t0, t1 = 0.0, (len(s) - 1) * G
    up, dn = merge_bounces(up, dn); ev = []
    a = _assign(up, t, h, P['TAP_TOL']); far = np.zeros(len(up), bool); used = np.zeros(len(t), bool); used[a[a >= 0]] = True
    un = np.where((a < 0) | (np.where(a >= 0, h[np.clip(a, 0, None)], 0) < 0.05))[0] if len(t) else np.array([], int)
    fr = np.where(~used & (h >= thr['2']))[0] if len(t) else np.array([], int)
    if len(un) and len(fr):
        fr = np.array([j for j in fr if not any((up[k] < t[j] < dn[k]) for k in range(len(up)) if k not in un)], int)
    if len(un) and len(fr):
        for q, b in zip(un, _assign(up[un], t[fr], h[fr], P['TAP_TOL2'])):
            if b >= 0:
                if a[q] >= 0: used[a[q]] = False
                a[q] = fr[b]; used[fr[b]] = True; far[q] = True
    cen = np.array(spec['calibration']['centers']); pr = np.array(spec['calibration']['p_real'])
    for k, (u, d) in enumerate(zip(up, dn)):
        j = a[k]; sc = h[j] if j >= 0 else 0.0; on = t[j] if j >= 0 else u
        edge = (u - t0 < P['TAP_EDGE_MS']) or (t1 - u < P['TAP_EDGE_MS'])
        ev.append(dict(k=k, tier='EDGE' if (edge and sc < 0.5) else 'PRESS', t_ptr=u, t_rel=d, t_on=on,
                       t_off=max(on + 40, d + (on - u)) if not far[k] else max(on + 40, on + (d - u)), score=float(sc), far=bool(far[k]),
                       p_real=float(np.interp(sc, cen, pr))))
    for j in np.where(~used)[0]:
        du = np.abs(up - t[j]).min() if len(up) else 1e9; inside = ((up < t[j]) & (dn > t[j])).any()
        ev.append(dict(k=-1, tier='DUP' if du <= P['TAP_TOL'] else ('INPRESS' if inside else 'ORPHAN'), t_ptr=np.nan, t_rel=np.nan,
                       t_on=t[j], t_off=t[j] + 150, score=float(h[j]), far=False, p_real=np.nan))
    return pd.DataFrame(ev), thr

# ------------------------------------------------------------------------------------------------ long press (claude_research_lp)
def hold_labels(up, dn, lab, g, kind, trust_pointer):
    n = len(g); hold = np.zeros(n, np.int8); hms = np.full(n, -1, np.int32); pid = np.full(n, -1, np.int32)
    act = drop_short_runs((lab == 1).any(1), P['MIN_ACT_RUN']); seen = ((lab == 0) | (lab == 1)).any(1); rise = g[1:][act[1:] & ~act[:-1]]
    sl = lambda a, b: slice(max(0, int(np.ceil(a / G))), max(0, min(n, int(np.floor(b / G)) + 1)))
    info = []
    for k, (u, d) in enumerate(zip(up, dn)):
        w = sl(u - P['CONF_BEFORE'], u + P['CONF_AFTER'])
        ok = 1 if act[w].any() else (0 if seen[w].any() else -1); is_long = (d - u) >= P['MIN_HOLD_MS']
        r = rise[(rise >= u - P['CONF_BEFORE']) & (rise <= u + P['ONSET_AFTER'])]
        on = float(r[np.abs(r - u).argmin()]) if (len(r) and ok == 1) else u - P['DEFAULT_LEAD']
        use = kind == 'hold' and is_long and (ok == 1 or trust_pointer)
        info.append(dict(k=k, t_ptr=float(u), t_rel=float(d), t_on=on, confirmed=ok, long=bool(is_long), held=bool(use)))
        pid[sl(min(on, u), d)] = k
        if kind in ('tap', 'swipe'): continue
        hold[sl(min(on, u) - (P['G_START'] if use else P['CONF_BEFORE']), d + P['G_REL_AFTER'])] = -1
    for x in info:
        if x['held'] and x['t_rel'] - P['G_REL_BEFORE'] > x['t_on']:
            q = sl(x['t_on'], x['t_rel'] - P['G_REL_BEFORE']); hold[q] = 1; hms[q] = (g[q] - x['t_on']).astype(np.int32)
    return hold, hms, pid, pd.DataFrame(info)

# ------------------------------------------------------------------------------------------------ one recording
def label_recording(path, tm, cm, kind='auto', opt=None):
    opt = opt or {}
    kind = recording_kind(path, kind)
    if kind is None: raise ValueError('cannot tell the recording type from its folder; pass --kind tap|swipe|hold|none')
    df, sha = read_recording(path)
    ts = df['Timestamp'].to_numpy(np.int64); t = (ts - ts[0]).astype(float); nrow = len(t)
    # channel state on its 10 ms grid
    F, g, _ = channel_features(df); lab, sc = channel_labels(F, cm, None); row = np.clip(np.searchsorted(g, t, side='right') - 1, 0, len(g) - 1)
    n_off = (lab == 2).sum(1).astype(np.int8)
    out = pd.DataFrame({'Timestamp': ts}); out['gesture_kind'] = np.int8(KIND_CODE[kind])
    tap = np.zeros(nrow, np.int8); tact = np.zeros(nrow, np.int8); teid = np.full(nrow, -1, np.int32)
    up, dn = raw_presses(df, t); events = []
    m_ = lambda lo, hi: (t >= lo) & (t <= hi)
    # --- taps
    if kind in ('tap', 'swipe'):
        score, tfold = tap_score(df, tm, sha)          # the fold model that never saw this recording
        if score is None:
            tap[:] = -1; tact[:] = -1
        else:
            E, thr = tap_events(score, up, dn, tm.spec); T_HI, T_ORPH = thr['1'], thr['5']
            if len(E):
                Pp = E[E.tier.isin(['PRESS', 'EDGE'])]; extra = E[E.tier.isin(['ORPHAN', 'INPRESS']) & (E.score >= T_ORPH)]
                for x in Pp.itertuples():
                    q = m_(min(x.t_ptr, x.t_on) - 150, max(x.t_rel, x.t_off) + 150); tap[q] = -1; tact[q] = -1
                for x in extra.itertuples():
                    q = m_(x.t_on - 200, x.t_on + 250); tap[q] = -1; tact[q] = -1
                good = (E.tier == 'PRESS') & ((E.score >= T_HI) | (opt.get('use_weak') & (E.p_real >= 0.5)))
                for x in E[good].itertuples():
                    tap[m_(x.t_on + P['TAP_WIN'][0], x.t_on + P['TAP_WIN'][1])] = 1; q2 = m_(x.t_on, x.t_off); tact[q2] = 1; teid[q2] = x.k
                if opt.get('promote_double') and 'doubletaps' in os.path.basename(path).lower():
                    ok_on = E[good].t_on.values
                    for x in extra[extra.score >= T_HI].itertuples():
                        if len(ok_on) and 150 <= np.abs(ok_on - x.t_on).min() <= 450:
                            tap[m_(x.t_on + P['TAP_WIN'][0], x.t_on + P['TAP_WIN'][1])] = 1; tact[m_(x.t_on, x.t_on + 120)] = 1
                E['labeled'] = np.where(good, 1, np.where(E.tier.isin(['PRESS', 'EDGE']) | E.index.isin(extra.index), -1, 0))
                E['type'] = np.where(E.tier.isin(['PRESS', 'EDGE']), 'tap_press', 'tap_sensor_only')
                events.append(E)
    # --- long presses
    # holds use the long-press research's press extraction; elsewhere only presses with a real down-edge count
    # (some negative recordings have PointerState stuck high from the first row to the last)
    hu, hd = hold_presses(df, t) if kind == 'hold' else merge_bounces(up, dn)
    hold_g, hms_g, hpid_g, H = hold_labels(hu, hd, lab, g, kind, opt.get('trust_pointer', False))
    hold = hold_g[row]; hms = np.where(hold == 1, hms_g[row] + (t - g[row]).astype(np.int32), -1).astype(np.int32); hpid = hpid_g[row]
    if kind == 'none':                                                # negative folders: every row is 0, no exception
        tap[:] = 0; tact[:] = 0; teid[:] = -1; hold[:] = 0; hms[:] = -1; hpid[:] = -1   # a pointer press changes nothing
    if kind == 'hold' and len(H):
        H['type'] = 'hold_press'; H['labeled'] = np.where(H.held, 1, -1); events.append(H)
        lo, hi = P['HOLD_TAP_MASK']; rlo, rhi = P['HOLD_TAP_MASK_REL']   # press starts (and some releases) look like taps
        for x in H.itertuples():
            q = m_(min(x.t_on, x.t_ptr) + lo, x.t_ptr + hi) | m_(x.t_rel + rlo, x.t_rel + rhi); tap[q] = -1; tact[q] = -1
    # --- disconnection rule
    rule = opt.get('disconnect_rule', 'all'); bad = n_off[row] >= P['MIN_CH_OFF']
    if rule in ('all', 'taps'): tap[bad] = 0; tact[bad] = 0; teid[bad] = -1
    if rule == 'all': hold[bad] = 0; hms[bad] = -1
    if events:
        for E in events:
            on_row = np.clip(np.searchsorted(t, E.t_on.values), 0, nrow - 1); E['n_ch_off_at_onset'] = n_off[row][on_row]
            E['rule_disconnect'] = (E.n_ch_off_at_onset >= P['MIN_CH_OFF']) & (rule != 'off') & (E.type.str.startswith('tap') | (rule == 'all'))
            E.loc[E.rule_disconnect, 'labeled'] = 0
    out['tap_label'] = tap; out['tap_active'] = tact; out['tap_event_id'] = teid
    out['hold_label'] = hold; out['hold_ms'] = hms; out['hold_press_id'] = hpid
    for c in range(3): out[f'ch{c + 1}_label'] = lab[row, c]
    for c in range(3): out[f'ch{c + 1}_contact'] = sc[row, c].round(3)
    out['n_ch_off'] = n_off[row]
    out['any_active'] = np.where((lab == 1).any(1), 1, np.where((lab == 0).all(1), 0, -1)).astype(np.int8)[row]
    ev = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    st = dict(kind=kind, rows=nrow, tap=np.bincount(tap + 1, minlength=3).tolist(), hold=np.bincount(hold + 1, minlength=3).tolist(),
              taps_labeled=int((ev.labeled == 1).sum()) if len(ev) and 'type' in ev and (ev.type == 'tap_press').any() else 0,
              holds_labeled=int(((ev.type == 'hold_press') & (ev.labeled == 1)).sum()) if len(ev) else 0,
              disconnect_rule_events=int(ev.rule_disconnect.sum()) if len(ev) else 0,
              rows_disconnected=int(bad.sum()),
              channels={f'ch{c + 1}': np.bincount(lab[row, c] + 1, minlength=4).tolist() for c in range(3)})
    return out, ev, st

# ------------------------------------------------------------------------------------------------ main
_WORKER = {}
def _init_worker(tap_win):
    """A labelling process: its own copy of the models, one thread each (the outputs do not depend on it)."""
    P['TAP_WIN'] = tuple(tap_win)
    _WORKER['tm'] = tap_teacher(); _WORKER['cm'] = contact_models()

def _label_job(job):
    """One recording in a worker: (rel, summary, events, error), the sidecar written as in the serial loop."""
    f, rel, dst, kind, opt = job
    try:
        out, ev, st = label_recording(f, _WORKER['tm'], _WORKER['cm'], kind, opt); out.to_csv(dst, index=False)
        return rel, st, ev, None
    except Exception as e:
        return rel, None, None, repr(e)

def main():
    ap = argparse.ArgumentParser(description='Offline y_true for taps, swipes, long presses and per-channel EMG state.')
    ap.add_argument('input', help='a .csv/.csv.gz recording, or a folder searched recursively')
    ap.add_argument('out', help='output folder')
    ap.add_argument('--kind', default='auto', choices=['auto', 'tap', 'swipe', 'hold', 'none'])
    ap.add_argument('--disconnect-rule', default='all', choices=['all', 'taps', 'off'],
                    help='where >=2 channels are confidently OFF: no tap and no hold (all), no tap only (taps), nothing (off)')
    ap.add_argument('--tap-win', nargs=2, type=int, default=list(P['TAP_WIN']), help='tap onset window in ms after the onset')
    ap.add_argument('--use-weak', action='store_true', help='also label weaker taps (p_real >= 0.5)')
    ap.add_argument('--promote-double-partners', action='store_true', help='label unpressed second taps in DoubleTaps recordings')
    ap.add_argument('--trust-pointer', action='store_true', help='label long presses as held even when the sensors do not confirm them')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('MUDRA_LABEL_WORKERS', '0') or 0),
                    help='labelling processes (0 = one per CPU, at most 16); the labels do not depend on it')
    a = ap.parse_args(); P['TAP_WIN'] = tuple(a.tap_win)
    opt = dict(disconnect_rule=a.disconnect_rule, use_weak=a.use_weak, promote_double=a.promote_double_partners, trust_pointer=a.trust_pointer)
    if os.path.isdir(a.input):
        files = sorted(set(glob.glob(os.path.join(a.input, '**', '*.csv.gz'), recursive=True) + glob.glob(os.path.join(a.input, '**', '*.csv'), recursive=True)))
        root = a.input
    else:
        files = [a.input]; root = os.path.dirname(a.input) or '.'
    os.makedirs(a.out, exist_ok=True); summary = {}; failed = {}; evs = []
    jobs = []
    for f in files:
        rel = os.path.relpath(f, root); dst = os.path.join(a.out, re.sub(r'\.csv(\.gz)?$', '', rel) + '.labels.csv.gz')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        jobs.append((f, rel, dst, a.kind, opt))
    workers = min(a.workers if a.workers > 0 else min(16, os.cpu_count() or 1), len(jobs))
    if workers <= 1:                                  # one process, as before
        _init_worker(P['TAP_WIN']); results = map(_label_job, jobs); pool = None
    else:                                             # recordings are independent: same labels, in file order
        for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=_init_worker,
                                   initargs=(list(P['TAP_WIN']),))
        results = pool.map(_label_job, jobs, chunksize=4)
        print(f'labelling {len(jobs)} recordings with {workers} processes', flush=True)
    for k, (rel, st, ev, err) in enumerate(results):
        if err is None:
            summary[rel] = st
            if len(ev): ev.insert(0, 'recording', rel); evs.append(ev)
        else:
            failed[rel] = err
        if (k + 1) % 200 == 0: print(f'{k + 1}/{len(files)}', flush=True)
    if pool is not None: pool.shutdown()
    if evs: pd.concat(evs, ignore_index=True).round(4).to_csv(os.path.join(a.out, '_events.csv.gz'), index=False)
    tot = dict(files=len(summary), failed=len(failed), rows=sum(s['rows'] for s in summary.values()),
               taps_labeled=sum(s['taps_labeled'] for s in summary.values()), holds_labeled=sum(s['holds_labeled'] for s in summary.values()),
               events_disconnect_rule=sum(s['disconnect_rule_events'] for s in summary.values()),
               rows_disconnected=sum(s['rows_disconnected'] for s in summary.values()),
               args={k: v for k, v in vars(a).items() if k not in ('input', 'out', 'workers')}, teacher=TEACHER)
    json.dump(dict(totals=tot, failed=failed, recordings=summary), open(os.path.join(a.out, '_summary.json'), 'w'), indent=1)
    print(json.dumps(tot))
    if failed: print('FAILED:', json.dumps(failed, indent=1)); sys.exit(1)

if __name__ == '__main__':
    main()
