#!/usr/bin/env python3
"""Offline labels (y_true) for Mudra long-press recordings.  One script, no training step.

    python claude_research_lp/label_long_press.py DATA/DATA_LONG_PRESS  DATA/DATA_LONG_PRESS_LABELS
    python claude_research_lp/label_long_press.py some_recording.csv.gz  out_dir  --kind hold

Needs: numpy pandas scipy lightgbm, and the folder `models/` that sits next to this script.

Output: one `<name>.labels.csv.gz` per recording (same sub-folder layout as the input), one row per source row, same order:

  Timestamp          copied from the source
  ch1_label ..       per-channel state:  0 REST   1 ACTIVE (muscle activity)   2 OFF (electrode not in proper contact)   -1 unknown
  ch1_contact ..     contact-problem score 0..1 behind the OFF decision
  any_active         1 if any channel is ACTIVE, 0 if every channel is REST, -1 otherwise
  hold_label         long-press state:  1 held   0 not held   -1 unknown
  hold_ms            ms since the labeled hold started (rows with hold_label 1), -1 elsewhere.
                     Use it to define your own target, e.g. y = hold_label==1 and hold_ms >= 500 for "fire after 500 ms".
  press_id           index of the pointer press this row belongs to, -1 outside presses
  press_confirmed    1 sensors show activity at the start of that press, 0 they do not,
                     -1 cannot tell (no channel with usable contact) or outside presses

Train on rows where the label you use is >= 0.  -1 means "do not put this row in the loss", not "negative".

How the labels are made (this is an OFFLINE labeler: it uses the whole recording, it is not the firmware model)
  1. Firmware columns are resampled to a 10 ms grid (zero-order hold).
  2. Contact score per channel: a LightGBM model trained on the contact protocol (CONNECTED / DISCONNECTED / LOOSE / TABLE).
     Five fold models are shipped; a recording is scored by the model that never saw that person. Unknown people get the average.
  3. OFF     = contact score >= 0.5, or WL below the amplifier-floor region (log10 WL < 2.2), held for at least 1 s.
     Shorter OFF runs become unknown (they are 1.3% of channel time in CONNECTED recordings and 0% after this rule).
     Good contact = contact score < 0.2 and log10 WL >= 2.35.
  4. Floor   = 5th percentile of the level 20*log10(WL) over good-contact frames of that recording and channel.
     REST    = good contact and level <= floor + 3 dB.   ACTIVE = good contact and level >= floor + 6 dB.   Between: unknown.
  5. The recording type comes from its folder (the folder is the ground truth):
       hold  : SOFT_/MID_/STRONG_LONG_PRESS, LONG_PRESS, DRAG_AND_DROP  -> pointer presses can be holds
       tap   : TAP, SWIPE                                               -> no holds; every row is hold_label 0
       none  : CONNECTED, DISCONNECTED, LOOSE, TABLE, NOTHING           -> no holds; every row is 0, a stray press is unknown
     For other folders or a single file pass --kind. Duration alone cannot do this: 7% of swipes and 2% of taps last over 1 s.
  6. In hold recordings, hold_label comes from PointerState, corrected with the sensors:
       - pointer bounces (re-press within 80 ms) are merged; presses shorter than 400 ms are not long presses -> unknown
       - a press is confirmed if some channel is ACTIVE for at least 30 ms between 300 ms before and 500 ms after pointer-down
         (by chance this happens in 3.7% of 800 ms windows at rest, against 84-96% at real long presses)
       - the hold starts at that activity onset (the sensors lead the pointer by about 100 ms), not at pointer-down
       - 150 ms before the start, and from 250 ms before to 150 ms after the pointer release, are unknown
       - unconfirmed presses are unknown for their whole length (--trust-pointer labels them held from pointer-down - 100 ms)
  The first second of every recording is unknown for the channel labels (history not filled).

Known limits: see claude_research_lp/LONG_PRESS_CHANNEL_STATE_RESEARCH_2026-11-17.md
"""
import sys, os, re, json, glob, argparse
import numpy as np, pandas as pd
from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d

G = 10                      # ms grid
OFF_T, ON_T = 0.5, 0.2      # contact score thresholds
LW_DEAD, LW_OK = 2.2, 2.35  # log10(WL): below 2.2 is amplifier floor, good contact needs >= 2.35
REST_DB, ACT_DB = 3.0, 6.0  # dB above the recording floor
MIN_GOOD_FRAMES = 200       # 2 s of good contact needed to estimate a floor
WARMUP = 100                # frames (1 s) at the start with unknown channel labels
MIN_OFF_RUN = 100           # frames (1 s): shorter OFF runs become unknown
BOUNCE_MS, MIN_HOLD_MS = 80, 400
CONF_BEFORE, CONF_AFTER = 300, 500          # window around pointer-down in which activity confirms a press
ONSET_AFTER = 200                           # the activity onset used as hold start must lie in [down-300, down+200]
MIN_ACT_RUN = 3                             # frames (30 ms): shorter ACTIVE bursts do not confirm a press
DEFAULT_LEAD = 100                          # ms the sensors lead the pointer when no clean onset edge is found
HOLD_FOLDERS = {'SOFT_LONG_PRESS', 'MID_LONG_PRESS', 'STRONG_LONG_PRESS', 'LONG_PRESS', 'DRAG_AND_DROP'}
TAP_FOLDERS = {'TAP', 'SWIPE'}
NONE_FOLDERS = {'CONNECTED', 'DISCONNECTED', 'LOOSE', 'TABLE', 'NOTHING'}
G_START, G_REL_BEFORE, G_REL_AFTER = 150, 250, 150


def lin(v):                 # firmware code = 10000*log10(1+power)
    return np.power(10.0, np.asarray(v, np.float64) / 10000.0) - 1.0

def db(p):
    return 10 * np.log10(np.maximum(p, 1e-4))

def past(x, n, fn):
    """causal statistic over the last n frames, current frame included"""
    o = (n - 1) // 2
    if fn == 'min': return minimum_filter1d(x, n, origin=o, mode='nearest')
    if fn == 'max': return maximum_filter1d(x, n, origin=o, mode='nearest')
    mu = uniform_filter1d(x, n, origin=o, mode='nearest')
    if fn == 'mean': return mu
    return np.sqrt(np.maximum(uniform_filter1d(x * x, n, origin=o, mode='nearest') - mu * mu, 0))

def channel_features(df):
    """feature name -> array (frames, 3 channels) on the 10 ms grid; also the grid and the row times (ms from first row)"""
    t = df['Timestamp'].to_numpy(np.float64); t = t - t[0]
    g = np.arange(0, t[-1] + G, G); idx = np.clip(np.searchsorted(t, g, side='right') - 1, 0, len(t) - 1)
    ch = (1, 2, 3); F = {}
    col = lambda name: np.stack([df[name.format(c=c)].to_numpy(np.float64)[idx] for c in ch], 1)
    lw = np.log10(col('p_core->wl[CH{c}]') + 1.0)
    wh, wt = col('p_core->wl_head[CH{c}]'), col('p_core->wl_tail[CH{c}]')
    pi = np.stack([lin(df[[f'p_core->emg_psd[0][CH{c}][{z}]' for z in range(5)]].to_numpy()[idx]) for c in ch], 1)
    pm = np.stack([lin(df[[f'p_core->emg_psd_mean[CH{c}][{z}]' for z in range(5)]].to_numpy()[idx]) for c in ch], 1)
    Pi, Pm = db(pi.sum(2)), db(pm.sum(2)); f = pm / np.maximum(pm.sum(2, keepdims=True), 1e-9); lz = db(pm)
    F['lw'] = lw; F['ht'] = np.log10((wh + 1.0) / (wt + 1.0)); F['wlq'] = lw - Pm / 20.0; F['Pi'] = Pi; F['Pm'] = Pm; F['dPim'] = Pi - Pm
    for z in range(5): F[f'f{z}'] = f[:, :, z]
    F['r01'] = lz[:, :, 0] - lz[:, :, 1]; F['r21'] = lz[:, :, 2] - lz[:, :, 1]; F['r32'] = lz[:, :, 3] - lz[:, :, 2]; F['r43'] = lz[:, :, 4] - lz[:, :, 3]
    roll = lambda fn, x, n: np.stack([past(x[:, c], n, fn) for c in range(3)], 1)
    F['sd1'] = roll('std', lw, 100); F['rng1'] = roll('max', lw, 100) - roll('min', lw, 100)
    F['min3'] = roll('min', lw, 300); F['max3'] = roll('max', lw, 300); F['sd3'] = roll('std', lw, 300)
    F['min6'] = roll('min', lw, 600); F['min10'] = roll('min', lw, 1000); F['max10'] = roll('max', lw, 1000); F['sd10'] = roll('std', lw, 1000)
    F['rise10'] = lw - F['min10']; F['Pmin10'] = roll('min', Pm, 1000); F['Prise10'] = Pm - F['Pmin10']
    return F, g, t


class ContactModel:
    def __init__(self, model_dir):
        import lightgbm as lgb
        spec = json.load(open(os.path.join(model_dir, 'contact_folds.json')))
        self.features = spec['features']; self.fold_of = spec['person_group_fold']
        self.models = [lgb.Booster(model_file=os.path.join(model_dir, f'contact_fold{k}.txt')) for k in range(5)]
    @staticmethod
    def group(name):
        m = re.match(r'[a-z]+', os.path.basename(name).lower()); return m.group(0)[:4] if m else ''
    def score(self, X, name):
        k = self.fold_of.get(self.group(name))
        if k is not None: return self.models[k].predict(X), k
        return np.mean([m.predict(X) for m in self.models], 0), -1


def channel_labels(F, cm, name):
    n = len(F['lw']); lab = np.full((n, 3), -1, np.int8); sc = np.zeros((n, 3), np.float32); fold = -1
    for c in range(3):
        X = np.stack([F[k][:, c] for k in cm.features], 1); s, fold = cm.score(X, name); lw = F['lw'][:, c]; A = 20 * lw
        off = (s >= OFF_T) | (lw < LW_DEAD); on = (s < ON_T) & (lw >= LW_OK); on[:WARMUP] = False
        if on.sum() >= MIN_GOOD_FRAMES:
            floor = np.percentile(A[on], 5); lab[on & (A <= floor + REST_DB), c] = 0; lab[on & (A >= floor + ACT_DB), c] = 1
        lab[off, c] = 2
        short = off & ~drop_short_runs(off, MIN_OFF_RUN); lab[short, c] = -1
        lab[:WARMUP, c] = -1; sc[:, c] = s
    return lab, sc, fold


def presses(ptr, t):
    """pointer presses as (down_ms, release_ms), bounces merged"""
    p = (np.asarray(ptr) > 0).astype(int)
    if not p.any(): return []
    e = np.diff(np.r_[0, p, 0]); up = np.where(e > 0)[0]; dn = np.where(e < 0)[0] - 1
    out = []
    for u, d in zip(up, dn):
        if out and t[u] - out[-1][1] < BOUNCE_MS: out[-1][1] = t[d]
        else: out.append([t[u], t[d]])
    return out


def recording_kind(path, forced='auto'):
    if forced != 'auto': return forced
    parts = {p.upper() for p in os.path.normpath(os.path.abspath(path)).split(os.sep)}
    if parts & HOLD_FOLDERS: return 'hold'
    if parts & TAP_FOLDERS: return 'tap'
    if parts & NONE_FOLDERS: return 'none'
    return None


def drop_short_runs(a, k):
    a = np.asarray(a, bool).copy()
    if k <= 1 or not a.any(): return a
    e = np.diff(np.r_[0, a.astype(int), 0]); s0 = np.where(e > 0)[0]; s1 = np.where(e < 0)[0]
    for x, y in zip(s0, s1):
        if y - x < k: a[x:y] = False
    return a


def hold_labels(pr, lab, g, kind, trust_pointer=False):
    """hold label on the 10 ms grid from pointer presses and channel labels"""
    n = len(g); hold = np.zeros(n, np.int8); hms = np.full(n, -1, np.int32); pid = np.full(n, -1, np.int32); conf = np.full(n, -1, np.int8)
    act = drop_short_runs((lab == 1).any(1), MIN_ACT_RUN); seen = ((lab == 0) | (lab == 1)).any(1); rise = g[1:][act[1:] & ~act[:-1]]
    sl = lambda a, b: slice(max(0, int(np.ceil(a / G))), max(0, min(n, int(np.floor(b / G)) + 1)))
    info = []
    for k, (u, d) in enumerate(pr):
        w = sl(u - CONF_BEFORE, u + CONF_AFTER)
        ok = 1 if act[w].any() else (0 if seen[w].any() else -1); is_long = (d - u) >= MIN_HOLD_MS
        r = rise[(rise >= u - CONF_BEFORE) & (rise <= u + ONSET_AFTER)]
        on = float(r[np.abs(r - u).argmin()]) if (len(r) and ok == 1) else u - DEFAULT_LEAD
        use = kind == 'hold' and is_long and (ok == 1 or trust_pointer)
        info.append(dict(press=k, down_ms=float(u), release_ms=float(d), onset_ms=on, confirmed=ok, long=bool(is_long), held=bool(use)))
        pid[sl(min(on, u), d)] = k; conf[sl(min(on, u), d)] = ok
        if kind == 'tap': continue                                   # taps and swipes are never holds: stay 0
        hold[sl(min(on, u) - (G_START if use else CONF_BEFORE), d + G_REL_AFTER)] = -1
    for x in info:
        if x['held'] and x['release_ms'] - G_REL_BEFORE > x['onset_ms']:
            q = sl(x['onset_ms'], x['release_ms'] - G_REL_BEFORE); hold[q] = 1; hms[q] = (g[q] - x['onset_ms']).astype(np.int32)
    return hold, hms, pid, conf, info


def label_recording(path, cm, kind='auto', trust_pointer=False):
    k_ = recording_kind(path, kind)
    if k_ is None: raise ValueError('cannot tell the recording type from its folder; pass --kind hold|tap|none')
    df = pd.read_csv(path, low_memory=False)
    F, g, t = channel_features(df); n = len(g)
    lab, sc, fold = channel_labels(F, cm, path)
    pr = presses(df['PointerState'].to_numpy(), t) if 'PointerState' in df else []
    hold, hms, pid, conf, info = hold_labels(pr, lab, g, k_, trust_pointer)
    row = np.clip(np.searchsorted(g, t, side='right') - 1, 0, n - 1)
    out = pd.DataFrame({'Timestamp': df['Timestamp'].values})
    for c in range(3): out[f'ch{c + 1}_label'] = lab[row, c]
    for c in range(3): out[f'ch{c + 1}_contact'] = sc[row, c].round(3)
    any_act = np.where((lab == 1).any(1), 1, np.where((lab == 0).all(1), 0, -1)).astype(np.int8)
    out['any_active'] = any_act[row]; out['hold_label'] = hold[row]
    out['hold_ms'] = np.where(hold[row] == 1, hms[row] + (t - g[row]).astype(np.int32), -1).astype(np.int32)
    out['press_id'] = pid[row]; out['press_confirmed'] = conf[row]
    stats = dict(rows=len(out), kind=k_, fold=int(fold), presses=len(info), presses_long=int(sum(x['long'] for x in info)),
                 presses_held=int(sum(x['held'] for x in info)),
                 presses_long_confirmed=int(sum(x['long'] and x['confirmed'] == 1 for x in info)),
                 presses_long_unverifiable=int(sum(x['long'] and x['confirmed'] == -1 for x in info)),
                 hold=np.bincount(hold[row] + 1, minlength=3).tolist(),                       # unknown, not held, held
                 channels={f'ch{c + 1}': np.bincount(lab[row, c] + 1, minlength=4).tolist() for c in range(3)})   # unknown, REST, ACTIVE, OFF
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('input', help='a .csv/.csv.gz recording, or a folder searched recursively')
    ap.add_argument('out', help='output folder')
    ap.add_argument('--models', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
    ap.add_argument('--kind', default='auto', choices=['auto', 'hold', 'tap', 'none'], help='recording type; auto = from the folder name')
    ap.add_argument('--trust-pointer', action='store_true', help='label long presses as held even when the sensors do not confirm them')
    a = ap.parse_args(); cm = ContactModel(a.models)
    if os.path.isdir(a.input):
        files = sorted(glob.glob(os.path.join(a.input, '**', '*.csv.gz'), recursive=True) + glob.glob(os.path.join(a.input, '**', '*.csv'), recursive=True)); root = a.input
    else:
        files = [a.input]; root = os.path.dirname(a.input) or '.'
    summary = {}; failed = {}
    for k, f in enumerate(files):
        rel = os.path.relpath(f, root); dst = os.path.join(a.out, re.sub(r'\.csv(\.gz)?$', '', rel) + '.labels.csv.gz')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            out, st = label_recording(f, cm, a.kind, a.trust_pointer); out.to_csv(dst, index=False); summary[rel] = st
        except Exception as e:                      # keep going, report at the end
            failed[rel] = repr(e)
        if (k + 1) % 200 == 0: print(f'{k + 1}/{len(files)}', flush=True)
    tot = dict(files=len(summary), failed=len(failed), rows=sum(s['rows'] for s in summary.values()),
               **{k: sum(s[k] for s in summary.values()) for k in ('presses', 'presses_long', 'presses_held', 'presses_long_confirmed', 'presses_long_unverifiable')})
    tot['args'] = dict(kind=a.kind, trust_pointer=a.trust_pointer)
    json.dump(dict(totals=tot, failed=failed, recordings=summary), open(os.path.join(a.out, '_label_summary.json'), 'w'), indent=1)
    print(json.dumps(tot))
    if failed: print('FAILED:', json.dumps(failed, indent=1)); sys.exit(1)


if __name__ == '__main__':
    main()
