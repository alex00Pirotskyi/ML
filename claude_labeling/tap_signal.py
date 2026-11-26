"""Signal layer of the tap labeller: per-axis, per-zone ACC and per-channel EMG, never mixed.

Every quantity here is computed per ACC axis (X, Y, Z), per ACC zone (acc_psd[0] zones 0-5) or per EMG channel
(CH1-CH3). Nothing is summed across axes or channels. The only sum is the legacy per-axis power, zones 1-5 of ONE
axis (unified_tap_labeler_deterministic.py: POWER_BINS = 1..5 per axis).

Units
  ACC  acc_psd codes are linear power (integers, rest ~1-30, taps 1e3-6e4). Everything is log10(1 + power), so
       "excess" is in decades above the recording's rest level (1 decade = 10x the power).
  EMG  emg_psd codes are 10000*log10(1 + power): code / 1000 is dB.

A jolt is a local peak of the strongest axis' excess (zones 1-5) above the rest level. jolts() returns one row per
jolt with the per-axis / per-zone / per-channel description the trigger's conditions are made of.
"""
from __future__ import annotations

import io
import gzip
import hashlib

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, maximum_filter1d

AXES = "XYZ"
ACC_COLS = [f"p_core->acc_psd[0][{a}][{z}]" for a in AXES for z in range(6)]          # (axis, zone) order
EMG_COLS = [f"p_core->emg_psd[0][CH{c}][{z}]" for c in (1, 2, 3) for z in range(5)]    # (channel, zone) order
WL_COLS = [f"p_core->wl[CH{c}]" for c in (1, 2, 3)]
BASE_COLS = ["Timestamp", "PointerState", "Metronome"]
POWER_ZONES = (1, 2, 3, 4, 5)       # the legacy per-axis power: zones 1-5 of one axis (zone 0 = slowest band)

SMOOTH_ROWS = 3                     # rolling median, as the legacy labeller
QUIET_PCT = 40                      # rest rows: strongest axis below this percentile of the recording ...
POINTER_GUARD_MS = 300              # ... and at least this far from any pointer edge
SCALE_FLOOR = 0.05                  # decades
JOLT_MIN = 0.5                      # decades above rest on the strongest axis (3.2x the power): every jolt we look at
PEAK_SEP_MS = 40                    # two jolts closer than this are one (the higher wins)


def read(path):
    """(signals dict, content SHA-256). Rows with a repeated or backwards timestamp are dropped."""
    raw = gzip.open(path, "rb").read() if str(path).endswith(".gz") else open(path, "rb").read()
    head = pd.read_csv(io.BytesIO(raw), nrows=0).columns
    use = [c for c in BASE_COLS + ACC_COLS + EMG_COLS + WL_COLS if c in head]
    df = pd.read_csv(io.BytesIO(raw), usecols=use, low_memory=False)
    return signals(df), hashlib.sha256(raw).hexdigest()


def signals(df):
    ts = df["Timestamp"].to_numpy(np.int64)
    t = (ts - ts[0]).astype(np.float64)
    n = len(t)
    col = lambda c, d=0.0: (pd.to_numeric(df[c], errors="coerce").fillna(d).to_numpy(np.float64) if c in df
                            else np.full(n, d))
    acc = np.stack([col(c) for c in ACC_COLS], 1).reshape(n, 3, 6)
    emg = np.stack([col(c) for c in EMG_COLS], 1).reshape(n, 3, 5)
    wl = np.stack([col(c) for c in WL_COLS], 1)
    ptr = (col("PointerState") > 0).astype(np.int8)
    met = (col("Metronome") > 0).astype(np.int8)
    return dict(ts=ts, t=t, acc=np.clip(acc, 0, None), emg=emg, wl=wl, ptr=ptr, met=met, n=n)


def presses(t, ptr):
    """Pointer presses (down_ms, up_ms, starts_down): up = the last row still pressed."""
    if not len(ptr) or not ptr.any():
        return np.zeros(0), np.zeros(0), False
    e = np.diff(np.r_[0, ptr.astype(np.int8), 0])
    up = np.flatnonzero(e > 0); dn = np.flatnonzero(e < 0) - 1
    return t[up].astype(float), t[np.minimum(dn, len(t) - 1)].astype(float), bool(ptr[0])


def beats(t, met):
    if not len(met) or not met.any():
        return np.zeros(0)
    e = np.diff(np.r_[0, met.astype(np.int8)])
    return t[np.flatnonzero(e > 0)].astype(float)


def window(t, a, b):
    """row slice of t in [a, b] (ms)"""
    return slice(int(np.searchsorted(t, a, "left")), int(np.searchsorted(t, b, "right")))


def prepare(s, guard_edges=None):
    """Log power per (axis, zone) and per-axis power (zones 1-5), 3-row median, rest levels from quiet rows."""
    t, acc = s["t"], s["acc"]
    L = np.log10(1.0 + acc)                                               # (n, 3, 6) decades
    S = np.log10(1.0 + acc[:, :, list(POWER_ZONES)].sum(2))               # (n, 3): one axis, zones 1-5
    E_db = s["emg"] / 1000.0                                              # (n, 3, 5) dB
    if SMOOTH_ROWS > 1 and len(t) >= SMOOTH_ROWS:
        L = median_filter(L, size=(SMOOTH_ROWS, 1, 1), mode="nearest")
        S = median_filter(S, size=(SMOOTH_ROWS, 1), mode="nearest")
    smax = S.max(1)
    quiet = smax <= np.percentile(smax, QUIET_PCT)
    if guard_edges is not None and len(guard_edges):
        near = np.zeros(len(t), bool)
        for e in guard_edges:
            near[window(t, e - POINTER_GUARD_MS, e + POINTER_GUARD_MS)] = True
        if (quiet & ~near).sum() >= 50:
            quiet = quiet & ~near
    if quiet.sum() < 20:
        quiet = np.ones(len(t), bool)
    med = lambda x: np.median(x[quiet], 0)
    mad = lambda x, m: np.maximum(1.4826 * np.median(np.abs(x[quiet] - m), 0), SCALE_FLOOR)
    bL, bS, bE = med(L), med(S), med(E_db)
    return dict(L=L, S=S, Edb=E_db, bL=bL, bS=bS, sS=mad(S, bS), bE=bE, quiet=quiet,
                ex=S - bS, exL=L - bL, exE=E_db - bE)


def _run(mask, k):
    """contiguous run of True around row k (inclusive bounds), or (k, k) if mask[k] is False"""
    if not mask[k]:
        return k, k
    a = k
    while a > 0 and mask[a - 1]:
        a -= 1
    b = k
    while b + 1 < len(mask) and mask[b + 1]:
        b += 1
    return a, b


def peaks(t, m, thr=JOLT_MIN, sep_ms=PEAK_SEP_MS):
    """local maxima of m (>= thr), at least sep_ms apart (the higher one wins)"""
    if len(m) < 3:
        return np.zeros(0, np.int64)
    cand = np.flatnonzero((m >= thr) & (m >= maximum_filter1d(m, 7, mode="nearest")))
    out = []
    for i in cand:
        if out and t[i] - t[out[-1]] < sep_ms:
            if m[i] > m[out[-1]]:
                out[-1] = i
        else:
            out.append(i)
    return np.asarray(out, np.int64)


def topo_prominence(t, x, k, half_ms=200.0):
    """Topographic prominence of the peak x[k] within +-half_ms: its height above the higher of the two lowest points
    (valleys) between it and the nearest higher point (or the window edge) on each side. Two taps 200 ms apart each
    keep their own prominence (the dip between them counts); a jolt riding on a bigger movement has little.
    Returns (prominence, left valley row, right valley row)."""
    lo = window(t, t[k] - half_ms, t[k]).start
    hi = window(t, t[k], t[k] + half_ms).stop
    left, right = x[lo:k], x[k + 1:hi]
    l0 = lo
    hl = np.flatnonzero(left > x[k])
    if len(hl):
        l0 = lo + hl[-1] + 1; left = left[hl[-1] + 1:]
    hr = np.flatnonzero(right > x[k])
    if len(hr):
        right = right[:hr[0]]
    if not len(left) or not len(right):
        lmin = left.min() if len(left) else x[k]; rmin = right.min() if len(right) else x[k]
        return float(x[k] - max(lmin, rmin)), k, k
    lmin, rmin = left.min(), right.min()
    # where the jolt starts / ends: the last (first) row below 10% of the peak's power (1 decade down), or its valley
    li = l0 + int(np.flatnonzero(left <= max(lmin, x[k] - 1.0))[-1])
    ri = k + 1 + int(np.flatnonzero(right <= max(rmin, x[k] - 1.0))[0])
    return float(x[k] - max(lmin, rmin)), li, ri


def jolts(s, p, thr=JOLT_MIN):
    """One row per jolt: per-axis / per-zone ACC excess at the peak, shape, isolation and per-channel EMG."""
    t, ex, exL, exE = s["t"], p["ex"], p["exL"], p["exE"]
    m = ex.max(1)
    pk = peaks(t, m, thr)
    ech = exE.mean(2)                                                      # (n, 3) EMG per channel, dB above rest
    rows = []
    for k in pk:
        e = ex[k]; d = int(np.argmax(e)); od = np.sort(e)[::-1]
        ed = ex[:, d]; h = ed[k]
        half_a, half_b = _run(ed >= h - 0.30103, k)                       # half power on the dominant axis
        ten_a, ten_b = _run(ed >= max(h - 1.0, 0.2), k)                    # 10% power (or 0.2 decade)
        run = slice(half_a, half_b + 1)
        cons = float(np.mean(np.argmax(ex[run], 1) == d))
        fl = np.r_[np.arange(window(t, t[k] - 200, t[k] - 60).start, window(t, t[k] - 200, t[k] - 60).stop),
                   np.arange(window(t, t[k] + 60, t[k] + 200).start, window(t, t[k] + 60, t[k] + 200).stop)]
        flank = float(m[fl].max()) if len(fl) else 0.0
        pre = m[window(t, t[k] - 250, t[k] - 50)]; post = m[window(t, t[k] + 80, t[k] + 300)]
        zl = exL[k]                                                        # (3, 6) decades per axis / zone
        lin = np.clip(10 ** p["L"][k] - 10 ** p["bL"], 0, None)            # linear excess power per axis / zone
        dz = lin[d, 1:]; share = dz / dz.sum() if dz.sum() > 0 else np.zeros(5)
        zd = zl[d]                                                         # dominant axis, zones 0-5 (decades)
        tp, vl, vr = topo_prominence(t, ed, k)
        r = dict(k=int(k), t=float(t[k]), axis=d, h=float(h), h2=float(od[1]), h3=float(od[2]),
                 w50=float(t[half_b] - t[half_a]), w10=float(t[ten_b] - t[ten_a]),
                 rise=float(t[k] - t[ten_a]), fall=float(t[ten_b] - t[k]), cons=cons,
                 flank=flank, pre=float(pre.max()) if len(pre) else 0.0, post=float(post.max()) if len(post) else 0.0,
                 cen=float((share * np.arange(1, 6)).sum()), tprom=tp, rise_v=float(t[k] - t[vl]), fall_v=float(t[vr] - t[k]),
                 z34=float(zd[3:5].mean()), mid_def=float(max(zd[1], zd[5]) - zd[3:5].mean()))
        for a in range(3):
            r[f"e{AXES[a]}"] = float(e[a])
            for z in range(6):
                r[f"z{AXES[a]}{z}"] = float(zl[a, z])
        for z in range(5):
            r[f"sh{z + 1}"] = float(share[z])
        # EMG per channel (dB above rest, mean over the 5 zones of that channel): at the jolt, before, after, and how
        # long the channel stays >= 3 dB up around it
        w_at = window(t, t[k] - 80, t[k] + 20)
        if w_at.stop <= w_at.start:
            w_at = slice(k, k + 1)
        for c in range(3):
            at = ech[w_at, c]; bef = ech[window(t, t[k] - 300, t[k] - 100), c]
            aft = ech[window(t, t[k] + 100, t[k] + 400), c]
            j = w_at.start + int(np.argmax(at))
            a_, b_ = _run(ech[:, c] >= 3.0, j)
            r[f"m{c + 1}"] = float(at.max())
            r[f"mb{c + 1}"] = float(bef.mean()) if len(bef) else 0.0
            r[f"ma{c + 1}"] = float(aft.mean()) if len(aft) else 0.0
            r[f"md{c + 1}"] = float(t[b_] - t[a_]) if ech[j, c] >= 3.0 else 0.0
            for z in range(5):
                r[f"m{c + 1}z{z}"] = float(exE[w_at, c, z].max())
        rows.append(r)
    return pd.DataFrame(rows)


# ================================================================================================ labels
# A tap is taught where the pointer (or, when the pointer missed it, the rhythm) says a tap happened AND the signal shows
# its contact jolt: the top of that jolt. What happens to the rest of the recording is the preset's choice (PRESETS);
# a recording the checks cannot trust is left out whole.
GATE = dict(h=1.0, tprom=0.8, w50=120.0, rise=110.0, cons=0.5, mid_def=0.8)     # a contact under a pointer press
GATE_LOOSE = dict(h=1.0, tprom=0.6, w50=150.0, rise=130.0, cons=0.5, mid_def=1.1)   # inclusive (opts gate="loose"), 2nd taps
STRICT = dict(h=1.5, tprom=1.2, w50=100.0, rise=90.0, cons=0.6, mid_def=0.5)    # a contact the pointer did not see
CONTACT_HALF = 90.0                # contact peak within +-90 ms of pointer-down + the recording's pointer delay
DELAY_SEARCH = (-150.0, 170.0)     # where the delay is first looked for (contact - pointer-down)
RELEASE_WIN = (-40.0, 120.0)       # release jolt relative to pointer-up
BOUNCE_MS = 40.0                   # pointer gaps shorter than this are one press
DELAY_DEFAULT = 20.0               # contact - pointer-down when a recording has too few clean presses
SECOND_MIN_MS = 120.0              # a second tap inside one press comes at least this long after the first
SECOND_DROP = 1.2                  # ... and is at most 1.2 decades weaker than the first
PARTNER_MS = (120.0, 450.0)        # double taps: a second tap the pointer never saw, after a confirmed first tap
RECOVER_TOL_MS = (60.0, 150.0)     # a recovered tap sits this close to the recording's beat lag (1.5 x its spread)
RECOVER_MAX_SPREAD_MS = 150.0      # ... and only where the person follows the beat that well
RHYTHM_TOL_NO_BEAT = 0.1           # no metronome: within 10% of a whole number of tap periods from a confirmed tap
STUCK_MS = 600.0                   # a single-tap press longer than this has a stuck pointer
TOP_BEFORE_MS, TOP_AFTER_MS = 10.0, 50.0   # the taught block: from 10 ms before the peak ...
TOP_DROP = 0.4                     # ... while the contact axis stays within 0.4 decades (40% power) of the peak
TOP_MIN_ROWS = 3
NEG_FOLDERS = {"FINGER", "WRIST", "ARM", "BAND", "NOTHING", "CONNECTED", "DISCONNECTED", "LOOSE", "TABLE"}
HOLD_FOLDERS = {"SOFT_LONG_PRESS", "MID_LONG_PRESS", "STRONG_LONG_PRESS", "LONG_PRESS", "DRAG_AND_DROP"}
POS_FOLDERS = {"TAP", "SWIPE"}
# Label presets (TAP_SIGNAL_LABELS.md, measured in training):
#   hybrid  the signal decides WHICH taps and WHERE (top of the contact jolt); around taps the rows the old labels left
#           out stay out (pointer press +-150 ms, every jolt with a role -200..+250 ms, the rise, weak presses, long-press
#           starts). Measured: same taps and fist protection as the current labels, band false taps -1.8/min, 240 bpm
#           taps +1.6 points.
#   legacy  the deterministic labeller's rule: only the tap block is 1, every other row 0 (releases, lead-in, rise, weak
#           presses), long-press starts are taps. No unknown rows. Measured worse: -2.9 points of taps, +2.3 fist
#           false taps/min at the same overall false rate, +25 ms latency, half the score margin.
PRESETS = dict(
    hybrid=dict(weak="unknown", lp_start="unknown", recover=True, second=True, rise="unknown", context="unknown",
                release="zero"),
    legacy=dict(weak="zero", lp_start="tap", recover=True, second=True, rise="zero", context="zero", release="zero"),
)
OPTIONS = dict(PRESETS["hybrid"])
VERSION = "signal-v2"


def passes(J, gate):
    """boolean per jolt: every condition of the gate (per axis / per zone of the jolt's own contact axis)"""
    if not len(J):
        return np.zeros(0, bool)
    return ((J.h >= gate["h"]) & (J.tprom >= gate["tprom"]) & (J.w50 <= gate["w50"]) & (J.rise_v <= gate["rise"])
            & (J.cons >= gate["cons"]) & (J.mid_def <= gate["mid_def"])).to_numpy()


def reasons(j, gate=GATE):
    out = []
    if j.h < gate["h"]: out.append("weak")
    if j.tprom < gate["tprom"]: out.append("not_isolated")
    if j.w50 > gate["w50"] or j.rise_v > gate["rise"]: out.append("slow")
    if j.cons < gate["cons"]: out.append("axis_changes")
    if j.mid_def > gate["mid_def"]: out.append("no_impact_band")
    return ";".join(out)


def recording_kind(folder, name):
    m = __import__("re").search(r"_(\d+)bpm", name)
    bpm = int(m.group(1)) if m else -1
    return dict(double="doubletap" in name.lower(), bpm=bpm, fast=bpm >= 240)


def top_block(t, ed, k, h):
    """rows taught for a tap peaking at row k: from <= TOP_BEFORE_MS before the peak, while the contact axis stays
    within TOP_DROP decades of the peak, up to TOP_AFTER_MS after it; at least TOP_MIN_ROWS rows"""
    a = k
    while a > 0 and t[a - 1] >= t[k] - TOP_BEFORE_MS and ed[a - 1] >= h - TOP_DROP:
        a -= 1
    b = k
    while b + 1 < len(t) and t[b + 1] <= t[k] + TOP_AFTER_MS and ed[b + 1] >= h - TOP_DROP:
        b += 1
    while b - a + 1 < TOP_MIN_ROWS and b + 1 < len(t):
        b += 1
    return a, b


def label(s, folder, name, opts=None, guard=True):
    """(y int8 per row: 1 tap / 0 not a tap / -1 left out, events DataFrame, quality dict). opts: a PRESETS name or a
    dict of OPTIONS overrides."""
    o = dict(PRESETS[opts]) if isinstance(opts, str) else dict(OPTIONS, **(opts or {}))
    gate = GATE_LOOSE if o.get("gate") == "loose" else GATE
    t = s["t"]; n = len(t)
    y = np.zeros(n, np.int8)
    rk = recording_kind(folder, name)
    down, up, starts_down = presses(t, s["ptr"])
    if len(down) > 1:                                              # bounces: a gap under BOUNCE_MS is one press
        keep = np.r_[True, (down[1:] - up[:-1]) >= BOUNCE_MS]; gi = np.cumsum(keep) - 1
        down = down[keep]; up = np.array([up[gi == i].max() for i in range(gi.max() + 1)])
    p = prepare(s, guard_edges=np.r_[down, up] if guard else None)
    J = jolts(s, p)
    q = dict(folder=folder, presses=int(len(down)), starts_down=bool(starts_down), jolts=int(len(J)),
             minutes=float(t[-1] / 60000) if n else 0.0, **rk)
    if not len(J):
        J = pd.DataFrame(columns=["k", "t", "axis", "h", "tprom", "w50", "rise_v", "cons", "mid_def"])
    J = J.reset_index(drop=True)
    J["crisp"] = passes(J, gate); J["loose"] = passes(J, GATE_LOOSE); J["strict"] = passes(J, STRICT)
    J["role"] = "other"; J["press"] = -1; J["y"] = 0
    q["crisp_per_min"] = float(J.crisp.sum() / max(q["minutes"], 1e-9))
    if folder not in (POS_FOLDERS | HOLD_FOLDERS):
        q.update(tier="negative", taps=0, tap_rows=0); return y, J, q
    tj = np.array(J.t, dtype=float); hh = np.array(J.h, dtype=float)
    crisp = np.array(J.crisp, bool); loose = np.array(J.loose, bool); strict = np.array(J.strict, bool)
    role = np.array(J.role, dtype=object); press = np.array(J.press, dtype=np.int64)
    no_edge = np.zeros(len(down), bool)
    if starts_down and len(down):
        no_edge[0] = True                                          # the first press began before the recording

    def pick(i, delay, ok, win=None):
        lo, hi = (down[i] + delay - CONTACT_HALF, down[i] + delay + CONTACT_HALF) if win is None else (down[i] + win[0], down[i] + win[1])
        c = np.flatnonzero((tj >= lo) & (tj <= hi) & (role == "other") & ok)
        if not len(c):
            return -1
        score = hh[c] - np.abs(tj[c] - down[i] - delay) / 100.0  # strong and close to where this pointer puts taps
        return int(c[np.argmax(score)])

    # 1) contacts under pointer presses, with the recording's own pointer delay
    first = [pick(i, DELAY_DEFAULT, crisp, DELAY_SEARCH) if not no_edge[i] else -1 for i in range(len(down))]
    d_ok = np.array([tj[j] - down[i] for i, j in enumerate(first) if j >= 0])
    delay = float(np.median(d_ok)) if len(d_ok) >= 5 else DELAY_DEFAULT
    jitter = float(np.subtract(*np.percentile(d_ok, [75, 25]))) if len(d_ok) >= 5 else np.nan
    contact = np.full(len(down), -1)
    anyj = np.ones(len(tj), bool)
    for i in range(len(down)):
        if no_edge[i]:
            continue
        j = pick(i, delay, crisp); r = "contact"
        if j < 0:
            j = pick(i, delay, anyj); r = "weak"
        if j >= 0:
            role[j] = r; press[j] = i; contact[i] = j
    # 2) inside each press: a second tap (merged double), the release, movement
    pos_kind = "lp" if folder in HOLD_FOLDERS else "tap"
    stuck = ((up - down) > STUCK_MS) if (folder == "TAP" and not rk["double"]) else np.zeros(len(down), bool)
    fast_p = 60000.0 / rk["bpm"] if rk["fast"] else np.nan
    for i in range(len(down)):
        if no_edge[i]:
            continue                                               # began before the recording: left to the rhythm
        c = contact[i]
        t0 = tj[c] if c >= 0 else down[i]
        end = up[i] + RELEASE_WIN[1] if not stuck[i] else t0 + STUCK_MS   # a stuck release says nothing
        inside = np.flatnonzero((tj > t0 + 50) & (tj <= end) & (role == "other"))
        if o["second"] and pos_kind == "tap" and (rk["double"] or rk["fast"]) and c >= 0 and role[c] == "contact":
            prev = c
            while True:                                            # double: one more tap; 240 bpm: one per period
                if rk["fast"]:
                    lo_, hi_ = tj[prev] + 0.6 * fast_p, tj[prev] + 1.4 * fast_p
                else:
                    lo_, hi_ = tj[prev] + SECOND_MIN_MS, np.inf
                cand = [j for j in inside if lo_ <= tj[j] <= min(hi_, up[i] + 60) and loose[j]
                        and hh[j] >= hh[prev] - SECOND_DROP and role[j] == "other"]
                if not cand:
                    break
                j2 = max(cand, key=lambda j: hh[j] + J.tprom[j])
                role[j2] = "second"; press[j2] = i
                inside = inside[inside != j2]; prev = j2
                if not rk["fast"]:
                    break
        for j in inside:
            role[j] = "release" if tj[j] >= up[i] + RELEASE_WIN[0] else "inpress"; press[j] = i
    # 3) taps the pointer did not see. With a metronome: on the recording's own beat lag (one tap per beat; double taps:
    #    the partner of a confirmed first tap). Without one: a whole number of tap periods from a confirmed tap.
    bt = beats(t, s["met"])
    conf_c = np.sort(tj[role == "contact"])
    if rk["double"] and len(conf_c):                               # double taps: the beat leads the pair's first tap
        conf_c = conf_c[np.r_[True, np.diff(conf_c) > PARTNER_MS[1]]]
    lag = spread = np.nan
    if len(bt) > 2 and len(conf_c) >= 8:
        bi = np.clip(np.searchsorted(bt, conf_c), 1, len(bt) - 1)
        nb = np.where(np.abs(conf_c - bt[bi - 1]) <= np.abs(conf_c - bt[bi]), bt[bi - 1], bt[bi])
        lg = conf_c - nb; lag = float(np.median(lg)); spread = float(np.subtract(*np.percentile(lg, [75, 25])))
    period = float(np.median(np.diff(bt))) if len(bt) > 2 else (float(np.median(np.diff(conf_c))) if len(conf_c) > 5 else np.nan)
    if o["recover"] and pos_kind == "tap" and len(conf_c):
        in_press = np.zeros(len(tj), bool)
        for i in range(len(down)):
            if not stuck[i] and not no_edge[i]:
                in_press |= (tj >= down[i] - 150) & (tj <= up[i] + 150)
        conf = np.sort(tj[np.isin(role, ["contact", "second"])])
        taken = set()
        if len(bt) > 2 and np.isfinite(lag):
            for x in conf:
                taken.add(int(np.argmin(np.abs(bt + lag - x))))
        tol = float(np.clip(1.5 * spread, RECOVER_TOL_MS[0], RECOVER_TOL_MS[1])) if np.isfinite(spread) else np.nan
        for j in np.flatnonzero((role == "other") & strict & ~in_press):
            before = conf[conf < tj[j]]
            ok = False
            if rk["double"] and len(before) and PARTNER_MS[0] <= tj[j] - before[-1] <= PARTNER_MS[1]:
                ok = True
            elif len(bt) > 2 and np.isfinite(lag) and spread <= RECOVER_MAX_SPREAD_MS:
                b_ = int(np.argmin(np.abs(bt + lag - tj[j])))
                ok = abs(tj[j] - bt[b_] - lag) <= tol and b_ not in taken
                if ok:
                    taken.add(b_)
            elif np.isfinite(period):
                after = conf[conf > tj[j]]
                for nb_ in ([before[-1]] if len(before) else []) + ([after[0]] if len(after) else []):
                    r = abs(tj[j] - nb_) / period
                    ok = ok or (round(r) >= 1 and abs(r - round(r)) <= RHYTHM_TOL_NO_BEAT)
            if ok:
                role[j] = "recovered"
    J["role"] = role; J["press"] = press
    # 4) labels: the top of every tap's contact jolt is 1, every other row 0
    pos_roles = {"contact", "second", "recovered"} | ({"weak"} if o["weak"] == "tap" else set())
    if pos_kind == "lp":
        pos_roles = {"contact"} if o["lp_start"] == "tap" else set()
    taps = 0
    unk = np.zeros(n, bool)
    if pos_kind == "lp" and o["lp_start"] == "unknown":               # research: long-press starts / releases left out
        for i in range(len(down)):
            unk[window(t, down[i] - 300, down[i] + 500)] = True; unk[window(t, up[i] - 300, up[i] + 500)] = True
    if o["weak"] == "unknown":                                         # research: weak presses left out
        for i in range(len(down)):
            if contact[i] >= 0 and role[contact[i]] == "weak":
                unk[window(t, down[i] - 100, up[i] + 150)] = True
    for j in np.flatnonzero(np.isin(role, list(pos_roles))):
        k = int(J.k[j]); d = int(J.axis[j])
        a, b = top_block(t, p["ex"][:, d], k, float(J.h[j]))
        if o["rise"] == "unknown":                                     # research: the rise before the block left out
            unk[window(t, t[k] - float(J.rise_v[j]), t[a] - 1e-6)] = True
        y[a:b + 1] = 1; J.loc[j, "y"] = 1; taps += 1
    if o["context"] == "unknown" and pos_kind == "tap":               # research: the old labels' unknown structure
        for i in range(len(down)):                                     # (press +-150 ms, around every jolt with a role
            unk[window(t, down[i] - 150, up[i] + 150)] = True          # and every crisp jolt -200..+250 ms)
        for j in np.flatnonzero((role != "other") | crisp):
            unk[window(t, tj[j] - 200, tj[j] + 250)] = True
    y[unk & (y == 0)] = -1
    if o["release"] == "block" and pos_kind == "tap":                 # research: only each release jolt's own top is
        for j in np.flatnonzero(role == "release"):                    # "not a tap" (the rest of the press unknown)
            a, b = top_block(t, p["ex"][:, int(J.axis[j])], int(J.k[j]), float(J.h[j]))
            y[a:b + 1] = np.where(y[a:b + 1] == 1, 1, 0)
    roles = pd.Series(role).value_counts().to_dict()
    q.update(delay=delay, jitter=jitter, period=period, beats=int(len(bt)), beat_lag=lag, beat_spread=spread,
             **{f"n_{r}": int(roles.get(r, 0)) for r in ("contact", "weak", "second", "recovered", "release", "inpress", "other")},
             no_jolt=int(sum(1 for i in range(len(down)) if contact[i] < 0 and not no_edge[i])),
             taps=taps, tap_rows=int(y.sum()))
    q["agreement"] = q["n_contact"] / max(1, q["presses"] - int(no_edge.sum()))
    q["expected"] = len(bt) * (2 if rk["double"] else 1) if len(bt) > 2 else np.nan
    q["tier"] = tier(q, folder)
    if q["tier"] == "broken":
        y[:] = -1
    return y, J, q


def tier(q, folder):
    """clean / repaired / weak / broken (left out). Only recordings that should hold taps are judged."""
    if folder not in POS_FOLDERS:
        return "hold" if folder in HOLD_FOLDERS else "negative"
    if q["presses"] < 3 and q["taps"] < 3:
        return "broken"
    if q["agreement"] < 0.3:
        return "broken"
    if q["agreement"] < 0.6:
        return "weak"
    return "repaired" if (q["n_second"] or q["n_recovered"]) else "clean"
