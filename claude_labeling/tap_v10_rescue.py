"""The legacy tap labeller (unified_tap_labeler_deterministic.py, v10), unchanged, run on a REPAIRED pointer.

The old labeller is trusted for what a tap is: one dominant ACC axis well above its own rest level, a short, locally
prominent impulse, total power rising too, one tap per press cycle, a 3-13 row block around the peak. What it cannot
fix is a wrong pointer: a double tap merged into one press gets one label, a tap the pointer missed gets none, a press
cut by a 1-row bounce becomes two cycles. Those taps then train as "not a tap".

Here the signal layer (tap_signal.py: per-axis / per-zone jolts, the recording's own pointer delay and tap rhythm)
repairs the pointer first, and the old labeller labels the repaired recording:

  bounce      a press interrupted for less than 40 ms is one press
  double      a press holding two taps (the 2nd tap of a double, or the next tap of a 240 bpm series) is split just
              before the 2nd tap, so each tap gets its own cycle
  missed      a tap the pointer never saw (found on the recording's metronome lag / rhythm, see tap_signal.py) gets a
              press of the recording's typical length, starting where the pointer usually starts before a tap
  broken      a recording whose pointer and signal disagree on most presses is left out (-1 for every row)

Every repaired cycle still has to pass the old labeller's own gates, so a repaired tap that does not look like the
old labeller's tap stays 0, as it always was. Nothing else changes: NOT-TAP rows (releases, lead-in, rest, movement
recordings) are 0 exactly as before.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd

import tap_signal as TS
import unified_tap_labeler_deterministic as V10

VERSION = "v10-rescue-1"
BOUNCE_MS = 40.0
GAP_ROWS = 2                     # inactive rows written between the two taps of a split press
PRESS_ROWS = (8, 20)             # length of an inserted press: the recording's median press, clipped
TAP_ROLES = ("contact", "second", "recovered")


def _runs(active):
    e = np.diff(np.r_[0, active.astype(np.int8), 0])
    return np.flatnonzero(e > 0), np.flatnonzero(e < 0) - 1          # first / last active row of every press


def _in_old_window(k, st, en):
    """is row k inside the old labeller's search window of an existing press (press - 20 .. press + 12 / 30 rows)?"""
    for p, r in zip(st, en):
        end = max(p + V10.MINIMUM_SEARCH_AFTER_PRESS, min(p + V10.SEARCH_SAMPLES_AFTER_PRESS, r + V10.RELEASE_GRACE_SAMPLES))
        if p - V10.SEARCH_SAMPLES_BEFORE_PRESS <= k <= end:
            return True
    return False


def repair_pointer(t, active, J, q, split=True, insert=True):
    """(repaired active array, counts). J / q: tap_signal.label() output for the recording (preset legacy).
    split: cut a press that holds two taps; insert: add a press for a tap no press covers."""
    a = np.asarray(active, bool).copy()
    n = len(a)
    counts = dict(bounces=0, splits=0, inserted=0)
    st, en = _runs(a)
    for i in range(1, len(st)):                                      # bounces
        if t[st[i]] - t[en[i - 1]] < BOUNCE_MS:
            a[en[i - 1] + 1:st[i]] = True; counts["bounces"] += 1
    st, en = _runs(a)
    lead = int(V10.EXPECTED_PEAK_OFFSET_FROM_PRESS)                  # rows from press to peak the old labeller expects
    plen = int(np.clip(np.median(en - st + 1) if len(st) else PRESS_ROWS[0], *PRESS_ROWS))
    if not len(J):
        return a, counts
    roles = np.asarray(J.role, object); rows = np.asarray(J.k, int)
    taps = sorted(int(r) for r, ro in zip(rows, roles) if ro in TAP_ROLES)
    guard = int(V10.BETWEEN_CYCLE_GUARD_SAMPLES) + 1
    for i, k in enumerate(taps):
        st, en = _runs(a)
        inside = np.flatnonzero((st <= k) & (en >= k))
        if len(inside):                                              # inside a press after another tap: split
            if not split:
                continue
            j = inside[0]
            prev = [x for x in taps[:i] if x >= st[j]]
            if prev:
                p2 = max(k - lead, prev[-1] + guard + 3)
                if p2 - GAP_ROWS > st[j] + 1 and p2 <= en[j]:
                    a[p2 - GAP_ROWS:p2] = False; counts["splits"] += 1
            continue
        if not insert or _in_old_window(k, st, en):                  # the old labeller's window already sees it
            continue
        lo = max(0, k - lead)                                        # a tap no press covers: insert one
        hi = lo
        while hi < min(n, lo + plen) and not np.any(a[hi:min(n, hi + guard + 1)]):
            hi += 1
        if hi - lo >= 3 and not np.any(a[max(0, lo - guard):lo]):
            a[lo:hi] = True; counts["inserted"] += 1
    return a, counts


@contextlib.contextmanager
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def label_frame(df, folder, name, workdir=None, split=True, insert=True):
    """(state int8 per row: 1 tap / 0 not a tap / -1 left out, info dict) for one recording (a DataFrame read from
    its CSV). Recordings outside TAP / SWIPE are all 0, as in the old labeller."""
    n = len(df)
    info = dict(labeller=VERSION, folder=folder)
    if folder not in V10.POSITIVE_FOLDERS:
        return np.zeros(n, np.int8), dict(info, status="hard_negative")
    s = TS.signals(df)
    _, J, q = TS.label(s, folder, name, "legacy")
    info.update(tier=q.get("tier"), presses=q.get("presses"), signal_taps=q.get("taps"))
    if q.get("tier") == "broken":
        return np.full(n, -1, np.int8), dict(info, status="left_out_broken")
    pcol = V10.resolve_column(df.columns, V10.POINTER_ALIASES)
    active = V10.pointer_values(df[pcol]) if pcol else np.zeros(n, bool)
    rep, counts = repair_pointer(s["t"], active, J, q, split=split, insert=insert)
    info.update(counts)
    df2 = df.copy()
    df2[pcol or "PointerState"] = rep.astype(np.int8)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        root = Path(tmp)
        p = root / (Path(name).name.removesuffix(".gz"))
        df2.to_csv(p, index=False)
        with _quiet():
            rep_v10, _events, _, out = V10.process_positive_file(p, root, p, 1)
    state = (pd.to_numeric(out["state"], errors="coerce").fillna(0).to_numpy() > 0).astype(np.int8)
    if len(state) != n:
        raise RuntimeError(f"{name}: {len(state)} labels for {n} rows")
    info.update(status=rep_v10.status, cycles=rep_v10.press_cycles, accepted=rep_v10.accepted_events,
                tap_rows=int(state.sum()))
    return state, info


def label_file(path, workdir=None, split=True, insert=True):
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)
    return label_frame(df, path.parent.name, path.name, workdir, split, insert)
