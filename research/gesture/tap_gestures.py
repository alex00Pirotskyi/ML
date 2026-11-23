"""Host twin of firmware/tap_gestures.h (single / double tap decision on the TAP_SWIPE score).

step()     straight port of tg_update(), one sample at a time (the reference; bit-exact with the C).
simulate() event-driven version for long replays: same events, it only walks the samples near onsets.

Scores and thresholds share one scale: Q20 integers for device logs and C parity, floats for Python model
probabilities. Times are milliseconds (the C uses uint32 ms; the twin uses floats, the same for t < 2^32).
"""
from dataclasses import dataclass, replace

import numpy as np

NONE, TAP, DOUBLE_TAP = 0, 1, 2
IDLE, PENDING, LOCKOUT = 0, 1, 2


@dataclass(frozen=True)
class Config:
    fire: float = 17            # TG_FIRE_Q20
    second: float = 12          # TG_SECOND_Q20
    dip: float = 4              # TG_DIP_Q20
    min_gap_ms: float = 150     # TG_MIN_GAP_MS
    window_ms: float = 400      # TG_WINDOW_MS
    lockout_ms: float = 300     # TG_LOCKOUT_MS
    immediate: bool = False     # TG_REPORT_TAP_IMMEDIATELY
    chain: bool = False         # TG_CHAIN_TAPS (immediate mode only): every tap re-arms the window, no lockout

    def __post_init__(self):
        assert not (self.chain and not self.immediate), "chain needs immediate reporting"

    def scaled(self, scale):
        """Same rule on another score scale (e.g. 1/20 for probabilities)."""
        return replace(self, fire=self.fire * scale, second=self.second * scale, dip=self.dip * scale)


FIRMWARE = Config()


class State:
    def __init__(self, cfg, now_ms=0.0):
        self.cfg = cfg
        self.reset(now_ms)

    def reset(self, now_ms=0.0):
        self.state, self.prev_q, self.dipped = IDLE, 0, False
        self.t_first = now_ms
        self.t_last = now_ms - self.cfg.lockout_ms - self.cfg.min_gap_ms

    def cancel(self):
        if self.state == PENDING:
            self.state = IDLE

    def step(self, q, now):
        c = self.cfg; ev = NONE
        rising = q >= c.fire and self.prev_q < c.fire
        self.prev_q = q
        if self.state == PENDING:
            dt = now - self.t_first
            if q < c.dip:
                self.dipped = True
            if c.min_gap_ms <= dt <= c.window_ms and self.dipped and q >= c.second:
                if c.chain:
                    self.t_first, self.t_last, self.dipped = now, now, False
                else:
                    self.state, self.t_last = LOCKOUT, now
                return DOUBLE_TAP
            if dt <= c.window_ms:
                return NONE
            self.state = IDLE
            if not c.immediate:
                ev = TAP
        if self.state == LOCKOUT:
            if now - self.t_last < c.lockout_ms:
                return NONE
            self.state = IDLE
        if rising and now - self.t_last >= c.min_gap_ms:
            self.state, self.t_first, self.t_last, self.dipped = PENDING, now, now, False
            if c.immediate:
                ev = TAP
        return ev


def step_all(q, t, cfg):
    """Reference: [(sample index, event)] from step()."""
    s = State(cfg, float(t[0]) if len(t) else 0.0)
    out = []
    for i, (qq, tt) in enumerate(zip(q, t)):
        e = s.step(qq, float(tt))
        if e:
            out.append((i, e))
    return out


def simulate(q, t, cfg, flush=True):
    """Event-driven twin of step_all(). Returns gestures and [(sample index, event)] exactly as step_all().
    Deferred mode: gestures are (kind, t_first, t_second), kind TAP or DOUBLE_TAP (t_second = NaN for TAP);
    flush: a tap still pending at the end of the stream is returned as a TAP (not reported by step_all).
    Immediate mode: one gesture (kind, t, NaN) per reported event (TAP, DOUBLE_TAP), i.e. one per click."""
    if cfg.immediate:
        return _simulate_immediate(q, t, cfg)
    q = np.asarray(q); t = np.asarray(t, np.float64); n = len(q)
    if not n:
        return [], []
    rising = np.flatnonzero((q >= cfg.fire) & (np.r_[0, q[:-1]] < cfg.fire))
    gestures, events = [], []
    t_last = t[0] - cfg.lockout_ms - cfg.min_gap_ms
    pos = 0                                   # first sample index the idle search may use
    while True:
        k = np.searchsorted(rising, pos)
        while k < len(rising) and t[rising[k]] - t_last < cfg.min_gap_ms:
            k += 1
        if k >= len(rising):
            break
        r = rising[k]; t_first = t[r]; t_last = t_first; dipped = False
        j = r + 1; done = False
        while j < n:
            dt = t[j] - t_first
            if q[j] < cfg.dip:
                dipped = True
            if cfg.min_gap_ms <= dt <= cfg.window_ms and dipped and q[j] >= cfg.second:
                gestures.append((DOUBLE_TAP, t_first, t[j])); events.append((j, DOUBLE_TAP))
                t_last = t[j]
                pos = int(np.searchsorted(t, t_last + cfg.lockout_ms, side="left"))
                pos = max(pos, j + 1); done = True
                break
            if dt > cfg.window_ms:
                gestures.append((TAP, t_first, np.nan)); events.append((j, TAP))
                pos = j; done = True                    # this sample may start the next tap
                break
            j += 1
        if not done:                                    # stream ended while pending
            if flush:
                gestures.append((TAP, t_first, np.nan))
            break
        if pos >= n:
            break
    return gestures, events


def _simulate_immediate(q, t, cfg):
    q = np.asarray(q); t = np.asarray(t, np.float64); n = len(q)
    if not n:
        return [], []
    rising = np.flatnonzero((q >= cfg.fire) & (np.r_[0, q[:-1]] < cfg.fire))
    gestures, events = [], []
    t_last = t[0] - cfg.lockout_ms - cfg.min_gap_ms
    pos = 0
    while True:
        k = np.searchsorted(rising, pos)
        while k < len(rising) and t[rising[k]] - t_last < cfg.min_gap_ms:
            k += 1
        if k >= len(rising):
            break
        r = rising[k]; t_first = t_last = t[r]; dipped = False
        gestures.append((TAP, t[r], np.nan)); events.append((int(r), TAP))
        j = r + 1; pos = n
        while j < n:
            dt = t[j] - t_first
            if q[j] < cfg.dip:
                dipped = True
            if cfg.min_gap_ms <= dt <= cfg.window_ms and dipped and q[j] >= cfg.second:
                gestures.append((DOUBLE_TAP, t[j], np.nan)); events.append((j, DOUBLE_TAP))
                t_last = t[j]
                if cfg.chain:
                    t_first, dipped = t[j], False
                    j += 1
                    continue
                pos = max(int(np.searchsorted(t, t_last + cfg.lockout_ms, side="left")), j + 1)
                break
            if dt > cfg.window_ms:
                pos = j
                break
            j += 1
        if pos >= n:
            break
    return gestures, events
