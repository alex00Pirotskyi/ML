"""firmware/tap_gestures.h vs tap_gestures.py: compiles the C (both reporting modes) and checks that the C,
the Python step-by-step port and the event-driven twin give identical events on synthetic score streams
(bursts, wobbles, irregular timing, uint32 clock wrap) in all three modes, plus a few hand cases that pin the rule.

    python research/gesture/test_tap_gestures.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tap_gestures as TG  # noqa: E402

HARNESS = r"""
#include <stdio.h>
#include "tap_gestures.h"
int main(void) {
    tg_state_t s = {0}; unsigned q; unsigned long t; long i = 0; int first = 1;
    while (scanf("%u %lu", &q, &t) == 2) {
        if (first) { tg_reset(&s, (uint32_t) t); first = 0; }
        tg_event_t e = tg_update(&s, (uint8_t) q, (uint32_t) t);
        if (e != TG_NONE) printf("%ld %d\n", i, (int) e);
        i++;
    }
    return 0;
}
"""


MODES = {"deferred": TG.FIRMWARE, "immediate": TG.Config(immediate=True),
         "immediate+chain": TG.Config(immediate=True, chain=True)}


def build(tmp, cfg):
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    assert cc, "no C compiler"
    src = tmp / "harness.c"; src.write_text(HARNESS)
    exe = tmp / f"h{int(cfg.immediate)}{int(cfg.chain)}"
    subprocess.run([cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", f"-I{HERE / 'firmware'}",
                    f"-DTG_REPORT_TAP_IMMEDIATELY={int(cfg.immediate)}", f"-DTG_CHAIN_TAPS={int(cfg.chain)}",
                    str(src), "-o", str(exe)], check=True)
    return exe


def c_events(exe, q, t):
    inp = "\n".join(f"{a} {b % 2**32}" for a, b in zip(q, t)) + "\n"
    out = subprocess.run([str(exe)], input=inp, capture_output=True, text=True, check=True).stdout.split()
    return [(int(out[i]), int(out[i + 1])) for i in range(0, len(out), 2)]


def stream(spec, dt=5):
    """spec: [(n samples, q)] -> q, t (ms, starting at 1000)."""
    q = np.concatenate([np.full(n, v, int) for n, v in spec]); return q, 1000 + dt * np.arange(len(q))


def kinds(q, t, cfg=TG.FIRMWARE):
    return [g[0] for g in TG.simulate(q, t.astype(float), cfg)[0]]


def hand_cases():
    T, D = TG.TAP, TG.DOUBLE_TAP
    # one tap, score back to 0: a single TAP once the window closes
    assert kinds(*stream([(20, 0), (10, 20), (200, 0)])) == [T]
    # a bump that only reaches 0.85 (Q20 17 < 18): no tap with the strict default, a TAP at the old 0.85
    assert kinds(*stream([(20, 0), (10, 17), (200, 0)])) == []
    assert kinds(*stream([(20, 0), (10, 17), (200, 0)]), TG.Config(fire=17, second=12)) == [T]
    # two taps 250 ms apart, second only 0.80 (Q20 16) after a dip: double
    assert kinds(*stream([(20, 0), (10, 20), (40, 0), (10, 16), (200, 0)])) == [D]
    # same but the second bump is 0.70 (Q20 14 < 15): one TAP (the 2nd is below 0.90, so not a tap on its own)
    assert kinds(*stream([(20, 0), (10, 20), (40, 0), (10, 14), (200, 0)])) == [T]
    # second bump without a dip below 0.20 (score stays at 0.30): not a double
    assert kinds(*stream([(20, 0), (10, 20), (40, 6), (10, 16), (200, 0)])) == [T]
    # second tap too late (window 400 ms): two single TAPs
    assert kinds(*stream([(20, 0), (10, 20), (100, 0), (10, 20), (200, 0)])) == [T, T]
    # triple tap 250 ms apart: one DOUBLE, the third is inside the 300 ms lockout
    assert kinds(*stream([(20, 0), (10, 20), (40, 0), (10, 20), (40, 0), (10, 20), (200, 0)])) == [D]
    # immediate mode: TAP at once, DOUBLE follows
    q, t = stream([(20, 0), (10, 20), (40, 0), (10, 16), (200, 0)])
    assert [e for _, e in TG.step_all(q, t.astype(float), TG.Config(immediate=True))] == [T, D]
    # fast series, 5 taps 250 ms apart (the 2nd and 4th only 0.80): lockout loses taps, chain keeps all
    q, t = stream([(20, 0)] + [x for k in range(5) for x in ((10, 20 if k % 2 == 0 else 16), (40, 0))] + [(200, 0)])
    assert [e for _, e in TG.step_all(q, t.astype(float), TG.Config(immediate=True))] == [T, D, T]
    assert [e for _, e in TG.step_all(q, t.astype(float), TG.Config(immediate=True, chain=True))] == [T, D, D, D, D]


def main():
    hand_cases()
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); exes = {m: build(tmp, cfg) for m, cfg in MODES.items()}
        n_ev = 0
        for k in range(200):
            n = int(rng.integers(50, 3000)); t = 5 + np.cumsum(rng.integers(1, 12, n)); q = np.zeros(n, int)
            for _ in range(int(rng.integers(1, 40))):
                a = int(rng.integers(0, n)); L = int(rng.integers(1, 80))
                q[a:a + L] = np.maximum(q[a:a + L], rng.integers(0, 21, size=len(q[a:a + L])))
            if k % 3 == 0:
                t = t + 2**32 - 3000                       # uint32 ms clock wraps inside the stream
            tf = t.astype(float)
            for m, cfg in MODES.items():
                ref = TG.step_all(q, tf, cfg)
                assert c_events(exes[m], q, t) == ref, f"C vs Python ({m}) case {k}"
                assert TG.simulate(q, tf, cfg)[1] == ref, f"event-driven twin ({m}) case {k}"
                n_ev += len(ref)
    print(f"TAP_GESTURES_OK hand cases + 200 random streams x {len(MODES)} modes, {n_ev} events identical in C and Python")


if __name__ == "__main__":
    main()
