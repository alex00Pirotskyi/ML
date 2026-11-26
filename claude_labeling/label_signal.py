#!/usr/bin/env python3
"""Signal tap labels for every TAP-task recording (claude_labeling/tap_signal.py; TAP_SIGNAL_LABELS.md).

    python claude_labeling/label_signal.py [OUT]       # default OUT = DATA/LABELS_SIGNAL

Writes OUT/labels.npz (one int8 array per recording, keyed by its path relative to DATA/: 1 tap, 0 not a tap, -1 the
recording is left out), OUT/events.csv.gz (every jolt with its role) and OUT/quality.csv.gz (one line per recording).
"""
from __future__ import annotations

import functools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import tap_signal as TS        # noqa: E402

TAP_DIRS = ["TAP", "SWIPE", "FINGER", "WRIST", "ARM", "BAND", "NOTHING"]
LP_DIRS = ["SOFT_LONG_PRESS", "MID_LONG_PRESS", "STRONG_LONG_PRESS", "LONG_PRESS", "DRAG_AND_DROP"]
EV_COLS = ["t", "k", "axis", "h", "h2", "tprom", "prom", "w50", "rise", "cons", "mid_def", "z34", "m1", "m2", "m3",
           "md1", "md2", "md3", "crisp", "strict", "role", "press", "y"]


def recordings(root=ROOT / "DATA"):
    out = []
    for d in TAP_DIRS:
        out += sorted((root / "DATA_TAP" / d).glob("*.csv.gz"))
    for d in LP_DIRS:
        out += sorted((root / "DATA_LONG_PRESS" / d).glob("*.csv.gz"))
    return out


def one(path, opts=None):
    path = Path(path)
    key = f"{path.parent.parent.name}/{path.parent.name}/{path.name}"
    s, sha = TS.read(path)
    y, J, q = TS.label(s, path.parent.name, path.name, opts)
    J = J.assign(prom=J.h - J.flank) if len(J) and "flank" in J else J
    ev = J[[c for c in EV_COLS if c in J.columns]].copy() if len(J) else pd.DataFrame(columns=EV_COLS)
    ev.insert(0, "file", key)
    q = dict(file=key, sha=sha, rows=int(len(y)), **q)
    return key, y, ev, q


def main(out: Path, opts=None):
    out.mkdir(parents=True, exist_ok=True)
    files = recordings(); t0 = time.time()
    Y, E, Q, seen = {}, [], [], {}
    with ProcessPoolExecutor(int(os.environ.get("SIG_WORKERS", os.cpu_count() or 2))) as ex:
        for i, (key, y, ev, q) in enumerate(ex.map(functools.partial(one, opts=opts), map(str, files), chunksize=4), 1):
            q["duplicate_of"] = seen.get(q["sha"], "")
            seen.setdefault(q["sha"], key)
            Y[key] = y; E.append(ev); Q.append(q)
            if i % 500 == 0:
                print(f"{i}/{len(files)} {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(out / "labels.npz", **Y)
    pd.concat(E, ignore_index=True).to_csv(out / "events.csv.gz", index=False)
    Q = pd.DataFrame(Q); Q.to_csv(out / "quality.csv.gz", index=False)
    used = dict(TS.PRESETS[opts]) if isinstance(opts, str) else dict(TS.OPTIONS, **(opts or {}))
    (out / "options.json").write_text(json.dumps(dict(used, version=TS.VERSION), indent=1))
    print(f"{len(Q)} recordings, {time.time() - t0:.0f} s | tiers {Q.tier.value_counts().to_dict()}")
    return Q


if __name__ == "__main__":
    # SIG_PRESET=hybrid|legacy (tap_signal.PRESETS), or SIG_OPTS='{"weak": "tap"}' overrides of tap_signal.OPTIONS
    opts = os.environ.get("SIG_PRESET") or json.loads(os.environ.get("SIG_OPTS", "{}"))
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "DATA" / "LABELS_SIGNAL", opts)
