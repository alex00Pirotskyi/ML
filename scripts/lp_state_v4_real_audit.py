#!/usr/bin/env python3
"""Execute V4 on the actual long-press recordings; report coverage, NOT accuracy.

No fake validation: PointerState is noisy weak context, not independently
synchronized physical force truth. 'Accepted events' must not be called recall.
This script never writes labels into the source tree.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from sys import path as python_path
python_path.insert(0, str(Path(__file__).resolve().parents[1]))
from lp_state_v4 import VERSION, GROUPS, LP_GROUPS, READ, label_recording


def stats_total(records):
    counts = {key: int(sum(r.get(key, 0) for r in records))
              for key in ("n", "rest", "active", "unknown", "pointer_events",
                          "verified_events")}
    n = counts["n"]
    denom = max(1, n)
    counts.update(
        files=len(records),
        files_with_both=int(sum(r.get("rest", 0) > 0 and r.get("active", 0) > 0
                                for r in records)),
        files_with_any_active=int(sum(r.get("active", 0) > 0 for r in records)),
        files_with_any_rest=int(sum(r.get("rest", 0) > 0 for r in records)),
        rest_fraction=round(counts["rest"] / denom, 5),
        active_fraction=round(counts["active"] / denom, 5),
        unknown_fraction=round(counts["unknown"] / denom, 5),
        labeled_fraction=round((counts["rest"] + counts["active"]) / denom, 5),
        reasons=dict(Counter(r.get("reason", "error") for r in records)))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("DATA/DATA_LONG_PRESS"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/lp_state_v4_real_audit.json"))
    args = parser.parse_args()
    root = args.input_dir.resolve()
    if not root.is_dir():
        parser.error("Missing source directory")
    records = []
    errors = []
    start = time.monotonic()
    for group in GROUPS:
        sources = sorted((root / group).glob("*.csv.gz"))
        print("GROUP_INPUT", group, len(sources), flush=True)
        for j, src in enumerate(sources, 1):
            item = {"file": src.relative_to(root).as_posix(), "group": group}
            try:
                df = pd.read_csv(src, compression="gzip", low_memory=False,
                                 usecols=lambda name: name in READ)
                y, audit = label_recording(df, group)
                item.update(audit)
                if len(y) != len(df):
                    raise ValueError("Label row count differs from source")
                if not set(np.unique(y.lp_state)).issubset({-1, 0, 1}):
                    raise ValueError("Unexpected label value")
                if not np.array_equal(y.lp_state_weight.to_numpy(),
                                      (y.lp_state.to_numpy() >= 0).astype(float)):
                    raise ValueError("Invalid label reliability")
                if item["n"] != item["rest"] + item["active"] + item["unknown"]:
                    raise ValueError("Incomplete row accounting")
            except Exception as e:
                item["reason"] = "error"
                item["exception"] = type(e).__name__ + ": " + str(e)[:300]
                errors.append(item["file"] + ": " + item["exception"])
            records.append(item)
            if j % 10 == 0 or j == len(sources):
                print("PROGRESS", group, j, len(sources),
                      "elapsed_s", round(time.monotonic()-start, 1), flush=True)
        print("AUDIT_GROUP", group, json.dumps(stats_total(
            [r for r in records if r["group"] == group]), sort_keys=True),
            flush=True)
    per_group = {g: stats_total([r for r in records if r["group"] == g])
                 for g in GROUPS}
    lp = stats_total([r for r in records if r["group"] in LP_GROUPS])
    control = stats_total([r for r in records if r["group"] not in LP_GROUPS])
    out = {
        "version": VERSION,
        "warning": "No independent physical press/release ground truth. All rates are coverage or weak-reference consistency, NOT accuracy.",
        "source": str(root),
        "selected_files": len(records),
        "counts": {"errors": len(errors), "event_support_on_pointer_reference": lp["verified_events"],
                   "pointer_reference_events": lp["pointer_events"]},
        "lp": lp,
        "controls": control,
        "per_group": per_group,
        "errors": errors,
        "elapsed_s": round(time.monotonic()-start, 1),
        "files": records
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print("AUDIT_TOTAL", json.dumps({k: v for k, v in out.items() if k != "files"},
                                    sort_keys=True), flush=True)
    print("AUDIT_REPORT_PATH", args.output, flush=True)
    if len(records) < 100:
        raise RuntimeError(f"Expected at least 100 LP/control recordings, got {len(records)}")
    if errors:
        raise RuntimeError(f"{len(errors)} recording errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
