"""Small TAP cache for newly added recordings, in the format of the full cache (train_unified.build_task_data),
so the research harness can add them without rebuilding the full cache (MUDRA_GR_EXTRA, see
tap_generalization_research.load).

    python claude_labeling/label_gestures.py <folder with the new .csv.gz, by class subfolder> <labels out>
    python research/gesture/tap_extra_cache.py <labels out> <extra cache out> <new raw .csv.gz> [...]

Steps: Lean-8 derived copies of the new raw files (generate_lean8_features.generate_file, as for every recording
of the full cache), then build_task_data("tap") over only these recordings with the given labels.
"""
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_unified as g                  # noqa: E402
import generate_lean8_features as lean8    # noqa: E402


def main(labels, out, raws):
    labels, out = Path(labels), Path(out)
    ev = labels / "_events.csv.gz"
    if not ev.exists():                    # no gesture events in these recordings (e.g. finger motion)
        cols = pd.read_csv("/tmp/L/DATA_TAP/_events.csv.gz", nrows=0).columns
        pd.DataFrame(columns=cols).to_csv(ev, index=False, compression="gzip")
    records = []
    for r in map(Path, raws):
        r = r.resolve()
        dest = g.SOURCE_TAP / r.relative_to(g.RAW_SOURCE_TAP)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            lean8.generate_file(r, dest)
        records.append((dest, g.SOURCE_TAP, labels))
    g.LABEL_TAP = g.LABEL_LP = labels      # events of these recordings (the LP events are not used for TAP)
    out.mkdir(parents=True, exist_ok=True)
    data, rpl = g.build_task_data("tap", records, out)[:2]
    print("EXTRA_CACHE", out / "cache", "recordings", len(rpl["recs"]), "train rows", len(data["y"]),
          "replay rows", rpl["X"].shape[0], "columns", len(data["columns"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
