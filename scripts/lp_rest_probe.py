#!/usr/bin/env python3
"""Read-only schema and quality probe for DATA/DATA_LONG_PRESS CSV.gz recordings."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import gzip
import json
import statistics

ROOT = Path("DATA/DATA_LONG_PRESS")
GROUPS = ["MID_LONG_PRESS", "SOFT_LONG_PRESS", "STRONG_LONG_PRESS", "LONG_PRESS",
          "DRAG_AND_DROP", "CONNECTED", "DISCONNECTED", "LOOSE", "NOTHING",
          "TABLE", "TAP", "SWIPE"]


def open_csv(path):
    return gzip.open(path, "rt", encoding="utf-8-sig", newline="")


def main():
    print("LP REST DATASET SCHEMA PROBE v1", flush=True)
    columns_by_count = Counter()
    type_by_columns = defaultdict(Counter)
    paths_by_group = {}
    for group in GROUPS:
        paths = sorted((ROOT / group).rglob("*.csv.gz"))
        paths_by_group[group] = paths
        for p in paths:
            with open_csv(p) as f:
                cols = next(csv.reader(f), [])
            columns_by_count[tuple(cols)] += 1
            type_by_columns[group][tuple(cols)] += 1

    print("FILE COUNTS", json.dumps({g: len(v) for g, v in paths_by_group.items()}))
    print("NUMBER OF UNIQUE SCHEMAS", len(columns_by_count))
    for index, (cols, count) in enumerate(columns_by_count.most_common(8), 1):
        print(f"SCHEMA {index}: count={count}, columns={len(cols)}")
        print("COLUMNS:", json.dumps(cols))

    for group, paths in paths_by_group.items():
        if not paths:
            continue
        # Pick the median-sized recording to avoid a tiny/outlier file.
        path = sorted(paths, key=lambda p: p.stat().st_size)[len(paths) // 2]
        with open_csv(path) as f:
            rows = csv.DictReader(f)
            columns = rows.fieldnames or []
            preview = []
            for index, row in enumerate(rows):
                if index in {0, 1, 10, 100, 300, 500, 900, 1500}:
                    selection = {
                        key: value for key, value in row.items()
                        if ("psd" not in key.lower()
                            and ("fft" not in key.lower()))
                    }
                    if len(selection) > 80:
                        selection = dict(list(selection.items())[:80])
                    preview.append({"row_index": index, "data": selection})
                if index >= 1500:
                    break
        interesting_columns = [c for c in columns
                               if "psd" not in c.lower() and "fft" not in c.lower()]
        print("\nGROUP:", group)
        print("SAMPLE:", path.as_posix(), "compressed-bytes", path.stat().st_size)
        print("SCHEMA_RANK:", type_by_columns[group].most_common(1)[0][1],
              "distinct-group-schemas:", len(type_by_columns[group]))
        print("NON_PSD_COLUMNS:", json.dumps(interesting_columns))
        print("PREVIEW:", json.dumps(preview, ensure_ascii=False))
    print("END SCHEMA PROBE")


if __name__ == "__main__":
    main()
