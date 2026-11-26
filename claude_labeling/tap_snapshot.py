"""Pack and read the released tap labels against the exact source recording bytes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FIELDS = ["tap_label", "provenance", "finger", "strength"]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pack(root, labels, output):
    with gzip.open(output, "xt", encoding="utf-8") as handle:
        for path in sorted(labels.rglob("*.labels.csv.gz")):
            key = path.relative_to(labels).as_posix().removesuffix(".labels.csv.gz")
            source = root / key
            df = pd.read_csv(path)
            ts = pd.read_csv(source, usecols=["Timestamp"]).Timestamp
            if not np.array_equal(ts, df.Timestamp):
                raise ValueError(f"Misaligned sidecar: {key}")
            vals = df[FIELDS]
            changes = np.flatnonzero(vals.ne(vals.shift()).any(axis=1))
            runs = [[int(i), int(row.tap_label), row.provenance, row.finger, row.strength]
                    for i, row in zip(changes, vals.iloc[changes].itertuples())]
            record = [key, digest(source), len(vals), runs]
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


class Snapshot:
    def __init__(self, path):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            self.records = {key: (sha, n, runs) for key, sha, n, runs in map(json.loads, handle)}

    def labels(self, root, key):
        sha, n, runs = self.records[key]
        path = Path(root) / key
        if digest(path) != sha:
            raise ValueError(f"Recording bytes changed: {key}")
        timestamps = pd.read_csv(path, usecols=["Timestamp"])
        if len(timestamps) != n:
            raise ValueError(f"Row count changed: {key}")
        starts, *columns = zip(*runs)
        lengths = np.diff([*starts, n])
        for field, values in zip(FIELDS, columns):
            timestamps[field] = np.repeat(values, lengths)
        return timestamps


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("labels", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    pack(args.source, args.labels, args.output)
