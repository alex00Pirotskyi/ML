"""Find tap-like windows using sensors alone and test dependence on time variation."""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

if __package__:
    from . import label_gestures as teacher
else:
    import label_gestures as teacher


def peaks(trace, threshold):
    smooth = uniform_filter1d(trace, 5)
    indices, _ = find_peaks(np.r_[0, smooth, 0], distance=10, height=threshold)
    indices -= 1
    return indices, smooth


def audit_recording(df, model):
    traces = teacher.tap_traces(df)
    n = len(traces[0])
    if n < 80:
        return [], dict(scored_ms=0, candidates=0, flat_candidates=0, flat_below=0, after_below=0)
    group = teacher.person_from_user(df)
    offsets = [teacher.O_A, teacher.O_R, teacher.O_E]
    full, flat, no_after = np.empty(n), np.empty(n), np.empty(n)
    for start in range(0, n, 4096):
        idx = np.arange(start, min(start + 4096, n))
        window = np.column_stack([teacher._win(tr, idx, off) for tr, off in zip(traces, offsets)])
        full[idx], _ = model.predict(window, group)
        # Same model, same center values and normalization. Remove within-window time variation.
        frozen = np.column_stack([np.tile(tr[idx], (1, len(off))) for tr, off in zip(traces, offsets)])
        flat[idx], _ = model.predict(frozen, group)
        # Preserve the entire approach to the center and hold all later offsets at the center.
        approach = np.column_stack([teacher._win(tr, idx, np.minimum(off, 0))
                                    for tr, off in zip(traces, offsets)])
        no_after[idx], _ = model.predict(approach, group)
    threshold = model.spec["thr_at_neg_false_per_min"]["1"]
    found, scores = peaks(full, threshold)
    flat_found, flat_scores = peaks(flat, threshold)
    # Expand the comparison to the same +/-150 ms alignment tolerance as the original teacher.
    after_scores = uniform_filter1d(no_after, 5)
    rows = []
    for p in found:
        nearby = flat_scores[max(0, p - 15):p + 16].max()
        after = after_scores[max(0, p - 15):p + 16].max()
        rows.append(dict(peak_ms=int(p * teacher.G), score=float(scores[p]),
                         flat_max_nearby=float(nearby), flat_below=bool(nearby < threshold),
                         approach_max_nearby=float(after), after_below=bool(after < threshold)))
    return rows, dict(scored_ms=n * teacher.G, candidates=len(found), flat_candidates=len(flat_found),
                      flat_below=sum(r["flat_below"] for r in rows), after_below=sum(r["after_below"] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("manifest", type=Path, help="gzipped list of deduplicated recording paths")
    p.add_argument("out", type=Path, help="new output directory")
    p.add_argument("--teacher-events", type=Path, help="attach existing accepted labels after sensor scoring")
    args = p.parse_args()
    model = teacher.FoldModels("tap_teacher_fold", "tap_teacher.json", "person_group_fold")
    records, candidates = [], []
    with gzip.open(args.manifest, "rt") as manifest:
        for key in map(str.strip, manifest):
            df = pd.read_csv(args.data / key, usecols=lambda c: c in teacher.TAP_COLS or c in {"Timestamp", "User"})
            rows, counts = audit_recording(df, model)
            folder = Path(key).parent.name
            records.append(dict(recording=key, folder=folder, **counts))
            candidates.extend(dict(recording=key, **row) for row in rows)
    args.out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(records).to_csv(args.out / "recordings.csv.gz", index=False)
    events = pd.DataFrame(candidates, columns=["recording", "peak_ms", "score", "flat_max_nearby",
                                               "flat_below", "approach_max_nearby", "after_below"])
    if args.teacher_events:
        previous = pd.read_csv(args.teacher_events, usecols=["recording", "type", "labeled", "t_on"])
        accepted = previous.loc[(previous.type == "tap_press") & (previous.labeled == 1),
                                ["recording", "t_on"]].rename(columns={"t_on": "peak_ms"}).drop_duplicates()
        events = events.merge(accepted.assign(accepted=True), on=["recording", "peak_ms"],
                              how="left", validate="one_to_one")
        events["accepted"] = events.accepted.eq(True)
    events.to_csv(args.out / "candidates.csv.gz", index=False)


if __name__ == "__main__":
    main()
