"""Build row-aligned tap and swipe supervision using the repository teacher and optional independent reviews."""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import label_gestures as teacher
else:
    import label_gestures as teacher


FINGERS = {"thumb", "index", "middle", "ring", "little", "unknown"}
STRENGTHS = {"soft", "normal", "strong", "unknown"}
COLUMNS = ["recording", "start_ms", "end_ms", "decision", "finger", "strength",
           "reviewer", "reference"]
FOLDERS = teacher.TAP_FOLDERS | teacher.SWIPE_FOLDERS | teacher.HOLD_FOLDERS | teacher.NONE_FOLDERS


def read_reviews(path):
    if path is None:
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(df.columns) != set(COLUMNS):
        raise ValueError(f"Review columns must be {COLUMNS}")
    validate_reviews(df)
    return dict(tuple(df.groupby("recording")))


def validate_reviews(df):
    for rec, group in df.groupby("recording", dropna=False):
        ends = []
        if not isinstance(rec, str) or not rec.strip():
            raise ValueError("Each review needs a recording")
        for r in group.itertuples():
            start, end = int(r.start_ms), int(r.end_ms)
            if float(r.start_ms) != start or float(r.end_ms) != end or not (0 <= start <= end):
                raise ValueError(f"Invalid interval: {r}")
            if not (str(r.reference).strip() and str(r.reviewer).strip()):
                raise ValueError("Each review needs a reviewer and an independent reference")
            if r.decision not in {"tap", "negative", "unknown"}:
                raise ValueError(f"Unknown decision: {r.decision}")
            if r.finger not in FINGERS or r.strength not in STRENGTHS:
                raise ValueError(f"Invalid tap attributes: {r}")
            if r.decision != "tap" and (r.finger, r.strength) != ("unknown", "unknown"):
                raise ValueError("Only taps may have finger or strength labels")
            if r.decision == "tap" and start != end:
                raise ValueError("A tap review identifies one contact onset")
            ends.append((start, end))
        ordered = sorted(ends)
        if any(b >= c for (_, b), (c, _) in zip(ordered, ordered[1:])):
            raise ValueError(f"Overlapping review windows: {rec}")


def label_rows(base, folder, reviews):
    ts = base.Timestamp.to_numpy()
    if not len(ts) or np.any(np.diff(ts) < 0):
        raise ValueError("Expected ordered recording timestamps")
    t = ts - ts[0]
    y = base.tap_label.to_numpy(copy=True)
    y[base.n_ch_off.to_numpy() >= 2] = -1
    # The absence of a teacher peak in a prompted tap trial cannot certify rest.
    if folder in {"TAP", "SWIPE"} and not (y == 1).any():
        y[:] = -1
    out = pd.DataFrame(dict(Timestamp=ts, tap_label=y, provenance=np.where(
        y == -1, "unknown", "silver"), finger="unknown", strength="unknown"))
    if reviews is not None:
        validate_reviews(reviews)
        ordered = reviews.astype({"start_ms": int, "end_ms": int}).sort_values("start_ms")
        next_start = [*ordered.start_ms.iloc[1:], t[-1] + 1]
        for r, following in zip(ordered.itertuples(), next_start):
            start, end = int(r.start_ms), int(r.end_ms)
            if end > t[-1]:
                raise ValueError(f"Review exceeds recording: {r}")
            hi = min(start + teacher.P["TAP_WIN"][1], following - 1) if r.decision == "tap" else end
            mask = (t >= start) & (t <= hi)
            if not mask.any():
                raise ValueError(f"Unrepresented or overlapping reviews: {r}")
            out.loc[mask, ["tap_label", "provenance", "finger", "strength"]] = [
                {"tap": 1, "negative": 0, "unknown": -1}[r.decision],
                "reviewed", r.finger, r.strength]
    return out


def build(root, out, reviews):
    files = sorted(p for p in root.rglob("*") if p.name.endswith((".csv", ".csv.gz")))
    if not files or set(reviews) - {p.relative_to(root).as_posix() for p in files}:
        raise ValueError("Empty input or unmatched recording in reviews")
    tm = teacher.FoldModels("tap_teacher_fold", "tap_teacher.json", "person_group_fold")
    cm = teacher.FoldModels("contact_fold", "contact_folds.json", "person_group_fold")
    counts = Counter()
    candidates = []
    for path in files:
        folder = path.parent.name.upper()
        if folder not in FOLDERS:
            raise ValueError(f"Unknown gesture folder: {path}")
        rel = path.relative_to(root)
        key = rel.as_posix()
        kind = ("tap" if folder == "TAP" else "swipe" if folder == "SWIPE" else
                "hold" if folder in teacher.HOLD_FOLDERS else "none")
        base, events, _ = teacher.label_recording(str(path), tm, cm, kind, {"use_weak": False})
        labels = label_rows(base, folder, reviews.get(key))
        dest = out / rel.parent / (rel.name + ".labels.csv.gz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(dest, index=False)
        counts.update(dict(labels.tap_label.value_counts().items()))
        if len(events):
            events.insert(0, "recording", key)
            events["prompt_soft_tap"] = folder == "TAP" and "softtap" in path.stem.lower()
            candidates.append(events)
    evidence = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame(columns=["recording"])
    evidence.to_csv(out / "teacher_events.csv.gz", index=False)
    if reviews:
        pd.concat(reviews.values(), ignore_index=True)[COLUMNS].to_csv(out / "reviews.csv", index=False)
    summary = dict(recordings=len(files), rows_by_target={str(k): int(v) for k, v in counts.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="directory of recordings in gesture folders")
    parser.add_argument("output", type=Path, help="new directory outside the input")
    parser.add_argument("--reviews", type=Path, help="independent review CSV")
    args = parser.parse_args()
    root, dest = args.input.resolve(), args.output.resolve()
    if dest.exists() or dest.is_relative_to(root) or not root.is_dir():
        parser.error("Specify an input directory and a new output outside it")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp) / "labels"
        staged.mkdir()
        build(root, staged, read_reviews(args.reviews))
        staged.rename(dest)


if __name__ == "__main__":
    main()
