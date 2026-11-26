"""Score emitted device events in independently reviewed intervals (milliseconds from recording start)."""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


if __package__:
    from .tap_only import read_reviews, validate_reviews
else:
    from tap_only import read_reviews, validate_reviews


def score(reviews, fires, early, late):
    validate_reviews(reviews)
    reviews = reviews.astype({"start_ms": int, "end_ms": int})
    if early < 0 or late < 0:
        raise ValueError("Timing tolerances must be nonnegative")
    if not set(fires.recording) <= set(reviews.recording):
        raise ValueError("Fires from recordings without reviews")
    slices = defaultdict(Counter)
    delays = []
    confusion = Counter()
    for rec, group in reviews.groupby("recording"):
        events = fires[fires.recording == rec].reset_index(drop=True)
        ft = events.fire_ms.to_numpy(dtype=float)
        if not np.isfinite(ft).all() or (ft < 0).any():
            raise ValueError(f"Invalid emission time in {rec}")
        taps = group[group.decision == "tap"].sort_values("start_ms").reset_index(drop=True)
        dt = ft[None, :] - taps.start_ms.to_numpy()[:, None]
        valid = (dt >= -early) & (dt <= late)
        # Make losing one match more expensive than all possible timing improvements.
        penalty = (1 + min(dt.shape)) * (early + late + 1)
        ti, fi = linear_sum_assignment(np.where(valid, np.abs(dt), penalty))
        matched = {i: j for i, j in zip(ti, fi) if valid[i, j]}
        for i, r in enumerate(taps.itertuples()):
            for key in ("all", "strength/" + r.strength, "finger/" + r.finger):
                slices[key]["taps"] += 1
                slices[key]["hits"] += int(i in matched)
            if i in matched:
                j = matched[i]
                delays.append(float(dt[i, j]))
                if r.finger != "unknown" and "finger" in events:
                    prediction = str(events.finger.iloc[j])
                    confusion[r.finger + "/" + prediction] += 1
                    slices["finger/" + r.finger]["correct_finger"] += int(r.finger == prediction)
        used = set(matched.values())
        neg = group[group.decision == "negative"].sort_values("start_ms")
        prev = -1
        for r in neg.itertuples():
            if r.start_ms <= prev or r.end_ms < r.start_ms:
                raise ValueError(f"Overlapping or reversed negative intervals in {rec}")
            prev = r.end_ms
            # Remove tap matching windows from the negative exposure at annotation boundaries.
            intervals = [(r.start_ms, r.end_ms)]
            for t in taps.start_ms:
                left, right = t - early, t + late
                intervals = [(a, b) for lo, hi in intervals
                             for a, b in [(lo, min(hi, left)), (max(lo, right), hi)] if b > a]
            exposure = sum(b - a for a, b in intervals)
            false = sum(j not in used and not valid[:, j].any() and any(a <= f <= b for a, b in intervals)
                        for j, f in enumerate(ft))
            for key in ("all", "gesture/" + rec.split("/")[-2] if "/" in rec else "gesture/unknown"):
                slices[key]["negative_ms"] += int(exposure)
                slices[key]["false_on_negative"] += int(false)
        slices["all"]["extra_fires_near_taps"] += int(sum(j not in used and valid[:, j].any() for j in range(len(ft))))
    for key, s in slices.items():
        if s["taps"]:
            s["recall"] = s["hits"] / s["taps"]
            if "finger" in fires and key.startswith("finger/") and key != "finger/unknown":
                s["correct_finger_per_tap"] = s["correct_finger"] / s["taps"]
        if s["negative_ms"]:
            s["false_per_min"] = s["false_on_negative"] * 60000 / s["negative_ms"]
    return dict(slices=dict(slices), latency_ms_p50_p95=np.percentile(delays, [50, 95]).tolist()
                if delays else [], finger_confusion=dict(confusion))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reviews")
    p.add_argument("fires", help="CSV with recording,fire_ms and optional finger")
    p.add_argument("--early", type=int, default=100)
    p.add_argument("--late", type=int, default=250)
    args = p.parse_args()
    print(json.dumps(score(pd.concat(read_reviews(args.reviews).values()), pd.read_csv(args.fires), args.early, args.late), indent=2))
