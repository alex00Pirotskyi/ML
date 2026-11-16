#!/usr/bin/env python3
"""Independent, event-level TAP labeler audit. Never uses pointer state as truth.

truth.csv: recording_id,time_ms,event_type,pair_id(optional)
pred.csv: recording_id,time_ms,score(optional)
recordings.csv: recording_id,duration_s,category,participant(optional),session(optional),contact(optional)
Only truth event_type=tap is positive. Every prediction in a record is counted.
Records with no truth events MUST be included for false-activation measurement.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

def rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def match_events(truth, pred, tolerance):
    """Optimal ordered one-to-one match: maximize count, then minimize total onset error."""
    a = sorted(truth)
    b = sorted(pred)
    n, m = len(a), len(b)
    # DP state (match count, negative timing error); track action.
    dp = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
    action = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [(dp[i-1][j], "skip_truth"), (dp[i][j-1], "skip_pred")]
            if abs(a[i-1] - b[j-1]) <= tolerance:
                prior = dp[i-1][j-1]
                choices.append(((prior[0]+1, prior[1]-abs(a[i-1]-b[j-1])), "match"))
            dp[i][j], action[i][j] = max(choices, key=lambda x: x[0])
    pairs = []
    i, j = n, m
    while i and j:
        step = action[i][j]
        if step == "match":
            pairs.append((a[i-1], b[j-1]))
            i -= 1
            j -= 1
        elif step == "skip_truth":
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))

def score(truth_path, pred_path, recordings_path, tolerance, threshold):
    rec = rows(recordings_path)
    gt = defaultdict(list)
    hypotheses = defaultdict(list)
    pairs = defaultdict(lambda: defaultdict(list))
    for r in rows(truth_path):
        if r["event_type"].strip().lower() != "tap":
            continue
        rid = r["recording_id"]
        t = float(r["time_ms"])
        gt[rid].append(t)
        if r.get("pair_id", "").strip():
            pairs[rid][r["pair_id"]].append(t)
    for r in rows(pred_path):
        if r.get("score", "").strip() and float(r["score"]) < threshold:
            continue
        hypotheses[r["recording_id"]].append(float(r["time_ms"]))
    ids = {r["recording_id"] for r in rec}
    missing = (set(gt) | set(hypotheses)) - ids
    if missing:
        raise ValueError("Missing recording metadata: " + ", ".join(sorted(missing)[:10]))
    out = []
    all_tp = all_fp = all_fn = 0
    n_pairs = n_correct_pairs = n_split_pairs = 0
    for r in rec:
        rid = r["recording_id"]
        dur = float(r["duration_s"])
        if dur <= 0:
            raise ValueError("Invalid duration for " + rid)
        matches = match_events(gt[rid], hypotheses[rid], tolerance)
        tp = len(matches)
        fp = len(hypotheses[rid]) - tp
        fn = len(gt[rid]) - tp
        all_tp += tp
        all_fp += fp
        all_fn += fn
        match_t = {a for a, _ in matches}
        pair_groups = [p for p in pairs[rid].values() if len(p) == 2]
        n_pairs += len(pair_groups)
        n_correct_pairs += sum(all(t in match_t for t in p) for p in pair_groups)
        n_split_pairs += sum(sum(t in match_t for t in p) == 1 for p in pair_groups)
        out.append({
            "recording_id": rid,
            "category": r["category"],
            "participant": r.get("participant", ""),
            "session": r.get("session", ""),
            "contact": r.get("contact", ""),
            "duration_s": dur,
            "tp": tp, "fp": fp, "fn": fn,
            "false_predictions_per_hour": fp * 3600 / dur,
        })
    def safe(a, b):
        return a / b if b else None
    negative = [r for r in out if r["category"].strip().upper() in
                {"FINGER", "WRIST", "ARM", "BAND", "NOTHING"}]
    neg_fp = sum(r["fp"] for r in negative)
    neg_hours = sum(r["duration_s"] for r in negative) / 3600
    by_category = {}
    for c in sorted({r["category"] for r in out}):
        subset = [r for r in out if r["category"] == c]
        by_category[c] = {
            "recordings": len(subset),
            "tp": sum(r["tp"] for r in subset),
            "fn": sum(r["fn"] for r in subset),
            "fp": sum(r["fp"] for r in subset),
            "fp_per_hour": safe(sum(r["fp"] for r in subset) * 3600,
                                sum(r["duration_s"] for r in subset)),
        }
    return {
        "matching_tolerance_ms": tolerance,
        "score_threshold": threshold,
        "recordings": len(rec),
        "true_events": all_tp + all_fn,
        "predicted_events": all_tp + all_fp,
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
        "precision": safe(all_tp, all_tp+all_fp),
        "recall": safe(all_tp, all_tp+all_fn),
        "f1": safe(2*all_tp, 2*all_tp+all_fp+all_fn),
        "negative_folder_fp_per_hour": safe(neg_fp, neg_hours),
        "double_pairs_with_two_true_events": n_pairs,
        "double_pairs_both_recovered": n_correct_pairs,
        "double_pairs_one_recovered": n_split_pairs,
        "double_pair_recall": safe(n_correct_pairs, n_pairs),
        "by_category": by_category,
        "by_recording": out,
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--tolerance-ms", type=float, default=80)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    result = score(a.truth, a.pred, a.recordings, a.tolerance_ms, a.threshold)
    report = json.dumps(result, indent=2, allow_nan=False)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(report + "\n", encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
