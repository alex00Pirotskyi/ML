"""Export the file-fold tap teacher for claude_labeling/label_gestures.py.

    python claude_labeling/teacher/export_teacher.py <work dir> claude_labeling/models

Reads <work>/models3.pkl (round-2 fold models), <work>/meta.pkl (sha -> fold), <work>/out3/summary.json (thresholds at
1 / 2 / 5 false events per minute in the negative folders, out of fold) and <work>/out3/calibration.csv, and writes
tap_teacher_file_fold<k>.txt and tap_teacher_file.json (key "file_fold": decompressed-content SHA-256 -> fold).
The labeller scores every recording with the fold model that never saw that recording (by content, so a renamed or
copied recording finds its fold); a recording none of them saw gets the mean of the five.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    work, out = Path(sys.argv[1]), Path(sys.argv[2])
    models = pickle.load(open(work / "models3.pkl", "rb"))
    meta = pd.read_pickle(work / "meta.pkl")
    summ = json.load(open(work / "out3" / "summary.json"))
    cal = pd.read_csv(work / "out3" / "calibration.csv")
    out.mkdir(parents=True, exist_ok=True)
    n_feat = None
    for k in range(5):
        b = models[k].booster_
        b.save_model(str(out / f"tap_teacher_file_fold{k}.txt"))
        n_feat = b.num_feature()
    folds = {str(s): int(f) for s, f in zip(meta.sha, meta.fold) if f == f}
    centers = ((cal.score_lo + cal.score_hi) / 2).round(5).tolist()
    spec = dict(grid_ms=10, offsets_acc_ms=[-150, 250, 10], offsets_rate_ms=[-100, 200, 20],
                offsets_emg_ms=[-240, 240, 20], n_features=int(n_feat),
                thr_at_neg_false_per_min={str(k): float(v) for k, v in summ["thr_at_neg_false_per_min"].items()},
                calibration=dict(centers=centers, p_real=cal.p_real.round(6).tolist()),
                file_fold=folds, split="gesture-aware file folds (claude_labeling/teacher/make_file_folds.py), never by person",
                note=("tap teacher round 2 (claude_research_tap V2 features and training), retrained on gesture-aware FILE "
                      "folds. file_fold: decompressed-content SHA-256 of a recording -> the fold whose model never saw it; "
                      "recordings not listed get the mean of the five models."))
    json.dump(spec, open(out / "tap_teacher_file.json", "w"), indent=0)
    print(json.dumps({k: v for k, v in spec.items() if k != "file_fold"}, indent=1), len(folds), "recordings")


if __name__ == "__main__":
    main()
