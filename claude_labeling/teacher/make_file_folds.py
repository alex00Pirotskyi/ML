"""Gesture-aware FILE folds for the tap teacher (never by person): folds.csv (name, sha, cat, stratum, fold).

    python claude_labeling/teacher/make_file_folds.py <teacher cache dir with meta.csv> folds.csv [--folds 5]

Every distinct recording (decompressed content hash) is its own group; identical copies share their fold. The folds are
stratified by recording type exactly as the trainer's file split: folder | gesture (from the file name) | bpm for TAP and
SWIPE, rare types pooled one level up (research/gesture/train_unified.py: recording_stratum, merge_rare).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "gesture"))
import train_unified as g  # noqa: E402

SEED = 20261126


def file_folds(meta: pd.DataFrame, n_folds: int = 5, seed: int = SEED) -> pd.DataFrame:
    m = meta[meta.get("err").isna()] if "err" in meta else meta
    m = m[["name", "cat", "sha"]].copy()
    uniq = m.drop_duplicates("sha").reset_index(drop=True)
    uniq["stratum"] = [g.recording_stratum(n, c) for n, c in zip(uniq.name, uniq.cat)]
    strata = g.merge_rare(uniq.stratum.to_numpy(object), max(g.STRATUM_MIN_FILES, n_folds))
    fold = np.full(len(uniq), -1, int)
    for k, (_, te) in enumerate(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                                .split(np.zeros(len(uniq)), strata)):
        fold[te] = k
    uniq["fold"] = fold; uniq["stratum_used"] = strata
    out = m.merge(uniq[["sha", "fold", "stratum", "stratum_used"]], on="sha", how="left")
    assert (out.fold >= 0).all()
    # an identical recording never sits in two folds
    assert out.groupby("sha").fold.nunique().max() == 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cache"); ap.add_argument("out"); ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    meta = pd.read_csv(Path(a.cache) / "meta.csv")
    f = file_folds(meta, a.folds)
    f.to_csv(a.out, index=False)
    print(f.fold.value_counts().sort_index().to_dict(), "files", len(f), "distinct", f.sha.nunique())
    print(pd.crosstab(f.cat, f.fold).to_string())


if __name__ == "__main__":
    main()
