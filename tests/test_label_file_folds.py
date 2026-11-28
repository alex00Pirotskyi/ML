"""No split or fold by person anywhere: the labeller's tap teacher is cross-fitted on gesture-aware FILE folds and picked
by the recording's content, the contact model is the mean of its folds, and the trainer's feature-ranking inner folds
are gesture-aware file folds like its CV folds."""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_labeling"))
sys.path.insert(0, str(ROOT / "claude_labeling" / "teacher"))


def test_file_folds_keep_copies_together_and_balance_types():
    import make_file_folds as F
    rows = []
    for k in range(60):
        for cat, gest in (("TAP", "MidTap"), ("SWIPE", "SwipeMID"), ("FINGER", "Pinky")):
            rows.append(dict(name=f"p{k % 3}_{gest}_Horizontal_Default_60bpm_2026-01-01_10-{k:02d}-00.csv.gz", cat=cat,
                             sha=f"{cat}{k}"))
    rows.append(dict(rows[0], name="copy_of_first.csv.gz"))              # identical content under another name
    f = F.file_folds(pd.DataFrame(rows), n_folds=5)
    assert f[f.sha == rows[0]["sha"]].fold.nunique() == 1
    counts = pd.crosstab(f.cat, f.fold)
    assert (counts.max(axis=1) - counts.min(axis=1) <= 1).all(), counts         # every type spread evenly over the folds
    # people p0/p1/p2 are in every fold: folds are by file, not by person
    f["person"] = f.name.str.extract(r"^(p\d)")[0]
    assert (f.dropna(subset=["person"]).groupby("person").fold.nunique() == 5).all()


def test_labeller_uses_file_folds_by_content():
    import label_gestures as L
    src = (ROOT / "claude_labeling" / "label_gestures.py").read_text()
    assert "person_group_fold" not in src and "person_from" not in src
    tm = L.tap_teacher()
    assert "file" in tm.spec["split"] and "person" not in tm.spec["split"].replace("never by person", "")
    keys = list(tm.fold_of)
    assert keys and all(re.fullmatch(r"[0-9a-f]{64}", k) for k in keys)
    assert set(tm.fold_of.values()) == {0, 1, 2, 3, 4}
    cm = L.contact_models()
    assert cm.fold_of == {}                                              # always the mean of the five
    X = np.zeros((3, tm.m[0].num_feature()), np.float32)
    p, k = tm.predict(X, "not-a-known-recording")
    assert k == -1 and np.allclose(p, np.mean([m.predict(X) for m in tm.m], 0))
    p, k = tm.predict(X, keys[0])
    assert k == tm.fold_of[keys[0]] and np.allclose(p, tm.m[k].predict(X))


TYPES = ["TAP|midtap|60", "TAP|softtap|120", "SWIPE|swipemid|60", "FINGER|pinky", "WRIST|wristcircle"]


def _rows_by_file():
    """Training rows of 12 files per recording type (people p0..p2 in every type); group = person|file."""
    g_, s_, y_ = [], [], []
    for st in TYPES:
        for k in range(12):
            n = 5 + k % 4
            g_ += [f"p{k % 3}|{st}_{k}.csv.gz"] * n; s_ += [st] * n; y_ += [k % 2] * n
    return dict(group=np.array(g_, object), stratum=np.array(s_, object), y=np.array(y_))


def _check_fold(data, fit_groups, val_groups):
    assert not (set(fit_groups) & set(val_groups))                       # a file is on one side only
    per_type = Counter(g.split("|", 1)[1].rsplit("_", 1)[0] for g in set(val_groups))
    assert set(per_type) == set(TYPES) and set(per_type.values()) == {4}  # every gesture, evenly (12 files / 3)
    people = lambda gs: {g.split("|")[0] for g in gs}
    assert people(fit_groups) == people(val_groups) == {"p0", "p1", "p2"}  # folds are by file, not by person


def test_feature_ranker_inner_folds_by_file_and_gesture():
    sys.path.insert(0, str(ROOT / "research" / "gesture"))
    import feature_selection as FS
    data = _rows_by_file()
    ids = np.arange(len(data["y"]))
    sp = FS.inner_splits(data, ids, seed=1)
    assert len(sp) == FS.INNER_FOLDS == 3
    seen = np.zeros(len(ids), int)
    for a, b in sp:
        _check_fold(data, data["group"][ids[a]], data["group"][ids[b]])
        seen[b] += 1
    assert (seen == 1).all()
    assert all((x[1] == z[1]).all() for x, z in zip(sp, FS.inner_splits(data, ids, seed=1)))     # reproducible
    plain = dict(group=data["group"], y=data["y"])                       # no strata: plain file-group folds
    for a, b in FS.inner_splits(plain, ids, seed=1):
        assert not (set(plain["group"][a]) & set(plain["group"][b]))


def test_trainer_folds_by_file_and_gesture():
    """The trainer's CV folds and its teacher folds (observable labels, distillation): gesture-aware file folds;
    without file groups it refuses (no person split)."""
    sys.path.insert(0, str(ROOT / "research" / "gesture"))
    import train_unified as U
    assert U.SPLIT_BY == "file"
    data = _rows_by_file()
    rows = np.arange(len(data["y"]))
    scored = np.zeros(len(rows), int)
    for fit, sc in U.teacher_folds(data, rows):
        _check_fold(data, data["group"][rows[fit]], data["group"][rows[sc]])
        scored[sc] += 1
    assert (scored == 1).all()
    for tr, va in U.cv_splits(data, rows):
        _check_fold(data, data["group"][rows[tr]], data["group"][rows[va]])
    import pytest
    with pytest.raises(ValueError):
        list(U.teacher_folds(dict(group=data["group"], y=data["y"]), rows))
    with pytest.raises(ValueError):
        U.cv_splits(dict(group=data["group"], y=data["y"]), rows)


if __name__ == "__main__":
    test_file_folds_keep_copies_together_and_balance_types(); print("ok folds")
    test_labeller_uses_file_folds_by_content(); print("ok labeller")
    test_feature_ranker_inner_folds_by_file_and_gesture(); print("ok ranker folds")
    test_trainer_folds_by_file_and_gesture(); print("ok trainer folds")
