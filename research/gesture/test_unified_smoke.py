#!/usr/bin/env python3
"""Real-source integration smoke. Never runs a full user Optuna experiment."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "gesture"))
import train_unified as g


def select_originals(source, category, per_person):
    files = sorted((source / category).glob("*.csv.gz"))
    selected, seen = [], set()
    for p in files:
        person = g.v6.person_group("", p.name[:-7])
        if person in seen:
            continue
        seen.add(person)
        selected.append(p)
        if len(selected) == per_person:
            break
    if not selected:
        raise ValueError(f"No actual recordings for {category}")
    return selected


def put_sample(orig, staging):
    dest = staging / orig.parent.name / orig.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(orig.resolve())


def main():
    sample_root=ROOT/"research"/"gesture"/"_smoke_input"
    tap_root=sample_root/"DATA_TAP"
    lp_root=sample_root/"DATA_LONG_PRESS"
    # Real recordings, diversified participants and gesture categories.
    for folder,n in (("TAP",22),("SWIPE",22),("FINGER",18),("ARM",16),
                     ("WRIST",14),("BAND",10)):
        for p in select_originals(g.SOURCE_TAP,folder,n):
            put_sample(p,tap_root)
    for folder,n in (("SOFT_LONG_PRESS",35),("MID_LONG_PRESS",35),
                     ("STRONG_LONG_PRESS",20),("DRAG_AND_DROP",35),
                     ("LONG_PRESS",4),("CONNECTED",16),("DISCONNECTED",16),
                     ("LOOSE",16),("TABLE",9),("TAP",12),("SWIPE",12)):
        for p in select_originals(g.SOURCE_LP,folder,n):
            put_sample(p,lp_root)
    g.SOURCE_TAP=tap_root
    g.SOURCE_LP=lp_root
    g.LABEL_TAP=ROOT/"research"/"gesture"/"_smoke_labels"/"DATA_TAP"
    g.LABEL_LP=ROOT/"research"/"gesture"/"_smoke_labels"/"DATA_LONG_PRESS"
    g.OUTPUT=ROOT/"research"/"gesture"/"_smoke_results"
    g.TRIALS_PER_MODEL=3
    g.TOP_K_PER_MODEL=2
    g.CV_FOLDS=2
    g.MAX_PER_CLASS_PER_RECORDING=16
    g.VERIFIED_HASHES=False
    # A smoke experiment needs actual source/label alignment, but no
    # CPU-hour-scale trial count or expensive full content digests.
    g.main()
    for task,stem in (("tap","TAP_SWIPE"),("lp","LONG_PRESS")):
        out=g.OUTPUT/task
        report=json.loads((out/"summary.json").read_text())
        assert report["status"]=="success"
        assert report["class_one"]==("RELEASE/REST" if task=="lp" else "TAP/SWIPE onset")
        assert report["fixed_threshold"]==0.85
        assert report["threshold_tuned"] is False
        assert len(report["ranking"])==2
        assert report["dataset"]["files"]>=20
        h=out/"models"/(stem+".h")
        text=h.read_text()
        assert f"#ifndef {stem}_MODEL_H" in text
        assert f"#define {stem}_predict {stem.lower()}_predict" in text
        assert f"#define {stem}_THRESHOLD 0.850000024f" in text
        assert "V6_RANK" not in text and "v6_rank" not in text
        schema=json.loads((out/"models"/(stem+"_feature_schema.json")).read_text())
        names=list(schema["feature_map"].values())
        assert names and all(s.startswith("p_core->") for s in names)
        assert not any(x in names for x in ("PointerState","Timestamp","hold_ms","ch1_contact"))
        print("UNIFIED_REAL_SOURCE_C_EXPORT_OK",task,report["dataset"],
              len(names),h,flush=True)
    print("UNIFIED_BOTH_TASKS_REAL_DATA_SMOKE_PASSED",flush=True)


if __name__=="__main__":
    main()
