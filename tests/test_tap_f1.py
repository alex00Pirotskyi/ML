"""The TAP F1 chain: file-level gesture-aware roadmap, inputs, dataset builder, the trainer's one line, event check."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "claude_labeling")]

import fold_split as FS  # noqa: E402
import tap_f1_features as F  # noqa: E402
import evaluate_tap_f1 as EV  # noqa: E402


# ------------------------------------------------------------------------------------------------ roadmap
def _csv(path: Path, rows: int, cls: int, seed: int, gz: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"class_id": cls, "Timestamp": np.arange(rows) * 10, "PointerState": 0,
                       "p_core->wl[CH1]": rng.integers(0, 100, rows)})
    text = df.to_csv(index=False).encode()
    if gz:
        path = path.with_name(path.name + ".gz")
        path.write_bytes(gzip.compress(text, mtime=0))
    else:
        path.write_bytes(text)
    return path


def _base(tmp: Path, files):
    doc = dict(schema_version=2, n_folds=5, default_fold_plan="person_disjoint_gesture_aware",
               identity_review_required={"alex": {}}, files=files, summary={}, warnings=[])
    p = tmp / "DATA" / "fold_roadmap.json"
    p.write_text(json.dumps(doc))
    return p


def _rec(fid, cat, gesture, rows, cls=1, bpm=60, sha=None, exclude=False, dup=None):
    name = f"p{fid}_{gesture}_Horizontal_Default_{bpm}bpm_2026-01-01_00-00-{int(fid):02d}.csv"
    return dict(file_id=f"file_{fid}", filename=name, gesture=gesture, bpm=bpm, sample_rows=rows,
                destinations=[dict(dataset="DATA_TAP", category=cat, class_id=cls,
                                   relative_csv_path=f"DATA_TAP/{cat}/{name}")],
                organized_content_sha256=sha or f"sha{fid}", content_sha256=sha or f"sha{fid}", recording_id=f"rec{fid}",
                fold_id=0, canonical_file_id=dup or f"file_{fid}", exclude_from_training=exclude, duplicate_of_file_id=dup)


def test_file_roadmap_is_stratified_stable_and_keeps_duplicates_together(tmp_path):
    files = [_rec(i, "TAP", "midtap", 100 + i) for i in range(10)]
    files += [_rec(20 + i, "FINGER", "fist", 50 + i, cls=3) for i in range(5)]
    files += [_rec(40, "TAP", "midtap", 100, sha="sha3", exclude=True, dup="file_3")]
    for f in files:
        d = f["destinations"][0]
        _csv(tmp_path / "DATA" / d["relative_csv_path"], f["sample_rows"], d["class_id"], int(f["file_id"][5:]))
    _csv(tmp_path / "DATA" / "DATA_TAP" / "FINGER" / "q_new_LeftHand_Pinky_Vertical_Default_60bpm_2026-11-24_15-08-08-246.csv",
         70, 3, 99)                                       # on disk, not in the base roadmap
    base = _base(tmp_path, files)
    out = tmp_path / "DATA" / "fold_roadmap_files.json"
    doc = FS.build_file_roadmap(base, tmp_path / "DATA", out, 5, log=lambda *a: None)
    by = {f["file_id"]: f for f in doc["files"]}
    assert doc["default_fold_plan"] == FS.FILE_PLAN and doc["identity_review_required"] == {}
    taps = [by[f"file_{i}"]["fold_id"] for i in range(10)]
    assert sorted(np.bincount(taps, minlength=5)) == [2, 2, 2, 2, 2]          # each gesture spread over all folds
    assert sorted(by[f"file_{20 + i}"]["fold_id"] for i in range(5)) == [0, 1, 2, 3, 4]
    assert by["file_40"]["fold_id"] == by["file_3"]["fold_id"]                  # a duplicate follows its recording
    new = [f for f in doc["files"] if f.get("added_by")]
    assert len(new) == 1 and new[0]["gesture"] == "pinky" and new[0]["destinations"][0]["class_id"] == 3
    # a rebuild keeps every fold; a new recording is dealt without moving the others
    _csv(tmp_path / "DATA" / "DATA_TAP" / "TAP" / "z_MidTap_Horizontal_Default_60bpm_2026-11-25_10-00-00.csv", 90, 1, 7)
    doc2 = FS.build_file_roadmap(base, tmp_path / "DATA", out, 5, log=lambda *a: None)
    by2 = {f["file_id"]: f for f in doc2["files"]}
    assert all(by2[k]["fold_id"] == v["fold_id"] for k, v in by.items())
    assert len(doc2["files"]) == len(doc["files"]) + 1


def test_roadmap_client_api(tmp_path):
    files = [_rec(i, "TAP", "midtap", 100) for i in range(5)] + [_rec(9, "TAP", "midtap", 100, sha="sha1", exclude=True, dup="file_1")]
    for f in files:
        _csv(tmp_path / "DATA" / f["destinations"][0]["relative_csv_path"], 100, 1, 1)
    base = _base(tmp_path, files)
    out = tmp_path / "DATA" / "fold_roadmap_files.json"
    FS.build_file_roadmap(base, tmp_path / "DATA", out, 5, log=lambda *a: None)
    c = FS.load_roadmap(out, tmp_path / "DATA", require_verified_identities=True)
    recs = c.get_records("DATA_TAP")
    assert len(recs) == 5 and all(r.path.name.endswith(".csv.gz") and r.path.is_file() for r in recs)
    name = files[2]["filename"]
    info = c.get_file_info(tmp_path / "elsewhere" / "TAP" / name, "DATA_TAP")          # a derived folder, plain .csv
    assert info.file_id == "file_2" and not info.exclude_from_training
    assert c.get_file_info(Path(name + ".gz"), "DATA_TAP").file_id == "file_2"
    assert c.get_file_info(Path(files[5]["filename"]), "DATA_TAP").exclude_from_training
    with pytest.raises(FS.RoadmapError):
        c.get_file_info(Path("unknown_file.csv"), "DATA_TAP")
    sp = c.get_split(0, "DATA_TAP")
    assert len(sp["test"]) == 1 and len(sp["train"]) == 4
    # the same recording content in train and test must stop the run
    doc = json.loads(out.read_text())
    leak = dict(doc["files"][0]); leak["file_id"] = "file_leak"; leak["exclude_from_training"] = False
    leak["fold_id"] = (doc["files"][0]["fold_id"] + 1) % 5
    leak["destinations"] = [dict(leak["destinations"][0], relative_csv_path="DATA_TAP/TAP/leak.csv")]
    doc["files"].append(leak); out.write_text(json.dumps(doc))
    with pytest.raises(FS.RoadmapError):
        FS.load_roadmap(out, tmp_path / "DATA").get_split(0, "DATA_TAP")


def test_repo_roadmap_is_file_level():
    p = ROOT / "DATA" / "fold_roadmap_files.json"
    if not p.exists():
        pytest.skip("no DATA/fold_roadmap_files.json in this checkout")
    c = FS.load_roadmap(p, ROOT / "DATA")
    assert c.plan == FS.FILE_PLAN and c.n_folds == 5
    recs = c.get_records("DATA_TAP")
    folds = np.bincount([r.fold_id for r in recs], minlength=5)
    assert folds.min() >= 0.95 * folds.max()
    for k in range(5):
        c.get_split(k, "DATA_TAP")


# ------------------------------------------------------------------------------------------------ inputs
def test_feature_pools_and_firmware_helpers():
    raw = [c for c in pd.read_csv(sorted((ROOT / "DATA" / "DATA_TAP" / "TAP").glob("*.csv.gz"))[0], nrows=0).columns
           if c.startswith("p_core->")] if (ROOT / "DATA" / "DATA_TAP" / "TAP").exists() else None
    if raw is None:
        pytest.skip("no DATA_TAP in this checkout")
    leg, newacc, new, lite = (F.pool(n, raw, wl=True) for n in F.POOLS)
    assert set(F.legacy_model_inputs()) <= set(leg) and len(leg) == 190
    assert set(leg) < set(newacc) < set(new) < set(lite)
    assert len(newacc) - len(leg) == 31          # 36 ACC ratios, 5 of them already inputs of the legacy model
    assert not F.needs_lite(new) and F.needs_lite(lite)
    # WL off (the default): no WL field or WL ratio anywhere, the legacy model's inputs all still there
    off = {n: F.pool(n, raw) for n in ("leg", "newacc", "new")}
    assert [len(off[n]) for n in ("leg", "newacc", "new")] == [181, 212, 346]
    assert not any(F.is_wl(x) for xs in off.values() for x in xs)
    assert set(F.legacy_model_inputs()) <= set(off["leg"])
    assert sum(F.is_wl(x) for x in new) == 18                                      # 9 fields + 9 ratios
    with pytest.raises(ValueError):
        F.pool("lite", raw)                                                        # Lean-lite is built from WL
    fm = F.FeatureMaker(["p_core->a", "p_core->b"], ["p_core->diff(p_core->a, p_core->b)",
                                                     "p_core->ratio(p_core->a, p_core->b)", "p_core->b"])
    X = fm(np.array([[30000, -30000], [5, 1]], np.float32))
    assert X[0, 0] == 32767 and X[1, 0] == 4                                       # int16 saturation
    assert X[1, 1] == pytest.approx(3.0) and X[0, 2] == -30000                     # (a + 1) / (b + 1)
    assert fm.float_columns == [1]
    with pytest.raises(KeyError):
        F.FeatureMaker(["p_core->a"], ["p_core->ratio(p_core->a, p_core->zz)"])


# ------------------------------------------------------------------------------------------------ builder + trainer
def _real_tap_file():
    d = ROOT / "DATA" / "DATA_TAP" / "TAP"
    fs = sorted(d.glob("*MidTap*.csv.gz")) if d.exists() else []
    return fs[0] if fs else None


def test_builder_writes_labels_inputs_and_passthrough(tmp_path):
    import build_tap_f1_dataset as B
    src = _real_tap_file()
    if src is None:
        pytest.skip("no DATA_TAP in this checkout")
    rel = src.relative_to(ROOT / "DATA").as_posix().removesuffix(".gz")
    out = tmp_path / "TAP" / Path(rel).name
    r = B.one((rel, str(out), "legacy", "newacc", 1.0, False))
    assert r["status"] == "written", r
    df = pd.read_csv(out)
    raw = pd.read_csv(src)
    assert len(df) == len(raw) and set(df.state.unique()) <= {0, 1} and (df.state == 1).sum() == r["tap_rows"] > 0
    assert (df.class_id == raw.class_id).all() and (df.timestamp == raw.Timestamp).all()
    names = F.pool("newacc", raw.columns)
    assert [c for c in df.columns if c.startswith("p_core->")] == names and not any(F.is_wl(c) for c in names)
    a, b = "p_core->acc_psd[0][Z][2]", "p_core->acc_psd_mean[Z][2]"
    np.testing.assert_allclose(df[f"p_core->ratio({a}, {b})"], (raw[a] + 1) / (raw[b] + 1), rtol=1e-6)


def test_trainer_one_line_drops_unknown_rows(tmp_path):
    pytest.importorskip("optuna"); pytest.importorskip("xgboost"); pytest.importorskip("pyarrow")
    import train_tap_f1 as T
    names = ["p_core->a", "p_core->b"]
    df = pd.DataFrame({"p_core->a": np.arange(10.0), "p_core->b": 1.0, "state": [0, 1, -1, -1, 1, 0, 0, -1, 1, 0],
                       "class_id": 1})
    p = tmp_path / "x.csv"; df.to_csv(p, index=False)
    res = T._process_file_worker((0, p, 2, 1), names, tmp_path, tmp_path, 1)
    assert res.success, res.error
    out = pd.read_parquet(tmp_path / "000000.parquet")
    assert len(out) == 7 and sorted(out.state.unique()) == [0, 1] and int(out.state.sum()) == 3
    assert list(out["p_core->a"]) == [0, 1, 4, 5, 6, 8, 9]                       # the -1 rows are gone, nothing else
    assert (out.roadmap_fold_id == 2).all()
    assert T.MAX_MODEL_SIZE_BYTES == 98304 and T.MIN_REALISTIC_SIZE == 5120 and T.DEFAULT_THRESHOLD == 0.85
    assert T.FOLD_ROADMAP_PATH.name == "fold_roadmap_files.json" and T.DATA_FOLDER == "DATA/DATA_TAP_F1/"


# ------------------------------------------------------------------------------------------------ event check
def test_events_refractory_presses_and_counts():
    t = np.arange(0, 2000, 10.0)
    s = np.zeros_like(t)
    s[(t >= 500) & (t < 530)] = 0.9; s[(t >= 600) & (t < 620)] = 0.95        # 2nd rise 100 ms later: refractory
    s[(t >= 1200) & (t < 1220)] = 0.9
    assert list(EV.events(t, s, 0.85)) == [500.0, 1200.0]
    ptr = np.zeros(len(t), bool); ptr[(t >= 480) & (t < 560)] = True
    ptr[(t >= 580) & (t < 600)] = True                                         # 30 ms after release: a bounce
    ptr[(t >= 1500) & (t < 1580)] = True                                       # no event: a missed tap
    assert list(EV.presses(t, ptr)) == [480.0, 1500.0]
    c = EV.Counts(ths=[0.85])
    c.add("TAP", "a_MidTap_x.csv", t, ptr, s)
    c.add("FINGER", "a_Fist&Stretch_x.csv", t, ptr, s)
    c.add("FINGER", "a_Pinky_x.csv", t, ptr, np.zeros_like(s))
    m = c.at(0.85)
    assert m["normal"] == 0.5 and m["taps"] == 0.5
    assert m["fpm_fist"] == pytest.approx(2 / (1990 / 6e4)) and m["fpm_finger"] == 0.0


def test_thin_tap_rows_only_thins_tap_recordings():
    import build_tap_f1_dataset as B
    y = np.array([0, 1, 1, 0, -1, 0, 1, 0], np.int8)
    t = B.thin_tap_rows(y, "TAP", 0.5)
    assert list(t) == [0, -1, 1, -1, -1, -1, 1, -1]                # every 2nd row trains; -1 stays -1
    assert list(B.thin_tap_rows(y, "FINGER", 0.5)) == list(y)       # movement recordings keep every row
    assert list(B.thin_tap_rows(y, "SWIPE", 1.0)) == list(y)
