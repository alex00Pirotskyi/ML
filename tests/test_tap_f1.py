"""The TAP F1 chain: file-level gesture-aware roadmap, inputs, dataset builder, the trainer's one line, event check."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
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
def _raw_columns():
    d = ROOT / "DATA" / "DATA_TAP" / "TAP"
    if not d.exists():
        pytest.skip("no DATA_TAP in this checkout")
    return [c for c in pd.read_csv(sorted(d.glob("*.csv.gz"))[0], nrows=0).columns if c.startswith("p_core->")]


def test_feature_pools_and_firmware_helpers():
    raw = _raw_columns()
    leg, newacc, new, lite = (F.pool(n, raw, wl=True) for n in ("leg", "newacc", "new", "lite"))
    assert set(F.legacy_model_inputs()) <= set(leg) and len(leg) == 190
    assert set(leg) < set(newacc) < set(new) < set(lite)
    assert len(newacc) - len(leg) == 31          # 36 ACC ratios, 5 of them already inputs of the legacy model
    assert not F.needs_lite(new) and F.needs_lite(lite)
    # WL off (the default): no WL field or WL ratio anywhere, the legacy model's inputs all still there
    off = {n: F.pool(n, raw) for n in ("leg", "newacc", "new", "lite")}
    assert [len(off[n]) for n in ("leg", "newacc", "new")] == [181, 212, 346]
    assert not any(F.is_wl(x) for xs in off.values() for x in xs)
    assert set(F.legacy_model_inputs()) <= set(off["leg"])
    assert sum(F.is_wl(x) for x in new) == 18                                      # 9 fields + 9 ratios
    assert sum(x in F.LITE_COLUMNS for x in off["lite"]) == 15                     # Lean-lite is relative: allowed
    with pytest.raises(ValueError):
        F.pool("rfe", raw)                                                         # needs the selected list
    assert F.pool("rfe", raw, selected=["p_core->acc_rate[X][0]"]) == ["p_core->acc_rate[X][0]"]
    fm = F.FeatureMaker(["p_core->a", "p_core->b"], ["p_core->diff(p_core->a, p_core->b)",
                                                     "p_core->ratio(p_core->a, p_core->b)", "p_core->b"])
    X = fm(np.array([[30000, -30000], [5, 1]], np.float32))
    assert X[0, 0] == 32767 and X[1, 0] == 4                                       # int16 saturation
    assert X[1, 1] == pytest.approx(3.0) and X[0, 2] == -30000                     # (a + 1) / (b + 1)
    assert fm.float_columns == [1]
    with pytest.raises(KeyError):
        F.FeatureMaker(["p_core->a"], ["p_core->ratio(p_core->a, p_core->zz)"])


def test_candidates_are_pairs_inside_one_modality_without_wl():
    raw = _raw_columns()
    c = F.candidates(raw, emg="all")
    assert len(c) == len(set(c)) == 5127
    assert not any(F.is_wl(x) for x in c) and sum(x in F.LITE_COLUMNS for x in c) == 15
    assert set(F.legacy_model_inputs()) <= set(c)                                 # the old pool is inside, same names
    calls = [x for x in c if x.startswith(("p_core->diff(", "p_core->ratio("))]
    assert len(calls) == 2 * 1431 + 2 * 990 + 153
    for x in calls:                                                                # never ACC with EMG
        assert not ("acc_" in x and "emg_" in x)
        if x.startswith("p_core->ratio("):                                         # ratios only over PSD codes >= 0
            assert "acc_rate" not in x
    fm = F.FeatureMaker(raw + F.LITE_COLUMNS, c)                                    # every candidate computes
    assert len(fm.names) == 5127
    assert len(F.candidates(raw, lite=False, emg="all")) == 5112


def test_wl_wl_head_and_wl_tail_are_all_removed_by_default():
    raw = _raw_columns()
    wl = [c for c in raw if re.search(r"->wl(_head|_tail)?\[", c)]
    assert len(wl) == 9 and all(F.is_wl(c) for c in wl)                  # wl, wl_head, wl_tail x 3 channels
    assert F.DEFAULT_WL is False
    for emg in F.EMG_MODES:                                                # no field, no ratio, no diff on any of them
        assert not any("wl" in x.lower() for x in F.candidates(raw, emg=emg))
    for name in F.POOLS:
        sel = F.candidates(raw)[:50] if name == "rfe" else None
        assert not any("wl" in x.lower() for x in F.pool(name, raw, selected=sel))
    for c in wl:                                                           # a WL input in the selection is refused
        with pytest.raises(ValueError):
            F.pool("rfe", raw, selected=[c])
    assert any("wl" in x.lower() for x in F.candidates(raw, wl=True))      # WL=1 brings them back
    import export_tap_bundle as EX
    assert all(EX.WL_FIELD.search(c) for c in wl) and not EX.WL_FIELD.search("p_core->cload_lite.rise10[CH1]")


def test_default_emg_candidates_are_diffs_inside_one_channel():
    raw = _raw_columns()
    assert F.DEFAULT_EMG == "within"
    c = F.candidates(raw)
    assert len(c) == len(set(c)) == 3237
    emg = [x for x in c if "emg_psd" in x]
    assert len(emg) == 135 and all(x.startswith("p_core->diff(") for x in emg)    # no ratio, no absolute code
    over_time = 0
    for x in emg:
        a, b = re.match(r"^p_core->diff\((p_core->.+?), (p_core->.+)\)$", x).groups()
        assert F.emg_within(a, b)
        over_time += F.emg_over_time(a, b)
    assert over_time == 45                                                           # + 90 zone vs zone
    assert F.emg_within("p_core->emg_psd[0][CH2][3]", "p_core->emg_psd[0][CH2][4]")              # shape
    assert not F.emg_within("p_core->emg_psd[0][CH2][3]", "p_core->emg_psd_mean[CH2][4]")        # slot and zone
    assert not F.emg_within("p_core->emg_psd[0][CH2][3]", "p_core->emg_psd[0][CH1][3]")          # channel vs channel
    assert [x for x in c if "emg_psd" not in x] == [x for x in F.candidates(raw, emg="all") if "emg_psd" not in x]
    assert len(F.FeatureMaker(raw + F.LITE_COLUMNS, c).names) == 3237


def test_augmentation_is_exact_on_codes_and_seeded_by_name():
    import tap_f1_augment as AUG
    raw = _raw_columns()
    rng = np.random.default_rng(0)
    B = rng.integers(0, 20000, (50, len(raw))).astype(np.float32)
    emg, wl = AUG.indices(raw)
    assert AUG.draw("a.csv")["gain"].tolist() == AUG.draw("a.csv")["gain"].tolist()       # same file, same draw
    assert AUG.draw("a.csv")["gain"].tolist() != AUG.draw("b.csv")["gain"].tolist()
    ident = dict(gain=np.ones(3), tilt=np.ones((3, 5)), add=np.zeros((3, 5)), mix=0.0)
    assert np.array_equal(AUG.apply(B, raw, ident), B)                                       # nothing changes
    g = dict(ident, gain=np.array([2.0, 1.0, 1.0]))                                          # +6 dB on CH1
    Y = AUG.apply(B, raw, g)
    p = np.power(10.0, B[:, emg[:, 0].reshape(-1)] / 1e4) - 1
    want = np.clip(np.round(1e4 * np.log10(1 + 4 * p)), 0, 65535)
    assert np.array_equal(Y[:, emg[:, 0].reshape(-1)], want)                                 # power x 4, re-encoded
    assert np.array_equal(Y[:, emg[:, 1:].reshape(-1)], B[:, emg[:, 1:].reshape(-1)])        # other channels kept
    acc = [i for i, c in enumerate(raw) if c.startswith("p_core->acc_")]
    for k in AUG.STRESS_KINDS + ("train",):
        d = AUG.draw("x.csv") if k == "train" else AUG.stress("x.csv", k)
        Y = AUG.apply(B, raw, d)
        assert np.array_equal(Y[:, acc], B[:, acc])                                          # ACC untouched
        assert np.isfinite(Y).all() and (Y[:, emg.reshape(-1)] >= 0).all()
    h = AUG.stress("x.csv", "hum2")["add"]
    assert (h[:, 2] > 0).all() and not h[:, [0, 1, 3, 4]].any()                              # one zone only


def test_robust_emg_candidates_are_change_over_time_inside_one_channel_and_zone():
    raw = _raw_columns()
    c = F.candidates(raw, emg="robust")
    assert len(c) == len(set(c)) == 3192
    emg = [x for x in c if "emg_psd" in x]
    assert len(emg) == 90 and all(x.startswith(("p_core->diff(", "p_core->ratio(")) for x in emg)   # no absolute code
    for x in emg:
        a, b = re.match(r"^p_core->(?:diff|ratio)\((p_core->.+?), (p_core->.+)\)$", x).groups()
        assert F.emg_over_time(a, b)                                               # same channel and zone, other slot
    assert F.emg_over_time("p_core->emg_psd[0][CH2][3]", "p_core->emg_psd_mean[CH2][3]")
    assert not F.emg_over_time("p_core->emg_psd[0][CH2][3]", "p_core->emg_psd_mean[CH1][3]")   # channel vs channel
    assert not F.emg_over_time("p_core->emg_psd[0][CH2][3]", "p_core->emg_psd[0][CH2][4]")     # zone vs zone
    assert [x for x in c if "emg_psd" not in x] == [x for x in F.candidates(raw) if "emg_psd" not in x]   # rest as before
    assert len(F.FeatureMaker(raw + F.LITE_COLUMNS, c).names) == 3192


def test_rfe_keeps_the_informative_input_and_drops_twins(tmp_path):
    pytest.importorskip("lightgbm")
    import select_tap_f1_features as SEL
    rng = np.random.default_rng(1)
    n, cols = 6000, [f"p_core->acc_psd[0][X][{i}]" for i in range(50)]
    B = rng.normal(100, 20, (n, len(cols))).astype(np.float32)
    B[:, 1] = B[:, 0]                                                              # an exact twin of field 0
    y = ((B[:, 2] - B[:, 3]) > 25).astype(np.int8)                                 # the tap depends on 2 - 3 only
    fold = np.repeat(np.arange(1, 5), n // 4).astype(np.int8)
    np.save(tmp_path / "base.npy", B); np.save(tmp_path / "y.npy", y); np.save(tmp_path / "fold.npy", fold)
    (tmp_path / "meta.json").write_text(json.dumps(dict(base_cols=cols)))
    key = "p_core->diff(p_core->acc_psd[0][X][2], p_core->acc_psd[0][X][3])"
    cands = cols + [key] + [f"p_core->ratio({cols[i]}, {cols[i + 1]})" for i in range(10, 30)]
    selected, report = SEL.select(tmp_path, cands, workers=1, threads=1, log=lambda *a: None)
    assert key in selected
    assert not (cols[0] in selected and cols[1] in selected)                       # twins: one of them at most
    assert SEL.MIN_KEEP <= len(selected) <= SEL.MAX_KEEP and report["rounds"][0]["n"] >= len(selected)
    assert selected == sorted(selected, key=cands.index)                           # candidate order
    assert "kept" in SEL.report_markdown(report)


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
    import build_tap_f1_dataset as B
    assert B.TEST_FOLD == T.TEST_FOLD_ID                         # the feature elimination never sees the test fold


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


# ------------------------------------------------------------------------------------------------ back test, bundle
def test_backtest_replay_counts_taps_doubles_and_false_events():
    import backtest_tap_f1 as BT
    t = np.arange(0, 6000, 10.0)
    q = np.zeros(len(t), np.int16)
    ptr = np.zeros(len(t), bool)
    for on in (500, 2500):                                   # two single taps, the 2nd one missed by the model
        ptr[(t >= on) & (t < on + 80)] = True
    q[(t >= 520) & (t < 560)] = 19                           # tap 1: onset 20 ms after the press
    r = BT.replay("TAP", "x_MidTap_Vertical_Default_60bpm_x.csv", t, ptr, q, fires=(18,))
    s = BT.summarize(r[18])
    assert s["taps"] == 0.5 and s["presses"] == 2 and s["single_as_double"] == 0.0
    pair = np.zeros(len(t), bool); qd = np.zeros(len(t), np.int16)
    pair[(t >= 1000) & (t < 1060)] = True; pair[(t >= 1250) & (t < 1310)] = True      # a double tap, 250 ms apart
    qd[(t >= 1010) & (t < 1050)] = 19; qd[(t >= 1260) & (t < 1290)] = 16             # dip between, 2nd >= SECOND
    d = BT.summarize(BT.replay("TAP", "x_DoubleTaps_Vertical_Default_40bpm_x.csv", t, pair, qd, fires=(18,))[18])
    assert d["double_pairs"] == 1.0 and d["taps"] == 1.0
    qn = np.zeros(len(t), np.int16); qn[(t >= 3000) & (t < 3050)] = 19            # one false tap in a fist recording
    n = BT.summarize(BT.replay("FINGER", "x_Fist&Stretch_Horizontal_x.csv", t, np.zeros(len(t), bool), qn, fires=(18,))[18])
    assert n["false_fist_pm"] == pytest.approx(1 / (5990 / 6e4)) and n["false_finger_pm"] == 0.0
    assert n["false_weighted_pm"] == pytest.approx(n["false_fist_pm"])
    assert n["U_se"] > 0                       # one event in 0.1 min: the false rate is far from known
    assert BT.levels([0.0, 0.0499, 0.05, 0.8999, 0.9, 1.0]).tolist() == [0, 0, 1, 17, 18, 20]


def test_backtest_fire_is_the_highest_within_one_standard_error_of_the_best():
    import backtest_tap_f1 as BT
    curve = {12: dict(U=75.0, U_se=4.0), 13: dict(U=74.0, U_se=4.0), 14: dict(U=71.5, U_se=4.0),
             15: dict(U=70.0, U_se=4.0), 16: dict(U=60.0, U_se=4.0)}
    assert BT.pick_fire(curve, 1.0) == (14, 12)       # 71.5 >= 75 - 4; 70.0 is not
    assert BT.pick_fire(curve, 0.0) == (12, 12)       # BACKTEST_SE=0: the best U as is
    assert BT.pick_fire(curve, 2.0) == (15, 12)


FAKE_MODEL = """#ifndef TAP_MODEL_H
#define TAP_MODEL_H
#include <math.h>
#include <stdint.h>
#include "core_type_def.h"
#define TAP_NUM_FEATURES 1u
#define TAP_NUM_TREES 1u
static const float tap_q20_raw_thresholds[19] = { %s };
static inline float tap_predict_raw_score(const t_core* p_core) { return (float) p_core->acc_rate[X][0] / 100.0f; }
static inline int8_t tap_predict_q20(const t_core* p_core, uint8_t threshold_q20)
{
    if (threshold_q20 == 0u) { return 1; }
    if (threshold_q20 >= 20u) { return 0; }
    return tap_predict_raw_score(p_core) >= tap_q20_raw_thresholds[threshold_q20 - 1u] ? 1 : 0;
}
static inline uint8_t tap_probability_q20(const t_core* p_core)
{
    uint8_t k = 0u;
    while (k < 19u && tap_predict_raw_score(p_core) >= tap_q20_raw_thresholds[k]) { ++k; }
    return k;
}
#endif /* TAP_MODEL_H */
"""


@pytest.mark.parametrize("immediate,chain", [(0, 0), (1, 0), (1, 1)])
def test_bundle_tap_logic_in_c_matches_the_python_twin(tmp_path, immediate, chain):
    """the generated tap_gestures.c (state machine, early-exit query while idle, full score while pending) fires
    exactly like research/gesture/tap_gestures.py on random score streams"""
    import ctypes
    import shutil
    import subprocess
    sys.path.insert(0, str(ROOT / "research" / "gesture"))
    import export_tap_bundle as EB
    import tap_gestures as TG
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        pytest.skip("no C compiler")
    logit = lambda p: np.log(p / (1 - p))
    thr = np.float32([logit(k / 20) for k in range(1, 20)])
    code, facts, fields = EB.model_code(FAKE_MODEL % ", ".join(f"{float(v):.9e}f" for v in thr))
    assert fields == ["p_core->acc_rate[X][0]"]
    choice = dict(fire_q20=17, second_q20=13, dip_q20=4, min_gap_ms=150, window_ms=400, lockout_ms=300, fold=0,
                  recordings=0, nominal={k: 0.0 for k in ("taps", "soft", "fast240", "double_pairs", "single_as_double",
                                                            "false_pm", "false_fist_pm", "false_finger_pm",
                                                            "false_wrist_pm", "false_arm_pm", "false_band_pm")})
    fw = tmp_path / "fw"; fw.mkdir()
    (fw / "core_typedef.h").write_text(EB.core_typedef_h("fake", fields, "now"))
    (fw / "tap_gestures.h").write_text(EB.tap_gestures_h("fake", choice, facts, None, "now"))
    (fw / "tap_gestures.c").write_text(EB.tap_gestures_c("fake", code, "now"))
    (tmp_path / "run.c").write_text("""#include <string.h>
#include "tap_gestures.h"
void run(const short* raw100, const unsigned* t, long n, signed char* ev)
{
    t_core c; tg_state_t s; memset(&c, 0, sizeof c); tg_init(&s, t[0]);
    for (long i = 0; i < n; ++i) { c.acc_rate[X][0] = raw100[i]; ev[i] = (signed char) tg_process(&s, &c, t[i]); }
}
""")
    lib = tmp_path / "libtg.so"
    r = subprocess.run([cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "-DTG_HOST_CORE",
                        f"-DTG_REPORT_TAP_IMMEDIATELY={immediate}", f"-DTG_CHAIN_TAPS={chain}", "-I", str(fw),
                        str(fw / "tap_gestures.c"), str(tmp_path / "run.c"), "-o", str(lib), "-lm"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    so = ctypes.CDLL(str(lib))
    so.run.argtypes = [ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_uint), ctypes.c_long, ctypes.POINTER(ctypes.c_byte)]
    cfg = TG.Config(fire=17, second=13, dip=4, min_gap_ms=150, window_ms=400, lockout_ms=300,
                    immediate=bool(immediate), chain=bool(chain))
    rng = np.random.default_rng(immediate * 2 + chain)
    total = 0
    for k in range(40):
        n = 3000
        t = np.cumsum(rng.integers(4, 9, n)).astype(np.uint32) + np.uint32(4294960000 if k == 0 else 0)  # k=0: clock wraps
        base = rng.normal(-4.0, 1.0, n)
        for on in rng.choice(n - 40, 25, replace=False):         # taps and close pairs
            base[on:on + rng.integers(3, 12)] = rng.uniform(0.5, 4.0)
        raw100 = np.clip(np.round(base * 100), -32000, 32000).astype(np.int16)
        q = (raw100.astype(np.float32)[:, None] / np.float32(100) >= thr[None, :]).sum(1)     # the C's level
        ev = np.zeros(n, np.int8)
        so.run(raw100.ctypes.data_as(ctypes.POINTER(ctypes.c_short)), t.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)), n,
               ev.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)))
        tt = t.astype(np.float64)
        if k == 0:                                                 # the twin uses floats: unwrap the clock
            tt = np.cumsum(np.r_[0.0, np.diff(t.astype(np.int64)) % 2 ** 32]) + float(t[0])
        want = TG.step_all(q, tt, cfg)
        got = [(int(i), int(e)) for i, e in zip(np.flatnonzero(ev), ev[ev != 0])]
        assert got == want, (k, got[:5], want[:5])
        total += len(want)
    assert total > 200
