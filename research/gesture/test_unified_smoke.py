#!/usr/bin/env python3
"""Checks for the unified trainer v2. Never runs a full user Optuna experiment.

1. Unit checks: TAP event replay, LP ten-sample press replay, exact budget trimming,
   C-header replay == Python, class weighting, Lean-8 emulator == a literal port of the
   firmware handler (and == the firmware C code when its header is available locally),
   export/compile/replay of a model that uses Lean-8 fields.
2. Real-source smoke: a small set of real recordings through Lean-8 generation, labels,
   sampling, replay CV, budget-trimmed C export, held-out C replay and baseline replay.
"""
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "gesture"))
import train_unified as g
import replay as R
import generate_lean8_features as L8


def lean8_handler_reference(wl_frames, psd_mean_frames):
    """Literal Python port of lean_8_features_handler (ring buffer, running integer sums,
    rescan on expiry), independent of the vectorised emulator."""
    N10, N3 = L8.HISTORY_10S, L8.HISTORY_3S
    hist = np.zeros((N10, 3), np.int64); sum3 = [0, 0, 0]; sumsq3 = [0, 0, 0]
    first = [0, 0, 0]; mn = [0, 0, 0]; mx = [0, 0, 0]; head = 0; seen = 0
    inv = np.float32(1.0) / (np.float32(N3) * np.float32(N3) * np.float32(4096) * np.float32(4096))
    out = np.zeros((len(wl_frames), 8, 3), np.float32)
    for f in range(len(wl_frames)):
        expire3 = (head + N10 - N3) % N10
        for ch in range(3):
            lw = L8.fw_log10_positive(np.array([np.float32(wl_frames[f, ch]) + np.float32(1.0)], np.float32))[0]
            q = min(int(np.float32(lw * np.float32(4096.0) + np.float32(0.5))), 65535)
            leaving10 = int(hist[head, ch]); leaving3 = int(hist[expire3, ch]) if seen >= N3 else 0
            hist[head, ch] = q
            if seen == 0:
                first[ch] = mn[ch] = mx[ch] = q
            elif seen < N10:
                mn[ch] = min(mn[ch], q); mx[ch] = max(mx[ch], q)
            elif leaving10 in (mn[ch], mx[ch]):
                mn[ch] = int(hist[:, ch].min()); mx[ch] = int(hist[:, ch].max())
            else:
                mn[ch] = min(mn[ch], q); mx[ch] = max(mx[ch], q)
            sum3[ch] += q; sumsq3[ch] += q * q
            if seen >= N3:
                sum3[ch] -= leaving3; sumsq3[ch] -= leaving3 * leaving3
            ts, tq = sum3[ch], sumsq3[ch]
            if seen + 1 < N3:
                pad = N3 - seen - 1; ts += pad * first[ch]; tq += pad * first[ch] * first[ch]
            var = np.float32(tq * N3 - ts * ts) * inv
            sd = np.float32(np.sqrt(var))
            min10 = np.float32(mn[ch]) * np.float32(1 / 4096); max10 = np.float32(mx[ch]) * np.float32(1 / 4096)
            x = L8.fw_psd_linear(psd_mean_frames[f, ch])
            power = np.where(x > np.float32(1e-4), x, np.float32(1e-4)); total = np.float32(0)
            for z in range(5):
                total = np.float32(total + x[z])
            total = max(total, np.float32(1e-4))
            lg = lambda v: L8.fw_log10_positive(np.array([v], np.float32))[0]
            out[f, :, ch] = (lw, np.float32(10) * lg(total), np.float32(10) * lg(power[0] / power[1]),
                             np.float32(10) * lg(power[3] / power[2]), sd, min10,
                             (lw - min10) if lw > min10 else np.float32(0), max10)
        seen = min(seen + 1, N10); head = (head + 1) % N10
    return out.reshape(len(wl_frames), 24)


def lean8_checks():
    rng = np.random.default_rng(5)
    n = 1300                                    # > 500 frames: warm-up, expiry rescans and steady state
    base = np.cumsum(rng.normal(0, 40, n)); wl = np.clip(np.stack([base + 900, base * 0.3 + 400, 2000 - base], 1), 0, None)
    wl[700:760] = 6000; wl[1000:1010, 1] = 0                 # bursts, a zero WL, ramps
    wl = np.round(wl).astype(np.int64)
    psd = rng.integers(0, 9000, size=(n, 3, 5)); psd[50:60, 2, :] = 0   # include zero-power zones
    ours = L8.lean8_per_frame(wl, psd)
    ref = lean8_handler_reference(wl, psd)
    assert np.array_equal(ours, ref), np.abs(ours - ref).max(axis=0)
    # Frame detection: PSD changes every 3 rows; one frame logged split (PSD first, WL next row).
    rows = 30; P = np.repeat(np.arange(10), 3)[:, None].repeat(15, 1); W = np.repeat(np.arange(10), 3)[:, None].repeat(3, 1)
    W[9] = W[8]                                            # frame 3 (rows 9-11): WL lands on row 10
    fr = L8.frame_rows(P, W)
    assert fr.tolist() == [0, 3, 6, 10, 12, 15, 18, 21, 24, 27], fr
    fw = L8.firmware_header()
    if fw is not None:
        files = sorted(g.RAW_SOURCE_TAP.glob("TAP/*.csv.gz"))[:3] + sorted(g.RAW_SOURCE_LP.glob("MID_LONG_PRESS/*.csv.gz"))[:2]
        par = L8.parity_check(files, fw)
        assert par["status"] == "passed" and par["max_abs_diff"] == 0.0, par
        lit = L8.FirmwareLean8(fw).run(wl, psd)[0]
        assert np.array_equal(lit, ours)
        print("LEAN8_FIRMWARE_C_PARITY_OK", par["frames"], "frames", flush=True)
    else:
        print("LEAN8_FIRMWARE_C_PARITY_SKIPPED (no local firmware header)", flush=True)
    print("LEAN8_EMULATOR_CHECKS_PASSED", flush=True)


def unit_checks():
    # TAP: onsets 1000/3000/5000; hits at 1020 and 3010 (+ a second edge at 3060 inside the refractory),
    # one false event on a negative row, one event on an unknown row (not counted), onset 5000 missed.
    t = np.arange(0, 10000, 5.0); lab = np.zeros(len(t), np.int8); lab[(t >= 8900) & (t <= 9200)] = -1
    for o in (1000, 3000, 5000):
        lab[(t >= o) & (t <= o + 80)] = 1
    d = np.zeros(len(t), bool)
    for a, b in ((1020, 1060), (3010, 3040), (3060, 3070), (7000, 7020), (9000, 9030)):
        d[(t >= a) & (t <= b)] = True
    c = R.TapCounts(); R.tap_replay_recording(d, t, lab, [1000, 3000, 5000], "TAP", c); r = c.report()
    assert (r["onsets"], r["hits"], r["false_events"]) == (3, 2, 1), r
    # LP: a hold released at 2000 ms; release values from 2000 ms -> declared at 2040 ms (9 of 10 samples = 0.9)
    t = np.arange(0, 4000, 5.0)
    c = R.LpCounts(); R.lp_replay_press(np.where(t < 2000, 0, 20), t, 0.0, 2000.0, "hold", "MID", c)
    assert c.kept == 1 and c.timely == 1 and c.latency == [40.0], c.latency
    c = R.LpCounts(); R.lp_replay_press(np.where(t < 1000, 0, 20), t, 0.0, 2000.0, "hold", "MID", c)
    assert c.kept == 0 and c.timely == 0                       # premature release breaks the hold
    # A model that NEVER releases must score ZERO.
    c = R.LpCounts(); R.lp_replay_press(np.zeros(len(t), int), t, 0.0, 2000.0, "hold", "MID", c)
    R.lp_replay_press(np.zeros(len(t), int), t, 0.0, 150.0, "tap", "TAP", c)
    rep = c.report(); assert rep["hold_kept"] == 1.0 and rep["timely_release"] == 0.0 and rep["objective_score"] == 0.0
    assert R.lp_declared_release(np.array([20] * 9 + [0]), np.arange(10) * 5.0, 0.0) is not None
    assert R.lp_declared_release(np.array([20] * 8 + [0, 0]), np.arange(10) * 5.0, 0.0) is None
    # LP release threshold is explicit: 0.5 releases when the ten-sample mean reaches 0.5 (sum of Q20 >= 100).
    assert R.lp_declared_release(np.array([10] * 10), np.arange(10) * 5.0, 0.0, threshold=0.5) is not None
    assert R.lp_declared_release(np.array([10] * 9 + [9]), np.arange(10) * 5.0, 0.0, threshold=0.5) is None
    assert R.lp_declared_release(np.array([10] * 10), np.arange(10) * 5.0, 0.0) is None    # default stays 0.85
    # Positive weight must change class weighting (it was inert in the old V6 helper).
    yy = np.array([0] * 8 + [1] * 8, dtype=np.int8)
    d = {"y": yy, "w": np.ones(16, np.float32), "file": np.array(["one"] * 8 + ["two"] * 8, dtype=object),
         "category": np.array(["SOFT_LONG_PRESS"] * 16, dtype=object)}
    a = g.weights_for_task(d, np.arange(16), "lp", 1.0); b = g.weights_for_task(d, np.arange(16), "lp", 6.0)
    assert np.allclose(a[yy == 0], b[yy == 0]) and np.allclose(b[yy == 1], 6 * a[yy == 1])
    # Budget trimming is exact, and the exported C header reproduces Python.
    import lightgbm as lgb
    rng = np.random.default_rng(0); X = rng.normal(size=(20000, 126)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 5] * X[:, 9] + rng.normal(size=20000) > 0).astype(int)
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, max_depth=8, verbosity=-1, n_jobs=2, max_bin=127).fit(X, y)
    for kb in (32, 24, 8):
        b = int(kb * 1024); k, size = g.trees_within(m, b)
        assert size <= b and g.static_bytes(g.truncated(m, k)) == size
        assert k == m.booster_.num_trees() or g.static_bytes(g.truncated(m, k + 1)) > b
    # Optimized sizing == original truncate-and-measure search, incl. the >64-node metadata layout and the
    # root-index width change; every prefix size equals the exporter's own accounting of the truncated model.
    big = lgb.LGBMClassifier(n_estimators=80, num_leaves=127, max_depth=10, min_child_samples=5, verbosity=-1, n_jobs=2,
                             max_bin=127).fit(X, y)
    for model in (m, big):
        for b in (2048, 6144, 8192, 24 * 1024, 32 * 1024, 32 * 1024 + 7, 10 ** 6):
            try:
                ref = g.trees_within_reference(model, b)
            except ValueError:
                ref = "too small"
            try:
                got = g.trees_within(model, b)
            except ValueError:
                got = "too small"
            assert ref == got, (b, ref, got)
        sz = g._sizing(model)
        for k in range(1, model.booster_.num_trees() + 1, 7):
            assert sz["prefix"](k) == g.static_bytes(g.truncated(model, k)), k
    assert not g.SIZING_FALLBACKS, g.SIZING_FALLBACKS
    # Replay gather: selected columns only, identical bytes to concatenate-then-select; predictions identical.
    Xmm = np.random.default_rng(3).normal(size=(5000, 40)).astype(np.float32)
    ranges = [(0, 7), (10, 900), (900, 905), (1200, 4999)]
    for sel in ([3, 0, 39, 7], list(range(40)), slice(None)):
        ref = np.ascontiguousarray(np.concatenate([Xmm[a:b] for a, b in ranges])[:, sel], np.float32)
        got = g._gather(Xmm, ranges, sel)
        assert got.flags.c_contiguous and got.dtype == np.float32 and np.array_equal(ref.view(np.uint32), got.view(np.uint32))
        fn = lambda Z: {"s": Z.sum(axis=1), "f": Z[:, 0] * 3}
        new = list(g._ranges_predict([fn], Xmm, ranges, sel, chunk=700))
        old = list(g._ranges_predict_reference([fn], Xmm, ranges, sel, chunk=700))
        assert [r for r, _ in new] == [r for r, _ in old]
        assert all(np.array_equal(a[0][k], b[0][k]) for (_, a), (_, b) in zip(new, old) for k in ("s", "f"))
    assert len(g.LEAN8_24) == 24
    g.export_plus.validate_feature_schema_accessors(g.LEAN8_24)
    out = Path(tempfile.mkdtemp()); cols = list(g.RAW_126)
    mk = g.truncated(m, g.trees_within(m, 24 * 1024)[0])
    h, size, _ = g.write_header(mk, cols, "TAP_SWIPE", out / "models", "tap", out)
    hm = R.HeaderModel(h, cols, out / "build", prefer="tap_swipe")
    Xt = rng.normal(size=(5000, 126)).astype(np.float32); p = mk.predict_proba(Xt)[:, 1]
    assert (hm.predict_bool(Xt).astype(bool) == (p >= 0.85)).all()
    assert (hm.value_q20(Xt) == R.q20(p)).all()
    # Selection guard: a CV winner that never fires on held-out people is skipped, the next one is promoted.
    tmp = Path(tempfile.mkdtemp()); cands = []
    for role, score in (("rank_01", 0.0), ("rank_02", 0.4), ("budget_6kb", 0.3)):
        d_ = tmp / "models" / "candidates" / role; d_.mkdir(parents=True)
        (d_ / "X.h").write_text(role); (d_ / "X_feature_schema.json").write_text("{}")
        cands.append(dict(role=role, trial=len(cands), header=str((d_ / "X.h").relative_to(tmp)),
                          holdout_c_header=dict(objective_score=score)))
    chosen, skipped = g.promote(tmp, "X", cands, [8192, 6144])
    assert chosen["role"] == "rank_02" and skipped[0]["role"] == "rank_01", (chosen, skipped)
    assert (tmp / "models" / "X.h").read_text() == "rank_02"
    assert (tmp / "models" / "budget_6kb" / "X.h").read_text() == "budget_6kb"
    # A model that uses Lean-8 fields exports, compiles against the 150-field t_core and replays exactly.
    cols = list(g.ALL_FEATURES)
    X8 = rng.normal(size=(20000, len(cols))).astype(np.float32)
    lw1 = cols.index("p_core->cload_lean_8.lw[CH1]"); r01 = cols.index("p_core->cload_lean_8.rise10[CH3]")
    y8 = (X8[:, lw1] + X8[:, r01] + 0.3 * X8[:, 0] + rng.normal(size=20000) > 0.5).astype(int)
    m8 = lgb.LGBMClassifier(n_estimators=60, num_leaves=15, verbosity=-1, n_jobs=2, max_bin=127).fit(X8, y8)
    h8, _, chk = g.write_header(m8, cols, "LONG_PRESS", out / "lean8_models", "lp", out)
    assert "cload_lean_8.lw[CH1]" in h8.read_text() and chk["status"] == "passed", chk
    hm8 = R.HeaderModel(h8, cols, out / "build8", prefer="long_press")
    p8 = m8.predict_proba(X8[:4000])[:, 1]
    assert (hm8.value_q20(X8[:4000]) == R.q20(p8)).all()
    print("UNIFIED_V2_UNIT_CHECKS_PASSED", flush=True)


def select_originals(source, category, per_person):
    files = sorted((source / category).glob("*.csv.gz"))
    selected, seen = [], set()
    for p in files:
        person = g.v6.person_group("", p.name[:-7])
        if person in seen:
            continue
        seen.add(person); selected.append(p)
        if len(selected) == per_person:
            break
    if not selected:
        raise ValueError(f"No actual recordings for {category}")
    return selected


def put_sample(orig, staging):
    dest = staging / orig.parent.name / orig.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.symlink_to(orig.resolve())


def real_source_smoke():
    sample_root = ROOT / "research" / "gesture" / "_smoke_input"
    for d in (sample_root, ROOT / "research" / "gesture" / "_smoke_labels", ROOT / "research" / "gesture" / "_smoke_results"):
        if d.exists():
            shutil.rmtree(d)
    tap_root = sample_root / "DATA_TAP"; lp_root = sample_root / "DATA_LONG_PRESS"
    for folder, n in (("TAP", 22), ("SWIPE", 22), ("FINGER", 18), ("ARM", 16), ("WRIST", 14), ("BAND", 10)):
        for p in select_originals(g.RAW_SOURCE_TAP, folder, n):
            put_sample(p, tap_root)
    for folder, n in (("SOFT_LONG_PRESS", 35), ("MID_LONG_PRESS", 35), ("STRONG_LONG_PRESS", 20),
                      ("DRAG_AND_DROP", 35), ("LONG_PRESS", 4), ("CONNECTED", 16), ("DISCONNECTED", 16),
                      ("LOOSE", 16), ("TABLE", 9), ("TAP", 12), ("SWIPE", 12)):
        for p in select_originals(g.RAW_SOURCE_LP, folder, n):
            put_sample(p, lp_root)
    # A local uncompressed sibling must not double a recording.
    original = sorted(tap_root.glob("TAP/*.csv.gz"))[0]
    with gzip.open(original, "rb") as stream, original.with_suffix("").open("wb") as dest:
        shutil.copyfileobj(stream, dest)
    g.USE_LEAN8 = True
    g.RAW_SOURCE_TAP = tap_root; g.RAW_SOURCE_LP = lp_root
    g.SOURCE_TAP = sample_root / "DATA_TAP_LEAN8"; g.SOURCE_LP = sample_root / "DATA_LONG_PRESS_LEAN8"
    g.LABEL_TAP = ROOT / "research" / "gesture" / "_smoke_labels" / "DATA_TAP"
    g.LABEL_LP = ROOT / "research" / "gesture" / "_smoke_labels" / "DATA_LONG_PRESS"
    g.OUTPUT = ROOT / "research" / "gesture" / "_smoke_results"
    g.TRIALS_PER_MODEL = 4; g.TOP_K_PER_MODEL = 3; g.CV_FOLDS = 2
    g.MODEL_BUDGET_KB = 8; g.ALT_BUDGET_KB = 6           # small budgets keep the smoke fast
    g.REPLAY_LP_TAPS_PER_RECORDING = 2
    fake = Path(tempfile.mkdtemp())
    g.BASELINES = dict(tap=[("missing_example", str(fake / "nope.h"))], lp=[("not_configured_example", "")])
    g.main()
    for task, stem in (("tap", "TAP_SWIPE"), ("lp", "LONG_PRESS")):
        out = g.OUTPUT / task
        report = json.loads((out / "summary.json").read_text())
        assert report["status"] == "success" and report["fixed_threshold"] == g.task_threshold(task)
        assert report["threshold_tuned"] is False and g.task_threshold("tap") == 0.85
        assert report["class_one"] == ("RELEASE/REST" if task == "lp" else "TAP/SWIPE onset")
        assert report["budgets"] == [8192, 6144]
        assert report["lean8"]["enabled"] is True and report["lean8"]["min_frames"] == g.LEAN8_MIN_FRAMES
        gate = report["dataset"]["counts"]["lean8_gate"]
        assert 0 < gate["train_rows_used"] <= gate["train_rows_built"], gate
        assert "Steady state only" in (out / "COMPARISON.md").read_text()
        assert all("holdout_c_header_steady_state" in c for c in report["candidates"])
        roles = [c["role"] for c in report["candidates"]]
        assert roles[0] == "rank_01" and "budget_6kb" in roles, roles
        sel = report["selected"]; assert sel["role"].startswith("rank_") and sel.get("default") is True, sel
        assert all("reason" in s_ for s_ in report["skipped"]), report["skipped"]
        if (sel.get("holdout_c_header") or {}).get("objective_score", 0) == 0:
            assert any("no candidate passed" in s_["reason"] for s_ in report["skipped"]), report["skipped"]
        assert (out / "models" / "budget_6kb" / (stem + ".h")).is_file()
        assert (out / "models" / (stem + ".h")).read_bytes() == (out / sel["header"]).read_bytes()
        for c in report["candidates"]:
            assert c["static_bytes"] <= c["budget_bytes"], c
            hold = c["holdout_c_header"]; assert hold and hold["objective_score"] >= 0
            par = c["c_vs_python_parity"]
            assert par is not None and par.get("decision_mismatch", par.get("q20_mismatch")) < 0.001, par
        h = out / "models" / (stem + ".h"); text = h.read_text()
        assert f"#define {stem}_predict {stem.lower()}_predict" in text
        assert f"#define {stem}_predict_value {stem.lower()}_predict_value" in text
        assert f"#define {stem}_THRESHOLD {g.export_plus._c_float(g.task_threshold(task))}" in text, stem
        assert "V6_RANK" not in text and "v6_rank" not in text
        names = list(json.loads((out / "models" / (stem + "_feature_schema.json")).read_text())["feature_map"].values())
        assert names and all(s.startswith("p_core->") for s in names)
        assert {b["status"] for b in report["baselines"]} <= {"not found", "not configured", "replayed", "not replayable"}
        assert (out / "COMPARISON.md").is_file()
        # A real header replayed as a baseline must give metrics without touching the file.
        copy = fake / task / (stem + ".h"); copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(h, copy)
        g.BASELINES[task] = [("copy_of_new_default", str(copy))]
        data, rpl, _ = g.build_task_data(task, g.selected_sources(task, g.canonical_files(g.RAW_SOURCE_TAP),
                                                                   g.canonical_files(g.RAW_SOURCE_LP)), out)
        tr, te = g.split_heldout(data, task)
        # Trial-invariant fold weights == the original per-trial computation, at several positive weights.
        for pf in g.prepared_folds(data, rpl, tr, task):
            for pw in (0.005, 0.37, 1.0, 6.0, 19.5):
                assert np.array_equal(g.fold_weights(pf, pw).view(np.uint32),
                                      g.weights_for_task(data, pf["tr"], task, pw).view(np.uint32)), (task, pw)
        # Study guard: a resumed study with other data is refused.
        st = g.optuna.create_study(direction="maximize")
        st.set_user_attr("data_content_sha256", "0" * 64)
        try:
            g.guard_study_data(st, data, Path(tempfile.mkdtemp()), None); raise AssertionError("guard did not fire")
        except RuntimeError:
            pass
        b = g.replay_baselines(task, data, rpl, te, out)[0]
        assert b["status"] == "replayed" and b["holdout"]["objective_score"] >= 0, b
        print("UNIFIED_V2_REAL_SOURCE_OK", task, json.dumps(report["dataset"]), report["selected"]["static_bytes"], flush=True)
    combined = g.OUTPUT / "both_models.c"
    combined.write_text('#include "TAP_SWIPE.h"\n#include "LONG_PRESS.h"\n'
                        'float both_models(const t_core *p_core) {\n'
                        '  return TAP_SWIPE_predict_value(p_core) + LONG_PRESS_predict_value(p_core) +\n'
                        '  (float)TAP_SWIPE_predict(p_core) + (float)LONG_PRESS_predict(p_core);\n}\n')
    cc = shutil.which("cc") or shutil.which("gcc")
    cmd = [cc, "-std=c11", "-O2", "-c", str(combined), "-I", str(g.OUTPUT / "lp" / "c_compiler_check"),
           "-I", str(g.OUTPUT / "tap" / "models"), "-I", str(g.OUTPUT / "lp" / "models"), "-o", str(g.OUTPUT / "both_models.o")]
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
    if p.returncode:
        raise RuntimeError("Combined firmware header C compile failed: " + p.stderr[-5000:])
    print("UNIFIED_V2_BOTH_HEADERS_COMPILE_TOGETHER", flush=True)
    man = json.loads((sample_root / "lean8_manifest.json").read_text())
    assert man["firmware_parity"]["status"] in ("passed", "not run"), man["firmware_parity"]
    derived = sorted(g.SOURCE_TAP.glob("TAP/*.csv.gz"))
    assert derived and not any(p.is_symlink() for p in derived)
    print("UNIFIED_V2_REAL_DATA_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    lean8_checks()
    unit_checks()
    if "--unit-only" not in sys.argv:
        real_source_smoke()
