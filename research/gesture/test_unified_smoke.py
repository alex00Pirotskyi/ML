#!/usr/bin/env python3
"""Checks for the unified trainer v2. Never runs a full user Optuna experiment.

1. Unit checks: TAP event replay, LP ten-sample press replay, exact budget trimming,
   C-header replay == Python, class weighting, Lean-8 emulator == a literal port of the
   firmware handler (and == the firmware C code when its header is available locally),
   export/compile/replay of a model that uses Lean-8 fields. Lean-lite: emulator == a literal port ==
   firmware/lean_lite.h compiled for the host (exact), same frames as Lean-8, export/compile/replay.
   Final TAP recipe: Lean-lite diff inputs per recording (nominal, augmented, stressed), press2 labels,
   one-to-one reference matching, content identity / dedupe / stale-reuse refusal.
2. Real-source smoke: a small set of real recordings through Lean-8 generation, labels,
   sampling, replay CV, budget-trimmed C export, held-out C replay and baseline replay; then the same
   in a second process with MUDRA_USE_LEAN8=lite, which must build exactly the same rows and labels.
"""
import os
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "gesture"))
import train_unified as g
import replay as R
import generate_lean8_features as L8
import generate_lean_lite_features as LL
import tap_features as TF


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


def lean_lite_reference(wl_frames, psd_mean_frames):
    """Literal frame-by-frame port of lean_lite_update() (independent of the vectorised emulator)."""
    B, RING, K = LL.BLOCK_FRAMES, LL.RING, LL.EMA_SHIFT
    sat = lambda v: max(-32768, min(32767, v))
    ring_min = [[0] * RING for _ in range(3)]; ring_max = [[0] * RING for _ in range(3)]
    cur_min = [0] * 3; cur_max = [0] * 3; agg_min = [0] * 3; agg_max = [0] * 3
    m_acc = [0] * 3; d_acc = [0] * 3; in_block = fill = head = seen = 0
    out = np.zeros((len(wl_frames), 24), np.int16)
    for f in range(len(wl_frames)):
        first = seen == 0
        for ch in range(3):
            w = int(wl_frames[f][ch]); v = w + (w != 0xFFFFFFFF); n = v.bit_length() - 1
            x = (n << 8) | (((v << (31 - n)) >> 23) & 0xFF)
            if in_block == 0:
                cur_min[ch] = cur_max[ch] = x
            else:
                cur_min[ch] = min(cur_min[ch], x); cur_max[ch] = max(cur_max[ch], x)
            lo, hi = cur_min[ch], cur_max[ch]
            if fill:
                lo = min(lo, agg_min[ch]); hi = max(hi, agg_max[ch])
            if first:
                m_acc[ch] = x << K; d_acc[ch] = 0
            m_acc[ch] += x - (m_acc[ch] >> K)
            xm = x - (m_acc[ch] >> K)
            d_acc[ch] += abs(xm) - (d_acc[ch] >> K)
            pm = psd_mean_frames[f][ch]
            for k, val in enumerate((lo, hi, x - lo, sat(d_acc[ch] >> LL.DEV_SHIFT), sat(int(pm[0]) - int(pm[1])),
                                     hi - lo, hi - x, sat(xm))):
                out[f, k * 3 + ch] = val
        in_block += 1
        if in_block >= B:
            for ch in range(3):
                ring_min[ch][head] = cur_min[ch]; ring_max[ch][head] = cur_max[ch]
            head = (head + 1) % RING; fill = min(fill + 1, RING)
            for ch in range(3):
                agg_min[ch] = min(ring_min[ch][:fill]); agg_max[ch] = max(ring_max[ch][:fill])
            in_block = 0
        seen = min(seen + 1, LL.READY_FRAMES)
    return out


def lean_lite_checks():
    import lightgbm as lgb
    # Vectorised emulator == literal port, on edge cases and on real frames.
    df = pd.read_csv(sorted(g.RAW_SOURCE_TAP.glob("TAP/*.csv.gz"))[0], low_memory=False)
    wl = df[L8.WL].to_numpy(np.float64); rows = L8.frame_rows(df[L8.PSD_NOW].to_numpy(np.int64), wl)
    real = (wl[rows].astype(np.uint64), df[L8.PSD_MEAN].to_numpy(np.int64).reshape(-1, 3, 5)[rows])
    for w, pm in LL.synthetic_cases()[:2] + [real]:
        ref = lean_lite_reference(w, pm)
        assert np.array_equal(LL.lite_per_frame(w, pm), ref[:, :15])
        assert np.array_equal(LL.lite_fields_per_frame(w, pm).reshape(len(w), -1), ref)
    # Emulator == firmware/lean_lite.h compiled for the host, exactly (real recordings + synthetic edge cases).
    files = sorted(g.RAW_SOURCE_TAP.glob("TAP/*.csv.gz"))[:3] + sorted(g.RAW_SOURCE_LP.glob("MID_LONG_PRESS/*.csv.gz"))[:2]
    par = LL.parity_check(files)
    assert par["status"] == "passed" and par["frames_mismatched"] == 0 and par["state_bytes"] == 164, par
    # Same handler calls as Lean-8: frame counter and new-frame rows are identical, so gates and rows are too.
    a, b = L8.compute(df), LL.compute(df)
    assert np.array_equal(a[L8.FRAMES_COLUMN], b[LL.FRAMES_COLUMN]) and np.array_equal(a[L8.NEW_FRAME_COLUMN], b[LL.NEW_FRAME_COLUMN])
    # A model on the 126 raw + 15 Lean-lite fields exports, compiles against t_core and replays exactly.
    rng = np.random.default_rng(9); cols = list(g.RAW_126) + list(g.LITE_15)
    X = rng.normal(size=(20000, len(cols))).astype(np.float32) * 300
    i1 = cols.index("p_core->cload_lite.rise10[CH2]"); i2 = cols.index("p_core->cload_lite.d01[CH1]")
    y = (X[:, i1] + X[:, i2] + 0.3 * X[:, 0] + rng.normal(size=20000) * 300 > 100).astype(int)
    m = lgb.LGBMClassifier(n_estimators=60, num_leaves=15, verbosity=-1, n_jobs=2, max_bin=127).fit(X, y)
    out = Path(tempfile.mkdtemp())
    h, _, chk = g.write_header(m, cols, "TAP_SWIPE", out / "lite_models", "tap", out)
    assert "cload_lite.rise10[CH2]" in h.read_text() and chk["status"] == "passed", chk
    hm = R.HeaderModel(h, cols, out / "build_lite", prefer="tap_swipe")
    p = m.predict_proba(X[:4000])[:, 1]
    assert (hm.value_q20(X[:4000]) == R.q20(p)).all()
    print("LEAN_LITE_CHECKS_PASSED", par["frames"], "frames exact", flush=True)


def tap_generalization_checks():
    """Derived TAP inputs and the placement transform: identity at zero change, and a model on raw + diff/ratio
    fields exported, compiled and replayed through the C helpers equals Python (including int16 saturation)."""
    import lightgbm as lgb
    raw = list(g.RAW_126)
    df = pd.read_csv(sorted(g.RAW_SOURCE_TAP.glob("TAP/*.csv.gz"))[0], usecols=lambda c: c in set(raw), low_memory=False)
    Xr = df[raw].to_numpy(np.float32)[::7]
    cols = raw + list(TF.DERIVED)
    X = np.hstack([Xr, TF.derived(Xr, raw)])
    n = len(X)
    same = TF.apply_placement(X, cols, np.ones((n, 3)), np.ones((n, 3, 5)), np.zeros((n, 3)))
    assert np.array_equal(same, X), "placement transform is not the identity at zero change"
    d = TF.stress_draws(1)
    moved = TF.apply_placement(X, cols, d["gain"][[0] * n], d["tilt"][[0] * n], d["noise"][[0] * n])
    acc = [cols.index(c) for c in raw if c.startswith("p_core->acc")]
    assert np.array_equal(moved[:, acc], X[:, acc]) and not np.array_equal(moved[:, :126], X[:, :126])
    assert np.array_equal(moved[:, 126:], TF.derived(moved[:, :126], raw)), "derived fields not recomputed"
    # Synthetic integer firmware values over the full uint16 range, so diff() saturates in int16.
    rng = np.random.default_rng(4)
    R_ = rng.integers(0, 65536, size=(20000, 126)).astype(np.float32)
    R_[:, [raw.index(c) for c in raw if "wl" in c]] = rng.integers(0, 20000, size=(20000, 9))
    F = np.hstack([R_, TF.derived(R_, raw)])
    i_d = cols.index(TF.DIFF_NAMES[0]); i_r = cols.index(TF.RATIO_NAMES[4])
    y = ((F[:, i_d] > 0).astype(int) ^ (F[:, i_r] > 1.0).astype(int))
    subset = list(TF.DIFF_NAMES[:30]) + list(TF.RATIO_NAMES) + [c for c in raw if c.startswith("p_core->acc")][:10]   # helpers, not their args
    ix = [cols.index(c) for c in subset]
    m = lgb.LGBMClassifier(n_estimators=40, num_leaves=15, verbosity=-1, n_jobs=2, max_bin=255).fit(F[:, ix], y)
    out = Path(tempfile.mkdtemp())
    h, _, chk = g.write_header(m, subset, "TAP_SWIPE", out / "diff_models", "tap", out)
    text = h.read_text()
    assert "diff(" in text and "ratio(" in text and chk["status"] == "passed", chk
    hm_cols = R.input_columns(subset)
    assert len(hm_cols) > len(subset)
    hm = R.HeaderModel(h, hm_cols, out / "build_diff", prefer="tap_swipe")
    Xh = F[:, [cols.index(c) for c in hm_cols]]
    p_ = m.predict_proba(F[:, ix])[:, 1]
    assert (hm.value_q20(Xh[:5000]) == R.q20(p_[:5000])).all(), "C helper replay differs from Python"
    try:
        R.HeaderModel(h, subset, out / "build_bad", prefer="tap_swipe"); raise AssertionError("missing args accepted")
    except ValueError:
        pass
    print("TAP_GENERALIZATION_CHECKS_PASSED", flush=True)


def tap_final_recipe_checks():
    """Final TAP recipe pieces (TAP_FOCUS_RESULTS.md, "Final recipe")."""
    import lightgbm as lgb
    # ---- Lean-lite diff inputs: trainer path (derived tree frame counter) == emulator from the raw recording.
    src = sorted(g.RAW_SOURCE_TAP.glob("TAP/*.csv.gz"))[0]
    df = pd.read_csv(src, low_memory=False)
    raw = list(g.RAW_126)
    X = df[raw].to_numpy(np.float32)
    rows = L8.frame_rows(df[L8.PSD_NOW].to_numpy(np.int64), df[L8.WL].to_numpy(np.float64))
    frames = np.cumsum(np.isin(np.arange(len(df)), rows))
    assert np.array_equal(LL.frame_rows_from_counter(frames), rows)
    got = TF.lite_rows(X, raw, frames)
    per = LL.lite_fields_per_frame(df[L8.WL].to_numpy(np.float64)[rows].astype(np.uint64), None)
    want = np.zeros_like(got); want[frames > 0] = per[:, [LL.FIELDS.index(f) for f in LL.DIFF_FEATURES]].reshape(len(per), -1)[frames[frames > 0] - 1]
    assert np.array_equal(got, want)
    same = TF.lite_rows_placement(X, raw, frames, np.ones(3), np.ones((3, 5)), np.zeros(3))
    assert np.array_equal(same, got), "Lean-lite placement is not the identity at zero change"
    moved = TF.lite_rows_placement(X, raw, frames, np.full(3, 2.0), np.ones((3, 5)), np.zeros(3))
    ri = [TF.LITE_COLUMNS.index(f"p_core->cload_lite.range10[CH{c}]") for c in (1, 2, 3)]
    assert np.abs(moved[frames > 46][:, ri] - got[frames > 46][:, ri]).max() <= 64, "range10 is not ~gain invariant"
    # Stress view: the lite columns of a slice are the recording's whole-history recomputation.
    cols = raw + list(TF.LITE_COLUMNS)
    Xa = np.hstack([X, got]).astype(np.float32)
    recs = pd.DataFrame(dict(start=[0], end=[len(Xa)]))
    d = TF.stress_draws(1)
    view = g._StressView(Xa, list(range(len(cols))), cols, np.zeros(len(Xa), np.int32), d, recs=recs, frames=frames)
    a_, b_ = len(Xa) // 2, len(Xa) - 3
    full = TF.lite_rows_placement(Xa, cols, frames, d["gain"][0], d["tilt"][0], d["noise"][0])
    assert np.array_equal(view[a_:b_][:, len(raw):], full[a_:b_])
    # A model on raw + Lean-lite diff fields exports, compiles against t_core (range10/xm) and replays exactly.
    rng = np.random.default_rng(11)
    Xr = rng.normal(size=(20000, len(cols))).astype(np.float32) * 300
    i1 = cols.index("p_core->cload_lite.xm[CH2]"); i2 = cols.index("p_core->cload_lite.range10[CH1]")
    y = (Xr[:, i1] - Xr[:, i2] + 0.3 * Xr[:, 0] + rng.normal(size=20000) * 300 > 50).astype(int)
    m = lgb.LGBMClassifier(n_estimators=40, num_leaves=7, verbosity=-1, n_jobs=2, max_bin=127).fit(Xr, y)
    out = Path(tempfile.mkdtemp())
    h, _, chk = g.write_header(m, cols, "TAP_SWIPE", out / "litediff_models", "tap", out)
    text = h.read_text()
    assert "cload_lite.xm[CH2]" in text and "cload_lite.range10[CH1]" in text and chk["status"] == "passed", chk
    hm = R.HeaderModel(h, cols, out / "build_litediff", prefer="tap_swipe")
    assert (hm.value_q20(Xr[:4000]) == R.q20(m.predict_proba(Xr[:4000])[:, 1])).all()
    # ---- press2 labels on a synthetic recording.
    t = np.arange(0, 3000, 4.0); y = np.zeros(len(t), np.int8)
    y[(t >= 500) & (t <= 580)] = 1                           # confirmed press at 500
    y[(t >= 980) & (t <= 1100)] = -1                         # unknown press with a teacher peak (score 0.8) at 1000
    y[(t >= 1990) & (t <= 2050)] = -1                        # unknown weak press (score 0.2) at 2000
    ev = pd.DataFrame(dict(type="tap_press", tier="PRESS", labeled=[1, -1, -1, -1], t_on=[500.0, 1000.0, 2000.0, 530.0],
                           t_ptr=[510.0, 1010.0, 2010.0, 540.0], score=[0.99, 0.8, 0.2, 0.1], rule_disconnect=False))
    y2, added, info = g.press2_labels(t, y, y, ev)
    at = lambda a, b: (t >= a) & (t <= b)
    assert (y2[at(500, 580)] == 1).all() and (y2[at(470, 499)] == -1).all() and (y2[at(581, 620)] == -1).all()
    assert (y2[at(1000, 1080)] == 1).all() and (y2[at(1081, 1100)] == -1).all() and (y2[at(970, 979)] == -1).all()
    assert (y2[at(1995, 2090)] == 1).all() and (y2[at(2091, 2120)] == -1).all() and (y2[at(1970, 1994)] == -1).all()
    assert (y2[at(1300, 1800)] == 0).all() and (y2[at(2200, 2900)] == 0).all()
    assert info["press2_anchored_presses"] == 1 and info["press2_soft_presses"] == 1     # 530: next to the confirmed one
    assert not added[at(500, 580)].any() and added[at(1995, 2090)].all() and np.array_equal(y, np.asarray(y))
    assert g.press2_recording(Path("DATA_TAP/TAP/A_RightHand_DoubleTaps_Horizontal.csv.gz"))
    assert g.press2_recording(Path("DATA_TAP/TAP/B_MidTap_Horizontal_Default_240bpm.csv.gz"))
    assert not g.press2_recording(Path("DATA_TAP/TAP/C_Tap_Horizontal_Default_60bpm.csv.gz"))
    assert not g.press2_recording(Path("DATA_TAP/SWIPE/D_DoubleTaps.csv.gz"))
    # ---- observable-pattern labels: per tap the best-scoring window of dur ms; other rows of the tap left out.
    tap = np.array([0] * 8 + [1] * 3 + [-1, 2]); off = np.r_[np.arange(-5, 75, 10.0), [0.0, 10, 20], [50.0], [3.0]]
    sc = np.r_[[0.1, 0.2, 0.9, 0.95, 0.9, 0.3, 0.2, 0.1], [0.5, 0.4, 0.6], [0.0], [0.2]]
    k = g.obs_select(tap, off, sc, 40.0, 10)
    assert k[:8].tolist() == [False, False, True, True, True, True, False, False], k     # rows 15..45 ms of tap 0
    assert k[8:11].all() and k[11] and k[12]          # short tap: window of all its rows; unowned / single rows kept
    k20 = g.obs_select(tap, off, sc, 20.0, 10)                 # tie 15+25 vs 25+35: the earlier window
    assert k20[:8].tolist() == [False, False, True, True, False, False, False, False], k20
    assert k20[8:11].tolist() == [False, True, True], k20      # 3-row tap, 20 ms: the best 2-row window
    # ---- MUDRA_SPLIT=file: one group per recording, in the rows and in the replay table, persons kept.
    recs_ = pd.DataFrame(dict(group=["a", "a", "b"], file=["f1", "f2", "f3"], start=[0, 5, 9], end=[5, 9, 12]))
    d_ = dict(group=np.array(["a", "a", "a", "b"], object), rec=np.array([0, 1, 1, 2]), content_sha256="x")
    g.group_by_file(d_, dict(recs=recs_))
    assert d_["group"].tolist() == ["a|f1", "a|f2", "a|f2", "b|f3"] and d_["person"].tolist() == ["a", "a", "a", "b"]
    assert recs_.group.tolist() == ["a|f1", "a|f2", "b|f3"] and recs_.person.tolist() == ["a", "a", "b"]
    assert d_["content_sha256"] != "x" and "stratum" not in d_
    # ---- gesture-aware file split: recording type from the file name; rare types pooled; stratified parts.
    gs = {"perry_SwipeMID_Horizontal_Up_Down__Up_Down_120bpm_2026-01-04_10-03-18.csv.gz": ("SWIPE", "SWIPE|swipemid|120"),
          "alex_new_LeftHand_Ring_Finger_Horizontal_Default_60bpm_2026-11-24_14-26-46-023.csv.gz": ("FINGER", "FINGER|ringfinger"),
          "x_LeftHand_Fist_&_Stretch_Vertical_Default_60bpm_2026-11-24_14-24-03-427.csv.gz": ("FINGER", "FINGER|fiststretch"),
          "Shira_TapOnArm_Horizontal_Default_60bpm_2026-01-29_11-59-49.csv.gz": ("ARM", "ARM|taponarm"),
          "AdiEdelshtein_SoftTap_Reverse_NONE_240bpm_2025-12-31_16-46-30.csv.gz": ("TAP", "TAP|softtap|240")}
    for name_, (fo_, want) in gs.items():
        assert g.recording_stratum("DATA/DATA_TAP/" + fo_ + "/" + name_, fo_) == want, (name_, want)
    m_ = g.merge_rare(["T|a|60"] * 6 + ["T|a|120"] * 3 + ["T|b|60"] * 2 + ["F|c"] * 7 + ["W|d"] * 5, 5)
    assert m_.tolist() == ["T|a|60"] * 6 + ["T"] * 5 + ["F|c"] * 7 + ["W|d"] * 5, m_.tolist()   # rare pooled per folder
    m_ = g.merge_rare(["T|a|60"] * 6 + ["T|a|120"] * 2 + ["T|b|60"] * 2 + ["W|d"], 5)
    assert m_.tolist() == ["T|a|60"] * 6 + ["other"] * 5, m_.tolist()                          # then across folders
    assert g.merge_rare(["A|x"] * 6 + ["B|y"], 5).tolist() == ["A|x"] * 7                       # a single one: largest
    n_f = 60; fo_ = np.array(["TAP"] * 20 + ["SWIPE"] * 10 + ["FINGER"] * 20 + ["WRIST"] * 10, object)
    kind_ = ["SoftTap"] * 10 + ["MidTap"] * 10 + ["SwipeMID"] * 10 + ["Pinky"] * 20 + ["Twist"] * 10
    nm_ = np.array([f"p{i % 7}_{kind_[i]}_Horizontal_Default_60bpm_{i}.csv.gz" for i in range(n_f)], object)
    recs_ = pd.DataFrame(dict(group=[f"p{i % 7}" for i in range(n_f)], file=nm_, folder=fo_))
    rec_ = np.repeat(np.arange(n_f), 2)
    d_ = dict(group=recs_.group.to_numpy(object)[rec_], rec=rec_, category=fo_[rec_], X=np.zeros((2 * n_f, 1)),
              y=np.where(np.isin(fo_[rec_], ["TAP", "SWIPE"]) & (np.arange(2 * n_f) % 2 == 0), 1, 0),
              content_sha256="x")
    g.group_by_file(d_, dict(recs=recs_))
    tr_, te_ = g.split_heldout(d_, "tap")
    bal_ = g.strata_balance(d_, tr_, te_)
    assert set(bal_) == {"TAP|softtap|60", "TAP|midtap|60", "SWIPE|swipemid|60", "FINGER|pinky", "WRIST|twist"}, bal_
    assert all(abs(v[1] / sum(v) - g.TEST_FRACTION) <= 0.1 for v in bal_.values()), bal_
    assert not set(d_["group"][tr_]) & set(d_["group"][te_])              # a file never on both sides
    assert set(d_["person"][tr_]) & set(d_["person"][te_])                # people are
    cv_ = g.cv_splits(d_, tr_)
    assert len(cv_) == g.CV_FOLDS
    seen_ = np.concatenate([np.unique(d_["group"][tr_[b]]) for _, b in cv_])
    assert len(seen_) == len(set(seen_)) == len(set(d_["group"][tr_]))     # every training file in one fold
    for _, b in cv_:                                                      # each fold gets its share of each type
        assert {st.split("|")[1] for st in d_["stratum"][tr_[b]]} == {"softtap", "midtap", "swipemid", "pinky", "twist"}
    # ---- distillation as binary rows: the pair's weighted log loss equals the cross-entropy with the soft target.
    rng_ = np.random.default_rng(3); t_ = np.r_[rng_.random(50), 0.0, 1.0]; w_ = rng_.random(52) + 0.5
    X_ = np.arange(52, dtype=np.float32)[:, None]; p_ = np.clip(rng_.random(52), 0.01, 0.99)
    X2_, y2_, w2_ = g.soft_rows(X_, t_, w_)
    assert len(y2_) == 2 * 50 + 2 and set(y2_.tolist()) == {0, 1}            # t = 0 and t = 1 give one row each
    p2_ = p_[X2_[:, 0].astype(int)]
    ll_pairs = -np.sum(w2_ * np.where(y2_ == 1, np.log(p2_), np.log(1 - p2_)))
    ll_soft = -np.sum(w_ * (t_ * np.log(p_) + (1 - t_) * np.log(1 - p_)))
    assert abs(ll_pairs - ll_soft) < 1e-3 * abs(ll_soft), (ll_pairs, ll_soft)
    # ---- finger-negative weight: FINGER negatives x k before balancing; teachers (finger=False) unchanged.
    d_ = dict(y=np.array([1, 1, 0, 0, 0, 0]), w=np.ones(6, np.float32), file=np.array(list("aabbcc"), object),
              category=np.array(["TAP", "TAP", "FINGER", "FINGER", "WRIST", "WRIST"], object))
    old_fw = g.TAP_FINGER_WEIGHT
    try:
        g.TAP_FINGER_WEIGHT = 2.0
        w2_ = g.weights_for_task(d_, np.arange(6), "tap", 1.0); w1_ = g.weights_for_task(d_, np.arange(6), "tap", 1.0, finger=False)
    finally:
        g.TAP_FINGER_WEIGHT = old_fw
    assert np.isclose(w2_[2:4].sum() / w2_[4:6].sum(), 2.0) and np.isclose(w1_[2:4].sum() / w1_[4:6].sum(), 1.0)
    assert np.isclose(w2_[d_["y"] == 0].sum(), w1_[d_["y"] == 0].sum())        # class balance unchanged
    # ---- TAP logit shift: exact on predict, truncation and the exporter's tree dump; 0.85 then means sigmoid(logit 0.85 + s).
    rng_ = np.random.default_rng(5); X_ = rng_.normal(size=(600, 4)); y_ = (X_[:, 0] + rng_.normal(size=600) > 0).astype(int)
    p_ = dict(n_estimators=12, num_leaves=7, verbose=-1, deterministic=True)
    m0_ = g.fit_model(X_, y_, np.ones(600), p_, 10 ** 6); m1_ = g.fit_model(X_, y_, np.ones(600), p_, 10 ** 6, shift=2.5)
    r0_ = m0_.booster_.predict(X_, raw_score=True); r1_ = m1_.booster_.predict(X_, raw_score=True)
    assert np.allclose(r0_ - 2.5, r1_, atol=1e-9)
    assert np.allclose(g.truncated(m1_, 5).booster_.predict(X_, raw_score=True),
                       m0_.booster_.predict(X_, raw_score=True, num_iteration=5) - 2.5, atol=1e-9)
    dump_ = m1_.booster_.dump_model(); walk_ = g.export_plus._walk_lgbm_dump_tree
    assert np.allclose([sum(walk_(t["tree_structure"], x) for t in dump_["tree_info"]) for x in X_[:20]], r1_[:20], atol=1e-5)
    sig_ = lambda z: 1 / (1 + np.exp(-z))
    assert ((sig_(r1_) >= 0.85) == (sig_(r0_) >= sig_(np.log(0.85 / 0.15) + 2.5))).all()
    # ---- balanced TAP objective: recording kinds, per-kind counts, worst movement; size-aware ladder choice.
    kinds_ = {("DATA/x/TAP/a_SoftTap_Horizontal_60bpm_2026.csv.gz", "TAP"): "tap:soft",
              ("DATA/x/TAP/a_DoubleTaps_Horizontal_60bpm_2026.csv.gz", "TAP"): "tap:double",
              ("DATA/x/TAP/a_MidTap_Horizontal_240bpm_2026.csv.gz", "TAP"): "tap:fast",
              ("DATA/x/TAP/a_StrongTap_Horizontal_120bpm_2026.csv.gz", "TAP"): "tap:strong",
              ("DATA/x/SWIPE/a_SwipeMID_Horizontal_60bpm_2026.csv.gz", "SWIPE"): "tap:swipe",
              ("DATA/x/FINGER/a_Fist&Stretch_NONE_60bpm_2026.csv.gz", "FINGER"): "move:fist",
              ("DATA/x/FINGER/a_Pinky_NONE_60bpm_2026.csv.gz", "FINGER"): "move:finger",
              ("DATA/x/WRIST/a_Twist_NONE_60bpm_2026.csv.gz", "WRIST"): "move:wrist",
              ("DATA/x/NOTHING/a_Nothing_NONE_0bpm_2026.csv.gz", "NOTHING"): None}
    for (f_, fo_), k_ in kinds_.items():
        assert g.tap_kind(f_, fo_) == k_, (f_, g.tap_kind(f_, fo_), k_)
    obj0, lim0 = R.TAP_OBJECTIVE, R.TAP_MOVEMENT_LIMIT
    try:
        R.TAP_OBJECTIVE, R.TAP_MOVEMENT_LIMIT = "balanced", 4.0
        tt_ = np.arange(0, 120000, 5.0); c_ = R.TapCounts()
        for kind_, n_hit in (("tap:soft", 20), ("tap:double", 30)):        # 40 onsets each: recall 0.5 and 0.75
            ons_ = np.arange(2000, 42000, 1000.0); d_ = np.zeros(len(tt_), bool)
            for o_ in ons_[:n_hit]:
                d_[int(o_ / 5) + 1] = True
            R.tap_replay_recording(d_, tt_, np.full(len(tt_), -1), ons_, "TAP", c_, kind=kind_)
        for kind_, every in (("move:fist", 7500), ("move:finger", 30000)):   # 8 and 2 false taps per minute
            d_ = np.zeros(len(tt_), bool); d_[np.arange(10, len(tt_), every // 5)] = True
            R.tap_replay_recording(d_, tt_, np.zeros(len(tt_), int), [], "FINGER", c_, kind=kind_)
        b_ = c_.report()["balanced"]
        assert b_["worst_movement"] == "fist" and abs(b_["worst_false_per_min"] - 8.0) < 1e-6, b_
        assert abs(b_["mean_recall"] - 0.625) < 1e-9 and abs(b_["score"] - 0.625 * 0.5) < 1e-9, b_
        assert c_.report()["objective_score"] == b_["score"]
        R.TAP_OBJECTIVE = "overall"
        assert c_.report()["objective_score"] != b_["score"]
    finally:
        R.TAP_OBJECTIVE, R.TAP_MOVEMENT_LIMIT = obj0, lim0
    pen0 = g.SIZE_PENALTY_PER_KB
    try:
        g.SIZE_PENALTY_PER_KB = 0.0001
        attrs_ = {"score_102400": 0.910, "max_bytes_102400": 102400, "score_65536": 0.907, "max_bytes_65536": 65536,
                  "score_32768": 0.890, "max_bytes_32768": 32768}
        b_, v_ = g.ladder_choice(attrs_, [102400, 65536, 32768])          # 64 KB: 0.907 - 0.0064 > 0.910 - 0.01
        assert b_ == 65536 and abs(v_ - (0.907 - 0.0064)) < 1e-9, (b_, v_)
    finally:
        g.SIZE_PENALTY_PER_KB = pen0
    # ---- one-to-one reference matching: one event cannot find two windows; crossing windows; boundaries.
    assert R.match_windows([1150.0], [850.0, 1050.0], [1250.0, 1450.0]).tolist() == [True, False]
    assert R.match_windows([5.0, 12.0], [0.0, 10.0], [20.0, 13.0]).tolist() == [True, True]
    assert R.match_windows([10.0], [10.0], [10.0]).tolist() == [True]
    assert R.match_windows([], [0.0], [1.0]).tolist() == [False] and len(R.match_windows([1.0], [], [])) == 0
    c = R.TapCounts(); tt = np.arange(0, 2000, 5.0); dd = (tt >= 1150) & (tt <= 1170)
    R.tap_replay_recording(dd, tt, np.zeros(len(tt), np.int8), [], "TAP", c,
                           extra={"press": (np.array([850.0, 1050.0]), np.array([1250.0, 1450.0]))})
    assert c.extra["press"] == [2, 1], c.extra
    # ---- content identity: gzip mtime ignored, content changes seen, duplicates dropped, stale reuse named.
    tmp = Path(tempfile.mkdtemp(dir=g.ROOT / "research" / "gesture"))
    try:
        a = tmp / "a.csv.gz"; b = tmp / "b.csv.gz"
        with gzip.GzipFile(a, "wb", mtime=1) as f_:
            f_.write(b"x,y\n1,2\n")
        with gzip.GzipFile(b, "wb", mtime=999) as f_:
            f_.write(b"x,y\n1,2\n")
        assert a.read_bytes() != b.read_bytes() and g.content_sha(a) == g.content_sha(b)
        lab = tmp / "labels"; (lab / tmp.name).mkdir(parents=True)
        for p_ in (a, b):
            shutil.copyfile(p_, g.sidecar(p_, tmp.parent, lab))
        (lab / "_events.csv.gz").write_bytes(gzip.compress(b"recording\n"))
        recs_ = [(a, tmp.parent, lab), (b, tmp.parent, lab)]
        kept, dup = g.dedupe_records(recs_)
        assert len(kept) == 1 and len(dup) == 1
        old = g.record_identity(kept)
        with gzip.GzipFile(a, "wb", mtime=1) as f_:
            f_.write(b"x,y\n1,3\n")                       # same size, other content
        new = g.record_identity(kept)
        msg = g.identity_changes(old, new)
        assert new["sha256"] != old["sha256"] and any("changed content" in m_ for m_ in msg), msg
        assert g.identity_changes(new, new) == [] and g.identity_changes(None, new)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("TAP_FINAL_RECIPE_CHECKS_PASSED", flush=True)


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
    # Selection guard uses CV only: a CV winner with a failed fold is skipped; the held-out score is ignored.
    tmp = Path(tempfile.mkdtemp()); cands = []
    for role, folds, held in (("rank_01", [0.5, 0.0], 0.9), ("rank_02", [0.4, 0.3], 0.0), ("budget_6kb", [0.3, 0.3], 0.1)):
        d_ = tmp / "models" / "candidates" / role; d_.mkdir(parents=True)
        (d_ / "X.h").write_text(role); (d_ / "X_feature_schema.json").write_text("{}")
        cands.append(dict(role=role, trial=len(cands), header=str((d_ / "X.h").relative_to(tmp)),
                          cv=dict(fold_scores=folds), holdout_c_header=dict(objective_score=held)))
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
    fs_unit_checks()
    print("UNIFIED_V2_UNIT_CHECKS_PASSED", flush=True)


def fs_unit_checks():
    import feature_selection as FS
    d = 10
    assert abs(FS.nogueira_stability([[0, 1, 2], [0, 1, 2], [0, 1, 2]], d) - 1.0) < 1e-12
    assert abs(FS.nogueira_stability([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], d) + 1.0) < 1e-12
    rng = np.random.default_rng(11); n = 4000
    a, b, c = rng.normal(size=(3, n))
    X = np.stack([a, a + 0.01 * rng.normal(size=n), a * 3 + 1, b, np.exp(b), c, rng.normal(size=n), rng.normal(size=n)], 1)
    cl = FS.correlation_clusters(X, seed=0)["clusters"]
    assert [0, 1, 2] in cl and [3, 4] in cl and [5] in cl, cl
    imp = np.array([0.0, 0.3, 0.1, 0.2, -0.1, 0.05, 0.0, -0.2]); gimp = np.array([0.5, 0.25, 0.05, 0.0, -0.1])
    clusters = [[0, 1, 2], [3, 4], [5], [6], [7]]
    order = FS.group_aware_order(imp, gimp, np.zeros(5), clusters)
    assert sorted(order) == list(range(8)) and order[:3] == [1, 3, 5], order      # best member of each strong group
    r = dict(order=order)
    assert FS.select_top_k(r, 3) == [1, 3, 5] and FS.select_top_k(r, 99) == list(range(8))
    try:
        FS.validate_fold_local(dict(outer_people=["a", "b"]), {"b"}); raise AssertionError("leak not detected")
    except AssertionError as e:
        assert "validation people" in str(e)
    print("FS_UNIT_CHECKS_PASSED", flush=True)


def cache_hash_compat_check(cache: Path):
    """With soft labels off, the cache content hash equals the hash of the earlier cache format (two-column
    tap_onsets.csv with confirmed onsets only, no sample weights): studies started before stay resumable."""
    old = Path(tempfile.mkdtemp())
    for f in cache.iterdir():
        if f.is_file():
            shutil.copyfile(f, old / f.name)
    o = pd.read_csv(cache / "tap_onsets.csv")
    o[o.kind == "strong"][["rec", "t_on"]].to_csv(old / "tap_onsets.csv", index=False)
    with np.load(cache / "train_meta.npz") as z:
        assert "w" in z.files and (z["w"] == 1).all()
        np.savez(old / "train_meta.npz", **{k: z[k] for k in z.files if k != "w"})
    assert g.cache_content_sha(old) == g.cache_content_sha(cache)
    # ... while a real change of the objective onsets is detected.
    o[o.kind == "strong"][["rec", "t_on"]].iloc[1:].to_csv(old / "tap_onsets.csv", index=False)
    assert g.cache_content_sha(old) != g.cache_content_sha(cache)
    shutil.rmtree(old)
    print("CACHE_HASH_COMPAT_OK", flush=True)


def validate_events_checks():
    """validate_events.py on synthetic logs: a device that follows the replay rules agrees; one with another
    refractory is detected and its refractory recovered; gold timing and the LP release are reproduced."""
    import validate_events as VE
    t = np.arange(0.0, 60000.0, 21.5); n = len(t)
    d = np.zeros(n, bool)
    onsets = []
    for k, c in enumerate(np.arange(1000.0, 59000.0, 1100.0)):
        i = int(np.searchsorted(t, c)); d[i:i + 3] = True; onsets.append(t[i])
        if k % 3 == 0:                                     # a second rising edge ~200 ms later
            j = int(np.searchsorted(t, c + 200.0)); d[j:j + 2] = True
    held = np.zeros(n, bool); q = np.zeros(n)
    releases = []
    for c in np.arange(2000.0, 58000.0, 4000.0):           # LP: press held 2 s, release probability 1 after it
        a, b = int(np.searchsorted(t, c)), int(np.searchsorted(t, c + 2000.0))
        q[b:b + 40] = 1.0
        rel = R.lp_declared_release(np.round(q[a:] * 20), t[a:], t[a], 0.5); releases.append(rel)
        held[a:int(np.searchsorted(t, rel))] = True

    class Fake:
        def decision(self, X):
            return X[:, 0] > 0.5

        def value_q20(self, X):
            return np.round(X[:, 1] * 20).astype(np.int64)

    tmp = Path(tempfile.mkdtemp())

    def log(name, device):
        f = tmp / name
        pd.DataFrame(dict(Timestamp=(1_700_000_000_000 + t).astype(np.int64), f0=d.astype(np.float32),
                          f1=q.astype(np.float32), DEV=device.astype(int))).to_csv(f, index=False)
        return f

    def device_taps(refractory):
        x = np.zeros(n, bool)
        for e in VE.tap_events(d, t, refractory):
            x[int(np.searchsorted(t, e))] = True
        return x

    same = VE.firmware_tap(Fake(), ["f0", "f1"], [log("same.csv", device_taps(R.TAP_REFRACTORY_MS))], "DEV", 30.0)
    assert same["agreement_f1"] == 1.0 and same["verdict"].startswith("replay rules agree"), same
    other = VE.firmware_tap(Fake(), ["f0", "f1"], [log("other.csv", device_taps(300))], "DEV", 30.0)
    assert other["agreement_f1"] < 0.98 and "disagree" in other["verdict"], other
    assert other["best_fit_f1"] == 1.0 and other["best_fit_refractory_ms"] > 200, other
    gold = pd.DataFrame(dict(recording="same", t_ms=np.asarray(onsets) + 20.0))
    gr = VE.gold_tap(Fake(), ["f0", "f1"], [tmp / "same.csv"], gold)
    doubles = len(range(0, len(onsets), 3))               # the second edges are 200 ms > refractory: false events
    assert gr["recall"] == 1.0 and gr["false_events"] == doubles and abs(gr["latency_ms_median"] + 20.0) < 1e-6, gr
    lp = VE.firmware_lp(Fake(), ["f0", "f1"], [log("lp.csv", held)], "DEV", 0.5, 30.0)
    assert lp["presses"] == len(releases) and lp["release_within_tol"] == 1.0, lp
    shutil.rmtree(tmp)
    print("VALIDATE_EVENTS_CHECKS_PASSED", flush=True)


def fold_parallel_check():
    """Concurrent CV folds give the same trees and the same objective as one fold at a time."""
    import hashlib
    import optuna
    ft = g.canonical_files(g.RAW_SOURCE_TAP); fl = g.canonical_files(g.RAW_SOURCE_LP)
    data, rpl, _ = g.build_task_data("tap", g.selected_sources("tap", ft, fl), g.OUTPUT / "tap")
    g.group_by_file(data, rpl)                             # as the trainer: file groups only
    tr, _ = g.split_heldout(data, "tap")
    budgets = [g.budget_bytes(g.MODEL_BUDGET_KB)]
    params = dict(n_features=len(data["columns"]) // 2, positive_weight=1.0, max_depth=4, num_leaves=15,
                  n_estimators=40, learning_rate=0.1, min_child_samples=50, reg_lambda=1.0, reg_alpha=1e-3,
                  colsample_bytree=0.8, subsample=0.8)
    if g.TAP_LOGIT_SHIFT:                                  # strict target / balanced objective: the shift is searched
        params["tap_logit_shift"] = 1.0
    fit, seen = g.fit_model, []

    def recording_fit(X, y, w, p, b, **kw):           # kw: distillation targets (MUDRA_TAP_DISTILL)
        m = fit(X, y, w, p, b, **kw)
        seen.append(hashlib.sha256(m.booster_.model_to_string().split("\nparameters:")[0].encode()).hexdigest())
        return m
    g.fit_model = recording_fit
    results = []
    try:
        for workers in (1, 2):
            g.FOLD_WORKERS, g.FOLD_THREADS = workers, 1
            seen.clear()
            t = optuna.trial.FixedTrial(params)
            results.append((g.objective(t, data, rpl, tr, "tap", budgets), sorted(seen),
                            [f["budgets"][str(budgets[0])]["objective_score"] for f in t.user_attrs["folds"]]))
    finally:
        g.fit_model = fit
    if g.TAP_DISTILL is None:
        assert results[0] == results[1], results
    else:
        # Distillation trains on two rows per frame (label 1 and label 0, same inputs): many exactly tied splits,
        # which LightGBM may break differently with another thread count (CI: 3-4 threads vs 1). The trees can then
        # differ in text while scoring identically; the scores must still match exactly.
        assert results[0][0] == results[1][0] and results[0][2] == results[1][2], results
        if results[0][1] != results[1][1]:
            print("FOLD_PARALLEL_TIED_SPLITS distillation: same scores, trees differ in tied splits", flush=True)
    print("FOLD_PARALLEL_EQUIVALENT", results[0][0], flush=True)


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


SMOKE_RESULTS = dict(full="_smoke_results", off="_smoke_results_no_lean8", lite="_smoke_results_lite")


def real_source_smoke(keep_input=False):
    """Runs in the trainer's own mode (MUDRA_USE_LEAN8). keep_input: reuse the sample and labels of a previous run."""
    sample_root = ROOT / "research" / "gesture" / "_smoke_input"
    results = ROOT / "research" / "gesture" / (SMOKE_RESULTS[g.LEAN8_MODE] + g.TAP_TAG + g.SPLIT_TAG)
    for d in ((results,) if keep_input else (sample_root, ROOT / "research" / "gesture" / "_smoke_labels", results)):
        if d.exists():
            shutil.rmtree(d)
    tap_root = sample_root / "DATA_TAP"; lp_root = sample_root / "DATA_LONG_PRESS"
    for folder, n in (() if keep_input else (("TAP", 22), ("SWIPE", 22), ("FINGER", 18), ("ARM", 16), ("WRIST", 14), ("BAND", 10))):
        for p in select_originals(g.RAW_SOURCE_TAP, folder, n):
            put_sample(p, tap_root)
    for folder, n in (() if keep_input else (("SOFT_LONG_PRESS", 35), ("MID_LONG_PRESS", 35), ("STRONG_LONG_PRESS", 20),
                      ("DRAG_AND_DROP", 35), ("LONG_PRESS", 4), ("CONNECTED", 16), ("DISCONNECTED", 16),
                      ("LOOSE", 16), ("TABLE", 9), ("TAP", 12), ("SWIPE", 12))):
        for p in select_originals(g.RAW_SOURCE_LP, folder, n):
            put_sample(p, lp_root)
    # A local uncompressed sibling must not double a recording.
    original = sorted(tap_root.glob("TAP/*.csv.gz"))[0]
    if not keep_input:
        with gzip.open(original, "rb") as stream, original.with_suffix("").open("wb") as dest:
            shutil.copyfileobj(stream, dest)
    g.RAW_SOURCE_TAP = tap_root; g.RAW_SOURCE_LP = lp_root
    g.SOURCE_TAP = sample_root / g.SOURCE_TAP.name; g.SOURCE_LP = sample_root / g.SOURCE_LP.name
    g.LABEL_TAP = ROOT / "research" / "gesture" / "_smoke_labels" / "DATA_TAP"
    g.LABEL_LP = ROOT / "research" / "gesture" / "_smoke_labels" / "DATA_LONG_PRESS"
    g.OUTPUT = results
    g.TRIALS_PER_MODEL = 4; g.TOP_K_PER_MODEL = 3; g.CV_FOLDS = 2
    import feature_selection as FS
    FS.INNER_FOLDS = 2; FS.SCREEN_PARAMS["n_estimators"] = 20; FS.REPEATS_GROUP = 1     # small data, fast ranker
    g.EVAL_MIN_FRAMES = 150              # short smoke recordings: enough events at this gate (default 500)
    g.MODEL_BUDGET_KB = 8; g.ALT_BUDGET_KB = 6           # small budgets keep the smoke fast
    if g.SIZE_LADDER_KB:                                   # final recipe: a small ladder, largest = MODEL_BUDGET_KB
        g.SIZE_LADDER_KB = (4.0, 6.0, 8.0)
    g.REPLAY_LP_TAPS_PER_RECORDING = 2
    g.FOLD_WORKERS = 2; g.FOLD_THREADS = 1                # CV folds and fold rankings run concurrently
    fake = Path(tempfile.mkdtemp())
    g.BASELINES = dict(tap=[("missing_example", str(fake / "nope.h"))], lp=[("not_configured_example", "")])
    g.main()
    for task, stem in (("tap", "TAP_SWIPE"), ("lp", "LONG_PRESS")):
        out = g.OUTPUT / task
        report = json.loads((out / "summary.json").read_text())
        assert report["status"] == "success" and report["fixed_threshold"] == g.task_threshold(task)
        assert report["threshold_tuned"] is False and g.task_threshold("tap") == 0.85
        assert report["class_one"] == ("RELEASE/REST" if task == "lp" else "TAP/SWIPE onset")
        assert report["budgets"] == ([8192, 6144, 4096] if task == "tap" and g.SIZE_LADDER_KB else [8192, 6144])
        if task == "tap" and g.TAP_OBJECTIVE == "balanced":    # final recipe: kinds, worst movement, size ladder
            cv_ = report["selected"]["cv"]
            assert cv_["objective"] == "balanced" and cv_["balanced"]["recall_by_kind"], cv_.get("balanced")
            assert abs(cv_["pooled_objective_score"] - cv_["balanced"]["score"]) < 1e-9
            assert report["selected"]["tap_logit_shift"] <= g.TAP_LOGIT_SHIFT[1]
            assert report["selected"]["positive_weight"] == 1.0
            if g.SIZE_LADDER_KB:
                assert report["selected"]["budget_bytes"] in report["budgets"], report["selected"]["budget_bytes"]
        assert report["lean8"]["enabled"] == g.LEAN8_TAG
        allowed = (set(g.ALL_FEATURES) | (set(TF.DERIVED) if task == "tap" and g.TAP_FEATURES == "diff" else set())
                   | (set(TF.LITE_COLUMNS) if task == "tap" and g.TAP_LITE == "diff" else set()))
        assert report["selected"]["features"] and set(report["selected"]["features"]) <= allowed
        assert report["experiment"] == "v3" and report["lean8"]["eval_min_frames"] == g.EVAL_MIN_FRAMES
        fs = report["feature_selection"]; assert fs and fs["default_k"] == len(report["selected"]["features"]), fs
        for f in ("fold_0_ranking.csv", "fold_1_ranking.csv", "selection_stability.json", "final_selected_features.json",
                  "correlation_clusters.json", "selected_features_by_trial.json", "report.md"):
            assert (out / "feature_selection" / f).is_file(), f
        held = json.loads((out / "holdout_report.json").read_text())
        assert [c["features_development"] for c in held["candidates"]] == \
               [c["features_holdout_candidate"] for c in report["candidates"]]
        if task == "tap":
            assert report["tap_target"]["source"].startswith("FALLBACK") and report["tap_target"]["value"] == 1.0
            assert report["tap_soft_labels"] is None                     # soft labels are off by default
            # Labeller-independent reference: every real pointer press of the TAP/SWIPE recordings.
            hold = report["selected"]["holdout_c_header"]
            assert hold is None or ("recall_press" in hold and "recall_strong" in hold), hold
            assert "recall_press" in (out / "COMPARISON.md").read_text()
            onsets = pd.read_csv(out / "cache" / "tap_onsets.csv")
            assert list(onsets.columns) == ["rec", "t_on", "kind", "lo", "hi"]
            assert set(onsets.kind) <= {"strong", "press"} and (onsets.kind == "press").any()
            assert (onsets.hi > onsets.lo).all()
            cache_hash_compat_check(out / "cache")
            if g.TAP_GENERALIZATION:
                tg = report["tap_generalization"]
                assert (tg["features"], tg["augmentation"], tg["derived_inputs"], tg["lean_lite_diff"], tg["labels"]) == \
                       (g.TAP_FEATURES, g.TAP_AUG, 144 if g.TAP_FEATURES == "diff" else 0, g.TAP_LITE, g.TAP_LABELS), tg
                n_lite = len(TF.LITE_COLUMNS) if g.TAP_LITE == "diff" else 0
                with np.load(out / "cache" / "replay_meta.npz") as z:
                    assert int(z["ncol"]) == 126 + (144 if g.TAP_FEATURES == "diff" else 0) + n_lite, int(z["ncol"])
                if n_lite:
                    with np.load(out / "cache" / "train_meta.npz") as z:
                        assert "row" in z.files
                counts = report["dataset"]["counts"]
                if g.TAP_LABELS == "press2":
                    assert counts.get("press2_recordings", 0) >= 1, counts
                assert tg["distill_label_weight"] == g.TAP_DISTILL and tg["finger_weight"] == g.TAP_FINGER_WEIGHT
                if g.TAP_DISTILL is not None:                         # teachers per fold, development, everyone
                    names = {p_.name.split("_")[1] for p_ in (out / "cache" / "distill").glob("distill_*.npz")}
                    assert {"cv0", "cv1", "development", "everyone"} <= names, names
                    assert "-distill" in g.TAP_TAG
                if g.TAP_OBS_MS is not None:                          # fold, development and everyone selections
                    assert tg["observable_ms"] == g.TAP_OBS_MS
                    names = {p_.name.split("_")[1] for p_ in (out / "cache" / "obs").glob("obs_*.npz")}
                    assert {"cv0", "cv1", "development", "everyone"} <= names, names
            assert report["data_identity"]["recordings"] > 0 and (out / "data_identity.json").is_file()
            sp = json.loads((out / "split.json").read_text())
            if g.SPLIT_BY == "file":                                   # one group per recording; people on both sides
                assert all("|" in x for x in sp["train_groups"] + sp["heldout_groups"])
                assert {x.split("|")[0] for x in sp["train_groups"]} & {x.split("|")[0] for x in sp["heldout_groups"]}
                assert sp["strata"] and sum(v[1] for v in sp["strata"].values()) == len(sp["heldout_groups"])
            else:                                                      # person split: nobody on both sides
                assert not set(sp["train_groups"]) & set(sp["heldout_groups"])
            # Completed output + changed data: the rerun refuses to reuse it and names the change.
            ident = json.loads((out / "data_identity.json").read_text())
            ident_bad = dict(ident, sha256="0" * 64, files=[[f[0], "x", f[2]] for f in ident["files"]])
            assert any("changed content" in m_ for m_ in g.identity_changes(ident_bad, ident))
            if g.LEAN8_MODE == "off":                                   # placement stress (report only)
                assert "Simulated band move" in (out / "COMPARISON.md").read_text()
                assert all(c["holdout_c_header_placement_stress"] for c in report["candidates"])
        gate = report["dataset"]["counts"]["lean8_gate"]
        assert 0 < gate["train_rows_used"] <= gate["train_rows_built"], gate
        assert "Steady state only" in (out / "COMPARISON.md").read_text()
        assert all("holdout_c_header_steady_state" in c for c in report["candidates"])
        roles = [c["role"] for c in report["candidates"]]
        ladder = bool(task == "tap" and g.SIZE_LADDER_KB)     # budget_<kb>kb only when the best raw score differs
        assert roles[0] == "rank_01" and (ladder or "budget_6kb" in roles), roles
        sel = report["selected"]; assert sel["role"].startswith("rank_") and sel.get("default") is True, sel
        assert all("reason" in s_ for s_ in report["skipped"]), report["skipped"]
        assert all("CV fold" in s_["reason"] for s_ in report["skipped"]), report["skipped"]
        for r_ in (roles if ladder else ["budget_6kb"]):
            if r_.startswith("budget_"):
                assert (out / "models" / r_ / (stem + ".h")).is_file(), r_
        assert (out / "models" / (stem + ".h")).read_bytes() == (out / sel["header"]).read_bytes()
        for c in report["candidates"]:
            assert c["static_bytes"] <= c["budget_bytes"], c
            hold = c["holdout_c_header"]; assert hold is None or hold["objective_score"] >= 0
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
        g.group_by_file(data, rpl)                         # as the trainer: file groups only
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
        assert b["status"] == "replayed" and (b["holdout"] is None or b["holdout"]["objective_score"] >= 0), b
        # Acceptance (report only): one row per rule with a verdict when both sides were replayed.
        acc = g.acceptance(task, report["candidates"], [b])
        assert all(isinstance(r["pass"], bool) and r["baseline"] == "copy_of_new_default" for r in acc), acc
        assert {r["metric"] for r in acc} <= {m for m, _, _ in g.ACCEPTANCE[task]}, acc
        assert "acceptance" in report
        if task == "lp":
            # Leakage: changing the labels of the validation and held-out people must not change a fold's ranking.
            pf = g.prepared_folds(data, rpl, tr, task)[0]
            r1 = FS.fit_fold_ranker(g, g.R, data, rpl, pf["tr"], task, seed=0, cache_dir=None, log=lambda *a: None)
            other = ~np.isin(np.arange(len(data["y"])), pf["tr"])
            y2 = data["y"].copy(); y2[other & (y2 >= 0)] = 1 - y2[other & (y2 >= 0)]
            r2 = FS.fit_fold_ranker(g, g.R, dict(data, y=y2), rpl, pf["tr"], task, seed=0, cache_dir=None, log=lambda *a: None)
            assert r1["order"] == r2["order"] and r1["mean_importance"] == r2["mean_importance"]
            FS.validate_fold_local(r1, pf["val_groups"]); FS.validate_fold_local(r1, set(data["group"][te]))
            print("FS_LEAKAGE_CHECK_PASSED", flush=True)
        print("UNIFIED_REAL_SOURCE_OK", task, json.dumps(report["dataset"]), report["selected"]["static_bytes"], flush=True)
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
    fold_parallel_check()
    man = json.loads((sample_root / ("lean_lite_manifest.json" if g.USE_LITE else "lean8_manifest.json")).read_text())
    assert man["firmware_parity"]["status"] in (("passed",) if g.USE_LITE else ("passed", "not run")), man["firmware_parity"]
    if g.USE_LITE:
        # Clean A/B: the Lean-lite run built exactly the rows, labels, frames and weights of the Lean-8 run.
        for task in ("tap", "lp"):
            a = ROOT / "research" / "gesture" / SMOKE_RESULTS["full"] / task / "cache"
            b = results / task / "cache"
            if not a.is_dir():
                print("LEAN_LITE_AB_CHECK_SKIPPED (no Lean-8 smoke cache)", task, flush=True); continue
            for npz in ("train_meta.npz", "replay_meta.npz"):
                with np.load(a / npz) as za, np.load(b / npz) as zb:
                    assert set(za.files) == set(zb.files), npz
                    for k in set(za.files) - {"ncol"}:          # ncol: 150 vs 141 input columns
                        assert np.array_equal(za[k], zb[k]), (task, npz, k)
            ra, rb = pd.read_csv(a / "recordings.csv"), pd.read_csv(b / "recordings.csv")
            for r in (ra, rb):                           # only the derived tree's folder name may differ
                r["file"] = r["file"].str.replace(r"/DATA_(TAP|LONG_PRESS)_(LEAN8|LITE)/", r"/DATA_\1_DERIVED/", regex=True)
            assert ra.equals(rb)
            for f in ("lp_windows.csv", "tap_onsets.csv"):
                if (a / f).is_file():
                    assert pd.read_csv(a / f).equals(pd.read_csv(b / f)), f
            print("LEAN_LITE_SAME_ROWS_AS_LEAN8", task, flush=True)
    derived = sorted(g.SOURCE_TAP.glob("TAP/*.csv.gz"))
    assert derived and not any(p.is_symlink() for p in derived)
    print("UNIFIED_V2_REAL_DATA_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    if "--lite-arm" in sys.argv or "--second-arm" in sys.argv:   # extra process in another mode: reuse sample, labels
        assert "--second-arm" in sys.argv or g.USE_LITE, "run with MUDRA_USE_LEAN8=lite"
        real_source_smoke(keep_input=True)
        sys.exit(0)
    lean8_checks()
    lean_lite_checks()
    tap_generalization_checks()
    tap_final_recipe_checks()
    unit_checks()
    validate_events_checks()
    if "--unit-only" not in sys.argv:
        real_source_smoke()
        if g.LEAN8_MODE == "full" and "--no-lite-arm" not in sys.argv:
            for name, extra in (("Lean-lite", dict(MUDRA_USE_LEAN8="lite")),
                                ("TAP diff + mild augmentation", dict(MUDRA_USE_LEAN8="0", MUDRA_TAP_FEATURES="diff",
                                                                      MUDRA_TAP_AUG="mild")),
                                ("TAP final recipe", dict(MUDRA_TAP_RECIPE="final14", MUDRA_TASKS="tap,lp")),
                                ("TAP final recipe + observable labels",
                                 dict(MUDRA_TAP_RECIPE="final14obs", MUDRA_TASKS="tap,lp")),
                                ("TAP final pipeline (balanced objective, size ladder, logit shift)",
                                 dict(MUDRA_TAP_RECIPE="final", MUDRA_TASKS="tap,lp")),
                                ("TAP maximum-protection recipe (obs40 + distillation + finger x2), split by file, "
                                 "strict target 1.5 (logit shift search)",
                                 dict(MUDRA_TAP_RECIPE="final14protect", MUDRA_TASKS="tap,lp", MUDRA_SPLIT="file",
                                      MUDRA_TAP_FALSE_PER_MIN_TARGET="1.5"))):
                p = subprocess.run([sys.executable, "-u", __file__, "--second-arm"], env=dict(os.environ, **extra))
                if p.returncode:
                    raise SystemExit(f"{name} real-source smoke failed ({p.returncode})")
