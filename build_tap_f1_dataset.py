#!/usr/bin/env python3
"""DATA/DATA_TAP_F1: the input of the TAP F1 trainer (train_tap_f1.py), one CSV per DATA_TAP recording.

Three passes over the recordings the file-level fold roadmap lists as trainable (DATA/fold_roadmap_files.json):

  1. labels   every recording is labelled once into _build/labels/ (kept while the labeller and the recording do not
              change). --labels legacy (default): claude_labeling/tap_signal.py, 1 = the top of each tap's contact jolt,
              0 = not a tap; the pointer is repaired (per-recording delay, bounces, the 2nd tap of a double inside one
              press, taps the pointer missed found on the rhythm). A recording whose pointer and signal disagree on most
              presses is left out. v10rescue / v10 / hybrid on request.
  2. inputs   --features rfe (default): select_tap_f1_features.py keeps the useful inputs of ~3,200 candidates (the
              raw ACC fields, every ACC-ACC diff and ratio, EMG PSD only as diffs inside one channel - over time and
              zone vs zone, --emg within -, Lean-lite; no WL) on a row sample of the TRAINING folds only; kept in
              _build/selected_features.json (+ .md report) while labels, data and settings do not change. Or a fixed
              pool: leg / newacc / new / lite.
              --aug 1 (default): every training-fold recording gets one band placement / skin contact / power-grid
              hum / rotation change (tap_f1_augment.py, seeded by its file name) before its inputs are computed, for
              the feature elimination and the CSVs alike; the held-out fold stays as recorded
              (research/gesture/EMG_PSD_ROBUST_FEATURES.md).
  3. CSVs     per recording: the inputs, state (1 tap / 0 not a tap / -1 not trained), class_id, and timestamp /
              PointerState (not inputs: the trainer only takes p_core-> columns; evaluate_tap_f1.py uses them).
              --tap-row-share 0.5 (default): TAP / SWIPE recordings train on every 2nd row only (the others are
              state -1, dropped by the trainer, still scored by the event check): the legacy model's mix, half the rows
              from movement recordings (research/gesture/TAP_F1_LEGACY.md, section 2).

    python build_tap_f1_dataset.py                 # -> DATA/DATA_TAP_F1
    python build_tap_f1_dataset.py --limit 3       # 3 recordings of every folder and fold (a quick check)

Re-running is incremental. _build/ also holds the manifest (one line per recording) and the build settings; the
trainer ignores folders that start with "_".
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "claude_labeling")]
import tap_f1_augment as AUG  # noqa: E402
import tap_f1_features as F  # noqa: E402
import tap_signal as TS  # noqa: E402
import tap_v10_rescue as RESCUE  # noqa: E402
import unified_tap_labeler_deterministic as V10  # noqa: E402
from fold_split import load_roadmap  # noqa: E402

DATA = ROOT / "DATA"
DEFAULT_OUT = DATA / "DATA_TAP_F1"
DEFAULT_LABELS = "legacy"
LABELLERS = ("legacy", "v10rescue", "v10", "hybrid")
MIN_ROWS = 20                      # a recording with fewer usable rows is not written
DEFAULT_TAP_ROW_SHARE = 0.5        # share of TAP / SWIPE rows that train (thin_tap_rows): the legacy data mix
DEFAULT_RFE_ROWS = 400_000         # rows of the training folds the feature elimination looks at
TEST_FOLD = 0                      # = train_tap_f1.TEST_FOLD_ID: never seen by the feature elimination
DEFAULT_AUG = True                 # training folds get the placement / contact / hum / rotation augmentation
POS_FOLDERS = ("TAP", "SWIPE")


def _ctx():
    """worker processes from a clean server process (no inherited OpenMP / BLAS threads)"""
    import multiprocessing as mp
    return mp.get_context("forkserver") if "forkserver" in mp.get_all_start_methods() else None


def _source(rel):
    p = DATA / rel
    return p if p.exists() else p.with_name(p.name + ".gz")


def _sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


# ================================================================================================ 1. labels
def labels(df, folder, name, preset, src):
    """(state per row: 1 / 0 / -1, info) of one recording"""
    if preset == "v10rescue":
        return RESCUE.label_frame(df, folder, name)
    if preset == "v10":
        if folder not in V10.POSITIVE_FOLDERS:
            return np.zeros(len(df), np.int8), dict(status="hard_negative")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep, _, _, out = V10.process_positive_file(Path(src), Path(src).parent.parent, Path(src), 1)
        return (out["state"].to_numpy() > 0).astype(np.int8), dict(status=rep.status, cycles=rep.press_cycles,
                                                                    accepted=rep.accepted_events)
    y, J, q = TS.label(TS.signals(df), folder, name, preset)
    return y, q


def label_signature(preset):
    code = {"v10rescue": ["tap_v10_rescue.py", "tap_signal.py", "unified_tap_labeler_deterministic.py"],
            "v10": ["unified_tap_labeler_deterministic.py"]}.get(preset, ["tap_signal.py"])
    sig = dict(labels=preset, options=dict(TS.PRESETS.get(preset, {})),
               version={"v10rescue": RESCUE.VERSION, "v10": V10.SCRIPT_VERSION}.get(preset, TS.VERSION),
               code=_sha([ROOT / "claude_labeling" / c for c in code]))
    sig["signature"] = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()
    return sig


def label_one(job):
    rel, path, preset = job
    src = _source(rel)
    try:
        df = pd.read_csv(src, low_memory=False)
        y, q = labels(df, src.parent.name, src.name, preset, src)
        if len(y) != len(df):
            raise RuntimeError(f"{len(y)} labels for {len(df)} rows")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(y, np.int8))
        keep = ("presses", "taps", "tier", "n_contact", "n_weak", "n_second", "n_recovered", "no_jolt", "agreement",
                "delay", "splits", "inserted", "bounces", "cycles", "accepted")
        return dict(rel=rel, label_status=str(q.get("status", "")), rows=len(df), tap_rows=int((y == 1).sum()),
                    left_out_rows=int((y < 0).sum()), **{k: q[k] for k in keep if k in q})
    except Exception as exc:
        return dict(rel=rel, label_status=f"error: {type(exc).__name__}: {exc}")


def thin_tap_rows(y, folder, share):
    """TAP / SWIPE recordings train on every k-th row only (k = round(1 / share)); the other rows become -1, which
    the trainer drops. Movement recordings keep every row. All rows stay in the CSV (the event check uses them)."""
    if share >= 1 or folder not in POS_FOLDERS:
        return y
    k = max(1, int(round(1.0 / share)))
    y = y.copy()
    y[(np.arange(len(y)) % k != 0) & (y >= 0)] = -1
    return y


# ================================================================================================ base fields
def base_fields(df, rel, lite, aug=False):
    """(rows x base fields float32, base column names): the raw p_core fields (+ Lean-lite from DATA_TAP_LEAN8).
    aug: the recording's training augmentation (tap_f1_augment.draw, seeded by its file name) is applied to the EMG
    PSD codes and WL first, so Lean-lite is computed from the changed WL, as the firmware would."""
    raw_cols = [c for c in df.columns if c.startswith("p_core->")]
    B = df[raw_cols].to_numpy(np.float32)
    if aug:
        B = AUG.apply(B, raw_cols, AUG.draw(Path(rel).name))
    if not lite:
        return B, raw_cols
    lp = _source(rel.replace("DATA_TAP/", "DATA_TAP_LEAN8/", 1))
    if not lp.exists():
        raise FileNotFoundError(f"{lp}: Lean-lite needs the Lean-8 frame counter (or --no-lite)")
    fr = pd.read_csv(lp, usecols=["Timestamp", "lean8_frames"])
    if len(fr) != len(df) or not np.array_equal(fr.Timestamp.to_numpy(), df.Timestamp.to_numpy()):
        raise RuntimeError(f"{lp} does not match the recording row for row")
    return np.hstack([B, F.lite_rows(B, raw_cols, fr.lean8_frames.to_numpy())]), raw_cols + F.LITE_COLUMNS


# ================================================================================================ 2. inputs
def sample_one(job):
    """every `every`-th trainable row of one training-fold recording: base fields (no WL), label, fold"""
    rel, label_path, fold, every, share, lite, phase, aug = job
    src = _source(rel)
    df = pd.read_csv(src, low_memory=False)
    y = thin_tap_rows(np.load(label_path), src.parent.name, share)
    rows = np.flatnonzero(y >= 0)
    rows = rows[(phase % every)::every]
    B, cols = base_fields(df, rel, lite, aug=aug)
    keep = [i for i, c in enumerate(cols) if not F.is_wl(c)]
    target = ((y[rows] > 0) & (df["class_id"].to_numpy()[rows] == 1)).astype(np.int8)
    return B[np.ix_(rows, keep)], target, np.full(len(rows), fold, np.int8), [cols[i] for i in keep]


def select_inputs(recs, label_dir, meta, share, wl, lite, rfe_rows, workers, raw_cols, lsig, roadmap_fp, log,
                  emg=F.DEFAULT_EMG, aug=DEFAULT_AUG):
    import select_tap_f1_features as SEL
    cands = F.candidates(raw_cols, wl=wl, lite=lite, emg=emg)
    train = [r for r in recs if r.fold_id != TEST_FOLD]
    data_fp = hashlib.sha256("".join(sorted(f"{r.file_id}:{_source(_rel(r)).stat().st_mtime_ns}" for r in train)).encode()).hexdigest()
    sig = dict(labels=lsig["signature"], tap_row_share=share, wl=wl, lite=lite, emg=emg, rfe_rows=rfe_rows,
               aug=AUG.VERSION if aug else None, aug_code=_sha([ROOT / "tap_f1_augment.py"]) if aug else None,
               candidates=hashlib.sha256("\n".join(cands).encode()).hexdigest(), rfe=SEL.SETTINGS,
               roadmap=roadmap_fp, data=data_fp, selector_code=_sha([ROOT / "select_tap_f1_features.py"]))
    sig_h = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()
    sel_path = meta / "selected_features.json"
    if sel_path.exists():
        old = json.loads(sel_path.read_text())
        if old.get("signature") == sig_h:
            log(f"[build] inputs: {len(old['selected'])} kept by the earlier feature elimination (unchanged settings)")
            return old["selected"]
    usable = sum(int(r.sample_rows) for r in train) * 0.6           # about 60 % of rows train (TAP rows halved)
    every = max(1, int(usable // max(1, rfe_rows)))
    log(f"[build] inputs: feature elimination over {len(cands)} candidates; sample: "
        f"{'every' if every == 1 else f'1 in {every}'} trainable row of {len(train)} training-fold recordings")
    jobs = [(_rel(r), str(label_dir / r.category / (Path(_rel(r)).name + ".npy")), r.fold_id, every, share, lite, k,
             aug) for k, r in enumerate(train)]
    with ProcessPoolExecutor(workers, mp_context=_ctx()) as ex:
        parts = list(ex.map(sample_one, jobs, chunksize=4))
    sdir = meta / "rfe_sample"
    shutil.rmtree(sdir, ignore_errors=True); sdir.mkdir(parents=True)
    np.save(sdir / "base.npy", np.concatenate([p[0] for p in parts]).astype(np.float32))
    np.save(sdir / "y.npy", np.concatenate([p[1] for p in parts]))
    np.save(sdir / "fold.npy", np.concatenate([p[2] for p in parts]))
    (sdir / "meta.json").write_text(json.dumps(dict(base_cols=parts[0][3], raw_cols=list(raw_cols), lite=lite)))
    selected, report = SEL.select(sdir, cands, log=log)
    report["signature"] = sig_h
    report["signature_parts"] = {k: v for k, v in sig.items() if k != "rfe"}
    sel_path.write_text(json.dumps(report, indent=1))
    (meta / "selected_features.md").write_text(SEL.report_markdown(report))
    shutil.rmtree(sdir, ignore_errors=True)
    return selected


# ================================================================================================ 3. CSVs
def write_one(job):
    rel, label_path, out_path, names, share, aug = job
    src = _source(rel)
    try:
        df = pd.read_csv(src, low_memory=False)
        y = thin_tap_rows(np.load(label_path), src.parent.name, share)
        if len(y) != len(df):
            raise RuntimeError(f"{len(y)} labels for {len(df)} rows")
        if int((y >= 0).sum()) < MIN_ROWS:
            return dict(rel=rel, status="skipped_left_out")
        B, cols = base_fields(df, rel, F.needs_lite(names), aug=aug)
        fm = F.FeatureMaker(cols, names)
        X = fm(B)
        fl = set(fm.float_columns)
        out = pd.DataFrame({n: (X[:, j] if j in fl else np.round(X[:, j]).astype(np.int32)) for j, n in enumerate(names)})
        out["state"] = y.astype(np.int8)
        out["class_id"] = df["class_id"].to_numpy()
        out["timestamp"] = df["Timestamp"].to_numpy()
        out["PointerState"] = df["PointerState"].to_numpy()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        out.to_csv(tmp, index=False, float_format="%.7g")
        os.replace(tmp, out_path)
        return dict(rel=rel, status="written", trained_rows=int((y >= 0).sum()), trained_tap_rows=int((y > 0).sum()),
                    bytes=out_path.stat().st_size, aug=AUG.VERSION if aug else "none")
    except Exception as exc:  # reported in the manifest; the build fails at the end
        return dict(rel=rel, status=f"error: {type(exc).__name__}: {exc}")


def one(job):
    """label + write one recording with a fixed pool, in memory (tests and quick checks)"""
    rel, out_path, preset, pool_name, share, wl = job
    src = _source(rel)
    tmp = Path(out_path).with_suffix(".labels.npy")
    info = label_one((rel, str(tmp), preset))
    if str(info.get("label_status", "")).startswith("error"):
        return dict(info, status=info["label_status"])
    cols = pd.read_csv(src, nrows=0).columns
    r = write_one((rel, str(tmp), out_path, F.pool(pool_name, cols, wl=wl), share, False))
    tmp.unlink(missing_ok=True)
    return dict(info, **r)


def _rel(r):
    return r.path.relative_to(DATA).as_posix().removesuffix(".gz")


# ================================================================================================ the build
def build(out=DEFAULT_OUT, preset=DEFAULT_LABELS, pool_name=F.DEFAULT_POOL, roadmap=DATA / "fold_roadmap_files.json",
          workers=None, limit=None, share=None, wl=None, lite=True, rfe_rows=DEFAULT_RFE_ROWS, log=print, emg=None,
          aug=None):
    share = DEFAULT_TAP_ROW_SHARE if share is None else float(share)
    aug = DEFAULT_AUG if aug is None else bool(aug)
    wl = F.DEFAULT_WL if wl is None else bool(wl)
    emg = F.DEFAULT_EMG if emg is None else str(emg)
    if emg not in F.EMG_MODES:
        raise SystemExit(f"--emg must be one of {F.EMG_MODES}")
    if preset not in LABELLERS:
        raise SystemExit(f"--labels must be one of {LABELLERS}")
    if pool_name not in F.POOLS:
        raise SystemExit(f"--features must be one of {F.POOLS}")
    out = Path(out); meta = out / "_build"; label_dir = meta / "labels"
    workers = int(workers or os.environ.get("TAP_F1_WORKERS") or max(1, (os.cpu_count() or 2) - 1))
    t0 = time.time()
    client = load_roadmap(roadmap, DATA)
    recs = client.get_records("DATA_TAP")
    if limit:                                    # a quick check: K recordings of every (folder, fold)
        per = {}
        recs = [r for r in recs if per.setdefault((r.category, r.fold_id), []).append(r) or len(per[(r.category, r.fold_id)]) <= int(limit)]
    raw_cols = [c for c in pd.read_csv(_source(_rel(recs[0])), nrows=0).columns if c.startswith("p_core->")]

    # 1. labels -------------------------------------------------------------------------------------------------
    lsig = label_signature(preset)
    sig_path = label_dir / "signature.json"
    if not sig_path.exists() or json.loads(sig_path.read_text()).get("signature") != lsig["signature"]:
        shutil.rmtree(label_dir, ignore_errors=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    sig_path.write_text(json.dumps(lsig, indent=1))
    lpath = {_rel(r): label_dir / r.category / (Path(_rel(r)).name + ".npy") for r in recs}
    jobs = [(rel, str(p), preset) for rel, p in lpath.items()
            if not p.exists() or p.stat().st_mtime < _source(rel).stat().st_mtime]
    log(f"[build] 1/3 labels ({preset}): {len(recs)} recordings, {len(recs) - len(jobs)} already labelled")
    infos = []
    with ProcessPoolExecutor(workers, mp_context=_ctx()) as ex:
        for i, r in enumerate(ex.map(label_one, jobs, chunksize=4), 1):
            infos.append(r)
            if i % 500 == 0:
                log(f"  {i}/{len(jobs)} labelled, {time.time() - t0:.0f} s")
    bad = [r for r in infos if str(r.get("label_status", "")).startswith("error")]
    if bad:
        raise SystemExit(f"{len(bad)} recordings failed to label, first: {bad[0]['rel']}: {bad[0]['label_status']}")

    # 2. inputs -------------------------------------------------------------------------------------------------
    selected = None
    if pool_name == "rfe":
        selected = select_inputs(recs, label_dir, meta, share, wl, lite, rfe_rows, workers, raw_cols, lsig,
                                 client.data_fingerprint, log, emg=emg, aug=aug)
    names = F.pool(pool_name, raw_cols, wl=wl, selected=selected)
    if F.needs_lite(names) and not lite:
        raise SystemExit("the inputs use Lean-lite but --no-lite was given")
    log(f"[build] 2/3 inputs: {pool_name}, {len(names)} inputs ({sum(n in F.LITE_COLUMNS for n in names)} Lean-lite, "
        f"{'WL on' if wl else 'WL, WL_head, WL_tail removed (9 fields, 9 ratios)'}, "
        f"EMG {emg if pool_name == 'rfe' else 'fixed pool'}, "
        f"augmentation {'on (training folds)' if aug else 'off'})")

    # 3. CSVs ---------------------------------------------------------------------------------------------------
    bsig = dict(labels=lsig["signature"], pool=pool_name, inputs=hashlib.sha256("\n".join(names).encode()).hexdigest(),
                tap_row_share=share, wl=wl, aug=AUG.VERSION if aug else None,
                writer=_sha([ROOT / "build_tap_f1_dataset.py", ROOT / "tap_f1_features.py", ROOT / "tap_f1_augment.py"]))
    bsig["signature"] = hashlib.sha256(json.dumps(bsig, sort_keys=True).encode()).hexdigest()
    old = json.loads((meta / "build.json").read_text()) if (meta / "build.json").exists() else {}
    if old.get("signature") != bsig["signature"]:
        n_old = 0
        for p in out.rglob("*.csv"):
            if not any(x.startswith("_") for x in p.relative_to(out).parts[:-1]):
                p.unlink(); n_old += 1
        if n_old:
            log(f"[build] settings or inputs changed: {n_old} old CSVs removed")
    dst = {_rel(r): out / r.category / Path(_rel(r)).name for r in recs}
    augmented = {_rel(r): bool(aug and r.fold_id != TEST_FOLD) for r in recs}   # the held-out fold stays as recorded
    man_path = meta / "manifest.csv"
    prev_aug = {}
    if man_path.exists():
        prev = pd.read_csv(man_path)
        if "aug" in prev.columns:
            prev_aug = dict(zip(prev.rel, prev.aug.fillna("none").astype(str)))
    want_aug = lambda rel: AUG.VERSION if augmented[rel] else "none"
    jobs = [(rel, str(lpath[rel]), str(d), names, share, augmented[rel]) for rel, d in dst.items()
            if not d.exists() or d.stat().st_mtime < max(_source(rel).stat().st_mtime, lpath[rel].stat().st_mtime)
            or prev_aug.get(rel, "none") != want_aug(rel)]
    need = int(sum(r.sample_rows for r in recs) * (len(names) * 5.5 + 30) * len(jobs) / max(len(recs), 1))
    free = shutil.disk_usage(out.parent if out.parent.exists() else DATA).free
    log(f"[build] 3/3 CSVs: {len(jobs)} to write ({len(recs) - len(jobs)} up to date), TAP/SWIPE rows that train: "
        f"{share:g}, ~{need / 1e9:.1f} GB, {free / 1e9:.1f} GB free")
    if need > free * 0.9:
        raise SystemExit(f"not enough disk space for {out}: need ~{need / 1e9:.1f} GB, {free / 1e9:.1f} GB free")
    res = []
    with ProcessPoolExecutor(workers, mp_context=_ctx()) as ex:
        for i, r in enumerate(ex.map(write_one, jobs, chunksize=4), 1):
            res.append(r)
            if i % 500 == 0:
                log(f"  {i}/{len(jobs)} written, {time.time() - t0:.0f} s")
    for r in res:                                # a recording left out must not keep an old CSV
        if str(r.get("status", "")).startswith("skipped"):
            dst[r["rel"]].unlink(missing_ok=True)
    if not limit:                                # a recording no longer trainable (deleted, now a duplicate) goes
        want = set(dst.values())
        for p in out.rglob("*.csv"):
            if not any(x.startswith("_") for x in p.relative_to(out).parts[:-1]) and p not in want:
                p.unlink(); log(f"[build] removed {p} (not a trainable roadmap recording any more)")
    man = pd.DataFrame(res)
    if len(infos):
        lab = pd.DataFrame(infos)
        man = lab.merge(man, on="rel", how="outer") if len(man) else lab
    if man_path.exists() and len(man):
        prev = pd.read_csv(man_path)
        man = pd.concat([prev[~prev.rel.isin(man.rel)], man], ignore_index=True)
    if len(man):
        man.to_csv(man_path, index=False)
    (meta / "build.json").write_text(json.dumps(dict(bsig, label_settings=lsig, inputs=names, roadmap=str(roadmap),
                                                     roadmap_fingerprint=client.data_fingerprint,
                                                     recordings=len(recs), limit=limit), indent=1))
    errors = [r for r in res if str(r.get("status", "")).startswith("error")]
    st = pd.Series([r.get("status") for r in res]).value_counts().to_dict() if res else {}
    log(f"[build] done in {time.time() - t0:.0f} s: {st}")
    if errors:
        raise SystemExit(f"{len(errors)} recordings failed, first: {errors[0]['rel']}: {errors[0]['status']}")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--labels", default=os.environ.get("TAP_F1_LABELS", DEFAULT_LABELS), choices=LABELLERS)
    ap.add_argument("--features", default=os.environ.get("TAP_F1_FEATURES", F.DEFAULT_POOL), choices=F.POOLS)
    ap.add_argument("--roadmap", type=Path, default=DATA / "fold_roadmap_files.json")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only K recordings of every folder and fold (a quick check)")
    ap.add_argument("--tap-row-share", type=float, default=float(os.environ.get("TAP_F1_TAP_ROW_SHARE", DEFAULT_TAP_ROW_SHARE)),
                    help="share of TAP / SWIPE rows that train (0.5 = every 2nd row; the rest is -1, dropped)")
    ap.add_argument("--wl", type=int, choices=(0, 1), default=int(os.environ.get("TAP_F1_WL", int(F.DEFAULT_WL))),
                    help="1: allow inputs on the absolute EMG WL level (off by default)")
    ap.add_argument("--no-lite", action="store_true", default=os.environ.get("TAP_F1_LITE", "1") == "0",
                    help="no Lean-lite candidates (the device firmware lacks lean_lite.h)")
    ap.add_argument("--rfe-rows", type=int, default=int(os.environ.get("TAP_F1_RFE_ROWS", DEFAULT_RFE_ROWS)),
                    help="rows of the training folds the feature elimination looks at")
    ap.add_argument("--emg", default=os.environ.get("TAP_F1_EMG", F.DEFAULT_EMG), choices=F.EMG_MODES,
                    help="within (default): EMG PSD candidates only as diffs inside one channel (over time, zone vs "
                         "zone); robust: only change over time; all: every pair and the absolute codes "
                         "(research/gesture/EMG_PSD_ROBUST_FEATURES.md)")
    ap.add_argument("--aug", type=int, choices=(0, 1), default=int(os.environ.get("TAP_F1_AUG", int(DEFAULT_AUG))),
                    help="1 (default): every training-fold recording gets one placement / contact / hum / rotation "
                         "change (tap_f1_augment.py); the held-out fold stays as recorded")
    a = ap.parse_args(argv)
    build(a.out, a.labels, a.features, a.roadmap, a.workers, a.limit, a.tap_row_share, bool(a.wl), not a.no_lite,
          a.rfe_rows, emg=a.emg, aug=bool(a.aug))
    return 0


if __name__ == "__main__":
    sys.exit(main())
