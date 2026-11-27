#!/usr/bin/env python3
"""DATA/DATA_TAP_F1: the input of the TAP F1 trainer (train_tap_f1.py), one CSV per DATA_TAP recording.

For every recording the file-level fold roadmap lists as trainable (DATA/fold_roadmap_files.json, fold_split.py):
  state      signal tap labels (claude_labeling/tap_signal.py, preset --labels): 1 = tap (the top of the tap's
             contact jolt), 0 = not a tap, -1 = unknown (the trainer drops those rows). The labeller repairs the
             pointer: taps the pointer missed are found on the recording's own rhythm, the second tap of a double
             tap merged into one press is split out, a late / early pointer is re-aligned per recording. A recording
             it judges broken (pointer and signal disagree on most presses) is not written, so it never trains.
  inputs     the input pool --features (tap_f1_features.py): raw firmware fields + diff / ratio inputs (+ Lean-lite)
  data mix   --tap-row-share 0.5 (default): TAP / SWIPE recordings train on every 2nd row only, the other rows are
             state -1 (dropped by the trainer, still scored by the event check). That gives back the legacy model's mix,
             half the rows from movement recordings (today's data has a third): the F1 objective then weighs a false
             tap on movement as it did, and the model is as strict (research/gesture/TAP_F1_LEGACY.md, section 2).
  class_id   copied from the recording (TAP / SWIPE 1, FINGER 3, WRIST 4, ARM 5, BAND 6, NOTHING 0)
  timestamp, PointerState   copied (not inputs: the trainer only takes p_core-> columns; evaluate_tap_f1.py uses them)

    python build_tap_f1_dataset.py                       # labels=legacy, features=newacc, tap rows 0.5 -> DATA/DATA_TAP_F1
    python build_tap_f1_dataset.py --limit 3             # 3 recordings of every folder and fold (a quick check)

Re-running is incremental: a recording whose output is newer than its source is kept, unless the labels, the inputs
or the labeller changed (then the folder is rebuilt from scratch). OUT/_build/ holds the manifest (one line per
recording: status, rows, tap rows, unknown rows, presses, taps, tier) and the build settings; the trainer ignores
folders that start with "_".
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "claude_labeling")]
import tap_f1_features as F  # noqa: E402
import tap_signal as TS  # noqa: E402
import tap_v10_rescue as RESCUE  # noqa: E402
import unified_tap_labeler_deterministic as V10  # noqa: E402
from fold_split import load_roadmap  # noqa: E402

DATA = ROOT / "DATA"
DEFAULT_OUT = DATA / "DATA_TAP_F1"
DEFAULT_LABELS = "legacy"
MIN_ROWS = 20                      # a recording with fewer usable rows is not written
DEFAULT_TAP_ROW_SHARE = 0.5        # share of TAP / SWIPE rows that train (see thin_tap_rows): the legacy data mix


def _source(rel):
    p = DATA / rel
    return p if p.exists() else p.with_name(p.name + ".gz")


LABELLERS = ("v10rescue", "v10", "legacy", "hybrid")


def labels(df, folder, name, preset, src):
    """(state per row, info). v10rescue: the old labeller on a repaired pointer (tap_v10_rescue.py); v10: the old
    labeller exactly; legacy / hybrid: the signal labeller's presets (tap_signal.py)."""
    if preset == "v10rescue":
        return RESCUE.label_frame(df, folder, name)
    if preset == "v10":
        if folder not in V10.POSITIVE_FOLDERS:
            return np.zeros(len(df), np.int8), dict(status="hard_negative")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep, _, _, out = V10.process_positive_file(Path(src), Path(src).parent.parent, Path(src), 1)
        return (out["state"].to_numpy() > 0).astype(np.int8), dict(status=rep.status, cycles=rep.press_cycles,
                                                                    accepted=rep.accepted_events)
    y, J, q = TS.label(TS.signals(df), folder, name, preset)
    return y, q


def thin_tap_rows(y, folder, share):
    """TAP / SWIPE recordings train on every k-th row only (k = round(1 / share)); the other rows become -1, which
    the trainer drops. Movement recordings keep every row. All rows stay in the CSV (the event check uses them)."""
    if share >= 1 or folder not in ("TAP", "SWIPE"):
        return y
    k = max(1, int(round(1.0 / share)))
    y = y.copy()
    drop = (np.arange(len(y)) % k != 0) & (y >= 0)
    y[drop] = -1
    return y


def one(job):
    rel, out_path, preset, pool_name, share = job
    src = _source(rel)
    folder, name = src.parent.name, src.name
    try:
        df = pd.read_csv(src, low_memory=False)
        raw_cols = [c for c in df.columns if c.startswith("p_core->")]
        names = F.pool(pool_name, raw_cols)
        y, q = labels(df, folder, name, preset, src)
        if len(y) != len(df):
            raise RuntimeError(f"{len(y)} labels for {len(df)} rows")
        y = thin_tap_rows(y, folder, share)
        info = dict(rel=rel, rows=len(df), tap_rows=int((y == 1).sum()), unknown_rows=int((y < 0).sum()),
                    presses=int(q.get("presses") or 0), tier=q.get("tier", ""),
                    **{k: q[k] for k in ("taps", "n_recovered", "n_second", "splits", "inserted", "bounces", "cycles",
                                         "accepted", "status") if k in q and k != "status"},
                    labeller_status=q.get("status", ""))
        if int((y >= 0).sum()) < MIN_ROWS:
            return dict(info, status="skipped_" + (q.get("tier") or "empty"))
        base_cols = list(raw_cols)
        B = df[raw_cols].to_numpy(np.float32)
        if F.needs_lite(names):
            lp = _source(rel.replace("DATA_TAP/", "DATA_TAP_LEAN8/", 1))
            fr = pd.read_csv(lp, usecols=["Timestamp", "lean8_frames"])
            if len(fr) != len(df) or not np.array_equal(fr.Timestamp.to_numpy(), df.Timestamp.to_numpy()):
                raise RuntimeError(f"{lp} does not match the recording row for row")
            B = np.hstack([B, F.lite_rows(B, raw_cols, fr.lean8_frames.to_numpy())])
            base_cols += list(F.TF.LITE_COLUMNS)
        fm = F.FeatureMaker(base_cols, names)
        X = fm(B)
        out = pd.DataFrame({n: (X[:, j] if j in set(fm.float_columns) else np.round(X[:, j]).astype(np.int32))
                            for j, n in enumerate(names)})
        out["state"] = y.astype(np.int8)
        out["class_id"] = df["class_id"].to_numpy()
        out["timestamp"] = df["Timestamp"].to_numpy()
        out["PointerState"] = df["PointerState"].to_numpy()       # not an input: evaluate_tap_f1.py's presses
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        out.to_csv(tmp, index=False, float_format="%.7g")
        os.replace(tmp, out_path)
        return dict(info, status="written", bytes=out_path.stat().st_size)
    except Exception as exc:  # reported in the manifest; the build fails at the end
        return dict(rel=rel, status=f"error: {type(exc).__name__}: {exc}")


def settings(preset, pool_name, share=1.0):
    code = {"v10rescue": ["tap_v10_rescue.py", "tap_signal.py", "unified_tap_labeler_deterministic.py"],
            "v10": ["unified_tap_labeler_deterministic.py"]}.get(preset, ["tap_signal.py"])
    sig = dict(labels=preset, tap_row_share=float(share), label_options=dict(TS.PRESETS.get(preset, {})),
               labeller_version={"v10rescue": RESCUE.VERSION, "v10": V10.SCRIPT_VERSION}.get(preset, TS.VERSION),
               features=pool_name, legacy_model_inputs=hashlib.sha256(F.LEGACY_MODEL.read_bytes()).hexdigest(),
               feature_code=hashlib.sha256((ROOT / "tap_f1_features.py").read_bytes()).hexdigest(),
               labeller_code={c: hashlib.sha256((ROOT / "claude_labeling" / c).read_bytes()).hexdigest() for c in code})
    sig["signature"] = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()
    return sig


def build(out=DEFAULT_OUT, preset=DEFAULT_LABELS, pool_name=F.DEFAULT_POOL, roadmap=DATA / "fold_roadmap_files.json",
          workers=None, limit=None, share=None, log=print):
    share = DEFAULT_TAP_ROW_SHARE if share is None else float(share)
    if preset not in LABELLERS:
        raise SystemExit(f"--labels must be one of {LABELLERS}")
    out = Path(out)
    meta = out / "_build"
    cfg = settings(preset, pool_name, share)
    old = json.loads((meta / "build.json").read_text()) if (meta / "build.json").exists() else {}
    if old.get("signature") != cfg["signature"] and out.exists():
        log(f"[build] settings changed ({old.get('labels')}/{old.get('features')} -> {preset}/{pool_name}): "
            f"rebuilding {out}")
        shutil.rmtree(out)
    client = load_roadmap(roadmap, DATA)
    recs = client.get_records("DATA_TAP")
    if limit:                                    # a quick check: K recordings of every (folder, fold)
        per = {}
        recs = [r for r in recs if per.setdefault((r.category, r.fold_id), []).append(r) or len(per[(r.category, r.fold_id)]) <= int(limit)]
    jobs, kept = [], 0
    for r in recs:
        rel = r.path.relative_to(DATA).as_posix().removesuffix(".gz")
        dst = out / r.category / Path(rel).name
        if dst.exists() and dst.stat().st_mtime >= _source(rel).stat().st_mtime:
            kept += 1; continue
        jobs.append((rel, str(dst), preset, pool_name, float(share)))
    rows = sum(r.sample_rows for r in recs)
    n_in = len(F.pool(pool_name, pd.read_csv(_source(recs[0].path.relative_to(DATA).as_posix()), nrows=0).columns))
    need = int(rows * n_in * 5.5 * len(jobs) / max(len(recs), 1))
    free = shutil.disk_usage(out.parent if out.parent.exists() else DATA).free
    log(f"[build] {len(recs)} recordings ({kept} up to date, {len(jobs)} to write), labels={preset}, "
        f"inputs={pool_name} ({n_in}), TAP/SWIPE rows that train: {share:g}, ~{need / 1e9:.1f} GB to write, "
        f"{free / 1e9:.1f} GB free")
    if need > free * 0.9:
        raise SystemExit(f"not enough disk space for {out}: need ~{need / 1e9:.1f} GB, {free / 1e9:.1f} GB free")
    meta.mkdir(parents=True, exist_ok=True)
    t0, res = time.time(), []
    workers = int(workers or os.environ.get("TAP_F1_WORKERS") or max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4), 1):
            res.append(r)
            if i % 250 == 0:
                log(f"  {i}/{len(jobs)} {time.time() - t0:.0f} s")
    if not limit:                                # a recording no longer trainable (deleted, now a duplicate) goes
        want = {out / r.category / Path(r.path.name.removesuffix(".gz")) for r in recs}
        for p in out.rglob("*.csv"):
            if not any(x.startswith("_") for x in p.relative_to(out).parts[:-1]) and p not in want:
                p.unlink(); log(f"[build] removed {p} (not a trainable roadmap recording any more)")
        for r in res:                            # skipped recordings must not keep an old output either
            if str(r.get("status", "")).startswith("skipped"):
                q = out / Path(r["rel"]).parent.name / Path(r["rel"]).name
                if q.exists():
                    q.unlink()
    man_path = meta / "manifest.csv"
    man = pd.DataFrame(res)
    if man_path.exists() and kept:
        prev = pd.read_csv(man_path)
        man = pd.concat([prev[~prev.rel.isin(man.rel)] if len(man) else prev, man], ignore_index=True)
    man.to_csv(man_path, index=False)
    errors = man[man.status.astype(str).str.startswith("error")] if len(man) else man
    (meta / "build.json").write_text(json.dumps(dict(cfg, roadmap=str(roadmap), roadmap_fingerprint=client.data_fingerprint,
                                                     recordings=len(recs), limit=limit), indent=1))
    if len(man):
        st = man.status.value_counts().to_dict()
        w = man[man.status == "written"]
        log(f"[build] done in {time.time() - t0:.0f} s: {st}; written tap rows {int(w.tap_rows.sum()):,}, "
            f"unknown rows {int(w.unknown_rows.sum()):,}")
    if len(errors):
        raise SystemExit(f"{len(errors)} recordings failed, first: {errors.iloc[0].rel}: {errors.iloc[0].status}")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--labels", default=os.environ.get("TAP_F1_LABELS", DEFAULT_LABELS))
    ap.add_argument("--features", default=os.environ.get("TAP_F1_FEATURES", F.DEFAULT_POOL), choices=F.POOLS)
    ap.add_argument("--roadmap", type=Path, default=DATA / "fold_roadmap_files.json")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only K recordings of every folder and fold (a quick check)")
    ap.add_argument("--tap-row-share", type=float, default=float(os.environ.get("TAP_F1_TAP_ROW_SHARE", DEFAULT_TAP_ROW_SHARE)),
                    help="share of TAP / SWIPE rows that train (0.5 = every 2nd row; the rest is -1, dropped)")
    a = ap.parse_args(argv)
    build(a.out, a.labels, a.features, a.roadmap, a.workers, a.limit, a.tap_row_share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
