#!/usr/bin/env python3
"""One entrypoint to train two embedded Mudra models from Claude's unified y_true.

    python -u research/gesture/train_unified.py

TAP_SWIPE: 1 = TAP/SWIPE onset, 0 = no onset, unknown = excluded.
LONG_PRESS: 1 = RELEASE/REST, 0 = sustained HOLD, unknown = excluded.

Only directly logged p_core-> fields can reach the models. PointerState,
recording folder, timestamps, contact teacher scores, hold_ms and subject
are labels, sampling aids, or split metadata; NEVER model features.

The successful previous TAP_SWIPE and LP models are NOT touched.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "tap_real"))
sys.path.insert(0, str(ROOT))
# Import sets environment thread controls BEFORE loading the numerical stack.
import train_claude_v2_optuna as previous_tap

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

v6 = previous_tap.v6
export_plus = v6.export_plus

# Intentionally no CLI. Modify only these simple research constants if desired.
SOURCE_TAP = ROOT / "DATA" / "DATA_TAP"
SOURCE_LP = ROOT / "DATA" / "DATA_LONG_PRESS"
LABEL_TAP = ROOT / "DATA" / "LABELS" / "DATA_TAP"
LABEL_LP = ROOT / "DATA" / "LABELS" / "DATA_LONG_PRESS"
OUTPUT = ROOT / "research" / "gesture" / "results_unified_20261117"
CLAUDE_LABELER = ROOT / "claude_labeling" / "label_gestures.py"
TRIALS_PER_MODEL = 40
TOP_K_PER_MODEL = 10
CV_FOLDS = 3
MAX_PER_CLASS_PER_RECORDING = 45
TEST_FRACTION = 0.22
FIXED_THRESHOLD = 0.85
CPU_THREADS = previous_tap.CPU_THREADS
SEED = 20261117
VERIFIED_HASHES = True

ALL_RAW = tuple(previous_tap.FULL_126)
assert len(ALL_RAW) == 126 and len(set(ALL_RAW)) == 126
assert all(x.startswith("p_core->") for x in ALL_RAW)

TAP_FOLDERS = frozenset(("TAP", "SWIPE"))
HOLD_FOLDERS = frozenset((
    "SOFT_LONG_PRESS", "MID_LONG_PRESS", "STRONG_LONG_PRESS",
    "LONG_PRESS", "DRAG_AND_DROP"))
TAP_NEGATIVES = frozenset(("ARM", "BAND", "FINGER", "WRIST", "NOTHING"))
LP_NEGATIVES = frozenset(("CONNECTED", "DISCONNECTED", "LOOSE", "TABLE", "NOTHING"))
TAP_CLASSES = TAP_FOLDERS | TAP_NEGATIVES
LP_CLASSES = HOLD_FOLDERS | TAP_FOLDERS | TAP_NEGATIVES | LP_NEGATIVES

TASKS = (
    dict(name="tap", header="TAP_SWIPE", class_one="TAP/SWIPE onset",
         class_zero="not an onset", metric="weighted grouped-CV silver F1"),
    dict(name="lp", header="LONG_PRESS", class_one="RELEASE/REST",
         class_zero="HELD", metric="weighted harmonic mean: 0.70 hold-safety / 0.30 release-recall"),
)


def notice(*values):
    print(*values, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    v6.save_json(path, obj)


def canonical_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing recording folder: {root}")
    files = sorted(root.glob("*/*.csv.gz"))
    if not files:
        raise ValueError(f"No compressed original recordings under {root}")
    seen = set()
    for file in files:
        rel = file.relative_to(root).as_posix()
        if rel in seen:
            raise ValueError(f"Duplicate recording: {rel}")
        seen.add(rel)
    return files


def sidecar(source: Path, source_root: Path, labels_root: Path) -> Path:
    return labels_root / source.relative_to(source_root).parent / (
        source.name[:-7] + ".labels.csv.gz")


def labels_complete(source: Path, output: Path, files: list[Path]) -> bool:
    summary = output / "_summary.json"
    if not summary.is_file():
        return False
    state = json.loads(summary.read_text(encoding="utf-8"))
    total = state.get("totals", {})
    args = total.get("args", {})
    if total.get("failed") != 0 or int(total.get("files", -1)) < len(files):
        return False
    if int(total["files"]) > len(files):
        # The older Claude tool scans both *.csv and *.csv.gz. Some users
        # have local uncompressed copies beside the canonical archive.
        # Reuse its per-canonical-file labels only when every canonical
        # sidecar exists; the row/timestamp audit below still applies.
        if next(source.glob("*/*.csv"), None) is None:
            return False
    if args.get("kind", "auto") != "auto":
        return False
    if args.get("disconnect_rule", "all") != "all":
        return False
    if args.get("use_weak", False) or args.get("promote_double_partners", False):
        return False
    if args.get("trust_pointer", False):
        return False
    if list(args.get("tap_win", [0, 80])) != [0, 80]:
        return False
    return all(sidecar(p, source, output).is_file() for p in files)


def ensure_labels(root: Path, label_dir: Path) -> list[Path]:
    """Reuse matching Claude outputs, otherwise generate without overwriting."""
    files = canonical_files(root)
    if labels_complete(root, label_dir, files):
        notice("UNIFIED_LABELS_REUSED", root.name, len(files), label_dir)
        return files
    if label_dir.exists():
        raise RuntimeError(
            f"Labels exist but are incomplete/incompatible: {label_dir}; "
            "move that directory out of the way, never overwrite it silently.")
    staging = label_dir.with_name(label_dir.name + ".building")
    if staging.exists():
        raise RuntimeError(f"Partial Claude output exists: {staging}; inspect it before retrying.")
    staging.parent.mkdir(parents=True, exist_ok=True)
    # Claude's labeler walks .csv.gz AND .csv. Use a canonical symlink tree
    # when a local uncompressed sibling exists, so no recording is doubled.
    extra_csv = next(root.glob("*/*.csv"), None) is not None
    input_dir = root
    symlink_root = None
    if extra_csv:
        symlink_root = OUTPUT / "_canonical_label_input" / root.name
        for p in files:
            dst = symlink_root / p.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                if dst.resolve() != p.resolve():
                    raise RuntimeError("Staging link points to different recording: " + str(dst))
            elif dst.exists():
                raise RuntimeError(f"Refuse to overwrite staged file {dst}")
            else:
                dst.symlink_to(p.resolve())
        input_dir = symlink_root
        notice("UNIFIED_CANONICAL_GZ_ONLY", root.name, len(files))
    notice("UNIFIED_LABELS_GENERATING", root.name, len(files))
    subprocess.run([sys.executable, str(CLAUDE_LABELER), str(input_dir), str(staging)],
                   check=True, cwd=ROOT)
    if not labels_complete(root, staging, files):
        raise ValueError(f"Claude unified label export failed audit: {staging}")
    staging.rename(label_dir)
    notice("UNIFIED_LABELS_READY", root.name, len(files), label_dir)
    return files


def selected_sources(task: str, tap_files: list[Path],
                     lp_files: list[Path]) -> list[tuple[Path, Path, Path]]:
    rows = []
    if task == "tap":
        rows.extend((p, SOURCE_TAP, LABEL_TAP)
                    for p in tap_files if p.parent.name in TAP_CLASSES)
        # Long-press starts/releases are UNKNOWN in unified tap labels.
        # Only trusted mid-hold or idle rows become additional negatives.
        rows.extend((p, SOURCE_LP, LABEL_LP)
                    for p in lp_files if p.parent.name in HOLD_FOLDERS)
    else:
        rows.extend((p, SOURCE_LP, LABEL_LP)
                    for p in lp_files if p.parent.name in LP_CLASSES)
    if not rows:
        raise ValueError(f"No usable {task} recordings")
    return sorted(rows, key=lambda a: (a[1].name, str(a[0])))


def common_p_core(records: list[tuple[Path, Path, Path]]) -> list[str]:
    common = set(ALL_RAW)
    missing = Counter()
    for p, _, _ in records:
        with gzip.open(p, "rt", newline="") as handle:
            names = set(next(csv.reader(handle)))
        for name in tuple(common):
            if name not in names:
                missing[name] += 1
        common.intersection_update(names)
    cols = [f for f in ALL_RAW if f in common]
    if not set(previous_tap.EMG_CURRENT + previous_tap.WL_FULL).issubset(cols):
        raise ValueError("Essential instantaneous EMG/WL firmware columns unavailable")
    if not cols or any(not x.startswith("p_core->") for x in cols):
        raise ValueError("Only original p_core fields allowed")
    export_plus.validate_feature_schema_accessors(cols)
    notice("UNIFIED_FEATURES", len(cols), "/", len(ALL_RAW),
           "missing", [x for x in ALL_RAW if x not in common][:18])
    return cols


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    have = set(columns)
    sets = {"all_available": columns[:]}
    for k, fs in previous_tap.FEATURE_SETS.items():
        if k == "full126":
            continue
        if set(fs).issubset(have):
            sets[k] = list(fs)
    # Guarantee an early full-feature candidate even in an interrupted run.
    if len(columns) == 126:
        sets["full126"] = columns[:]
    return sets


def spread(a: np.ndarray, cap: int) -> np.ndarray:
    if len(a) <= cap:
        return a
    return a[np.linspace(0, len(a)-1, num=cap, dtype=int)]


def sample_training_rows(label: np.ndarray, events: np.ndarray,
                         target: str) -> np.ndarray:
    chosen = []
    for cls in (0, 1):
        pool = np.flatnonzero(label == cls)
        if not len(pool):
            continue
        anchors = np.zeros(0, dtype=int)
        if (target == "tap" and cls == 1) or (target == "lp" and cls == 0):
            event_ids = np.unique(events[pool])
            anchored = [pool[events[pool] == k][len(pool[events[pool] == k])//2]
                        for k in event_ids if k >= 0]
            anchors = spread(np.asarray(anchored, dtype=int),
                             MAX_PER_CLASS_PER_RECORDING)
        remaining = MAX_PER_CLASS_PER_RECORDING - len(anchors)
        filler = spread(np.setdiff1d(pool, anchors, assume_unique=True),
                        max(0, remaining)) if remaining > 0 else np.zeros(0,int)
        selected = np.unique(np.r_[anchors, filler])
        chosen.append(selected)
    return np.sort(np.concatenate(chosen)) if chosen else np.zeros(0,int)


def dataset(task: str, records, output: Path):
    columns = common_p_core(records)
    feature_set = feature_sets(columns)
    data = dict(X=[], y=[], w=[], group=[], file=[], category=[], event=[], hold_ms=[])
    info = Counter()
    sources = []
    selected_cols = set(columns) | {"Timestamp", "User"}
    for i, (source, root, labels_root) in enumerate(records, 1):
        label_file = sidecar(source, root, labels_root)
        if not label_file.is_file():
            raise FileNotFoundError(label_file)
        cols = ["Timestamp", "tap_label", "tap_event_id"] if task == "tap" else [
            "Timestamp", "hold_label", "hold_press_id", "hold_ms"]
        lab = pd.read_csv(label_file, usecols=cols, compression="gzip", low_memory=False)
        original = pd.to_numeric(lab["tap_label" if task=="tap" else "hold_label"],
                                  errors="raise").to_numpy(np.int8)
        if not np.isin(original, (-1, 0, 1)).all():
            raise ValueError(f"Invalid silver labels in {label_file}")
        # Claude: hold=1. Firmware: class 1 means RELEASE. This inversion
        # is explicit and never affects UNKNOWN rows.
        y = np.where(original < 0, -1,
                     original if task == "tap" else 1 - original).astype(np.int8)
        folder = source.parent.name
        if task=="tap" and folder in HOLD_FOLDERS and (y==1).any():
            raise ValueError(f"Unexpected positive tap onset in hold folder {source}")
        if task=="lp" and folder in TAP_FOLDERS and (y==0).any():
            raise ValueError(f"Unexpected HOLD in TAP or SWIPE folder {source}")
        # Skip unsupervised gesture files instead of declaring their
        # unconfirmed gestures negative.
        if task=="tap" and folder in TAP_FOLDERS and not (y==1).any():
            info["positive_bags_without_confirmed_event"] += 1
            continue
        if task=="lp" and folder in HOLD_FOLDERS and not (y==0).any():
            info["hold_bags_without_confirmed_event"] += 1
            continue
        events = lab["tap_event_id" if task=="tap" else "hold_press_id"].to_numpy(np.int32)
        sample = sample_training_rows(y, events, task)
        if not len(sample):
            info["empty_labeled_bags"] += 1
            continue
        df = pd.read_csv(source, compression="gzip",
                         usecols=lambda c: c in selected_cols, low_memory=False)
        if len(df) != len(lab):
            raise ValueError(f"Row count mismatch: {source}")
        if not np.array_equal(pd.to_numeric(df["Timestamp"],errors="raise").to_numpy(np.int64),
                              lab["Timestamp"].to_numpy(np.int64)):
            raise ValueError(f"Timestamp mismatch: {source}")
        if not set(columns).issubset(df):
            raise ValueError(f"Missing raw feature in {source}")
        xx = df.iloc[sample][columns].apply(pd.to_numeric,errors="raise").to_numpy(np.float32)
        if not np.isfinite(xx).all():
            raise ValueError(f"Nonfinite firmware input in {source}")
        yy = y[sample]
        user = str(df["User"].iloc[0]) if "User" in df.columns else ""
        who = v6.person_group(user, source.name[:-7])
        file_key = source.relative_to(ROOT).as_posix()
        data["X"].append(xx)
        data["y"].append(yy)
        data["w"].extend([1.0]*len(sample))
        data["group"].extend([who]*len(sample))
        data["file"].extend([file_key]*len(sample))
        data["category"].extend([folder]*len(sample))
        data["event"].extend(events[sample].tolist())
        data["hold_ms"].extend(
            lab["hold_ms"].to_numpy(np.int32)[sample].tolist()
            if task == "lp" else [-1]*len(sample))
        sources.append({"file":file_key,"source_sha256":sha256(source) if VERIFIED_HASHES else None,
                        "source_bytes":source.stat().st_size,
                        "label_sha256":sha256(label_file) if VERIFIED_HASHES else None,
                        "selected_rows":len(sample)})
        info["read_sources"] += 1
        info[folder+"_files"] += 1
        info["class_1_rows"] += int((yy==1).sum())
        info["class_0_rows"] += int((yy==0).sum())
        if i%350==0:
            notice("UNIFIED_DATASET_EXTRACT",task,i,len(records),len(data["y"]))
    if not data["X"]:
        raise ValueError(f"No supervised {task} recordings")
    result = dict(X=np.concatenate(data["X"]).astype(np.float32),
                  y=np.concatenate(data["y"]).astype(np.int8),
                  w=np.asarray(data["w"],dtype=np.float32),
                  group=np.asarray(data["group"],dtype=object),
                  file=np.asarray(data["file"],dtype=object),
                  category=np.asarray(data["category"],dtype=object),
                  event=np.asarray(data["event"],dtype=np.int32),
                  hold_ms=np.asarray(data["hold_ms"],dtype=np.int32),
                  columns=columns, feature_sets=feature_set)
    if set(np.unique(result["y"])) != {0, 1}:
        raise ValueError(f"{task} has no both supervised classes")
    if len(set(result["group"])) < 7:
        raise ValueError(f"Only {len(set(result['group']))} person groups in {task}")
    manifest = {
        "task": task,
        "features": columns,
        "feature_sets": feature_set,
        "n_sources": len(sources),
        "n_samples": len(result["y"]),
        "subjects": len(set(result["group"])),
        "counts": dict(info),
        "labeler_sha256": sha256(CLAUDE_LABELER),
        "fixed_decision_threshold": FIXED_THRESHOLD,
        "positive_semantics": (TASKS[0] if task=="tap" else TASKS[1])["class_one"],
        "negative_semantics": (TASKS[0] if task=="tap" else TASKS[1])["class_zero"],
        "source_manifest": sources,
        "provenance_warning": "Silver labels from offline teacher, not physical ground truth; user grouping inferred from metadata",
    }
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"dataset_manifest.json",manifest)
    return result,manifest


def split_heldout(data, task):
    g = data["group"];y=data["y"];c=data["category"]
    splitter=GroupShuffleSplit(n_splits=70,test_size=TEST_FRACTION,
                               random_state=SEED+int(task=="lp"))
    for tr,te in splitter.split(data["X"],y,g):
        if set(y[tr]) != {0,1} or set(y[te]) != {0,1}:
            continue
        if len(set(g[tr])) < CV_FOLDS + 1:
            continue
        if task=="tap" and not {"TAP","SWIPE","FINGER"}.issubset(set(c[te])):
            continue
        if task=="lp" and not (
                {"SOFT_LONG_PRESS","MID_LONG_PRESS"}.issubset(set(c[te]))):
            continue
        if set(g[tr]) & set(g[te]):
            raise AssertionError("Subject overlap in holdout")
        return tr,te
    raise RuntimeError(f"No usable person-disjoint {task} split, inspect category coverage")


def weights_for_task(data, indices, task, positive_weight):
    """File-normalized class balancing, with a REAL tunable positive weight.

    In the old v6 helper positive_weight cancels during per-class
    renormalization. Here it is applied only AFTER balancing. For LP,
    release examples inside actual hold recordings receive half the total
    release-class weight, rather than being swamped by many TAP/SWIPE
    and ordinary non-gesture files.
    """
    y = data["y"][indices]
    w = data["w"][indices].astype(np.float32).copy()
    file_counts = Counter(data["file"][indices])
    w /= np.asarray([math.sqrt(file_counts[f]) for f in data["file"][indices]],
                    dtype=np.float32)
    for cls in (0, 1):
        take = y == cls
        total = float(w[take].sum())
        if total <= 0:
            raise ValueError(f"Missing class {cls} in train or evaluation weights")
        w[take] *= len(indices) / (2.0 * total)
    if task == "lp":
        is_hold_folder = np.isin(data["category"][indices],list(HOLD_FOLDERS))
        inside = (y == 1) & is_hold_folder
        outside = (y == 1) & ~is_hold_folder
        sw = float(w[inside].sum())
        ow = float(w[outside].sum())
        if sw > 0 and ow > 0:
            total = sw + ow
            w[inside] *= (0.5 * total) / sw
            w[outside] *= (0.5 * total) / ow
    w[y==1] *= float(positive_weight)
    return w.astype(np.float32)


def metrics(task, y, p, weights=None):
    m = v6.metrics(y,p,FIXED_THRESHOLD,weights)
    if task == "tap":
        return dict(m, objective_score=m["f1"])
    # LP must avoid premature release AND must eventually release. A simple
    # weighted arithmetic mean rewarded the degenerate never-release model.
    y=np.asarray(y,int);p=np.asarray(p,float)
    w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    keep_hold=y==0; keep_release=y==1
    hs=float(np.average(p[keep_hold]<FIXED_THRESHOLD,weights=w[keep_hold]))
    rr=float(np.average(p[keep_release]>=FIXED_THRESHOLD,weights=w[keep_release]))
    score=(1.0/(0.70/hs+0.30/rr)) if hs>0 and rr>0 else 0.0
    return dict(m, hold_safety=hs, release_recall=rr,
                objective_score=score)


def build_params(trial):
    result=v6.model_params(trial,SEED)
    result["n_jobs"]=CPU_THREADS
    return result


def objective(trial,data,train_ids,task):
    name=trial.suggest_categorical("feature_set",list(data["feature_sets"]))
    selected=[data["columns"].index(x) for x in data["feature_sets"][name]]
    positive_weight=trial.suggest_float("positive_weight",1.0,18.0,log=True) if task=="lp" else trial.suggest_float("positive_weight",.75,3.0,log=True)
    params=build_params(trial)
    out_idx=[];out_p=[];max_size=0;fold_records=[]
    for fold,(a,b) in enumerate(GroupKFold(n_splits=CV_FOLDS).split(
            train_ids, data["y"][train_ids], data["group"][train_ids])):
        tr=train_ids[a];val=train_ids[b]
        if set(np.unique(data["y"][tr])) != {0,1} or set(np.unique(data["y"][val])) != {0,1}:
            raise optuna.TrialPruned("Missing class in person-grouped fold")
        model=v6.fit_one(data["X"][tr][:,selected],data["y"][tr],
                         weights_for_task(data,tr,task,positive_weight),params)
        p=model.predict_proba(data["X"][val][:,selected])[:,1]
        size=int(export_plus._estimate_compact_tree_payload_legacy(model)
                 ["total_static_data_bytes"])
        fold_records.append(dict(fold=fold, holdout_people=len(set(data["group"][val])),
                                 metrics=metrics(task,data["y"][val],p,
                                     weights_for_task(data,val,task,1.0)),bytes=size))
        max_size=max(max_size,size)
        out_idx.extend(val.tolist());out_p.extend(p.tolist())
    vi=np.asarray(out_idx,dtype=np.int64)
    report=metrics(task,data["y"][vi],np.asarray(out_p),
                   weights_for_task(data,vi,task,1.0))
    trial.set_user_attr("training_params",params)
    trial.set_user_attr("selected_features",list(data["feature_sets"][name]))
    trial.set_user_attr("folds",fold_records)
    trial.set_user_attr("cv",report)
    trial.set_user_attr("fixed_decision_threshold",FIXED_THRESHOLD)
    trial.set_user_attr("positive_semantics","RELEASE" if task=="lp" else "TAP_OR_SWIPE_ONSET")
    return float(report["objective_score"]),max_size


def choose_trials(study, data):
    picked=v6.pareto_select(study,TOP_K_PER_MODEL)
    full=[t for t in study.trials
          if t.state==optuna.trial.TrialState.COMPLETE and t.values is not None
          and t.params.get("feature_set")=="all_available"]
    if not full:raise ValueError("No all-available-feature model trial completed")
    best=max(full,key=lambda t:float(t.values[0]))
    if best.number not in {t.number for t in picked}:
        picked[-1]=best
    return picked


def uppercase_alias(header: Path, stem: str) -> None:
    source=header.read_text(encoding="utf-8")
    guard=f"{stem}_MODEL_H"
    final=f"#endif /* {guard} */"
    native=stem.lower()+"_predict"
    if final not in source or f"{native}(const t_core*" not in source:
        raise ValueError(f"Header {header} lacks native prediction function or expected guard")
    if "V6_RANK" in source or "v6_rank" in source:
        raise ValueError("Trial ID leaked into C firmware API")
    # Use the value API for firmware 10-prediction probability smoothing.
    # For LGBM the exporter returns a [0, 1] Q20-quantized float (0.05 steps).
    aliases = {
        f"{stem}_predict": native,
        f"{stem}_predict_value": stem.lower()+"_predict_value",
        f"{stem}_probability_q20": stem.lower()+"_probability_q20",
    }
    for public, private in aliases.items():
        if f"{private}(" not in source:
            raise ValueError(f"Missing native C prediction API {private} in {header}")
        if f"#define {public} " not in source:
            source=source.replace(final,
                f"#ifndef {public}\n#define {public} {private}\n#endif\n\n"+final)
    header.write_text(source,encoding="utf-8",newline="\n")


def check_header(header: Path, stem: str, output: Path) -> dict:
    uppercase_alias(header,stem)
    check=previous_tap.compile_header_126(header,stem.lower(),output)
    if check["status"] != "passed":
        raise RuntimeError(f"C compiler not available / syntax failed: {check}")
    # Prove the stable probability and Q20 aliases compile as well as the
    # binary prediction API against the exact same synthetic t_core.
    stub=output/"c_compiler_check"
    src=stub/(stem.lower()+"_value_api.c")
    src.write_text('#include "'+header.name+'"\n'
        'float model_value_api(const t_core *p_core) { return '
        +stem+'_predict_value(p_core) + '
        +stem+'_probability_q20(p_core)*0.05f; }\n')
    cc=shutil.which("cc") or shutil.which("gcc")
    command=[cc,"-std=c11","-O2","-c",str(src),"-I",str(stub),
             "-I",str(header.parent),"-o",str(stub/(stem.lower()+"_value_api.o"))]
    result=subprocess.run(command,text=True,capture_output=True,timeout=90)
    if result.returncode:
        raise RuntimeError("Stable probability C API failed to compile: "+result.stderr[-2600:])
    check["stable_probability_api"]="passed"
    return check


def export_one(data, task, stem, tr,te, selected, outdir):
    model_dir=outdir/"models"
    model_dir.mkdir(parents=True,exist_ok=True)
    y=data["y"];cols=data["columns"]
    selection=[]
    for rank,trial in enumerate(selected,1):
        if "decision_threshold" in trial.params:
            raise ValueError("Threshold tuning is forbidden in unified trainer")
        train_params=trial.user_attrs["training_params"]
        subset=trial.user_attrs["selected_features"]
        ix=[cols.index(x) for x in subset]
        pw=float(trial.params["positive_weight"])
        # Honest holdout metrics BEFORE the final full-data refit.
        prefit=v6.fit_one(data["X"][tr][:,ix],y[tr],
                          weights_for_task(data,tr,task,pw),train_params)
        predictions=prefit.predict_proba(data["X"][te][:,ix])[:,1]
        before=metrics(task,y[te],predictions,weights_for_task(data,te,task,1.0))
        before["unweighted"]=metrics(task,y[te],predictions)
        subgroups={}
        for category in sorted(set(data["category"][te])):
            mask=data["category"][te]==category
            yy=y[te][mask]
            if set(np.unique(yy))=={0,1}:
                subgroups[category]=metrics(task,yy,predictions[mask])
            else:
                subgroups[category]=dict(samples=len(yy),class_one=int(yy.sum()),
                                         trigger_rate=float((predictions[mask]>=FIXED_THRESHOLD).mean()))
        before["per_folder"]=subgroups
        # Freeze one model per rank; all candidates have identical firmware API.
        full_idx=np.arange(len(y))
        final=v6.fit_one(data["X"][:,ix],y,
                         weights_for_task(data,full_idx,task,pw),train_params)
        candidate=model_dir/"candidates"/f"rank_{rank:02d}"
        candidate.mkdir(parents=True,exist_ok=True)
        schema=candidate/(stem+"_feature_schema.json")
        write_json(schema,{"feature_map":{str(k):name for k,name in enumerate(subset)},
                           "positive_class_id":1,
                           "positive_semantics":("RELEASE/REST" if task=="lp" else "TAP/SWIPE ONSET"),
                           "source":"Claude unified labels, original p_core fields only"})
        info=dict(group_name=stem,selected_indices=list(range(len(subset))),
                  decision_threshold=FIXED_THRESHOLD,target_class_id=1,
                  preprocessing="none",params=train_params)
        header=candidate/(stem+".h")
        saved=export_plus.export_to_c_header(
            dict(model=final,preprocessing="none",indices=list(range(len(subset)))),
            info,header,schema,validate=True)
        checked=check_header(header,stem,outdir)
        size=int(saved["total_static_data_bytes"])
        estimate=int(export_plus._estimate_compact_tree_payload_legacy(final)
                     ["total_static_data_bytes"])
        if size != estimate:raise ValueError(f"Exported size differs: {size}!={estimate}")
        filename=f"rank{rank:02d}_trial{trial.number:03d}"
        joblib.dump(dict(model=final,preprocessing="none",indices=list(range(len(subset))),
                         feature_names=subset),
                    model_dir/(filename+".pkl"),compress=3)
        model_report=dict(rank=rank,trial=trial.number,task=task,positive_class_id=1,
                          class_one=("RELEASE/REST" if task=="lp" else "TAP/SWIPE ONSET"),
                          fixed_threshold=FIXED_THRESHOLD,
                          optuna=dict(score=trial.values[0],static_bytes=trial.values[1],
                                      params=trial.params),
                          cv=trial.user_attrs["cv"],holdout=before,
                          features=subset,estimated_static_bytes=estimate,
                          static_bytes=size,header=str(header.relative_to(model_dir)),
                          c_stub=checked,source_pickle=filename+".pkl",
                          limitations=["Teacher-derived silver labels, not physical event truth",
                          "C syntactic smoke uses synthetic t_core, not nRF52832 firmware",
                          "LP class 1 is RELEASE; source hold_label==1 was inverted"])
        write_json(model_dir/(filename+".json"),model_report)
        selection.append(dict(rank=rank,trial=trial.number,score=float(trial.values[0]),
                              cv_f1=float(trial.user_attrs["cv"]["f1"]),
                              holdout_score=float(before["objective_score"]),
                              holdout_f1=float(before["f1"]),
                              bytes=size,features=len(subset),
                              header=str(header.relative_to(model_dir))))
        notice("UNIFIED_MODEL_EXPORT",json.dumps(dict(task=task,**selection[-1])))
    return selection


def default_candidate(ranks):
    best=max(float(x["score"]) for x in ranks)
    keep=[x for x in ranks if x["score"]>=best-.015]
    return min(keep,key=lambda x:(x["bytes"],-x["score"],x["rank"]))


def promote_default(output: Path, stem: str, ranks):
    default=default_candidate(ranks)
    folder=output/"models"
    src=folder/default["header"]
    for suffix in (".h","_feature_schema.json"):
        from_file=src if suffix==".h" else src.parent/(stem+suffix)
        shutil.copyfile(from_file,folder/(stem+suffix))
    write_json(folder/(stem+"_model_info.json"),
               dict(selected=default,fixed_threshold=FIXED_THRESHOLD,
                    stable_header=stem+".h",api=stem+"_predict(&m_feat_core)",
                    native_api=stem.lower()+"_predict(&m_feat_core)",
                    all_candidates=ranks,
                    rationale="Smallest static C model within 0.015 grouped-CV objective of best"))
    return default


def train_task(task: str, tap_files, lp_files):
    definition=next(v for v in TASKS if v["name"]==task)
    out=OUTPUT/task
    summary_file=out/"summary.json"
    if summary_file.is_file():
        saved=json.loads(summary_file.read_text())
        if (saved.get("status")=="success" and saved.get("fixed_threshold")==FIXED_THRESHOLD
                and saved.get("class_one")==definition["class_one"]
                and (out/"models"/(definition["header"]+".h")).is_file()):
            notice("UNIFIED_COMPLETE_REUSED",task,str(out))
            return saved
        raise ValueError("Existing output is not matching completed study: "+str(out))
    out.mkdir(parents=True,exist_ok=True)
    data,manifest=dataset(task,selected_sources(task,tap_files,lp_files),out)
    tr,te=split_heldout(data,task)
    notice("UNIFIED_TRAIN_BEGIN",json.dumps({
        "task":task,"subjects":len(set(data["group"])),
        "sampled_rows":len(data["y"]),"train_rows":len(tr),"holdout_rows":len(te),
        "model_features":len(data["columns"]),"class_one":definition["class_one"],
        "class_zero":definition["class_zero"],"fixed_threshold":FIXED_THRESHOLD}))
    split_info=dict(train_groups=sorted(set(data["group"][tr])),
                    heldout_groups=sorted(set(data["group"][te])),
                    split_seed=SEED+int(task=="lp"))
    write_json(out/"split.json",split_info)
    study=optuna.create_study(
        directions=["maximize","minimize"],
        sampler=optuna.samplers.TPESampler(seed=SEED+int(task=="lp")),
        storage="sqlite:///"+str((out/"optuna_study.sqlite3").resolve()),
        study_name="unified_claude_"+task,load_if_exists=True)
    if any("decision_threshold" in t.params for t in study.trials):
        raise ValueError("Found old threshold-tuned trial: refuse reuse")
    n_left=max(0,TRIALS_PER_MODEL-len(study.trials))
    if len(study.trials)==0:
        if task == "lp":
            # These are model TRAINING class-weight hyperparameters, not
            # decision thresholds. A strict .85 cutoff needs calibrated,
            # high-confidence release predictions to be useful.
            study.enqueue_trial(dict(feature_set="all_available",positive_weight=6.0))
            if "emg_wl54" in data["feature_sets"]:
                study.enqueue_trial(dict(feature_set="emg_wl54",positive_weight=6.0))
        else:
            study.enqueue_trial(dict(feature_set="all_available"))
            if "emg_wl54" in data["feature_sets"]:
                study.enqueue_trial(dict(feature_set="emg_wl54"))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for _ in range(n_left):
        def fn(trial):
            z=objective(trial,data,tr,task)
            notice("UNIFIED_TRIAL",json.dumps({"task":task,"trial":trial.number,
                "score":z[0],"bytes":z[1],"feature_set":trial.params["feature_set"]}))
            return z
        study.optimize(fn,n_trials=1,n_jobs=1,show_progress_bar=False)
    done=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE
          and t.values is not None]
    if len(done)<TOP_K_PER_MODEL:
        raise RuntimeError(f"{task}: {len(done)} completed trials, need {TOP_K_PER_MODEL}")
    picked=choose_trials(study,data)
    ranked=export_one(data,task,definition["header"],tr,te,picked,out)
    best=promote_default(out,definition["header"],ranked)
    manifest_summary=dict(status="success",task=task,model_header=definition["header"]+".h",
                          class_one=definition["class_one"],class_zero=definition["class_zero"],
                          fixed_threshold=FIXED_THRESHOLD,threshold_tuned=False,
                          selected=best,ranking=ranked,trials=len(done),
                          dataset=dict(rows=manifest["n_samples"],
                                       files=manifest["n_sources"],
                                       subjects=manifest["subjects"]),
                          cv_objective=definition["metric"],
                          note="No production firmware overwrite; silver labels; LP class 1 is RELEASE.")
    write_json(out/"summary.json",manifest_summary)
    notice("UNIFIED_TRAIN_END",json.dumps({
        "task":task,"selected":best,
        "firmware_header":str(out/"models"/(definition["header"]+".h")),
        "class_one":definition["class_one"]}))
    return manifest_summary


def main():
    t0=time.monotonic()
    notice("UNIFIED_TRAINER_START","cpu_threads",CPU_THREADS,
           "fixed_threshold",FIXED_THRESHOLD,
           "features","only p_core->, up to 126",
           "labels","Claude unified y_true")
    files_t=ensure_labels(SOURCE_TAP,LABEL_TAP)
    files_l=ensure_labels(SOURCE_LP,LABEL_LP)
    a=train_task("tap",files_t,files_l)
    b=train_task("lp",files_t,files_l)
    write_json(OUTPUT/"UNIFIED_SUMMARY.json",
               dict(status="success",tap=a,lp=b,elapsed_s=round(time.monotonic()-t0,1),
                    baseline_unchanged=True))
    notice("UNIFIED_BOTH_READY",str(OUTPUT/"tap"/"models"/"TAP_SWIPE.h"),
           str(OUTPUT/"lp"/"models"/"LONG_PRESS.h"))


if __name__ == "__main__":
    main()
