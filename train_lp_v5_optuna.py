#!/usr/bin/env python3
"""Train one tiny release/rest LightGBM for nRF52832 from V5 pseudo-labels.

Optuna: 2 objectives (maximize participant-held-out teacher-imitation score,
minimize EXACT export_plus.py generated static data bytes). All Optuna trials
execute; no early-stopping or hand-picked "good" results. Fits one final
full-dataset model and validates C header parity with export_plus.py.

Only raw, current firmware t_core fields are student inputs. There are no
student-side log, EMA, scaler, ratios, future data, folder or pointer features.

This cannot prove physical press/release accuracy. v20 remains untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GroupKFold

from export_plus import (
    estimate_compact_tree_payload, export_to_c_header,
    validate_feature_schema_accessors
)
from lp_v3_labeler import EXTRA_READ, normalize_person, train_contact_quality
from lp_v5_labeler import INPUT_ROOT, LP_GROUPS, GROUPS, infer_quality
from lp_state_v5 import BIN_COUNTS, VERSION as TEACHER_VERSION, label_recording

# The precise CSV keys are the precise, UNMODIFIED t_core expressions emitted by
# export_plus.py. Do NOT train on log1p(WL), rolling EMA, or host-only features.
CORE18 = (
    [f"p_core->wl[CH{ch}]" for ch in range(1, 4)]
    + [f"p_core->emg_psd_mean[CH{ch}][{z}]" for ch in range(1, 4) for z in range(5)]
)
INSTANT15 = [
    f"p_core->emg_psd[0][CH{ch}][{z}]" for ch in range(1, 4) for z in range(5)
]
WL_HALVES6 = (
    [f"p_core->wl_head[CH{ch}]" for ch in range(1, 4)]
    + [f"p_core->wl_tail[CH{ch}]" for ch in range(1, 4)]
)
FEATURES = CORE18 + INSTANT15 + WL_HALVES6
SETS = {"core18": list(range(18)), "instant33": list(range(33)),
        "full39": list(range(39))}
assert len(FEATURES) == 39 and len(set(FEATURES)) == 39
LABEL_CONTRACT = {"1": "REST/RELEASE positive", "0": "ACTIVE/HOLD",
                  "-1": "UNKNOWN/exclude; never train as zero"}
FIRMWARE_GROUP = "lp_v5"
OUT_DEFAULT = Path("LP_V5_EMBEDDED_OPTUNA")


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def raw_features(df: pd.DataFrame) -> np.ndarray:
    """Use only fields that export_plus can access directly in C.

    Historical CSVs marked psd_mean_applied=1 divided already-LOG-coded
    per-zone values. Undo the old conversion in training only. On-device
    firmware data must be in the original log-zone representation.
    """
    absent = [k for k in FEATURES if k not in df.columns]
    if absent:
        raise KeyError(f"Missing required firmware feature(s): {absent[:4]}")
    arr = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if "psd_mean_applied" in df:
        marker = pd.to_numeric(df["psd_mean_applied"], errors="coerce")
        if marker.eq(1).all():
            factor = np.tile(BIN_COUNTS, 3).astype(np.float32)
            arr[:, 3:18] *= factor[None, :]
            arr[:, 18:33] *= factor[None, :]
        elif not marker.eq(0).all():
            raise ValueError("Mixed/invalid PSD legacy-encoding marker")
    return arr


def stratified_sample(y: np.ndarray, weight: np.ndarray, cap: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Sample accepted rows only; keep both hold and release evidence."""
    eligible = (y >= 0) & (weight > 0)
    all_ids = np.flatnonzero(eligible)
    if len(all_ids) <= cap:
        return all_ids
    by_class = [all_ids[y[all_ids] == cls] for cls in (0, 1)]
    if not len(by_class[0]) or not len(by_class[1]):
        return np.sort(rng.choice(all_ids, cap, replace=False))
    # Do not erase short rest intervals in long gesture files; use a balanced
    # per-recording sample. Class and recording weights are reapplied at fit.
    count0 = min(len(by_class[0]), cap // 2)
    count1 = min(len(by_class[1]), cap - count0)
    remaining = cap - count0 - count1
    if remaining:
        extra0 = min(remaining, len(by_class[0]) - count0)
        count0 += extra0
        remaining -= extra0
        count1 += min(remaining, len(by_class[1]) - count1)
    out = np.concatenate([
        rng.choice(by_class[0], count0, replace=False),
        rng.choice(by_class[1], count1, replace=False),
    ])
    return np.sort(out)


def load_training_data(root: Path, max_rows_per_file: int, seed: int,
                       max_files_per_group: int = 0):
    """In-memory dataset; V5 teacher computed on full recordings before sampling."""
    root = root.resolve()
    if root != INPUT_ROOT.resolve():
        raise ValueError("Only DATA/DATA_LONG_PRESS is allowed as source")
    rng = np.random.default_rng(seed)
    print("CONTACT_MODEL: cross-fitting weak per-channel quality, by person", flush=True)
    contact, by_person, known, _ref, quality_eval = train_contact_quality(root, .7)
    collected_x, collected_y, collected_w = [], [], []
    people, source_groups, paths = [], [], []
    file_audit, skipped, all_source = [], Counter(), []
    start = time.monotonic()
    for group in GROUPS:
        sources = sorted((root / group).glob("*.csv.gz"))
        if max_files_per_group:
            sources = sources[:max_files_per_group]
        for src in sources:
            if root not in src.resolve().parents:
                raise ValueError("Input path escapes the required source root")
            all_source.append((src.relative_to(root).as_posix(), src.stat().st_size))
            try:
                df = pd.read_csv(
                    src, compression="gzip", usecols=lambda name: name in EXTRA_READ,
                    low_memory=False
                )
                bad, method = infer_quality(src, group, df, by_person, contact, known)
                labels, teacher_report = label_recording(df, bad, group=group)
                y = labels.lp_state.to_numpy(np.int8)
                w = labels.lp_state_weight.to_numpy(np.float32)
                if np.any(w[y < 0] > 0):
                    raise ValueError("Unknown V5 labels must have zero sample weight")
                if not set(np.unique(y)).issubset({-1, 0, 1}):
                    raise ValueError("Unexpected teacher label value")
                chosen = stratified_sample(y, w, max_rows_per_file, rng)
                person = normalize_person(
                    df["User"].iloc[0] if "User" in df else None, src.name
                )
                file_audit.append({
                    "path": src.relative_to(root).as_posix(), "person": person,
                    "quality_source": method,
                    "rows": len(df), "accepted": int(np.sum(y >= 0)),
                    "sampled": len(chosen), "class0": int(np.sum(y[chosen] == 0)),
                    "class1": int(np.sum(y[chosen] == 1)),
                    "teacher_reason": teacher_report["reason"],
                })
                if len(chosen) == 0:
                    skipped["all_unknown"] += 1
                    continue
                x = raw_features(df.iloc[chosen])
                finite = np.all(np.isfinite(x), axis=1)
                skipped["non_finite_selected"] += int(np.sum(~finite))
                if not np.any(finite):
                    skipped["all_nonfinite"] += 1
                    continue
                x, chosen = x[finite], chosen[finite]
                collected_x.append(x)
                collected_y.append(y[chosen])
                collected_w.append(w[chosen])
                people.extend([person] * len(chosen))
                source_groups.extend([group] * len(chosen))
                paths.extend([str(src)] * len(chosen))
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                # Fail rather than silently claim the corpus was processed.
                raise RuntimeError(f"V5 training input failed: {src}: {exc}") from exc
        print("PREPARED", group, "files", len(sources), "elapsed_s",
              round(time.monotonic() - start, 1), flush=True)
    if not collected_x:
        raise RuntimeError("No valid V5 training labels from the source corpus")
    x = np.concatenate(collected_x).astype(np.float32, copy=False)
    y = np.concatenate(collected_y).astype(np.int8, copy=False)
    weight = np.concatenate(collected_w).astype(np.float32, copy=False)
    uid = np.asarray(people, dtype=object)
    group = np.asarray(source_groups, dtype=object)
    source_paths = np.asarray(paths, dtype=object)
    if len(np.unique(uid)) < 3 or len(np.unique(y)) != 2:
        raise RuntimeError("Insufficient subjects or classes for person-held-out CV")
    if not np.isfinite(x).all() or not np.isfinite(weight).all():
        raise RuntimeError("Training contains invalid data")
    # Source manifest is names and sizes; individual source files remain read-only.
    digest = hashlib.sha256(json.dumps(all_source, sort_keys=True).encode()).hexdigest()
    report = {
        "source_root": str(root),
        "source_inventory_sha256": digest,
        "teacher_version": TEACHER_VERSION,
        "file_count": len(file_audit),
        "sampled_rows": len(y),
        "sampled_hold": int(np.sum(y == 0)),
        "sampled_rest": int(np.sum(y == 1)),
        "participants": int(len(np.unique(uid))),
        "groups": dict(Counter(group)),
        "skipped": dict(skipped),
        "contact_model_proxy_CV": quality_eval,
        "files": file_audit,
    }
    return (x, y, weight, uid, group, source_paths), report


def weighted_fit_weights(y: np.ndarray, source_w: np.ndarray) -> np.ndarray:
    """Balanced class weights retaining the V5 confidence of each row."""
    base = source_w.astype(np.float64, copy=True)
    counts = np.bincount(y.astype(np.int32), minlength=2)
    if not np.all(counts > 0):
        raise ValueError("Training fold has a missing label class")
    base[y == 0] *= len(y) / (2 * counts[0])
    base[y == 1] *= len(y) / (2 * counts[1])
    return base.astype(np.float32)


def model_params(params: dict, workers: int, seed: int) -> dict:
    return {
        "objective": "binary", "boosting_type": "gbdt",
        "n_estimators": int(params["trees"]),
        "num_leaves": int(params["leaves"]),
        "max_depth": int(params["depth"]),
        "min_child_samples": int(params["min_child"]),
        "learning_rate": float(params["lr"]),
        "colsample_bytree": float(params["colsample"]),
        "subsample": float(params["subsample"]),
        "subsample_freq": 1,
        "reg_lambda": float(params["l2"]),
        "max_bin": 127,
        "verbosity": -1, "n_jobs": workers,
        "force_col_wise": True, "random_state": seed,
    }


def score_predictions(y_true: np.ndarray, probability: np.ndarray,
                      threshold: float) -> dict:
    pred = (probability >= threshold).astype(np.int8)
    hold_recall = float(recall_score(y_true, pred, pos_label=0, zero_division=0))
    rest_recall = float(recall_score(y_true, pred, pos_label=1, zero_division=0))
    rest_f1 = float(f1_score(y_true, pred, pos_label=1, zero_division=0))
    balanced = float(balanced_accuracy_score(y_true, pred))
    # During a tap-initiated LP, a false REST can prematurely release.
    # Hold safety gets extra weight, but actual eventual release still matters.
    score = .45 * balanced + .35 * hold_recall + .20 * rest_f1
    return {"score": score, "balanced_accuracy": balanced,
            "hold_recall": hold_recall, "rest_recall": rest_recall,
            "rest_f1": rest_f1,
            "false_release_while_hold": 1.0 - hold_recall,
            "precision_rest": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
            "confusion_0_hold_1_rest": confusion_matrix(y_true, pred, labels=[0, 1]).tolist()}


def grouped_cv(x, y, weight, uid, folds):
    if folds < 2 or len(np.unique(uid)) < folds:
        raise ValueError("Not enough subjects for requested GroupKFold")
    splits = list(GroupKFold(n_splits=folds).split(x, y, groups=uid))
    for fi, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[te])) != 2:
            raise ValueError(f"GroupKFold fold {fi} has only one class")
    return splits


def trial_parameters(trial: optuna.Trial, max_trees: int) -> dict:
    # Prefix/size trade-off is a true Optuna search dimension.
    choices = list(range(24, max_trees + 1, 8))
    if not choices:
        raise ValueError("max_trees must be >=24")
    return {
        "feature_set": trial.suggest_categorical("feature_set", list(SETS)),
        "trees": trial.suggest_categorical("trees", choices),
        "leaves": trial.suggest_int("leaves", 4, 20, step=2),
        "depth": trial.suggest_int("depth", 2, 6),
        "min_child": trial.suggest_int("min_child", 30, 170, step=20),
        "lr": trial.suggest_float("lr", .035, .17, log=True),
        "colsample": trial.suggest_float("colsample", .65, 1.0),
        "subsample": trial.suggest_float("subsample", .7, 1.0),
        "l2": trial.suggest_float("l2", .1, 20., log=True),
        # Q20 thresholds are exact in the generated LightGBM predict_q20 API.
        "threshold": trial.suggest_categorical(
            "threshold", [.40, .45, .50, .55, .60, .65, .70, .75, .80, .85]
        ),
    }


def objective_builder(ds, splits, workers, seed, max_trees):
    x, y, weights, _uid, _group, _paths = ds
    def objective(trial):
        p = trial_parameters(trial, max_trees)
        chosen = SETS[p["feature_set"]]
        # Memory reduction: slice one shared compact X, not all three sets.
        selected_x = np.ascontiguousarray(x[:, chosen], dtype=np.float32)
        scores, sizes, metrics = [], [], []
        for fi, (tr, te) in enumerate(splits):
            m = lgb.LGBMClassifier(**model_params(p, workers, seed + fi))
            m.fit(selected_x[tr], y[tr],
                  sample_weight=weighted_fit_weights(y[tr], weights[tr]))
            probability = m.predict_proba(selected_x[te])[:, 1]
            metric = score_predictions(y[te], probability, float(p["threshold"]))
            size = estimate_compact_tree_payload(m)["total_static_data_bytes"]
            scores.append(metric["score"])
            sizes.append(int(size))
            metrics.append({"fold": fi, "test_rows": int(len(te)),
                            "static_bytes": int(size), **metric})
        # Multiobjective Pareto: *both* score and exact embedded array bytes.
        # Penalize unstable folds lightly so one lucky participant split loses.
        avg_score = float(np.mean(scores) - .08 * np.std(scores))
        avg_size = float(np.mean(sizes))
        trial.set_user_attr("folds", metrics)
        trial.set_user_attr("score_avg", float(np.mean(scores)))
        trial.set_user_attr("score_std", float(np.std(scores)))
        trial.set_user_attr("static_bytes_max_fold", int(max(sizes)))
        return avg_score, avg_size
    return objective


def rank_candidates(study: optuna.Study, max_static_bytes: int) -> list:
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and
                t.values is not None and all(math.isfinite(v) for v in t.values)]
    feasible = [t for t in complete if t.values[1] <= max_static_bytes]
    # A small size penalty chooses a compact Pareto knee over a <1 point
    # gain requiring substantially more Flash, while remaining under budget.
    return sorted(feasible, key=lambda t: (
        -(t.values[0] - .02 * t.values[1] / max_static_bytes),
        t.values[1], t.number,
    ))


def emit_export(out: Path, model, params: dict, threshold: float,
                selected: list, budget: int) -> dict:
    schema_path = out / "lp_v5_feature_schema.json"
    schema = {"schema_name": "lp_v5_direct_fw_features",
              "feature_map": {str(i): name for i, name in enumerate(FEATURES)},
              "selected_indices": selected,
              "available_sets": SETS,
              "note": "Raw firmware WL and ALREADY LOG-ENCODED PSD zones; no log/EMA preprocessing."}
    validate_feature_schema_accessors([FEATURES[k] for k in selected])
    schema_path.write_text(json.dumps(schema, indent=2))
    model_path = out / "lp_v5_model.pkl"
    info_path = out / "lp_v5_model_info.json"
    header_path = out / "lp_v5_model.h"
    container = {"model": model, "indices": selected, "preprocessing": "none",
                 "decision_threshold": threshold}
    info = {
        "group_name": FIRMWARE_GROUP, "target_class_id": 1,
        "task": "rest/release-positive binary classifier",
        "selected_indices": selected, "preprocessing": "none",
        "decision_threshold": threshold,
        "threshold_q20": int(round(threshold * 20)),
        "positive_label": "lp_state=1 means resting/released",
        "params": params,
    }
    joblib.dump(container, model_path, compress=3)
    info_path.write_text(json.dumps(info, indent=2))
    # Exact validation of generated firmware tree inference vs Python model.
    report = export_to_c_header(
        model=container, trial_info=info, output_path=header_path,
        feature_schema_path=schema_path, validate=True
    )
    if report["total_static_data_bytes"] > budget:
        raise ValueError("Final C header exceeds requested static Flash budget")
    if report.get("validation", {}).get("decision_disagreements", 0):
        raise RuntimeError("Python vs exported C tree prediction disagreement")
    return {"model_file": str(model_path), "header_file": str(header_path),
            "info_file": str(info_path), "schema_file": str(schema_path),
            "model_file_sha256": hash_file(model_path),
            "header_sha256": hash_file(header_path),
            **report}


def compile_stub(header: Path) -> dict:
    """Host C type-check of generated header. Not a real nRF firmware build."""
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        return {"compiled": False, "reason": "No host GCC/CC found"}
    with tempfile.TemporaryDirectory(prefix="lp-v5-c-check-") as scratch:
        work = Path(scratch)
        shutil.copy2(header, work / "lp_v5_model.h")
        (work / "core_type_def.h").write_text(
            "#ifndef LP_TEST_CORE_TYPE_DEF_H\n#define LP_TEST_CORE_TYPE_DEF_H\n"
            "#include <stdint.h>\n"
            "#define CH1 0\n#define CH2 1\n#define CH3 2\n"
            "typedef struct {\n"
            "  uint16_t wl[3];\n"
            "  uint16_t wl_head[3];\n"
            "  uint16_t wl_tail[3];\n"
            "  uint16_t emg_psd_mean[3][5];\n"
            "  uint16_t emg_psd[1][3][5];\n"
            "} t_core;\n#endif\n"
        )
        (work / "check.c").write_text(
            '#include "lp_v5_model.h"\n'
            'int main(void) { t_core c = {0}; '
            'volatile float p = lp_v5_predict_value(&c); '
            'return (p >= 0.0f && p <= 1.0f && lp_v5_predict(&c) >= 0) ? 0 : 1; }\n'
        )
        cmd = [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
               "-O2", "-c", str(work / "check.c"), "-o", str(work / "check.o")]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if p.returncode:
            raise RuntimeError("Generated C header failed host C compilation:\n" +
                               (p.stdout + p.stderr)[-5000:])
        return {"compiled": True, "compiler": compiler, "mode": "host stub types only"}


def post_export_replay(model, selected: list, threshold: float,
                       ds, splits, train_params: dict, root: Path,
                       models, quality_full, known):
    """Held-out full-rate V5 pseudo-event replay, with 10-prob trailing mean.

    This is NOT a physically verified release/hold latency test. It ensures
    whole-timeline behavior is inspected rather than just isolated CV rows.
    """
    x, y, w, uid, groups, paths = ds
    results = []
    paths = np.asarray(paths, dtype=object)
    for fold, (tr, te) in enumerate(splits):
        m = lgb.LGBMClassifier(**model_params(train_params, 2, 740+fold))
        m.fit(x[tr][:, selected], y[tr],
              sample_weight=weighted_fit_weights(y[tr], w[tr]))
        test_files = np.unique(paths[te])
        for path_text in test_files:
            p = Path(path_text)
            group = p.parent.name
            if group not in LP_GROUPS:
                continue
            df = pd.read_csv(p, compression="gzip", usecols=lambda k: k in EXTRA_READ,
                             low_memory=False)
            bad, _ = infer_quality(p, group, df, models, quality_full, known)
            lab, _ = label_recording(df, bad, group=group)
            valid = lab.lp_state.to_numpy(np.int8) >= 0
            if not valid.any():
                continue
            feat = raw_features(df)
            finite = np.all(np.isfinite(feat[:, selected]), axis=1)
            if not finite.all():
                continue
            prob = m.predict_proba(feat[:, selected])[:, 1]
            # The C exporter returns Q20 probability in 0.05 steps; firmware
            # averages the last 10 probability predictions after tap activation.
            q20 = np.clip(np.floor(prob * 20 + .5), 0, 20) / 20.
            rolling = pd.Series(q20).rolling(10, min_periods=10).mean().to_numpy()
            ready = np.isfinite(rolling) & valid
            yp = lab.lp_state.to_numpy(np.int8)
            held = ready & (yp == 0)
            released = ready & (yp == 1)
            results.append({
                "fold": fold, "file": p.name, "group": group,
                "observed_hold_rows": int(held.sum()),
                "observed_rest_rows": int(released.sum()),
                "pseudo_false_rest_during_hold_rate": (
                    float(np.mean(rolling[held] >= threshold)) if held.any() else None
                ),
                "pseudo_release_recognition_rate": (
                    float(np.mean(rolling[released] >= threshold)) if released.any() else None
                ),
            })
        print("REPLAY_FOLD", fold, "LP files", len(test_files), flush=True)
    held_n = sum(r["observed_hold_rows"] for r in results)
    rest_n = sum(r["observed_rest_rows"] for r in results)
    held_error = sum(r["pseudo_false_rest_during_hold_rate"] * r["observed_hold_rows"]
                     for r in results if r["pseudo_false_rest_during_hold_rate"] is not None)
    rest_correct = sum(r["pseudo_release_recognition_rate"] * r["observed_rest_rows"]
                       for r in results if r["pseudo_release_recognition_rate"] is not None)
    return {"warning": "Pseudo-label timing is NOT physical hold/release truth; replay is un-gated",
            "smoothing": "trailing mean of 10 Q20 outputs (0.05 increments)",
            "test_files": len(results), "held_label_rows": held_n,
            "rest_label_rows": rest_n,
            "false_rest_during_pseudo_hold": held_error / held_n if held_n else None,
            "rest_detected_during_pseudo_release": rest_correct / rest_n if rest_n else None,
            "per_file": results}


def run(args) -> dict:
    if args.trials < 1 or args.max_static_bytes < 1024:
        raise ValueError("Trials >=1 and static byte budget >=1024 required")
    if args.max_rows_per_file < 48 or args.workers < 1 or args.max_trees < 24:
        raise ValueError("Require >=48 samples/file, >=1 worker, >=24 max trees")
    out = args.output.resolve()
    if out == INPUT_ROOT.resolve() or INPUT_ROOT.resolve() in out.parents:
        raise ValueError("Output must not be written under source recordings")
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n!.gitignore\n")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    random.seed(args.seed)
    np.random.seed(args.seed)
    start = time.monotonic()
    ds, audit = load_training_data(
        INPUT_ROOT, args.max_rows_per_file, args.seed, args.limit_per_group
    )
    x, y, w, uid, groups, paths = ds
    splits = grouped_cv(x, y, w, uid, args.cv_folds)
    (out / "dataset_audit.json").write_text(json.dumps(audit, indent=2))
    print("TRAINING_DATA", json.dumps({
        "rows": len(y), "participants": len(np.unique(uid)),
        "hold": int(np.sum(y==0)), "rest": int(np.sum(y==1)),
        "features": len(FEATURES), "groups": audit["groups"],
    }), flush=True)
    sampler = optuna.samplers.NSGAIISampler(seed=args.seed, population_size=16)
    study = optuna.create_study(
        directions=["maximize", "minimize"], sampler=sampler,
        study_name="lp_v5_score_vs_exact_fw_static_bytes"
    )
    objective = objective_builder(ds, splits, args.workers, args.seed,
                                  args.max_trees)
    print("OPTUNA_BEGIN", args.trials, "trials, 2 objectives; size limit",
          args.max_static_bytes, "bytes", flush=True)
    for n in range(args.trials):
        study.optimize(objective, n_trials=1, n_jobs=1, catch=())
        trial = study.trials[-1]
        print("TRIAL", n+1, "/", args.trials, "score",
              round(trial.values[0], 5), "static_bytes",
              round(trial.values[1]), "feature_set",
              trial.params["feature_set"], "trees", trial.params["trees"], flush=True)
    study.trials_dataframe().to_csv(out / "optuna_trials.csv", index=False)
    candidates = rank_candidates(study, args.max_static_bytes)
    if not candidates:
        raise RuntimeError("No trial fits the firmware static-data budget; "
                           "increase --max-static-bytes or expand Optuna trials.")
    # Refitting with all participants can change the trained tree shapes;
    # check actual bytes, not just average fold estimates.
    chosen_trial, fitted, chosen_size = None, None, None
    for candidate in candidates:
        p = candidate.params
        indices = SETS[p["feature_set"]]
        model = lgb.LGBMClassifier(**model_params(p, args.workers, args.seed))
        model.fit(x[:, indices], y, sample_weight=weighted_fit_weights(y, w))
        estimated = estimate_compact_tree_payload(model)["total_static_data_bytes"]
        if estimated <= args.max_static_bytes:
            chosen_trial, fitted, chosen_size = candidate, model, estimated
            break
    if chosen_trial is None:
        raise RuntimeError("All feasible CV trials exceed budget when refit on full data")
    params = dict(chosen_trial.params)
    indices = list(SETS[params["feature_set"]])
    threshold = float(params["threshold"])
    result = emit_export(out, fitted, params, threshold, indices, args.max_static_bytes)
    compiler = compile_stub(out / "lp_v5_model.h") if not args.skip_host_compile else {
        "compiled": False, "reason": "Disabled by user"
    }
    replay = None
    if not args.skip_replay and not args.limit_per_group:
        # Compute contact quality only for cross-fitted reference membership;
        # no quality-model outputs ever enter the firmware student.
        quality_full, quality_by_person, known, _ref, _audit = train_contact_quality(INPUT_ROOT, .7)
        replay = post_export_replay(
            fitted, indices, threshold, ds, splits, params, INPUT_ROOT,
            quality_by_person, quality_full, known
        )
        (out / "pseudo_event_replay.json").write_text(json.dumps(replay, indent=2))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "single binary REST/RELEASE model (positive=1) for embedded LP",
        "warning": "Teacher imitation and pseudo-event replay are NOT actual pressed/released truth",
        "baseline": "Existing v20 firmware model is unchanged; must A/B test physically",
        "teacher_version": TEACHER_VERSION,
        "label_contract": LABEL_CONTRACT,
        "optuna_trials_completed": len(study.trials),
        "optuna_objectives": ["maximize: .45*BA + .35*hold_recall + .20*rest_F1 - .08*fold_SD",
                              "minimize: export_plus exact generated static data bytes"],
        "selected_trial": chosen_trial.number,
        "selected_cv_score": chosen_trial.values[0],
        "selected_cv_mean_static_bytes": chosen_trial.values[1],
        "selected_trial_folds": chosen_trial.user_attrs.get("folds"),
        "selected_params": params,
        "selected_firmware_feature_indices": indices,
        "selected_firmware_feature_names": [FEATURES[i] for i in indices],
        "positive_class": 1,
        "decision_threshold": threshold,
        "decision_threshold_q20": int(round(threshold * 20)),
        "firmware_static_byte_budget": args.max_static_bytes,
        "actual_static_bytes": chosen_size,
        "export": result, "host_compiler": compiler,
        "whole_rate_pseudo_event_replay": (
            {k:v for k,v in replay.items() if k != "per_file"} if replay else None
        ),
        "training_samples": len(y), "training_persons": len(np.unique(uid)),
        "training_groups": audit["groups"],
        "original_source_inventory_sha256": audit["source_inventory_sha256"],
        "source_files_hash": {
            "trainer": hash_file(Path(__file__)),
            "teacher": hash_file(Path("lp_state_v5.py")),
            "labeler": hash_file(Path("lp_v5_labeler.py")),
            "export_plus": hash_file(Path("export_plus.py")),
        },
        "elapsed_s": round(time.monotonic() - start, 2)
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2))
    print("RESULT", json.dumps({
        "selected_trial": chosen_trial.number, "score": report["selected_cv_score"],
        "size_bytes": chosen_size, "threshold": threshold,
        "features": len(indices), "trees": params["trees"],
        "c_equivalence": result.get("validation"),
        "host_compilation": compiler, "out": str(out),
        "elapsed_s": report["elapsed_s"]
    }), flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=96, help="Full Optuna multiobjective trials")
    p.add_argument("--max-static-bytes", type=int, default=24576,
                   help="Generated C static array bytes, not entire firmware image")
    p.add_argument("--max-trees", type=int, default=128, help="Tree count search upper bound")
    p.add_argument("--max-rows-per-file", type=int, default=480)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=OUT_DEFAULT)
    p.add_argument("--limit-per-group", type=int, default=0,
                   help="Testing only; 0 means full source corpus")
    p.add_argument("--skip-host-compile", action="store_true")
    p.add_argument("--skip-replay", action="store_true",
                   help="Skip full-rate held-out pseudo-event 10-prediction replay")
    a = p.parse_args(argv)
    run(a)


if __name__ == "__main__":
    main()
