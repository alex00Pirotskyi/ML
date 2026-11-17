#!/usr/bin/env python3
"""Publish stable TAP_SWIPE firmware headers from already trained Claude V2 models.

No dataset loading, Optuna, or retraining. Reads saved per-rank .pkl files and
reports produced by train_claude_v2_optuna.py.

Outputs:
  models/TAP_SWIPE.h                            default balanced candidate
  models/TAP_SWIPE_feature_schema.json          matching CSV -> C feature map
  models/TAP_SWIPE_model_info.json              selected source trial and metrics
  models/TAP_SWIPE_VARIANTS.json                all candidate comparisons
  models/candidates/rank_01/TAP_SWIPE.h         one stable-name model per folder
  .../rank_10/TAP_SWIPE.h

Every generated header has TAP_SWIPE_MODEL_H, TAP_SWIPE_* macros, the normal
tap_swipe_predict(), and the optional TAP_SWIPE_predict() alias. Include only
one candidate at a time, because all candidates intentionally share C symbols.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Callable

import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import export_plus
sys.path.insert(0, str(ROOT / "research" / "tap_real"))
from train_optuna_student_v6 import FIXED_DECISION_THRESHOLD

DEFAULT_OUTPUT = ROOT / "research" / "tap_real" / "results_claude_v2_optuna_126_thr085"
MODEL_NAME = "TAP_SWIPE"
MAX_CV_F1_GAP = 0.015  # smallest model within 1.5 percentage points of best CV F1


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _add_uppercase_predict_alias(header: Path) -> None:
    """Keep export_plus's native function and provide the requested uppercase API."""
    src = header.read_text(encoding="utf-8")
    closing = "#endif /* TAP_SWIPE_MODEL_H */"
    if src.count(closing) != 1:
        raise ValueError(f"Unexpected TAP_SWIPE header guard: {header}")
    if "tap_swipe_predict(const t_core*" not in src:
        raise ValueError(f"Expected tap_swipe_predict API missing in {header}")
    if "#define TAP_SWIPE_predict" not in src:
        src = src.replace(closing,
            "#ifndef TAP_SWIPE_predict\n"
            "#define TAP_SWIPE_predict tap_swipe_predict\n"
            "#endif\n\n" + closing)
        header.write_text(src, encoding="utf-8", newline="\n")
    if "V6_RANK" in src or "v6_rank" in src:
        raise ValueError(f"Legacy trial identifier leaked into C header: {header}")


def _select_default(rows: list[dict]) -> dict:
    """Use training-CV F1, not held-out F1, to select the balanced default."""
    best = max(float(r["cv_f1"]) for r in rows)
    eligible = [r for r in rows if float(r["cv_f1"]) >= best - MAX_CV_F1_GAP]
    return min(eligible,
               key=lambda r: (int(r["actual_static_bytes"]),
                              -float(r["cv_f1"]), int(r["rank"])))


def publish(output: Path = DEFAULT_OUTPUT,
            compile_header: Callable | None = None) -> dict:
    output = Path(output).resolve()
    models = output / "models"
    summary_file = output / "summary.json"
    if not summary_file.is_file():
        raise FileNotFoundError(
            f"Completed Optuna summary missing: {summary_file}. "
            "Finish Claude V2 training before publishing.")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary.get("status") != "success":
        raise ValueError("Cannot publish unsuccessful or incomplete training")
    if (summary.get("fixed_decision_threshold") != FIXED_DECISION_THRESHOLD
        or summary.get("threshold_optimized_by_optuna") is not False):
        raise ValueError("Legacy Optuna study tuned its threshold; do not republish "
                         "its scores as fixed-0.85. Train a new 0.85-fixed study.")
    rows = summary.get("ranking", [])
    if not rows or len({int(x["rank"]) for x in rows}) != len(rows):
        raise ValueError("Missing/ambiguous ranked models in completed summary")
    if len(rows) != int(summary.get("selected_models", -1)):
        raise ValueError("Rank count differs from completed training summary")

    published = []
    rewritten_reports = []
    legacy_headers_to_delete = []
    for row in sorted(rows, key=lambda r: int(r["rank"])):
        rank = int(row["rank"])
        # Saved research model names are independent of the firmware header.
        # An existing experiment might have old rank-named C files, while new
        # experiments already place TAP_SWIPE.h inside candidate folders.
        metadata_tag = Path(row["json"]).stem
        source_model = models / (metadata_tag + ".pkl")
        source_report = models / row["json"]
        if not source_model.is_file() or not source_report.is_file():
            raise FileNotFoundError(f"Missing frozen model/report for rank {rank}")
        details = json.loads(source_report.read_text(encoding="utf-8"))
        if int(details["rank"]) != rank or int(details["optuna_trial"]) != int(row["trial"]):
            raise ValueError(f"Model provenance mismatches Optuna summary at rank {rank}")
        old_name = str(row["header"])
        # Only clean unmodified legacy-generated headers, after all replacement
        # C headers have been regenerated, validated and compiled.
        if re.fullmatch(r"tap_swipe_v6_rank\d{2}_trial\d{3}\.h", old_name):
            old_header = models / old_name
            if old_header.exists():
                expected_sha = details.get("files", {}).get("header_sha256")
                if expected_sha and _sha(old_header) == expected_sha:
                    legacy_headers_to_delete.append((old_header, expected_sha))
                else:
                    print("PRESERVE_EDITED_LEGACY_HEADER", str(old_header), flush=True)
        container = joblib.load(source_model)
        if not isinstance(container, dict) or "model" not in container:
            raise ValueError(f"Invalid saved model container: {source_model}")
        feature_names = list(container.get("feature_names", ()))
        if len(feature_names) != int(row["feature_count"]):
            raise ValueError(f"Feature-count mismatch at rank {rank}")
        if feature_names != details["final_model"]["features"]:
            raise ValueError(f"Feature order mismatch at rank {rank}")
        if container.get("preprocessing") != "none":
            raise ValueError("Unexpected preprocessing in model")
        if list(container.get("indices", ())) != list(range(len(feature_names))):
            raise ValueError(f"Unexpected selected feature indices at rank {rank}")

        candidate = models / "candidates" / f"rank_{rank:02d}"
        candidate.mkdir(parents=True, exist_ok=True)
        schema = candidate / "TAP_SWIPE_feature_schema.json"
        _write_json(schema, {
            "feature_map": {str(i): col for i, col in enumerate(feature_names)},
            "positive_class_id": 1,
            "target_semantics": "TAP or SWIPE gesture onset (binary)",
            "origin": "Claude V2 strong-only onset silver labels",
        })
        header = candidate / "TAP_SWIPE.h"
        if float(details["final_model"]["decision_threshold"]) != FIXED_DECISION_THRESHOLD:
            raise ValueError(f"Rank {rank} was not trained/evaluated at fixed 0.85")
        threshold = FIXED_DECISION_THRESHOLD
        export_report = export_plus.export_to_c_header(
            container,
            {"group_name": MODEL_NAME,
             "selected_indices": list(range(len(feature_names))),
             "decision_threshold": threshold,
             "target_class_id": 1,
             "preprocessing": "none",
             "params": details["final_model"]["params"]},
            header, schema, validate=True)
        _add_uppercase_predict_alias(header)
        static_bytes = int(export_report["total_static_data_bytes"])
        if static_bytes != int(row["actual_static_bytes"]):
            raise ValueError(f"Static model size changed for rank {rank}: "
                             f"old={row['actual_static_bytes']} new={static_bytes}")
        compile_result = (compile_header(header, "tap_swipe", output)
                          if compile_header else {"status": "not_run"})
        if compile_header and compile_result.get("status") != "passed":
            raise ValueError(f"C compile smoke failed for rank {rank}")
        info = {
            "firmware_header": "TAP_SWIPE.h",
            "firmware_api": "TAP_SWIPE_predict(&m_feat_core)",
            "native_api": "tap_swipe_predict(&m_feat_core)",
            "include_guard": "TAP_SWIPE_MODEL_H",
            "feature_schema": schema.name,
            "rank": rank,
            "optuna_trial": int(row["trial"]),
            "source_pickle": str(source_model.relative_to(output)),
            "source_pickle_sha256": _sha(source_model),
            "header_sha256": _sha(header),
            "num_features": len(feature_names),
            "static_model_bytes": static_bytes,
            "decision_threshold": threshold,
            "cv_silver_f1": float(row["cv_f1"]),
            "holdout_silver_f1": float(row["holdout_f1"]),
            "compiled_c_stub": compile_result,
            "evaluation_warning": "F1 is against silver labels, not physical tap ground truth",
        }
        _write_json(candidate / "TAP_SWIPE_model_info.json", info)
        # Update the experiment's output references, NOT trial parameters.
        # Repeated publish() calls stay valid after old .h files are cleaned.
        relative_header = str(header.relative_to(models))
        row["header"] = relative_header
        row["header_text_bytes"] = header.stat().st_size
        details.setdefault("files", {}).update({
            "header": relative_header,
            "header_sha256": _sha(header),
            "header_text_bytes": header.stat().st_size,
            "feature_schema": str(schema.relative_to(models)),
        })
        details["export_plus"] = export_report
        details["compiled_c_stub"] = compile_result
        rewritten_reports.append((source_report, details))
        published.append(info)
        print("STABLE_TAP_SWIPE_EXPORT", json.dumps({
            "rank": rank, "trial": info["optuna_trial"],
            "header": str(header.relative_to(output)),
            "bytes": static_bytes, "cv_f1": info["cv_silver_f1"]}), flush=True)

    selected = _select_default(rows)
    default_rank = int(selected["rank"])
    selected_dir = models / "candidates" / f"rank_{default_rank:02d}"
    for filename in ("TAP_SWIPE.h", "TAP_SWIPE_feature_schema.json",
                     "TAP_SWIPE_model_info.json"):
        shutil.copyfile(selected_dir / filename, models / filename)

    selection_info = {
        "selected_rank": default_rank,
        "selected_optuna_trial": int(selected["trial"]),
        "selection_rule": "Smallest static model within 0.015 absolute grouped-CV silver F1 at fixed threshold 0.85 of best",
        "fixed_decision_threshold": FIXED_DECISION_THRESHOLD,
        "threshold_optimized_by_optuna": False,
        "cv_f1_gap_absolute": MAX_CV_F1_GAP,
        "stable_default_header": "models/TAP_SWIPE.h",
        "stable_default_macro_prefix": "TAP_SWIPE_",
        "call": "TAP_SWIPE_predict(&m_feat_core)",
        "native_call": "tap_swipe_predict(&m_feat_core)",
        "candidates": published,
        "note": "Candidate headers intentionally have identical C APIs: use one header at a time",
    }
    _write_json(models / "TAP_SWIPE_VARIANTS.json", selection_info)
    for source_report, details in rewritten_reports:
        _write_json(source_report, details)
    _write_json(summary_file, summary)
    removed = 0
    for old_header, old_sha in legacy_headers_to_delete:
        if old_header.is_file() and _sha(old_header) == old_sha:
            old_header.unlink()
            removed += 1
    print("STABLE_TAP_SWIPE_LEGACY_HEADERS_CLEANED",
          json.dumps({"removed": removed, "checked": len(legacy_headers_to_delete)}),
          flush=True)
    print("STABLE_TAP_SWIPE_READY", json.dumps({
        "default_rank": default_rank,
        "default_header": str((models / "TAP_SWIPE.h").relative_to(output)),
        "total_variants": len(published),
        "no_retraining": True}), flush=True)
    return selection_info


if __name__ == "__main__":
    # Re-export previously trained models without launching Optuna again.
    from train_claude_v2_optuna import compile_header_126
    publish(DEFAULT_OUTPUT, compile_header=compile_header_126)
