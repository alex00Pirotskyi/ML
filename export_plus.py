#!/usr/bin/env python3
"""Export supported sklearn/LightGBM/XGBoost tree models to a compact C header.

Key properties
--------------
* Binary classifiers and regressors are supported.
* Multi-output RF regressors export continuous soft targets with a compact leaf-only
  side table, checked array-input inference, and an optional t_core wrapper.
* Selected feature indices always refer to the original feature schema.
* No logarithmic preprocessing is accepted.
* Per-feature affine preprocessing is folded into tree thresholds, so
  StandardScaler/MinMaxScaler/MaxAbsScaler/RobustScaler add no runtime FW math.
* Row-wise L1/L2/max normalization is reproduced in C with one normalization
  factor per prediction.
* Cortex-M4F balanced storage uses aligned float32 thresholds plus packed metadata:
  RF is 6/7 bytes per node and small LightGBM trees are 6 bytes per node.
* LightGBM binary probability uses Q20 (0.05 steps) and exact inverse-sigmoid threshold LUTs;
  generated production inference contains no expf() or logf().
* Conservative grouped suffix bounds provide exact early exit for LightGBM decisions.
* Callable feature members such as p_core->diff(...) and p_core->ratio(...) are preserved and
  lazily cached once per prediction to avoid repeated helper work across trees.
* LightGBM ±1e-35 zero canonicalization and missing routing are reproduced.
* Export equivalence is validated against the Python model before the header is written.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import warnings
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableSequence, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
from sklearn.utils.validation import validate_data

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:  # LightGBM is optional for sklearn-only export environments.
    import lightgbm as lgb
except ImportError:  # pragma: no cover - exercised only in minimal deployments.
    lgb = None  # type: ignore[assignment]


SUPPORTED_PREPROCESSING = {
    "none",
    "standard",
    "minmax_0_1",
    "minmax_m1_1",
    "maxabs",
    "robust_25_75",
    "robust_10_90",
    "normalize_l1",
    "normalize_l2",
    "normalize_max",
}

AFFINE_PREPROCESSING = {
    "standard",
    "minmax_0_1",
    "minmax_m1_1",
    "maxabs",
    "robust_25_75",
    "robust_10_90",
}

ROW_NORMALIZATION = {
    "normalize_l1": "l1",
    "normalize_l2": "l2",
    "normalize_max": "max",
}

_PREPROCESSING_ALIASES = {
    "raw": "none",
    "identity": "none",
    "standardization": "standard",
    "standard_scaler": "standard",
    "zscore": "standard",
    "z_score": "standard",
    "minmax": "minmax_0_1",
    "min_max": "minmax_0_1",
    "max_abs": "maxabs",
    "robust": "robust_25_75",
    "l1": "normalize_l1",
    "l2": "normalize_l2",
    "unit_l1": "normalize_l1",
    "unit_l2": "normalize_l2",
    "unit_max": "normalize_max",
}


# =================================================================================================
# EXPORT CONFIGURATION — EDIT GLOBALS HERE, NO COMMAND-LINE ARGUMENTS
# =================================================================================================
# Cortex-M4F balanced profile: exact float32 thresholds, aligned SoA storage, compact packed
# metadata when the tree shape permits it, a 0.05 probability LUT for LightGBM, and exact
# raw-score comparisons for runtime-selectable probability thresholds. The generated binary
# path never calls expf/logf. CMSIS force-inline controls are used when available.
EXPORT_PROFILE = "cortex_m4f_balanced"
FLASH_STORAGE_LAYOUT = "exact_float32_soa_bitpacked_flags"
EXPORT_VALIDATE = True

# LightGBM probability interface. Q20 means one integer step is 0.05 probability.
# Two tiny float tables are emitted: exact threshold boundaries (0.05..0.95) and midpoint
# boundaries (0.025..0.975) for nearest-step probability output. Total cost: 156 bytes.
LIGHTGBM_PROBABILITY_STEP = 0.05
LIGHTGBM_PROBABILITY_LEVELS = 20
LIGHTGBM_ENABLE_Q20_PROBABILITY = True
LIGHTGBM_EARLY_EXIT_GROUP_SIZE = 4

# Preserve helper calls exactly as t_core member calls, e.g. p_core->diff(a, b). This assumes
# t_core exposes callable members with matching semantics. Set to "free_function" only when
# firmware helpers are ordinary functions/macros such as diff(a, b).
FEATURE_HELPER_CALL_STYLE = "p_core_member"  # "p_core_member" or "free_function"

# Used only when running export_plus.py directly. Training calls export_to_c_header() itself.
STANDALONE_MODEL_PATH = "TAP_MODELS_F1_V225_FAST_TARGET_GLOBAL/TAP_lgbm_rank1_20260803_033633_model.pkl"
STANDALONE_INFO_PATH = "TAP_MODELS_F1_V225_FAST_TARGET_GLOBAL/TAP_lgbm_rank1_20260803_033633_info.json"
STANDALONE_SCHEMA_PATH = "TAP_MODELS_F1_V225_FAST_TARGET_GLOBAL/TAP_feature_schema.json"
STANDALONE_OUTPUT_PATH = "gesture_model.h"

RF_U8_NODE_BYTES = 6   # float threshold + packed uint16(feature | right_child << 8)
RF_U16_NODE_BYTES = 7  # float threshold + uint16 right child + uint8 feature
LGBM_PACKED_NODE_BYTES = 6  # float threshold + uint8(child6|flags2) + uint8 feature
LGBM_LARGE_BASE_NODE_BYTES = 6  # float threshold + uint8 right child + uint8 feature metadata
LIGHTGBM_ZERO_THRESHOLD = np.float32(1.0e-35)

# Feature schemas may contain either direct t_core field accessors or firmware-side
# computed expressions. Keep function calls explicit and auditable instead of
# accepting arbitrary C text. Extend this set only for pure, already-available
# firmware helpers/macros whose return value is a float-compatible feature.
ALLOWED_FEATURE_FUNCTIONS = frozenset({
    "diff",
    "ratio",
    "fabsf",
    "sqrtf",
    "fminf",
    "fmaxf",
    "min",
    "max",
})

# Historical feature generators sometimes serialize pure helper calls as if the
# helper were a t_core member, for example ``p_core->ratio(a, b)``. When this is
# enabled, any syntactically safe helper written in that legacy-prefixed form is
# normalized to ``ratio(a, b)`` automatically. The helper name is recorded in
# the generated header/report so firmware integration remains auditable.
AUTO_ALLOW_LEGACY_PREFIXED_HELPERS = True

# Set this to the firmware header that declares helper functions/macros such as
# diff() and ratio(). Leave as None when core_type_def.h already provides them
# or the including C file declares them before including the generated model.
FEATURE_HELPERS_HEADER: str | None = None

_FEATURE_TOKEN_RE = re.compile(
    r"\s*(->|<<|>>|[A-Za-z_]\w*|"
    r"(?:(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFlLuU]*)|"
    r"[\[\]\(\),.\+\-*/%&|^~])"
)



@dataclass(frozen=True)
class ExportContext:
    group_name: str
    group_var: str
    group_macro: str
    preprocessing: str
    runtime_normalization: str | None
    selected_indices: Tuple[int, ...]
    feature_names: Tuple[str, ...]
    decision_threshold: float
    is_regression: bool
    model_type: str


@dataclass(frozen=True)
class SklearnNode:
    threshold: float
    right_child: int
    feature_index: int
    is_leaf: bool


@dataclass(frozen=True)
class LgbmNode:
    threshold: float
    left_child: int
    right_child: int
    feature_index: int
    default_left: bool
    missing_is_zero: bool
    is_leaf: bool


@dataclass(frozen=True)
class ExportValidation:
    sample_count: int
    max_abs_error: float
    mean_abs_error: float
    decision_disagreements: int


@dataclass(frozen=True)
class StorageLayout:
    code: str
    total_nodes: int
    max_tree_nodes: int
    root_c_type: str
    root_bytes: int
    child_c_type: str
    child_bytes: int
    metadata_layout: str
    node_data_bytes: int
    root_table_bytes: int
    tree_payload_bytes: int
    legacy_8byte_payload_bytes: int
    flag_storage_bytes: int = 0
    feature_metadata_layout: str = "plain_u8"
    default_left_encoding: str = "not_applicable"
    missing_is_zero_encoding: str = "not_applicable"



def _root_index_storage(total_nodes: int) -> Tuple[str, int]:
    if total_nodes <= np.iinfo(np.uint16).max:
        return "uint16_t", 2
    if total_nodes <= np.iinfo(np.uint32).max:
        return "uint32_t", 4
    raise ValueError(f"Combined model has {total_nodes} nodes; uint32 root offsets are insufficient.")


def _pack_boolean_bitset(values: Sequence[bool]) -> Tuple[int, ...]:
    """Pack one Boolean value per node, least-significant bit first."""

    packed = [0] * ((len(values) + 7) // 8)
    for index, value in enumerate(values):
        if value:
            packed[index >> 3] |= 1 << (index & 7)
    return tuple(packed)


def _boolean_state(values: Sequence[bool]) -> str:
    """Return constant_0, constant_1, or mixed for a non-empty Boolean sequence."""

    if not values or not any(values):
        return "constant_0"
    if all(values):
        return "constant_1"
    return "mixed"


def _collect_lgbm_dump_nodes(node: Mapping[str, Any], output: MutableSequence[LgbmNode]) -> None:
    """Collect pre-order LightGBM metadata without needing preprocessing state."""

    if "leaf_value" in node:
        output.append(LgbmNode(0.0, 0, 0, 0xFF, False, False, True))
        return
    output.append(
        LgbmNode(
            0.0,
            0,
            0,
            int(node["split_feature"]),
            bool(node.get("default_left", True)),
            str(node.get("missing_type", "None")) == "Zero",
            False,
        )
    )
    _collect_lgbm_dump_nodes(node["left_child"], output)
    _collect_lgbm_dump_nodes(node["right_child"], output)


def _lgbm_large_metadata_plan(nodes: Sequence[LgbmNode]) -> Dict[str, Any]:
    """Choose the smallest exact metadata representation for large LightGBM trees.

    Leaf nodes already use feature byte 0xFF. For internal nodes:
    * up to 63 actually-used features: both flags occupy feature bits 6 and 7;
    * up to 127 actually-used features: one varying flag occupies feature bit 7;
    * otherwise: varying flags use one-bit bitsets.

    Constant flags consume no flash and compile to literal 0/1 values.
    """

    internal = [node for node in nodes if not node.is_leaf]
    maximum_feature = max((int(node.feature_index) for node in internal), default=-1)
    default_state = _boolean_state([bool(node.default_left) for node in internal])
    missing_state = _boolean_state([bool(node.missing_is_zero) for node in internal])

    feature_layout = "plain_u8"
    default_encoding = default_state
    missing_encoding = missing_state

    if maximum_feature <= 62:
        # 0xFF remains exclusively the leaf marker because feature 63 is reserved.
        feature_layout = "u6_feature_default_left_missing_is_zero"
        default_encoding = "feature_bit6"
        missing_encoding = "feature_bit7"
    elif maximum_feature <= 126:
        # Reserve 0xFF for leaves; one flag can use the high feature bit.
        if default_state == "mixed":
            feature_layout = "u7_feature_default_left"
            default_encoding = "feature_bit7"
        elif missing_state == "mixed":
            feature_layout = "u7_feature_missing_is_zero"
            missing_encoding = "feature_bit7"

    default_bits: Tuple[int, ...] = ()
    missing_bits: Tuple[int, ...] = ()
    if default_encoding == "mixed":
        default_encoding = "bitset"
        default_bits = _pack_boolean_bitset(
            [bool(node.default_left) if not node.is_leaf else False for node in nodes]
        )
    if missing_encoding == "mixed":
        missing_encoding = "bitset"
        missing_bits = _pack_boolean_bitset(
            [bool(node.missing_is_zero) if not node.is_leaf else False for node in nodes]
        )

    return {
        "feature_metadata_layout": feature_layout,
        "default_left_encoding": default_encoding,
        "missing_is_zero_encoding": missing_encoding,
        "default_left_bits": default_bits,
        "missing_is_zero_bits": missing_bits,
        "flag_storage_bytes": len(default_bits) + len(missing_bits),
        "maximum_split_feature": maximum_feature,
    }


def _lgbm_feature_metadata_literal(node: LgbmNode, layout: str) -> str:
    if node.is_leaf:
        return "0xFFu"
    feature = int(node.feature_index)
    if layout == "u6_feature_default_left_missing_is_zero":
        if feature > 62:
            raise AssertionError("Six-bit LightGBM feature metadata overflow.")
        value = feature | ((1 if node.default_left else 0) << 6) | (
            (1 if node.missing_is_zero else 0) << 7
        )
    elif layout == "u7_feature_default_left":
        if feature > 126:
            raise AssertionError("Seven-bit LightGBM feature metadata overflow.")
        value = feature | ((1 if node.default_left else 0) << 7)
    elif layout == "u7_feature_missing_is_zero":
        if feature > 126:
            raise AssertionError("Seven-bit LightGBM feature metadata overflow.")
        value = feature | ((1 if node.missing_is_zero else 0) << 7)
    elif layout == "plain_u8":
        value = feature
    else:
        raise AssertionError(f"Unknown LightGBM feature metadata layout: {layout}")
    if value == 0xFF:
        raise AssertionError("Internal LightGBM metadata collided with the 0xFF leaf marker.")
    return f"0x{value:02X}u"


def _bitset_array_code(group_var: str, flag_name: str, packed: Sequence[int]) -> str:
    if not packed:
        return ""
    return _format_c_array(
        "uint8_t",
        f"{group_var}_node_{flag_name}_bits",
        [f"0x{int(value):02X}u" for value in packed],
        per_line=16,
    ).rstrip()


def _xgb_dump_trees(model_obj: Any) -> List[Dict[str, Any]]:
    """Parse binary XGBoost or identity-link regression trees, preserving C routing."""
    if xgb is None or not isinstance(model_obj, (xgb.XGBClassifier, xgb.XGBRegressor)):
        raise TypeError("XGBoost export requires a fitted XGBClassifier or XGBRegressor")
    booster = model_obj.get_booster()
    config = json.loads(booster.save_config())
    learner = config["learner"]
    gb = learner["gradient_booster"]
    if gb.get("name") != "gbtree":
        raise ValueError("Only XGBoost gbtree (not dart/linear) can be exported")
    is_regression = isinstance(model_obj, xgb.XGBRegressor)
    required_objective = "reg:squarederror" if is_regression else "binary:logistic"
    if learner["objective"]["name"] != required_objective:
        raise ValueError(
            f"Only XGBoost {required_objective} is supported for "
            f"{'regressors' if is_regression else 'classifiers'}; "
            "non-identity regression links are not exportable as a plain tree sum."
        )
    if int(learner["learner_model_param"].get("num_target", "1")) != 1:
        raise ValueError("Only single-output XGBoost models can be exported")
    tree_params = gb.get("gbtree_model_param", {})
    if int(tree_params.get("num_parallel_tree", "1")) != 1:
        raise ValueError("Parallel forests within XGBoost are not supported")
    if hasattr(model_obj, "best_iteration"):
        try:
            best = model_obj.best_iteration
        except AttributeError:
            best = None
        if best is not None and int(best) + 1 < int(booster.num_boosted_rounds()):
            raise ValueError("XGBoost early-stopped model: refit with best_iteration+1 trees before C export")
    result = [json.loads(raw) for raw in booster.get_dump(dump_format="json")]
    if not result:
        raise ValueError("XGBoost contains no trees")
    return result


def _xgb_tree_node_count(node: Mapping[str, Any]) -> int:
    if "leaf" in node:
        return 1
    if "split" not in node or len(node.get("children", [])) != 2:
        raise ValueError("Unsupported XGBoost tree: expected a binary numerical split")
    if str(node.get("split", "")).startswith("f") is False:
        raise ValueError("Only f0/f1 numerical XGBoost features are supported")
    return 1 + sum(_xgb_tree_node_count(child) for child in node["children"])


def _xgb_raw_init_score(model_obj: Any) -> float:
    """Preserve binary:logistic logit base score (legacy classifier semantics)."""
    config = json.loads(model_obj.get_booster().save_config())
    base = str(config["learner"]["learner_model_param"]["base_score"]).strip("[] ")
    pieces = [part.strip() for part in base.split(",") if part.strip()]
    if len(pieces) != 1:
        raise ValueError("Only single-output XGBoost base_score is supported")
    probability = float(pieces[0])
    if not 0.0 < probability < 1.0:
        raise ValueError(f"XGBoost base_score probability must lie in (0,1): {probability}")
    return _float32(math.log(probability / (1.0 - probability)))


def _xgb_regression_init_score(model_obj: Any) -> float:
    """reg:squarederror uses base_score in output units (NOT logit units)."""
    config = json.loads(model_obj.get_booster().save_config())
    base = str(config["learner"]["learner_model_param"]["base_score"]).strip("[] ")
    pieces = [part.strip() for part in base.split(",") if part.strip()]
    if len(pieces) != 1:
        raise ValueError("Only single-output XGBoost base_score is supported")
    return _float32(float(pieces[0]))


def _fold_xgb_strict_less(
    threshold: float,
    feature_index: int,
    scalar_transform: Callable[[int, np.float32], np.float32],
    *,
    affine_folded: bool,
) -> float:
    """Represent an XGBoost x < threshold split using an exact float32 x <= bound.

    In the affine case find the last float32 raw value whose float32 transformed
    value is strictly below the XGBoost float32 threshold. No extra FW math.
    """
    t = np.float32(threshold)
    if not np.isfinite(t):
        raise ValueError("XGBoost split threshold is not finite")
    if not affine_folded:
        return _float32(np.nextafter(t, np.float32(-np.inf), dtype=np.float32))

    def goes_left(raw_key: int) -> bool:
        raw = _ordered_key_to_float32(raw_key)
        value = scalar_transform(feature_index, raw)
        return bool(np.float32(value) < t)

    low = _float32_to_ordered_key(np.float32(-np.finfo(np.float32).max))
    high = _float32_to_ordered_key(np.float32(np.finfo(np.float32).max))
    if not goes_left(low):
        raise ValueError("XGBoost split has no finite raw values routing left")
    if goes_left(high):
        return _float32(np.finfo(np.float32).max)
    lo, hi = low, high
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2
        if goes_left(mid):
            lo = mid
        else:
            hi = mid - 1
    return _float32(_ordered_key_to_float32(lo))


def _flatten_xgb_tree(
    node: Mapping[str, Any],
    flattened: MutableSequence[LgbmNode | None],
    scalar_transform: Callable[[int, np.float32], np.float32],
    affine_folded: bool,
) -> int:
    """Use the existing 6-byte/node boosted-tree layout with XGB-specific routing."""
    position = len(flattened)
    if "leaf" in node:
        flattened.append(LgbmNode(_float32(node["leaf"]), 0, 0, 0xFF, False, False, True))
        return position
    _xgb_tree_node_count(node) if len(flattened) == 0 else None
    split = str(node["split"])
    if not re.fullmatch(r"f[0-9]+", split):
        raise ValueError(f"Unsupported XGBoost feature {split!r}; categorical splits not supported")
    feature = int(split[1:])
    if feature > 254:
        raise ValueError("Compact C exporter supports up to 254 features")
    yes, no, missing = int(node["yes"]), int(node["no"]), int(node["missing"])
    children = {int(child["nodeid"]): child for child in node["children"]}
    if yes not in children or no not in children or yes == no or missing not in (yes, no):
        raise ValueError("Invalid XGBoost yes/no/missing child relationship")
    flattened.append(None)
    left = _flatten_xgb_tree(children[yes], flattened, scalar_transform, affine_folded)
    right = _flatten_xgb_tree(children[no], flattened, scalar_transform, affine_folded)
    if left != position + 1:
        raise AssertionError("XGB flatten must have implicit left child")
    cutoff = _fold_xgb_strict_less(
        float(node["split_condition"]), feature, scalar_transform,
        affine_folded=affine_folded,
    )
    flattened[position] = LgbmNode(cutoff, left, right, feature, missing == yes, False, False)
    return position


def _build_xgb_trees(
    model_obj: Any,
    scalar_transform: Callable[[int, np.float32], np.float32],
    affine_folded: bool,
) -> List[List[LgbmNode]]:
    result: List[List[LgbmNode]] = []
    for root in _xgb_dump_trees(model_obj):
        nodes: List[LgbmNode | None] = []
        if _flatten_xgb_tree(root, nodes, scalar_transform, affine_folded) != 0:
            raise AssertionError("Invalid XGB root index")
        if len(nodes) > 255:
            raise ValueError("XGBoost max tree nodes exceeds uint8 index; tune max_leaves <= 128")
        if any(n is None for n in nodes):
            raise AssertionError("Unfilled XGBoost node")
        result.append([n for n in nodes if n is not None])
    return result


def _estimate_compact_tree_payload_legacy(model: Any) -> Dict[str, Any]:
    """Return exact generated static-data bytes for the compact SoA layout."""

    raw_model = model.get("model") if isinstance(model, Mapping) else model
    if raw_model is None:
        raise ValueError("Cannot estimate payload without a fitted model.")

    flag_storage_bytes = 0
    feature_metadata_layout = "not_applicable"
    default_left_encoding = "not_applicable"
    missing_is_zero_encoding = "not_applicable"
    metadata_layout = ""

    if isinstance(raw_model, (RandomForestClassifier, RandomForestRegressor)):
        counts = [int(estimator.tree_.node_count) for estimator in raw_model.estimators_]
        if not counts:
            raise ValueError("RandomForest has no estimators.")
        max_nodes = max(counts)
        child_bytes = 1 if max_nodes <= np.iinfo(np.uint8).max else 2
        if max_nodes > np.iinfo(np.uint16).max:
            raise ValueError("A RandomForest tree exceeds uint16 local child indexing.")
        node_bytes = RF_U8_NODE_BYTES if child_bytes == 1 else RF_U16_NODE_BYTES
        family = "sklearn"
        total_nodes = int(sum(counts))
        node_data_bytes = total_nodes * node_bytes
        n_outputs = int(getattr(raw_model, "n_outputs_", 1))
        if isinstance(raw_model, RandomForestClassifier) and n_outputs > 1:
            # Output 0 reuses the legacy leaf threshold slot. Outputs 1..N-1
            # are stored in the per-node side table used by the multi-output
            # single-traversal firmware path.
            node_data_bytes += total_nodes * (n_outputs - 1) * 4
        metadata_layout = (
            (
                "rf_u8_child_feature_packed_u16_multioutput_side_table"
                if child_bytes == 1
                else "rf_u16_child_separate_feature_multioutput_side_table"
            )
            if isinstance(raw_model, RandomForestClassifier) and n_outputs > 1
            else (
                "rf_u8_child_feature_packed_u16" if child_bytes == 1
                else "rf_u16_child_separate_feature"
            )
        )
    elif lgb is not None and isinstance(raw_model, lgb.LGBMModel):
        dump = raw_model.booster_.dump_model()
        tree_info = dump.get("tree_info", [])
        counts = [2 * int(tree["num_leaves"]) - 1 for tree in tree_info]
        if not counts:
            raise ValueError("LightGBM model has no trees.")
        max_nodes = max(counts)
        if max_nodes > np.iinfo(np.uint8).max:
            raise ValueError("A LightGBM tree exceeds uint8 local child indexing.")
        child_bytes = 1
        node_bytes = LGBM_PACKED_NODE_BYTES
        family = "lgbm"
        total_nodes = int(sum(counts))
        if max_nodes <= 64:
            node_data_bytes = total_nodes * LGBM_PACKED_NODE_BYTES
            metadata_layout = "lgbm_u6_child_flags_packed"
            feature_metadata_layout = "plain_u8"
            default_left_encoding = "child_bit6"
            missing_is_zero_encoding = "child_bit7"
        else:
            metadata_nodes: List[LgbmNode] = []
            for tree in tree_info:
                _collect_lgbm_dump_nodes(tree["tree_structure"], metadata_nodes)
            if len(metadata_nodes) != total_nodes:
                raise AssertionError(
                    f"LightGBM metadata count mismatch: {len(metadata_nodes)} != {total_nodes}."
                )
            plan = _lgbm_large_metadata_plan(metadata_nodes)
            flag_storage_bytes = int(plan["flag_storage_bytes"])
            feature_metadata_layout = str(plan["feature_metadata_layout"])
            default_left_encoding = str(plan["default_left_encoding"])
            missing_is_zero_encoding = str(plan["missing_is_zero_encoding"])
            node_data_bytes = (
                total_nodes * LGBM_LARGE_BASE_NODE_BYTES + flag_storage_bytes
            )
            metadata_layout = "lgbm_u8_child_feature_metadata_bitpacked_flags"
    elif xgb is not None and isinstance(raw_model, (xgb.XGBClassifier, xgb.XGBRegressor)):
        trees = _xgb_dump_trees(raw_model)
        counts = [_xgb_tree_node_count(tree) for tree in trees]
        max_nodes = max(counts)
        if max_nodes > 255:
            raise ValueError("XGBoost tree exceeds 255 nodes")
        child_bytes = 1
        node_bytes = LGBM_PACKED_NODE_BYTES
        family = "xgb"
        total_nodes = int(sum(counts))
        if max_nodes <= 64:
            node_data_bytes = total_nodes * 6
            metadata_layout = "xgb_u6_child_flags_packed"
        else:
            # The two missing flags are constant/bitpacked as necessary.
            flat_trees = _build_xgb_trees(raw_model, lambda _i, value: value, False)
            nodes = [node for tree in flat_trees for node in tree]
            plan = _lgbm_large_metadata_plan(nodes)
            flag_storage_bytes = int(plan["flag_storage_bytes"])
            feature_metadata_layout = str(plan["feature_metadata_layout"])
            default_left_encoding = str(plan["default_left_encoding"])
            missing_is_zero_encoding = str(plan["missing_is_zero_encoding"])
            node_data_bytes = total_nodes * 6 + flag_storage_bytes
            metadata_layout = "xgb_u8_child_feature_metadata_bitpacked_flags"
    elif isinstance(raw_model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        counts = [int(raw_model.tree_.node_count)]
        max_nodes = counts[0]
        child_bytes = 1 if max_nodes <= np.iinfo(np.uint8).max else 2
        if max_nodes > np.iinfo(np.uint16).max:
            raise ValueError("DecisionTree exceeds uint16 local child indexing.")
        node_bytes = RF_U8_NODE_BYTES if child_bytes == 1 else RF_U16_NODE_BYTES
        family = "sklearn"
        total_nodes = int(sum(counts))
        node_data_bytes = total_nodes * node_bytes
        metadata_layout = (
            "rf_u8_child_feature_packed_u16" if child_bytes == 1
            else "rf_u16_child_separate_feature"
        )
    else:
        raise TypeError(
            f"Unsupported model for compact payload estimation: {type(raw_model).__name__}."
        )

    root_c_type, root_bytes = _root_index_storage(total_nodes)
    root_table_bytes = len(counts) * root_bytes
    tree_payload_bytes = node_data_bytes + root_table_bytes
    probability_lut_bytes = 0
    early_exit_bounds_bytes = 0
    if family in {"lgbm", "xgb"} and (
        (family == "lgbm" and lgb is not None and isinstance(raw_model, lgb.LGBMClassifier))
        or (family == "xgb" and xgb is not None and isinstance(raw_model, xgb.XGBClassifier))
    ):
        probability_lut_bytes = (19 + 20) * 4 if LIGHTGBM_ENABLE_Q20_PROBABILITY else 0
        if LIGHTGBM_EARLY_EXIT_GROUP_SIZE > 0:
            group_count = (
                len(counts) + LIGHTGBM_EARLY_EXIT_GROUP_SIZE - 1
            ) // LIGHTGBM_EARLY_EXIT_GROUP_SIZE
            early_exit_bounds_bytes = (group_count + 1) * 2 * 4

    return {
        "model_type": family,
        "tree_count": len(counts),
        "total_nodes": total_nodes,
        "max_tree_nodes": max_nodes,
        "node_bytes": node_bytes,
        "base_node_bytes": node_bytes,
        "effective_node_bytes": node_data_bytes / total_nodes if total_nodes else 0.0,
        "child_bytes": child_bytes,
        "root_bytes": root_bytes,
        "root_c_type": root_c_type,
        "metadata_layout": metadata_layout,
        "feature_metadata_layout": feature_metadata_layout,
        "default_left_encoding": default_left_encoding,
        "missing_is_zero_encoding": missing_is_zero_encoding,
        "flag_storage_bytes": flag_storage_bytes,
        "node_data_bytes": node_data_bytes,
        "root_table_bytes": root_table_bytes,
        "tree_payload_bytes": tree_payload_bytes,
        "probability_lut_bytes": probability_lut_bytes,
        "early_exit_bounds_bytes": early_exit_bounds_bytes,
        "total_static_data_bytes": tree_payload_bytes + probability_lut_bytes + early_exit_bounds_bytes,
        "legacy_8byte_payload_bytes": total_nodes * 8,
    }

def _sanitize_c_identifier(value: str, *, uppercase: bool = False) -> str:
    text = re.sub(r"[^0-9A-Za-z_]", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_") or "gesture"
    if text[0].isdigit():
        text = f"model_{text}"
    return text.upper() if uppercase else text.lower()


def _float32(value: Any) -> float:
    result = float(np.float32(value))
    if not math.isfinite(result):
        raise ValueError(f"Cannot export non-finite float value: {value!r}")
    if result == 0.0:
        return 0.0
    return result


def _c_float(value: Any) -> str:
    number = _float32(value)
    text = format(number, ".9g")
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return f"{text}f"


def _tokenize_safe_feature_expression(expression: str) -> List[str]:
    """Tokenize the intentionally small, side-effect-free C expression grammar."""

    tokens: List[str] = []
    position = 0
    while position < len(expression):
        match = _FEATURE_TOKEN_RE.match(expression, position)
        if match is None:
            preview = expression[position : position + 24]
            raise ValueError(
                "Unsupported token in firmware feature expression at offset "
                f"{position}: {preview!r}"
            )
        tokens.append(match.group(1))
        position = match.end()
    return tokens


def _validate_safe_feature_expression(
    expression: str,
    original: str,
    *,
    allowed_functions: Iterable[str] = ALLOWED_FEATURE_FUNCTIONS,
) -> Tuple[str, ...]:
    """Reject statements, assignments and unsafe function-call syntax.

    The grammar supports direct ``p_core->field`` access, arithmetic, symbolic
    indices, ordinary approved helper calls, and approved callable members of
    ``p_core`` such as ``p_core->ratio(a, b)``. The returned tuple records the
    helper names required by the generated firmware expression.
    """

    forbidden = (";", "#", "{", "}", '"', "'", "`", "\\", "//", "/*", "*/", "?")
    if any(token in expression for token in forbidden):
        raise ValueError(f"Unsafe token in firmware feature expression: {original!r}")

    allowed = frozenset(str(value) for value in allowed_functions)
    tokens = _tokenize_safe_feature_expression(expression)
    if not tokens:
        raise ValueError("Feature expression cannot be empty.")

    helpers: set[str] = set()
    stack: List[str] = []
    matching = {")": "(", "]": "["}
    for index, token in enumerate(tokens):
        if token in ("(", "["):
            stack.append(token)
        elif token in matching:
            if not stack or stack.pop() != matching[token]:
                raise ValueError(f"Unbalanced delimiters in feature expression: {original!r}")

        if re.fullmatch(r"[A-Za-z_]\w*", token):
            next_token = tokens[index + 1] if index + 1 < len(tokens) else None
            previous = tokens[index - 1] if index else None
            if next_token == "(":
                if previous == ".":
                    raise ValueError(
                        f"Dot member-function calls are not supported: {original!r}"
                    )
                if previous == "->":
                    receiver = tokens[index - 2] if index >= 2 else None
                    if receiver != "p_core":
                        raise ValueError(
                            "Only p_core callable members are supported in firmware feature "
                            f"expressions: {original!r}"
                        )
                if token not in allowed:
                    allowed_text = ", ".join(sorted(allowed))
                    raise ValueError(
                        f"Feature expression calls unapproved helper {token!r}. "
                        f"Allowed helpers: {allowed_text}. Expression: {original!r}"
                    )
                helpers.add(token)
            if token == "p_core" and next_token != "->":
                raise ValueError(
                    f"p_core must be dereferenced with '->' in feature expression: {original!r}"
                )

    if stack:
        raise ValueError(f"Unbalanced delimiters in feature expression: {original!r}")
    if "p_core" not in tokens:
        raise ValueError(
            "Firmware feature expression must reference p_core at least once: "
            f"{original!r}"
        )
    return tuple(sorted(helpers))


def _convert_feature_expression(key: str) -> Tuple[str, Tuple[str, ...]]:
    """Return ``(safe_c_expression, required_helpers)`` for one schema feature."""

    value = str(key).strip()
    if not value:
        raise ValueError("Feature name cannot be empty.")

    expression = value.replace("psd_features.", "p_core->").replace("p_core.", "p_core->")
    helper_pattern = re.compile(r"\bp_core->([A-Za-z_]\w*)\s*(?=\()")
    legacy_helpers = frozenset(helper_pattern.findall(expression))
    allowed_helpers = set(ALLOWED_FEATURE_FUNCTIONS)
    if AUTO_ALLOW_LEGACY_PREFIXED_HELPERS:
        allowed_helpers.update(legacy_helpers)

    if FEATURE_HELPER_CALL_STYLE == "free_function":
        def normalize_helper(match: re.Match[str]) -> str:
            helper = match.group(1)
            return helper if helper in allowed_helpers else match.group(0)
        expression = helper_pattern.sub(normalize_helper, expression)
    elif FEATURE_HELPER_CALL_STYLE != "p_core_member":
        raise ValueError(
            "FEATURE_HELPER_CALL_STYLE must be 'p_core_member' or 'free_function'."
        )

    begins_with_helper = any(
        re.match(rf"^{re.escape(helper)}\s*\(", expression)
        for helper in allowed_helpers
    )
    if "p_core->" not in expression and not begins_with_helper and not expression.startswith("("):
        expression = "p_core->" + expression

    helpers = _validate_safe_feature_expression(
        expression,
        value,
        allowed_functions=allowed_helpers,
    )
    return expression, helpers


def csv_key_to_c_accessor(key: str) -> str:
    """Convert one schema feature name to a safe firmware C expression.

    With the default Cortex-M4 profile, legacy helper expressions remain callable
    members, for example ``p_core->ratio(a, b)``. Set
    ``FEATURE_HELPER_CALL_STYLE='free_function'`` to emit ``ratio(a, b)`` instead.
    """

    expression, _ = _convert_feature_expression(key)
    return expression


def feature_expression_helpers(key: str) -> Tuple[str, ...]:
    """Return helper functions or callable members required by one schema feature."""

    _, helpers = _convert_feature_expression(key)
    return helpers


def validate_feature_schema_accessors(feature_names: Sequence[str]) -> Dict[str, Any]:
    """Preflight an entire feature schema before expensive model training."""

    helpers: set[str] = set()
    for index, feature_name in enumerate(feature_names):
        try:
            _, required = _convert_feature_expression(feature_name)
        except Exception as exc:
            raise ValueError(
                f"Feature schema entry {index} is not firmware-exportable: {feature_name!r}. {exc}"
            ) from exc
        helpers.update(required)
    return {
        "feature_count": len(feature_names),
        "required_feature_helpers": sorted(helpers),
        "helper_call_style": FEATURE_HELPER_CALL_STYLE,
        "auto_legacy_helpers": bool(AUTO_ALLOW_LEGACY_PREFIXED_HELPERS),
    }


def get_model_trees(model_dict: Mapping[str, Any]) -> Tuple[List[Any], bool]:
    """Return sklearn estimators and whether they form an ensemble."""

    if "ensemble" in model_dict:
        trees = list(model_dict["ensemble"])
        if not trees:
            raise ValueError("The model container has an empty 'ensemble'.")
        return trees, len(trees) > 1

    model = model_dict.get("model")
    if model is None:
        raise ValueError("Could not find a 'model' key in the model container.")
    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        return list(model.estimators_), True
    if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        return [model], False
    raise TypeError(f"{type(model).__name__} is not an sklearn RF/DT model.")


def get_model_and_type(model_dict: Mapping[str, Any]) -> Tuple[Any, str, bool]:
    """Return ``(model, platform, is_regression)`` with strict validation."""

    model = model_dict.get("model")
    if model is None:
        # Retain compatibility with legacy containers that stored one estimator
        # under ``ensemble`` but reject multi-model custom voting semantics.
        ensemble = model_dict.get("ensemble")
        if isinstance(ensemble, Sequence) and len(ensemble) == 1:
            model = ensemble[0]
        else:
            raise ValueError("Could not find a supported 'model' in the .pkl file.")

    if isinstance(
        model,
        (
            RandomForestClassifier,
            DecisionTreeClassifier,
            RandomForestRegressor,
            DecisionTreeRegressor,
        ),
    ):
        n_outputs = int(getattr(model, "n_outputs_", 1))
        is_regression = isinstance(model, (RandomForestRegressor, DecisionTreeRegressor))
        if n_outputs != 1 and not isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            raise ValueError(
                "Multi-output export is supported only for sklearn RandomForestClassifier/Regressor; "
                f"got {type(model).__name__} with model.n_outputs_={n_outputs}."
            )
        return model, "sklearn", is_regression

    if xgb is not None and isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
        return model, "xgb", isinstance(model, xgb.XGBRegressor)

    if lgb is not None and isinstance(model, (lgb.LGBMModel, lgb.LGBMClassifier, lgb.LGBMRegressor)):
        is_regression = isinstance(model, lgb.LGBMRegressor)
        if not is_regression:
            objective = str(model.get_params().get("objective", "")).lower()
            is_regression = any(token in objective for token in ("regression", "mse", "rmse", "l1", "l2"))
        return model, "lgbm", is_regression

    raise TypeError(
        f"Model type {type(model).__module__}.{type(model).__name__} is not supported. "
        "Supported models are sklearn DecisionTree/RandomForest, LightGBM, "
        "XGBClassifier and XGBRegressor."
    )


def _resolve_preprocessing(model_dict: Mapping[str, Any], trial_info: Mapping[str, Any]) -> Tuple[str, Any | None]:
    params = trial_info.get("params")
    params = params if isinstance(params, Mapping) else {}

    # Explicitly reject legacy log models. Silently exporting one as raw is unsafe.
    legacy_log = bool(params.get("use_log_transform", False) or trial_info.get("use_log_transform", False))
    if legacy_log:
        raise ValueError(
            "This model was trained with a logarithmic transform. LOG export is intentionally disabled "
            "because it is too expensive for the target firmware. Retrain with a supported non-log mode."
        )

    raw_candidates = [
        ("model.preprocessing", model_dict.get("preprocessing")),
        ("trial.preprocessing", trial_info.get("preprocessing")),
        ("trial.params.preprocessing", params.get("preprocessing")),
    ]
    normalized_candidates: List[Tuple[str, str]] = []
    for source, raw_value in raw_candidates:
        if raw_value is None:
            continue
        raw_text = str(raw_value).strip().lower()
        normalized = _PREPROCESSING_ALIASES.get(raw_text, raw_text)
        normalized_candidates.append((source, normalized))

    distinct = {value for _, value in normalized_candidates}
    if len(distinct) > 1:
        details = ", ".join(f"{source}={value!r}" for source, value in normalized_candidates)
        raise ValueError(f"Conflicting preprocessing metadata: {details}.")

    if normalized_candidates:
        name = normalized_candidates[0][1]
    else:
        legacy_standard = bool(
            params.get("use_standardization", False)
            or trial_info.get("use_standardization", False)
        )
        name = "standard" if legacy_standard else "none"

    if name not in SUPPORTED_PREPROCESSING:
        raise ValueError(
            f"Unsupported preprocessing mode {name!r}. Supported non-log modes: "
            + ", ".join(sorted(SUPPORTED_PREPROCESSING))
        )

    preprocessor = model_dict.get("preprocessor")
    if preprocessor is None:
        preprocessor = model_dict.get("scaler")

    if name in AFFINE_PREPROCESSING and preprocessor is None:
        raise ValueError(
            f"Model metadata says preprocessing={name!r}, but no fitted preprocessor/scaler is stored in the .pkl."
        )
    if name in ROW_NORMALIZATION and preprocessor is not None and not isinstance(preprocessor, Normalizer):
        raise TypeError(
            f"preprocessing={name!r} expects sklearn.preprocessing.Normalizer or None, "
            f"not {type(preprocessor).__name__}."
        )
    if isinstance(preprocessor, MinMaxScaler) and bool(getattr(preprocessor, "clip", False)):
        raise ValueError(
            "MinMaxScaler(clip=True) is not supported by zero-runtime threshold folding. "
            "Retrain with clip=False (the repaired trainer already uses clip=False)."
        )
    return name, preprocessor


def _extract_selected_indices(
    model_dict: Mapping[str, Any], trial_info: Mapping[str, Any], model_obj: Any, schema_size: int
) -> Tuple[int, ...]:
    raw_candidates = [
        ("trial.selected_indices", trial_info.get("selected_indices")),
        ("trial.used_indices", trial_info.get("used_indices")),
        ("model.indices", model_dict.get("indices")),
    ]
    candidates: List[Tuple[str, Tuple[int, ...]]] = []
    for source, raw_values in raw_candidates:
        if raw_values is None:
            continue
        values = tuple(int(value) for value in np.asarray(raw_values).reshape(-1).tolist())
        candidates.append((source, values))
    if not candidates:
        raise KeyError("Could not find selected_indices/used_indices/indices in metadata.")

    indices = candidates[0][1]
    for source, candidate in candidates[1:]:
        if candidate != indices:
            details = ", ".join(f"{name}={list(value)}" for name, value in candidates)
            raise ValueError(f"Conflicting selected-feature metadata: {details}.")

    if not indices:
        raise ValueError("The selected feature list is empty.")
    if len(indices) > 254:
        raise ValueError(
            f"The compact node format supports at most 254 model features; received {len(indices)}."
        )
    if len(set(indices)) != len(indices):
        raise ValueError("selected_indices contains duplicate original feature indices.")
    if min(indices) < 0 or max(indices) >= schema_size:
        raise IndexError(
            f"selected_indices must stay in [0, {schema_size - 1}], got min={min(indices)}, max={max(indices)}."
        )

    model_n_features = getattr(model_obj, "n_features_in_", None)
    if model_n_features is not None and int(model_n_features) != len(indices):
        raise ValueError(
            "Model input width does not match selected_indices: "
            f"model.n_features_in_={model_n_features}, selected={len(indices)}."
        )
    return indices


def _resolve_threshold(
    trial_info: Mapping[str, Any],
    model_dict: Mapping[str, Any],
    *,
    is_regression: bool,
) -> float:
    params = trial_info.get("params")
    params = params if isinstance(params, Mapping) else {}
    raw_candidates = [
        ("trial.decision_threshold", trial_info.get("decision_threshold")),
        ("trial.threshold", trial_info.get("threshold")),
        ("model.decision_threshold", model_dict.get("decision_threshold")),
        ("trial.params.decision_threshold", params.get("decision_threshold")),
        ("trial.params.threshold", params.get("threshold")),
    ]
    values: List[Tuple[str, float]] = []
    for source, raw_value in raw_candidates:
        if raw_value is None:
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{source} must be finite, got {value}.")
        values.append((source, value))

    threshold = values[0][1] if values else 0.5
    for source, value in values[1:]:
        if not math.isclose(value, threshold, rel_tol=0.0, abs_tol=1.0e-12):
            details = ", ".join(f"{name}={item:.17g}" for name, item in values)
            raise ValueError(f"Conflicting decision-threshold metadata: {details}.")

    if not is_regression and not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Classifier decision threshold must be in [0, 1], got {threshold}.")
    return threshold


def _ensure_binary_classifier(model_obj: Any, model_type: str) -> int:
    classes = np.asarray(getattr(model_obj, "classes_", []))
    if classes.ndim != 1 or classes.size != 2:
        raise ValueError(
            "Only single-output binary classifiers are supported by the compact exporter; "
            f"model classes are {classes.tolist()}."
        )
    positive = np.flatnonzero(classes == 1)
    if positive.size != 1:
        raise ValueError(
            "Binary classifier export expects the positive class label to be exactly 1; "
            f"model classes are {classes.tolist()}."
        )
    positive_index = int(positive[0])
    if model_type in {"lgbm", "xgb"} and positive_index != 1:
        raise ValueError(
            "LightGBM binary raw scores represent classes_[1]. The positive label 1 must therefore "
            f"be classes_[1], but classes_={classes.tolist()}."
        )
    return positive_index


def _multioutput_binary_positive_indices(model_obj: Any) -> Tuple[int, ...]:
    """Return positive-class indices for a multi-output binary RF.

    sklearn stores one ``classes_`` array per output.  The multi-output export
    path intentionally supports only binary outputs whose positive label is 1.
    """

    n_outputs = int(getattr(model_obj, "n_outputs_", 1))
    if n_outputs <= 1:
        raise ValueError("Multi-output RF export requires model.n_outputs_ > 1.")
    if not isinstance(model_obj, RandomForestClassifier):
        raise TypeError("Multi-output export is supported only for RandomForestClassifier.")

    classes_per_output = getattr(model_obj, "classes_", None)
    if not isinstance(classes_per_output, (list, tuple)) or len(classes_per_output) != n_outputs:
        raise ValueError(
            "Expected one classes_ array per output for multi-output RF export; "
            f"n_outputs_={n_outputs}."
        )

    result: List[int] = []
    for output_index, output_classes in enumerate(classes_per_output):
        classes = np.asarray(output_classes)
        if classes.ndim != 1 or classes.size != 2:
            raise ValueError(
                f"Output {output_index} is not binary; classes={classes.tolist()}."
            )
        positive = np.flatnonzero(classes == 1)
        if positive.size != 1:
            raise ValueError(
                f"Output {output_index} must contain positive class label 1; "
                f"classes={classes.tolist()}."
            )
        result.append(int(positive[0]))
    return tuple(result)


def _select_or_validate_vector(values: Any, n_features: int, selected_indices: Sequence[int], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == n_features:
        selected = array
    elif array.size > max(selected_indices):
        # Compatibility with the old trainer, which fitted StandardScaler before
        # slicing selected features and therefore stored full-schema statistics.
        selected = array[np.asarray(selected_indices, dtype=np.int64)]
    else:
        raise ValueError(
            f"Fitted preprocessor {label} has {array.size} values, but the model needs {n_features}."
        )
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"Fitted preprocessor {label} contains NaN/Inf.")
    return selected


def _affine_coefficients(
    preprocessing: str,
    preprocessor: Any,
    n_features: int,
    selected_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return vectors ``a, b`` for transformed feature ``z = a*x + b``."""

    if preprocessing == "none":
        return np.ones(n_features, dtype=np.float64), np.zeros(n_features, dtype=np.float64)

    if preprocessing == "standard":
        if not isinstance(preprocessor, StandardScaler):
            raise TypeError(f"standard preprocessing requires StandardScaler, got {type(preprocessor).__name__}.")
        scale = (
            _select_or_validate_vector(preprocessor.scale_, n_features, selected_indices, "scale_")
            if preprocessor.with_std
            else np.ones(n_features, dtype=np.float64)
        )
        mean = (
            _select_or_validate_vector(preprocessor.mean_, n_features, selected_indices, "mean_")
            if preprocessor.with_mean
            else np.zeros(n_features, dtype=np.float64)
        )
        a = 1.0 / scale
        b = -mean / scale

    elif preprocessing in {"minmax_0_1", "minmax_m1_1"}:
        if not isinstance(preprocessor, MinMaxScaler):
            raise TypeError(f"{preprocessing} requires MinMaxScaler, got {type(preprocessor).__name__}.")
        a = _select_or_validate_vector(preprocessor.scale_, n_features, selected_indices, "scale_")
        b = _select_or_validate_vector(preprocessor.min_, n_features, selected_indices, "min_")

    elif preprocessing == "maxabs":
        if not isinstance(preprocessor, MaxAbsScaler):
            raise TypeError(f"maxabs preprocessing requires MaxAbsScaler, got {type(preprocessor).__name__}.")
        scale = _select_or_validate_vector(preprocessor.scale_, n_features, selected_indices, "scale_")
        a = 1.0 / scale
        b = np.zeros(n_features, dtype=np.float64)

    elif preprocessing in {"robust_25_75", "robust_10_90"}:
        if not isinstance(preprocessor, RobustScaler):
            raise TypeError(f"{preprocessing} requires RobustScaler, got {type(preprocessor).__name__}.")
        center = (
            _select_or_validate_vector(preprocessor.center_, n_features, selected_indices, "center_")
            if preprocessor.with_centering
            else np.zeros(n_features, dtype=np.float64)
        )
        scale = (
            _select_or_validate_vector(preprocessor.scale_, n_features, selected_indices, "scale_")
            if preprocessor.with_scaling
            else np.ones(n_features, dtype=np.float64)
        )
        a = 1.0 / scale
        b = -center / scale

    else:
        raise ValueError(f"{preprocessing!r} is not an affine preprocessing mode.")

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Affine preprocessing produced non-finite coefficients.")
    if np.any(a <= 0.0):
        raise ValueError(
            "Tree-threshold folding requires a strictly positive per-feature scale; "
            f"minimum affine coefficient is {float(a.min())}."
        )
    return a, b


def _quantize_split_threshold_down(value: float) -> float:
    """Choose a float32 split threshold without rounding the boundary upward."""

    exact = float(value)
    quantized = np.float32(exact)
    if float(quantized) > exact:
        quantized = np.nextafter(quantized, np.float32(-np.inf), dtype=np.float32)
    return _float32(quantized)


def _make_scalar_transform(
    preprocessing: str,
    preprocessor: Any | None,
    n_features: int,
    selected_indices: Sequence[int],
) -> Callable[[int, np.float32], np.float32]:
    """Reproduce sklearn's float32 scalar operations for threshold folding."""

    if preprocessing == "none":
        return lambda _feature, value: np.float32(value)

    if preprocessing == "standard":
        assert isinstance(preprocessor, StandardScaler)
        mean = (
            _select_or_validate_vector(preprocessor.mean_, n_features, selected_indices, "mean_")
            if preprocessor.with_mean
            else np.zeros(n_features, dtype=np.float64)
        )
        scale = (
            _select_or_validate_vector(preprocessor.scale_, n_features, selected_indices, "scale_")
            if preprocessor.with_std
            else np.ones(n_features, dtype=np.float64)
        )

        mean32 = mean.astype(np.float32)
        scale32 = scale.astype(np.float32)

        def transform(feature: int, value: np.float32) -> np.float32:
            result = np.float32(value)
            if preprocessor.with_mean:
                result = np.float32(result - mean32[feature])
            if preprocessor.with_std:
                result = np.float32(result / scale32[feature])
            return result

        return transform

    if preprocessing in {"minmax_0_1", "minmax_m1_1"}:
        assert isinstance(preprocessor, MinMaxScaler)
        scale = _select_or_validate_vector(preprocessor.scale_, n_features, selected_indices, "scale_")
        offset = _select_or_validate_vector(preprocessor.min_, n_features, selected_indices, "min_")

        scale32 = scale.astype(np.float32)
        offset32 = offset.astype(np.float32)

        def transform(feature: int, value: np.float32) -> np.float32:
            result = np.float32(np.float32(value) * scale32[feature])
            return np.float32(result + offset32[feature])

        return transform

    if preprocessing == "maxabs":
        assert isinstance(preprocessor, MaxAbsScaler)
        scale = _select_or_validate_vector(preprocessor.scale_, n_features, selected_indices, "scale_")
        scale32 = scale.astype(np.float32)
        return lambda feature, value: np.float32(np.float32(value) / scale32[feature])

    if preprocessing in {"robust_25_75", "robust_10_90"}:
        assert isinstance(preprocessor, RobustScaler)
        center = (
            _select_or_validate_vector(preprocessor.center_, n_features, selected_indices, "center_")
            if preprocessor.with_centering
            else np.zeros(n_features, dtype=np.float64)
        )
        scale = (
            _select_or_validate_vector(preprocessor.scale_, n_features, selected_indices, "scale_")
            if preprocessor.with_scaling
            else np.ones(n_features, dtype=np.float64)
        )

        center32 = center.astype(np.float32)
        scale32 = scale.astype(np.float32)

        def transform(feature: int, value: np.float32) -> np.float32:
            result = np.float32(value)
            if preprocessor.with_centering:
                result = np.float32(result - center32[feature])
            if preprocessor.with_scaling:
                result = np.float32(result / scale32[feature])
            return result

        return transform

    # Row normalization is not per-feature affine and is executed in C instead.
    if preprocessing in ROW_NORMALIZATION:
        return lambda _feature, value: np.float32(value)
    raise AssertionError(preprocessing)


def _float32_to_ordered_key(value: np.float32) -> int:
    """Map finite float32 values to monotonically increasing uint32 keys."""

    bits = int(np.asarray(value, dtype=np.float32).view(np.uint32))
    return ((~bits) & 0xFFFFFFFF) if (bits & 0x80000000) else (bits ^ 0x80000000)


def _ordered_key_to_float32(key: int) -> np.float32:
    """Inverse of :func:`_float32_to_ordered_key` for finite float32 values."""

    key &= 0xFFFFFFFF
    bits = (key ^ 0x80000000) if (key & 0x80000000) else ((~key) & 0xFFFFFFFF)
    return np.asarray(bits, dtype=np.uint32).view(np.float32)[()]


def _fold_threshold(
    threshold: float,
    feature_index: int,
    affine_a: np.ndarray,
    affine_b: np.ndarray,
    scalar_transform: Callable[[int, np.float32], np.float32],
    *,
    affine_folded: bool,
) -> float:
    if not affine_folded:
        return _quantize_split_threshold_down(float(threshold))

    def goes_left(raw_value: np.float32) -> bool:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            transformed = scalar_transform(feature_index, raw_value)
        return float(transformed) <= float(threshold)

    # The affine coefficient is required to be positive, therefore goes_left()
    # is monotonic over all finite float32 inputs. Search the complete finite
    # float32 domain by ordered bit key instead of walking with nextafter().
    # The previous 4096-step walk failed for perfectly valid MinMax boundaries
    # when many adjacent raw float32 values collapsed to one transformed value.
    min_key = _float32_to_ordered_key(np.float32(-np.finfo(np.float32).max))
    max_key = _float32_to_ordered_key(np.float32(np.finfo(np.float32).max))

    if not goes_left(_ordered_key_to_float32(min_key)):
        # No finite raw value routes left. A threshold below -FLT_MAX preserves
        # that behavior for all finite firmware inputs.
        return _float32(-np.finfo(np.float32).max)
    if goes_left(_ordered_key_to_float32(max_key)):
        # Every finite raw value routes left.
        return _float32(np.finfo(np.float32).max)

    lo, hi = min_key, max_key
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2
        if goes_left(_ordered_key_to_float32(mid)):
            lo = mid
        else:
            hi = mid - 1

    candidate = _ordered_key_to_float32(lo)
    next_key = lo + 1
    if next_key <= max_key and goes_left(_ordered_key_to_float32(next_key)):
        raise AssertionError(
            f"Internal boundary-search failure for feature {feature_index}: result is not maximal."
        )
    return _float32(candidate)


def _sklearn_tree_nodes(
    tree: _tree.Tree,
    tree_idx: int,
    *,
    is_regression: bool,
    positive_class_index: int,
    affine_a: np.ndarray,
    affine_b: np.ndarray,
    scalar_transform: Callable[[int, np.float32], np.float32],
    affine_folded: bool,
) -> List[SklearnNode]:
    """Reorder an sklearn tree into compact depth-first/implicit-left form.

    sklearn can use either depth-first or best-first builders. With
    ``max_leaf_nodes`` the original left child is *not* guaranteed to be
    ``node + 1``. The old exporter silently relied on that false assumption.
    We recursively reindex every tree so the compact representation is always
    valid, regardless of sklearn's internal node order.
    """

    if tree.node_count > 65535:
        raise ValueError(f"sklearn tree {tree_idx} has {tree.node_count} nodes; uint16 child indices allow 65535.")

    result: List[SklearnNode | None] = []
    visited: set[int] = set()

    def append_subtree(original_index: int) -> int:
        if original_index in visited:
            raise ValueError(
                f"sklearn tree {tree_idx}: node {original_index} is referenced more than once."
            )
        if not 0 <= original_index < tree.node_count:
            raise ValueError(
                f"sklearn tree {tree_idx}: invalid original node index {original_index}."
            )
        visited.add(original_index)

        compact_index = len(result)
        result.append(None)
        left = int(tree.children_left[original_index])
        right = int(tree.children_right[original_index])

        if left == _tree.TREE_LEAF:
            if right != _tree.TREE_LEAF:
                raise ValueError(
                    f"sklearn tree {tree_idx}, node {original_index}: inconsistent leaf children."
                )
            if is_regression:
                leaf_value = float(tree.value[original_index][0][0])
            else:
                values = np.asarray(tree.value[original_index][0], dtype=np.float64)
                if positive_class_index >= values.size:
                    raise ValueError(
                        f"sklearn tree {tree_idx}, node {original_index}: class-value width {values.size} is invalid."
                    )
                total = float(values.sum())
                leaf_value = float(values[positive_class_index]) / total if total > 0.0 else 0.0
            result[compact_index] = SklearnNode(_float32(leaf_value), 0, 0xFF, True)
            return compact_index

        feature_index = int(tree.feature[original_index])
        if not 0 <= feature_index <= 0xFE:
            raise ValueError(
                f"sklearn tree {tree_idx}, node {original_index}: feature index {feature_index} exceeds uint8/0xFE."
            )

        compact_left = append_subtree(left)
        if compact_left != compact_index + 1:
            raise AssertionError("Depth-first reindexing failed to make the left child implicit.")
        compact_right = append_subtree(right)
        raw_threshold = _fold_threshold(
            tree.threshold[original_index],
            feature_index,
            affine_a,
            affine_b,
            scalar_transform,
            affine_folded=affine_folded,
        )
        result[compact_index] = SklearnNode(
            raw_threshold, compact_right, feature_index, False
        )
        return compact_index

    root = append_subtree(0)
    if root != 0 or len(visited) != tree.node_count:
        raise ValueError(
            f"sklearn tree {tree_idx}: visited {len(visited)} of {tree.node_count} nodes."
        )
    if any(node is None for node in result):
        raise AssertionError("sklearn reindexing left an empty node placeholder.")
    return [node for node in result if node is not None]


def _sklearn_multioutput_tree_nodes(
    tree: _tree.Tree,
    tree_idx: int,
    *,
    positive_class_indices: Sequence[int],
    affine_a: np.ndarray,
    affine_b: np.ndarray,
    scalar_transform: Callable[[int, np.float32], np.float32],
    affine_folded: bool,
) -> Tuple[List[SklearnNode], List[Tuple[float, ...]]]:
    """Compact one multi-output RF tree.

    The first output probability is stored in the legacy leaf threshold slot.
    Outputs 1..N-1 are stored in a side table indexed by compact absolute node.
    This preserves the existing node format and still traverses each tree once.
    """

    if tree.node_count > 65535:
        raise ValueError(
            f"sklearn tree {tree_idx} has {tree.node_count} nodes; uint16 child indices allow 65535."
        )
    n_outputs = len(positive_class_indices)
    if n_outputs < 2:
        raise ValueError("Multi-output RF tree export requires at least two outputs.")

    result: List[SklearnNode | None] = []
    extra_probabilities: List[Tuple[float, ...] | None] = []
    visited: set[int] = set()

    def append_subtree(original_index: int) -> int:
        if original_index in visited:
            raise ValueError(
                f"sklearn tree {tree_idx}: node {original_index} is referenced more than once."
            )
        if not 0 <= original_index < tree.node_count:
            raise ValueError(
                f"sklearn tree {tree_idx}: invalid original node index {original_index}."
            )
        visited.add(original_index)

        compact_index = len(result)
        result.append(None)
        extra_probabilities.append(None)
        left = int(tree.children_left[original_index])
        right = int(tree.children_right[original_index])

        if left == _tree.TREE_LEAF:
            if right != _tree.TREE_LEAF:
                raise ValueError(
                    f"sklearn tree {tree_idx}, node {original_index}: inconsistent leaf children."
                )
            values = np.asarray(tree.value[original_index], dtype=np.float64)
            if values.ndim != 2 or values.shape[0] != n_outputs:
                raise ValueError(
                    f"sklearn tree {tree_idx}, node {original_index}: expected value shape "
                    f"({n_outputs}, classes), got {values.shape}."
                )
            probs: List[float] = []
            for output_index, positive_class_index in enumerate(positive_class_indices):
                output_values = values[output_index]
                if positive_class_index >= output_values.size:
                    raise ValueError(
                        f"sklearn tree {tree_idx}, node {original_index}, output {output_index}: "
                        f"class-value width {output_values.size} is invalid."
                    )
                total = float(output_values.sum())
                probs.append(
                    float(output_values[positive_class_index]) / total if total > 0.0 else 0.0
                )
            result[compact_index] = SklearnNode(_float32(probs[0]), 0, 0xFF, True)
            extra_probabilities[compact_index] = tuple(_float32(v) for v in probs[1:])
            return compact_index

        feature_index = int(tree.feature[original_index])
        if not 0 <= feature_index <= 0xFE:
            raise ValueError(
                f"sklearn tree {tree_idx}, node {original_index}: feature index {feature_index} exceeds uint8/0xFE."
            )
        compact_left = append_subtree(left)
        if compact_left != compact_index + 1:
            raise AssertionError("Depth-first reindexing failed to make the left child implicit.")
        compact_right = append_subtree(right)
        raw_threshold = _fold_threshold(
            tree.threshold[original_index],
            feature_index,
            affine_a,
            affine_b,
            scalar_transform,
            affine_folded=affine_folded,
        )
        result[compact_index] = SklearnNode(raw_threshold, compact_right, feature_index, False)
        extra_probabilities[compact_index] = tuple(0.0 for _ in range(n_outputs - 1))
        return compact_index

    append_subtree(0)
    if len(visited) != tree.node_count or any(node is None for node in result):
        raise AssertionError(f"sklearn tree {tree_idx}: compact traversal did not cover every node.")
    return (
        [node for node in result if node is not None],
        [tuple(v) if v is not None else tuple(0.0 for _ in range(n_outputs - 1)) for v in extra_probabilities],
    )


def _lgbm_canonicalize_zero(value: Any) -> np.float32:
    """Match LightGBM's dense/sparse zero canonicalization around ±1e-35."""

    value32 = np.float32(value)
    if -LIGHTGBM_ZERO_THRESHOLD <= value32 <= LIGHTGBM_ZERO_THRESHOLD:
        return np.float32(0.0)
    return value32


def _lgbm_decision_type(node: Mapping[str, Any]) -> str:
    decision = str(node.get("decision_type", "<=")).strip()
    if decision not in {"<="}:
        raise ValueError(
            f"LightGBM categorical/unsupported decision_type={decision!r} cannot be represented by the compact numeric node."
        )
    return decision


def _flatten_lgbm_tree(
    node: Mapping[str, Any],
    flat_nodes: MutableSequence[LgbmNode | None],
    affine_a: np.ndarray,
    affine_b: np.ndarray,
    scalar_transform: Callable[[int, np.float32], np.float32],
    affine_folded: bool,
) -> int:
    if "leaf_value" in node:
        flat_nodes.append(LgbmNode(_float32(node["leaf_value"]), 0, 0, 0xFF, False, False, True))
        return len(flat_nodes) - 1

    _lgbm_decision_type(node)
    feature_index = int(node["split_feature"])
    if not 0 <= feature_index <= 0xFE:
        raise ValueError(f"LightGBM split feature {feature_index} exceeds the compact uint8 limit.")

    missing_type = str(node.get("missing_type", "None"))
    if missing_type not in {"None", "NaN", "Zero"}:
        raise ValueError(f"Unsupported LightGBM missing_type={missing_type!r}.")
    if missing_type == "Zero" and affine_folded:
        # With shifted affine preprocessing, LightGBM's transformed zero no longer
        # corresponds to raw zero. The 8-byte node has no room for a per-node raw
        # zero sentinel, so reject instead of silently routing missing values wrong.
        transformed_raw_zero = scalar_transform(feature_index, np.float32(0.0))
        if transformed_raw_zero != np.float32(0.0):
            raise ValueError(
                "LightGBM zero_as_missing cannot be folded through shifted affine preprocessing "
                f"for feature {feature_index} (raw 0 transforms to {float(transformed_raw_zero):.9g}). "
                "Disable zero_as_missing or use a zero-preserving preprocessing mode."
            )

    current_idx = len(flat_nodes)
    flat_nodes.append(None)
    left_idx = _flatten_lgbm_tree(
        node["left_child"], flat_nodes, affine_a, affine_b, scalar_transform, affine_folded
    )
    right_idx = _flatten_lgbm_tree(
        node["right_child"], flat_nodes, affine_a, affine_b, scalar_transform, affine_folded
    )

    raw_threshold = _fold_threshold(
        float(node["threshold"]),
        feature_index,
        affine_a,
        affine_b,
        scalar_transform,
        affine_folded=affine_folded,
    )
    flat_nodes[current_idx] = LgbmNode(
        threshold=raw_threshold,
        left_child=left_idx,
        right_child=right_idx,
        feature_index=feature_index,
        default_left=bool(node.get("default_left", True)),
        missing_is_zero=missing_type == "Zero",
        is_leaf=False,
    )
    return current_idx


def _build_lgbm_trees(
    model_obj: Any,
    affine_a: np.ndarray,
    affine_b: np.ndarray,
    scalar_transform: Callable[[int, np.float32], np.float32],
    affine_folded: bool,
) -> Tuple[Mapping[str, Any], List[List[LgbmNode]]]:
    booster = getattr(model_obj, "booster_", None)
    if booster is None:
        raise ValueError("LightGBM model is not fitted (booster_ is missing).")
    model_dump = booster.dump_model()
    if bool(model_dump.get("average_output", False)):
        raise ValueError("LightGBM average_output models are not supported by this compact sum-of-trees exporter.")

    def lgbm_scalar_transform(feature_index: int, raw_value: np.float32) -> np.float32:
        return _lgbm_canonicalize_zero(scalar_transform(feature_index, raw_value))

    trees: List[List[LgbmNode]] = []
    for tree_info in model_dump.get("tree_info", []):
        flat: List[LgbmNode | None] = []
        root_index = _flatten_lgbm_tree(
            tree_info["tree_structure"],
            flat,
            affine_a,
            affine_b,
            lgbm_scalar_transform,
            affine_folded,
        )
        if root_index != 0:
            raise AssertionError("Flattened LightGBM root must be node 0.")
        if len(flat) > 255:
            raise ValueError(
                f"LightGBM tree {tree_info.get('tree_index', len(trees))} has {len(flat)} nodes; "
                "uint8 child indices permit at most 255 nodes."
            )
        if any(node is None for node in flat):
            raise AssertionError("LightGBM flattening left an unfilled node placeholder.")
        trees.append([node for node in flat if node is not None])

    if not trees:
        raise ValueError("LightGBM dump contains no trees.")
    return model_dump, trees


def _walk_lgbm_dump_tree(node: Mapping[str, Any], row: np.ndarray) -> float:
    current = node
    while "leaf_value" not in current:
        _lgbm_decision_type(current)
        feature_index = int(current["split_feature"])
        value = float(_lgbm_canonicalize_zero(row[feature_index]))
        missing_type = str(current.get("missing_type", "None"))
        is_missing = math.isnan(value) or (missing_type == "Zero" and value == 0.0)
        if is_missing:
            go_left = bool(current.get("default_left", True))
        else:
            go_left = value <= float(current["threshold"])
        current = current["left_child"] if go_left else current["right_child"]
    return float(current["leaf_value"])


def _lgbm_init_score(model_obj: Any, model_dump: Mapping[str, Any]) -> float:
    n_features = int(model_obj.n_features_in_)
    reference = np.zeros((1, n_features), dtype=np.float64)
    raw_prediction = float(model_obj.booster_.predict(reference, raw_score=True)[0])
    tree_sum = sum(
        _walk_lgbm_dump_tree(tree_info["tree_structure"], reference[0])
        for tree_info in model_dump["tree_info"]
    )
    return _float32(raw_prediction - tree_sum)


def _lgbm_sigmoid_parameter(model_obj: Any, model_dump: Mapping[str, Any]) -> float:
    params = model_obj.get_params()
    if params.get("sigmoid") is not None:
        value = float(params["sigmoid"])
    else:
        objective = str(model_dump.get("objective", params.get("objective", "")))
        match = re.search(r"(?:^|\s)sigmoid:([+\-0-9.eE]+)", objective)
        value = float(match.group(1)) if match else 1.0
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid LightGBM sigmoid parameter: {value}.")
    return _float32(value)


def _format_c_array(c_type: str, name: str, values: Sequence[str], *, per_line: int = 12) -> str:
    if not values:
        raise ValueError(f"Cannot emit empty C array {name}.")
    chunks = [", ".join(values[index : index + per_line]) for index in range(0, len(values), per_line)]
    body = ",\n    ".join(chunks)
    return f"static const {c_type} {name}[{len(values)}] = {{\n    {body}\n}};\n"


def _tree_roots(tree_collection: Sequence[Sequence[Any]]) -> Tuple[List[int], int]:
    roots: List[int] = []
    offset = 0
    for tree in tree_collection:
        roots.append(offset)
        offset += len(tree)
    return roots, offset


def _sklearn_storage_code(
    trees: Sequence[Sequence[SklearnNode]], group_var: str
) -> StorageLayout:
    roots, total_nodes = _tree_roots(trees)
    max_tree_nodes = max(len(tree) for tree in trees)
    if max_tree_nodes > np.iinfo(np.uint16).max:
        raise ValueError("sklearn tree exceeds uint16 local child indexing.")
    root_c_type, root_bytes = _root_index_storage(total_nodes)

    flat = [node for tree in trees for node in tree]
    thresholds = [_c_float(node.threshold) for node in flat]
    root_values = [f"{value}u" for value in roots]

    if max_tree_nodes <= np.iinfo(np.uint8).max:
        # Balanced Cortex-M4 layout: one aligned uint16 metadata load contains both
        # model-local feature index and right-child index. Flash remains 6 bytes/node.
        metadata = [
            f"0x{(((int(node.right_child) & 0xFF) << 8) | (0xFF if node.is_leaf else int(node.feature_index))):04X}u"
            for node in flat
        ]
        code = "\n".join(
            [
                _format_c_array("float", f"{group_var}_node_thresholds", thresholds, per_line=6).rstrip(),
                _format_c_array(root_c_type, f"{group_var}_tree_roots", root_values).rstrip(),
                _format_c_array("uint16_t", f"{group_var}_node_metadata", metadata, per_line=10).rstrip(),
                "",
            ]
        )
        child_c_type = "uint8_t"
        child_bytes = 1
        metadata_layout = "rf_u8_child_feature_packed_u16"
        node_bytes = RF_U8_NODE_BYTES
    else:
        right_children = [f"{int(node.right_child)}u" for node in flat]
        features = ["0xFFu" if node.is_leaf else f"{int(node.feature_index)}u" for node in flat]
        code = "\n".join(
            [
                _format_c_array("float", f"{group_var}_node_thresholds", thresholds, per_line=6).rstrip(),
                _format_c_array(root_c_type, f"{group_var}_tree_roots", root_values).rstrip(),
                _format_c_array("uint16_t", f"{group_var}_node_right_children", right_children).rstrip(),
                _format_c_array("uint8_t", f"{group_var}_node_features", features).rstrip(),
                "",
            ]
        )
        child_c_type = "uint16_t"
        child_bytes = 2
        metadata_layout = "rf_u16_child_separate_feature"
        node_bytes = RF_U16_NODE_BYTES

    node_data_bytes = total_nodes * node_bytes
    root_table_bytes = len(trees) * root_bytes
    return StorageLayout(
        code=code,
        total_nodes=total_nodes,
        max_tree_nodes=max_tree_nodes,
        root_c_type=root_c_type,
        root_bytes=root_bytes,
        child_c_type=child_c_type,
        child_bytes=child_bytes,
        metadata_layout=metadata_layout,
        node_data_bytes=node_data_bytes,
        root_table_bytes=root_table_bytes,
        tree_payload_bytes=node_data_bytes + root_table_bytes,
        legacy_8byte_payload_bytes=total_nodes * 8,
    )


def _sklearn_multioutput_storage_code(
    trees: Sequence[Sequence[SklearnNode]],
    extra_probabilities: Sequence[Sequence[Sequence[float]]],
    group_var: str,
    n_outputs: int,
) -> StorageLayout:
    """Reuse the legacy RF node layout and append per-node probabilities for outputs 1..N-1."""

    base = _sklearn_storage_code(trees, group_var)
    if len(extra_probabilities) != len(trees):
        raise ValueError("Multi-output RF extra-probability tree count mismatch.")

    flattened: List[str] = []
    expected_width = n_outputs - 1
    for tree, values in zip(trees, extra_probabilities):
        if len(values) != len(tree):
            raise ValueError("Multi-output RF extra-probability node count mismatch.")
        for vector in values:
            if len(vector) != expected_width:
                raise ValueError("Multi-output RF probability width mismatch.")
            flattened.extend(_c_float(value) for value in vector)

    if not flattened:
        raise ValueError("Multi-output RF probability side table is empty.")
    side_code = _format_c_array(
        "float",
        f"{group_var}_node_extra_probabilities",
        flattened,
        per_line=6,
    ).rstrip()
    extra_bytes = len(flattened) * 4
    return StorageLayout(
        code=base.code.rstrip() + "\n" + side_code + "\n",
        total_nodes=base.total_nodes,
        max_tree_nodes=base.max_tree_nodes,
        root_c_type=base.root_c_type,
        root_bytes=base.root_bytes,
        child_c_type=base.child_c_type,
        child_bytes=base.child_bytes,
        metadata_layout=base.metadata_layout + "_multioutput_side_table",
        node_data_bytes=base.node_data_bytes + extra_bytes,
        root_table_bytes=base.root_table_bytes,
        tree_payload_bytes=base.tree_payload_bytes + extra_bytes,
        legacy_8byte_payload_bytes=base.legacy_8byte_payload_bytes,
        flag_storage_bytes=base.flag_storage_bytes,
        feature_metadata_layout=base.feature_metadata_layout,
        default_left_encoding=base.default_left_encoding,
        missing_is_zero_encoding=base.missing_is_zero_encoding,
    )


def _lgbm_storage_code(
    trees: Sequence[Sequence[LgbmNode]], group_var: str
) -> StorageLayout:
    roots, total_nodes = _tree_roots(trees)
    max_tree_nodes = max(len(tree) for tree in trees)
    if max_tree_nodes > np.iinfo(np.uint8).max:
        raise ValueError("LightGBM tree exceeds uint8 local child indexing.")
    root_c_type, root_bytes = _root_index_storage(total_nodes)

    flat: List[LgbmNode] = []
    for tree_index, tree in enumerate(trees):
        for node_index, node in enumerate(tree):
            if not node.is_leaf and node.left_child != node_index + 1:
                raise AssertionError(
                    f"LightGBM tree {tree_index}: left child {node.left_child} is not implicit "
                    f"node+1 for node {node_index}."
                )
        flat.extend(tree)

    thresholds = [_c_float(node.threshold) for node in flat]
    root_values = [f"{value}u" for value in roots]
    flag_storage_bytes = 0
    feature_metadata_layout = "plain_u8"
    default_left_encoding = "child_bit6"
    missing_is_zero_encoding = "child_bit7"

    if max_tree_nodes <= 64:
        features = ["0xFFu" if node.is_leaf else f"{int(node.feature_index)}u" for node in flat]
        child_flags = []
        for node in flat:
            if node.is_leaf:
                packed = 0
            else:
                if int(node.right_child) > 63:
                    raise AssertionError("Packed LightGBM child index exceeds six bits.")
                packed = (
                    int(node.right_child)
                    | ((1 if node.default_left else 0) << 6)
                    | ((1 if node.missing_is_zero else 0) << 7)
                )
            child_flags.append(f"0x{packed:02X}u")
        code = "\n".join(
            [
                _format_c_array("float", f"{group_var}_node_thresholds", thresholds, per_line=6).rstrip(),
                _format_c_array(root_c_type, f"{group_var}_tree_roots", root_values).rstrip(),
                _format_c_array("uint8_t", f"{group_var}_node_child_flags", child_flags).rstrip(),
                _format_c_array("uint8_t", f"{group_var}_node_features", features).rstrip(),
                "",
            ]
        )
        metadata_layout = "lgbm_u6_child_flags_packed"
        node_data_bytes = total_nodes * LGBM_PACKED_NODE_BYTES
    else:
        plan = _lgbm_large_metadata_plan(flat)
        feature_metadata_layout = str(plan["feature_metadata_layout"])
        default_left_encoding = str(plan["default_left_encoding"])
        missing_is_zero_encoding = str(plan["missing_is_zero_encoding"])
        flag_storage_bytes = int(plan["flag_storage_bytes"])

        right_children = [f"{int(node.right_child)}u" for node in flat]
        features = [
            _lgbm_feature_metadata_literal(node, feature_metadata_layout)
            for node in flat
        ]
        arrays = [
            _format_c_array("float", f"{group_var}_node_thresholds", thresholds, per_line=6).rstrip(),
            _format_c_array(root_c_type, f"{group_var}_tree_roots", root_values).rstrip(),
            _format_c_array("uint8_t", f"{group_var}_node_right_children", right_children).rstrip(),
            _format_c_array("uint8_t", f"{group_var}_node_features", features).rstrip(),
        ]
        default_code = _bitset_array_code(
            group_var, "default_left", plan["default_left_bits"]
        )
        missing_code = _bitset_array_code(
            group_var, "missing_is_zero", plan["missing_is_zero_bits"]
        )
        if default_code:
            arrays.append(default_code)
        if missing_code:
            arrays.append(missing_code)
        arrays.append("")
        code = "\n".join(arrays)
        metadata_layout = "lgbm_u8_child_feature_metadata_bitpacked_flags"
        node_data_bytes = (
            total_nodes * LGBM_LARGE_BASE_NODE_BYTES + flag_storage_bytes
        )

    root_table_bytes = len(trees) * root_bytes
    return StorageLayout(
        code=code,
        total_nodes=total_nodes,
        max_tree_nodes=max_tree_nodes,
        root_c_type=root_c_type,
        root_bytes=root_bytes,
        child_c_type="uint8_t",
        child_bytes=1,
        metadata_layout=metadata_layout,
        node_data_bytes=node_data_bytes,
        root_table_bytes=root_table_bytes,
        tree_payload_bytes=node_data_bytes + root_table_bytes,
        legacy_8byte_payload_bytes=total_nodes * 8,
        flag_storage_bytes=flag_storage_bytes,
        feature_metadata_layout=feature_metadata_layout,
        default_left_encoding=default_left_encoding,
        missing_is_zero_encoding=missing_is_zero_encoding,
    )


def _compiler_optimization_code(context: ExportContext) -> str:
    """Compiler-portable Cortex-M4F inlining and branch-layout hints."""

    return f"""
#ifndef {context.group_macro}_FORCE_INLINE
  #if defined(__STATIC_FORCEINLINE)
    #define {context.group_macro}_FORCE_INLINE __STATIC_FORCEINLINE
  #elif defined(__GNUC__) || defined(__clang__)
    #define {context.group_macro}_FORCE_INLINE static inline __attribute__((always_inline))
  #else
    #define {context.group_macro}_FORCE_INLINE static inline
  #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define {context.group_macro}_LIKELY(value)   __builtin_expect(!!(value), 1)
  #define {context.group_macro}_UNLIKELY(value) __builtin_expect(!!(value), 0)
#else
  #define {context.group_macro}_LIKELY(value)   (value)
  #define {context.group_macro}_UNLIKELY(value) (value)
#endif
"""


def _helper_feature_slots(context: ExportContext) -> Dict[int, int]:
    slots: Dict[int, int] = {}
    for model_index, feature_name in enumerate(context.feature_names):
        if feature_expression_helpers(feature_name):
            slots[model_index] = len(slots)
    return slots


def _feature_accessor_code(context: ExportContext) -> str:
    helper_slots = _helper_feature_slots(context)
    cache_count = len(helper_slots)
    cache_words = (cache_count + 31) // 32
    cache_bytes = cache_count * 4 + cache_words * 4 if cache_count else 1
    lines = [
        f"#define {context.group_macro}_HELPER_CACHE_COUNT {cache_count}u",
        f"#define {context.group_macro}_HELPER_CACHE_BYTES {cache_bytes}u",
        "typedef struct {",
        *(
            [f"    float values[{cache_count}];", f"    uint32_t valid_mask[{cache_words}];"]
            if cache_count
            else ["    uint8_t unused;"]
        ),
        f"}} {context.group_var}_feature_cache_t;",
        "",
        "/** Return one selected feature; helper expressions are cached lazily once per prediction. */",
        f"{context.group_macro}_FORCE_INLINE float {context.group_var}_get_raw_model_feature(",
        f"    const t_core* p_core, uint16_t model_feature_index, {context.group_var}_feature_cache_t* cache)",
        "{",
        *( ["    (void)cache;"] if cache_count == 0 else [] ),
        "    switch (model_feature_index) {",
    ]
    for model_index, (original_index, feature_name) in enumerate(
        zip(context.selected_indices, context.feature_names)
    ):
        accessor = csv_key_to_c_accessor(feature_name)
        safe_comment = str(feature_name).replace(chr(10), " ").replace(chr(13), " ")
        if model_index in helper_slots:
            slot = helper_slots[model_index]
            word = slot // 32
            bit = slot % 32
            mask = 1 << bit
            lines.extend(
                [
                    f"        case {model_index}u:",
                    f"            if ((cache->valid_mask[{word}] & 0x{mask:08X}u) == 0u) {{",
                    f"                cache->values[{slot}] = (float)({accessor});",
                    f"                cache->valid_mask[{word}] |= 0x{mask:08X}u;",
                    "            }",
                    f"            return cache->values[{slot}]; "
                    f"// original[{original_index}] {safe_comment}",
                ]
            )
        else:
            lines.append(
                f"        case {model_index}u: return (float)({accessor}); "
                f"// original[{original_index}] {safe_comment}"
            )
    lines.extend(
        [
            "        default: return 0.0f;",
            "    }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _normalization_code(context: ExportContext) -> str:
    if context.runtime_normalization is None:
        return (
            f"{context.group_macro}_FORCE_INLINE float {context.group_var}_runtime_normalization_factor(\n"
            f"    const t_core* p_core, {context.group_var}_feature_cache_t* cache)\n"
            "{\n"
            "    (void)p_core;\n"
            "    (void)cache;\n"
            "    return 1.0f;\n"
            "}\n\n"
        )

    mode = context.runtime_normalization
    lines = [
        f"{context.group_macro}_FORCE_INLINE float {context.group_var}_runtime_normalization_factor(",
        f"    const t_core* p_core, {context.group_var}_feature_cache_t* cache)",
        "{",
    ]
    if mode == "l1":
        lines.extend(
            [
                "    float denominator = 0.0f;",
                f"    for (uint16_t i = 0u; i < {context.group_macro}_NUM_FEATURES; ++i) {{",
                f"        denominator += fabsf({context.group_var}_get_raw_model_feature(p_core, i, cache));",
                "    }",
            ]
        )
    elif mode == "l2":
        lines.extend(
            [
                "    float sum_squares = 0.0f;",
                f"    for (uint16_t i = 0u; i < {context.group_macro}_NUM_FEATURES; ++i) {{",
                f"        const float value = {context.group_var}_get_raw_model_feature(p_core, i, cache);",
                "        sum_squares += value * value;",
                "    }",
                "    const float denominator = sqrtf(sum_squares);",
            ]
        )
    elif mode == "max":
        lines.extend(
            [
                "    float denominator = 0.0f;",
                f"    for (uint16_t i = 0u; i < {context.group_macro}_NUM_FEATURES; ++i) {{",
                f"        const float magnitude = fabsf({context.group_var}_get_raw_model_feature(p_core, i, cache));",
                "        if (magnitude > denominator) { denominator = magnitude; }",
                "    }",
            ]
        )
    else:  # pragma: no cover - context construction guarantees the set.
        raise AssertionError(f"Unknown runtime normalization mode: {mode}")

    lines.extend(
        [
            "    return denominator > 1.0e-20f ? (1.0f / denominator) : 1.0f;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _walker_code(context: ExportContext, storage: StorageLayout) -> str:
    root_type = storage.root_c_type
    inline = f"{context.group_macro}_FORCE_INLINE"
    unlikely = f"{context.group_macro}_UNLIKELY"

    if context.model_type == "sklearn":
        if storage.metadata_layout == "rf_u8_child_feature_packed_u16":
            metadata_load = f"""
        const uint16_t metadata = {context.group_var}_node_metadata[absolute_index];
        const uint8_t feature_index = (uint8_t)(metadata & 0x00FFu);
        if ({unlikely}(feature_index == 0xFFu)) {{
            return {context.group_var}_node_thresholds[absolute_index];
        }}
        const float feature_value =
            {context.group_var}_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;
        node_index = feature_value <= {context.group_var}_node_thresholds[absolute_index]
            ? (uint16_t)(node_index + 1u)
            : (uint16_t)(metadata >> 8);
"""
        else:
            metadata_load = f"""
        const uint8_t feature_index = {context.group_var}_node_features[absolute_index];
        if ({unlikely}(feature_index == 0xFFu)) {{
            return {context.group_var}_node_thresholds[absolute_index];
        }}
        const float feature_value =
            {context.group_var}_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;
        node_index = feature_value <= {context.group_var}_node_thresholds[absolute_index]
            ? (uint16_t)(node_index + 1u)
            : (uint16_t){context.group_var}_node_right_children[absolute_index];
"""
        return f"""
{inline} float {context.group_var}_walk_sklearn_tree(
    {root_type} root_index,
    const t_core* p_core,
    float normalization_factor,
    {context.group_var}_feature_cache_t* cache)
{{
    uint16_t node_index = 0u;
    for (;;) {{
        const uint32_t absolute_index = (uint32_t)root_index + (uint32_t)node_index;{metadata_load}    }}
}}
"""

    if storage.metadata_layout in {"lgbm_u6_child_flags_packed", "xgb_u6_child_flags_packed"}:
        feature_decode = f"""
        const uint8_t feature_metadata = {context.group_var}_node_features[absolute_index];
        if ({unlikely}(feature_metadata == 0xFFu)) {{
            return {context.group_var}_node_thresholds[absolute_index];
        }}
        const uint8_t feature_index = feature_metadata;
"""
        metadata_decode = f"""
        const uint8_t child_flags = {context.group_var}_node_child_flags[absolute_index];
        const uint8_t right_child = (uint8_t)(child_flags & 0x3Fu);
        const uint8_t default_left = (uint8_t)((child_flags >> 6) & 0x01u);
        const uint8_t missing_is_zero = (uint8_t)(child_flags >> 7);
"""
    else:
        layout = storage.feature_metadata_layout
        if layout == "u6_feature_default_left_missing_is_zero":
            feature_decode = f"""
        const uint8_t feature_metadata = {context.group_var}_node_features[absolute_index];
        if ({unlikely}(feature_metadata == 0xFFu)) {{
            return {context.group_var}_node_thresholds[absolute_index];
        }}
        const uint8_t feature_index = (uint8_t)(feature_metadata & 0x3Fu);
"""
        elif layout in {"u7_feature_default_left", "u7_feature_missing_is_zero"}:
            feature_decode = f"""
        const uint8_t feature_metadata = {context.group_var}_node_features[absolute_index];
        if ({unlikely}(feature_metadata == 0xFFu)) {{
            return {context.group_var}_node_thresholds[absolute_index];
        }}
        const uint8_t feature_index = (uint8_t)(feature_metadata & 0x7Fu);
"""
        elif layout == "plain_u8":
            feature_decode = f"""
        const uint8_t feature_metadata = {context.group_var}_node_features[absolute_index];
        if ({unlikely}(feature_metadata == 0xFFu)) {{
            return {context.group_var}_node_thresholds[absolute_index];
        }}
        const uint8_t feature_index = feature_metadata;
"""
        else:
            raise AssertionError(f"Unknown LightGBM feature metadata layout: {layout}")

        def flag_decode(flag_name: str, encoding: str) -> str:
            if encoding == "constant_0":
                return f"        const uint8_t {flag_name} = 0u;"
            if encoding == "constant_1":
                return f"        const uint8_t {flag_name} = 1u;"
            if encoding == "feature_bit6":
                return f"        const uint8_t {flag_name} = (uint8_t)((feature_metadata >> 6) & 0x01u);"
            if encoding == "feature_bit7":
                return f"        const uint8_t {flag_name} = (uint8_t)(feature_metadata >> 7);"
            if encoding == "bitset":
                return (
                    f"        const uint8_t {flag_name} = (uint8_t)(("
                    f"{context.group_var}_node_{flag_name}_bits[absolute_index >> 3u] >> "
                    f"(absolute_index & 7u)) & 0x01u);"
                )
            raise AssertionError(f"Unknown {flag_name} encoding: {encoding}")

        metadata_decode = "\n".join(
            [
                f"        const uint8_t right_child = {context.group_var}_node_right_children[absolute_index];",
                flag_decode("default_left", storage.default_left_encoding),
                flag_decode("missing_is_zero", storage.missing_is_zero_encoding),
                "",
            ]
        )

    boosted_walker = "walk_xgb_tree" if context.model_type == "xgb" else "walk_lgbm_tree"
    zero_canonicalization = "" if context.model_type == "xgb" else f"""
        if (feature_value >= -1.0e-35f && feature_value <= 1.0e-35f) {{
            feature_value = 0.0f;
        }}"""
    return f"""
{inline} float {context.group_var}_{boosted_walker}(
    {root_type} root_index,
    const t_core* p_core,
    float normalization_factor,
    {context.group_var}_feature_cache_t* cache)
{{
    uint8_t node_index = 0u;
    for (;;) {{
        const uint32_t absolute_index = (uint32_t)root_index + (uint32_t)node_index;{feature_decode}
        float feature_value =
            {context.group_var}_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;
{zero_canonicalization}{metadata_decode}
        const uint8_t is_missing = (uint8_t)(
            (feature_value != feature_value) || ((missing_is_zero != 0u) && feature_value == 0.0f));

        if ({unlikely}(is_missing != 0u)) {{
            node_index = default_left != 0u
                ? (uint8_t)(node_index + 1u)
                : right_child;
        }} else {{
            node_index = feature_value <= {context.group_var}_node_thresholds[absolute_index]
                ? (uint8_t)(node_index + 1u)
                : right_child;
        }}
    }}
}}
"""


def _sklearn_multioutput_walker_code(
    context: ExportContext,
    storage: StorageLayout,
    n_outputs: int,
) -> str:
    """One RF traversal that accumulates all output probabilities at the reached leaf."""

    inline = f"{context.group_macro}_FORCE_INLINE"
    unlikely = f"{context.group_macro}_UNLIKELY"
    extra_width = n_outputs - 1
    if storage.child_bytes == 1:
        metadata = f"""
        const uint16_t metadata = {context.group_var}_node_metadata[absolute_index];
        const uint8_t feature_index = (uint8_t)(metadata & 0x00FFu);
        if ({unlikely}(feature_index == 0xFFu)) {{
            probabilities[0] += {context.group_var}_node_thresholds[absolute_index];
            for (uint16_t output = 1u; output < {context.group_macro}_NUM_OUTPUTS; ++output) {{
                probabilities[output] += {context.group_var}_node_extra_probabilities[
                    absolute_index * {extra_width}u + (uint32_t)(output - 1u)];
            }}
            return;
        }}
        const float feature_value =
            {context.group_var}_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;
        node_index = feature_value <= {context.group_var}_node_thresholds[absolute_index]
            ? (uint16_t)(node_index + 1u)
            : (uint16_t)(metadata >> 8);
"""
    else:
        metadata = f"""
        const uint8_t feature_index = {context.group_var}_node_features[absolute_index];
        if ({unlikely}(feature_index == 0xFFu)) {{
            probabilities[0] += {context.group_var}_node_thresholds[absolute_index];
            for (uint16_t output = 1u; output < {context.group_macro}_NUM_OUTPUTS; ++output) {{
                probabilities[output] += {context.group_var}_node_extra_probabilities[
                    absolute_index * {extra_width}u + (uint32_t)(output - 1u)];
            }}
            return;
        }}
        const float feature_value =
            {context.group_var}_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;
        node_index = feature_value <= {context.group_var}_node_thresholds[absolute_index]
            ? (uint16_t)(node_index + 1u)
            : (uint16_t){context.group_var}_node_right_children[absolute_index];
"""

    return f"""
{inline} void {context.group_var}_walk_sklearn_tree_multioutput(
    {storage.root_c_type} root_index,
    const t_core* p_core,
    float normalization_factor,
    {context.group_var}_feature_cache_t* cache,
    float probabilities[{context.group_macro}_NUM_OUTPUTS])
{{
    uint16_t node_index = 0u;
    for (;;) {{
        const uint32_t absolute_index = (uint32_t)root_index + (uint32_t)node_index;{metadata}    }}
}}
"""


def _lgbm_raw_decision_threshold(probability_threshold: float, sigmoid: float) -> float:
    threshold = float(probability_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "LightGBM raw threshold requires probability strictly between 0 and 1."
        )
    return _float32(math.log(threshold / (1.0 - threshold)) / float(sigmoid))


def _lgbm_probability_tables(sigmoid: float) -> Tuple[List[float], List[float]]:
    """Return exact Q20 thresholds and nearest-step midpoint boundaries."""

    if LIGHTGBM_PROBABILITY_LEVELS != 20 or not math.isclose(
        LIGHTGBM_PROBABILITY_STEP, 0.05, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("The Cortex-M4 balanced exporter currently requires Q20 / 0.05 steps.")
    exact = [
        _lgbm_raw_decision_threshold(index / 20.0, sigmoid)
        for index in range(1, 20)
    ]
    midpoints = [
        _lgbm_raw_decision_threshold((index + 0.5) / 20.0, sigmoid)
        for index in range(20)
    ]
    return exact, midpoints


def _lgbm_probability_lut_code(group_var: str, sigmoid: float) -> str:
    exact, midpoints = _lgbm_probability_tables(sigmoid)
    return "\n".join(
        [
            "/* Q20 probability LUT: one unit equals 0.05 probability. */",
            _format_c_array(
                "float",
                f"{group_var}_q20_raw_thresholds",
                [_c_float(value) for value in exact],
                per_line=5,
            ).rstrip(),
            _format_c_array(
                "float",
                f"{group_var}_q20_raw_midpoints",
                [_c_float(value) for value in midpoints],
                per_line=5,
            ).rstrip(),
            "",
        ]
    )


def _outward_float32(value: float, *, upward: bool) -> float:
    candidate = np.float32(value)
    direction = np.float32(np.inf if upward else -np.inf)
    if (upward and float(candidate) < value) or ((not upward) and float(candidate) > value):
        candidate = np.nextafter(candidate, direction, dtype=np.float32)
    return float(candidate)


def _lgbm_early_exit_code(
    trees: Sequence[Sequence[LgbmNode]], group_var: str
) -> Tuple[str, int]:
    """Emit conservative suffix leaf bounds checked once per small tree group."""

    group_size = max(1, int(LIGHTGBM_EARLY_EXIT_GROUP_SIZE))
    tree_mins = [min(float(node.threshold) for node in tree if node.is_leaf) for tree in trees]
    tree_maxs = [max(float(node.threshold) for node in tree if node.is_leaf) for tree in trees]
    group_count = (len(trees) + group_size - 1) // group_size

    suffix_min = [0.0] * (group_count + 1)
    suffix_max = [0.0] * (group_count + 1)
    epsilon = np.finfo(np.float32).eps
    for group in range(group_count - 1, -1, -1):
        begin = group * group_size
        end = min(len(trees), begin + group_size)
        min_terms = tree_mins[begin:end]
        max_terms = tree_maxs[begin:end]
        raw_min = sum(min_terms) + suffix_min[group + 1]
        raw_max = sum(max_terms) + suffix_max[group + 1]
        # Allow for sequential float32 accumulation before rounding outward.
        remaining_terms = len(trees) - begin
        min_margin = (sum(abs(x) for x in tree_mins[begin:]) + 1.0) * remaining_terms * epsilon * 2.0
        max_margin = (sum(abs(x) for x in tree_maxs[begin:]) + 1.0) * remaining_terms * epsilon * 2.0
        suffix_min[group] = _outward_float32(raw_min - min_margin, upward=False)
        suffix_max[group] = _outward_float32(raw_max + max_margin, upward=True)

    code = "\n".join(
        [
            f"#define {group_var.upper()}_EARLY_EXIT_GROUP_SIZE {group_size}u",
            f"#define {group_var.upper()}_EARLY_EXIT_GROUPS {group_count}u",
            _format_c_array(
                "float",
                f"{group_var}_remaining_min",
                [_c_float(value) for value in suffix_min],
                per_line=6,
            ).rstrip(),
            _format_c_array(
                "float",
                f"{group_var}_remaining_max",
                [_c_float(value) for value in suffix_max],
                per_line=6,
            ).rstrip(),
            "",
        ]
    )
    return code, group_count


def _predictor_code(
    context: ExportContext,
    *,
    tree_count: int,
    storage: StorageLayout,
    lgbm_init_score: float = 0.0,
    lgbm_sigmoid: float = 1.0,
    lgbm_early_exit_groups: int = 0,
) -> str:
    lines: List[str] = []
    inline = f"{context.group_macro}_FORCE_INLINE"

    if context.model_type == "sklearn":
        lines.extend(
            [
                "/** Return the sum of tree outputs. */",
                f"{inline} float {context.group_var}_predict_sum(const t_core* p_core)",
                "{",
                f"    {context.group_var}_feature_cache_t cache = {{0}};",
                f"    const float normalization_factor = {context.group_var}_runtime_normalization_factor(p_core, &cache);",
                "    float sum = 0.0f;",
            ]
        )
        if tree_count == 1:
            lines.append(
                f"    sum = {context.group_var}_walk_sklearn_tree("
                f"{context.group_var}_tree_roots[0], p_core, normalization_factor, &cache);"
            )
        else:
            lines.extend(
                [
                    f"    for (uint16_t tree_index = 0u; tree_index < {context.group_macro}_NUM_TREES; ++tree_index) {{",
                    f"        sum += {context.group_var}_walk_sklearn_tree("
                    f"{context.group_var}_tree_roots[tree_index], p_core, normalization_factor, &cache);",
                    "    }",
                ]
            )
        sum_threshold = _float32(context.decision_threshold * tree_count)
        lines.extend(
            [
                "    return sum;",
                "}",
                "",
                "/** Return probability for classifiers or averaged RF regression output. */",
                f"{inline} float {context.group_var}_predict_value(const t_core* p_core)",
                "{",
                f"    return {context.group_var}_predict_sum(p_core) / {tree_count}.0f;",
                "}",
                "",
                "/** Binary path avoids the RF probability division. */",
                f"{inline} int8_t {context.group_var}_predict(const t_core* p_core)",
                "{",
                f"    return {context.group_var}_predict_sum(p_core) >= {_c_float(sum_threshold)} ? 1 : 0;",
                "}",
                "",
            ]
        )
        return "\n".join(lines)

    boosted_walker = "walk_xgb_tree" if context.model_type == "xgb" else "walk_lgbm_tree"
    lines.extend(
        [
            "/** Return boosted-tree raw score before the logistic link. */",
            f"{inline} float {context.group_var}_predict_raw_score(const t_core* p_core)",
            "{",
            f"    {context.group_var}_feature_cache_t cache = {{0}};",
            f"    const float normalization_factor = {context.group_var}_runtime_normalization_factor(p_core, &cache);",
            f"    float raw_score = {_c_float(lgbm_init_score)};",
        ]
    )
    if tree_count == 1:
        lines.append(
            f"    raw_score += {context.group_var}_{boosted_walker}("
            f"{context.group_var}_tree_roots[0], p_core, normalization_factor, &cache);"
        )
    else:
        lines.extend(
            [
                f"    for (uint16_t tree_index = 0u; tree_index < {context.group_macro}_NUM_TREES; ++tree_index) {{",
                f"        raw_score += {context.group_var}_{boosted_walker}("
                f"{context.group_var}_tree_roots[tree_index], p_core, normalization_factor, &cache);",
                "    }",
            ]
        )
    lines.extend(["    return raw_score;", "}", ""])

    if context.is_regression:
        # The boosted-regression C API keeps its original float and binary entry
        # points. Add the same Q20 call surface as boosted classifiers so an LP
        # model can be swapped without changing existing firmware call sites.
        # Regressor predictions are not necessarily probabilities: only this
        # optional Q20 adapter saturates them to [0, 1] before quantization.
        default_q20 = max(0, min(20, int(round(context.decision_threshold * 20.0))))
        lines.extend(
            [
                f"{inline} float {context.group_var}_predict_value(const t_core* p_core)",
                "{",
                f"    return {context.group_var}_predict_raw_score(p_core);",
                "}",
                "",
                "/** Q20 adapter: clamp continuous regression output to 0..1, then round to 0..20. */",
                f"{inline} uint8_t {context.group_var}_probability_q20(const t_core* p_core)",
                "{",
                f"    const float value = {context.group_var}_predict_value(p_core);",
                "    if (!(value > 0.0f)) { return 0u; }",
                "    if (value >= 1.0f) { return 20u; }",
                "    return (uint8_t)(value * 20.0f + 0.5f);",
                "}",
                "",
                "/** Compare unquantized regression output with an exact Q20 decision boundary. */",
                f"{inline} int8_t {context.group_var}_predict_q20(",
                "    const t_core* p_core, uint8_t threshold_q20)",
                "{",
                "    if (threshold_q20 == 0u) { return 1; }",
                "    if (threshold_q20 >= 20u) { return 0; }",
                f"    return {context.group_var}_predict_value(p_core) >= ",
                "        (float)threshold_q20 * 0.05f ? 1 : 0;",
                "}",
                "",
                f"#define {context.group_macro}_DEFAULT_THRESHOLD_Q20 {default_q20}u",
                f"{inline} int8_t {context.group_var}_predict(const t_core* p_core)",
                "{",
                f"    return {context.group_var}_predict_raw_score(p_core) >= "
                f"{context.group_macro}_THRESHOLD ? 1 : 0;",
                "}",
                "",
            ]
        )
        return "\n".join(lines)

    default_q20 = int(round(context.decision_threshold * 20.0))
    default_q20 = max(0, min(20, default_q20))
    lines.extend(
        [
            "/** Convert a raw score to nearest 0.05 probability, returned as Q20 [0..20]. */",
            f"{inline} uint8_t {context.group_var}_probability_q20_from_raw(float raw_score)",
            "{",
            "    uint8_t low = 0u;",
            "    uint8_t high = 20u;",
            "    while (low < high) {",
            "        const uint8_t middle = (uint8_t)((low + high) >> 1u);",
            f"        if (raw_score < {context.group_var}_q20_raw_midpoints[middle]) {{",
            "            high = middle;",
            "        } else {",
            "            low = (uint8_t)(middle + 1u);",
            "        }",
            "    }",
            "    return low;",
            "}",
            "",
            f"{inline} uint8_t {context.group_var}_probability_q20(const t_core* p_core)",
            "{",
            f"    return {context.group_var}_probability_q20_from_raw("
            f"{context.group_var}_predict_raw_score(p_core));",
            "}",
            "",
            "/** Approximate probability in 0.05 steps; maximum rounding error is 0.025. */",
            f"{inline} float {context.group_var}_predict_value(const t_core* p_core)",
            "{",
            f"    return 0.05f * (float){context.group_var}_probability_q20(p_core);",
            "}",
            "",
            "/** Runtime-selectable exact threshold for Q20 values 0.05..0.95; no expf/logf. */",
            f"{inline} int8_t {context.group_var}_predict_q20("
            "    const t_core* p_core, uint8_t threshold_q20)",
            "{",
            "    if (threshold_q20 == 0u) { return 1; }",
            "    if (threshold_q20 >= 20u) { return 0; }",
            f"    const float raw_threshold = {context.group_var}_q20_raw_thresholds[threshold_q20 - 1u];",
        ]
    )

    if lgbm_early_exit_groups > 0:
        lines.extend(
            [
                f"    {context.group_var}_feature_cache_t cache = {{0}};",
                f"    const float normalization_factor = {context.group_var}_runtime_normalization_factor(p_core, &cache);",
                f"    float raw_score = {_c_float(lgbm_init_score)};",
                f"    for (uint16_t group = 0u; group < {context.group_macro}_EARLY_EXIT_GROUPS; ++group) {{",
                f"        const uint16_t first_tree = (uint16_t)(group * {context.group_macro}_EARLY_EXIT_GROUP_SIZE);",
                f"        uint16_t end_tree = (uint16_t)(first_tree + {context.group_macro}_EARLY_EXIT_GROUP_SIZE);",
                f"        if (end_tree > {context.group_macro}_NUM_TREES) {{ end_tree = {context.group_macro}_NUM_TREES; }}",
                "        for (uint16_t tree_index = first_tree; tree_index < end_tree; ++tree_index) {",
                f"            raw_score += {context.group_var}_{boosted_walker}("
                f"{context.group_var}_tree_roots[tree_index], p_core, normalization_factor, &cache);",
                "        }",
                f"        if (raw_score + {context.group_var}_remaining_max[group + 1u] < raw_threshold) {{ return 0; }}",
                f"        if (raw_score + {context.group_var}_remaining_min[group + 1u] >= raw_threshold) {{ return 1; }}",
                "    }",
                "    return raw_score >= raw_threshold ? 1 : 0;",
            ]
        )
    else:
        lines.extend(
            [
                f"    return {context.group_var}_predict_raw_score(p_core) >= raw_threshold ? 1 : 0;",
            ]
        )

    lines.extend(
        [
            "}",
            "",
            f"#define {context.group_macro}_DEFAULT_THRESHOLD_Q20 {default_q20}u",
            f"{inline} int8_t {context.group_var}_predict(const t_core* p_core)",
            "{",
            f"    return {context.group_var}_predict_q20(p_core, {context.group_macro}_DEFAULT_THRESHOLD_Q20);",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _sklearn_multioutput_predictor_code(context: ExportContext, tree_count: int) -> str:
    inline = f"{context.group_macro}_FORCE_INLINE"
    return "\n".join(
        [
            "/** Traverse each RF tree once and return all positive-class probabilities. */",
            f"{inline} void {context.group_var}_predict_probabilities(",
            f"    const t_core* p_core, float probabilities[{context.group_macro}_NUM_OUTPUTS])",
            "{",
            f"    {context.group_var}_feature_cache_t cache = {{0}};",
            f"    const float normalization_factor = {context.group_var}_runtime_normalization_factor(p_core, &cache);",
            f"    for (uint16_t output = 0u; output < {context.group_macro}_NUM_OUTPUTS; ++output) {{",
            "        probabilities[output] = 0.0f;",
            "    }",
            f"    for (uint16_t tree_index = 0u; tree_index < {context.group_macro}_NUM_TREES; ++tree_index) {{",
            f"        {context.group_var}_walk_sklearn_tree_multioutput(",
            f"            {context.group_var}_tree_roots[tree_index], p_core, normalization_factor, &cache, probabilities);",
            "    }",
            f"    const float scale = 1.0f / {tree_count}.0f;",
            f"    for (uint16_t output = 0u; output < {context.group_macro}_NUM_OUTPUTS; ++output) {{",
            "        probabilities[output] *= scale;",
            "    }",
            "}",
            "",
        ]
    )


def _apply_python_preprocessor(preprocessor: Any | None, preprocessing: str, raw: np.ndarray) -> np.ndarray:
    data = np.asarray(raw, dtype=np.float32).copy(order="C")
    if preprocessing == "none":
        return data
    if preprocessing in AFFINE_PREPROCESSING or preprocessing in ROW_NORMALIZATION:
        if preprocessor is None:
            # Normalizer has no fitted numeric state; recreate exact behavior when
            # old containers stored only the mode name.
            norm = ROW_NORMALIZATION.get(preprocessing)
            if norm is None:
                raise ValueError(f"Missing fitted preprocessor for {preprocessing}.")
            preprocessor = Normalizer(norm=norm, copy=False).fit(data)
        transformed = preprocessor.transform(data)
        return np.asarray(transformed, dtype=np.float32)
    raise AssertionError(preprocessing)


def _runtime_norm_factor(row: np.ndarray, runtime_normalization: str | None) -> np.float32:
    if runtime_normalization is None:
        return np.float32(1.0)
    row32 = np.asarray(row, dtype=np.float32)
    if runtime_normalization == "l1":
        denominator = np.sum(np.abs(row32), dtype=np.float32)
    elif runtime_normalization == "l2":
        denominator = np.sqrt(np.sum(row32 * row32, dtype=np.float32), dtype=np.float32)
    elif runtime_normalization == "max":
        denominator = np.max(np.abs(row32), initial=np.float32(0.0))
    else:
        raise AssertionError(runtime_normalization)
    return np.float32(1.0 / denominator) if denominator > np.float32(1.0e-20) else np.float32(1.0)


def _walk_sklearn_nodes(nodes: Sequence[SklearnNode], row: np.ndarray, factor: np.float32) -> np.float32:
    index = 0
    row32 = np.asarray(row, dtype=np.float32)
    while True:
        node = nodes[index]
        if node.is_leaf:
            return np.float32(node.threshold)
        value = np.float32(row32[node.feature_index] * factor)
        index = index + 1 if value <= np.float32(node.threshold) else node.right_child


def _walk_lgbm_nodes(nodes: Sequence[LgbmNode], row: np.ndarray, factor: np.float32, *, xgb_mode: bool = False) -> np.float32:
    index = 0
    row32 = np.asarray(row, dtype=np.float32)
    while True:
        node = nodes[index]
        if node.is_leaf:
            return np.float32(node.threshold)
        value = np.float32(row32[node.feature_index] * factor)
        if not xgb_mode:
            value = _lgbm_canonicalize_zero(value)
        missing = bool(np.isnan(value) or (node.missing_is_zero and value == np.float32(0.0)))
        if missing:
            index = node.left_child if node.default_left else node.right_child
        else:
            index = node.left_child if value <= np.float32(node.threshold) else node.right_child


def _validation_samples(
    n_features: int,
    raw_thresholds: Mapping[int, Sequence[float]],
    preprocessing: str,
    preprocessor: Any | None,
    *,
    random_state: int = 71237,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    centers = np.zeros(n_features, dtype=np.float64)
    scales = np.ones(n_features, dtype=np.float64)

    if preprocessing in AFFINE_PREPROCESSING and preprocessor is not None:
        if hasattr(preprocessor, "mean_") and getattr(preprocessor, "mean_") is not None:
            values = np.asarray(preprocessor.mean_, dtype=np.float64).reshape(-1)
            if values.size == n_features:
                centers = values.copy()
        elif hasattr(preprocessor, "center_") and getattr(preprocessor, "center_") is not None:
            values = np.asarray(preprocessor.center_, dtype=np.float64).reshape(-1)
            if values.size == n_features:
                centers = values.copy()
        elif hasattr(preprocessor, "data_min_"):
            low = np.asarray(preprocessor.data_min_, dtype=np.float64).reshape(-1)
            high = np.asarray(preprocessor.data_max_, dtype=np.float64).reshape(-1)
            if low.size == n_features:
                centers = ((low + high) * 0.5).copy()
                scales = np.maximum((high - low) * 0.5, 1.0e-3).copy()

        if hasattr(preprocessor, "scale_") and getattr(preprocessor, "scale_") is not None:
            candidate = np.asarray(preprocessor.scale_, dtype=np.float64).reshape(-1)
            if candidate.size == n_features and isinstance(preprocessor, (StandardScaler, RobustScaler, MaxAbsScaler)):
                scales = np.maximum(np.abs(candidate), 1.0e-3).copy()

    for feature_index, thresholds in raw_thresholds.items():
        finite = np.asarray([x for x in thresholds if math.isfinite(x)], dtype=np.float64)
        if finite.size:
            centers[feature_index] = float(np.median(finite))
            spread = float(np.percentile(finite, 90) - np.percentile(finite, 10)) if finite.size > 1 else 0.0
            scales[feature_index] = max(spread, abs(centers[feature_index]) * 0.1, 1.0)

    samples = centers + rng.normal(size=(256, n_features)) * scales

    boundary_rows: List[np.ndarray] = []
    for feature_index, thresholds in raw_thresholds.items():
        for threshold in list(thresholds)[:4]:
            if not math.isfinite(threshold):
                continue
            epsilon = max(abs(threshold) * 2.0e-5, 2.0e-5)
            for value in (threshold - epsilon, threshold, threshold + epsilon):
                row = centers.copy()
                row[feature_index] = value
                boundary_rows.append(row)
    if boundary_rows:
        samples = np.vstack([samples, np.asarray(boundary_rows, dtype=np.float64)])

    return np.asarray(samples, dtype=np.float32)


def _validate_equivalence(
    *,
    model_obj: Any,
    model_type: str,
    is_regression: bool,
    positive_class_index: int,
    preprocessing: str,
    preprocessor: Any | None,
    runtime_normalization: str | None,
    sklearn_trees: Sequence[Sequence[SklearnNode]],
    lgbm_trees: Sequence[Sequence[LgbmNode]],
    lgbm_init_score: float,
    lgbm_sigmoid: float,
    decision_threshold: float,
    scalar_transform: Callable[[int, np.float32], np.float32],
) -> ExportValidation:
    raw_thresholds: Dict[int, List[float]] = {}
    tree_collection: Iterable[Sequence[Any]] = sklearn_trees if model_type == "sklearn" else lgbm_trees
    for tree_nodes in tree_collection:
        for node in tree_nodes:
            if not node.is_leaf:
                raw_thresholds.setdefault(node.feature_index, []).append(float(node.threshold))

    raw_samples = _validation_samples(
        int(model_obj.n_features_in_), raw_thresholds, preprocessing, preprocessor
    )
    if preprocessing in AFFINE_PREPROCESSING:
        transformed = np.empty_like(raw_samples, dtype=np.float32)
        for feature_index in range(raw_samples.shape[1]):
            transformed[:, feature_index] = np.fromiter(
                (scalar_transform(feature_index, value) for value in raw_samples[:, feature_index]),
                dtype=np.float32,
                count=raw_samples.shape[0],
            )
    else:
        transformed = _apply_python_preprocessor(preprocessor, preprocessing, raw_samples)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names.*",
            category=UserWarning,
        )
        if model_type == "lgbm":
            expected = np.asarray(model_obj.booster_.predict(transformed), dtype=np.float64)
        elif is_regression:
            expected = np.asarray(model_obj.predict(transformed), dtype=np.float64)
        else:
            expected = np.asarray(
                model_obj.predict_proba(transformed)[:, positive_class_index], dtype=np.float64
            )

    actual_values: List[float] = []
    fast_binary_decisions: List[bool] = []
    default_q20 = max(0, min(20, int(round(float(decision_threshold) * 20.0))))
    quantized_decision_threshold = default_q20 / 20.0
    q20_exact_thresholds: List[float] = []
    q20_midpoints: List[float] = []
    raw_decision_threshold = None
    if model_type in {"lgbm", "xgb"} and not is_regression:
        q20_exact_thresholds, q20_midpoints = _lgbm_probability_tables(lgbm_sigmoid)
        if 0 < default_q20 < 20:
            raw_decision_threshold = q20_exact_thresholds[default_q20 - 1]
    for row in raw_samples:
        factor = _runtime_norm_factor(row, runtime_normalization)
        if model_type == "sklearn":
            accumulator = np.float32(0.0)
            for tree_nodes in sklearn_trees:
                accumulator = np.float32(accumulator + _walk_sklearn_nodes(tree_nodes, row, factor))
            actual = np.float32(accumulator / np.float32(len(sklearn_trees)))
        else:
            accumulator = np.float32(lgbm_init_score)
            for tree_nodes in lgbm_trees:
                accumulator = np.float32(accumulator + _walk_lgbm_nodes(
                    tree_nodes, row, factor, xgb_mode=(model_type == "xgb")
                ))
            if is_regression:
                actual = accumulator
            else:
                if default_q20 == 0:
                    fast_binary_decisions.append(True)
                elif default_q20 >= 20:
                    fast_binary_decisions.append(False)
                else:
                    fast_binary_decisions.append(
                        bool(accumulator >= np.float32(raw_decision_threshold))
                    )
                q20 = int(np.searchsorted(
                    np.asarray(q20_midpoints, dtype=np.float32),
                    np.float32(accumulator),
                    side="right",
                ))
                actual = np.float32(q20 * LIGHTGBM_PROBABILITY_STEP)
        actual_values.append(float(actual))

    actual_array = np.asarray(actual_values, dtype=np.float64)
    errors = np.abs(expected - actual_array)
    disagreements = 0
    if not is_regression:
        comparison_threshold = (
            quantized_decision_threshold
            if model_type in {"lgbm", "xgb"}
            else decision_threshold
        )
        expected_decisions = expected >= comparison_threshold
        if model_type in {"lgbm", "xgb"}:
            # The Q20 probability API is intentionally approximate, but predict_q20()
            # compares against an exact inverse-sigmoid boundary for every 0.05 step.
            fast_array = np.asarray(fast_binary_decisions, dtype=bool)
            disagreements = int(np.count_nonzero(expected_decisions != fast_array))
        else:
            probability_decisions = actual_array >= comparison_threshold
            disagreements = int(np.count_nonzero(expected_decisions != probability_decisions))

    validation = ExportValidation(
        sample_count=int(raw_samples.shape[0]),
        max_abs_error=float(errors.max(initial=0.0)),
        mean_abs_error=float(errors.mean() if errors.size else 0.0),
        decision_disagreements=disagreements,
    )

    # Float32 nodes necessarily round sklearn/LightGBM's double thresholds. A
    # small numeric delta is acceptable; changed binary decisions are not.
    tolerance = (
        LIGHTGBM_PROBABILITY_STEP * 0.5 + 5.0e-6
        if model_type in {"lgbm", "xgb"} and not is_regression
        else 2.5e-4 if model_type in {"lgbm", "xgb"} else 7.5e-5
    )
    if validation.max_abs_error > tolerance or validation.decision_disagreements != 0:
        raise ValueError(
            "Generated C representation failed equivalence validation: "
            f"max_abs_error={validation.max_abs_error:.8g}, "
            f"decision_disagreements={validation.decision_disagreements}, "
            f"samples={validation.sample_count}."
        )
    return validation


def _validate_multioutput_rf_equivalence(
    *,
    model_obj: RandomForestClassifier,
    preprocessing: str,
    preprocessor: Any | None,
    runtime_normalization: str | None,
    sklearn_trees: Sequence[Sequence[SklearnNode]],
    extra_probabilities: Sequence[Sequence[Sequence[float]]],
    scalar_transform: Callable[[int, np.float32], np.float32],
) -> Dict[str, Any]:
    raw_thresholds: Dict[int, List[float]] = {}
    for tree_nodes in sklearn_trees:
        for node in tree_nodes:
            if not node.is_leaf:
                raw_thresholds.setdefault(node.feature_index, []).append(float(node.threshold))

    raw_samples = _validation_samples(
        int(model_obj.n_features_in_), raw_thresholds, preprocessing, preprocessor
    )
    if preprocessing in AFFINE_PREPROCESSING:
        transformed = np.empty_like(raw_samples, dtype=np.float32)
        for feature_index in range(raw_samples.shape[1]):
            transformed[:, feature_index] = np.fromiter(
                (scalar_transform(feature_index, value) for value in raw_samples[:, feature_index]),
                dtype=np.float32,
                count=raw_samples.shape[0],
            )
    else:
        transformed = _apply_python_preprocessor(preprocessor, preprocessing, raw_samples)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names.*",
            category=UserWarning,
        )
        python_outputs = model_obj.predict_proba(transformed)
    expected = np.column_stack(
        [np.asarray(output, dtype=np.float64)[:, 1] for output in python_outputs]
    )

    actual = np.zeros_like(expected, dtype=np.float64)
    for sample_index, row in enumerate(raw_samples):
        factor = _runtime_norm_factor(row, runtime_normalization)
        row32 = np.asarray(row, dtype=np.float32)
        accumulator = np.zeros(expected.shape[1], dtype=np.float32)
        for nodes, tree_extras in zip(sklearn_trees, extra_probabilities):
            node_index = 0
            while True:
                node = nodes[node_index]
                if node.is_leaf:
                    accumulator[0] = np.float32(accumulator[0] + np.float32(node.threshold))
                    extras = tree_extras[node_index]
                    for output_index, value in enumerate(extras, start=1):
                        accumulator[output_index] = np.float32(
                            accumulator[output_index] + np.float32(value)
                        )
                    break
                value = np.float32(row32[node.feature_index] * factor)
                node_index = (
                    node_index + 1
                    if value <= np.float32(node.threshold)
                    else node.right_child
                )
        actual[sample_index] = accumulator.astype(np.float64) / float(len(sklearn_trees))

    errors = np.abs(expected - actual)
    result = {
        "sample_count": int(raw_samples.shape[0]),
        "output_count": int(expected.shape[1]),
        "max_abs_error": float(errors.max(initial=0.0)),
        "mean_abs_error": float(errors.mean() if errors.size else 0.0),
    }
    if result["max_abs_error"] > 7.5e-5:
        raise ValueError(
            "Generated multi-output RF representation failed equivalence validation: "
            f"max_abs_error={result['max_abs_error']:.8g}, "
            f"samples={result['sample_count']}."
        )
    return result


def _export_to_c_header_legacy(
    model: Mapping[str, Any],
    trial_info: Mapping[str, Any],
    output_path: str | Path,
    feature_schema_path: str | Path,
    *,
    validate: bool = True,
) -> Dict[str, Any]:
    """Generate and validate a compact C header.

    The return value is a machine-readable export report. Existing callers may
    ignore it; the function signature remains compatible with the old script.
    """

    schema_path = Path(feature_schema_path)
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if "feature_map" not in schema:
        raise KeyError(f"Feature schema {schema_path} does not contain 'feature_map'.")
    feature_map = {int(key): str(value) for key, value in schema["feature_map"].items()}
    if not feature_map:
        raise ValueError("Feature schema contains an empty feature_map.")
    schema_size = max(feature_map) + 1
    if set(feature_map) != set(range(schema_size)):
        raise ValueError("feature_map indices must be contiguous and start at 0.")

    model_obj, model_type, is_regression = get_model_and_type(model)
    selected_indices = _extract_selected_indices(model, trial_info, model_obj, schema_size)
    preprocessing, preprocessor = _resolve_preprocessing(model, trial_info)
    decision_threshold = _resolve_threshold(
        trial_info, model, is_regression=is_regression
    )
    target_class_raw = trial_info.get(
        "target_class_id",
        trial_info.get(
            "positive_class_id",
            schema.get("target_class_id", schema.get("positive_class_id")),
        ),
    )
    target_class_id = None if target_class_raw is None else int(target_class_raw)

    group_name = str(trial_info.get("group_name", "gesture"))
    group_var = _sanitize_c_identifier(group_name)
    group_macro = _sanitize_c_identifier(group_name, uppercase=True)
    feature_names = tuple(feature_map[index] for index in selected_indices)
    runtime_normalization = ROW_NORMALIZATION.get(preprocessing)

    context = ExportContext(
        group_name=group_name,
        group_var=group_var,
        group_macro=group_macro,
        preprocessing=preprocessing,
        runtime_normalization=runtime_normalization,
        selected_indices=selected_indices,
        feature_names=feature_names,
        decision_threshold=decision_threshold,
        is_regression=is_regression,
        model_type=model_type,
    )

    if preprocessing in AFFINE_PREPROCESSING:
        affine_a, affine_b = _affine_coefficients(
            preprocessing, preprocessor, len(selected_indices), selected_indices
        )
    else:
        affine_a = np.ones(len(selected_indices), dtype=np.float64)
        affine_b = np.zeros(len(selected_indices), dtype=np.float64)
    scalar_transform = _make_scalar_transform(
        preprocessing, preprocessor, len(selected_indices), selected_indices
    )
    affine_folded = preprocessing in AFFINE_PREPROCESSING

    n_outputs = int(getattr(model_obj, "n_outputs_", 1)) if model_type == "sklearn" else 1
    is_multioutput_rf = bool(
        model_type == "sklearn"
        and isinstance(model_obj, RandomForestClassifier)
        and n_outputs > 1
    )
    positive_class_index = 0
    multioutput_positive_indices: Tuple[int, ...] = ()
    if is_multioutput_rf:
        multioutput_positive_indices = _multioutput_binary_positive_indices(model_obj)
    elif not is_regression:
        positive_class_index = _ensure_binary_classifier(model_obj, model_type)

    sklearn_trees: List[List[SklearnNode]] = []
    multioutput_extra_probabilities: List[List[Tuple[float, ...]]] = []
    lgbm_trees: List[List[LgbmNode]] = []
    storage: StorageLayout | None = None
    lgbm_init_score = 0.0
    lgbm_sigmoid = 1.0
    lgbm_probability_code = ""
    lgbm_early_exit_code = ""
    lgbm_early_exit_groups = 0

    if model_type == "sklearn":
        estimators, _ = get_model_trees({"model": model_obj})
        for tree_idx, estimator in enumerate(estimators):
            if is_multioutput_rf:
                nodes, extras = _sklearn_multioutput_tree_nodes(
                    estimator.tree_,
                    tree_idx,
                    positive_class_indices=multioutput_positive_indices,
                    affine_a=affine_a,
                    affine_b=affine_b,
                    scalar_transform=scalar_transform,
                    affine_folded=affine_folded,
                )
                sklearn_trees.append(nodes)
                multioutput_extra_probabilities.append(extras)
            else:
                nodes = _sklearn_tree_nodes(
                    estimator.tree_,
                    tree_idx,
                    is_regression=is_regression,
                    positive_class_index=positive_class_index,
                    affine_a=affine_a,
                    affine_b=affine_b,
                    scalar_transform=scalar_transform,
                    affine_folded=affine_folded,
                )
                sklearn_trees.append(nodes)
        tree_count = len(sklearn_trees)
        storage = (
            _sklearn_multioutput_storage_code(
                sklearn_trees,
                multioutput_extra_probabilities,
                group_var,
                n_outputs,
            )
            if is_multioutput_rf
            else _sklearn_storage_code(sklearn_trees, group_var)
        )
    elif model_type in {"lgbm", "xgb"}:
        if model_type == "xgb":
            lgbm_trees = _build_xgb_trees(model_obj, scalar_transform, affine_folded)
            lgbm_init_score = (
                _xgb_regression_init_score(model_obj)
                if is_regression else _xgb_raw_init_score(model_obj)
            )
            lgbm_sigmoid = 1.0
        else:
            model_dump, lgbm_trees = _build_lgbm_trees(
                model_obj, affine_a, affine_b, scalar_transform, affine_folded
            )
            lgbm_init_score = _lgbm_init_score(model_obj, model_dump)
            if not is_regression:
                lgbm_sigmoid = _lgbm_sigmoid_parameter(model_obj, model_dump)
        tree_count = len(lgbm_trees)
        storage = _lgbm_storage_code(lgbm_trees, group_var)
        if model_type == "xgb":
            storage = replace(storage, metadata_layout=storage.metadata_layout.replace("lgbm_", "xgb_", 1))
        if not is_regression:
            if LIGHTGBM_ENABLE_Q20_PROBABILITY:
                lgbm_probability_code = _lgbm_probability_lut_code(group_var, lgbm_sigmoid)
            if LIGHTGBM_EARLY_EXIT_GROUP_SIZE > 0:
                lgbm_early_exit_code, lgbm_early_exit_groups = _lgbm_early_exit_code(
                    lgbm_trees, group_var
                )
    else:
        raise TypeError(f"Unsupported exporter family: {model_type}")

    if storage is None:
        raise AssertionError("C storage layout was not constructed.")

    probability_lut_bytes = (
        (19 + 20) * 4
        if model_type in {"lgbm", "xgb"} and not is_regression and LIGHTGBM_ENABLE_Q20_PROBABILITY
        else 0
    )
    early_exit_bounds_bytes = (
        (lgbm_early_exit_groups + 1) * 2 * 4 if lgbm_early_exit_groups > 0 else 0
    )
    total_static_data_bytes = (
        storage.tree_payload_bytes + probability_lut_bytes + early_exit_bounds_bytes
    )

    validation = None
    if validate:
        if is_multioutput_rf:
            validation = _validate_multioutput_rf_equivalence(
                model_obj=model_obj,
                preprocessing=preprocessing,
                preprocessor=preprocessor,
                runtime_normalization=runtime_normalization,
                sklearn_trees=sklearn_trees,
                extra_probabilities=multioutput_extra_probabilities,
                scalar_transform=scalar_transform,
            )
        else:
            validation = _validate_equivalence(
                model_obj=model_obj,
                model_type=model_type,
                is_regression=is_regression,
                positive_class_index=positive_class_index,
                preprocessing=preprocessing,
                preprocessor=preprocessor,
                runtime_normalization=runtime_normalization,
                sklearn_trees=sklearn_trees,
                lgbm_trees=lgbm_trees,
                lgbm_init_score=lgbm_init_score,
                lgbm_sigmoid=lgbm_sigmoid,
                decision_threshold=decision_threshold,
                scalar_transform=scalar_transform,
            )

    guard = f"{group_macro}_MODEL_H"
    task = (
        "MULTI_OUTPUT_BINARY_CLASSIFIER"
        if is_multioutput_rf
        else ("REGRESSOR" if is_regression else "BINARY_CLASSIFIER")
    )
    preprocessing_execution = (
        "runtime row normalization"
        if runtime_normalization is not None
        else "folded into raw-domain tree thresholds"
        if preprocessing in AFFINE_PREPROCESSING
        else "raw features; no preprocessing"
    )
    selected_comment = ", ".join(str(index) for index in selected_indices)
    required_feature_helpers = sorted({
        helper
        for feature_name in context.feature_names
        for helper in feature_expression_helpers(feature_name)
    })
    helper_feature_count = len(_helper_feature_slots(context))
    helper_cache_words = (helper_feature_count + 31) // 32
    helper_cache_bytes = (
        helper_feature_count * 4 + helper_cache_words * 4
        if helper_feature_count
        else 1
    )
    helper_comment = (
        "Required feature helpers/macros: none"
        if not required_feature_helpers
        else "Required feature helpers/macros: " + ", ".join(required_feature_helpers)
    )

    target_comment = (
        "Binary target class: not specified"
        if target_class_id is None
        else f"Binary target class_id: {target_class_id}"
    )
    target_define = (
        []
        if target_class_id is None or is_regression or is_multioutput_rf
        else [f"#define {group_macro}_TARGET_CLASS_ID {target_class_id}"]
    )

    helper_include = (
        []
        if not FEATURE_HELPERS_HEADER
        else [f'#include "{FEATURE_HELPERS_HEADER}"']
    )

    header_parts = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "/*",
        f" * Generated: {datetime.now(timezone.utc).isoformat()}",
        f" * Model platform: {model_type.upper()}",
        f" * Task: {task}",
        f" * {target_comment}",
        f" * Training preprocessing: {preprocessing}",
        f" * Firmware preprocessing: {preprocessing_execution}",
        f" * Original selected feature indices: {selected_comment}",
        f" * {helper_comment}",
        f" * Export profile: {EXPORT_PROFILE}",
        f" * Flash storage: {FLASH_STORAGE_LAYOUT}; {storage.tree_payload_bytes} tree-data bytes.",
        f" * Total generated static model data: {total_static_data_bytes} bytes.",
        f" * Metadata layout: {storage.metadata_layout}",
        f" * Feature metadata: {storage.feature_metadata_layout}; "
        f"default_left={storage.default_left_encoding}; "
        f"missing_is_zero={storage.missing_is_zero_encoding}; "
        f"flag bytes={storage.flag_storage_bytes}.",
        " */",
        "",
        "#include <math.h>",
        "#include <stdint.h>",
        '#include "core_type_def.h"',
        *helper_include,
        "",
        f"#define {group_macro}_NUM_FEATURES {len(selected_indices)}u",
        f"#define {group_macro}_NUM_TREES {tree_count}u",
        *([f"#define {group_macro}_NUM_OUTPUTS {n_outputs}u"] if is_multioutput_rf else []),
        f"#define {group_macro}_THRESHOLD {_c_float(decision_threshold)}",
        *target_define,
        f'#define {group_macro}_PREPROCESSING "{preprocessing}"',
        "",
        _compiler_optimization_code(context).strip("\n"),
        "",
        _feature_accessor_code(context).rstrip(),
        "",
        _normalization_code(context).rstrip(),
        "",
        storage.code.rstrip(),
        "",
        lgbm_probability_code.rstrip(),
        "" if not lgbm_probability_code else "",
        lgbm_early_exit_code.rstrip(),
        "" if not lgbm_early_exit_code else "",
        (
            _sklearn_multioutput_walker_code(context, storage, n_outputs)
            if is_multioutput_rf
            else _walker_code(context, storage)
        ).strip("\n"),
        "",
        (
            _sklearn_multioutput_predictor_code(context, tree_count)
            if is_multioutput_rf
            else _predictor_code(
                context,
                tree_count=tree_count,
                storage=storage,
                lgbm_init_score=lgbm_init_score,
                lgbm_sigmoid=lgbm_sigmoid,
                lgbm_early_exit_groups=lgbm_early_exit_groups,
            )
        ).rstrip(),
        "",
        f"#endif /* {guard} */",
        "",
    ]
    header = "\n".join(header_parts)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(header, encoding="utf-8", newline="\n")
    temporary.replace(destination)

    report: Dict[str, Any] = {
        "output_path": str(destination),
        "model_type": model_type,
        "task": task,
        "tree_count": tree_count,
        "output_count": n_outputs if is_multioutput_rf else 1,
        "selected_feature_count": len(selected_indices),
        "preprocessing": preprocessing,
        "firmware_preprocessing": preprocessing_execution,
        "decision_threshold": decision_threshold,
        "target_class_id": target_class_id,
        "required_feature_helpers": required_feature_helpers,
        "auto_legacy_helpers": bool(AUTO_ALLOW_LEGACY_PREFIXED_HELPERS),
        "feature_helper_call_style": FEATURE_HELPER_CALL_STYLE,
        "helper_feature_count": helper_feature_count,
        "helper_cache_bytes_per_prediction": helper_cache_bytes,
        "feature_helpers_header": FEATURE_HELPERS_HEADER,
        "export_profile": EXPORT_PROFILE,
        "storage_layout": FLASH_STORAGE_LAYOUT,
        "metadata_layout": storage.metadata_layout,
        "feature_metadata_layout": storage.feature_metadata_layout,
        "default_left_encoding": storage.default_left_encoding,
        "missing_is_zero_encoding": storage.missing_is_zero_encoding,
        "flag_storage_bytes": storage.flag_storage_bytes,
        "total_nodes": storage.total_nodes,
        "max_tree_nodes": storage.max_tree_nodes,
        "child_index_bytes": storage.child_bytes,
        "root_index_bytes": storage.root_bytes,
        "node_data_bytes": storage.node_data_bytes,
        "root_table_bytes": storage.root_table_bytes,
        "tree_payload_bytes": storage.tree_payload_bytes,
        "legacy_8byte_payload_bytes": storage.legacy_8byte_payload_bytes,
        "payload_savings_bytes": storage.legacy_8byte_payload_bytes - storage.tree_payload_bytes,
        "payload_savings_percent": (
            100.0 * (storage.legacy_8byte_payload_bytes - storage.tree_payload_bytes)
            / storage.legacy_8byte_payload_bytes
            if storage.legacy_8byte_payload_bytes else 0.0
        ),
        "binary_fast_path": bool(model_type in {"lgbm", "xgb"} and not is_regression),
        "lightgbm_probability_step": (
            LIGHTGBM_PROBABILITY_STEP if model_type == "lgbm" and not is_regression else None
        ),
        "xgboost_probability_step": (
            LIGHTGBM_PROBABILITY_STEP if model_type == "xgb" and not is_regression else None
        ),
        "lightgbm_probability_levels": (
            LIGHTGBM_PROBABILITY_LEVELS if model_type == "lgbm" and not is_regression else None
        ),
        "lightgbm_probability_lut_bytes": probability_lut_bytes,
        "lightgbm_early_exit_bounds_bytes": early_exit_bounds_bytes,
        "total_static_data_bytes": total_static_data_bytes,
        "lightgbm_exp_runtime": False,
        "lightgbm_early_exit_group_size": (
            LIGHTGBM_EARLY_EXIT_GROUP_SIZE if model_type == "lgbm" and not is_regression else 0
        ),
        "lightgbm_early_exit_groups": lgbm_early_exit_groups if model_type == "lgbm" else 0,
        "xgboost_early_exit_groups": lgbm_early_exit_groups if model_type == "xgb" else 0,
        "xgboost_raw_base_margin": lgbm_init_score if model_type == "xgb" else None,
        "boosted_probability_uses_q20": model_type in {"lgbm", "xgb"} and not is_regression,
    }
    if validation is not None:
        if isinstance(validation, Mapping):
            report["validation"] = dict(validation)
        else:
            report["validation"] = {
                "sample_count": validation.sample_count,
                "max_abs_error": validation.max_abs_error,
                "mean_abs_error": validation.mean_abs_error,
                "decision_disagreements": validation.decision_disagreements,
            }

    validation_text = ""
    if validation is not None:
        if isinstance(validation, Mapping):
            validation_text = (
                f", validated on {validation['sample_count']} samples across "
                f"{validation['output_count']} outputs "
                f"(max error {validation['max_abs_error']:.3g})"
            )
        else:
            validation_text = (
                f", validated on {validation.sample_count} samples "
                f"(max error {validation.max_abs_error:.3g}, decisions identical)"
            )
    print(
        f"✓ Exported {model_type.upper()} {task} to {destination} "
        f"with preprocessing={preprocessing}{validation_text}."
    )
    return report


# =================================================================================================
# MULTI-OUTPUT RANDOM-FOREST REGRESSION — CONTINUOUS SOFT-TARGET EXPORT
# =================================================================================================
# The existing scalar/classifier exporter above is retained.  This separate
# dispatch path adds regression vectors without changing its C ABI or layout.


FIRMWARE_NORMALIZER_MIN_NORM = np.float32(10.0 * np.finfo(np.float32).eps)

# "auto" keeps compact threshold folding when validation succeeds, and switches
# to exact per-feature runtime preprocessing if any split routes differently.
# "runtime" forces runtime preprocessing for affine multi-output regressors;
# "folded" keeps the original strict behavior (raises on mismatch).
MULTIOUTPUT_AFFINE_EXPORT_POLICY = "auto"  # "auto", "runtime", "folded"


class FirmwareNormalizer(Normalizer):
    """Dense float32 row normalization with the same arithmetic as emitted C.

    The C implementation accumulates one feature at a time in selected-schema
    order, computes one float32 reciprocal, and multiplies each feature by it.
    sklearn's ordinary Normalizer may use a different summation order and
    division, so it can choose different leaves immediately beside a split.

    Norms below ten float32 epsilons use a factor of one, matching sklearn's
    near-zero convention. Finite inputs whose norm accumulation overflows
    produce a NaN row, matching the new C predictor's failure-status policy.
    Store this importable export_plus class in joblib, never a __main__ class.
    """

    firmware_arithmetic_version = "float32_sequential_reciprocal_v1"

    def _normalization_factors(self, data: np.ndarray) -> np.ndarray:
        if self.norm not in {"l1", "l2", "max"}:
            raise ValueError(f"Unsupported normalization norm {self.norm!r}.")
        values = np.asarray(data, dtype=np.float32)
        denominator = np.zeros(values.shape[0], dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for feature in range(values.shape[1]):
                column = values[:, feature]
                if self.norm == "l1":
                    np.add(denominator, np.abs(column), out=denominator)
                elif self.norm == "l2":
                    term = np.multiply(column, column, dtype=np.float32)
                    np.add(denominator, term, out=denominator)
                else:
                    np.maximum(denominator, np.abs(column), out=denominator)
            if self.norm == "l2":
                np.sqrt(denominator, out=denominator)
            factor = np.ones_like(denominator)
            valid = np.isfinite(denominator)
            divide = valid & (denominator >= FIRMWARE_NORMALIZER_MIN_NORM)
            np.divide(np.float32(1.0), denominator, out=factor, where=divide)
            factor[~valid] = np.nan
        return factor

    def transform(self, X: Any, copy: bool | None = None) -> np.ndarray:
        copy = self.copy if copy is None else copy
        data = validate_data(
            self, X, reset=False, accept_sparse=False, copy=copy,
            dtype=np.float32, force_writeable=True, ensure_all_finite=True,
        )
        factor = self._normalization_factors(data)
        with np.errstate(over="ignore", invalid="ignore"):
            np.multiply(data, factor[:, None], out=data)
        return data


def _is_multioutput_rf_regressor(model: Any) -> bool:
    raw = model.get("model") if isinstance(model, Mapping) else model
    return isinstance(raw, RandomForestRegressor) and int(getattr(raw, "n_outputs_", 1)) > 1


def _make_multioutput_scalar_transform(
    preprocessing: str,
    preprocessor: Any,
    n_features: int,
    selected_indices: Sequence[int],
) -> Callable[[int, np.float32], np.float32]:
    """Match RobustScaler's parameter precision without changing legacy paths.

    sklearn 1.8 RobustScaler performs in-place NumPy operations into float32
    data using the stored parameter dtypes (its scale is commonly float64).
    StandardScaler explicitly casts parameters to the input dtype instead.
    Rounding RobustScaler's scale to float32 changes some split boundaries.
    The one-element in-place operations below reproduce the actual transformer;
    they run only while folding thresholds during Python export.
    """
    if preprocessing not in {"robust_25_75", "robust_10_90"}:
        return _make_scalar_transform(preprocessing, preprocessor, n_features, selected_indices)
    if not isinstance(preprocessor, RobustScaler):
        raise TypeError(f"{preprocessing} requires a fitted RobustScaler.")
    center = None
    scale = None
    if preprocessor.with_centering:
        original = np.asarray(preprocessor.center_)
        center = _select_or_validate_vector(original, n_features, selected_indices, "center_").astype(original.dtype)
    if preprocessor.with_scaling:
        original = np.asarray(preprocessor.scale_)
        scale = _select_or_validate_vector(original, n_features, selected_indices, "scale_").astype(original.dtype)

    def transform(feature: int, value: np.float32) -> np.float32:
        data = np.asarray([value], dtype=np.float32)
        if center is not None:
            np.subtract(data, center[feature:feature + 1], out=data)
        if scale is not None:
            np.divide(data, scale[feature:feature + 1], out=data)
        return data[0]

    return transform


def _multioutput_regressor_payload(model: RandomForestRegressor) -> Dict[str, Any]:
    if not getattr(model, "estimators_", None):
        raise ValueError("RandomForestRegressor has no fitted estimators.")
    n_outputs = int(model.n_outputs_)
    if n_outputs < 2 or n_outputs > 65535:
        raise ValueError("Multi-output regression requires 2..65535 outputs.")
    counts = [int(est.tree_.node_count) for est in model.estimators_]
    leaves = [int(np.count_nonzero(est.tree_.children_left == _tree.TREE_LEAF)) for est in model.estimators_]
    total_nodes, total_leaves = sum(counts), sum(leaves)
    max_nodes = max(counts)
    if max_nodes > 65535:
        raise ValueError("A RandomForestRegressor tree exceeds uint16 local node indexing.")
    child_bytes = 1 if max_nodes <= 255 else 2
    node_bytes = RF_U8_NODE_BYTES if child_bytes == 1 else RF_U16_NODE_BYTES
    root_type, root_bytes = _root_index_storage(total_nodes)
    leaf_root_type, leaf_root_bytes = _root_index_storage(total_leaves)
    base_node_bytes = total_nodes * node_bytes
    side_table_bytes = total_leaves * (n_outputs - 1) * 4
    root_table_bytes = len(counts) * root_bytes
    leaf_root_table_bytes = len(counts) * leaf_root_bytes
    total = base_node_bytes + side_table_bytes + root_table_bytes + leaf_root_table_bytes
    return {
        "model_type": "sklearn", "task": "MULTI_OUTPUT_REGRESSOR",
        "tree_count": len(counts), "output_count": n_outputs,
        "total_nodes": total_nodes, "total_leaves": total_leaves,
        "max_tree_nodes": max_nodes, "node_bytes": node_bytes,
        "base_node_bytes": node_bytes, "effective_node_bytes": (base_node_bytes + side_table_bytes) / total_nodes,
        "child_bytes": child_bytes, "root_bytes": root_bytes, "root_c_type": root_type,
        "leaf_root_bytes": leaf_root_bytes, "leaf_root_c_type": leaf_root_type,
        "metadata_layout": ("rf_u8_child_feature_packed_u16" if child_bytes == 1 else "rf_u16_child_separate_feature") + "_multioutput_leaf_side_table",
        "feature_metadata_layout": "plain_u8", "default_left_encoding": "not_applicable_finite_only",
        "missing_is_zero_encoding": "not_applicable", "flag_storage_bytes": 0,
        "node_data_bytes": base_node_bytes + side_table_bytes,
        "base_node_data_bytes": base_node_bytes, "leaf_side_table_bytes": side_table_bytes,
        "root_table_bytes": root_table_bytes, "leaf_root_table_bytes": leaf_root_table_bytes,
        "tree_payload_bytes": total, "probability_lut_bytes": 0, "early_exit_bounds_bytes": 0,
        "total_static_data_bytes": total,
        "legacy_8byte_payload_bytes": total_nodes * 8,
        "all_node_side_table_payload_bytes": total_nodes * (node_bytes + 4 * (n_outputs - 1)) + root_table_bytes,
    }


def estimate_compact_tree_payload(model: Any) -> Dict[str, Any]:
    """Exact emitted array bytes; compiler/linker padding and executable code are excluded."""
    if _is_multioutput_rf_regressor(model):
        raw = model.get("model") if isinstance(model, Mapping) else model
        return _multioutput_regressor_payload(raw)
    return _estimate_compact_tree_payload_legacy(model)


def _multioutput_regressor_tree(
    tree: _tree.Tree,
    tree_index: int,
    n_outputs: int,
    affine_a: np.ndarray,
    affine_b: np.ndarray,
    scalar_transform: Callable[[int, np.float32], np.float32],
    affine_folded: bool,
) -> Tuple[List[SklearnNode], List[Tuple[float, ...]]]:
    """Depth-first nodes plus outputs 1..N-1 stored only for reached leaves.

    A leaf's otherwise unused right-child field holds its tree-local leaf ID.
    Output zero occupies the leaf threshold slot, matching the scalar layout.
    """
    if tree.node_count > 65535:
        raise ValueError(f"RF tree {tree_index} exceeds uint16 indexing.")
    result: List[SklearnNode | None] = []
    extras: List[Tuple[float, ...]] = []
    visited: set[int] = set()

    def append(original: int) -> int:
        if not 0 <= original < tree.node_count or original in visited:
            raise ValueError(f"RF tree {tree_index}: invalid or repeated node {original}.")
        visited.add(original)
        compact = len(result)
        result.append(None)
        left, right = int(tree.children_left[original]), int(tree.children_right[original])
        if left == _tree.TREE_LEAF:
            if right != _tree.TREE_LEAF:
                raise ValueError("Inconsistent leaf children.")
            values = np.asarray(tree.value[original], dtype=np.float64)
            if values.shape != (n_outputs, 1) or not np.isfinite(values).all():
                raise ValueError(f"Unexpected/nonfinite regression leaf shape {values.shape}.")
            leaf_id = len(extras)
            extras.append(tuple(_float32(x) for x in values[1:, 0]))
            result[compact] = SklearnNode(_float32(values[0, 0]), leaf_id, 0xFF, True)
            return compact
        feature = int(tree.feature[original])
        if not 0 <= feature <= 0xFE:
            raise ValueError("Compact RF export requires model-local feature IDs below 255.")
        if append(left) != compact + 1:
            raise AssertionError("Depth-first left-child invariant failed.")
        compact_right = append(right)
        threshold = _fold_threshold(
            tree.threshold[original], feature, affine_a, affine_b,
            scalar_transform, affine_folded=affine_folded,
        )
        result[compact] = SklearnNode(threshold, compact_right, feature, False)
        return compact

    append(0)
    if len(visited) != tree.node_count or any(node is None for node in result):
        raise AssertionError("RF traversal failed to include every node.")
    return [node for node in result if node is not None], extras


def _multioutput_regressor_storage(
    trees: Sequence[Sequence[SklearnNode]],
    extras: Sequence[Sequence[Sequence[float]]],
    group_var: str,
    payload: Mapping[str, Any],
) -> str:
    base = _sklearn_storage_code(trees, group_var)
    leaf_roots: List[int] = []
    flat_extras: List[str] = []
    leaf_count = 0
    for tree, tree_extras in zip(trees, extras):
        if len(tree_extras) != sum(node.is_leaf for node in tree):
            raise AssertionError("Leaf side-table/tree mismatch.")
        leaf_roots.append(leaf_count)
        leaf_count += len(tree_extras)
        for vector in tree_extras:
            flat_extras.extend(_c_float(x) for x in vector)
    expected = base.tree_payload_bytes + len(flat_extras) * 4 + len(leaf_roots) * int(payload["leaf_root_bytes"])
    if expected != int(payload["total_static_data_bytes"]):
        raise AssertionError("Exported static-array byte count does not match the estimator.")
    return "\n".join([
        base.code.rstrip(),
        _format_c_array(str(payload["leaf_root_c_type"]), f"{group_var}_tree_leaf_roots", [f"{x}u" for x in leaf_roots]).rstrip(),
        _format_c_array("float", f"{group_var}_leaf_extra_outputs", flat_extras, per_line=6).rstrip(),
        "",
    ])


def _c_double(value: Any) -> str:
    """Round-trip hexadecimal C double literal, retaining the fitted dtype."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Cannot emit a nonfinite fitted preprocessing constant")
    return value.hex()


def _multioutput_runtime_affine_code(
    preprocessing: str, preprocessor: Any, n_features: int,
    selected_indices: Sequence[int],
) -> Tuple[str, int]:
    """Generate unrolled, sequential preprocessing of RAW float32 inputs.

    Keep the same intermediate roundings as sklearn's float32 in-place
    transformers. StandardScaler explicitly casts mean/scale to float32;
    MinMax/Robust/MaxAbs instead keep the fitted parameter's NumPy dtype.
    Fitted float64 constants use a double operation followed by a float32
    assignment after EACH operation (critical for RobustScaler).
    There are no extra declared static model arrays: constants are emitted
    directly in instructions/literal pools. Firmware code/literal-pool size is
    NOT part of the static model-array budget.
    """
    def coeff(obj: Any, attr: str, default: float | None = None) -> np.ndarray:
        raw = getattr(obj, attr, None)
        if raw is None:
            if default is None:
                raise ValueError(f"Missing fitted preprocessing attribute {attr}")
            return np.full(n_features, default, dtype=np.float32)
        original = np.asarray(raw)
        vector = _select_or_validate_vector(original, n_features, selected_indices, attr)
        if preprocessing == "standard":
            return vector.astype(np.float32)
        return vector.astype(original.dtype, copy=False)

    if preprocessing == "standard":
        if not isinstance(preprocessor, StandardScaler):
            raise TypeError("Expected fitted StandardScaler")
        means = coeff(preprocessor, "mean_", 0.0) if preprocessor.with_mean else None
        scales = coeff(preprocessor, "scale_", 1.0) if preprocessor.with_std else None
        operations = [("-", means), ("/", scales)]
    elif preprocessing in ("minmax_0_1", "minmax_m1_1"):
        if not isinstance(preprocessor, MinMaxScaler) or bool(preprocessor.clip):
            raise ValueError("Runtime MinMaxScaler requires clip=False")
        operations = [("*", coeff(preprocessor, "scale_")), ("+", coeff(preprocessor, "min_"))]
    elif preprocessing == "maxabs":
        if not isinstance(preprocessor, MaxAbsScaler) or bool(getattr(preprocessor, "clip", False)):
            raise ValueError("Expected unclipped MaxAbsScaler")
        operations = [("/", coeff(preprocessor, "scale_"))]
    elif preprocessing in ("robust_25_75", "robust_10_90"):
        if not isinstance(preprocessor, RobustScaler):
            raise TypeError("Expected fitted RobustScaler")
        center = coeff(preprocessor, "center_") if preprocessor.with_centering else None
        scale = coeff(preprocessor, "scale_") if preprocessor.with_scaling else None
        operations = [("-", center), ("/", scale)]
    else:
        raise ValueError(f"Unsupported affine runtime preprocessing: {preprocessing}")

    lines = ["    float model_features[" + str(n_features) + "];", 
             "    /* Runtime affine fallback; same dtype and operation order as sklearn. */"]
    double_features = set()
    for feature in range(n_features):
        lines.append(f"    float v{feature} = features[{feature}];")
        for op, vec in operations:
            if vec is None:
                continue
            datum = vec[feature]
            if np.asarray(vec).dtype == np.float64:
                double_features.add(feature)
                # C double arithmetic is used, then explicitly rounded to float32.
                lines.append(f"    v{feature} = (float)((double)v{feature} {op} {_c_double(datum)});")
            else:
                lines.append(f"    v{feature} = (float)(v{feature} {op} {_c_float(datum)});")
        lines.append(f"    if (!isfinite(v{feature})) return 0;")
        lines.append(f"    model_features[{feature}] = v{feature};")
    return "\n".join(lines), len(double_features)


def _multioutput_regressor_array_code(
    context: ExportContext, payload: Mapping[str, Any], preprocessor: Any = None,
    *, runtime_affine: bool = False,
) -> str:
    g, m = context.group_var, context.group_macro
    outputs, trees = int(payload["output_count"]), int(payload["tree_count"])
    minimum_norm = _c_float(FIRMWARE_NORMALIZER_MIN_NORM)
    if int(payload["child_bytes"]) == 1:
        decode = f"""            const uint16_t metadata = {g}_node_metadata[absolute_index];
            const uint8_t feature_index = (uint8_t)(metadata & 0x00FFu);
            const uint16_t right_or_leaf = (uint16_t)(metadata >> 8);"""
    else:
        decode = f"""            const uint8_t feature_index = {g}_node_features[absolute_index];
            const uint16_t right_or_leaf = {g}_node_right_children[absolute_index];"""
    if runtime_affine:
        if context.preprocessing not in AFFINE_PREPROCESSING:
            raise AssertionError("Runtime affine requested without fitted affine preprocessing")
        preprocessing_code, _ = _multioutput_runtime_affine_code(
            context.preprocessing, preprocessor, len(context.selected_indices), context.selected_indices
        )
        model_array_name = "model_features"
    else:
        preprocessing_code = ""
        model_array_name = "features"
    mode = context.runtime_normalization
    if mode is None:
        normalization = "    const float normalization_factor = 1.0f;"
    elif mode == "l1":
        normalization = f"""    float denominator = 0.0f;
    for (uint16_t j = 0u; j < {m}_NUM_FEATURES; ++j) denominator += fabsf(features[j]);
    if (!isfinite(denominator)) return 0;
    const float normalization_factor = denominator >= {minimum_norm} ? 1.0f / denominator : 1.0f;"""
    elif mode == "l2":
        normalization = f"""    float sum_squares = 0.0f;
    for (uint16_t j = 0u; j < {m}_NUM_FEATURES; ++j) sum_squares += features[j] * features[j];
    const float denominator = sqrtf(sum_squares);
    if (!isfinite(denominator)) return 0;
    const float normalization_factor = denominator >= {minimum_norm} ? 1.0f / denominator : 1.0f;"""
    elif mode == "max":
        normalization = f"""    float denominator = 0.0f;
    for (uint16_t j = 0u; j < {m}_NUM_FEATURES; ++j) {{
        const float magnitude = fabsf(features[j]);
        if (magnitude > denominator) denominator = magnitude;
    }}
    const float normalization_factor = denominator >= {minimum_norm} ? 1.0f / denominator : 1.0f;"""
    else:
        raise AssertionError(mode)
    return f"""
/**
 * Input: selected RAW features, in the exact model-local order documented above.
 * Output: {outputs} continuous forest means. No thresholds, states, sigmoid or softmax.
 * Return 1 on success. Return 0 and write NaNs for nonfinite/missing inputs.
 * Compile without -ffast-math/-ffinite-math-only and with -ffp-contract=off.
 * Row normalization, when enabled, matches FirmwareNormalizer's float32 order.
 */
{m}_FORCE_INLINE int {g}_predict_probabilities_array(
    const float features[{m}_NUM_FEATURES], float probabilities[{m}_NUM_OUTPUTS])
{{
    if (probabilities == 0) return 0;
    for (uint16_t output = 0u; output < {m}_NUM_OUTPUTS; ++output) probabilities[output] = NAN;
    if (features == 0) return 0;
    for (uint16_t j = 0u; j < {m}_NUM_FEATURES; ++j) {{
        if (!isfinite(features[j])) return 0;
    }}
{normalization}
{preprocessing_code}
    for (uint16_t output = 0u; output < {m}_NUM_OUTPUTS; ++output) probabilities[output] = 0.0f;
    for (uint32_t tree_index = 0u; tree_index < {m}_NUM_TREES; ++tree_index) {{
        const uint32_t root = (uint32_t){g}_tree_roots[tree_index];
        uint16_t node_index = 0u;
        for (;;) {{
            const uint32_t absolute_index = root + (uint32_t)node_index;
{decode}
            if (feature_index == 0xFFu) {{
                probabilities[0] += {g}_node_thresholds[absolute_index];
                const uint32_t leaf = (uint32_t){g}_tree_leaf_roots[tree_index] + (uint32_t)right_or_leaf;
                for (uint16_t output = 1u; output < {m}_NUM_OUTPUTS; ++output) {{
                    probabilities[output] += {g}_leaf_extra_outputs[leaf * {outputs - 1}u + (uint32_t)(output - 1u)];
                }}
                break;
            }}
            const float value = {model_array_name}[feature_index] * normalization_factor;
            node_index = value <= {g}_node_thresholds[absolute_index]
                ? (uint16_t)(node_index + 1u) : right_or_leaf;
        }}
    }}
    const float scale = 1.0f / {trees}.0f;
    for (uint16_t output = 0u; output < {m}_NUM_OUTPUTS; ++output) probabilities[output] *= scale;
    return 1;
}}

#ifndef {m}_ARRAY_ONLY
/** Optional firmware t_core adapter.  Array-only users define {m}_ARRAY_ONLY. */
{m}_FORCE_INLINE int {g}_predict_probabilities(
    const t_core* p_core, float probabilities[{m}_NUM_OUTPUTS])
{{
    if (p_core == 0) return {g}_predict_probabilities_array(0, probabilities);
    {g}_feature_cache_t cache = {{0}};
    float features[{m}_NUM_FEATURES];
    for (uint16_t j = 0u; j < {m}_NUM_FEATURES; ++j) {{
        features[j] = {g}_get_raw_model_feature(p_core, j, &cache);
    }}
    return {g}_predict_probabilities_array(features, probabilities);
}}
#endif
"""


def _validate_multioutput_regressor(
    model: RandomForestRegressor,
    context: ExportContext,
    preprocessor: Any,
    trees: Sequence[Sequence[SklearnNode]],
    extras: Sequence[Sequence[Sequence[float]]],
    *,
    runtime_affine: bool = False,
    validation_raw: np.ndarray | None = None,
) -> Dict[str, Any]:
    thresholds: Dict[int, List[float]] = {}
    for tree in trees:
        for node in tree:
            if not node.is_leaf:
                thresholds.setdefault(node.feature_index, []).append(node.threshold)
    raw = (
        _validation_samples(int(model.n_features_in_), thresholds, context.preprocessing, preprocessor)
        if validation_raw is None else np.asarray(validation_raw, dtype=np.float32)
    )
    transformed = _apply_python_preprocessor(preprocessor, context.preprocessing, raw)
    if runtime_affine and context.preprocessing not in AFFINE_PREPROCESSING:
        raise AssertionError("Runtime affine validation requires an affine preprocessor")
    routing_rows = transformed if runtime_affine else raw
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
        expected = np.asarray(model.predict(transformed), dtype=np.float64)
    actual = np.zeros_like(expected, dtype=np.float32)
    scale = np.float32(1.0 / len(trees))
    for i, row in enumerate(raw):
        factor = (
            preprocessor._normalization_factors(row[None, :])[0]
            if context.runtime_normalization is not None else np.float32(1.0)
        )
        for tree, side in zip(trees, extras):
            node_id = 0
            while not tree[node_id].is_leaf:
                node = tree[node_id]
                value = np.float32(np.float32(routing_rows[i, node.feature_index]) * factor)
                node_id = node_id + 1 if value <= np.float32(node.threshold) else node.right_child
            leaf = tree[node_id]
            actual[i, 0] = np.float32(actual[i, 0] + np.float32(leaf.threshold))
            for output, value in enumerate(side[leaf.right_child], start=1):
                actual[i, output] = np.float32(actual[i, output] + np.float32(value))
        actual[i] *= scale
    error = np.abs(expected - actual.astype(np.float64))
    report = {
        "sample_count": len(raw), "output_count": int(model.n_outputs_),
        "max_abs_error": float(error.max(initial=0.0)), "mean_abs_error": float(error.mean()),
        "method": (
            "Python compact-tree emulator with original sklearn float32 transformed inputs; "
            "compiled-C arithmetic validation is a separate gate" if runtime_affine else
            "Python emulator of float32 compact representation; compiled-C validation is a separate gate"
        ),
        "affine_execution": "runtime" if runtime_affine else "threshold_folded",
    }
    if report["max_abs_error"] > 7.5e-5:
        raise ValueError(f"Multi-output regression export failed representation validation: {report}.")
    return report


def _export_multioutput_regressor(
    model: Mapping[str, Any], trial_info: Mapping[str, Any], output_path: str | Path,
    feature_schema_path: str | Path, *, validate: bool,
) -> Dict[str, Any]:
    raw = model["model"]
    payload = _multioutput_regressor_payload(raw)
    schema = json.loads(Path(feature_schema_path).read_text(encoding="utf-8"))
    feature_map = {int(k): str(v) for k, v in schema["feature_map"].items()}
    if not feature_map or set(feature_map) != set(range(len(feature_map))):
        raise ValueError("feature_map must contain contiguous indices starting at zero.")
    indices = _extract_selected_indices(model, trial_info, raw, len(feature_map))
    preprocessing, preprocessor = _resolve_preprocessing(model, trial_info)
    if preprocessing in ROW_NORMALIZATION:
        if not isinstance(preprocessor, FirmwareNormalizer):
            raise ValueError(
                "Multi-output RF row normalization requires export_plus.FirmwareNormalizer; "
                "retrain with its exact float32 firmware arithmetic. Ordinary sklearn "
                "Normalizer can disagree with C at split boundaries."
            )
        if preprocessor.norm != ROW_NORMALIZATION[preprocessing]:
            raise ValueError("Saved FirmwareNormalizer.norm conflicts with preprocessing metadata.")
    group_name = str(trial_info.get("group_name", "emg_rf"))
    group_var = _sanitize_c_identifier(group_name)
    group_macro = _sanitize_c_identifier(group_name, uppercase=True)
    context = ExportContext(group_name, group_var, group_macro, preprocessing,
                            ROW_NORMALIZATION.get(preprocessing), indices,
                            tuple(feature_map[i] for i in indices), 0.5, True, "sklearn")
    if preprocessing in AFFINE_PREPROCESSING:
        a, b = _affine_coefficients(preprocessing, preprocessor, len(indices), indices)
    else:
        a, b = np.ones(len(indices)), np.zeros(len(indices))
    scalar_transform = _make_multioutput_scalar_transform(preprocessing, preprocessor, len(indices), indices)
    trees: List[List[SklearnNode]] = []
    extras: List[List[Tuple[float, ...]]] = []
    for i, estimator in enumerate(raw.estimators_):
        tree, side = _multioutput_regressor_tree(estimator.tree_, i, raw.n_outputs_, a, b,
                                               scalar_transform, preprocessing in AFFINE_PREPROCESSING)
        trees.append(tree)
        extras.append(side)
    policy = str(MULTIOUTPUT_AFFINE_EXPORT_POLICY).lower()
    if policy not in {"auto", "runtime", "folded"}:
        raise ValueError("MULTIOUTPUT_AFFINE_EXPORT_POLICY must be auto, runtime, or folded")
    runtime_affine = bool(policy == "runtime" and preprocessing in AFFINE_PREPROCESSING)
    folded_failure = None
    validation_rows = None
    if validate:
        thresholds: Dict[int, List[float]] = {}
        for tree in trees:
            for node in tree:
                if not node.is_leaf:
                    thresholds.setdefault(node.feature_index, []).append(node.threshold)
        validation_rows = _validation_samples(len(indices), thresholds, preprocessing, preprocessor)
        if not runtime_affine:
            try:
                validation = _validate_multioutput_regressor(
                    raw, context, preprocessor, trees, extras, validation_raw=validation_rows
                )
            except ValueError as exc:
                if (policy != "auto" or preprocessing not in AFFINE_PREPROCESSING
                        or "representation validation" not in str(exc)):
                    raise
                folded_failure = str(exc)
                runtime_affine = True
                warnings.warn(
                    "Affine split-threshold folding disagreed with sklearn; "
                    "switching to runtime affine preprocessing instead of relaxing parity tolerance. "
                    + folded_failure,
                    RuntimeWarning, stacklevel=2,
                )
    if runtime_affine:
        # Tree thresholds now remain in the same preprocessed domain as sklearn.
        # The C function transforms raw inputs only once per prediction.
        trees = []
        extras = []
        for i, estimator in enumerate(raw.estimators_):
            tree, side = _multioutput_regressor_tree(
                estimator.tree_, i, raw.n_outputs_, a, b, scalar_transform, False
            )
            trees.append(tree)
            extras.append(side)
    if validate and runtime_affine:
        validation = _validate_multioutput_regressor(
            raw, context, preprocessor, trees, extras, runtime_affine=True,
            validation_raw=validation_rows,
        )
    if not validate:
        validation = None
    storage = _multioutput_regressor_storage(trees, extras, group_var, payload)
    output_names = list(trial_info.get("output_names", model.get("target_columns", schema.get("output_names", []))))
    if not output_names:
        output_names = [f"output_{i}" for i in range(raw.n_outputs_)]
    output_names = [str(x) for x in output_names]
    if len(output_names) != raw.n_outputs_ or len(set(output_names)) != len(output_names):
        raise ValueError("Output names must be unique and match model.n_outputs_.")
    output_macros = [_sanitize_c_identifier(x, uppercase=True) for x in output_names]
    if len(set(output_macros)) != len(output_macros):
        raise ValueError("Output names collide after conversion to C identifiers.")
    required_helpers = sorted({h for feature in context.feature_names for h in feature_expression_helpers(feature)})
    helper_count = len(_helper_feature_slots(context))
    helper_cache_bytes = helper_count * 4 + ((helper_count + 31) // 32) * 4 if helper_count else 1
    guard = f"{group_macro}_MODEL_H"
    parts = [
        f"#ifndef {guard}", f"#define {guard}", "",
        "/*", f" * Generated: {datetime.now(timezone.utc).isoformat()}",
        " * Task: MULTI_OUTPUT_REGRESSOR; continuous soft-target random forest.",
        " * One traversal per tree returns every output; no class argmax or state machine.",
        f" * Static model arrays: {payload['total_static_data_bytes']} bytes (excludes code/linker padding).",
        f" * Training preprocessing: {preprocessing}; affine operations are folded into thresholds.",
        f" * Define {group_macro}_ARRAY_ONLY to omit t_core integration.",
        " * Nonfinite selected inputs or normalization overflow: return 0 and NaNs.",
        " * Use -fno-fast-math -ffp-contract=off for reproducible float32 arithmetic.",
        f" * Affine preprocessing execution: {'runtime to preserve split routing' if runtime_affine else 'threshold folding'}.",
        " * Runtime affine constants are part of generated code/literal pools (not static model arrays).",
        " * Input array order:",
        *[f" *   features[{j}] = original[{i}] {feature_map[i]}" for j, i in enumerate(indices)],
        " * Output order:", *[f" *   probabilities[{i}] = {name}" for i, name in enumerate(output_names)],
        " */", "", "#include <math.h>", "#include <stdint.h>",
        f"#ifndef {group_macro}_ARRAY_ONLY", '#include "core_type_def.h"',
        *([] if not FEATURE_HELPERS_HEADER else [f'#include "{FEATURE_HELPERS_HEADER}"']),
        "#endif", "",
        f"#define {group_macro}_NUM_FEATURES {len(indices)}u",
        f"#define {group_macro}_NUM_TREES {payload['tree_count']}u",
        f"#define {group_macro}_NUM_OUTPUTS {raw.n_outputs_}u",
        f"#define {group_macro}_STATIC_MODEL_BYTES {payload['total_static_data_bytes']}u",
        *[f"#define {group_macro}_OUTPUT_{name} {i}u" for i, name in enumerate(output_macros)],
        "", _compiler_optimization_code(context).strip(), "",
        f"#ifndef {group_macro}_ARRAY_ONLY", _feature_accessor_code(context).rstrip(), "#endif", "",
        storage.rstrip(), "", _multioutput_regressor_array_code(
            context, payload, preprocessor, runtime_affine=runtime_affine
        ).strip(), "",
        f"#endif /* {guard} */", "",
    ]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    temporary.replace(destination)
    report = {
        **payload, "output_path": str(destination), "output_names": output_names,
        "selected_feature_count": len(indices), "selected_indices": list(indices),
        "selected_feature_names": list(context.feature_names), "preprocessing": preprocessing,
        "firmware_preprocessing": (
            "runtime affine preprocessing" if runtime_affine else
            "raw features" if preprocessing == "none" else
            "affine threshold folding" if preprocessing in AFFINE_PREPROCESSING else "row normalization"
        ),
        "affine_execution": "runtime" if runtime_affine else "threshold_folded",
        "folded_validation_failure": folded_failure,
        "runtime_affine_constants_scope": (
            "code and literal pools; not separately declared static model arrays"
            if runtime_affine else None
        ),
        "decision_threshold": None, "has_state_machine": False, "has_softmax": False,
        "finite_input_policy": "failure_status_and_all_nan_outputs",
        "normalization_implementation": (
            FirmwareNormalizer.firmware_arithmetic_version if preprocessing in ROW_NORMALIZATION else None
        ),
        "normalization_near_zero_guard": (
            float(FIRMWARE_NORMALIZER_MIN_NORM) if preprocessing in ROW_NORMALIZATION else None
        ),
        "array_function": f"{group_var}_predict_probabilities_array",
        "core_function": f"{group_var}_predict_probabilities", "array_only_macro": f"{group_macro}_ARRAY_ONLY",
        "required_feature_helpers": required_helpers, "helper_cache_bytes_per_prediction": helper_cache_bytes,
        "core_adapter_feature_array_bytes": 4 * len(indices),
        "storage_size_scope": "sum of generated static model array sizeof values; excludes executable code, stack and linker padding",
        "max_supported_features": 254, "validation": validation,
    }
    print(
        f"Exported {raw.n_outputs_}-output RF regressor to {destination}; "
        f"{payload['total_static_data_bytes']} static-array bytes; "
        f"affine execution={'runtime' if runtime_affine else 'folded'}."
    )
    return report


def export_to_c_header(
    model: Mapping[str, Any], trial_info: Mapping[str, Any], output_path: str | Path,
    feature_schema_path: str | Path, *, validate: bool = True,
) -> Dict[str, Any]:
    """Backward-compatible public API with a separate continuous multi-output RF path."""
    if _is_multioutput_rf_regressor(model):
        return _export_multioutput_regressor(model, trial_info, output_path, feature_schema_path, validate=validate)
    return _export_to_c_header_legacy(model, trial_info, output_path, feature_schema_path, validate=validate)


def main() -> int:
    model_path = Path(STANDALONE_MODEL_PATH)
    info_path = Path(STANDALONE_INFO_PATH)
    schema_path = Path(STANDALONE_SCHEMA_PATH)
    output_path = Path(STANDALONE_OUTPUT_PATH)
    for path, label in ((model_path, "model"), (info_path, "metadata"), (schema_path, "schema")):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label.capitalize()} file not found: {path}. Edit the STANDALONE_* globals "
                "at the top of export_plus.py."
            )

    model_container = joblib.load(model_path)
    if not isinstance(model_container, Mapping):
        raise TypeError(
            f"Model file must contain a mapping/dict, got {type(model_container).__name__}."
        )
    with info_path.open("r", encoding="utf-8") as handle:
        trial_info = json.load(handle)

    export_to_c_header(
        model=model_container,
        trial_info=trial_info,
        output_path=output_path,
        feature_schema_path=schema_path,
        validate=EXPORT_VALIDATE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
