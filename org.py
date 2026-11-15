#!/usr/bin/env python3

"""Organize gesture CSV files into one or more dataset trees.

Each GESTURE_MAP entry has this shape:

    "keyword": (TARGET_FOLDER_OR_FOLDERS, "SUB_FOLDER", CLASS_ID)

TARGET_FOLDER_OR_FOLDERS may be:

- one string, for example: FOLDER_TAP

- a tuple/list/set of strings, for example:

      (FOLDER_TAP, FOLDER_LONG_PRESS)

      {FOLDER_TAP, FOLDER_LONG_PRESS}

For multi-destination entries, the CSV is parsed and serialized only once.

The generated classified CSV is then copied to every requested dataset.

"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import time

import shutil

from pathlib import Path

from typing import Iterable
from collections import Counter, defaultdict
import hashlib
import json
import re
import unicodedata
import numpy as np

import pandas as pd

# =============================================================================

# 1. CONFIGURATION

# =============================================================================

INPUT_FOLDER = Path("TAP_DATA_VALID")

OUTPUT_BASE = Path("DATA")

UNMATCHED_FOLDER = Path("DATA_UNMATCHED")

FOLDER_TAP = "DATA_TAP"

FOLDER_LONG_PRESS = "DATA_LONG_PRESS"

FOLDER_FINGER_TAP = "DATA_FINGER_TAP"

#FOLDER_FIT_CHECK = "DATA_FIT_CHECK"
FOLDER_FIT_CHECK = FOLDER_LONG_PRESS

FOLDER_OTHER = "DATA_OTHER"

# Used to make set-based destinations deterministic.

MAIN_FOLDER_ORDER = (

    FOLDER_TAP,

    FOLDER_LONG_PRESS,

    FOLDER_FINGER_TAP,

    FOLDER_FIT_CHECK,

    FOLDER_OTHER,

)

OVERWRITE_EXISTING = True

CSV_READ_KWARGS = {"low_memory": False}

# Wide CSVs are independent; process them on multiple CPU cores.  For a
# machine with fast NVMe and ample RAM, 8-16 workers is a good starting point.
# Set to 1 for sequential operation (and easy debugging).
CSV_WORKERS = min(12, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))

# Reuse one internal scoring buffer instead of allocating an array on every
# candidate. The original greedy/move/swap search and all tie-breaks remain.
FOLD_REUSE_SCORE_BUFFER = True

# -------------------- FOLD ROADMAP CONFIGURATION ---------------------------
FOLD_ROADMAP_ENABLED = True
FOLD_ROADMAP_PATH = OUTPUT_BASE / "fold_roadmap.json"
FOLD_COUNT = 5
FOLD_SEED = 20261011
FOLD_RESTARTS = 12
FOLD_REFINE_PASSES = 6
FOLD_DETECT_DUPLICATES = True
FOLD_EXCLUDE_DUPLICATES = True  # Only a JSON training flag; CSV copies unchanged.
FOLD_REUSE_EXISTING_ROADMAP = True  # Freeze old users on subsequent runs.
# Conservative mode may merge DIFFERENT people who share a first name.
# It avoids potential cross-fold leakage; audit identity_review_required.
# Alternative: "exact_normalized_name" after verifying your participant IDs.
FOLD_PERSON_GROUPING = "conservative_first_name"
# Example: {"separate_person_id": ["Raw User column name", "filename_prefix"]}
# Explicit aliases take precedence over automatic first-name grouping.
# Dataset-specific evidence: nine FIT_CHECK CSVs are named Ohad_LeftHand_* but
# have User="tws". Conservatively bind them to the Ohad group to avoid leakage.
# Review this alias against recording provenance before a final benchmark.
FOLD_PERSON_ALIASES = {"ohad": ["tws"]}
FOLD_BALANCE_WEIGHTS = {
    "overall_rows": 6.0, "overall_files": 5.0, "subject_groups": 1.0,
    "dataset_class_rows": 5.0, "dataset_class_files": 5.0,
    "dataset_category_rows": 4.0, "dataset_category_files": 4.0,
    "dataset_gesture_rows": 4.0, "dataset_gesture_files": 5.0,
    "orientation_files": 1.5, "direction_files": 0.9,
    "hand_files": 0.5, "bpm_files": 0.9,
}
FOLD_COVERAGE_STRENGTH = 1.5


# =============================================================================

# 2. GESTURE MAPPING

# =============================================================================

# Format:

#   "lowercase keyword": (TARGET_FOLDER_OR_FOLDERS, "SUB_FOLDER", CLASS_ID)

#

# TARGET_FOLDER_OR_FOLDERS can be:

#   FOLDER_TAP

#   (FOLDER_TAP, FOLDER_LONG_PRESS)

#   {FOLDER_TAP, FOLDER_LONG_PRESS}

#

# Longer keywords are checked first.

GESTURE_MAP = {

    # FINGER TAPS -> DATA_FINGER_TAP

    "indexfingertap": (FOLDER_FINGER_TAP, "INDEX_FINGER_TAP", 1),

    "midfingertap": (FOLDER_FINGER_TAP, "MID_FINGER_TAP", 2),

    "ringfingertap": (FOLDER_FINGER_TAP, "RING_FINGER_TAP", 3),

    "pinkyfingertap": (FOLDER_FINGER_TAP, "PINKY_FINGER_TAP", 4),

    "thumbfingertap": (FOLDER_FINGER_TAP, "THUMB_FINGER_TAP", 5),

    # LONG PRESS / PRESSURE -> DATA_LONG_PRESS

    # Repeating the exact same dict key would overwrite earlier entries,

    # so specific long-press variants use separate keys.

    "softlongpress": (FOLDER_LONG_PRESS, "SOFT_LONG_PRESS", 1),

    "midlongpress": (FOLDER_LONG_PRESS, "MID_LONG_PRESS", 1),

    "stronglongpress": (FOLDER_LONG_PRESS, "STRONG_LONG_PRESS", 1),

    "longpress": (FOLDER_LONG_PRESS, "LONG_PRESS", 1),

    "drag": (FOLDER_LONG_PRESS, "DRAG_AND_DROP", 1),

    "drop": (FOLDER_LONG_PRESS, "DRAG_AND_DROP", 1),

    # STANDARD TAP -> DATA_TAP

    "tapincontraction": (FOLDER_TAP, "TAP", 1),

    "tapincontaction": (FOLDER_TAP, "TAP", 1),

    "tapin": (FOLDER_TAP, "TAP", 1),

    "tapwithcontaction": (FOLDER_TAP, "TAP", 1),

    "tapwiththepalmbentup": (FOLDER_TAP, "TAP", 1),

    "tapwiththepalmbentdown": (FOLDER_TAP, "TAP", 1),

    "taponmotion": ({FOLDER_TAP, FOLDER_LONG_PRESS}, "TAP", 1),

    "doubletaps": ({FOLDER_TAP, FOLDER_LONG_PRESS}, "TAP", 1),

    "strongtap": ({FOLDER_TAP, FOLDER_LONG_PRESS}, "TAP", 1),

    "softtap": ({FOLDER_TAP, FOLDER_LONG_PRESS}, "TAP", 1),

    "midtap": ({FOLDER_TAP, FOLDER_LONG_PRESS}, "TAP", 1),

    "pinch": ({FOLDER_TAP, FOLDER_LONG_PRESS}, "TAP", 1),

    # SHARED NEGATIVE / SUPPORT CLASSES

    # These entries are written to both dataset trees.

    "swipe": ({FOLDER_TAP, FOLDER_LONG_PRESS}, "SWIPE", 1),

    "roll": ({FOLDER_TAP, FOLDER_LONG_PRESS}, "SWIPE", 1),

    "ingest": ({FOLDER_TAP, FOLDER_LONG_PRESS}, "SWIPE", 1),

    "outgest": ({FOLDER_TAP, FOLDER_LONG_PRESS}, "SWIPE", 1),

    # FINGER MOTION

    "indexfinger": (FOLDER_TAP, "FINGER", 3),

    "midfinger": (FOLDER_TAP, "FINGER", 3),

    "ringfinger": (FOLDER_TAP, "FINGER", 3),

    "pinky": (FOLDER_TAP, "FINGER", 3),

    "thumb": (FOLDER_TAP, "FINGER", 3),

    "fist": (FOLDER_TAP, "FINGER", 3),

    "stretch": (FOLDER_TAP, "FINGER", 3),

    # WRIST

    "wristextraction_contraction": (

        {FOLDER_TAP},

        "WRIST",

        4,

    ),

    "taponhand": ({FOLDER_TAP}, "WRIST", 4),

    "wrist": ({FOLDER_TAP}, "WRIST", 4),

    "twist": (FOLDER_TAP, "WRIST", 4),

    # ARM

    "arm": ({FOLDER_TAP}, "ARM", 5),

    # BAND

    "band": ({FOLDER_TAP}, "BAND", 6),

    # OBJECT

    "surface": (FOLDER_TAP, "OBJECT", 7),

    "keyboard": (FOLDER_TAP, "OBJECT", 7),

    "mouse": (FOLDER_TAP, "OBJECT", 7),

    "phone": (FOLDER_TAP, "OBJECT", 7),

    "slap": (FOLDER_TAP, "OBJECT", 7),

    # SYSTEM / IDLE

    "nothing": ({FOLDER_TAP, FOLDER_LONG_PRESS}, "NOTHING", 0),

    "connected": (FOLDER_FIT_CHECK, "CONNECTED", 1),

    "disconnected": (FOLDER_FIT_CHECK, "DISCONNECTED", 2),

    "loose": (FOLDER_FIT_CHECK, "LOOSE", 3),

    "elect_on_top_horizontal": (FOLDER_FIT_CHECK, "LOOSE", 4),
    # Also recognize Vertical and Reverse, which the specific kweyword misses.
    "elect_on_top": (FOLDER_FIT_CHECK, "LOOSE", 5),

    "table_with_power": (FOLDER_FIT_CHECK, "TABLE", 6),

    "table": (FOLDER_FIT_CHECK, "TABLE", 7),

}

# =============================================================================

# 3. HELPERS

# =============================================================================

def normalize_target_folders(

    raw_targets: str | Iterable[str],

) -> tuple[str, ...]:

    """Normalize one or many destination-folder names.

    Strings remain one destination. Lists, tuples, and sets are deduplicated.

    Set order is made deterministic using MAIN_FOLDER_ORDER.

    """

    if isinstance(raw_targets, str):

        candidates = [raw_targets]

    else:

        candidates = [str(value) for value in raw_targets]

    candidates = list(dict.fromkeys(candidates))

    if not candidates:

        raise ValueError("A mapping entry has no target folders.")

    known_order = {

        folder_name: index

        for index, folder_name in enumerate(MAIN_FOLDER_ORDER)

    }

    candidates.sort(

        key=lambda value: (

            known_order.get(value, len(known_order)),

            value,

        )

    )

    return tuple(candidates)

def validate_gesture_map() -> None:

    """Fail early when a mapping entry is malformed."""

    for keyword, specification in GESTURE_MAP.items():

        if keyword != keyword.lower():

            raise ValueError(

                f"GESTURE_MAP key must be lowercase: {keyword!r}"

            )

        if not isinstance(specification, tuple) or len(specification) != 3:

            raise ValueError(

                f"Invalid mapping for {keyword!r}: expected "

                "(target_folder_or_folders, sub_folder, class_id)."

            )

        raw_targets, sub_folder, class_id = specification

        target_folders = normalize_target_folders(raw_targets)

        if not sub_folder or not isinstance(sub_folder, str):

            raise ValueError(

                f"Invalid subfolder for {keyword!r}: {sub_folder!r}"

            )

        if isinstance(class_id, bool) or not isinstance(class_id, int):

            raise ValueError(

                f"class_id for {keyword!r} must be an integer."

            )

        unknown = [

            folder

            for folder in target_folders

            if folder not in MAIN_FOLDER_ORDER

        ]

        if unknown:

            raise ValueError(

                f"Unknown target folder(s) for {keyword!r}: {unknown}"

            )

def set_class_id_first(

    dataframe: pd.DataFrame,

    class_id: int,

) -> pd.DataFrame:

    """Insert or overwrite class_id and keep it as the first column."""

    result = dataframe.copy()

    if "class_id" in result.columns:

        result["class_id"] = class_id

        result = result.loc[

            :,

            [

                "class_id",

                *(column for column in result.columns if column != "class_id"),

            ],

        ]

    else:

        result.insert(0, "class_id", class_id)

    return result

def atomic_write_dataframe(

    dataframe: pd.DataFrame,

    destination: Path,

) -> None:

    """Write a CSV atomically."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not OVERWRITE_EXISTING:

        raise FileExistsError(f"Destination already exists: {destination}")

    temporary = destination.with_name(destination.name + ".tmp")

    try:

        dataframe.to_csv(temporary, index=False)

        os.replace(temporary, destination)

    finally:

        if temporary.exists():

            temporary.unlink()

def atomic_copy_file(

    source: Path,

    destination: Path,

) -> None:

    """Copy one already-generated CSV atomically."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not OVERWRITE_EXISTING:

        raise FileExistsError(f"Destination already exists: {destination}")

    temporary = destination.with_name(destination.name + ".tmp")

    try:

        shutil.copy2(source, temporary)

        os.replace(temporary, destination)

    finally:

        if temporary.exists():

            temporary.unlink()

def find_mapping(

    filename: str,

    sorted_keywords: list[str],

) -> tuple[str, tuple[object, str, int]] | None:

    """Return the first longest-keyword match."""

    filename_lower = filename.lower()

    for keyword in sorted_keywords:

        if keyword in filename_lower:

            return keyword, GESTURE_MAP[keyword]

    return None

# =============================================================================

# 5. PROCESSING

# =============================================================================

# =============================================================================
# 4. REPRODUCIBLE GROUP-AWARE FIVE-FOLD ROADMAP
# =============================================================================
# No CLI parameters. Change the global settings near the top of this script.
# Only successfully organized source CSVs enter the roadmap. Each source is
# counted ONCE globally even when it was copied into multiple dataset trees.


def _fold_key(value: str) -> str:
    """Case/whitespace/punctuation-tolerant identifier (Unicode-safe)."""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return "_".join(re.findall(r"[^\W_]+", value, flags=re.UNICODE)).strip("_")


def _person_tokens(value: str) -> list[str]:
    words = _fold_key(value).split("_")
    ignored = {"left", "right", "hand", "lefthand", "righthand",
               "noise", "new", "offset"}
    kept = []
    for word in words:
        if word in ignored or re.fullmatch(r"psd\d*", word):
            continue
        # Firmware-era names sometimes concatenate hand labels: ohadright.
        for suffix in ("lefthand", "righthand", "left", "right"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
        if word:
            kept.append(word)
    return kept


def _manual_alias_lookup() -> dict[str, str]:
    alias_to_person: dict[str, str] = {}
    for canonical, aliases in FOLD_PERSON_ALIASES.items():
        person_key = _fold_key(canonical)
        if not person_key:
            raise ValueError("An empty canonical person ID is not allowed")
        for spelling in [canonical, *aliases]:
            spelling_key = _fold_key(spelling)
            if spelling_key in alias_to_person and alias_to_person[spelling_key] != person_key:
                raise ValueError(f"Conflicting person alias: {spelling!r}")
            alias_to_person[spelling_key] = person_key
    return alias_to_person


def _parse_filename_details(filename: str, match_keyword: str,
                            user_raw: str = "") -> dict:
    """Recover full gesture, identity prefix, and conditions from filenames.

    GESTURE_MAP is only for routing. Do not mistake a substring match such as
    'arm' in 'TaponArm', or 'pinch' in 'SlowPinch', for the full gesture name.
    """
    stem = Path(filename).stem
    # Common file-copy suffixes: ..._15-45-48(1).csv, ..._15-45-48 (2).csv
    stem = re.sub(r"(?:\s*\(\d+\)|_\d+)$", "", stem)
    date_match = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})(?:-\d{1,6})?$", stem)
    date_value = date_match.group(1) if date_match else None
    base = stem[:date_match.start()] if date_match else stem
    bpm_match = re.search(r"_(\d+)bpm$", base, flags=re.IGNORECASE)
    bpm = int(bpm_match.group(1)) if bpm_match else None
    if bpm_match:
        base = base[:bpm_match.start()]
    orient_match = re.search(
        r"_(horizontal|vertical|reverse|default|neutral|none)_",
        base, flags=re.IGNORECASE,
    )
    if orient_match:
        before = base[:orient_match.start()]
        orientation = orient_match.group(1).upper()
        direction = _fold_key(base[orient_match.end():]).upper() or "UNKNOWN"
    else:
        before = base
        orientation = "UNKNOWN"
        direction = "UNKNOWN"
    # Some filenames repeat a directional option: Up_Down___Up_Down.
    direction_parts = direction.split("_")
    half = len(direction_parts) // 2
    if half > 0 and len(direction_parts) % 2 == 0 and (
        direction_parts[:half] == direction_parts[half:]
    ):
        direction = "_".join(direction_parts[:half])

    # The raw 'User' field usually identifies how much of the prefix belongs to
    # the person. Its formatting can differ from the filename ('Adi Edelshtein'
    # vs 'AdiEdelshtein'). Try all prefix lengths, preferring the longest match.
    components = before.split("_")
    person_length = None
    name_tokens = _person_tokens(user_raw)
    raw_compact = "".join(name_tokens)
    generic_names = {"", "na", "none", "nan", "null", "unknown", "test", "tt", "s"}
    if raw_compact not in generic_names and len(raw_compact) > 2:
        full_matches = []
        first_name_matches = []
        for length in range(1, len(components)):
            prefix = "_".join(components[:length])
            prefix_tokens = _person_tokens(prefix)
            if not prefix_tokens:
                continue
            compact = "".join(prefix_tokens)
            if compact == raw_compact:
                full_matches.append(length)
            elif name_tokens and prefix_tokens[0] == name_tokens[0]:
                first_name_matches.append(length)
        if full_matches:
            person_length = max(full_matches)
        elif first_name_matches:
            # For partially identified users, do not consume additional words
            # beyond the first possible person token (which may be a gesture).
            person_length = min(first_name_matches)
    if person_length is None:
        # For generic User values like 'tt', the gesture is normally the final
        # filename component before the orientation; earlier parts are identity.
        # If the keyword starts at a component boundary, use that as a fallback.
        person_length = max(1, len(components) - 1)
        for idx in range(1, len(components)):
            if "_".join(components[idx:]).casefold().startswith(match_keyword.casefold()):
                person_length = idx
                break
    person_length = min(person_length, max(0, len(components) - 1))
    filename_person = "_".join(components[:person_length]).strip("_ -.")
    gesture_raw = "_".join(components[person_length:]).strip("_ -.") or match_keyword
    gesture = _fold_key(gesture_raw) or _fold_key(match_keyword)
    hand_words = _fold_key(filename_person).split("_")
    hand = "LEFT" if "left" in hand_words or "lefthand" in hand_words or any(
        w.endswith("left") and len(w) > 6 for w in hand_words) else (
        "RIGHT" if "right" in hand_words or "righthand" in hand_words or any(
            w.endswith("right") and len(w) > 7 for w in hand_words)
        else "UNSPECIFIED"
    )
    return {
        "filename_person": filename_person,
        "gesture": gesture,
        "gesture_raw": gesture_raw,
        "orientation": orientation,
        "direction": direction,
        "hand": hand,
        "bpm": bpm,
        "recording_date": date_value,
    }


def _get_user_metadata(dataframe: pd.DataFrame) -> tuple[str, list[str]]:
    """Read user metadata WITHOUT using the unreliable PointerState/state."""
    if "User" not in dataframe.columns:
        return "", []
    raw = dataframe["User"].dropna()
    if len(raw) == 0:
        return "", []
    candidates = sorted({str(v).strip() for v in raw.unique() if str(v).strip()})
    return (candidates[0] if candidates else ""), candidates


def _subject_for_file(user_raw: str, filename_person: str,
                      alias_lookup: dict[str, str]) -> tuple[str, str, bool]:
    """Conservative grouping prevents possible same-first-name leakage.

    Different surnames with the same first name may be over-grouped. The JSON
    explicitly flags that ambiguity. Override confirmed identities via
    FOLD_PERSON_ALIASES, or use strict grouping if the user verifies names.
    """
    generic = {"", "na", "n_a", "none", "nan", "null", "unknown",
               "test", "tt", "s", "user", "anonymous"}
    raw_key = _fold_key(user_raw)
    raw_tokens = _person_tokens(user_raw)
    low_quality = raw_key in generic or (len(raw_key) <= 2)
    chosen = filename_person if (low_quality and filename_person) else user_raw
    tokens = _person_tokens(chosen)
    if not tokens:
        tokens = _person_tokens(filename_person)
    if not tokens:
        raise ValueError(f"Cannot determine subject from User={user_raw!r}, "
                         f"filename_person={filename_person!r}. "
                         "Add an explicit alias or correct the filename.")
    full_key = "_".join(tokens)
    # Explicit aliases always win and can separate known same-first-name people.
    for possible in (_fold_key(user_raw), _fold_key(filename_person),
                     _fold_key(chosen), full_key):
        if possible in alias_lookup:
            return alias_lookup[possible], full_key, low_quality
    if FOLD_PERSON_GROUPING == "conservative_first_name":
        return tokens[0], full_key, low_quality
    if FOLD_PERSON_GROUPING == "exact_normalized_name":
        return full_key, full_key, low_quality
    raise ValueError(f"Invalid FOLD_PERSON_GROUPING={FOLD_PERSON_GROUPING!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_for_file(source_path: Path, dataframe: pd.DataFrame,
                     keyword: str, targets: tuple[str, ...],
                     sub_folder: str, class_id: int,
                     output_paths: list[Path], aliases: dict[str, str]) -> dict:
    relative_source = source_path.relative_to(INPUT_FOLDER).as_posix()
    filename = source_path.name
    user_raw, user_values = _get_user_metadata(dataframe)
    if len(user_values) > 1:
        # One CSV with multiple people cannot safely be treated as one group.
        raise ValueError(f"Multiple User values in one recording: {filename}: "
                         f"{user_values}. Resolve this before assigning folds.")
    details = _parse_filename_details(filename, keyword, user_raw)
    subject_id, subject_full_hint, low_quality = _subject_for_file(
        user_raw, details["filename_person"], aliases
    )
    data_sha = _sha256_file(source_path) if FOLD_DETECT_DUPLICATES else None
    file_id = "file_" + hashlib.sha256(relative_source.encode("utf-8")).hexdigest()[:24]
    destinations = [
        {"dataset": target, "category": sub_folder, "class_id": int(class_id),
         "relative_csv_path": path.relative_to(OUTPUT_BASE).as_posix()}
        for target, path in zip(targets, output_paths)
    ]
    return {
        "file_id": file_id,
        "source_relative_path": relative_source,
        "filename": filename,
        "user_raw": user_raw or None,
        "user_values": user_values if len(user_values) > 1 else None,
        "filename_person": details.pop("filename_person"),
        "subject_id": subject_id,
        "subject_full_name_hint": subject_full_hint,
        "identity_low_quality_user": low_quality,
        **details,
        "sample_rows": int(len(dataframe)),
        "destinations": destinations,
        "content_sha256": data_sha,
        "organized_content_sha256": None,
        "recording_id": "rec_" + data_sha[:24] if data_sha else file_id,
        "fold_id": None,
        "canonical_file_id": file_id,
        "exclude_from_training": len(dataframe) == 0,
        "duplicate_of_file_id": None,
    }


class _DisjointSet:
    def __init__(self, keys: Iterable[str]):
        self.parent = {key: key for key in keys}

    def root(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.root(parent)
        return self.parent[key]

    def unite(self, a: str, b: str) -> None:
        a, b = self.root(a), self.root(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _mark_duplicates_and_groups(records: list[dict]) -> list[str]:
    """Prevent leakage and overcounting of both RAW and ORGANIZED duplicates.

    Raw input hashes alone are insufficient: two recordings can have different
    original class_id values or formatting, but serialize to identical output
    CSVs after class_id normalization. A union-find handles transitive links.
    """
    warnings: list[str] = []
    subject_groups = _DisjointSet(r["subject_id"] for r in records)
    file_groups = _DisjointSet(r["file_id"] for r in records)
    seen_raw: dict[str, dict] = {}
    seen_organized: dict[str, dict] = {}
    for record in sorted(records, key=lambda row: row["file_id"]):
        if record["sample_rows"] == 0:
            continue
        for label, key_name, seen in (
            ("raw", "content_sha256", seen_raw),
            ("organized", "organized_content_sha256", seen_organized),
        ):
            digest = record.get(key_name)
            if not digest:
                continue
            first = seen.setdefault(digest, record)
            if first is record:
                continue
            if first["sample_rows"] != record["sample_rows"]:
                raise AssertionError("Identical CSV hashes with different row counts")
            if file_groups.root(first["file_id"]) != file_groups.root(record["file_id"]):
                if label == "organized" and first["content_sha256"] != record["content_sha256"]:
                    warnings.append(
                        "Same organized CSV despite different raw input hashes: "
                        f"{first['filename']} vs {record['filename']}; "
                        "only one copy will be trainable"
                    )
                file_groups.unite(first["file_id"], record["file_id"])
            if first["subject_id"] != record["subject_id"]:
                warnings.append(
                    "Identical CSVs with different participant IDs: "
                    f"{first['filename']} vs {record['filename']}; "
                    "groups joined to prevent leakage"
                )
                subject_groups.unite(first["subject_id"], record["subject_id"])
            if {(d["dataset"], d["class_id"], d["category"])
                for d in first["destinations"]} != {
                (d["dataset"], d["class_id"], d["category"])
                for d in record["destinations"]}:
                warnings.append(
                    "Identical CSVs routed to different labels/datasets: "
                    f"{first['filename']} vs {record['filename']}"
                )

    for record in records:
        canonical = file_groups.root(record["file_id"])
        record["canonical_file_id"] = canonical
        if record["file_id"] != canonical:
            record["duplicate_of_file_id"] = canonical
            if FOLD_EXCLUDE_DUPLICATES:
                record["exclude_from_training"] = True
        record["subject_id"] = subject_groups.root(record["subject_id"])
    return list(dict.fromkeys(warnings))


def _features_by_subject(records: list[dict]) -> tuple[list[str], np.ndarray, list[tuple], list[str]]:
    """Construct dynamic balancing dimensions from actual data distributions."""
    subject_counters: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        subj = r["subject_id"]
        subject_counters[subj]  # Include empty-only subjects as fold groups.
        if r["exclude_from_training"]:
            continue
        count = int(r["sample_rows"])
        counter = subject_counters[subj]
        counter[("overall_files",)] += 1
        counter[("overall_rows",)] += count
        # Multiple destinations are evaluated independently. A source counts
        # only once in the global overall total.
        for d in r["destinations"]:
            dataset, category, cls = d["dataset"], d["category"], d["class_id"]
            for level, suffix in (
                ("dataset_class", str(cls)),
                ("dataset_category", category),
                ("dataset_gesture", category + "/" + r["gesture"]),
            ):
                counter[(level + "_files", dataset, suffix)] += 1
                counter[(level + "_rows", dataset, suffix)] += count
            counter[("orientation_files", dataset, r["orientation"])] += 1
            counter[("direction_files", dataset, r["direction"])] += 1
            counter[("hand_files", dataset, r["hand"])] += 1
            if r["bpm"] is not None:
                counter[("bpm_files", dataset, str(r["bpm"]))] += 1
    for subj in subject_counters:
        subject_counters[subj][("subject_groups",)] = 1
    subjects = sorted(subject_counters)
    feature_keys = sorted({key for c in subject_counters.values() for key in c})
    arr = np.asarray([[c.get(key, 0) for key in feature_keys]
                      for subj, c in sorted(subject_counters.items())], dtype=np.float64)
    tiers = [key[0] for key in feature_keys]
    return subjects, arr, feature_keys, tiers


def _choose_folds(subjects: list[str], raw_features: np.ndarray,
                  feature_keys: list[tuple], tiers: list[str],
                  frozen: dict[str, int]) -> tuple[dict[str, int], float]:
    """Greedy rare-gesture stratification + deterministic local swaps/moves.

    Hard constraint: all files for the same subject share ONE fold. Existing
    assignments are immutable when FOLD_REUSE_EXISTING_ROADMAP is true.
    """
    n, k = len(subjects), FOLD_COUNT
    if n < k:
        raise ValueError(f"Need >= {k} independent groups, found {n}. "
                         "Reduce FOLD_COUNT or collect more participants.")
    # Ignore unfixable dimensions represented by only one independent group.
    supports = (raw_features > 0).sum(axis=0)
    use = (supports >= 2) & (raw_features.sum(axis=0) > 0)
    if not np.any(use):
        # Subject-count remains if >=2 groups.
        raise ValueError("No nonempty balancing features")
    used_keys = [feature_keys[i] for i in np.flatnonzero(use)]
    used_tiers = [tiers[i] for i in np.flatnonzero(use)]
    totals = raw_features[:, use].sum(axis=0)
    per_fold_targets = totals / k
    matrix = raw_features[:, use] / per_fold_targets[None, :]
    tier_sizes = Counter(used_tiers)
    weights = np.array([
        FOLD_BALANCE_WEIGHTS.get(tier, 0.25) / np.sqrt(tier_sizes[tier])
        for tier in used_tiers
    ], dtype=np.float64)
    cover_columns = np.array([
        index for index, (key, support) in enumerate(zip(used_keys, supports[use]))
        if key[0] in ("dataset_class_files", "dataset_category_files", "dataset_gesture_files")
        and support >= 2
    ], dtype=np.int64)
    cover_weights = np.array([
        FOLD_COVERAGE_STRENGTH / np.sqrt(tier_sizes[used_tiers[i]])
        for i in cover_columns
    ], dtype=np.float64)
    target_coverage = np.minimum(k, supports[use][cover_columns])
    identity_weight = {s: i for i, s in enumerate(subjects)}
    fixed = {identity_weight[s]: int(fold) for s, fold in frozen.items()
             if s in identity_weight}
    for fold in fixed.values():
        if not 0 <= fold < k:
            raise ValueError(f"Previous roadmap has invalid fold {fold}")
    fixed_indices = set(fixed)

    # Reuse temporary storage to avoid two large NumPy allocations on every
    # candidate. Operations and reduction order are identical to the original.
    score_buffer = np.empty((k, matrix.shape[1]), dtype=np.float64)

    def score(sums: np.ndarray) -> float:
        if FOLD_REUSE_SCORE_BUFFER:
            np.subtract(sums, 1.0, out=score_buffer)
            np.square(score_buffer, out=score_buffer)
            loss = float(np.sum(weights * np.sum(score_buffer, axis=0)))
        else:
            difference = sums - 1.0
            loss = float(np.sum(weights * np.sum(difference * difference, axis=0)))
        if len(cover_columns):
            represented = np.count_nonzero(sums[:, cover_columns] > 1e-12, axis=0)
            loss += float(np.dot(cover_weights,
                                 np.maximum(0, target_coverage - represented)))
        return loss

    best_loss = float("inf")
    best_assignment: np.ndarray | None = None
    base_rng = np.random.default_rng(FOLD_SEED)
    runs = max(1, int(FOLD_RESTARTS))
    for restart in range(runs):
        fold_of = np.full(n, -1, dtype=np.int16)
        sums = np.zeros((k, matrix.shape[1]), dtype=np.float64)
        fold_subject_counts = np.zeros(k, dtype=int)
        for idx, fold in fixed.items():
            fold_of[idx] = fold
            sums[fold] += matrix[idx]
            fold_subject_counts[fold] += 1
        # Largest / rarest subjects first, with reproducible perturbations to
        # explore alternative allocations across restarts.
        rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
        unassigned = [idx for idx in range(n) if idx not in fixed_indices]
        magnitude = matrix @ weights
        if restart:
            jitter = rng.uniform(0.65, 1.35, size=n)
        else:
            jitter = np.ones(n)
        unassigned.sort(key=lambda idx: (-magnitude[idx] * jitter[idx], subjects[idx]))
        for position, idx in enumerate(unassigned):
            left = len(unassigned) - position
            empty = np.flatnonzero(fold_subject_counts == 0)
            choices = empty if len(empty) == left else np.arange(k)
            # Avoid systematically favoring fold 0 for completely tied options.
            tiebreak = rng.permutation(k)
            pick = None
            minimum = float("inf")
            for fold in choices:
                sums[fold] += matrix[idx]
                candidate = score(sums)
                sums[fold] -= matrix[idx]
                if candidate < minimum - 1e-9 or (
                    abs(candidate - minimum) <= 1e-9 and
                    (pick is None or np.where(tiebreak == fold)[0][0] <
                     np.where(tiebreak == pick)[0][0])
                ):
                    minimum, pick = candidate, int(fold)
            assert pick is not None
            fold_of[idx] = pick
            sums[pick] += matrix[idx]
            fold_subject_counts[pick] += 1        current = score(sums)
        # Coordinate descent with whole-subject moves and swaps. A person is
        # never split, and frozen subjects never move. Stop at a local optimum.
        movable = [i for i in range(n) if i not in fixed_indices]
        for _ in range(FOLD_REFINE_PASSES):
            best_action = None
            next_best = current - 1e-8
            for i in movable:
                a = int(fold_of[i])
                if fold_subject_counts[a] <= 1:
                    continue
                for b in range(k):
                    if b == a:
                        continue
                    sums[a] -= matrix[i]
                    sums[b] += matrix[i]
                    trial = score(sums)
                    sums[a] += matrix[i]
                    sums[b] -= matrix[i]
                    if trial < next_best:
                        next_best, best_action = trial, ("move", i, a, b)
            for pos, i in enumerate(movable):
                a = int(fold_of[i])
                for j in movable[pos + 1:]:
                    b = int(fold_of[j])
                    if a == b:
                        continue
                    difference = matrix[j] - matrix[i]
                    sums[a] += difference
                    sums[b] -= difference
                    trial = score(sums)
                    sums[a] -= difference
                    sums[b] += difference
                    if trial < next_best:
                        next_best, best_action = trial, ("swap", i, j, a, b)
            if best_action is None:
                break
            if best_action[0] == "move":
                _, i, a, b = best_action
                sums[a] -= matrix[i]
                sums[b] += matrix[i]
                fold_of[i] = b
                fold_subject_counts[a] -= 1
                fold_subject_counts[b] += 1
            else:
                _, i, j, a, b = best_action
                difference = matrix[j] - matrix[i]
                sums[a] += difference
                sums[b] -= difference
                fold_of[i], fold_of[j] = b, a
            current = next_best
        if current < best_loss - 1e-8:
            best_loss, best_assignment = current, fold_of.copy()
    assert best_assignment is not None
    result = {subj: int(best_assignment[i]) for i, subj in enumerate(subjects)}
    assert all(fold in result.values() for fold in range(k))
    assert all(result[s] == f for s, f in frozen.items() if s in result)
    return result, best_loss


def _load_frozen_subject_assignments(records: list[dict]) -> tuple[dict[str, int], list[str]]:
    if not FOLD_REUSE_EXISTING_ROADMAP or not FOLD_ROADMAP_PATH.is_file():
        return {}, []
    with FOLD_ROADMAP_PATH.open("r", encoding="utf-8") as stream:
        previous = json.load(stream)
    if previous.get("schema_version") != 2 or previous.get("n_folds") != FOLD_COUNT:
        raise ValueError("Existing roadmap has a different version/fold count: "
                         f"{FOLD_ROADMAP_PATH}. Set FOLD_REUSE_EXISTING_ROADMAP=False "
                         "to regenerate deliberately, or use a new output path.")
    prev_by_file = {r["file_id"]: int(r["fold_id"]) for r in previous.get("files", [])
                    if r.get("fold_id") is not None}
    prev_by_subject = {s: int(f) for s, f in
                       previous.get("subject_assignments", {}).items()}
    locked_by_group: dict[str, set[int]] = defaultdict(set)
    for record in records:
        subject = record["subject_id"]
        if subject in prev_by_subject:
            locked_by_group[subject].add(prev_by_subject[subject])
        if record["file_id"] in prev_by_file:
            locked_by_group[subject].add(prev_by_file[record["file_id"]])
    conflicts = {s: sorted(ids) for s, ids in locked_by_group.items() if len(ids) > 1}
    if conflicts:
        raise ValueError("Identity grouping changes conflict with previously frozen "
                         f"folds: {conflicts}. Resolve aliases, then intentionally "
                         "regenerate with FOLD_REUSE_EXISTING_ROADMAP=False.")
    fixed = {s: next(iter(ids)) for s, ids in locked_by_group.items()}
    return fixed, ([f"Reused {len(fixed)} participant fold assignments from the existing JSON"]
                   if fixed else [])


def _fold_summary(records: list[dict], assignments: dict[str, int]) -> tuple[list[dict], list[dict]]:
    warnings: list[dict] = []
    summary: list[dict] = []
    active = [r for r in records if not r["exclude_from_training"]]
    # Every destination may represent a different semantic class_id.
    dims: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in active:
        for dest in record["destinations"]:
            base = dest["dataset"]
            for name, value in (
                ("class", str(dest["class_id"])),
                ("category", dest["category"]),
                ("gesture", dest["category"] + "/" + record["gesture"]),
            ):
                dims[f"{base}/{name}/{value}"]["subjects"].add(record["subject_id"])
                dims[f"{base}/{name}/{value}"]["folds"].add(record["fold_id"])
    for feature, counts in sorted(dims.items()):
        possible = min(FOLD_COUNT, len(counts["subjects"]))
        observed = len(counts["folds"])
        if observed < FOLD_COUNT or observed < possible:
            warnings.append({
                "stratum": feature,
                "independent_subjects": len(counts["subjects"]),
                "folds_represented": observed,
                "max_possible_folds": possible,
                "reason": "insufficient_independent_subjects" if possible < FOLD_COUNT
                          else "optimizer_tradeoff_or_frozen_assignments",
                "additional_avoidable_gap": observed < possible,
            })
    for fold in range(FOLD_COUNT):
        fold_records = [r for r in active if r["fold_id"] == fold]
        by_dataset: dict[str, dict] = {}
        for rec in fold_records:
            for dest in rec["destinations"]:
                group = by_dataset.setdefault(dest["dataset"], {
                    "file_count": 0, "sample_rows": 0,
                    "by_class": defaultdict(lambda: {"files": 0, "rows": 0}),
                    "by_gesture": defaultdict(lambda: {"files": 0, "rows": 0}),
                })
                group["file_count"] += 1
                group["sample_rows"] += rec["sample_rows"]
                c = group["by_class"][str(dest["class_id"])]
                c["files"] += 1
                c["rows"] += rec["sample_rows"]
                g = group["by_gesture"][dest["category"] + "/" + rec["gesture"]]
                g["files"] += 1
                g["rows"] += rec["sample_rows"]
        summary.append({
            "fold_id": fold,
            "subject_groups": sum(1 for f in assignments.values() if f == fold),
            "unique_source_files": len(fold_records),
            "unique_sample_rows": sum(r["sample_rows"] for r in fold_records),
            "excluded_duplicate_or_empty_files": sum(
                1 for r in records if r["fold_id"] == fold and r["exclude_from_training"]
            ),
            "datasets": by_dataset,
        })
    return summary, warnings


def _validate_roadmap(records: list[dict], assignments: dict[str, int]) -> None:
    assert len(set(r["file_id"] for r in records)) == len(records), "Duplicate file IDs"
    by_canonical = {}
    for record in records:
        fold = record["fold_id"]
        assert fold == assignments[record["subject_id"]]
        assert isinstance(fold, int) and 0 <= fold < FOLD_COUNT
        canonical = record["canonical_file_id"]
        if canonical in by_canonical:
            assert by_canonical[canonical] == fold, "Identical recordings leaked between folds"
        by_canonical[canonical] = fold
        for dest in record["destinations"]:
            assert (OUTPUT_BASE / dest["relative_csv_path"]).is_file(), (
                f"Missing organized output {dest['relative_csv_path']}"
            )
    assert len(set(assignments.values())) == FOLD_COUNT, "Some folds are empty"


def _atomic_write_json(data: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as out:
            json.dump(data, out, indent=2, ensure_ascii=False)
            out.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _export_roadmap(records: list[dict], unmatched_count: int,
                    failed_count: int) -> None:
    if not records:
        print("[FOLDS] No successfully organized CSVs: roadmap not overwritten")
        return
    warnings = _mark_duplicates_and_groups(records)
    if not FOLD_EXCLUDE_DUPLICATES and any(r["duplicate_of_file_id"] for r in records):
        warnings.append("Duplicate exclusion disabled; duplicates are counted multiple times "
                        "in the balancing objective and training loaders must handle them")
    frozen, prev_messages = _load_frozen_subject_assignments(records)
    subjects, matrix, feature_keys, tiers = _features_by_subject(records)
    assignments, objective = _choose_folds(subjects, matrix, feature_keys, tiers, frozen)
    for record in records:
        record["fold_id"] = assignments[record["subject_id"]]
    _validate_roadmap(records, assignments)
    folds, feature_warnings = _fold_summary(records, assignments)
    identity_review: dict[str, dict] = {}
    grouped = defaultdict(list)
    for record in records:
        grouped[record["subject_id"]].append(record)
    for subject, members in sorted(grouped.items()):
        unique_names = sorted({r["subject_full_name_hint"] for r in members})
        user_values = sorted({r["user_raw"] for r in members if r["user_raw"]})
        questionable = (len(unique_names) > 1 or len(user_values) > 1 or
                        any(r["identity_low_quality_user"] for r in members))
        if questionable or any(r["user_values"] for r in members):
            identity_review[subject] = {
                "full_name_hints": unique_names,
                "user_values": user_values,
                "source_file_count": len(members),
                "warning": "Review before claiming verified person-level separation",
            }
    fingerprint = hashlib.sha256("\n".join(
        f"{r['file_id']}|{r['content_sha256']}|{r['sample_rows']}"
        for r in sorted(records, key=lambda item: item["file_id"])
    ).encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 2,
        "status": ("IDENTITIES_REQUIRE_REVIEW" if identity_review
                   else "PERSON_GROUPING_PROVISIONAL"),
        "default_fold_plan": "person_disjoint_gesture_aware",
        "n_folds": FOLD_COUNT,
        "fold_index_base": 0,
        "seed": FOLD_SEED,
        "source_folder": str(INPUT_FOLDER),
        "output_folder": str(OUTPUT_BASE),
        "data_fingerprint_sha256": fingerprint,
        "person_grouping_mode": FOLD_PERSON_GROUPING,
        "optimization_objective": objective,
        "reuse_existing_roadmap": FOLD_REUSE_EXISTING_ROADMAP,
        "frozen_subject_group_count": len(frozen),
        "summary": {
            "organized_source_files": len(records),
            "total_input_rows": sum(r["sample_rows"] for r in records),
            "unique_usable_files": sum(not r["exclude_from_training"] for r in records),
            "unique_usable_rows": sum(r["sample_rows"] for r in records
                                      if not r["exclude_from_training"]),
            "duplicated_source_files": sum(bool(r["duplicate_of_file_id"]) for r in records),
            "independent_group_count": len(subjects),
            "unmatched_source_files": unmatched_count,
            "failed_source_files": failed_count,
        },
        "subject_assignments": assignments,
        "folds": folds,
        "identity_review_required": identity_review,
        "coverage_warnings": feature_warnings,
        "warnings": list(dict.fromkeys(warnings + prev_messages)),
        "files": sorted(records, key=lambda r: r["file_id"]),
    }
    # Write last: never publish a partially validated / partially written JSON.
    _atomic_write_json(manifest, FOLD_ROADMAP_PATH)
    print("\n[FOLDS] Five-fold roadmap exported:", FOLD_ROADMAP_PATH.resolve())
    print("[FOLDS] Group count:", len(subjects), "unique usable recordings:",
          manifest["summary"]["unique_usable_files"])
    for fold in folds:
        print(f"[FOLDS] {fold['fold_id']}: groups={fold['subject_groups']} "
              f"files={fold['unique_source_files']} "
              f"rows={fold['unique_sample_rows']:,}")
    print("[FOLDS] Identity groups requiring review:", len(identity_review))
    print("[FOLDS] Coverage limitations / warnings:", len(feature_warnings))



def _process_one_file(source_path: Path, sorted_keywords: tuple[str, ...],
                      fold_aliases: dict[str, str]) -> tuple[str, int, dict | None, str]:
    """Isolated worker: one CSV, at most one read and one serialization.

    Returning only a small JSON-compatible record (never the wide dataframe)
    keeps IPC traffic and worker memory low.  Parent emits logs in filename
    order, so results are deterministic regardless of completion order.
    """
    match = find_mapping(source_path.name, sorted_keywords)
    if match is None:
        try:
            unmatched_destination = UNMATCHED_FOLDER / source_path.name
            atomic_copy_file(source_path, unmatched_destination)
            return ("unmatched", 0, None, f"⚠️  UNMATCHED: {source_path.name}")
        except Exception as exc:
            return ("failed", 0, None,
                    f"❌ Error copying unmatched {source_path.name}: {exc}")

    keyword, specification = match
    raw_targets, target_sub_folder, target_class_id = specification
    target_main_folders = normalize_target_folders(raw_targets)
    output_paths = [
        OUTPUT_BASE / main_folder / target_sub_folder / source_path.name
        for main_folder in target_main_folders
    ]
    try:
        dataframe = pd.read_csv(source_path, **CSV_READ_KWARGS)
        dataframe = set_class_id_first(dataframe, target_class_id)
        record = (_record_for_file(
            source_path, dataframe, keyword, target_main_folders,
            target_sub_folder, target_class_id, output_paths, fold_aliases,
        ) if FOLD_ROADMAP_ENABLED else None)
        atomic_write_dataframe(dataframe, output_paths[0])
        del dataframe  # Free the wide dataframe before copying/hash computation.
        for additional_output in output_paths[1:]:
            atomic_copy_file(output_paths[0], additional_output)
        if record is not None and FOLD_DETECT_DUPLICATES:
            record["organized_content_sha256"] = _sha256_file(output_paths[0])
        destinations = ", ".join(
            str(path.parent.relative_to(OUTPUT_BASE)) for path in output_paths
        )
        message = (f"✅ {source_path.name} [{keyword}] -> {destinations} "
                   f"(ID: {target_class_id})")
        return ("matched", len(output_paths), record, message)
    except Exception as exc:
        return ("failed", 0, None, f"❌ Error processing {source_path.name}: {exc}")


def process_files() -> None:
    run_start = time.perf_counter()
    validate_gesture_map()
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    UNMATCHED_FOLDER.mkdir(parents=True, exist_ok=True)
    if not INPUT_FOLDER.is_dir():
        print(f"❌ Error: source folder {str(INPUT_FOLDER)!r} was not found.")
        return
    files = sorted(path for path in INPUT_FOLDER.iterdir()
                   if path.is_file() and path.suffix.lower() == ".csv")
    print(f"📂 Found {len(files)} CSV files. Processing with "
          f"{min(CSV_WORKERS, len(files))} worker(s)...", flush=True)
    sorted_keywords = tuple(sorted(GESTURE_MAP,
                                   key=lambda value: (-len(value), value)))
    aliases = _manual_alias_lookup() if FOLD_ROADMAP_ENABLED else {}

    matched_input_files = 0
    generated_output_files = 0
    unmatched_files = 0
    failed_input_files = 0
    fold_records: list[dict] = []

    def collect(results):
        nonlocal matched_input_files, generated_output_files
        nonlocal unmatched_files, failed_input_files
        for status, copies, record, message in results:
            print(message)
            if status == "matched":
                matched_input_files += 1
                generated_output_files += copies
                if record is not None:
                    fold_records.append(record)
            elif status == "unmatched":
                unmatched_files += 1
            else:
                failed_input_files += 1

    if CSV_WORKERS < 1:
        raise ValueError("CSV_WORKERS must be >= 1")
    if CSV_WORKERS == 1 or len(files) <= 1:
        collect(map(_process_one_file, files,
                    repeat(sorted_keywords), repeat(aliases)))
    else:
        # map(..., chunksize=1) is memory-bounded in the workers and yields
        # results in original filename order even though CSVs run in parallel.
        with ProcessPoolExecutor(max_workers=min(CSV_WORKERS, len(files))) as pool:
            collect(pool.map(_process_one_file, files,
                             repeat(sorted_keywords), repeat(aliases), chunksize=1))

    csv_elapsed = time.perf_counter() - run_start
    print("\n" + "=" * 48)
    print("🎉 Done")
    print(f"Matched input files:    {matched_input_files}")
    print(f"Generated CSV copies:   {generated_output_files}")
    print(f"Unmatched input files:  {unmatched_files}")
    print(f"Failed input files:     {failed_input_files}")
    print(f"Output root:            {OUTPUT_BASE.resolve()}")
    print(f"[PERF] CSV organization: {csv_elapsed:.2f}s "
          f"({len(files) / max(csv_elapsed, 1e-9):.2f} input CSVs/s)")
    if FOLD_ROADMAP_ENABLED:
        fold_start = time.perf_counter()
        _export_roadmap(fold_records, unmatched_files, failed_input_files)
        print(f"[PERF] Fold roadmap: {time.perf_counter() - fold_start:.2f}s")
    print(f"[PERF] Total runtime: {time.perf_counter() - run_start:.2f}s")

if __name__ == "__main__":

    process_files()