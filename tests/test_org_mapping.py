"""org.py routing regressions: participant names cannot override the gesture, finger taps stay finger taps."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import org  # noqa: E402

KEYS = tuple(sorted(org.GESTURE_MAP, key=lambda value: (-len(value), value)))


def category(name: str):
    hit = org.find_mapping(name, list(KEYS))
    return None if hit is None else hit[1][1]


def test_participant_name_with_index_does_not_route_to_finger():
    assert category("indexuser_RightHand_Swipe_Horizontal_Default_60bpm_2026-01-01_12-00-00.csv") == "SWIPE"


def test_index_finger_motion_names_stay_finger():
    assert category("omer_gilinsky_RightHand_Index_Finger_Horizontal_Default_60bpm_2026-09-23_12-39-04-017.csv") == "FINGER"
    assert category("Nikol_IndexFinger_Vertical_Default_60bpm_2026-01-29_12-05-39.csv") == "FINGER"


def test_finger_taps_are_not_finger_motion():
    assert category("Someone_IndexFingerTap_Horizontal_Default_60bpm_2026-01-01_12-00-00.csv") == "INDEX_FINGER_TAP"


def test_no_bare_index_alias():
    assert "index" not in org.GESTURE_MAP


def test_current_recordings_keep_their_folder():
    root = Path(__file__).resolve().parents[1] / "DATA"
    files = [p for sub in ("DATA_TAP", "DATA_LONG_PRESS") for p in (root / sub).glob("*/*.csv.gz")]
    wrong = [p.name for p in files if category(p.name) not in (None, p.parent.name)]
    assert not wrong, wrong[:5]
