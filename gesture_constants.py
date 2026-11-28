"""Project constants read by the TAP F1 trainer (train_tap_f1.py).

Only the two model-size limits are project decisions (96 KB maximum, 5 KB "realistic" minimum); every other value
is the trainer's own built-in default, written out so nothing is implicit. N_TRIALS can be lowered for a quick
check with LEGACY_N_TRIALS=<n> (the Makefile's smoke target does that); a normal run uses 100 per model family.
"""
import os

import numpy as np

FEATURES_VAR_TYPE = np.float32
GESTURE_GROUPS_ID = {"TAP": 1, "NONE": 0}
GESTURE_GROUPS_ID_TRAIN = {"TAP": 1}

MAX_MODEL_SIZE_BYTES = 96 * 1024        # 98304: the largest tree payload the firmware takes
MIN_REALISTIC_SIZE = 5 * 1024           # 5120: below this, fewer bytes are not rewarded by the size objective

N_FOLDS = 5                             # must match DATA/fold_roadmap_files.json
N_TRIALS = int(os.environ.get("LEGACY_N_TRIALS", "100"))
