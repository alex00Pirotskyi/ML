#!/usr/bin/env bash
# The TAP model, end to end (make train starts this in the background):
#   1. file-level, gesture-aware fold roadmap   fold_split.py build          -> DATA/fold_roadmap_files.json
#   2. labels, feature elimination, CSVs        build_tap_f1_dataset.py      -> DATA/DATA_TAP_F1/
#   3. the legacy F1 trainer                    train_tap_f1.py              -> TAP_MODELS_F1_FILEFOLDS/
#   4. event check of every saved model         evaluate_tap_f1.py           -> TAP_MODELS_F1_FILEFOLDS/event_eval.md
# Settings (environment): TAP_F1_LABELS (legacy | v10rescue | v10 | hybrid), TAP_F1_FEATURES (rfe | newacc | leg | new |
# lite), TAP_F1_TAP_ROW_SHARE (0.5: TAP / SWIPE recordings train on every 2nd row), TAP_F1_WL (0: no absolute-WL input),
# TAP_F1_LITE (1: Lean-lite candidates), TAP_F1_RFE_ROWS (400000), LEGACY_N_TRIALS (trials per family, 100), PYTHON.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=TAP_MODELS_F1_FILEFOLDS
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] TAP_F1 1/4 fold roadmap (recording files, gesture aware; old files keep their fold)"
$PY fold_split.py build
LITE_ARG=""; [ "${TAP_F1_LITE:-1}" = 0 ] && LITE_ARG="--no-lite"
echo "[$(stamp)] TAP_F1 2/4 labels (${TAP_F1_LABELS:-legacy}), inputs (${TAP_F1_FEATURES:-rfe}, WL ${TAP_F1_WL:-0}, Lean-lite ${TAP_F1_LITE:-1}), TAP/SWIPE rows ${TAP_F1_TAP_ROW_SHARE:-0.5} -> DATA/DATA_TAP_F1"
$PY -u build_tap_f1_dataset.py --labels "${TAP_F1_LABELS:-legacy}" --features "${TAP_F1_FEATURES:-rfe}" --tap-row-share "${TAP_F1_TAP_ROW_SHARE:-0.5}" --wl "${TAP_F1_WL:-0}" --rfe-rows "${TAP_F1_RFE_ROWS:-400000}" $LITE_ARG
if [ -d "$OUT" ]; then
  old="$OUT.done_$(date +%Y%m%d_%H%M%S)"
  echo "[$(stamp)] previous results moved to $old"
  mv "$OUT" "$old"
fi
echo "[$(stamp)] TAP_F1 3/4 legacy F1 trainer (Optuna, F1 at 0.85 + size, max 96 KB), trials per family: ${LEGACY_N_TRIALS:-100}"
rc=0
$PY -u train_tap_f1.py || rc=$?
if ls "$OUT"/*_model.pkl >/dev/null 2>&1; then
  echo "[$(stamp)] TAP_F1 4/4 event check (taps per press, false taps per minute of movement recordings)"
  $PY evaluate_tap_f1.py "$OUT"
fi
if [ "$rc" != 0 ]; then echo "[$(stamp)] TAP_F1_FAILED trainer exit code $rc (see the lines above and $OUT/*failures*)"; exit "$rc"; fi
echo "[$(stamp)] TAP_F1_DONE models, headers and event_eval.md in $OUT/"
