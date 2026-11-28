#!/usr/bin/env bash
# The TAP model, end to end (make train starts this in the background):
#   1. file-level, gesture-aware fold roadmap   fold_split.py build          -> DATA/fold_roadmap_files.json
#   2. labels -> no WL -> candidates (ACC pairs, EMG diffs inside one channel, Lean-lite) with the band / contact /
#      hum / rotation augmentation on the training folds -> feature elimination (RFE) -> CSVs
#                                               build_tap_f1_dataset.py      -> DATA/DATA_TAP_F1/
#   3. the legacy F1 trainer                    train_tap_f1.py              -> TAP_MODELS_F1_FILEFOLDS/
#   4. back test of every saved model through the firmware tap logic (clean + band changes), choice of the model and
#      its thresholds                          backtest_tap_f1.py           -> TAP_MODELS_F1_FILEFOLDS/backtest.md
#   5. firmware bundle of the chosen model, compiled and replayed against the back test
#                                               export_tap_bundle.py         -> TAP_MODELS_F1_FILEFOLDS/firmware/
#                                               (core_typedef.h, tap_gestures.h, tap_gestures.c, BUNDLE.md)
# Settings (environment): TAP_F1_LABELS (legacy | v10rescue | v10 | hybrid), TAP_F1_FEATURES (rfe | newacc | leg | new |
# lite), TAP_F1_TAP_ROW_SHARE (0.5: TAP / SWIPE recordings train on every 2nd row), TAP_F1_WL (0: no absolute-WL input),
# TAP_F1_LITE (1: Lean-lite candidates), TAP_F1_EMG (within | robust | all), TAP_F1_AUG (1: training augmentation),
# TAP_F1_RFE_ROWS (400000), LEGACY_N_TRIALS (trials per family, 100), BACKTEST_LAMBDA (2), BACKTEST_SE (1),
# BACKTEST_PREFER (XGBoost), BACKTEST_STRESS_TOP (all),
# TAP_F1_PARITY_FILES (held-out recordings replayed in C; 0 = all), PYTHON.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=TAP_MODELS_F1_FILEFOLDS
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] TAP_F1 1/5 fold roadmap (recording files, gesture aware; old files keep their fold)"
$PY fold_split.py build
LITE_ARG=""; [ "${TAP_F1_LITE:-1}" = 0 ] && LITE_ARG="--no-lite"
echo "[$(stamp)] TAP_F1 2/5 labels (${TAP_F1_LABELS:-legacy}), inputs (${TAP_F1_FEATURES:-rfe}, WL ${TAP_F1_WL:-0}, EMG ${TAP_F1_EMG:-within}, Lean-lite ${TAP_F1_LITE:-1}, augmentation ${TAP_F1_AUG:-1}), TAP/SWIPE rows ${TAP_F1_TAP_ROW_SHARE:-0.5} -> DATA/DATA_TAP_F1"
$PY -u build_tap_f1_dataset.py --labels "${TAP_F1_LABELS:-legacy}" --features "${TAP_F1_FEATURES:-rfe}" --tap-row-share "${TAP_F1_TAP_ROW_SHARE:-0.5}" --wl "${TAP_F1_WL:-0}" --emg "${TAP_F1_EMG:-within}" --aug "${TAP_F1_AUG:-1}" --rfe-rows "${TAP_F1_RFE_ROWS:-400000}" $LITE_ARG
if [ -d "$OUT" ]; then
  old="$OUT.done_$(date +%Y%m%d_%H%M%S)"
  echo "[$(stamp)] previous results moved to $old"
  mv "$OUT" "$old"
fi
echo "[$(stamp)] TAP_F1 3/5 legacy F1 trainer (Optuna, F1 at 0.85 + size, max 96 KB), trials per family: ${LEGACY_N_TRIALS:-100}"
rc=0
$PY -u train_tap_f1.py || rc=$?
brc=0
if ls "$OUT"/*_model.pkl >/dev/null 2>&1; then
  echo "[$(stamp)] TAP_F1 4/5 back test: held-out fold through the firmware tap logic, clean and with band changes"
  $PY -u backtest_tap_f1.py "$OUT" || brc=$?
  if [ "$brc" = 0 ]; then
    echo "[$(stamp)] TAP_F1 5/5 firmware bundle: core_typedef.h, tap_gestures.h, tap_gestures.c (compiled, replayed in C)"
    $PY -u export_tap_bundle.py "$OUT" || brc=$?
  fi
fi
if [ "$brc" != 0 ]; then echo "[$(stamp)] TAP_F1_FAILED back test / bundle exit code $brc (see the lines above)"; exit "$brc"; fi
if [ "$rc" != 0 ]; then echo "[$(stamp)] TAP_F1_FAILED trainer exit code $rc (see the lines above and $OUT/*failures*); the back test and bundle used the models that were saved"; exit "$rc"; fi
echo "[$(stamp)] TAP_F1_DONE firmware bundle in $OUT/firmware/ (BUNDLE.md), back test in $OUT/backtest.md"
