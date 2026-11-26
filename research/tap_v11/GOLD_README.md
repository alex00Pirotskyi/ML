# GOLD TAP label correction (v11.1)

This is the **small, conservative replacement for the proposed large V11 redesign**. It is an executable correction pass over an existing ACC + EMG sensor teacher export. It does not train another teacher, fit a runtime model, access participant names, or change the production trainer.

## What is grounded in evidence

The original dominant-axis ACC script labels a narrow impulse but only within pointer cycles; a missed pointer press therefore cannot create a positive. The published fusion labeler detects independent sensor peaks, and already has double-partner promotion. In reference commit `28efd6e1cb992475f1f8e4caf9b216543b91c779`, [label_gestures.py lines 294–303](https://github.com/alex00Pirotskyi/ML/blob/28efd6e1cb992475f1f8e4caf9b216543b91c779/claude_labeling/label_gestures.py#L294-L303) promotion is limited to DoubleTaps filenames, and the promoted frame labels do not receive distinct event IDs or a consistently positive event-report entry. The new code repairs those issues for newly promoted candidates.

The same repository's [historical labeling results](https://github.com/alex00Pirotskyi/ML/blob/28efd6e1cb992475f1f8e4caf9b216543b91c779/claude_research_tap/TAP_SWIPE_LABELING_SYSTEM_V2_2026-11-16.md#L92-L113) show sensor-aligned 0–80 ms labels at 79.0% pointer-press recall versus 57.4% for raw PointerState at matched 2 FP/min. The [generalization report](https://github.com/alex00Pirotskyi/ML/blob/28efd6e1cb992475f1f8e4caf9b216543b91c779/research/gesture/TAP_GENERALIZATION_RESULTS.md#L55-L75) reports ACC-only recall 0.597 versus approximately 0.865 for raw126 at matched 2 FP/min. Thus keep the existing fused teacher, not an ACC-only or amplitude-only detector. These are historical reports from old experiments, not newly measured V11 performance. **Do not reuse their person splits.**

Claude's newer archive analysis (reported by the owner) found 92–98% positive agreement, 1,552 inferred missed taps and 599 likely second/fast partner events. The archived sensor values/row counts differ from current raw data. Those counts are hypotheses and are not independently verified here. We therefore do not join the old archive to new raw files by row, trust every old zero, or assert an independently measured real-life FPR.

## What changes

| Input | GOLD action |
|---|---|
| Existing confirmed tap/soft tap/swipe | Keep original positive block exactly, except confirmed disconnection |
| Unknown strong independent sensor peak after confirmed tap in DoubleTaps or >=240 bpm file | Recover one short onset if all guards pass; distinct ID and positive event report |
| Weak, close, ambiguous, or unsupported candidate | Leave unknown; never force a second tap |
| Existing known zero | Never overwrite with a recovered positive |
| Valid hard-negative gesture folder | Lock tap label to zero |
| Two or more confidently OFF electrodes | Lock tap label to zero, preserving runtime disconnection protection |
| Hold/drag recording | Keep base tap labels, including unknown transitions; only the existing OFF rule may override |
| Confident quiet unknown background | Optional separate ablation, outside every sensor/pointer guard |
| Uncertain/movement background | Unknown by default; optional low-score experimental arm requires dense fused teacher score |
| All other intervals | Preserve original labels |

The current fused teacher already includes EMG and WL context. This is the primary confirmation, not an absolute EMG floor. An optional `--require-emg-active` arm requires existing contact-model ACTIVE evidence near recovered partners; measure it rather than enabling by default. Soft tap positives from the base remain unchanged. This version **does not invent a general weak-soft-tap recovery rule** or claim to find every missed tap when the base teacher does not propose it. We deliberately target the measured second/fast-tap issue first. No confidence boosting/self-training loop or new teacher.

The proposed recovery uses the existing calibrated strong threshold, an original confirmed anchor 150–450 ms earlier, at least 150 ms separation from every accepted anchor/recovery, good contact throughout the new onset, and a wholly unknown window. Recovered events never become anchors. An active merged pointer extent does not block a distinct second onset. When the current teacher supplies `window_start_ms/window_end_ms` for an event, use that short observable interval; otherwise a 60 ms sensor-aligned block is the comparison default. No release is invented.

Background uses contact-confirmed REST and a conservative legacy-style robust ACC quiet test, with guard zones around all exported tap proposals (including weak/rejected ones), raw pointer edges/active state, existing positives, acquisition gaps, start/end and holds. It does not assert that a missing sparse peak is proof of a non-tap. The optional movement-background arm additionally needs a calibrated low dense fused score; do not enable it without that evidence. If no safe regions exist, zero additions is correct.

## Strict file-fold contract

GOLD itself **has no fold logic** and reads no User/participant fields. It consumes the **actual latest local production** gesture-aware file-split teacher output. The published reference `main` still used person-fold dispatch in `claude_labeling/label_gestures.py` and person-fold trainer CV at inspection; do not run that code to generate GOLD inputs.

Use the owner's existing file/subtype split and keep identical recordings together. For nested evaluation, fit/calibrate the teacher and generate any training pseudo-labels within each outer training partition. Never use validation or reserved files to fit thresholds or choose label variants. No User-based split, folds, features or calibration. A small metadata file is an explicit contract with the caller, not cryptographic proof of how the teacher was trained. The integration owner must verify the actual fold manifest and calibration source.

## Input format and one-command execution

The current production pipeline must export the existing sidecar schema (or adapt it once):

- Raw tree: each original `.csv` or `.csv.gz`, with exact Timestamp in ms and optional PointerState, ACC bins 1–5 for quiet-background arm.
- Base sidecar tree: same relative recording paths, suffix `.labels.csv.gz`; Timestamp, gesture_kind (0 none, 1 tap, 2 swipe, 3 hold), tap_label, tap_active, tap_event_id, hold_label, n_ch_off, ch1_label..ch3_label. Each row corresponds exactly to raw Timestamp, and each channel label is -1 unknown, 0 REST, 1 ACTIVE, 2 OFF. Counts must agree. No interpolation or fuzzy join.
- Base `_events.csv.gz`: recording (relative raw CSV path), type, tier, t_on/t_ptr/t_rel in relative ms, score, labeled, k. Preserve *all* candidate events, including ORPHAN and INPRESS. A filtered positive-only event table cannot recover missed partners. Keep full-precision scores when possible. Optional observable window columns as above. 
- Base `_summary.json`: totals.failed=0 and recordings mapping keyed by exact relative raw filenames.
- Actual file-fold teacher calibration JSON: required keys `split_policy: "gesture_aware_file"`, `feature_families: ["ACC", "EMG", "WL"]`, `run_id` from the actual run, `strong_score` at its chosen negative FPM constraint. Do not copy 0.966 or any historical person-fold threshold. For experimental movement restoration, also include calibrated `low_score` and export `tap_teacher_score` aligned per raw row. Do not fabricate those values.

The script only requires numpy and pandas. From the existing repository:

```bash
git fetch origin research/tap-v11-reviewed-seed
git restore --source=FETCH_HEAD --worktree -- research/tap_v11/gold_labeler.py research/tap_v11/test_gold_labeler.py research/tap_v11/GOLD_README.md
python -m unittest discover -s research/tap_v11 -p 'test_gold_labeler.py' -v
```

Use the existing virtual environment. Set these paths to the real current file-fold export for TAP. The example uses /tmp for new variants to keep production untouched.

```bash
export RAW=DATA/DATA_TAP
export BASE=/actual/path/current_file_fold_labels/DATA_TAP
export TEACHER_SPEC=/actual/path/current_file_fold_teacher_calibration.json
python -u research/tap_v11/gold_labeler.py \
  --raw "$RAW" --base "$BASE" --teacher-spec "$TEACHER_SPEC" \
  --out /tmp/gold_recover --mode recover
```

For the controlled four-arm experiment:

```bash
# A = unchanged "$BASE"
# B = /tmp/gold_recover (above)
python -u research/tap_v11/gold_labeler.py \
  --raw "$RAW" --base "$BASE" --teacher-spec "$TEACHER_SPEC" \
  --out /tmp/gold_background --mode background
python -u research/tap_v11/gold_labeler.py \
  --raw "$RAW" --base "$BASE" --teacher-spec "$TEACHER_SPEC" \
  --out /tmp/gold_both --mode both
```

Run the same process for the relevant DATA_LONG_PRESS label root only if the current trainer consumes that root for TAP negatives. Never delete its hold/drag protection. Do not include long-press starts/releases as known negatives merely because of folder name.

GOLD writes a new directory atomically, refuses to overwrite inputs or output, and fails if files/columns/time alignment do not match. It produces standard sidecars, corrected `_events.csv.gz`, original `_events_reference.csv.gz`, updated `_summary.json`, and `_gold_report.json` with per-recording counts and reasons. The CLI does not relabel raw source CSVs.

### In-memory integration

If the local pipeline no longer has the historical sidecar schema, adapt at the export point (not by reintroducing the old person teacher):

```python
from gold_labeler import Policy, correct_recording

corrected_labels, corrected_events, audit = correct_recording(
    raw_dataframe, current_labels, all_current_teacher_events,
    recording_relative_path,
    Policy(strong_score=current_calibrated_threshold, mode="recover"),
)
```

Use the actual local trainer's existing label import or add a minimal adapter. The public reference trainer's `labels_complete` and label root constants may reject new sidecars; don't change them silently. Do not use the public person-fold trainer to test this. Integrate the actual current file-fold implementation.

**Critical:** in A/B/C/D comparison, use the same original event references and actual firmware replay for every arm. If the trainer currently reads `_events.csv.gz` for both sampling and scoring, point scoring to the original frozen base `_events.csv.gz` (or output `_events_reference.csv.gz`) separately, and use corrected events only for training/sampling. Otherwise adding new positives can change the answer key and artificially improve the metric. No independent physical ground truth is created by this correction.

Compare at fixed 0.85, identical file folds/seeds/features/size ladder to the current 100 KB ceiling, same worst-movement constraints, negative exposure and actual firmware replay. Measure finger/fist/all movement FP/min, normal/soft/first/second/fast/double-pair recall and latency, exact C bytes and parity, and nominal/stress protection. SlowPinch is separate. Select the smallest real improvement only at equal or better protection. Keep A if B/C/D fails.

## Validation status and limitations

Synthetic tests are provided. A successful full-data run and physical/firmware test are not yet available from this session because its execution environment could not open the uploads/run Python. GitHub source was inspected and this is an implementation candidate, **not a certified gold standard**. Defaults are conservative starting choices based on the existing promotion rule/legacy support gate, not newly optimized thresholds. The script does not train the teacher, execute the production trainer, or validate the claimed split provenance. The current local code and data must be used to execute the tests/benchmark. No changes were made to production main or raw datasets.
