# Lean-8 + unified gesture training workflow

## Goal

Train TAP/SWIPE and LONG_PRESS using the same features that the firmware consumes:

1. Raw recording CSV
2. Offline Lean-8 feature generation
3. Unified binary labeling
4. Unified replay-scored LightGBM training
5. C export through `export_plus.py`
6. Firmware compile + real-life test

The raw recordings are never modified.

## 1. Generate Lean-8 features

Run:

```bash
python -u research/gesture/generate_lean8_features.py
```

Input is **only**:

- `DATA/DATA_TAP`
- `DATA/DATA_LONG_PRESS`

Output is derived:

- `DATA/DATA_TAP_LEAN8`
- `DATA/DATA_LONG_PRESS_LEAN8`

Each CSV keeps the original rows and columns and appends 24 Lean-8 fields plus
`lean8_features_ready`.

EMG feature rate is **50 Hz**.

- 3-second history = 150 frames
- 10-second history = 500 frames
- model inference is allowed only after the 10-second history is ready

## 2. Label

Labels are still generated from the **raw recordings**, not from generated Lean-8 values.

```text
DATA/DATA_TAP              -> DATA/LABELS/DATA_TAP
DATA/DATA_LONG_PRESS       -> DATA/LABELS/DATA_LONG_PRESS
```

The labeler remains the offline teacher. It is not part of firmware inference.

Unknown `-1` remains excluded.

### TAP/SWIPE

- 1 = onset
- 0 = no onset
- -1 = unknown

### LONG_PRESS

The training script converts the unified hold label to firmware polarity:

- 1 = RELEASE/REST
- 0 = HELD
- -1 = unknown

This keeps the production binary approach.

## 3. Train

Run:

```bash
python -u research/gesture/train_unified.py
```

The trainer automatically:

1. generates/reuses Lean-8 datasets;
2. generates/reuses unified labels from the raw source;
3. reads the augmented Lean-8 CSVs;
4. exposes feature sets:
   - raw126
   - lean8_24
   - all_available (150)
5. removes Lean-8 warm-up rows from supervised training;
6. excludes warm-up from replay;
7. performs person-grouped CV;
8. scores TAP by event replay;
9. scores LP by the firmware 10-sample Q20 average;
10. exports production-compatible C headers.

## 4. Firmware parity

The Lean-8 feature names are represented as firmware accessors:

```text
p_core->cload_lean_8.lw[CH1..3]
p_core->cload_lean_8.Pm[CH1..3]
p_core->cload_lean_8.r01[CH1..3]
p_core->cload_lean_8.r32[CH1..3]
p_core->cload_lean_8.sd3[CH1..3]
p_core->cload_lean_8.min10[CH1..3]
p_core->cload_lean_8.rise10[CH1..3]
p_core->cload_lean_8.max10[CH1..3]
```

The host generator uses the 50 Hz EMG stream and causal 150/500-frame histories.

**Important:** the ML repository does not contain the private firmware header itself.
The exact C header must remain the final authority. Before production promotion,
run a C-vs-Python parity test against the actual firmware implementation and verify
the scaling/rounding of PSD conversion and the exact warm-up condition.

## 5. Model policy

Keep LP **binary** for now.

Do not switch to regression.

The current firmware architecture remains:

- TAP starts the press.
- LP cannot initiate a press.
- LP predicts RELEASE/REST.
- Firmware averages 10 LP Q20 predictions.
- release threshold is 0.85.

## 6. What must not enter the model

Never use:

- PointerState
- labels
- timestamps
- folder name
- subject/person ID
- contact labels/scores
- hold_ms

Those are labeling, sampling, or validation information.

## 7. Final acceptance

A candidate is not production-ready from offline CV alone.

Required:

1. person-disjoint CV;
2. C-header replay;
3. firmware compilation;
4. TAP false-event test;
5. LP soft-hold survival;
6. LP eventual release;
7. bad-contact recordings;
8. real-life test on people not used for training.

## Provenance

The derived Lean-8 datasets are disposable and ignored by Git. The raw recording
directories remain the only source data.
