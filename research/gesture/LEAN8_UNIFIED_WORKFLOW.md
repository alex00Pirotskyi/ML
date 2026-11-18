# Lean-8 + unified gesture training workflow

## Goal

Train TAP/SWIPE and LONG_PRESS on exactly the inputs the firmware computes, including the 24
`p_core->cload_lean_8` fields:

1. Raw recording CSV (never modified)
2. Offline Lean-8 generation that is **bit-exact with the firmware handler**
3. Unified binary labels (from the raw recordings)
4. Replay-scored LightGBM training (raw126, lean8_24, raw126 + instant Lean-8, all 150)
5. C export through `export_plus.py`
6. Firmware compile and real-life test

Everything runs from one command, `python -u research/gesture/train_unified.py`, which does steps 2–5.
The generator can also be run alone:

```bash
python -u research/gesture/generate_lean8_features.py
```

## 1. Lean-8 generation

| | Path |
|---|---|
| Input (read only) | `DATA/DATA_TAP`, `DATA/DATA_LONG_PRESS` |
| Output (git-ignored, disposable) | `DATA/DATA_TAP_LEAN8`, `DATA/DATA_LONG_PRESS_LEAN8`, `DATA/lean8_manifest.json` |

Every original line is kept byte-for-byte. 27 columns are appended:

- the 24 firmware fields
- `lean8_frames`: number of `lean_8_features_handler()` calls so far
- `lean8_new_frame`: 1 on the row where a frame was handed to the handler
- `lean8_features_ready`: `lean_8_features_ready()`

Generation is incremental: a file is redone only if its source or the generator changed. A full run takes
about 8 min on 2 cores and adds about 1.2 GB.

### How the logged data maps onto the firmware

Measured on all 5,352 recordings:

| Fact | Value | Consequence |
|---|---|---|
| Logged rows per second | ~140–210 (every new p_core state, mostly ACC) | rows are **not** handler calls |
| EMG frames per second (rows where `emg_psd` changes) | 46.4 (p5–p95: 45.0–46.7) | the handler is stepped **once per EMG frame**; values hold between frames |
| Frames logged split across two rows | ~11% (PSD on row i, WL on row i+1) | the frame is complete, and handed to the handler, on row i+1 |
| `emg_psd_mean` vs `emg_psd` change rows | identical | `Pm`/`r01`/`r32` use `emg_psd_mean`, as the firmware does |
| History lengths (`LEAN8_FEATURE_UPDATE_HZ` = 50) | 150 / 500 calls | = **3.2 s / 10.8 s** at 46.4 frames/s. If you want true 3 s / 10 s, set the define to 46 in the firmware and here |

### Firmware parity

The generator reproduces the firmware arithmetic exactly:

- WL history as Q12 `uint16`.
- Exact integer `sd3`, with nearest-edge padding during warm-up.
- Partial-window min/max during warm-up, with a rescan on expiry.
- `rise10` clipped at 0.
- Zone power from the 32-entry LUT plus the expm1 polynomial, clamped at 1e-4; total power clamped at 1e-4.
- `log10` via the IEEE-754 polynomial.
- All float32, in C operation order.

It is checked three ways:

1. Against a line-by-line Python port of `lean_8_features_handler` (unit test; runs in CI).
2. Against **the firmware C code itself**, compiled for the host with stub `arm_math.h` /
   `psd_configuration_table.h`. This runs on 24 sample recordings at every generation, whenever the
   header is available locally. Measured: **max difference 0.0** over 25,990 frames (40 recordings),
   ready flag identical. A mismatch aborts generation.
3. CI checks the same exported model against a Lean-8-capable synthetic `t_core` for C-compilation.

The firmware header is **not committed** (this repository is public). Put it at
`research/gesture/_lean8_reference/core_type_def.h` (git-ignored), or point
`MUDRA_LEAN8_FIRMWARE_HEADER` at it. Without it, generation still runs and the parity check is
reported as "not run".

Remaining differences from the device:

- The host build has no FMA; ARM GCC may fuse multiply-adds, giving ±1 ulp.
- The real `emg_wl_type` / `emg_psd_type` widths are stubbed as `uint32_t` / `uint16_t`.

### What the earlier generator got wrong

Measured on 25 recordings:

| Feature | Median error | p95 error | Spread of the feature (IQR) |
|---|---|---|---|
| `Pm` | 2.3 | 11.4 | 11.2 |
| `r01` | 2.1 | 6.7 | 4.3 |
| `r32` | 1.6 | 4.8 | 2.5 |
| `sd3` | 0.03 | 0.17 | 0.17 |
| `min10` / `rise10` | — | 0.32 | 0.33 |

The causes:

- It stepped per row (150/500 rows ≈ 0.9 s / 3 s).
- It used the instant `emg_psd[0]` instead of `emg_psd_mean`.
- It clamped at 1e-12, without Q12 quantisation.

## 2. Warm-up policy (please decide)

Each recording is treated as a fresh sensor stream (`lean_8_features_reset()`). With the firmware's
500-frame `lean_8_features_ready()`, most recordings never become ready:

| Rows used from | ≈ time | Labelled taps kept | Holds kept |
|---|---|---|---|
| frame 50 (default) | 1.1 s | 93% | 97% |
| frame 150 (`sd3` full) | 3.2 s | 68% | 94% |
| frame 500 (`lean_8_features_ready()`) | 10.8 s | **14%** (SWIPE 8%) | 73% |

In addition, every CONNECTED, DISCONNECTED, LOOSE and TABLE recording is shorter than 500 frames.

The trainer uses two gates (`PIPELINE.md`, `FEATURE_SELECTION_RESULTS.md` §2):

- **Training examples** from frame 50 (`MUDRA_TRAIN_MIN_FRAMES`). This skips the least reliable first second
  and keeps the firmware's warm-up values (nearest-edge padding), which still teach the model.
- **Scoring, ranking and choice** from frame 500 (`MUDRA_EVAL_MIN_FRAMES`), because the firmware returns from
  `model_execution()` until `lean_8_features_ready()`.

If the firmware is changed to run the models once `seen >= 50`, set `MUDRA_EVAL_MIN_FRAMES=50` so that the
search scores those rows too.

The `raw126_lean8_instant12` feature set adds only `lw`, `Pm`, `r01` and `r32`. These have no history,
so they behave identically during and after warm-up.

## 3. Labels

Labels come from the **raw recordings** (`DATA/DATA_TAP` → `DATA/LABELS/DATA_TAP`, etc.) via
`claude_labeling/label_gestures.py`, never from Lean-8 values. Unknown `-1` is excluded.

| Task | 1 | 0 |
|---|---|---|
| TAP/SWIPE | onset | no onset |
| LONG_PRESS | RELEASE/REST (firmware polarity) | HELD |

## 4. Train

```bash
python -u research/gesture/train_unified.py
```

The trainer:

1. Refreshes the Lean-8 trees incrementally and checks parity.
2. Labels the raw recordings.
3. Reads raw and Lean-8 fields from the derived files.
4. Searches an integer number K of fields, taken from a ranking learned inside each CV fold over all 150
   fields (`feature_selection.py`). The named sets below are kept for research scripts and reports:

| Feature set | Fields |
|---|---|
| `raw126` | the 126 raw p_core fields |
| `lean8_24` | the 24 Lean-8 fields |
| `raw126_lean8_instant12` | raw126 + `lw`, `Pm`, `r01`, `r32` |
| `all_available` | all 150 |

5. Trains on rows from frame `MUDRA_TRAIN_MIN_FRAMES` (50) on and scores rows from `MUDRA_EVAL_MIN_FRAMES`
   (500) on. Taps before the scoring gate are not counted as misses.
6. Scores TAP by event replay and LP by the 10-sample Q20 average, on person-grouped folds.
7. Exports headers. The compile check uses a `t_core` that includes `cload_lean_8`.
8. Replays held-out people through the C headers, on all usable rows and on steady-state rows.

Changing the Lean-8 settings starts a separate Optuna study and refuses to reuse an output folder built
with different settings.

## 5. Firmware integration

Exported headers read the fields as `p_core->cload_lean_8.<field>[CH1..CH3]`. Call
`lean_8_features_handler(&m_feat_core)` once per new EMG frame, before the models. Reset on reconnect.

Production policy is unchanged:

- TAP starts a press. LP cannot start one.
- LP predicts RELEASE/REST.
- The firmware averages 10 LP Q20 values.
- The LP release threshold is 0.5 (10-sample Q20 mean ≥ 0.5; was 0.85 until 2026-11-18). The TAP onset
  threshold stays 0.85.
- LP stays binary.

## 6. Never model inputs

The following are labelling, sampling or validation information only:

- PointerState
- labels
- timestamps
- folder name
- subject
- contact scores
- `hold_ms`
- `lean8_frames`, `lean8_new_frame`, `lean8_features_ready`

## 7. Acceptance before production

1. Person-disjoint CV and held-out C-header replay, both all-rows and steady-state.
2. Firmware build with the real `core_type_def.h`.
3. On device: TAP false events, LP soft-hold survival, eventual release, bad-contact recordings.
4. A real-life test on people not used for training.
