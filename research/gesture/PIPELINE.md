# Gesture models: the pipeline on one page

Two embedded LightGBM models for the Mudra band, exported as C headers with a stable firmware API:

- **TAP_SWIPE**: fires a tap/swipe onset when `TAP_SWIPE_predict()` ≥ 0.85.
- **LONG_PRESS**: class 1 = RELEASE. The firmware averages ten `LONG_PRESS_predict_value()` Q20 outputs and
  releases when the mean is ≥ 0.5.

## Run

    git pull --ff-only
    python -u research/gesture/train_unified.py

There are no flags; settings are environment variables (README, "Settings"). The defaults are the measured
choices. Output goes to `research/gesture/results_unified_v3_feature_selection/`. A run resumes where it
stopped, also after a code update that does not change the data.

The EMG context inputs are an A/B switch with identical rows, labels and split in every arm:

    MUDRA_USE_LEAN8=1     python -u research/gesture/train_unified.py   # 24 float Lean-8 fields (default)
    MUDRA_USE_LEAN8=0     python -u research/gesture/train_unified.py   # none
    MUDRA_USE_LEAN8=lite  python -u research/gesture/train_unified.py   # 15 integer Lean-lite fields (LEAN_LITE.md)

TAP generalization (TAP_GENERALIZATION_RESULTS.md): EMG diffs + placement augmentation, TAP only, scored from frame 50:

    MUDRA_USE_LEAN8=0 MUDRA_TAP_FEATURES=diff MUDRA_TAP_AUG=mild MUDRA_TASKS=tap MUDRA_EVAL_MIN_FRAMES=50 \
        python -u research/gesture/train_unified.py

With `MUDRA_USE_LEAN8=0` the TAP `COMPARISON.md` also replays every model and baseline under a simulated band move /
contact change on the held-out people.

**FAILED EXPERIMENT: balanced pipeline** (`RECIPE=final`; no mid / strong taps on the band, refused unless `MUDRA_ALLOW_FAILED_RECIPE=1`, TAP_FILESPLIT_RESULTS.md): the final14obs data recipe, the
balanced objective (every tap kind counts the same; no movement kind above 7 false taps per minute), one operating
control (the logit shift 0..3, weight 1), and the size ladder (8-100 KB; a model must be 1 point better per 100 KB):

    MUDRA_ALLOW_FAILED_RECIPE=1 MUDRA_TAP_RECIPE=final python -u research/gesture/train_unified.py   # research only

**Final TAP recipe** (TAP_FOCUS_RESULTS.md, "Final recipe"): diffs + augmentation + the 15 gain-invariant Lean-lite
diff inputs + press2 labels, 14 KB, 4-15 leaves, TAP only:

    MUDRA_TAP_RECIPE=final14 python -u research/gesture/train_unified.py
    MUDRA_TAP_RECIPE=final14obs python -u research/gesture/train_unified.py   # + observable-pattern labels (60 ms)
    MUDRA_TAP_RECIPE=final14protect python -u research/gesture/train_unified.py   # obs 40 + distillation + finger x2

The held-out split and the CV folds group by recording file, always (a person split is not allowed; `MUDRA_SPLIT`
accepts only `file`): every person takes part in every fold and in the held-out part (results folder suffix
`_splitfile`). The shipped header is refitted on every file. It is
gesture-aware: the held-out files and the CV folds are stratified by recording type (folder, gesture from the file
name, bpm for TAP / SWIPE), so each part holds its share of every gesture (the log line `UNIFIED_SPLIT_GESTURES` and
`split.json` show the counts).

The repository `Makefile` wraps this on the server: `make train` updates the code (`git pull --ff-only`) and starts
`final14` with the file split and the 10 KB alternative in the background, with a timestamped log under `logs/`.
It refuses to start while another trainer process runs (`FORCE=1` overrides) and prints the log if the run stops
within 20 s. Before starting it runs `prepare_run.py` with the same settings: label folders that no longer match the
recordings (new or changed files) are moved aside as `<folder>.old_<time>` and rebuilt, and a finished run of the
same settings is renamed `<results folder>.done_<time>`, so every `make train` trains from scratch on the current
recordings (an unfinished run resumes). Nothing is deleted. `make status` shows the trials done and any error; `make stop` stops every running trainer. Settings:
`RECIPE`, `SPLIT`, `ALT_KB`, `EXTRA` (e.g. `make train EXTRA="MUDRA_TAP_FALSE_PER_MIN_TARGET=3"`), `PULL=0`.

A completed run is reused only if every recording, label file and event table is byte-for-byte the one it was
trained on (else it stops and names the change); byte-identical copies of a recording are used once.

## Stages and what each one guarantees

| Stage | What it does | Guarantee / check |
|---|---|---|
| 1. Lean-8 features | `generate_lean8_features.py` emulates `lean_8_features_handler` frame by frame, giving 24 firmware fields next to the 126 raw ones | Bit-exact with the firmware C code on 24 recordings at every generation (`firmware_parity`) |
| 2. Labels | `claude_labeling/label_gestures.py`: tap teacher + PointerState fusion (TAP onsets; LP holds, releases and taps) | Silver labels; graded independently by `recall_press` (below) |
| 3. Data | Training rows from Lean-8 frame 50 with explicit hard negatives; replay matrix of every recording | Content hash stored in the study; a study never resumes on different data |
| 4. Search | Optuna, 40 trials per model, 3 person-grouped CV folds. K = 1…150 fields from a ranking learned inside each fold | Objective = the firmware replay on rows from frame 500 (where the firmware runs the models): TAP events at 0.85 with a 150 ms refractory, LP ten-sample Q20 release at 0.5 |
| 5. Choice | Best CV candidate inside the 32 KB budget that works for the people of every CV fold | The 22% held-out people never influence the choice; they are replayed once, through the compiled C header |
| 6. Export | Refit on everyone, trimmed to the budget, C header with `TAP_SWIPE_predict` / `LONG_PRESS_predict_value` | C-vs-Python decision and Q20 parity; both headers compile together |

CI (`gesture-unified-trainer-smoke.yml`) runs the whole chain on a reduced real-data set on every change.
It also runs these unit checks:

- Replay counting and LP averaging.
- Budget trimming.
- Fold-local ranking without leakage.
- Cache-hash compatibility.
- `validate_events.py` on synthetic device logs.

## Where the numbers stand (development people, 3 folds, all 150 fields)

TAP at the firmware threshold 0.85, on the rows where the firmware runs the models (frame ≥ 500):

- About 98% of confirmed taps and **92–94% of all real pointer presses** are found.
- The cost is about **4.5–5 false events per minute** on the hard-negative recordings (finger, wrist, band, arm).

At stricter budgets the same models find fewer of all real presses: about 82% at ≤ 2 false events per minute
and 87% at ≤ 3. The TAP budget is set from the previous production header, so a new model is never less
sensitive than the one it replaces. Details: `FEATURE_SELECTION_RESULTS.md`.

| Question | Answer (measured) |
|---|---|
| Do the Lean-8 fields help? | TAP: all 150 fields give 0.7 fewer false events per minute at the same recall than the 126 raw fields, in 25 of 27 fold × rule cells. LP: small gains |
| Would a feature subset help? | TAP: no subset beats all 150 in any fold. LP: within fold noise. Optuna still searches K |
| Would teaching unconfirmed ("soft") taps help? | No: same recall of all real presses, +1 false event per minute. Off |
| Do the replay-rule assumptions change any choice? | No: 9 TAP and 18 LP rule variants keep every comparison; only absolute numbers move |

## Before a header goes into the firmware

1. Read `COMPARISON.md`, held-out people, replayed through the C headers, new default vs the previous header:
   - **TAP:** `recall_press` and `recall` should be at least the previous header's, minus one point, and
     `false_per_min` should be no higher.
   - **LP:** `hold_kept`, `timely_release` and `tap_released` should be no worse.
2. Build the header in the real firmware (the C replay uses an all-float `t_core`, not the nRF52832 ABI).
3. Once per change of the firmware decision logic, record a session in which the device logs its own
   decisions, and run `validate_events.py` (below). Pass: TAP F1 ≥ 0.98 at the configured refractory; LP
   ≥ 95% of releases within 30 ms.

## What is measured and what is still assumed

| Item | Status |
|---|---|
| Lean-8 features | Measured: bit-exact with the firmware |
| Labels | Silver. TAP is also graded against every real pointer press (`recall_press`), which needs no labeller |
| Replay rules (refractory, hit window, LP timing) | Assumed. Conclusions are robust to them; absolute numbers need a firmware log (`validate_events.py --device-column`) |
| Tap timing (latency) | PointerState has a per-recording clock offset (−47 to +62 ms), so latency needs gold taps (`validate_events.py --gold`) |

## What would raise the ceiling

At strict false-event budgets the weakest taps look like deliberate finger, wrist and band movements in
these fields. More labels or more search move along the same trade-off curve. Only **new information**
moves the curve:

- Better sensor contact or new signals.
- Per-user calibration.
- A gold set (≈ 15–20 people × 3 min, fully annotated, synchronised video or a touch surface). This would
  replace silver grading for the final numbers.

## Files

| File | Role |
|---|---|
| `train_unified.py` | The trainer (stages 3–6) |
| `feature_selection.py` | Fold-local ranking, clusters, stability |
| `replay.py` | Firmware replay rules and metrics; C header loader |
| `generate_lean8_features.py` | Lean-8 emulator (stage 1); see `LEAN8_UNIFIED_WORKFLOW.md` |
| `validate_events.py` | Checks the replay rules and a header against firmware logs or gold taps |
| `fs_research.py` | Evidence runs: `gate`, `rank`, `kcurve`, `soft`, `rules`; results in `results_fs_research/` |
| `bench_unified.py` | Speed and exact-equivalence harness |
| `test_unified_smoke.py` | CI smoke test |
| `FEATURE_SELECTION_RESULTS.md` | All measurements behind the defaults |
