# One script to train TAP/SWIPE and LONG PRESS from Claude's unified labels

Run from the existing ML repository:

    git pull --ff-only
    python -u research/gesture/train_unified.py

No command-line flags. The script auto-generates/reuses Claude's new unified
per-row labels for both source roots, trains TWO **separate binary LightGBM
models** with Optuna, and exports stable C headers:

    research/gesture/results_unified_20261117/tap/models/TAP_SWIPE.h
    research/gesture/results_unified_20261117/lp/models/LONG_PRESS.h

Also writes 10 Pareto/size-diverse alternative candidate headers per task:

    .../tap/models/candidates/rank_01/TAP_SWIPE.h
    .../lp/models/candidates/rank_01/LONG_PRESS.h

Each candidate has the same simple C API, so choose only ONE header for each
task in the firmware. Model/trial identity stays in JSON, never C symbol names.
The default header selects the smallest candidate within 0.015 absolute of the
best grouped-CV score. Model static size is reported separately from the
human-readable header text length.

## Inputs and labels

Claude's consolidated offline labeler is
`claude_labeling/label_gestures.py`. Its default parameters are used:
- Raw sources: `DATA/DATA_TAP` and `DATA/DATA_LONG_PRESS`, ORIGINAL
  compressed `.csv.gz` only (a local `.csv` sibling cannot double a file).
- Label cache: `DATA/LABELS/DATA_TAP` and
  `DATA/LABELS/DATA_LONG_PRESS` (both Git-ignored). If matching complete
  label caches exist, they are reused; incomplete/unexpected caches are
  **rejected**, never silently overwritten.
- Every label row is checked against the ORIGINAL source's timestamp and row
  order. The full run SHA-256 fingerprints both data and labels.
- Unknown `-1` is NEVER converted to negative.
- TAP/SWIPE training sources: gesture and ordinary negative recordings
  in DATA_TAP, plus ONLY hold folders from DATA_LONG_PRESS. In those added
  hold recordings, starts/releases that look like taps are UNKNOWN; trusted
  mid-hold and idle samples are negative. Duplicated TAP/SWIPE recordings
  in DATA_LONG_PRESS are excluded from the TAP task.
- LP training sources: DATA_LONG_PRESS hold, TAP/SWIPE, contact protocol, and
  ordinary negative folders. Hold folders without confirmed holds are excluded
  rather than assumed to have no holds.

## Firmware prediction polarity (important)

The **new Claude labeler** defines `hold_label=1` as held and
`hold_label=0` as not held. The **existing Mudra LP release logic** expects
positive probability to mean RELEASE/REST. The trainer intentionally inverts
that label:

| Task | Model class 1 (probability >=0.85) | Model class 0 |
|---|---|---|
| TAP_SWIPE | TAP/SWIPE onset | no onset |
| LONG_PRESS | **RELEASE/REST** | **still HELD** |

The inversion preserves `-1` as unknown. It does not affect source labels.

    #include "TAP_SWIPE.h"
    #include "LONG_PRESS.h"

    bool onset = TAP_SWIPE_predict(&m_feat_core);
    bool release = LONG_PRESS_predict(&m_feat_core);

`LONG_PRESS_predict()` returning false **does not start a new press**;
only the separate TAP model may start one. Integrate the release score into
the actual 10-prediction averaged firmware policy. These generated APIs
output a hard decision at 0.85; where the firmware needs *probabilities*,
use the header's generated native probability API, not the boolean method.

**Never run both rank variants of the same header simultaneously** because
the C APIs are deliberately identical.

## Training policy

- **Only original p_core-> columns.** The up-to-126 default features are
  exactly the ACC PSD (including temporal history/mean/rate), EMG PSD
  (instant/oldest/mean) and WL (full/head/tail) used by the successful TAP
  pipeline. If a genuinely missing column is found in older LP recordings,
  the script uses the common validated p_core subset and records it.
- Neither PointerState, input folder, timestamps, teacher contact scores,
  synthetic EMA/engineered features, hold_ms, subject IDs nor label columns
  enter model inference.
- Feature-set search can select full, ACC/EMG/WL subsets **but never builds
  new firmware features**.
- 40 Optuna trials for EACH model; three participant-held-out CV folds;
  maximize score, minimize exact export_plus static C model bytes.
- **Threshold fixed at 0.85 for every trial, CV score, holdout evaluation and
  exported header. Optuna never searches thresholds.**
- TAP objective: grouped-CV silver-label weighted F1.
- LP objective: `0.70 * hold_safety + 0.30 * release_recall`, at threshold 0.85.
  Hold safety means the fraction of held samples not prematurely classified
  RELEASE. This prioritizes sustained holds instead of optimizing majority
  RELEASE row-F1; neither metric constitutes real hardware safety.
- CPU: 80% of affinity/cgroup-constrained logical CPUs (same policy as the
  current TAP experiment). No new runtime preprocessing.
- Train-only person groups are used for CV and a separate unseen-person
  holdout; model is refitted on all known rows **after** reporting its holdout
  metrics and before C export.
- Results/study data are isolated under
  `research/gesture/results_unified_20261117`. Previously successful
  `TAP_SWIPE.h` and LP v20 remain completely untouched.

## Limitations and acceptance criteria

Claude's `y_true` means strong, selective **offline silver labels**;
`PointerState` and whole-recording EMG floors still contribute to their
creation. The new trainer does NOT generate independent physical truth.
A high holdout score is not proof of soft long-press or physical release
performance.

Before promotion to production:
1. Build with the real nRF52832 firmware type declarations, not just the
   generated C **synthetic t_core** compile check.
2. Compare every candidate against the existing field-tested TAP model and
   the best LP v20 baseline; record soft/mid/strong sustained holds, releases,
   accidental starts, and double/soft TAP behavior on multiple wearers.
3. Replay the exact firmware 10-prediction averaging and minimum hold timing.
   LP labels intentionally exclude ambiguous rows close to releases, so the
   reported 0.85 holdout release accuracy does **not** establish actual
   release latency or 0 false releases.
4. Preserve winning model bytes, feature schema and JSON audit.

## Integration smoke

GitHub Actions runs a **real-recording** small study with 3 trials per head,
2 selected exports per head, both Claude labelers, and C compilation. The
test is in `research/gesture/test_unified_smoke.py`. The full default study
runs only when the user executes the command above. 
