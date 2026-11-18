# Unified TAP/SWIPE + LONG PRESS trainer (v2, 2026-11-18)

Run from the repository root:

    git pull --ff-only
    python -u research/gesture/train_unified.py

There are no command-line flags. The script reuses or generates Claude's unified labels
(`claude_labeling/label_gestures.py`). It trains two separate binary LightGBM models with Optuna and
exports C headers with the stable firmware API:

    research/gesture/results_unified_v2/tap/models/TAP_SWIPE.h
    research/gesture/results_unified_v2/lp/models/LONG_PRESS.h
    research/gesture/results_unified_v2/COMPARISON.md        <- new models vs baselines, held-out people

## Why v2 (what was wrong with v1)

| v1 | Effect | v2 |
|---|---|---|
| Optuna minimized model bytes as a second objective, and the default was "smallest within 0.015 of best" | Always shipped the smallest model | Performance first, inside a byte **budget** (default 32 KB, alternative 24 KB) |
| At most 45 training rows per class per recording | Hard negatives (finger/wrist/band motion) barely seen | ~1.0 M TAP / ~0.77 M LP rows, with explicit hard negatives |
| Frame-level F1 on silver labels | Does not measure what the firmware does | TAP: event replay at 0.85. LP: hold/release sequence replay with the 10-sample average |
| Trees 20–160, leaves up to 31 | No room to grow | Trees 80–400, depth 3–10, leaves 8–127, trimmed to the budget |

## Training samples

All rows come from the original `p_core->` columns only. Rows are thinned by time (one row per time bin)
rather than capped per recording. The strides are constants at the top of `train_unified.py`.

**TAP** (class 1 = onset window 0–80 ms after sensor onset):

| Kind | Rule | Stride |
|---|---|---|
| positive | `tap_label == 1` | 10 ms |
| hard_near_press | negatives within 500 ms of any onset, pointer press or release | 20 ms |
| hard_muscle_activity | negatives with any channel ACTIVE (FINGER/WRIST/ARM/BAND motion, holds) | 20 ms |
| background | other negatives | 40 ms |

Unknown (`-1`) rows are never used. In the full data set this is about 142 k positives and 882 k negatives.

**LP** (class 1 = RELEASE/REST, class 0 = still HELD; the inverse of `hold_label`):

| Kind | Rule | Stride |
|---|---|---|
| hold | `hold_label == 1`; every row in the first 600 ms of a hold | 10 ms |
| release_after_press | within 1 s after a hold release; within 600 ms after a tap release | 10 ms / 40 ms |
| background | other release/rest rows | 100 ms |

Rows inside a tap press in TAP/SWIPE recordings are excluded from LP training; firmware only runs LP
after the TAP model has started a press.

## Replay metrics (what Optuna optimizes)

Every trial is scored by replaying whole recordings of the validation people through the model at the
fixed threshold 0.85, in the firmware's arithmetic (`research/gesture/replay.py`).

**TAP:** each rising edge of `TAP_SWIPE_predict` is an event, with a 150 ms refractory.

- An event counts as a hit if it falls within [onset − 100 ms, onset + 250 ms].
- Any other event on a labelled row is a false event.
- false/min is computed over negative time.
- `score = recall × min(1, 1.0 / false_per_min)`, so beyond one false event per minute every extra
  false event costs score.

Latency is also reported.

**LP:** for each confirmed hold and up to three confirmed taps per TAP/SWIPE recording:

1. Start at the press start (from the TAP model).
2. Feed `LONG_PRESS_predict_value()` (Q20 steps) into a 10-sample moving average.
3. Release is declared when the average ≥ 0.85.

The following counts are taken:

- **hold_kept:** no release before (true release − 250 ms).
- **timely_release:** released by true release + 500 ms. A press that never releases is stuck.
- **tap_released:** a tap press released within 500 ms, so it does not become a long press.

`score` = weighted harmonic mean, hold_kept 0.5, timely_release 0.3, tap_released 0.2. It is 0 if any part
is 0, so a model that never releases scores 0.

The objective is the mean of per-fold scores (3 person-grouped folds), which penalizes settings that only
work for some people. The pooled score is also stored.

**Firmware rules assumed (please confirm).** These are constants in `replay.py`:

| Constant | Value |
|---|---|
| tap refractory | 150 ms |
| hit window | −100/+250 ms |
| LP minimum hold before release | 0 ms |
| LP premature margin | 250 ms |
| release-on-time tolerance | 500 ms |
| tap becomes long press after | 500 ms |

## Search space and model-data budget

| Hyperparameter | Range |
|---|---|
| n_estimators | 80–400 |
| max_depth | 3–10 |
| num_leaves | 8–127, and ≤ 2^depth |
| learning_rate | 0.02–0.3 |
| min_child_samples | 10–300 |
| positive weight | TAP 0.2–10, LP 0.2–20 |
| feature subsets | of the 126 p_core features |
| colsample / subsample / L1 / L2 | also searched |

The export_plus format costs 6 bytes per node. 32 KB holds about 5,400 nodes, roughly:

| Trees | Leaves per tree |
|---|---|
| 400 | 7 |
| 170 | 15 |
| 85 | 31 |
| 21 | 127 |

The full range, 400 trees × 127 leaves, therefore cannot fit. Each model is trimmed: boosting is additive,
so the first k trees are a valid k-tree model. An exact binary search on the exported size picks the
largest k inside the budget. Optuna then trades depth against tree count by score. Every trial is scored
at both budgets.

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MUDRA_MODEL_BUDGET_KB` | 32 | primary budget (default header) |
| `MUDRA_ALT_BUDGET_KB` | 24 | second budget; `none` or `0` disables it |
| `MUDRA_TAP_BASELINE_HEADER` | `research/tap_real/results_claude_v2_optuna_126_thr085/models/TAP_SWIPE.h` | previous successful TAP model |
| `MUDRA_LP_V20_HEADER` | not set | **LP v20 header; set this to get the v20 comparison** |

## Selection and outputs

- 40 trials per model (TPE, median pruner after fold 1). Two anchors are queued first: 15 leaves/depth 6
  and 63 leaves/depth 10. LP anchors use positive weight 6.
- The top 5 by CV score inside the primary budget, plus the best inside 24 KB, are exported. For each one:
  - It is refit without the held-out people (22%, never used for CV).
  - It is exported to C and replayed **through the compiled header**, with C-vs-Python parity checked.
  - It is then refit on everyone, trimmed and exported.
- **Default** = the best CV candidate whose held-out C replay is not a failure (it fires / releases on
  unseen people). Skipped candidates and the reason are listed in `summary.json`.

```
results_unified_v2/
  COMPARISON.md, UNIFIED_SUMMARY.json
  tap/ and lp/
    models/TAP_SWIPE.h (LONG_PRESS.h)        default, stable API
    models/budget_24kb/                      best inside 24 KB
    models/candidates/rank_01..05, budget_24kb/   every exported candidate
    models/*.json, *.pkl                     per-candidate audit (CV, held-out, parity, params)
    summary.json, COMPARISON.md, optuna_study.sqlite3 (resumable)
    cache/                                   training rows + replay matrix (rebuilt if data/labels change)
```

## Baselines (read-only)

The previous successful TAP model and LP v20 are not retrained and not modified: a SHA-256 is taken before
and after. Each is compiled and replayed on the same held-out people with the same metrics, as rows in
`COMPARISON.md`. A missing or not-configured baseline is reported, never fatal. The previous unified
(2026-11-17) headers are compared too.

## Firmware polarity (unchanged)

| Task | class 1 (p ≥ 0.85) | class 0 |
|---|---|---|
| TAP_SWIPE | TAP/SWIPE onset | no onset |
| LONG_PRESS | **RELEASE/REST** | **still HELD** |

    bool onset = TAP_SWIPE_predict(&m_feat_core);
    float release_q = LONG_PRESS_predict_value(&m_feat_core);   // Q20 steps, average 10 of these

`LONG_PRESS_predict()` returning false does not start a press; only the TAP model starts one. Average
`LONG_PRESS_predict_value()`, not the boolean.

## Runtime and resources (measured on a 2-CPU, 7 GB box)

| Item | Measurement |
|---|---|
| Full TAP cache build | 105 s: 3,005 recordings, 1.02 M train rows, 6.6 M replay rows |
| Disk | 3.7 GB for the TAP cache; the LP cache is smaller |
| One 3-fold TAP trial | ≈ 95 s for a 15-leaf model |
| Peak memory | 4.2 GB |

Expect roughly 1.5–3 h per model for 40 trials on 2 CPUs, less with more cores. The study is resumable:
rerunning continues the sqlite study.

## Limitations

- Labels are offline silver labels (teacher + PointerState fusion), not physical truth.
- The replay timing rules above are assumptions about the firmware.
- C replay uses a synthetic all-float `t_core`, not the nRF52832 ABI. Build the chosen headers in the
  real firmware before field tests.

## Smoke test

    python -u research/gesture/test_unified_smoke.py            # unit checks + real-data reduced run
    python -u research/gesture/test_unified_smoke.py --unit-only

The unit checks cover:

- replay counting
- LP averaging latency
- budget trimming exactness
- C header vs Python decisions and Q20
- the default-selection guard

The real-data run uses one recording subset per person, 4 trials, 8/6 KB budgets and baseline replay, and
compiles both headers together. CI runs it on every change (`gesture-unified-trainer-smoke.yml`).
