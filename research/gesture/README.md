# Unified TAP/SWIPE + LONG PRESS trainer (v2, 2026-11-18)

Run from the repository root:

    git pull --ff-only
    python -u research/gesture/train_unified.py

There are no command-line flags. The script:

1. Refreshes the firmware Lean-8 features (`generate_lean8_features.py`, incremental, bit-exact with the
   firmware handler; see **LEAN8_UNIFIED_WORKFLOW.md**).
2. Reuses or generates Claude's unified labels (`claude_labeling/label_gestures.py`).
3. Trains two separate binary LightGBM models with Optuna on raw126 and/or Lean-8 fields.
4. Exports C headers with the stable firmware API:

    research/gesture/results_unified_v2/tap/models/TAP_SWIPE.h
    research/gesture/results_unified_v2/lp/models/LONG_PRESS.h
    research/gesture/results_unified_v2/COMPARISON.md        <- new models vs baselines, held-out people

## v3 (default): learned feature selection

`python -u research/gesture/train_unified.py` runs the **v3** experiment
(`FEATURE_SELECTION_RESEARCH_AND_EXECUTION_PLAN.md`; evidence in `FEATURE_SELECTION_RESULTS.md`). Set
`MUDRA_EXPERIMENT=v2` for the earlier feature-set menu; its outputs and studies are untouched and still
resumable.

What changes in v3:

| | v2 | v3 |
|---|---|---|
| Features | Optuna picks one of a hand-written menu of sets | Optuna picks an integer **K = 1…150** over all available firmware fields; each CV fold uses its own ranking, learned from that fold's training people only (`feature_selection.py`) |
| Rows scored | from Lean-8 frame 50 | from frame **500** = `lean_8_features_ready()`, the rows on which the firmware runs the models (`MUDRA_EVAL_MIN_FRAMES`) |
| Training rows | from frame 50 | from frame 50 by default (`MUDRA_TRAIN_MIN_FRAMES`, see the results) |
| TAP false-event budget | 1 per minute | the previous production TAP header's false rate on the development people (`MUDRA_TAP_FALSE_PER_MIN_TARGET=baseline`, or a number) |
| Default model | could skip candidates that failed on the held-out people | chosen on CV only; the held-out report is written once, before anything is fitted on those people (`holdout_report.json`) |
| Production model | refit on everyone | ranking refitted on everyone at the locked K, then refit |
| Starting trials | – | K = all fields (all-feature control) |

The ranking method:

1. Build correlation clusters from the training rows: Spearman |ρ|, average linkage, cut chosen by
   silhouette.
2. Run an inner person-grouped cross-fit. In each split, train a screening LightGBM on all fields.
3. For each feature and each cluster (shifted jointly), apply a **within-recording circular shift** and
   measure the drop in the firmware replay score. The operating point is fixed from the unperturbed
   screening model.
4. Order the features: clusters by joint importance, members by their own importance.

This is block-permutation importance, not conditional sampling. The ordering is only a search path: K,
retraining and replay decide.

Outputs (`results_unified_v3_feature_selection/<task>/`):

- `feature_selection/fold_*_ranking.csv`, `development_ranking.csv`, `final_ranking.csv`: every field with its
  rank, importance, cluster, cluster importance and LightGBM gain share.
- `correlation_clusters.json`, `selection_stability.json` (Nogueira), `selected_features_by_trial.json`,
  `final_selected_features.json`, `report.md`.
- `holdout_report.json`.
- The same model outputs as v2.

Cost of the rankings: one ranking per CV fold, plus a development and a final ranking, five per task.
Each runs about 250 replay perturbations on its validation people. On the 2-CPU research box a TAP fold
ranking took 26–31 min and an LP fold ranking about 12 min. On the server the LightGBM part uses all
threads, and the single-threaded replay scoring then dominates. Rankings are cached
(`feature_selection/cache/`) and computed only when Optuna first tries K < all fields.

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

Every trial is scored by replaying whole recordings of the validation people through the model at fixed
firmware thresholds, in the firmware's arithmetic (`research/gesture/replay.py`):

- TAP onset at p ≥ **0.85**.
- LP release when the 10-sample Q20 mean is ≥ **0.5** (`MUDRA_LP_RELEASE_THRESHOLD`; was 0.85 until
  2026-11-18).

Optuna never tunes a threshold.

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
3. Release is declared when the average ≥ the LP release threshold (0.5 by default). **The firmware's
   release comparison must use the same value**: mean of the 10 values ≥ 0.5, i.e. a sum of the 10 Q20
   codes ≥ 100.

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
| positive weight | TAP 0.2–10; LP 0.005–20 at a 0.5 release (0.2–20 at 0.85) |
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
| `MUDRA_USE_LEAN8` | 1 | 0 = raw126 only, read straight from the raw recordings |
| `MUDRA_LEAN8_MIN_FRAMES` | 50 | rows are used once the Lean-8 handler has run this many EMG frames (~1.1 s); 500 = `lean_8_features_ready()` (~10.8 s, keeps only ~14% of taps) |
| `MUDRA_LEAN8_FIRMWARE_HEADER` | `research/gesture/_lean8_reference/core_type_def.h` | firmware header for the C parity check (git-ignored; public repo) |
| `MUDRA_LEAN8_PARITY_FILES` | 24 | recordings compared with the firmware C code on each generation (0 = off) |
| `MUDRA_LEAN8_VERIFY_SOURCE_SHA` | 0 | 1 = also SHA-256 every source recording before reusing its derived file (size + mtime is the default check) |
| `MUDRA_LP_RELEASE_THRESHOLD` | 0.5 | firmware LP release threshold on the 10-sample Q20 mean; part of the LP study name |
| `MUDRA_LP_BASELINE_THRESHOLD` | 0.85 | release threshold used to replay the LP baselines |
| `MUDRA_EXPERIMENT` | v3 | v3 = learned feature selection; v2 = earlier feature-set menu |
| `MUDRA_EVAL_MIN_FRAMES` | 500 (v3) | rows scored and ranked = rows the firmware runs the models on |
| `MUDRA_TRAIN_MIN_FRAMES` | 50 (v3) | rows used as training examples |
| `MUDRA_TAP_FALSE_PER_MIN_TARGET` | baseline (v3), 1.0 (v2) | TAP false-event budget: `baseline` = previous TAP header's rate on the development people, or a number |
| `MUDRA_CPU_THREADS` | 80% of CPUs | LightGBM threads; does not change the models (deterministic=True). Choose with `bench_unified.py threads` |

## Selection and outputs

- 40 trials per model (TPE, median pruner after fold 1). Two anchors are queued first: 15 leaves/depth 6
  and 63 leaves/depth 10.
  - TAP anchors use positive weight 1.
  - LP anchors use 0.02 at a 0.5 release threshold. Measured: the LP replay optimum sits at threshold
    1.0 / 0.95 / 0.85 / 0.7 for weights 6 / 1.5 / 0.2 / 0.05.
- Study names:
  - TAP: `unified_v2_tap_lean81_min50`, unchanged, so a running study resumes.
  - LP: `unified_v2_lp_lean81_min50_rel0.50`. It carries the release threshold, so an LP study run at
    0.85 is never resumed under 0.5.
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
    timing.jsonl                             per-trial stage seconds (fit, size, replay gather/predict/score)
    run_manifest.json                        commit, versions, CPU/memory, hashes, search space, constants
    backups/optuna_study_<time>.sqlite3      SQLite-API copy taken before every resume
    study_snapshot_<time>.json               trials (params, values, fold scores) before new trials start
    cache/                                   training rows + replay matrix (rebuilt if data/labels change)
```

### Resuming a running study after a code update

1. Stop the trainer between trials, or let it finish.
2. `git pull`, then start it again with the same command and environment.

On start the trainer:

- Copies the SQLite study into `backups/` and writes a snapshot of every trial.
- Records the dataset's content hash in the study. On later runs it **refuses to resume** if the training
  or replay data differ.
- If the cache is rebuilt, hashes the old cache first and logs whether the rebuild is identical
  (`UNIFIED_CACHE_REBUILD identical content`).

The TAP study name, search space, feature sets, seeds, sampler and pruner are unchanged, so Optuna
continues the same 40-trial study.

## Baselines (read-only)

The previous successful TAP model and LP v20 are not retrained and not modified: a SHA-256 is taken before
and after. Each is compiled and replayed on the same held-out people with the same metrics, as rows in
`COMPARISON.md`. A missing or not-configured baseline is reported, never fatal. The previous unified
(2026-11-17) headers are compared too.

LP baselines are replayed at the release threshold they were built for: 0.85
(`MUDRA_LP_BASELINE_THRESHOLD`). New LP models are replayed at 0.5. Each is therefore compared at its own
firmware setting.

## Lean-8 features

All the details are in `LEAN8_UNIFIED_WORKFLOW.md`.

- The 24 `p_core->cload_lean_8.*` fields are computed offline with the firmware's own arithmetic. The
  handler is stepped **once per EMG frame**: 46.4 frames/s, against 140–210 logged rows/s.
- They are checked frame by frame against the firmware C code, compiled for the host: max difference
  0.0.
- Feature sets searched:

| Feature set | Fields |
|---|---|
| `raw126` | the 126 raw p_core fields |
| `lean8_24` | the 24 Lean-8 fields |
| `raw126_lean8_instant12` | raw126 + `lw`, `Pm`, `r01`, `r32` (no history, so no warm-up effect) |
| `all_available` | all 150 |

- Rows are used from frame `MUDRA_LEAN8_MIN_FRAMES` (default 50) on.
- `COMPARISON.md` has a second table: held-out C replay on rows after 500 frames, the firmware's
  `lean_8_features_ready()` steady state.
- Exported headers are compile-checked against a synthetic `t_core` that includes `cload_lean_8`.

## Firmware polarity (unchanged)

| Task | class 1 | class 0 | Firmware decision |
|---|---|---|---|
| TAP_SWIPE | TAP/SWIPE onset | no onset | p ≥ 0.85 |
| LONG_PRESS | **RELEASE/REST** | **still HELD** | mean of 10 Q20 values ≥ 0.5 (`LONG_PRESS_THRESHOLD` 0.5f) |

    bool onset = TAP_SWIPE_predict(&m_feat_core);
    float release_q = LONG_PRESS_predict_value(&m_feat_core);   // Q20 steps, average 10 of these

`LONG_PRESS_predict()` returning false does not start a press; only the TAP model starts one. Average
`LONG_PRESS_predict_value()`, not the boolean.

## Runtime and resources (measured on a 2-CPU, 7 GB box)

| Item | Measurement |
|---|---|
| Lean-8 generation (first run, then incremental) | ≈ 8 min for 5,352 recordings; +1.2 GB derived data |
| Full TAP cache build | 113 s: 3,005 recordings, 1.02 M train rows (0.95 M after the 50-frame gate), 6.6 M replay rows |
| Disk | 4.0 GB for the TAP cache (150 columns); the LP cache is smaller |
| One 3-fold TAP trial | ≈ 106 s for a 15-leaf model on all 150 features |
| Peak memory | 4.2 GB |

Expect roughly 1.5–3 h per model for 40 trials on 2 CPUs, less with more cores. The study is resumable:
rerunning continues the sqlite study.

## Speed and equivalence harness

`research/gesture/bench_unified.py` implements the measurement and parity side of
`TRAINING_PERFORMANCE_OPTIMIZATION_PLAN.md`. Its implementation status and results are at the top of
that file.

    python research/gesture/bench_unified.py run tap new.json [--study results_unified_v2/tap/optuna_study.sqlite3 --limit 5]
    python research/gesture/bench_unified.py compare old.json new.json      # ALL_EQUIVALENT or a list of mismatches
    python research/gesture/bench_unified.py stages tap                     # old vs optimized time per changed stage
    python research/gesture/bench_unified.py threads tap 8,16,24,32,48,64,76  # time + exact-model check per thread count

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
