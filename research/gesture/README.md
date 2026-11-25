# Unified TAP/SWIPE + LONG PRESS trainer (2026-11-18)

**Start with `PIPELINE.md`**: one page on what runs, what is checked and what still needs outside data.

Run from the repository root:

    git pull --ff-only
    python -u research/gesture/train_unified.py

There are no command-line flags. The script:

1. Refreshes the firmware Lean-8 features (`generate_lean8_features.py`, incremental, bit-exact with the
   firmware handler; see **LEAN8_UNIFIED_WORKFLOW.md**).
2. Reuses or generates Claude's unified labels (`claude_labeling/label_gestures.py`).
3. Trains two separate binary LightGBM models with Optuna, choosing among all 150 firmware fields.
4. Exports C headers with the stable firmware API:

    research/gesture/results_unified_v3_feature_selection/tap/models/TAP_SWIPE.h
    research/gesture/results_unified_v3_feature_selection/lp/models/LONG_PRESS.h
    research/gesture/results_unified_v3_feature_selection/COMPARISON.md   <- new models vs baselines, held-out people

## How models are searched and chosen

The design is in `FEATURE_SELECTION_RESEARCH_AND_EXECUTION_PLAN.md`; the evidence is in
`FEATURE_SELECTION_RESULTS.md`. The earlier feature-set menu ("v2") was removed once its study finished; it
is in git history (commit 53e3e83d), and `MUDRA_EXPERIMENT=v2` now stops with that message.

| | |
|---|---|
| Features | Optuna picks an integer **K = 1…150** over all available firmware fields; each CV fold uses its own ranking, learned from that fold's training people only (`feature_selection.py`). Both starting trials use K = all fields. |
| Rows scored | from Lean-8 frame **500** = `lean_8_features_ready()`, the rows on which the firmware runs the models (`MUDRA_EVAL_MIN_FRAMES`) |
| Training rows | from frame 50 (`MUDRA_TRAIN_MIN_FRAMES`); the warm-up rows still help (results, §2) |
| TAP false-event budget | the previous production TAP header's false rate on the development people (`MUDRA_TAP_FALSE_PER_MIN_TARGET=baseline`, or a number). A fixed 1/min made the search give up ~25% of taps. The score is recall x min(1, target / false rate): below the target only recall counts, so the search picks the most sensitive model that stays just under it. With the file split (about 6/min) that makes taps very sensitive; `make train TARGET=4` asks for at most 4 (finger false taps -37%, fist -35%, 1.2 points of taps; TARGET=3: -57% / -58%, 3 points; TAP_FILESPLIT_RESULTS.md). A number gets its own folder and study (`-target4`). Below 4/min Optuna also searches a logit shift 0..3, with the TAP positive weight fixed at 1 (the shift is subtracted from the first tree's leaves, so the firmware's 0.85 means a stricter probability of the fitted model). Recommended: `make train KB=32 ALT_KB=14 TARGET=3`; 1.5 lost taps on the band (TAP_FILESPLIT_RESULTS.md, "Band test of 64 KB / 1.5") |
| TAP labels | confirmed taps only. Soft labels (`MUDRA_TAP_SOFT_LABELS`) are off: measured, they do not help (results, §5). |
| Default model | chosen on CV only; the held-out report is written once, before anything is fitted on those people (`holdout_report.json`) |
| Production model | ranking refitted on everyone at the locked K, then refit |

The ranking method:

1. Build correlation clusters from the training rows: Spearman |ρ|, complete linkage, cut chosen by
   silhouette among strong-correlation cuts (|ρ| ≥ 0.7).
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
- `models/`, `COMPARISON.md`, `summary.json`, `run_manifest.json`, `timing.jsonl` (below).

Cost of the rankings: one ranking per CV fold, plus a development and a final ranking, five per task.
Each runs about 250 replay perturbations on its validation people. On the 2-CPU research box a TAP fold
ranking took 26–31 min and an LP fold ranking about 12 min. On the server the LightGBM part uses all
threads, and the single-threaded replay scoring then dominates. Rankings are cached
(`feature_selection/cache/`) and computed only when Optuna first tries K < all fields.

## What the unified trainer fixed (v1 → v2, kept since)

| v1 | Effect | Now |
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
| `MUDRA_USE_LEAN8` | 1 | EMG context inputs next to the 126 raw fields. `1` = the 24 float Lean-8 fields. `0` = none (firmware built with `NEW_CLOAD_FEATURES 0`); output `results_unified_v3_feature_selection_no_lean8/`, studies `lean80`. `lite` = the 15 integer Lean-lite fields of `firmware/lean_lite.h` (see `LEAN_LITE.md`); data in `DATA/DATA_*_LITE`, output `results_unified_v3_feature_selection_lean_lite/`, studies `lean8lite`. Rows, labels, frame gates and the held-out split are identical in all three, so they are clean A/B arms |
| `MUDRA_TAP_FEATURES` | raw | `diff` = add 135 within-channel EMG PSD diffs and 9 WL ratios to the TAP inputs, exported as the firmware helpers `p_core->diff(a, b)` / `p_core->ratio(a, b)` (TAP_GENERALIZATION_RESULTS.md). Needs `MUDRA_USE_LEAN8=0`; output folder and TAP study get the suffix `_tap-<features>-<aug>` |
| `MUDRA_TAP_AUG` | none | `mild` = placement augmentation of the TAP training rows (random EMG gain, spectral tilt and contact noise per recording; replay and held-out scores use the real recordings). Needs `MUDRA_USE_LEAN8=0` |
| `MUDRA_TAP_LITE` | off | `diff` = add the 15 gain-invariant Lean-lite diff inputs `p_core->cload_lite.{rise10,fall10,range10,dev3,xm}[CH1..3]` (firmware/lean_lite.h), computed per recording from the logged WL (no new data files) and recomputed under augmentation / placement stress. Needs `MUDRA_USE_LEAN8=0`; folder suffix `-litediff` |
| `MUDRA_TAP_LABELS` | confirmed | `press2` = "trust the press" in DoubleTaps and 240 bpm TAP recordings: unknown presses become training positives, negatives right at a press are left out (TAP_FOCUS_RESULTS.md). Training labels only; scores use the original labels. Folder suffix `-press2` |
| `MUDRA_TAP_OBS_MS` | off | e.g. `60` = observable-pattern labels (TAP_OBSERVABLE_RESULTS.md): of each tap's positive rows only the 60 ms that a teacher cross-fitted by person finds most distinguishable are trained, the rest left out; done inside every CV fold's, the development and everyone's training people. Folder suffix `-obs60` |
| `MUDRA_NUM_LEAVES` | 8,127 | LightGBM `num_leaves` search range `low,high`; a non-default range (or budget) adds `-kb<budget>-leaves<low>-<high>` to the TAP folder |
| `MUDRA_TAP_RECIPE` | (none) | `final14` = the final TAP recipe as defaults: `MUDRA_TASKS=tap MUDRA_USE_LEAN8=0 MUDRA_TAP_FEATURES=diff MUDRA_TAP_AUG=mild MUDRA_TAP_LITE=diff MUDRA_TAP_LABELS=press2 MUDRA_MODEL_BUDGET_KB=14 MUDRA_ALT_BUDGET_KB=0 MUDRA_NUM_LEAVES=4,15 MUDRA_TRAIN_MIN_FRAMES=50 MUDRA_EVAL_MIN_FRAMES=50` (explicit variables still win). `final14obs` = the same + `MUDRA_TAP_OBS_MS=60` (the best balance with the file split, TAP_FILESPLIT_RESULTS.md; the `make train` default). `final14protect` = `final14` + `MUDRA_TAP_OBS_MS=40 MUDRA_TAP_DISTILL=0.5 MUDRA_TAP_FINGER_WEIGHT=2` (far fewer finger / fist false taps at 0.85 for about 1 point of taps at 14 KB; at 32 KB and above with a stricter target `final14obs` protects more: `make train KB=32 ALT_KB=14 TARGET=3`) |
| `MUDRA_TAP_DISTILL` | off | label weight `a` of distillation (e.g. `0.5`): the TAP model learns `a*label + (1-a)*teacher`, the teacher being a full-size LightGBM (no budget) on all inputs, cross-fitted by group inside every training set (CV folds, development, everyone); trained as the usual binary model (two weighted rows per frame), so size and C export are unchanged. Folder suffix `-distill50` |
| `MUDRA_N_ESTIMATORS` | 80,400 | tree-count range of the search; widen for bigger budgets (400 trees of 4 leaves fill about 18 KB). `make train KB=32` sets `MUDRA_MODEL_BUDGET_KB=32` and `80,800`; add `ALT_KB=14` for a 14 KB candidate from the same search |
| `MUDRA_TAP_FINGER_WEIGHT` | 1 | FINGER-folder negatives (finger motion, fist) weighted x k in the TAP model's training before class balancing; teachers keep plain weights. Folder suffix `-fingerw2` |
| `MUDRA_SPLIT` | person | what the held-out split and the CV folds group by. `person`: nobody is on both sides (scores = people the model never saw). `file`: every recording file is its own group, so the same person can be on both sides and every person takes part in every fold; scores then mostly describe people the model was trained on. The file split is gesture-aware: held-out files and CV folds are stratified by recording type (folder, gesture from the file name, bpm for TAP / SWIPE; types with fewer than 5 files pooled per folder), so every part gets its share of each gesture; `split.json` lists the files per type on each side. The shipped header is refitted on everyone either way. Folder suffix `_splitfile`, study suffix `_splitfile_gesture1` |
| `MUDRA_TASKS` | tap,lp | train only these tasks, e.g. `tap` |
| `MUDRA_TAP_BASELINE_EXTRA` | (none) | path of one more TAP header replayed as a baseline on the same held-out people, e.g. today's no-Lean `TAP_SWIPE.h` for an A/B in one table |
| `MUDRA_LEAN_LITE_PARITY_FILES` | 24 | recordings on which `generate_lean_lite_features.py` proves exact equality with `firmware/lean_lite.h` compiled for the host (synthetic edge cases always) |
| `MUDRA_LEAN8_FIRMWARE_HEADER` | `research/gesture/_lean8_reference/core_type_def.h` | firmware header for the C parity check (git-ignored; public repo) |
| `MUDRA_LEAN8_PARITY_FILES` | 24 | recordings compared with the firmware C code on each generation (0 = off) |
| `MUDRA_LEAN8_VERIFY_SOURCE_SHA` | 0 | 1 = also SHA-256 every source recording before reusing its derived file (size + mtime is the default check) |
| `MUDRA_LP_RELEASE_THRESHOLD` | 0.5 | firmware LP release threshold on the 10-sample Q20 mean; part of the LP study name |
| `MUDRA_LP_BASELINE_THRESHOLD` | 0.85 | release threshold used to replay the LP baselines |
| `MUDRA_EVAL_MIN_FRAMES` | 500 | rows scored and ranked = rows the firmware runs the models on (`lean_8_features_ready()`) |
| `MUDRA_TRAIN_MIN_FRAMES` | 50 | rows used as training examples, counted in Lean-8 frames (~1.1 s); the older name `MUDRA_LEAN8_MIN_FRAMES` is still read |
| `MUDRA_TAP_FALSE_PER_MIN_TARGET` | baseline | TAP false-event budget: `baseline` = previous TAP header's rate on the development people, or a number |
| `MUDRA_TAP_LOGIT_SHIFT` | auto | TAP logit-shift range of the search: `auto` = 0,3 when the target is a number below 4 (positive weight then fixed at 1), else none; `off`; or `lo,hi`. Own folder and study (`-shift0-3-pw1`) |
| `MUDRA_TAP_SOFT_LABELS` | off | a teacher score (e.g. 0.5): unconfirmed taps with at least that score become weighted positives and count in the objective. Measured not to help; part of the TAP study name when set |
| `MUDRA_CPU_THREADS` | 80% of CPUs | all threads the trainer uses (CPUs the process may use, cgroup quota respected); does not change the models (deterministic=True) |
| `MUDRA_FOLD_WORKERS` | 3 when `MUDRA_CPU_THREADS` ≥ 24 (2 when ≥ 16) | CV folds of a trial trained at the same time, each LightGBM with `MUDRA_CPU_THREADS / workers` threads. Same trees and scores as one fold at a time (smoke: `FOLD_PARALLEL_EQUIVALENT`); 1 = one fold at a time |

## Selection and outputs

- 40 trials per model (TPE, median pruner after fold 1). Two anchors are queued first: 15 leaves/depth 6
  and 63 leaves/depth 10.
  - TAP anchors use positive weight 1.
  - LP anchors use 0.02 at a 0.5 release threshold. Measured: the LP replay optimum sits at threshold
    1.0 / 0.95 / 0.85 / 0.7 for weights 6 / 1.5 / 0.2 / 0.05.
- Study names carry every setting that changes the objective or the data, so a study is never resumed under
  other settings: `unified_v3_tap_fsfs-2_lean81_train50_eval500_fpm<target>` (plus `_soft<score>` when soft
  labels are on) and `unified_v3_lp_fsfs-2_lean81_train50_eval500_rel0.50`.
- The top 5 by CV score inside the primary budget, plus the best inside 24 KB, are exported. For each one:
  - It is refit without the held-out people (22%, never used for CV).
  - It is exported to C and replayed **through the compiled header**, with C-vs-Python parity checked.
  - It is then refit on everyone, trimmed and exported.
- **Default** = the best CV candidate that did not fail on any CV fold (fold score 0: never fires / never
  releases for those people). The held-out people never influence the choice. Skipped candidates and the
  reason are listed in `summary.json`.

```
results_unified_v3_feature_selection/
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

The content hash covers only what training and the objective see. Report-only additions (unit sample
weights, the pointer-press reference windows in `tap_onsets.csv`) are left out, so a study started on
commit 53e3e83d resumes after this update: on a 1/25 subset the old and the new cache give the same hash for
TAP and LP, and the smoke test checks it (`CACHE_HASH_COMPAT_OK`). The study names, search space, seeds,
sampler and pruner are unchanged, so Optuna continues the same 40-trial study.

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

- Training rows are used from frame `MUDRA_TRAIN_MIN_FRAMES` (default 50) on; rows are scored from
  `MUDRA_EVAL_MIN_FRAMES` (default 500).
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

    python research/gesture/bench_unified.py run tap new.json [--study results_unified_v3_feature_selection/tap/optuna_study.sqlite3 --limit 5]
    python research/gesture/bench_unified.py compare old.json new.json      # ALL_EQUIVALENT or a list of mismatches
    python research/gesture/bench_unified.py stages tap                     # old vs optimized time per changed stage
    python research/gesture/bench_unified.py threads tap 8,16,24,32,48,64,76  # time + exact-model check per thread count

## Limitations

- Labels are offline silver labels (teacher + PointerState fusion), not physical truth. TAP reports also give
  `recall_press`: recall of every real pointer press in the TAP/SWIPE recordings, which does not depend on
  the labeller (but uses a wide window, so it does not grade timing).
- The replay timing rules above are assumptions about the firmware. No model comparison changes under other
  rules (results, §6), but absolute numbers do: confirm them with `validate_events.py` on a firmware log.
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
