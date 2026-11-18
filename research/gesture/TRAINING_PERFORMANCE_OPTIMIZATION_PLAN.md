# Performance optimization plan — unified TAP/SWIPE + LONG_PRESS trainer

**Status:** implementation plan only; no training behavior changes in this document.  
**Repository:** alex00Pirotskyi/ML  
**Reviewed source:** main at commit 235a8b07c7dbedfab2c172715044fa489c9b07f1 (2026-11-18)  
**Primary files:** research/gesture/train_unified.py, research/gesture/replay.py, research/gesture/generate_lean8_features.py, research/gesture/test_unified_smoke.py, export_plus.py  
**Goal:** reduce wall-clock time and unnecessary CPU/memory/disk work without sacrificing a single recording, feature, training example, evaluated event, Optuna trial, validation fold, model-size constraint, or production acceptance criterion.

## Executive decision

**Optimize implementation overhead, NOT the ML problem.** We have already generated the firmware Lean-8 feature values **before** Optuna starts. No Lean-8 features are recomputed inside a trial. The slow part is training and evaluating new candidate models, repeated for every person-grouped fold.

Do not replace the existing experiment with a cheaper one. Preserve the running Optuna database, all completed trials and the ability to resume. The first changes should be profiling, reuse of immutable data/weights, fewer copies of the replay matrix, and equivalence-checked replay/model-size optimizations. Apply LightGBM thread tuning only after measuring it.

**A zero-sacrifice optimization is accepted only when the original and optimized implementations produce equivalent results under the contracts below. Speed alone is not a reason to merge a change.** No speedup percentage is promised before the measurements exist.

## 1. Current workload and where time goes

The user's observed run (2026-11-18) reported:

| Item | Observed |
| --- | ---: |
| Raw TAP files | 2,921 |
| Raw LP files | 2,431 |
| TAP dataset recordings | 3,005 |
| Available firmware feature columns | 150 = 126 raw + 24 Lean-8 |
| TAP training matrix after configured frame gate | 946,678 rows × 150 |
| TAP held-out training rows | 211,579 |
| TAP replay matrix | 6,596,284 rows × 150 (~3.96 GB float32) |
| Subject count | 57 |
| LightGBM threads reported | 76 |
| TAP trials requested | 40 |
| LP trials requested | 40 |
| Folds per trial | 3, except trials pruned under the existing Optuna policy |
| Primary / alternate model budgets | 32 / 24 KiB |
| TAP threshold / LP release threshold | 0.85 |
| LP smoothing | mean of 10 Q20 predictions |

The reported TAP trial times ranged from ~3.8 to ~466 seconds. Times vary strongly with selected features, tree depth, number of trees, pruning and replay cost. **Do not infer which of these is dominant from total trial time alone.**

### Current execution path

~~~text
main()
  lean8.ensure_datasets()              # preprocessing BEFORE Optuna
  ensure_labels()                      # reusable labels
  train_task("tap")
    build_task_data()                  # reusable train and replay caches
    split_heldout()                    # person-disjoint
    40-trial sequential Optuna study
      objective(trial)
        for each of 3 GroupKFold folds:
          calculate per-row weights
          fit LightGBM candidate
          determine exact 32/24-KiB tree counts
          gather selected replay rows/columns
          predict on EVERY corresponding replay row
          event-level TAP or press-level LP replay
          report fold score to the existing pruner
    candidate C export, held-out replay, parity, baseline comparison
  train_task("lp")
    same training/validation framework; different replay objective
~~~

### Already implemented; do not optimize a nonexistent problem

- Lean-8 features are computed offline and saved to augmented CSV files. They are **not** regenerated per trial or per fold.
- Lean-8 generator has a per-file incremental manifest (<code>_lean8_files.json</code>).
- Training/replay data have a fingerprinted cache and a memory-mapped replay matrix.
- The trial is serial (<code>study.optimize(..., n_jobs=1)</code>); LightGBM itself uses the thread count.
- The trainer already has a <code>MedianPruner</code> that checks after folds.
- <code>raw_fns()</code> already shares leading-tree raw-score evaluation between the two budget variants; do not accidentally evaluate identical trees twice.
- Separate 32/24-KiB budgets, TAP event replay, LP 10-sample Q20 press replay and final C parity already exist.

**Initial generation of 5,352 source files is a one-time/warm-cache concern. The principal repeated cost is the 3-fold fit + full validation replay per trial.**

## 2. Hard invariants: what must NEVER change

Every optimization must preserve the following unless a separate, deliberately versioned research study authorizes a behavioral change:

1. **Inputs:** exactly the same raw recordings, timestamps, row order, generated Lean-8 values, label sidecars, user/group mapping and feature-column order.
2. **Labels:** existing unified binary rules, unknown (-1) exclusions, the same TAP onset labels and LP polarity (class 1 = RELEASE/REST, class 0 = HOLD). No regression conversion.
3. **Sampling:** identical selected training-row indices, class weights, per-recording weighting, hard negatives, frame gate and all source folders. Do not cap rows, subsample or shrink replay sets.
4. **Validation:** identical 22% person-disjoint holdout, the same 3 GroupKFold splits, same independent held-out people and same recording coverage. No mixing of people across sets.
5. **Training:** identical search space, hyperparameter candidates, seeds, LightGBM objectives, 40 requested trials per task, learning parameters, and existing pruning policy. No shorter tree search or lower model budget.
6. **Firmware behavior:** same TAP rising-edge/refractory and matching rules; same Q20 conversion; same LP ten-prediction mean, 0.85 threshold and timing/release rules; same frame gate and chronological sequence boundaries.
7. **Selection:** identical scoring formula, 32/24-KiB exact export-size checks, rank/selection logic, final refit, C header API and baseline comparisons.
8. **Precision:** retain float32 feature storage and the same floating-point math used by the validated training and firmware paths. No FP16, approximate thresholds, approximate model bytes, approximate PSD features or relaxed C parity.
9. **Trial persistence:** resume the existing Optuna SQLite study without deleting, renumbering, replaying or replacing its completed trials.
10. **Production safeguard:** previous TAP and LP v20 headers are read-only baselines. Training never silently overwrites firmware production headers.

### Explicitly excluded shortcuts

**Not permitted under this plan:** fewer folds, fewer trials, only a sample of replay rows, heuristic early elimination of candidates, stronger pruning, changing threshold/labels, reducing the feature universe, GPU training without an exactness proof, half-precision replay, replacing exact tree-size accounting with a guess, or running simultaneous Optuna trials merely to increase CPU use.

Even if such changes sometimes improve the score, they alter the experiment. They belong in an optional, separately named research track, not a zero-sacrifice optimization.

### Equivalence levels

- **Data equivalence:** row counts, file/subject/category identities, ordered row IDs, labels, weights, feature schemas, feature float32 bytes, replay-window boundaries and gate masks match.
- **Model equivalence (preferred):** with the same parameters and seed, serialized LightGBM tree structures and exact exported C header bytes match. If they do not, the optimization is not “zero-change” by default.
- **Prediction equivalence:** all original/optimized probabilities (with declared numerical tolerances), Q20 integer outputs, threshold decisions and event/release times match on the complete replay set.
- **Metric equivalence:** TAP onset hits, false events/minute, onset latency; LP holds kept, premature releases, timely releases, tap release, stuck cases, release latency; aggregate/fold scores, selected candidate and export bytes match.

**Strict promotion gate:** no changed Q20 value, TAP decision, LP release decision, event attribution or candidate ranking on the frozen acceptance corpus. If floating-point differences appear in training under a changed thread setting, keep the original thread setting or treat the result as a *new* experiment, not a proven exact optimization.

## 3. Phase 0 — Freeze the active experiment and measure before editing

**Priority: P0. Risk: minimal. Benefit: critical for choosing the real bottleneck.**

Do not edit the training code while the current Linux process is running. Do not remove the SQLite study or derived caches. Save a reproducibility manifest containing:

- commit SHA; Python, NumPy, pandas, LightGBM, Optuna, OpenMP and compiler versions;
- CPU topology (physical/logical/NUMA), available memory, CPU affinity and storage filesystem;
- hashes of generator, labels, source-file manifest, training/replay cache fingerprint and firmware reference header;
- exact feature-set list, 40-trial study identifier, sampling and replay constants, budgets and frame gate;
- frozen study trial numbers, trial states, parameters, values and intermediate fold scores.

Add timestamped stage timers without changing function outputs:

| Timer | Around |
| --- | --- |
| <code>features_s</code> | <code>lean8.ensure_datasets()</code> |
| <code>labels_s</code> | <code>ensure_labels()</code> |
| <code>cache_build_s</code> | <code>build_task_data()</code> |
| <code>fold_prepare_s</code> | feature view + weight/split preparation |
| <code>fit_s</code> | <code>fit_model()</code> |
| <code>size_s</code> | <code>trees_within()</code> for both budgets |
| <code>replay_gather_s</code> | <code>_ranges_predict()</code> data extraction |
| <code>replay_predict_s</code> | LightGBM <code>booster.predict()</code> calls |
| <code>replay_score_s</code> | TAP/LP replay scoring functions |
| <code>study_io_s</code> | Optuna SQLite/report writes |
| <code>export_s</code> | finalist refit, export, C replay and parity |

Use <code>time.perf_counter()</code> and record per trial + per fold JSONL, with <code>psutil</code> optionally reporting peak RSS/CPU and process I/O. Do not introduce a heavy profiler in every run: do one representative <code>cProfile</code> or <code>py-spy</code> run after the lightweight timers identify the slow section.

Also record actual OpenMP worker utilization. 76 requested threads does **not** prove that the 76 threads are productive.

**Exit criterion:** identify how much time is fit vs model sizing vs replay-data gather vs replay prediction vs replay accounting. Never optimize by intuition when measured data are available.

## 4. Phase 1 — Keep features/labels/datasets truly incremental

**Priority: P1. Risk: low. Scope: startup, not Optuna model scoring.**

The generator already uses a manifest with source size/mtime and a generator hash. Keep this design and harden it:

1. Preserve each raw CSV byte-for-byte. Derived outputs only under <code>DATA/DATA_TAP_LEAN8</code> and <code>DATA/DATA_LONG_PRESS_LEAN8</code>.
2. For an unchanged source and identical feature-algorithm version, verify the output file exists and reuse it. Print generated/reused counts.
3. Use atomic output files and atomic manifest updates. A power loss or interrupted multiprocessing run must not produce a false cache hit.
4. Cache the *feature algorithm* version separately from orchestration-only code formatting, progress messages and worker-count settings, so a harmless generator edit does not force 5,352 recomputations. The algorithm hash must include all numerical inputs, LUTs, frame-detection rules, quantization and warm-up behavior.
5. Keep an optional strong-integrity source SHA/checkpoint audit rather than relying **only** on size + mtime for irreplaceable data.
6. Reuse the fingerprinted training/replay cache when all relevant input+label+schema hashes match. Changing only an unrelated Markdown file must not invalidate it.
7. On any actual numerical change to Lean-8, labels or frame alignment, intentionally invalidate relevant derived caches and rebuild. Never conceal stale results in pursuit of speed.
8. Avoid duplicate full decompression of unchanged CSVs. All 150 features remain available to training and export.

**Acceptance:** repeat runs produce zero regenerated files and reuse train/replay caches. Compare derived CSV numerical columns and a complete dataset fingerprint against the baseline. Initial/warm-cache costs are reported separately.

## 5. Phase 2 — Prepare immutable folds and weights ONCE

**Priority: P1. Risk: low to medium. Likely saves substantial Python overhead.**

Today <code>objective()</code> creates the 3 GroupKFold splits on each trial, and <code>weights_for_task()</code> repeatedly builds per-file counts, loops over file names and recomputes class/file normalization.

The split depends on the **dataset and groups**, not on Optuna hyperparameters. Most weight terms depend on **fold + labels + recording identity**, not on hyperparameters. Only the trial's positive-class multiplier changes.

Create a read-only <code>PreparedTask</code> and <code>PreparedFold</code> once after <code>split_heldout()</code>:

~~~python
prepared = prepare_task_once(data, rpl, train_ids, task)
# Each fold holds the EXACT same ordered indices/groups, X/y views,
# fixed base per-file/per-class weights and reusable replay metadata.

def objective(trial, prepared):
    params = suggest_params(trial, prepared.task)  # unchanged Optuna sequence
    positive_weight = trial.suggest_float(...)     # unchanged distribution
    selected = prepared.feature_set[trial_feature_set]
    for fold in prepared.folds:
        w = fold.base_weights.copy()
        w[fold.positive_mask] *= positive_weight  # same original arithmetic/dtype
        model = fit_model(fold.X[:, selected], fold.y, w, params, prepared.primary_budget)
        # same exact trimming and full person-held-out replay
~~~

This pseudocode is conceptual: **preserve the existing order of <code>trial.suggest_*</code> calls** and positive-weight multiplication as in the current objective; do not move suggestions just because the example is arranged differently.

Implementation rules:

- Precompute only invariant operations. Retain the exact order of row indices and exact class/recording weight arithmetic.
- Avoid Python loops over hundreds of thousands of per-row string file identifiers in every trial; encode file IDs once and use NumPy count operations if parity is exact.
- Do not normalize sample weights differently. Compare the precomputed vectors **elementwise** at several positive-weight settings, including the LP special balancing of release examples.
- Precompute/retain fold group and replay-range membership rather than recreating pandas <code>group.isin(...)</code> masks thousands of times.
- Cache per-fold selected-feature index arrays; do not mutate the master feature list or allow metadata to become a model input.

**Acceptance:** same 3 fold indices, same sample count and weight bytes, same Optuna suggestions, same LightGBM serialized model and same scores on frozen trials.

## 6. Phase 3 — Eliminate avoidable replay matrix copies, not replay rows

**Priority: P1. Risk: medium. Major possible impact due to ~3.96-GB replay source.**

Currently <code>_ranges_predict()</code> batches recording ranges, concatenates full 150-column chunks from a float32 memmap, then performs <code>[:, selected]</code>. For many feature sets this scans and copies columns that are not used by the model. Similar copies recur for every fold/trial.

**Safe approach:**

1. Preserve original recording order, chronological row order and all expected replay rows.
2. Precompute the exact per-fold list of replay spans and their row-to-recording metadata once.
3. Read/cache only the **selected columns**, in the same feature order and float32 dtype, instead of materializing a full 150-column chunk and then taking a subset.
4. Use bounded reusable buffers rather than allocating new temporary gigabyte arrays repeatedly. Do not grow RAM consumption without limit.
5. Optionally prepare on-disk, memory-mapped per-feature-set replay views (or a bounded LRU cache) when a feature set is used repeatedly. A full 150-column copy for each of many feature sets is wasteful; choose based on actual available RAM/NVMe throughput.
6. Preserve exact chunk boundaries *as logical recording boundaries* for event accounting. Changing **I/O chunk size** is allowed only if the numerical predictions and sequence-state resets are unchanged.
7. All rows included in the original replay must still be predicted. No downsampling, no event-window cropping, no “quick replay”.
8. Provide a fallback to the current <code>_ranges_predict()</code> if a cached view is missing/invalid or parity fails.

**Acceptance:** exact same ordered X float32 elements for every chosen feature set, same per-recording index spans, full row count, same per-row probabilities/Q20 and same TAP/LP events.

**Important:** the 6.6M rows are already prepared; optimizing replay I/O must not cause Lean-8 feature recomputation.

## 7. Phase 4 — Speed up predictions without changing tree math

**Priority: P1/P2. Risk: medium. Potentially the largest repeated gain, depending on profiling.**

The current <code>raw_fns()</code> predicts separate leading-tree segments for 32-KiB and 24-KiB candidates, accumulating raw margins and converting them with <code>R.sigmoid()</code>. This is already better than running two completely independent full-tree predictions.

Possible safe improvements:

- Reuse immutable matrix views and inference buffers between model budgets and replay chunks.
- Deduplicate equal tree counts: if 32-KiB and 24-KiB trim to the same <code>k</code>, predict once and reuse the result.
- Cache trial-invariant validation metadata (onset times, negative-time masks, LP press windows, usable-frame masks).
- Avoid converting the same predictions to Q20 repeatedly for LP when the selected model/budget count is identical.
- Evaluate a faster **exact** CPU inference wrapper only if it reproduces the current LightGBM tree traversal, raw margins, Q20 and threshold decisions on every replay row. An alternative predictor that is merely “close” is **not** accepted by the strict plan.
- Keep a single model's inference behavior deterministic; do not silently switch to a different numerical precision or tree threshold semantics.

**Do not** change <code>R.q20()</code>, sigmoid conventions, <code>start_iteration</code>/<code>num_iteration</code> behavior, TAP 0.85 comparison or LP moving average. Small differences near the threshold can change an entire hold-release event.

**Acceptance:** full replay probability/Q20 equality and identical event/release timestamps. Benchmark before/after prediction CPU seconds, replay GB read and peak RSS.

## 8. Phase 5 — Exact model-size calculation without repeated model serialization

**Priority: P2. Risk: medium/high. Do only if timers show it matters.**

Current <code>trees_within()</code> performs binary search; each query may serialize a truncated <code>lgb.Booster</code> and invoke the <code>export_plus</code> size estimator. This can be expensive, especially for large trial models.

Build an internal exact-prefix size index from the exporter's **real byte accounting**:

~~~text
booster trained once
  -> enumerate actual tree structures once
  -> calculate exact static bytes of each tree
  -> prefix sum of exact tree bytes plus fixed model overhead
  -> choose largest k that fits 32 KiB and 24 KiB
  -> verify result with the original exporter
~~~

Caveats:

- Inspect <code>export_plus.py</code> to determine exactly what is counted (node layout, leaf values, constants, alignment and any per-model overhead). Do **not** assume every tree costs 6 × number_of_nodes without checking.
- If any output representation has cross-tree compression, nonlinear overhead or other dependence on selected <code>k</code>, a prefix sum alone may be invalid.
- For each accepted trial, verify the proposed <code>k</code> and <code>k+1</code> against the existing <code>static_bytes(truncated(model, k))</code>. Maintain the current binary-search fallback.
- Cache size calculations for the same booster/tree count/budget; use immutable model identity, not a guess based on hyperparameters.

**Acceptance:** identical chosen tree count and byte count for both budgets across many actual models (shallow/deep, 8–127 leaves), including boundary sizes near 24/32 KiB; exported C headers remain within the exact budgets.

## 9. Phase 6 — Benchmark CPU threads and memory topology

**Priority: P2. Risk: low to benchmark, higher to deploy if models differ.**

The server reports 76 LightGBM threads, while Optuna runs one trial at a time. Many threads can hurt on memory-bound or NUMA workloads, but **76 threads being “too many” has not yet been established**.

Benchmark the SAME frozen trial/fold with worker counts such as 8, 16, 24, 32, 48, 64 and 76. Use the same data/cache, parameter set, random seed, version and fold. Capture:

- fit time and replay time separately;
- CPU utilization and context switching;
- NUMA placement, physical vs logical CPU topology and CPU affinity;
- RAM bandwidth/peak RSS, disk cache effects and thermal/clock behavior;
- serialized model hash and all output predictions.

Select a faster setting **only if the strict model/result equivalence gate passes**. OpenMP and LightGBM scheduling may change floating-point reduction order; claiming zero behavioral change without comparison is unsafe.

Keep the current serial Optuna proposal order. Running multiple trials simultaneously, even with carefully allocated cores, changes when TPE sees previous scores and can change future candidates. That is NOT an exact same-study optimization; leave it out of the strict implementation.

### Optional later research branch (not approved by this document)

Concurrent trials with separate processes or a shared RDB may increase throughput, but modify TPE's adaptive proposal sequence. Likewise, GPU fitting or altered tree histograms may change models. Research these in a different experiment only, never silently as a zero-sacrifice patch.

## 10. Phase 7 — Optional LightGBM data-structure reuse, only with proof

**Priority: P3. Risk: high; postpone until profiling and lower-risk wins.**

LightGBM repeatedly bins the same training data. It might be possible to reuse preconstructed <code>Dataset</code> objects or cached binning across trials, especially for repeated feature sets.

This is tempting but potentially changes models because of feature prefiltering, bin construction, reference datasets, changing <code>min_child_samples</code>, feature order and the sklearn-wrapper/native-API difference.

Requirements before implementing:

1. Preserve training row order, sample weights, labels and feature set.
2. Preserve <code>max_bin=127</code>, bin sample behavior, all regularization/histogram parameters and seed.
3. If changing <code>feature_pre_filter</code>, document how it affects model equivalence; a “better” but different model is not a zero-change optimization.
4. Compare serialized trees and full inference after every trial. Reject/revert if they differ.
5. Keep the original sklearn <code>LGBMClassifier.fit()</code> path as the correctness reference and fallback.

Do not do this just for theoretical speed; prove the binning step is material first.

## 11. Phase 8 — Finalist export/replay and repeated output work

**Priority: P2/P3. Risk: low to medium.**

Candidate selection and final export happen *after* Optuna and should remain fully rigorous:

- Preserve all original top-k candidate rankings, the alternative-budget candidate, original holdout and full-data final refit.
- Cache immutable finalist-specific intermediate results when inputs and model hash match, but never share an intermediate model between different train/held-out splits.
- Avoid exporting/compiling the same exact candidate model twice if the model bytes, feature schema, target architecture and compiler configuration are identical.
- Retain original C-vs-Python parity (up to <code>PARITY_ROWS=200000</code> as configured), all full held-out event replays, and the steady-state Lean-8 evaluation.
- Keep prior TAP model, LP v20 and prior unified baselines read-only and hash-check their contents.
- Do not let a cached failed parity check be replaced by a “pass” based on timestamps or a different compiler.

**Acceptance:** same finalist headers, exact byte sizes, same candidate selection, comparison table and baseline results.

## 12. Existing study and cache safety on the active server

**Important:** The current 40-trial study is already in progress. These are the rules for updating safely:

1. Finish or stop at an ordinary trial boundary. Do not replace imported Python modules mid-process.
2. Back up <code>research/gesture/results_unified_v2/tap/optuna_study.sqlite3</code> using SQLite's backup API or a safe paused-process copy. Do the equivalent for LP if it already exists. Preserve trial attributes and intermediate fold scores.
3. Back up the existing <code>split.json</code>, cache fingerprint files and available summary/metrics. Never wipe the result directory.
4. Make optimization changes on a separate Git branch; benchmark against the frozen base commit and frozen trial parameters in a separate results directory.
5. Avoid changing <code>OUTPUT</code>, study name, feature-set categorical options/order, sampler/pruner behavior or random seeds when resuming the original study.
6. After strict parity passes, merge only behavior-preserving changes; restart/resume the same study. Optuna must see the existing completed/pruned trials and continue to the original total.
7. A mismatch in dataset fingerprint or Optuna schema must **fail closed** with a clear diagnostic. Do not silently regenerate an incompatible study under the same name.

Do not treat “number of completed trials” and “number of attempted/pruned trials” as interchangeable; preserve the current definition used by the training loop.

## 13. Test suite and acceptance gates

Implement the tests **before** optimizing the slow path.

### A. Data and cache tests

- Raw input and labels never modified; all expected files present.
- Derived generator produces the same 24 Lean-8 columns (float32 values), row count, frame markers and timestamp alignment.
- Warm-up 50-frame and steady-state 500-frame gates match on every row.
- On cache hit, source CSVs are not read/recomputed and train/replay row identity remains unchanged.
- On one changed input or altered numerical algorithm, only the valid dependency set is rebuilt; no stale cache accepted.
- Compare exact float32 bytes, frame counts and ordered (recording, row) fingerprints.

### B. Training tests

- For a frozen collection of actual trial parameters from the ongoing Optuna SQLite DB, each of the three person-grouped fold index arrays is identical.
- Supervised X/y, weights, row order and feature subset ordering are identical.
- All relevant LightGBM model structures match for every proposed exact optimization.
- All 32/24-KiB trim results and model bytes match.

### C. Replay tests

- Entire validation dataset, not a quick subset.
- TAP: exact event onsets, accepted rising edges, refractory suppression, false positives/minute, recall and latency.
- LP: exact Q20 sequence, ten-value moving averages, release timestamp, hold survival, taps released, no stuck holds added/lost and latency.
- Original vs optimized per-trial fold scores and final CV score unchanged.
- No reduced person/recording/folder coverage, including soft holds and bad electrode contact.

### D. Export and end-to-end

- Full emitted C header API compiles with the existing synthetic <code>t_core</code>.
- C/Python prediction parity and *actual firmware compilation* still required.
- Final selected trial, primary and alternate budget winners, C bytes, held-out reports and baseline hashes unchanged.
- Run original <code>research/gesture/test_unified_smoke.py</code> and the new optimization-specific parity suite.

### E. Throughput acceptance

Use an unmodified baseline and optimized candidate on the same host and cached data, with warm-up runs and multiple repeats. Report:

| Stage | Baseline seconds | Optimized seconds | Speedup | Parity |
| --- | ---: | ---: | ---: | --- |
| Incremental features/labels | TBD | TBD | TBD | pending |
| TAP fold prepare | TBD | TBD | TBD | pending |
| TAP model fit | TBD | TBD | TBD | pending |
| TAP size selection | TBD | TBD | TBD | pending |
| TAP full replay + score | TBD | TBD | TBD | pending |
| LP model fit | TBD | TBD | TBD | pending |
| LP full replay + score | TBD | TBD | TBD | pending |
| Finalist C export/parity | TBD | TBD | TBD | pending |
| Complete unchanged study (if measured) | TBD | TBD | TBD | pending |

A patch is accepted if it passes strict parity and provides a repeatable throughput or memory improvement without regressing other key stages. If parity fails: **revert**, even if the change is faster.

## 14. Concrete implementation order

| Step | Change | Files | Expected risk | Merge condition |
| --- | --- | --- | --- | --- |
| 0 | Freeze baselines, add timings and machine profile | new benchmark helper, train_unified.py | Low | No changed outputs |
| 1 | Strengthen incremental source/dataset caching | generate_lean8_features.py, train_unified.py | Low | Same generated features and matrices |
| 2 | Precompute fold indices, fixed weight vectors and replay ranges | train_unified.py | Low/medium | Same X/y/weights/folds |
| 3 | Avoid full-column replay gathers; bounded reusable feature-set buffers | train_unified.py | Medium | Same row/column bytes and replay |
| 4 | Reuse exact prefix prediction work and validation metadata | train_unified.py, replay.py | Medium | Exact Q20/events/scores |
| 5 | Replace repeated model-size serialization with proven exact accounting | train_unified.py, export_plus.py | Medium/high | Same tree counts/C bytes |
| 6 | Benchmark threads and NUMA; use only a strictly equivalent faster setting | benchmark helper, configuration | Medium | Same model/header/predictions |
| 7 | Consider cached LightGBM binning only if large measured cost and model equality | train_unified.py | High | Identical fitted models |
| 8 | Cache immutable finalist work, run full acceptance and CI | train_unified.py, tests, CI | Low/medium | Full output parity |

**Each step is independently reviewable and reversible.** No large rewrite or new model objective.

## 15. Measured success criterion

The target is **less time doing exactly the same work**, not a new “fast mode.”

We retain all 5,352 original recordings, all eligible training rows, all 150 available firmware features, all validation rows, 3 person-grouped CV folds, 40 Optuna trials per model, two exact flash budgets, original Q20 behavior, strong replay validation, and original C export checks.

Success means that a fair before/after benchmark reports lower wall-clock time and/or peak memory with no change in the trained model, replay decisions, candidate ranking or firmware artifacts.

**No change can be advertised as “without sacrificing anything” until these equivalence tests pass.**

## 16. Separate outstanding issue — firmware feature parity

The user's log includes:

~~~text
LEAN8_FIRMWARE_PARITY {"status": "not run", "reason": "firmware header not found (set MUDRA_LEAN8_FIRMWARE_HEADER)"}
~~~

This is **not** the cause of slow Optuna trials. It is, however, an independent production-validation gap. Set the path to the actual firmware header and run full Lean-8 C/Python parity before promoting any trained model to production. Do not skip parity to save time.

---

### Recommendation

**Start with timers + frozen-trial parity, then immutable fold/weight preparation and replay matrix copy reduction.** These target repeated work without changing the ML experiment. Optimize exact model-size accounting next only if it is measurably expensive. Tune thread counts with a controlled benchmark rather than assuming 76 is optimal. Keep Optuna sequential and its 40-trial/3-fold protocol unchanged.
