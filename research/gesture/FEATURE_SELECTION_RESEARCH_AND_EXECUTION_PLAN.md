# Feature selection research and execution plan
## Unified TAP/SWIPE + LONG_PRESS — all firmware features, no hand-written feature subsets

**Status:** research-backed execution plan; no training code or active study changed by this document.  
**Repository snapshot:** main at commit aa37e9e2ea2264f76d2ee2eccffa7dd8c7c6e93d (2026-11-18).  
**Primary implementation:** research/gesture/train_unified.py, research/gesture/replay.py, research/gesture/ablate_lean8.py, research/gesture/bench_unified.py.  
**Research references:** see [Research basis](#14-research-basis).

---

## 1. Executive recommendation

Replace the current hand-written feature_set menu with a **task-specific, group-aware, nested feature-selection pipeline that starts with every available firmware feature and learns the subset from data**.

The recommended approach is:

> **Cross-fitted conditional/block permutation ranking + replay-scored wrapper selection of K + selection-stability analysis + one untouched person-held-out test.**

This is not “pick ACC”, “pick EMG”, “pick raw126”, “pick Lean-8”, or “keep the top 32 features”. There will be no manually curated feature allow-list, deny-list, or fixed K shortlist.

- Every one of the 150 available p_core features is eligible.
- TAP and LP learn separate feature rankings because they have different targets and firmware objectives.
- Correlations and redundant features are discovered from the training data, not from hand-written feature-family rules.
- K is learned from the full integer range 1–150.
- The selected subset is judged by the real firmware replay objective, not frame-level accuracy, univariate F-score, or feature importance alone.
- K=150 remains the all-feature control. If it wins, we keep all 150. There is no requirement to reduce the feature count.

There is no universally best feature-selection algorithm independent of the task. The gold standard is the **evaluation protocol**: selection must happen entirely inside training data, validation must be person-disjoint, the actual deployment metric must choose the subset, and the final holdout must remain untouched. The algorithm below is designed around those requirements.

## 2. Latest repository audit

The latest commit reviewed is aa37e9e:

- It adds the zero-change training optimizations and equivalence harness.
- It adds research/gesture/ablate_lean8.py.
- It preserves the existing TAP 0.85 threshold.
- It changes the LP release threshold to 0.5 in a separately versioned LP study.
- It records the feature-selection performance plan as a future area, not an implemented selector.

### 2.1 Current feature universe

The unified trainer currently has:

| Source | Feature count | Current contents |
| --- | ---: | --- |
| Original firmware fields | 126 | ACC 72 + EMG 45 + WL 9 |
| Firmware Lean-8 | 24 | 8 fields × 3 EMG channels |
| **Total** | **150** | **All currently available firmware fields** |

These family names are useful for reporting and diagnosis only. They must not constrain the selector.

### 2.2 What the current code actually does

In train_unified.py:

1. common_p_core() finds the feature columns present in all selected recordings.
2. feature_sets() constructs named subsets including all_available, raw126, lean8_24, raw126_lean8_instant12, and the manually defined subsets imported from previous_tap.FEATURE_SETS.
3. objective() asks Optuna to choose feature_set with suggest_categorical.
4. The same fixed menu is available to TAP and LP.

The imported previous_tap.FEATURE_SETS contains hand-written combinations such as instant36, acc72, emg45, acc_emg117, emg_wl54, and acc_current18.

This is a **feature-family ablation menu**, not general feature selection. Optuna chooses among a small number of predefined groups; it does not learn an arbitrary subset of the 150 columns.

### 2.3 The current ablation script is not a selector

ablate_lean8.py compares four named sets (raw126, lean8_24, raw126_lean8_instant12, all_available) and two fixed model configurations. It is useful for answering whether Lean-8 helps, but it cannot discover arbitrary useful combinations of the 150 individual columns.

Do not extend this script by adding more hand-written feature sets. Replace its role with a generic selection experiment that accepts an ordered list of feature names produced by a fitted selector.

### 2.4 Important issues to resolve before trusting selection results

**A. The final holdout is not currently fully untouched.**  
promote() examines held-out C replay and skips a candidate when its holdout score is zero, then promotes the next candidate. That means the held-out people influence which candidate is selected. The new protocol must select the candidate using development/CV data only, then report the holdout result once. A failed holdout result must be reported, not used to choose a different candidate.

**B. Lean-8 readiness does not currently match the firmware gate shown in the earlier firmware integration.**  
The current trainer defaults to LEAN8_MIN_FRAMES = 50, while LEAN8_STEADY_FRAMES is 500 and the generator marks lean_8_features_ready() true at 500 frames. The firmware code shown earlier returns from model_execution() until lean_8_features_ready() is true. If that is still the deployed behavior, rows 50–499 are not inference-eligible and must not be treated as normal production rows during feature selection. Before starting the new study, confirm the deployed firmware gate and make the dataset/replay gate match it exactly. Do not silently train on a feature state the device never consumes.

**C. TAP labels are conservative and have a known coverage ceiling.**  
The current labeler requires a matched press with teacher score at or above the 1-false-event/min threshold (about 0.9664) for a positive label. Weaker/unsupported presses and disputed spans are generally unknown (-1). The committed label statistics report 12,330/15,150 TAP-folder presses and 7,289/9,166 SWIPE-folder presses as labeled positives. Feature selection cannot learn to detect events that the teacher never labels positive. Keep label sensitivity as a separate audit; do not silently change labels inside this feature-selection experiment.

## 3. Research conclusion: what “gold standard” means here

Feature importance and feature selection are different:

- **Importance** estimates how much a fitted model relies on a feature.
- **Selection** chooses a subset that improves or preserves generalization when the model is retrained using that subset.
- **Deployment selection** must consider the actual embedded event behavior, model-size budget and firmware timing—not only a statistical importance score.

For this project, the most defensible standard is:

1. Person-grouped nested selection: no person's rows may appear in both sides of a selection/evaluation split.
2. All data-driven operations (ranking, correlation clustering, K choice, selector tuning) are fit using training people only.
3. The selection objective is the actual chronological firmware replay metric.
4. Correlated predictors are treated carefully; ordinary single-column permutation can understate the importance of interchangeable predictors.
5. Selection stability is measured across people/folds/resamples.
6. A final person-held-out cohort is used once, after the pipeline and candidate have been selected.

### Why not use a simple filter?

- SelectKBest(f_classif) and other univariate filters can miss nonlinear interactions and features that are useful only in combination.
- LightGBM split/gain importance is useful for screening but is not sufficient as the final selection criterion; tree impurity importance can be biased and can reflect overfitting.
- SHAP is useful for explaining a fitted model and generating a ranking, but correlated features can share or exchange attribution. SHAP alone does not prove that a feature is necessary.
- Ordinary row-wise permutation breaks temporal structure and can create unrealistic feature combinations. It is also misleading when a correlated feature can substitute for the permuted one.
- Boruta is an all-relevant screening method, not necessarily the smallest subset that maximizes our event-level firmware objective. It can be a comparator, not the production decision rule.
- Exhaustive search over all 2^150 subsets is infeasible. The selection algorithm must search intelligently without preselecting a hand-written feature family.

## 4. The proposed selection method

### 4.1 Candidate pool: every available firmware feature

For each task, start with:

    candidate_columns = common_p_core(records)  # all available firmware p_core fields
    assert len(candidate_columns) == 150        # for the current complete dataset

If a future dataset genuinely lacks a field, record that as a schema/data-availability issue. Do not remove columns because their names belong to ACC, EMG, WL, or Lean-8.

There must be no code equivalent to:

- require ACC features;
- prohibit EMG-only or WL-only features;
- force a minimum number from a family;
- only allow raw126, lean8_24, or another hand-written group;
- manually nominate “important” feature names.

Family membership may appear in reports to help explain the learned result. It must never filter the candidate pool or affect selection scores.

### 4.2 Select separately for TAP and LP

Use the same selection machinery, but fit it independently for each task.

| Task | Binary target | Exact deployment evaluation |
| --- | --- | --- |
| TAP/SWIPE | 1 = onset; 0 = no onset; -1 excluded | Fixed TAP threshold 0.85; chronological rising-edge/refractory replay; event recall, false events/min and latency |
| LONG_PRESS | 1 = RELEASE/REST; 0 = HELD; -1 excluded | Latest LP firmware release threshold 0.5; Q20 values averaged over 10 predictions; hold survival, eventual/timely release, tap release and latency |

Do not force the two tasks to share a selected feature subset. Their labels, class polarity and operational objectives differ.

### 4.3 Fold-local, out-of-person feature relevance

The feature-ranking process must not inspect the outer validation people.

For each of the existing three person-grouped CV folds:

1. Take only that fold's training people.
2. Run a second, group-aware cross-fitting pass inside those training people.
3. In each inner split, fit a screening model using all 150 candidate features and the existing binary labels/weights.
4. Evaluate that model on the inner validation people using complete chronological recordings.
5. Estimate out-of-person feature relevance using the replay metric and correlation-aware perturbations.
6. Aggregate relevance across inner folds and people to produce a feature ranking for this outer fold.

The ranking for outer fold 0 must never use labels, events or model predictions from outer fold 0's validation people. Repeat independently for folds 1 and 2.

This ranking is computed once per fold and cached by the exact data/label/feature-schema/ranker version. It is not recomputed for every Optuna trial.

### 4.4 Primary relevance measure: conditional and block-aware permutation

Use a model-agnostic permutation measure tied to the actual replay objective:

    I_j = S_baseline - S_perturbed,j

where S is the same task-specific replay score used by the trainer.

The perturbation must respect this dataset's structure:

- Never shuffle individual time rows globally.
- Keep recording boundaries intact.
- Use contiguous temporal blocks or within-recording circular/block shifts so the perturbation does not destroy every feature's autocorrelation.
- Estimate feature dependencies from the training partition only.
- For highly correlated predictors, measure both individual conditional importance and joint importance of automatically discovered correlated groups.
- Where conditional sampling is used, fit the conditional generator on inner-training people only; never fit it using the validation people.
- Repeat perturbations with deterministic seeds and retain the mean, dispersion and confidence interval.

**Correlation handling is data-driven.** Build a correlation/dependency structure from the training partition; do not hard-code feature families. Use a data-driven clustering/cut criterion and report the discovered clusters. Jointly perturb a correlated cluster to determine whether the group contains predictive information. Then estimate each member's incremental/conditional contribution inside that group.

A low *marginal* importance for one feature is not sufficient reason to discard it if its correlated group is important. Conversely, a large group importance does not mean every member must be retained.

If a conditional sampler cannot generate realistic in-distribution replacements, fail that conditional-importance calculation and fall back to a clearly labeled group-level block-importance result. Do not silently report ordinary row-wise permutation as conditional importance.

### 4.5 Use model-based importance as a cross-check, not the final decision

Record the following from the same all-feature screening fits:

- LightGBM gain importance;
- split count;
- mean absolute out-of-person TreeSHAP contribution, calculated in chunks;
- conditional/block permutation importance;
- correlation-cluster importance.

These are complementary diagnostics. Do not select the final subset from gain, split count or SHAP alone.

The final subset must win or remain non-inferior on the real replay objective after the model is retrained with that subset.

### 4.6 Search K from 1 to 150 — not from a hard-coded list

Replace Optuna's categorical feature_set parameter with an integer:

    n_features = trial.suggest_int("n_features", 1, len(candidate_columns))

For each outer fold:

1. Use that fold's own training-only feature ranking.
2. Select the first K features from that ranking.
3. Preserve the original canonical feature-column order when building the model matrix and C schema.
4. Fit LightGBM using the same trial hyperparameters, weights, training rows and model-data budget.
5. Replay every validation sequence using the existing firmware logic.
6. Aggregate the exact per-fold objective.

K is therefore a real learned parameter, not a list such as [32, 48, 64, 96, 128, 150].

**K=150 is the all-feature control.** It must be represented in the study and compared with the selected models. The optimizer is allowed to choose 150. There is no goal to reduce K at any cost.

The ranking is only an ordering heuristic. The wrapper evaluation (retrain with the selected subset and replay) determines whether K is actually useful.

### 4.7 Protect interactions and avoid a ranking-only selector

Top-K ranking alone can miss features that are useful only in combination. The implementation should therefore include a floating wrapper refinement around promising subsets:

- propose adding a feature outside the current subset;
- propose removing a selected feature;
- propose swapping a selected feature with an unselected feature;
- evaluate each proposal with the same person-grouped training/validation discipline and exact replay objective;
- retain changes only when the inner selection criterion supports them.

All 150 columns remain eligible for these moves. The refinement must not be restricted to a manually chosen family or to features with a particular name.

Use the ranked top-K subset to make the search tractable; use the wrapper replay score to decide. The final reported subset must be the exact subset that was trained and exported, not a post-hoc list assembled from feature importance.

## 5. Validation design and leakage controls

### 5.1 Preserve the existing person-level split

Keep the current development/holdout split and three person-grouped CV folds for the first implementation:

- 22% of people are the final holdout.
- The remaining development people are used for feature ranking, K selection and Optuna.
- All folds remain grouped by person.
- No recording from a held-out person can enter the selector, ranking, correlation model, hyperparameter search or feature stability calculation.

If repeated group resampling is used for stability, it must also split by person—not by row.

### 5.2 The holdout is an audit, not a selector

Change the current promotion logic:

- Candidate selection must be based on development/CV scores and the predeclared model-size constraints.
- Do not inspect holdout replay to decide which candidate to promote.
- Do not choose a different K, feature ranking, threshold or model because one candidate looked better on the holdout.
- After the development selection is locked, evaluate the selected pipeline once on the holdout and report the complete results.
- If it fails on the holdout, report the failure. Do not try the next candidate against the same holdout and call that an independent test.

The current promote() behavior (skip candidates based on whether their holdout replay score is zero) must be replaced for the new experiment.

### 5.3 Selection must be fitted inside every fold

The following operations are data-dependent and must be fitted using only that fold's training people:

- correlation matrix and correlation clusters;
- conditional feature generators;
- feature-importance ranking;
- rank aggregation and K selection;
- any feature normalization or imputation (if later introduced);
- feature-subset refinement.

The outer validation fold is only used to score the candidate model selected from the fold's training data.

### 5.4 The reported CV score is a search score

Optuna compares many candidate combinations. Its best CV score is therefore a selection criterion and can be optimistic. The final person-held-out score is the independent estimate of the chosen pipeline, provided it is not reused for promotion or tuning.

If a later publication-grade performance estimate is needed, add a fully nested repeated person-grouped outer CV as a separate expensive evaluation. Do not confuse the Optuna search score with an unbiased final performance estimate.

## 6. Feature-selection stability

A subset that changes completely when a few people change is not a reliable production explanation, even if one fold has a high score.

For every feature and automatically discovered correlation group, report:

- selection frequency across outer CV folds;
- selection frequency across repeated person-group resamples;
- median rank and rank dispersion;
- conditional/group importance mean and uncertainty;
- whether the feature is selected only as a substitute for another correlated feature;
- performance when the feature or its correlated group is removed.

Calculate the Nogueira feature-selection stability measure, with confidence intervals where the resampling design supports them.

Do not impose an arbitrary “selected in at least 80% of folds” rule. Stability is evidence, not a manual whitelist. If individual features exchange places while their correlated group is stable, report that explicitly.

Use stability to diagnose whether the selected set is reproducible and to break close ties only when the predictive evidence is genuinely indistinguishable. Do not trade away a material improvement in event performance merely to make the feature list look stable.

## 7. Exact task objectives and reporting

### TAP/SWIPE

Do not use frame accuracy or F1 as the selector's primary objective.

Use the existing fixed-threshold chronological event replay:

- TAP decision threshold: 0.85;
- rising-edge event detection;
- existing 150 ms refractory;
- existing one-to-one event/onset matching windows;
- false events per minute on trusted negative time;
- event recall and latency.

Keep the current objective formula unchanged for the first feature-selection experiment. Also report the full recall/false-event trade-off and per-person/per-folder results, so a small number of difficult people or negative categories cannot be hidden by a pooled score.

Report at minimum:

- TAP and SWIPE separately;
- recall;
- false events/min;
- median and p90 latency;
- worst-person and person-distribution metrics;
- FINGER, WRIST, ARM, BAND, NOTHING and other negative categories separately;
- soft/slow gesture subtypes where the event metadata supports them.

Do not tune the firmware threshold during this experiment.

### LONG_PRESS

Use the latest firmware-aligned LP threshold from commit aa37e9e:

- binary target: 1 = RELEASE/REST, 0 = HELD;
- release threshold: 0.5 on the mean of the last 10 Q20 values;
- preserve the exact Q20 quantization and chronological ten-sample average.

Report hold-kept, premature release, timely/eventual release, stuck holds, tap release and release latency. Keep the LP threshold/versioned study separate from historical 0.85 LP baselines.

### Model constraints

For each selected subset:

- exact primary 32 KiB model-data budget;
- exact alternative 24 KiB budget;
- exact export_plus size accounting;
- real C header export and compile;
- C/Python parity;
- firmware execution/latency measurement.

Do not assume fewer input features automatically means a smaller C model. Measure the exported bytes and runtime.

## 8. Resolve the Lean-8 readiness contract first

The current repository defaults to 50 frames for LEAN8_MIN_FRAMES, while the firmware lean_8_features_ready() implementation is documented as requiring 500 EMG feature frames.

Before feature selection starts, verify the current deployed firmware behavior:

- If the firmware still blocks model execution until lean_8_features_ready() is true at 500 frames, the model-training and replay gate must use that same 500-frame condition.
- If the product intentionally wants inference after 50 frames, firmware must expose and use a readiness contract that actually becomes true at 50, and the partial-history feature values must be validated against that firmware behavior.
- If the product needs TAP inference before Lean-8 is fully ready, evaluate an explicit two-state firmware/model architecture (pre-ready model using only currently valid fields, then ready-state model). That is a separate design and must be evaluated as such—not simulated by training on rows the deployed model cannot see.

Do not let the feature selector decide this by optimizing whichever gate happens to be in the CSV. The gate is part of the firmware contract and must be fixed first.

## 9. Label coverage is a separate limitation

The current labeler uses a high-confidence sensor-confirmation rule and marks weak/unsupported/disputed presses unknown. That is intentional and supported by the earlier label research, but it means a model cannot learn positives that are absent from tap_label == 1.

For this feature-selection study:

- freeze the current labels and labeler hash;
- do not silently enable --use-weak;
- do not turn unknown -1 into negative;
- report labeled-positive coverage by person, folder and gesture subtype;
- compare model recall against confirmed labels and pointer-press coverage separately.

After the selector is evaluated, run a separate label-sensitivity experiment (strong-only vs weak/weighted labels) with its own study name and holdout. Do not mix label changes with feature-selection changes; otherwise it becomes impossible to tell which change helped.

## 10. Proposed code architecture

### New module

Create research/gesture/feature_selection.py with a small, testable API:

    fit_fold_ranker(
        X, y, groups, recordings, replay_data, task, sample_weight, feature_names,
        seed, selector_version
    ) -> FoldFeatureRanking

    select_top_k(ranking, k) -> list[int]

    report_stability(rankings, feature_names, cluster_metadata) -> dict

    validate_selection_is_fold_local(selector, train_groups, validation_groups) -> None

The exact signatures can change during implementation, but the separation of responsibilities should remain.

### FoldFeatureRanking should retain

- ordered feature names and original column indices;
- individual conditional importance;
- correlation-group importance;
- importance uncertainty/repeat values;
- per-person and per-inner-fold ranks;
- learned correlation clusters;
- selector/data/label/firmware hashes;
- fold training-group hash;
- ranker seed and configuration.

### train_unified.py changes

Replace:

    trial.suggest_categorical("feature_set", list(data["feature_sets"]))

with:

    n_features = trial.suggest_int("n_features", 1, len(data["columns"]))

For every CV fold, use that fold's own training-only ranking to produce the subset of size K. Store:

- n_features;
- selected_features_by_fold;
- selected_indices_by_fold;
- ranking/cluster report references;
- feature-selection version/hash.

Do not store one global feature list as though the selected identities were identical in all folds.

The final export flow must:

1. fit the selector on development people only;
2. derive the final development subset at the selected K;
3. train the pre-holdout candidate with that subset;
4. export and replay the candidate on the holdout exactly once;
5. after the holdout report is frozen, refit the selector on all eligible people at the locked K;
6. train and export the final production candidate;
7. store the exact final selected feature names in the C feature schema and model report.

The all-feature K=150 candidate must remain available for direct comparison.

### Cache and reproducibility

Cache the expensive fold rankings because they do not change per Optuna trial. The cache key must include:

- source dataset content hash;
- label/event hash;
- feature schema and column order;
- frame/readiness gate;
- task and label polarity;
- person-fold assignments;
- ranker code/version;
- conditional sampler and correlation-clustering version;
- model screening configuration and seed.

A mismatch must invalidate the ranking cache. Never reuse a ranking produced using a different labeler, data split, feature schema or firmware gate.

## 11. Execution phases

| Phase | Work | Deliverable | Gate |
| --- | --- | --- | --- |
| 0 | Freeze latest commit, Optuna DBs, current headers, labels and dataset fingerprints | immutable baseline manifest | Existing studies untouched |
| 1 | Confirm actual firmware Lean-8 readiness and make train/replay gate identical | readiness contract + test | Firmware/data gate parity |
| 2 | Remove static feature-set menu in a new experiment branch; create the all-150 candidate contract | feature schema tests | Every available feature eligible |
| 3 | Implement fold-local nested ranker and correlation-aware block/conditional importance | feature_selection.py + unit tests | No outer-fold/holdout leakage |
| 4 | Implement integer K search 1–150 and fold-specific selected subsets | new Optuna study | K=150 reproduces all-feature baseline |
| 5 | Add stability/resampling reports and correlated-group diagnostics | selection CSV/JSON/MD | Reproducible rankings |
| 6 | Full CV search for TAP and LP independently | resumable study DBs | Exact replay objective and model budgets |
| 7 | Select using development CV only; evaluate the locked pipeline on holdout once | held-out C replay report | No holdout-based candidate switching |
| 8 | Refit selector/model on all eligible people; export stable C headers | final model + feature schema | C compile, parity and size checks |
| 9 | Real-world A/B test against current production headers | per-person/per-gesture report | Improved sensitivity without unacceptable false triggers |

### Branch and study safety

- Do not overwrite results_unified_v2, its SQLite studies, or current headers.
- Create a new branch for implementation.
- Use a new output directory, for example research/gesture/results_unified_v3_feature_selection.
- Use new versioned study names, e.g. unified_v3_tap_fs... and unified_v3_lp_fs..._rel0.50.
- Do not resume the old study after changing feature_set from a categorical string to a dynamic integer K. That is a different Optuna search space.
- Keep the old V2 study, all completed trials, baselines and model headers read-only for comparisons.

## 12. Required tests and acceptance criteria

### Data/selection correctness

- The candidate pool contains every expected available firmware field exactly once.
- No family allow-list, deny-list or forced feature is present.
- Feature order is deterministic and recorded.
- Fold rankings use only the fold's training people.
- Altering validation/holdout labels cannot alter the corresponding training-fold ranking.
- Correlation clusters are derived from training data only.
- Block perturbations preserve recording boundaries and temporal block structure.
- Unknown labels remain excluded.
- No labels, timestamps, PointerState, subject, folder, hold_ms, contact labels or event metadata enter the model feature matrix.

### Baseline equivalence

- K=150, with identical trial parameters, labels, training rows, weights, feature order and seeds, must reproduce the current all-available model/replay behavior.
- The new code must match the old all-feature baseline before any selected-K result is interpreted.
- Compare training matrix hashes, weights, model bytes, predictions, TAP decisions, LP Q20 sequences and replay event counts.

### Selection correctness

- K is allowed to be any integer 1–150.
- K=150 is not excluded or penalized.
- Selected features are retrained from scratch; no feature is removed only from a post-hoc importance report.
- Fold-specific selected feature identities are stored.
- The final exported header uses exactly the final refitted selector's feature list.
- The selected subset must not be chosen by inspecting the holdout.

### Product acceptance

- Compare selected K against K=150 on the same held-out people.
- Report per-person and per-folder results, not only the pooled mean.
- Confirm TAP sensitivity at the fixed 0.85 threshold and false-event rate.
- Confirm LP behavior at the 0.5 ten-sample Q20 threshold.
- Confirm exact 32/24 KiB limits, C compilation, C/Python parity and firmware feature-readiness parity.
- Run a real-world A/B test with the current production header as the control.

**No subset is automatically better because it is smaller.** If the selected model does not improve event performance or measured deployment cost, keep the all-feature model.

## 13. Expected outputs

Under the new experiment directory:

    feature_selection/
      tap/
        fold_0_ranking.csv
        fold_1_ranking.csv
        fold_2_ranking.csv
        correlation_clusters.json
        selection_stability.json
        selected_features_by_trial.json
        final_selected_features.json
        report.md
      lp/
        same reports, independently learned

    tap/
      optuna_study.sqlite3
      models/
        TAP_SWIPE.h
        TAP_SWIPE_feature_schema.json
        TAP_SWIPE_model_info.json
      COMPARISON.md

    lp/
      optuna_study.sqlite3
      models/
        LONG_PRESS.h
        LONG_PRESS_feature_schema.json
        LONG_PRESS_model_info.json
      COMPARISON.md

    FEATURE_SELECTION_SUMMARY.md
    run_manifest.json

The report must show:

- all 150 features and their learned ranks;
- individual and correlation-group importance;
- selection frequency/stability;
- K selected by the development study;
- exact feature names in each fold and in the final model;
- K=150 baseline versus selected-K metrics;
- exact model bytes and firmware replay results;
- holdout results, clearly labeled as a one-time independent evaluation.

## 14. Research basis

These references support the methodology. They do **not** imply that any single selector is universally best.

1. **Nogueira, Sechidis & Brown (2018), “On the Stability of Feature Selection Algorithms,” Journal of Machine Learning Research.** Provides a rigorous feature-selection stability measure and statistical treatment of stability.  
   https://www.jmlr.org/papers/v18/17-514.html

2. **Cawley & Talbot (2010), “On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation,” JMLR.** Explains why repeatedly selecting models/features on a finite validation criterion can overfit that criterion, motivating a truly independent outer evaluation.  
   https://www.jmlr.org/papers/v11/cawley10a.html

3. **Strobl et al. (2008), “Conditional variable importance for random forests,” BMC Bioinformatics.** Shows why correlated predictors can distort ordinary variable-importance measures and develops conditional permutation importance.  
   https://doi.org/10.1186/1471-2105-9-307

4. **scikit-learn documentation, “Permutation feature importance.”** Explains held-out permutation importance, its dependence on the scoring metric, the limitations of impurity importance, and why correlated features can make single-column permutation misleading.  
   https://scikit-learn.org/stable/modules/permutation_importance.html

5. **Consensus features nested cross-validation (2020), Bioinformatics.** Discusses feature stability within nested CV and using consensus across inner folds.  
   https://doi.org/10.1093/bioinformatics/btaa046

6. **Hamer & Dupont (2021), “An Importance Weighted Feature Selection Stability Measure,” JMLR.** Discusses stability measures that account for feature importance and correlated feature spaces.  
   https://www.jmlr.org/papers/v22/20-366.html

---

## Final decision

Implement **all-150, task-specific, fold-local feature selection**, not another feature-family ablation.

The selector should learn feature relevance from person-held-out cross-fitting, account for correlation and temporal structure, and let exact firmware replay choose K. Stability is measured, not enforced by an arbitrary feature rule. The 22% person holdout remains untouched until the selected pipeline is locked.

The first implementation milestone is not training a new model: it is proving that the selector cannot see validation/holdout information and that K=150 reproduces the existing all-feature baseline. Only after those checks pass should the new Optuna study begin.
