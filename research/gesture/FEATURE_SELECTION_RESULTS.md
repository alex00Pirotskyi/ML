# Feature selection, readiness gate and TAP sensitivity: research results and integration (2026-11-18)

This document implements and reviews `FEATURE_SELECTION_RESEARCH_AND_EXECUTION_PLAN.md`. All numbers come
from the full dataset: 5,352 recordings, 150 firmware fields, frame-exact Lean-8. Only **development
people** were used; the 22% person holdout was never touched.

Reproduce with:

- `research/gesture/fs_research.py`: `gate`, `rank`, `kcurve` and `report` modes.
- `research/gesture/ablate_lean8.py`.
- `research/gesture/bench_unified.py`: equivalence checks.

## Answers

1. **Why the latest TAP model is insensitive.** The TAP objective scored models as
   `recall × min(1, 1 / false events per minute)`. At the firmware threshold 0.85, every TAP model
   catches 94–99% of taps but fires 4–11 false events per minute on the hard negative recordings. The
   search could only reach 1 per minute with models that rarely exceed 0.85, which miss 25–50% of taps.
   v3 sets this budget from the previous production TAP header (§1).
2. **Learned feature selection:**
   - **TAP:** no gain. K = 150 is best on average and at least as good as any subset in every fold. The
     learned ranking is still far better than random subsets of the same size.
   - **LP:** some subsets (K ≈ 10–100) are up to +0.03 better on average, but only in 2 of 3 folds,
     within fold noise.

   Optuna keeps K in its search with K = 150 as a candidate, so it can only help (§3, §4).
3. **Readiness gate.** v3 scores and ranks only the rows the firmware runs on (frame 500). Training still
   uses rows from frame 50: on firmware-gated rows TAP is +0.07 better with them, and LP is neutral (§2).
4. **Holdout.** It no longer influences which model becomes the default. It is reported once, before
   anything is fitted on those people.

## 1. TAP sensitivity is set by the false-event budget, not by the features

These are the all-feature (K = 150) models of the K-curve: mean over 3 folds of unseen people, on
steady-state rows (frame ≥ 500).

| Operating point | Taps caught | False events/min |
|---|---|---|
| firmware threshold 0.85 | 96.9% | 4.8 |
| best point under a 1/min budget | 72.4% | ≤ 1 |
| under a 2/min budget | 91.3% | ≤ 2 |
| under a 3/min budget | 94.1% | ≤ 3 |
| under a 5/min budget | 96.1% | ≤ 5 |

At 0.85, fold 0's false events per minute by recording type:

| Recording type | False events/min |
|---|---|
| BAND | 17.2 |
| FINGER | 12.8 |
| WRIST | 7.0 |
| SWIPE | 6.9 |
| TAP | 2.2 |
| SOFT_LONG_PRESS | 1.9 |
| ARM | 1.9 |
| NOTHING | 0.6 |

These are deliberate hand movements, the hardest negatives. Under the v2 budget of 1 per minute, a model
that gives up a quarter of the taps scores up to five times higher than one that catches them all. The
search followed that, which is the insensitive model you saw.

v3 replays the previous production TAP header on the development people (same rows, same rules) and uses
its false-event rate as the budget. The new model must catch more taps at no more false alarms than the
model in the field. If that header is missing, v3 falls back to 1.0 and says so; set
`MUDRA_TAP_FALSE_PER_MIN_TARGET` to a number in that case.

## 2. Readiness gate

The firmware runs the models only after `lean_8_features_ready()`, i.e. 500 EMG frames, about 10.8 s.

| Rows from | Labelled taps kept | Holds kept |
|---|---|---|
| frame 50 | 93% | 97% |
| frame 500 | 14% | 73% |

Each cell is a mean over 3 folds, scored **only on steady-state rows** (what the device sees):

| Model | Train from frame 50 | Train from frame 500 |
|---|---|---|
| TAP, all 150 fields | **0.765** (0.908 / 0.798 / 0.589) | 0.694 (0.915 / 0.788 / 0.379) |
| TAP, raw 126 | 0.678 | 0.669 |
| LP, all 150 fields | 0.561 | 0.544 |
| LP, raw 126 | 0.543 | 0.556 |

TAP also catches 91.3% of taps at ≤ 2/min when trained from frame 50, against 83.3% when trained from
frame 500. The frame-500 training set has 5–7× fewer labelled taps.

v3 therefore uses:

- **Evaluation gate = firmware gate** (`MUDRA_EVAL_MIN_FRAMES=500`): scoring, ranking and selection.
- **Training gate 50** (`MUDRA_TRAIN_MIN_FRAMES=50`): warm-up rows serve as extra training examples
  only, never as rows to be judged on.

If the firmware gate changes, set `MUDRA_EVAL_MIN_FRAMES` to match.

## 3. Learned rankings (fold-local, block permutation on firmware replay)

**Cluster cut.**

| Linkage, cut | Clusters | Largest | Min \|ρ\| inside a cluster | Silhouette |
|---|---|---|---|---|
| average, 0.55 (silhouette-best over 0.05–0.6) | 17 | 54 | – | 0.466 |
| complete, 0.50 (silhouette-best over 0.05–0.6) | 24 | 54 | 0.51 | 0.481 |
| **complete, 0.25 (silhouette-best over \|ρ\| ≥ 0.7)** | 52 | 11 | 0.75 | 0.328 |

A 54-field block at |ρ| 0.51 is a sensor family, not a group of interchangeable fields. The rankings
used 52–66 clusters per fold.

**TAP.** The strongest correlated groups in every fold:

- **Instantaneous ACC spectrum zones** (`acc_psd[0]` X/Y/Z): shifting one group jointly costs 0.2–0.4
  of replay score. This is the ACC burst found in the original tap-label research.
- **Instant EMG spectrum on CH1** (0.22–0.30).
- **Lean-8 `rise10`** (activity above the 10-s floor) at ranks 7–12 in every fold.

**LP.**

- **Instant EMG spectrum on CH2** (`emg_psd[0][CH2][0..3]`), joint importance 0.16–0.33.
- **CH1 EMG.**
- **ACC Z zone 0 / history.**
- **`rise10` on CH1/CH2.**
- **`wl_head`.**

**Stability (Nogueira Φ) of the top-K sets across the 3 fold rankings:**

| Task | K=5 | K=10 | K=20 | K=40 | K=60 |
|---|---|---|---|---|---|
| TAP | 0.38 | 0.29 | 0.31 | 0.22 | 0.19 |
| LP | 0.59 | 0.43 | 0.35 | 0.30 | 0.35 |

The same groups lead in every fold, while their interchangeable members swap places. That is why the
joint group measure is needed.

Note: grouped K-fold balances rows, not people, so two inner splits (TAP folds 1 and 2) held a single
person. Their importances are noisy; the other splits held 7–14 people.

## 4. K-curves: retrained models from each fold's own ranking

Fixed configuration: 31 leaves, depth 8, ≤ 32 KB. Scores are the mean best replay score over the 3 folds;
"vs K=150" is per fold.

**TAP:**

| Subset | Score | vs K=150 per fold | Steady-state | Taps caught at ≤ 1 / ≤ 2 false/min |
|---|---|---|---|---|
| K5 | 0.186 | −0.79 / −0.30 / −0.35 | 0.251 | 0.15 / 0.15 |
| K20 | 0.494 | −0.26 / −0.06 / −0.19 | 0.625 | 0.45 / 0.69 |
| random 20 | 0.376 | −0.41 / −0.24 / −0.21 | 0.486 | 0.33 / 0.54 |
| K40 | 0.559 | −0.21 / −0.04 / −0.07 | 0.684 | 0.52 / 0.77 |
| random 40 | 0.494 | −0.23 / −0.05 / −0.22 | 0.628 | 0.45 / 0.70 |
| K60 | 0.598 | −0.12 / +0.00 / −0.08 | 0.700 | 0.53 / 0.78 |
| K100 | 0.591 | −0.14 / −0.07 / −0.01 | 0.707 | 0.55 / 0.84 |
| **K150** | **0.663** | 0 | **0.765** | **0.63 / 0.82** |

**LP:**

| Subset | Score | vs K=150 per fold | Steady-state | At release threshold 0.5 |
|---|---|---|---|---|
| K5 | 0.491 | −0.13 / −0.08 / −0.06 | 0.475 | 0.446 |
| K10 | 0.594 | −0.02 / +0.01 / +0.05 | 0.570 | 0.560 |
| K20 | 0.567 | −0.07 / +0.00 / +0.03 | 0.531 | 0.520 |
| random 20 | 0.580 | −0.04 / +0.03 / +0.01 | 0.542 | 0.526 |
| K60 | 0.594 | −0.06 / +0.06 / +0.03 | 0.567 | 0.523 |
| K100 | 0.611 | −0.02 / +0.06 / +0.05 | 0.583 | 0.507 |
| K150 | 0.580 | 0 | 0.561 | 0.453 |

Reading:

- TAP needs (almost) all fields. Any ranking-driven cut loses, although the ranking beats random subsets
  by 0.06–0.12.
- LP is less sensitive to the subset. Mid-size subsets can be slightly better, but the gain is the same
  size as the fold spread, and for LP the ranking beats random only at some K.
- The large LP differences at the firmware threshold are mostly calibration under a fixed class weight,
  which the trainer's weight search adjusts.

Optuna's integer K, searched jointly with the other settings, is the right way to use this. It can find
the LP subsets if they hold up, and it keeps K = 150 for TAP.

Full tables, per-fold rankings and stability are in `results_fs_research/` (`report_tap.md`,
`report_lp.md`, JSON).


## How v3 is implemented (`feature_selection.py`, `train_unified.py`)

| Plan item | Implementation |
|---|---|
| All 150 fields eligible, no family rules | Candidate pool = `common_p_core()`. Families appear only as report labels. |
| Fold-local relevance (4.3) | One ranking per CV fold, fitted on that fold's training people. Inside them, an inner person-grouped cross-fit (3 splits): a screening LightGBM on all fields, replayed on the inner-validation people. |
| Correlation handled from data (4.4) | Spearman \|ρ\| on the fold's training rows, complete linkage, silhouette-chosen cut among strong-correlation cuts (\|ρ\| ≥ 0.7). |
| Block-aware perturbation (4.4) | A **within-recording circular shift** of a feature, or of a whole cluster jointly (same shift). Recording boundaries and autocorrelation are kept. The score is the firmware replay score at an operating point fixed from the unperturbed screening model. This is labelled block permutation importance (marginal), **not** conditional sampling. |
| Gain as cross-check (4.5) | LightGBM gain share stored next to every importance. |
| Integer K 1…150 (4.6) | `trial.suggest_int("n_features", 1, n_fields)`. Each fold uses its own top-K, in canonical column order. Both starting trials are K = 150. |
| Holdout = audit (5.2) | `promote()` uses CV fold scores only. `holdout_report.json` is written before any ranking or model is fitted on the held-out people. |
| Final export (10) | Ranking on the development people → pre-holdout candidate → holdout replay, once. Then ranking on everyone at the locked K → production model. |
| Stability (6) | Nogueira Φ of the top-K sets across the outer-fold rankings and across all inner-split importances. |
| Cache | Keyed on: data content hash, columns, gates, task, fold people, TAP target, ranker version and settings. |
| Readiness gate (8) | `MUDRA_EVAL_MIN_FRAMES` (rows scored and ranked; default 500 = `lean_8_features_ready()`) is separate from `MUDRA_TRAIN_MIN_FRAMES` (training examples). |
| New outputs / studies (11) | `results_unified_v3_feature_selection/`. Study names carry the ranker version, gates, TAP target or LP threshold. v2 is untouched (`MUDRA_EXPERIMENT=v2`). |

## Checks

- **K = 150 reproduces the all-feature model (plan 12):** v3 at K=150 against v2 `all_available`, same
  frozen trials: `ALL_EQUIVALENT`. Identical training bytes, models, 10,598 LP Q20 sequences and
  objective values; for TAP, identical training bytes, models, 4,802 TAP decision sequences and objective values.
- **No leakage:** flipping the labels of every validation and held-out person leaves a fold's ranking
  byte-identical (smoke test `FS_LEAKAGE_CHECK_PASSED`). Every ranking also records its people and is
  validated against the validation and held-out people.
- **Unit tests:**
  - Nogueira Φ is 1 for identical sets and −1 for disjoint halves.
  - The cluster finder recovers planted groups.
  - The ordering is a permutation, with the best member of each strong group first.
  - Top-K returns the canonical column order.
- **End-to-end v3 smoke on real recordings:** exports, C compile, parity and reports pass. CI runs it.

## Where this deviates from the plan, and why

1. **TAP objective (plan 7: "keep unchanged").** Changed: this is the measured cause of the insensitive
   TAP model (see above). The v3 default anchors the false-event budget on the previous production TAP
   header. A number can still be set.
2. **Conditional sampling (4.4).** Not implemented. Realistic conditional generators for 150
   autocorrelated, partly redundant sensor fields would be a study of their own. The plan's fallback is
   used and labelled: joint block shifts of mutually correlated groups. These answer the substitution
   question directly.
3. **Cluster cut.** Silhouette alone picks block-level partitions on this data, so the cut search is
   limited to strong correlation (table above).
4. **Floating add/drop/swap (4.7).** Not implemented, because of the K-curve: for TAP no subset beats K = 150 in any fold, and for LP the subset gains are inside fold noise. Add/drop/swap moves (each costing 3 inner retrains and replays per outer fold) would mostly optimise noise. Worth adding only if the server study's K search clearly and repeatedly prefers K < 150.
5. **TreeSHAP (4.5).** Not computed. Gain plus replay-based importance are recorded; SHAP would add cost,
   and its attribution splits across correlated features.
6. **Repeated person resampling for stability (6).** Stability uses the 3 outer-fold rankings and the 9
   inner-split importance vectors (person-disjoint resamples). More resampling multiplies ranking cost.
