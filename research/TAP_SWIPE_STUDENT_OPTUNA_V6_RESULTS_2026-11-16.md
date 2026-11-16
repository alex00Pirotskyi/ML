# Mudra V6 — full 40-trial TAP+SWIPE Optuna student results

**2026-11-16 — real-data execution completed and code exported.**  
Full training CI: [Actions run #43861744110](https://github.com/alex00Pirotskyi/ML/actions/runs/43861744110) (**success**).  
Full model + study artifact: [download ten C headers and JSON reports](https://github.com/alex00Pirotskyi/ML/actions/runs/43861744110/artifacts/14223876591).  
Git model branch: [research/tap-swipe-optuna-v6-run-43861744110](https://github.com/alex00Pirotskyi/ML/tree/research/tap-swipe-optuna-v6-run-43861744110/research/tap_real/exports_v6/43861744110).  
Training source: [train_optuna_student_v6.py](https://github.com/alex00Pirotskyi/ML/blob/main/research/tap_real/train_optuna_student_v6.py).  
Exporter: [export_plus.py](https://github.com/alex00Pirotskyi/ML/blob/main/export_plus.py).

## Experiment contract

Trained directly against original DATA/DATA_TAP raw, causal firmware features and separately versioned V5 OOF silver labels. All source SHA-256, row count and timestamps were verified against the V5 export. Dataset: **2,790 unique recordings**, with **117,132 sampled known-labeled frames** from the complete 2,790-file corpus; **98,994 training frames** and **18,138 outer held-out frames**, split by normalized participant strings.

TAP and SWIPE are jointly positive. FINGER, WRIST, ARM, BAND, NOTHING are negative. Unknown V5 rows are never supervised. This is a combined binary gesture detector; it does not identify whether an event is a TAP or SWIPE.

Optuna performed **40 successful trials**, each with three subject-group validation folds. It optimized exactly **two objectives**: maximize confidence-weighted F1 versus V5 silver labels, and minimize the exact generated export_plus static payload bytes. The model search covered raw PSD feature groups (15 EMG, optional WL and ACC), leaf/depth/number of trees, learning rate, regularization, sampling, class weighting and decision threshold. Ten distinct Pareto/quality-size candidates were selected and **all ten** were exported as compact C headers and matching JSON files. Every export passed export_plus's Python parity representation validation and GCC C compilation with a synthetic t_core stub; no real nRF52832 firmware build was performed.

## Top ten actual model exports

| Rank | Optuna trial | 3-fold weighted silver F1 | Subject-heldout silver F1 | Subject-heldout AP | Static model bytes | Firmware features |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23 | **0.8042** | **0.7540** | **0.7717** | 11,524 | 36 |
| 2 | 9 | 0.6988 | 0.7118 | 0.5216 | **1,084** | 18 |
| 3 | 29 | 0.7745 | 0.7345 | 0.6766 | **1,084** | 36 |
| 4 | 31 | 0.7875 | 0.7406 | 0.6961 | 2,004 | 36 |
| 5 | 37 | 0.7917 | 0.7390 | 0.7460 | 3,644 | 36 |
| 6 | 5 | 0.7956 | 0.7460 | 0.7204 | 3,844 | 36 |
| 7 | 26 | 0.7997 | 0.7482 | 0.7326 | 6,568 | 36 |
| 8 | 24 | 0.8017 | 0.7484 | 0.7507 | 7,524 | 36 |
| 9 | 39 | 0.7852 | 0.7369 | 0.6880 | 2,004 | 36 |
| 10 | 32 | 0.7692 | 0.7305 | 0.6637 | **1,084** | 36 |

**Rank 1** maximizes Optuna F1, but uses about 11.5 KB in generated static arrays. **Rank 3** is a more compact trial, about 1.08 KB static data, with approximately 0.735 heldout silver-label F1. These ranks are tradeoffs rather than production-approved selections. Note all above are based on automatic V5 temporal pseudo labels, not independently annotated physical gestures.

The actual final static size can differ from the fold-estimated trial size; each row above uses the **refitted model's** exact export report. The human-readable C header text is larger than flash static arrays. Actual firmware flash also contains executable instructions, literal pools, alignment and linker overhead.

## Critical hard-negative finding — do not deploy yet

Despite moderate silver-label F1, the candidate classifiers still react too frequently to other muscle gestures on the outer heldout sampled data:

| Category / metric | Rank 1 | Rank 3 | Rank 5 |
|---|---:|---:|---:|
| FINGER frame false-positive fraction | 0.631 | 0.712 | **0.594** |
| WRIST frame false-positive fraction | 0.458 | 0.531 | **0.402** |
| NOTHING frame false-positive fraction | 0.956 | **0.044** | 0.244 |
| SOFT TAP silver-positive-frame recall | **0.891** | 0.833 | 0.822 |
| TAP silver-positive-frame recall | **0.880** | 0.862 | 0.827 |
| SWIPE silver-positive-frame recall | **0.889** | 0.882 | 0.812 |

These fractions are computed on **subsampled, labeled test frames**, not false activations per real-world hour. NOTHING has only a handful of recordings, so that estimate has high uncertainty. Nevertheless **FINGER confusion is much too high to infer field reliability from F1 alone**. A firmware trigger must be tested with a true event-level reference, real-time smoothing/refractory logic and hard daily-use negatives. The full per-model JSON reports preserve confidence-weighted and unweighted metrics, subgroup statistics, TP/FP/TN/FN, ROC AUC, PR AUC, MCC, kappa, Brier, ECE, export parity, feature importances and precise static size.

## Validation and Git details

- All 40 full-data Optuna trials completed; no failed trial.
- Ten exact, distinct header/model JSON pairs, ten feature schemas and ten Python pickles exported.
- All 10 headers independently re-read, matched to SHA-256 recorded in their JSON and compiled using a minimal C stub t_core.
- The trainer uses 15 EMG PSD zone inputs and optional 3 WL and 18 ACC zone inputs, with no Oculus/pointer, metronome, filename, class label or post-event teacher features in the inference inputs.
- Data split uses participant-string grouping; matching aliases may still be imperfect. Outer holdout measures silver-label agreement on the **pre-refit** model. Final exported models are refitted on all sampled labeled subjects; that distinction is recorded.
- All generated metrics, hyperparameter trials, SQLite Optuna study, source-binding manifests and C compilation logs are available in the GitHub Actions artifact. Header+JSON model files and summary are also committed on a **versioned Git branch**, not on main.

**Release decision:** the Optuna training and export pipeline is implemented, executed and reproducible; ten firmware candidate models exist. The models are **research candidates**, not production-certified tap/swipe models, principally because the held-out FINGER and other hard-negative responses remain high.
