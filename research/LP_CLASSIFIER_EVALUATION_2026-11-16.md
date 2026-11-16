# Long-press regression pseudo-labels: student classifier benchmark

**Executed:** 2026-11-16  
**Repo:** alex00Pirotskyi/ML  
**Dataset:** DATA/DATA_LONG_PRESS (all 2,431 .csv.gz files)  
**Student input:** WL and EMG PSD features only, without PointerState, TriggerPressure, class_id or User  
**Teacher:** lp_rest_labeler.py (lp_rest_teacher_v1)  
**Split:** 5-fold GroupKFold, normalized User ID, out-of-fold predictions  
**Evidence:** [successful full run](https://github.com/alex00Pirotskyi/ML/actions/runs/43798124489) and [reproducible script](../scripts/lp_classifier_benchmark.py)

> WARNING: THESE ARE PSEUDO-LABEL IMITATION METRICS. They do not measure true voluntary rest, pressing, physiological activity, or electrode-contact correctness. The teacher uses per-recording future information and lacks independent ground truth.

## 1. Dataset and evaluation

- **2,431 recordings processed; zero failed files**.
- **194,480 sampled rows**, up to 80 evenly distributed rows per recording.
- **64 grouped participant identifiers**; obvious variants beginning with ohad, nikol or shahar were merged. Other aliases might still remain; split is not perfect unique-human identity resolution.
- One sample per selected original row. Whole-file feature extraction was applied before row sampling.
- Teacher-positive global resting fraction **10.73%**; teacher-binary zero fraction **89.27%** (includes high activity, suspected disconnection AND unknown).
- Five-fold participants-held-out evaluation. Each participant was kept entirely on one side of the train/test boundary.
- The production branch, recording contents and labels were never modified by this benchmark.

### Classification inputs

- **Raw (21 features):** current WL on 3 channels, PSD total power on 3, normalized 5-zone PSD shape for each channel.
- **Causal history (44 features):** raw features plus trailing 180 ms WL/PSD medians, trailing ~2.2 s medians/variability, relative fast-to-slow ratios, elapsed time and observed sampling interval. These involve no future observations.
- **Oracle (59 features, NOT deployable):** the causal features **plus** whole-recording 15th-percentile floors, whole-recording channel dynamic ranges, and ratios to those floors. Oracle is included to expose how strongly pseudo-labels depend on non-causal reference information.
- LightGBM with 200 trees and fixed regularization; sample_weight derived from teacher's lp_state_weight, with a 0.1 floor. Classification at threshold 0.5, without test-fold threshold tuning.
- Logistic baseline uses the same causal inputs and weights. Majority baseline predicts zero from the training-class prevalence.

## 2. Binary lp_state results: all five held-out-person folds

| Model | Accuracy | Balanced accuracy | F1 (rest=1) | Precision (rest) | Recall (rest) | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority (predict non-rest) | 89.27% | 50.00% | 0.000 | 0.000 | 0.000 | 0.451 |
| LightGBM: raw only | 88.17% | 71.57% | 0.478 | 0.454 | 0.504 | 0.883 |
| Logistic: causal history | 88.85% | 76.09% | 0.535 | 0.484 | 0.598 | 0.902 |
| **LightGBM: causal history** | **92.57%** | **83.50%** | **0.675** | **0.636** | **0.720** | **0.953** |
| LightGBM: non-causal oracle | 95.79% | 94.50% | 0.826 | 0.743 | 0.929 | 0.989 |

Causal LightGBM also scored **84.09% teacher-confidence-weighted balanced accuracy** and **94.43% weighted accuracy**.

Do NOT report the oracle score as an achievable real-time classifier score; it observes whole-session facts that are unavailable at the time of inference.

### Confusion matrix: causal LightGBM

Rows are teacher state, columns are predicted state:

| | Pred 0 (non-rest) | Pred 1 (rest) |
|---|---:|---:|
| **Teacher 0** | 165,006 | 8,606 |
| **Teacher 1** | 5,853 | 15,015 |

Rest is the **positive class**. Thus rest recall = 71.95%, non-rest specificity = 95.04%. The model labeled 5,853 teacher-rest rows as non-rest and 8,606 teacher-nonrest rows as rest.

These are errors **relative to the pseudo-teacher**, not proven false presses or premature releases.

## 3. Breakdown by recording condition: causal LightGBM

| Recording category | Samples | Balanced accuracy | F1 rest | Recall rest | Specificity non-rest |
|---|---:|---:|---:|---:|---:|
| CONNECTED | 3,120 | 78.3% | 0.885 | 96.0% | 60.6% |
| DISCONNECTED | 3,360 | 73.6% | 0.589 | 47.5% | 99.7% |
| LOOSE | 3,360 | 72.9% | 0.471 | 51.7% | 94.0% |
| NOTHING | 400 | 68.4% | 0.793 | 97.3% | 39.5% |
| TABLE | 720 | 100.0%* | 0.000 | n/a* | 100.0% |
| MID_LONG_PRESS | 3,280 | 77.1% | 0.574 | 64.3% | 90.0% |
| SOFT_LONG_PRESS | 2,880 | 80.9% | 0.669 | 78.3% | 83.5% |
| **STRONG_LONG_PRESS** | **1,600** | **55.5%** | **0.200** | **12.0%** | **98.9%** |
| LONG_PRESS (contraction LP) | 320 | 85.4% | 0.784 | 72.5% | 98.2% |
| DRAG_AND_DROP | 2,960 | 84.8% | 0.743 | 75.8% | 93.8% |
| TAP | 98,480 | 82.5% | 0.669 | 70.3% | 94.8% |
| SWIPE | 74,000 | 82.2% | 0.623 | 68.2% | 96.3% |

*TABLE had no teacher-positive rest rows, so its nominal balanced accuracy of 100% is just a trivial all-nonrest result. Do not treat it as a two-class score.

**Most important model weakness:** On STRONG_LONG_PRESS, the model retrieves only **12.0% of teacher-rest windows**, even though it rarely mistakes teacher-nonrest for rest. The oracle model on the same category achieves **86.9% teacher-rest recall**, exposing a major reliance on per-recording baselines. This likely requires improving causal baseline estimation, not simply increasing tree count.

**Class-balance caution:** TAP and SWIPE together contribute 172,480 of 194,480 rows, so global metrics are dominated by those contexts. Per-condition results are essential.

## 4. Regression: nine continuous channel targets

Nine independent LightGBM regressors, trained on the **first held-out-person split only** (38,880 test samples, 12 held-out grouped users), using the 44 causal features.

| Pseudo-regression target | CH1 MAE / R² | CH2 MAE / R² | CH3 MAE / R² |
|---|---:|---:|---:|
| Resting | 0.123 / 0.792 | 0.145 / 0.707 | 0.137 / 0.726 |
| High activity | 0.125 / 0.790 | 0.148 / 0.709 | 0.149 / 0.673 |
| Disconnection evidence | 0.0044 / 0.944 | 0.0132 / 0.807 | 0.0190 / 0.624 |

Scores are on the teacher's 0–1 numerical scale; MAE is absolute prediction error. **Disconnection targets are very sparse** (~3% of test samples at >=0.5 for each channel), and their low MAE is therefore partly dominated by zero-label samples. These values cannot be interpreted as per-channel physical disconnection precision/recall.

Regression numbers above are **one held-out fold**, not five-fold cross-validated results.

## 5. Additional four-class per-channel classifier

Argmax over the four pseudo-state scores (REST, ACTIVE, DISCONNECTION, UNKNOWN) for each channel; first held-out fold, causal inputs.

| Channel | Accuracy | Macro F1 (4 classes) |
|---|---:|---:|
| CH1 | 89.68% | 0.687 |
| CH2 | 84.58% | 0.681 |
| CH3 | 85.41% | 0.591 |

WARNING: An uncalibrated multiclass argmax score hides per-class weaknesses. E.g. on CH3 the disconnection pseudo-class had 1,326 test examples but only 485 correctly classified (~36.6% recall). UNKNOWN was extremely sparse on CH1 and under-recovered. Four-class accuracy is NOT physical sensor-state accuracy.

## 6. Quick comparison: smaller 475-file sample

An independent three-fold test using all 275 non-tap/swipe recordings plus 100 TAP and 100 SWIPE (475 files, 38,000 rows, 57 participants) produced:

- **Causal LightGBM:** 90.07% accuracy, **81.49% balanced accuracy**, rest F1 0.689, ROC AUC 0.934.
- **Raw LightGBM:** 86.35% accuracy, 74.78% balanced accuracy.
- **Oracle LightGBM:** 96.04% accuracy, **94.73% balanced accuracy**.

See [quick-run logs](https://github.com/alex00Pirotskyi/ML/actions/runs/43798242869).

This cross-check shows the same gap between a causal student and a student with non-causal teacher-floor information; overall accuracy changes with group composition and class prevalence.

## 7. Interpretation and recommended next experiment

1. **Feasible:** A small LightGBM can learn a substantial portion of the teacher's rest/non-rest distinction using WL/PSD and short history, even for unseen participants.
2. **Not solved:** It cannot reproduce many resting periods in STRONG_LONG_PRESS sessions without file-wide low-energy baseline information.
3. **Classification ≠ truth:** Neither student nor teacher has independent physiological rest/press or per-electrode contact ground truth, so neither model's reported score is true LP-state accuracy.
4. **Uncertainty matters:** The teacher uses lp_state=0 for high activity, contact fault and UNKNOWN. Downweight low-confidence zeros via lp_state_weight, or train abstention explicitly rather than treating all zeros as verified active.
5. **Next engineering test:** Replace the offline per-file 15th-percentile baseline with a **causal, contact-gated, adaptive baseline** that can recover from startup-active recordings. Compare against this fixed 5-fold split and particularly report STRONG_LONG_PRESS rest recall, true no-press false-activation, and channel-level contact errors once independently annotated data exist.
6. **Do not deploy based on these figures:** A confident binary/resting model without sensor-quality truth can still label contact faults or sustained contractions incorrectly.

## 8. Reproduce

    git fetch origin
    git switch research/lp-classifier-eval
    python -m pip install numpy pandas scikit-learn lightgbm
    python scripts/lp_classifier_benchmark.py --input-dir DATA/DATA_LONG_PRESS --per-file 80 --folds 5 --out /tmp/lp_classifier_benchmark.json

Outputs are printed in the CI job log and saved as a JSON artifact associated with the successful [full benchmark run](https://github.com/alex00Pirotskyi/ML/actions/runs/43798124489).

The script reads source CSV.gz and generates pseudo-labels **in memory** via lp_rest_labeler.py. No input files or original labels were written or changed.
