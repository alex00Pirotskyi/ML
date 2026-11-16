# Mudra LP labeling — final no-PSD-bin-mean data audit and model evaluation

**Date:** 2026-11-16  
**Input:** DATA/DATA_LONG_PRESS at data commit 36f02ac757166ba6dee848385ce70aa2f276a7d4  
**Teacher:** lp_causal_fw_teacher.py, version lp_causal_fw_v2  
**Code branch:** research/lp-no-psd-mean-20261116  
**Important:** All classifier and regression numbers below are **imitation of pseudo-labels**, NOT verified muscle rest, actual physical finger press/release, or physical electrode-contact correctness.

## 1. Executive verdict

1. Removing the compile-time EMG_PSD_MEAN and ACC_PSD_MEAN blocks is consistent with firmware already recording with both macros set to 0. Those blocks were never executed under that configuration.
2. The entire new long-press dataset was checked, **2,431 of 2,431 decompressed CSV files byte-for-byte identical** to the preceding commit. The new commit recompressed the files, but did **not** alter their physiological data.
3. The Python teacher uses the existing encoded log-domain PSD-zone features without zone-bin division, which matches the recorded configuration. Do not apply the legacy psd_mean.py conversion.
4. Three-fold person-held-out LightGBM on the balanced 475-file benchmark scored **85.75% balanced accuracy**, **92.65% accuracy**, **rest F1 0.710**, and **ROC AUC 0.958**, exactly matching the earlier same-code benchmark.
5. Activity evidence separated long-press recording sessions from quiet connected sessions with **AUC 0.987** and **92.19% fixed-threshold balanced accuracy**, but this is session-level context testing, not real press detection.
6. Contact/disconnection evidence was markedly weaker: **AUC 0.762, balanced accuracy 71.43%** using provisional file/channel contact proxies. It missed **48/84** provisional disconnected-channel cases. A model imitating those contact scores can get a high R² without physically identifying disconnection.
7. Strong-LP rest remains a serious weakness: only **17.0% of the teacher's rest windows** recovered by the student in strong LP files, while the teacher marked only **~5.4%** of strong LP rows as confidently resting.

## 2. Full data provenance verification

User's commit message: DATA NO PSD MEAN. SHA: 36f02ac757166ba6dee848385ce70aa2f276a7d4. The prior data state is 2e14d95252b2ef94fdd1c8c92c643220ff432855.

We compared decompress(old gzip CSV) against decompress(new gzip CSV) for **every** DATA_LONG_PRESS recording:

| Verification | Result |
|---|---:|
| Input compressed files | 2,431 |
| Byte-identical decompressed CSV contents | **2,431** |
| Different decompressed CSV contents | **0** |
| Errors | **0** |
| Independently sampled compressed files with changed gzip bytes | 36/36 |
| Independently sampled decompressed CSVs still identical | 36/36 |

Complete audit: https://github.com/alex00Pirotskyi/ML/actions/runs/43821967044

All CSV headers across the full set were inspected. Column schema variations:

| Columns | Recordings |
|---|---:|
| 134 | 1,506 |
| 135 | 793 |
| 139 | 132 |
| Total | **2,431** |

Every folder includes 3 waveform-length channels, 15 emg_psd_mean channel/zone fields, and 30 EMG PSD history fields. No header contains the legacy psd_mean_applied marker. The present data are consistent with the user's raw-acquisition mode, but numerical provenance cannot in general be proven from missing markers alone; this study also established that they are the **same CSV contents as before**.

### Firmware arithmetic: two different means

The FW macro EMG_PSD_MEAN=0 (and ACC_PSD_MEAN=0) bypasses FFT-zone count division. The PSD feature is the quantized log of a sum of squared FFT magnitudes:

    zone_log = uint16(round(10000 * log10(1 + sum(FFT_magnitude_squared_in_zone))))

The separately computed emg_psd_mean feature is an arithmetic **temporal mean of 15 encoded log-zone history values**. That temporal mean remains in the data even though FFT-bin averaging was disabled. Commenting the disabled conditional blocks does not remove or change this temporal operation.

The Python psd_mean.py postprocessing script divides the already-encoded log-zone number by bin counts. That division is NOT equivalent to a mean of pre-log linear FFT powers. The current v2 teacher no longer requires this postprocessing. At the time of this audit psd_mean.py remains in the repository as a legacy script; avoid invoking it in the training pipeline.

## 3. Pseudo-label definitions and important caveats

For each channel CH1/CH2/CH3, the teacher produces:
- lp_resting_ch1..3
- lp_high_activity_ch1..3
- lp_disconnection_ch1..3
- lp_unknown_ch1..3

These are 0..1 heuristic scores, **not calibrated probabilities** and not independently measured contact truth. The four outputs within each channel form a soft partition.

The binary lp_state=1 only when **all 3 channels appear calibrated, connected and resting**, with a resting dwell; otherwise lp_state=0. Zero combines high muscle activity, suspected contact fault, and insufficient calibration/UNKNOWN. Therefore **zero is not equivalent to verified physical pressing or even verified active muscle**.

lp_state_weight should be used to downweight or discard uncertain teacher labels. PointerState, TriggerPressure and class_id are NOT treated as supervision or model inputs. This matters because pointer state is unreliable and electrode quality may vary within a single recording.

## 4. Balanced held-out-participant experiment

Training/evaluation:
- 475 source files: all 275 non-TAP/SWIPE files plus 100 TAP and 100 SWIPE files.
- 80 evenly distributed samples per file: **38,000 rows**.
- 57 normalized User identifiers and 3-fold GroupKFold, so files from the same normalized subject identifier are held out together.
- Fixed LightGBM configuration, 200 trees, max depth 8, 31 leaves.
- 21 raw features; 44 causal-history features; 59 explicit noncausal/oracle features.
- No subject, directory, class_id or pointer-state input.
- Sample weights from teacher lp_state_weight; threshold fixed at 0.5.

Model results against **V2 generated lp_state**, not physical ground truth:

| Model | Accuracy | Balanced accuracy | Rest F1 | Rest recall | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Majority predicts 0 | 88.27% | 50.00% | 0.000 | 0.00% | 0.448 |
| LightGBM raw only | 85.00% | 72.16% | 0.464 | 55.38% | 0.847 |
| Logistic causal history | 82.17% | 80.94% | 0.511 | 79.34% | 0.879 |
| **LightGBM causal history** | **92.65%** | **85.75%** | **0.710** | **76.74%** | **0.958** |
| LightGBM whole-file oracle, NONDEPLOYABLE | 93.78% | 87.97% | 0.752 | 80.37% | 0.971 |

Causal LightGBM rest precision **66.07%**, nonrest specificity **94.76%**; teacher 0 vs 1 confusion matrix:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Teacher 0 | 31,785 | 1,757 |
| Teacher 1 | 1,037 | 3,421 |

These scores exactly reproduce the earlier benchmark (run 43816669080), which is expected because the underlying decompressed recordings are unchanged.

### Hard condition: strong long press

| File context | Teacher rest fraction | Student recall of teacher-rest |
|---|---:|---:|
| CONNECTED | 60.22% | 88.08% |
| SOFT_LONG_PRESS | 24.76% | 69.00% |
| MID_LONG_PRESS | 9.21% | 62.91% |
| **STRONG_LONG_PRESS** | **5.88%** | **17.02%** |
| DISCONNECTED | 3.45% | 50.86% |

The old challenge remains. The teacher itself may label genuine rests as active/unknown, and the student also misses many of the sparse teacher-rest intervals in strong contractions. Neither can be evaluated for correct physical release latency without independent release timestamps.

### Nine regression targets: one held-out fold

| Teacher target | CH1 MAE / R² | CH2 MAE / R² | CH3 MAE / R² |
|---|---:|---:|---:|
| Resting | 0.145 / 0.702 | 0.154 / 0.606 | 0.148 / 0.697 |
| High activity | 0.160 / 0.280 | 0.146 / 0.525 | 0.141 / 0.160 |
| Disconnection | 0.0111 / 0.917 | 0.0072 / 0.943 | 0.0191 / 0.787 |

R² and MAE evaluate **student ability to reconstruct teacher soft scores**. High-activity regression is weak especially for CH3. Disconnection targets are sparse, so impressive R²/low MAE should NOT be mistaken for correct detection of real electrode disconnection.

Evidence: https://github.com/alex00Pirotskyi/ML/actions/runs/43821234257 (job quick, uploaded JSON).

## 5. Teacher sanity testing against weak recording contexts

These tests are independent of the student but still rely on imperfect session labels. CH3 in DISCONNECTED recordings is deliberately excluded from a disconnected-positive label.

**Contact proxy:** 117 connected-reference channel examples and 84 provisionally disconnected CH1/2 examples.

| Metric | Result |
|---|---:|
| ROC AUC | **0.762** |
| Balanced accuracy at disconnection score >=0.5 | **71.43%** |
| Correctly unflagged connected examples | 117/117 |
| Flagged provisional disconnected examples | 36/84 |
| Missed provisional disconnected examples | **48/84** |

**Activity/session proxy:** For 101 LP and 39 CONNECTED sessions, use the max across channels of the recording's P90 activity score (after a 5-second initial period). This is retrospective session classification, NOT real-time LP-event classification.

| Metric | Result |
|---|---:|
| ROC AUC | **0.987** |
| Balanced accuracy at >=0.5 | **92.19%** |
| LP sessions with clear activity evidence | 93/101 |
| Connected sessions incorrectly flagged | 3/39 |

Mean teacher-positive lp_state resting fractions: CONNECTED 60.1%; NOTHING 69.0%; SOFT LP 24.6%; MID LP 9.2%; STRONG LP 5.4%; DISCONNECTED 3.4%; LOOSE 9.6%; TABLE 0%. These are output distributions, not independent resting-time proportions.

Evidence: https://github.com/alex00Pirotskyi/ML/actions/runs/43821738519

## 6. Scientific interpretation

**Adaptive resting baselines:** EMG onset literature has used single, double and adaptive threshold algorithms; thresholds are sensitive to signal-to-noise ratio, preprocessing and dwell. No single universally valid threshold exists. A causal per-channel baseline is justified, but its calibration must be verified rather than inferred solely from a low percentile or quiet time. Reference: Carvalho et al. (2023), Journal of NeuroEngineering and Rehabilitation: https://link.springer.com/article/10.1186/s12984-023-01268-8

**Electrode artifacts:** Poor electrode contact, placement, cable/electrode motion, power-line interference and baseline wander can create both high apparent activity and distorted low-energy signals. High stationary WL is a useful marker of some severe faults, not a complete disconnection test. Reference: Reducing Noise, Artifacts and Interference in Single-Channel EMG Signals (2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10059683/

**Subject and placement variation:** Surface EMG changes with electrode position, muscle anatomy and the skin-electrode interface. Train and evaluate by participant/device/session, not random rows from the same recording. Reference: Analysis and Biophysics of Surface EMG (2020): https://pmc.ncbi.nlm.nih.gov/articles/PMC7594523

## 7. Engineering recommendation

1. **Keep current FW FFT-zone bin averaging disabled.** There is no expected feature change from commenting out code that was under macros set to 0.
2. **Stop running psd_mean.py** on future raw recordings. It is legacy post-log scaling, not a physically correct frontend change.
3. **Retain emg_psd_mean temporal history for this model** until a separately measured latency/accuracy experiment shows whether instantaneous PSD history slot 0 performs better.
4. **Treat disconnection output as severe-artifact evidence**, not definite electrode contact. Weak-context test sensitivity is insufficient for full contact-state supervision.
5. **Keep UNKNOWN and train with lp_state_weight** (or exclude low-confidence zeros). Unknown should not be coerced into true muscle activity.
6. **Focus next experiment on soft vs strong held contractions with independently measured release timing**, plus controlled per-channel electrode lift-off, a verified quiet baseline, and startup-active trials.
7. Revalidate under held-out people, hands, devices and acquisition days. Report false-REST-during-hold, false-ACTIVE-at-rest, false disconnected while connected, time-to-rest after release, and abstention rate.

## 8. Reproducible research scripts

- Complete decompressed data-identity audit: https://github.com/alex00Pirotskyi/ML/actions/runs/43821967044
- Full CSV schema/provenance audit: https://github.com/alex00Pirotskyi/ML/actions/runs/43821003142
- Balanced classification and regression: https://github.com/alex00Pirotskyi/ML/actions/runs/43821234257
- Weak-context activity/contact tests: https://github.com/alex00Pirotskyi/ML/actions/runs/43821738519
- Existing teacher: https://github.com/alex00Pirotskyi/ML/blob/main/lp_causal_fw_teacher.py
- Existing student: https://github.com/alex00Pirotskyi/ML/blob/main/scripts/lp_classifier_benchmark_v2.py
- Research scripts: https://github.com/alex00Pirotskyi/ML/tree/research/lp-no-psd-mean-20261116/scripts

These investigations are read-only and do not change data, existing firmware or production model weights. The research lives on a separate branch.
