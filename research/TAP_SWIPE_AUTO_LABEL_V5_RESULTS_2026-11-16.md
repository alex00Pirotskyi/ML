# Mudra TAP + SWIPE automatic labeler V5 — complete real-data research and release audit

**Date:** November 16, 2026  
**Repository:** `alex00Pirotskyi/ML`  
**Source:** `DATA/DATA_TAP` exclusively  
**Implementation branch:** [`research/tap-swipe-auto-label-v5`](https://github.com/alex00Pirotskyi/ML/tree/research/tap-swipe-auto-label-v5)  
**Verified end-to-end GitHub Actions run:** [43855399972](https://github.com/alex00Pirotskyi/ML/actions/runs/43855399972)  
**Dataset/model artifact:** [`tap-swipe-auto-label-v5-full`](https://github.com/alex00Pirotskyi/ML/actions/runs/43855399972/artifacts/14219439371) (time-limited Actions artifact; re-run workflow to reproduce).

## Release classification

**GO for automatically creating source-bound, confidence-weighted TAP/SWIPE SILVER training labels under the owner's trusted recording-folder ground truth.**

**NOT a certification of independently verified physical tap/swipe onset accuracy or the final embedded production classifier.** This is an automatic *weakly supervised temporal annotation* pipeline. Folder labels guarantee whether the target gesture is present, not the exact frame boundaries. The output explicitly separates `1=auto-proposed event`, `0=trusted negative`, and `-1=unknown` rather than fabricating gold labels for all frames.

## Correct ground truth

The dataset owner specified:

| Source folder | Known target gesture content | Automatic training-label treatment |
|---|---|---|
| `TAP` | Contains TAP events | Localize TAP positives; other intervals UNKNOWN |
| `SWIPE` | Contains SWIPE events | Localize SWIPE positives; other intervals UNKNOWN |
| `FINGER` | Never contains TAP or SWIPE | All usable frames negative |
| `WRIST` | Never contains TAP or SWIPE | All usable frames negative |
| `ARM` | Never contains TAP or SWIPE | All usable frames negative |
| `BAND` | Never contains TAP or SWIPE | All usable frames negative |
| `NOTHING` | Never contains TAP or SWIPE | All usable frames negative |

**Correction to earlier TAP-only experiments:** `SWIPE` must be **positive**, not a hard negative, when training a combined TAP/SWIPE gesture detector. Do not use earlier V1–V4 TAP-only label exports or AUC comparisons as a verdict for this task.

## Completed full-corpus results (independently re-opened CSVs)

| Audit measure | Verified |
|---|---:|
| Source compressed recording archives | 2,921 |
| Unique recordings labeled | **2,790** |
| Duplicates skipped by source content | 131 |
| Source/CSV/export failures | **0** |
| TAP recordings successfully assigned target witnesses | **1,246 / 1,246** |
| SWIPE recordings successfully assigned target witnesses | **896 / 896** |
| Negative-folder recordings assigned positive labels | **0 / 648** |
| Automatically selected TAP event hypotheses | **11,652** |
| Automatically selected SWIPE event hypotheses | **7,054** |
| Total automatically selected events | **18,706** |
| DoubleTaps morphology/timing pair labels | **523** |
| Positive-frame labels (`1`) | **429,354** |
| Negative-frame labels (`0`) | **1,602,078** |
| Unknown-frame labels (`-1`) | **3,475,266** |
| Total labeled/retained frames | **5,506,698** |

**Caution on coverage:** 100% positive-recording *coverage* includes **115 low-weight fallback event witnesses** (0.61% of 18,706 selected events). The pipeline does not claim that every real physical tap or swipe was recovered. Negative-folder zero positives is guaranteed by the trusted folder gate, not a direct estimate of firmware false-positive rate.

### Model discrimination on held-out participant strings

The event-shape teacher is trained with five-fold `GroupKFold` and source-level, out-of-fold (OOF) predictions. The group identifiers are normalized `User` strings with possible unresolved aliases.

| Weak-supervised model experiment | TAP+SWIPE vs other recording-category ROC AUC |
|---|---:|
| Binary all candidate events | 0.824 |
| Binary two-stage MIL-like training | 0.799 |
| **Three-class TAP/SWIPE/other** | **0.837** |
| Three-class two-stage MIL-like training | 0.830 |

Selected three-class model: `AUC=0.836898`, `AP=0.928801`. This score measures **recording-category discrimination** derived from OOF event scores; it is **not** actual per-event localization precision/recall, onset error, or a calibrated probability. TAP-versus-SWIPE recording discrimination was about 0.746 via the TAP softmax score.

### Confidence evidence on 18,706 selected events

We inspected the downloaded **actual full-corpus selected-event CSVs** from the successful workflow artifact and counted event-level provenance:

| Provenance | Events | Fraction | Use |
|---|---:|---:|---|
| `AUTO_POINTER_EMG_STRONG` | **11,465** | **61.29%** | Higher-weight selected events with internally consistent temporal cue agreement |
| `AUTO_POINTER_EMG_WEAK` | 4,492 | 24.01% | Medium-weight plausible physiological events with weaker timing corroboration |
| `AUTO_MIL_HIGH` | 2,321 | 12.41% | Model-high candidates without independent pointer alignment |
| `AUTO_MODEL_UNCERTAIN_POINTER` | 313 | 1.67% | Lower-weight candidates under unreliable pointer evidence |
| `AUTO_BAG_WITNESS_LOW` | 115 | 0.61% | **Very-low-weight fallback** from known-positive recording bag |

Strong + weak temporal cue events: **15,957 / 18,706 (85.3%)**. Timing cues are neither full physical truth nor proof of no false positive. The positive timestamps come from EMG episode onsets and are corroborated by pointer timing when robust; they are **not** copied from a metronome or filename.

Observed event-count distribution by folder subtype, from all final `.selected_events.csv` files:

| Subset | Unique files | Selected events | Median events/file | Strong provenance fraction |
|---|---:|---:|---:|---:|
| SoftTap | 276 | 2,401 | 9 | 68.8% |
| MidTap | 302 | 2,911 | 10 | 68.8% |
| StrongTap | 273 | 2,242 | 9 | 72.2% |
| DoubleTaps | 118 | 1,703 | 15 | 63.2% |
| Tap-on-motion | 104 | 954 | 10 | 52.7% |
| Other TAP | 173 | 1,441 | 9 | 50.8% |
| SWIPE | 896 | 7,054 | 9 | 55.0% |

These counts make it possible to target downstream student training by evidence quality and activity type instead of treating all proposed labels as equivalent.

## How V5 creates automatic temporal labels

1. Read original files from **only** `DATA/DATA_TAP`; verify category against explicit 7-category allowlist.
2. Extract the V3 physiological episode candidate table (77,377 EMG event shapes). The offline teacher can use complete event-shape/decay features; the embedded student **must not** use future information it cannot obtain at inference time.
3. Train LightGBM TAP/SWIPE/non-target on **recording-category** supervision. The training-time model feature vector **excludes** `PointerState`, `TriggerPressure`, `Metronome`, filename, BPM, person ID and source-folder ID.
4. Cross-fit out-of-fold scores using subject-string-grouped 5 folds, with file-balanced weights to prevent long/high-event-count recordings dominating.
5. In known-positive bags, select distinct EMG episodes using recording-class compatibility, shape quality, pointer/EMG lag fitted on part of the recording and verified on the remainder, temporal de-duplication and bounded event rate.
6. Emit `gesture_label=1` in short onset-centered TAP/SWIPE windows, per-event `gesture_kind` (1 TAP, 2 SWIPE), `gesture_event_id`, `gesture_pair_id`, `gesture_weight`, physiological candidate ID, cue and model scores, and provenance; leave uncertain positive-bag frames `-1`.
7. In negative folders, label all usable frames as `0` by authoritative recording truth, retaining `-1` for warmup or timestamp discontinuities.
8. Deduplicate source CSV by decompressed SHA-256; verify every source hash against the feature-model input snapshot and reject contradictory source categories.
9. Export to a **separate, versioned destination** with an audit manifest; independently re-open each output CSV and compare original timestamps, row counts, label weights, event IDs, class isolation and double pair IDs.

### Adversarial timing stress test

The earlier pointer/EMG alignment algorithm was too permissive. After strengthening the split-half temporal consistency tests, a local adversarial replay using borrowed positive pointer timing applied to non-target EMG recordings showed:

- Positive-recording strong alignment: **62/95**.
- Borrowed pointer timing falsely accepted as strong agreement: **39/752 (5.2%)**.

This was a **synthetic pairing stress** under a selected sample, *not* a full end-to-end false-event-rate estimate. It motivates lower weights for weak cues and independent firmware validation.

## Training guidance

To train a compact TAP/SWIPE student:

- Train on rows where `gesture_label` is 0 or 1; **exclude `-1` from supervised loss**. Use `gesture_weight` as per-example weight, preserving the strict distinction between high- and low-confidence positives.
- Use the combined `gesture_label` for a binary TAP-or-SWIPE trigger. If TAP and SWIPE must be differentiated, `gesture_kind=1` TAP, `2` SWIPE supplies a weak subtype label. The legacy `tap_label` alias in V5 means the **combined TAP/SWIPE** class, not TAP-only.
- Balance the **effective weighted positive/negative loss**; the exported data contain far more trusted negative frames than short positive windows. Compare strong-only/weighted-all ablations for soft-tap recall, swipe recall and negative interference.
- Split evaluations by participant and recording session, not random frames; resolve participant aliases before claiming human-independent generalization.
- Keep the embedded student inputs strictly causal and firmware-available: EMG/ACC/WL features are candidates; **do not train the embedded student to rely on Oculus, metronome, provenance, source folder, filename or future offline teacher windows**.
- A robust final release requires field tests on held-out users and normal-daily-activity hard negatives. Folder-level truth guarantees presence/absence, **not** the exact event onset of every physical gesture.

## Reproducibility, source and generated artifacts

- [Implementation and usage (branch README)](https://github.com/alex00Pirotskyi/ML/blob/research/tap-swipe-auto-label-v5/research/tap_real/README_AUTO_LABEL_V5.md)
- [V5 fit and auto-export implementation](https://github.com/alex00Pirotskyi/ML/blob/research/tap-swipe-auto-label-v5/research/tap_real/auto_label_v5.py)
- [Independent stream-level audit](https://github.com/alex00Pirotskyi/ML/blob/research/tap-swipe-auto-label-v5/research/tap_real/audit_auto_label_v5.py)
- [Regression tests](https://github.com/alex00Pirotskyi/ML/blob/research/tap-swipe-auto-label-v5/research/tap_real/test_auto_label_v5.py)
- [Successful full end-to-end workflow](https://github.com/alex00Pirotskyi/ML/actions/runs/43855399972) — CI ran `ALL_V5_TESTS_PASSED`, `fit`, `export`, then `V5_INTEGRITY_RESULT` with 2,790 checks.
- [Full 25.7 MB zipped label/model/output artifact](https://github.com/alex00Pirotskyi/ML/actions/runs/43855399972/artifacts/14219439371) — 2,790 label CSVs, selected-events CSVs, OOF teacher scores, training model, frozen source-hash manifest, full export/verification summaries and logs.

### Example: regenerate automatically without manual onset annotations

```bash
python -m pip install numpy pandas scipy scikit-learn lightgbm

python research/tap_real/auto_label_v5.py fit \
  --features research/tap_real/results_v3/v3_features.csv.gz \
  --input-dir DATA/DATA_TAP \
  --out-dir research/tap_real/results_v5_model

python research/tap_real/auto_label_v5.py export \
  --input-dir DATA/DATA_TAP \
  --scores research/tap_real/results_v5_model/oof_event_scores.csv.gz \
  --source-hashes research/tap_real/results_v5_model/source_sha256.csv \
  --strict-provenance \
  --out-dir research/tap_real/results_v5_labels

python research/tap_real/audit_auto_label_v5.py \
  --input-dir DATA/DATA_TAP \
  --output-dir research/tap_real/results_v5_labels \
  --min-target-coverage 0.70
```

For the exact verified result and frozen input data, use the Actions artifact rather than rerunning with changed library versions or feature extraction settings.

## Release decision

**The user's requested automatic train-label generation is implemented and passes full source/serialization integrity; the generated dataset is ready for experimental student training as confidence-weighted weak labels.** It is **not** correct to state that all 18,706 automatically selected timestamps are independently verified true physical events, or to declare the resulting untrained/untested embedded classifier production-validated. Preserve the V5 dataset and metrics as a reproducible baseline for student model optimization and field tests.
