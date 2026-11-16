# Mudra V5 — fully automatic TAP + SWIPE weak-supervised training labels

**Status:** source-aware, reproducible automatic pseudo-label pipeline. The operator's recording-folder ground truth makes the target classes authoritative, but **individual onset times remain inferred**; do not call the generated temporal positives independently verified physical ground truth.

## Authoritative bag-level ground truth

| Source folder | Target presence | Export semantics |
|---|---|---|
| `DATA/DATA_TAP/TAP` | Contains taps | Auto-locate TAP events, `gesture_kind=1` |
| `DATA/DATA_TAP/SWIPE` | Contains swipes | Auto-locate SWIPE events, `gesture_kind=2` |
| `FINGER`, `WRIST`, `ARM`, `BAND`, `NOTHING` | Contains **no taps/swipes** | Trusted `gesture_label=0` negative windows |

**Critical correction:** earlier V3/V4 reports treated SWIPE as a negative while studying TAP-only discrimination. This is not the intended *combined TAP+SWIPE* task. Training on the old TAP-only labels would incorrectly teach the student to reject legitimate SWIPE events.

A source filename/folder **does not** supply onset times. For TAP and SWIPE files, an uncertain EMG interval remains `-1` instead of treating every frame as gesture-positive.

## Pipeline

1. **Physiological candidate episodes.** The existing V3 causal PSD-zone event-envelope and re-arming state machine finds possible muscle onsets and preserves separately observed close taps. Only features that exist in `DATA_TAP` are used.
2. **Weak recording-level model.** A small 3-class LightGBM ranks events as TAP, SWIPE, or non-target using EMG/ACC/WL event shape. `PointerState`, `Metronome`, file names, source category, person ID, BPM and class_id are **excluded from the model feature matrix**.
3. **Holdout against participant leakage.** Five-fold `GroupKFold` by normalized user string creates **out-of-fold (OOF)** event scores, so labels for an existing source recording are scored by a model that did not train on that subject group. Groups are approximate aliases; perfect identity resolution is not asserted. A final model is also trained on all data for potential future unseen recordings.
4. **Automatic onset reconciliation.** In known-positive recording bags, EMG episodes are correlated with real recorded Oculus rising edges. The lag is fitted using only the first half of pointer edges and tested on the remaining half. A strong match requires substantial train/test coverage and small timing error. Strong corroboration gets a higher training weight. Unstable timing is downgraded rather than blindly trusted; high-EMG-score model-only events are allowed. When a reliable candidate exists but the model/timing quality is low, it can become an explicitly **low-weight bag witness**.
5. **Precise label serialization.** Per-source sidecar CSVs contain combined `gesture_label\in\{-1,0,1\}`, per-label weight, `gesture_kind=1/2`, distinct event and double-pair IDs, model scores, source-safe candidate IDs, and provenance. `tap_label` is provided as a **legacy alias for the combined TAP+SWIPE class**, not a TAP-only label. Double pairs are assigned only within DoubleTaps recordings when two actual selected episodes occur 65–500 ms apart.
6. **Safety gates.** Only read files from a root named `DATA_TAP`. Deduplicate decompressed SHA-256. Fail if the same CSV appears in contradictory folders. Freeze and verify source hashes against the feature-model training source. Keep output separate, atomically publish, preserve immutable versioned run folders. The independent audit reopens every saved CSV and verifies source timestamps, class isolation, positive event references, pair IDs, weight invariants and positive-bag coverage.

## Tested evaluation results

Feature corpus: **2,921 source archives, 2,790 unique recordings, 77,377 candidate EMG episodes, 56 normalized participant strings**.

| Method | 5-fold person-string-heldout ROC AUC — positive TAP/SWIPE recording vs other category |
|---|---:|
| Binary recording-weighted event-shape model (all bag events weakly labeled) | **0.824** |
| Binary two-stage MIL-like event selection | 0.799 |
| Three-class TAP/SWIPE/other, all weak-bag events | **0.837** |
| Three-class two-stage selection | 0.830 |

These are **recording-category bag-level AUCs**, not physical event precision/recall or onset timing accuracy. The three-class model also reaches 0.746 AUC for distinguishing TAP from SWIPE *recordings* via its TAP score. Model softmax values are ranking scores, **not validated calibrated event probabilities**.

### Initial real-recording replay

On **147 unique representative recordings** (68 TAP, 27 SWIPE, 52 non-target), the hardened V5 auto-export passed its independent post-serialization audit:

- **95/95 positive TAP or SWIPE recordings** got at least one automatic proposed event.
- **0/52 negative recordings** got a positive label, by authoritative category gating.
- 973 selected positive event candidates, 106 double-pair candidate associations, 18,583 positive rows, 133,235 negative rows, 151,834 unknown rows.
- Candidate provenance and weights distinguish high-confidence consistent pointer/EMG observations from weaker model-only and fallback hypotheses.
- A borrowed-pointer adversarial stress test produced **62/95 strong alignments in real positive recordings** but **39/752 (5.2%)** when positive recordings' pointer patterns were transplanted onto non-target muscle signals. This is a useful stress test, **not a calibrated false-positive rate**.

## Command-line usage

```bash
python -m pip install numpy pandas scipy scikit-learn lightgbm

# Inputs: existing V3 77,377-event feature table and source recordings.
# The source hash binding is required for strict production-data export.
python research/tap_real/auto_label_v5.py fit \
  --features research/tap_real/results_v3/v3_features.csv.gz \
  --input-dir DATA/DATA_TAP \
  --out-dir research/tap_real/results_v5_model

# Fully automatic OOF-scored labeling (no manually annotated onset CSV).
python research/tap_real/auto_label_v5.py export \
  --input-dir DATA/DATA_TAP \
  --scores research/tap_real/results_v5_model/oof_event_scores.csv.gz \
  --source-hashes research/tap_real/results_v5_model/source_sha256.csv \
  --strict-provenance \
  --out-dir research/tap_real/results_v5_labels

# Optional: append every label column to the original full source CSV rows
# with --full-rows when disk space allows.
python research/tap_real/audit_auto_label_v5.py \
  --input-dir DATA/DATA_TAP \
  --output-dir research/tap_real/results_v5_labels \
  --min-target-coverage 0.70
```

For a fresh run, use a *new* output directory rather than overwriting earlier generated labels. Avoid adding generated artifacts to `DATA/DATA_TAP` or committing large training CSV exports to the Git repository.

## Reproducible GitHub Actions

- [Full TAP+SWIPE V5 Actions workflow](https://github.com/alex00Pirotskyi/ML/blob/research/tap-swipe-auto-label-v5/.github/workflows/tap-swipe-auto-label-v5.yml)
- [Automatic labeling source](https://github.com/alex00Pirotskyi/ML/blob/research/tap-swipe-auto-label-v5/research/tap_real/auto_label_v5.py)
- [Independent output integrity audit](https://github.com/alex00Pirotskyi/ML/blob/research/tap-swipe-auto-label-v5/research/tap_real/audit_auto_label_v5.py)
- [Deterministic regression tests](https://github.com/alex00Pirotskyi/ML/blob/research/tap-swipe-auto-label-v5/research/tap_real/test_auto_label_v5.py)

## Release interpretation and limits

**Automatic training-label generation requires no manual tap timestamp annotations.** This solves the *pipeline automation* requirement under the user's recording-level truth assumption. The output is appropriate as a *weighted, provenance-tracked pseudo-label dataset* for training and experiment iteration.

The physical onset of each soft tap, missing tap, or swipe is **not identifiable with absolute certainty from recording-class labels alone**. This study cannot provide actual event-level precision/recall, reliably quantify missing soft taps, or prove that a student trained on these labels will meet production runtime behavior. Validate the trained firmware on held-out physical events and negative daily activity before claiming an embedded model is production-ready.

This document is a research release record; compare V5 weighted student models against V3/V4 and the archived production model before replacing the deployed firmware.
