# Mudra TAP V3 — event shape, physiological re-arming, and full-corpus research

**Research date:** November 16, 2026  
**Baseline:** [V1/V2 full dataset research](TAP_DEEP_RESEARCH_RESULTS_2026-11-16.md)  
**Experiment branch:** [`research/tap-real-benchmark-20261116`](https://github.com/alex00Pirotskyi/ML/tree/research/tap-real-benchmark-20261116)  
**Full GitHub Actions evidence:** [V3 2,790-file experiment](https://github.com/alex00Pirotskyi/ML/actions/runs/43838651326)  
**V3 regression tests:** [Successful CI](https://github.com/alex00Pirotskyi/ML/actions/runs/43840550961)

> **Research only. Do not use V3 pseudo-labels as physical ground truth, or deploy the offline event-shape model to nRF52832.** The 2026-11-16 study has no independently measured physical tap onset times and therefore cannot establish real tap recall, precision, false-tap rate, or firmware latency.

## Executive conclusion

V3 substantially improves the *recording-category weak-label ranking* compared with V1/V2 event-density metrics, and a physiological re-arming state machine improves synthetic double-tap recovery without splitting tested broad single pulses. This does **not** yet solve tap recognition: FINGER gestures remain a major confounder. The strongest held-out V3 weak-label recording-category AUC is about **0.70** on all unique files, but the primary TAP-vs-FINGER AUC is only **0.52**. A more FINGER-conscious model improves that comparison to **0.62**, still far from deployment quality.

The decision is **continue with V3 as an offline candidate miner and weak-supervision benchmark, preserve abstention, and obtain genuine annotated physical tap onset labels before production training**.

## V3 design — not another thresholded peak counter

1. **Per-zone, causal PSD transforms:** process each of the 15 values `p_core->emg_psd[0][CHx][zone]` independently. These are log-coded spectral-power features, **not raw sEMG**, so do not sum the stored log values as if they were linear powers. A timestamp-aware fast envelope (~17 ms) and asymmetric resting baseline (up ~850 ms, down ~230 ms) estimate within-zone activation changes. Aggregate the strongest few zones per electrode, allowing an authentic one-channel soft tap.
2. **Two complementary activation states:** a slow-baseline contrast identifies the first event, while a short-lag (~26 ms) rise identifies a second onset without waiting for full baseline recovery.
3. **Event-shape state machine:** `IDLE -> RISE -> FALL -> IDLE/REARMED`. A candidate can only rearm after observed physiological decay/valley and renewed rise. Event duration, rise/fall, compactness, channel and zone recruitment, waveform-length change and optional ACC change are measured. A broad sustained contraction must not become multiple confirmed taps just because it has local maxima.
4. **Conservative pair tagging:** two existing episodes within an experimental 80–500 ms separation are tagged with a shared `pair_id` only when both pass morphology/valley checks. The algorithm never fabricates a second tap from `DoubleTaps`, `PointerState`, `TriggerPressure`, or `Metronome`.
5. **Abstention by default:** CLI emits auditable candidate IDs, `tap_pair_id`, shape scores, and `UNKNOWN = -1`; generating heuristic positive TAP rows requires the explicit flag `--experimental-positive-labels`. Folder-category hard negatives are useful weak class information, **not verification of every instantaneous movement**.
6. **Offline weak-label discriminator:** train a small LightGBM event-shape model using recording folders for weak TAP vs non-TAP labels, with group-held-out participant strings. A two-stage variant mines top predicted TAP-bag events before refitting. No filename, class_id, BPM, participant identifier, Oculus, or metronome fields enter the feature matrix. There is still label noise in TAP bags.

**Important:** Event features using complete fall/decay or +10..+150 ms context are **offline teacher** descriptors; they must not be assumed available for zero-latency firmware inference. A separately verified causal student and C/Python parity tests are required.

### Files and workflow

- [V3 state-machine / labeler](https://github.com/alex00Pirotskyi/ML/blob/research/tap-real-benchmark-20261116/research/tap_real/tap_event_shape_v3.py) — Python CLI and sidecar output
- [V3 weak shape models and participant CV](https://github.com/alex00Pirotskyi/ML/blob/research/tap-real-benchmark-20261116/research/tap_real/weak_shape_mil_v3.py)
- [V3 deterministic regression tests](https://github.com/alex00Pirotskyi/ML/blob/research/tap-real-benchmark-20261116/research/tap_real/test_event_shape_v3.py)
- [Full-corpus CI workflow](https://github.com/alex00Pirotskyi/ML/blob/research/tap-real-benchmark-20261116/.github/workflows/tap-v3-shape-full.yml)

The source code is isolated on the research branch pending review; the source links above target the experimental branch until a reviewed merge. The full GitHub Actions run provides the actual audit artifacts, including feature tables and out-of-fold per-event probabilities.

## Full real-data results

Source: `DATA/DATA_TAP` exclusively; 2,921 archives scanned, **2,790 unique** compressed blobs, **131 duplicates excluded**, **0 parsing failures**. The state machine extracted **77,377 candidate episodes**. Source duration is approximately **9.12 hours**: 4.01 hours TAP and 5.11 hours non-TAP. CSV user strings are normalized imperfectly (56 canonical strings in the V3 extraction), so participant independence is approximate, not guaranteed.

**File category ROC AUC (higher is better, physical-event accuracy not measured):**

| Method / evaluation | Held-out TAP versus non-TAP file ROC AUC |
|---|---:|
| V1 accepted-event density, earlier full audit | 0.435 |
| V2 accepted-event density, earlier full audit | 0.450 |
| V3 state shape + binary LightGBM, 5-fold person-string held out | **0.7011** |
| V3 no-ACC two-stage weak MIL-like selection | 0.6936 |
| V3 early-only feature ablation | 0.6946 |

These are different types of scores: the first two use only event density, and the V3 scores use the trained shape discriminator. They are not direct estimates of physical tap event precision, recall or latency. The early-only ablation still uses event-peak identification and must not be interpreted as firmware readiness.

### Hard-negative failure at probability threshold 0.70

Using the **no-ACC two-stage** held-out model (all values are `p >= 0.7` events/hour, not verified firmware FPs):

| True recording group | High-scoring candidate episodes per hour |
|---|---:|
| TAP | **209.9** |
| SWIPE | 42.5 |
| FINGER | **519.6** |
| WRIST | 66.5 |
| ARM | 67.3 |
| BAND | 171.0 |
| NOTHING | 0.0 |

The model ranks FINGER events as tap-like more often than TAP episodes. This is unacceptable as a deployable tap trigger. It also indicates that threshold-only and shape-only pseudo-labeling would still corrupt training.

Pairwise TAP-vs-category AUC using the held-out no-ACC two-stage model:

- TAP vs SWIPE: **0.720**
- TAP vs FINGER: **0.517**
- TAP vs WRIST: **0.794**
- TAP vs ARM: **0.861**
- TAP vs BAND: **0.756**
- TAP vs NOTHING: **0.637** (only five NOTHING files)

### Alternative architectures attempted and compared on the same complete extracted-feature corpus

Additional participant-grouped 5-fold CV experiments, using complete event shapes and weak folder labels:

| Variant | TAP vs all file ROC AUC | TAP vs FINGER file ROC AUC |
|---|---:|---:|
| Binary baseline | 0.702 | 0.520 |
| Binary, FINGER negatives ×3 | 0.679 | 0.588 |
| Binary, FINGER negatives ×6 | 0.644 | 0.600 |
| Seven-class LightGBM, equal file weighting | **0.725** | 0.570 |
| Seven-class LightGBM, FINGER ×3 | 0.709 | **0.620** |

**Interpretation:** emphasizing FINGER reduces its confusions, but simultaneously suppresses TAP events and harms overall ranking. It does not establish physical specificity. A geometric/additive ensemble did not meaningfully fix the TAP-vs-FINGER ambiguity. A separate PSD-jump gate also failed to provide convincing discrimination and was rejected.

## Controlled synthetic event-level regression

Exactly **480 two-pulse scenarios**: 5 amplitude pairs × 6 delays (70–400 ms) × 4 pulse widths (18–60 ms) × 4 frame spacings (5–30 ms). One-to-one event matching at ±90 ms.

| Method | Matched synthetic events / 960 | Synthetic F1 | Exactly two events and no extras |
|---|---:|---:|---:|
| V1 | 142 / 960 | 0.258 | 0 / 480 |
| V2 | 417 / 960 | 0.606 | 160 / 480 |
| **V3 event-shape FSM** | **558 / 960** | **0.735** | **233 / 480** |

For V3, 558 synthetic events were matched, 402 missed and zero additional `TAP_SHAPE` predictions in that clean test. This is **not** real-data false-positive rate. V3 recovered both synthetic soft–soft taps in **53 of 96** cases. A separate 30-case sweep of *single broad pulses* passed the no-false-double regression test, and stuck pointer signals did not influence event onsets. Two independent synthetic events remain inseparable when their physiological waveforms overlap strongly; returning UNKNOWN is preferable to inventing a pair.

## Physical data limitations and release gate

- The CSVs contain only processed log-coded EMG PSD zones, waveform length and ACC-derived features—not synchronized raw 24-bit sEMG contact onsets. Physically identical-looking tap and isolated finger-flexion signals may be indistinguishable at this abstraction level.
- `PointerState` has missing edges, temporal offsets and even stuck-high NOTHING recordings. Its agreement cannot be a final score. `TriggerPressure` is not independent in inspected files.
- Some recording labels/filename aliases may be wrong, and participant normalization is incomplete. Duplicates must never cross splits.
- The V3 event-shape teacher uses some post-onset context. An nRF52832 causal implementation and measured CPU/flash/RAM do not yet exist; this version is explicitly offline.
- **Before production:** build a small, high-quality, manually verified physical-event set with normal/soft/soft–soft/double taps and hard FINGER negatives; hold out participants and recording sessions; measure event precision, soft-tap recall, pair split/merge, latency and false activations/hour at fixed sensitivity. Compare trained binary, multiclass and separate FINGER-veto approaches; deploy only a model that passes all gates.

## Usage (research branch)

```bash
python -m pip install numpy pandas scipy scikit-learn lightgbm

# Safe default: detect candidate episodes, leave TAP positives UNKNOWN
python research/tap_real/tap_event_shape_v3.py \
  --input-dir DATA/DATA_TAP \
  --out-dir DATA/DATA_TAP_EVENTS_V3

# Full, slow, read-only corpus evaluation (writes to research only)
python research/tap_real/weak_shape_mil_v3.py \
  --input-dir DATA/DATA_TAP \
  --out-dir research/tap_real/results_v3
```

To inspect **unverified** heuristic-positive behavior, explicitly add `--experimental-positive-labels`; those labels must not be used as production truth.

## Decision

**Keep V3, not V1/V2, as the experiment branch’s event candidate miner; keep trained scores in research, not firmware.** Its stateful rearm logic is a meaningful improvement for double taps, and its weak-label event-shape model shows some value across the real corpus. The hard FINGER negatives remain an unresolved blocker, so do not automatically promote these pseudo-labels into final embedded model training.


## Supplemental investigation — soft taps versus finger-only gestures

A more targeted analysis exposed that the general two-stage V3 model is especially weak for soft taps. On the full held-out predictions, **SoftTap recording vs FINGER recording AUC = 0.407**, versus **0.778 for DoubleTaps vs FINGER**. At a 0.70 threshold the general model produces **53.7 high-probability events/hour** in SoftTap recordings, versus **519.5/hour in FINGER**. This directly contradicts a claim of reliable soft-tap recognition.

A specialist, separately trained binary SoftTap-vs-FINGER model was therefore evaluated using the same V3 event-shape features and five-fold participant-string-grouped holdout. Source: all **276 unique SoftTap files + 339 unique FINGER files** (615 files; 16,773 extracted event episodes). **No Oculus or metronome input** was used.

| Specialist model | SoftTap vs FINGER file ROC AUC | SoftTap high-p≥0.70 events/hour | FINGER high-p≥0.70 events/hour |
|---|---:|---:|---:|
| All event features, weak bag labels | 0.743 | 2,106 | 1,448 |
| Two-stage weak event selection | **0.754** | 958 | 435 |
| Two-stage without ACC | 0.749 | 997 | 448 |
| Early-feature two-stage | 0.743 | 773 | 444 |

**Interpretation:** Focusing the objective on the hard FINGER class improves ranking of *recordings* (0.407 → 0.754), but still passes hundreds of FINGER episodes/hour as high-scoring candidates. Raising its candidate threshold from 0.70 to 0.90 reduces FINGER candidates from **435/hour to 25/hour**, while SoftTap candidates fall from **958/hour to 101/hour**; this is not a deployable soft-tap operating point. That is not a production false-positive rate, yet it shows that weak folder labels and these feature summaries alone cannot establish a safe operating point. The specialist result is exploratory and must be verified against manually annotated physical tap times, especially soft taps with poor electrode placement.

Reproducible specialist script: [`soft_vs_finger_v3.py`](https://github.com/alex00Pirotskyi/ML/blob/research/tap-real-benchmark-20261116/research/tap_real/soft_vs_finger_v3.py). It requires the extracted `v3_features.csv.gz` from the full V3 workflow artifact.


## Full V3 candidate-only labeling export — independent integrity pass

**Verified GitHub Actions run:** [V3 export/integrity audit](https://github.com/alex00Pirotskyi/ML/actions/runs/43841132120)  
**Download full candidate-only sidecars:** [GitHub Actions artifact `tap-v3-full-candidate-sidecars`](https://github.com/alex00Pirotskyi/ML/actions/runs/43841132120/artifacts/14214160145) (GitHub artifact retention: 14 days)  
**Export branch:** `research/tap-real-benchmark-20261116`

The source-only `DATA/DATA_TAP` export finished successfully on 2026-11-16. The workflow independently re-opened each written `.labels.csv.gz` and verified the output schema and label states:

| Audit check | Verified result |
|---|---:|
| Input source archives | 2,921 |
| Unique recordings with written labels | **2,790** |
| Duplicates excluded | **131** |
| Export / verification failures | **0** |
| Unique DoubleTaps files | **118** |
| DoubleTap morphology-only candidate pairs | **349** |
| Candidate pairs across all recording classes | 6,178 |
| Confirmed physical tap positives written | **0** |

All 2,790 labels files contained `tap_label`, `tap_event_id`, `tap_candidate_id`, `tap_pair_id` and `tap_candidate_shape_score`. Every `tap_label` was 0 or -1 and every `tap_event_id` was -1 in the conservative default run. The pair counts are **unverified physiological candidate associations**, not independent confirmed double taps; the 6,178 all-class pair candidates further demonstrate that local temporal pairing alone is highly nonspecific.

The export also writes one `.events.csv` per recording and a `manifest.jsonl` with provenance, SHA-256 of each source CSV, and quality counters. It does not overwrite any original recording.

**Release decision unchanged:** pass the generated candidates through independent human-verified event annotation before promoting anything to a positive training label. The already-measured FINGER confusion is the major unresolved false-tap risk.
