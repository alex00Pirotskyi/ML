# Mudra TAP labeling: full-corpus research and honest evaluation

**Date:** 2026-11-16  
**Repository:** `alex00Pirotskyi/ML`, main dataset `DATA/DATA_TAP`  
**Benchmark branch:** `research/tap-real-benchmark-20261116`  
**GitHub Actions real-corpus run:** https://github.com/alex00Pirotskyi/ML/actions/runs/43833576665  
**Status:** comprehensive dataset audit and pseudo-labeler comparison completed. **No independently synchronized physical-tap ground truth exists in this study; no production recall/precision/false-tap figures are justified.**

## Executive decision

**DO NOT select or deploy v1, v2, or a simple EMG peak threshold as the final tap labeler.** Although v2 improves synthetic double-tap recall, a 2,790-unique-recording, 9.12-hour real-corpus replay demonstrates severe over-segmentation of general muscle activity. The right approach is a high-recall *candidate* generator followed by a gesture-specific discriminator/weakly supervised event learner trained to distinguish target TAP morphology from SWIPE/FINGER/WRIST/ARM/BAND/NOTHING. Preserve an UNKNOWN label. Require an independent gold event set before calling any result accurate.

## Dataset coverage and integrity

- 2,921 compressed files enumerated, 2,790 unique by *compressed* SHA256, 131 duplicates, **zero processing failures**; 9.124 hours of recordings.
- TAP: 1,246 unique files, 4.011 hours. All other classes: 1,544 files, 5.112 hours.
- Unique file-category breakdown: SWIPE 896, FINGER 339, WRIST 149, ARM 113, BAND 42, NOTHING 5.
- 75 distinct strings in the CSV User field. These may contain aliases: do not treat as 75 verified unique human participants without normalization.
- 118 unique DoubleTaps recordings (130 pre-dedup). 276 unique SoftTap recordings.
- Median frame spacing by recording ranges **1 to 23 ms**, with across-file median 5 ms. Never equate a fixed number of rows to a time interval.
- Historic recording schemas are not identical; read named columns and validate configuration. Native `emg_psd[0]` values are *log-coded* spectral power, not raw 24-bit sEMG or linearly summed raw PSD.
- Oculus PointerState is not physical ground truth, can be stuck high even during NOTHING; TriggerPressure replicates pointer edges in inspected examples and is not independent.
- `class_id=1` is assigned to both TAP and SWIPE by `org.py`; do not use it as a tap-specific target.

## Methods tested on the real corpus

1. **v1:** causal fast/slow sEMG PSD novelty and conservative candidate confirmation.
2. **v2:** local EMG short-rise retrigger, double-pair support, candidate abstention, and modified positive logic.
3. **rise08, rise12, rise20, rise35:** pure short-lag EMG rise detectors with thresholds 0.08, 0.12, 0.20, 0.35, relative prominence and a 65ms peak merge. These are candidate finders, not trained TAP classifiers.

**Fair negative test:** v1 and v2 were run under a *counterfactual TAP protocol* even on non-TAP recordings. This is crucial: their normal folder-category gate labels all non-TAP rows as negatives by construction. If we evaluated those gated outputs directly, the zero false positives would be a methodological artifact.

### Real file-level event acceptance

Accepted onset candidates per hour of recording:

| Method | TAP | All non-TAP | Non-TAP / TAP event density |
|---|---:|---:|---:|
| v1 | 7,159 | 8,391 | 1.172 |
| v2 | 15,861 | 17,829 | 1.124 |
| rise08 | 19,769 | 20,645 | 1.044 |
| rise12 | 16,925 | 18,193 | 1.075 |
| rise20 | 12,481 | 14,216 | 1.139 |
| rise35 | 8,010 | 9,693 | 1.210 |

**Interpretation:** raw event density is greater in non-TAP recordings for every method. This does *not* mean that an eventual trained classifier must fail. It means these event detectors are nonspecific and should never directly supply positive target labels without morphology/context evidence.

### File-category discrimination using only event density

ROC AUC (TAP file versus non-TAP file) using accepted candidate events per second: v1 = **0.435**; v2 = **0.450**; rise08 = **0.436**; rise12 = **0.423**; rise20 = **0.409**; rise35 = **0.424**. These are file-level AUCs on a very weak scalar feature, NOT event-level detector accuracy. They are worse than chance-level ranking (AUC=0.5) because normal movement can create more EMG onsets than intentional taps.

Block-bootstrap (500 draws of CSV `User` string clusters) produced approximate v2 file-AUC interval **0.410–0.492**, and v2 non-TAP/TAP density ratio **1.066–1.187**. These are descriptive intervals with incompletely resolved participant aliases and must not be used as clinical CIs.

### Double-tap warning

118 unique DoubleTaps files contain 2,145 Oculus rising edges and 1,528 metronome rising edges. V1 accepted **3,754** positive events; V2 accepted **8,663**, tagged **2,717** pairs, and a simple adjacent-event pairing proxy counted **2,958** pairs. The loose short-rise threshold counted **9,405** candidate events and **4,639** proxy pairs. **None of these numbers is verified true tap count.** Oculus can omit events and can record a single press for a pair; the metronome can tick on other subdivisions. The much higher V2 output is consistent with spurious fragmentation of muscle activity, especially alongside high event density during SWIPE, WRIST and NOTHING.

Raw Oculus-edge one-to-one matching at a ±125ms tolerance was 91.1% for v1 and 97.3% for v2 in TAP recordings. **This is not physical recall.** A dense onset generator can match many externally recorded edges while producing numerous other events, and that pointer signal is itself imperfect and was part of some pseudo-label decisions.

### Synthetic event-level timing benchmark

A separate suite of **480 synthetic two-event scenarios** (5 strength combinations × 6 intervals × 4 Gaussian widths × 4 sample periods, 960 known generated events) scored one-to-one event matches within ±90ms:

| Method | Synthetic event recall | Synthetic F1 | Exact two detected/no extra |
|---|---:|---:|---:|
| v1 | 14.8% | 0.258 | 0/480 |
| v2 | 43.4% | 0.606 | 160/480 |
| rise08 | 78.5% | 0.880 | 289/480 |
| rise12 | 65.2% | 0.789 | 206/480 |
| rise20 | 40.4% | 0.576 | 102/480 |

Synthetic tests are *deliberately clean*. None of these F1 scores are real-world precision/recall. A targeted 32-case baseline (4 amplitudes × 8 gaps) found v2 split 7/8 strong–strong, 5/8 strong–soft, 8/8 soft–strong, and **0/8 soft–soft**. A broad activation can create two local feature peaks for a *single* contraction, while two overlapping soft taps can produce only one smooth pulse. This is why a generic peak rule cannot solve the physical event identification problem.

## What we should build next: V3 approach

**Stage A — high-recall candidate mining.** Calculate a short-lag, timestamp-aware rise from each of 15 EMG PSD log zones rather than only the sum of all zones. Estimate a causal per-zone baseline, noise floor, saturation/contact state and slope. Cluster activation into *episodes*, with local prominence and per-channel evidence. A high-recall detector should produce candidate episodes and UNKNOWN labels, not immediately confirmed TAP positives.

**Stage B — target-gesture discriminator.** Use the reliable recording-category weak labels to learn which *episodes* resemble intended TAP versus SWIPE, FINGER, WRIST, ARM and BAND. Start with an event-level LightGBM baseline, event-window features and positive-bag / negative-bag training. Avoid `class_id` confusion. Train with group-held-out participants and an event sparsity prior; otherwise the model can simply learn 'any muscle activity' as positive. Validate performance of multi-instance top-k pooling against conservative bootstrapped example mining. `NOTHING` alone is not enough negative coverage; use all 1,544 non-TAP recordings.

**Stage C — specialized double-tap modeling.** Each physical pulse has independent onset, event ID and quality/confidence. The firmware sees the same compact *single-tap* classifier for each event and employs a 2-event timing automaton. No forced 'two taps per beat' labels. Re-arm on physiological decay and renewed rise; prevent a wide first pulse from being counted twice. An uncertain second pulse becomes UNKNOWN, not a training-negative. An offline model may use wider context to label; the deployed model must remain causal.

**Stage D — calibration and audit.** Compare the following independent objectives: (i) manually verified soft tap event recall at one-to-one onset tolerance; (ii) event precision and doubles split/merge; (iii) false activations per hour in held-out real non-TAP activity; (iv) event latency; (v) firmware model RAM/flash/CPU. A method is accepted only if it improves an outcome on a held-out physical reference, not because it scores its own pseudo-labels highly.

**Stage E — nRF52832 deployment.** Use a compact causal tree/student with few cheap EMA/short-difference features, 3-channel health checks, and a one-shot TAP state machine. Preserve separate existing long-press (LP) logic; TAP initiates LP, LP holds/releases. Reproduce exact firmware PSD/log conventions. Verify C/Python parity on same features and measure CPU occupancy before enabling new inference.

## Quality-control and potential pitfalls

- Beware **label leakage**: PointerState, metronome, TriggerPressure, filename, class_id, offline entire-recording normalization, and future events must never enter student features.
- Beware **weak label circularity**: optimizing against Oculus edges then reporting agreement with Oculus rewards the teacher by construction.
- Beware **person leak**: normalize name aliases; group by user/session/acquisition cohort and keep duplicates from splitting across folds.
- Beware **candidate selection bias**: score all events, not only selected strong pulses. A simple threshold that always finds top-ranked peaks in every recording cannot be expected to reject idle noise.
- Beware **PSD processing mismatch**: legacy per-bin divisions after logarithmic encoding are not equivalent to physical linear-bin averaging; verify sensor extraction format per cohort.
- Beware **illusory low false-positive rate** if code gets the ground-truth category as input and hardcodes negatives.

## Research literature

- Comparison of sEMG onset methods in 55 people, TKEO + CFAR raw-sEMG approach: https://pmc.ncbi.nlm.nih.gov/articles/PMC10885915/
- Onset timing changes significantly with SNR: https://pmc.ncbi.nlm.nih.gov/articles/PMC8395734/
- Adaptive onset with nonstationary noise: https://pmc.ncbi.nlm.nih.gov/articles/PMC9103856/
- Bayesian changepoint approaches and multi-burst caveats: https://pmc.ncbi.nlm.nih.gov/articles/PMC5425195/

**Decision:** Keep V1 and V2 for reproducibility. Do **not** train a production TAP student using the V2 positives as truth. Next research must test recording-label-supervised event morphology learning plus honest independent event labels, and show lower non-TAP false activations while protecting genuinely soft tap recall.