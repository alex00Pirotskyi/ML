# TAP pattern research V7 — deep audit of DATA/DATA_TAP

**Date:** November 16, 2026  
**Release decision:** RESEARCH ONLY. No production-positive labels were changed.

**Reproducible source verification:** [Full 2,790-file waveform extraction](https://github.com/alex00Pirotskyi/ML/actions/runs/43866509437), [download full waveform artifact](https://github.com/alex00Pirotskyi/ML/actions/runs/43866509437/artifacts/14225039026), [V7 waveform extractor on isolated research branch](https://github.com/alex00Pirotskyi/ML/blob/research/tap-pattern-v7-deep-20261116/research/tap_real/extract_psd_motifs_v7.py).

## Main conclusion

The V5 labeler has a **weak-supervision defect**: its initial teacher labels every physiological candidate inside a TAP recording as TAP (or inside SWIPE as SWIPE), regardless of which candidate was actually the gesture. Its final candidate filter and low-confidence bag-witness fallback do not establish independent physical onset truth. Excellent file-level label consistency is not event-level accuracy.

**Tap patterns are not simply strong activity, short peaks, or repetitive bursts.** Those signals are also prominent and sometimes even more repetitive in FINGER/WRIST/ARM recordings. There is useful information in full channel-by-frequency temporal shape, but the strongest tested pattern score still distinguishes held-out TAP recordings from held-out FINGER only modestly. Day-to-day/session shifts are large enough to invalidate apparent personalized accuracy for some wearers.

## Data and ground-truth contract

- All 2,921 source archives scanned; **2,790 distinct recordings** after excluding 131 duplicate archives.
- The trusted folders are TAP (contains taps), SWIPE (contains swipes), and FINGER/WRIST/ARM/BAND/NOTHING (contain no target gesture). Folder labels do not identify exact onset time. A physical tap on the device or arm in a negative-folder instruction is still a hard negative for the target class.
- V3 physiological candidate extractor reproduced **77,377** episodes; 15-channel PSD waveforms were recoverable for **72,016** episodes with adequate pre/post-event context. **17,618** selected V5 positive hypotheses were covered.
- Original source CSVs were verified against the frozen V5 **uncompressed SHA-256** manifest. No raw recording was modified.
- Time-aligned input: 23 samples at 20 ms intervals from **−200 to +240 ms around a candidate peak**, 3 channels × 5 spectral zones. Smooth, subtract pre-event baseline, normalize and sort electrode channels by response. This is an **offline noncausal** teacher signature, not a zero-latency firmware feature.
- Normalized participant aliases create ~56 groups, but aliases are imperfect; explicitly cross-date testing matters.

## Experiment A — V5 pseudo-positive versus hard-negative event shapes

**Full 77,377 event study.** A classifier was trained using 7,583 high-confidence, pointer+EMG-selected TAP events from V5 as **weak positive hypotheses**, versus 24,267 physiological events from five trusted negative folders. Five-fold subject-group heldout.

| Feature family | OOF ROC AUC against pseudo-positive event targets |
|---|---:|
| Energy/amplitude | 0.795 |
| Temporal curve | 0.770 |
| Spectral zones | 0.817 |
| Electrode recruitment | 0.754 |
| Rise, decay and compactness | 0.785 |
| Shape + spectrum + channel response | 0.832 |
| All V3 event features | **0.837** |

**Important:** These AUCs use *V5's chosen positive hypotheses as the label*, and can be deceptively high. A retrospective operating point selecting 1% of all negative candidate episodes retained only ~7% of those V5 strong-positive hypotheses. The threshold was not calibrated prospectively, so it is only an overlap diagnostic.

Using recording-level TAP/FINGER folder truth and max event score within each heldout file:

| Scoring method | Heldout TAP vs FINGER recording AUC |
|---|---:|
| Energy-only | 0.469 |
| Shape + spectral + channel response | 0.537 |
| All V3 event features | 0.533 |

The shape/spectrum method had a descriptive participant-block-bootstrap AUC interval of approximately **0.454–0.639**. Event-level pseudo-label ranking does **not** translate into dependable TAP-versus-FINGER recording discrimination.

## Experiment B — unsupervised repeated motifs independent of V3, V5, Oculus

**109 stratified source recordings**; 240 ms original PSD windows resampled at 20 ms, with temporal/zone normalization and within-record matrix-profile-style nearest-subsequence search. Self-matches within 400 ms excluded. Ten strongest nonoverlapping repeats measured per recording.

| Folder | Median repeated-window cosine similarity |
|---|---:|
| TAP | 0.912 |
| FINGER | **0.958** |
| WRIST | 0.926 |
| ARM | 0.895 |
| SWIPE | 0.881 |
| BAND | 0.801 |
| NOTHING | 0.795 |

The selected FINGER sessions showed **greater** recurrence than TAP (two-sided Mann–Whitney p≈0.00018). This is a convenience-stratified sample with only 15 FINGER recordings, not a population-level theorem. It falsifies **generic recurrence by itself** as a tap-confirmation rule.

A separate 154-file V3-candidate pattern study found median adjacent-event cosine similarity: TAP 0.713, FINGER 0.756, WRIST 0.802, ARM 0.801. Similarity alone cannot solve tap localization. Close pairs in selected DoubleTaps recordings were not consistently more morphologically similar than pairs in hard negatives.

## Experiment C — exhaustive original PSD waveforms, not just V3 summary statistics

Five-fold subject-group-heldout LightGBM, trained only on high-confidence V5 TAP hypotheses (about 7,194 source-valid event windows) and 23,004 negative-folder events. **72,016** source-verified waveforms were scored. All categories were available for evaluation. Recording-level AUC uses trusted folder labels.

| Representation | Pseudo-positive event AUC | TAP vs FINGER recording AUC | TAP vs all negative recording AUC |
|---|---:|---:|---:|
| Peak strength + generic morphology | 0.813 | 0.544 | 0.625 |
| Old normalized activation curve | 0.749 | 0.465 | 0.530 |
| New compact 31-dimensional PSD shape | 0.811 | 0.525 | 0.620 |
| **Full 345-value PSD temporal waveform** | **0.854** | **0.607** | 0.707 |
| **Full 345 waveform + 10 morphology descriptors** | **0.861** | **0.620** | **0.720** |
| Compact 31 + morphology | 0.842 | 0.589 | 0.693 |

The best TAP-versus-FINGER AUC 0.620 is **modest**. Subject-stratified bootstrap (1,200 repetitions) yielded AUC 95% interval ~**0.527–0.721**; the improvement over the basic morphology model was +0.076 with 95% interval **−0.003 to +0.141**. This is promising but **not statistically conclusive** in this exploratory setup.

At a *retrospectively selected* threshold allowing ~5% of FINGER recording bags above threshold, only **11% of TAP recordings** had a candidate above threshold. The extra waveform features have not solved reliable event discovery.

**Crucial limitation:** Exact event onset/success cannot be measured from folder labels. The event AUC numbers are *not* true physical tap precision/recall; this teacher sees up to 240 ms AFTER the event peak.

## Experiment D — within-wearer repetition and session drift

A full PSD waveform model trained/tested across different recordings *within the same wearer* achieved **median TAP-vs-FINGER bag AUC ~0.882** among eight wearer groups. This result can be optimistic because different files from the *same date/session* may share electrode placement and setup.

A stricter leave-entire-recording-date-out evaluation was feasible for only two wearers with multiple dates containing both target classes:

| Wearer alias | Heldout files, same-wearer AUC | Entire-date-heldout AUC |
|---|---:|---:|
| Michael | 0.884 | **0.414** |
| Ohad | 0.915 | **0.858** |

The Michael result collapses across sessions while Ohad remains comparatively robust. Recorded event-relative PSD activation scale also shifted substantially between Michael's recording dates. This **does not prove** contact was the cause; sensor positioning, task style, user physiology or technical settings may contribute. Most users lack enough both-class dates to estimate this reliably.

A production-generalization evaluation must hold out **wearers AND dates/sessions** and report the worst cases—not just the average file-level CV.

## Experiment E — weak bag supervision without forced per-event positives

To address the original problem directly:

- **True bag-level multiple-instance neural network**: TAP bags contain at least one tap; negative bags contain none. Uses event-shape MLP, top-3 event pooling, negative-bag and positive-sparsity penalties. No V5 positives used for labels. Five-fold heldout TAP-versus-FINGER recording AUC **0.570**; TAP-versus-all-negative AUC **0.652**.
- **Recording-shape-distribution classifier**, quantiles of all candidate shape features with no forced event-positive annotations: heldout TAP-versus-FINGER AUC **0.657**, TAP-versus-all-negative AUC **0.775**. This is recording classification only; it does not identify which individual episodes are real taps.

A better weak-label objective alone cannot make physically overlapping EMG shapes separable. Context across a whole recording has some signal, but that information cannot automatically be assigned to exact onsets.

## Design implications for V7 production labeler

1. Candidate generation must remain high-recall and contact-aware, per-zone, with independent rise/fall/re-arm evidence. **Missing candidate** is not a negative event.
2. Do **not** initialize positive event targets by copying the TAP/SWIPE source folder to every candidate. Train a multi-instance gesture pattern model on positive/negative bags; enforce hard FINGER rejection.
3. Model **several TAP prototypes**, including soft/normal and channel-specific patterns; compare support relative to a learned bank of FINGER/WRIST/ARM/BAND negative motifs, not generic event energy or self-recurrence.
4. Use complete temporal zone trajectories and dynamic baseline-relative features in the offline teacher, then distill to strictly causal firmware features only after independent validation.
5. Treat repeated motifs and Oculus/metronome timing as **corroboration only**, after negative-pattern exclusion. The Oculus pointer is missing, offset or occasionally stuck; never use it as unquestioned onset truth.
6. Use independent wearer **and date/session** splits and explicitly test contact failure and channel dropouts.
7. Emit 0 for trusted non-target folder epochs, 1 only for high-specificity selected target intervals and **−1 for ambiguous positive-bag periods**; never force an arbitrary positive event if there is no trustworthy physiological witness.
8. A **small manually/independently time-aligned physical reference set** is still needed to certify event-level accuracy. This is not a request to manually label all source data. Without it, genuine tap event precision, false activations/hour, and double-tap split/merge rates remain unknown.

## Release decision

**NO-GO for automatically certifying exact positive physical onset timestamps from V5/V7 patterns.** The pipeline and full research are reproducible and the 345-value waveform offers a measurable but modest ranking improvement. V5/V6 should remain frozen baselines; do not silently replace their selected event labels with another pseudo-labeler trained on the same unverified guesses.

## References

- Yang et al., *STCNet: Spatio-Temporal Cross Network with subject-aware contrastive learning for hand gesture recognition in surface EMG* (2025), https://pubmed.ncbi.nlm.nih.gov/39674068/
- Yuan et al., *Exploring pattern-specific components associated with hand gestures through different sEMG measures* (2024), https://doi.org/10.1186/s12984-024-01526-3
- *Comparison of sEMG Onset Detection Methods for Occupational Exoskeletons on Extensive Close-to-Application Data* (2024), https://pmc.ncbi.nlm.nih.gov/articles/PMC10885915/
- Liu et al., *Weakly-supervised temporal action localization using multi-branch attention weighting* (2024), https://doi.org/10.1007/s00530-024-01445-2

These published systems use different sensors/signals/datasets and cannot be numerically compared directly to these log-coded PSD feature recordings.

## Reproducibility

- [Full source-bound extraction and download](https://github.com/alex00Pirotskyi/ML/actions/runs/43866509437)
- [Source code on isolated V7 branch](https://github.com/alex00Pirotskyi/ML/blob/research/tap-pattern-v7-deep-20261116/research/tap_real/extract_psd_motifs_v7.py)
- [Previous V5 auto-labeler](https://github.com/alex00Pirotskyi/ML/blob/main/research/tap_real/auto_label_v5.py)
- [Previous V6 student Optuna baseline](https://github.com/alex00Pirotskyi/ML/blob/main/research/TAP_SWIPE_STUDENT_OPTUNA_V6_RESULTS_2026-11-16.md)


## Additional falsification: class-enriched unsupervised prototypes

We also tested a balanced, normalized EMG-pattern clustering approach rather than a single classifier: 24 or 40 compact morphology clusters were fitted on training groups; cluster candidates were retained if the cluster was more frequent in weak V5 TAP seed events than in FINGER events by factors of 1.5×, 2×, or 2.5×. Five-fold held-out participant strings were scored. The positive rows are still **unverified V5 pseudo labels**.

| Clusters | Minimum seed-vs-FINGER enrichment | V5 TAP pseudo-positive recall | Hard-negative FINGER candidate acceptance |
|---|---:|---:|---:|
| 24 | 1.5× | 18.3% | 15.5% |
| 24 | 2× | 2.9% | 7.6% |
| 40 | 1.5× | 36.9% | 37.8% |
| 40 | 2× | 16.4% | 19.4% |
| 40 | 2.5× | 7.6% | 13.7% |

The apparent positive-enrichment clusters **did not generalize as tap-specific**: for 40 clusters at 1.5×, 93.2% of TAP recordings but **96.5% of FINGER recordings** contained a matched prototype; for 40 clusters at 2×, TAP 61.4% and FINGER **64.9%**. This rejects a naive "unsupervised motifs + cluster enrichment" labeler. These figures are exploratory diagnostics, not false activation rates in firmware.

The primary **independent GitHub Actions reproduction** of the original full 2,790-source pattern experiment also passed:
[full V7 raw waveform ablation and source audit, run #43870045765](https://github.com/alex00Pirotskyi/ML/actions/runs/43870045765), [artifact 14226917060](https://github.com/alex00Pirotskyi/ML/actions/runs/43870045765/artifacts/14226917060); reproducible code is on the [research branch](https://github.com/alex00Pirotskyi/ML/blob/research/tap-pattern-v7-deep-20261116/research/tap_real/benchmark_psd_motifs_v7.py).

**Confirmed full benchmark values:** simple morphology TAP-vs-FINGER recording AUC **0.54419**, full 345-dimensional waveform plus morphology **0.62034**, paired participant-block bootstrap improvement 95% descriptive interval **−0.00290 to +0.14065** (includes no improvement). No physical event labels were introduced.


## Held-out group imbalance — interpretation safeguard

The 72,016-waveform corpus contains **56 normalized participant strings**, but their sizes are highly unequal: one alias (Ohad) contributes **15,430 waveforms, 387 recordings, 21.4% of all waveform events**. Under five-fold GroupKFold, that participant formed an entire fold alone; the other folds held out about 13–14 aliases each. This *prevents direct participant overlap*, but still makes the pooled AUC sensitive to a large single person's behavior and to unmerged aliases. The bootstrap results are descriptive, not formal prospective performance guarantees. In future work, optimize and report **macro-by-person/date metrics** and leave-date-out tests separately; use fully verified person IDs for group construction.
