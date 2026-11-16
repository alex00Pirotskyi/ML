# Mudra EMG V3 — robust rest, activity and contact pseudo-labeling research

**Date:** 2026-11-16  
**Dataset:** DATA/DATA_LONG_PRESS — all 2,431 .csv.gz recordings  
**Experiment branch:** research/lp-v3-rest-contact  
**Implementation:** lp_v3_teacher.py and lp_v3_labeler.py  
**Teacher version:** lp_offline_bistate_contact_v3  
**Status:** Fully implemented and exercised on all recordings. **Experimental weak labels, not independently verified ground truth.**

## Executive finding

We built and tested a more defensible weak-supervision mechanism that:

- Detects candidate resting and active states **inside** a recording, including sessions beginning during a contraction.
- Uses WL, logged EMG PSD, its instantaneous history slot, per-channel spectral and temporal changes, WL head/tail features and ACC motion features.
- Uses a person-held-out, contact-proxy model to avoid automatically calling strong muscle contractions disconnected.
- Preserves an **UNKNOWN** class and deliberately abstains on poorly placed, nonresponsive or ambiguous samples.
- Produces the nine requested continuous regression targets plus global binary lp_state and confidence weights.
- Leaves original recordings, V2 labels and firmware unchanged.
- Generates a review queue with file-level problems and time intervals, so a human can inspect difficult examples rather than silently trusting pseudo-labels.

**The most important caution:** No algorithm using only these CSV features can know true force-button release timestamps or definitively measure electrode contact. PointerState is unreliable, the folder label is only a weak session context, and some DISCONNECTED recordings retained a connected CH3. An empirically high teacher-imitation score must not be advertised as true physiological accuracy.

## 1. Source semantics and the complete input audit

The source firmware uses SNC_QUEUE_SIZE=256 filtered sEMG samples. Each WL feature is approximately 1000 times the sum of consecutive absolute differences within this window.

The frequency-domain output is encoded logarithmically:

    encoded_PSD_zone = clip_uint16(round(10000 * log10(1 + FFT_zone_sum)))

The user recorded with both EMG_PSD_MEAN=0 and ACC_PSD_MEAN=0, so bin-count division is disabled. The separate 15-entry history average in the emg_psd_mean fields remains intact; emg_psd[0] holds the newer instantaneous spectrum. V3 uses both and never divides the encoded PSD zones by FFT-bin counts.

All 2,431 compressed long-press source files were verified byte-for-byte identical after decompressing the prior and current commits: https://github.com/alex00Pirotskyi/ML/actions/runs/43821967044. Thus the user’s DATA NO PSD MEAN commit changed gzip representation, not numerical signal values. There was no new signal transformation to credit with a score gain.

Available feature families include:
- Three channels each of WL, WL head and WL tail.
- Three channels each of five encoded EMG PSD zones: instantaneous history index 0, history end, and 15-frame average.
- ACC PSD, ACC-rate and axis-by-zone summaries.
- Time, participant and task-context metadata.

No raw 24-bit sEMG voltage sequence, calibrated per-electrode impedance measurement or independent physical force-switch timestamps exist in these CSVs.

## 2. Why V2 could not be our final label teacher

Earlier V2 was causal and respected firmware log-PSD math, but:
- In strong-LP recordings it labeled only ~5.4% of rows as confident global resting.
- In the physical-contact weak-context test, 48 of 84 provisional disconnected-channel examples were missed.
- Several per-channel muscle-activity regression targets had low held-out-person R².
- Many recordings were poorly or ambiguously placed, and a quiet electrode could not be proven connected.
- The V2 binary student achieved 85.78% balanced accuracy against its own teacher on 2,431 recordings, **not against physical rest/release truth**.

A basic signal-processing limitation is that a physiological rest and sustained very weak contraction can be electrically indistinguishable if the recording contains no observed high/low transition or valid quiet calibration interval. Likewise, high, stationary electrode noise can resemble a sustained high contraction. V3 addresses this by preferring UNKNOWN over invented labels.

## 3. Experiment A: find low/high states independent of PointerState

We fitted one- and two-component Gaussian mixture models to approximately 180-ms robust windows of the feature pair

    X_ch(t) = [
      ln(1 + WL_ch(t)),
      median_over_5_PSD_zones(emg_psd_mean[ch,z](t))/10000
    ]

A credible two-state structure requires:
- Two-state Bayesian information criterion (BIC) improvement >10 versus one-state;
- both modes persistently occupied, roughly between 9% and 91%;
- higher mode has WL greater by >ln(1.4) and logged PSD greater by >0.018;
- both signal families agree on the direction of activation.

A second weaker tier permits small but still concordant splits; these receive lower confidence and more boundary uncertainty. This is an exploratory selection rule, not a universal sEMG threshold.

At least one channel with a credible two-state split:

| Recording context | Credible files |
|---|---:|
| Strong long press | **20/20** |
| Mid long press | **34/41** |
| Soft long press | **21/36** |
| Contraction long press | **4/4** |
| Drag/drop | 35/37 |
| Connected reference | **3/39** |
| Disconnected | 19/42 |
| Table | 3/9 |
| Sampled taps | 65/98 |
| Sampled swipes | 53/99 |

Source experiment: https://github.com/alex00Pirotskyi/ML/actions/runs/43827970309.

**What this actually demonstrates:** The strong-LP files often contain a low/high *electrical activity structure* that does not require PointerState to discover. It does NOT prove that every lower-energy interval is genuinely released, nor that every high-energy interval represents an intended LP. No physical force truth is present.

## 4. Experiment B: sensor contact and domain shift

Training labels (weak context proxies only):
- CONNECTED folder, all three channels: 117 nominally connected examples.
- DISCONNECTED folder, only CH1 and CH2: 84 provisionally disconnected examples.
- DISCONNECTED CH3 excluded from positive labels because it was not consistently physically disconnected.
- LP files are not treated as guaranteed good-contact sources.

Contact features combine per-channel WL percentiles, PSD distribution and shape, temporal modulation, PSD latest-versus-average differences, WL head/tail imbalance, and ACC-related changes.

**Fatal issue of the baseline:** A classifier trained only on relaxed connected/disconnected sessions flagged all 20/20 strong-LP recordings as having a disconnected-like channel, despite the possibility that the high amplitude was real contraction.

A person-grouped exploratory ablation added clearly responsive LP channel examples as downweighted nominally-good examples. They remain weak labels, not measured electrode truth.

| Training | Provisional contact balanced accuracy | Strong LP files flagged bad |
|---|---:|---:|
| Connected/disconnected only | 91.3% | 20/20 |
| + responsive LP examples, weight 0.12 | 91.2% | 8/20 |
| + responsive LP examples, weight 0.70 | 91.0% | 2/20 |

The integrated cross-fitted version uses a regularized logistic regression with weight 0.7. Across **201 weakly labeled channel examples, 14 connected-reference subject identifiers**, five held-out-person folds:

- ROC AUC **0.9605**;
- balanced accuracy **89.19%**;
- confusion matrix [normal, bad] = [[114,3],[16,68]];
- provisional disconnected-channel recall **68/84**;
- connected-reference channel specificity **114/117**.

The exact calibration differs slightly from the initial ablation because of person normalization and training details. These results are exploratory: weak folder labels and retrospective parameter selection limit independent validity. The reduction of bad-contact flags in LP files does **not** establish the electrodes were physically good.

Evidence: https://github.com/alex00Pirotskyi/ML/actions/runs/43828311747 and https://github.com/alex00Pirotskyi/ML/actions/runs/43831105557.

## 5. V3 labeling mechanism and its safety safeguards

### 5.1 Offline physiological segmentation

V3 deliberately makes the *teacher* noncausal: it can inspect the entire recording to find the lower-energy component even if the session began during a sustained contraction. The eventual embedded student MUST work only with current and past features.

For each channel:
1. Build WL/log-PSD windows and fit a two-state model only when separation is credible.
2. Identify low- and high-energy modes, compute a soft activity posterior, and use relative WL/PSD response as supporting evidence.
3. Apply limited temporal smoothing and mark segments near state changes as UNKNOWN, since precise press/release timing is unverified.
4. Use the current, unsmoothed emg_psd[0] alongside emg_psd_mean as weak transient evidence. An abrupt mismatch coinciding with a WL jump lowers label confidence.
5. Use WL head/tail imbalance and ACC-rate change only as artifact/timing warnings; arm movement is not automatically an electrode failure or non-gesture.
6. Allow single trustworthy responsive channels; do not force 2/3-channel majority agreement that loses soft activity.

### 5.2 Unresponsive and poor-contact channels

If a gesture recording never exhibits an independent low/high EMG anchor, a low steady signal is **not** proof of true rest: the user may be holding a soft contraction with very little detectable sEMG or using badly placed electrodes. Mark such channels UNKNOWN.

A dedicated CONNECTED or NOTHING context may weakly establish quiet rest if the signal is stable, the contact model supports it and amplitude is within plausible reference limits. These samples have limited training weight because the directory is not an independent time-resolved rest annotation.

For other channels in a recording with an established activity anchor, a low stable, valid-contact channel can be provisionally interpreted as resting while its companion channel provides timing context.

A nearly silent nonzero electrode is UNKNOWN unless external contact evidence exists; severe high-static-noise/flatline/quantizer clipping can support contact-artifact evidence but cannot definitively prove physical disconnect. Conflicting contact and muscle evidence increases UNKNOWN instead of forcing a diagnosis.

### 5.3 Output contract

Every original sample receives:
- lp_resting_ch1, lp_resting_ch2, lp_resting_ch3
- lp_high_activity_ch1, lp_high_activity_ch2, lp_high_activity_ch3
- lp_disconnection_ch1, lp_disconnection_ch2, lp_disconnection_ch3
- lp_unknown_ch1, lp_unknown_ch2, lp_unknown_ch3
- lp_state (strict binary, 1 means all three channels appear resting and adequately connected)
- lp_state_weight (0 to 1; uncertain zeros have near-zero weight)
- lp_teacher_version = lp_offline_bistate_contact_v3

The four per-channel scores sum to 1, up to rounding. These are **heuristic evidence scores** and must not be treated as calibrated probabilities. A binary 0 caused by UNKNOWN is NOT a verified active-muscle class.

V3 outputs a label_summary.json and review_queue.json. The latter identifies suspicious files and segments with start/end times and dominant unknown channel for manual review.

## 6. Experiment C: audit all 2,431 original recordings

The V3 dry-run completed with:
- **2,431/2,431 files processed**, zero errors.
- **25 short/insufficiently calibrated recordings** retained and handled conservatively, rather than silently dropped.
- no writes to the original recordings or previous V2 labels.

V3 teacher fraction of rows marked confident GLOBAL rest:

| Context | Files | V3 rest fraction | Files with any rest label | V3 uncertain fraction |
|---|---:|---:|---:|---:|
| CONNECTED | 39 | 74.41% | 31 | 16.38% |
| NOTHING | 5 | 76.39% | 5 | 9.12% |
| DISCONNECTED | 42 | **0%** | **0** | 70.81% |
| TABLE | 9 | **0%** | **0** | 44.44% |
| LOOSE | 42 | 9.91% | 7 | 78.22% |
| SOFT_LONG_PRESS | 36 | 16.86% | 17 | 65.41% |
| MID_LONG_PRESS | 41 | 14.63% | 26 | 55.21% |
| STRONG_LONG_PRESS | 20 | 8.00% | 11 | 55.69% |
| LONG_PRESS | 4 | 17.24% | 4 | 45.25% |
| DRAG_AND_DROP | 37 | 13.71% | 23 | 61.29% |
| TAP | 1,231 | 4.20% | 291 | 90.29% |
| SWIPE | 925 | 1.71% | 117 | 94.59% |

**These numbers are label-coverage distributions, NOT true resting-time proportions, and should never be reported as physiological detection recall.** In many TAP/SWIPE examples the teacher rightly abstains. In 9/20 STRONG_LONG_PRESS files, no defensible all-channel resting interval emerged, even though the recording may physically include rest.

Evidence: https://github.com/alex00Pirotskyi/ML/actions/runs/43831105557.

### Comparison to V2

Against the same recordings, V2 produced ~5.43% rest rows in STRONG_LONG_PRESS and 3.43% in DISCONNECTED sessions. V3 produced ~8.0% and 0%, respectively, while generally increasing explicit uncertainty in LP and gesture sessions.

**This is not proof that V3 is more correct.** It may exclude valid rest in some files; it may still mistake subtle activity for rest. The value of V3 is an interpretable segmentation method plus abstention and a review queue, not an unverified score increase.

## 7. Experiment D: can a causal student reproduce V3 labels?

### 7.1 Standard benchmark

We trained LightGBM on approximately 475 recordings, with 38,000 sampled rows and three person-held-out folds, using existing causal feature history. The task was to IMITATE V3 weak pseudo-labels, not physical press/release truth.

On 37,760 successfully sampled rows across 472 files / 56 normalized person identifiers:

| Learner | Balanced accuracy | Rest F1 | ROC AUC |
|---|---:|---:|---:|
| Causal LightGBM (44 features) | **73.74%** | 0.514 | 0.869 |
| Always predict binary 0 | 50.00% | 0 | approximately 0.5 |

This is lower than the previously reported V2-teacher imitation score (85.75%) on a similarly sampled benchmark. **The targets are different**, and V3 intentionally abstains more often. The score difference is *not* a direct measure of physical label quality. It exposes a real *deployability challenge*: a causal model cannot necessarily reproduce an offline teacher that uses future session evidence.

Evidence: https://github.com/alex00Pirotskyi/ML/actions/runs/43830210845.

### 7.2 Past-only, multi-timescale features and confidence selection

A second study added ~168 total features:
- rolling/current WL and logged-PSD, several past-only EMAs over 0.25–20 seconds;
- asymmetric online floor estimates (fast downward, extremely slow upward);
- spectral history/current discrepancy;
- WL head/tail and ACC motion;
- per-channel cross-signal and shape features.

The same 37,760-row, 56-person, three-fold grouping was used.

| Student training | Balanced accuracy on ALL V3 rows | Rest F1 on ALL rows |
|---|---:|---:|
| 44-feature causal baseline (matched study) | 74.24% | 0.524 |
| 168-feature multi-timescale causal model | 74.93% | 0.551 |
| Multi-timescale student TRAINED on high-confidence teacher rows | **80.39%** | 0.467 |

The high-confidence training variant had high resting recall but many resting false positives on all uncertain rows. Therefore it should NOT be advertised as an all-samples finished detector.

**Selective evaluation is the useful result.** On teacher rows with lp_state_weight >=0.65, covering **25.3% of all samples**, the high-confidence-trained student achieved:

- **90.25% balanced accuracy** against V3 pseudo-labels;
- **0.905 rest F1** against V3 pseudo-labels;
- **25.3% selective data coverage**.

On all rows, however, that same classifier scores only 80.39% balanced accuracy and rest F1 0.467. The reported 90% result is **conditional on a confidence filter defined by the teacher**; it is not independently measured real-world release accuracy.

**Engineering choice:** For the first student experiment, train primarily on high-confidence samples rather than treating UNKNOWN=0 as a true negative. For all-row real-time deployment, build a separate causal confidence/quality or abstention mechanism and measure selective risk on independent truth.

### 7.3 Channel-level regression learnability

A one-fold experiment with the longer causal features found substantial variation:
- Rest-score R² around 0.26–0.32 on some channels.
- High-activity R² around 0.06–0.44, weakest on CH3.
- Channel-specific confidence weighting improved CH3 disconnection-score regression R² from about 0.11 to **0.50**, whereas other target heads changed modestly.

This is not sufficient to claim high-quality nine-target physical prediction. Treat each channel as its own confidence/training problem, and do not blindly reuse the binary global weight for all nine regression heads.

Evidence: https://github.com/alex00Pirotskyi/ML/actions/runs/43831059618.

## 8. What still cannot be inferred reliably

1. **True force release / long-press state.** The sEMG amplitude may remain high due gripping after a button releases, or a soft contraction may resemble rest.
2. **Physical electrode connection.** EMG features cannot uniquely identify impedance or open lead-off. A confident disconnect score is evidence of a pattern, not contact truth.
3. **Low-effort contraction vs rest without an activity reference.** If no high/low split or independently measured rest exists, the system must abstain rather than invent labels.
4. **Exact EMG onset/offset timing.** The firmware history-mean PSD and 180-ms aggregation smear boundaries; pointer state itself has known offsets/FP/FN. V3 explicitly lowers confidence around transitions.
5. **Generalization to truly new hardware/user populations.** The limited connected/disconnected reference protocol has only ~14 usable person identifiers, and the current experiments cannot fully separate collection campaign/domain shift from physiology.
6. **Good-vs-bad electrode for CH3 in disconnected captures.** Channel-specific truth must come from independent measurements.

These are scientific identifiability limitations, not algorithmic excuses. Claiming 98–99% accuracy from self-generated pseudo-labels without physical truth would be misleading.

## 9. Independent sEMG / sensor literature

1. **Li, Zhou & Aruin (2007), Teager-Kaiser EMG onset detection** — TKEO can improve onset detection when raw sEMG is available. Our historical CSVs do not contain the original full raw waveform; reconstructing TKEO from WL and five logged PSD zones is not justified. https://pubmed.ncbi.nlm.nih.gov/17473984/
2. **Comparison of sEMG onset detection methods for occupational exoskeletons (2024)** — tests 55 participants with independent reference assessment and emphasizes false-alarm/latency thresholds. Its quoted sensitivities are not transferable to Mudra. https://pmc.ncbi.nlm.nih.gov/articles/PMC10885915/
3. **Systematic investigation of low-SNR EMG detectors (2024)** — demonstrates explicit evaluation of false positives during verified rest, missed active episodes and detection latency. https://pmc.ncbi.nlm.nih.gov/articles/PMC10997989/
4. **Long-term electrode-skin impedance variation (2023)** — the contact interface and noise floor vary over time and across electrodes, so absolute amplitude alone is not enough. https://pmc.ncbi.nlm.nih.gov/articles/PMC10610867/
5. **Review of noise and artifacts in single-channel EMG (2023)** — motion artifacts, poor contact and interference can create false muscle features; quality needs separate handling. https://pmc.ncbi.nlm.nih.gov/articles/PMC10059683/
6. **Analysis and biophysics of surface EMG (2020)** — amplitude, electrode placement and filtering depend on subject and setup. https://pmc.ncbi.nlm.nih.gov/articles/PMC7594523/
7. **TI ADS1293 datasheet** — ADS1293 supports DC and AC lead-off diagnostics. The AC mechanism is potentially relevant to dry/capacitive electrodes, but must be checked for Mudra's actual reference routing, firmware, interference and safety before enabling it. https://www.ti.com/lit/ds/symlink/ads1293.pdf

## 10. How to use the experimental labeler

From the repository root after fetching this research branch:

    git fetch origin
    git switch research/lp-v3-rest-contact
    python -m pip install numpy pandas scipy scikit-learn
    python lp_v3_labeler.py

The default output is DATA/DATA_LONG_PRESS_LABELED_V3, a separate directory with the original folder structure and the additional 15 label/weight/version columns. The original gzip recordings are never modified. Existing outputs are skipped unless overwrite is explicitly requested.

To verify without writing any labeled CSV:

    python lp_v3_labeler.py --dry-run

To test two source files from each folder:

    python lp_v3_labeler.py --limit-per-group 2 --dry-run

Use the resulting label_summary.json and review_queue.json. Filter on lp_state_weight for high-confidence binary teacher training and on each channel's lp_unknown_chN score for its individual regression target; avoid treating uncertain zeros as confirmed contraction.

**The V3 teacher uses future data and must not be dropped directly into firmware as an online decision rule.** Train a separate causal model with only firmware-realizable feature history, then validate independently.

## 11. Prioritized next experiments

1. **Instrument a small truth set.** Independently synchronize a physical press switch/force measurement with sEMG, plus verify actual electrode lift-off/re-attachment per CH. Include already-active starts, quiet rest, soft/mid/strong holds, tap/swipe/drag and movement artifacts.
2. **Review difficult files first.** Use review_queue.json for uncertain intervals, especially the 9/20 strong LP files without any confident global rest and the 19/36 soft LP files with no confident all-channel rest.
3. **Test hardware lead-off/impedance.** The ADS1293 AC lead-off approach may provide better objective contact truth if compatible with Mudra wiring; do not enable it blindly on shipped hardware.
4. **Select an embedded feature subset.** The 168-feature experiment is useful for identifying long-history information but is not a recommended nRF52832 model as-is. Use group-held-out ablations, modest EMAs and compact trees.
5. **Optimize selective physical error**, not pseudo-label F1: false-REST while force held, false-ACTIVE during verified rest, time-to-rest after release, false-disconnected under verified good contact, abstention rate.
6. **Hold out people AND recording campaigns** when tuning model or teacher parameters. Some historical folders come from different collection protocols and contain imperfect contact metadata.

## 12. Reproducibility and GitHub evidence

- V3 source: https://github.com/alex00Pirotskyi/ML/blob/research/lp-v3-rest-contact/lp_v3_teacher.py
- V3 labeling entrypoint: https://github.com/alex00Pirotskyi/ML/blob/research/lp-v3-rest-contact/lp_v3_labeler.py
- All-folder feature research and GMM: https://github.com/alex00Pirotskyi/ML/actions/runs/43827970309
- Contact augmentation / held-out participants: https://github.com/alex00Pirotskyi/ML/actions/runs/43828311747
- 2,431-recording full dry-run: https://github.com/alex00Pirotskyi/ML/actions/runs/43831105557
- V3 versus V2 observed teacher outputs: https://github.com/alex00Pirotskyi/ML/actions/runs/43829993556
- Causal student test: https://github.com/alex00Pirotskyi/ML/actions/runs/43830210845
- Multiscale causal and high-confidence selective evaluation: https://github.com/alex00Pirotskyi/ML/actions/runs/43831059618

### Final recommendation

V3 is a meaningful research improvement in **segmentation, contact-domain robustness, transparent uncertainty, and reviewability**, not a validated finished physical-ground-truth system. **Use its high-confidence windows as experimental teacher labels and abstain on the rest.** The most valuable next leap is not another complex model trained on the same weak labels, but an independently instrumented study that breaks the physical ambiguities between rest, weak effort, and electrode contact failure.
