# sEMG Rest, Muscle Activity, and Electrode Disconnection
## Deep experimental research on Mudra long-press recordings

**Date:** 2026-11-15  
**Dataset:** DATA/DATA_LONG_PRESS on repository alex00Pirotskyi/ML  
**Research branch:** research/lp-rest-signatures  
**Status:** Completed exploratory dataset audit and multiple recorded-data experiments; **no firmware-ready classifier validated**.  
**Authoring principle:** Treat this as a fresh investigation. Prior long-press models, thresholds, pointer labels, and assumptions are not authoritative.

> **Central conclusion:** The strongest generalizable evidence for low versus high muscle activity is **change relative to an established, reliable, per-channel resting baseline**. Electrode disconnection requires a **separate contact-quality assessment**; a high or a low EMG amplitude alone is NOT enough. However, no available recording provides independent, trustworthy press/release timestamps or channel-level impedance truth, so exact release detection and clinical-quality electrode identification cannot yet be verified.

---

# 1. Executive summary

We analyzed the full available set of 2,431 compressed recordings at the dataset level and ran targeted temporal, spectral, contact, and unsupervised experiments.

**What worked:**

1. **Strong physical-contact signature:** In 39 *matched person-and-orientation* connected-versus-disconnected recording pairs from 13 people, median waveform length increased by **10.24× on CH1** and **11.00× on CH2** under disconnected-labelled conditions; CH3 instead fell to **0.234×**. Corresponding spectral-power median folds were **17.73×, 31.01×, and 0.0575×**. Channel effects are markedly asymmetric.
2. **A held-out-person file-level connected/disconnected classifier** achieved **0.997 ROC AUC / 98.8% balanced accuracy** on 81 context-labelled recordings. A provisional **channel-level** model achieved **0.976 AUC / 92.8% balanced accuracy** on 201 channel examples, excluding CH3 from the disconnected-positive labels.
3. **A conservative hand-designed noisy-electrode marker** (recording median WL above 5,000 AND WL 90th/10th percentile ratio below 1.5 on at least one channel) flagged **32/42 disconnected contexts, 0/39 connected contexts, 0/101 long-press contexts, 15/42 loose contexts, and 7/9 table contexts**. **This rule was selected while looking at the same data; it is a hypothesis, not independently validated accuracy.**
4. **Activity commonly creates a relative rise in total EMG spectral power**, typically across several bands. Across 101 long-press recordings, median high-WL/low-WL PSD-power ratios were **2.77×, 4.30×, and 1.96×** for CH1–3. No single PSD zone emerged as a universal resting marker.
5. **Unsupervised low/high separation exists in many sessions:** a two-component Gaussian-mixture analysis with occupancy and separation requirements identified a credible two-state structure in **86/101 long-press sessions**, versus **3/39 connected controls**. But it also identified **83/90 sampled taps** and **79/96 sampled swipes**: an activity detector is not a long-press-intent detector.

**What failed or remains risky:**

6. **Contact model cross-domain failure:** A model trained only on relaxed connected-versus-disconnected conditions marked at least one channel as disconnected in **20/20 strong long-press recordings** in one stress experiment. Another architecture flagged **19/20**. These are **unverified contact flags**, not established false positives, because actual placement varies and data campaigns differ. They prove that a high cross-validation score within one contact protocol is not enough.
7. Adding weakly labelled long-press channels as nominally connected examples **reduced held-out-person strong-LP disconnection flags from 85.0% to 16.7% of channels**, but also reduced genuine connected/disconnected proxy balanced accuracy from **89.3% to 85.6%**. This is a trade-off, **not proof of correct physical contact**.
8. **A strict two-of-three-channel rule loses soft activity:** under one one-second online-like temporal test, single-channel WL+PSD criterion flagged **17/36 soft LP files**, versus **13/36** for the two-channel criterion (different threshold settings noted below). Physically attached electrodes can be unresponsive to a given action.
9. **Timestamp and label limitations are material:** PointerState and TriggerPressure are not independent annotations. In 27/101 long-press recordings, pointer-on windows had lower relative WL-derived activity than pointer-off windows. Some recording types contain duplicated millisecond timestamps; the data come from different collection campaigns.
10. **No exact ground-truth high/low/release performance can be claimed**, because class_id is a recording context, not a verified per-window muscle-state annotation, and contact quality was not measured independently.

**Recommended direction:** A multi-stage, per-channel **QUALITY → CALIBRATED ACTIVITY → TEMPORAL STATE** system with an explicit UNKNOWN state, plus new independent annotations and controlled electrode lift-off trials. Use simple interpretable quality diagnostics as a candidate fallback while collecting better truth.

---

# 2. Definitions: three different questions, not one classifier

The goal is to distinguish:

| Concept | Physical meaning | What sEMG can observe | What sEMG cannot guarantee |
|---|---|---|---|
| **Low muscle activity / rest candidate** | Recorded forearm muscles are electrically quiet relative to their established baseline | Low and stable WL/PSD *relative to reliable prior rest* | The user is deliberately not pressing, or all electrodes are actually connected |
| **High muscle activity** | More muscle electrical activity than local baseline, sometimes sustained | Relative WL and broadband spectral-power increase, possibly one channel only | A keyboard/mouse press is occurring; motion/other contractions can look similar |
| **Disconnected / bad electrode** | Electrode-skin coupling is absent or unreliable | Abnormal baseline, spectrum, instability, flatline, inter-channel asymmetry and/or failed response | Definitive physical disconnection without impedance/lead-off or independent contact measurements |
| **Unknown** | Baseline or sensor quality is insufficient | Confidence is low or contradictory | Must not be forcibly classified as REST |
| **Pointer press/release** | UI/firmware/application action | PointerState and TriggerPressure columns reflect system behavior | Not trustworthy independent supervision in these recordings |

These distinctions matter. A wrist held still while muscles actively grip may have **low variance at high energy**. A disconnected electrode may have **high stationary noise**. A weak signal may be **genuine relaxation or missing contact**. Conversely, physiological activation during a swipe is real muscle activity but not necessarily a long-press action.

The output needed for robust physiology is therefore best represented as **per-channel contact confidence plus low/high/unknown activity**, not a single opaque “pressed or not pressed” score.

---

# 3. Data inventory, scope, and provenance

All measurements below were computed from the compressed CSV recordings on the research branch, without editing their bytes. The production main branch and firmware were not changed.

## 3.1 Recordings

| Folder / recording context | CSV.gz files |
|---|---:|
| TAP | 1,231 |
| SWIPE | 925 |
| MID_LONG_PRESS | 41 |
| SOFT_LONG_PRESS | 36 |
| STRONG_LONG_PRESS | 20 |
| LONG_PRESS / contraction LP | 4 |
| DRAG_AND_DROP | 37 |
| CONNECTED | 39 |
| DISCONNECTED | 42 |
| LOOSE (includes electrode-on-top cases) | 42 |
| NOTHING | 5 |
| TABLE (including powered conditions) | 9 |
| **Total** | **2,431** |

The comprehensive file-level audit successfully extracted **2,429** recordings. Two very short TAP files could not meet that script’s minimum duration. The temporal benchmark required at least 12 quarter-second windows and used **2,358** files; **73** were excluded primarily because they were too short. The unsupervised experiment used **all 101 LP recordings**, all 39 connected references, selected contact controls, and sampled taps/swipes; it successfully analyzed **454** recordings. Every result below states its relevant denominator.

The LP cohorts are small: **36 soft**, **41 mid**, **20 strong**, **4 contraction**. The explicit NOTHING set has only **5** files, and TABLE has **9** files from **2 user identifiers**. Do not over-generalize from these small categories.

## 3.2 Schemas and time bases

Three full CSV schema variants exist: **134, 135, and 139 columns**. They differ mainly in metadata such as Hand and additional time fields. The analysis selected only relevant signal columns; some audit logs therefore show smaller selected-column counts (47, 48, or 50), **not** different physical data schemas.

The median per-recording Timestamp step varied:

| Context | Median timestamp step |
|---|---:|
| MID / SOFT LP | ~6 ms |
| STRONG LP | ~7 ms |
| Connected / disconnected / loose | ~8 ms |
| TABLE | ~13 ms |
| TAP | ~6 ms |
| SWIPE | ~5 ms |

In the September contact-protocol recordings, the median fraction of consecutive **identical millisecond Timestamps** was approximately **4%**; these timestamps are coarser than the underlying sampling and should not automatically be interpreted as lost samples. The newest schema includes MonotonicTimeUs and SampleIndex, which should be preferred when available after validating monotonicity and units.

**Engineering consequence:** Keep all temporal thresholds in milliseconds or seconds, not hard-coded sample counts. Record effective rate explicitly; verify sample-rate changes by acquisition pipeline.

## 3.3 Feature families actually available

- **WL (waveform length):** three EMG channels, along with head/tail subfeatures.
- **EMG PSD:** five zones per channel across three spectral snapshots/groups, including the mean group.
- **Accelerometer-derived features:** rate and PSD summaries for X/Y/Z, not necessarily raw calibrated acceleration vectors.
- **Metadata:** User, hand/orientation in some files, Timestamp, PointerState, TriggerPressure, class_id, other optional timing fields.
- **Unavailable for this study:** synchronized optical ground truth of presses/releases, measured fingertip force, actual electrode impedance, lead-off hardware diagnostic, and the raw sEMG waveform used to calculate the PSD.

For exploratory spectral totals, the scripts divide zone sums by the bin counts **[8, 10, 12, 46, 51]**, following psd_mean.py. **Important conversion limitation:** recordings lack the new psd_mean_applied provenance flag. If an older unmarked file had already been converted, applying the divisors again changes absolute powers and spectral weighting. Future data acquisition MUST record PSD normalization mode explicitly; repeat the comparisons on verified raw-versus-normalized cohorts before locking firmware constants.

---

# 4. Ground-truth and label audit

**Do not train release-state labels from PointerState by default.**

- **class_id = 1** appears throughout LP, TAP, SWIPE, DRAG_AND_DROP, and CONNECTED recordings. It cannot mean “pressed now.”
- Connected, disconnected and loose capture folders mainly use class_id **1**, **2**, and **3** respectively; some LOOSE recordings contain **4/5**, TABLE contains **6/7**, and NOTHING uses **0**. These are recording **contexts**, not verified instantaneous physiological states.
- Thresholded PointerState and TriggerPressure **agreed in all analyzed files** at >0.5 in this audit. They should not be treated as two independent truth sources.
- In **27/101** LP recordings, the median relative-WL activity during pointer-on windows was lower than during pointer-off windows. These differences do not prove the pointer timestamps are uniformly wrong; they show that the pointer signal and muscle activity do not align reliably enough to support ground-truth inference.
- File-level placement labels are incomplete. In the DISCONNECTED experiment, **CH3 remained physically connected in some recordings** despite the recording’s directory label. Hence CH3 was NOT counted as a “disconnected” channel in supervised channel-target experiments. This does NOT prove CH3 always has good contact.

### Distinct sources of uncertainty

1. **Physiology:** recruitment and activation vary by individual, muscle, task, hand, strength, electrode placement.
2. **Hardware contact:** an electrode can be tight, loose, disconnected, touching another conductor, or on a table.
3. **Measurement:** PSD bin normalization, time stamps, sensor versions, filtering, firmware state, saturation.
4. **Labels:** filenames and application PointerState need not represent current physiological state.
5. **Domain shift:** contact-protocol data are largely September captures, while many LP recordings are from earlier campaigns.

Any high training score can be driven by some combination of these factors. We explicitly tested cross-condition behavior rather than only reporting training accuracy.

---

# 5. Experimental methods

## 5.1 Audits and evaluation units

**File-level audit:** 2,429 usable files. Recorded summary statistics on each channel: WL 10th/50th/90th percentiles, total normalized PSD power, PSD shape, WL variability, cross-channel and amplitude imbalance, metadata and timing quality.

**Contact comparisons:** 39 paired connected/disconnected captures, matched by normalized User and horizontal/vertical/reverse orientation, spanning 13 people. Use within-pair fold-change, not across-person raw amplitude comparison.

**Contact model:** 5-fold **GroupKFold**, grouped by normalized User. All orientations and files for the same normalized identifier are in the same fold. File classifier positive class = disconnected folder; channel classifier positive = **CH1/CH2 in disconnected folder only**, while all three channels in CONNECTED are provisional connected examples. This is weak supervision, not electrode-impedance ground truth.

**Temporal benchmark:** 250 ms windows (quantized to the available sample cadence); base WL/PSD is the 20th percentile of the **first two seconds**, for a causal-like initialization. A separate **oracle20** benchmark uses the 20th percentile of the **entire file**, which is NONCAUSAL and can falsely look better.

**Spectral benchmark:** Compare high-WL and low-WL 0.5-second window quintiles *within recording*. Both high and low groups are retrospective amplitude proxies; they are not independently labelled press/rest.

**Unsupervised benchmark:** For each recording, fit one- versus two-component diagonal Gaussian mixtures to within-file normalized log WL and PSD features. “Credible two-state” demanded BIC improvement >10, each state occupying 10–90% of windows, and average log-energy separation >ln(1.4). This is model selection for signal structure, NOT event-level activity accuracy.

## 5.2 Why these numbers are not final classifier accuracy

An LP folder means **the recording session was an LP task**, not that every sample is a true press. A file flagged as containing sustained activity is therefore “LP recording coverage,” not sensitivity to the actual hold intervals. Conversely, TAP/SWIPE recordings may contain prolonged genuine muscle activation, so their flags are not automatically “false presses.” Release latency cannot be measured without independent reference timestamps.

The only ROC AUC values in this study apply to **session-context or provisional channel-contact labels** and are subject to label imperfection and domain shift.

---

# 6. Experiment A — what changes physically during disconnection?

Matched across the same user and orientation, disconnected-labelled versus connected-labelled recording medians gave:

| Measure | CH1 | CH2 | CH3 |
|---|---:|---:|---:|
| Median WL fold | **10.24×** | **11.00×** | **0.234×** |
| 10th–90th percentile range of paired WL fold | 1.76–28.48× | 2.20–31.79× | 0.034–2.22× |
| Median total PSD-power fold | **17.73×** | **31.01×** | **0.0575×** |
| Pairs with WL increasing | 37/39 | 37/39 | 10/39 |

**Interpretation:** The affected channels can show dramatically **higher** apparent sEMG activity when contact fails; this looks more like interference/artifact than a simple loss of amplitude. Other channels may drop sharply. Thus “disconnected = WL near zero” is a poor general rule.

The CH3 asymmetry also demonstrates why the labels must be **channel specific**. A channel physically left attached can still be influenced by changing the rest of the electrode array and reference conditions.

**Hypothesis A1 (supported by these captures):** A combination of unexpectedly high WL/PSD with unusually little temporal modulation is a useful marker of severe electrode artifact. It is not enough to catch quiet open circuits or every kind of loose contact.

**Hypothesis A2 (not independently verified):** Frequency distribution and response to voluntary calibration motion may distinguish quiet disconnection from genuine resting contact. Without a hardware impedance reference this remains probabilistic.

---

# 7. Experiment B — what can supervised contact models learn?

## 7.1 Participant-held-out contact results

| Task | Model | Samples | ROC AUC | Balanced accuracy | Confusion matrix [normal,bad] |
|---|---|---:|---:|---:|---|
| File context: CONNECTED vs DISCONNECTED | Random forest | 81 files, 14 users | **0.997** | **98.8%** | [[39,0],[1,41]] |
| Channel proxy: connected CH1–3 vs disconnected CH1–2 | Random forest | 201 channel examples, 14 users | **0.976** | **92.8%** | [[110,7],[7,77]] |
| Same channel proxy | Logistic regression | 201 | 0.968 | 91.0% | [[107,10],[8,76]] |

The high within-protocol results are genuine **held-out-user evaluations**, but are not a license to deploy. Contact sessions come from a fairly narrow protocol and era.

## 7.2 Which feature families mattered?

A separate 201-channel held-out-person experiment compared feature subsets (logistic unless indicated):

| Channel features | ROC AUC | Balanced accuracy |
|---|---:|---:|
| WL median only | 0.935 | 84.7% |
| PSD-power median only | 0.944 | 85.4% |
| All amplitude percentiles | 0.951 | 86.4% |
| Time variability only | 0.876 | 82.0% |
| Spectral zone fractions and entropy only | 0.915 | 85.8% |
| WL head/tail ratio + WL-to-PSD correlation | 0.671 | 65.0% |
| All features (logistic) | 0.964 | 89.9% |
| All features (forest) | 0.969 | 91.1% |

These ablation features differ slightly from the first full contact model, so the rows are informative comparisons **within that experiment**, not direct ranking across two feature-extraction implementations.

**Important finding:** Spectral **shape** alone retains meaningful contact-context information even without absolute amplitude. It is worth retaining a few zone-fraction or low-zone-ratio features in a compact embedded model, but it must be tested across sensing revisions and physiological activity.

---

# 8. Experiment C — contact model failure under strong contractions

A contact-only forest trained on CONNECTED/DISCONNECTED recordings did **not** generalize as a reliable physical-contact detector to LP activity.

- In one in-sample cross-condition stress test, **20/20 strong-LP recordings** had at least one channel flagged disconnected.
- A second implementation flagged **19/20** strong LP as having at least one bad channel.
- These recordings were not independently inspected electrode-by-electrode, so some may genuinely have poor placement. The important conclusion is **poor portability and untrustworthy interpretation**, not that all flags are necessarily wrong.

A controlled held-out-person comparison trained two models with identical test groups:

| Held-out-person diagnostic | Contact-only training | Contact + weak LP-as-connected training |
|---|---:|---:|
| AUC on 201 labelled contact-channel proxy examples | 0.972 | 0.915 |
| Balanced accuracy on those contact labels | 89.3% | 85.6% |
| Strong-LP channels flagged disconnected | **85.0%** | **16.7%** |
| Mid-LP channels flagged disconnected | 50.4% | 6.5% |
| Soft-LP channels flagged disconnected | 20.4% | 10.2% |
| Drag/drop channels flagged disconnected | 46.8% | 9.9% |
| Disconnected proxy channels correctly flagged | 71/84 | 67/84 |

**Interpretation:** Adding active-muscle examples reduces confusion with muscle activation but harms some of the contact classification. We deliberately marked the LP-as-connected labels **weak**, since placement was not confirmed. These numbers show a trade-off between model behaviors, not that the augmented model became a definitive quality monitor.

**Recommended resolution:** Collect new sessions with **simultaneously verified contact state and independently measured muscle activity**, including high effort and controlled electrode lift-off. Train contact quality with both resting and active **good-contact** examples, and an explicit UNKNOWN region, rather than silently assuming LP files have valid placement.

---

# 9. Experiment D — test simple, interpretable bad-contact heuristics

The following rules were examined over the existing recordings. All use **per-recording channel summary statistics**, and are therefore offline. They must be revalidated on sliding windows before firmware use.

| Rule: ANY channel qualifies | DISCONNECTED | CONNECTED | SOFT LP | MID LP | STRONG LP | TABLE |
|---|---:|---:|---:|---:|---:|---:|
| Median WL >3,000 | 38/42 | 5/39 | 8/36 | 8/41 | 7/20 | 9/9 |
| Median WL >5,000 | 33/42 | 0/39 | 5/36 | 6/41 | 5/20 | 8/9 |
| WL >3,000 and WL(P90)/WL(P10) <1.5 | 34/42 | 5/39 | 1/36 | 0/41 | 0/20 | 8/9 |
| **WL >5,000 and WL(P90)/WL(P10) <1.5** | **32/42** | **0/39** | **0/36** | **0/41** | **0/20** | **7/9** |
| WL >3,000, stable, low-zone share >0.34 | 27/42 | 0/39 | 1/36 | 0/41 | 0/20 | 6/9 |

The last two rules were **selected after seeing this dataset**, so all apparent specificity numbers are in-sample and optimistic. The 5,000 WL cut-off is in dataset-specific device feature units, not a transferable physiological unit.

**Candidate high-confidence severe-artifact indicator:**

    noisy_static_channel =
        median(WL over stable multi-second window) > 5000
        AND (P90(WL)+1)/(P10(WL)+1) < 1.5

It is an indicator of **some severe high-noise states**, not a complete disconnection classifier. Quiet open electrodes, intermittent contact, motion-corrupted pads, and bad reference wiring require additional tests. It should never be used as proof that a high-amplitude responsive channel is disconnected.

A soft fusion of amplitude, temporal modulation, and spectral shape is more promising than an absolute-only threshold.

---

# 10. Experiment E — what does EMG spectral analysis actually tell us?

Normalized per-zone PSD is useful but is highly coupled to WL.

Median **high-WL window / low-WL window total PSD power**:

| Session context | CH1 | CH2 | CH3 |
|---|---:|---:|---:|
| All LP (101) | 2.77× | 4.30× | 1.96× |
| MID LP (41) | 3.37× | 5.19× | 2.15× |
| SOFT LP (36) | 2.05× | 3.28× | 1.60× |
| STRONG LP (20) | 5.32× | 4.00× | 2.27× |
| CONNECTED (39) | 1.15× | 1.19× | 1.15× |
| TABLE (9) | 1.05× | 1.02× | 1.02× |

Within LP sessions, median log-scale WL/total-power correlation was approximately **0.76–0.86** by channel. The five zone gains generally moved in the same direction. Relative PSD *shape* drift between high- and low-WL windows was comparatively small for ordinary LP recordings (roughly 0.05–0.09 median total-variation distance by channel, depending on strength); this is not a clear standalone rest signature.

Examples from the independent per-zone experiment:

- SOFT LP: CH1 zone gains approximately **2.21, 2.00, 1.95, 1.73, 1.88×**.
- MID LP: CH2 zone gains approximately **4.80, 4.07, 5.54, 4.49, 4.15×**.
- CONNECTED reference: per-zone gains generally around **1.03–1.24×**.

**Hypothesis E1 (supported):** The broad rise in physiological muscle activity is more robust than any particular zone’s response. Use WL/total PSD to estimate **energy above baseline**, not a special “resting frequency.”

**Hypothesis E2 (worth testing):** Low-index-zone fraction or spectral entropy, especially during *steady* high-amplitude signals, adds useful contact-artifact evidence. Shape-only held-out contact AUC around 0.915 supports this as a feature family, but is not proof of a universally correct band.

**Non-result:** We cannot identify literal 50/60 Hz power-line interference from these five coarse zone totals without confirming actual sensor sampling rate, mapping of spectral bins to frequency, frontend filtering, and access to finer spectrum or raw samples. Do not label a specific zone as 50 Hz solely from its index.

---

# 11. Experiment F — temporal activity detection and its trade-offs

## 11.1 What was tested

- 250 ms windows computed with each recording’s observed median time step.
- Baseline per channel = 20th percentile WL and PSD power during the **first 2 s**.
- A channel is active if normalized energy exceeds a tested ratio.
- An event is counted if the required number of channels remain above the criterion for a duration.
- Reporting unit = **recording contains at least one such event**. This is NOT true per-event press recall.

For the thresholds shown below, simultaneous WL and PSD conditions use the SAME channel but do **not** represent independent evidence.

## 11.2 Representative first-two-second baseline results

| Criterion | SOFT LP (36) | MID LP (41) | STRONG LP (20) | CONNECTED (39) | DISCONNECTED (42) | TAP (1,169) |
|---|---:|---:|---:|---:|---:|---:|
| Any 1 channel WL ≥1.6× for ≥1 s | 18 | 36 | 15 | 1 | 6 | 679 |
| ≥2 channels WL ≥1.4× for ≥1 s | 16 | 34 | 13 | 1 | 0 | 546 |
| Any 1 channel WL & PSD ≥1.6× for ≥1 s | 17 | 36 | 14 | 1 | 3 | 628 |
| ≥2 channels WL & PSD ≥1.4× for ≥1 s | 13 | 34 | 12 | 1 | 0 | 505 |
| Any 1 channel WL ≥1.8× & PSD ≥1.6× for ≥2 s | 12 | 32 | 11 | 0 | 1 | 305 |
| ≥2 channels WL ≥1.8× & PSD ≥1.6× for ≥2 s | 3 | 16 | 7 | 0 | 0 | 144 |

Many TAP/SWIPE sessions contain sustained physiological activation. Hence a muscle-activity detector cannot independently infer the *intended* long-press command.

**Counterintuitive result:** Longer duration and multi-channel requirements suppress some noise but eliminate much of the SOFT LP cohort. Requiring all three channels would be even more fragile.

## 11.3 Duration sweep of one combined WL/PSD activation criterion

Configuration: any channel WL ≥1.6× and PSD ≥1.4× versus first-2-s baseline.

| Required duration | MID LP /41 | SOFT LP /36 | STRONG LP /20 | CONNECTED /39 | TAP /1,169 | SWIPE /914 |
|---|---:|---:|---:|---:|---:|---:|
| ≥0.25 s | 41 | 36 | 19 | 5 | 1,127 | 890 |
| ≥0.50 s | 37 | 24 | 19 | 2 | 1,021 | 810 |
| ≥1.00 s | 36 | 17 | 14 | 1 | 640 | 544 |
| ≥1.50 s | 34 | 15 | 13 | 0 | 477 | 421 |
| ≥2.00 s | 33 | 14 | 12 | 0 | 383 | 337 |
| ≥3.00 s | 28 | 14 | 8 | 0 | 248 | 251 |
| ≥4.00 s | 24 | 12 | 7 | 0 | 177 | 176 |

A simple long-duration threshold does **not** solve the problem. Shortening it raises detection coverage but includes many short/other gestures; lengthening it excludes soft contractions and sometimes strong sessions whose recorded muscle response is not continuously elevated.

## 11.4 Single-channel ablation

For the first-2-s WL/PSD ≥1.6× criterion held for ≥1 s, SOFT LP recording coverage was:

| Input channels available | SOFT LP files with event |
|---|---:|
| CH1 only | 7/36 |
| CH2 only | 13/36 |
| CH3 only | 12/36 |
| CH1 + CH2 | 15/36 |
| All three | 17/36 |

This validates a practical design constraint: **do not demand two reliable channels before admitting any activity**. Quality-weighted evidence from the best single responsive channel must remain possible. However, accepting only one channel creates susceptibility to single-channel contact artifacts, so quality assessment and time consistency must accompany it.

---

# 12. Experiment G — startup baseline versus retrospective oracle

For **any channel WL ≥1.6× for ≥1 s**:

| Initialization | STRONG LP files flagged | DISCONNECTED context files flagged |
|---|---:|---:|
| Causal-like first 2 seconds | 15/20 | 6/42 |
| NONCAUSAL per-file 20th percentile | 20/20 | 20/42 |

This is an important warning. The full-file “oracle” often finds a better low point in recordings that began with muscle activity or noise; it may improve LP coverage while creating more disconnected/noise triggers. It is not something firmware can use in real time.

**Recommendation:** Begin in UNCALIBRATED / UNKNOWN. Build a trustworthy rest floor from prolonged high-confidence quiet periods; never silently assume the initial two seconds were relaxed or that low within-file energy automatically signifies good contact.

A moving floor must adapt conservatively and **freeze when activation/contact uncertainty occurs**. Otherwise a held contraction can be incorporated into the resting baseline and later appear falsely “relaxed.”

---

# 13. Experiment H — leave-people-out context-level activity detection

A five-fold participant-grouped training selection chose the **PSD-only, one-channel ≥1.8× for ≥1 s** criterion in all five folds when asked the limited question:

> Does this recording look like it contains substantial activity (LP context), versus a controlled CONNECTED reference context?

On **101 LP recordings plus 39 connected reference recordings** (26 normalized user identifiers):

- **Balanced accuracy:** 90.0%.
- **Confusion matrix [connected,LP]:** [[37,2],[15,86]].
- **LP recordings containing a sustained PSD event:** 86/101.
- **Connected-reference recordings containing such an event:** 2/39.

These are useful **context-level feasibility results** but are **not** activity/rest ground-truth classification accuracy and are not false-release performance. LP contexts contain rest windows and can have poor sensor placement. Connected reference contexts can include incidental contractions.

Do not use this headline number to claim 90% actual press/release detection.

---

# 14. Experiment I — unsupervised two-state discovery, with NO pointer labels

We also asked: does a given recording naturally exhibit distinct low and high sEMG regimes, even without trusting pointer state?

A two-component Gaussian mixture fit the within-file relative log-WL/PSD features. A credible split required:

- BIC(1-component) minus BIC(2-component) >10.
- Both states occupy 10–90% of the session.
- High versus low state mean log-energy difference >ln(1.4).

| Group | Sessions with a credible two-state split |
|---|---:|
| MID LP | **36/41** |
| SOFT LP | **30/36** |
| STRONG LP | **17/20** |
| Contraction LP | **3/4** |
| **All LP** | **86/101** |
| CONNECTED reference | 3/39 |
| DISCONNECTED | 7/38 |
| LOOSE | 11/42 |
| TABLE | 1/6 |
| NOTHING | 1/5 |
| DRAG_AND_DROP | 35/37 |
| Sampled TAP | 83/90 |
| Sampled SWIPE | 79/96 |

Selected median longest high-state episode durations: MID LP **3.22 s**, SOFT LP **2.52 s**, STRONG LP **4.27 s**, CONNECTED reference **1.24 s**. These durations are of **model-defined states**, not verified button holds.

**Interpretation:** There often is an independently discoverable *low-energy/high-energy structure* inside LP sessions. This is evidence for pursuing a multi-state muscle activity model. But high/low structure also appears in taps, swipes and drag/drop, so it is **not specific to LP**.

**Limitation:** The mixture uses all windows retrospectively and assumes two Gaussian components; it cannot directly provide causal real-time thresholds, contact truth, or validated press/release timing. The low component could also be faulty-contact signal. It is a candidate for proposing intervals for human review.

---

# 15. Synthesis — what seems genuinely common across people?

### Pattern P1: Relative energy matters more than absolute amplitude

A legitimate activity response often increases WL and broadband PSD *relative to the same channel’s previously observed low level*. Baselines vary greatly across people, hands, session placement, hardware contact, and contraction strength. Absolute-scale classifiers are brittle.

### Pattern P2: Disconnected and table/noise conditions can be high-amplitude but temporally flat

Bad contact in these experiments frequently produces high absolute WL/PSD with low dynamic range. This is a valuable high-confidence *artifact marker*. It is not the only type of disconnection.

### Pattern P3: Active muscle can also form a plateau

A genuine sustained contraction may stop changing rapidly while remaining elevated compared with its **pre-contraction baseline**. Stability alone is never a sufficient definition of relaxation. The reference floor must be frozen or guarded during suspected activity.

### Pattern P4: Healthy activity is often channel-asymmetric

The channel with the clearest muscle response differs between subjects and recording conditions. Strong contractions may recruit several channels, while soft activity often appears predominantly in one. The best two channels are not permanently CH1+CH2 and cannot be selected by a hard-wired preference.

### Pattern P5: Spectral **shape** helps contact, but spectral **total power** largely tracks WL

Use spectral shape for contact/noise characterization, and WL/PSD total to estimate amplitude above baseline. Do not double-count correlated evidence as though it were two independent sensors.

### Pattern P6: Absolute “quiet” is insufficient to certify valid contact

A nearly silent channel may be relaxed muscle, poor placement, saturated frontend, open circuit, or lost reference. Without controlled activation and/or hardware impedance detection, “low EMG = connected and resting” is unsafe.

### Pattern P7: Context remains necessary

High muscle activity can occur during tapping, swiping, resting grip, repositioning and other non-LP actions. A muscle activity model should express activity and confidence; **application semantics** for a long press require other evidence and verified event annotations.

---

# 16. Candidate embedded architecture (NOT yet validated)

## 16.1 Data processing: 3 channels, several time horizons

Use a short primary window of approximately **250 ms** for robust activity-energy summaries; consider a faster 50–100 ms envelope for responsiveness and a longer **2–5 s** horizon for stable contact and baseline estimation. The exact horizons must be tuned against independently annotated transitions and the actual firmware update rate.

For each channel i in {1,2,3}:

    W_i(t) = robust window estimate of waveform length
    P_i(t) = sum of five correctly normalized EMG PSD zones
    Fw_i   = learned trusted resting floor for WL
    Fp_i   = learned trusted resting floor for PSD power
    rw_i   = ln((W_i + eps) / (Fw_i + eps))
    rp_i   = ln((P_i + eps) / (Fp_i + eps))

WL and PSD both describe muscle electrical amplitude. An exploratory continuous activation score could be:

    A_i = 0.6 * rw_i + 0.4 * rp_i

**The 0.6/0.4 weights are a hypothesis only**, not fitted by these experiments. Simpler WL-only or PSD-only baselines must be retained as ablations. Requiring both WL and PSD to exceed hard thresholds often loses sensitivity because the two signals are related, not independent.

## 16.2 Quality scoring per channel

For each channel keep:

- **Absolute/noise context:** log WL and PSD power, with calibration-version provenance.
- **Temporal response:** short versus multi-second WL/PSD dynamic range, robust derivative/change, flatline and clipping indicators.
- **Spectral shape:** five zone proportions, coarse spectral entropy and low-index zone fraction.
- **Cross-channel context:** amplitude imbalance, possible correlations, and whether plausible contractions are visible on other channels; no absolute 2-of-3 requirement.
- **Hardware diagnostics, if available:** lead-off/impedance measurement is far more direct for contact than inferred EMG activity.

Outputs: per-channel quality score Q_i from 0 to 1, plus flags (STATIC_HIGH_NOISE, FLATLINE, UNSTABLE_CONTACT, UNCALIBRATED, POSSIBLY_GOOD). The precise decision model must be fitted on new truth-labelled data and validated out of domain.

**Conservative fallback:** Severe static-high-noise can temporarily lower Q_i when WL median over a suitable *online* multi-second window is unusually high and relative variation is tiny. Do not apply the offline 5,000/1.5 cut-offs directly without window revalidation.

## 16.3 Rest-floor estimation

A robust floor should be established **only when contact seems valid and physiological activity is confidently low**:

1. At start, status UNKNOWN until enough quality-controlled observations exist.
2. Gather several quiet candidate windows, favoring low quantiles rather than means; check stability and sensor quality.
3. Freeze or strongly limit upward baseline movement during high activity, unstable placement or unknown quality.
4. Permit slow recalibration during later verified rest periods.
5. Keep floors per channel and, if possible, per contact/placement session; reset or flag on large structural changes in PSD shape or impedance.
6. If no credible floor is available, report UNCALIBRATED/UNKNOWN rather than inventing confidence.

**Risk to explicitly test:** Feedback loop where an incorrect ACTIVE or bad-contact inference blocks baseline updates forever. Recovery needs a supervised or controlled calibration path.

## 16.4 Decision states

Recommended output is not a single binary flag:

| State | Enter only if… | Meaning |
|---|---|---|
| **REST** | Usable contact; calibrated; low relative energy across trustworthy available channels; stable enough | Low recorded muscle activity, NOT verified lack of physical pressing |
| **ACTIVE** | Reliable channel(s) rise sufficiently relative to prior floor; temporal support; no stronger artifact explanation | Evidence of muscle activity |
| **AMBIGUOUS** | Weak rise, conflicting channels, changing placement, startup uncertainty | Abstain, don't overstate confidence |
| **BAD_CONTACT** | Reliable evidence of faulty contact on relevant channels | Channel(s) cannot support muscle-state judgment |
| **UNCALIBRATED** | No trustworthy baseline yet | No rested-state decision |

A quality-weighted aggregate could prioritize the **best trustworthy channel**, with bonuses for agreeing independent physiological channels:

    valid = [i for i where Q_i exceeds validated minimum]
    if no valid channels:
        global_state = BAD_CONTACT or UNKNOWN
    else:
        top = max(Q_i * A_i for i in valid)
        agreement = bounded_bonus_from_other_reliable_channels()
        score = top + agreement

Use hysteresis and minimum dwell times to prevent flicker. **Thresholds are not fixed here**: the demonstrated 1.4×–2.2× ranges are *swept hypotheses*, not validated universal on/off values. Pressing semantics must be decided at a higher layer, informed by independent event evidence.

## 16.5 Pseudocode for a first prototype

    every observation (timestamped, not sample-count driven):
        for each channel i:
            calculate WL_i, PSD_power_i, zone_shape_i
            update rolling noise/contact features
            Q_i = contact_quality_model(features_i, context)

            if Q_i is poor:
                channel_state_i = BAD_CONTACT
                freeze_rest_floor(i)
                continue

            if trusted_rest_floor_i does not exist:
                channel_state_i = UNCALIBRATED
                collect_calibration_candidate_only_if_quiet_and_stable(i)
                continue

            activity_i = relative_energy(WL_i, PSD_power_i,
                                         trusted_rest_floor_i)

            if activity_i clearly high for validated persistence:
                channel_state_i = ACTIVE
                freeze_rest_floor(i)
            else if activity_i convincingly low for validated persistence:
                channel_state_i = REST_CANDIDATE
                cautiously_update_rest_floor(i)
            else:
                channel_state_i = AMBIGUOUS
                freeze_or_slow_rest_floor(i)

        global_state = quality_weighted_temporal_fusion(channel_states)
        output global_state and per-channel confidence/reasons

This intentionally avoids assuming PointerState is ground truth or that fixed low activity always equals valid contact.

## 16.6 Firmware resource goals

For nRF-class embedded implementation:

- Reuse already-computed 3×5 PSD and three WL features where available.
- Prefer fixed-size circular buffers, stable online quantiles approximations, EMA/robust change features, and bounded memory.
- Use compact lookup-table approximations or monotonic ratio thresholds instead of expensive repeated logarithms, if profiling demands it.
- A shallow tree or small fixed-point classifier can estimate Q_i, with separate temporal hysteresis and calibration logic.
- Benchmark CPU, RAM and update frequency on the **actual firmware build**. This report did not measure embedded cycles or model size.
- Keep diagnostic telemetry: per-channel quality, baseline validity, energy ratios, reason codes, and state transitions.

---

# 17. Competing hypotheses that still need falsification

| Hypothesis | Why it might work | How to falsify it |
|---|---|---|
| Adaptive per-channel baseline identifies rest | Relative energy changes are consistent; absolute amplitudes vary | Begin session already pressing, change position mid-hold, observe false REST |
| Static high WL + low variability indicates severe bad contact | Observed in disconnected/table captures | Hold a very steady strong contraction; test whether it falsely trips |
| Low spectral-zone fraction / entropy detects electrode artifacts | Shape-only contact model retains AUC | Test across firmware versions, country power-line frequencies, healthy active users |
| One reliable responsive channel is sufficient for soft activity | Strict two-channel rule misses soft LP | Create controlled one-channel intermittent contact noise and measure false ACTIVE |
| Two-component low/high state model generalizes | 86/101 LP recordings show two-state structure | Test prolonged continuous contraction, mixed gestures, sensor lift-off, first 2 s already active |
| Good quality can be inferred entirely from PSD/WL | Some contact conditions separable | Quiet disconnected electrode and silent valid electrode under the same physical test |
| Accelerometer movement helps reject artifact | Some motion artifacts create correlated movement | Active press while moving and rest while walking; quantify trade-off |
| Cross-channel correlation helps verify physiology | Multi-muscle responses may co-modulate | Independent single-channel muscle activation and coupled electrical interference |

A scientifically honest detector must have measurable UNKNOWN and abstention behavior, not only positive classifications.

---

# 18. What data to collect next for *actual* validation

## 18.1 Independent truth instrumentation

The fastest path to real accuracy is a carefully controlled, smaller, higher-quality dataset:

- **Independent press/release timestamps** from a physical force switch/load cell or synchronized external input signal, not PointerState generated by the system under test.
- **Independent muscle-state annotation** when possible, with explicit “relaxed”, “lightly tensed without press”, “strong isometric contraction”, “moving while tensed”, and “quiet while moving” tasks.
- **Electrode-level contact truth:** controlled unplug/replug, known lifted contact, loose and normal placement, and ideally direct electrode impedance or a supported hardware lead-off readout. Log which CH was affected.
- Preserve firmware version, filtering, PSD bin normalization mode, device ID, sampling clock, person, hand, position, pressure target, and exact artifact type.
- Capture quiet baseline both **before and after** presses to measure eventual return-to-rest. Include runs that begin in a held contraction without a preceding rest.

## 18.2 Core factorial test matrix

For each volunteer, vary independently:

1. **Activity:** rest, soft, mid, strong, sustained, repeated bursts, taps, swipe, drag, voluntary gripping without UI press.
2. **Contact:** all good, one channel lifted, two lifted, all lifted, loose pressure, electrode touching fabric/table, charging/power nearby.
3. **Motion:** stationary, arm rotating, walking, wrist movement, rapid reposition.
4. **Setup:** both hands, multiple band orientations, session re-donning, different days and devices.
5. **Timeline:** initial rest, initial already-active, short hold, long hold, release to rest, spontaneous contact failure during active hold.

Avoid subject/day leakage: keep all recordings from a person, or for stricter transfer tests a device/day, in held-out groups. Also hold out **entire acquisition campaigns** to expose firmware/schema domain shift.

## 18.3 Required evaluation metrics

**Contact quality**
- Precision/recall/specificity for each CH with independently confirmed contact truth.
- False bad-contact warnings **during true strong activity**.
- Time-to-detect lifted electrode, time-to-recover after reattachment.
- UNKNOWN/abstention rate and coverage-conditional accuracy.

**Muscle activity**
- Per-window low/high confusion versus independently annotated physiology.
- True rest false-activation events **per hour**, including noisy environments.
- False-REST rate during a confirmed sustained contraction.
- Detected onset and return-to-rest delay distributions (median, 90th/99th).
- Sensitivity under soft effort and poor-but-usable single-channel conditions.

**Application long press** (if needed later)
- Release timing while holding, eventual correct release, false long press on taps/swipes.
- Distinguish physiological “inactive” from the UI’s physical “released” state.

Report metrics **per subject and per contact condition**, not just global sample-weighted F1. Bootstrap at **participant level**, not individual rows. Tune thresholds only inside the training folds.

---

# 19. Suggested experimental roadmap

### Phase 0 — data integrity / reproduction
- Confirm raw versus mean-converted PSD per file; add explicit conversion metadata, no inference from amplitude alone.
- Standardize available monotonic time fields and report sensor sample-rate version.
- Resolve user aliases for subject leakage and identify channel-contact ground-truth problems.
- Preserve the original recording bytes and existing best models separately; do not retroactively relabel original data.

### Phase 1 — robust contact model
- Train on normal rest, normal active, every controlled disconnected channel, partial/loose contact and high-noise off-wrist/table states.
- Compare 3-family feature subsets: amplitude, temporal response, spectral shape; use no-contact impulse tests.
- Optimize **false bad-contact alarms during strong contractions**, not just overall contact AUC.
- Expose an UNKNOWN decision.

### Phase 2 — physiological low/high state estimator
- Verify per-channel WL/PSD rest floor and causal update protocol.
- Compare absolute amplitude, relative WL, relative PSD, weighted combination, and simple 2-state temporal segmentation.
- Simulate startup-active and contact changes; explicitly test model recovery.
- Evaluate per-event and per-window truth, including soft holds.

### Phase 3 — resource-constrained firmware
- Reduce to the smallest set of robust features validated under domain shift.
- Implement fixed-memory windows, compact contact model and hysteresis.
- Benchmark execution and power on the actual nRF firmware.
- Run long unsupervised in-the-wild rest periods and collect error logs for review.

### Phase 4 — hard failure review
- Review every false REST during a true hold, every false ACTIVE during verified rest, and every false BAD_CONTACT during a confirmed good-contact contraction.
- Cluster errors by person, channel, device, orientation, temperature, power environment and recording campaign.
- Add adversarial counterexamples rather than optimizing only easy examples.

---

# 20. Limits of this research — explicit nonclaims

1. **No manually verified per-window rest/press/release labels.** We cannot state true press recall, release latency or false-rest rate.
2. **No complete electrode-level contact truth.** In particular CH3 should not inherit DISCONNECTED file labels.
3. **No raw sEMG or impedance measurements**, so spectral zone ratios are not a substitute for direct hardware contact proof.
4. **The LP cohort is only 101 files.** SOFT=36 and contraction LP=4; not a large population.
5. **Dataset and protocol shift:** connected/disconnected controls use a newer collection protocol than many LP sessions.
6. **Partial and potentially noisy subject identifiers:** normalized User strings are a useful grouping but aliases may correspond to the same person.
7. **Post-hoc rule selection:** the high-WL/static-noise thresholds came from looking at these recordings; their apparent false-alarm performance is not an independent test.
8. **Weak label augmentation:** labelling all LP electrodes as physically connected is known to be imperfect and can hide genuine poor contact.
9. **Retrospective baselines/GMM:** comparisons using the full recording do not provide online guarantees.
10. **Correlation, not independence:** WL and total PSD are related representations of the same EMG, so agreement is not multiple independent sensors.
11. **No confirmed universal feature:** a single amplitude/spectral band is not proven to detect both resting and disconnection in every condition.

A detector claiming high accuracy from these recordings alone without reporting the above would be overconfident.

---

# 21. Reproducibility: runnable scripts, evidence and logged results

All code resides in branch **research/lp-rest-signatures** and reads existing CSV.gz files without modifying them.

| Experiment | Script | GitHub Actions evidence |
|---|---|---|
| Original schema/column probe | scripts/lp_rest_probe.py | https://github.com/alex00Pirotskyi/ML/actions/runs/43753550480 |
| Preliminary WL/PSD 0.5 s audit | scripts/lp_rest_audit.py | https://github.com/alex00Pirotskyi/ML/actions/runs/43753771785 |
| Five-zone spectral audit | scripts/lp_rest_spectral.py | https://github.com/alex00Pirotskyi/ML/actions/runs/43753975241 |
| Entire dataset and contact cross-validation | scripts/lp_deep_audit.py | https://github.com/alex00Pirotskyi/ML/actions/runs/43754837877 |
| Temporal thresholds, startup, channel ablation | scripts/lp_temporal_benchmark.py | https://github.com/alex00Pirotskyi/ML/actions/runs/43754930614 |
| Contact-feature ablations and matched examples | scripts/lp_contact_ablation.py | https://github.com/alex00Pirotskyi/ML/actions/runs/43755025541 |
| Contact-vs-activation training domain stress | scripts/lp_contact_activity_stress.py | https://github.com/alex00Pirotskyi/ML/actions/runs/43755169530 |
| GMM two-state experiment (corrected run) | scripts/lp_unsupervised_states.py | https://github.com/alex00Pirotskyi/ML/actions/runs/43755381302 |
| Matched-fold LP-versus-contact OOD comparison | scripts/lp_contact_heldout.py | https://github.com/alex00Pirotskyi/ML/actions/runs/43755316326 |

**How to reproduce:**

    git fetch origin
    git checkout research/lp-rest-signatures
    python -m pip install numpy pandas scikit-learn
    python scripts/lp_deep_audit.py
    python scripts/lp_temporal_benchmark.py
    python scripts/lp_contact_ablation.py
    python scripts/lp_contact_activity_stress.py
    python scripts/lp_contact_heldout.py
    python scripts/lp_unsupervised_states.py

The scripts require the DATA/DATA_LONG_PRESS/**/*.csv.gz files to exist. They are designed to be read-only. Check the current branch and avoid accidentally committing results into main.

**Evidence distinction:** All numerical dataset findings in Sections 3–14 are observed experimental output from the listed scripts. Algorithm suggestions and thresholds in Sections 15–19 are *proposed engineering hypotheses* unless explicitly marked as measured.

---

# 22. Outside scientific references (context, not claims about our data)

1. **Besomi et al., 2020** — Analysis and Biophysics of Surface EMG for Physiotherapists and Kinesiologists. Reviews amplitude/frequency dependence on person, electrode placement, normalization, onset threshold limitations, and physiological interpretation. https://pmc.ncbi.nlm.nih.gov/articles/PMC7594523/
2. **Review of electromyography onset detection methods for real-time control of robotic exoskeletons, 2023.** Describes fixed, dual and adaptive onset thresholds, temporal duration, and lack of universal standard thresholds. https://pmc.ncbi.nlm.nih.gov/articles/PMC10594734/
3. **Reducing Noise, Artifacts and Interference in Single-Channel EMG Signals: A Review, 2023.** Details power-line interference, motion artifacts and poor-contact artifacts, motivating a separate quality path. https://pmc.ncbi.nlm.nih.gov/articles/PMC10059683/
4. **Tutorial: Surface EMG detection, conditioning and pre-processing — Best practices, 2020.** Electrode-skin interface, CMRR, filtering, power-line noise, sampling. https://doi.org/10.1016/j.jelekin.2020.102440
5. **Tutorial: Surface electromyogram amplitude estimation — Best practices, 2023.** Interpretation and estimation of EMG envelope/RMS/rectified amplitude. https://pubmed.ncbi.nlm.nih.gov/37552918/
6. **Tutorial: Frequency analysis of the surface EMG signal — Best practices, 2024.** Stationarity, spectral features, coherence and the limitations of frequency-domain interpretation. https://doi.org/10.1016/j.jelekin.2024.102937

---

# Final recommendation

**Prototype a transparent three-layer model rather than another monolithic long-press classifier:**

1. **Per-channel contact confidence** built from absolute WL/PSD, response variability, spectral shape, and verified hardware diagnostics when possible.
2. **Causal per-channel rest normalization** from high-confidence quiet periods, tracking WL and broadband PSD changes without assuming any PointerState truth.
3. **Temporal low/high/ambiguous decision logic** using the best usable channels, adaptive thresholds, hysteresis and explicit UNKNOWN.

A safe provisional insight from these recordings is that **high stationary noise often indicates bad contact**, while **repeatable elevation above a trusted prior low-energy baseline** often indicates muscle activation. Neither statement alone identifies intent or guarantees reliable contact. The next decisive investment is **independent per-electrode and per-event ground truth**—without it, even a seemingly excellent classifier may fail on strong contractions or soft holds.

**Research status: hypothesis and evidence established; next step is controlled validation, not claims of production accuracy.**
