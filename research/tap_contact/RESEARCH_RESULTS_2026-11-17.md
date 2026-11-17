# EMG contact filtering for TAP/SWIPE — measured full-corpus research

**17 November 2026 — reproducible experimental ablation, NOT a production model.**

- **Data:** Frozen Claude V2 strong-only TAP/SWIPE labels on `DATA/DATA_TAP`; independent connected/disconnected reference contexts from `DATA/DATA_LONG_PRESS`.
- **Code:** [contact_research.py](./contact_research.py)
- **Verified full GitHub Actions run:** [44052431135](https://github.com/alex00Pirotskyi/ML/actions/runs/44052431135). Download the attached `tap-contact-deep-research` artifact for full JSON and audit.
- **Isolation:** Research branch `research/tap-emg-contact-20261117` only. No changes to main training, golden labels, `export_plus.py`, or the current high-performing firmware `TAP_SWIPE.h`.
- **Status:** The user-proposed *simple* bad-contact positive-recording deletion did NOT demonstrate a large benefit on available recordings. Contact state still correlates with poorer sensitivity and merits localized, independently verified research.

## Experimental question

Does discarding TAP/SWIPE **positive training recordings** judged to have bad EMG contact measurably improve detection on the **same complete, unfiltered held-out recordings**, without losing soft/double taps or increasing false triggers?

This is **not** a physical contact measurement. The data lack per-channel electrode impedance/contact truth and independently synchronized real-world tap onset labels. Every contact prediction is a *proxy*; every tap recall value below is *against Claude V2 strong silver events*.

## Input audit

- 2,737 source recordings used after original Claude label cleaning: **2,089 positive TAP/SWIPE** recordings and **648 negative** source recordings.
- 115,571 sampled training/evaluation rows: 86,411 silver-positive, 29,160 nominal-negative.
- 18,487 selected, sensor-confirmed silver events; 56 normalized participant IDs overall, 51 with eligible silver-positive events.
- 126 original `p_core->` fields; 3 participant-held-out folds with no person shared between training and test per fold.
- Reference quality controls: 39 `CONNECTED` and 42 `DISCONNECTED` recordings, 13 normalized people, with **159 nominally good channel examples** (all three `CONNECTED` plus **CH3 of `DISCONNECTED`**) and **84 nominally bad** (`DISCONNECTED` CH1/CH2). CH3 remained physically connected during those recordings.
- 0 read/feature extraction errors over 2,737 TAP/SWIPE/negative recordings.

## Quality screens

1. **Sensor plausibility** (fixed from prior LP research): independently per channel, 10th-percentile WL in `[65,3500]`, 10th-percentile log-encoded mean PSD `<=0.60`, with saturation, invalidity and flatline guards. Uses the complete recording to assign an **offline training-quality proxy**, not an online inference feature.
2. **Learned contact proxy**: regularized logistic regression on per-channel WL distribution, PSD spectral shape/dynamics, current-vs-average difference, head/tail imbalance and static/noise indicators; trained on the weak 243-channel control reference only. A high-confidence accepted channel requires estimated bad-contact probability <0.20, **a research hypothesis, not a calibrated physical-contact threshold**. No gesture labels were used to train the quality model.

On participant-held-out weak control references:
- Physical plausibility screen passes 95.60% of *nominal-good* and 32.14% of *nominal-bad* examples. It does **not** prove physical connectivity.
- Learned proxy: ROC AUC **0.9206**, balanced accuracy **0.8443**, confusion [nominal-good TN=136, FP=23; nominal-bad FN=14, TP=70] against weak context labels.
- Extrapolation from those controls to real tapping is a serious domain-shift risk: muscle contraction and contact artifacts change the same summary features.

### How many positive training recordings does each screen remove?

- At least **one** plausible channel: **2,089 / 2,089 pass**; removes nothing.
- At least **two** plausible channels: **2,081 / 2,089 pass**; removes 8 (0.38%).
- At least **one learned high-confidence channel** (`p_bad < 0.20`): **1,108 / 2,089 pass**; removes **981 (46.96%)**. A proxy calling 47% of known-gesture recordings all-bad is not reliable enough to control hard deletion.

## Paired held-out model experiment

**Same model** in every arm: 100-tree LightGBM, 13 leaves, depth 4, 126 raw features, same grouped folds and sampling, no hyperparameter search. All students trained and evaluated at **fixed threshold 0.85**, with class weighting calculated after filtering. Negatives are **never removed** by any positive-label filter; all held-out positive and negative recordings remain in evaluation.

For nominal negative folders, each fold's model is replayed at full file duration on its held-out negative recordings. Trigger clusters are counted at threshold 0.85 with a 150 ms refractory rule. This is **not the firmware tap state machine** and is not a physically verified false-trigger rate; some nominal-negative recordings may contain real gesture-like motion.

| Training policy | Silver event recall, all test events | Δ recall (percentage points) | Assumed-negative events / min | Estimated static model bytes |
|---|---:|---:|---:|---:|
| Baseline: accept all strong silver positives | **95.1966%** | — | **10.2412** | 15,528–15,564 |
| Drop positive recordings with no plausible channel | 95.1966% | **0.0000** | 10.2412 | 15,528–15,564 |
| Drop unless ≥2 plausible channels | **95.2669%** | **+0.0703** | **9.9890** | 15,540–15,564 |
| Drop unless learned proxy sees ≥1 high-confidence channel | 90.6367% | **−4.5599** | 9.6422 | 15,480–15,564 |
| Downweight positive samples by best-channel proxy | 95.1425% | −0.0541 | 10.4367 | 15,552–15,564 |

The negative results cover **158.57 recorded minutes** across negative categories; the baseline had 1,624 threshold clusters, two-channel policy 1,584, hard learned filter 1,529, and learned weighting 1,655.

### Participant-cluster bootstrap confidence intervals

- ≥2 plausible channels: **+0.07 pp**, 95% bootstrap interval **−0.14 to +0.31 pp**. This includes zero: **no demonstrated improvement**.
- Learned hard filter: **−4.56 pp**, interval **−7.41 to −2.36 pp**: a substantial measured silver-event recall loss.
- Learned weighting: **−0.05 pp**, interval **−0.38 to +0.23 pp**. No established overall gain.

The bootstrap unit is the person, not independent adjacent 6-ms rows. These intervals describe sampling variability *conditional on the existing weak labels and research design*; they do not address label or contact-truth error.

### Soft-tap, double-tap and swipe breakdown

| Subgroup | Events | Baseline recall | ≥2 plausible recall | Learned hard-filter recall |
|---|---:|---:|---:|---:|
| All TAP | 11,457 | 95.46% | 95.52% | 90.64% |
| All SWIPE | 7,030 | 94.77% | 94.85% | 90.63% |
| Soft TAP | 2,457 | **92.55%** | 92.92% | 92.67% |
| Double TAP | 1,696 | **96.11%** | 96.34% | **88.38%** |
| Soft SWIPE | 2,207 | **91.16%** | 91.26% | 91.75% |
| Mid SWIPE | 2,758 | **96.34%** | 96.56% | 91.77% |

**Strong negative finding:** The learned contact-only hard filter reduces double-tap silver-event recall by ~7.72 percentage points. Filtering on a quality proxy can preferentially remove information useful for harder gestures.

### Does lower-quality EMG correlate with worse detection?

Yes, **as an association in this teacher/screen setting**:

- 3 plausible channels: 17,449 silver events, **95.52%** baseline recall.
- 2 plausible channels: 1,018 events, **89.78%** recall.
- 1 plausible channel: 20 events, **85.0%** recall; very small sample.

That supports investing in contact robustness but **does not establish that deleting whole positive recordings is the correct intervention**, nor that the plausibility flag is physically accurate. Lower-quality captures may also be harder because of wearers, protocol, tap force and domain shift.

## Interpretation

1. **The user's hypothesis contains a real signal:** partially invalid/atypical channels correlate with lower detected-event coverage.
2. **Simple whole-recording deletion is not a major lever:** the conservative physical screen has essentially no zero-good positive files; requiring two good channels removes just 8 positive files and yields an inconclusive <0.1 pp gain.
3. **Naive learned-contact deletion is actively dangerous:** the contact reference has weak supervision and domain shift; its strict threshold marks 981 genuine-gesture recordings as all-bad and significantly damages recall.
4. **There is no evidence to replace the successful real-device model.** This experiment uses a new matched 100-tree research baseline, not the exact selected Optuna header the user field-tested.
5. **The absolute 95% silver-event recall here cannot be compared with Claude's earlier ~79% at 2 false/min.** It is a different student/configuration/threshold and runs at ~10.24 nominal-negative triggers/min; it is neither matched event-level false-alarm calibration nor physical precision.

## Highest-value next experiment

**Localized quality-aware label auditing**, not whole-file deletion.

- On each independently confirmed onset, examine per-channel validity across nearby **pre-onset** windows; compare spectral flatline/clipping, high stationary noise, head/tail imbalance and loss/recovery, while not misclassifying strong contractions as disconnection.
- Distinguish **electrode disconnected**, **poor but usable**, **good with a weak gesture**, and **ACC-supported valid tap with low EMG activity**. None can be perfectly inferred from the existing logs without measured contact.
- Quarantine only high-confidence physically invalid *event windows*, and keep uncertain valid/ACC-only events unknown or at reduced weight rather than making them negative or deleting their recording.
- Add independently observed contact lift/recontact events with per-channel status; test on new wearers and different placements. Re-run the exact same grouped-heldout **fixed-threshold** score, plus full negative-folder replay and device A/B.
- Preserve the currently successful **TAP_SWIPE.h** and v20 LP baseline until demonstrated real-life improvement.

## Reproduce

Checkout `research/tap-emg-contact-20261117` and execute:

```bash
python -m pip install numpy pandas scipy scikit-learn lightgbm optuna joblib
python -u research/tap_contact/contact_research.py
```

The script verifies source/label hashes, exports labels if absent, preserves the source, and writes `research/tap_contact/results/report.json`. It uses ~80% of available affinity/cgroup CPUs and has no mandatory CLI options. Its source and results are isolated from firmware and the original trainer. The exact completed-run results are available as GitHub Actions artifact `tap-contact-deep-research` from run 44052431135.

**Research status:** Matched same-data ablation completed and verified. No new physical-contact ground truth, firmware release-rate test, contact-label model, or production deployment was created.
