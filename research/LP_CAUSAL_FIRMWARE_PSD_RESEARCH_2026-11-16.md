# Causal resting-baseline research: exact Mudra firmware PSD semantics and LP labeling v2

**Date:** 2026-11-16  
**Status:** Source-grounded investigation, scientific literature review, implemented/tested causal teacher on research branch. **Not independently validated physiological ground truth.**  
**Firmware source:** User-provided feat_extraction.c (1,803 lines), plus user-provided macros for queue sizes, quantization, and EMG/ACC_PSD_MEAN values at recording time.  
**Data:** 2,431 CSV.gz recordings under DATA/DATA_LONG_PRESS in alex00Pirotskyi/ML.  
**Implementation:** lp_causal_fw_teacher.py called by lp_rest_labeler.py. Original v1 source preserved in research/legacy/lp_rest_labeler_v1.py.

## 1. Research objective and outcome

Objective: replace the previous noncausal whole-recording 15th-percentile reference with a causal, per-electrode resting baseline that operates with the actual firmware feature encoding, responds to release after a sustained contraction, and does not mistake bad contact for physiological resting.

The prior classifier imitated v1 offline pseudo-labels at 83.50% balanced accuracy on five subject-held-out folds, but retrieved only **12.0% of v1 teacher-rest samples in STRONG_LONG_PRESS**, versus 86.9% for a model with *non-deployable whole-recording reference features*. This suggested a baseline-access problem rather than simply a classifier-size problem. These are pseudo-label imitation scores, not true physiological accuracy.

The correction has three parts:

1. Respect the actual **log-scaled and quantized** spectral feature units.
2. Use **only information known before or at each sample** to initialize, hold, lower, and (very slowly) raise each channel's rest floor.
3. Separate activity and contact uncertainty; do not mislabel a sustained high-amplitude contraction at startup as a definitely disconnected electrode.

The current artifact is a **heuristic teacher**. It generates continuous regression targets; it is not yet a measured-contact or physiologically verified binary classifier.

## 2. Firmware audit: the precise signal transformations

### 2.1 Input ADC path and EMG queue

In feat_extraction_snc_new_data_handler, the ADS1293 produces int32 raw SNC data. Values are converted to float using ADC_SCALE = 1 / ECG_MAX_RES, processed through the configured ADS IIR biquad filter bank, and pushed into three 256-element SNC buffers. The exact filter coefficients, ECG_MAX_RES, ADS1293 sample rate and SNC packet cadence are not supplied here, so do not infer exact electrical units or signal bandwidth.

The important configuration is **SNC_QUEUE_SIZE=256** and **ACC_QUEUE_SIZE=32**. Those are raw filtered-sample queue lengths, **not counts of logged CSV rows**.

### 2.2 Waveform length is not RMS or an average

FW emg_estimate_wl computes, per channel:

    WL_ch(t) = uint16(clip(1000 * sum_{k=0}^{254} abs(x[k+1]-x[k]), 0, 65535))

The source also computes two half-window features, wl_tail (first 128 samples) and wl_head (last 128 samples), using 127 differences each. Comments saying 32-sample segments are stale. WL is a time-domain measure of fluctuation, not literally muscle force or activity; motion/noise/poor contact can also raise WL.

The FW computes RMS separately over 256 samples, scaled by 1000, but the inspected recording CSV schema contains no RMS field. A labeler cannot retroactively use absent RMS.

### 2.3 EMG PSD is **log(power-zone sum)**, not linear PSD

The SNC PSD path:

1. Multiply 256 filtered samples by a Blackman window.
2. Run a 256-point real FFT.
3. Set the DC bin to zero and calculate squared complex magnitudes for positive-frequency bins.
4. Sum squared magnitudes within each of five frequency-zone mappings (mapping table not provided).
5. Apply the logarithm and store in uint16:

    zone_sum(ch,z,t) = sum_{k in zone_z} |FFT(ch,k,t)|^2
    E(ch,z,t) = clip_uint16(round(10000 * log10(1 + zone_sum(ch,z,t))))

This follows directly from LOG10_CONVERTER_EMG = FLOAT_TO_U15_EMG_SCALE * INV_LOG_10, the arm_vlog_f32 invocation, the 1.0 offset and uint16 saturation logic.

EMG_PSD_MEAN = 0 at recording means **DO NOT divide by the number of FFT bins in each zone**. It does **not** mean that no temporal mean is performed.

### 2.4 The similarly named mean is a DIFFERENT operation

Separately, psd_estemate_psd_mean loops over the **15** stored EMG log-code history entries and calculates:

    E_mean(ch,z,t) = (1/15) * sum_{j=0}^{14} E(ch,z,t-j)

This runs from the ACC-side feature update path, so the mean can retain stale EMG entries when SNC and ACC events are not synchronized. It is a mean of **quantized logs**, NOT a log of averaged linear power:

    mean(log10(1 + P_j)) != log10(1 + mean(P_j))

Therefore the CSV field named p_core->emg_psd_mean[CHx][z] is ALREADY temporally smoothed. Its integer-like numerical values are encoded log-power estimates; neither a second long moving average nor a second log transform is justified without measuring the latency penalty.

The analogous ACC path uses ACC_QUEUE_SIZE=32, FLOAT_TO_U15_ACC_SCALE=100000, a 10-frame temporal PSD mean and optional ACC bin-average which was OFF during recording. ACC's spectral representation must be interpreted similarly, but the current v2 teacher makes its physiological decisions using EMG only.

### 2.5 Consequence for Python psd_mean.py

The repository's psd_mean.py divides the already logged, rounded uint16 zone values by zone-bin counts [8,10,12,46,51] for EMG. This operation is NOT physically equivalent to dividing linear zone power before taking logarithm.

Example: if zone_sum = 10 and the zone has 8 bins:

    FW recorded log-sum:     round(10000*log10(1+10)) = 10414
    Legacy after-log / 8:   approximately 1302
    Correct pre-log mean:   round(10000*log10(1+10/8)) = 3522

Thus the old Python script is a reversible-ish *numeric rescaling* of the existing log-coded feature, not a faithful simulation of EMG_PSD_MEAN=1 firmware. The v2 labeler **does not** apply any zone-bin division to unconverted recordings. When a file has an explicit psd_mean_applied=1 marker, it multiplies the codes by zone counts to approximately recover the original code before inference, allowing for rounding error. There is NO safe way to determine whether a legacy *unmarked* recording had been postconverted from its content alone.

### 2.6 Data provenance check

A 69-file profile sampled from all 12 recording folders found no psd_mean_applied markers and spectral values consistent with the provided raw-log recording configuration, but this is not an exhaustive proof of every byte of all 2,431 files. Examples of median zone log-codes:

| Recording example | Zone 0 | Zone 1 | Zone 2 | Zone 3 | Zone 4 |
|---|---:|---:|---:|---:|---:|
| Ohad connected | 1,156 | 436 | 286 | 174 | 804 |
| Ohad disconnected | 24,306 | 19,606 | 15,685 | 15,716 | 22,209 |
| Ben soft LP | 61 | 185 | 430 | 397 | 657 |
| Strong LP example | 448 | 1,120 | 1,618 | 666 | 1,965 |

These values are NOT linear power. Their relative changes may provide useful information, but ratios between the encoded values are not physical power ratios.

**Current teacher:** use the median of five *decoded-to-original-scale log-codes*, and relative **differences divided by 10,000**:

    delta_log10P(ch,t) = [PSD_log_index(ch,t) - PSD_log_floor(ch)] / 10000

For a single UNSATURATED zone with the same source encoding, this approximately represents a log10 ratio of (1 + spectral power) across two times. With an average/median over different zones and a 15-frame history, it is an evidence index, NOT a calibrated total PSD power or precise dB amplitude.

## 3. What external sEMG research recommends (and what it does not)

### 3.1 EMG amplitude depends on many factors

Clancy et al., 2023, emphasize that surface EMG amplitude is an estimate of voltage variation/physiological excitation and is affected by electrode geometry, muscle position, skin and frontend conditioning. RMS and average rectified value are common amplitude envelopes. No absolute amplitude across people or electrode positions can be treated as a universal muscular-rest threshold.

**Source:** Clancy et al., Tutorial. Surface electromyogram (sEMG) amplitude estimation: Best practices, Journal of Electromyography and Kinesiology 72 (2023), 102807. https://doi.org/10.1016/j.jelekin.2023.102807

The CEDE international consensus normalization matrix advises choosing normalization references based on purpose, task and population. A per-channel *resting floor* is useful for detecting relative activity but is NOT equivalent to MVC or normalized muscle force.

**Source:** Besomi et al., CEDE Amplitude normalization matrix, Journal of Electromyography and Kinesiology 53 (2020), 102438. https://doi.org/10.1016/j.jelekin.2020.102438

### 3.2 Threshold + duration + adaptation are valid starting points

A 2023 systematic review of 156 papers describes single-threshold, double-threshold and adaptive-threshold onset detection. The best threshold and latency depend on SNR and task; there is no standardized universally correct threshold. A causal rolling baseline and hysteresis are sensible engineering candidates, **but their numeric settings must be validated for Mudra**.

**Source:** Review of electromyography onset detection methods for real-time control of robotic exoskeletons (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10594734/

### 3.3 PSD and movement artifacts should not be conflated

Muceli and Merletti (2024) stress that short-window EMG spectra are noisy, non-stationarity matters, and spectral interpretation depends on actual FFT, filtering and sampling. Windowing and power-spectrum averaging are not interchangeable with averaging the LOGGED power representations.

**Source:** Tutorial. Frequency analysis of the surface EMG signal: Best practices (2024). https://doi.org/10.1016/j.jelekin.2024.102937

Reviews of artifact removal and electrode design show that skin-electrode motion, contact impedance, ambient powerline interference and changes in electrode placement all contaminate sEMG features. High, low or stationary activity can be contact artifacts; none uniquely proves physiological rest.

**Sources:** Artifacts review, Computers in Biology and Medicine (2025), 109651: https://doi.org/10.1016/j.compbiomed.2025.109651 ; Flexible electrode review, npj Flexible Electronics 7 (2023), Article 39: https://www.nature.com/articles/s41528-023-00273-0

### 3.4 Hardware-quality assessment is the most definitive next step

The FW uses a Texas Instruments ADS1293 frontend. According to the TI datasheet, ADS1293 hardware provides DC, analog-AC and digital-AC **lead-off detection (LOD)** modes. The datasheet cautions that DC LOD can misidentify capacitive dry electrodes because DC test current is blocked, and AC LOD requires an appropriate reference/return path. Whether lead-off can safely be enabled in Mudra's actual electrode routing, processing pipeline, and power profile must be confirmed experimentally.

This hardware facility is preferable to insisting that high noisy WL alone proves an electrode physically disconnected.

**Source:** TI ADS1293 datasheet, sections 8.3.15–8.3.18: https://www.ti.com/lit/ds/symlink/ads1293.pdf

## 4. Prior teacher weaknesses specifically relevant to strong LP

The v1 teacher used the **15th percentile of the full recording** as a reference for WL and an incorrectly named PSD power sum. Therefore:

- It used *future* quiet intervals to label earlier activity.
- An online classifier could not reproduce its reference without hindsight.
- Constant high-noise contact could appear stable and therefore "rest" unless a separate contact guard was applied.
- Starting during a contraction is inherently underdetermined without a verified rest observation.
- Its held-out-person five-fold LightGBM imitation result was 83.50% balanced accuracy across 2,431 files, but only **12.0% rest recall against its own pseudo-labels on STRONG_LONG_PRESS**. The oracle with future floor information reached 86.9% in STRONG_LONG_PRESS.
- A classifier imitating a teacher is not an independent test of physiological truth.

See prior research report: research/LONG_PRESS_DEEP_RESEARCH_2026-11-15.md on research/lp-rest-signatures, and new benchmark: research/LP_CLASSIFIER_EVALUATION_2026-11-16.md on research/lp-classifier-eval.

## 5. Proposed correction: quality-gated causal baseline

Implemented as lp_causal_fw_teacher.py and integrated into lp_rest_labeler.py.

### 5.1 Filter-time assumptions

Logged CSV rows are feature frames. The code estimates median spacing from the first 32 valid monotonic timestamps. It does NOT assume that the queue length of 256 equals 256 recorded rows. The early bootstrap interval is explicitly UNKNOWN with near-zero training weight.

Default research candidates:

| Parameter | V2 hypothesis |
|---|---:|
| Trailing WL median | 120 ms |
| Trailing encoded PSD median | 90 ms |
| Rolling quality/stability window | 1.8 s |
| Initial bootstrap before considering rest | ≥450 ms, at least 33 rows |
| Quiet-candidate calibration | 650 ms |
| Following clear large energy drop | 300 ms calibration |
| Binary REST dwell | 180 ms |
| Falling floor adaptation time constant | 700 ms |
| Upward floor adaptation time constant | 60 s |
| Severe static noise evidence | WL ~3,500–8,000 AND low modulation |
| Strong contact-fault claim without reference | Deferred until ≥5 s, otherwise UNKNOWN |
| PSD activity ramp | Δlog10 encoded ~log10(1.07) to log10(1.90) |
| WL activity ramp | ~1.06× to ~1.90× resting WL |
| Per-channel WL/PSD weights | 0.76/0.24 (experimental) |

These are NOT scientifically calibrated thresholds; firmware update-rate and release-timing tests are required.

### 5.2 Causal floor behavior

Per channel, v2 tracks WL baseline Fw_i and encoded PSD floor Fp_i. A channel begins UNCALIBRATED. It may initialize only from sustained plausible quiet observations with non-negligible signal, no clipping, no high-static-noise signature and no contradictory rapid changes. Explicit cross-channel agreement is not mandatory because healthy soft contractions can be channel-asymmetric.

During a high-activity window, the baseline is **frozen**. During plausible rest, downward floor adaptation is comparatively fast but upward adaptation is extremely slow. A constant high contraction must not be absorbed as new rest after 1–2 seconds.

Strong activity followed by a large energy decrease is special. V2 tracks a **decaying past WL peak** for each channel. When the signal falls substantially below the previously observed peak and remains nonzero and plausible, it can establish a new floor in ~300 ms even if the slower quality window still remembers the contraction. This is a release-like *physiological change* hypothesis, not proof of a button release.

### 5.3 Log-domain activity estimate

    WL_ratio = (WL_fast + 25)/(WL_floor + 25)
    EMG_log_change = (PSD_log_index_fast - PSD_log_floor)/10000
    activity_ch = 0.76 * smooth_ramp(log(WL_ratio))
                  + 0.24 * smooth_ramp(EMG_log_change)

The PSD term is not an independent vote; WL and PSD share the same muscle signal and correlated noise.

### 5.4 Contact, low rest, and UNKNOWN

Per-channel contact features include:

- Stably high absolute WL over seconds (severe noise hypothesis).
- Low relative temporal modulation, not just high amplitude.
- Clipping of WL/PSD uint16.
- Obvious dead-signal flatline where both WL and PSD approaches zero.
- Evidence of earlier physiological response that should protect a genuine held contraction from a spurious disconnected label.

If a channel has insufficient evidence to establish a good baseline or sensor quality, it is UNKNOWN rather than implicitly connected.

**Critical identifiability limit:** A strong contraction already present at time zero can be observationally indistinguishable from a high, stationary, disconnected electrode without independent impedance/lead-off information. V2 delays severe-disconnection claims during such uncertain startup; indefinite activity or high noise can still be confounded. The extra UNKNOWN target makes that limitation explicit.

The four soft outputs per channel sum to 1:

    resting_ch + high_activity_ch + disconnection_ch + unknown_ch = 1

The first three are requested regression targets; UNKNOWN is necessary to avoid inventing labels. The values are **heuristic evidence scores, not calibrated probabilities**.

### 5.5 Global binary semantics

    lp_state = 1   only when all three channels are calibrated,
                   appear connected, and appear resting for >=180 ms
    lp_state = 0   otherwise (activity OR disconnection OR unknown)

Zero in uncertain conditions is **NOT a verified active/press label**. lp_state_weight is 0.02 during uncalibrated/strongly UNKNOWN states, and larger only where evidence supports a decision. The user-requested binary convention is preserved.

## 6. Testing and provenance

The old teacher is preserved in research/legacy/lp_rest_labeler_v1.py; the updated public entrypoint is lp_rest_labeler.py.

**Synthetic safeguards implemented:**

1. Connected, quiet startup acquires rest conservatively.
2. One-channel sustained contraction scores active and returns to rest after release.
3. Stationary high electrode noise does not produce confident all-channel REST.
4. Very quiet but nonzero channel is UNKNOWN rather than proved disconnected.
5. Begin while actively contracting: does NOT label the initial high state as confident rest or immediate physical disconnect.
6. Prefix invariance: changing ANY future signal does not change already produced labels.
7. Legacy PSD post-log division with explicit marker is reversed to near-identical labels.
8. Missing PSD is UNKNOWN.
9. PointerState changes do not affect labels.
10. Six older output-format and synthetic-regression tests remain intact.

**Real-data smoke testing:** 36 CSV.gz files, three per each of the 12 groups, processed with no exceptions, using a separate output path. All original recording bytes are untouched.

**Data-provenance audit:** sampled all categories; no raw/legacy markers present in tested files. An optional --psd-mode switch supports raw, legacy-divided or auto(marker) selection. If the file is missing its provenance marker but was already processed, auto cannot detect that condition.

**Important limitation:** These tests establish internal consistency and code correctness, NOT true muscle-rest accuracy or verified electrode disconnection. Additional recording-level comparisons, per-subject validation, and controlled contact trials remain necessary.



## 6.1 Real-data comparison: V1 versus V2 (important negative results)

The read-only comparative experiment processed **all 101 long-press recordings**,
all 39 CONNECTED, 42 DISCONNECTED, 42 LOOSE, 5 NOTHING, 9 TABLE,
37 DRAG_AND_DROP, and 110 sampled TAP + 110 sampled SWIPE recordings.
All source files remained unchanged. These are descriptive fractions of
pseudo-labels, NOT true rest annotations.

| Context | Files | V1 fraction lp_state=1 | V2 fraction lp_state=1 | V2 fraction of rows with low confidence (<0.05) |
|---|---:|---:|---:|---:|
| CONNECTED | 39 | 65.61% | 60.10% | 25.30% |
| DISCONNECTED | 42 | 2.37% | 3.43% | 92.61% |
| LOOSE | 42 | 8.08% | 9.63% | 82.88% |
| NOTHING | 5 | 56.18% | 69.04% | 5.79% |
| TABLE | 9 | 0.00% | 0.00% | 100.00% |
| MID_LONG_PRESS | 41 | 14.90% | 9.19% | 21.63% |
| SOFT_LONG_PRESS | 36 | 23.30% | 24.63% | 25.27% |
| STRONG_LONG_PRESS | 20 | 12.30% | 5.43% | 31.05% |
| LONG_PRESS / contraction | 4 | 12.07% | 5.36% | 47.29% |
| DRAG_AND_DROP | 37 | 18.11% | 8.27% | 42.00% |
| TAP (sample) | 110 | 11.41% | 2.78% | 81.94% |
| SWIPE (sample) | 110 | 6.96% | 1.56% | 85.95% |

**This does NOT prove improvement in real-world strong LP recognition.**
V2 is explicitly more conservative and typically produces fewer confident-rest
rows in strong LP sessions. Some recordings have no identifiable all-channel
rest interval under V2 because one channel never obtains sufficient baseline
or contact evidence. Without independently recorded ground truth, we cannot
call that a true-negative improvement or a false-rest regression.

In 39 connected references, V2 marks **94.9% of files with at least one**
confident resting interval (compared with 84.6% for V1), albeit with smaller
total resting coverage; this suggests V2 often calibrates but is stricter about
transitions. In DISCONNECTED sessions, CH1/CH2 are largely UNKNOWN or
static-noise candidates; CH3 is not assumed disconnected.

**Next validation priority:** label a small collection of precisely
synchronized strong-press and release intervals independently, and measure
time-to-rest and false-REST-while-held against those timestamps. It would be
unsound to optimize V2 merely to reproduce a larger portion of V1's
noncausal labels.

## 7. How to run

From the repo root:

    git pull --ff-only origin main
    python -m pip install numpy pandas
    python lp_rest_labeler.py

Defaults:

    --input-dir DATA/DATA_LONG_PRESS
    --output-dir DATA/DATA_LONG_PRESS_LABELED_V2
    --psd-mode auto

The separate V2 output folder prevents mixing legacy v1 and v2 labels. The labeler leaves all original CSV.gz fields intact and appends the same nine requested continuous targets, three uncertainty scores, lp_state, lp_state_weight and lp_teacher_version.

**For a converted but unmarked dataset:** explicitly pass --psd-mode legacy-divided. **For known raw capture:** pass --psd-mode raw (or auto when source has no conversion flag). Do not mix ambiguous data formats in a training batch.

## 8. Follow-up validation needed before firmware deployment

1. Confirm sample clock synchronization, actual raw SNC rate, FFT-zone tables, IIR filters, and duration of the 15-entry history mean.
2. Record controlled **start-active**, rest, soft hold, strong hold, contact lift-off and reattachment, one/two/all electrodes affected, with an independent force-switch event signal.
3. Compare teacher/rest-window marks to independent truth on held-out participants and acquisition campaigns. Report false-REST during strong holds, false-ACTIVE at rest, rest detection latency after release, false disconnected under strong contractions, and abstention rate.
4. Compare raw log-PSD code with direct normalized spectral zone data acquired simultaneously, and confirm that post-converted legacy files are explicitly marked.
5. Consider TI ADS1293 AC LOD hardware if compatible with this electrode design; confirm safety, operation with the actual reference/RLD path, sampling interference and power cost before firmware change.
6. Profile 120 ms/90 ms/180 ms windows on actual nRF52832 CPU; Python processing speed does not predict embedded latency.
7. Keep source quality metadata so the next ML training stage can distinguish physical bad contact, missing calibration, and normal activity, rather than learning mislabeled UNKNOWN zeros.

**Research conclusion:** A causal baseline with frozen-in-activity updates, release-aware recovery, and explicit UNKNOWN is better justified for real-time use than a retrospective recording percentile. The present data can verify code behavior and broad signal patterns, but do not permit a statistically honest true-rest or true-contact accuracy claim.
