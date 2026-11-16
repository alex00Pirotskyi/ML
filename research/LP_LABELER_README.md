# Firmware-aware causal LP pseudo-label teacher — v2

The entrypoint lp_rest_labeler.py now calls lp_causal_fw_teacher.py. V2 was
derived from a direct audit of the provided feature-extraction firmware,
external sEMG research, all 101 long-press recordings, contact controls, and
synthetic causal/encoding tests.

**It creates pseudo-labels, NOT verified physical ground truth.**

## Running from the repository root

    python -m pip install numpy pandas
    python lp_rest_labeler.py

Defaults:
- Input: DATA/DATA_LONG_PRESS (raw CSV or CSV.gz)
- Output: DATA/DATA_LONG_PRESS_LABELED_V2, preserving subfolders and original columns
- Workers: 80% of logical CPU cores capped at 12, to avoid RAM exhaustion
- PSD encoding: auto, meaning use explicit psd_mean_applied flag when present;
  otherwise assume the raw acquisition encoding specified by user

Output path is ignored by Git; input files are never modified. The old v1
output DATA/DATA_LONG_PRESS_LABELED is not touched or mixed with v2.

Run a small test:

    python lp_rest_labeler.py --limit-per-group 2 --workers 2

Explicit original raw acquisition mode:

    python lp_rest_labeler.py --psd-mode raw

If the CSV has been processed by the old post-log psd_mean.py WITHOUT a
conversion marker, select the legacy mode and a dedicated output directory:

    python lp_rest_labeler.py --psd-mode legacy-divided --output-dir DATA/DATA_LP_LEGACY_LABELED_V2

Use only when provenance is independently confirmed. An unmarked file cannot
be reliably classified as raw or post-converted by numerical values alone.

## Feature math — essential differences from v1

- The firmware uses SNC_QUEUE_SIZE=256 filtered EMG samples, Blackman window,
  real FFT, squared magnitudes, sum over five frequency zones, then
  E = uint16(round(10000 * log10(1 + zone_sum))). It clips at 65535.
- EMG_PSD_MEAN=0 during recording means DO NOT average FFT bins within
  each zone. The field named emg_psd_mean is STILL the arithmetic temporal
  mean of 15 previously encoded log-zone frames.
- V1 incorrectly called a sum of logged-zone encodings "PSD power".
- V2 uses a robust median of encoded log zones and *differences* of that
  index relative to an established per-channel resting floor, scaled
  by 1/10000. This is only a normalized LOG-DOMAIN evidence index.
- WL still represents scaled absolute sample differences over 256
  filtered EMG samples, not RMS, muscle force, or linear power.

## Outputs

Per original CSV row, on a 0–1 *heuristic score* scale:

- lp_resting_ch1, lp_resting_ch2, lp_resting_ch3
- lp_high_activity_ch1, lp_high_activity_ch2, lp_high_activity_ch3
- lp_disconnection_ch1, lp_disconnection_ch2, lp_disconnection_ch3
- lp_unknown_ch1, lp_unknown_ch2, lp_unknown_ch3

The four per-channel scores add to 1, up to CSV rounding. The
disconnection signal is **artifact evidence**, not electrode-impedance
truth. A silent or initially strong channel may remain UNKNOWN.

- lp_state: binary, 1 ONLY when all three channels are calibrated,
  appear connected and resting for >=180ms. Zero means active OR
  disconnected OR unknown.
- lp_state_weight: 0–1 label reliability. Uncalibrated/strongly UNKNOWN
  rows receive about 0.02 rather than being treated as confirmed active.
- lp_teacher_version: lp_causal_fw_v2.
- label_summary.json: per-directory file counts and label/uncertainty ratios.

Train regression heads on continuous targets. When training binary
lp_state, use sample_weight and consider dropping highly uncertain
samples; do not use zero labels with tiny weights as verified pressing.

## Causal initialization/release

- Start in UNKNOWN with short timestamp bootstrap.
- Require quiet contact-plausible observations before establishing
  per-channel resting WL and log-PSD floors.
- Freeze floors during elevated signal; lower cautiously during
  confident rest; increase VERY slowly (~60s time constant).
- Track previously seen WL peaks; a sufficiently large energy drop
  permits accelerated recalibration even when a long quality window
  still contains prior muscle activation.
- Do not call a high stationary startup contraction "disconnected"
  immediately. Use UNKNOWN until enough evidence is available.
- A true release cannot be proven without an independent pressure or
  event-synchronization reference.

## PSD conversion caveat

The old psd_mean.py divides ALREADY-ENCODED uint16 LOG spectral values by
[8,10,12,46,51] per zone. That is not equal to applying the bin average
inside the firmware before taking the logarithm. V2 reverses that
numerical division approximately when psd_mean_applied=1 is present,
but rounding and clipping make exact recovery impossible.

Do not apply psd_mean.py as a required preprocessing step for V2.

## Validation and reproducibility

- tests/test_lp_causal_fw_teacher.py: causal prefix invariance,
  startup-active, release, steady rest, noise, ambiguous quiet
  channels, raw/legacy PSD numerical agreement and pointer immunity.
- tests/test_lp_rest_labeler.py: six existing tests including
  non-destructive CSV handling.
- scripts/lp_psd_provenance.py: actual recording PSD encoding audit.
- scripts/lp_teacher_compare.py: read-only v1-v2 comparison across all
  LP/contact folders.
- scripts/lp_classifier_benchmark_v2.py: held-out-person
  student-imitation benchmark using firmware-compatible log PSD features.
- Detailed report:
  research/LP_CAUSAL_FIRMWARE_PSD_RESEARCH_2026-11-16.md

No code here changes the actual nRF firmware. All thresholds are
research hypotheses. Person-held-out imitation of a pseudo-label teacher
does NOT prove real muscle activity/disconnection/press-release accuracy.
