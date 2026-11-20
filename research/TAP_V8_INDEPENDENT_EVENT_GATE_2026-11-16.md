# TAP V8: independent physical-event evaluation gate (research only)

**Date:** 2026-11-16. **Status:** not production validated. **Branch:** `research/tap-v8-ground-truth-gate-20261116`.

## Why

V3/V5/V7 candidate discovery cannot prove the true onset of a physical tap. V5 initially assigned the recording-level TAP class to *every* candidate. That creates label contamination. V7 waveform models improved recording-level TAP-vs-FINGER AUC only to about 0.620 and did not establish onset accuracy. Several wearers show large cross-session shifts. The new evaluation code intentionally does not use Oculus, filenames, metronome or pseudo-labels as physical truth.

**This commit implements an event-level evaluation gate, not a new certified labeling algorithm. Do not promote V8 to the production dataset just because the test suite passes.**

## Ground truth collection contract

Create a separate, small, independently synchronized reference set (for instance device-visible contact sensor or synchronized high-frame-rate contact video with clock alignment), covering several people, multiple days and re-donned bands. Annotation must come from the physical tap/contact observation, not from Oculus, EMG candidate peaks or the model. Include soft single taps, normal taps, soft-soft/normal doubles with short and long gaps, intentionally broad contractions, isolated finger movements, wrist/arm/band movements, rest, partial electrode contact, one- or two-electrode dropouts. Record both positives and full-duration negatives.

Prepare UTF-8 CSV files:

* `recordings.csv`: `recording_id,duration_s,category,participant,session,contact`
* `truth.csv`: `recording_id,time_ms,event_type,pair_id` where `event_type=tap` and `pair_id` groups **exactly two** annotated taps in a double gesture.
* `pred.csv`: `recording_id,time_ms,score` with actual model onset predictions. Include **every** prediction, including in FINGER/NONE recordings. `score` is optional; if present, thresholds are applied.

The event timestamp is measured relative to recording start, in milliseconds. Keep the device-to-reference clock alignment error documented and smaller than the predeclared matching tolerance. Do not tune the tolerance after looking at results.

## Run

```bash
python research/tap_real/evaluate_physical_events_v8.py \
  --recordings recordings.csv \
  --truth truth.csv \
  --pred pred.csv \
  --tolerance-ms 80 \
  --threshold 0.5 \
  --out research/tap_real/results_v8/physical_gate.json

python -m unittest discover -s research/tap_real -p 'test_evaluate_physical_events_v8.py'
```

The evaluator performs an optimal ordered one-to-one tolerance-constrained matching. It reports precision, recall, F1, per-folder false predictions/hour and double-pair both-recovered count. Unlike recording-level AUC, these metrics can measure actual event quality when the reference annotations are independent. It reports `null` for undefined ratios.

**Caution:** The current metrics count a predicted onset outside the event matching tolerance as FP and the missed onset as FN; negative-folder predictions per hour are measured over full negative duration. Pair success requires both annotated taps to be separately matched. Pair hallucinations, accidental pairings, contact-conditioned results, calibration, latency quantiles, per-participant confidence intervals and causal firmware CPU/flash/RAM remain additional required gates.

## Proposed V8 development direction, NOT yet verified

1. Keep V3 physiological onset/rearm mining **high recall** without using Oculus as ground truth. Preserve one-channel events and independent per-electrode contact quality, but reject missing or invalid channels; do not require all three channels.
2. Offline train an abstaining **multi-instance** teacher using *recording-level bags*: positive bags contain at least one target gesture, negative bags contain no target gesture. Do not label every candidate in positive bags as tap. Candidate scores must incorporate temporal PSD and negative finger prototypes; separate TAP from SWIPE in the final target if needed.
3. Score candidates only on held-out participant **and session/date** splits. Treat Oculus with an estimated per-recording lag interval and unknown FN/FP as weak corroboration, never event-onset truth.
4. Auto-export positive silver labels only if a candidate exceeds an independently selected high-confidence threshold and has low hard-negative support. Positive recordings with no high-confidence events produce **no forced positives** and remain UNKNOWN.
5. Negative source recordings can supply negative examples, but quarantine signal-corrupted/uncertain frames. Candidate-free stretches in positive recordings must not be assumed negative near possible missing taps.
6. Evaluate doubles as two physical onsets with a real valley/rearm, not inferred from name or metronome. Compare pair recovery and false splits.
7. Freeze the existing V5/V6 models and output, and do **not** merge a new labeler into training until independently validated soft-tap, FINGER specificity, session shift and pair accuracy pass preregistered gates.

## Reporting checklist

Report per-gesture event precision/recall and onset error; per-negative-category FP/hour; double pair recovery and false splitting; per-person and per-session worst-case; partial-contact and channel-dropout slices; frozen threshold; subject/session held-out splits; firmware causal parity and resource usage. Declare baseline and pass/fail thresholds before calibration. Do not substitute high pseudo-label AUC for actual event performance.

See [V7 full pattern research](../TAP_PATTERN_DISCOVERY_V7_2026-11-16.md).
