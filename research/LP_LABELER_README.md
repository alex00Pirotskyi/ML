# LP sEMG pseudo-label teacher

The script `lp_rest_labeler.py` generates per-row, per-channel exploratory
**regression pseudo-labels** for physiological resting, high muscle activity,
and contact-fault/disconnection evidence. It intentionally does NOT use
PointerState, TriggerPressure, class_id or folder labels as ground truth.

## Run

From the repository root on Ubuntu, with numpy/pandas installed:

```bash
python lp_rest_labeler.py
```

Default source: `DATA/DATA_LONG_PRESS`  
Default destination: `DATA/DATA_LONG_PRESS_LABELED`

Outputs preserve all original CSV cell text and source folder structure.
The input files are **never modified**. Original .csv.gz files stay compressed.
Existing outputs are skipped unless you pass `--overwrite`.

Quick run across every folder, 2 files from each folder:

```bash
python lp_rest_labeler.py --limit-per-group 2 --workers 2
```

Process the entire dataset with at most 12 workers by default (roughly 80%
of logical cores but capped to limit RAM use), or override with `--workers 16`.
Outputs are ignored by Git unless explicitly force-added, preventing a huge
automatic push from `git add .`.

## Columns (one row per input sample)

For CH1/CH2/CH3, 0..1 continuous scores:

- `lp_resting_ch1` ... `lp_resting_ch3`: resting evidence.
- `lp_high_activity_ch1` ... `lp_high_activity_ch3`: elevated muscle activity evidence.
- `lp_disconnection_ch1` ... `lp_disconnection_ch3`: severe contact-fault / disconnection evidence.
- `lp_unknown_ch1` ... `lp_unknown_ch3`: **abstention** where the other three are insufficient.

Within each channel, the four outputs add up to 1 (up to CSV rounding).
These are **heuristic scores, not calibrated physiological probabilities**.

Global:
- `lp_state` (binary): **1** only after all three channels appear connected and
  confidently resting continuously for ~180 ms; **0** for activity, suspected
  disconnection, uncertain contact, or startup / insufficient evidence.
- `lp_state_weight` (0..1): pseudo-label confidence for downstream learning.
  **Do not train on lp_state=0 as a verified active class when this weight is
  low.** Handle uncertainty and class imbalance appropriately.
- `lp_teacher_version`: reproducibility marker.
- `label_summary.json` summarizes per-group scores, counts and errors.

## Method / limitations

- Activity is estimated from 180 ms trailing medians of WL and total PSD per
  channel, normalized by that file's 15th percentile. Uses a smooth continuous
  response between ~1.08x and ~2x the low-energy floor.
- Disconnection is intentionally conservative: abnormally **high AND flat**
  WL on multi-second windows with a generally flat recording, plus obvious
  dead-sensor flatlines. Other bad-contact mechanisms remain UNKNOWN.
- PSD bands are summed without re-dividing by bin counts: older source files
  may have been mean-converted without an explicit conversion flag.
- The 15th percentile is calculated from the ENTIRE recording, so this is an
  **offline, non-causal teacher**. Firmware and streaming inference must learn
  or estimate resting floors causally.
- Contact scores may still confuse muscle activity, and quiet disconnections
  cannot be conclusively detected without independent electrode tests.
  Validate against real per-channel contact truth and real press/release times.
- Any channel suspected disconnected makes the global binary label 0, even
  when the other channels appear at rest, as specifically requested.
- Do not silently treat `lp_state` as pointer/gesture ground truth.

Research evidence and known failure modes:
`research/LONG_PRESS_DEEP_RESEARCH_2026-11-15.md` on research branch.
