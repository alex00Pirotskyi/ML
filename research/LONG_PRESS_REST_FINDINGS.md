# Long-press resting-state discovery — exploratory audit

Date: 2026-11-15. Source: DATA/DATA_LONG_PRESS in this repository,
2,431 compressed recordings in 12 folders. This is a read-only investigation
and NOT a trained classifier or a validated per-timestamp rest/press label.

## Ground rules for future LP work

1. Start from scratch: old LP models, thresholds and labels are NOT assumptions.
2. Goal is to discover common sEMG patterns when **not pressing/resting**.
3. PointerState and TriggerPressure are UNRELIABLE observations, not truth.
4. class_id is NOT a press/release label; LP, tap, and connected examples all
   contain class_id=1.
5. Contact quality differs by person, session and individual electrode. Directory
   names DISCONNECTED/LOOSE cannot be imposed on all CH1/CH2/CH3 at once.
6. All scripts here are read-only; main and dataset files must remain untouched.

## Dataset and procedure

- All 101 LP recordings: 41 MID, 36 SOFT, 20 STRONG, 4 contraction LP.
- 427 successfully analyzed comparison recordings: 37 drag/drop, 39 connected,
  42 disconnected, 42 loose, 5 nothing, 9 table, 123 tap, 130 swipe.
  Seven additional selected tap files were skipped as too short or invalid.
- Three distinct input schemas: 134, 135, 139 columns.
- Analyze half-second windows based on median recording timestamp spacing.
- Per recording, compute per-channel low baseline as 20th percentile of
  waveform length (WL). Compare lowest versus highest 20% of windows by
  median relative WL; these are exploratory low/high activity **proxies**,
  not verified rest/press labels.
- Spectral zone totals divided by 8, 10, 12, 46, 51 bins respectively.
- Baselines use the WHOLE recording (retrospective), not an online algorithm.

## Findings

### Sustained relative coactivation is common, but not universal

Heuristic: >=2 channels have WL >=1.4x their per-recording low baseline
for >=1 second. It is observed in:

| Recording group | Recordings with criterion |
| --- | ---: |
| Strong long press | 19/20 |
| Mid long press | 33/41 |
| Soft long press | 21/36 |
| Contraction long press | 4/4 |
| All long press | 77/101 |
| Connected reference | 2/39 |

The same exploratory criterion also occurs in 68/123 sampled tap and 65/130
sampled swipe recordings, so it CANNOT uniquely identify LP. It would also
miss many soft presses. These numbers are NOT accuracy, recall or precision.

### PSD power rises during WL-high periods

Median EMG PSD total power high-WL versus low-WL windows, per channel:

| Recording group | CH1 | CH2 | CH3 |
| --- | ---: | ---: | ---: |
| All LP (101) | 2.77x | 4.30x | 1.96x |
| Mid LP (41) | 3.37x | 5.19x | 2.15x |
| Soft LP (36) | 2.05x | 3.28x | 1.60x |
| Strong LP (20) | 5.32x | 4.00x | 2.27x |
| Connected reference (39) | 1.15x | 1.19x | 1.15x |
| Table/noise (9) | 1.05x | 1.02x | 1.02x |

Log-scale WL versus PSD total power within recordings are highly correlated
in LP files (~0.76 to 0.86 median by channel). These are related measurements
rather than independent votes. All five PSD zones commonly rise together:
there is no convincing single PSD zone shared across people that is uniquely
a rest indicator. Any shape/contact distinction needs separate verification.

### Fixed absolute WL thresholds cannot work across conditions

Median of per-recording WL low baselines:

| Group | CH1 | CH2 | CH3 |
| --- | ---: | ---: | ---: |
| Connected reference | 674 | 563 | 703 |
| Disconnected-labelled | 8,498 | 6,626 | 123 |
| Table/noise | 11,866 | 9,821 | 14,873 |

Contact noise can be very high but stationary; low signal on one channel may
also reflect placement. Directory contact labels must not override per-channel
evidence. These numbers are descriptive, not learned contact thresholds.

### Pointer State must not supervise training

In 27 of 101 LP recordings, pointer-on windows have LOWER median relative
WL activity than pointer-off windows. Therefore do not treat PointerState
or TriggerPressure as a true press/release timeline.

## Candidate architecture — to validate, NOT firmware-ready

1. **Separate per-electrode contact quality from press/rest activity.**
   Unreliable-all-channels should produce UNKNOWN, not confident REST.
2. **Estimate per-channel adaptive low-activity baselines** (WL plus total PSD
   power). Update only during high-confidence rest; freeze across holds.
3. **Rest candidate:** signal energy returns near its OWN baseline and remains
   stable across consecutive windows. Do not equate stationarity alone
   with rest; a held contraction can form a stable elevated plateau.
4. **Quality-weighted soft voting**, not mandatory 2/3-channel activation;
   single clean responsive channels must remain possible for soft holds.
5. **Temporal context/hysteresis:** avoid isolated tap bursts and
   motion interference. Motion alone is not a press signal.
6. No PointerState-derived ground truth, no unvalidated performance claims.

## Key limitation and next validation

No independent timestamps of true rest/press/release were available.
We only compared relative low/high WL windows in recording contexts.
Manually verify press/rest intervals across participants, intensities,
hands and known poor electrode placements. Then measure false-active at rest,
false-rest during sustained holds, and correct return-to-rest timing,
using participant-held-out evaluation.

## Replication

Read-only audit scripts on research/lp-rest-signatures branch:
- scripts/lp_rest_probe.py
- scripts/lp_rest_audit.py
- scripts/lp_rest_spectral.py

GitHub Actions:
- Schema probe: https://github.com/alex00Pirotskyi/ML/actions/runs/43753550480
- Relative activity audit: https://github.com/alex00Pirotskyi/ML/actions/runs/43753771785
- Spectral-zone audit: https://github.com/alex00Pirotskyi/ML/actions/runs/43753975241

The main branch and all recording bytes were not changed by this analysis.
