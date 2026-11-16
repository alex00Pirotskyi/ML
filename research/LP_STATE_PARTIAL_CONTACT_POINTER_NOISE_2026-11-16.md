# LP state: partial contact and noisy pointer, real-data audit (2026-11-16)

## Status: experimental, NOT production approved

The user correctly notes that:
- Not all recordings have good contact.
- One channel can be good and responsive while the others are unusable.
- PointerState is affected by FP, FN and variable time offset.

Our earlier V4.1 hard requirement for two usable channels and exact pointer agreement is an inappropriate production assumption. This experiment evaluates an alternative V4.2 that forms labels from independently screened EMG channels and uses pointer only as a post-hoc diagnostic.

## Implementation

Source: lp_state_v4_partial.py
Tests: tests/test_lp_state_v4_partial.py
Audit: scripts/lp_state_v4_partial_audit.py
GitHub Actions success: https://github.com/alex00Pirotskyi/ML/actions/runs/43861828499

All 12 baseline V4 tests plus 10 V4.2 pointer-invariance and partial-contact tests passed. Full audit: 238/238 source recordings processed with zero errors; 101 LP, 44 quiet, 42 disconnected, 42 loose, 9 table.

V4.2 LP label contract: 1 = EMG-supported quiet/rest, 0 = EMG-supported active, -1 = unknown/exclude. The three channel-specific state and usability flags are NOT equivalent to physically validated signal-quality probabilities. PointerState NEVER makes or vetoes labels.

## Measured LP sample coverage (not accuracy)

| Group | V4.1 | V4.2 | V4.2 active-like rows |
|---|---:|---:|---:|
| Soft long press | 26.053% | 54.746% | 29.913% |
| Mid long press | 43.570% | 67.961% | 48.680% |
| Strong long press | 55.179% | 61.171% | 52.647% |
| Long press | 51.065% | 70.517% | 62.116% |
| All 101 LP files | 40.542% | 62.354% | 43.904% |

V4.2 accepted some active signal in 98/101 LP files. Per-file responsive-channel counts: 74 with 3 channels, 19 with 2, 5 with 1, 3 with 0. This is algorithmic recognition, NOT ground-truth contact: the electrode-quality gate can be wrong.

## Stress controls: an important blocker

| Group | Any active-like labels in files | % of rows active-like |
|---|---:|---:|
| CONNECTED + NOTHING (44 files) | 12 | 5.253% |
| DISCONNECTED (42 files) | 24 | 20.005% |
| LOOSE (42 files) | 27 | 17.008% |
| TABLE (9 files) | 0 | 0% |

These are NOT verified false-positive rates: real movement/contractions may exist in controls, and CH3 remained connected in DISCONNECTED captures. However, they demonstrate that simple one-channel EMG responsiveness is vulnerable to movement/contact artifacts.

Pointer/EMG IoU (diagnostic only): 0.509 no shift, 0.527 with best approximate -600..600 ms shift. The reference is noisy, so this is NOT accuracy or recall.

## Engineering conclusion

1. Keep independent per-channel quality and evidence; one well-supported responsive channel can be enough for a candidate label.
2. A high-dynamic channel is NOT necessarily good contact. Add validated artifact/contact checks; otherwise abstain rather than force activity.
3. Pointer is a weak event-level clue, never a hard framewise label; handle FP, FN and per-event offsets and allow EMG events with missing pointer support only when physically strong.
4. Keep all uncertain regions UNKNOWN (-1); do not map them to binary zero.
5. Independently validate sustained hold, false release while held, eventual release and timing on synchronized physical truth and compare the causal embedded model to the frozen v20 baseline under real tap-gated 10-prediction averaging.
6. Do not claim the nine CH1-3 rest/activity/signal-quality targets are production ready: signal quality cannot be identified reliably from these historical EMG features alone.

Neither V4.1 nor V4.2 has a verified production-grade accuracy score. High coverage alone is not the goal. Existing main, V3 and v20 are untouched.
