# Single / double tap on the device (2026-11-19)

`firmware/tap_gestures.h` turns the TAP_SWIPE onset score (Q20, one value per model run) into `TG_TAP` or
`TG_DOUBLE_TAP`. It uses integers only: 12 bytes of state and 67 Cortex-M4 instructions, with no calls.
`tap_gestures.py` is its host twin. `test_tap_gestures.py` (run in CI) checks that the C and Python give
identical events in both reporting modes on 200 random streams, including a wrap of the uint32 clock.

## Stricter defaults: 0.90 / 0.75 (2026-11-26)

`TG_FIRE_Q20` 17 → **18** (tap onset 0.85 → 0.90) and `TG_SECOND_Q20` 12 → **15** (2nd tap 0.60 → 0.75). Nothing
else changes. Back to the previous rule: `-DTG_FIRE_Q20=17 -DTG_SECOND_Q20=12`.

**Why the 2nd-tap threshold.** With the top-label model (`make train`'s `final14top`), the bump that follows a
single tap completes a false double: 5.7% of single taps come out as `TG_DOUBLE_TAP`. Real second taps now score
high: 93% reach Q20 ≥ 18 within 100 ms of their start. The false second bumps are spread over 0.60–1.0, and half
of them stay below 0.85. The relaxed 0.60 was chosen for the earlier model, whose second taps peaked at 0.55–0.80
on the device (below).

**Why the onset threshold.** False taps during movement peak lower than real taps. Within 100 ms of the onset,
real taps reach a median of 0.994 (10% below 0.96), while finger, fist and wrist false taps reach 0.91–0.94.
Requiring a higher level after the onset ("must reach X within N ms") did no better than the higher onset
threshold itself, so the rule keeps its form.

**Setup.** The research fold models of `tap_generalization_research.py` (gesture-aware file folds, each model
scores only its own validation files): the top-label arm (`final14top` recipe) and the onset arm (`final14`, the
model on the band now). Scores go to the device Q20 scale with one log-odds shift. Device 0.85 is anchored where
the research model has 4.4 false taps/min (pooled), the band baseline header's held-out rate; 2.8 and 6.2 are the
sensitivity check. "Scores −1 logit" lowers every score by one logit, as on the band, where live scores sat
below the recordings'. The data: 511 double pairs, 4,562 isolated single taps, 3,717 swipes and 120 minutes of
finger, fist, wrist, arm and band movement.

Deferred mode (the default), top-label model:

| `FIRE` / `SECOND` | Single → TAP | Single → DOUBLE | Double → DOUBLE | Swipe → TAP | Movement false taps/min (finger / fist / wrist / band) | Movement false doubles/min |
|---|---|---|---|---|---|---|
| 0.85 / 0.60 (before) | 84.0% | 5.68% | 94.1% | 83.1% | 5.14 (5.92 / 9.81 / 5.53 / 4.99) | 1.20 |
| **0.90 / 0.75 (default)** | **84.2%** | **3.59%** | **93.0%** | **82.3%** | **3.93 (4.59 / 7.68 / 3.98 / 3.88)** | **0.72** |
| 0.90 / 0.60 | 82.3% | 5.52% | 94.1% | 81.4% | 3.92 | 0.93 |
| 0.85 / 0.75 | 86.0% | 3.66% | 92.8% | 84.0% | 5.16 | 0.92 |
| 0.90 / 0.80 | 85.0% | 2.76% | 91.4% | 82.6% | 3.94 | 0.64 |
| 0.95 / 0.75 (strictest tested) | 81.4% | 3.55% | 92.4% | 77.8% | 2.44 (2.86 / 5.23 / 2.13 / 2.55) | 0.52 |
| *scores −1 logit:* 0.85 / 0.60 | 81.8% | 2.92% | 90.8% | 77.9% | 2.33 | 0.43 |
| *scores −1 logit:* **0.90 / 0.75** | 80.2% | 1.60% | 87.3% | 74.7% | 1.64 | 0.25 |
| *scores −1 logit:* 0.95 / 0.75 | 74.6% | 1.45% | 84.9% | 67.5% | 0.83 | 0.16 |

HID clicks (`TG_REPORT_TAP_IMMEDIATELY 1`, `TG_CHAIN_TAPS 1`), top-label model:

| `FIRE` / `SECOND` | 240 bpm taps clicked | Single tap + extra click | Movement clicks/min | Scores −1 logit: taps clicked / extra click |
|---|---|---|---|---|
| 0.85 / 0.60 (before) | 95.9% | 5.53% | 6.93 | 90.2% / 2.73% |
| **0.90 / 0.75 (default)** | **93.6%** | **3.42%** | **4.98** | 87.4% / 1.41% |
| 0.90 / 0.60 | 95.5% | 5.33% | 5.44 | 89.6% / 2.56% |
| 0.95 / 0.60 | 93.4% | 5.22% | 3.63 | 88.3% / 2.31% |

The onset model (on the band now), deferred mode:

| `FIRE` / `SECOND` | Single → TAP | Single → DOUBLE | Double → DOUBLE | Swipe → TAP | Movement false taps/min | Movement false doubles/min |
|---|---|---|---|---|---|---|
| 0.85 / 0.60 (before) | 86.2% | 4.06% | 90.2% | 83.7% | 6.12 | 1.55 |
| 0.90 / 0.75 | 86.2% | 2.08% | 87.5% | 82.5% | 4.36 | 0.83 |

Sensitivity, top-label model, 0.85 / 0.60 → 0.90 / 0.75:
- **Anchor 2.8/min:** movement false taps 2.47 → 1.75. Single → DOUBLE 3.00% → 1.67%. Single taps −1.3 points, doubles −2.9, swipes −2.9.
- **Anchor 6.2/min:** 8.47 → 6.45 and 8.57% → 5.77%. Single taps +1.9 points, doubles −0.4.

**Reading the tables:**
- **0.90 / 0.75 on the top-label model:** about a quarter fewer false taps during movement (finger and fist −22%, wrist −28%, band −22%) and 40% fewer false doubles there. Single taps read as doubles drop by a third. The cost is about 1 point of double taps and of swipes. Single taps stay level, because fewer of them turn into doubles. With scores one logit lower, it costs 1.6 points of single taps, 3.5 of doubles and 3.2 of swipes.
- **0.95 (`TG_FIRE_Q20 19`):** about half the movement false taps. It costs 2.6 points of single taps and 5.3 of swipes, and 7 and 10 with lower scores. Try it on the band only after 0.90.
- **The onset model:** the same setting removes 29% of movement false taps and halves single → DOUBLE, but it costs 2.7 points of doubles, because that model's second taps score lower.

Reproduce: `MUDRA_GR_CACHE=<research cache> MUDRA_GR_OUT=<research results with the fold models> python
research/gesture/tap_gestures_strict.py` writes `results_tap_gestures/strict_2026-11-26.json`.

## Rule

| Step | Condition |
|---|---|
| Tap onset | rising edge of the score to ≥ 0.90 (`TG_FIRE_Q20` 18; 0.85 before 2026-11-26), 150 ms refractory |
| 2nd tap of a double | inside 150–400 ms after the 1st onset, the score first falls below **0.20** (`TG_DIP_Q20` 4), then reaches **0.75** (`TG_SECOND_Q20` 15; 0.60 before 2026-11-26) |
| Single tap | the 400 ms window closes without a 2nd tap |
| After a double | onsets are ignored for 300 ms, so a triple tap is not reported as a double plus a single |
| Reporting | default: TAP and DOUBLE_TAP are exclusive, and a single TAP is reported 400 ms after its onset. `TG_REPORT_TAP_IMMEDIATELY 1`: TAP at once, then DOUBLE_TAP |
| `tg_cancel()` | drops a pending tap, e.g. when a swipe or long press is recognised inside the window |
| Fast taps, `TG_CHAIN_TAPS 1` | needs immediate reporting (HID clicks). There is no lockout: every accepted tap re-arms the window, so each next tap of a fast series (4–5 taps/s) needs only `TG_SECOND_Q20` after a dip. One click per `TG_TAP` and per `TG_DOUBLE_TAP`; the host forms double clicks itself |

Why the lower 2nd-tap threshold (2026-11-19, the earlier model; the sections below describe that model and the 0.85 / 0.60 rule): second taps score lower than first taps.

- **Device, new model:** 20 double taps. 18 first taps fired at 0.85, but only 11 second taps did; the missed second taps peaked at 0.55–0.80.
- **Replay:** 75% of second taps fire against 95% of first taps.

Why the dip: it stops the release of a single tap from counting as a 2nd tap. On the device, no single tap
showed any bump after its dip. That covers 41 single taps from the new model and 50 from Lean-lite (a bump
there would have peaked at 0.25 at most). In real double taps, the score falls below 0.20 between the taps
in 99.4% of pairs (replay).

## Device logs (the real exported models)

| Log | Two plain onsets in 400 ms | This rule (0.60 / 0.20) | 0.50 / 0.20 |
|---|---|---|---|
| 43 single taps, new model (165801) | 41 TAP, 0 DOUBLE | 41 TAP, 0 DOUBLE | 41 TAP, 0 DOUBLE |
| 5 double taps (175749) | 4 DOUBLE, 1 TAP | 4 DOUBLE, 1 TAP | 5 DOUBLE |
| 15 double-tap bursts (175806) | 7 DOUBLE, 7 TAP | 10 DOUBLE, 4 TAP | 10 DOUBLE, 4 TAP |
| 20 swipes, old model (201928) | 16 TAP, 0 DOUBLE | 16 TAP, 0 DOUBLE | 16 TAP, 0 DOUBLE |
| 60 s movement (202641) | nothing | nothing | nothing |

In 175806, two pairs had both taps at 0.80. Neither tap reached 0.85, so both pairs are missed under every
rule. Three bursts there have no second bump and may have been single taps.

## Fast taps (HID clicks, immediate reporting)

**Device logs:** three runs of fast tapping (191651, 191706, 191725) at 210–225 ms intervals, 267–284 taps/min.
- **Counting the taps:** score bumps ≥ 0.30 whose starts are less than 120 ms apart are merged into one tap. That gives 45, 53 and 48 taps (146 in all), which matches the tapping rhythm (46, 54 and 48).
- **Before the change:** the 300 ms lockout after a double tap swallowed every third tap, and a tap below 0.85 that did not follow a click was dropped.

| Rule | Fast taps clicked (device) | Clicks with no tap | Single / double / swipe / movement logs |
|---|---|---|---|
| Lockout, 2nd ≥ 0.60 (before) | 85/146 (58%) | 0 | unchanged |
| **Chain, next ≥ 0.60, 400 ms (default)** | **127/146 (87%)** | 0 | identical clicks |
| Chain, next ≥ 0.50 | 136/146 (93%) | 0 | 1 more double tap recognised |
| Chain, next ≥ 0.40 | 143/146 (98%) | 0 | 1 more double tap recognised |

At 0.60, 14 of the 19 missed fast taps peaked between 0.30 and 0.60.

The same rules on held-out replay, using the research models with thresholds on the device scale as below:

| Rule | 240 bpm taps clicked | 2nd tap of a double | Single tap + extra click | Movement clicks/min |
|---|---|---|---|---|
| Plain clicks at 0.85 | 71.9% | 76.3% | 2.09% | 3.60 |
| Lockout, 0.60 (before) | 64.6% | 85.9% | 2.51% | 3.71 |
| **Chain, 0.60, 400 ms** | **78.6%** | 86.4% | 2.51% | 3.82 |
| Chain, 0.50, 400 ms | 81.1% | 88.2% | 3.27% | 3.97 |
| Chain, 0.50, 500 ms | 82.3% | 88.2% | 3.39% | 4.12 |

The replay covers 491 taps from the 240 bpm recordings, 552 second taps and 4,308 isolated single taps.

**Reading the tables:**
- **Default (chain at 0.60):** it costs nothing extra on single taps compared with the lockout rule.
- **0.50:** it adds about 6 points of fast taps on the device. In replay it costs about 0.8 points more single taps followed by an extra click. Over HID, that extra click can turn a single tap into a double click on the host.

## Held-out replay (44 development people, 3 person folds)

**Setup:**
- **Model:** the research `raw126_diff+augm` models (`tap_generalization_research.py`), one per fold, each scoring only its own held-out people.
- **Calibration:** their probabilities are not on the exported header's scale. The trainer weights positives by 0.21 and the research models by 1, which shifts the log-odds. Thresholds are therefore given on the device scale and mapped by one log-odds shift. The shift is anchored so that device 0.85 lands where the research model has the deployed model's false rate (2.8/min). That puts device 0.85 at research 0.98, a shift of 2.16. The table also checks a second mapping, ln(1/0.21) = 1.54.
- **Truth:** every real pointer press.
- **Pairs:** consecutive presses less than 500 ms apart in the DoubleTaps recordings (546 pairs).
- **Singles and swipes:** presses with no other press within 600 ms (4,999 singles and 3,999 swipes). The 240 bpm single-tap recordings have double-tap timing and are left out.

| Rule (device scale) | Double → DOUBLE | Double → one TAP | Double → two TAPs | Single → TAP | Single → DOUBLE | Swipe → DOUBLE | Movement TAP/min | Movement DOUBLE/min |
|---|---|---|---|---|---|---|---|---|
| Two plain onsets in 400 ms | 71.6% | 23.3% | 3.5% | 84.7% | 1.34% | 0.78% | 3.25 | 0.16 |
| 2nd ≥ 0.70, dip 0.20 | 78.4% | 16.9% | 3.1% | 84.9% | 1.08% | 0.83% | 3.17 | 0.24 |
| **2nd ≥ 0.60, dip 0.20 (default)** | **81.9%** | 13.4% | 3.1% | 84.3% | **1.64%** | 0.95% | 3.09 | 0.31 |
| 2nd ≥ 0.50, dip 0.20 | 83.9% | 11.5% | 2.9% | 83.6% | 2.30% | 1.33% | 3.00 | 0.40 |
| 2nd ≥ 0.50, dip 0.20, window 350 ms | 79.5% | 11.9% | 7.0% | 83.7% | 2.22% | 1.05% | 3.06 | 0.34 |
| 2nd ≥ 0.50, dip 0.20, window 450 ms | 86.3% | 11.5% | 0.6% | 83.4% | 2.42% | 1.53% | 2.87 | 0.48 |
| Plain, second mapping (shift 1.54) | 78.6% | 17.2% | 3.5% | 87.0% | 2.30% | 1.43% | 5.25 | 0.38 |
| 0.60 / 0.20, second mapping (shift 1.54) | 85.9% | 10.1% | 3.3% | 86.7% | 2.56% | 1.35% | 4.85 | 0.75 |

**Reading the table:**
- **The default rule:** it recognises about 10 points more double taps. The cost is about +0.3 points of single taps read as doubles and about +0.15 double taps/min during movement, because some false taps there become false doubles.
- **Remaining misses:** what remains under "double → one TAP" is mostly a tap the model scored too low. Raising those scores needs model training, not this rule.
- **0.50 instead of 0.60:** it adds 2 more points of doubles at about +0.7 points of single taps read as doubles. On the device logs, 0.50 recovered one more pair (5/5 in 175749) and turned no single tap into a double. It is a one-define change (`TG_SECOND_Q20 10`) if device testing keeps false doubles at zero.
- **The 400 ms window:** the next tap of a double starts 247 ms after the first onset (median), and 99% start within 394 ms.

Reproduce (needs the research cache and the saved fold models):

    python research/gesture/tap_gestures_research.py        # results_tap_gestures/results.json
    python research/gesture/test_tap_gestures.py            # C vs Python
