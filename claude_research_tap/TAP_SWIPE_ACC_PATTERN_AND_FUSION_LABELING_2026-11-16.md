# DATA_TAP: the tap/swipe-specific pattern and a labeling recipe

2026-11-16 · source: `DATA/DATA_TAP` only

## Bottom line

The pattern that separates taps and swipes from everything else is in the accelerometer PSD, not the EMG. A tap or swipe onset is a short, compact burst in the ACC zones that rises and falls within about 200 ms, together with an EMG burst. The earlier V1–V7 research looked almost only at EMG shape, which is why it stalled at TAP-vs-FINGER AUC of 0.52–0.62.

On subjects the model never saw, at 2 false events per minute in the negative folders:

| Sensor family used | TAP presses recovered | SWIPE presses recovered |
| --- | --- | --- |
| EMG + WL only | 13% | 6% |
| ACC only | 78% | 79% |
| ACC + EMG + WL | 83% | 80% |
| ACC + EMG + WL, seeing at most 60 ms after onset | 81% | 79% |

The best labels do not come from the sensor pattern alone. They come from fusing it with PointerState: the pointer says roughly when, the sensor pattern confirms it, fixes the timing, and exposes the pointer's false and missed presses. That recipe is implemented and run on all 2,790 unique recordings.

Every number here is out-of-fold: each recording was scored by a model trained without that person.

## What the data contains

DATA/DATA_TAP holds 2,921 archives, 2,790 unique after removing 131 exact duplicates: TAP 1,246, SWIPE 896, FINGER 339, WRIST 149, ARM 113, BAND 42, NOTHING 5. That is about 6.5 hours of TAP/SWIPE and 2.5 hours of negatives.

Four facts about the recordings shaped everything else:

- **PointerState is never active in the negative folders.** All 648 negative recordings have zero pointer edges, zero TriggerPressure and constant InputX/InputY. So the pointer's false presses cannot be measured there; negatives must be judged from sensors alone.
- **In TAP/SWIPE the pointer is a real per-gesture anchor.** There are 22,599 usable presses. Tap presses last about 110–165 ms (median by subtype; SlowPinch about 550 ms) with no InputX/Y travel; swipe presses last about 350 ms with travel. No recording is stuck high or empty.
- **The metronome is not a timing reference.** At 60 bpm, presses land anywhere within about 450 ms either side of the tick. It is useful only as an expected count.
- **Logging rate differs by folder.** Negatives are mostly logged every 3 ms, positives every 5–6 ms. The features themselves update at a fixed rate (EMG about 46 Hz, ACC up to about 200 Hz), so everything was resampled to a 10 ms time grid. Any row-count-based feature would leak the folder.

The folders are also different cohorts. Most negatives are from Dec 2025–Feb 2026; a large part of TAP/SWIPE is from Jul–Aug 2026 with other people. The next section controls for this with same-person, same-day comparisons.

## The pattern and the evidence for it

The two weak truths were used against each other: the folder says what happened, the pointer says when. A pattern counts only if it is locked to pointer presses inside TAP/SWIPE files and absent from the negative folders.

**Method.** Each moment is described by a window of ACC zones, EMG zones and WL around it, baseline-corrected per recording. Positives are windows at pointer presses, snapped to the nearest ACC onset. Negatives are every 20 ms of every negative recording, plus the quiet parts of TAP/SWIPE files. A gradient-boosted model (LightGBM) is cross-fitted in five folds grouped by person.

**What the pattern looks like.** Averaged around the pointer press, a tap or swipe shows all six ACC zones rising together about 100 ms before the press, peaking about 40 ms after, and returning to baseline by 200 ms. The hard negatives differ in shape:

- Finger flexion: ACC rises and stays elevated for 400 ms or more, with a longer EMG burst.
- Tap on the band, hand or arm: a larger ACC burst with a dip before it and almost no EMG.
- Wrist and arm motion: sustained high ACC with no compact burst.

**Frame-level separation, held-out people, first-pass model.** Pointer-press windows versus all negative-folder frames: ACC alone AUC 0.980, EMG alone 0.880, WL alone 0.857, all together 0.986.

**Cohort control.** For 17 person-days where the same person recorded both TAP/SWIPE and negatives on the same date, ACC-only AUC ranges 0.957–0.999 and the combined model 0.975–0.999. The model had never seen that person. The separation is not a cohort or firmware artifact.

**Event-level, final model.** Share of pointer presses that have a detection within 150 ms, at a fixed false-event rate in negative folders:

| False events per minute in negatives | TAP | SWIPE |
| --- | --- | --- |
| 0.5 | 64% | 62% |
| 1 | 74% | 72% |
| 2 | 83% | 80% |
| 5 | 90% | 88% |
| 10 | 93% | 92% |

For scale, negative recordings contain about 53 paced movements per minute, so 2 per minute means roughly 4% of those movements fire. The EMG-only row in the first table comes from the first-pass model (20 ms features, 120 ms matching); its gap to ACC is far larger than that protocol difference.

**Where it is weak.** At the 1-per-minute threshold, false events concentrate in Fist&Stretch (4.0/min) and TaponHand (3.2/min); single-finger recordings give 0.5–1.0/min, ARM and wrist motion near zero. SlowPinch is the one positive subtype the pattern largely misses (45% at that threshold versus 68–80% for the others): a slow pinch has no impact.

## How wrong PointerState is

The pointer is a good anchor with three measurable defects.

**Timing offset.** Measured as ACC onset time minus pointer-down, on 19,962 presses with a clear ACC burst:

| Quantity | Value |
| --- | --- |
| Median offset, all presses | −15 ms (sensor leads the pointer) |
| Per-recording median offset, 5th–95th percentile | −70 to +50 ms |
| Jitter inside one recording (median absolute deviation) | 20 ms typical, 58 ms at the 95th percentile |
| Presses within ±50 ms of the sensor onset, raw | 69% |
| Presses within ±50 ms after removing the recording's own offset | 83% |
| Presses still more than 100 ms off after that | 7% |

So the offset is part per-recording and part per-press. A single shift per file is not enough; each press has to be snapped to its own sensor onset.

**Presses with no sensor support (likely false presses).** 3.3% of pointer presses have no pattern within 150 ms (score below 0.05). They have weaker ACC and EMG and longer press times (median 276 ms versus 195 ms). SlowPinch is the exception at 14%, where the gesture is real but has no impact. Nine recordings have more than half their presses unsupported.

**Missed presses.** 1,707 strong sensor events in TAP/SWIPE files sit more than 300 ms from any press, 4.4 per minute against 1.0 per minute for the same threshold in negatives. Independent check from rhythm: of the 856 that fall between two presses, 76% are in a gap at least 1.7 times the recording's normal press interval. Such gaps are only 6.6% of all gaps. That is where a skipped press would be. The other 851 are before the first or after the last press of a recording.

Rough size: about 3% of presses look false and about 3–6% of gestures have no press.

## Labeling recipe and what it produced

Label a gesture only where the pointer and the sensor pattern agree, take the time from the sensors, and mark every disagreement as unknown rather than guessing.

1. Resample each recording to a 10 ms grid and compute the out-of-fold sensor score (the model for a person is trained without that person).
2. Take score peaks at least 150 ms apart.
3. For each pointer press, find the strongest peak within 150 ms of pointer-down. The peak time becomes the gesture onset.
4. The gesture ends at pointer release shifted by the same amount. TAP or SWIPE comes from the folder.
5. Frames inside the gesture get label 1; a 100 ms guard band on each side and every disputed span get −1; everything else in the file gets 0. Negative folders are 0 throughout.

Result on all 2,790 recordings, with the confirm threshold set where negative folders give 1 false event per minute (score 0.851):

| Tier | Meaning | Count | Label |
| --- | --- | --- | --- |
| A | Press confirmed by a strong sensor pattern | 16,441 | 1, weight 1.0, sensor-timed |
| B | Press with a weaker pattern (score 0.05–0.85) | 5,402 | 1, weight 0.5, sensor-timed |
| C | Press with no sensor support | 756 | −1 |
| Edge | Press too close to the file start or end to score | 487 | −1 |
| D | Strong sensor event with no press nearby | 1,707 | −1 |
| E | Second strong event inside a long press | 614 | −1 |

Per frame that is 1,146,410 positive, 3,215,806 negative and 1,144,482 unknown. Tier A onsets moved by a median of −17 ms relative to the pointer, 5th–95th percentile −79 to +58 ms.

Tier B is real but softer evidence: 17–28% of presses in most subtypes and 41% of SlowPinch. Train with it at lower weight, and compare against a Tier-A-only run before deciding to keep it.

## For the embedded model, limits, next steps

**The pattern is available early.** A model restricted to data up to 60 ms after onset keeps almost all of the separation: 81% of TAP and 79% of SWIPE presses at 2 false events per minute, against 83% and 80% for the full offline window. The V6 students treated ACC as optional; ACC zones should be core inputs.

**Limits of this analysis.**

- There is still no independent onset truth. The evidence is agreement between three weak sources: folder, pointer, and press rhythm.
- PointerState was assumed to come from an independent physical input, not from an existing sensor-based detector. Its symmetric jitter around the sensor onset and the press durations fit that. If it is a detector output, the positives inherit that detector's bias.
- If the pointer device is what the finger strikes, part of the ACC burst may come from that contact. Check on a few taps recorded without the device before relying on ACC in the product.
- Negatives are scripted movements. Fist&Stretch and TaponHand are the weakest, and there is no free-living data.
- SlowPinch is poorly covered by the pattern; its labels are mostly Tier B or unknown.
- The offline scores normalise each channel by the whole recording's 10th percentile. Firmware needs a running baseline instead.
- People were grouped by the first four letters of the normalised User string, which merges aliases such as the several "ohad" and "nikol" entries but is not verified identity.

**Next steps.**

- [ ] Train the student on labels 0 and 1 with the weights, excluding −1, with ACC zones in the feature set.
- [ ] Score students by false events per minute in negative folders per subtype, and by confirmed-press recall per person, not by frame F1.
- [ ] Record more Fist&Stretch and TaponHand negatives, and some free movement.
- [ ] Keep the pointer device active during negative sessions so its false presses can be measured directly.
- [ ] Hand-check about 100 Tier C and Tier D spans to calibrate the false and missed press estimates.

## Reproducibility note

The per-recording label files, the event table with tiers and the scripts as run were delivered separately as `data_tap_fusion_labels.zip`; they are not committed here. Labels are offline teacher labels built from out-of-fold scores; −1 means unknown and must be excluded from the loss.
