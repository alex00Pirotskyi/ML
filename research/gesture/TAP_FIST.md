# Fists fire the TAP model: cause and fix (2026-11-26)

**Answer:** it is not normalization and not the long-press labels. The 14 KB model has no room to tell a fist
from a tap. At the same false-tap rate, reweighting only moves false taps from fists to wrist motion. A bigger
model removes them: **at the same taps as today, a 64 KB model fires 30% less on fists and 40–50% less on finger,
wrist and band motion**, and it keeps more taps when live scores drop. To get that protection instead of extra
taps, give it a stricter target:

    make train KB=64 ALT_KB=14 TARGET=5 FIST=3

`TARGET=5` keeps the big model at about today's tap recall; at the default target it would spend its capacity on
taps instead (+1.8 points, fists unchanged). `FIST=3` (`MUDRA_TAP_FIST_WEIGHT`) weights fist negatives ×3; with the
bigger model it takes fists from −30% to −36%. `KB=90` should be at least as good: in earlier research 128 KB beat
64 KB slightly. Costs and cautions are at the end.

## What the device shows

Three `model_test` logs with fist clenches (64 s, tap and long-press scores only, the model on the band now;
`tap_fist_research.py devlogs`):
- **27 false taps (25/min).** 11 of them reach the maximum score 1.0, so no threshold removes them. `tap_gestures.h` at 0.90 / 0.75 leaves 23, at 0.95 17, and at 1.0 11.
- **Each fist starts like a perfect tap.** The tap score reaches 0.95–1.0 for 60–140 ms.
- **Then the long press takes over.** The long-press score passes 0.90 within 400 ms in all 27, with a median of 200 ms after the onset, and stays up for 300–400 ms. To the models, a fist clench is the start of a long press.

The held-out recordings show the same thing. Fists are where the model fires most: 19.4/min at 0.85, against
12.5 for finger motion and 14.5 for wrist motion. The fist data is 61 Fist & Stretch recordings at 60 bpm, from 16
people: 2% of the replay rows, and a single protocol.

## What it is not

Research models: `tap_generalization_research.py` arms on the `final14top` data recipe, 14 KB, gesture-aware file
folds, scored only on validation files (`tap_top_research.py compare`, `results_tap_fist/`).

| Hypothesis | Test | Result |
|---|---|---|
| Normalization (`TAP_SWIPE_PREPROCESSING "none"`) | none needed | Trees give the same splits under any fixed per-feature scaling. The inputs already include differences against running means and gain-invariant Lean-lite fields |
| Long-press starts are left unknown, so "a press that becomes a hold" is never "not a tap" | `lpneg`: the labeller's unknown windows of long-press recordings (−300..+500 ms around every pointer-down and release, about 25,000 rows per fold) as negatives | fists unchanged (+0.3/min at 0.85); finger −0.9\*, band −1.6\* |
| Too little weight on fists | `fistw3`, `fistw10` (fist negatives ×3 / ×10) | at the same taps: fists −29% / −45%, but wrist +22% / +46% and finger +11% / +16%. The false taps move rather than disappear (the 2026-11-20 finding for `fingerw3` too) |
| Decide later, after the contact | `shift100`, `shift150` (positives 100 / 150 ms later) | at 6.2/min: taps −8 / −13 points, finger false taps +4.5 / +6 per min |

Is the information there? Rows where the research model fires, fist fires against real-tap fires, with a
dedicated classifier on the same inputs (file-grouped CV, `tap_fist_research.py separability`):

| when | AUC | fist fires removed, keeping 98% of taps |
|---|---|---|
| the moment it fires | 0.93 | 45% |
| +100 ms | 0.95 | 66% |
| +150 ms | 0.955 | 69% |

So the inputs separate fists from taps fairly well at the firing moment. The 14 KB model cannot also keep every
other separation it needs.

## What fixes it: capacity

Every arm at the 14 KB model's tap recall (93.9%, its 6.2/min point; `tap_fist_research.py match`,
`results_tap_fist/same_taps.txt`), false taps per minute of each movement:

| model | taps | fist | finger | wrist | arm | band | all |
|---|---|---|---|---|---|---|---|
| 14 KB (`final14top` data) | 93.9% | 10.66 | 6.78 | 7.55 | 1.97 | 6.43 | 6.09 |
| 14 KB, fists ×3 | 94.2% | 7.57 | 7.51 | 9.20 | 2.37 | 5.87 | 6.29 |
| 14 KB, long-press starts as negatives | 94.0% | 11.20 | 6.68 | 7.48 | 2.06 | 6.21 | 5.99 |
| **64 KB** | 94.1% | **7.46** | **3.95** | **4.46** | 1.45 | **3.21** | **4.91** |
| **64 KB, fists ×3** | 93.9% | **6.83** | 4.03 | 4.98 | 1.23 | 2.88 | 4.94 |
| *band moved:* 14 KB | 92.3% | 12.69 | 7.77 | 6.62 | 1.05 | 4.43 | 6.05 |
| *band moved:* 64 KB, fists ×3 | 93.0% | 8.32 | 5.19 | 3.81 | 0.61 | 2.22 | 4.94 |
| *scores −1 logit:* 14 KB | 89.1% | 5.44 | 3.08 | 2.78 | 1.18 | 3.33 | 4.11 |
| *scores −1 logit:* 64 KB, fists ×3 | **91.2%** | 4.59 | 2.36 | 2.75 | 0.97 | 1.55 | 3.96 |

At the device threshold 0.85, the 64 KB model has the same taps as the 14 KB model (96.2% mean over the tap kinds),
but fires 35% less on fists (12.7 vs 19.4/min), 42% less on finger motion, 34% less on wrist motion and 56% less on
band motion.

## Cautions

- **Research models vs the trainer.** These are research models with fixed settings, compared with each other. The `make train` header is chosen by Optuna, so expect the same direction, not the same numbers. `TARGET=5` is the trainer-scale counterpart of "the 64 KB model at today's taps" (4.9 false taps/min overall).
- **CPU and flash.** A 64 KB model is about 370 trees of up to 15 leaves, against about 170 of 7 at 14 KB, so the tree walk per frame costs a few times more.
- **More fist data helps any model.** All fist recordings follow one protocol. Record more variants with the normal recorder: fast clench, clench and hold, clench with wrist movement, different arm positions. Put them in `DATA_TAP/FINGER` with "Fist" in the file name, and `FIST=3` makes them count three times.
- **Strictness stacks.** The 0.90 / 0.75 default of `tap_gestures.h` and a stricter target both cost taps. If taps feel missed on the band, return the gesture layer to 0.85 / 0.60 first.
