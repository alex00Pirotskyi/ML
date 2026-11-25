# TAP research with the gesture-aware file split (2026-11-24)

**Question:** now that the split and folds are by recording file (the change that made the band about ten times
better on new people), which labels, training method, model size and inputs give the best 14 KB TAP model?

**Answer:** use the 60 ms observable-pattern labels, recipe `final14obs`. It is the only change that is better in
every view, confirmed on 3 training seeds. Everything else either trades one kind of error for another or is worse.
At the false-tap level `make train` aims for, every 14 KB variant tested is within a quarter point of tap recall
(except trees of 31 leaves, -1.3). The labels and training tricks now shift *which* errors happen, not how many. Model size still matters: 10 KB and
8 KB lose taps.

`make train` now uses `final14obs` by default. A second recipe, `final14protect`, is there for the case where finger
false taps stay the main complaint on the band (section 3).

## Setup

- **Harness:** `tap_generalization_research.py` with `MUDRA_GR_SPLIT=file`, the same split as `make train`:
  - 22% of the recording files of every type are held out.
  - 3 cross-validation folds, stratified by recording type, over 2,339 training files.
  - Cache of 3,005 recordings, built before the 29 finger recordings of 2026-11-24.
- **Every variant:** the same fixed LightGBM settings (no Optuna), 14 KB and trees of 7 leaves unless named, and the
  final14 inputs (diff + Lean-lite diff + mild placement augmentation + press2 labels).
- **Yardstick:** the validation files of each fold, scored like the firmware:
  - recall of every real pointer press, per gesture type;
  - false taps per minute, per kind of non-tap recording.
- **Four views:**
  - **0.85:** the firmware threshold, as trained.
  - **Strict:** each variant at the threshold that gives 2.0 false taps/min over all non-tap recordings. This is
    high protection (threshold about 0.997).
  - **Trainer:** recall of confirmed onsets at 6.2 false taps/min in the trainer's own measure, the level Optuna
    aims for today. Moving the threshold stands in for Optuna's positive-class weight.
  - **Band moved:** the trainer view under the simulated placement change (EMG gain, tilt, contact noise).

## Results

Baseline values in the first row; the other rows are differences to it. Recall is in points and false taps in
events per minute of that kind of recording; negative false-tap numbers are better. Seed 0. The two recipes were
repeated with seeds 1 and 2 (next section).

| Variant | KB | 0.85 normal | 0.85 soft | 0.85 finger/min | 0.85 fist/min | 0.85 all/min | strict normal | strict soft | strict double 2nd | strict fast 240 | strict swipe | trainer recall | band moved |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline (final14-like) | 13.9 | 98.3 | 95.0 | 14.2 | 23.8 | 10.3 | 61.8 | 60.6 | 39.3 | 45.1 | 51.9 | 98.50 | 98.01 |
| **observable labels 60 ms** | 13.9 | -0.1 | -0.2 | **-1.7** | **-2.1** | **-0.71** | +0.6 | +1.2 | +1.7 | **+3.4** | +0.7 | +0.04 | +0.28 |
| observable labels 40 ms | 13.9 | -0.5 | -0.5 | -2.7 | -1.2 | -1.26 | +1.5 | +1.9 | +2.0 | +5.7 | +1.7 | -0.01 | +0.24 |
| soft taps as positives everywhere | 13.9 | +0.2 | +0.4 | +2.0 | +2.5 | +0.70 | +1.1 | +0.4 | -1.1 | +1.7 | +0.9 | -0.07 | -0.11 |
| no press2 labels | 13.9 | -0.1 | +0.1 | -0.3 | -1.4 | -0.51 | +1.4 | +1.1 | **-4.2** | **-3.3** | +0.6 | +0.12 | +0.25 |
| distillation, half label / half teacher | 13.9 | -0.1 | -0.7 | -0.9 | +0.1 | -0.63 | +3.1 | -0.5 | +1.0 | +1.7 | +1.3 | -0.21 | -0.02 |
| distillation, 1/4 label | 13.9 | -0.3 | -0.7 | -1.0 | -1.7 | -1.09 | +2.8 | -1.0 | +2.5 | +1.8 | +1.2 | -0.13 | -0.17 |
| finger / fist negatives x2 | 13.9 | -0.5 | -0.6 | -3.3 | -4.7 | -0.86 | +1.6 | +1.1 | +2.8 | +3.1 | +1.1 | -0.12 | +0.22 |
| obs 40 + distillation | 13.9 | -0.7 | -1.2 | -4.0 | -3.1 | -2.00 | +1.4 | -0.1 | +1.6 | +2.5 | -0.2 | -0.08 | +0.13 |
| **obs 40 + distillation + finger x2 (`final14protect`)** | 13.9 | -0.9 | -1.3 | **-5.8** | **-7.8** | **-2.34** | **+4.6** | +2.1 | **+7.6** | **+8.1** | +3.3 | -0.14 | +0.21 |
| 10 KB | 9.9 | +0.1 | -0.4 | +2.1 | +3.2 | +0.69 | -0.9 | -0.3 | -2.6 | -1.9 | -0.9 | -0.38 | -0.33 |
| 8 KB | 7.9 | -0.0 | -0.6 | +3.7 | +7.8 | +1.36 | -1.3 | -0.8 | -3.5 | -5.5 | -1.5 | -0.82 | -0.65 |
| trees of 4 leaves | 14.0 | +0.1 | -0.6 | +1.3 | +1.7 | -0.05 | -1.0 | -2.7 | -3.1 | -2.4 | -0.6 | -0.19 | -0.25 |
| trees of 15 leaves | 13.9 | +0.2 | -0.1 | +1.4 | +3.9 | +0.54 | +1.5 | +2.9 | +1.2 | +2.8 | +1.8 | -0.19 | +0.01 |
| trees of 31 leaves | 13.9 | +0.1 | -0.8 | +3.5 | **+16.7** | +2.01 | +1.6 | **-9.7** | -1.4 | -4.9 | -2.9 | **-1.30** | **-1.05** |
| no absolute EMG levels (acc_emgrel) | 13.9 | -0.0 | -0.1 | +0.3 | -1.6 | +0.15 | -0.2 | +0.4 | +2.8 | -0.2 | +0.1 | -0.05 | +0.06 |
| no augmentation | 13.9 | -0.0 | -0.3 | -0.1 | +0.6 | -1.41 | +1.5 | +0.7 | -0.8 | +2.1 | +2.5 | +0.24 | **-0.76** |

### Seeds

3 LightGBM seeds per arm; the difference is taken to the baseline with the same seed. Mean, with the
[smallest, largest] seed in brackets.

| | 0.85: all false/min | 0.85: finger / fist per min | 0.85: normal / soft recall | strict: normal / soft / double 2nd / fast 240 | trainer recall at 6.2 |
|:--|--:|--:|--:|--:|--:|
| obs 60 ms (`final14obs`) | -0.74 [-0.87, -0.63] | -2.0 / -1.4 | -0.16 / -0.27 | +0.6 / +1.0 / +1.2 / +2.9 | +0.06 [+0.03, +0.12] |
| `final14protect` | -2.09 [-2.34, -1.87] | -6.0 / -7.0 | -1.0 / -1.5 | +3.4 / +1.6 / +6.5 / +5.9 (each seed > 0) | -0.17 [-0.22, -0.14] |

At the trainer's own false-tap level (6.2/min, where Optuna puts the model), the mean over the 3 seeds is:
- **`final14obs`:** keeps slightly more taps (normal +0.1, soft +0.4, double +0.4, fast +0.5 points).
- **`final14protect`:** moves false taps from finger motion (-2.3/min, -20%) to the wrist and band recordings
  (+2.0 and +4.5/min) and keeps 0.1-0.3 points fewer taps.
- **Timing:** `final14protect` fires about 5 ms later (median at 0.85).

## What it means

1. **Labels: observable 60 ms (`final14obs`).** This is the consistent gain:
   - fewer false taps at 0.85 (finger -2.0, fist -1.4, band -2.5 per min) for 0.2 points of tap recall;
   - at strict protection, more taps kept (soft +1.0, fast +2.9);
   - more robust when the band moves (+0.28).

   40 ms is a bit more protective and a bit less sensitive than 60 ms.
2. **Keep press2.** Without it, fast and double taps are lost at strict protection (-3.3 fast 240, -4.2 double 2nd).
3. **Don't teach soft taps as positives.** It adds false taps (finger +2.0, fist +2.5 per min at 0.85) and keeps
   no more taps at the trainer's level. Same verdict as with the person split.
4. **Distillation** (the small model learns a full-size teacher's confidence, half label, half teacher) helps
   only at strict protection. On its own it costs soft taps. It is useful only inside `final14protect`.
5. **Maximum protection: `final14protect`** (obs 40 ms + distillation + finger/fist negatives x2).
   - At 0.85 it fires far less on finger motion and fists (-6.0 and -7.0 per min, about -40% and -30%) for about
     1 point of taps.
   - At strict protection it keeps the most taps of everything tested (+3 to +8 points).
   - At Optuna's level it mostly moves false taps from fingers to wrist/band motion.

   Use it if finger false taps on the band still matter more than the last point of soft taps. Optuna spends any
   protection margin on recall up to the false-tap target, so to keep its protection give it a stricter target
   (`MUDRA_TAP_FALSE_PER_MIN_TARGET`).
6. **Size: 14 KB is still the best budget.** 10 KB and 8 KB keep fewer taps and fire more on fists.
7. **Tree size: keep 4-15 leaves.** 31 leaves is clearly worse (fist +16.7/min, soft -9.7 at strict); 4 leaves on
   its own is worse than 7; 15 is mixed.
8. **Keep the mild augmentation.** Without it the scores on the recordings as recorded look better, but recall
   drops when the band moves (-0.76).
9. **Absolute EMG levels make no difference** (acc_emgrel ≈ baseline); the current inputs stay.

## Maximum protection at 85-90% of taps (2026-11-25)

The question: keep 85-90% of taps and fire as little as possible on finger motion, fists, wrist and band motion.

False taps per minute at the same share of normal taps kept (threshold moved until normal-tap recall matches):

| normal taps kept | config | trainer target | soft taps | finger | fist | wrist | band |
|---|---|---|---|---|---|---|---|
| 88% | 14 KB obs60 (`final14obs`) | 1.8 | 83.4% | 1.86 | 4.10 | 2.81 | 1.61 |
| 88% | 14 KB protect | 1.9 | 83.2% | 1.64 | 4.00 | 3.31 | 2.83 |
| 88% | **32 KB obs60, 15 leaves** | **1.5** | 84.5% | **1.06** | **2.55** | **1.41** | 1.37 |
| 88% | 32 KB protect, 15 leaves | 1.6 | 83.1% | 1.03 | 3.07 | 1.72 | 1.79 |
| 85% | 32 KB obs60, 15 leaves | 1.3 | 81.3% | 0.76 | 1.56 | 1.09 | 1.10 |
| 90% | 32 KB obs60, 15 leaves | 1.7 | 86.6% | 1.50 | 3.42 | 1.95 | 1.60 |

For comparison, the 14 KB obs60 model at the old header's target (6.2) keeps 97.9% of normal taps and fires
10.3 / 18.3 / 14.8 / 17.2 per minute (finger / fist / wrist / band).

1. **32 KB is what buys protection.** At 88% of taps it fires about 40% less on fingers and fists than 14 KB.
2. **At 32 KB, `final14obs` beats `final14protect`.** Finger false taps are the same. Protect fires more on fists,
   wrist and band, and keeps fewer soft taps. The finger weight only moves errors to other motions.
3. **The firmware threshold stays 0.85, so the trainer needs another way to reach a strict target.** A lower
   positive-class weight does not work. At 32 KB and 0.85:

   | positive weight | false taps/min | recall |
   |---|---|---|
   | 1 | 5.5 | 0.989 |
   | 0.05 | 2.37 | 0.956 |
   | 0.01 | 2.25 | 0.938 |

   The weight saturates. At the same false rate it also keeps 1-2 points fewer taps than moving the threshold.
   The trainer's 0.2 floor stops near 3/min.

   So with a numeric target below 4/min, Optuna also searches a **logit shift** `s` in 0..5:
   - `s` is subtracted from every leaf of the first tree. That is exact in Python and in the C export, whose
     leaves are float.
   - The firmware's 0.85 then means probability `sigmoid(logit(0.85) + s)` of the fitted model. For example,
     `s = 3` gives 0.991, the 1.5/min point.
   - Setting: `MUDRA_TAP_LOGIT_SHIFT` (auto / off / lo,hi). The run gets its own folder and study (`-shift0-5`).

**Command:** `make train KB=32 ALT_KB=14 TARGET=1.5`. This keeps about 88% of taps on people in the data, and
somewhat fewer on new people. If the band test keeps too few taps, use `TARGET=2` (about 92%); for even more
protection, use `TARGET=1.2` (about 83%).

## More taps at the same protection (2026-11-25, after the band test)

**Band test of the 32 KB / target 1.5 model:**
- Pinky and taps on the hand: very good.
- Twist: better.
- Taps: much less sensitive, many more misses.

The research picture agrees. Against the old 6.2/min point, these tap types are lost:

| tap type | points lost |
|---|---|
| normal | 10 |
| soft | 11 |
| second tap of a double | 21 |
| fast (240 bpm) | 16 |
| swipe | 15 |

The question: what keeps the protection and brings taps back?

The candidates were compared at the trainer's 1.5/min, which is what `make train TARGET=1.5` produces. Movement
false taps come out equal or lower than for the tested model.

| config | normal | soft | 2nd of double | fast 240 | swipe | finger /min | fist /min | wrist /min | band /min |
|---|---|---|---|---|---|---|---|---|---|
| 32 KB (band-tested) | 88.2% | 84.7% | 74.7% | 80.8% | 78.9% | 1.10 | 2.64 | 1.44 | 1.41 |
| **64 KB** | **90.4%** | **86.1%** | **80.3%** | **85.2%** | **81.5%** | 1.02 | 2.75 | 1.43 | 0.85 |
| 128 KB | 90.9% | 85.5% | 82.3% | 85.9% | 82.2% | 1.08 | 2.73 | 1.21 | 0.98 |

All three rows use the same recipe (`final14obs`: obs60, press2, diff + lite, mild augmentation) with 15 leaves.

**Under placement stress:** at the same movement false rate, the gains are about half:
- 64 KB: normal +1.1, second taps +4.4, fast +2.4.
- 128 KB: +2.2, +6.6, +3.7.

**Tested and left out.** Differences to 32 KB at the same movement false taps (finger + fist + wrist + band).
For scale, seed noise is about ±2 points (±3 on second taps).

| variant | normal | soft | 2nd of double | fast | verdict |
|---|---|---|---|---|---|
| XGBoost, depth 4, same bytes | −0.6 | −1.7 | −1.8 | −0.5 | no gain |
| XGBoost, depth 5, same bytes | −2.7 | −4.5 | −5.5 | −5.6 | worse |
| firmware rule: 2 frames in a row ≥ threshold | +0.1 | +0.5 | +0.7 | 0.0 | no gain |
| 100 ms window instead of 60 (`obs100`) | −0.6 | −1.6 | +0.2 | −0.9 | no gain, more latency |
| soft taps of every recording as positives | +0.2 | −0.8 | −4.0 | +1.5 | no gain |
| double / 240 bpm positives ×2 (`pw2`), 32 and 64 KB | −0.5 to +0.2 | −0.8 to −1.8 | +1.4 to +2.6 | −0.7 to +1.8 | noise |
| 64 KB, 31 leaves, depth 10 | −1.2 vs 64 KB / 15 leaves | | | | smaller trees win |
| 64 KB + ACC differences | +0.4 vs 64 KB | | | | fist +0.9/min: left out |

**Cost:** 64 KB is about 367 trees of up to 15 leaves, against 183 at 32 KB, so about twice the tree walk per
frame; 128 KB is about 4×. The C export supports both: root offsets switch to uint32 past 65,535 nodes.

**Command:** `make train KB=64 ALT_KB=32 TARGET=1.5`. If flash and CPU allow, use `KB=128 ALT_KB=64` for a
little more on second taps and fast taps.

Reproduce: same as above with `P=raw126_diff+augm+litediff+press2`:
- the arms `$P+kb64+leaves15+obs60` and `$P+kb128+leaves15+obs60+trees900`;
- `$P+kb32+obs60+xgb` (needs `pip install xgboost`);
- `rule $P+kb32+leaves15+obs60 min2` (re-scores saved fold models with another decision rule).

### Can 1–4 extra inputs replace the bigger model? (2026-11-25)

No. Each candidate was added to the 32 KB model (the same tree bytes). Differences to 32 KB at the trainer's 1.5/min:

| extra inputs | normal | soft | 2nd of double | fast | fist /min | under placement stress |
|---|---|---|---|---|---|---|
| `lrscore`: one stacked linear tap score (logistic regression on all inputs, ~1.1 KB of weights) | +0.2 | −0.8 | +1.5 | +3.3 | +0.44 | gains vanish |
| `accsum`: ACC totals (now, now − mean, high zones, rate) | 0.0 | −0.9 | −0.2 | +0.3 | −0.06 | within noise |
| `emgsum`: rise10 sum / min / max and xm sum across channels | −0.3 | −1.1 | −0.5 | +0.6 | +0.10 | within noise |
| (for scale) 48 KB, no new inputs | +1.6 | +1.1 | +4.0 | +3.7 | +0.31 | about half |
| (for scale) 64 KB, no new inputs | +2.2 | +1.4 | +5.6 | +4.4 | +0.11 | about half |

The gain is capacity, not missing information. If flash is the limit, a 4-byte node instead of 6 bytes would hold
1.5x the trees in the same flash: int16 thresholds per feature scale and int16 leaves, instead of float. The
per-frame work would then be that of the bigger model.

### Follow-up tap rule for the strict model (2026-11-25)

`firmware/tap_gestures.h` (immediate + chain) clicks a second time when the score dips below `dip` and comes back
above `second`, 150-400 ms after a tap. Both thresholds are on the device scale, so the logit shift makes them
stricter as well: at s = 3, device 0.60 means 0.968 of the fitted model. Tested with `tap_followup_rule.py` at the
trainer's 1.5/min, using the same models:

| model | second / dip (device) | 2nd tap of double | 240 bpm taps | single tap + extra click | movement clicks /min |
|---|---|---|---|---|---|
| 32 KB | no rule | 76.3% | 81.6% | 1.41% | 1.17 |
| 32 KB | 0.60 / 0.20 (today) | 87.6% | 87.9% | 1.73% | 1.32 |
| 32 KB | **0.40 / 0.10** | **91.6%** | **90.7%** | 2.28% | 1.42 |
| 32 KB | 0.25 / 0.05 | 92.7% | 91.8% | 3.33% | 1.60 |
| 64 KB | 0.60 / 0.20 (today) | 88.9% | 90.6% | 1.72% | 1.21 |
| 64 KB | **0.40 / 0.10** | **92.0%** | **92.5%** | 2.31% | 1.25 |
| 64 KB | 0.25 / 0.05 | 92.9% | 93.5% | 3.03% | 1.39 |

The rule only acts within 400 ms of a tap, so single taps and movement far from taps keep their protection.
`TG_SECOND_Q20 = 8, TG_DIP_Q20 = 2` (0.40 / 0.10) wins back most of the second taps and fast taps for about half a
point more extra clicks.

## Band test of 64 KB / 1.5: taps lost, trainer changed (2026-11-25)

**On the band:** protection worked, but even mid taps were missed. It was the worst tap model so far.

**What the trainer picked** (`logs/tap_final14obs_splitfile_20261125_155615.log`):
- Trial 29: positive weight **5.7** plus logit shift **4.7**.
- Device 0.85 therefore meant **0.9984** of the fitted model.
- Lowering the device threshold could not help: device 0.50 = 0.991, and 0.20 = 0.964.
- The other top trials used shifts of 3.3–5.0, with weights from 0.2 to 5.7.
- Offline everything looked fine: cross-validation 91.8% at 1.52/min, held-out files 93.1%, parity exact.

**On held-out people** (person split, the research cache), the same weight / threshold pair keeps 86.4% of
normal taps. That is within 1–2 points of weight 1 at the same movement false taps. So the offline data does not
show the collapse. The band does: live scores sit lower than in the recordings, and an operating point this deep
in the score's tail has no margin for that.

**Trainer changes:**
1. With the shift searched, the TAP positive weight is fixed at 1. The two moved the same thing, and Optuna paired
   them.
2. The shift range is 0..3. At s = 3, device 0.20 is 0.83 of the model, so the device threshold can always return
   to the old operating point.
3. The study and folder tag is `-shift0-3-pw1`, so the old trials are not reused.

**Strictness on known vs new people**, at the threshold of each trainer target (file split = known people,
person split = new people; same model threshold):

| model | target | known: normal / soft / 2nd / fast | known: finger / fist | new: normal / soft / 2nd / fast | new: finger / fist |
|---|---|---|---|---|---|
| 32 KB | 6.2 (default) | 98.1 / 95.4 / 95.7 / 96.8 | 10.3 / 19.1 | 98.1 / 94.6 / 93.3 / 95.2 | 14.5 / 31.0 |
| 32 KB | **3** | **95.6 / 92.2 / 90.3 / 92.9** | **3.9 / 7.4** | **95.1 / 90.5 / 87.2 / 86.8** | **6.2 / 14.2** |
| 32 KB | 1.5 | 88.2 / 84.7 / 74.7 / 80.8 | 1.1 / 2.6 | 86.4 / 81.5 / 71.5 / 70.0 | 2.3 / 4.8 |

**Recommendation:**
1. Roll back to the last good header.
2. Next, `make train KB=32 ALT_KB=14 TARGET=3`: about 40% of the default's false taps, for 2.5–5 points of taps (normal −2.5, second taps −5).
3. Go stricter only in steps, checked on the band, and with more recordings from more people.

**Model size limit:** `MAX_MODEL_SIZE_BYTES = 102400` (100 KB) is fixed in the trainer. Larger budgets are refused
at start (and by `make train`), and every export is checked against it.

## Final pipeline (2026-11-25): `make train`

The goal: every tap kind works, no movement makes taps, and Optuna keeps the model as small as it can. No extra
tricks.

**Data recipe** (`final14obs`, unchanged, all researched above):
- press2 labels, so every press of a double / 240 bpm recording is a tap;
- observable 60 ms windows;
- diff + Lean-lite inputs;
- mild placement augmentation;
- trees of 4–15 leaves;
- the file split.

**What is new:**
1. **Balanced objective.** Score = mean recall over the tap kinds × min(1, 7 / the worst movement kind's false
   taps per minute).
   - Tap kinds (from the file name): soft, mid, strong, double, fast (240 bpm), other taps, swipes.
   - Movement kinds: finger, fist (Fist & Stretch), wrist (twist, wrist moves, tap on hand), arm, band.
   - So a soft or double tap counts as much as a strong one, and fist cannot hide behind the other movements.
     `make train LIMIT=5` is stricter.
2. **One operating control:** the threshold shift (0..3, positive weight fixed at 1). The device threshold keeps its
   full range.
3. **Size ladder.**
   - Each trial is fitted once at 100 KB and scored at 8, 12, 16, 24, 32, 48, 64 and 100 KB. The first trees of a
     boosted model are exactly the smaller model.
   - Optuna maximises score − 0.0001 per KB, so a bigger model has to be 1 point better per 100 KB.
   - The header is the best (trial, size). The best raw score is exported beside it (`models/budget_<kb>kb`).
4. **No finger weighting, distillation, soft labels or strict false-tap target.**

**Expected level** (known people, fist ≈ 7/min):

| model | normal | soft | double | fast | swipe | finger /min | wrist /min | band /min |
|---|---|---|---|---|---|---|---|---|
| 32 KB | 95.3% | 92.0% | 90.0% | 92.6% | 89.0% | 3.8 | 5.7 | 4.3 |
| 64 KB | 96.0% | 92.5% | 91.3% | 93.7% | 90.0% | 3.6 | 4.9 | 4.0 |

Compare the old default: fist 19/min, finger 10/min, for taps of about 98%.

## Limits

- **These numbers describe people who are in the data.** File-split scores are optimistic for a new user. The
  literature on record-wise vs subject-wise validation of wearable data shows the same (Saeb et al. 2017,
  "The need to approximate the use-case in clinical machine learning"). The band test with new people decides.
- **The bootstrap marks in the working tables resample files.** Files of one person are correlated, so those
  intervals are too narrow. That is why the two recipes were repeated with 3 seeds.
- **Every variant used fixed LightGBM settings.** `make train` re-tunes each recipe with Optuna (K inputs, leaves,
  positive weight).
- **Not measured here:** electrode disconnection (no recordings of the trigger yet) and the newest recordings.

## Trainer options added

- **`MUDRA_TAP_DISTILL=a`:** soft targets `a*label + (1-a)*teacher`.
  - The teacher is a full-size LightGBM (no budget) on all inputs, cross-fitted by file inside each training set.
  - Trained as the usual binary model, two weighted rows per frame. The size accounting and C export are unchanged.
- **`MUDRA_TAP_FINGER_WEIGHT=k`:** FINGER-folder negatives weighted x k before class balancing; the teachers keep
  plain weights.
- **Recipe `final14protect`:** `final14` + `MUDRA_TAP_OBS_MS=40` + `MUDRA_TAP_DISTILL=0.5` + `MUDRA_TAP_FINGER_WEIGHT=2`.

Each option gets its own results folder and study (`-distill50`, `-fingerw2`).

## Reproduce

    export MUDRA_GR_SPLIT=file MUDRA_GR_STRESS=nominal,placement
    B=raw126_diff+augm+litediff+press2+kb14+leaves7
    python research/gesture/tap_generalization_research.py "$B" "$B+obs60" "$B+obs40" "$B+softall" \
        "$B+distill50" "$B+distill25" "$B+fingerw2" "$B+obs40+distill50" "$B+obs40+distill50+fingerw2" \
        raw126_diff+augm+litediff+kb14+leaves7 raw126_diff+augm+litediff+press2+kb10+leaves7 \
        raw126_diff+augm+litediff+press2+kb8+leaves7 raw126_diff+augm+litediff+press2+kb14+leaves4 \
        raw126_diff+augm+litediff+press2+kb14+leaves15 raw126_diff+augm+litediff+press2+kb14+leaves31 \
        acc_emgrel+augm+litediff+press2+kb14+leaves7 raw126_diff+litediff+press2+kb14+leaves7
    # seeds: append +seed1 / +seed2 to $B, $B+obs60 and $B+obs40+distill50+fingerw2
    python research/gesture/tap_generalization_research.py focus_table "$B" "$B+obs60"   # per-type tables
    # maximum protection (32 KB): P=raw126_diff+augm+litediff+press2+kb32
    #   "$P+leaves7+obs60" "$P+leaves15+obs60" "$P+leaves15+obs40+distill50+fingerw2"
    #   "$P+leaves15+obs60+posw5" "$P+leaves15+obs60+posw1"   (positive weight 0.05 / 0.01)

Results: `results_tap_filesplit/<arm>.json` and `focus_<arm>.npz`.
