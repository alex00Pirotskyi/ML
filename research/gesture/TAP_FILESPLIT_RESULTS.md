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

Results: `results_tap_filesplit/<arm>.json` and `focus_<arm>.npz`.
