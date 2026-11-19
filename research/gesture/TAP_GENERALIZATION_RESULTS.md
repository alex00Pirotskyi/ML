# TAP generalization research (2026-11-19)

**Question:** which features, training methods and labels make the TAP model hold up when the band moves or
the electrode contact changes, without losing taps or adding false taps?

**Answer:** keep the 126 raw firmware fields, add **135 within-channel EMG diffs + 9 WL ratios**, and train with
**mild placement augmentation** (random EMG gain, spectral tilt and contact noise per recording). Keep today's labels.

Compared with today's no-Lean model, at the same false-tap rate this gives:
- +2.2 points TAP recall with nothing changed (95% person-bootstrap interval +0.7 to +3.8).
- +4.4 points under a simulated band shift (+2.6 to +6.1).
- False taps stay at about 1.7 per minute under the shift, where today's model rises to 2.5 per minute.

The model now takes 72–84% of its split gain from ACC, 14–25% from relative EMG (diffs, ratios), and only ~2% from
absolute EMG levels. Today's model takes 13–18% from absolute EMG levels.

## Method

- **Data.** The full TAP cache: 3,005 recordings with the 126 raw fields and Lean-8.
  - Development people only (44). The trainer's 22% person holdout was not touched.
  - 3 folds split by person.
  - Confirmed labels only (no soft rows).
  - Rows from frame 50 are trained on and scored. That's the "usable" view, which matches a device that runs the
    model from connect.
- **Model.** The same LightGBM configuration for every arm (31 leaves, depth 8, learning rate 0.1), cut to the
  32 KB firmware budget (88 trees). There is no Optuna in this research, so differences come from the approach,
  not the tuning.
- **Evaluation (identical for every arm).** The firmware replay (`replay.tap_replay_recording`) on every row of the
  validation people, using the original labels:
  - recall of confirmed TAP and SWIPE onsets, per folder;
  - false events per minute on trusted-negative rows, per activity.

  Arms are compared at exactly 2 and 5 false taps/min (threshold interpolated on the grid).
- **Stress tests.** These simulate the band moving. Each validation recording gets its own random draw, the same for
  every arm; ACC is unchanged:
  - **gain:** EMG amplitude ×2^±1 per channel (contact impedance). Exact on WL and on the PSD codes, which are
    10000·log10(1+power).
  - **tilt:** amplitude ×2^±0.5 per channel and per PSD zone (a different muscle or filter coupling).
  - **noise:** broadband noise power 0.1–1.0 per channel (loose contact). WL rises in quadrature, using
    WL ≈ 850·√power, which fits this data with log-log correlation 0.93.
  - **all three combined:** the "placement" condition in the tables.

  The deployment threshold is fixed on the nominal data and then kept under each stress.
- **Worst score.** recall × min(1, budget / false taps per minute), the trainer's objective, taken as the minimum
  over the four stresses. It penalizes a model whose false taps inflate under stress.
- **Does the stress test reproduce the device?** Yes. On the device, Lean features looked good on the bench but
  broke when the band moved or reconnected. In the usable view, Lean-8 is worse than raw 126 at 2 false/min:
  - −2.5 points in the first 10 s after connect;
  - −4.9 points under the placement stress (interval −8.3 to −2.0).

  The trainer's steady-only view (frame ≥ 500) hides this.

## Results

### At 2 false taps per minute

| Arm | TAP | SWIPE | TAP, gain | TAP, tilt | TAP, noise | TAP, all three | Worst TAP | False/min, all three | Worst score | EMG gain share |
|---|---|---|---|---|---|---|---|---|---|---|
| **raw 126 + EMG diffs + mild placement aug** | 0.890 | 0.855 | 0.880 | 0.879 | 0.873 | 0.852 | 0.852 | 1.94 | 0.852 | 0.21 |
| raw 126 + EMG diffs + placement aug | 0.857 | 0.814 | 0.845 | 0.853 | 0.870 | 0.851 | 0.845 | 1.85 | 0.845 | 0.20 |
| ACC + EMG diffs + mild placement aug | 0.884 | 0.854 | 0.873 | 0.871 | 0.871 | 0.842 | 0.842 | 1.79 | 0.842 | 0.19 |
| raw 126 + EMG diffs + gain aug | 0.896 | 0.860 | 0.883 | 0.884 | 0.846 | 0.820 | 0.820 | 1.87 | 0.820 | 0.20 |
| raw 126 + placement augmentation | 0.836 | 0.807 | 0.813 | 0.831 | 0.859 | 0.833 | 0.813 | 1.91 | 0.813 | 0.13 |
| raw 126 + EMG diffs | 0.886 | 0.857 | 0.856 | 0.869 | 0.844 | 0.811 | 0.811 | 2.01 | 0.806 | 0.27 |
| raw 126 + Lean-8 | 0.875 | 0.844 | 0.831 | 0.867 | 0.847 | 0.812 | 0.812 | 2.04 | 0.797 | 0.17 |
| raw 126 + EMG diffs + placement aug on half | 0.888 | 0.857 | 0.866 | 0.879 | 0.885 | 0.859 | 0.859 | 2.17 | 0.791 | 0.23 |
| ACC + EMG diffs + placement aug | 0.806 | 0.764 | 0.796 | 0.796 | 0.805 | 0.780 | 0.780 | 1.59 | 0.780 | 0.19 |
| ACC + EMG diffs | 0.881 | 0.849 | 0.862 | 0.855 | 0.827 | 0.771 | 0.771 | 1.61 | 0.771 | 0.27 |
| raw 126 + mild placement augmentation | 0.865 | 0.836 | 0.844 | 0.855 | 0.855 | 0.836 | 0.836 | 2.28 | 0.733 | 0.13 |
| raw 126 + gain augmentation | 0.866 | 0.835 | 0.843 | 0.861 | 0.847 | 0.824 | 0.824 | 2.36 | 0.700 | 0.13 |
| ACC + exact EMG log-ratios | 0.868 | 0.841 | 0.868 | 0.851 | 0.719 | 0.699 | 0.699 | 1.29 | 0.699 | 0.26 |
| labels: pure TAP | 0.836 | 0.801 | 0.784 | 0.822 | 0.808 | 0.768 | 0.768 | 2.33 | 0.659 | 0.17 |
| ACC + EMG code ratios | 0.870 | 0.847 | 0.869 | 0.844 | 0.776 | 0.736 | 0.736 | 2.31 | 0.638 | 0.20 |
| ACC only | 0.597 | 0.569 | 0.597 | 0.597 | 0.597 | 0.597 | 0.597 | 2.00 | 0.597 | 0.00 |
| raw 126 (today's no-Lean model) | 0.865 | 0.841 | 0.807 | 0.852 | 0.849 | 0.806 | 0.806 | 2.87 | 0.562 | 0.16 |
| labels: ACC teacher + re-anchor | 0.813 | 0.770 | 0.750 | 0.799 | 0.800 | 0.758 | 0.750 | 2.75 | 0.552 | 0.19 |
| labels: ACC teacher (keep/drop) | 0.860 | 0.837 | 0.811 | 0.849 | 0.850 | 0.813 | 0.811 | 2.99 | 0.543 | 0.13 |
| raw 126, EMG 8 bins | 0.870 | 0.848 | 0.829 | 0.863 | 0.849 | 0.819 | 0.819 | 3.09 | 0.530 | 0.14 |
| raw 126, EMG split penalty 0.3 | 0.859 | 0.835 | 0.821 | 0.852 | 0.855 | 0.828 | 0.821 | 3.18 | 0.521 | 0.03 |
| labels: ACC-evidence rule | 0.826 | 0.771 | 0.784 | 0.820 | 0.820 | 0.793 | 0.784 | 3.50 | 0.454 | 0.12 |

### At 5 false taps per minute

| Arm | TAP | SWIPE | TAP, all three | Worst TAP | False/min, all three | Worst score |
|---|---|---|---|---|---|---|
| ACC + EMG diffs + mild placement aug | 0.965 | 0.952 | 0.950 | 0.950 | 3.88 | 0.950 |
| **raw 126 + EMG diffs + mild placement aug** | 0.969 | 0.953 | 0.958 | 0.958 | 4.60 | 0.949 |
| raw 126 + EMG diffs + placement aug | 0.957 | 0.933 | 0.950 | 0.950 | 4.32 | 0.947 |
| ACC + EMG diffs | 0.969 | 0.955 | 0.939 | 0.939 | 4.61 | 0.939 |
| raw 126 + EMG diffs | 0.968 | 0.954 | 0.947 | 0.947 | 6.24 | 0.758 |
| raw 126 + Lean-8 | 0.961 | 0.952 | 0.947 | 0.947 | 7.13 | 0.664 |
| raw 126 (today's no-Lean model) | 0.966 | 0.957 | 0.953 | 0.952 | 9.62 | 0.496 |

At 5 false/min, today's model holds its recall under stress, but its false taps nearly double (9.6/min). The winner
stays under budget (4.6/min).

### Person-bootstrap intervals (44 people, 2,000 resamples, threshold on the grid at ≤ 2 false/min)

| Arm | TAP, nominal | vs today | TAP, placement stress | vs today | False/min, placement |
|---|---|---|---|---|---|
| raw 126 (today) | 0.852 (0.814–0.885) | — | 0.787 (0.750–0.824) | — | 2.50 (1.80–3.65) |
| **raw 126 + EMG diffs + mild aug** | 0.874 (0.839–0.902) | **+0.022 (+0.007, +0.038)** | 0.831 (0.792–0.866) | **+0.044 (+0.026, +0.061)** | 1.68 (1.23–2.48) |
| ACC + EMG diffs + mild aug | 0.874 (0.843–0.899) | +0.022 (+0.006, +0.045) | 0.828 (0.787–0.861) | +0.042 (+0.020, +0.060) | 1.64 (1.21–2.40) |
| raw 126 + EMG diffs | 0.861 (0.821–0.893) | +0.010 (−0.005, +0.021) | 0.773 (0.732–0.817) | −0.014 (−0.027, +0.003) | 1.62 (1.03–2.69) |
| raw 126 + Lean-8 | 0.825 (0.767–0.864) | −0.027 (−0.056, −0.004) | 0.738 (0.681–0.783) | −0.049 (−0.083, −0.020) | 1.36 (1.04–1.91) |

## What moved the result and what didn't

- **EMG is necessary.** ACC alone loses 27 points of TAP recall at 2 false/min. EMG is what rejects ACC impulses that
  aren't taps (arm, wrist, band). The goal is to use EMG robustly, not to drop it.
- **Absolute EMG levels are the fragile part.** Under the placement stress, today's model's false taps rise mostly
  on arm movement (1.3 → 4.6/min), wrist (1.0 → 5.3) and band handling (1.1 → 3.6). With the winner they stay
  flat: arm 0.6 → 0.6, wrist 0.9 → 1.0, band 0.9 → 1.1.
- **Within-channel EMG diffs** (zone vs zone, and now vs history vs mean) raise nominal recall (+2 points). With
  augmentation they also stay robust. Without augmentation they cancel gain but not tilt or noise.
  - The firmware's `diff()` on the PSD codes is the right helper.
  - Ratios of the codes, and even exact log-power ratios, are fragile to contact noise.
- **Augmentation decides robustness.** It has to cover all three effects (gain, tilt, noise), and it should be mild:
  - Gain-only augmentation leaves noise fragility.
  - Strong augmentation on every recording costs ~3 points nominal.
  - Strong augmentation on half the recordings is about as good as mild augmentation everywhere, but false taps rise
    a little.
- **EMG penalty and coarse EMG bins don't help.** They push weight away from EMG, but false taps under stress get
  worse (3.1–3.2/min).
- **Labels: keep today's.** Every relabeling was worse:
  - Pure TAP (ignore swipe presses): −3 points TAP, −4 points SWIPE. Swipe positives help the tap model.
  - ACC-evidence rule: keeps 70–76% of taps and 60–63% of swipes; −4 points TAP, and the most false taps under
    stress.
  - ACC-only teacher, cross-fitted by person: keeps 91–93% of taps and 86–93% of swipes. So a learned ACC detector
    does find a tap-like ACC onset in most swipes; the crude energy rule underestimated this. But keep/drop gave no
    gain (0.860 vs 0.865), and re-anchoring the window on the ACC peak lost 5 points. The current labeler's onset is
    the better target.
- **Lean-8** is worse in the first ~10 s after connect and under the placement stress. This confirms the device
  finding.

## Recommendation and what it needs

1. **Trainer option (built).** Run:

   ```bash
   MUDRA_USE_LEAN8=0 MUDRA_TAP_FEATURES=diff MUDRA_TAP_AUG=mild MUDRA_TASKS=tap MUDRA_EVAL_MIN_FRAMES=50 \
   MUDRA_TAP_BASELINE_EXTRA=research/gesture/results_unified_v3_feature_selection_no_lean8/tap/models/TAP_SWIPE.h \
       python -u research/gesture/train_unified.py
   ```

   - **Derived fields:** the 144 fields are computed from the 126 raw columns and stored in the cache; no new data files.
   - **Augmentation:** the training rows get the mild placement draw (`tap_features.py`, identical to the research).
   - **Header calls:** the exporter writes `p_core->diff(a, b)` / `p_core->ratio(a, b)`. The host replay provides
     both helpers and is checked against Python, including int16 saturation.
   - **Stress in COMPARISON.md:** it replays every candidate and baseline under the simulated band move on the
     held-out people.
   - **Scoring from frame 50:** `MUDRA_EVAL_MIN_FRAMES=50` scores Optuna on rows from frame 50. Without Lean inputs
     there is no 10 s warm-up, and this uses ~7× more confirmed onsets than the steady gate.
   - **Same-table A/B:** `MUDRA_TAP_BASELINE_EXTRA` replays today's no-Lean header on the same held-out people,
     same rules, same stress.
2. **Firmware, nothing new:**
   - The trees call `diff()` and `ratio()` through `t_core`. Each needed value is computed once per prediction:
     up to ~110 helper calls, roughly 2–3k cycles, next to ~50–60k for the tree walk.
   - Please confirm the helper definitions match the research: `diff(a, b) = a − b` saturated to int16, and
     `ratio(a, b) = (a + 1) / (b + 1)` in float.
3. **Device test:**
   - The same tapping, plus 2–3 minutes of movement only, in 3–4 placements (shifted, rotated, loose, tight) and
     after a reconnect.
   - Run it on today's no-Lean model and on the new model.

## Caveats

- **The stress is a simulation.** Gain, tilt and noise are the main effects of a band shift, but not all of them
  (e.g. a large rotation moves electrodes to other muscles and rotates the ACC axes). The device test decides.
- **Same family as the augmentation.** The augmentation and the stress test use the same three effects, with
  different random draws; the training draw is milder (tilt ±0.25 octave, noise 0.03–0.3) than the test draw.
- **Fixed model configuration.** Optuna in the trainer will change the absolute numbers; the ranking is what this
  research measures.
- **Small differences are noise.** Near the top of the table, differences under ~1.5 points are within the
  person-bootstrap noise.

## Reproduce

```bash
python research/gesture/tap_generalization_research.py raw126 "raw126_diff+augm" ...   # arms (see ARMS, '+' combines)
python research/gesture/tap_generalization_research.py table                         # interpolated comparison
python research/gesture/tap_generalization_research.py rescore raw126 "raw126_diff+augm" lean8   # bootstrap
```

Uses the full TAP cache (`MUDRA_GR_CACHE`, built by `train_unified.build_task_data` with Lean-8 on) and writes to
`results_tap_generalization/`. One arm takes about 4 minutes on 2 CPUs.
