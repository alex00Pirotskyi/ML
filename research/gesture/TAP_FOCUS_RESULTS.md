# TAP focus research: double taps, fast taps, finger protection (2026-11-20)

**Goal.** Make double taps and fast taps (240 bpm) stronger in the model itself, so the firmware tap rule
(`firmware/tap_gestures.h`) has less to do. Cut false taps from finger motion and fist, and keep soft taps at
least as good as they are.

**Recommendation.** Take `litediff + press2` (12–15 Lean-lite diff features plus press-anchored labels for
DoubleTaps and 240 bpm) into the trainer and test it on the device.

| Goal | Result | Verdict |
|---|---|---|
| Double and fast taps | second taps of doubles +12.9 points, fast taps +7.3, at the same false rate. The model alone now catches about as many fast taps as the old model with the firmware rule | Met |
| Soft taps | +4.1 points | Met |
| Swipes and normal taps | +2.0 and +1.6 points | Kept, slightly up |
| Finger / fist protection | not improved; this needs more people in the finger and fist recordings | Not met |

`+accdiff` (81 ACC zone and slot differences, `p_core->diff` only) is the alternative if normal taps and
swipes matter more than double taps. It gives about +3.5 points on both, at a smaller double-tap gain.

## Method

**Harness:** `tap_generalization_research.py`. Same data, folds and model settings as `TAP_GENERALIZATION_RESULTS.md`:
- 44 development people, 3 person folds.
- Research models with 88 trees, model size capped at 32 KB.
- Rows from EMG frame 50.

**What the report counts:** the new `focus` report counts every real pointer press, not only labeller-confirmed
ones, per gesture type:
- normal taps (60/120 bpm Mid/Strong);
- soft taps (SoftTap recordings);
- the first and second tap of a double (second = less than 500 ms after the previous press);
- 240 bpm fast taps;
- swipes.

A press is clicked when an event fires within −100..+250 ms of it. False events per minute are counted per
no-gesture recording type: finger motion, fist (Fist&Stretch), wrist, arm, band. Unmatched events on
trusted-negative rows of tap recordings are counted as well.

**How arms are compared:**
- **Same false rate:** every arm runs at its own threshold for **2.8 false events/min overall** (the deployed model's rate).
- **Placement:** the same threshold under placement stress (gain ±1 octave, tilt ±0.5 octave, noise).
- **Warm-up:** the same threshold on rows before frame 500, the first ~10 s after a (re)connect.
- **Uncertainty:** 95% intervals from a paired person bootstrap.
- **Scale:** absolute recalls are on the research models' scale, so compare arms with each other, not with device numbers.

## Arms

| Token | What it does |
|---|---|
| `press2` | In DoubleTaps and 240 bpm recordings every real press is a tap. The labeller's soft rows of those presses are trained on with full weight. Presses with no onset at all get positives at [press − 5, press + 90] ms. Negatives within −30..+120 ms of a press are dropped (there were none). Per fold this adds about 2,000–2,600 rows and about 60 presses |
| `fingerw3` | Finger-motion and fist negatives weighted ×3 |
| `fmine` | Hard negatives: finger and fist rows that the baseline fold model still scores ≥ 0.3, mined from the training people only, one every 20 ms |
| `movew3` | All movement negatives (finger, fist, wrist, arm, band) weighted ×3 |
| `pw3` | Positives of DoubleTaps and 240 bpm recordings weighted ×3 |
| `litediff` | Lean-lite context features in the log domain only. x = log2(WL+1), per channel: `rise10` = x − min10, `fall10` = max10 − x, `range10` = max10 − min10, `dev3` (mean absolute deviation), `xm` = x − EMA mean. A contact gain adds a constant to x, so none of them change with gain. The absolute `min10`, `max10` and `d01` (the parts that did not generalise on the device) are **not** used. The values are recomputed from the transformed WL under augmentation and stress. They match `lean_lite.h` exactly (lo / hi / rise / dev) |
| `accdiff` | ACC: the current spectrum's zone-vs-zone differences per axis, plus current vs oldest history and current vs mean per axis and zone (81 `p_core->diff`) |

## Results

### Every press, at the same false rate (2.8 events/min), nominal

Differences to the current recipe (`raw126_diff+augm`), in points of recall or false events/min.
`*` marks a 95% interval that excludes 0.

| Arm | Normal | Soft | Double 1st | Double 2nd | Fast 240 | Swipe | Finger /min | Fist /min |
|---|---|---|---|---|---|---|---|---|
| Current, absolute | 78.8% | 69.9% | 83.8% | 49.8% | 51.2% | 66.8% | 1.50 | 3.13 |
| press2 | −0.4 | −1.7 | +1.3 | **+8.3\*** | +0.3 | −0.6 | +0.16 | +0.11 |
| fingerw3 | −0.9 | +0.4 | −0.2 | +0.7 | 0.0 | +0.6 | −0.31 | −0.59 |
| fmine | +0.1 | +0.1 | +0.1 | +3.4\* | +1.1 | +0.8 | −0.11 | −0.03 |
| **litediff** | **+2.1\*** | **+5.2\*** | +0.9 | **+6.1\*** | **+5.8\*** | **+2.3\*** | +0.27 | −0.09 |
| litediff + accdiff | +3.7\* | +4.9\* | +4.1\* | +1.6 | +3.2 | +3.7\* | −0.14 | +0.41 |
| **litediff + press2** | +1.6 | **+4.1\*** | +2.8\* | **+12.9\*** | **+7.3\*** | +2.0 | +0.26 | −0.38 |
| litediff + press2 + movew3 | +1.4 | +4.7\* | +3.1\* | +13.2\* | +9.6\* | +2.2\* | +0.18 | +0.08 |
| litediff + press2 + pw3 | +1.1 | +2.5\* | +2.7 | +18.7\* | +9.0\* | +1.2 | +0.47 | +0.15 |
| litediff + press2 + accdiff | +3.5\* | +4.2\* | +3.9\* | +9.0\* | +6.9\* | +3.6\* | −0.05 | +0.60 |

**Robustness:**
- **Placement stress:** `litediff + press2` keeps its gains: second taps +11.5\*, normal +1.3, swipe +1.8, overall false rate −0.10.
- **Warm-up (first ~10 s):** second taps +14.8\*, fast +7.3\*, soft +3.7\*. Unlike Lean-8, the diff features need no 10 s gate.
- **Older harness metric:** confirmed onsets at 2 events/min, worst stress. `litediff` matches the current recipe on taps (0.890), and its false rate under placement is lower (1.68 vs 1.94).

**Finger and fist weighting** moves false taps around instead of removing them. At the same overall rate,
`fingerw3` has −0.31 finger and −0.59 fist, but +0.40\* wrist, +0.28 arm and +0.30 band. `movew3` does
not change movement false taps either.

### At the same normal-tap recall (spending the gain on fewer false taps)

| Arm | Soft | Double 2nd | Fast 240 | Finger /min | Fist /min | All /min |
|---|---|---|---|---|---|---|
| litediff | +3.3\* | +2.2 | +3.1 | +0.10 | −0.68\* | −0.15 |
| litediff + press2 | +2.4 | +9.7\* | +5.5\* | +0.08 | −0.62 | −0.10 |
| litediff + press2 + accdiff | +0.9 | +3.0 | +0.9 | −0.39 | −0.17 | −0.25\* |

### How much the firmware rule still adds

Replay with `tap_gestures.h` (immediate, chain, next tap ≥ 0.60 after a dip below 0.20). The thresholds are
mapped to each model's scale; each model runs at its 2.8 events/min threshold.

| Model | Rule | 2nd tap of double clicked | 240 bpm taps clicked | Single tap + extra click | Movement clicks/min |
|---|---|---|---|---|---|
| Current | model only | 50.4% | 52.3% | 0.93% | 1.07 |
| Current | + rule | 72.1% | 61.1% | 1.36% | 1.15 |
| litediff + press2 | model only | 63.9% | 58.9% | 1.37% | 1.24 |
| litediff + press2 | + rule | **81.0%** | **69.7%** | 1.51% | 1.35 |
| litediff + press2 + accdiff | + rule | 79.5% | 68.2% | 1.76% | 1.24 |

- **Model only:** the new model catches about as many fast taps as the current model with the rule. It catches more than half of the second taps the rule used to add.
- **With the rule kept:** it is cheap, and with it the second taps and fast taps go up another 9 points each.

### Why finger protection did not move

- **Who records finger and fist:** only 12 of the 44 development people have finger or fist recordings.
- **Where the false taps come from:** two people produce 60% of the current model's finger and fist false taps (42 and 19 of about 104 events; 5.8 and 6.1 per minute against 0–3.4 for the others).
- **Most-triggering activity:** Fist&Stretch triggers most (3.2/min); single-finger motion triggers 0.4–1.3/min.

Weighting and mining teach the model those people's patterns, but a new person moves differently, so person-fold
validation shows no transfer. Finger and fist motion needs recordings from more people, and more sessions from
the people who trigger most. The person-fold set-up can measure the effect of new data directly.

## Would a 10–20 ms tap delay help finger protection? (2026-11-20)

The idea: a tap starts with finger motion and only then makes contact, so waiting 10–20 ms could let the
model see the contact. Two ways of spending the delay were tested, both compared at the same normal-tap recall:

1. **Firmware only** (`focus_hold`): an event fires only once the score has stayed above the threshold for 10 or 20 ms.
2. **Shifted labels** (`shift10`, `shift20`): the positive windows move 10 or 20 ms later, so the model learns to decide later.

| Model | Rule | Median latency | Finger /min | Fist /min | All /min | Soft | Double 2nd | Fast 240 |
|---|---|---|---|---|---|---|---|---|
| Current | as now | +2.5 ms | 1.50 | 3.13 | 2.80 | 69.9% | 49.8% | 51.2% |
| Current | hold 10 ms | +17.5 | 1.33 | 3.21 | 2.73\* | 70.2% | 51.7%\* | 49.4% |
| Current | hold 20 ms | +22.5 | 1.30\* | 2.68\* | 2.67\* | 70.1% | 50.0% | 47.3%\* |
| litediff + press2 | as now | +2.5 | 1.58 | 2.61 | 2.70 | 72.3% | 59.4% | 56.7% |
| litediff + press2 | hold 20 ms | +22.5 | 1.52 | 2.48 | 2.64 | 72.5% | 61.7% | 56.5% |
| litediff + press2 + shift10 | as now | +7.5 | 1.65 | 3.82 | 2.71 | 71.4% | 59.5% | 55.0% |
| litediff + press2 + shift20 | as now | +17.5 | 1.37 | 3.05 | 2.64 | 67.8% | 55.9% | 56.5% |

In the current-model rows, `*` marks a 95% person-bootstrap interval of the difference to the model as it is now that excludes 0.

**What this means:**
- **Hold time:** 20 ms removes about 15% of the finger and fist false taps of the current model, for about +20 ms latency and about −4 points of fast taps. On the new model the effect is smaller (−0.06 overall).
- **Shifted labels:** teaching the model to decide later does not help, and costs soft taps and second taps.
- **Why:** the model already fires at the contact: its median is +2.5 ms after the press is registered. The false taps
  that remain are short at the deployed threshold (fist: median 3 ms above the threshold, 92% under 20 ms; finger:
  median 12 ms), but 31% of real taps are that short too. Making a fist or bending a finger produces a short
  contact-like event, not an early part of a tap.
- **On the device:** a hold rule is cheap to try (a `TG_HOLD_MS` in `tap_gestures.h`). The device's Q20 scores and 4.4 ms model rate differ from replay, so a device test with finger and fist motion would show whether the blips there are shorter than in replay.

## Remove the two people who trigger most? (2026-11-22)

Two people produce 60% of the current model's finger false taps (recorded as `alex` and `hadas`).

**Data check:**

| Person, session | Taps clicked (own recordings, held out) | EMG rise during a tap | Finger /min | Fist /min | Wrist / arm / band /min |
|---|---|---|---|---|---|
| hadas, 2025-12-29 | 95% (swipes 94%) | 1.99× | 3.8 | 13.3 | 2.8 / 0 / – |
| alex, 2025-12-28 | 24% | **1.52×** | 1.7 | 8.4 | – |
| alex, 2026-02-02 | (no taps recorded) | – | 7.6 | – | 0.4 / 0 / 0 |
| Everyone else | median 82% (per person) | median 2.84× over 68 sessions (p10 1.92×) | | | |

EMG rise = WL peak within 150 ms of a press divided by the resting WL. The alex session of 2025-12-28 has the
weakest tap response of all 71 sessions, and its taps are rarely detected, which points to a contact or setup
problem in that session. The other two sessions look normal: their taps are detected well, or other movements
in the same session stay quiet.

**Retraining without them** (`litediff + press2 + dropalex`, `+ drophadas`), compared on the other 42 people
at the same false rate:

| Trained without | Normal | Soft | Fast 240 | Swipe | Finger /min | Fist /min |
|---|---|---|---|---|---|---|
| alex | −1.1\* | −1.2 | −2.6\* | −0.6 | −0.19 | +0.39 |
| alex and hadas | −0.3 | −0.9 | −3.2\* | −1.1\* | −0.22 | **+1.22\*** |

Under placement stress, dropping alex costs −3.5\* soft, −4.9\* fast and −2.0\* swipes.

**What this means:** removing them makes the model worse for everyone else. Their finger and fist recordings
are the hard examples the model learns protection from, and fist protection for others gets worse without
hadas. Keep them, in training and in the evaluation.

## New finger recordings: omer_gilinsky (2026-11-22)

**The data:** 18 finger and fist recordings (Fist&Stretch, index, middle, ring, pinky, thumb × three
orientations), about 12 s each and 3.6 min in total. The existing finger data is 339 recordings, 14 people
and 85 min.

**How they were added:**
- Labelled with `label_gestures.py` (no gesture events).
- Cached with `tap_extra_cache.py` (Lean-8 derived copies, then `build_task_data` on the 18 files only).
- Added to the research data with `MUDRA_GR_EXTRA` / `MUDRA_GR_EXTRA_MODE`. The person split and folds stay
  unchanged, and the new recordings get their own stress draws.

**A person the models never saw** (all three fold models, threshold for 2.8 events/min):

| Model | Finger /min | Fist /min | Under placement: finger / fist |
|---|---|---|---|
| Current | 0.00 | 0.84 (1 event) | 0.00 / 1.68 |
| litediff + press2 | 0.00 | 0.00 | 0.17 / 0.84 |

The held-out people's finger rates range from 0 to 5.8 per minute. This person is among the easy ones.

**Trained with them** (`+extra`, in every training fold), compared on the 44 development people: finger
−0.13 [−0.43, +0.15], fist +0.22 [−1.15, +1.26], other recalls within ±1 point. Only fast taps moved
outside the interval (−1.9 [−4.2, −0.0]), and only just. So there is no measurable change: 4% more finger
data from a person who rarely triggers adds little to learn from.

The `Tap_on_Hand` and `Tap_on_motion` recordings are not in the dataset yet, because `org.py` has no mapping for
them (`DATA_UNMATCHED`).

## Tighter labels: EMG confirmation and a shorter window (2026-11-22)

**The EMG rise of a tap:** the largest WL of any channel in [onset − 20, onset + 120] ms, divided by the
resting floor. The floor is the lower of the recording's and the session's 5th-percentile WL, so 240 bpm
recordings, which never rest, are covered (`emg_rise_table`). Taps that fail the gate are left out of
training; they are never made negatives.

| Presses with EMG rise below | 1.2× | 1.5× | 2.0× |
|---|---|---|---|
| Normal taps | 0.0% | 0.4% | 2.2% |
| Soft taps | 0.3% | 4.4% | 16% |
| Double tap, 2nd | 0.0% | 0.1% | 0.4% |
| Fast 240 bpm | 0.9% | 4.3% | 11% |
| Swipes | 0.2% | 1.6% | 5.5% |

Nearly every labelled tap already shows a clear EMG rise. The labeller's teacher already reads EMG and WL, so
the confirmation is largely there.

Compared with `litediff + press2` at the same false rate:

| Arm | Taps left out | Normal | Soft | Double 2nd | Fast 240 | Finger /min | Fist /min |
|---|---|---|---|---|---|---|---|
| `emg1.2` | 1–4 per fold | −0.3 | −0.7 | −2.3 | +0.2 | −0.04 | +0.27 |
| `emg1.5` | 23–89 per fold (~1%) | +0.4 | +0.1 | +1.8 | −1.1 | +0.12 | +0.14 |
| `win40` (positives [−5, 40] ms) | 0, about 42,000 later rows | −1.6\* | −0.3 | **−9.7\*** | **−6.9\*** | −0.19 | +0.96 |

**What this means:**
- **EMG confirmation:** harmless, but it changes nothing measurable, including finger and fist protection.
- **Shorter window:** clearly worse, mostly for second and fast taps.
- **Retraining noise:** `emg1.2` removes almost nothing from training, yet under placement it still shows
  −3.1 on second taps. Differences of about 2–3 points between single retrains are noise.

## 60 ms window, EMG confirmation, weak-recording gate, and training noise (2026-11-23)

**Question:** not every recording is good (about 85% are), so do slightly tighter labels make the model more robust?

**Variants tested:**
- a 60 ms window (positives [−5, 60] ms instead of 90);
- EMG confirmation at 1.2× and 1.5×;
- a weak-recording gate (`recq50`): TAP recordings, other than DoubleTaps and 240 bpm, where the labeller
  confirmed fewer than half the presses give no positives. That is 73 recordings and about 1% of the positive rows.

**Training noise was measured two ways:**
- **Seeds:** the reference was retrained with two other LightGBM seeds. Spread across the 3 runs: normal 0.2,
  soft 1.4, second taps 0.5, fast 1.4 points (placement: up to 2.3).
- **Near-identical data:** `emg1.2` removes only 1–4 taps per fold, yet it moves second taps by −2.3 (placement
  −2.6) and soft taps by −1.5. Removing a few rows changes the bagging draws, which is a larger perturbation
  than a seed.

So differences up to about ±3 points on second taps and ±2 on the other types are noise.

**Differences to the reference** (mean of the 3 seeds), at 2.8 events/min. Each cell is nominal / under placement stress:

| Variant | Normal | Soft | Double 2nd | Fast 240 | Swipe | Finger /min | Fist /min |
|---|---|---|---|---|---|---|---|
| emg1.2 | −0.4 / −0.5 | −1.5 / −1.7 | −2.3 / −2.6 | +0.9 / −1.2 | +0.2 / −0.1 | −0.09 / −0.11 | +0.04 / −0.38 |
| emg1.5 | +0.3 / +0.6 | −0.6 / +0.9 | +1.8 / +2.7 | −0.6 / −0.4 | +0.3 / +1.0 | +0.05 / +0.03 | −0.08 / −0.39 |
| win60 | −0.6 / +1.3 | −0.9 / +2.3 | −3.8 / −3.7 | −1.1 / +0.9 | −1.2 / +1.5 | −0.20 / +0.08 | +0.48 / +0.11 |
| win60 + emg1.2 | +0.6 / +1.5 | +0.8 / +2.9 | −2.2 / −2.3 | +0.4 / +3.5 | +0.7 / +2.1 | −0.16 / +0.02 | +0.77 / +0.24 |
| win60 + emg1.5 | −0.9 / +0.3 | −0.5 / +2.1 | −4.1 / −3.5 | −3.2 / −1.1 | −0.6 / +1.2 | −0.13 / −0.01 | +0.36 / +0.34 |
| emg1.2 + recq50 | −0.4 / −0.3 | −1.1 / −0.8 | −1.1 / −0.3 | +0.2 / +0.0 | −0.5 / +0.1 | +0.05 / −0.04 | +0.78 / −0.41 |
| win40 | −1.6 / −1.3 | −1.1 / −0.4 | **−9.6 / −11.3** | **−6.2 / −2.6** | −1.9 / −0.4 | −0.21 / −0.15 | +0.89 / +0.26 |

**What this means:**
- **EMG confirmation and the weak-recording gate:** within noise. The labeller already protects against weak
  recordings: in recordings whose taps raise the EMG less than 2×, it confirms only 70–75% of the presses,
  against 91% elsewhere.
- **60 ms window:** all three 60 ms runs cost second taps (−2 to −4). Under placement, the variants with
  EMG 1.2 or no EMG gate gained soft, normal and swipes (+1.5 to +3, at the edge of noise); with EMG 1.5
  those gains were smaller. For the double-tap goal it is not worth it.
- **40 ms window:** clearly worse.
- **Recommendation:** keep the 80 ms window. EMG ≥ 1.2× can stay as a safeguard (it removes the rare taps with
  no EMG at all), but don't expect it to move the numbers.

## 14 KB model budget and overfitting (2026-11-23)

The device allows 14 KB for the model. The trainer defaults to 32 KB (alternative 24 KB). The last exported
TAP_SWIPE.h is 32.7 KB with 130 trees.

**Model size and tree shape**, same data and labels, at 2.8 events/min (nominal; the placement row is in the
focus files):

| Model | Trees | Normal | Soft | Double 2nd | Fast 240 | Swipe | Finger /min | Fist /min |
|---|---|---|---|---|---|---|---|---|
| litediff + press2, 32 KB, 31 leaves | 88 | 80.4 | 74.0 | 62.5 | 58.5 | 68.8 | 1.73 | 2.87 |
| litediff + press2, 14 KB, 31 leaves | 38 | 74.9 | 58.2 | 55.0 | 44.5 | 57.7 | 2.22 | 4.46 |
| litediff + press2, 14 KB, 15 leaves | 79 | 76.0 | 69.6 | 54.3 | 51.5 | 64.5 | 1.84 | 3.30 |
| **litediff + press2, 14 KB, 7 leaves** | 172 | 78.4 | 71.3 | 59.4 | 55.6 | 68.1 | 1.87 | **2.33** |
| litediff + press2, 14 KB, 4 leaves | 308 | 78.5 | 69.3 | 59.4 | 52.5 | 67.5 | 1.79 | 2.15 |
| Current recipe, 14 KB, 31 leaves | 38 | 74.9 | 55.6 | 52.0 | 41.2 | 59.5 | 1.94 | 6.94 |
| Current recipe, 14 KB, 7 leaves | 172 | 77.2 | 68.5 | 47.2 | 47.8 | 66.4 | 1.67 | 2.56 |

**What this means:**
- **Tree shape:** at 14 KB, small trees are essential. With 31 leaves only 38 trees fit and the model collapses
  (soft −16 points, fist false taps ×1.5). With 7 leaves the 14 KB model is within 1–3 points of the 32 KB one,
  and its fist protection is better.
- **Recipe at 14 KB / 7 leaves:** the new recipe against the current one gives second taps +12.2\*, fast
  +7.8\*, first taps of doubles +4.8\*, soft +2.7\*, normal +1.1, swipes +1.7, with finger and fist unchanged.
  The label and feature recommendation holds at 14 KB.

**Overfitting: people seen in training vs held-out people,** at the same threshold
(`overfit_gap.json`, a random half of each fold's training people):

| Model | Normal | Fast 240 | Finger /min | Fist /min |
|---|---|---|---|---|
| 32 KB, 31 leaves | 86.8 / 80.4 | 80.4 / 58.5 | 0.23 / 1.73 (7.5×) | 0.57 / 2.88 (5.1×) |
| 14 KB, 31 leaves | 82.2 / 74.9 | 61.8 / 44.5 | 0.77 / 2.22 (2.9×) | 1.81 / 4.46 (2.5×) |
| 14 KB, 7 leaves | 84.2 / 78.4 | 73.8 / 55.6 | 0.49 / 1.87 (3.8×) | 0.88 / 2.33 (2.6×) |

**What this means:**
- **Selection is not overfitting:** the trainer's held-out results match its cross-validation.
- **People are memorized:** for people it was trained on, the 32 KB model has 7× fewer finger false taps and 5× fewer fist false taps than for new people.
- **Small trees help:** they shrink this gap and generalise better per byte. That fits the finger/fist finding: protection learned per person does not carry over to new people.
- **Recommendation for the trainer:** 14 KB budget, and Optuna `num_leaves` limited to about 4–15 (7 as the reference).

## Final recipe (2026-11-23): research + GPT review, in the trainer

One line trains it (TAP only; LP is not touched):

    MUDRA_TAP_RECIPE=final14 python -u research/gesture/train_unified.py

The preset sets these defaults (any `MUDRA_*` set explicitly still wins):

| Setting | Value | Why |
|---|---|---|
| `MUDRA_USE_LEAN8` | `0` | no Lean-8 float fields |
| `MUDRA_TAP_FEATURES` / `MUDRA_TAP_AUG` | `diff` / `mild` | PSD diffs + WL ratios, placement augmentation (as before) |
| `MUDRA_TAP_LITE` | `diff` | 15 Lean-lite diff inputs: `rise10`, `fall10`, `range10`, `dev3`, `xm` per channel |
| `MUDRA_TAP_LABELS` | `press2` | trust the press in DoubleTaps / 240 bpm TAP recordings |
| `MUDRA_MODEL_BUDGET_KB` / `MUDRA_ALT_BUDGET_KB` | `14` / `0` | the device limit; no second header |
| `MUDRA_NUM_LEAVES` | `4,15` | small trees fit 14 KB (7 leaves = 172 trees) and memorize people less |
| `MUDRA_TRAIN_MIN_FRAMES` / `MUDRA_EVAL_MIN_FRAMES` | `50` / `50` | the model runs from 1.1 s, as on the device |
| `MUDRA_TASKS` | `tap` | |

Results go to `results_unified_v3_feature_selection_no_lean8_tap-diff-mild-litediff-press2-kb14-leaves4-15/tap/`.

**How each piece is implemented** (all bit-exact with the research arms):
- **Lean-lite diff inputs:** computed per recording from the logged WL and the frame counter, from the
  recording's first row (the firmware state after a reset). No new data files.
  - Under augmentation and under the placement stress report they are recomputed from the changed WL of the
    whole recording, because they depend on ~10 s of history.
  - The emulator equals `firmware/lean_lite.h` compiled for the host on real recordings and edge cases, and
    equals the research's values on every frame.
- **press2 labels:** training labels only; every replay reference and score uses the original labels. Per
  press of those recordings:
  - **Confirmed:** unchanged.
  - **Unknown with a teacher peak** (score ≥ 0.5): its unknown rows in the 0–80 ms onset window become
    positives.
  - **Other unknown presses:** rows from −5 to +90 ms around the onset estimate become positives.
  - **Negatives** from −30 to +120 ms around any press are left out.

**Tested in this round and left out:**
- **GPT's E2** (drop every provisional negative inside TAP/SWIPE recordings, `+notapneg`, 14 KB / 7 leaves) was
  no better. At 2.8 false/min: normal −0.7, soft −1.3, fast −2.8, second taps +0.5, finger −0.03/min, fist
  +0.56/min, median latency +5 ms. With placement stress, second taps were −2.9.
- **Dropping `fall10`** (12 inputs) gave taps +1 to +2 points but fist +0.4 to +0.8/min with placement stress
  (one training run; the seed spread is ±1.4 points). Fist protection has priority, and `fall10` costs one
  subtraction, so the recipe keeps the researched 15-input set.

**Correctness fixes from the GPT review** (they apply to every configuration):

| Issue | Fix |
|---|---|
| A completed run was reused without checking its data | A content hash of every recording, label file and event table is stored with the run. A rerun on changed data refuses and names the change. |
| The cache fingerprint used path, size and mtime | It uses content hashes. The gzip time and stored name are ignored, so regenerated identical labels still match. |
| 137 byte-identical recordings counted twice | Dropped before the split, first path kept, logged. 6 of them are copies under another participant name. |
| `recall_press` / `recall_strong` reference windows were not one-to-one (one tap could find two presses) | Maximum one-to-one matching. The objective's onset matching was already one-to-one. |
| `org.py`: the alias `index` matched participant names | Replaced by `index_finger`. All 5,370 current files map as before; covered by regression tests. |

**Firmware for this model:**
- **Lean-lite fields:** `lean_lite.h` and the private `core_type_def.h` gain `range10`, `fall10` and `xm`.
  These are appended fields, so the offsets of older fields are unchanged.
- **Cost:** 48 bytes of model inputs, the same 164 bytes of state, 192 Cortex-M4 instructions per EMG frame.
  There is no float and the only calls are CMSIS min/max.
- **Build:** `NEW_CLOAD_FEATURES 1`. Call `lean_lite_features_handler()` once per EMG frame and reset it at
  stream start.
- **Warm-up:** run the TAP model after the 1.1 s warm-up. Do not wait for `lean_lite_features_ready()`.
- **Tap rule:** unchanged (chain mode for fast taps).

**GPT items not done here (do not affect what is trained):**
- **Not done:**
  - the frozen manifest and contract system
  - an independent-truth validator format
  - continuous replay for frame-500 reports
  - the LP validator denominator fix
  - a development-only E0–E3 runner
  - E3 (LP rows)
  - V9 coverage
- **E1** (keep the labeller's 22 double-tap onsets) is covered more broadly by press2. The labeller is
  unchanged.

## Reproduce

    python research/gesture/tap_generalization_research.py focus "raw126_diff+augm"
    python research/gesture/tap_generalization_research.py "raw126_diff+augm+press2" "raw126_diff+augm+litediff" \
        "raw126_diff+augm+litediff+press2" "raw126_diff+augm+litediff+press2+accdiff"   # ... any arm above
    python research/gesture/tap_generalization_research.py focus_table "raw126_diff+augm" "raw126_diff+augm+litediff+press2"
    python research/gesture/tap_generalization_research.py focus_match "raw126_diff+augm" "raw126_diff+augm+litediff+press2"
    python research/gesture/tap_focus_fw_rule.py "raw126_diff+augm" "raw126_diff+augm+litediff+press2"

The results are in `results_tap_generalization/`:
- `focus_<arm>.json`: summary at 2.0 and 2.8 events/min.
- `focus_<arm>.npz`: per-person counts on the threshold grid.
- `fw_rule_eval.json`: the firmware-rule comparison.

The Lean-lite values of every replay row are cached next to the TAP cache (`lite_diff_v1.npy`).
