# TAP: labels and inputs from the observable pattern (2026-11-23)

**Question:** don't teach the embedded model "a tap happened". Teach it the smallest observable pattern that tells a
tap apart from everything else, keep soft, double and fast taps, and find the smallest model with the best
performance on people it has never seen.

**Setup:**
- **Data:** 44 development people, 3 person folds; the trainer's held-out people are never used.
- **Reference:** the final recipe (`raw126_diff+augm+litediff+press2`, 14 KB, 7 leaves).
- **Metrics:** taps per type (every real pointer press) and false taps per minute per look-alike type, compared at
  the same overall false rate (2.8 and 2.0 per minute), nominal and under a simulated band move (placement).
- **Noise:** a plain retrain (`win80`, which drops 0.07% of rows) moves results by about ±1 point, ±2 on second
  taps. `*` marks a 95% person-bootstrap interval that excludes 0.

Scripts: `tap_observable_research.py` (timeline, event shapes, tails) and the arms in
`tap_generalization_research.py` (`obs<ms>`, `obsfirst<ms>`, `cons<k>`, `acc_emgrel`, `acc_emgevent`, `kb<k>`).
Results: `results_tap_observable/` and `results_tap_generalization/focus_*`.

## Time resolution of the inputs

| Input | Updates every |
|---|---|
| Rows (model calls) | 5.5 ms |
| ACC fields | 5.5 ms (every row) |
| EMG PSD and WL | **22 ms** (one EMG frame) |

So a 20 ms pattern is one EMG frame and 40 ms is two.

## A. When is a tap distinguishable?

**Method:**
- **Taps:** every real press of the TAP and SWIPE recordings (16,000 per offset). For each, the row at onset + τ.
- **Look-alikes:** rows with muscle activity in finger-motion, fist, wrist, arm and band recordings (80,000).
- **Model:** a LightGBM per τ, cross-fitted by person.
- **Measure:** the share of taps found while only 1% of look-alike rows fire.

| τ from onset | −44 ms | −22 ms | **0** | **+22** | +44 | +66 | +88 | +110 |
|---|---|---|---|---|---|---|---|---|
| All inputs | 24% | 58% | **85%** | **85%** | 83% | 83% | 81% | 72% |
| ACC only | 27% | 60% | 79% | **82%** | 76% | 70% | 64% | 54% |
| EMG only | 10% | 9% | 9% | 11% | 11% | 12% | 12% | 11% |

All inputs at the onset, by tap type:

| | Normal | Soft | Double 2nd | Fast 240 | Swipe |
|---|---|---|---|---|---|
| Taps found | 90% | 88% | 74% | 78% | 84% |

Per-person AUC (a person's taps vs that person's look-alikes): median 0.995, worst 0.98.

**What this means:**
- **Onset:** the pattern appears at the onset and peaks in the first one or two EMG frames (0–44 ms).
- **ACC:** the contact jolt carries most of it. ACC alone peaks at +22 ms and fades after about 44 ms.
- **EMG:** on its own it does not tell a tap from finger or fist activity at a strict threshold (about 10%), and
  that is the same 66 ms before the tap. Alone, EMG describes the context (what kind of recording, which person),
  not the event.

## B. Observable-pattern labels

**Method:**
- **Teacher:** a full-size model, cross-fitted by person inside each fold's training people, scores every
  positive training row.
- **Selection per tap:** keep only the D ms of positive rows that the teacher finds most distinguishable
  (`obs<D>`: highest mean score), or the earliest window within 90% of that (`obsfirst<D>`).
- **The rest of the tap:** left out of training, never made a negative.
- **Coverage:** every tap keeps a window, so soft, second and fast taps stay in training.
- **Where the windows start** (p10 / p50 / p90): best 20 ms 4 / 22 / 60 ms after the onset, best 40 ms
  2 / 16 / 42, best 60 ms 1 / 11 / 25, earliest 0 / 5 / 23–25.

**Differences to the reference at 2.8 false/min** (nominal / placement):

| Window per tap | Normal | Soft | Double 1st | Double 2nd | Fast 240 | Swipe | Finger /min | Fist /min | Latency p50 |
|---|---|---|---|---|---|---|---|---|---|
| retrain (noise) | −0.3 / −0.4 | +0.7 / 0.0 | −0.9 / −0.7 | −0.5 / −2.1 | −0.8 / −0.2 | 0.0 / +0.1 | +0.04 / +0.16 | −0.19 / +0.15 | 0 |
| best 20 ms | −0.3 / +0.1 | 0.0 / 0.0 | −1.8\* / −1.6 | +2.1 / +1.2 | −0.1 / +2.2 | +0.3 / +0.2 | −0.55 / −0.26 | 0.0 / +0.37 | +5 ms |
| earliest 20 ms | −0.2 / −0.2 | +1.7 / +0.8 | −4.8\* / −3.8\* | **−8.1\* / −9.7\*** | −2.1 / −0.8 | −0.7 / +0.2 | −0.23 / +0.21 | +0.20 / +0.72 | −5 ms |
| best 40 ms | +1.0\* / +1.8\* | +0.3 / +2.7\* | −0.8 / −0.1 | +3.0\* / +3.9\* | +1.9\* / +3.2\* | +1.5\* / +3.4\* | −0.32 / −0.26 | +0.84\* / +1.21\* | +5 ms |
| earliest 40 ms | +0.5 / +0.9\* | +1.6 / +2.3\* | −3.1\* / −2.0\* | −3.6\* / −3.8\* | −0.6 / −0.8 | +0.5 / +1.4 | −0.29 / +0.04 | +0.13 / +0.88\* | 0 |
| **best 60 ms** | +1.2\* / +1.6\* | +1.4\* / +3.7\* | −0.8 / +0.8 | +2.2 / +2.0 | +1.7 / +4.9\* | +1.5\* / +2.9\* | −0.38 / −0.02 | +0.43 / +0.48 | +5 ms |

**At 2.0 false/min:**
- **Best 60 ms:** normal +1.3\* / +2.1\*, soft +1.6\* / +2.9\*, swipes +2.6\* / +2.7\*, fist +0.08 / +0.22.
- **Best 40 ms:** similar gains, fist +0.06 / −0.01.
- **Earliest 20 ms:** −8 to −9 on second taps.

**At the same normal-tap recall** (spending the gain on fewer false taps), best 60 ms:
- **Finger:** 1.87 → 1.44/min (−23%\*).
- **Fist:** +0.21/min (not significant).
- **All false taps:** −0.07/min.

**What this means:**
- **Pattern length:** 20 ms is not enough. The model needs about 40–60 ms of evidence per tap, and it has to be the
  most distinguishable part. The earliest frames alone cost second taps 8–10 points.
- **Tap gains:** the best 40–60 ms windows raise every tap type by 1–5 points (soft included) for 5 ms more
  latency. The model no longer spends capacity on frames that look like everything else.
- **Protection:** the labels move the errors, they don't remove them. There are fewer finger false taps but not
  fewer fist false taps.

## What the remaining false taps look like

**Method:**
- **Events:** every event of the reference fold models on their held-out people, at the 2.8/min threshold.
- **Signals:** ACC energy (sum of log1p of the current ACC spectrum) and EMG level (mean log2(WL + 1)).
- **Baseline:** each signal minus its median 600–300 ms before the event.
- **Counts:** 5,473 matched taps, 86 finger false events, 23 fist false events. Ten of the fist events come from
  one person, so this is a direction, not a precise number.

| At the event | Taps (normal + soft), median | Fist false | Finger false |
|---|---|---|---|
| ACC jump at 0 / +22 ms | +55 / +73 | +57 / +70 (AUC 0.47–0.50) | +57 / +61 (AUC 0.49–0.64) |
| EMG 22 ms before (octaves) | +0.61 | +0.21 (AUC 0.52) | **+1.36 (AUC 0.29)** |
| EMG 260 ms after | +0.25 | **+0.80 (AUC 0.36)** | +0.46 (AUC 0.43) |

**What this means:**
- **Fist:** the residual fist false taps have the same contact jolt and the same EMG as a tap at the moment of
  firing. Making a fist is a contact: the fingers hit the palm. The only measurable difference comes about 250 ms
  later, because a fist is held and a tap releases. No onset label or onset input can separate what is not yet
  observable.
- **Finger:** finger-motion false taps are observable at the onset, with too much EMG before and at the event.
  That is why the observable windows reduced them.

## Which inputs are needed

| Inputs (all with ACC + Lean-lite diff) | Count | 2.8/min: normal / soft / 2nd / fast / swipe | Finger / fist / band /min |
|---|---|---|---|
| Reference: raw EMG levels + all EMG diffs + WL ratios | 285 | — | — |
| **No absolute EMG levels** (`acc_emgrel`: EMG diffs + WL ratios) | 231 | −0.2 / −1.3\* / +0.8 / −1.5 / −0.5 | −0.15 / +0.08 / +0.31 |
| Only event-local EMG (`acc_emgevent`: now vs history, WL ratios) | 141 | +0.4 / +1.2 / **−4.8** / **+8.0\*** / −1.2 | −0.54 / +0.36 / **+2.60\*** |
| 24 / 48 / 96 most person-consistent single inputs (`cons<k>`) | 24–96 | **−31 / −8 / −6** normal, soft −47 / −25 / −12 | fist **+9.3 / +6.3 / +2.5** |

At 2.0/min, the variant without absolute EMG levels is better: normal +1.1\*, double 1st +3.2\*, double 2nd +3.1\*, swipe +1.2\*. Finger and fist are unchanged.

**What this means:**
- **Absolute EMG levels are not needed.** Without them the model is as good (better at the strict point) and
  gain-invariant by construction.
- **The spectral-shape diffs are needed.** Without them band-handling false taps rise by 2.6/min and second taps
  drop 5–8 points.
- **A "minimal consistent" input set fails badly.** The inputs that separate taps in most people one at a time are
  almost all ACC (the jolt), and ACC alone fires on fists. Fist rejection needs the combination of ACC with
  relative EMG, which no single input shows on its own. Minimizing the inputs one at a time throws that away.
  Consistency has to be judged on the whole model across people, which is what person-fold validation does.

## Combined: observable 60 ms labels and no absolute EMG levels

Both changes together (`acc_emgrel+...+obs60`: 231 inputs, 14 KB), compared with the reference:

| | Normal | Soft | Double 1st | Double 2nd | Fast 240 | Swipe | Finger /min | Fist /min |
|---|---|---|---|---|---|---|---|---|
| 2.8/min, nominal / placement | +0.9 / 0.0 | 0.0 / +2.3\* | +1.6\* / −1.0 | +1.2 / +0.4 | +1.4 / +2.3\* | +0.7 / +2.4\* | −0.42 / −0.19 | +0.49 / +0.54 |
| 2.0/min, nominal / placement | +1.6 / +1.5 | +2.2\* / +3.5\* | +3.0\* / +2.9\* | **+5.5\* / +3.7\*** | **+4.1\* / +4.0\*** | +2.6\* / +3.7\* | −0.04 / −0.13 | −0.06 / +0.24 |

**Worst cases** (12 people with finger and fist recordings) are unchanged in every variant:
- **Finger:** 7 people above 1/min at 2.8/min; the worst person is at 5–7/min.
- **Fist:** 6–7 people above 1/min; the worst person (hadas) is at 13/min.

## D. Size: the smallest model at the best performance

Observable 60 ms labels, 7 leaves, budget from 4 to 14 KB. Differences to 14 KB at 2.8/min (nominal):

| Budget | Trees | Normal | Soft | Double 2nd | Fast 240 | Swipe | Finger /min | Fist /min |
|---|---|---|---|---|---|---|---|---|
| 14 KB | 172 | 79.5% | 72.7% | 61.6% | 57.3% | 69.6% | 1.53 | 2.87 |
| 10 KB | 122 | −1.3\* | −2.3\* | −2.3\* | −3.6\* | −2.2\* | +0.08 | +0.03 |
| 8 KB | 97 | −4.1\* | −5.5\* | −6.4\* | −7.9\* | −5.1\* | +0.12 | +0.26 |
| 6 KB | 72 | −6.4\* | −9.2\* | −10.2\* | −13.4\* | −8.8\* | +0.30\* | +0.52 |
| 4 KB | 47 | −8.1\* | −16.3\* | −16.9\* | −20.6\* | −14.4\* | +1.10\* | +0.75 |
| 14 KB, today's labels | 172 | −1.2\* | −1.4\* | −2.2 | −1.7 | −1.5\* | +0.38 | −0.43 |

**What this means:**
- **14 KB is still on the rising part of the curve.** Every KB below it costs taps, first fast taps and second
  taps, then soft taps. False taps stay about the same down to 8 KB.
- **Smaller is not more general here.** Finger and fist false taps do not improve at any smaller size. The
  protection limit is not model capacity.
- **The labels buy back about 4 KB.** Observable labels at 10 KB match today's labels at 14 KB.

## Conclusions

1. **The observable tap pattern** starts at the contact and needs about 40–60 ms: two to three EMG frames, plus
   the 5.5 ms ACC jolt. 20 ms (one EMG frame) is not enough, and the earliest frames alone hurt double taps.
2. **Labels from the observable pattern work for taps:**
   - **Method:** per tap, the best 40–60 ms as judged by a person-crossfitted teacher, with the rest left out.
   - **Result:** +1 to +5 points on every tap type, soft included, and +5 ms latency.
   - **Size:** worth about 4 KB of model.
3. **Inputs:** EMG is needed only in relative form (spectral shape and change). Absolute EMG levels can go. A
   "minimal" set of individually consistent inputs fails, because fist rejection lives in the combination of ACC
   and relative EMG.
4. **Fist protection cannot come from labels or onset inputs.** At the moment of firing, the remaining fist false
   taps show the same contact jolt and EMG as taps; a fist is a contact. The difference appears about 250 ms
   later, when the fist is held. Finger false taps are separable at the onset (too much EMG), and observable labels
   cut them by about 23% at the same tap recall.
5. **Ways to better fist protection:**
   - **A hold check:** the event is decided once the contraction has released (about 250 ms latency), or the
     click is cancelled when the contraction is held. This is a product decision.
   - **More fist data:** recordings from many more people. Today there are 10 minutes of fist recordings in total.

**Recommendation:**
- **Next candidate:** today's final recipe plus observable 60 ms labels and no absolute EMG levels.
- **Before it goes into the trainer:** confirm it with 2 more training seeds (one run per variant here; the
  gains are 1–3× the retraining noise).
- **Size:** keep 14 KB. If the device needs room, 10 KB with observable labels equals today's 14 KB.

## Reproduce

    python research/gesture/tap_observable_research.py timeline
    python research/gesture/tap_observable_research.py shapes
    B=raw126_diff+augm+litediff+press2+kb14+leaves7
    python research/gesture/tap_generalization_research.py "$B+obs40" "$B+obs60" "$B+obs20" "$B+obsfirst20" "$B+obsfirst40"
    python research/gesture/tap_generalization_research.py "acc_emgrel+augm+litediff+press2+kb14+leaves7" "$B+cons48"
    python research/gesture/tap_generalization_research.py "acc_emgrel+augm+litediff+press2+kb14+leaves7+obs60" \
        "raw126_diff+augm+litediff+press2+kb10+leaves7+obs60"    # ... kb8 / kb6 / kb4
    python research/gesture/tap_generalization_research.py focus_table "$B" "$B+obs60"
    python research/gesture/tap_observable_research.py tails "$B" "$B+obs60"
