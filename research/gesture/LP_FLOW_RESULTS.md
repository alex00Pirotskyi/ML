# Long press: why the probability looks inconsistent, and a flow that latches (2026-11-25)

**Problem seen on the band.** While holding, the LONG_PRESS probability sits near zero. It goes up when the hand
moves, and when the finger lifts it goes up and then comes back down. Nothing in it reads clearly as "holding"
or "released".

**Why.** The sensors barely show the hold itself. They show the changes:

- **A soft hold has almost no lasting muscle signal.** Activity is present 1–2 s into the hold in only 30% of soft
  long presses and 69% of mid ones (`claude_research_lp/LONG_PRESS_CHANNEL_STATE_RESEARCH_2026-11-17.md`).
- **The lift is a short transient.** After it, resting looks like holding again, so a per-frame "released?"
  probability rises at the lift and falls back within about a second.
- **Hand motion makes similar transients.**

Each panel in `figures/lp_flow_traces.png` shows one long press from people the models never saw (green = finger
down).

A per-channel regression (EMG level per channel) would not fix this. During a soft hold there is no steady level
to regress; the information is in the changes.

## The flow

The band reports a state instead of a probability. A press starts at the TAP onset (that already exists), and
two small models run after it:

1. **Release-event model:** "did the finger just lift?". It is trained on the 80 ms after each pointer release
   against the rest of the press (including the press itself) and against background motion.
2. **State model:** "does it look released?". This is the current LONG_PRESS model, retrained with the Lean-lite
   inputs.

A release is a release event ≥ 0.6, confirmed by the state model's ten-sample average ≥ 0.7 within 150 ms.

    onset (TAP fires)          -> PENDING
    confirmed release <= 600 ms -> TAP (report the tap as today)
    none by 600 ms             -> HOLDING (long press starts; tg_cancel() drops the pending tap)
    HOLDING                    -> stays HOLDING until the next confirmed release -> RELEASED

Once HOLDING, the state is latched: the probability falling back after the lift, or a single motion spike,
no longer changes it.

## Results

Setup: 3 folds by recording (stratified by folder), 2,231 recordings, 1,047 holds and 5,180 tap/swipe presses
scored on held-out recordings. Inputs are the 126 raw fields + diffs. Each model is 14 KB with trees of 7
leaves. The onset is the labelled press start.

- **Strict criteria:** the hold is recognised, the release is declared no more than 100 ms early, and it is
  declared at most 150 ms after the finger lifts.
- **Lenient criteria:** 250 ms early and 500 ms late, as in `replay.py`.

| Flow | Criteria | Holds right | Soft | Mid | Strong | Drag | Taps kept as taps | Swipes kept |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| Today's approach: state model, average ≥ 0.5 once | strict | 17% | 18% | 19% | 19% | 11% | 97% | 95% |
| | lenient | 20% | 20% | 23% | 23% | 13% | 97% | 95% |
| One model (state + Lean-lite), average ≥ 0.8 held 50 ms, latched | strict | 57% | 47% | 63% | 61% | 56% | 90% | 82% |
| | lenient | 67% | 59% | 76% | 71% | 62% | 90% | 82% |
| **Two models (release event + state, Lean-lite), latched** | strict | **61%** | 53% | 72% | 60% | 55% | 91% | 80% |
| | lenient | 66% | 58% | 77% | 68% | 61% | 91% | 80% |

- **Today's approach mostly fails at the start.** Only 31% of holds are recognised: the press transient pushes
  the average over 0.5 and it is read as a release.
- **The latched flows recognise 82–85% of holds.** With two models the release is declared at the lift (median
  latency about 0 ms, 90% within about 110 ms). The one-model version is about 50 ms later.
- **Remaining errors:**
  - about 17% of holds are taken for taps at the start (the press transient looks like a lift);
  - half of the "early" releases come after 89% of the hold has passed (the muscle relaxes before the pointer
    reports the lift);
  - soft holds stay the hardest.
- **The decision time hardly matters between 600 and 1000 ms.** At 400 ms more swipes are taken for holds
  (68% kept).
- **Tested and not better:**
  - dropping the background rows;
  - press-anchored inputs (EMG against its pre-press level and its peak since the onset, time since onset). These
    recognise more holds (66% strict) but keep only 67% of taps.

**The ceiling is the data:** 136 long-press and drag recordings (about 1,070 holds, few people) against about
2,200 tap recordings. For taps, more people was the biggest gain; the same is expected here.

## Reproduce

    python research/gesture/lp_flow_prepare.py   # Lean-lite + anchored fields for the LP cache rows (needs DATA_LONG_PRESS_LEAN8)
    MUDRA_LPF_LITE=/tmp/v3data/lp/lite python research/gesture/lp_flow_research.py
    python research/gesture/lp_flow_research.py --evaluate     # re-score flows from the stored traces

`MUDRA_LPF_NOBG=1` and `MUDRA_LPF_ANCHOR=1` select the variants. Results are in `results_lp_flow*/summary.json`.
