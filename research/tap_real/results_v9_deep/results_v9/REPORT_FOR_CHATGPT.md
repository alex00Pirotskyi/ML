# V9 multiscale TAP research — automated aggregate summary

**Research-only. Never interpret bag AUC as real event precision, recall, double-tap accuracy or firmware false positives.**

Sources: 2790 unique recordings, 3052 duplicates removed. 
Trials: 10583 recorded, 10583 successful, 0 failed.

## Top models (recording-level, participant OOF)

| Trial | Task | Window (before / after ms) | Representation | Selection | FINGER weight | TAP vs FINGER AUC | Target vs negatives AUC |
|---|---|---|---|---|---:|---:|---:|
| 737b1851fa53b520 | combined | 60/640 | wave/relative/native | mil1 | 3.0 | 0.774 | 0.778 |
| 5b825b7967e634c4 | combined | 100/640 | wave/relative/native | mil3 | 3.0 | 0.772 | 0.775 |
| d2ad11f12fb6f604 | combined | 320/640 | temporal/relative/native | mil8 | 3.0 | 0.771 | 0.782 |
| 981a1ffb5adbfd7f | combined | 320/640 | temporal/relative/native | mil1 | 3.0 | 0.770 | 0.781 |
| d262941775f4367d | combined | 160/640 | wave/relative/native | mil1 | 3.0 | 0.768 | 0.777 |
| 07ceeb13044b71e8 | combined | 160/640 | wave/relative/native | mil3 | 3.0 | 0.767 | 0.777 |
| c53aaf9693010c0c | combined | 100/640 | wave/relative/native | mil1 | 3.0 | 0.767 | 0.775 |
| 5cbeb39fb6836087 | combined | 320/640 | temporal/relative/native | mil3 | 3.0 | 0.767 | 0.779 |
| 5af5619f3f0641b2 | combined | 100/640 | wave/relative/native | mil8 | 3.0 | 0.766 | 0.773 |
| 258747696dd4b5cd | combined | 240/640 | temporal/relative/native | mil3 | 3.0 | 0.765 | 0.772 |
| 80267bcf9c112bf2 | combined | 160/640 | wave/relative/native | mil8 | 3.0 | 0.765 | 0.776 |
| f28c3873bc76a2ef | combined | 240/640 | temporal/relative/native | mil8 | 3.0 | 0.764 | 0.773 |
| 6097eeb1c5e52c52 | combined | 240/640 | temporal/relative/native | mil1 | 3.0 | 0.762 | 0.771 |
| bc2e5336d61d5959 | combined | 240/640 | wave/relative/native | mil1 | 3.0 | 0.761 | 0.776 |
| ef6d0f70640205be | combined | 320/640 | wave/relative/native | mil8 | 3.0 | 0.760 | 0.780 |
| 9747e47a7f53f65b | combined | 320/640 | wave/relative/native | mil3 | 3.0 | 0.759 | 0.777 |
| 3df48f10a42e9c15 | combined | 240/640 | wave/relative/native | mil3 | 3.0 | 0.759 | 0.777 |
| c433428fb1000850 | combined | 60/640 | wave/relative/native | mil3 | 3.0 | 0.759 | 0.771 |
| 2f133e8a6b4da3e1 | combined | 60/640 | wave/relative/native | mil8 | 3.0 | 0.757 | 0.768 |
| 9a02c7a5c634d9eb | combined | 160/640 | temporal/relative/native | mil1 | 3.0 | 0.757 | 0.766 |
| c5a4a385d0d609cb | combined | 240/640 | wave/relative/native | mil8 | 3.0 | 0.756 | 0.777 |
| 518ab074ac0dfa88 | combined | 320/640 | wave/relative/native | all | 3.0 | 0.756 | 0.776 |
| 62e6e21a955f5d0f | combined | 320/480 | temporal/relative/native | mil3 | 3.0 | 0.755 | 0.775 |
| 2a3ed94688594c76 | combined | 320/640 | temporal/relative/native | mil3 | 1.0 | 0.754 | 0.781 |
| 59b00e7782f73598 | combined | 160/640 | temporal/relative/native | mil8 | 3.0 | 0.754 | 0.767 |

## Important caveats

No synchronized physical tap onset reference exists in these folders. SoftTap and DoubleTaps are recording instructions, NOT true event timestamps.
TAP vs FINGER bag AUC is a **diagnostic for weak supervised pattern separation**, not a deployability score.
Selection of best trial from this leaderboard is optimistic without an independent final heldout set.
Event candidate population is V3 plus a weak-rise complement; real taps absent from both are unmeasurable.
No signal-quality gold annotations for independently disconnected/partially connected electrodes were supplied.

## Files for follow-up

- `SUMMARY_FOR_CHATGPT.json` small aggregate findings
- `leaderboard.csv` all successful configurations
- `trials.jsonl` all successes and failures with fold details
- `pattern_templates_by_category.csv` candidate shape averages (if a winning trial is revisited)
- `candidate_intervals_not_ground_truth.csv` double-gap hypotheses (if winner revisited)
- `out_of_fold_predictions.csv.gz` local detailed event predictions, **do not publish publicly**
