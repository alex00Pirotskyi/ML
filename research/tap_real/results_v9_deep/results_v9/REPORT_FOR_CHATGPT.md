# V9 multiscale TAP research — automated aggregate summary

**Research-only. Never interpret bag AUC as real event precision, recall, double-tap accuracy or firmware false positives.**

Sources: 2790 unique recordings, 3052 duplicates removed. 
Trials: 5243 recorded, 5243 successful, 0 failed.

## Top models (recording-level, participant OOF)

| Trial | Task | Window (before / after ms) | Representation | Selection | FINGER weight | TAP vs FINGER AUC | Target vs negatives AUC |
|---|---|---|---|---|---:|---:|---:|
| 737b1851fa53b520 | combined | 60/640 | wave/relative/native | mil1 | 3.0 | 0.774 | 0.778 |
| c433428fb1000850 | combined | 60/640 | wave/relative/native | mil3 | 3.0 | 0.759 | 0.771 |
| 2f133e8a6b4da3e1 | combined | 60/640 | wave/relative/native | mil8 | 3.0 | 0.757 | 0.768 |
| d8f9c865eabd51de | tap | 100/640 | temporal/relative/native | mil8 | 3.0 | 0.749 | 0.785 |
| c68a5235b9408bf7 | tap | 100/640 | wave/relative/native | mil3 | 3.0 | 0.749 | 0.782 |
| 35e9e2bb235b7794 | tap | 320/640 | temporal/relative/native | mil1 | 3.0 | 0.749 | 0.788 |
| b3152126d7c66b2c | tap | 320/640 | wave/relative/native | mil3 | 3.0 | 0.749 | 0.792 |
| 812187cb0006c72d | tap | 320/640 | temporal/relative/native | mil3 | 3.0 | 0.748 | 0.793 |
| c8da92b1c4d98446 | tap | 100/640 | temporal/relative/native | mil1 | 3.0 | 0.748 | 0.777 |
| feeaf5c87bc4946c | tap | 60/640 | temporal/relative/native | mil3 | 3.0 | 0.747 | 0.779 |
| 6899acbbf922096e | tap | 60/640 | wave/relative/native | mil1 | 3.0 | 0.746 | 0.775 |
| 294d894748aaf538 | combined | 60/480 | wave/relative/native | mil1 | 3.0 | 0.746 | 0.767 |
| c45e6987335a3dfb | tap | 320/640 | temporal/relative/native | mil8 | 3.0 | 0.745 | 0.793 |
| 2d0a78d8013d4a60 | tap | 100/640 | temporal/relative/native | mil3 | 3.0 | 0.745 | 0.782 |
| c6a64c1fc39f85c1 | combined | 60/640 | wave/relative/native | mil1 | 1.0 | 0.744 | 0.763 |
| 8a9155247e19518f | combined | 60/640 | wave/relative/native | mil3 | 1.0 | 0.744 | 0.768 |
| 3c138af40c08f03e | tap | 100/640 | wave/relative/native | mil1 | 3.0 | 0.744 | 0.778 |
| 2028ba2cc9fd1bfe | combined | 60/640 | temporal/relative/native | mil3 | 3.0 | 0.743 | 0.752 |
| 88dbadf4f52a968a | tap | 60/640 | wave/relative/native | mil3 | 3.0 | 0.743 | 0.779 |
| 71065578ff48f472 | tap | 320/640 | wave/relative/native | mil1 | 3.0 | 0.743 | 0.787 |
| 9faed1f20b934d8a | tap | 60/640 | temporal/relative/native | mil8 | 3.0 | 0.742 | 0.775 |
| 76e2435a8f7bb254 | tap | 100/640 | wave/relative/native | mil8 | 3.0 | 0.741 | 0.778 |
| 659cd9db13ee565d | tap | 320/640 | wave/relative/native | mil8 | 3.0 | 0.740 | 0.786 |
| 2e43ae203814f996 | combined | 60/640 | wave/relative/native | mil8 | 1.0 | 0.739 | 0.769 |
| d3ee4a081870a96a | combined | 60/640 | temporal/relative/native | mil8 | 3.0 | 0.739 | 0.752 |

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
