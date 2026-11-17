# V9 multiscale TAP research — automated aggregate summary

**Research-only. Never interpret bag AUC as real event precision, recall, double-tap accuracy or firmware false positives.**

Sources: 2790 unique recordings, 3052 duplicates removed. 
Trials: 3460 recorded, 3460 successful, 0 failed.

## Top models (recording-level, participant OOF)

| Trial | Task | Window (before / after ms) | Representation | Selection | FINGER weight | TAP vs FINGER AUC | Target vs negatives AUC |
|---|---|---|---|---|---:|---:|---:|
| d8f9c865eabd51de | tap | 100/640 | temporal/relative/native | mil8 | 3.0 | 0.749 | 0.785 |
| c68a5235b9408bf7 | tap | 100/640 | wave/relative/native | mil3 | 3.0 | 0.749 | 0.782 |
| c8da92b1c4d98446 | tap | 100/640 | temporal/relative/native | mil1 | 3.0 | 0.748 | 0.777 |
| feeaf5c87bc4946c | tap | 60/640 | temporal/relative/native | mil3 | 3.0 | 0.747 | 0.779 |
| 6899acbbf922096e | tap | 60/640 | wave/relative/native | mil1 | 3.0 | 0.746 | 0.775 |
| 2d0a78d8013d4a60 | tap | 100/640 | temporal/relative/native | mil3 | 3.0 | 0.745 | 0.782 |
| 3c138af40c08f03e | tap | 100/640 | wave/relative/native | mil1 | 3.0 | 0.744 | 0.778 |
| 88dbadf4f52a968a | tap | 60/640 | wave/relative/native | mil3 | 3.0 | 0.743 | 0.779 |
| 9faed1f20b934d8a | tap | 60/640 | temporal/relative/native | mil8 | 3.0 | 0.742 | 0.775 |
| 76e2435a8f7bb254 | tap | 100/640 | wave/relative/native | mil8 | 3.0 | 0.741 | 0.778 |
| 7e4ef06a0d4bc161 | tap | 60/640 | hybrid/relative/native | mil1 | 3.0 | 0.738 | 0.778 |
| 249ee565df020057 | tap | 60/640 | temporal/relative/native | mil1 | 3.0 | 0.737 | 0.772 |
| 8438ed5a316bc324 | tap | 60/640 | wave/relative/native | mil8 | 3.0 | 0.735 | 0.774 |
| be3da5fd7e0c1373 | tap | 100/480 | temporal/relative/native | mil1 | 3.0 | 0.734 | 0.769 |
| 483a240ed9594ae1 | tap | 100/480 | temporal/relative/native | mil3 | 3.0 | 0.733 | 0.773 |
| 73a535884d3b9922 | tap | 100/480 | wave/relative/native | mil3 | 3.0 | 0.733 | 0.778 |
| 0b812117f222ba0e | tap | 100/480 | temporal/relative/native | mil8 | 3.0 | 0.732 | 0.773 |
| 8a00cea34217d1c5 | tap | 100/480 | wave/relative/native | mil1 | 3.0 | 0.731 | 0.777 |
| 7b48e58d67060355 | tap | 60/480 | wave/relative/native | mil3 | 3.0 | 0.730 | 0.774 |
| ad2f4bf2e72c98fe | tap | 60/480 | temporal/relative/native | mil8 | 3.0 | 0.730 | 0.762 |
| 77c6ab56adbe5801 | tap | 60/640 | hybrid/relative/native | mil3 | 3.0 | 0.730 | 0.773 |
| 561989d25a20bdaf | tap | 60/480 | temporal/relative/native | mil3 | 3.0 | 0.729 | 0.763 |
| d40a9a739e04e2ce | tap | 100/480 | wave/relative/native | mil8 | 3.0 | 0.729 | 0.777 |
| f3614bec77111362 | tap | 60/480 | wave/relative/native | mil1 | 3.0 | 0.728 | 0.770 |
| c55d62ccaa4ea26b | tap | 60/480 | temporal/relative/native | mil1 | 3.0 | 0.728 | 0.759 |

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
