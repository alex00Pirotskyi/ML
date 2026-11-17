# claude_research_lp

Labels for long press and per-channel EMG state in `DATA/DATA_LONG_PRESS`, and the research behind them.

- `label_long_press.py` – **the labeler. One script, no training step.** Use its output as `y_true`.
- `models/` – contact models the labeler needs (keep next to the script).
- `LONG_PRESS_CHANNEL_STATE_RESEARCH_2026-11-17.md` – the research report. Start here for how and how well.
- `research/` – the pipeline as run, and a causal REST/ACTIVE/OFF tool (`research/state_tool/lp_state.py`).
- `results/`, `figures/` – label statistics, disconnection delay measurements, plots.

## Label the data (about 2 minutes for the whole folder)

```bash
python -m pip install numpy pandas scipy lightgbm
python claude_research_lp/label_long_press.py DATA/DATA_LONG_PRESS DATA/DATA_LONG_PRESS_LABELS
```

One `<recording>.labels.csv.gz` per recording, same sub-folders, one row per source row in source order, plus
`_label_summary.json`. A single file: `python claude_research_lp/label_long_press.py rec.csv.gz out/ --kind hold`.

| Column | Values | Meaning |
| --- | --- | --- |
| `Timestamp` | | copied from the source row |
| `ch1_label`, `ch2_label`, `ch3_label` | 0 / 1 / 2 / −1 | REST / ACTIVE (muscle activity) / OFF (electrode not in proper contact) / unknown |
| `ch1_contact` … | 0–1 | contact-problem score behind OFF |
| `any_active` | 1 / 0 / −1 | some channel ACTIVE / every channel REST / otherwise |
| `hold_label` | 1 / 0 / −1 | long press held / not held / unknown |
| `hold_ms` | ms or −1 | time since the labeled hold started |
| `press_id` | int or −1 | pointer press the row belongs to |
| `press_confirmed` | 1 / 0 / −1 | sensors show activity at that press / do not / cannot tell |

**−1 means "leave this row out of the loss", not "negative".**

Suggested targets:

| Model | y_true | Rows to train on |
| --- | --- | --- |
| Long-press state | `hold_label` | `hold_label >= 0` |
| Long-press event that fires after N ms | `hold_label == 1 and hold_ms >= N` | `hold_label >= 0` |
| Per-channel state | `ch<c>_label` (3 classes) | `ch<c>_label >= 0` |
| Contact alarm | `ch<c>_label == 2` | `ch<c>_label >= 0` |
| Any muscle activity | `any_active` | `any_active >= 0` |

## What the labels are, in short

- **The folder is the ground truth for what kind of recording it is.** Presses can be holds only in SOFT_/MID_/STRONG_LONG_PRESS,
  LONG_PRESS and DRAG_AND_DROP. TAP and SWIPE recordings are all "not held" (hard negatives). The contact protocol folders are all "not held".
- **Inside long-press recordings, PointerState says roughly when; the sensors confirm and time it.** A hold is labeled only if some channel
  shows at least 30 ms of activity from 300 ms before to 500 ms after pointer-down. It starts at that activity onset (the sensors lead the
  pointer by about 100 ms) and ends 250 ms before release. Unconfirmed presses are unknown (`--trust-pointer` labels them anyway).
- **Channel state is per recording and channel.** OFF comes from a contact model trained on the CONNECTED / DISCONNECTED / LOOSE / TABLE
  protocol and scored by the fold model that never saw that person. REST and ACTIVE are relative to the recording's own resting floor
  (within 3 dB / at least 6 dB above it).

Result on the full folder: 1,075 holds labeled (534,902 rows held, 4,268,274 not held, 207,089 unknown), no failures.
Confirmation rates of verifiable long presses: strong 95.6%, mid 95.4%, drag 91.5%, soft 84.7%.

Keep recordings under their original file names. The person is read from the start of the file name to pick the contact model that
never saw them; an unknown or renamed file gets the average of the five models, which is the right choice for new people.

This is an **offline** labeler: it uses the whole recording (a floor per recording, a 1 s minimum for OFF). It is meant for `y_true`,
not as the firmware model. A causal version is in `research/state_tool/`.

> **Labels:** `claude_labeling/label_gestures.py` now labels taps, swipes and long presses in one script and reproduces these hold and channel labels exactly, adding the disconnection rule.
