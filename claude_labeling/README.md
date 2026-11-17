# claude_labeling

**One script that writes `y_true` for taps, swipes, long presses and per-channel EMG state.**

```bash
python -m pip install numpy pandas scipy lightgbm
python claude_labeling/label_gestures.py DATA/DATA_TAP        DATA/LABELS/DATA_TAP
python claude_labeling/label_gestures.py DATA/DATA_LONG_PRESS DATA/LABELS/DATA_LONG_PRESS
```

About 3 minutes per data folder. One `<recording>.labels.csv.gz` per recording (rows in source order), plus `_events.csv.gz` and `_summary.json`.

| Column | Use |
| --- | --- |
| `tap_label` | tap/swipe onset trigger target: 1 / 0 / −1 |
| `hold_label`, `hold_ms` | long-press target; `hold_ms` lets you set a firing delay |
| `ch1_label` … `ch3_label` | per channel 0 REST, 1 ACTIVE, 2 OFF |
| `gesture_kind` | 0 none, 1 tap, 2 swipe, 3 long press (from the folder) |
| `n_ch_off` | channels confidently disconnected; at 2 or more there is no tap and no hold |

**−1 means "leave the row out of the loss".** Full column list, rules, evidence and limits: `GESTURE_LABELING_2026-11-17.md`.

`models/` must stay next to the script. Keep recordings under their original file names (they select the held-out fold model per person).

This replaces running `claude_research_tap/pipeline/export_labels.py` and `claude_research_lp/label_long_press.py` separately;
their tap, hold and channel labels are reproduced exactly, with the disconnection and cross-gesture rules added.
