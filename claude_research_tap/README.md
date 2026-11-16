# claude_research_tap

Research on labeling taps and swipes in `DATA/DATA_TAP`.

- `TAP_SWIPE_LABELING_SYSTEM_V2_2026-11-16.md` – current report. Start here.
- `TAP_SWIPE_ACC_PATTERN_AND_FUSION_LABELING_2026-11-16.md` – first report (the ACC finding). Its label counts and pointer-error estimates are superseded by V2.
- `pipeline/` – scripts. `results/` – event table, calibration, folds, student comparison, figures.

## Get the labels (no model needed, about one minute)

```bash
python -m pip install numpy pandas
python claude_research_tap/pipeline/export_labels.py DATA/DATA_TAP \
    claude_research_tap/results/events_v3.csv.gz claude_research_tap/results/summary.json \
    DATA/DATA_TAP_LABELS_V3
```

One `<recording>.labels.csv.gz` per unique recording, rows in source order:
`Timestamp, onset_label, active_label, gesture_kind, event_id, weight`.
`onset_label` is the training target for an onset trigger: 1 in the 0–80 ms after a confirmed gesture onset, 0 trusted negative, −1 unknown. Exclude −1 from the loss.

Options: `--win A B` (onset window in ms), `--use-weak` (add weaker presses at weight `p_real`), `--promote-double-partners` (label unpressed second taps in DoubleTaps recordings at weight 0.7).

## Rebuild everything

Run from an empty working directory; scripts read and write files there. Needs `numpy pandas scipy scikit-learn lightgbm matplotlib`, about 7 GB RAM and 6 GB disk. `P=claude_research_tap/pipeline`.

| Step | Command | Produces |
| --- | --- | --- |
| 1 | `python $P/build_cache.py DATA/DATA_TAP cache` | per-recording arrays, `cache/meta.csv` |
| 2 | `cp claude_research_tap/results/folds.csv . && python $P/make_meta.py` | `meta.pkl` with person groups and folds |
| 3 | `python $P/presses.py` | `presses.pkl`, every pointer press |
| 4 | `python $P/v2.py` | teacher round 1, `scores2.pkl` (out-of-fold) |
| 5 | `python $P/v3.py` | teacher round 2 on cleaned labels, `models3.pkl` |
| 6 | `python $P/score_full.py` | `scores3f.pkl`, every 10 ms of every recording |
| 7 | `python $P/label3.py scores3f.pkl ALL3 out3` | `out3/events.csv`, `summary.json`, `calibration.csv` |
| 8 | `python $P/export_labels.py DATA/DATA_TAP out3/events.csv out3/summary.json labels` | per-row labels |
| 9 | `python $P/recache.py && python $P/student_build.py` | 126 firmware columns on a 10 ms grid |
| 10 | `python $P/student.py out3/events.csv F126 P_raw,P_on,F_ext,F_on` and `python $P/student2.py out3/events.csv S_0_80,S_0_80_nobg` | student out-of-fold scores per label variant |
| 11 | `python $P/student_eval.py out3/events.csv st_score_out3_F126_*.npy` | event-level comparison |

`timing.py` is the pointer-offset study and `sheet.py` draws the contact sheets. The scripts are the code as run with paths made portable; they compile but the full chain was not re-run from a clean checkout.
