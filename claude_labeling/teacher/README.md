# Tap teacher, retrained on gesture-aware file folds

The labeller (`claude_labeling/label_gestures.py`) decides which pointer presses are real taps with a tap teacher: a
LightGBM model on ACC, EMG and WL around every 10 ms frame (the `claude_research_tap` V2 teacher, two rounds). Every
recording is scored by a model that never saw it.

Until 2026-11-26 the five teacher models were cross-fitted on **person** folds and the labeller picked the model by the
person in the file name. That is gone: the teacher is now cross-fitted on **gesture-aware file folds**, the same kind of
split as the trainer, and the labeller picks the model by the recording's content hash. No step of labelling or training
splits or folds by person.

| | before | now |
|---|---|---|
| Folds | 5 person groups (first letters of the user name) | 5 file folds, stratified by recording type (folder / gesture / bpm), identical copies together |
| Which model scores a recording | the fold of its person | the fold that never saw the recording (content SHA-256); the mean of the five for new recordings |
| Models | `models/tap_teacher_fold*.txt`, `tap_teacher.json` (kept, no longer used) | `models/tap_teacher_file_fold*.txt`, `tap_teacher_file.json` |
| Contact model | the fold of the person | the mean of its five folds (it was trained on other recordings' channels; no dispatch) |

## Retrain (about an hour on 2 CPUs, 7 GB RAM)

From an empty working directory (`T=<repo>/claude_labeling/teacher`, `D=<repo>/DATA/DATA_TAP`):

```bash
python $T/build_cache.py $D cache                 # per-recording arrays + cache/meta.csv (content hashes)
python $T/make_file_folds.py cache folds.csv      # gesture-aware file folds (5), never person folds
python $T/make_meta.py                            # meta.pkl (requires folds.csv)
python $T/presses.py                              # every pointer press
TAP_CACHE=cache python $T/v2.py                   # round 1, out-of-fold scores
TAP_CACHE=cache python $T/v3.py                   # round 2 on cleaned labels
TAP_CACHE=cache python $T/score_full.py           # every 10 ms of every recording
python $T/label3.py scores3f.pkl ALL3 out3        # thresholds (false events / min in negative folders), calibration
python $T/export_teacher.py . <repo>/claude_labeling/models
```

The scripts are `claude_research_tap/pipeline` as run, with three changes: the folds come only from `folds.csv`
(file folds; `make_meta.py` refuses to draw person folds), round 1 trains only the feature family round 2 uses, and the
training matrices have room for the larger data set. After a retrain, `make train` rebuilds the labels by itself
(the label summary records the teacher, `label_gestures.TEACHER`).
