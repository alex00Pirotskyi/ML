# ML

Python utilities for gesture dataset validation, organization, PSD processing, and model export.

## Automatically commit CSV files as `.csv.gz`

Enable the repository's Git hook **once per local clone**:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

Then use ordinary Git commands:

```bash
git add .
git commit -m "Add recordings"
git push
```

The pre-commit hook automatically replaces **staged** `*.csv` entries with
losslessly compressed `*.csv.gz` files in the commit. Original `.csv` files
remain on your disk; no uncompressed CSV data is included in the new commit.
It also works for CSVs in nested folders and files with spaces in their names.
The archives have deterministic contents (gzip timestamp set to zero).
Already compressed `.csv.gz` files are left as-is. Python 3 is required.

To read compressed CSVs directly:

```python
import pandas as pd

df = pd.read_csv("DATA/recording.csv.gz")
```

**Important:** Git does not automatically commit new, untracked files or push
commits. You still run `git add`, `git commit`, and `git push`; the hook takes
care of compression during `git commit`. Hooks are local Git configuration,
so the one-time setup must be done separately on each clone. If you bypass or
do not enable the hook, raw CSV files can be committed normally.

The existing `valid.py`, `org.py`, and `psd_mean.py` currently expect plain
`.csv` files, so keep or decompress local CSVs when using those scripts.
For archives approaching 100 MB each, use Git LFS instead of regular Git.

### Faster compression

The pre-commit hook processes CSVs in parallel (80% of detected logical CPU cores by default, rounded), uses
fast gzip level 3 instead of level 9, and stages thousands of archives with
batched Git operations. To adjust speed vs archive size for one commit:

```bash
CSV_GZIP_WORKERS=12 CSV_GZIP_LEVEL=1 git commit -m "Update recordings"
```

Set `CSV_GZIP_LEVEL=9` for the smallest archives (slower), or keep level 3
for a balance of size and speed. Git reads the exact **staged** CSV contents,
so edits made after `git add` do not silently affect the compressed commit.
If you update the hook while a commit is already running, that running commit
still uses the old hook; the replacement takes effect on the next attempt.

## TAP model: `make train`

```bash
make train      # file folds -> signal labels + inputs -> the legacy F1 trainer -> event check, in the background
make status     # step, Optuna trials done, last lines, errors; the event table when finished
make stop
make smoke      # the whole chain on 2 recordings per folder and fold, 3 trials (a few minutes)
```

The legacy F1 trainer (`train_tap_f1.py`: Optuna LightGBM + XGBoost, F1 at 0.85 + model size, 96 KB maximum,
`gesture_constants.py`) unchanged except its paths and **one line that drops unknown rows** (`state < 0`). Folds are
**recording files**, stratified by gesture (`fold_split.py` -> `DATA/fold_roadmap_files.json`); no person split or
fold. The data (`build_tap_f1_dataset.py` -> `DATA/DATA_TAP_F1/`): signal labels with a repaired pointer
(`claude_labeling/tap_signal.py`), the legacy model's inputs + ACC ratios (`tap_f1_features.py`), TAP / SWIPE
recordings training on every 2nd row (the legacy model's data mix). Settings: `LABELS=legacy|v10rescue|v10`,
`FEATURES=newacc|leg|new|lite`, `TAPROWS=0.5`, `TRIALS=100`. Results: `TAP_MODELS_F1_FILEFOLDS/` (models, C headers,
`event_eval.md`). Evidence: [research/gesture/TAP_F1_LEGACY.md](research/gesture/TAP_F1_LEGACY.md). The previous
pipeline is `make train-unified` ([research/gesture/PIPELINE.md](research/gesture/PIPELINE.md)).

## Long-press sEMG pseudo-labeling

Generate firmware-aware, causal and non-destructive per-sample pseudo-labels for each sEMG
channel: resting, high activity, and suspected disconnection (0-1), plus a
strict global binary `lp_state` where **1 = confidently resting on all
channels**, otherwise **0**. Unknown cases have `lp_state_weight` so they
are not treated as verified active negatives.

```bash
python -m pip install numpy pandas
python lp_rest_labeler.py
```

Default input: `DATA/DATA_LONG_PRESS`. Output:
`DATA/DATA_LONG_PRESS_LABELED_V2` (ignored by Git by default).
Full documentation: [LP pseudo-label teacher](research/LP_LABELER_README.md).
The teacher does not use PointerState or class_id as truth; scores are not
validated physical muscle-state or electrode-impedance probabilities.


Firmware-corrected teacher v2:
- EMG_PSD_MEAN=0 at acquisition; PSD zones are uint16 log10(sum squared FFT-bin magnitudes+1) * 10000.
- The emg_psd_mean field is still a 15-frame temporal average of logged spectra.
- Per-channel resting floors and activity are causal; uncalibrated/poor-contact rows are UNKNOWN and low-weight.
- Legacy teacher v1 is preserved under research/legacy/; corrected research report: research/LP_CAUSAL_FIRMWARE_PSD_RESEARCH_2026-11-16.md.


## Experimental LP sensor labeling V3 (offline, opt-in)

The existing V2 labeler remains unchanged. V3 is an experimental offline
weak-supervision teacher with a separate output folder, never a direct
real-time firmware inference algorithm. It uses WL, logged five-zone
PSD (current and temporal mean), motion, and head/tail WL to propose
rest, high-activity, disconnection-artifact and UNKNOWN scores per channel.

Run from repository root:

```bash
python -m pip install numpy pandas scipy scikit-learn
python lp_v3_labeler.py --dry-run --limit-per-group 2
python lp_v3_labeler.py
```

The output is `DATA/DATA_LONG_PRESS_LABELED_V3` (Git-ignored).
Original recordings and V2 outputs stay unchanged. V3 emits
`label_summary.json` and `review_queue.json` to prioritize
uncertain samples and time spans for checking against physical truth.

**Warning:** These are pseudo-labels, not measured muscle states or
electrode impedance. V3 intentionally examines the entire recording to
identify low/high regimes, and the final firmware student must be
trained/validated using causal features. It uses folder context only
as weak reference supervision for CONNECTED/NOTHING samples.
Low-confidence `lp_state=0` means UNKNOWN, not confirmed pressing.
Do not train on every zero as a verified negative.

[Full research, measurements and limitations](research/LP_V3_SENSOR_LABELING_RESEARCH_2026-11-16.md).
