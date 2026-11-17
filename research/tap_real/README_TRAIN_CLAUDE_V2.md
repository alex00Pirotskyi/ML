# Train Claude V2 onset labels for an embedded TAP+SWIPE detector

This training path is experimental. It combines TAP and SWIPE as one binary onset
class. It does **not** certify physical tap recall or free-living false activations.

## Run

From the root of the `alex00Pirotskyi/ML` repository:

```bash
git pull --ff-only
python3 -m pip install -U numpy pandas scipy scikit-learn lightgbm optuna joblib
python3 -u research/tap_real/train_claude_v2_optuna.py
```

The script uses 80% of affinity/cgroup-constrained logical CPU threads, with a
minimum of one thread. This is a hardcoded constant; there is **no --threads**
argument. Training with 40 Optuna trials may take substantial time.

If absent, the script first calls the committed Claude V2 label exporter to
create `DATA/DATA_TAP_LABELS_CLAUDE_V2_20261117` (strong-only labels,
0–80 ms onset, weak press partners not promoted). The original data are read-only.

The trainer validates the exact original recording hashes using
`claude_research_tap/results/folds.csv`. It creates a temporary V6-schema
adapter under `research/tap_real/cache_claude_v2_v6_schema/`, without changing
the original labels, and trains from **known** rows only. A positive bag with
no strong event is omitted, not incorrectly treated as a negative. The
default excludes unverified background rows in positive bags from negative
training and uses trusted negative folders for negative examples.

## Optimization and outputs

The runner reuses `research/tap_real/train_optuna_student_v6.py` with:
- multiobjective Optuna: maximize grouped out-of-fold **silver-label F1** and
  minimize compact static C-model data bytes as estimated by `export_plus.py`;
- three subject-group CV folds, a separate subject-held-out evaluation, and
  forty trials by default;
- ten C header exports, matching JSON, pickle files and a C syntax smoke
  compilation against a synthetic `t_core` stub;
- raw, directly addressable firmware EMG-PSD, ACC-PSD and WL features. The
  126-column features from Claude's experiment are not exported in this run;
  they require separate firmware schema and C/Python parity validation.

Outputs go to `research/tap_real/results_claude_v2_optuna/`, including
`summary.json`, `CLAUDE_V2_PROVENANCE.json`, and `models/`.

The upstream V6 exporter has legacy filenames (`tap_swipe_v6_...`) and
some documentation referencing V5. The wrapper appends explicit
Claude V2 provenance to its output reports.

Do not rerun against a nonempty result directory: archive or rename the old
experiment first. The generated label cache and outputs are Git-ignored.

**Limitations:** Cross-validated F1 means agreement with silver labels, not
independent physical-onset recall. True deployment evaluation needs
one-to-one event matching (especially double taps), held-out wearer/session
recordings, free-living false activations/hour, nRF52832 build, real CPU/RAM/
flash measurement, and actual firmware C parity. Previous V5/V6 models remain
untouched.
