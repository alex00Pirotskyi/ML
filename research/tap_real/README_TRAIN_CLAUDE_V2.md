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

**The classifier decision threshold is hardcoded at 0.85 everywhere**:
Optuna DOES NOT sample or optimize it. The 0.85 operating threshold is
identical for all CV folds, held-out testing, candidate ranking and
`TAP_SWIPE.h` export. Optuna optimizes model structure, feature groups,
positive-class weight, and other training hyperparameters versus C model size.

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
- all **126** direct raw `p_core->` columns from the original recordings: 72 ACC (current/oldest/mean/rate), 45 EMG (current/oldest/mean) and 9 WL (full/head/tail). No Python-only derived features are added.
  - Optuna feature-group comparisons include the full **126** and original **36** (explicitly seeded first), ACC-only 72, EMG-only 45, and additional group ablations.
  - The C-header compiler smoke test uses a synthetic `t_core` covering 126 fields; the real nRF52832 firmware ABI/macro values and computational cost still need validation.

Outputs go to `research/tap_real/results_claude_v2_optuna_126_thr085/`, including
`summary.json`, `CLAUDE_V2_PROVENANCE.json`, and `models/`.

A small backwards-compatible hook seeds the first two Optuna trials with `full126` and `instant36`, guaranteeing a direct side-by-side training baseline. At least one of the 10 exported C models is guaranteed to be a `full126` model; the other nine prioritize the quality/size trade-off.

## Stable firmware header and existing results

The training wrapper automatically publishes a **stable firmware interface**:

- `models/TAP_SWIPE.h` — default firmware header, using plain
  `TAP_SWIPE_MODEL_H` include guard and `TAP_SWIPE_*` constants.
- `TAP_SWIPE_predict(&m_feat_core)` — convenient uppercase macro alias
  to the generated native `tap_swipe_predict(&m_feat_core)` function.
- `models/TAP_SWIPE_feature_schema.json` — exact C input feature mapping.
- `models/TAP_SWIPE_model_info.json` and `models/TAP_SWIPE_VARIANTS.json`
  — frozen source trial and balanced-model selection information.
- `models/candidates/rank_01/TAP_SWIPE.h` through
  `rank_10/TAP_SWIPE.h` — all ten models with **identical API**. Use
  only one candidate's header in a firmware build.

Default selection chooses the **smallest static model** whose **training
cross-validation silver F1** is within 0.015 absolute of the best CV F1.
This avoids using the held-out test F1 for the selection. For each new fixed-0.85 study, the default is selected only from **CV F1
measured at 0.85**. Prior study rankings are not reused.

The earlier successful 40-trial study in
`research/tap_real/results_claude_v2_optuna_126/` **optimized a variable
decision threshold**, so its F1 numbers and rank ordering are NOT valid for the
new fixed-0.85 objective. The old study is deliberately preserved and will
not be silently republished as if it used 0.85.

To run the correct fixed-0.85 study, use:

```bash
git pull --ff-only
python -u research/tap_real/train_claude_v2_optuna.py
```

This starts a **new experiment** in `results_claude_v2_optuna_126_thr085`.
Original labels are reused, so relabeling is unnecessary. If that new study
has already completed, the same command republishes its stable C headers
without retraining; alternatively run:

```bash
python -u research/tap_real/publish_tap_swipe_headers.py
```

**No rank/trial names in firmware C:** The exporter now directly emits
`models/candidates/rank_XX/TAP_SWIPE.h` with the `TAP_SWIPE_MODEL_H`
include guard and `TAP_SWIPE_*` macros. It no longer creates versioned
per-trial `.h` files for new Claude training runs. Saved `.pkl` and
research `.json` reports still include trial identifiers for traceability.

Existing completed fixed-0.85 runs can be repaired without retraining:

```bash
git pull --ff-only
python -u research/tap_real/publish_tap_swipe_headers.py
```

This republishes all candidate headers with the stable API, updates their
output references and removes **only original unmodified** rank-named generated
headers after successful export and validation. An edited legacy header is
preserved and reported rather than silently deleted. The publisher is
idempotent. New firmware should include only
`models/TAP_SWIPE.h` from the fixed-0.85 experiment.

The upstream V6 **research model/report** filenames still include
`tap_swipe_v6_...`; this does not appear in any newly generated C header.
For incomplete studies without `summary.json`, the trainer refuses to
overwrite the output directory. Generated label cache and training outputs are
Git-ignored.

**Limitations:** Cross-validated F1 means agreement with silver labels, not
independent physical-onset recall. True deployment evaluation needs
one-to-one event matching (especially double taps), held-out wearer/session
recordings, free-living false activations/hour, nRF52832 build, real CPU/RAM/
flash measurement, and actual firmware C parity. Previous V5/V6 models remain
untouched.
