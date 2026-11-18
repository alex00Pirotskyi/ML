# Gesture correlation research

This folder contains an independent research pipeline for discovering what the Mudra sensors observe in each named recording class.

It is deliberately separate from the production trainer. It does not change labels, Optuna studies, exported models, or firmware.

## Research question

Using only the recording taxonomy in the repository (class folder + filename subtype), determine:

- What sensor patterns occur in TAP recordings?
- Which patterns distinguish TAP from FINGER, WRIST, ARM, BAND and NOTHING?
- What differs in SWIPE, tap subtypes and the other recording classes?
- Which original firmware features are informative, redundant, or correlated?
- Do these patterns generalize to people whose recordings were not used to fit the analysis model?

## Ground-truth contract

The script uses:

- The immediate parent folder as the broad class label, for example TAP, SWIPE, FINGER, WRIST, ARM, BAND, NOTHING, LONG_PRESS and the other existing folders.
- Filename tokens after the participant prefix as a descriptive subtype, for example MidTap, SoftTap, DoubleTaps or IndexFinger.
- The first filename token, normalized with the existing TAP-research alias convention, only to keep one participant entirely inside one validation fold.

It does **not** read or use:

- PointerState
- LABELS or any label sidecars
- tap_label / hold_label
- event tables or event timestamps
- teacher-model predictions
- the CSV User column
- filename tokens, folder names, participant IDs, duration, or timestamps as model inputs

The class and subtype are weak recording-level truth. There is no claim that every row in a TAP recording is a tap.

## Method

### 1. Dataset inventory and deduplication

By default, both recording roots are scanned:

- DATA/DATA_TAP
- DATA/DATA_LONG_PRESS

The second root contains some TAP/SWIPE recordings also present in the first root. Recordings are deduplicated by the SHA-256 of their uncompressed CSV payload, so a plain .csv and its .csv.gz copy count as one recording. If identical sensor data appears under conflicting class folders, all copies of that content hash are excluded and reported rather than assigning an arbitrary label.

Current repository folder taxonomy includes:

- DATA_TAP: ARM, BAND, FINGER, NOTHING, SWIPE, TAP, WRIST
- DATA_LONG_PRESS: CONNECTED, DISCONNECTED, DRAG_AND_DROP, LONG_PRESS, LOOSE, MID_LONG_PRESS, NOTHING, SOFT_LONG_PRESS, STRONG_LONG_PRESS, SWIPE, TABLE, TAP

The script discovers the actual folders at runtime; this list is documentation, not a hard-coded class allow-list.

### 2. Firmware features

All common numeric p_core columns are used. The script does not choose sensor families in advance.

- If the derived Lean-8 trees exist, it uses the 24 Lean-8 columns together with the original firmware columns.
- With the default --lean8 auto setting, missing derived trees are generated using the existing incremental research/gesture/generate_lean8_features.py.
- Use --lean8 required to fail rather than silently fall back to raw features.
- Use --lean8 off to run the original p_core feature set only.

The expected complete schema is 126 original fields + 24 Lean-8 fields = 150. The script reports the actual common count and exact names.

If the real firmware header is configured through MUDRA_LEAN8_FIRMWARE_HEADER, the existing generator can run its C/Python parity check. If no header is available, Lean-8 calculations can still be generated, but parity must be reported as not run. Do not call that firmware-verified.

### 3. Recording-level feature bags

A recording is one independent labelled sample. Its time series is summarized at fixed durations, by default:

- 100 ms
- 200 ms
- 400 ms

Windows overlap by 50%. For each original firmware feature, the script records:

- Whole-recording median, standard deviation, quantiles and median absolute deviation.
- Time-normalized absolute first-difference statistics.
- Distribution of window medians across the recording.
- Distribution of within-window standard deviation.
- Distribution of within-window p90-p10 range.

Window summaries are pooled into one vector per recording. The p90/p99 window summaries preserve evidence of short, high-activity periods without falsely declaring every window in a TAP file positive.

Duration, row count, logging cadence, folder, subtype and participant are reported as diagnostics/grouping metadata, never as model inputs.

### 4. Feature-pattern statistics

For every class and every original firmware feature, the script calculates one-vs-rest recording-level distribution differences:

- Median values by class.
- Cliff's delta effect size.
- Mann-Whitney p-value.
- Benjamini-Hochberg q-value.

It also performs paired participant comparisons for class pairs where the same participants have recordings in both classes. These use participant-level medians and a Wilcoxon signed-rank test. This is especially useful for TAP vs FINGER, TAP vs WRIST, TAP vs ARM, TAP vs BAND, TAP vs NOTHING, and SWIPE vs the same negative classes.

No feature family is forced or excluded. ACC/EMG/WL/Lean-8 labels are used only to make the resulting report readable.

### 5. Feature correlation

The correlation matrix is calculated from within-class rank residuals of recording-level feature medians. This reduces correlations caused only by different class means.

Hierarchical clusters are selected from the observed correlation structure. They are diagnostics for redundancy, not a feature allow-list or deny-list.

### 6. Recording-level classifier

For every class with enough independent participant groups, a one-vs-rest classifier is trained using all pooled p_core descriptors.

- Validation is StratifiedGroupKFold by filename-derived participant.
- No participant appears in both sides of a fold.
- Model inputs contain sensor features only.
- Metrics are recording-level ROC-AUC, average precision, balanced accuracy and sensitivity at 0.5.
- Feature gain/split importance is aggregated back to the original p_core feature.

This model answers whether the *recording's sensor profile* identifies its filename class. It is not a per-frame trigger model and it does not estimate tap event recall or latency.

## Run

From the ML repository root:

\`\`\`bash
git pull --ff-only
python -m pip install numpy pandas scipy scikit-learn lightgbm matplotlib
python -u gesture_correlation/analyze.py
\`\`\`

Recommended first full run:

\`\`\`bash
MUDRA_CPU_THREADS=60 python -u gesture_correlation/analyze.py --lean8 required --windows-ms 100,200,400 --folds 5
\`\`\`

If you want to validate the pipeline without fitting classifiers first:

\`\`\`bash
python -u gesture_correlation/analyze.py --skip-models
\`\`\`

Useful options:

| Option | Meaning |
| --- | --- |
| --roots PATH... | Override the default two data roots |
| --out PATH | Results directory |
| --lean8 auto|required|off | Use/generate Lean-8, require it, or disable it |
| --lean8-parity-files N | Number of files passed to the generator's optional C parity check |
| --windows-ms 100,200,400 | Window durations in milliseconds |
| --folds N | Maximum participant-grouped CV folds |
| --min-paired-participants N | Minimum overlapping participants for paired tests |
| --skip-models | Run descriptive statistics/correlation without classifiers |
| --rebuild-cache | Recompute recording descriptors |

The script caches the expensive recording-level descriptor matrix. A rerun with unchanged source CSV content, feature files, schema and window sizes reuses it. Use --rebuild-cache to force recomputation.

## Outputs

Results are written under gesture_correlation/results by default and are not committed.

| File | Contents |
| --- | --- |
| REPORT.md | Generated research report and interpretation |
| run_manifest.json | Dataset fingerprint, inputs, class counts, feature schema and run settings |
| recordings.csv | One row per unique recording, including class, subtype, participant and acquisition diagnostics |
| class_summary.csv | Counts, participants, duration and sampling-rate diagnostics by class |
| feature_availability.csv | Union of p_core fields and how many recordings contain each field |
| subtype_summary.csv | Counts by filename subtype |
| cache/recording_descriptors.npz | Cached recording-level feature matrix |
| descriptor_schema.json | Exact mapping from pooled descriptors back to p_core columns |
| feature_effects.csv.gz | One-vs-rest effect sizes and corrected p-values |
| subtype_feature_effects.csv.gz | Within-folder subtype contrasts, e.g. SoftTap vs MidTap |
| paired_participant_effects.csv.gz | Participant-paired class contrasts |
| feature_correlation_pairs.csv | All pairwise feature correlations, sorted by absolute value |
| feature_correlation_clusters.csv | Data-driven correlation clusters |
| within_class_spearman.csv | Correlation matrix |
| recording_level_cv_metrics.csv | Person-grouped one-vs-rest metrics |
| recording_level_feature_importance.csv | Gain/split importance aggregated to original features |
| model_skips.csv | Classes/folds with insufficient independent groups |
| plots/ | Class counts, feature correlation, class effects and CV plots |

## How to read the results

Start with REPORT.md.

For TAP vs FINGER, look at the paired participant table first, then compare its direction with the one-vs-rest feature effects and the grouped classifier importance. Repeat for WRIST, ARM, BAND and NOTHING.

A feature with a large unpaired effect but no consistent paired effect may be reflecting participant or session differences. A feature with strong class effect but high correlation with several others may be redundant rather than uniquely necessary.

The recording-level classifier is a separability diagnostic. A high AUC does not prove that an embedded detector will fire at the right instant.

## Important limitations

1. Filename/folder truth says what a recording session was intended to contain, not the exact times of individual gestures.
2. The method cannot distinguish a true tap window from a rest window inside the same TAP recording. It only studies the distribution of patterns within that recording bag.
3. A positive recording can contain missed gestures, pauses, accidental movements or recording artifacts.
4. Participant identity is inferred from the filename prefix. Review recordings.csv for aliases that should be merged or separated.
5. Collection sessions and recording protocols can confound classes. The report exposes duration and cadence differences, and the validation groups by participant, but this does not guarantee session-independent ground truth.
6. The classifier is for discovery only. It is not exported to firmware.
7. Exact per-event sensitivity and latency require independently verified event times, which this study intentionally does not invent.

## Tests

\`\`\`bash
python -m unittest gesture_correlation.test_analyze
\`\`\`
