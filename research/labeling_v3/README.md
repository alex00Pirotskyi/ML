# TAP labeling V3 research

Run from the repository root:

    python -u research/labeling_v3/analyze.py

The script uses DATA/DATA_TAP compressed recordings only, deduplicated by decompressed
SHA-256. It re-runs the existing offline teacher on the current data, caches each
recording in an isolated results directory, and benchmarks seven label strategies
with identical subject splits, model settings, positive-weight grid, fixed 0.85
threshold and a 14 KiB exact exporter-accounted static-data limit. No production
label files, headers, or studies are modified. It includes both raw126 and
fold-local selected relative EMG features, mild training placement augmentation,
and normal plus simulated placement-stress evaluation.

The adaptive variant uses inner participant-disjoint causal models to select one
contiguous 50 ms interval per confirmed event; it does not discard low-scoring
events. This is an experiment, not a guaranteed improvement. Negative folders
remain authoritative; uncertain positive-bag rows are excluded. Baseline80 uses
the original labels. Weak/unconfirmed presses are an evaluation reference, not
automatically promoted. No long-press recordings are included in this TAP-only
benchmark. Data quality is inherited from the existing teacher; video-confirmed
physical reference events are still needed.

The script compares three loss weights per variant at the same fixed threshold.
It uses a fixed tree configuration rather than per-variant Optuna tuning to
isolate labeling effects. Results are checkpointed per fold/variant/weight;
rerunning resumes. The baseline-derived FINGER/fist FPM limits are defaults,
not product acceptance targets. Set --finger-fpm and --fist-fpm to explicit
targets if desired. If no candidate meets the limits it reports NONE.

Outputs: research/labeling_v3/results/run_*/REPORT.md and summary.json.
The report contains only development CV; the reserved participants remain
untouched unless --holdout is supplied after selection. All reference matches
are one-to-one. Pointer presses remain imperfect reference events. Double pairs
are inferred from the press spacing, not independent video truth.

Quick unit checks:
    python research/labeling_v3/analyze.py --self-test

Optional shorter run (separate output, not a representative benchmark):
    python -u research/labeling_v3/analyze.py --limit-per-folder 50 --folds 2 --variants baseline80,trusted50 --weights 1 --output /tmp/labeling-v3-smoke

Full run then reserved-participant audit:
    python -u research/labeling_v3/analyze.py --holdout

Send REPORT.md and summary.json for interpretation; don't commit the generated
recording caches or participant-level manifests.
