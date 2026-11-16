# Mudra TAP V9 — exhaustive multiscale pattern research

**Status: research, not production labels.** This runner is designed to run unattended on a Linux server and write persistent results you can share back with ChatGPT through an aggregate GitHub branch. It does not read or modify any production label exports and never creates certified per-event positive labels.

## What it tests

1. **Independent physiological event hypotheses:** V3 event-state/re-arm candidates plus low-threshold EMG-rise candidates. The candidate pool is fixed across window/representation comparisons (no Oculus pointer or metronome used to mine events).
2. **Window/context brute force:** 60–320 ms before candidate peaks and 60–640 ms after peaks (5 × 7 deep combinations; sampled at 20 ms by default). The entire 3-electrode × 5 PSD-zone trajectory is retained. The decoded PSD zone values are log-coded; we use per-zone differences, never sum log numbers as powers.
3. **Representations:** amplitude, statistical moments, temporal timecourse, channel-by-zone spectral evolution, full waveform, hybrid. Compare ranked vs physical electrode order, event-normalized vs absolute amplitude.
4. **Weak event selectivity:** original all-events-as-positive baseline vs top-1, top-3, top-8 weak-bag selection. We explicitly **do not** mistake any of these choices for verified physical taps. Hard FINGER negative weighting ×1/×3.
5. **Tasks:** TAP-only vs FINGER and other true recording-level negatives, plus combined TAP/SWIPE vs hard negatives (SWIPE is POSITIVE for the combined target).
6. **Participant-held-out** five-fold scoring, plus recording-subtype SoftTap/DoubleTaps/StrongTap comparisons. Each recording has equal supervision weight, regardless of length or number of candidate events.
7. Optional **independent motif clustering** (20 fold-heldout cluster-size/context experiments, fitting PCA and k-means on TRAIN participants only, using bag-frequency enrichment instead of V5 positives).
8. Optional **simulated electrode dropout**: mask each channel alone or each pair, rebuild features and measure OOF score degradation. This is a stress test, not a substitute for actual bad-contact records.
9. Source SHA-256 de-duplication, quarantine of cross-folder label conflicts and validation, chunked resumable extraction, per-trial JSONL checkpoints and report generation.

**What it cannot establish:** actual physical tap timing, missed soft taps not mined as candidates, true double-tap split accuracy or real firmware false taps/hour. For that, use an independent synchronized physical reference and the separate V8 evaluator. Candidate-gap histograms and AUC against recording folders **must not** be interpreted as timing ground truth.

## Automatic CPU threads (not configurable)

The runner always allocates **80% of available logical CPU threads** to LightGBM, rounded down (minimum one thread). It uses Linux CPU affinity/cpuset when available, otherwise `os.cpu_count()`. The ratio is hardcoded as `CPU_THREAD_FRACTION = 0.80` in Python. The CLI intentionally has **no `--threads` flag**. At startup it prints `V9_CPU_BUDGET logical_threads ... worker_threads ...`. On a server with 32 available logical threads, this uses 25 workers. Other libraries may independently use threads, so this is a LightGBM worker limit, not a strict whole-process CPU quota.

## Install on the server

Put `tap_multiscale_research_v9.py` next to the existing `research/tap_real/tap_event_shape_v3.py` in `alex00Pirotskyi/ML`. Keep the `DATA/DATA_TAP` folder in the project repository. On Linux (Python 3.10+; recommend >=16 GB RAM and enough disk for the source data and temporary outputs):

```bash
cd ML
python3 -m venv .venv
source .venv/bin/activate
pip install -U numpy pandas scipy scikit-learn lightgbm
```

First run a small smoke search after full extraction (does not compare against real onset ground truth):

```bash
python -u research/tap_real/tap_multiscale_research_v9.py all \
  --input-dir DATA/DATA_TAP \
  --out-dir research/tap_real/results_v9_run1 \
  --preset smoke --winner-analysis
```

Then for a serious all-data **hours-long standard grid** (576 full CV configurations, plus optional 20 unsupervised motif experiments and channel-dropout stress):

```bash
python -u research/tap_real/tap_multiscale_research_v9.py all \
  --input-dir DATA/DATA_TAP \
  --out-dir research/tap_real/results_v9_standard \
  --preset standard --folds 5 --trees 160 \
  --winner-analysis --motif-scan --dropout-stress \
  2>&1 | tee tap_v9_standard.log
```

For **complete deep brute force** use `--preset deep` instead (6,720 configurations × 5 grouped CV folds, each training one or two models; runtime may be many days depending on the server). `--hours-limit 12` stops *gracefully after the next completed trial*, and rerunning the identical command reuses source chunks and skips completed trials. **Default `--hours-limit 0` runs every configuration without time-based early stop.** Use `tmux` or `screen` if SSH may disconnect. Large grids may need substantial compute budget; progress is printed after each trial.

### Crash recovery

Rerun the **identical** command and out directory. The runner validates the original SHA-256 source hashes and skips completed extraction chunks/trials. If a source archive or study configuration has changed, use a **new versioned output directory** rather than silently mixing incompatible results. An incomplete chunk reports the exact chunk that needs investigation; do not blindly delete everything.

### Results to send back

The most important files are:

- `results_v9/SUMMARY_FOR_CHATGPT.json` — aggregate scores and top configuration.
- `results_v9/REPORT_FOR_CHATGPT.md` — readable top-25 summary and limitations.
- `results_v9/leaderboard.csv` — every completed successful experiment.
- `results_v9/motif_summary.json` and `motif_cluster_scan.csv` — independent unsupervised motif search results, when enabled.
- `results_v9/SIMULATED_CHANNEL_DROPOUT.json` — contact robustness proxy, when enabled.
- `results_v9/DATE_SLICES_DESCRIPTIVE.json` — local-only OOF date slices; **not** independent leave-date-out training.

Detailed `source_manifest.json`, cached event waveforms, per-file predictions, participant names, CSVs and source archives stay local. **Do not publish them in a public repository without explicit review.** The summary score and selected window can be optimistic due to comparison of thousands of configurations on the same OOF folds; a truly independent test set must decide production readiness.

To let ChatGPT inspect an aggregate run without uploading large files, copy `publish_v9_aggregate.sh` into your repository and run it **with GitHub push credentials**:

```bash
bash publish_v9_aggregate.sh research/tap_real/results_v9_standard my-run-01
```

The helper builds an isolated, orphan Git worktree (no second checkout of the large dataset), then pushes a dedicated `research/tap-v9-aggregate-my-run-01` branch containing ONLY the aggregate reports. It does not change the current working branch. It prints a GitHub URL you can paste into the next chat. Or upload `SUMMARY_FOR_CHATGPT.json`, `leaderboard.csv`, and `REPORT_FOR_CHATGPT.md` directly here.

## Interpretation and decision discipline

The score to prioritize is *held-out recording-level TAP-vs-FINGER ROC AUC*, with SoftTap/DoubleTaps subtype results and simulated dropout checks. A high bag AUC **does not make the selected event onsets correct**. Weak multiple-instance top-k selection can latch onto contextual artifacts; motif clustering may separate recording sessions without separating a true tap. The complete waveform includes future samples so it is appropriate for an offline teacher, **not** real-time embedded inference. Do not export training positives until independently aligned physical onsets verify precision, recall, hard-negative false activations/hour and double-tap recovery.
