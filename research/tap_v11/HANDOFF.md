# V11 TAP labeling — implementation and experimental handoff for Claude Code

**Purpose:** Build, execute, and validate an automatic correction layer around the manually reviewed September dominant-axis ACC labeling data. Do not merely write another research proposal.

**Repository:** https://github.com/alex00Pirotskyi/ML  
**Inspected reference commit:** `28efd6e1cb992475f1f8e4caf9b216543b91c779`  
**Isolated implementation branch:** `research/tap-v11-reviewed-seed`  
**Reviewed archive supplied by the owner:** `result_TAPANDSWIPE2709_newlbl_mehod123 (4).zip`  
**Original labeler supplied by the owner:** `Pasted text.txt`, version `v10.0-deterministic-restored-2026-09-17`  
**Additional raw input:** `DATA/DATA_TAP` and/or the owner's `DATA_TAP.zip`.

## 0. Direct instruction to Claude

You have access to a working local Python environment. Read the supplied original labeler and manually reviewed archive, audit their actual contents, then implement and run V11. Do not ask the owner to manually fix thousands of PointerState errors. Do not overwrite the original reviewed archive or the current production trainer. Use the latest local implementation of `make train` if it differs from the inspected reference commit. Record that implementation's SHA and configuration.

The objective is to retain the original dominant-axis ACC pattern that worked well for the owner while automatically correcting false pointer events, missed taps, inaccurate timing and merged double taps. EMG must supply useful physiological corroboration and improve discrimination against FINGER and FIST, not merely measure muscle activity. Strong, mid, soft, fast, double, motion and swipe onsets are priorities. SlowPinch is optional.

**Non-negotiable constraints:**

- NEVER split by user, participant, person, name, or user identifier. Use gesture/subtype-aware **recording-file-level** splits for every development, inner teacher, calibration, contact-model and self-learning fold. Keep exact/near duplicate recordings together; do not split rows from the same recording. No person-based split anywhere in newly executed V11 experiments.
- No PointerState, TriggerPressure, Metronome, filename, folder, User, tap_id, state, label_*, regression or future information in the embedded student's feature matrix. Offline teacher may use future sensor context, but must not use pointer or metadata as a feature. Folder/subtype is supervision/stratification only.
- The manually reviewed archive is a protected seed/reference, not disposable pseudo-labels. Do not assume every zero in a positive file was individually reviewed. Only explicit verified labels are immutable; otherwise preserve source state and audit corrections separately.
- Trusted valid FINGER, WRIST, ARM, BAND and NOTHING files cannot become positives. Keep TAP and SWIPE jointly positive for onset detection. No LP training data in the TAP classifier. A separate contact-quality protocol is permissible only for contact-quality training, not as extra TAP gesture data.
- Discard bad recordings with two or more confidently and persistently disconnected electrodes. Do not convert invalid recordings into useful negative gesture training data. Mask uncertain/transient invalid intervals; allow two good channels.
- No changes to firmware decision architecture, fixed threshold 0.85, current file split protocol, protection objective or size ladder during the label-only comparisons. Current owner-specified ceiling is 100 KB (verify exact byte convention against the local trainer); do not impose the old 14 KiB cap. Use the existing worst-movement protection limit/soft penalty. No separate mandatory runtime veto.
- Run real experiments. Do not claim success without exact results, artifacts, executable tests and reproducibility commands.

## 1. Evidence ledger: what was actually inspected and what it proves

The following source code and historical result files were read directly from the GitHub reference commit above. Historical numeric results are **repository-reported**; they were not freshly rerun in the previous ChatGPT environment. Old studies often used person-disjoint folds. Their observations are useful hypotheses and evidence about failure modes, but **do not repeat that split** or treat their numerical values as comparable with V11 file-split results.

### E1 — Original reviewed labeler: the useful inductive bias and the specific defects

Source: owner's attached `Pasted text.txt`, approximately lines 45–156, 379–541, 576–820, 869–914, 1053–1333, 1531–1554.

Directly inspected:

| Code observation | Implication for V11 |
|---|---|
| ACC PSD bins 1–5 summed per axis; bin 0 excluded. | Preserve the original feature calculation exactly as a baseline. Verify encoding before attempting any power conversion; do not silently change its behavior. |
| Dominant-axis rest ratio >=1.50 and robust score >=6; total rest ratio >=1.20 and robust score >=5; axis dominance >=1.15, score share >=0.38; local prominence >=3; support >=2 samples; width 2–16 samples; dominant-axis fraction >=0.50. | This is a selective short mechanical-impulse pattern, not just EMG amplitude or generic activity. Preserve it initially. |
| `candidate_metrics` uses local flanks, support, half-height width and axis consistency. `choose_candidate` ranks candidate maxima and returns at most one accepted candidate per pointer cycle. | A good baseline candidate descriptor exists. Decouple discovery from pointer cycles; preserve descriptors and compare against real negatives. |
| Search -20/+30 samples around press, release grace 8, minimum search after press 12, between-cycle guard 3, minimum pointer press phase 8. | A missed pointer cycle cannot generate a label; a short pointer pulse can be discarded before any sensor evaluation; one stuck/merged cycle cannot yield two labels. Sample-based timing also changes with logging rate. |
| Rest baseline uses all valid `PointerState == 0` rows. | Missing presses may contaminate the rest estimate; evaluate a pointer-independent robust baseline as an isolated ablation. This is a code-derived risk, not a measured effect yet. |
| `apply_label` expands around the dominant-axis peak and clamps width to 3–13 samples. | The reviewed positives are observable impulse blocks, not fixed full-gesture labels. Preserve them. For recovered events, learn/derive equivalent short blocks and benchmark against fixed 60 ms. |
| `initialize_labels` sets all state=0; rejected/missing-pointer cycles and processing failures can remain zero. | Genuine missed/soft taps may be mislabeled as negatives. Use an explicit unknown state (-1) for unsupported/ambiguous positive-bag candidates, and exclude invalid files rather than silently writing zeros. |
| `drop_previous_generated_columns` removes state, tap_id, regression and label_*; main rewrites CSVs in place, even on error. | Never rerun this destructive workflow on the protected manually reviewed archive. Use sidecars or a separate output directory. |

**Evidence boundary:** The owner reports good real-life finger protection from this dataset and method. The script proves the restrictive pattern and pointer dependency, not a measured real-life FPR. The ZIP contents must be audited on your machine before making claims about the exact coverage or accuracy of its labels.

### E2 — Short sensor-aligned targets beat raw pointer targets

Source: [Claude TAP/SWIPE labeling V2, label design table](https://github.com/alex00Pirotskyi/ML/blob/28efd6e1cb992475f1f8e4caf9b216543b91c779/claude_research_tap/TAP_SWIPE_LABELING_SYSTEM_V2_2026-11-16.md#L92-L113).

Historical 126-feature student, recall of pointer presses at matched negative false-event rates:

| Label design | Recall @0.5 FP/min | Recall @2 FP/min |
|---|---:|---:|
| Sensor onset 0–80 ms, strong presses | 61.4% | 79.0% |
| Sensor onset 0–50 ms | 62.6% | 78.5% |
| Sensor onset +20–100 ms | 64.1% | 79.3% |
| Default, negatives only from negative folders | 63.6% | 79.8% |
| Default, teacher false alarms up-weighted | 58.5% | 75.8% |
| Full extent | 53.0% | 70.8% |
| Pointer-down onset, no cleaning | 53.7% | 73.9% |
| Raw PointerState | 36.2% | 57.4% |

The same report states that sensor timing improved SoftTap recall among confirmed presses at 2 FP/min from 70.5% to 80.5%; SwipeSoft from 62.4% to 71.3%. StrongTap was roughly unchanged. It also reports no benefit from positive-bag background negatives and harm from indiscriminate teacher false-alarm upweighting.

**Design consequence:** Keep short sensor-aligned observable blocks and abstain on disputed regions. Do not treat every rejected pointer or positive-file background as a hard negative. Do not assume a fixed 60 ms block beats the reviewed observable block: test it. These are pointer-reference metrics, not verified physical tap recall, and old models were not necessarily size-constrained.

### E3 — Pure EMG amplitude, generic shape and recurrence are not TAP identity

Source: [V7 pattern discovery](https://github.com/alex00Pirotskyi/ML/blob/28efd6e1cb992475f1f8e4caf9b216543b91c779/research/TAP_PATTERN_DISCOVERY_V7_2026-11-16.md).

- Repository reports 2,921 scanned archives, 2,790 unique recordings and 131 duplicates in that historical snapshot.
- On 109 stratified recordings, median repeated-window cosine similarity was 0.912 for TAP and **0.958 for FINGER**. Repetition is not a valid sufficient tap-confirmation rule.
- Full PSD temporal waveform + morphology reached a pseudo-positive event AUC of 0.861, but TAP-vs-FINGER recording-bag AUC only 0.620. The event target was itself weakly supervised.
- V7's generic energy and morphology alone do not prove that finger impulses are taps. Repetition and rhythm can corroborate but must not manufacture labels.

Source: [V9 aggregate report](https://github.com/alex00Pirotskyi/ML/blob/28efd6e1cb992475f1f8e4caf9b216543b91c779/research/tap_real/results_v9_deep/results_v9/REPORT_FOR_CHATGPT.md) and [machine-readable summary](https://github.com/alex00Pirotskyi/ML/blob/28efd6e1cb992475f1f8e4caf9b216543b91c779/research/tap_real/results_v9_deep/results_v9/SUMMARY_FOR_CHATGPT.json).

- 10,583 successful configurations on 2,790 unique recordings.
- Best reported TAP-vs-FINGER recording AUC 0.774382, target-vs-all-hard-negative AUC 0.778423; best representation used relative waveform, 60 ms before and 640 ms after, MIL1, FINGER weight 3.
- This is **recording-bag AUC**, not event recall, exact timing, firmware FP/min or deployability. Even with large offline context, ambiguity remains.

**Design consequence:** Use the reviewed event seed and negative morphology contrast. Do not copy folder positives to every candidate, use generic activity as confirmation, or launch another 10,000-trial bag-AUC search. Offline teacher success must be validated at event level and by downstream embedded replay.

### E4 — EMG helps distinguish non-target ACC impulses

Source: [TAP generalization research](https://github.com/alex00Pirotskyi/ML/blob/28efd6e1cb992475f1f8e4caf9b216543b91c779/research/gesture/TAP_GENERALIZATION_RESULTS.md#L55-L75).

Reported historical TAP recall at 2 false events/min:

| Feature configuration | Nominal TAP recall | Simulated all-placement stress recall | Stress FP/min |
|---|---:|---:|---:|
| ACC only | 0.597 | 0.597 | 2.00 |
| Raw126 baseline | 0.865 | 0.806 | 2.87 |
| Raw126 + EMG diffs + mild placement augmentation | 0.890 | 0.852 | 1.94 |

ACC-only loses approximately 27 percentage points relative to raw126 at the reported matched false-event rate. Within-channel relative EMG and mild placement augmentation improved robustness in that historical stress simulation. This supports including EMG in both teacher verification and the existing causal student features. It does **not** establish that an arbitrary EMG amplitude gate, a two-teacher ensemble or a particular fusion rule is best.

### E5 — Existing unified labeler has concrete code paths worth fixing

Source: [claude_labeling/label_gestures.py](https://github.com/alex00Pirotskyi/ML/blob/28efd6e1cb992475f1f8e4caf9b216543b91c779/claude_labeling/label_gestures.py).

- L64–70: bounce=80 ms, tap window 0–80 ms, contact thresholds.
- L93–119: person-based selection of fold models. **Do not reuse these person-fold dispatch paths in V11.**
- L126–151: sensor teacher uses offline windows around frames, independent of pointer input. This is a useful architectural precedent.
- L184–242: pointer extraction, `merge_bounces`, one-to-one assignment, 150/400 ms matching, minimum peak distance. Bounces are merged before matching. An independent second sensor event should remain a candidate even when pointer is merged or absent.
- L268–306: pointer-confirmed positives, masks, optional promotion of double partners. The original labeler is not a true independent all-event detector.
- L311–312: stray press masks in negative-folder recordings. V11 should keep valid negative folder labels authoritative.
- L319–326: confidently disconnected rows become 0; V11 instead excludes bad recordings and masks transient invalid intervals.

Source: [existing contact-quality research](https://github.com/alex00Pirotskyi/ML/blob/28efd6e1cb992475f1f8e4caf9b216543b91c779/claude_labeling/GESTURE_LABELING_2026-11-17.md#L95-L128). The historical disconnection rule removed zero confirmed tap events and covered 71.5% of DISCONNECTED and 79.7% of TABLE time, with 0% of good CONNECTED and FINGER/WRIST/ARM/BAND/NOTHING time in the reported study. This supports starting from the existing contact features and sustained OFF rule, not inventing a raw amplitude cutoff. Reproduce using file-based folds; old contact teacher was person-folded.

### E6 — Do not execute the old labeling_v3 benchmark unchanged

Source: [research/labeling_v3/analyze.py](https://github.com/alex00Pirotskyi/ML/blob/28efd6e1cb992475f1f8e4caf9b216543b91c779/research/labeling_v3/analyze.py).

- L26–27: fixed threshold 0.85 but hard-coded 14 KiB budget.
- L220–235: trusted variants use authoritative negative folders, confirmed positive windows and unknown elsewhere.
- L306–335: adaptive interval selection uses inner GroupKFold based on person groups.
- L448–462: GroupShuffleSplit and GroupKFold by person for development/holdout.
- L470–493: constrained selection using FINGER and fist FP/min on normal and stress data.

Reuse safe ideas, not the old split or budget. The newer local pipeline may already implement gesture-aware file folds, observable 60 ms labels, worst-movement penalty and a size ladder. Inspect it first. Owner's newer reported three-seed observable-label gain and 64 KB/100 KB size measurements were not independently accessible from the inspected GitHub main. Keep those local results as baselines if reproducible; do not replace them with old proxy results.

### E7 — What is not verified and must not be used as evidence

ChatGPT's previous execution tools failed when opening the reviewed ZIP and generating the local Markdown. Earlier conversation also contained preliminary claimed ZIP inventory numbers and proxy experiment tables (911 recordings, adaptive/fixed 60 ms, etc.). No underlying successful execution logs or reproducible artifacts were accessible in this handoff. **Do not cite or rely on those numbers. Recompute them if useful.** No V11 experiment, full reviewed-archive audit, or implementation has been completed by ChatGPT. The evidence above consists of source inspection and historical repository-reported results.

## 2. Start here: exact commands

Run in the owner's existing clone or use a clean worktree. Do not reset/overwrite uncommitted work. If the actual latest production trainer exists only in a local branch, use that as the baseline and preserve it; the research branch below points to the inspected historical commit.

```bash
git status --short
git rev-parse HEAD
git branch --show-current
git log -5 --oneline
git fetch origin research/tap-v11-reviewed-seed
# Inspect rather than blindly replacing the current worktree:
git show FETCH_HEAD:research/tap_v11/HANDOFF.md
git worktree add --detach ../ML-v11-research FETCH_HEAD
cd ../ML-v11-research
git rev-parse HEAD
python --version
python -c 'import numpy,pandas,scipy,sklearn,lightgbm; print("deps OK")'
nproc
```

If dependencies are missing, use the owner's current virtual environment or install compatible versions in an isolated environment:

```bash
python -m pip install numpy pandas scipy scikit-learn lightgbm
```

Locate both source inputs locally. The user may need to place the uploads in the working environment; never upload the archives to GitHub.

```bash
find .. -name 'result_TAPANDSWIPE2709_newlbl_mehod123*.zip' -o -name 'Pasted text.txt' -o -name 'DATA_TAP.zip'
# Set paths to the actual local files:
export REVIEWED_ZIP='/actual/path/result_TAPANDSWIPE2709_newlbl_mehod123 (4).zip'
export LEGACY_SCRIPT='/actual/path/Pasted text.txt'
export RAW_ROOT='/actual/path/ML/DATA/DATA_TAP'
export V11_OUT='/actual/path/v11_results'
mkdir -p "$V11_OUT"
sha256sum "$REVIEWED_ZIP" "$LEGACY_SCRIPT" > "$V11_OUT/input_sha256.txt"
unzip -l "$REVIEWED_ZIP" | head -60
find "$RAW_ROOT" -maxdepth 2 -type f | head -20
python -m py_compile "$LEGACY_SCRIPT"
# Locate the real current trainer and observable-label implementation:
find . -name Makefile -o -name '*label*' -o -name '*train*' | head -100
git grep -n -E 'observable|worst.movement|finger.fpm|fist.fpm|MUDRA_TAP|TAP_REFRACTORY_MS' -- '*.py' Makefile | head -150
```

The original script is valid source input; never execute its in-place rewrite against the reviewed data.

## 3. Required implementation: minimal, auditable layers

Implement a separate `research/tap_v11/` package and CLI. Keep it readable and deterministic. Use existing exporter/replay/trainer utilities where safe. All below commands in Section 5 are the **required new CLI contract**, not commands that already exist.

### Layer 0 — inventory, quality and split manifest

- Read CSV and CSV.GZ from raw and reviewed sources, including ZIP members, without modifying either source. Ignore report CSVs, __MACOSX, hidden files. Validate columns, timestamps, monotonicity, cadence, finite values and pointer encoding.
- Inventory folder/subtype counts, seconds, rows, seed positive runs, pointer edges, stuck/absent pointer, label durations, per-file coverage and invalid-contact rates. Report no-pointer cases and disagreements. Preserve reviewed binary state and tap_id. Distinguish explicitly manually marked labels from original auto-labels if metadata exists; if it doesn't, say so.
- Compute exact decompressed hash and canonical sensor-content hash excluding generated labels/diagnostics and identifying metadata. Verify same recording across reviewed/raw sources by sensor rows + timestamp alignment. Never join on filename alone; do not double count reviewed/raw versions. Quarantine conflicting folder labels, mismatched rows and suspected near duplicates until resolved.
- Freeze one gesture/subtype-aware file-level development/holdout manifest before training. Each underlying recording content identity belongs to exactly one fold. Stratify by gesture/subtype; merge sparse strata into parent class transparently or reduce fold count. If a subtype lacks independent exposure, report it as unmeasurable. Do not use User or participant strings for any split.
- Reuse contact feature extraction and sustained OFF logic from E5/E6, but train/validate any new contact teacher using file-aware folds on the separate CONNECTED/DISCONNECTED/LOOSE/TABLE protocol. Do not infer electrode OFF merely from low muscle activity. If protocol data unavailable, explicitly flag an unvalidated quality gate; do not pretend bad files are correctly recognized. Inspect the current local contact pipeline first.
- Establish a sustained confidently >=2 OFF-channel recording-level rejection rule from the contact protocol (time and duration, not number of CSV rows). Report excluded files and exposure per subtype. Mask transient invalid or uncertain regions and exclude rejected recordings from both train and evaluation denominators; report excluded fraction so protection is not inflated by dropping difficult valid files. Preserve valid two-channel data.
- Cache all deterministic extraction keyed by raw hashes, configuration, script/model versions and split hash.

### Layer 1 — legacy seed and mechanical candidates

- Preserve reviewed positive blocks and original legacy parameters. Reproduce the original method read-only on matching raw inputs to quantify agreement; don't force reviewed labels to match the script.
- Separate candidate discovery from pointer cycles. Run original dominant-axis feature extraction across the whole valid recording. Enumerate local maxima/physiological impulses and store shape, rise, decay, width, support, dominance, energy ratio, prominence, axis identity and timestamps.
- Generate a high-recall weak candidate path using relative local ACC changes and EMG/WL rises; also include raw pointer-proposed neighborhoods as candidate proposals. A candidate proposal is not a positive label.
- Preserve multiple distinct impulses in a single pointer cycle. Deduplicate ringing/overlapping proposals by waveform valley, decay, renewed rise and local morphology; do not impose a broad suppression window that destroys real second taps. Compare raw pointer gaps rather than merging blindly.
- Fix timestamp handling in a separate ablation. Estimate true millisecond ranges corresponding to the legacy samples from actual reviewed files. Retain source-row mapping; do not resample away short impulses or create extra apparent evidence by duplicated frames.
- First use legacy rest baseline; separately test robust pointer-independent quiet reference excluding provisional impulses/invalid frames. Handle no reliable baseline by abstention. Do not hardcode cadence assumptions.

### Layer 2 — offline ACC + EMG identity teacher

- Train on reviewed positive event blocks and valid hard-negative candidate windows from FINGER/FIST/WRIST/ARM/BAND/NOTHING. Treat all unverified positive-bag candidate regions as unlabeled, not negative. Include trusted verified negative regions if actually marked.
- Train three comparable arms: ACC-only, ACC+EMG fused LightGBM, and separate ACC/EMG heads with calibrated fusion. Start with the simplest fused teacher; keep two heads only if measured incremental value warrants them. Both modalities are correlated; do not claim statistical independence.
- Offline feature context: multi-scale candidate-aligned ACC PSD/rate/relative axis morphology plus EMG PSD zones, WL/head/tail, relative temporal/spectral differences, per-channel recruitment, EMG-to-ACC timing, rise/decay and contact. Use pre/post context to identify events; no metadata/pointer feature. Use shorter and longer windows as controlled ablations, not a giant search.
- Verify PSD encoding before summing; keep legacy ACC sums unchanged in legacy arm. Relative features must avoid division instabilities and disconnected-channel contamination.
- Balance candidate exposure by recording and gesture/subtype; don't give every teacher false alarm arbitrary extra weight. Calibrate candidate thresholds with file-held-out negative exposure, especially fist and individual fingers. Report candidate precision/recall against reviewed seed only where annotations are trusted; FPM on full valid negative recordings, not just sampled rows.
- A strong contraction or high EMG probability by itself cannot establish a tap. EMG verifies gesture-specific evidence relative to known competing negatives. Do not require an absolute EMG amplitude floor; recover soft taps when the full evidence supports them.
- When teacher and legacy disagree, preserve explicit manually verified decisions and emit an audit record. Do not auto-delete reviewed positives based on teacher confidence.

### Layer 3 — automatic PointerState reconstruction

Construct a sensor event list first. Associate pointer edges one-to-one by calibrated timing/offset and morphology, using a bounded matching algorithm. Pointer is weak evidence only. Preserve unmatched events and classify them into explicit categories:

| Case | Policy on unreviewed data |
|---|---|
| Supported legacy impulse + consistent teacher + matched pointer | Positive, retain legacy observable window |
| Supported legacy impulse + strong calibrated teacher + no pointer | Recover missing tap |
| Weak ACC + strong tap-specific fused evidence + pointer | Soft/weak recovery candidate; accept only at validated specificity |
| Weak ACC + strong EMG without pointer | Most conservative tier; abstain unless validated to add useful recovery |
| Unsupported pointer, no sensor event | Reject pointer hypothesis; unknown locally, not automatically negative |
| Two distinct supported impulses in one pointer cycle | Recover two event IDs and distinct label windows |
| Conflicting timing/identity or ambiguous release/ringing | Unknown; do not manufacture extra events |
| Valid negative-folder event | Always 0, never promote |
| Invalid contact | Exclude recording or interval before labeling |

Emit `pointer_matched`, `recovered_missing`, `recovered_double`, `rejected_pointer`, `soft_recovered`, `ambiguous`, `manual_preserved` and contact status with event IDs and reason codes. Reconstruct event/edge proposals for auditing without overwriting original PointerState. Do not infer exact physical release timing from onset evidence.

### Layer 4 — conservative self-learning, no manual intervention

This is semi-supervised learning from a trusted seed, not fully unsupervised clustering. Generic repetition does not distinguish TAP/FINGER (E3).

- Generation 0: teacher trained on reviewed positives/trusted negatives.
- Generation 1: add only high-confidence recovered candidates, selected with **inner gesture-aware file-disjoint out-of-fold** predictions, calibrated by negative exposure. Keep seeds permanent; cap pseudo-label contribution and record source generation.
- Generation 2: repeat only if it improves held-out evidence at unchanged or better protection. Stop on worsening worst-movement protection, unstable pseudo-labels or no material gain. Test at most two expansion rounds initially. Recovered positives must have sensor evidence, not merely a high probability or an expected gesture count.
- For each outer development fold, all teacher fitting, calibration, self-learning and student training must use only its training files; inner OOF predictions are computed inside that partition. Never train a global teacher/pseudo-label bank across outer validation files and then reuse it in cross-validation. Score outer validation independently. A final reserved file holdout is not used to choose generation, thresholds, label method or model size.
- On unreviewed ambiguous candidates, abstain rather than promote; no forcing at least one positive per positive file.

### Layer 5 — export labels for current trainer

Write sidecar labels in original row order with source hashes and timestamps. Required semantics:

```python
if excluded_recording_or_invalid_interval:
    label = -1       # whole bad recording excluded separately
elif trusted_negative_folder:
    label = 0
elif explicitly_verified_seed_label:
    label = seed_label
elif accepted_event_and_within_observable_window:
    label = 1
else:
    label = -1       # disputed/background positive-bag interval
```

Do not erase original labels. Keep `state_original`, `state_v11`, `tap_id_v11`, `label_source`, `label_reason`, `confidence`, `contact_status`, `Timestamp`. Provide adapter for the current trainer to ignore -1 (assert this; never cast -1 to truthy binary). For manually reviewed positive blocks preserve exact block shape. For recovered events use observable morphology-derived blocks, with a fixed sensor-aligned 60 ms comparison. Explicitly test short double separation; no overlapping positive blocks should collapse event IDs. Keep existing production observable labels as a frozen comparator.

## 4. Experiment design and acceptance

Run controlled paired comparisons on exactly the same file folds, features, augmentation, training seeds, fixed threshold, replay semantics, loss/size ladder and evaluation references. Do not change multiple components at once.

| Arm | Label method |
|---|---|
| B0 | Current frozen production labels and trainer |
| B1 | Original reviewed/legacy labels, read-only |
| A1 | B1 + pointer-independent discovery and FP masking/recovery |
| A2 | A1 + calibrated ACC-only teacher |
| A3 | A2 comparison with fused ACC+EMG teacher |
| A4 | Separate calibrated heads, only if useful |
| A5 | Best above + one self-learning round |
| A6 | Best above + second round, conditional |
| A7 | Winning teacher + independent baseline / timestamp ablations |
| W | Same events, observable block vs fixed sensor-aligned 60 ms |
| S | Include/exclude SlowPinch positives; report separate recall |

Contact exclusion must be a controlled factor: report full baseline and matched valid-file subset; compare all labeling arms on the same subset. B1 may have fewer reviewed files than the full corpus; report matched-seed comparisons separately from expansion experiments. Never compare different exposure as if equivalent.

Use existing file-stratified CV and preferably three seeds if affordable. Teacher may be offline large, student uses current raw126 + relative EMG diffs and mild placement augmentation if supported by the current production trainer. Reuse actual C header export and firmware replay (verify fixed 0.85, actual smoothing/mean/refractory; historical replay uses 150 ms). Do not tune threshold. Measure exact exported bytes and linked size where possible, use owner's current size ladder up to 100 KB. Evaluate normal and simulated placement-shift conditions with identical stress draws. Keep worst-movement protection objective and soft penalty; separately report whether any candidate meets the actual protection limits.

**Required results:**

- File counts, dedup, cadence, quality exclusions, per subtype and exposure.
- Candidate coverage relative to reviewed event annotations; report missing/unsupported/ambiguous events and correction provenance.
- Pointer recovery counts by subtype; these are hypotheses until independently verified. Do not call every teacher-supported orphan a proven FN.
- One-to-one event matches on common references: normal, strong, mid, soft, fast, moving, swipe, double first and second, double-pair recall, SlowPinch separately.
- FINGER, FIST/stretch and every other negative subtype FP/min, worst movement, worst-file and pooled rates; report counts and duration, nominal and stress. Duplicate triggers on positive recordings; onset latency relative to reference, with uncertainty.
- Results on reviewed seed references, noisy pointer references and independent physical references clearly separated. If the reviewed archive has no independent event truth, say so; do not relabel the validation reference to favor the current variant.
- Export bytes, C/Python parity, CPU/RAM where measurable. Report seed-to-seed variation and file-block uncertainty, no person split/bootstrap.
- Best model at equal or better protection, not best F1. SlowPinch is secondary and must not drive selection. If no candidate improves baseline, preserve B0/B1 and report the failure.

Real-life acceptance requires the band. The best offline replay model is a candidate, not a production guarantee. A small independently synchronized current-firmware tap/reference recording set is valuable, but manual annotation of the full database is explicitly not required.

## 5. Deliver runnable commands and actual outputs

Implement these entry points (or documented equivalent exact commands). First run a synthetic/unit smoke and then full-data audit. Use 80% of available CPU threads; keep preprocessing, prediction and training parallelism under that total. Resume cached stages when safe.

```bash
export OMP_NUM_THREADS="$(python -c 'import os; print(max(1,int((os.cpu_count() or 1)*0.8)))')"
python -m pytest -q research/tap_v11/tests
python -u research/tap_v11/run.py audit \
  --reviewed-zip "$REVIEWED_ZIP" --legacy-script "$LEGACY_SCRIPT" \
  --raw-root "$RAW_ROOT" --output "$V11_OUT"
python -u research/tap_v11/run.py prepare \
  --output "$V11_OUT" --folds 5 --seed 42 --threads "$OMP_NUM_THREADS"
python -u research/tap_v11/run.py benchmark \
  --output "$V11_OUT" --seeds 42,43,44 \
  --threads "$OMP_NUM_THREADS"
# Only after development selection:
python -u research/tap_v11/run.py holdout --output "$V11_OUT"
```

The CLI must load frozen input/split manifests and fail if they change. Document the actual flags, contact protocol location, production trainer entry point and available budget. If the current trainer can't import sidecars yet, build a minimal adapter with tests rather than silently changing it.

Required regression tests:

- Exact duplicates across reviewed/raw never cross folds; different users are not used for grouping.
- No pointer present, stuck high, stuck low, false pointer, very short press, two genuine pulses in one merged press; no fabricated second tap from broad single pulse/ringing.
- Soft event not rejected solely by absolute EMG; hard-negative folder never positive.
- Two OFF electrodes excluded; one OFF accepted if two good; transient/uncertain masked; no failed-file zero fallback.
- Immutable reviewed source bytes; sidecar timestamp/row alignment; -1 never enters binary loss.
- Inner self-learning predictions disjoint by recording; outer validation/holdout never used for fitting or selecting.
- One-to-one matching; fixed threshold; exact export budget/parity.

Artifacts (local, not raw data in Git):
`AUDIT.json`, `SPLIT_MANIFEST.json`, `CONTACT_REPORT.json`, `LABEL_CORRECTIONS.csv.gz`, `BENCHMARK.json`, `REPORT.md`, logs, exported candidate headers, test results, exact reproduction commands and source commit hashes. Store sensitive recording identifiers locally; commit only source/tests/docs and sanitized aggregate results.

**Final response to owner:** provide actual measured results, the winning or rejected arms, changes to reviewed-label agreement, recovery counts, finger/fist protection, soft/double/fast recall, model size, and exact commands. Explicitly state remaining evidence gaps and any assumptions. Do not substitute a design document for execution.

## 6. Core principle

Preserve the smallest observable mechanical pattern already present in the reviewed data. Use a sensor-only offline teacher and calibrated EMG corroboration to recover true patterns missed by PointerState, not to invent a new gesture definition. Keep hard-negative protection central, abstain where evidence is insufficient, and prove every new layer with gesture-aware recording-file folds and identical embedded replay.
