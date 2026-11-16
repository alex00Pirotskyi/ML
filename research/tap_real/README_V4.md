# Mudra Tap Labeler V4 — fail-closed labeling pipeline

**Status:** export/integrity mechanics tested. **Not an independently validated automatic tap-labeling oracle.** A physical timestamp reference is still missing for the existing V3 candidate dataset. The phrase *production-safe export* means `tap_label=1` cannot be created by a folder name, a PSD peak, Oculus PointerState, or a metronome alone.

## Why V4 exists

V1/V2 accepted thousands of spurious EMG activations/hour in hard negatives. V3's physiological event-shape/refractory state improved double-tap segmentation, but its out-of-person TAP-vs-FINGER recording-level AUC remained poor (0.517 for the two-stage model; a tuned seven-class approach reached only about 0.620). V3 produced **349 unverified double-tap candidate pairs** in 118 unique DoubleTaps recordings; it did **not** establish physical ground truth. V4 deliberately avoids turning V3's candidate flags into production positive labels.

## Architecture

1. Read **only** a directory actually named `DATA_TAP`, with category folders `TAP`, `FINGER`, `SWIPE`, `WRIST`, `ARM`, `BAND`, `NOTHING`. `class_id` is **not** a trusted source label (in existing data some SWIPE files have `class_id=1`, also seen in TAP).
2. Deduplicate by SHA-256 of **decompressed** source CSV bytes, not basename or gzip wrapper. Check source stat before/after each file. Read only supported EMG PSD, ACC PSD, WL, Timestamp, PointerState columns.
3. Generate V3 candidate episodes with physiological refractory/re-arming, independent of Oculus/Metronome and filename. Preserve `tap_candidate_id`, `tap_candidate_pair_id`, `tap_candidate_score`, and detailed `.events.csv`.
4. Detect **optional** stable one-to-one Oculus/EMG correspondence, split-half lag checked. This is a **silver review cue**, never positive training truth. Add `POINTER_ONLY_UNVERIFIED` review opportunities where V3 missed a possible soft event. Record uncertainty and rejection reasons.
5. For **TAP** files, default every unreviewed sample to `UNKNOWN = -1`, weight 0. Do not assume all peaks are true, or that all remaining frames are safe negatives.
6. For **non-TAP** instructed gesture recordings, label known-class negatives `0` with reduced *weak category* weight 0.70; abstain over timestamp gaps and warmup. These are not independently verified zero-contact moments, so their provenance says `RECORDED_CATEGORY_NEGATIVE`.
7. For physically verified TAP onset annotations, emit `1` only with **two distinct synchronized independent reviewers** agreeing within 45 ms, or a directly synchronized physical contact sensor with ≤10 ms verified clock uncertainty. At most 25 ms synchronization uncertainty for all other physical sources. Never accept `PointerState`, `TriggerPressure`, metronome, file class, or V3 pseudo-confidence as a physical ground-truth evidence type.
8. Gold annotation time is **relative to the first `Timestamp` row**, in milliseconds, and tied to the exact source SHA-256. Even when V3 missed a very soft tap, a valid independent onset can create a positive window (the quality gate still checks usable EMG). Each verified event gets a separate numeric `tap_event_id`; adjacent gold onsets at 80–500 ms are paired with `tap_pair_id` without collapsing either event.
9. If a recording has no verified onsets, it produces **zero positive labels**. An entirely unannotated TAP session stays UNKNOWN, including its quiet-looking intervals. All incomplete, stale, out-of-range, incorrectly synchronized and conflicting annotations are rejected or abstained; unresolved cases are listed in the manifest.
10. Write to a separate, new, versioned output directory. Final results are atomically renamed into place; errors clean the staged output. Any positive without independent annotation is a fatal invariant violation.

## Installation

```bash
python -m pip install numpy pandas scipy
```

## Run the candidate-only audit first

```bash
python research/tap_real/production_labeler_v4.py \
  --input-dir DATA/DATA_TAP \
  --out-dir DATA/DATA_TAP_V4_CANDIDATES \
  --policy candidate
```

Read `review_queue.csv` (including soft / ambiguous TAPs, HARD_FINGER cases, and some pointer-only misses), `.events.csv`, and `manifest.jsonl`. Select a **separate synchronized physical reference** for onset review (e.g. calibrated video of the hand or an independently synchronized physical switch). Raw sEMG shape itself is useful for finding candidates but **not** an independent check that the event is truly a tap.

## Annotation format

The script creates `annotations_template.csv` with these required fields:

```csv
source,source_sha256,onset_rel_ms,decision,reviewer_id,evidence_type,sync_error_ms,notes
TAP/example.csv.gz,<exact-sha256-from-manifest>,1540,TAP,reviewer_a,synchronized_video,10,physical start of first tap
TAP/example.csv.gz,<exact-sha256-from-manifest>,1551,TAP,reviewer_b,synchronized_video,10,independently reviewed
TAP/example.csv.gz,<exact-sha256-from-manifest>,1790,TAP,reviewer_a,synchronized_video,10,physical start of second tap
TAP/example.csv.gz,<exact-sha256-from-manifest>,1798,TAP,reviewer_b,synchronized_video,10,independently reviewed
```

Illustration only; **there is no known real `example.csv.gz` or verified timing in the dataset above**. Replace file path, hash, time and reviewer evidence with actual observations. Onsets mean **start of the intended tap movement**, not a delayed Oculus pointer transition. The evidence clocks must be independently synchronized to each CSV.

Allowed `decision`: `TAP`, `NOT_TAP`, `UNCERTAIN`. Allowed positive `evidence_type`: `synchronized_video`, `synchronized_kinematics`, `physical_contact_sensor`. A physical contact sensor may be accepted with one instrument record only when `sync_error_ms` ≤10 ms; video or kinematic evidence requires two independent reviewers. Two observers copying the same machine-generated timestamp are **not** independent validation.

## Produce verified-only labels

```bash
python research/tap_real/production_labeler_v4.py \
  --input-dir DATA/DATA_TAP \
  --out-dir DATA/DATA_TAP_V4_GOLD_RELEASE_001 \
  --annotations DATA/approved_tap_annotations.csv \
  --policy production
```

If zero independently verified usable tap onsets are available, the production export exits with an error and **publishes no output directory**. There is no option to silently relabel silver or V3 candidates as gold.

### Output semantics

| Field | Meaning |
|---|---|
| `tap_label` | `1`: independently verified physical-onset core; `0`: instructed non-TAP negative; `-1`: unknown / excluded |
| `tap_label_weight` | Gold positive = 1.0, weak instructed category negative = 0.70, unknown = 0 |
| `tap_event_id` | Positive event ordinal; **-1 everywhere else** |
| `tap_pair_id` | Two independently verified physical taps within 80–500 ms, else 0 |
| `tap_provenance` | `VERIFIED_GOLD`, `RECORDED_CATEGORY_NEGATIVE`, `UNKNOWN`, `SIGNAL_UNAVAILABLE` or related explicit state |
| `tap_candidate_id`, `tap_candidate_pair_id` | Candidate IDs from V3, **not physical truth** |
| `tap_candidate_score` | V3 shape heuristic, not a calibrated posterior probability |
| `tap_silver_proposal` | Oculus/EMG support flag for reviewer attention only; **never a gold label** |

### Strict release gate

1. Zero bad hashes, duplicate annotation keys, source mutations, inconsistent annotation timing, failed recordings, or incomplete sidecars.
2. In the output, **every `tap_label=1` maps to accepted independent physical evidence**. A gold onset need not have a V3 candidate (prevent soft-tap false exclusions).
3. Verify each double tap has two distinct positive IDs, and that single broad activations never create fake doubles.
4. Before training a **production** tap classifier, obtain a **separate held-out, physically annotated evaluation set** with soft-soft, strong-soft, one-tap, short-interval doubles, poor contact, FINGER-only, SWIPE, BAND, ARM, WRIST, NOTHING and unstructured daily activity. Measure **true event precision, recall, onset error and false activations per hour**, per subject and cohort, and compare with the archived V3/v20 baselines where appropriate. No currently available automatic method has passed this gate.
5. Freeze alias maps and deduplicated subject/session splits before measuring reported accuracy. Filename and person are never ML features.

## Tests and reproducibility

```bash
python research/tap_real/test_production_labeler_v4.py
```

Tests cover annotation review agreement, physical evidence allowlisting, missing annotations, stale source hash, path traversal, conflicting decisions, out-of-bounds times, strict non-TAP labels, double IDs, zero-leak invariants, and atomic rollback.

**Observed pilot (154 sampled source recordings):** strict silver agreement supported only 2/75 TAP recordings (17 supporting events) and 0/27 SWIPE recordings. Relaxing multiple thresholds did not materially improve coverage. This is a **proxy**, not independent tap precision; the silver gate is therefore intentionally review-only. Full-corpus labels and further audit counts are generated by the GitHub CI workflow.

## External research context

The 2024 CFAR/TKEO EMG onset study used *raw* sEMG and independent expert/kinematic reference events (https://pmc.ncbi.nlm.nih.gov/articles/PMC10885915/). It cannot be transplanted as a performance claim for Mudra's log-coded PSD feature frames. Cross-user EMG gesture work illustrates the difficulty of rejecting unseen gestures despite high classifier confidence (https://pmc.ncbi.nlm.nih.gov/articles/PMC11459555/). Hence explicit unknown labels, cross-participant validation, and independent timing evidence.