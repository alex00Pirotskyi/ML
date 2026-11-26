# Tap-only targets for the wristband

Train one output to fire once at intentional tap contact. A swipe, a long press and a finger flex are competing gestures. Finger identity is a conditional target on a known tap. The existing unified trainer uses a different product definition that accepts both taps and swipes. Its default command reads its own labels. To consume this release, use the loader below in the tap student's data path.

## Use the labels

From the repository root, install `numpy pandas scipy lightgbm`. The compressed snapshot keys are paths relative to `DATA`. It can be used without running the offline teacher. Each decoded frame is in the original row order. The loader verifies the source file hash before returning labels.

```python
from pathlib import Path
import gzip
import pandas as pd
from claude_labeling.tap_snapshot import Snapshot

data = Path("DATA")
s = Snapshot("claude_labeling/tap_labels.jsonl.gz")
with gzip.open("claude_labeling/tap_recordings.txt.gz", "rt") as f:
    keys = f.read().splitlines()  # deduplicated training pool
key = next(k for k in keys if "/TAP/" in k and "SoftTap" in k)
target = s.labels(data, key)
source = pd.read_csv(data / key)
features = [c for c in source if c.startswith("p_core->")]
use = target.tap_label.ne(-1)
X, y = source.loc[use, features], target.loc[use, "tap_label"]
```

Use `tap_recordings.txt.gz` to select one copy of each distinct decompressed CSV across both source trees. In this release all duplicate groups agreed on gesture folder. Recompute the manifest before using new data. Resolve aliases and put every session from the same wearer on the same side of a split. Weight or subsample repeated rows so long recordings and densely sampled events do not dominate. Confirm every selected feature has a causal firmware accessor. The repository's 126 raw features plus within-channel EMG differences and mild placement augmentation are a measured starting point. Re-select features and operating threshold for this tap-only target. [Repository experiment](../research/gesture/TAP_GENERALIZATION_RESULTS.md)

| Field | Meaning |
| --- | --- |
| `tap_label` | 1 in the 0–80 ms onset window, 0 for a supervised non-trigger row, −1 to exclude from loss. |
| `provenance` | `silver` comes from the folder, pointer and sensor teacher rules. `reviewed` has an explicit independent annotation. `unknown` excludes an unresolved row. |
| `finger` | thumb, index, middle, ring, little, or unknown. Compute finger loss only on known positive fingers. |
| `strength` | soft, normal, strong, or unknown. A SoftTap instruction is useful for stratification. Assign actual strength only with independent evidence and a documented criterion. |

The teacher proposes tap timing from pointer alignment, accelerometer spectra and three sEMG channels. The established high-confidence onset rule supplies silver positives. Swipe onsets become hard negatives. Existing uncertainty around hold starts/releases and unexpected pointer activity remains masked. Two suspected disconnected channels mask supervision. A prompted TAP file without any accepted positives is excluded across the entire recording. Quiet contacts in other files can still be missed or mislabeled as background. The independent review overrides all teacher rules, including the contact mask. Handle firmware signal-quality gating separately.

## Release coverage

The snapshot covers 5,370 source paths at commit `28efd6e1`. The deduplicated pool contains 3,072 recordings and 6,623,066 rows. Of those, 151,123 are positive onset rows, 3,785,043 are negative rows and 2,686,900 are excluded. All 357 FINGER and 896 SWIPE recordings contain zero positive tap rows. These counts describe the targets available for training.

Across 276 distinct SoftTap recordings the teacher accepted 2,457 pointer-linked proposals and left 496 unresolved. The latter are listed in [the soft-tap review queue](soft_tap_review_queue.csv.gz) with pointer and proposed onset times for retrieval. Review complete clips blind first on held-out data. The fraction accepted measures teacher coverage of prompted presses. It cannot measure recall of actual soft contacts. [Per-gesture counts](tap_label_counts.json)

## Add independent evidence

```bash
OMP_NUM_THREADS=1 python claude_labeling/tap_only.py DATA labels-tap-only --reviews reviews.csv
python -m claude_labeling.tap_snapshot DATA labels-tap-only reviewed-labels.jsonl.gz
```

Omit `--reviews` to reproduce the automatic targets. The output includes the row sidecars and `teacher_events.csv.gz`, with original teacher scores, tiers and decisions. The event file describes the joint teacher's proposals. Its `labeled` field includes swipes. Use the row target above for the tap-only loss.

Reviews use the following CSV header. Times are integer milliseconds relative to the first source timestamp. Reference IDs should resolve to a retained video frame or contact trace and its clock alignment. Use pseudonymous reviewer IDs.

```csv
recording,start_ms,end_ms,decision,finger,strength,reviewer,reference
```

- **Tap.** Mark the first physical contact. Set start and end equal. The exporter expands the onset to 80 ms and truncates it before the next reviewed interval. Label each contact in a double tap separately. A synchronized contact switch or force trace gives timing, and synchronized video can identify the tapping finger. Confirm intent against the task and observed motion.
- **Negative.** Mark the full interval observed to contain no intended tap. Include swipes, long-press starts and releases, drag/drop, isolated finger flexion, desk knocks, walking and ordinary use while wearing the band. Use unknown for both attributes. The sensor response to an accidental impact belongs in this set when the reference confirms the cause.
- **Unknown.** Mark intervals with occlusion, uncertain intent, ambiguous timing, or unresolved disagreement. If a spurious teacher event was displaced from a verified tap, explicitly review its old interval too. Unreviewed rows retain silver supervision.
- **Soft taps.** Sample random full clips independently of teacher score and also inspect weak, unmatched and sensor-only events from SoftTap trials. A teacher-only review queue misses taps with no peak. For new collection, randomize finger, strength, posture and surface within each wearer. Record multiple dons. Define physical strength bins in advance on a calibrated reference, retain raw force and units, and use unknown when force is unavailable.
- **Adjudication.** Two people independently annotate held-out sessions blind to the model score. Keep original annotations and evidence, reconcile event count, timing, finger and strength, and export one consensus file. Measure onset differences and finger agreement before adjudication. Keep the final test wearers and negative streams out of mining and threshold selection.

## Measure device behavior

Replay continuous recordings through the compiled model, feature warmup, quantization, threshold, and refractory logic used on the device. Log each emitted event as `recording,fire_ms,finger` (finger optional). Use the same relative clock for references and emissions. Run:

```bash
python claude_labeling/score_tap_reviews.py heldout-reviews.csv device-fires.csv
```

The scorer assigns at most one detection to each reviewed tap. The defaults match the repository replay tolerance (100 ms early, 250 ms late). Record tolerances before tuning. The report includes soft and per-finger recall, true/predicted finger counts, correct-finger detections per actual tap, latency, extra fires around taps, and false events per minute of reviewed negative exposure by gesture folder. A missing denominator means the metric is unmeasured. Count reviewed minutes and people alongside rates. Compare candidate models at the same validation false-trigger budget and evaluate once on held-out people. Include onset and release of other gestures as their own reviewed challenges. Long presses and swipes may initially resemble taps, so inspect the latency cost of waiting for more evidence.

A best-finger comparison needs within-wearer trials of each finger at matched strength, placement and activity. Rank fingers using held-out per-finger results with participant-level uncertainty. The released recordings in FINGER exercise distractors. They cannot establish which finger produced a successful tap. The snapshot assigns unknown to physical finger and strength throughout.

## Research basis

TapID evaluated touch-event detection and finger identification separately and measured performance across people. Its wrist sensor configuration and seated task differ from this device. [TapID, IEEE VR 2021](https://static.siplab.org/papers/vr2021-TapID.pdf)

TapWristband examined how finger tapping transmits vibrations to wrist sensors. It motivates testing across surfaces and postures with the actual band. [TapWristband, IEEE Transactions on Mobile Computing 2025](https://ieeexplore.ieee.org/document/10759301/)

The wrist-sEMG neuromotor interface study used multi-participant data and evaluated closed-loop input. It motivates independent wearer splits and testing real device emissions. [Nature 2025](https://www.nature.com/articles/s41586-025-09255-w)

The repository's own ablations found worse tap recall for accelerometer-only re-anchoring, and the feature-selection study measured extra false events from promoting weak teacher scores. Those experiments used joint tap/swipe targets. This release preserves the existing timing teacher and changes the target semantics. [Generalization experiments](../research/gesture/TAP_GENERALIZATION_RESULTS.md) · [Feature-selection experiments](../research/gesture/FEATURE_SELECTION_RESULTS.md)
