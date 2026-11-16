# LP State V4 — strict, auditable labeling candidate

## Decision and scope

This is a **candidate**, not an approved physical ground-truth labeling system. It deliberately focuses on **lp_state**. Unlike V3, it does not fit a GMM, estimate contact probability from trained classifiers, or create nine channel-score targets. It operates only on:

- DATA/DATA_LONG_PRESS/{SOFT_LONG_PRESS,MID_LONG_PRESS,STRONG_LONG_PRESS,LONG_PRESS}
- Additional quiet-rest reference controls from CONNECTED and NOTHING

TAP, SWIPE, DRAG_AND_DROP, TABLE, LOOSE, and DISCONNECTED are **not** treated as trustworthy long-press state labels. This is intentional; exclude rather than contaminate the primary LP training dataset.

**Input and output are separate.** Originals, V2/V3 labelers and the successful v20 model must remain unchanged.

## Label contract — IMPORTANT breaking change relative to V3

| Output | Meaning | Train? |
|---|---|---|
| lp_state = 1 | Stable pointer-off **AND** sufficiently quiet measured EMG relative to a locally observed quiet reference | Yes, tentative rest/release |
| lp_state = 0 | Stable pointer-on **AND** sufficiently elevated WL and logged PSD in at least one usable EMG channel | Yes, tentative active/hold |
| lp_state = -1 | Any other sample (timing boundary, uncertain event, poor contact, missing features, mismatched signals) | **NO. EXCLUDE** |
| lp_state_weight = 1 | Candidate accepted by the strict rule | Yes |
| lp_state_weight = 0 | Unknown/ignored | **NO** |

The original binary V3 benchmarking/training scripts are **not compatible as-is** with -1. Never cast these labels directly to uint8 and never map -1 to 0. A training-loader change must explicitly select rows with \`lp_state_weight > 0\` and \`lp_state >= 0\`. Preserve 1=resting semantics.

**No "98% accuracy" claim is made.** PointerState is from an imperfect input device (Oculus/camera/trackpad), not a calibrated muscle-force reference. Agreement can improve precision, but the two signals may both be wrong about physical intent.

## Deterministic labeling rules

1. Use recorded monotonic/sample timestamps. If missing or invalid, keep all labels UNKNOWN. Exclude clock gaps.
2. Filter one-frame pointer glitches with a short median filter. Exclude 250 ms around each transition and sample-integrity anomalies.
3. Require a usable pointer-off region of at least 700 ms for local per-channel reference levels. No valid off baseline: no labels.
4. Reject flat/implausible channel amplitudes, missing PSD zones, invalid/overflow samples, and files with fewer than two usable channels.
5. Compute a ~90 ms robust WL/PSD median. The PSD is **already log10-encoded by firmware**; use differences, not linear power ratios. Respect explicit \`psd_mean_applied\` conversion flags.
6. Active support: WL relative to reference >=1.55 and logged-PSD increase >=0.018, continuously >=180 ms in any usable channel.
7. Quiet support: WL relative to reference <=1.28 and logged-PSD increase <=0.028, continuously >=180 ms across all usable channels.
8. A candidate LP pointer event must contain at least 180 ms of active-supporting signal and >=18% support in its stable interior. Otherwise do not train on that event. If no LP event is supported, label the **whole file UNKNOWN**.
9. Require rest agreement with pointer-off and active agreement with pointer-on. Ambiguous/mismatched rows remain -1.
10. Generate a file-level per-group audit and a manual review queue. Thresholds are **hypotheses**, not tuned or validated on real physical truth.

The minimum two-usable-channel rule avoids good-looking labels on badly placed bands, but it may exclude valid soft holds and cannot certify actual electrode contact. This trade-off must be quantified on real recordings.

## Usage

From repo root:

\`\`\`bash
python -m pip install numpy pandas scipy
python -m unittest tests/test_lp_state_v4.py
python lp_state_v4.py --dry-run --limit-per-group 2 --report /tmp/lp_state_v4_pilot.json
python lp_state_v4.py --dry-run --report /tmp/lp_state_v4_full_audit.json
python lp_state_v4.py --report /tmp/lp_state_v4_written_audit.json
\`\`\`

Default output: \`DATA/DATA_LONG_PRESS_LP_STATE_V4\`. This is a different directory from V2/V3. Existing outputs cause a failure, rather than silently mixing labeler revisions; regenerate with \`--overwrite\` only when the full experiment needs replacing.

**Avoid mass-training before reviewing the audit.** The audit reports samples per class by folder, accepted LP events, unusable data, and files needing review.

## Acceptance gates before production

1. **Real-data coverage:** run on every eligible LP/control source file; inspect file-level coverage and return reasons. Review at least 20 recordings across soft/mid/strong and multiple users; include excluded files.
2. **Event-level timing:** use independently synchronized physical button/force truth on a representative, person-held-out test set. Measure premature-release-in-hold, missed true release, release delay, missed soft holds, and labels assigned to physically disconnected channels. Keep manual uncertainty around transitions.
3. **Generalization:** hold out both participants and acquisition campaigns. Do not tune threshold decisions on the evaluation set.
4. **Embedded relevance:** train a *causal*, compact student with explicit -1 exclusion. Evaluate *online after the separate tap model activates LP* and with the actual firmware 10-prediction averaging/release logic.
5. **Baseline:** preserve the successful v20 model/export/settings. Only promote V4 if it shows better sustained-hold reliability AND no degradation in eventual release, while remaining within nRF52832 CPU/RAM limits.
6. **Reproducibility:** pin input hashes, labeler commit, thresholds, sampling mode and training code in a manifest before publishing a production dataset.

## Known limitations

- V4 is intentionally conservative and may have much lower coverage than V3.
- LP labels are contingent on pointer quality and relative EMG change; very light, steady effort with minimal signal variation can remain unknown.
- The per-file quiet baseline assumes that some pointer-off samples are genuinely relaxed. It is **not** independent physiological truth, and a biased off baseline can hurt labels.
- If the input pointer reference is systematically offset by more than 250 ms, boundary masking is insufficient.
- Offline median filtering uses neighboring samples; the student must never rely on future features.
- No independent physical-validation dataset was available for this implementation. Passing synthetic unit tests does not imply real-world accuracy.

## Next step

Run the full audit on GitHub's real dataset, review the report and waveforms, then compare the candidate *on held-out real events* to V3 and v20. **No production sign-off until the physical truth checks pass.**
