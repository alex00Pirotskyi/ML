# LP State V4 / V4.1 — real-recording audit (2026-11-16)

**Scope:** \`DATA/DATA_LONG_PRESS\`, all 101 available LP recordings across SOFT_LONG_PRESS, MID_LONG_PRESS, STRONG_LONG_PRESS, LONG_PRESS, plus 39 CONNECTED and 5 NOTHING controls. **Only these 145 of the repository's 2,431 long-press-tree files** were evaluated by this deliberately narrow labeler.

**Status:** Experimental; **NOT approved for production training**. All 145 selected files processed with zero errors. The experimental model was evaluated by GitHub Actions on the actual compressed CSVs, not estimated from synthetic data.

- Original V4 audit: https://github.com/alex00Pirotskyi/ML/actions/runs/43858086782
- V4 threshold diagnostics: https://github.com/alex00Pirotskyi/ML/actions/runs/43858456058
- Contact-control audit: https://github.com/alex00Pirotskyi/ML/actions/runs/43858721611
- Revised V4.1 audited run (successful): https://github.com/alex00Pirotskyi/ML/actions/runs/43859161508
- Revision commit: \`b50cb4a5edd30510bc610601b2f0d2ec00c55c2b\`; teacher version: \`lp_state_strict_v4_1_candidate\`
- Draft PR: https://github.com/alex00Pirotskyi/ML/pull/2

## Measured coverage, not accuracy

The CSVs contain no independently synchronized physical press/release force switch; recorded PointerState is noisy and time-offset. As V4 uses pointer context for accepting labels, agreement with PointerState is **tautological** on accepted samples and must **never** be described as accuracy or precision.

| LP class | Files | Original V4 labeled samples | Revised V4.1 labeled samples | Original V4 accepted files | Revised V4.1 accepted files |
|---|---:|---:|---:|---:|---:|
| SOFT_LONG_PRESS | 36 | 8.446% | **26.053%** | 11 | **26** |
| MID_LONG_PRESS | 41 | 22.747% | **43.570%** | 19 | **33** |
| STRONG_LONG_PRESS | 20 | 30.196% | **55.179%** | 11 | **17** |
| LONG_PRESS | 4 | 40.328% | **51.065%** | 3 | **4** |
| **All LP** | **101** | **20.307%** | **40.542%** | **44** | **80** |

Actual 707,860 long-press source rows, V4.1:
- **244,866** \`lp_state=0\` (supported active/held); **34.592%**.
- **42,114** \`lp_state=1\` (supported rest/released); **5.949%**.
- **420,880** \`lp_state=-1\` (UNKNOWN and must be discarded); **59.458%**.
- **576/898** pointer-reported hold events have sufficient independent EMG support (**64.14% event-support coverage, NOT recall**).
- **80/101** files have at least one accepted active label; **77/101** have at least one accepted rest label.
- Of 21 files with no accepted labels, 12 have insufficient stable off-reference time and 9 have no verified LP event under the rules.

V4.1 quiet-control audit, 75,483 total rows: 43/44 control recordings contain accepted rest; 51,068 rest rows, 24,415 unknown, **zero active**. This is a useful sanity check, NOT validated resting specificity.

## Critical bug discovered using real data: PSD validity gate had its sign backwards

Original code required \`floor_psd >= 0.02\`, on the assumption that quiet PSD was suspicious. Actual CONNECTED control recordings often have much lower values, whereas disconnected CH1/CH2 often show **much higher** PSD due to artifacts.

We audited 159 nominal-good channel recordings (39 CONNECTED x3 + DISCONNECTED CH3 x42) and 84 nominal-bad channel recordings (DISCONNECTED CH1/2 x42; CH3 excluded from bad labels because it was physically connected in those acquisitions).

| Channel pass fraction | Original gate (WL 65..3500, PSD 0.02..5.5) | V4.1 gate (WL 65..3500, PSD <=0.60) |
|---|---:|---:|
| Nominal-good controls | 57.862% | **94.969%** |
| Nominal-bad controls | 25.000% | **26.190%** |

Removing the erroneous **PSD minimum** substantially improved preservation of nominal-good contacts without materially changing nominal-bad admissions. The high-PSD cap \`<=0.60\` remains a heuristic, **not** a validated impedance classifier. A lower upper cap, e.g. 0.30, passed about 89.94% of nominal-good channels and 16.67% of nominal-bad channels in this SAME control set; do **not** tune on it further and call that independent validation.

Connected/disconnected are weak folder-level proxies, not physical electrode measurements. In particular, V4.1 **still admits ~26%** of nominally-disconnected channel records through the per-channel validity gate, so channel quality remains a material production risk.

## Boundary sensitivity study (original V4 gate)

Changing exclusion around pointer transitions from 250 ms to 120 ms and minimum active evidence from 180 ms to 120 ms raised original soft LP coverage from 8.45% to about 10.99%, still inadequate. Shorter transition guards increase mislabeled-time risk given the known VR pointer offsets. We did **not** adopt the shorter guard. The major improvement came from correcting the unjustified fixed PSD minimum, not loosening the temporal safety margin.

## Testing

- 12/12 synthetic safety/contract tests pass under V4.1 after making their PSD fixture represent realistic nominal-connected levels.
- Full real-data V4.1 audit: 145/145 recordings processed, zero processing errors.
- Reproducible \`lp_state\` audit, boundary sensitivity and contact-control JSON artifacts are stored in the successful GitHub Actions run for 30 days.
- Read-only processing; originals, V2/V3, and the successful v20 exported model are unchanged.
- The output contract remains \`1=rest\`, \`0=active\`, \`-1=unknown\`. Existing V3 binary \`uint8\` loaders **must not** consume V4.1 files without explicitly excluding \`-1\`.

## Decision

**Reject V4.1 as a production-grade source of "correct" LP state labels for now.** It supplies a more useful, reproducible *candidate* training subset, but 59.46% LP-row abstention, 21 completely excluded LP files, weak/disconnected-quality ambiguity, and especially missing soft-press event support are not solved by more heuristic tuning.

Next scientifically valid gate: manually or physically verify a representative held-out set of press/hold/release events, measure false-rest-while-held, time-to-correct-release, and hold continuity, then compare a causal embedded student with the frozen successful v20 baseline. The data folder does not contain independent physical press/release ground truth, so any purported "95% accuracy" before that is fictitious.

### Reproduction

\`\`\`bash
git fetch origin research/lp-state-strict-v4
git switch research/lp-state-strict-v4
python -m pip install numpy pandas scipy
python -m unittest discover -s tests -p "test_lp_state_v4.py" -v
python scripts/lp_state_v4_real_audit.py --output /tmp/lp_state_v4_real_audit.json
python scripts/lp_state_v4_diagnostics.py
python scripts/lp_state_v4_contact_controls.py
\`\`\`
