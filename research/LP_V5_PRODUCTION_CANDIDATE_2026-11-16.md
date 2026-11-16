# LP V5 production-candidate research audit — 2026-11-16

> **Release status: EXPERIMENTAL / NOT YET APPROVED FOR PRODUCT.**
> This is the strongest *current label-generation candidate*, not verified physical ground truth. Preserve the original recordings, earlier V2/V3/V4, and the successful v20 firmware model.

## Scope and reproducibility

- Repository: `alex00Pirotskyi/ML`.
- Implementation: `lp_state_v5.py`; safe exporter: `lp_v5_labeler.py`.
- PSD/contact preprocessing: `scripts/lp_v3_feature_probe.py` (correct historical `psd_mean_applied` in the *log-encoded* PSD).
- Causal LightGBM student and nine-head benchmark: `scripts/lp_v5_student_benchmark.py`.
- Tests: `tests/test_lp_state_v5.py`, `tests/test_lp_v5_export_quality.py`.
- [Successful full GitHub Actions real-data run](https://github.com/alex00Pirotskyi/ML/actions/runs/43896735761), including reproducible, downloadable per-file audit and student JSON.
- All 14 synthetic/state/export/PSD tests passed; **238/238** real source recordings processed, zero errors.
- Evaluated **101 LP** + **44 quiet-reference** + **93 contact-stress** recordings; 9 source groups in `DATA/DATA_LONG_PRESS`.
- `--dry-run` emitted no labeled files and originals were unchanged. The optional full exporter writes to `DATA/DATA_LONG_PRESS_LABELED_V5` only, atomic per-file, and rejects stale existing outputs unless explicitly `--overwrite`.

### Product constraints explicitly addressed

1. **Partial electrode contact**: score CH1/CH2/CH3 independently and allow a single reliable channel with sustained repeatable EMG events to support LP state. Do *not* require all channels to connect or contract.
2. **Noisy pointer**: `PointerState`, `class_id`, `TriggerPressure` are not fed to the teacher's event decisions, not used as labels, and not included in the causal student's features. Synthetic FP/FN/delay/removal tests demonstrate invariance.
3. **Sustained holds and eventual release**: prefer repeated activity/rest episodes on one trusted channel (default >=3 within-file episodes, each >=350 ms), and retain uncertain transitions as `lp_state=-1`.
4. **Contact artifacts**: use person-held-out weak contact-quality probabilities and recording-specific WL/PSD physics checks; respond conservatively to clipping, flatlines, invalid timestamps and saturated PSD.
5. **Firmware**: V5 offline teacher uses future recording context for label calibration and **cannot be deployed directly on nRF52832**. Causal student must be tested with the separate tap initiation model and actual 10-prediction averaging/hysteresis and release logic.

## Output label contract

- `lp_state=1`: channel-supported quiet/rest; `lp_state=0`: channel-supported muscle activity; `lp_state=-1`: UNKNOWN (**never cast to uint8 or use as a class**).
- `lp_state_weight`: global evidence weight: 1 for accepted LP events/quiet, 0 for unknown; known quiet-control rest is weakly supervised (weight 0.55).
- Nine requested main targets:
  - `lp_resting_ch1..3`: quiet evidence on each accepted channel.
  - `lp_high_activity_ch1..3`: contraction-like evidence on each accepted channel.
  - `lp_signal_quality_ch1..3`: **usable-signal/contact *proxy*** based on quality model and data validity, **not measured impedance and not a calibrated probability of actual contact**.
- Additional safety/provenance: `lp_unknown_ch1..3`, `lp_target_weight_ch1..3`, `lp_chN_event_supported`, `lp_teacher_version`. Rest/high values are not calibrated probabilities of physical muscle force.
- For global binary student training, include only `lp_state>=0` and `lp_state_weight>0`. For **each** resting/high activity regressor, select **that channel's own** `lp_target_weight_chN>=0.8` and ignore zero values on unknown rows. Quality regression must be evaluated separately against *independent measured electrode quality* before product sign-off.

## GitHub CI: actual recording coverage

These are **coverage / accepted evidence**, not measured accuracy, recall or physical false-positive rates:

| Folder | Recordings | Rows | Accepted rest+active rows | Coverage | Files with activity | Accepted single-channel files |
|---|---:|---:|---:|---:|---:|---:|
| SOFT_LONG_PRESS | 36 | 230,836 | 88,901 | **38.513%** | 23 | 12 |
| MID_LONG_PRESS | 41 | 303,506 | 162,262 | **53.463%** | 33 | 9 |
| STRONG_LONG_PRESS | 20 | 145,736 | 70,765 | **48.557%** | 17 | 5 |
| LONG_PRESS | 4 | 27,782 | 19,501 | **70.193%** | 4 | 1 |
| **All LP** | **101** | **707,860** | **341,429** | **48.235%** | **77** | **27** |

Of all LP rows: 230,072 accepted active, 111,357 accepted rest, 366,431 unknown. (The sums must be understood as accepted EMG evidence, not switch truth.) If future runs differ, reference the attached CI JSON rather than copying this table.

Quiet reference recordings:
- CONNECTED: 38,149/54,012 weak-rest rows, zero active *exported*.
- NOTHING: 13,626/21,471 weak-rest rows, zero active *exported*.
- Contact-stress `DISCONNECTED`, `LOOSE`, `TABLE`: **all `lp_state=-1` by policy**, so their zero exported active labels are NOT evidence of physical specificity.
- Independently re-running the detector on controls *without* the folder veto: 0 active in DISCONNECTED, 0.824% of LOOSE rows and 0.945% of NOTHING rows labeled active-like. None is a verified error rate; real activity and CH3 connected cases may occur.

Quality proxy held-out-person calibration (201 weak contact labels, 14 people; no measured impedance):
- ROC AUC **0.9605**, balanced accuracy **89.19%**, confusion nominal-good/nominal-bad [[114,3],[16,68]].
- CH3 of DISCONNECTED sessions is **not** treated as physically disconnected; it often remained connected.
- Results are exploratory because both folder-level truth and LP-responsive synthetic-good augmentation are weak.

## Causal student imitation benchmark — participant-held-out

GitHub Actions run: [full reproducible benchmark](https://github.com/alex00Pirotskyi/ML/actions/runs/43896735761).

- **238** recordings, 38,080 sampled windows, three `GroupKFold` folds by normalized participant identifier, **13,203 accepted global labels** evaluated.
- Only current and past WL, firmware log-PSD, log-PSD slot0, WL head/tail and 0.25/1/3-second past-only EMA features were provided to the student. No PointerState/folder/user/teacher-confidence inputs.
- Compact 48-feature, 60-tree LightGBM: balanced accuracy **88.39%**, resting F1 **0.8898**, ROC AUC **0.9521** **against selective V5 pseudo-labels**. This is not physical hold/release accuracy. Mean exported *LightGBM text* size around **65,907 bytes**, which does **not** equal TFLite/micro/firmware Flash/RAM/CPU usage.
- 120-tree compact variation gave similar balanced accuracy, indicating that simply adding trees is unlikely to solve physical accuracy.

### Nine continuous outputs: R² *to selected V5 teacher labels*

| Target | CH1 | CH2 | CH3 |
|---|---:|---:|---:|
| Rest (only channel-confident windows) | 0.807 | 0.804 | 0.753 |
| High activity (only channel-confident windows) | 0.794 | 0.790 | 0.781 |
| Signal-quality proxy (all windows) | 0.786 | 0.748 | 0.508 |

- Rest and high activity were evaluated on 4,981 CH1, 6,947 CH2, 3,607 CH3 confident sampled rows. Signal-quality imitation on all 38,080 sampled rows.
- **Crucial rule**: an unknown target's stored zero is missing supervision, NOT evidence that the muscle is resting/active at level zero. Mixing unknown into regression massively biases R². Use per-channel masks.
- The nine targets are heuristic/derived and have different distributions from V3. These R² values do **not** show that V5 is physically better than V3, nor that its 9 continuous outputs are calibrated.
- Firmware input parity, actual compiled model size/latency and output smoothing remain unverified.

## Remaining blockers before a shipped-product release

1. **Independent truth capture**: log real synchronized press/hold/release using a physical switch or force/touch measurement and actual electrode lift/contact per channel (including CH3). Include disconnected while gripping, quiet rest, weak/steady holds, movement artifacts, and varying skin/electrode placement.
2. **Hold continuity**: score premature-rest **during a verified hold**, eventual correct release and time-to-release **after verified release** at event level (not rowwise pseudo-label agreement). Include tap gating and real 10-prediction averaging on the exact firmware.
3. **Participant and campaign holdout**: freeze the V5 teacher and thresholds before testing on new people, devices, capture dates, electrode placement and motion; report confidence intervals, event coverage and abstention. Avoid using the same control folders to both tune and judge actual contact errors.
4. **Quality calibration**: measure whether `lp_signal_quality` tracks actual electrode contact/usable SNR, including quiet healthy channels, partially connected bands, disconnected CH1/2 while CH3 stays connected, electrode motion, and fatigue.
5. **Embedded implementation**: export the compact student in the actual inference runtime; measure nRF52832 Flash, RAM, CPU %, BLE timing and on-device behavior. Keep prior v20 model/settings unchanged and compare head-to-head on six+ existing people and a new participant set.
6. **Production gate**: merge/replace v20 only if the new model improves reliable sustained holds and does not worsen eventual release, soft hold sensitivity, or resource limits on independently verified events. A numerical threshold must be agreed against actual measured user behavior; do not infer one from pseudo-label F1.

## Reproduction

From the repository root on the research branch:

```bash
python -m pip install numpy pandas scipy scikit-learn lightgbm
python -m unittest discover -s tests -p "test_lp_state_v5.py" -v
python -m unittest discover -s tests -p "test_lp_v5_export_quality.py" -v
python lp_v5_labeler.py --dry-run --report /tmp/lp_v5_dry_run_audit.json
python scripts/lp_v5_student_benchmark.py
# Candidate export to an isolated V5 directory:
python lp_v5_labeler.py
```

Never overwrite `DATA/DATA_LONG_PRESS`, `DATA/DATA_LONG_PRESS_LABELED_V3` or the previous successful v20 trainer/model. All work is kept on [standalone V5 draft PR](https://github.com/alex00Pirotskyi/ML/pulls) and is not merged.
