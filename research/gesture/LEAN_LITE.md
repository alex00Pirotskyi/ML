# Lean-lite: integer EMG context features for the gesture models (2026-11-18)

`firmware/lean_lite.h` replaces the float Lean-8 handler. It keeps the part of Lean-8 that the trees
actually use and computes it with integer arithmetic. It uses the CMSIS intrinsics where they map to a
single instruction:
- `__CLZ` for the log,
- `__SSAT` for saturation,
- `arm_min_q15` / `arm_max_q15` for the once-per-second block scan.

## What was kept and why

Tree splits per Lean-8 field in the two headers flashed on 2026-11-18 (TAP 2,240 splits, LP 2,559):

| Lean-8 field | TAP splits | LP splits | Lean-lite |
|---|---|---|---|
| min10, max10, rise10, sd3 (10 s / 3 s history) | 529 | 777 | kept: `min10`, `max10`, `rise10`, `dev3` |
| r01 (PSD-mean zone 0 / zone 1) | 56 | 215 | kept: `d01` |
| r32, Pm | 37 | 51 | dropped |
| lw = log10(WL+1) | 10 | 0 | dropped: a tree splits on `wl` with the same result (monotone transform) |

## Definitions (per EMG channel; one call per EMG frame)

L = log2(WL + 1) in Q8, computed as `CLZ` for the integer part plus the 8 bits after the leading one
(piecewise-linear log2, at most 0.086 octave off, exact at powers of two).

| Field | Definition | Replaces |
|---|---|---|
| `min10` | minimum of L over the current 1 s block and the 9 complete blocks before it (46 frames per block) | min10: exact 500-frame sliding minimum |
| `max10` | the same, maximum | max10 |
| `rise10` | L − min10 | rise10 |
| `dev3` | running mean of \|L − running mean of L\|, both with time constant 2^7 = 128 frames (~2.8 s), shifts only, ×8 | sd3: 150-frame population std |
| `d01` | `emg_psd_mean[ch][0] − emg_psd_mean[ch][1]`, saturated to int16. The codes are 10000·log10(1+power), so this is the zone ratio in dB×1000 at any real power | r01 |
| `range10` | max10 − min10 (added 2026-11-23) | – |
| `fall10` | max10 − L (added 2026-11-23) | – |
| `xm` | L − running mean of L, signed, saturated to int16 (added 2026-11-23) | – |

The final TAP model (`MUDRA_TAP_RECIPE=final14`, TAP_FOCUS_RESULTS.md) reads only the gain-invariant differences
`rise10`, `fall10`, `range10`, `dev3`, `xm` and runs from frame 50, without waiting for `lean_lite_ready()`. The three
added fields are appended to `t_cload_lite` (48 bytes), so older models keep their field offsets.

Rank correlation with the Lean-8 fields they replace, on 41 recordings (all 7 TAP folders, steady frames):

| Field | Correlation |
|---|---|
| rise10 | 0.99 |
| max10 | 0.99 |
| min10 | 0.98 |
| dev3 vs sd3 | 0.87 |
| d01 vs r01 | 0.86 |

The models are retrained on the new definitions, so this is a sanity check, not a requirement.

## Cost on a Cortex-M4F

These counts come from emulating the real Cortex-M4 machine code:
- Both handlers were built with clang `-O2` and the real CMSIS headers, then run under Unicorn on 3,062
  real EMG frames.
- The "cycles" are a rough model: loads and stores 2, VDIV/VSQRT 14, +2 per taken branch. Flash wait
  states are not modeled.

| Per EMG frame (46 per second) | Instructions, mean | Instructions, worst | Rough cycles, mean | Rough cycles, worst |
|---|---|---|---|---|
| Lean-8 (`lean_8_features_handler`) | 1,601 | 12,795 (a 10 s extremum expires: rescan of 3 × 500 samples) | 2,256 | 15,245 |
| **Lean-lite** | **249** | **822** (a block closes, once per second) | **363** | **1,141** |

| Memory | Lean-8 | Lean-lite |
|---|---|---|
| State | 3,000 B history + running sums | **164 B** |
| t_core fields | 96 B (24 floats) | **30 B** (15 int16) |

Lean-lite has no float, no division, no libm and no 64-bit arithmetic.

**Lean-8 was never most of the CPU load; the models are.** Same emulation, the headers flashed on
2026-11-18, per ACC tick (~225 per second):

| Per ACC tick | Rough cycles |
|---|---|
| TAP, full Q20 probability (`model_debug`, every tree) | 57,000–61,000 |
| TAP, decision only (`TAP_SWIPE_predict`, early exit) | 45,000–52,000 |
| LP, full Q20 probability | ~38,000 |

Running both probabilities every tick costs roughly 100,000 cycles per tick. Lean-8 was about 0.5% of
that and Lean-lite is under 0.1%. The real CPU savings are:
- Run LP only while a press is active.
- Use the decision API in production.
- Make the exported tree walker cheaper. Each node currently costs about 35 instructions (switch-based
  feature fetch, normalization, missing-value checks).

## Firmware integration

```c
// t_core (packed): the models read p_core->cload_lite.<field>[CHn]
t_cload_lite cload_lite;

// module state, zero at boot (= reset)
static lean_lite_state_t m_lean_lite;

// SNC branch, once per EMG frame, AFTER the EMG PSD mean of this frame is computed:
emg_estimate_wl();
emg_estimate_rms();
psd_estemate_psd_mean(...);
{
    uint32_t wl[3]; uint16_t pm[15]; t_cload_lite out;
    for (uint32_t ch = 0u; ch < 3u; ++ch) {
        wl[ch] = (uint32_t) m_feat_core.wl[ch];
        for (uint32_t z = 0u; z < 5u; ++z) pm[ch * 5u + z] = (uint16_t) m_feat_core.emg_psd_mean[ch][z];
    }
    lean_lite_update(&m_lean_lite, wl, pm, &out);
    memcpy((void*) &m_feat_core.cload_lite, &out, sizeof(out));
}
// models: run only when lean_lite_ready(&m_lean_lite)  (500 frames, ~10.8 s after a reset)
// lean_lite_reset(&m_lean_lite) when the sensor stream restarts
```

The copies in and out keep every access aligned although `t_core` is packed. They cost a few dozen
instructions.

## Training

```bash
MUDRA_USE_LEAN8=lite python -u research/gesture/train_unified.py
```

This mode:
- Generates `DATA/DATA_TAP_LITE` and `DATA/DATA_LONG_PRESS_LITE` once (incremental), with the 15 fields
  appended to the original lines.
- Trains on the 126 raw fields + 15 Lean-lite fields.
- Writes to `results_unified_v3_feature_selection_lean_lite/`, with studies named `lean8lite`.

The frame counter comes from the same frame detection as Lean-8, so rows, labels, gates and the held-out
split are identical to the Lean-8 and no-Lean-8 runs. The three are clean A/B arms. Their fingerprints and
study names are unchanged, so a running `MUDRA_USE_LEAN8=0` study resumes as before.

## How it is verified

- **Bit-exact parity.** `generate_lean_lite_features.py` compiles `lean_lite.h` for the host
  (`-DLEAN_LITE_HOST`, plain C instead of the intrinsics). The emulator must equal it exactly before
  anything is generated. The check covers 24 recordings plus synthetic edge cases: WL 0 and 2^32−1, every
  power of two, saturating PSD differences, flat stretches and ramps. A deliberate one-constant error in
  the emulator fails it.
- **Real CMSIS path.** The Cortex-M4 build with the real CMSIS-DSP `arm_min_q15` / `arm_max_q15`,
  executed under emulation, equals the emulator on 3,062 real frames.
- **Machine-code checks.** `firmware/check_arm_build.sh` (also in CI) cross-compiles against CMSIS 5.9.0
  and CMSIS-DSP 1.16.2 with `-Werror`. It asserts that CLZ and SSAT are used, that there is no
  floating-point instruction, and that the only calls are `arm_min_q15` / `arm_max_q15`.
- **Smoke test.** `test_unified_smoke.py` checks:
  - the vectorized emulator against a literal frame-by-frame port;
  - export, compile and replay of a model on the 141 fields;
  - a second real-source run with `MUDRA_USE_LEAN8=lite` that must build the same rows, labels, frames
    and weights as the Lean-8 run.

To check the device, log the 15 int16 `cload_lite` fields next to the sensor fields for one session. The
offline values from the same logged `wl` / `emg_psd_mean` must match exactly.
