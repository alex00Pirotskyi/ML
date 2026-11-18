### lp: training gate (rows from frame 50 vs 500), scored on all usable rows and on steady-state rows

| features | train gate | train rows | positives | usable: best score | steady: best score | steady: score at firmware threshold |
|---|---|---|---|---|---|---|
| raw126 | 50 | 414567 | 223675 | 0.573 | 0.543 | 0.483 |
| raw126 | 500 | 216414 | 62764 | 0.590 | 0.556 | 0.511 |
| all_available | 50 | 414567 | 223675 | 0.580 | 0.561 | 0.440 |
| all_available | 500 | 216414 | 62764 | 0.568 | 0.544 | 0.441 |

### lp: K-curve (fold-local ranking, fixed configuration, retrained per K)

| subset | usable: best score | vs K=150 per fold | steady: best score | at firmware threshold |
|---|---|---|---|---|
| K5 | 0.491 | [-0.131, -0.084, -0.055] | 0.475 | 0.446 |
| K10 | 0.594 | [-0.018, 0.008, 0.051] | 0.570 | 0.560 |
| K15 | 0.553 | [-0.092, -0.022, 0.032] | 0.523 | 0.514 |
| K20 | 0.567 | [-0.071, 0.001, 0.029] | 0.531 | 0.520 |
| K30 | 0.570 | [-0.065, 0.017, 0.016] | 0.541 | 0.515 |
| K40 | 0.577 | [-0.094, 0.059, 0.024] | 0.542 | 0.506 |
| K60 | 0.594 | [-0.056, 0.064, 0.033] | 0.567 | 0.523 |
| K80 | 0.598 | [-0.024, 0.029, 0.048] | 0.564 | 0.473 |
| K100 | 0.611 | [-0.019, 0.058, 0.054] | 0.583 | 0.507 |
| K125 | 0.557 | [-0.067, -0.036, 0.032] | 0.533 | 0.467 |
| K150 | 0.580 | [0.0, 0.0, 0.0] | 0.561 | 0.453 |
| random20 | 0.580 | [-0.042, 0.031, 0.009] | 0.542 | 0.526 |
| random40 | 0.526 | [-0.059, -0.042, -0.062] | 0.507 | 0.489 |

### lp: ranking stability (Nogueira) of the top-K sets

| K | across outer folds | across inner splits |
|---|---|---|
| 5 | 0.59 | 0.49 |
| 10 | 0.43 | 0.47 |
| 15 | 0.33 | 0.42 |
| 20 | 0.35 | 0.39 |
| 30 | 0.35 | 0.32 |
| 40 | 0.30 | 0.29 |
| 60 | 0.35 | 0.28 |
| 80 | 0.33 | 0.14 |
| 100 | 0.08 | 0.08 |
| 125 | 0.07 | 0.07 |

Top 15 per outer fold (family: name):

- fold 0 (cut 0.3, 66 clusters, 720 s): WL:wl_head[CH2], WL:wl_head[CH1], ACC:acc_psd_mean[Z][0], Lean-8:cload_lean_8.rise10[CH2], Lean-8:cload_lean_8.rise10[CH1], EMG:emg_psd_mean[CH3][0], ACC:acc_psd_mean[Y][0], EMG:emg_psd_mean[CH2][2], EMG:emg_psd[EMG_PSD_HISTORY_SIZE - 1][CH1][1], Lean-8:cload_lean_8.r01[CH3], ACC:acc_psd[0][Y][5], ACC:acc_psd[0][Y][0], Lean-8:cload_lean_8.r01[CH1], Lean-8:cload_lean_8.r32[CH2], Lean-8:cload_lean_8.r01[CH2]
- fold 1 (cut 0.3, 66 clusters, 741 s): EMG:emg_psd[0][CH2][2], Lean-8:cload_lean_8.rise10[CH2], ACC:acc_psd_mean[Z][0], WL:wl_head[CH1], EMG:emg_psd[EMG_PSD_HISTORY_SIZE - 1][CH1][0], ACC:acc_psd_mean[Y][0], EMG:emg_psd_mean[CH2][1], EMG:emg_psd[0][CH3][3], Lean-8:cload_lean_8.r01[CH3], Lean-8:cload_lean_8.r01[CH2], ACC:acc_psd[ACC_PSD_HISTORY_SIZE - 1][X][0], Lean-8:cload_lean_8.max10[CH2], ACC:acc_psd[0][Z][5], Lean-8:cload_lean_8.sd3[CH2], ACC:acc_psd[0][X][0]
- fold 2 (cut 0.3, 64 clusters, 748 s): WL:wl_head[CH2], WL:wl_head[CH1], ACC:acc_psd_mean[Z][0], Lean-8:cload_lean_8.rise10[CH1], EMG:emg_psd_mean[CH3][0], ACC:acc_psd[ACC_PSD_HISTORY_SIZE - 1][Y][0], Lean-8:cload_lean_8.rise10[CH2], Lean-8:cload_lean_8.r32[CH2], EMG:emg_psd_mean[CH1][0], EMG:emg_psd[EMG_PSD_HISTORY_SIZE - 1][CH2][4], ACC:acc_psd[0][Y][5], Lean-8:cload_lean_8.r32[CH1], Lean-8:cload_lean_8.sd3[CH3], Lean-8:cload_lean_8.max10[CH2], ACC:acc_psd_mean[Z][3]
