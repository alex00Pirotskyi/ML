### tap: training gate (rows from frame 50 vs 500), scored on all usable rows and on steady-state rows

| features | train gate | train rows | positives | usable: best score | steady: best score | steady: score at firmware threshold | steady: recall @<=2 false/min |
|---|---|---|---|---|---|---|---|
| raw126 | 50 | 490066 | 70727 | 0.613 | 0.678 | 0.223 | 0.900 |
| raw126 | 500 | 170121 | 10155 | 0.572 | 0.669 | 0.232 | 0.814 |
| all_available | 50 | 490066 | 70727 | 0.663 | 0.765 | 0.263 | 0.913 |
| all_available | 500 | 170121 | 10155 | 0.575 | 0.694 | 0.292 | 0.833 |

### tap: K-curve (fold-local ranking, fixed configuration, retrained per K)

| subset | usable: best score | vs K=150 per fold | steady: best score | at firmware threshold | recall @<=1 / @<=2 false/min |
|---|---|---|---|---|---|
| K5 | 0.186 | [-0.785, -0.297, -0.351] | 0.251 | 0.038 | 0.151 / 0.151 |
| K10 | 0.444 | [-0.42, -0.099, -0.139] | 0.466 | 0.066 | 0.360 / 0.557 |
| K20 | 0.494 | [-0.257, -0.062, -0.19] | 0.625 | 0.093 | 0.453 / 0.688 |
| K30 | 0.535 | [-0.195, -0.029, -0.163] | 0.632 | 0.109 | 0.525 / 0.689 |
| K40 | 0.559 | [-0.206, -0.037, -0.071] | 0.684 | 0.121 | 0.518 / 0.766 |
| K60 | 0.598 | [-0.122, 0.002, -0.075] | 0.700 | 0.131 | 0.526 / 0.777 |
| K80 | 0.586 | [-0.145, -0.05, -0.038] | 0.668 | 0.133 | 0.532 / 0.806 |
| K100 | 0.591 | [-0.141, -0.068, -0.008] | 0.707 | 0.136 | 0.547 / 0.838 |
| K150 | 0.663 | [0.0, 0.0, 0.0] | 0.765 | 0.159 | 0.629 / 0.824 |
| random20 | 0.376 | [-0.408, -0.242, -0.213] | 0.486 | 0.069 | 0.327 / 0.541 |
| random40 | 0.494 | [-0.234, -0.05, -0.224] | 0.628 | 0.102 | 0.447 / 0.700 |

### tap: ranking stability (Nogueira) of the top-K sets

| K | across outer folds | across inner splits |
|---|---|---|
| 5 | 0.38 | 0.44 |
| 10 | 0.29 | 0.49 |
| 15 | 0.28 | 0.44 |
| 20 | 0.31 | 0.43 |
| 30 | 0.19 | 0.43 |
| 40 | 0.22 | 0.37 |
| 60 | 0.19 | 0.27 |
| 80 | 0.16 | 0.18 |
| 100 | 0.13 | 0.09 |
| 125 | 0.06 | 0.05 |

Top 15 per outer fold (family: name):

- fold 0 (cut 0.25, 55 clusters, 1876 s): ACC:acc_psd_mean[X][1], ACC:acc_psd_mean[Z][4], WL:wl_head[CH1], ACC:acc_psd_mean[X][4], ACC:acc_psd_mean[Y][4], ACC:acc_psd[0][Z][0], EMG:emg_psd[0][CH2][4], ACC:acc_psd[0][Y][0], Lean-8:cload_lean_8.rise10[CH3], ACC:acc_psd_mean[Y][2], ACC:acc_rate[Y][0], Lean-8:cload_lean_8.r01[CH1], Lean-8:cload_lean_8.r01[CH3], ACC:acc_rate[Z][2], ACC:acc_rate[Y][2]
- fold 1 (cut 0.25, 52 clusters, 1551 s): ACC:acc_psd[0][Z][0], ACC:acc_psd[0][X][1], ACC:acc_psd_mean[Z][4], WL:wl_head[CH1], EMG:emg_psd[0][CH2][2], ACC:acc_psd[0][Z][2], Lean-8:cload_lean_8.rise10[CH2], ACC:acc_rate[Z][0], ACC:acc_rate[Z][5], EMG:emg_psd_mean[CH2][0], EMG:emg_psd[EMG_PSD_HISTORY_SIZE - 1][CH2][0], EMG:emg_psd_mean[CH1][0], EMG:emg_psd_mean[CH3][0], ACC:acc_rate[X][5], ACC:acc_rate[Y][0]
- fold 2 (cut 0.2, 62 clusters, 1738 s): ACC:acc_psd_mean[Z][4], EMG:emg_psd[0][CH1][4], ACC:acc_psd_mean[X][1], ACC:acc_psd[ACC_PSD_HISTORY_SIZE - 1][Y][0], ACC:acc_psd_mean[Y][4], ACC:acc_psd_mean[X][4], ACC:acc_psd_mean[Y][0], Lean-8:cload_lean_8.rise10[CH1], EMG:emg_psd[0][CH2][2], Lean-8:cload_lean_8.rise10[CH3], EMG:emg_psd_mean[CH1][0], Lean-8:cload_lean_8.rise10[CH2], EMG:emg_psd_mean[CH3][0], ACC:acc_rate[Z][5], EMG:emg_psd_mean[CH2][0]
