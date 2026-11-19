# tap: held-out people, replayed through the C headers

| Model | Bytes | recall | recall_press | false_per_min | latency_ms_median | objective_score |
| --- | --- | --- | --- | --- | --- | --- |
| new rank_01 (trial 36, K=195) **default** | 32668 | 0.967 | 0.898 | 2.832 | -6.000 | 0.967 |
| new budget_24kb (trial 29, K=193) | 24484 | 0.966 | 0.893 | 2.819 | -5.000 | 0.966 |
| new rank_02 (trial 29, K=193) | 32656 | 0.971 | 0.898 | 2.713 | -5.000 | 0.971 |
| new rank_03 (trial 30, K=168) | 32692 | 0.965 | 0.896 | 2.700 | -5.000 | 0.965 |
| new rank_04 (trial 26, K=178) | 28484 | 0.968 | 0.899 | 2.858 | -6.000 | 0.968 |
| baseline previous_tap_thr085 (replayed) | - | 0.977 | 0.920 | 4.202 | -5.000 | 0.977 |
| baseline unified_20261117 (replayed) | - | 0.986 | 0.948 | 15.161 | -11.000 | 0.406 |
| baseline extra_baseline (replayed) | - | 0.980 | 0.909 | 2.292 | -7.000 | 0.980 |

Steady state only (rows after 500 Lean-8 frames, ~10.8 s, i.e. lean_8_features_ready()):

| Model | Bytes | recall | recall_press | false_per_min | latency_ms_median | objective_score |
| --- | --- | --- | --- | --- | --- | --- |
| new rank_01 (trial 36, K=195) **default** | 32668 | 0.960 | 0.869 | 1.833 | -6.000 | 0.960 |
| new budget_24kb (trial 29, K=193) | 24484 | 0.954 | 0.859 | 1.706 | -6.000 | 0.954 |
| new rank_02 (trial 29, K=193) | 32656 | 0.964 | 0.869 | 1.680 | -7.000 | 0.964 |
| new rank_03 (trial 30, K=168) | 32692 | 0.958 | 0.864 | 1.833 | -6.000 | 0.958 |
| new rank_04 (trial 26, K=178) | 28484 | 0.961 | 0.869 | 1.680 | -7.000 | 0.961 |
| baseline previous_tap_thr085 (replayed) | - | 0.961 | 0.884 | 2.953 | -5.000 | 0.961 |
| baseline unified_20261117 (replayed) | - | 0.976 | 0.899 | 11.023 | -12.000 | 0.553 |
| baseline extra_baseline (replayed) | - | 0.973 | 0.888 | 1.553 | -8.000 | 0.973 |

Simulated band move / contact change on the same held-out people (TAP_GENERALIZATION_RESULTS.md: EMG gain +-1 octave, spectral tilt +-0.5 octave, contact noise; ACC unchanged; the same draw for every model). A model that generalizes keeps recall without more false events:

| Model | Bytes | recall | recall_press | false_per_min | latency_ms_median | objective_score |
| --- | --- | --- | --- | --- | --- | --- |
| new rank_01 (trial 36, K=195) **default** | 32668 | 0.965 | 0.893 | 3.649 | -6.000 | 0.965 |
| new budget_24kb (trial 29, K=193) | 24484 | 0.957 | 0.884 | 3.491 | -5.000 | 0.957 |
| new rank_02 (trial 29, K=193) | 32656 | 0.961 | 0.891 | 3.557 | -6.000 | 0.961 |
| new rank_03 (trial 30, K=168) | 32692 | 0.962 | 0.891 | 3.596 | -6.000 | 0.962 |
| new rank_04 (trial 26, K=178) | 28484 | 0.963 | 0.894 | 4.031 | -6.000 | 0.963 |
| baseline previous_tap_thr085 (replayed) | - | 0.972 | 0.912 | 9.668 | -3.000 | 0.628 |
| baseline unified_20261117 (replayed) | - | 0.988 | 0.952 | 33.075 | -10.000 | 0.187 |
| baseline extra_baseline (replayed) | - | 0.969 | 0.897 | 5.559 | -5.000 | 0.969 |

Acceptance of the default header against each replayed baseline, held-out people (PIPELINE.md; report only, the choice is made on CV):

| Baseline | Metric | New | Baseline | Rule | Result |
| --- | --- | --- | --- | --- | --- |
| previous_tap_thr085 | recall_press | 0.898 | 0.920 | >= baseline - 0.01 | **review** |
| previous_tap_thr085 | recall | 0.967 | 0.977 | >= baseline - 0.01 | pass |
| previous_tap_thr085 | false_per_min | 2.832 | 4.202 | <= baseline x 1.1 + 0.1 | pass |
| unified_20261117 | recall_press | 0.898 | 0.948 | >= baseline - 0.01 | **review** |
| unified_20261117 | recall | 0.967 | 0.986 | >= baseline - 0.01 | **review** |
| unified_20261117 | false_per_min | 2.832 | 15.161 | <= baseline x 1.1 + 0.1 | pass |
| extra_baseline | recall_press | 0.898 | 0.909 | >= baseline - 0.01 | **review** |
| extra_baseline | recall | 0.967 | 0.980 | >= baseline - 0.01 | **review** |
| extra_baseline | false_per_min | 2.832 | 2.292 | <= baseline x 1.1 + 0.1 | **review** |

