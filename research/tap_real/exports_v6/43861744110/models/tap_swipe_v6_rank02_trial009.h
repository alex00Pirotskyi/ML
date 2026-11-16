#ifndef TAP_SWIPE_V6_RANK02_TRIAL009_MODEL_H
#define TAP_SWIPE_V6_RANK02_TRIAL009_MODEL_H

/*
 * Generated: 2026-11-16T11:57:07.121218+00:00
 * Model platform: LGBM
 * Task: BINARY_CLASSIFIER
 * Binary target class_id: 1
 * Training preprocessing: none
 * Firmware preprocessing: raw features; no preprocessing
 * Original selected feature indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
 * Required feature helpers/macros: none
 * Export profile: cortex_m4f_balanced
 * Flash storage: exact_float32_soa_bitpacked_flags; 880 tree-data bytes.
 * Total generated static model data: 1084 bytes.
 * Metadata layout: lgbm_u6_child_flags_packed
 * Feature metadata: plain_u8; default_left=child_bit6; missing_is_zero=child_bit7; flag bytes=0.
 */

#include <math.h>
#include <stdint.h>
#include "core_type_def.h"

#define TAP_SWIPE_V6_RANK02_TRIAL009_NUM_FEATURES 18u
#define TAP_SWIPE_V6_RANK02_TRIAL009_NUM_TREES 20u
#define TAP_SWIPE_V6_RANK02_TRIAL009_THRESHOLD 0.349999994f
#define TAP_SWIPE_V6_RANK02_TRIAL009_TARGET_CLASS_ID 1
#define TAP_SWIPE_V6_RANK02_TRIAL009_PREPROCESSING "none"

#ifndef TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE
  #if defined(__STATIC_FORCEINLINE)
    #define TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE __STATIC_FORCEINLINE
  #elif defined(__GNUC__) || defined(__clang__)
    #define TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE static inline __attribute__((always_inline))
  #else
    #define TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE static inline
  #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define TAP_SWIPE_V6_RANK02_TRIAL009_LIKELY(value)   __builtin_expect(!!(value), 1)
  #define TAP_SWIPE_V6_RANK02_TRIAL009_UNLIKELY(value) __builtin_expect(!!(value), 0)
#else
  #define TAP_SWIPE_V6_RANK02_TRIAL009_LIKELY(value)   (value)
  #define TAP_SWIPE_V6_RANK02_TRIAL009_UNLIKELY(value) (value)
#endif

#define TAP_SWIPE_V6_RANK02_TRIAL009_HELPER_CACHE_COUNT 0u
#define TAP_SWIPE_V6_RANK02_TRIAL009_HELPER_CACHE_BYTES 1u
typedef struct {
    uint8_t unused;
} tap_swipe_v6_rank02_trial009_feature_cache_t;

/** Return one selected feature; helper expressions are cached lazily once per prediction. */
TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE float tap_swipe_v6_rank02_trial009_get_raw_model_feature(
    const t_core* p_core, uint16_t model_feature_index, tap_swipe_v6_rank02_trial009_feature_cache_t* cache)
{
    (void)cache;
    switch (model_feature_index) {
        case 0u: return (float)(p_core->emg_psd[0][CH1][0]); // original[0] p_core->emg_psd[0][CH1][0]
        case 1u: return (float)(p_core->emg_psd[0][CH1][1]); // original[1] p_core->emg_psd[0][CH1][1]
        case 2u: return (float)(p_core->emg_psd[0][CH1][2]); // original[2] p_core->emg_psd[0][CH1][2]
        case 3u: return (float)(p_core->emg_psd[0][CH1][3]); // original[3] p_core->emg_psd[0][CH1][3]
        case 4u: return (float)(p_core->emg_psd[0][CH1][4]); // original[4] p_core->emg_psd[0][CH1][4]
        case 5u: return (float)(p_core->emg_psd[0][CH2][0]); // original[5] p_core->emg_psd[0][CH2][0]
        case 6u: return (float)(p_core->emg_psd[0][CH2][1]); // original[6] p_core->emg_psd[0][CH2][1]
        case 7u: return (float)(p_core->emg_psd[0][CH2][2]); // original[7] p_core->emg_psd[0][CH2][2]
        case 8u: return (float)(p_core->emg_psd[0][CH2][3]); // original[8] p_core->emg_psd[0][CH2][3]
        case 9u: return (float)(p_core->emg_psd[0][CH2][4]); // original[9] p_core->emg_psd[0][CH2][4]
        case 10u: return (float)(p_core->emg_psd[0][CH3][0]); // original[10] p_core->emg_psd[0][CH3][0]
        case 11u: return (float)(p_core->emg_psd[0][CH3][1]); // original[11] p_core->emg_psd[0][CH3][1]
        case 12u: return (float)(p_core->emg_psd[0][CH3][2]); // original[12] p_core->emg_psd[0][CH3][2]
        case 13u: return (float)(p_core->emg_psd[0][CH3][3]); // original[13] p_core->emg_psd[0][CH3][3]
        case 14u: return (float)(p_core->emg_psd[0][CH3][4]); // original[14] p_core->emg_psd[0][CH3][4]
        case 15u: return (float)(p_core->wl[CH1]); // original[15] p_core->wl[CH1]
        case 16u: return (float)(p_core->wl[CH2]); // original[16] p_core->wl[CH2]
        case 17u: return (float)(p_core->wl[CH3]); // original[17] p_core->wl[CH3]
        default: return 0.0f;
    }
}

TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE float tap_swipe_v6_rank02_trial009_runtime_normalization_factor(
    const t_core* p_core, tap_swipe_v6_rank02_trial009_feature_cache_t* cache)
{
    (void)p_core;
    (void)cache;
    return 1.0f;
}

static const float tap_swipe_v6_rank02_trial009_node_thresholds[140] = {
    456.5f, 316.5f, -0.166211411f, -0.0830998495f, 1700.5f, 0.0545422435f,
    -0.0139202243f, 542.5f, 366.5f, -0.133837715f, -0.0461386703f, 2131.5f,
    0.056613218f, -0.0159454253f, 668.5f, 94.5f, -0.118202642f, -0.0293961018f,
    1114.5f, 0.0873787254f, 0.00215984532f, 678.5f, 401.5f, -0.106503531f,
    -0.0275881384f, 10029.5f, 0.0492045991f, -0.0548614264f, 678.5f, 341.5f,
    -0.113354698f, -0.0313351378f, 2597.5f, 0.053768225f, -0.0188471396f, 423.5f,
    303.5f, -0.123777151f, -0.0597197115f, 1319.5f, -0.00372215779f, 0.0560373329f,
    412.5f, 303.5f, -0.117199332f, -0.0595465899f, 1671.5f, -0.00219650776f,
    0.0537194572f, 655.5f, 329.5f, -0.0994602889f, -0.0274204239f, 755.5f,
    0.0622197054f, -0.00525388448f, 696.5f, 366.5f, -0.0818555057f, -0.0121701825f,
    1555.5f, 0.0761048123f, -0.00116195087f, 906.5f, 389.5f, -0.0724108964f,
    -0.00894975755f, 1555.5f, 0.0756043121f, 0.000905119407f, 467.5f, 74.5f,
    -0.0110408422f, -0.0742441788f, 10029.5f, 0.0221232083f, -0.0599667653f, 1839.5f,
    702.5f, 0.00256592757f, -0.0440809131f, 3778.5f, 0.0692880452f, -0.00825611409f,
    412.5f, 289.5f, -0.100112066f, -0.0376405604f, 11355.5f, 0.0165231023f,
    -0.0597202182f, 1488.5f, 109.5f, 0.0162528474f, -0.0332190171f, 1114.5f,
    0.0666799769f, 0.00230388599f, 2340.5f, 702.5f, 0.00454036472f, -0.0388157666f,
    3561.5f, 0.0650045127f, -0.00284296623f, 729.5f, 303.5f, -0.0897908285f,
    -0.0177896358f, 608.5f, 0.053415928f, -0.00265422091f, 1365.5f, 109.5f,
    0.0176174194f, -0.0297913365f, 11355.5f, 0.0358589739f, -0.0471130796f, 341.5f,
    273.5f, -0.100391373f, -0.0466365144f, 10029.5f, 0.0119343065f, -0.0474466756f,
    401.5f, 46.5f, -0.00247183233f, -0.0658596456f, 100.5f, 0.0696044117f,
    0.000703822647f, 3430.5f, 702.5f, 0.00532645592f, -0.0309926476f, 11278.5f,
    0.0520906597f, -0.0558949485f
};
static const uint16_t tap_swipe_v6_rank02_trial009_tree_roots[20] = {
    0u, 7u, 14u, 21u, 28u, 35u, 42u, 49u, 56u, 63u, 70u, 77u,
    84u, 91u, 98u, 105u, 112u, 119u, 126u, 133u
};
static const uint8_t tap_swipe_v6_rank02_trial009_node_child_flags[140] = {
    0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u,
    0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u,
    0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u,
    0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u,
    0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u,
    0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u,
    0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u,
    0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u,
    0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u,
    0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u,
    0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u,
    0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u
};
static const uint8_t tap_swipe_v6_rank02_trial009_node_features[140] = {
    15u, 15u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 15u, 15u, 0xFFu, 0xFFu, 10u,
    0xFFu, 0xFFu, 2u, 2u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 15u, 15u, 0xFFu,
    0xFFu, 5u, 0xFFu, 0xFFu, 15u, 15u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 15u,
    15u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 15u, 15u, 0xFFu, 0xFFu, 3u, 0xFFu,
    0xFFu, 15u, 15u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 3u, 15u, 0xFFu, 0xFFu,
    10u, 0xFFu, 0xFFu, 3u, 15u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 15u, 5u,
    0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 3u, 10u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu,
    15u, 15u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 2u, 5u, 0xFFu, 0xFFu, 10u,
    0xFFu, 0xFFu, 3u, 10u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 15u, 15u, 0xFFu,
    0xFFu, 10u, 0xFFu, 0xFFu, 2u, 5u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 15u,
    15u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 15u, 5u, 0xFFu, 0xFFu, 10u, 0xFFu,
    0xFFu, 3u, 10u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu
};

/* Q20 probability LUT: one unit equals 0.05 probability. */
static const float tap_swipe_v6_rank02_trial009_q20_raw_thresholds[19] = {
    -2.94443893f, -2.19722462f, -1.73460102f, -1.38629436f, -1.09861231f,
    -0.847297847f, -0.619039237f, -0.405465096f, -0.200670689f, 0.0f,
    0.200670689f, 0.405465096f, 0.619039237f, 0.847297847f, 1.09861231f,
    1.38629436f, 1.73460102f, 2.19722462f, 2.94443893f
};
static const float tap_swipe_v6_rank02_trial009_q20_raw_midpoints[20] = {
    -3.66356158f, -2.51230574f, -1.9459101f, -1.55059743f, -1.23676264f,
    -0.969400585f, -0.730887532f, -0.510825634f, -0.302280873f, -0.100083455f,
    0.100083455f, 0.302280873f, 0.510825634f, 0.730887532f, 0.969400585f,
    1.23676264f, 1.55059743f, 1.9459101f, 2.51230574f, 3.66356158f
};

#define TAP_SWIPE_V6_RANK02_TRIAL009_EARLY_EXIT_GROUP_SIZE 4u
#define TAP_SWIPE_V6_RANK02_TRIAL009_EARLY_EXIT_GROUPS 5u
static const float tap_swipe_v6_rank02_trial009_remaining_min[6] = {
    -1.78236616f, -1.25759757f, -0.803797483f, -0.531200826f, -0.269260257f, 0.0f
};
static const float tap_swipe_v6_rank02_trial009_remaining_max[6] = {
    1.08774114f, 0.839992344f, 0.614240587f, 0.371115625f, 0.169489473f, 0.0f
};

TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE float tap_swipe_v6_rank02_trial009_walk_lgbm_tree(
    uint16_t root_index,
    const t_core* p_core,
    float normalization_factor,
    tap_swipe_v6_rank02_trial009_feature_cache_t* cache)
{
    uint8_t node_index = 0u;
    for (;;) {
        const uint32_t absolute_index = (uint32_t)root_index + (uint32_t)node_index;
        const uint8_t feature_metadata = tap_swipe_v6_rank02_trial009_node_features[absolute_index];
        if (TAP_SWIPE_V6_RANK02_TRIAL009_UNLIKELY(feature_metadata == 0xFFu)) {
            return tap_swipe_v6_rank02_trial009_node_thresholds[absolute_index];
        }
        const uint8_t feature_index = feature_metadata;

        float feature_value =
            tap_swipe_v6_rank02_trial009_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;

        if (feature_value >= -1.0e-35f && feature_value <= 1.0e-35f) {
            feature_value = 0.0f;
        }
        const uint8_t child_flags = tap_swipe_v6_rank02_trial009_node_child_flags[absolute_index];
        const uint8_t right_child = (uint8_t)(child_flags & 0x3Fu);
        const uint8_t default_left = (uint8_t)((child_flags >> 6) & 0x01u);
        const uint8_t missing_is_zero = (uint8_t)(child_flags >> 7);

        const uint8_t is_missing = (uint8_t)(
            (feature_value != feature_value) || ((missing_is_zero != 0u) && feature_value == 0.0f));

        if (TAP_SWIPE_V6_RANK02_TRIAL009_UNLIKELY(is_missing != 0u)) {
            node_index = default_left != 0u
                ? (uint8_t)(node_index + 1u)
                : right_child;
        } else {
            node_index = feature_value <= tap_swipe_v6_rank02_trial009_node_thresholds[absolute_index]
                ? (uint8_t)(node_index + 1u)
                : right_child;
        }
    }
}

/** Return boosted-tree raw score before the logistic link. */
TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE float tap_swipe_v6_rank02_trial009_predict_raw_score(const t_core* p_core)
{
    tap_swipe_v6_rank02_trial009_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank02_trial009_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t tree_index = 0u; tree_index < TAP_SWIPE_V6_RANK02_TRIAL009_NUM_TREES; ++tree_index) {
        raw_score += tap_swipe_v6_rank02_trial009_walk_lgbm_tree(tap_swipe_v6_rank02_trial009_tree_roots[tree_index], p_core, normalization_factor, &cache);
    }
    return raw_score;
}

/** Convert a raw score to nearest 0.05 probability, returned as Q20 [0..20]. */
TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE uint8_t tap_swipe_v6_rank02_trial009_probability_q20_from_raw(float raw_score)
{
    uint8_t low = 0u;
    uint8_t high = 20u;
    while (low < high) {
        const uint8_t middle = (uint8_t)((low + high) >> 1u);
        if (raw_score < tap_swipe_v6_rank02_trial009_q20_raw_midpoints[middle]) {
            high = middle;
        } else {
            low = (uint8_t)(middle + 1u);
        }
    }
    return low;
}

TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE uint8_t tap_swipe_v6_rank02_trial009_probability_q20(const t_core* p_core)
{
    return tap_swipe_v6_rank02_trial009_probability_q20_from_raw(tap_swipe_v6_rank02_trial009_predict_raw_score(p_core));
}

/** Approximate probability in 0.05 steps; maximum rounding error is 0.025. */
TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE float tap_swipe_v6_rank02_trial009_predict_value(const t_core* p_core)
{
    return 0.05f * (float)tap_swipe_v6_rank02_trial009_probability_q20(p_core);
}

/** Runtime-selectable exact threshold for Q20 values 0.05..0.95; no expf/logf. */
TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE int8_t tap_swipe_v6_rank02_trial009_predict_q20(    const t_core* p_core, uint8_t threshold_q20)
{
    if (threshold_q20 == 0u) { return 1; }
    if (threshold_q20 >= 20u) { return 0; }
    const float raw_threshold = tap_swipe_v6_rank02_trial009_q20_raw_thresholds[threshold_q20 - 1u];
    tap_swipe_v6_rank02_trial009_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank02_trial009_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t group = 0u; group < TAP_SWIPE_V6_RANK02_TRIAL009_EARLY_EXIT_GROUPS; ++group) {
        const uint16_t first_tree = (uint16_t)(group * TAP_SWIPE_V6_RANK02_TRIAL009_EARLY_EXIT_GROUP_SIZE);
        uint16_t end_tree = (uint16_t)(first_tree + TAP_SWIPE_V6_RANK02_TRIAL009_EARLY_EXIT_GROUP_SIZE);
        if (end_tree > TAP_SWIPE_V6_RANK02_TRIAL009_NUM_TREES) { end_tree = TAP_SWIPE_V6_RANK02_TRIAL009_NUM_TREES; }
        for (uint16_t tree_index = first_tree; tree_index < end_tree; ++tree_index) {
            raw_score += tap_swipe_v6_rank02_trial009_walk_lgbm_tree(tap_swipe_v6_rank02_trial009_tree_roots[tree_index], p_core, normalization_factor, &cache);
        }
        if (raw_score + tap_swipe_v6_rank02_trial009_remaining_max[group + 1u] < raw_threshold) { return 0; }
        if (raw_score + tap_swipe_v6_rank02_trial009_remaining_min[group + 1u] >= raw_threshold) { return 1; }
    }
    return raw_score >= raw_threshold ? 1 : 0;
}

#define TAP_SWIPE_V6_RANK02_TRIAL009_DEFAULT_THRESHOLD_Q20 7u
TAP_SWIPE_V6_RANK02_TRIAL009_FORCE_INLINE int8_t tap_swipe_v6_rank02_trial009_predict(const t_core* p_core)
{
    return tap_swipe_v6_rank02_trial009_predict_q20(p_core, TAP_SWIPE_V6_RANK02_TRIAL009_DEFAULT_THRESHOLD_Q20);
}

#endif /* TAP_SWIPE_V6_RANK02_TRIAL009_MODEL_H */
