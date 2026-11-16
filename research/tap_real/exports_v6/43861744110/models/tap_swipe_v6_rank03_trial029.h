#ifndef TAP_SWIPE_V6_RANK03_TRIAL029_MODEL_H
#define TAP_SWIPE_V6_RANK03_TRIAL029_MODEL_H

/*
 * Generated: 2026-11-16T11:57:08.559836+00:00
 * Model platform: LGBM
 * Task: BINARY_CLASSIFIER
 * Binary target class_id: 1
 * Training preprocessing: none
 * Firmware preprocessing: raw features; no preprocessing
 * Original selected feature indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
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

#define TAP_SWIPE_V6_RANK03_TRIAL029_NUM_FEATURES 36u
#define TAP_SWIPE_V6_RANK03_TRIAL029_NUM_TREES 20u
#define TAP_SWIPE_V6_RANK03_TRIAL029_THRESHOLD 0.449999988f
#define TAP_SWIPE_V6_RANK03_TRIAL029_TARGET_CLASS_ID 1
#define TAP_SWIPE_V6_RANK03_TRIAL029_PREPROCESSING "none"

#ifndef TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE
  #if defined(__STATIC_FORCEINLINE)
    #define TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE __STATIC_FORCEINLINE
  #elif defined(__GNUC__) || defined(__clang__)
    #define TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE static inline __attribute__((always_inline))
  #else
    #define TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE static inline
  #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define TAP_SWIPE_V6_RANK03_TRIAL029_LIKELY(value)   __builtin_expect(!!(value), 1)
  #define TAP_SWIPE_V6_RANK03_TRIAL029_UNLIKELY(value) __builtin_expect(!!(value), 0)
#else
  #define TAP_SWIPE_V6_RANK03_TRIAL029_LIKELY(value)   (value)
  #define TAP_SWIPE_V6_RANK03_TRIAL029_UNLIKELY(value) (value)
#endif

#define TAP_SWIPE_V6_RANK03_TRIAL029_HELPER_CACHE_COUNT 0u
#define TAP_SWIPE_V6_RANK03_TRIAL029_HELPER_CACHE_BYTES 1u
typedef struct {
    uint8_t unused;
} tap_swipe_v6_rank03_trial029_feature_cache_t;

/** Return one selected feature; helper expressions are cached lazily once per prediction. */
TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE float tap_swipe_v6_rank03_trial029_get_raw_model_feature(
    const t_core* p_core, uint16_t model_feature_index, tap_swipe_v6_rank03_trial029_feature_cache_t* cache)
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
        case 18u: return (float)(p_core->acc_psd[0][X][0]); // original[18] p_core->acc_psd[0][X][0]
        case 19u: return (float)(p_core->acc_psd[0][X][1]); // original[19] p_core->acc_psd[0][X][1]
        case 20u: return (float)(p_core->acc_psd[0][X][2]); // original[20] p_core->acc_psd[0][X][2]
        case 21u: return (float)(p_core->acc_psd[0][X][3]); // original[21] p_core->acc_psd[0][X][3]
        case 22u: return (float)(p_core->acc_psd[0][X][4]); // original[22] p_core->acc_psd[0][X][4]
        case 23u: return (float)(p_core->acc_psd[0][X][5]); // original[23] p_core->acc_psd[0][X][5]
        case 24u: return (float)(p_core->acc_psd[0][Y][0]); // original[24] p_core->acc_psd[0][Y][0]
        case 25u: return (float)(p_core->acc_psd[0][Y][1]); // original[25] p_core->acc_psd[0][Y][1]
        case 26u: return (float)(p_core->acc_psd[0][Y][2]); // original[26] p_core->acc_psd[0][Y][2]
        case 27u: return (float)(p_core->acc_psd[0][Y][3]); // original[27] p_core->acc_psd[0][Y][3]
        case 28u: return (float)(p_core->acc_psd[0][Y][4]); // original[28] p_core->acc_psd[0][Y][4]
        case 29u: return (float)(p_core->acc_psd[0][Y][5]); // original[29] p_core->acc_psd[0][Y][5]
        case 30u: return (float)(p_core->acc_psd[0][Z][0]); // original[30] p_core->acc_psd[0][Z][0]
        case 31u: return (float)(p_core->acc_psd[0][Z][1]); // original[31] p_core->acc_psd[0][Z][1]
        case 32u: return (float)(p_core->acc_psd[0][Z][2]); // original[32] p_core->acc_psd[0][Z][2]
        case 33u: return (float)(p_core->acc_psd[0][Z][3]); // original[33] p_core->acc_psd[0][Z][3]
        case 34u: return (float)(p_core->acc_psd[0][Z][4]); // original[34] p_core->acc_psd[0][Z][4]
        case 35u: return (float)(p_core->acc_psd[0][Z][5]); // original[35] p_core->acc_psd[0][Z][5]
        default: return 0.0f;
    }
}

TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE float tap_swipe_v6_rank03_trial029_runtime_normalization_factor(
    const t_core* p_core, tap_swipe_v6_rank03_trial029_feature_cache_t* cache)
{
    (void)p_core;
    (void)cache;
    return 1.0f;
}

static const float tap_swipe_v6_rank03_trial029_node_thresholds[140] = {
    445.5f, 20.5f, -0.0690223053f, -0.257446617f, 113.5f, 0.10018681f,
    -0.0536520891f, 542.5f, 45.5f, -0.0186123028f, -0.206480563f, 178.5f,
    0.0168138929f, 0.165942997f, 401.5f, 54.5f, -0.0847453028f, -0.208612084f,
    101.5f, 0.0821086988f, -0.0575681292f, 678.5f, 15.5f, -0.0171026085f,
    -0.18479003f, 293.5f, 0.0720253587f, -0.0817982182f, 696.5f, 14.5f,
    0.0218865834f, -0.139404282f, 1858.5f, 0.125658512f, -0.00787219778f, 366.5f,
    14.5f, -0.0720866546f, -0.169486195f, 236.5f, -0.00521249371f, 0.126338154f,
    54.5f, 354.5f, -0.10671591f, 0.0709253848f, 1555.5f, -0.150567561f,
    0.0225361455f, 292.5f, 37.5f, 0.00312389806f, 0.153415993f, 139.5f,
    -0.162855878f, 0.0257151853f, 22.5f, 389.5f, -0.0584095679f, 0.080080606f,
    757.5f, -0.138578847f, 0.0154751716f, 200.5f, 79.5f, 0.00484341988f,
    0.144271985f, 403.5f, -0.135921031f, 0.0486699641f, 690.5f, 11.5f,
    0.0163063351f, -0.11816068f, 1531.5f, 0.0473678932f, -0.120337829f, 26.5f,
    125.5f, -0.0485544614f, 0.0619478785f, 154.5f, -0.0828617364f, 0.0600978062f,
    632.5f, 39.5f, -0.00643065106f, 0.120932929f, 4113.5f, -0.155676261f,
    0.00292221224f, 588.5f, 7.5f, -0.00921404641f, -0.121596351f, 363.5f,
    0.00272955536f, 0.114674263f, 2925.5f, 139.5f, -0.0126664303f, 0.0946008712f,
    1026.5f, -0.205858693f, -0.0725488737f, 14.5f, 91.5f, -0.0764770731f,
    0.0616508648f, 1140.5f, -0.091408208f, 0.0189073365f, 68.5f, 690.5f,
    -0.00962819718f, 0.0649622455f, 65.5f, -0.0920875147f, 0.0362344906f, 316.5f,
    4193.5f, -0.132863566f, 0.150011897f, 848.5f, 0.0209854133f, -0.0869842023f,
    6.5f, 412.5f, -0.0175058395f, 0.075134404f, 950.5f, -0.0581136458f,
    0.0339271501f, 608.5f, 53.5f, -0.0104528703f, 0.0973534361f, 630.5f,
    -0.164607465f, -0.0137528647f
};
static const uint16_t tap_swipe_v6_rank03_trial029_tree_roots[20] = {
    0u, 7u, 14u, 21u, 28u, 35u, 42u, 49u, 56u, 63u, 70u, 77u,
    84u, 91u, 98u, 105u, 112u, 119u, 126u, 133u
};
static const uint8_t tap_swipe_v6_rank03_trial029_node_child_flags[140] = {
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
static const uint8_t tap_swipe_v6_rank03_trial029_node_features[140] = {
    15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 33u,
    0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu,
    0xFFu, 24u, 0xFFu, 0xFFu, 3u, 24u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 15u,
    35u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 2u, 0xFFu,
    0xFFu, 30u, 22u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu,
    2u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 15u, 30u,
    0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 30u, 2u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu,
    30u, 22u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu, 0xFFu, 34u,
    0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 30u, 12u, 0xFFu,
    0xFFu, 2u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu, 15u,
    16u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 3u, 0xFFu,
    0xFFu, 24u, 28u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu
};

/* Q20 probability LUT: one unit equals 0.05 probability. */
static const float tap_swipe_v6_rank03_trial029_q20_raw_thresholds[19] = {
    -2.94443893f, -2.19722462f, -1.73460102f, -1.38629436f, -1.09861231f,
    -0.847297847f, -0.619039237f, -0.405465096f, -0.200670689f, 0.0f,
    0.200670689f, 0.405465096f, 0.619039237f, 0.847297847f, 1.09861231f,
    1.38629436f, 1.73460102f, 2.19722462f, 2.94443893f
};
static const float tap_swipe_v6_rank03_trial029_q20_raw_midpoints[20] = {
    -3.66356158f, -2.51230574f, -1.9459101f, -1.55059743f, -1.23676264f,
    -0.969400585f, -0.730887532f, -0.510825634f, -0.302280873f, -0.100083455f,
    0.100083455f, 0.302280873f, 0.510825634f, 0.730887532f, 0.969400585f,
    1.23676264f, 1.55059743f, 1.9459101f, 2.51230574f, 3.66356158f
};

#define TAP_SWIPE_V6_RANK03_TRIAL029_EARLY_EXIT_GROUP_SIZE 4u
#define TAP_SWIPE_V6_RANK03_TRIAL029_EARLY_EXIT_GROUPS 5u
static const float tap_swipe_v6_rank03_trial029_remaining_min[6] = {
    -2.97959781f, -2.12224936f, -1.49992347f, -1.02221692f, -0.447673559f, 0.0f
};
static const float tap_swipe_v6_rank03_trial029_remaining_max[6] = {
    2.00962615f, 1.58934796f, 1.11300004f, 0.779325604f, 0.387463301f, 0.0f
};

TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE float tap_swipe_v6_rank03_trial029_walk_lgbm_tree(
    uint16_t root_index,
    const t_core* p_core,
    float normalization_factor,
    tap_swipe_v6_rank03_trial029_feature_cache_t* cache)
{
    uint8_t node_index = 0u;
    for (;;) {
        const uint32_t absolute_index = (uint32_t)root_index + (uint32_t)node_index;
        const uint8_t feature_metadata = tap_swipe_v6_rank03_trial029_node_features[absolute_index];
        if (TAP_SWIPE_V6_RANK03_TRIAL029_UNLIKELY(feature_metadata == 0xFFu)) {
            return tap_swipe_v6_rank03_trial029_node_thresholds[absolute_index];
        }
        const uint8_t feature_index = feature_metadata;

        float feature_value =
            tap_swipe_v6_rank03_trial029_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;

        if (feature_value >= -1.0e-35f && feature_value <= 1.0e-35f) {
            feature_value = 0.0f;
        }
        const uint8_t child_flags = tap_swipe_v6_rank03_trial029_node_child_flags[absolute_index];
        const uint8_t right_child = (uint8_t)(child_flags & 0x3Fu);
        const uint8_t default_left = (uint8_t)((child_flags >> 6) & 0x01u);
        const uint8_t missing_is_zero = (uint8_t)(child_flags >> 7);

        const uint8_t is_missing = (uint8_t)(
            (feature_value != feature_value) || ((missing_is_zero != 0u) && feature_value == 0.0f));

        if (TAP_SWIPE_V6_RANK03_TRIAL029_UNLIKELY(is_missing != 0u)) {
            node_index = default_left != 0u
                ? (uint8_t)(node_index + 1u)
                : right_child;
        } else {
            node_index = feature_value <= tap_swipe_v6_rank03_trial029_node_thresholds[absolute_index]
                ? (uint8_t)(node_index + 1u)
                : right_child;
        }
    }
}

/** Return boosted-tree raw score before the logistic link. */
TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE float tap_swipe_v6_rank03_trial029_predict_raw_score(const t_core* p_core)
{
    tap_swipe_v6_rank03_trial029_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank03_trial029_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t tree_index = 0u; tree_index < TAP_SWIPE_V6_RANK03_TRIAL029_NUM_TREES; ++tree_index) {
        raw_score += tap_swipe_v6_rank03_trial029_walk_lgbm_tree(tap_swipe_v6_rank03_trial029_tree_roots[tree_index], p_core, normalization_factor, &cache);
    }
    return raw_score;
}

/** Convert a raw score to nearest 0.05 probability, returned as Q20 [0..20]. */
TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE uint8_t tap_swipe_v6_rank03_trial029_probability_q20_from_raw(float raw_score)
{
    uint8_t low = 0u;
    uint8_t high = 20u;
    while (low < high) {
        const uint8_t middle = (uint8_t)((low + high) >> 1u);
        if (raw_score < tap_swipe_v6_rank03_trial029_q20_raw_midpoints[middle]) {
            high = middle;
        } else {
            low = (uint8_t)(middle + 1u);
        }
    }
    return low;
}

TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE uint8_t tap_swipe_v6_rank03_trial029_probability_q20(const t_core* p_core)
{
    return tap_swipe_v6_rank03_trial029_probability_q20_from_raw(tap_swipe_v6_rank03_trial029_predict_raw_score(p_core));
}

/** Approximate probability in 0.05 steps; maximum rounding error is 0.025. */
TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE float tap_swipe_v6_rank03_trial029_predict_value(const t_core* p_core)
{
    return 0.05f * (float)tap_swipe_v6_rank03_trial029_probability_q20(p_core);
}

/** Runtime-selectable exact threshold for Q20 values 0.05..0.95; no expf/logf. */
TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE int8_t tap_swipe_v6_rank03_trial029_predict_q20(    const t_core* p_core, uint8_t threshold_q20)
{
    if (threshold_q20 == 0u) { return 1; }
    if (threshold_q20 >= 20u) { return 0; }
    const float raw_threshold = tap_swipe_v6_rank03_trial029_q20_raw_thresholds[threshold_q20 - 1u];
    tap_swipe_v6_rank03_trial029_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank03_trial029_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t group = 0u; group < TAP_SWIPE_V6_RANK03_TRIAL029_EARLY_EXIT_GROUPS; ++group) {
        const uint16_t first_tree = (uint16_t)(group * TAP_SWIPE_V6_RANK03_TRIAL029_EARLY_EXIT_GROUP_SIZE);
        uint16_t end_tree = (uint16_t)(first_tree + TAP_SWIPE_V6_RANK03_TRIAL029_EARLY_EXIT_GROUP_SIZE);
        if (end_tree > TAP_SWIPE_V6_RANK03_TRIAL029_NUM_TREES) { end_tree = TAP_SWIPE_V6_RANK03_TRIAL029_NUM_TREES; }
        for (uint16_t tree_index = first_tree; tree_index < end_tree; ++tree_index) {
            raw_score += tap_swipe_v6_rank03_trial029_walk_lgbm_tree(tap_swipe_v6_rank03_trial029_tree_roots[tree_index], p_core, normalization_factor, &cache);
        }
        if (raw_score + tap_swipe_v6_rank03_trial029_remaining_max[group + 1u] < raw_threshold) { return 0; }
        if (raw_score + tap_swipe_v6_rank03_trial029_remaining_min[group + 1u] >= raw_threshold) { return 1; }
    }
    return raw_score >= raw_threshold ? 1 : 0;
}

#define TAP_SWIPE_V6_RANK03_TRIAL029_DEFAULT_THRESHOLD_Q20 9u
TAP_SWIPE_V6_RANK03_TRIAL029_FORCE_INLINE int8_t tap_swipe_v6_rank03_trial029_predict(const t_core* p_core)
{
    return tap_swipe_v6_rank03_trial029_predict_q20(p_core, TAP_SWIPE_V6_RANK03_TRIAL029_DEFAULT_THRESHOLD_Q20);
}

#endif /* TAP_SWIPE_V6_RANK03_TRIAL029_MODEL_H */
