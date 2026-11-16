#ifndef TAP_SWIPE_V6_RANK10_TRIAL032_MODEL_H
#define TAP_SWIPE_V6_RANK10_TRIAL032_MODEL_H

/*
 * Generated: 2026-11-16T11:57:24.759929+00:00
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

#define TAP_SWIPE_V6_RANK10_TRIAL032_NUM_FEATURES 36u
#define TAP_SWIPE_V6_RANK10_TRIAL032_NUM_TREES 20u
#define TAP_SWIPE_V6_RANK10_TRIAL032_THRESHOLD 0.449999988f
#define TAP_SWIPE_V6_RANK10_TRIAL032_TARGET_CLASS_ID 1
#define TAP_SWIPE_V6_RANK10_TRIAL032_PREPROCESSING "none"

#ifndef TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE
  #if defined(__STATIC_FORCEINLINE)
    #define TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE __STATIC_FORCEINLINE
  #elif defined(__GNUC__) || defined(__clang__)
    #define TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE static inline __attribute__((always_inline))
  #else
    #define TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE static inline
  #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define TAP_SWIPE_V6_RANK10_TRIAL032_LIKELY(value)   __builtin_expect(!!(value), 1)
  #define TAP_SWIPE_V6_RANK10_TRIAL032_UNLIKELY(value) __builtin_expect(!!(value), 0)
#else
  #define TAP_SWIPE_V6_RANK10_TRIAL032_LIKELY(value)   (value)
  #define TAP_SWIPE_V6_RANK10_TRIAL032_UNLIKELY(value) (value)
#endif

#define TAP_SWIPE_V6_RANK10_TRIAL032_HELPER_CACHE_COUNT 0u
#define TAP_SWIPE_V6_RANK10_TRIAL032_HELPER_CACHE_BYTES 1u
typedef struct {
    uint8_t unused;
} tap_swipe_v6_rank10_trial032_feature_cache_t;

/** Return one selected feature; helper expressions are cached lazily once per prediction. */
TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE float tap_swipe_v6_rank10_trial032_get_raw_model_feature(
    const t_core* p_core, uint16_t model_feature_index, tap_swipe_v6_rank10_trial032_feature_cache_t* cache)
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

TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE float tap_swipe_v6_rank10_trial032_runtime_normalization_factor(
    const t_core* p_core, tap_swipe_v6_rank10_trial032_feature_cache_t* cache)
{
    (void)p_core;
    (void)cache;
    return 1.0f;
}

static const float tap_swipe_v6_rank10_trial032_node_thresholds[140] = {
    456.5f, 20.5f, -0.0506374352f, -0.202343181f, 113.5f, 0.0809858665f,
    -0.0408691168f, 423.5f, 45.5f, -0.054394044f, -0.181532368f, 178.5f,
    0.00746342586f, 0.122391678f, 542.5f, 18.5f, -0.0127570862f, -0.164043814f,
    131.5f, 0.075901188f, -0.0392876975f, 678.5f, 14.5f, -0.0115730762f,
    -0.148703098f, 239.5f, 0.060010165f, -0.056749586f, 389.5f, 17.5f,
    -0.0429739207f, -0.142862767f, 43.5f, 0.0499240384f, -0.0332118645f, 696.5f,
    17.5f, 0.0153941084f, -0.115018994f, 1555.5f, 0.104822755f, 0.00213113124f,
    401.5f, 42.5f, -0.0545742922f, -0.138655782f, 178.5f, -0.0014389446f,
    0.106291458f, 226.5f, 678.5f, -0.028118005f, 0.0614710897f, 1839.5f,
    -0.13054803f, 0.0242996011f, 26.5f, 366.5f, -0.0664023906f, 0.0656572431f,
    1427.5f, -0.104735598f, 0.0248517431f, 56.5f, 341.5f, -0.0853587762f,
    0.0472351089f, 191.5f, -0.0938297361f, 0.0495442078f, 632.5f, 45.5f,
    -0.0052075889f, 0.114811644f, 790.5f, -0.137621239f, 0.0100929532f, 21.5f,
    125.5f, -0.0341066942f, 0.0571076684f, 868.5f, -0.102432624f, 0.017169822f,
    292.5f, 22.5f, -0.000662887003f, 0.0981874913f, 139.5f, -0.114319988f,
    0.0258231349f, 590.5f, 9.5f, 0.00772388792f, -0.0977804661f, 249.5f,
    0.0030294864f, 0.0997236595f, 1531.5f, 139.5f, -0.00968647189f, 0.0947254971f,
    666.5f, -0.14438802f, -0.0284407791f, 38.5f, 205.5f, -0.0449253432f,
    0.0439694859f, 950.5f, -0.0837237537f, 0.0204563029f, 10.5f, 389.5f,
    -0.0294489041f, 0.0636208653f, 757.5f, -0.079108879f, 0.0145584913f, 68.5f,
    666.5f, -0.0118171778f, 0.0517222881f, 292.5f, -0.0720293149f, 0.0427570194f,
    2925.5f, 53.5f, -0.0102246813f, 0.0785762146f, 1004.5f, -0.0926444978f,
    -0.181083962f, 380.5f, 83.5f, -0.00221279333f, 0.0908380449f, 627.5f,
    -0.100639276f, 0.0186581742f
};
static const uint16_t tap_swipe_v6_rank10_trial032_tree_roots[20] = {
    0u, 7u, 14u, 21u, 28u, 35u, 42u, 49u, 56u, 63u, 70u, 77u,
    84u, 91u, 98u, 105u, 112u, 119u, 126u, 133u
};
static const uint8_t tap_swipe_v6_rank10_trial032_node_child_flags[140] = {
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
static const uint8_t tap_swipe_v6_rank10_trial032_node_features[140] = {
    15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 33u,
    0xFFu, 0xFFu, 15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu,
    0xFFu, 24u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 24u, 0xFFu, 0xFFu, 3u,
    24u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 33u, 0xFFu,
    0xFFu, 30u, 15u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu,
    2u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 22u,
    0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 2u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu,
    30u, 22u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 2u, 24u, 0xFFu, 0xFFu, 34u,
    0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 30u, 4u, 0xFFu,
    0xFFu, 3u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 30u,
    15u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 28u, 0xFFu, 0xFFu, 18u, 0xFFu,
    0xFFu, 30u, 34u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu
};

/* Q20 probability LUT: one unit equals 0.05 probability. */
static const float tap_swipe_v6_rank10_trial032_q20_raw_thresholds[19] = {
    -2.94443893f, -2.19722462f, -1.73460102f, -1.38629436f, -1.09861231f,
    -0.847297847f, -0.619039237f, -0.405465096f, -0.200670689f, 0.0f,
    0.200670689f, 0.405465096f, 0.619039237f, 0.847297847f, 1.09861231f,
    1.38629436f, 1.73460102f, 2.19722462f, 2.94443893f
};
static const float tap_swipe_v6_rank10_trial032_q20_raw_midpoints[20] = {
    -3.66356158f, -2.51230574f, -1.9459101f, -1.55059743f, -1.23676264f,
    -0.969400585f, -0.730887532f, -0.510825634f, -0.302280873f, -0.100083455f,
    0.100083455f, 0.302280873f, 0.510825634f, 0.730887532f, 0.969400585f,
    1.23676264f, 1.55059743f, 1.9459101f, 2.51230574f, 3.66356158f
};

#define TAP_SWIPE_V6_RANK10_TRIAL032_EARLY_EXIT_GROUP_SIZE 4u
#define TAP_SWIPE_V6_RANK10_TRIAL032_EARLY_EXIT_GROUPS 5u
static const float tap_swipe_v6_rank10_trial032_remaining_min[6] = {
    -2.53544021f, -1.83880079f, -1.3117044f, -0.873078644f, -0.432862818f, 0.0f
};
static const float tap_swipe_v6_rank10_trial032_remaining_max[6] = {
    1.5703131f, 1.23101187f, 0.908494115f, 0.621367872f, 0.284758627f, 0.0f
};

TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE float tap_swipe_v6_rank10_trial032_walk_lgbm_tree(
    uint16_t root_index,
    const t_core* p_core,
    float normalization_factor,
    tap_swipe_v6_rank10_trial032_feature_cache_t* cache)
{
    uint8_t node_index = 0u;
    for (;;) {
        const uint32_t absolute_index = (uint32_t)root_index + (uint32_t)node_index;
        const uint8_t feature_metadata = tap_swipe_v6_rank10_trial032_node_features[absolute_index];
        if (TAP_SWIPE_V6_RANK10_TRIAL032_UNLIKELY(feature_metadata == 0xFFu)) {
            return tap_swipe_v6_rank10_trial032_node_thresholds[absolute_index];
        }
        const uint8_t feature_index = feature_metadata;

        float feature_value =
            tap_swipe_v6_rank10_trial032_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;

        if (feature_value >= -1.0e-35f && feature_value <= 1.0e-35f) {
            feature_value = 0.0f;
        }
        const uint8_t child_flags = tap_swipe_v6_rank10_trial032_node_child_flags[absolute_index];
        const uint8_t right_child = (uint8_t)(child_flags & 0x3Fu);
        const uint8_t default_left = (uint8_t)((child_flags >> 6) & 0x01u);
        const uint8_t missing_is_zero = (uint8_t)(child_flags >> 7);

        const uint8_t is_missing = (uint8_t)(
            (feature_value != feature_value) || ((missing_is_zero != 0u) && feature_value == 0.0f));

        if (TAP_SWIPE_V6_RANK10_TRIAL032_UNLIKELY(is_missing != 0u)) {
            node_index = default_left != 0u
                ? (uint8_t)(node_index + 1u)
                : right_child;
        } else {
            node_index = feature_value <= tap_swipe_v6_rank10_trial032_node_thresholds[absolute_index]
                ? (uint8_t)(node_index + 1u)
                : right_child;
        }
    }
}

/** Return boosted-tree raw score before the logistic link. */
TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE float tap_swipe_v6_rank10_trial032_predict_raw_score(const t_core* p_core)
{
    tap_swipe_v6_rank10_trial032_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank10_trial032_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t tree_index = 0u; tree_index < TAP_SWIPE_V6_RANK10_TRIAL032_NUM_TREES; ++tree_index) {
        raw_score += tap_swipe_v6_rank10_trial032_walk_lgbm_tree(tap_swipe_v6_rank10_trial032_tree_roots[tree_index], p_core, normalization_factor, &cache);
    }
    return raw_score;
}

/** Convert a raw score to nearest 0.05 probability, returned as Q20 [0..20]. */
TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE uint8_t tap_swipe_v6_rank10_trial032_probability_q20_from_raw(float raw_score)
{
    uint8_t low = 0u;
    uint8_t high = 20u;
    while (low < high) {
        const uint8_t middle = (uint8_t)((low + high) >> 1u);
        if (raw_score < tap_swipe_v6_rank10_trial032_q20_raw_midpoints[middle]) {
            high = middle;
        } else {
            low = (uint8_t)(middle + 1u);
        }
    }
    return low;
}

TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE uint8_t tap_swipe_v6_rank10_trial032_probability_q20(const t_core* p_core)
{
    return tap_swipe_v6_rank10_trial032_probability_q20_from_raw(tap_swipe_v6_rank10_trial032_predict_raw_score(p_core));
}

/** Approximate probability in 0.05 steps; maximum rounding error is 0.025. */
TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE float tap_swipe_v6_rank10_trial032_predict_value(const t_core* p_core)
{
    return 0.05f * (float)tap_swipe_v6_rank10_trial032_probability_q20(p_core);
}

/** Runtime-selectable exact threshold for Q20 values 0.05..0.95; no expf/logf. */
TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE int8_t tap_swipe_v6_rank10_trial032_predict_q20(    const t_core* p_core, uint8_t threshold_q20)
{
    if (threshold_q20 == 0u) { return 1; }
    if (threshold_q20 >= 20u) { return 0; }
    const float raw_threshold = tap_swipe_v6_rank10_trial032_q20_raw_thresholds[threshold_q20 - 1u];
    tap_swipe_v6_rank10_trial032_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank10_trial032_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t group = 0u; group < TAP_SWIPE_V6_RANK10_TRIAL032_EARLY_EXIT_GROUPS; ++group) {
        const uint16_t first_tree = (uint16_t)(group * TAP_SWIPE_V6_RANK10_TRIAL032_EARLY_EXIT_GROUP_SIZE);
        uint16_t end_tree = (uint16_t)(first_tree + TAP_SWIPE_V6_RANK10_TRIAL032_EARLY_EXIT_GROUP_SIZE);
        if (end_tree > TAP_SWIPE_V6_RANK10_TRIAL032_NUM_TREES) { end_tree = TAP_SWIPE_V6_RANK10_TRIAL032_NUM_TREES; }
        for (uint16_t tree_index = first_tree; tree_index < end_tree; ++tree_index) {
            raw_score += tap_swipe_v6_rank10_trial032_walk_lgbm_tree(tap_swipe_v6_rank10_trial032_tree_roots[tree_index], p_core, normalization_factor, &cache);
        }
        if (raw_score + tap_swipe_v6_rank10_trial032_remaining_max[group + 1u] < raw_threshold) { return 0; }
        if (raw_score + tap_swipe_v6_rank10_trial032_remaining_min[group + 1u] >= raw_threshold) { return 1; }
    }
    return raw_score >= raw_threshold ? 1 : 0;
}

#define TAP_SWIPE_V6_RANK10_TRIAL032_DEFAULT_THRESHOLD_Q20 9u
TAP_SWIPE_V6_RANK10_TRIAL032_FORCE_INLINE int8_t tap_swipe_v6_rank10_trial032_predict(const t_core* p_core)
{
    return tap_swipe_v6_rank10_trial032_predict_q20(p_core, TAP_SWIPE_V6_RANK10_TRIAL032_DEFAULT_THRESHOLD_Q20);
}

#endif /* TAP_SWIPE_V6_RANK10_TRIAL032_MODEL_H */
