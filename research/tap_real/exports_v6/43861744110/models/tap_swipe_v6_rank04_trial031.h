#ifndef TAP_SWIPE_V6_RANK04_TRIAL031_MODEL_H
#define TAP_SWIPE_V6_RANK04_TRIAL031_MODEL_H

/*
 * Generated: 2026-11-16T11:57:10.308655+00:00
 * Model platform: LGBM
 * Task: BINARY_CLASSIFIER
 * Binary target class_id: 1
 * Training preprocessing: none
 * Firmware preprocessing: raw features; no preprocessing
 * Original selected feature indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
 * Required feature helpers/macros: none
 * Export profile: cortex_m4f_balanced
 * Flash storage: exact_float32_soa_bitpacked_flags; 1760 tree-data bytes.
 * Total generated static model data: 2004 bytes.
 * Metadata layout: lgbm_u6_child_flags_packed
 * Feature metadata: plain_u8; default_left=child_bit6; missing_is_zero=child_bit7; flag bytes=0.
 */

#include <math.h>
#include <stdint.h>
#include "core_type_def.h"

#define TAP_SWIPE_V6_RANK04_TRIAL031_NUM_FEATURES 36u
#define TAP_SWIPE_V6_RANK04_TRIAL031_NUM_TREES 40u
#define TAP_SWIPE_V6_RANK04_TRIAL031_THRESHOLD 0.449999988f
#define TAP_SWIPE_V6_RANK04_TRIAL031_TARGET_CLASS_ID 1
#define TAP_SWIPE_V6_RANK04_TRIAL031_PREPROCESSING "none"

#ifndef TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE
  #if defined(__STATIC_FORCEINLINE)
    #define TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE __STATIC_FORCEINLINE
  #elif defined(__GNUC__) || defined(__clang__)
    #define TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE static inline __attribute__((always_inline))
  #else
    #define TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE static inline
  #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define TAP_SWIPE_V6_RANK04_TRIAL031_LIKELY(value)   __builtin_expect(!!(value), 1)
  #define TAP_SWIPE_V6_RANK04_TRIAL031_UNLIKELY(value) __builtin_expect(!!(value), 0)
#else
  #define TAP_SWIPE_V6_RANK04_TRIAL031_LIKELY(value)   (value)
  #define TAP_SWIPE_V6_RANK04_TRIAL031_UNLIKELY(value) (value)
#endif

#define TAP_SWIPE_V6_RANK04_TRIAL031_HELPER_CACHE_COUNT 0u
#define TAP_SWIPE_V6_RANK04_TRIAL031_HELPER_CACHE_BYTES 1u
typedef struct {
    uint8_t unused;
} tap_swipe_v6_rank04_trial031_feature_cache_t;

/** Return one selected feature; helper expressions are cached lazily once per prediction. */
TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE float tap_swipe_v6_rank04_trial031_get_raw_model_feature(
    const t_core* p_core, uint16_t model_feature_index, tap_swipe_v6_rank04_trial031_feature_cache_t* cache)
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

TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE float tap_swipe_v6_rank04_trial031_runtime_normalization_factor(
    const t_core* p_core, tap_swipe_v6_rank04_trial031_feature_cache_t* cache)
{
    (void)p_core;
    (void)cache;
    return 1.0f;
}

static const float tap_swipe_v6_rank04_trial031_node_thresholds[280] = {
    456.5f, 20.5f, -0.0603070147f, -0.244916722f, 113.5f, 0.0979061201f,
    -0.0496597141f, 542.5f, 34.5f, -0.00861050468f, -0.191388816f, 178.5f,
    0.0159116704f, 0.159516826f, 401.5f, 51.5f, -0.0819316059f, -0.201182544f,
    126.5f, 0.0752971098f, -0.0613557622f, 678.5f, 15.5f, -0.0164577626f,
    -0.177586272f, 363.5f, 0.0188956559f, 0.155266538f, 690.5f, 10.5f,
    -0.00287720188f, -0.149995074f, 183.5f, 0.0689016432f, -0.0676799491f, 366.5f,
    48.5f, -0.0880717337f, -0.174087778f, 47.5f, 0.0545415357f, -0.044339966f,
    54.5f, 389.5f, -0.0806020796f, 0.0710359514f, 1488.5f, -0.149876088f,
    0.0255054254f, 292.5f, 37.5f, 0.00142054271f, 0.149576887f, 139.5f,
    -0.154287234f, 0.0303997695f, 22.5f, 341.5f, -0.0872750059f, 0.0712038204f,
    667.5f, -0.117957272f, 0.0326328576f, 757.5f, 14.5f, 0.0289556701f,
    -0.118034758f, 1555.5f, 0.0931151956f, -0.0256590005f, 201.5f, 65.5f,
    0.0240797084f, -0.102468215f, 2925.5f, 0.126788482f, -0.144621536f, 1191.5f,
    26.5f, 0.0258454252f, -0.0869240016f, 74.5f, 0.0131143956f, 0.126816019f,
    632.5f, 47.5f, -0.00533632003f, 0.126026869f, 325.5f, -0.146068409f,
    0.015262411f, 590.5f, 17.5f, -0.00519719953f, -0.120851271f, 1114.5f,
    0.0814238265f, -0.0140937455f, 341.5f, 16.5f, -0.0598577f, -0.143215463f,
    249.5f, -0.00746685313f, 0.0901141614f, 1106.5f, 125.5f, -0.00898280367f,
    0.109682053f, 790.5f, -0.165620178f, -0.0216148607f, 10.5f, 389.5f,
    -0.0388716906f, 0.0721209124f, 339.5f, -0.0886809006f, 0.0127021177f, 226.5f,
    58.5f, 0.00109793688f, 0.107717089f, 591.5f, -0.10564287f, 0.0403374508f,
    1531.5f, 37.5f, -0.00765721966f, 0.097668618f, 710.5f, -0.149530917f,
    -0.027088277f, 316.5f, 4193.5f, -0.124929145f, 0.166397795f, 6.5f,
    0.0619003065f, -0.011077188f, 740.5f, 64.5f, -0.0084277736f, 0.0867498294f,
    3938.5f, -0.157875121f, -0.0269219428f, 101.5f, 599.5f, -0.0234393999f,
    0.0466216728f, 2361.5f, -0.0984517857f, 0.0144604156f, 4734.0f, 292.5f,
    -0.00917780399f, 0.0854534954f, 8296.0f, -0.119222037f, -0.201670438f, 38.5f,
    678.5f, -0.00785068143f, 0.0578539111f, 1671.5f, -0.0688010827f, 0.0313502178f,
    316.5f, 3.5f, -0.0729216337f, -0.148221359f, 399.5f, 0.0199494585f,
    -0.0508726239f, 6.5f, 91.5f, -0.0618635491f, 0.0547627397f, 22.5f,
    -0.0423578359f, 0.0370336585f, 896.5f, 154.5f, -0.00889284909f, 0.0717971697f,
    5004.0f, -0.182512164f, -0.0669661909f, 144.5f, 15.5f, -0.00614053104f,
    0.0767868459f, 790.5f, -0.0938871801f, 0.0287636984f, 10029.5f, 1329.5f,
    -0.0159670021f, 0.0493944287f, 10681.5f, -0.0984326825f, 0.124075755f, 18.5f,
    1.00000002e-35f, -0.0206671469f, 0.0498852916f, 301.5f, -0.0793349221f, 0.0047347974f,
    192.5f, 61.5f, 0.00409814715f, 0.0850693658f, 2340.5f, -0.0847218484f,
    0.0257083438f, 303.5f, 4403.5f, -0.112017632f, 0.156039402f, 82.5f,
    -0.00526250247f, 0.059893135f, 533.5f, 54.5f, -0.00636386499f, 0.0689162239f,
    801.5f, -0.125783429f, -0.00317027862f, 5.5f, 1502.5f, 0.0245922059f,
    0.0996020585f, 576.5f, -0.0717095509f, 0.00142906443f, 1868.0f, 412.5f,
    -0.0545569062f, 0.0112860473f, 3569.5f, -0.158072278f, -0.209833175f, 30.5f,
    8.5f, 0.00396077428f, 0.0805912912f, 2462.5f, -0.0723925382f, 0.0123888599f,
    6021.5f, 98.5f, -0.00605187053f, 0.0600007623f, 677.5f, -0.115146376f,
    -0.18081367f, 10029.5f, 533.5f, -0.0137109263f, 0.0359620973f, 10681.5f,
    -0.0858893096f, 0.119220637f, 26.5f, 1.00000002e-35f, -0.0214633159f, 0.0390397683f,
    43.5f, -0.0509864427f, 0.0193761215f, 178.5f, 24.5f, 0.00315889716f,
    0.0685519949f, 245.5f, -0.0794721022f, 0.012317677f
};
static const uint16_t tap_swipe_v6_rank04_trial031_tree_roots[40] = {
    0u, 7u, 14u, 21u, 28u, 35u, 42u, 49u, 56u, 63u, 70u, 77u,
    84u, 91u, 98u, 105u, 112u, 119u, 126u, 133u, 140u, 147u, 154u, 161u,
    168u, 175u, 182u, 189u, 196u, 203u, 210u, 217u, 224u, 231u, 238u, 245u,
    252u, 259u, 266u, 273u
};
static const uint8_t tap_swipe_v6_rank04_trial031_node_child_flags[280] = {
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
    0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u,
    0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u,
    0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u,
    0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u,
    0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u,
    0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u,
    0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u,
    0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u,
    0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u,
    0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u,
    0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u,
    0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u,
    0x00u, 0x46u, 0x00u, 0x00u
};
static const uint8_t tap_swipe_v6_rank04_trial031_node_features[280] = {
    15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 33u,
    0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu,
    0xFFu, 34u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu, 0xFFu, 24u, 0xFFu, 0xFFu, 15u,
    35u, 0xFFu, 0xFFu, 24u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 2u, 0xFFu,
    0xFFu, 30u, 22u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu,
    3u, 0xFFu, 0xFFu, 2u, 30u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 34u, 30u,
    0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 3u, 30u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu,
    30u, 28u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 2u, 24u, 0xFFu, 0xFFu, 10u,
    0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu,
    0xFFu, 34u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u,
    34u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 21u, 0xFFu, 0xFFu, 28u, 0xFFu,
    0xFFu, 15u, 16u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 24u, 28u, 0xFFu, 0xFFu,
    3u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 30u, 34u,
    0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu,
    15u, 34u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 30u, 12u, 0xFFu, 0xFFu, 22u,
    0xFFu, 0xFFu, 18u, 34u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu,
    0xFFu, 34u, 0xFFu, 0xFFu, 5u, 7u, 0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu, 30u,
    18u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 3u, 0xFFu,
    0xFFu, 15u, 16u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu,
    28u, 0xFFu, 0xFFu, 30u, 12u, 0xFFu, 0xFFu, 15u, 0xFFu, 0xFFu, 18u, 15u,
    0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu,
    30u, 22u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 5u, 8u, 0xFFu, 0xFFu, 16u,
    0xFFu, 0xFFu, 30u, 18u, 0xFFu, 0xFFu, 19u, 0xFFu, 0xFFu, 30u, 19u, 0xFFu,
    0xFFu, 32u, 0xFFu, 0xFFu
};

/* Q20 probability LUT: one unit equals 0.05 probability. */
static const float tap_swipe_v6_rank04_trial031_q20_raw_thresholds[19] = {
    -2.94443893f, -2.19722462f, -1.73460102f, -1.38629436f, -1.09861231f,
    -0.847297847f, -0.619039237f, -0.405465096f, -0.200670689f, 0.0f,
    0.200670689f, 0.405465096f, 0.619039237f, 0.847297847f, 1.09861231f,
    1.38629436f, 1.73460102f, 2.19722462f, 2.94443893f
};
static const float tap_swipe_v6_rank04_trial031_q20_raw_midpoints[20] = {
    -3.66356158f, -2.51230574f, -1.9459101f, -1.55059743f, -1.23676264f,
    -0.969400585f, -0.730887532f, -0.510825634f, -0.302280873f, -0.100083455f,
    0.100083455f, 0.302280873f, 0.510825634f, 0.730887532f, 0.969400585f,
    1.23676264f, 1.55059743f, 1.9459101f, 2.51230574f, 3.66356158f
};

#define TAP_SWIPE_V6_RANK04_TRIAL031_EARLY_EXIT_GROUP_SIZE 4u
#define TAP_SWIPE_V6_RANK04_TRIAL031_EARLY_EXIT_GROUPS 10u
static const float tap_swipe_v6_rank04_trial031_remaining_min[11] = {
    -5.22029686f, -4.40516329f, -3.77687097f, -3.30929685f, -2.73351264f, -2.26470733f,
    -1.73789346f, -1.25139868f, -0.876885116f, -0.397162855f, 0.0f
};
static const float tap_swipe_v6_rank04_trial031_remaining_max[11] = {
    3.5635469f, 3.0755167f, 2.73142552f, 2.3134737f, 1.90620458f, 1.46228361f,
    1.18559301f, 0.962288439f, 0.547213018f, 0.286814392f, 0.0f
};

TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE float tap_swipe_v6_rank04_trial031_walk_lgbm_tree(
    uint16_t root_index,
    const t_core* p_core,
    float normalization_factor,
    tap_swipe_v6_rank04_trial031_feature_cache_t* cache)
{
    uint8_t node_index = 0u;
    for (;;) {
        const uint32_t absolute_index = (uint32_t)root_index + (uint32_t)node_index;
        const uint8_t feature_metadata = tap_swipe_v6_rank04_trial031_node_features[absolute_index];
        if (TAP_SWIPE_V6_RANK04_TRIAL031_UNLIKELY(feature_metadata == 0xFFu)) {
            return tap_swipe_v6_rank04_trial031_node_thresholds[absolute_index];
        }
        const uint8_t feature_index = feature_metadata;

        float feature_value =
            tap_swipe_v6_rank04_trial031_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;

        if (feature_value >= -1.0e-35f && feature_value <= 1.0e-35f) {
            feature_value = 0.0f;
        }
        const uint8_t child_flags = tap_swipe_v6_rank04_trial031_node_child_flags[absolute_index];
        const uint8_t right_child = (uint8_t)(child_flags & 0x3Fu);
        const uint8_t default_left = (uint8_t)((child_flags >> 6) & 0x01u);
        const uint8_t missing_is_zero = (uint8_t)(child_flags >> 7);

        const uint8_t is_missing = (uint8_t)(
            (feature_value != feature_value) || ((missing_is_zero != 0u) && feature_value == 0.0f));

        if (TAP_SWIPE_V6_RANK04_TRIAL031_UNLIKELY(is_missing != 0u)) {
            node_index = default_left != 0u
                ? (uint8_t)(node_index + 1u)
                : right_child;
        } else {
            node_index = feature_value <= tap_swipe_v6_rank04_trial031_node_thresholds[absolute_index]
                ? (uint8_t)(node_index + 1u)
                : right_child;
        }
    }
}

/** Return boosted-tree raw score before the logistic link. */
TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE float tap_swipe_v6_rank04_trial031_predict_raw_score(const t_core* p_core)
{
    tap_swipe_v6_rank04_trial031_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank04_trial031_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t tree_index = 0u; tree_index < TAP_SWIPE_V6_RANK04_TRIAL031_NUM_TREES; ++tree_index) {
        raw_score += tap_swipe_v6_rank04_trial031_walk_lgbm_tree(tap_swipe_v6_rank04_trial031_tree_roots[tree_index], p_core, normalization_factor, &cache);
    }
    return raw_score;
}

/** Convert a raw score to nearest 0.05 probability, returned as Q20 [0..20]. */
TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE uint8_t tap_swipe_v6_rank04_trial031_probability_q20_from_raw(float raw_score)
{
    uint8_t low = 0u;
    uint8_t high = 20u;
    while (low < high) {
        const uint8_t middle = (uint8_t)((low + high) >> 1u);
        if (raw_score < tap_swipe_v6_rank04_trial031_q20_raw_midpoints[middle]) {
            high = middle;
        } else {
            low = (uint8_t)(middle + 1u);
        }
    }
    return low;
}

TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE uint8_t tap_swipe_v6_rank04_trial031_probability_q20(const t_core* p_core)
{
    return tap_swipe_v6_rank04_trial031_probability_q20_from_raw(tap_swipe_v6_rank04_trial031_predict_raw_score(p_core));
}

/** Approximate probability in 0.05 steps; maximum rounding error is 0.025. */
TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE float tap_swipe_v6_rank04_trial031_predict_value(const t_core* p_core)
{
    return 0.05f * (float)tap_swipe_v6_rank04_trial031_probability_q20(p_core);
}

/** Runtime-selectable exact threshold for Q20 values 0.05..0.95; no expf/logf. */
TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE int8_t tap_swipe_v6_rank04_trial031_predict_q20(    const t_core* p_core, uint8_t threshold_q20)
{
    if (threshold_q20 == 0u) { return 1; }
    if (threshold_q20 >= 20u) { return 0; }
    const float raw_threshold = tap_swipe_v6_rank04_trial031_q20_raw_thresholds[threshold_q20 - 1u];
    tap_swipe_v6_rank04_trial031_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank04_trial031_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t group = 0u; group < TAP_SWIPE_V6_RANK04_TRIAL031_EARLY_EXIT_GROUPS; ++group) {
        const uint16_t first_tree = (uint16_t)(group * TAP_SWIPE_V6_RANK04_TRIAL031_EARLY_EXIT_GROUP_SIZE);
        uint16_t end_tree = (uint16_t)(first_tree + TAP_SWIPE_V6_RANK04_TRIAL031_EARLY_EXIT_GROUP_SIZE);
        if (end_tree > TAP_SWIPE_V6_RANK04_TRIAL031_NUM_TREES) { end_tree = TAP_SWIPE_V6_RANK04_TRIAL031_NUM_TREES; }
        for (uint16_t tree_index = first_tree; tree_index < end_tree; ++tree_index) {
            raw_score += tap_swipe_v6_rank04_trial031_walk_lgbm_tree(tap_swipe_v6_rank04_trial031_tree_roots[tree_index], p_core, normalization_factor, &cache);
        }
        if (raw_score + tap_swipe_v6_rank04_trial031_remaining_max[group + 1u] < raw_threshold) { return 0; }
        if (raw_score + tap_swipe_v6_rank04_trial031_remaining_min[group + 1u] >= raw_threshold) { return 1; }
    }
    return raw_score >= raw_threshold ? 1 : 0;
}

#define TAP_SWIPE_V6_RANK04_TRIAL031_DEFAULT_THRESHOLD_Q20 9u
TAP_SWIPE_V6_RANK04_TRIAL031_FORCE_INLINE int8_t tap_swipe_v6_rank04_trial031_predict(const t_core* p_core)
{
    return tap_swipe_v6_rank04_trial031_predict_q20(p_core, TAP_SWIPE_V6_RANK04_TRIAL031_DEFAULT_THRESHOLD_Q20);
}

#endif /* TAP_SWIPE_V6_RANK04_TRIAL031_MODEL_H */
