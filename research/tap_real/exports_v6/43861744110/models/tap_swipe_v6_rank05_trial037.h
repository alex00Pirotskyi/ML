#ifndef TAP_SWIPE_V6_RANK05_TRIAL037_MODEL_H
#define TAP_SWIPE_V6_RANK05_TRIAL037_MODEL_H

/*
 * Generated: 2026-11-16T11:57:12.485072+00:00
 * Model platform: LGBM
 * Task: BINARY_CLASSIFIER
 * Binary target class_id: 1
 * Training preprocessing: none
 * Firmware preprocessing: raw features; no preprocessing
 * Original selected feature indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
 * Required feature helpers/macros: none
 * Export profile: cortex_m4f_balanced
 * Flash storage: exact_float32_soa_bitpacked_flags; 3360 tree-data bytes.
 * Total generated static model data: 3644 bytes.
 * Metadata layout: lgbm_u6_child_flags_packed
 * Feature metadata: plain_u8; default_left=child_bit6; missing_is_zero=child_bit7; flag bytes=0.
 */

#include <math.h>
#include <stdint.h>
#include "core_type_def.h"

#define TAP_SWIPE_V6_RANK05_TRIAL037_NUM_FEATURES 36u
#define TAP_SWIPE_V6_RANK05_TRIAL037_NUM_TREES 60u
#define TAP_SWIPE_V6_RANK05_TRIAL037_THRESHOLD 0.5f
#define TAP_SWIPE_V6_RANK05_TRIAL037_TARGET_CLASS_ID 1
#define TAP_SWIPE_V6_RANK05_TRIAL037_PREPROCESSING "none"

#ifndef TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE
  #if defined(__STATIC_FORCEINLINE)
    #define TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE __STATIC_FORCEINLINE
  #elif defined(__GNUC__) || defined(__clang__)
    #define TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE static inline __attribute__((always_inline))
  #else
    #define TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE static inline
  #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define TAP_SWIPE_V6_RANK05_TRIAL037_LIKELY(value)   __builtin_expect(!!(value), 1)
  #define TAP_SWIPE_V6_RANK05_TRIAL037_UNLIKELY(value) __builtin_expect(!!(value), 0)
#else
  #define TAP_SWIPE_V6_RANK05_TRIAL037_LIKELY(value)   (value)
  #define TAP_SWIPE_V6_RANK05_TRIAL037_UNLIKELY(value) (value)
#endif

#define TAP_SWIPE_V6_RANK05_TRIAL037_HELPER_CACHE_COUNT 0u
#define TAP_SWIPE_V6_RANK05_TRIAL037_HELPER_CACHE_BYTES 1u
typedef struct {
    uint8_t unused;
} tap_swipe_v6_rank05_trial037_feature_cache_t;

/** Return one selected feature; helper expressions are cached lazily once per prediction. */
TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE float tap_swipe_v6_rank05_trial037_get_raw_model_feature(
    const t_core* p_core, uint16_t model_feature_index, tap_swipe_v6_rank05_trial037_feature_cache_t* cache)
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

TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE float tap_swipe_v6_rank05_trial037_runtime_normalization_factor(
    const t_core* p_core, tap_swipe_v6_rank05_trial037_feature_cache_t* cache)
{
    (void)p_core;
    (void)cache;
    return 1.0f;
}

static const float tap_swipe_v6_rank05_trial037_node_thresholds[540] = {
    456.5f, 20.5f, -0.0655842349f, -0.262071759f, 113.5f, 0.104835875f,
    249.5f, -0.150344014f, 0.114358589f, 542.5f, 34.5f, -0.00883996394f,
    -0.202795625f, 154.5f, 39.5f, 0.043374408f, 0.21151498f, -0.0635209009f,
    401.5f, -0.157120258f, 48.5f, 49.5f, 0.0769498646f, 0.212790072f,
    211.5f, -0.108069293f, 0.101875715f, 678.5f, 14.5f, -0.0137052378f,
    -0.18518135f, 1486.5f, 0.0951356888f, 663.5f, -0.0594850518f, 0.130204588f,
    696.5f, 15.5f, 354.5f, -0.108794697f, 0.0489619933f, -0.142179623f,
    1858.5f, 0.121606633f, -0.00349776819f, 412.5f, -0.115581371f, 282.5f,
    47.5f, 0.0134504465f, 0.160698488f, 562.5f, -0.154230595f, 0.0630421266f,
    226.5f, 690.5f, 14.5f, 0.0254040416f, -0.111021049f, 0.0773650035f,
    2682.5f, -0.169832334f, 0.00561313005f, 24.5f, 341.5f, -0.0912063122f,
    0.0722564384f, 906.5f, -0.112977535f, 133.5f, -0.00940546487f, 0.131594121f,
    332.5f, 107.5f, 18.5f, 0.0482421219f, -0.0480564944f, 0.143377438f,
    1839.5f, -0.152093619f, -0.00011823995f, 1531.5f, 757.5f, 9.5f,
    0.0378825292f, -0.0995637625f, 181.5f, 0.0233413856f, 0.16520451f, -0.140719861f,
    136.5f, 464.5f, -0.0166912191f, 0.0658752695f, 403.5f, -0.104517721f,
    4734.0f, 0.0881101117f, -0.196105853f, 632.5f, 79.5f, 26.5f,
    0.0315450393f, -0.0577635057f, 239.5f, -0.141044557f, 0.138736859f, -0.0904940069f,
    125.5f, -0.089279905f, 292.5f, 15.5f, 0.0107393339f, 0.113887869f,
    138.5f, -0.119662262f, 0.048196435f, 590.5f, 25.5f, -0.0041444134f,
    -0.133824155f, 10029.5f, 47.5f, 0.0216184817f, 0.12802121f, -0.0780849233f,
    316.5f, -0.127082974f, 6.5f, 1193.5f, 0.0532673113f, 0.124354057f,
    1427.5f, -0.0541514121f, 0.0271896441f, 2925.5f, 292.5f, 65.5f,
    0.0207040068f, -0.0644656643f, 400.5f, -0.135884583f, 0.136297077f, -0.146185592f,
    366.5f, 6.5f, -0.029502809f, -0.129508346f, 740.5f, 53.5f,
    0.00134254934f, 0.105840765f, -0.114427529f, 510.5f, 39.5f, 9.5f,
    0.0374577567f, -0.03558103f, 0.0838119835f, 790.5f, -0.118590333f, 0.0209626835f,
    677.5f, 769.5f, 74.5f, -0.00601847237f, -0.114721626f, 139.5f,
    0.0146484328f, 0.130008474f, -0.142458975f, 303.5f, -0.119920239f, 47.5f,
    9.5f, 0.00820923876f, 0.0972359776f, 1839.5f, -0.0862643644f, 0.0273590107f,
    192.5f, 27.5f, 4.5f, 0.0454203822f, -0.0191208199f, 0.079240486f,
    2829.5f, -0.0870930031f, 0.0315010548f, 6021.5f, 729.5f, 54.5f,
    -0.00813634135f, -0.100452296f, 591.5f, 0.010639865f, 0.121953554f, -0.171478838f,
    316.5f, -0.0991551206f, 101.5f, 55.5f, 0.0122958096f, 0.108575329f,
    93.5f, -0.071469754f, 0.0251547396f, 905.5f, 54.5f, 13.5f,
    0.0262444802f, -0.0283080693f, 284.5f, -0.106562428f, 0.107139446f, -0.0723957419f,
    1301.5f, 588.5f, 28.5f, -0.0108665498f, -0.0932106897f, 790.5f,
    0.00843686238f, 0.120748915f, -0.165220171f, 39.5f, 12.5f, 0.00587791903f,
    0.0854437351f, 1928.5f, -0.0711548105f, 10029.5f, 0.0578159764f, -0.0665448755f,
    289.5f, -0.113723248f, 1300.5f, 233.5f, 0.00219971244f, 0.0908176079f,
    2505.5f, -0.118462406f, 0.0356280915f, 98.5f, 10.5f, -0.0023099652f,
    0.0592674427f, 790.5f, -0.068873629f, 1868.0f, 0.0668887943f, -0.175246105f,
    6620.5f, 1515.5f, 56.5f, 0.00499371346f, -0.0632638335f, 0.0576630123f,
    10681.5f, -0.0636511296f, 0.160326868f, 2.5f, 0.0489744768f, 668.5f,
    97.5f, -0.0232787263f, -0.104320914f, 728.5f, 0.0710534975f, -0.0192675982f,
    91.5f, -0.0777167603f, 74.5f, 1798.5f, 0.0517812707f, 0.203619927f,
    67.5f, -0.0105820531f, 0.0482917763f, 389.5f, 7.5f, -0.0315095186f,
    -0.128798366f, 124.5f, 0.0602075756f, 47.5f, -0.0131160831f, 0.0448665619f,
    1301.5f, 1.00000002e-35f, 1502.5f, -0.0660357699f, 0.0347798578f, 5.5f,
    0.0579650588f, -0.000663683866f, -0.115021385f, 2320.0f, 296.5f, 420.5f,
    0.00452314643f, -0.0746093541f, 118.5f, -0.176068082f, 0.103024192f, -0.146077916f,
    454.5f, 72.5f, 192.5f, 0.00308421277f, -0.0679193661f, 136.5f,
    -0.103157319f, 0.0714637116f, -0.0872209445f, 554.5f, 94.5f, 0.023011297f,
    -0.0333624221f, 1114.5f, 0.0724464282f, 10029.5f, 0.00822770782f, -0.0766465738f,
    6021.5f, 86.5f, 380.5f, 0.00247683702f, -0.0794158354f, 446.5f,
    -0.144923776f, 0.0698743165f, -0.13808544f, 8766.5f, 533.5f, 233.5f,
    -0.00491145859f, 0.0579442233f, 1026.5f, -0.117194302f, 0.00211113878f, 0.0867412388f,
    107.5f, 182.5f, -0.0794771612f, 0.0707575679f, 3.5f, 0.043538224f,
    2579.5f, -0.0162296072f, 0.0317889564f, 1868.0f, 1.00000002e-35f, 2.5f,
    0.00011641108f, -0.0715474263f, 9.5f, 0.0367580317f, -0.0099532418f, -0.157032788f,
    678.5f, 1067.5f, 46.5f, 0.0426026098f, -0.050630141f, 0.0626685172f,
    5729.5f, 0.0248724464f, -0.0335139483f, 10396.0f, 3358.5f, 2251.5f,
    -0.00268585002f, 0.0702640489f, 2505.5f, -0.0509701855f, 0.0656143874f, 0.129790589f,
    273.5f, -0.108035177f, 89.5f, 130.5f, 0.0119159771f, 0.101906218f,
    148.5f, -0.0779199079f, 0.00192442746f, 1.5f, 1381.5f, 0.0143149011f,
    0.116193727f, 264.5f, -0.0358236842f, 630.5f, 0.0393841006f, -0.018757822f,
    10124.5f, 63.5f, 31.5f, -0.00202459004f, 0.0683865696f, 368.5f,
    -0.100468315f, -0.0220319126f, 0.0545489378f, 10681.5f, 273.5f, -0.103356183f,
    100.5f, 0.0350775495f, -0.00611000042f, 1386.5f, 0.225383833f, 0.0293186195f,
    91.5f, -0.0611147135f, 1114.5f, 1584.5f, 0.000432829489f, 0.073757112f,
    1493.5f, -0.050412938f, 0.00957392156f, 256.5f, 485.5f, 0.000860656786f,
    -0.0720100254f, 384.5f, -0.130259156f, 1868.0f, 0.0937069803f, -0.130310953f,
    1.00000002e-35f, -0.0352575891f, 263.5f, 729.5f, 0.00350401644f, 0.07855919f,
    13229.0f, 0.000809825142f, -0.0710734874f, 8176.0f, 10396.0f, 12583.5f,
    -0.000768464059f, -0.079545103f, 15916.5f, 0.169777632f, -0.165244415f, 0.141816571f,
    2.5f, 0.0331804641f, 2640.5f, 3768.5f, -0.0240440071f, 0.084760718f,
    815.5f, 0.0991665944f, 0.000153963221f, 896.5f, 143.5f, 5.5f,
    0.0160893705f, -0.0180152617f, 322.5f, -0.0859629065f, 0.0689179301f, -0.0996434093f,
    46.5f, 115.5f, -0.0578213781f, 0.071841605f, 98.5f, -0.0717146397f,
    221.5f, -0.00534525141f, 0.0380774438f, 8176.0f, 955.5f, 74.5f,
    -0.00416798424f, 0.0489900149f, 2003.5f, -0.159959435f, -0.0471637584f, 0.12685056f,
    70.5f, 1032.5f, 0.0221263114f, 0.152041361f, 415.5f, -0.0719014704f,
    74.5f, 0.0820387602f, -0.0029139223f, 366.5f, 534.5f, 36.5f,
    0.0105764549f, -0.0237587839f, 491.5f, -0.150176838f, 0.0752338395f, -0.065478988f,
    9333.0f, 336.5f, 19.5f, -0.0063761468f, 0.0292881243f, 2505.5f,
    -0.122744881f, 0.0205493327f, 0.0691632032f, 6021.5f, 852.5f, 632.5f,
    0.00223730854f, -0.0537607707f, 715.5f, -0.137828872f, 0.120730802f, -0.119532101f,
    8357.0f, 10029.5f, 2.5f, -0.0143465409f, 0.0236634221f, -0.0603568479f,
    7806.5f, 0.145940199f, -0.0861020982f, 54.5f, 0.0441246442f, 154.5f,
    1004.5f, -0.0592626333f, 0.0707938895f, 6006.5f, 0.0101403221f, -0.0291996673f
};
static const uint16_t tap_swipe_v6_rank05_trial037_tree_roots[60] = {
    0u, 9u, 18u, 27u, 36u, 45u, 54u, 63u, 72u, 81u, 90u, 99u,
    108u, 117u, 126u, 135u, 144u, 153u, 162u, 171u, 180u, 189u, 198u, 207u,
    216u, 225u, 234u, 243u, 252u, 261u, 270u, 279u, 288u, 297u, 306u, 315u,
    324u, 333u, 342u, 351u, 360u, 369u, 378u, 387u, 396u, 405u, 414u, 423u,
    432u, 441u, 450u, 459u, 468u, 477u, 486u, 495u, 504u, 513u, 522u, 531u
};
static const uint8_t tap_swipe_v6_rank05_trial037_node_child_flags[540] = {
    0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x48u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u,
    0x00u, 0x48u, 0x47u, 0x00u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x48u, 0x00u, 0x00u,
    0x46u, 0x45u, 0x44u, 0x00u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u,
    0x45u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u, 0x46u, 0x45u, 0x44u, 0x00u, 0x00u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x48u, 0x00u, 0x00u,
    0x46u, 0x45u, 0x44u, 0x00u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u,
    0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u,
    0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u,
    0x00u, 0x48u, 0x47u, 0x00u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u,
    0x44u, 0x43u, 0x00u, 0x00u, 0x48u, 0x47u, 0x00u, 0x00u, 0x00u, 0x46u, 0x45u, 0x44u,
    0x00u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u,
    0x00u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u,
    0x46u, 0x45u, 0x44u, 0x00u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u,
    0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u,
    0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u,
    0x00u, 0x46u, 0x00u, 0x48u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x48u, 0x00u, 0x00u,
    0x46u, 0x45u, 0x44u, 0x00u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u,
    0x45u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x48u, 0x00u, 0x00u,
    0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u,
    0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u,
    0x00u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x48u, 0x00u, 0x00u,
    0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u,
    0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u,
    0x46u, 0x45u, 0x44u, 0x00u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u,
    0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x48u, 0x00u, 0x00u,
    0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x46u, 0x43u, 0x00u,
    0x45u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x48u, 0x00u, 0x00u,
    0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u,
    0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u,
    0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u,
    0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u,
    0x48u, 0x45u, 0x44u, 0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x48u, 0x45u, 0x44u,
    0x00u, 0x00u, 0x47u, 0x00u, 0x00u, 0x00u, 0x46u, 0x45u, 0x44u, 0x00u, 0x00u, 0x00u,
    0x48u, 0x00u, 0x00u, 0x42u, 0x00u, 0x46u, 0x45u, 0x00u, 0x00u, 0x48u, 0x00u, 0x00u
};
static const uint8_t tap_swipe_v6_rank05_trial037_node_features[540] = {
    15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 34u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu,
    0xFFu, 24u, 22u, 0xFFu, 0xFFu, 0xFFu, 15u, 0xFFu, 30u, 34u, 0xFFu, 0xFFu,
    34u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu, 0xFFu, 10u, 0xFFu, 34u, 0xFFu, 0xFFu,
    3u, 24u, 15u, 0xFFu, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 15u, 0xFFu, 24u,
    22u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 30u, 15u, 30u, 0xFFu, 0xFFu, 0xFFu,
    2u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 3u, 0xFFu, 28u, 0xFFu, 0xFFu,
    30u, 34u, 30u, 0xFFu, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 2u, 30u,
    0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 0xFFu, 30u, 2u, 0xFFu, 0xFFu, 34u, 0xFFu,
    30u, 0xFFu, 0xFFu, 30u, 34u, 30u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 0xFFu,
    2u, 0xFFu, 30u, 22u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu, 2u, 24u, 0xFFu,
    0xFFu, 5u, 21u, 0xFFu, 0xFFu, 0xFFu, 15u, 0xFFu, 30u, 13u, 0xFFu, 0xFFu,
    2u, 0xFFu, 0xFFu, 30u, 34u, 30u, 0xFFu, 0xFFu, 7u, 0xFFu, 0xFFu, 0xFFu,
    15u, 30u, 0xFFu, 0xFFu, 24u, 28u, 0xFFu, 0xFFu, 0xFFu, 30u, 19u, 30u,
    0xFFu, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 18u, 15u, 30u, 0xFFu, 0xFFu, 28u,
    0xFFu, 0xFFu, 0xFFu, 15u, 0xFFu, 24u, 22u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu,
    30u, 19u, 30u, 0xFFu, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 15u, 30u,
    0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 0xFFu, 15u, 0xFFu, 30u, 34u, 0xFFu, 0xFFu,
    19u, 0xFFu, 0xFFu, 30u, 22u, 30u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 0xFFu,
    18u, 15u, 30u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 0xFFu, 24u, 20u, 0xFFu,
    0xFFu, 3u, 0xFFu, 5u, 0xFFu, 0xFFu, 15u, 0xFFu, 30u, 32u, 0xFFu, 0xFFu,
    34u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu, 34u, 0xFFu, 18u, 0xFFu, 0xFFu,
    5u, 7u, 30u, 0xFFu, 0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu, 30u, 0xFFu, 2u,
    30u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 12u, 0xFFu, 5u, 17u, 0xFFu, 0xFFu,
    28u, 0xFFu, 0xFFu, 15u, 34u, 0xFFu, 0xFFu, 10u, 0xFFu, 28u, 0xFFu, 0xFFu,
    24u, 18u, 12u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 0xFFu, 24u, 28u, 30u,
    0xFFu, 0xFFu, 8u, 0xFFu, 0xFFu, 0xFFu, 18u, 19u, 30u, 0xFFu, 0xFFu, 3u,
    0xFFu, 0xFFu, 0xFFu, 8u, 5u, 0xFFu, 0xFFu, 10u, 0xFFu, 5u, 0xFFu, 0xFFu,
    30u, 22u, 30u, 0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu, 0xFFu, 3u, 24u, 32u,
    0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 0xFFu, 12u, 8u, 0xFFu, 0xFFu, 30u, 0xFFu,
    3u, 0xFFu, 0xFFu, 18u, 18u, 30u, 0xFFu, 0xFFu, 24u, 0xFFu, 0xFFu, 0xFFu,
    15u, 16u, 5u, 0xFFu, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 17u, 10u, 8u,
    0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 0xFFu, 15u, 0xFFu, 10u, 0u, 0xFFu, 0xFFu,
    12u, 0xFFu, 0xFFu, 30u, 14u, 0xFFu, 0xFFu, 1u, 0xFFu, 10u, 0xFFu, 0xFFu,
    2u, 24u, 19u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 0xFFu, 16u, 15u, 0xFFu,
    10u, 0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 12u, 0xFFu, 10u, 7u, 0xFFu, 0xFFu,
    13u, 0xFFu, 0xFFu, 28u, 30u, 0xFFu, 0xFFu, 7u, 0xFFu, 18u, 0xFFu, 0xFFu,
    18u, 0xFFu, 10u, 15u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 15u, 17u, 10u,
    0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 0xFFu, 30u, 0xFFu, 7u, 3u, 0xFFu, 0xFFu,
    10u, 0xFFu, 0xFFu, 18u, 19u, 24u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 0xFFu,
    5u, 14u, 0xFFu, 0xFFu, 7u, 0xFFu, 32u, 0xFFu, 0xFFu, 15u, 24u, 22u,
    0xFFu, 0xFFu, 7u, 0xFFu, 0xFFu, 0xFFu, 5u, 13u, 0xFFu, 0xFFu, 16u, 0xFFu,
    10u, 0xFFu, 0xFFu, 18u, 31u, 30u, 0xFFu, 0xFFu, 9u, 0xFFu, 0xFFu, 0xFFu,
    3u, 18u, 20u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 0xFFu, 30u, 28u, 30u,
    0xFFu, 0xFFu, 7u, 0xFFu, 0xFFu, 0xFFu, 16u, 5u, 22u, 0xFFu, 0xFFu, 0xFFu,
    10u, 0xFFu, 0xFFu, 5u, 0xFFu, 1u, 16u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu
};

/* Q20 probability LUT: one unit equals 0.05 probability. */
static const float tap_swipe_v6_rank05_trial037_q20_raw_thresholds[19] = {
    -2.94443893f, -2.19722462f, -1.73460102f, -1.38629436f, -1.09861231f,
    -0.847297847f, -0.619039237f, -0.405465096f, -0.200670689f, 0.0f,
    0.200670689f, 0.405465096f, 0.619039237f, 0.847297847f, 1.09861231f,
    1.38629436f, 1.73460102f, 2.19722462f, 2.94443893f
};
static const float tap_swipe_v6_rank05_trial037_q20_raw_midpoints[20] = {
    -3.66356158f, -2.51230574f, -1.9459101f, -1.55059743f, -1.23676264f,
    -0.969400585f, -0.730887532f, -0.510825634f, -0.302280873f, -0.100083455f,
    0.100083455f, 0.302280873f, 0.510825634f, 0.730887532f, 0.969400585f,
    1.23676264f, 1.55059743f, 1.9459101f, 2.51230574f, 3.66356158f
};

#define TAP_SWIPE_V6_RANK05_TRIAL037_EARLY_EXIT_GROUP_SIZE 4u
#define TAP_SWIPE_V6_RANK05_TRIAL037_EARLY_EXIT_GROUPS 15u
static const float tap_swipe_v6_rank05_trial037_remaining_min[16] = {
    -7.25300932f, -6.4457221f, -5.86640263f, -5.2363534f, -4.70952702f, -4.19898891f,
    -3.7346499f, -3.20452595f, -2.8300066f, -2.35908771f, -1.86044037f, -1.61496758f,
    -1.21970737f, -0.859695792f, -0.405939817f, 0.0f
};
static const float tap_swipe_v6_rank05_trial037_remaining_max[16] = {
    6.58147764f, 5.91250134f, 5.42114449f, 4.88563585f, 4.38300848f, 3.96605515f,
    3.54909897f, 3.18516088f, 2.68992114f, 2.38499713f, 2.12084675f, 1.71027291f,
    1.24902809f, 0.832600296f, 0.406629443f, 0.0f
};

TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE float tap_swipe_v6_rank05_trial037_walk_lgbm_tree(
    uint16_t root_index,
    const t_core* p_core,
    float normalization_factor,
    tap_swipe_v6_rank05_trial037_feature_cache_t* cache)
{
    uint8_t node_index = 0u;
    for (;;) {
        const uint32_t absolute_index = (uint32_t)root_index + (uint32_t)node_index;
        const uint8_t feature_metadata = tap_swipe_v6_rank05_trial037_node_features[absolute_index];
        if (TAP_SWIPE_V6_RANK05_TRIAL037_UNLIKELY(feature_metadata == 0xFFu)) {
            return tap_swipe_v6_rank05_trial037_node_thresholds[absolute_index];
        }
        const uint8_t feature_index = feature_metadata;

        float feature_value =
            tap_swipe_v6_rank05_trial037_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;

        if (feature_value >= -1.0e-35f && feature_value <= 1.0e-35f) {
            feature_value = 0.0f;
        }
        const uint8_t child_flags = tap_swipe_v6_rank05_trial037_node_child_flags[absolute_index];
        const uint8_t right_child = (uint8_t)(child_flags & 0x3Fu);
        const uint8_t default_left = (uint8_t)((child_flags >> 6) & 0x01u);
        const uint8_t missing_is_zero = (uint8_t)(child_flags >> 7);

        const uint8_t is_missing = (uint8_t)(
            (feature_value != feature_value) || ((missing_is_zero != 0u) && feature_value == 0.0f));

        if (TAP_SWIPE_V6_RANK05_TRIAL037_UNLIKELY(is_missing != 0u)) {
            node_index = default_left != 0u
                ? (uint8_t)(node_index + 1u)
                : right_child;
        } else {
            node_index = feature_value <= tap_swipe_v6_rank05_trial037_node_thresholds[absolute_index]
                ? (uint8_t)(node_index + 1u)
                : right_child;
        }
    }
}

/** Return boosted-tree raw score before the logistic link. */
TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE float tap_swipe_v6_rank05_trial037_predict_raw_score(const t_core* p_core)
{
    tap_swipe_v6_rank05_trial037_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank05_trial037_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t tree_index = 0u; tree_index < TAP_SWIPE_V6_RANK05_TRIAL037_NUM_TREES; ++tree_index) {
        raw_score += tap_swipe_v6_rank05_trial037_walk_lgbm_tree(tap_swipe_v6_rank05_trial037_tree_roots[tree_index], p_core, normalization_factor, &cache);
    }
    return raw_score;
}

/** Convert a raw score to nearest 0.05 probability, returned as Q20 [0..20]. */
TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE uint8_t tap_swipe_v6_rank05_trial037_probability_q20_from_raw(float raw_score)
{
    uint8_t low = 0u;
    uint8_t high = 20u;
    while (low < high) {
        const uint8_t middle = (uint8_t)((low + high) >> 1u);
        if (raw_score < tap_swipe_v6_rank05_trial037_q20_raw_midpoints[middle]) {
            high = middle;
        } else {
            low = (uint8_t)(middle + 1u);
        }
    }
    return low;
}

TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE uint8_t tap_swipe_v6_rank05_trial037_probability_q20(const t_core* p_core)
{
    return tap_swipe_v6_rank05_trial037_probability_q20_from_raw(tap_swipe_v6_rank05_trial037_predict_raw_score(p_core));
}

/** Approximate probability in 0.05 steps; maximum rounding error is 0.025. */
TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE float tap_swipe_v6_rank05_trial037_predict_value(const t_core* p_core)
{
    return 0.05f * (float)tap_swipe_v6_rank05_trial037_probability_q20(p_core);
}

/** Runtime-selectable exact threshold for Q20 values 0.05..0.95; no expf/logf. */
TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE int8_t tap_swipe_v6_rank05_trial037_predict_q20(    const t_core* p_core, uint8_t threshold_q20)
{
    if (threshold_q20 == 0u) { return 1; }
    if (threshold_q20 >= 20u) { return 0; }
    const float raw_threshold = tap_swipe_v6_rank05_trial037_q20_raw_thresholds[threshold_q20 - 1u];
    tap_swipe_v6_rank05_trial037_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank05_trial037_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t group = 0u; group < TAP_SWIPE_V6_RANK05_TRIAL037_EARLY_EXIT_GROUPS; ++group) {
        const uint16_t first_tree = (uint16_t)(group * TAP_SWIPE_V6_RANK05_TRIAL037_EARLY_EXIT_GROUP_SIZE);
        uint16_t end_tree = (uint16_t)(first_tree + TAP_SWIPE_V6_RANK05_TRIAL037_EARLY_EXIT_GROUP_SIZE);
        if (end_tree > TAP_SWIPE_V6_RANK05_TRIAL037_NUM_TREES) { end_tree = TAP_SWIPE_V6_RANK05_TRIAL037_NUM_TREES; }
        for (uint16_t tree_index = first_tree; tree_index < end_tree; ++tree_index) {
            raw_score += tap_swipe_v6_rank05_trial037_walk_lgbm_tree(tap_swipe_v6_rank05_trial037_tree_roots[tree_index], p_core, normalization_factor, &cache);
        }
        if (raw_score + tap_swipe_v6_rank05_trial037_remaining_max[group + 1u] < raw_threshold) { return 0; }
        if (raw_score + tap_swipe_v6_rank05_trial037_remaining_min[group + 1u] >= raw_threshold) { return 1; }
    }
    return raw_score >= raw_threshold ? 1 : 0;
}

#define TAP_SWIPE_V6_RANK05_TRIAL037_DEFAULT_THRESHOLD_Q20 10u
TAP_SWIPE_V6_RANK05_TRIAL037_FORCE_INLINE int8_t tap_swipe_v6_rank05_trial037_predict(const t_core* p_core)
{
    return tap_swipe_v6_rank05_trial037_predict_q20(p_core, TAP_SWIPE_V6_RANK05_TRIAL037_DEFAULT_THRESHOLD_Q20);
}

#endif /* TAP_SWIPE_V6_RANK05_TRIAL037_MODEL_H */
