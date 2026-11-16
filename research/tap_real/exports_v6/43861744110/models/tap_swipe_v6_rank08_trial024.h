#ifndef TAP_SWIPE_V6_RANK08_TRIAL024_MODEL_H
#define TAP_SWIPE_V6_RANK08_TRIAL024_MODEL_H

/*
 * Generated: 2026-11-16T11:57:21.376116+00:00
 * Model platform: LGBM
 * Task: BINARY_CLASSIFIER
 * Binary target class_id: 1
 * Training preprocessing: none
 * Firmware preprocessing: raw features; no preprocessing
 * Original selected feature indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
 * Required feature helpers/macros: none
 * Export profile: cortex_m4f_balanced
 * Flash storage: exact_float32_soa_bitpacked_flags; 7040 tree-data bytes.
 * Total generated static model data: 7524 bytes.
 * Metadata layout: lgbm_u6_child_flags_packed
 * Feature metadata: plain_u8; default_left=child_bit6; missing_is_zero=child_bit7; flag bytes=0.
 */

#include <math.h>
#include <stdint.h>
#include "core_type_def.h"

#define TAP_SWIPE_V6_RANK08_TRIAL024_NUM_FEATURES 36u
#define TAP_SWIPE_V6_RANK08_TRIAL024_NUM_TREES 160u
#define TAP_SWIPE_V6_RANK08_TRIAL024_THRESHOLD 0.400000006f
#define TAP_SWIPE_V6_RANK08_TRIAL024_TARGET_CLASS_ID 1
#define TAP_SWIPE_V6_RANK08_TRIAL024_PREPROCESSING "none"

#ifndef TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE
  #if defined(__STATIC_FORCEINLINE)
    #define TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE __STATIC_FORCEINLINE
  #elif defined(__GNUC__) || defined(__clang__)
    #define TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE static inline __attribute__((always_inline))
  #else
    #define TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE static inline
  #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define TAP_SWIPE_V6_RANK08_TRIAL024_LIKELY(value)   __builtin_expect(!!(value), 1)
  #define TAP_SWIPE_V6_RANK08_TRIAL024_UNLIKELY(value) __builtin_expect(!!(value), 0)
#else
  #define TAP_SWIPE_V6_RANK08_TRIAL024_LIKELY(value)   (value)
  #define TAP_SWIPE_V6_RANK08_TRIAL024_UNLIKELY(value) (value)
#endif

#define TAP_SWIPE_V6_RANK08_TRIAL024_HELPER_CACHE_COUNT 0u
#define TAP_SWIPE_V6_RANK08_TRIAL024_HELPER_CACHE_BYTES 1u
typedef struct {
    uint8_t unused;
} tap_swipe_v6_rank08_trial024_feature_cache_t;

/** Return one selected feature; helper expressions are cached lazily once per prediction. */
TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE float tap_swipe_v6_rank08_trial024_get_raw_model_feature(
    const t_core* p_core, uint16_t model_feature_index, tap_swipe_v6_rank08_trial024_feature_cache_t* cache)
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

TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE float tap_swipe_v6_rank08_trial024_runtime_normalization_factor(
    const t_core* p_core, tap_swipe_v6_rank08_trial024_feature_cache_t* cache)
{
    (void)p_core;
    (void)cache;
    return 1.0f;
}

static const float tap_swipe_v6_rank08_trial024_node_thresholds[1120] = {
    456.5f, 20.5f, -0.0584034175f, -0.23427318f, 113.5f, 0.0927161276f,
    -0.0479405299f, 542.5f, 54.5f, -0.0226376802f, -0.194285735f, 292.5f,
    0.0154534662f, 0.151934087f, 401.5f, 40.5f, -0.0753594562f, -0.191111028f,
    93.5f, 0.0774793178f, -0.0505224839f, 678.5f, 10.5f, -0.00676325429f,
    -0.160644621f, 176.5f, 0.0717052594f, -0.0531767458f, 690.5f, 21.5f,
    0.00759870326f, -0.150844827f, 332.5f, 0.0732439309f, -0.0547855161f, 389.5f,
    6.5f, -0.0535253845f, -0.154243633f, 26.5f, 0.0801414698f, -0.0238471888f,
    1427.5f, 36.5f, 0.0226065926f, -0.130682305f, 249.5f, 0.0139481267f,
    0.152548745f, 696.5f, 56.5f, 0.0129428655f, -0.131278023f, 1555.5f,
    0.111710362f, -0.00751830032f, 366.5f, 16.5f, -0.0589822978f, -0.142298207f,
    211.5f, -0.00843326095f, 0.108135521f, 226.5f, 79.5f, 0.00440075062f,
    0.140901983f, 403.5f, -0.135413438f, 0.0400985144f, 632.5f, 39.5f,
    -0.00616298662f, 0.123008631f, 790.5f, -0.151540995f, 0.00625608955f, 18.5f,
    354.5f, -0.08279448f, 0.0665842518f, 757.5f, -0.112978749f, 0.0152328936f,
    1531.5f, 54.5f, -0.0111040175f, 0.107874751f, 666.5f, -0.163173392f,
    -0.0238091219f, 136.5f, 341.5f, -0.0977886319f, 0.0364476964f, 1839.5f,
    -0.108843669f, 0.0337392464f, 9.5f, 401.5f, -0.0354027636f, 0.0725419298f,
    757.5f, -0.0892097428f, 0.0177248567f, 2925.5f, 154.5f, -0.0139862932f,
    0.0984121412f, 1090.5f, -0.176043555f, -0.064357698f, 52.5f, 678.5f,
    -0.00948479772f, 0.0627917871f, 65.5f, -0.0812488422f, 0.0415773615f, 316.5f,
    10.5f, -0.0547927991f, -0.135995567f, 608.5f, 0.0166712049f, -0.102633916f,
    6.5f, 412.5f, -0.0158319063f, 0.0679959208f, 906.5f, -0.0558977649f,
    0.0324741192f, 380.5f, 83.5f, -0.0031754123f, 0.103815496f, 139.5f,
    -0.117956959f, 0.023446288f, 47.5f, 9.5f, 0.00247066212f, 0.0901724994f,
    1385.5f, -0.0964282155f, 0.0241024103f, 4734.0f, 249.5f, -0.0101680961f,
    0.0857695788f, 896.5f, -0.108039051f, -0.196506336f, 13.5f, 303.5f,
    -0.0838856697f, 0.0447548255f, 53.5f, -0.0474535413f, 0.0441041663f, 200.5f,
    16.5f, 0.0014366695f, 0.0827517286f, 117.5f, -0.0926436707f, 0.0235126521f,
    588.5f, 17.5f, -0.0121868411f, -0.117453419f, 10029.5f, 0.0294951685f,
    -0.0743010789f, 740.5f, 53.5f, -0.00760431821f, 0.0755042434f, 1026.5f,
    -0.139191449f, -0.0182067584f, 316.5f, 3.5f, -0.0623894669f, -0.143010944f,
    677.5f, 0.0116748111f, -0.122074775f, 5.5f, 1.00000002e-35f, -0.000121148507f,
    0.0662819147f, 590.5f, -0.0635379851f, 0.0128810229f, 98.5f, 39.5f,
    -0.000365460437f, 0.0846148804f, 627.5f, -0.0830649137f, 0.0358228907f, 65.5f,
    4.5f, 0.0046562627f, 0.067463845f, 2340.5f, -0.0625860319f, 0.0332894772f,
    6021.5f, 383.5f, -0.00690164557f, 0.0741999f, 677.5f, -0.100678734f,
    -0.177325875f, 25.5f, 6.5f, 0.00750586204f, 0.0729771107f, 2462.5f,
    -0.0716903806f, 0.017973721f, 303.5f, 9.5f, -0.0664617121f, -0.140261322f,
    10029.5f, 0.0127857393f, -0.064924255f, 1300.5f, 139.5f, -0.0064059468f,
    0.0758782104f, 562.5f, -0.113830231f, 0.00106831221f, 22.5f, 1.00000002e-35f,
    -0.0209183581f, 0.0400090516f, 301.5f, -0.0782506913f, 0.00370509643f, 289.5f,
    551.5f, -0.115948826f, 0.0299460273f, 2131.5f, 0.0169463754f, -0.0311041549f,
    1301.5f, 86.5f, -0.00812505465f, 0.0627654418f, 2366.0f, -0.093471311f,
    -0.179074228f, 2.5f, 630.5f, 0.0138533469f, 0.0870913118f, 363.5f,
    -0.0513697155f, 0.0079746237f, 13.5f, 3.5f, 0.00820181239f, 0.0677841604f,
    511.5f, -0.0680653453f, 0.00647683255f, 282.5f, 47.5f, -0.00381745468f,
    0.0730533153f, 4130.5f, -0.0932210311f, 0.0253974404f, 91.5f, 182.5f,
    -0.0916240066f, 0.0599144436f, 1114.5f, 0.0219824258f, -0.0225905646f, 4.5f,
    1502.5f, 0.0173111726f, 0.0955780521f, 37.5f, -0.0247850027f, 0.0315015204f,
    4734.0f, 245.5f, -0.00617466308f, 0.0590729639f, 2118.5f, -0.143010125f,
    -0.0531102158f, 131.5f, 24.5f, 0.00387969846f, 0.0662626773f, 133.5f,
    -0.0599643812f, 0.0315895975f, 677.5f, 121.5f, -0.00529948436f, 0.0546237752f,
    1819.5f, -0.144124791f, -0.0352336876f, 366.5f, 6.5f, -0.0236155465f,
    -0.133018598f, 702.5f, 0.0253832191f, -0.0154245012f, 513.5f, 77.5f,
    -0.0013012538f, -0.0774371773f, 1114.5f, 0.0689198971f, -0.0137142092f, 1868.0f,
    502.5f, -0.0052172849f, 0.0610931776f, 2366.0f, -0.0963238403f, -0.162597671f,
    289.5f, 3.5f, -0.0577380881f, -0.139840126f, 50.5f, 0.074307695f,
    -0.00145590922f, 848.5f, 66.5f, -0.00385371852f, 0.0599836372f, 3938.5f,
    -0.0923050344f, 0.0206901971f, 39.5f, 8.5f, 0.00226425496f, 0.0539640114f,
    2579.5f, -0.0528891534f, 0.0254182946f, 769.5f, 65.5f, -0.00839669723f,
    -0.106890678f, 755.5f, 0.0525306463f, -0.0130127948f, 107.5f, 79.5f,
    -0.0133548966f, -0.103863433f, 6620.5f, 0.0133203184f, -0.037154302f, 2320.0f,
    133.5f, -0.00493764505f, 0.0571442619f, 4544.5f, -0.140702963f, -0.0484789461f,
    1515.5f, 305.5f, -0.00437675184f, -0.0966256633f, 1364.5f, 0.0701228231f,
    -0.0108012836f, 2.5f, 1044.5f, 0.0129167782f, 0.0955044776f, 348.5f,
    -0.0389868356f, 0.00530008692f, 744.5f, 510.5f, 0.00240141363f, -0.0769374296f,
    526.5f, -0.143153965f, 0.0864230469f, 26.5f, 1.00000002e-35f, -0.0250313636f,
    0.0299504176f, 143.5f, -0.0345199034f, 0.0259272475f, 1868.0f, 256.5f,
    -0.00322119216f, 0.057294596f, 2366.0f, -0.0816356614f, -0.153648645f, 1301.5f,
    245.5f, -0.00336684007f, 0.0475584418f, 5371.5f, -0.107065342f, 0.0376220122f,
    273.5f, 304.5f, -0.106238991f, 0.0430518612f, 74.5f, 0.0489904061f,
    -0.0040737479f, 6620.5f, 1515.5f, -0.00945995934f, 0.0428330116f, 3448.5f,
    -0.0971454978f, -0.00674191816f, 10681.5f, 13229.0f, 0.00295021641f, -0.0843347684f,
    1493.5f, 0.178263471f, 0.0357263647f, 14.5f, 1.00000002e-35f, -0.0219045822f,
    0.0259560589f, 464.5f, -0.0730991736f, 0.000416874595f, 6021.5f, 700.5f,
    -0.00291993795f, 0.0551886596f, 1831.5f, -0.131395325f, -0.0578556918f, 10396.0f,
    8766.5f, -0.00504507544f, 0.0645855963f, 13229.0f, 0.167395532f, -0.0790500641f,
    183.5f, 19.5f, -0.00340284617f, 0.0476602204f, 325.5f, -0.06564679f,
    0.0247704145f, 2.5f, 1044.5f, 0.0131852226f, 0.0857657641f, 198.5f,
    -0.0447976887f, 0.00196819822f, 454.5f, 162.5f, -0.00280421297f, 0.0509577617f,
    2707.5f, -0.119157054f, 0.00403346494f, 10681.5f, 13229.0f, 0.00213110726f,
    -0.0756303966f, 2867.5f, 0.164980784f, 0.024524685f, 46.5f, 142.5f,
    0.0653451309f, -0.117083788f, 415.5f, -0.0553262904f, 0.00192887313f, 10.5f,
    1.00000002e-35f, -0.0122201778f, 0.0322913677f, 239.5f, -0.0580997914f, -0.00196312368f,
    159.5f, 19.5f, -0.0002410405f, 0.0467155874f, 1191.5f, -0.0628100559f,
    0.00666759489f, 94.5f, 592.5f, 0.00741481502f, 0.101532534f, 85.5f,
    -0.0729139075f, -0.000623205677f, 10396.0f, 3358.5f, 0.00534709916f, -0.0342484489f,
    14497.0f, 0.146941304f, -0.0885783359f, 8176.0f, 702.5f, 0.010364851f,
    -0.015969662f, 2471.5f, 0.177426592f, 0.0592398979f, 91.5f, 165.5f,
    -0.0803811178f, 0.00984664168f, 124.5f, 0.0380052961f, -0.00325990072f, 649.5f,
    279.5f, -0.00474654417f, -0.0822366029f, 566.5f, 0.0767567679f, -0.00091923424f,
    9333.0f, 5.5f, 0.0147032505f, -0.0125796618f, 4995.5f, 0.115546554f,
    0.0322156586f, 64.5f, 94.5f, 0.0325568877f, -0.0160065293f, 118.5f,
    -0.132274389f, 0.0400609337f, 2913.5f, 54.5f, -0.00488436129f, 0.0347706638f,
    3948.5f, -0.133289427f, -0.0712769628f, 10681.5f, 10029.5f, 0.00321065448f,
    -0.0516062863f, 1412.5f, 0.174028829f, 0.0181101933f, 2359.5f, 3958.5f,
    -0.00101548503f, -0.0705120713f, 2575.5f, 0.0890736952f, 0.0047543156f, 82.5f,
    23.5f, 0.0643091351f, -0.0672539398f, 94.5f, 0.0426320992f, -0.0023063093f,
    769.5f, 1117.5f, -0.0263899025f, 0.055461973f, 608.5f, 0.042437654f,
    -0.00810014829f, 10396.0f, 10124.5f, -0.00479454361f, 0.0388266556f, 14497.0f,
    0.136177823f, -0.0794381425f, 1868.0f, 51.5f, -0.00559299672f, 0.028357327f,
    2366.0f, -0.0675872788f, -0.136749476f, 273.5f, 18.5f, 0.0187716819f,
    -0.0951551497f, 74.5f, 0.0386419035f, -0.0034196123f, 1679.0f, 534.5f,
    -0.002569156f, 0.0509652682f, 2505.5f, -0.089253068f, 0.0201943368f, 6021.5f,
    296.5f, -0.00180103688f, 0.048686754f, 3948.5f, -0.117267139f, -0.0393122621f,
    307.5f, 16.5f, -0.00425302517f, 0.0324352682f, 1254.5f, -0.106548466f,
    -0.00931531191f, 65.5f, 1.5f, -0.00784337893f, 0.025566902f, 335.5f,
    -0.0914960206f, -0.00448501762f, 4.5f, 1966.5f, 0.00372423744f, 0.0674978048f,
    2579.5f, -0.0187414996f, 0.0198167097f, 6006.5f, 2251.5f, -0.00194491399f,
    0.0545255095f, 784.5f, 0.0742261112f, -0.0374792069f, 10396.0f, 12583.5f,
    0.000920523598f, -0.0610709898f, 15916.5f, 0.121952564f, -0.075794667f, 91.5f,
    106.5f, -0.0173067339f, -0.108777784f, 74.5f, 0.0391396172f, -0.00202693674f,
    8176.0f, 366.5f, 0.000928663241f, -0.0524216741f, 2231.5f, 0.164288431f,
    0.0485316031f, 10681.5f, 39.5f, -0.00637536822f, 0.0192201305f, 1386.5f,
    0.160274118f, 0.0149107035f, 46.5f, 193.5f, 0.0479301624f, -0.12150424f,
    154.5f, -0.0388677567f, 0.0026249676f, 3.5f, 1165.5f, 0.0078932019f,
    0.0697212592f, 276.5f, -0.0310290325f, 0.00216333056f, 263.5f, 714.5f,
    -0.00719621172f, 0.0525848903f, 138.5f, -0.0150058735f, 0.0444009453f, 17.5f,
    1.5f, -0.00661746785f, 0.0289476309f, 1671.5f, -0.0314171836f, 0.0136643201f,
    485.5f, 33.5f, -0.00178050017f, 0.0371071845f, 457.5f, -0.0545325056f,
    0.031522084f, 5729.5f, 1733.5f, -0.00434338627f, 0.0321833305f, 4214.5f,
    -0.06242406f, 0.00308457506f, 2913.5f, 666.5f, -0.00251249014f, 0.0332440399f,
    3491.5f, -0.123302363f, -0.0574734882f, 1.5f, 992.5f, 0.00621520821f,
    0.0913881063f, 1.00000002e-35f, -0.0382168405f, 0.000873900543f, 38.5f, 1.5f,
    -0.00401265826f, 0.0252134725f, 445.5f, -0.0909565687f, -0.00700296881f, 578.5f,
    563.5f, -0.00207556644f, 0.0398214944f, 2707.5f, -0.107543766f, -0.00685036974f,
    4.5f, 1977.5f, 0.00818492007f, 0.0676757842f, 901.5f, 0.0100258701f,
    -0.0181652643f, 94.5f, 171.5f, 0.0292429253f, -0.133573949f, 400.5f,
    -0.0307050236f, 0.00434239814f, 1868.0f, 67.5f, -0.00354095339f, 0.0291144308f,
    2366.0f, -0.0538907759f, -0.128899336f, 10396.0f, 3358.5f, 0.00423701992f,
    -0.0250715837f, 13229.0f, 0.116747595f, -0.0506814048f, 10681.5f, 10029.5f,
    0.00199370319f, -0.0403685346f, 1386.5f, 0.154563501f, 0.0115840975f, 273.5f,
    4.5f, -0.0398695953f, -0.128560826f, 50.5f, 0.0392307304f, -0.00126280938f,
    3448.5f, 3958.5f, 0.00118706981f, -0.0502158701f, 987.5f, 0.123139516f,
    0.0103774946f, 56.5f, 149.5f, -0.0172309075f, -0.143359631f, 90.5f,
    -0.00156911591f, 0.0297719259f, 149.5f, 16.5f, -0.00226280699f, 0.0340765752f,
    1202.5f, -0.0802302361f, -0.0025918677f, 19.5f, 1454.5f, 0.00259447354f,
    0.0641593412f, 4.5f, -0.0471820459f, 0.000879140862f, 23.5f, 46.5f,
    -0.0214377921f, 0.0875486284f, 296.5f, -0.0681571215f, 0.000577868777f, 8176.0f,
    351.5f, 0.00177248672f, -0.0355184525f, 2231.5f, 0.158890173f, 0.0411178544f,
    399.5f, 49.5f, -0.00149552815f, 0.0310390778f, 233.5f, -0.0575987399f,
    0.0178669877f, 118.5f, 157.5f, -0.00529375719f, 0.0670928136f, 139.5f,
    -0.0747112185f, -0.00112239574f, 10396.0f, 1204.5f, 0.00576551352f, -0.0140557764f,
    11355.5f, 0.109180287f, -0.0302869659f, 41147.5f, 907.5f, -0.00143221254f,
    0.0625748113f, 7371.0f, -0.149623394f, -0.0461255535f, 8176.0f, 366.5f,
    0.0005735007f, -0.0460442081f, 2538.5f, 0.148086935f, 0.0384161063f, 273.5f,
    4.5f, -0.0353961028f, -0.125501305f, 744.5f, -0.00116767734f, 0.0342341736f,
    1.5f, 2103.5f, 0.00468537305f, 0.0724095255f, 1.00000002e-35f, -0.0425124727f,
    -0.000461311662f, 2913.5f, 72.5f, -0.00381780765f, 0.0216717329f, 3948.5f,
    -0.110103011f, -0.0398570821f, 39.5f, 2.5f, -0.00376497186f, 0.0255510919f,
    348.5f, -0.0855977237f, -0.00506183691f, 263.5f, 1272.5f, 0.0042623342f,
    0.085503228f, 363.5f, -0.0120585887f, 0.0377120487f, 101.5f, 1.5f,
    -0.0049477485f, 0.0181745905f, 544.5f, -0.0742133632f, -0.0039324672f, 10153.5f,
    7.5f, 0.00971543416f, -0.00773104699f, 2131.5f, 0.020949183f, 0.148529649f,
    290.5f, 176.5f, -0.0717702881f, 0.126556218f, 114.5f, 0.0226643793f,
    -0.00350568653f, 1301.5f, 22.5f, -0.00353163062f, 0.0196078382f, 7028.0f,
    -0.121360496f, -0.0221272707f, 8357.0f, 10396.0f, -0.00213942095f, 0.0742513686f,
    1440.5f, 0.1544303f, -0.013517987f, 17.5f, 115.5f, 0.00667832373f,
    0.103322469f, 74.5f, -0.0469406769f, 0.000895815087f, 8296.0f, 1193.5f,
    -0.00130642299f, 0.0401493981f, 608.5f, -0.0234969854f, -0.111696325f, 720.5f,
    338.5f, -0.00550190127f, 0.0608486049f, 165.5f, -0.0123392623f, 0.0415932611f,
    670.5f, 31.5f, -0.00167088013f, 0.0254973583f, 2144.5f, -0.0741709396f,
    0.00882590655f, 38.5f, 1.00000002e-35f, -0.0152717112f, 0.015128403f, 445.5f,
    -0.0845693424f, -0.0047698142f, 7272.5f, 4956.5f, 0.000560227374f, 0.0772534981f,
    847.5f, 0.0687179267f, -0.0352410488f, 173.5f, 1774.5f, 0.0028204557f,
    0.0903890505f, 169.5f, -0.0475250147f, 0.0013462879f, 11400.0f, 503.5f,
    0.000788244361f, -0.0420194641f, 4.5f, -0.00719510112f, 0.0877644643f, 23.5f,
    115.5f, -0.014976914f, 0.0785790756f, 319.5f, -0.0475292504f, 0.000670903479f,
    720.5f, 553.5f, -0.014773733f, 0.0340368114f, 562.5f, -0.00993572362f,
    0.0587738082f, 896.5f, 705.5f, -0.0013854556f, 0.0310919154f, 6277.5f,
    -0.103851266f, -0.00548072672f, 104.5f, 367.5f, 0.0222951677f, -0.146792114f,
    217.5f, -0.0447767861f, 0.00102062628f, 8059.5f, 4110.5f, 0.000370106311f,
    0.0581147745f, 455.5f, 0.102403641f, -0.0328875594f, 8357.0f, 1.5f,
    0.0211214982f, -0.00434242049f, 1386.5f, 0.141214401f, -0.00353105459f, 366.5f,
    508.5f, -0.00115263171f, 0.0321451761f, 2707.5f, -0.0794020221f, 0.0182695948f,
    41147.5f, 907.5f, -0.000612154021f, 0.0451170355f, 7371.0f, -0.146160632f,
    -0.0334315188f, 79.5f, 350.5f, -0.00223395461f, 0.0602694079f, 148.5f,
    -0.0482712165f, 0.000227842494f, 10153.5f, 4995.5f, 0.00264299987f, -0.0175469927f,
    2131.5f, 0.0205340236f, 0.124900125f, 2851.5f, 3958.5f, 0.000278929423f,
    -0.0483396575f, 1307.5f, 0.072924763f, 0.000649925787f, 1301.5f, 705.5f,
    -0.00115391507f, 0.0283348225f, 2003.5f, -0.139381886f, -0.0225993134f, 10124.5f,
    332.5f, 0.00198128517f, -0.0326951444f, 1805.5f, 0.0945145562f, 0.00566461822f,
    1.00000002e-35f, 2308.5f, -0.00701936288f, -0.109946363f, 901.5f, 0.0149704162f,
    -0.00707746949f, 841.5f, 485.5f, -0.00528123556f, -0.132956624f, 490.5f,
    0.0481607094f, -0.00244978792f, 256.5f, 573.5f, -0.0871688947f, 0.0430516303f,
    50.5f, 0.0281475931f, -0.00154426461f, 5150.5f, 4751.5f, 0.000826243428f,
    -0.0347182639f, 987.5f, 0.128827482f, 0.0140527692f
};
static const uint16_t tap_swipe_v6_rank08_trial024_tree_roots[160] = {
    0u, 7u, 14u, 21u, 28u, 35u, 42u, 49u, 56u, 63u, 70u, 77u,
    84u, 91u, 98u, 105u, 112u, 119u, 126u, 133u, 140u, 147u, 154u, 161u,
    168u, 175u, 182u, 189u, 196u, 203u, 210u, 217u, 224u, 231u, 238u, 245u,
    252u, 259u, 266u, 273u, 280u, 287u, 294u, 301u, 308u, 315u, 322u, 329u,
    336u, 343u, 350u, 357u, 364u, 371u, 378u, 385u, 392u, 399u, 406u, 413u,
    420u, 427u, 434u, 441u, 448u, 455u, 462u, 469u, 476u, 483u, 490u, 497u,
    504u, 511u, 518u, 525u, 532u, 539u, 546u, 553u, 560u, 567u, 574u, 581u,
    588u, 595u, 602u, 609u, 616u, 623u, 630u, 637u, 644u, 651u, 658u, 665u,
    672u, 679u, 686u, 693u, 700u, 707u, 714u, 721u, 728u, 735u, 742u, 749u,
    756u, 763u, 770u, 777u, 784u, 791u, 798u, 805u, 812u, 819u, 826u, 833u,
    840u, 847u, 854u, 861u, 868u, 875u, 882u, 889u, 896u, 903u, 910u, 917u,
    924u, 931u, 938u, 945u, 952u, 959u, 966u, 973u, 980u, 987u, 994u, 1001u,
    1008u, 1015u, 1022u, 1029u, 1036u, 1043u, 1050u, 1057u, 1064u, 1071u, 1078u, 1085u,
    1092u, 1099u, 1106u, 1113u
};
static const uint8_t tap_swipe_v6_rank08_trial024_node_child_flags[1120] = {
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
    0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u,
    0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u,
    0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u,
    0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u,
    0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u,
    0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u,
    0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u,
    0x00u, 0x46u, 0x00u, 0x00u
};
static const uint8_t tap_swipe_v6_rank08_trial024_node_features[1120] = {
    15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 34u,
    0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu,
    0xFFu, 24u, 0xFFu, 0xFFu, 15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u,
    30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 2u, 30u, 0xFFu, 0xFFu, 34u, 0xFFu,
    0xFFu, 3u, 30u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu,
    34u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 22u,
    0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu,
    30u, 22u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 3u,
    0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu,
    0xFFu, 28u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu, 15u,
    35u, 0xFFu, 0xFFu, 24u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 3u, 0xFFu,
    0xFFu, 30u, 34u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu,
    3u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 30u, 15u,
    0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 30u, 22u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu,
    15u, 24u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 24u, 28u, 0xFFu, 0xFFu, 28u,
    0xFFu, 0xFFu, 15u, 34u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 30u, 18u, 0xFFu,
    0xFFu, 2u, 0xFFu, 0xFFu, 24u, 19u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u,
    22u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 18u, 0xFFu,
    0xFFu, 24u, 22u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 15u, 31u, 0xFFu, 0xFFu,
    5u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 30u, 18u,
    0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 15u, 16u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu,
    18u, 22u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 30u, 16u, 0xFFu, 0xFFu, 2u,
    0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 24u, 28u, 0xFFu,
    0xFFu, 3u, 0xFFu, 0xFFu, 12u, 8u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 30u,
    12u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu, 30u, 32u, 0xFFu, 0xFFu, 28u, 0xFFu,
    0xFFu, 30u, 19u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 18u, 19u, 0xFFu, 0xFFu,
    33u, 0xFFu, 0xFFu, 15u, 34u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 8u, 30u,
    0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 18u, 34u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu,
    15u, 22u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 30u, 19u, 0xFFu, 0xFFu, 3u,
    0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu,
    0xFFu, 10u, 0xFFu, 0xFFu, 12u, 10u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 24u,
    28u, 0xFFu, 0xFFu, 15u, 0xFFu, 0xFFu, 7u, 30u, 0xFFu, 0xFFu, 10u, 0xFFu,
    0xFFu, 30u, 17u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 34u, 30u, 0xFFu, 0xFFu,
    7u, 0xFFu, 0xFFu, 30u, 18u, 0xFFu, 0xFFu, 32u, 0xFFu, 0xFFu, 18u, 28u,
    0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 24u, 32u, 0xFFu, 0xFFu, 15u, 0xFFu, 0xFFu,
    15u, 7u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 5u, 7u, 0xFFu, 0xFFu, 8u,
    0xFFu, 0xFFu, 16u, 5u, 0xFFu, 0xFFu, 13u, 0xFFu, 0xFFu, 24u, 18u, 0xFFu,
    0xFFu, 2u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 27u, 0xFFu, 0xFFu, 17u,
    3u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 24u, 20u, 0xFFu, 0xFFu, 28u, 0xFFu,
    0xFFu, 30u, 17u, 0xFFu, 0xFFu, 1u, 0xFFu, 0xFFu, 18u, 19u, 0xFFu, 0xFFu,
    34u, 0xFFu, 0xFFu, 16u, 5u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 5u, 29u,
    0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu, 30u, 18u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu,
    30u, 19u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 10u, 4u, 0xFFu, 0xFFu, 13u,
    0xFFu, 0xFFu, 17u, 10u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 15u, 10u, 0xFFu,
    0xFFu, 16u, 0xFFu, 0xFFu, 12u, 2u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 8u,
    30u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 3u, 24u, 0xFFu, 0xFFu, 5u, 0xFFu,
    0xFFu, 34u, 5u, 0xFFu, 0xFFu, 8u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu,
    28u, 0xFFu, 0xFFu, 16u, 5u, 0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 8u, 5u,
    0xFFu, 0xFFu, 14u, 0xFFu, 0xFFu, 7u, 5u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu,
    15u, 7u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 17u, 2u, 0xFFu, 0xFFu, 5u,
    0xFFu, 0xFFu, 18u, 19u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 15u, 0u, 0xFFu,
    0xFFu, 5u, 0xFFu, 0xFFu, 30u, 31u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u,
    28u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 24u, 28u, 0xFFu, 0xFFu, 15u, 0xFFu,
    0xFFu, 30u, 18u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 24u, 12u, 0xFFu, 0xFFu,
    3u, 0xFFu, 0xFFu, 5u, 8u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 17u, 10u,
    0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 12u, 10u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu,
    15u, 18u, 0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu, 16u, 19u, 0xFFu, 0xFFu, 17u,
    0xFFu, 0xFFu, 5u, 29u, 0xFFu, 0xFFu, 1u, 0xFFu, 0xFFu, 30u, 11u, 0xFFu,
    0xFFu, 1u, 0xFFu, 0xFFu, 10u, 4u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu, 24u,
    18u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 19u, 0xFFu, 0xFFu, 28u, 0xFFu,
    0xFFu, 5u, 7u, 0xFFu, 0xFFu, 8u, 0xFFu, 0xFFu, 24u, 31u, 0xFFu, 0xFFu,
    28u, 0xFFu, 0xFFu, 30u, 17u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 30u, 18u,
    0xFFu, 0xFFu, 15u, 0xFFu, 0xFFu, 18u, 31u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu,
    24u, 17u, 0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 5u, 34u, 0xFFu, 0xFFu, 7u,
    0xFFu, 0xFFu, 18u, 28u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 17u, 10u, 0xFFu,
    0xFFu, 5u, 0xFFu, 0xFFu, 16u, 5u, 0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 15u,
    22u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 8u, 5u, 0xFFu, 0xFFu, 11u, 0xFFu,
    0xFFu, 13u, 10u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu, 24u, 28u, 0xFFu, 0xFFu,
    15u, 0xFFu, 0xFFu, 35u, 12u, 0xFFu, 0xFFu, 23u, 0xFFu, 0xFFu, 5u, 11u,
    0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu,
    30u, 34u, 0xFFu, 0xFFu, 19u, 0xFFu, 0xFFu, 10u, 0u, 0xFFu, 0xFFu, 12u,
    0xFFu, 0xFFu, 17u, 10u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 35u, 28u, 0xFFu,
    0xFFu, 7u, 0xFFu, 0xFFu, 15u, 18u, 0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu, 15u,
    22u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 24u, 14u, 0xFFu, 0xFFu, 18u, 0xFFu,
    0xFFu, 24u, 19u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu,
    2u, 0xFFu, 0xFFu, 10u, 7u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 22u,
    0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu,
    17u, 8u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 18u, 22u, 0xFFu, 0xFFu, 34u,
    0xFFu, 0xFFu, 16u, 17u, 0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 5u, 14u, 0xFFu,
    0xFFu, 7u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 24u, 0xFFu, 0xFFu, 17u,
    8u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 30u, 19u, 0xFFu, 0xFFu, 8u, 0xFFu,
    0xFFu, 30u, 18u, 0xFFu, 0xFFu, 15u, 0xFFu, 0xFFu, 5u, 17u, 0xFFu, 0xFFu,
    10u, 0xFFu, 0xFFu, 5u, 12u, 0xFFu, 0xFFu, 8u, 0xFFu, 0xFFu, 3u, 24u,
    0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 5u, 14u, 0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu,
    17u, 15u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 18u, 31u, 0xFFu, 0xFFu, 34u,
    0xFFu, 0xFFu, 5u, 29u, 0xFFu, 0xFFu, 6u, 0xFFu, 0xFFu, 5u, 17u, 0xFFu,
    0xFFu, 10u, 0xFFu, 0xFFu, 16u, 30u, 0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 18u,
    31u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 35u, 28u, 0xFFu, 0xFFu, 7u, 0xFFu,
    0xFFu, 10u, 4u, 0xFFu, 0xFFu, 12u, 0xFFu, 0xFFu, 15u, 5u, 0xFFu, 0xFFu,
    10u, 0xFFu, 0xFFu, 8u, 5u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 24u, 31u,
    0xFFu, 0xFFu, 7u, 0xFFu, 0xFFu, 2u, 30u, 0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu,
    18u, 1u, 0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 15u, 22u, 0xFFu, 0xFFu, 0u,
    0xFFu, 0xFFu, 15u, 16u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 8u, 5u, 0xFFu,
    0xFFu, 11u, 0xFFu, 0xFFu
};

/* Q20 probability LUT: one unit equals 0.05 probability. */
static const float tap_swipe_v6_rank08_trial024_q20_raw_thresholds[19] = {
    -2.94443893f, -2.19722462f, -1.73460102f, -1.38629436f, -1.09861231f,
    -0.847297847f, -0.619039237f, -0.405465096f, -0.200670689f, 0.0f,
    0.200670689f, 0.405465096f, 0.619039237f, 0.847297847f, 1.09861231f,
    1.38629436f, 1.73460102f, 2.19722462f, 2.94443893f
};
static const float tap_swipe_v6_rank08_trial024_q20_raw_midpoints[20] = {
    -3.66356158f, -2.51230574f, -1.9459101f, -1.55059743f, -1.23676264f,
    -0.969400585f, -0.730887532f, -0.510825634f, -0.302280873f, -0.100083455f,
    0.100083455f, 0.302280873f, 0.510825634f, 0.730887532f, 0.969400585f,
    1.23676264f, 1.55059743f, 1.9459101f, 2.51230574f, 3.66356158f
};

#define TAP_SWIPE_V6_RANK08_TRIAL024_EARLY_EXIT_GROUP_SIZE 4u
#define TAP_SWIPE_V6_RANK08_TRIAL024_EARLY_EXIT_GROUPS 40u
static const float tap_swipe_v6_rank08_trial024_remaining_min[41] = {
    -14.3085175f, -13.5276194f, -12.9600306f, -12.4172935f, -11.87955f, -11.4880095f,
    -11.0181284f, -10.5545454f, -10.1595144f, -9.71088314f, -9.31883621f, -8.99915791f,
    -8.4817028f, -8.08952522f, -7.70911264f, -7.27050877f, -6.90949345f, -6.5884223f,
    -6.21828508f, -5.9778614f, -5.67024994f, -5.34746075f, -5.00961304f, -4.60494518f,
    -4.36406088f, -4.15264845f, -3.98919511f, -3.62910914f, -3.29773164f, -2.93517756f,
    -2.7040484f, -2.39179277f, -2.06760263f, -1.88797843f, -1.63436973f, -1.35157883f,
    -1.17925286f, -0.880939841f, -0.602758169f, -0.364791453f, 0.0f
};
static const float tap_swipe_v6_rank08_trial024_remaining_max[41] = {
    11.5066442f, 11.1123323f, 10.6942377f, 10.2551842f, 9.9395113f, 9.68786144f,
    9.38405609f, 9.2007637f, 8.9011879f, 8.74226665f, 8.45128441f, 8.17018604f,
    7.95991182f, 7.71888638f, 7.48256969f, 7.26113272f, 6.96489286f, 6.60870075f,
    6.29495859f, 5.82218933f, 5.55168343f, 5.18937635f, 4.93062544f, 4.77286959f,
    4.46996021f, 4.02766275f, 3.87676764f, 3.68703532f, 3.44419646f, 3.09744f,
    2.75272226f, 2.48279953f, 2.20636725f, 1.92858422f, 1.52464783f, 1.38300967f,
    1.0490123f, 0.834439933f, 0.555688679f, 0.235011414f, 0.0f
};

TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE float tap_swipe_v6_rank08_trial024_walk_lgbm_tree(
    uint16_t root_index,
    const t_core* p_core,
    float normalization_factor,
    tap_swipe_v6_rank08_trial024_feature_cache_t* cache)
{
    uint8_t node_index = 0u;
    for (;;) {
        const uint32_t absolute_index = (uint32_t)root_index + (uint32_t)node_index;
        const uint8_t feature_metadata = tap_swipe_v6_rank08_trial024_node_features[absolute_index];
        if (TAP_SWIPE_V6_RANK08_TRIAL024_UNLIKELY(feature_metadata == 0xFFu)) {
            return tap_swipe_v6_rank08_trial024_node_thresholds[absolute_index];
        }
        const uint8_t feature_index = feature_metadata;

        float feature_value =
            tap_swipe_v6_rank08_trial024_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;

        if (feature_value >= -1.0e-35f && feature_value <= 1.0e-35f) {
            feature_value = 0.0f;
        }
        const uint8_t child_flags = tap_swipe_v6_rank08_trial024_node_child_flags[absolute_index];
        const uint8_t right_child = (uint8_t)(child_flags & 0x3Fu);
        const uint8_t default_left = (uint8_t)((child_flags >> 6) & 0x01u);
        const uint8_t missing_is_zero = (uint8_t)(child_flags >> 7);

        const uint8_t is_missing = (uint8_t)(
            (feature_value != feature_value) || ((missing_is_zero != 0u) && feature_value == 0.0f));

        if (TAP_SWIPE_V6_RANK08_TRIAL024_UNLIKELY(is_missing != 0u)) {
            node_index = default_left != 0u
                ? (uint8_t)(node_index + 1u)
                : right_child;
        } else {
            node_index = feature_value <= tap_swipe_v6_rank08_trial024_node_thresholds[absolute_index]
                ? (uint8_t)(node_index + 1u)
                : right_child;
        }
    }
}

/** Return boosted-tree raw score before the logistic link. */
TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE float tap_swipe_v6_rank08_trial024_predict_raw_score(const t_core* p_core)
{
    tap_swipe_v6_rank08_trial024_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank08_trial024_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t tree_index = 0u; tree_index < TAP_SWIPE_V6_RANK08_TRIAL024_NUM_TREES; ++tree_index) {
        raw_score += tap_swipe_v6_rank08_trial024_walk_lgbm_tree(tap_swipe_v6_rank08_trial024_tree_roots[tree_index], p_core, normalization_factor, &cache);
    }
    return raw_score;
}

/** Convert a raw score to nearest 0.05 probability, returned as Q20 [0..20]. */
TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE uint8_t tap_swipe_v6_rank08_trial024_probability_q20_from_raw(float raw_score)
{
    uint8_t low = 0u;
    uint8_t high = 20u;
    while (low < high) {
        const uint8_t middle = (uint8_t)((low + high) >> 1u);
        if (raw_score < tap_swipe_v6_rank08_trial024_q20_raw_midpoints[middle]) {
            high = middle;
        } else {
            low = (uint8_t)(middle + 1u);
        }
    }
    return low;
}

TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE uint8_t tap_swipe_v6_rank08_trial024_probability_q20(const t_core* p_core)
{
    return tap_swipe_v6_rank08_trial024_probability_q20_from_raw(tap_swipe_v6_rank08_trial024_predict_raw_score(p_core));
}

/** Approximate probability in 0.05 steps; maximum rounding error is 0.025. */
TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE float tap_swipe_v6_rank08_trial024_predict_value(const t_core* p_core)
{
    return 0.05f * (float)tap_swipe_v6_rank08_trial024_probability_q20(p_core);
}

/** Runtime-selectable exact threshold for Q20 values 0.05..0.95; no expf/logf. */
TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE int8_t tap_swipe_v6_rank08_trial024_predict_q20(    const t_core* p_core, uint8_t threshold_q20)
{
    if (threshold_q20 == 0u) { return 1; }
    if (threshold_q20 >= 20u) { return 0; }
    const float raw_threshold = tap_swipe_v6_rank08_trial024_q20_raw_thresholds[threshold_q20 - 1u];
    tap_swipe_v6_rank08_trial024_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank08_trial024_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t group = 0u; group < TAP_SWIPE_V6_RANK08_TRIAL024_EARLY_EXIT_GROUPS; ++group) {
        const uint16_t first_tree = (uint16_t)(group * TAP_SWIPE_V6_RANK08_TRIAL024_EARLY_EXIT_GROUP_SIZE);
        uint16_t end_tree = (uint16_t)(first_tree + TAP_SWIPE_V6_RANK08_TRIAL024_EARLY_EXIT_GROUP_SIZE);
        if (end_tree > TAP_SWIPE_V6_RANK08_TRIAL024_NUM_TREES) { end_tree = TAP_SWIPE_V6_RANK08_TRIAL024_NUM_TREES; }
        for (uint16_t tree_index = first_tree; tree_index < end_tree; ++tree_index) {
            raw_score += tap_swipe_v6_rank08_trial024_walk_lgbm_tree(tap_swipe_v6_rank08_trial024_tree_roots[tree_index], p_core, normalization_factor, &cache);
        }
        if (raw_score + tap_swipe_v6_rank08_trial024_remaining_max[group + 1u] < raw_threshold) { return 0; }
        if (raw_score + tap_swipe_v6_rank08_trial024_remaining_min[group + 1u] >= raw_threshold) { return 1; }
    }
    return raw_score >= raw_threshold ? 1 : 0;
}

#define TAP_SWIPE_V6_RANK08_TRIAL024_DEFAULT_THRESHOLD_Q20 8u
TAP_SWIPE_V6_RANK08_TRIAL024_FORCE_INLINE int8_t tap_swipe_v6_rank08_trial024_predict(const t_core* p_core)
{
    return tap_swipe_v6_rank08_trial024_predict_q20(p_core, TAP_SWIPE_V6_RANK08_TRIAL024_DEFAULT_THRESHOLD_Q20);
}

#endif /* TAP_SWIPE_V6_RANK08_TRIAL024_MODEL_H */
