#ifndef TAP_SWIPE_V6_RANK06_TRIAL005_MODEL_H
#define TAP_SWIPE_V6_RANK06_TRIAL005_MODEL_H

/*
 * Generated: 2026-11-16T11:57:14.860031+00:00
 * Model platform: LGBM
 * Task: BINARY_CLASSIFIER
 * Binary target class_id: 1
 * Training preprocessing: none
 * Firmware preprocessing: raw features; no preprocessing
 * Original selected feature indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
 * Required feature helpers/macros: none
 * Export profile: cortex_m4f_balanced
 * Flash storage: exact_float32_soa_bitpacked_flags; 3520 tree-data bytes.
 * Total generated static model data: 3844 bytes.
 * Metadata layout: lgbm_u6_child_flags_packed
 * Feature metadata: plain_u8; default_left=child_bit6; missing_is_zero=child_bit7; flag bytes=0.
 */

#include <math.h>
#include <stdint.h>
#include "core_type_def.h"

#define TAP_SWIPE_V6_RANK06_TRIAL005_NUM_FEATURES 36u
#define TAP_SWIPE_V6_RANK06_TRIAL005_NUM_TREES 80u
#define TAP_SWIPE_V6_RANK06_TRIAL005_THRESHOLD 0.400000006f
#define TAP_SWIPE_V6_RANK06_TRIAL005_TARGET_CLASS_ID 1
#define TAP_SWIPE_V6_RANK06_TRIAL005_PREPROCESSING "none"

#ifndef TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE
  #if defined(__STATIC_FORCEINLINE)
    #define TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE __STATIC_FORCEINLINE
  #elif defined(__GNUC__) || defined(__clang__)
    #define TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE static inline __attribute__((always_inline))
  #else
    #define TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE static inline
  #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define TAP_SWIPE_V6_RANK06_TRIAL005_LIKELY(value)   __builtin_expect(!!(value), 1)
  #define TAP_SWIPE_V6_RANK06_TRIAL005_UNLIKELY(value) __builtin_expect(!!(value), 0)
#else
  #define TAP_SWIPE_V6_RANK06_TRIAL005_LIKELY(value)   (value)
  #define TAP_SWIPE_V6_RANK06_TRIAL005_UNLIKELY(value) (value)
#endif

#define TAP_SWIPE_V6_RANK06_TRIAL005_HELPER_CACHE_COUNT 0u
#define TAP_SWIPE_V6_RANK06_TRIAL005_HELPER_CACHE_BYTES 1u
typedef struct {
    uint8_t unused;
} tap_swipe_v6_rank06_trial005_feature_cache_t;

/** Return one selected feature; helper expressions are cached lazily once per prediction. */
TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE float tap_swipe_v6_rank06_trial005_get_raw_model_feature(
    const t_core* p_core, uint16_t model_feature_index, tap_swipe_v6_rank06_trial005_feature_cache_t* cache)
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

TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE float tap_swipe_v6_rank06_trial005_runtime_normalization_factor(
    const t_core* p_core, tap_swipe_v6_rank06_trial005_feature_cache_t* cache)
{
    (void)p_core;
    (void)cache;
    return 1.0f;
}

static const float tap_swipe_v6_rank06_trial005_node_thresholds[560] = {
    445.5f, 13.5f, -0.0506975427f, -0.219902828f, 113.5f, 0.0891232193f,
    -0.0474154167f, 542.5f, 45.5f, -0.0165015142f, -0.184741676f, 178.5f,
    0.0155694559f, 0.14755553f, 401.5f, 21.5f, -0.0792254284f, -0.191353515f,
    101.5f, 0.0738032833f, -0.0509617925f, 678.5f, 15.5f, -0.0157362893f,
    -0.166445807f, 239.5f, 0.0671964288f, -0.0636501759f, 729.5f, 10.5f,
    -0.000222495393f, -0.138474762f, 292.5f, 0.0127327153f, 0.136832491f, 366.5f,
    48.5f, -0.0827899054f, -0.164761066f, 47.5f, 0.0517079085f, -0.0415437035f,
    226.5f, 588.5f, -0.0428969674f, 0.0641562194f, 236.5f, -0.141622618f,
    0.0298162401f, 26.5f, 354.5f, -0.0858718306f, 0.0697834045f, 696.5f,
    -0.120850205f, 0.0336609371f, 332.5f, 107.5f, 0.000403804152f, 0.141195878f,
    1839.5f, -0.141946673f, 0.0191554576f, 18.5f, 389.5f, -0.0533316359f,
    0.0703796595f, 1188.5f, -0.107086338f, 0.0261706952f, 60.5f, 690.5f,
    -0.0097546624f, 0.0707640275f, 201.5f, -0.0946161449f, 0.0541356206f, 1300.5f,
    47.5f, -0.0121963127f, 0.112912923f, 343.5f, -0.160550639f, -0.0187279247f,
    590.5f, 14.5f, 0.0238071475f, -0.107562378f, 1555.5f, 0.0754416287f,
    -0.0243907161f, 329.5f, 6.5f, -0.0742900297f, -0.145725071f, 47.5f,
    -0.00762298144f, 0.0811882243f, 632.5f, 92.5f, -0.00664665923f, 0.106861904f,
    790.5f, -0.135702282f, 0.0036534227f, 38.5f, 94.5f, -0.0592622235f,
    0.0447124541f, 1191.5f, -0.0869241357f, 0.0292107631f, 6.5f, 316.5f,
    -0.0601068512f, 0.0671785101f, 757.5f, -0.0750448853f, 0.0189467277f, 2925.5f,
    236.5f, -0.0107534099f, 0.098443985f, 6021.5f, -0.0902409926f, -0.19080101f,
    54.5f, 8.5f, 0.014050927f, 0.1023532f, 54.5f, -0.0791779533f,
    0.0327457786f, 200.5f, 16.5f, 0.000289392367f, 0.0900381655f, 139.5f,
    -0.0951276943f, 0.0303684324f, 740.5f, 53.5f, -0.00893061981f, 0.0790815577f,
    5051.0f, -0.132538289f, 0.0535693653f, 316.5f, 4193.5f, -0.113316901f,
    0.176395625f, 10.5f, 0.051068902f, -0.0132259643f, 1531.5f, 154.5f,
    -0.00739122322f, 0.0836569592f, 595.5f, -0.134151161f, -0.0182203408f, 389.5f,
    51.5f, -0.0304745436f, -0.128055036f, 4734.0f, 0.0143457996f, -0.152161658f,
    24.5f, 1.00000002e-35f, -0.0189686436f, 0.0460486971f, 511.5f, -0.078556262f,
    0.014221698f, 485.5f, 37.5f, -0.00254251179f, 0.0845436975f, 2219.5f,
    -0.107846655f, 0.0152692422f, 316.5f, 10.5f, -0.0300199594f, -0.120689414f,
    403.5f, -0.00354824541f, 0.0712487698f, 144.5f, 39.5f, -0.00301104179f,
    0.0785438493f, 790.5f, -0.0913221091f, 0.0274944287f, 10029.5f, 649.5f,
    -0.0156615544f, 0.0461740643f, 915.5f, 0.0913613588f, -0.0981897786f, 5.5f,
    412.5f, -0.00799963344f, 0.0625112578f, 668.5f, -0.057007879f, 0.0117106959f,
    121.5f, 52.5f, 0.00776886055f, 0.0842006281f, 1839.5f, -0.0776571482f,
    0.0234521758f, 896.5f, 67.5f, -0.00777058955f, 0.0654387027f, 1868.0f,
    -0.0818680227f, -0.183706149f, 17.5f, 4.5f, 0.00851253513f, 0.072375752f,
    1998.5f, -0.0760943294f, 0.00389419706f, 8.5f, 91.5f, -0.0549032502f,
    0.04304998f, 54.5f, -0.0326673985f, 0.0333247222f, 955.5f, 92.5f,
    -0.0032829775f, 0.0729284361f, 508.5f, -0.142008498f, -0.0313023068f, 98.5f,
    16.5f, -0.000268472737f, 0.0627497509f, 3188.5f, -0.0901655406f, 0.00826769881f,
    6021.5f, 94.5f, -0.00572855026f, 0.052617494f, 1131.5f, -0.116796412f,
    -0.177824348f, 10029.5f, 513.5f, -0.0139302528f, 0.0347182527f, 10681.5f,
    -0.0835614949f, 0.126843795f, 65.5f, 678.5f, -0.00598840043f, 0.0372550003f,
    301.5f, -0.0968900174f, -0.00511292415f, 25.5f, 1.00000002e-35f, -0.023013331f,
    0.0292546619f, 2579.5f, -0.0525991693f, 0.0267284159f, 289.5f, 2768.5f,
    -0.110260986f, 0.134295955f, 6933.0f, 0.0116481446f, -0.0469723567f, 2.5f,
    642.5f, 0.0157862883f, 0.087898545f, 348.5f, -0.0458709262f, 0.00707254233f,
    1364.5f, 1584.5f, -0.00446414575f, 0.0788257569f, 700.5f, -0.045337528f,
    0.0616846196f, 677.5f, 363.5f, -0.004018337f, 0.0662725642f, 2926.5f,
    -0.151427716f, -0.0327013545f, 399.5f, 33.5f, -0.00240795431f, 0.0569092073f,
    234.5f, -0.0827691928f, 0.0248367507f, 303.5f, 3.5f, -0.0559230633f,
    -0.142061591f, 1114.5f, 0.0188745465f, -0.0215630196f, 4.5f, 1716.5f,
    0.0208364762f, 0.0953732878f, 688.5f, -0.0470278226f, 0.00347148767f, 21.5f,
    1.00000002e-35f, -0.0169878025f, 0.0327671356f, 97.5f, -0.0384120308f, 0.0241027605f,
    1301.5f, 121.5f, -0.00549933966f, 0.0461879931f, 2366.0f, -0.0905498639f,
    -0.172264919f, 2271.0f, 37.5f, -0.00526469294f, 0.0440858901f, 956.0f,
    -0.111011796f, -0.0101461764f, 39.5f, 8.5f, 0.00479363976f, 0.0628863201f,
    2462.5f, -0.0652714223f, 0.0114442548f, 1301.5f, 43.5f, -0.00472234841f,
    0.0421531014f, 962.5f, -0.13858901f, -0.0380225107f, 136.5f, 19.5f,
    0.00397047307f, 0.0523806065f, 668.5f, -0.106135823f, -0.00872213673f, 289.5f,
    3.5f, -0.0603381209f, -0.141176164f, 50.5f, 0.0733796135f, -0.00127126579f,
    107.5f, 79.5f, -0.0141314203f, -0.104729667f, 702.5f, 0.0240861196f,
    -0.0152635667f, 454.5f, 168.5f, -0.00355689856f, 0.0581296794f, 2707.5f,
    -0.121085614f, -0.00570780551f, 6.5f, 1762.5f, 0.0178255755f, 0.088809669f,
    239.5f, -0.0583109744f, -0.000381394726f, 6021.5f, 245.5f, -0.00585898524f,
    0.0447188616f, 740.5f, -0.06909132f, -0.141213477f, 6620.5f, 1515.5f,
    -0.00809326954f, 0.0394905321f, 784.5f, 0.0793456882f, -0.0501655266f, 77.5f,
    62.5f, 0.0578021221f, 0.00536717987f, 906.5f, -0.0605973601f, 0.00715545565f,
    389.5f, 9.5f, -0.0153501118f, -0.128178641f, 124.5f, 0.0547812209f,
    -0.00286091026f, 467.5f, 43.5f, -0.00730483141f, -0.108932354f, 627.5f,
    4.01577672e-05f, 0.0703985989f, 307.5f, 25.5f, -0.00264518056f, 0.0471930355f,
    4328.5f, -0.0844194815f, 0.0170662589f, 380.5f, 19.5f, -0.000384165993f,
    0.0447020121f, 172.5f, -0.0725270733f, 0.0197433326f, 2913.5f, 518.5f,
    -0.00332654826f, 0.0459690765f, 5371.5f, -0.144476086f, -0.0496813506f, 10396.0f,
    769.5f, -0.0191438235f, 0.00910030678f, 13229.0f, 0.17112048f, -0.102681331f,
    6006.5f, 2251.5f, -0.000199083399f, 0.0594328158f, 3633.5f, -0.0891084224f,
    -0.00586929359f, 2.5f, 1044.5f, 0.0138084097f, 0.0941230133f, 423.5f,
    -0.0496668443f, -0.000328119204f, 1114.5f, 726.5f, -0.00269338186f, 0.0560102016f,
    1057.5f, -0.0306782946f, 0.0559000298f, 1868.0f, 273.5f, -0.0918417424f,
    0.00334379543f, 2366.0f, -0.0774090439f, -0.160143986f, 46.5f, 136.5f,
    0.0672172979f, -0.121063225f, 154.5f, -0.0427510589f, 0.00272180326f, 7.5f,
    1.00000002e-35f, -0.0174764991f, 0.0306243971f, 464.5f, -0.0511882566f, -0.00131939317f,
    1868.0f, 133.5f, -0.00354517903f, 0.0429176316f, 3569.5f, -0.100327857f,
    -0.163261443f, 848.5f, 39.5f, -0.00278417324f, 0.0407754481f, 562.5f,
    -0.0762866288f, 0.0182658937f, 86.5f, 6.5f, -0.0033024943f, 0.0350669995f,
    1202.5f, -0.0776748732f, -0.00513405167f, 18.5f, 1.00000002e-35f, -0.0143301832f,
    0.0276199076f, 8766.5f, -0.0174554698f, 0.0626662895f, 702.5f, 742.5f,
    -0.0128439683f, 0.0425223298f, 138.5f, -0.0267261788f, 0.0405795462f, 159.5f,
    10396.0f, 0.00565320579f, 0.135498822f, 3938.5f, -0.0447844155f, 0.0259893853f,
    91.5f, 118.5f, -0.0270139761f, -0.126817256f, 74.5f, 0.0525036789f,
    -0.00204039901f, 3358.5f, 20.5f, -0.00105143816f, 0.0420033112f, 1493.5f,
    -0.0859480649f, -0.00643193536f
};
static const uint16_t tap_swipe_v6_rank06_trial005_tree_roots[80] = {
    0u, 7u, 14u, 21u, 28u, 35u, 42u, 49u, 56u, 63u, 70u, 77u,
    84u, 91u, 98u, 105u, 112u, 119u, 126u, 133u, 140u, 147u, 154u, 161u,
    168u, 175u, 182u, 189u, 196u, 203u, 210u, 217u, 224u, 231u, 238u, 245u,
    252u, 259u, 266u, 273u, 280u, 287u, 294u, 301u, 308u, 315u, 322u, 329u,
    336u, 343u, 350u, 357u, 364u, 371u, 378u, 385u, 392u, 399u, 406u, 413u,
    420u, 427u, 434u, 441u, 448u, 455u, 462u, 469u, 476u, 483u, 490u, 497u,
    504u, 511u, 518u, 525u, 532u, 539u, 546u, 553u
};
static const uint8_t tap_swipe_v6_rank06_trial005_node_child_flags[560] = {
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
    0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u
};
static const uint8_t tap_swipe_v6_rank06_trial005_node_features[560] = {
    15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 33u,
    0xFFu, 0xFFu, 15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu,
    0xFFu, 24u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 15u,
    35u, 0xFFu, 0xFFu, 24u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 33u, 0xFFu,
    0xFFu, 30u, 15u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu,
    3u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 30u, 15u,
    0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 22u, 0xFFu, 0xFFu, 27u, 0xFFu, 0xFFu,
    2u, 30u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 15u, 31u, 0xFFu, 0xFFu, 21u,
    0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 2u, 0xFFu,
    0xFFu, 3u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 30u,
    34u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 30u, 22u, 0xFFu, 0xFFu, 22u, 0xFFu,
    0xFFu, 30u, 22u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 24u, 28u, 0xFFu, 0xFFu,
    15u, 0xFFu, 0xFFu, 15u, 16u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 30u, 34u,
    0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu,
    30u, 18u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 22u, 0xFFu, 0xFFu, 3u,
    0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 24u, 19u, 0xFFu,
    0xFFu, 34u, 0xFFu, 0xFFu, 5u, 8u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 30u,
    15u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 3u, 0xFFu,
    0xFFu, 18u, 28u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu,
    4u, 0xFFu, 0xFFu, 30u, 12u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu, 24u, 27u,
    0xFFu, 0xFFu, 27u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu,
    30u, 22u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 5u, 8u, 0xFFu, 0xFFu, 16u,
    0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 24u, 18u, 0xFFu,
    0xFFu, 3u, 0xFFu, 0xFFu, 15u, 16u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 30u,
    16u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 10u, 7u, 0xFFu, 0xFFu, 34u, 0xFFu,
    0xFFu, 18u, 34u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 19u, 0xFFu, 0xFFu,
    28u, 0xFFu, 0xFFu, 15u, 22u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 30u, 12u,
    0xFFu, 0xFFu, 4u, 0xFFu, 0xFFu, 30u, 18u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu,
    18u, 19u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 30u, 22u, 0xFFu, 0xFFu, 27u,
    0xFFu, 0xFFu, 24u, 21u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 24u, 28u, 0xFFu,
    0xFFu, 28u, 0xFFu, 0xFFu, 30u, 19u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 15u,
    22u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 12u, 10u, 0xFFu, 0xFFu, 10u, 0xFFu,
    0xFFu, 18u, 19u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 17u, 0xFFu, 0xFFu,
    2u, 0xFFu, 0xFFu, 30u, 32u, 0xFFu, 0xFFu, 24u, 0xFFu, 0xFFu, 5u, 7u,
    0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 30u, 5u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu,
    15u, 34u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 15u, 29u, 0xFFu, 0xFFu, 34u,
    0xFFu, 0xFFu, 24u, 28u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 20u, 0xFFu,
    0xFFu, 22u, 0xFFu, 0xFFu, 24u, 32u, 0xFFu, 0xFFu, 15u, 0xFFu, 0xFFu, 17u,
    15u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 5u, 8u, 0xFFu, 0xFFu, 8u, 0xFFu,
    0xFFu, 30u, 17u, 0xFFu, 0xFFu, 15u, 0xFFu, 0xFFu, 10u, 3u, 0xFFu, 0xFFu,
    34u, 0xFFu, 0xFFu, 18u, 15u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 5u, 29u,
    0xFFu, 0xFFu, 1u, 0xFFu, 0xFFu, 24u, 18u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu,
    18u, 28u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 30u, 19u, 0xFFu, 0xFFu, 28u,
    0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu, 15u, 0xFFu, 0xFFu, 30u, 18u, 0xFFu,
    0xFFu, 3u, 0xFFu, 0xFFu, 10u, 15u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu, 30u,
    17u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 12u, 10u, 0xFFu, 0xFFu, 5u, 0xFFu,
    0xFFu, 10u, 20u, 0xFFu, 0xFFu, 13u, 0xFFu, 0xFFu
};

/* Q20 probability LUT: one unit equals 0.05 probability. */
static const float tap_swipe_v6_rank06_trial005_q20_raw_thresholds[19] = {
    -2.94443893f, -2.19722462f, -1.73460102f, -1.38629436f, -1.09861231f,
    -0.847297847f, -0.619039237f, -0.405465096f, -0.200670689f, 0.0f,
    0.200670689f, 0.405465096f, 0.619039237f, 0.847297847f, 1.09861231f,
    1.38629436f, 1.73460102f, 2.19722462f, 2.94443893f
};
static const float tap_swipe_v6_rank06_trial005_q20_raw_midpoints[20] = {
    -3.66356158f, -2.51230574f, -1.9459101f, -1.55059743f, -1.23676264f,
    -0.969400585f, -0.730887532f, -0.510825634f, -0.302280873f, -0.100083455f,
    0.100083455f, 0.302280873f, 0.510825634f, 0.730887532f, 0.969400585f,
    1.23676264f, 1.55059743f, 1.9459101f, 2.51230574f, 3.66356158f
};

#define TAP_SWIPE_V6_RANK06_TRIAL005_EARLY_EXIT_GROUP_SIZE 4u
#define TAP_SWIPE_V6_RANK06_TRIAL005_EARLY_EXIT_GROUPS 20u
static const float tap_swipe_v6_rank06_trial005_remaining_min[21] = {
    -8.56662178f, -7.80399609f, -7.23812771f, -6.73378611f, -6.2577467f, -5.81748438f,
    -5.28521872f, -4.88672018f, -4.4700861f, -4.10685205f, -3.69592381f, -3.34298182f,
    -3.03267407f, -2.54550624f, -2.07235527f, -1.76205039f, -1.36797976f, -0.982038081f,
    -0.618958652f, -0.284277141f, 0.0f
};
static const float tap_swipe_v6_rank06_trial005_remaining_max[21] = {
    5.6410532f, 5.26324797f, 4.94065428f, 4.54529953f, 4.23700523f, 3.87891126f,
    3.52536154f, 3.2449162f, 2.94135165f, 2.69020271f, 2.44419312f, 2.07686734f,
    1.87291682f, 1.67758155f, 1.46958768f, 1.19889712f, 0.981811762f, 0.611158788f,
    0.453958511f, 0.272529364f, 0.0f
};

TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE float tap_swipe_v6_rank06_trial005_walk_lgbm_tree(
    uint16_t root_index,
    const t_core* p_core,
    float normalization_factor,
    tap_swipe_v6_rank06_trial005_feature_cache_t* cache)
{
    uint8_t node_index = 0u;
    for (;;) {
        const uint32_t absolute_index = (uint32_t)root_index + (uint32_t)node_index;
        const uint8_t feature_metadata = tap_swipe_v6_rank06_trial005_node_features[absolute_index];
        if (TAP_SWIPE_V6_RANK06_TRIAL005_UNLIKELY(feature_metadata == 0xFFu)) {
            return tap_swipe_v6_rank06_trial005_node_thresholds[absolute_index];
        }
        const uint8_t feature_index = feature_metadata;

        float feature_value =
            tap_swipe_v6_rank06_trial005_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;

        if (feature_value >= -1.0e-35f && feature_value <= 1.0e-35f) {
            feature_value = 0.0f;
        }
        const uint8_t child_flags = tap_swipe_v6_rank06_trial005_node_child_flags[absolute_index];
        const uint8_t right_child = (uint8_t)(child_flags & 0x3Fu);
        const uint8_t default_left = (uint8_t)((child_flags >> 6) & 0x01u);
        const uint8_t missing_is_zero = (uint8_t)(child_flags >> 7);

        const uint8_t is_missing = (uint8_t)(
            (feature_value != feature_value) || ((missing_is_zero != 0u) && feature_value == 0.0f));

        if (TAP_SWIPE_V6_RANK06_TRIAL005_UNLIKELY(is_missing != 0u)) {
            node_index = default_left != 0u
                ? (uint8_t)(node_index + 1u)
                : right_child;
        } else {
            node_index = feature_value <= tap_swipe_v6_rank06_trial005_node_thresholds[absolute_index]
                ? (uint8_t)(node_index + 1u)
                : right_child;
        }
    }
}

/** Return boosted-tree raw score before the logistic link. */
TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE float tap_swipe_v6_rank06_trial005_predict_raw_score(const t_core* p_core)
{
    tap_swipe_v6_rank06_trial005_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank06_trial005_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t tree_index = 0u; tree_index < TAP_SWIPE_V6_RANK06_TRIAL005_NUM_TREES; ++tree_index) {
        raw_score += tap_swipe_v6_rank06_trial005_walk_lgbm_tree(tap_swipe_v6_rank06_trial005_tree_roots[tree_index], p_core, normalization_factor, &cache);
    }
    return raw_score;
}

/** Convert a raw score to nearest 0.05 probability, returned as Q20 [0..20]. */
TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE uint8_t tap_swipe_v6_rank06_trial005_probability_q20_from_raw(float raw_score)
{
    uint8_t low = 0u;
    uint8_t high = 20u;
    while (low < high) {
        const uint8_t middle = (uint8_t)((low + high) >> 1u);
        if (raw_score < tap_swipe_v6_rank06_trial005_q20_raw_midpoints[middle]) {
            high = middle;
        } else {
            low = (uint8_t)(middle + 1u);
        }
    }
    return low;
}

TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE uint8_t tap_swipe_v6_rank06_trial005_probability_q20(const t_core* p_core)
{
    return tap_swipe_v6_rank06_trial005_probability_q20_from_raw(tap_swipe_v6_rank06_trial005_predict_raw_score(p_core));
}

/** Approximate probability in 0.05 steps; maximum rounding error is 0.025. */
TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE float tap_swipe_v6_rank06_trial005_predict_value(const t_core* p_core)
{
    return 0.05f * (float)tap_swipe_v6_rank06_trial005_probability_q20(p_core);
}

/** Runtime-selectable exact threshold for Q20 values 0.05..0.95; no expf/logf. */
TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE int8_t tap_swipe_v6_rank06_trial005_predict_q20(    const t_core* p_core, uint8_t threshold_q20)
{
    if (threshold_q20 == 0u) { return 1; }
    if (threshold_q20 >= 20u) { return 0; }
    const float raw_threshold = tap_swipe_v6_rank06_trial005_q20_raw_thresholds[threshold_q20 - 1u];
    tap_swipe_v6_rank06_trial005_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank06_trial005_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t group = 0u; group < TAP_SWIPE_V6_RANK06_TRIAL005_EARLY_EXIT_GROUPS; ++group) {
        const uint16_t first_tree = (uint16_t)(group * TAP_SWIPE_V6_RANK06_TRIAL005_EARLY_EXIT_GROUP_SIZE);
        uint16_t end_tree = (uint16_t)(first_tree + TAP_SWIPE_V6_RANK06_TRIAL005_EARLY_EXIT_GROUP_SIZE);
        if (end_tree > TAP_SWIPE_V6_RANK06_TRIAL005_NUM_TREES) { end_tree = TAP_SWIPE_V6_RANK06_TRIAL005_NUM_TREES; }
        for (uint16_t tree_index = first_tree; tree_index < end_tree; ++tree_index) {
            raw_score += tap_swipe_v6_rank06_trial005_walk_lgbm_tree(tap_swipe_v6_rank06_trial005_tree_roots[tree_index], p_core, normalization_factor, &cache);
        }
        if (raw_score + tap_swipe_v6_rank06_trial005_remaining_max[group + 1u] < raw_threshold) { return 0; }
        if (raw_score + tap_swipe_v6_rank06_trial005_remaining_min[group + 1u] >= raw_threshold) { return 1; }
    }
    return raw_score >= raw_threshold ? 1 : 0;
}

#define TAP_SWIPE_V6_RANK06_TRIAL005_DEFAULT_THRESHOLD_Q20 8u
TAP_SWIPE_V6_RANK06_TRIAL005_FORCE_INLINE int8_t tap_swipe_v6_rank06_trial005_predict(const t_core* p_core)
{
    return tap_swipe_v6_rank06_trial005_predict_q20(p_core, TAP_SWIPE_V6_RANK06_TRIAL005_DEFAULT_THRESHOLD_Q20);
}

#endif /* TAP_SWIPE_V6_RANK06_TRIAL005_MODEL_H */
