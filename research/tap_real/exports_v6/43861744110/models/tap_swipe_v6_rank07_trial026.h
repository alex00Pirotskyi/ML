#ifndef TAP_SWIPE_V6_RANK07_TRIAL026_MODEL_H
#define TAP_SWIPE_V6_RANK07_TRIAL026_MODEL_H

/*
 * Generated: 2026-11-16T11:57:17.957440+00:00
 * Model platform: LGBM
 * Task: BINARY_CLASSIFIER
 * Binary target class_id: 1
 * Training preprocessing: none
 * Firmware preprocessing: raw features; no preprocessing
 * Original selected feature indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
 * Required feature helpers/macros: none
 * Export profile: cortex_m4f_balanced
 * Flash storage: exact_float32_soa_bitpacked_flags; 6124 tree-data bytes.
 * Total generated static model data: 6568 bytes.
 * Metadata layout: lgbm_u6_child_flags_packed
 * Feature metadata: plain_u8; default_left=child_bit6; missing_is_zero=child_bit7; flag bytes=0.
 */

#include <math.h>
#include <stdint.h>
#include "core_type_def.h"

#define TAP_SWIPE_V6_RANK07_TRIAL026_NUM_FEATURES 36u
#define TAP_SWIPE_V6_RANK07_TRIAL026_NUM_TREES 140u
#define TAP_SWIPE_V6_RANK07_TRIAL026_THRESHOLD 0.449999988f
#define TAP_SWIPE_V6_RANK07_TRIAL026_TARGET_CLASS_ID 1
#define TAP_SWIPE_V6_RANK07_TRIAL026_PREPROCESSING "none"

#ifndef TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE
  #if defined(__STATIC_FORCEINLINE)
    #define TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE __STATIC_FORCEINLINE
  #elif defined(__GNUC__) || defined(__clang__)
    #define TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE static inline __attribute__((always_inline))
  #else
    #define TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE static inline
  #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define TAP_SWIPE_V6_RANK07_TRIAL026_LIKELY(value)   __builtin_expect(!!(value), 1)
  #define TAP_SWIPE_V6_RANK07_TRIAL026_UNLIKELY(value) __builtin_expect(!!(value), 0)
#else
  #define TAP_SWIPE_V6_RANK07_TRIAL026_LIKELY(value)   (value)
  #define TAP_SWIPE_V6_RANK07_TRIAL026_UNLIKELY(value) (value)
#endif

#define TAP_SWIPE_V6_RANK07_TRIAL026_HELPER_CACHE_COUNT 0u
#define TAP_SWIPE_V6_RANK07_TRIAL026_HELPER_CACHE_BYTES 1u
typedef struct {
    uint8_t unused;
} tap_swipe_v6_rank07_trial026_feature_cache_t;

/** Return one selected feature; helper expressions are cached lazily once per prediction. */
TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE float tap_swipe_v6_rank07_trial026_get_raw_model_feature(
    const t_core* p_core, uint16_t model_feature_index, tap_swipe_v6_rank07_trial026_feature_cache_t* cache)
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

TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE float tap_swipe_v6_rank07_trial026_runtime_normalization_factor(
    const t_core* p_core, tap_swipe_v6_rank07_trial026_feature_cache_t* cache)
{
    (void)p_core;
    (void)cache;
    return 1.0f;
}

static const float tap_swipe_v6_rank07_trial026_node_thresholds[974] = {
    456.5f, 20.5f, -0.0476344451f, -0.193472281f, 113.5f, 0.0773055404f,
    -0.0392334647f, 542.5f, 54.5f, -0.0185928289f, -0.162771955f, 307.5f,
    0.0142021328f, 0.127563536f, 401.5f, 14.5f, -0.0639382973f, -0.163282186f,
    101.5f, 0.0643914565f, -0.043929752f, 678.5f, 10.5f, -0.00624361169f,
    -0.136587664f, 239.5f, 0.0585762002f, -0.0536946617f, 588.5f, 19.5f,
    -0.00346127502f, -0.137133554f, 257.5f, 0.0596369393f, -0.0470719896f, 401.5f,
    42.5f, -0.0543442816f, -0.137345254f, 249.5f, -0.00169941527f, 0.103596568f,
    54.5f, 366.5f, -0.081559442f, 0.0580157079f, 1555.5f, -0.121385485f,
    0.022033304f, 226.5f, 590.5f, -0.0192604419f, 0.0646846145f, 1671.5f,
    -0.122982211f, 0.0216516946f, 24.5f, 366.5f, -0.0623424947f, 0.0612919889f,
    667.5f, -0.0975304767f, 0.0246919617f, 36.5f, 341.5f, -0.0767377689f,
    0.0477086641f, 211.5f, -0.076845713f, 0.0555971935f, 632.5f, 45.5f,
    -0.00605334016f, 0.109386787f, 790.5f, -0.130793124f, 0.00984889735f, 757.5f,
    14.5f, 0.0243636109f, -0.0928981528f, 1486.5f, 0.0739566088f, -0.0161225088f,
    136.5f, 15.5f, 0.00536286226f, 0.0956087783f, 82.5f, -0.0969985351f,
    0.0307993907f, 690.5f, 17.5f, -0.00757493405f, -0.106248364f, 363.5f,
    0.00440484472f, 0.100308381f, 1531.5f, 154.5f, -0.00924021658f, 0.0941155106f,
    666.5f, -0.137417048f, -0.027100483f, 389.5f, 48.5f, -0.0386507809f,
    -0.11591664f, 10.5f, 0.0611313246f, -0.0097700879f, 380.5f, 79.5f,
    -0.00182077533f, 0.0919448137f, 139.5f, -0.105738007f, 0.0134253893f, 329.5f,
    10.5f, -0.0341165513f, -0.11359144f, 9.5f, 0.0535500385f, -0.0104011511f,
    906.5f, 56.5f, 0.0101035265f, -0.083290346f, 4734.0f, 0.0446880013f,
    -0.140449613f, 690.5f, 7.5f, 0.0123747345f, -0.0703366995f, 403.5f,
    0.00326400227f, 0.0912458226f, 101.5f, 26.5f, -0.00011278184f, 0.0868202746f,
    1839.5f, -0.0943855718f, 0.0166666042f, 2046.5f, 132.5f, -0.0090036625f,
    0.0749218464f, 1090.5f, -0.126496688f, -0.0265566949f, 316.5f, 10.5f,
    -0.0445396155f, -0.109696768f, 26.5f, 0.0351861492f, -0.0166798439f, 467.5f,
    20.5f, -0.00246503111f, -0.0839042291f, 59.5f, -0.00473906239f, 0.0634417459f,
    848.5f, 54.5f, -0.00367056672f, 0.0792784765f, 996.5f, -0.105761603f,
    -0.00321936351f, 200.5f, 55.5f, 0.00327888783f, 0.0775943324f, 127.5f,
    -0.0805860981f, 0.0205851346f, 22.5f, 678.5f, -0.00076272845f, 0.0556295589f,
    1427.5f, -0.0627321303f, 0.0145674925f, 608.5f, 125.5f, -0.00643509394f,
    0.0668802336f, 5371.5f, -0.099733226f, 0.0558313131f, 47.5f, 9.5f,
    0.00323931198f, 0.0665962398f, 1385.5f, -0.0759966448f, 0.0151665071f, 6.5f,
    412.5f, -0.0158309508f, 0.0547143854f, 615.5f, -0.0549505465f, 0.010189252f,
    4734.0f, 67.5f, -0.0076760673f, 0.059521649f, 896.5f, -0.0802379996f,
    -0.158918127f, 65.5f, 690.5f, -0.00688807061f, 0.0399097838f, 64.5f,
    -0.0556912422f, 0.0242679976f, 303.5f, 9.5f, -0.0643682554f, -0.117053419f,
    10029.5f, 0.012214547f, -0.0632856041f, 677.5f, 363.5f, -0.00512944069f,
    0.068844609f, 2926.5f, -0.139176235f, -0.0462559238f, 25.5f, 6.5f,
    0.00547204493f, 0.06473656f, 1928.5f, -0.0557155125f, 0.0238016043f, 2925.5f,
    178.5f, -0.00513764145f, 0.0644024909f, 962.5f, -0.121082939f, -0.0410007089f,
    303.5f, 607.5f, -0.0935233384f, 0.0247986745f, 5.5f, 0.0422460325f,
    -0.00692136073f, 183.5f, 39.5f, -0.00321126869f, 0.0597977936f, 325.5f,
    -0.0767794251f, 0.018383529f, 65.5f, 678.5f, -0.00740494998f, 0.034511745f,
    353.5f, -0.0831366181f, -0.00172296294f, 955.5f, 53.5f, -0.00583880628f,
    0.0533597171f, 1026.5f, -0.116528846f, -0.0215377379f, 5.5f, 630.5f,
    0.00469318125f, 0.0585768856f, 363.5f, -0.0544379614f, 0.00385859748f, 10029.5f,
    577.5f, -0.0124644339f, 0.0330502726f, 784.5f, 0.0887553468f, -0.0760230348f,
    14.5f, 3.5f, 0.00769375684f, 0.0580768399f, 1088.5f, -0.0601346418f,
    0.00621184148f, 332.5f, 39.5f, -0.000353877695f, 0.058258947f, 627.5f,
    -0.0714517012f, 0.023089638f, 1301.5f, 86.5f, -0.00575877866f, 0.0472108386f,
    2366.0f, -0.073093079f, -0.148727119f, 303.5f, 3.5f, -0.0547124371f,
    -0.116628446f, 1114.5f, 0.0181757715f, -0.0205151606f, 3.5f, 1.00000002e-35f,
    0.00102029671f, 0.0569665357f, 37.5f, -0.0206171088f, 0.0241295844f, 26.5f,
    1.00000002e-35f, -0.0206490625f, 0.0300214309f, 301.5f, -0.0620379858f, 0.000781860959f,
    896.5f, 133.5f, -0.00427206326f, 0.0537551269f, 1868.0f, -0.0568367727f,
    -0.133204892f, 1106.5f, 69.5f, -0.00383168994f, 0.0504717864f, 3938.5f,
    -0.0884370804f, 0.00948253088f, 316.5f, 4.5f, -0.040259935f, -0.111506671f,
    6620.5f, 0.0117211211f, -0.0376206189f, 43.5f, 8.5f, 0.00224281987f,
    0.0488165542f, 2579.5f, -0.0491952188f, 0.0225627758f, 91.5f, 165.5f,
    -0.0860890523f, 0.00244922051f, 1700.5f, 0.0166059006f, -0.0229744632f, 6021.5f,
    273.5f, -0.00392327318f, 0.052180808f, 677.5f, -0.0703027472f, -0.126957968f,
    18.5f, 1.00000002e-35f, -0.015575164f, 0.030863706f, 464.5f, -0.0573514253f,
    0.000621416606f, 307.5f, 32.5f, -0.00347709656f, 0.0464345776f, 505.5f,
    -0.0742858797f, 0.0184851848f, 289.5f, 541.5f, -0.0869324282f, 0.0219298303f,
    6620.5f, 0.00966464728f, -0.0323392563f, 2.5f, 1009.5f, 0.0155082662f,
    0.082604371f, 696.5f, -0.0229271483f, 0.0128832962f, 1301.5f, 245.5f,
    -0.00407128455f, 0.0465228595f, 5371.5f, -0.0948287845f, 0.0260584597f, 178.5f,
    27.5f, 0.00236107153f, 0.0504970476f, 162.5f, -0.0558545589f, 0.0141148865f,
    13.5f, 1.00000002e-35f, -0.018610172f, 0.025703311f, 2150.5f, -0.0406734087f,
    0.011490874f, 1868.0f, 700.5f, -0.00311550265f, 0.0507196672f, -0.119768627f,
    1204.5f, 1584.5f, -0.00417828187f, 0.0673964545f, 700.5f, -0.0314156003f,
    0.0450002626f, 109.5f, 5.5f, 0.00185246393f, 0.0399811119f, 668.5f,
    -0.0809424371f, -0.00545926252f, 6021.5f, 389.5f, -0.0375087f, 0.00684479764f,
    626.5f, -0.0630501062f, -0.119540393f, 107.5f, 176.5f, -0.0622425191f,
    0.0338170715f, 74.5f, 0.04935655f, -0.00186310394f, 2.5f, 1009.5f,
    0.0125052491f, 0.0755671784f, 322.5f, -0.0315697156f, 0.00369156362f, 508.5f,
    162.5f, -0.00271304557f, 0.0459828451f, 2707.5f, -0.104226016f, -0.00259436201f,
    441.5f, 64.5f, -0.000531675061f, 0.0425696149f, 193.5f, -0.0590057001f,
    0.0219963398f, 273.5f, 227.5f, -0.0894785225f, 0.00791284908f, 50.5f,
    0.0583735742f, -0.00247588009f, 2131.5f, 870.5f, -0.00514707714f, 0.0460605845f,
    13229.0f, -0.00895639323f, -0.0804347172f, 7.5f, 1.00000002e-35f, -0.0142921731f,
    0.0266372468f, 2340.5f, -0.0252800565f, 0.0199202094f, 2320.0f, 53.5f,
    -0.0048335758f, 0.033076629f, 8766.5f, -0.107525766f, -0.0317462757f, 10.5f,
    1.00000002e-35f, -0.00813012943f, 0.030269647f, 20.5f, -0.0222454611f, 0.0141024794f,
    149.5f, 27.5f, -0.00060904125f, 0.0411069728f, 3327.5f, -0.066931203f,
    0.0014883203f, 702.5f, 729.5f, -0.0127292974f, 0.0414021686f, 790.5f,
    -0.0230622496f, 0.0487143025f, 6021.5f, 10681.5f, -0.000147596715f, 0.108214982f,
    3948.5f, -0.10925854f, -0.0428299308f, 6620.5f, 1515.5f, -0.00657964218f,
    0.0330877379f, 455.5f, 0.095230341f, -0.0387226045f, 39.5f, 46.5f,
    0.0515894443f, 0.00401797332f, 2579.5f, -0.0379307605f, 0.0179298855f, 273.5f,
    4.5f, -0.0497182384f, -0.110206299f, 62.5f, 0.0429199859f, -0.00218841573f,
    115.5f, 106.5f, -0.0102033364f, -0.0900050625f, 653.5f, 0.0174603481f,
    -0.00959256385f, 10396.0f, 3358.5f, 0.00419361563f, -0.029215917f, 13229.0f,
    0.130316928f, -0.059130054f, 10681.5f, 13229.0f, 0.00216101203f, -0.0609580874f,
    1386.5f, 0.147125229f, 0.0204971004f, 82.5f, 23.5f, 0.0543213077f,
    -0.0617620647f, 94.5f, 0.0397086032f, -0.0025202434f, 8766.5f, 65.5f,
    0.0045159054f, -0.02525701f, 1767.5f, 0.107500926f, 0.0390924588f, 423.5f,
    43.5f, -0.00850881729f, -0.0948828608f, 133.5f, -0.00142885395f, 0.0444530919f,
    1868.0f, 47.5f, -0.00448319502f, 0.0280649886f, 2366.0f, -0.0485018268f,
    -0.116199389f, 1.5f, 1044.5f, 0.0112145431f, 0.0869465768f, 827.5f,
    -0.0197324902f, 0.00676317839f, 6021.5f, 245.5f, -0.00372661254f, 0.0333477631f,
    1240.0f, -0.103420056f, -0.0455603153f, 677.5f, 121.5f, -0.00270079589f,
    0.0349160656f, 2707.5f, -0.104197972f, -0.0185697507f, 93.5f, 1.5f,
    -0.0073546744f, 0.0202675778f, 348.5f, -0.0870678127f, -0.00669960771f, 2.5f,
    1502.5f, 0.0124172745f, 0.0732670128f, 162.5f, -0.0360576734f, 0.000570962264f,
    10396.0f, 3358.5f, 0.00358014437f, -0.0256155748f, 13229.0f, 0.124443226f,
    -0.0526791736f, 503.5f, 43.5f, -0.00270942296f, 0.036329221f, 4328.5f,
    -0.0701706409f, 0.010332657f, 91.5f, 69.5f, -0.00758314412f, -0.0791438892f,
    74.5f, 0.0373326913f, -0.0016987609f, 10681.5f, 10029.5f, 0.00264865183f,
    -0.0442242883f, 1412.5f, 0.137092948f, 0.0132578174f, 9333.0f, 399.5f,
    0.00235814741f, -0.0326487869f, 4.5f, 0.000721718883f, 0.0686307698f, 10396.0f,
    2359.5f, -0.00642765127f, 0.0175901037f, 5467.5f, 0.113017246f, -0.0292461999f,
    513.5f, 292.5f, -0.00420403481f, -0.0762014836f, 677.5f, 0.0535292737f,
    -0.00515599176f, 124.5f, 486.5f, -0.00391711993f, 0.0643719882f, 139.5f,
    -0.0763212517f, -0.00131365901f, 10681.5f, 13229.0f, 0.0019274666f, -0.0526209585f,
    1437.5f, 0.11880172f, 0.022790432f, 8176.0f, 143.5f, -0.00553883705f,
    0.0208109356f, 2231.5f, 0.130032972f, 0.0499674715f, 46.5f, 97.5f,
    0.0463045239f, -0.0965897068f, 415.5f, -0.0417172536f, 0.00162297126f, 2913.5f,
    221.5f, -0.00288444501f, 0.0288776848f, 2736.0f, -0.103363402f, -0.0403636508f,
    6.5f, 2999.5f, 0.00826831628f, 0.0753532872f, 33.5f, -0.0151637504f,
    0.0130396942f, 3448.5f, 4542.5f, 0.000549013319f, -0.0578009449f, 1307.5f,
    0.0897360891f, 0.00315749529f, 366.5f, 534.5f, -0.00233398029f, 0.0383827426f,
    562.5f, -0.0866972134f, 0.00282733585f, 1300.5f, 39.5f, -0.00337574887f,
    0.0269232709f, 2193.5f, -0.100784868f, -0.0105609968f, 4.5f, 1.00000002e-35f,
    -0.00931879133f, 0.0251027104f, 276.5f, -0.0388736278f, -0.000407496787f, 273.5f,
    227.5f, -0.075654909f, 0.0222783647f, 94.5f, 0.028233679f, -0.00326926075f,
    98.5f, 4.5f, -0.00290476694f, 0.0232906286f, 790.5f, -0.0364593379f,
    0.0269926898f, 702.5f, 1162.5f, -0.00166601082f, 0.0547917075f, 256.5f,
    -0.0171849336f, 0.0447827913f, 56.5f, 8.5f, 0.00131427124f, 0.0345864743f,
    393.5f, -0.0668056384f, -0.00339761982f, 1868.0f, 9333.0f, -0.000247308606f,
    0.0518301316f, 2366.0f, -0.041196119f, -0.107734665f, 10396.0f, 5.5f,
    0.0105992788f, -0.00780232297f, 14497.0f, 0.105885208f, -0.0569520183f, 2913.5f,
    325.5f, -0.000996545306f, 0.0375615023f, -0.0918338671f, 485.5f, 19.5f,
    -0.00257288408f, 0.0240543075f, 1733.5f, -0.079479143f, -0.00233670464f, 94.5f,
    520.5f, -0.00462824479f, 0.0592065081f, 148.5f, -0.0555190556f, -0.000368320761f,
    6006.5f, 2251.5f, -0.000961501035f, 0.0454237014f, 3633.5f, -0.0630624294f,
    -0.00211241865f, 273.5f, 369.5f, -0.0720194504f, 0.0271292944f, 46.5f,
    0.0429134034f, -0.000808505109f, 10396.0f, 3358.5f, 0.00288622081f, -0.0220831446f,
    14497.0f, 0.0949184299f, -0.0505359992f, 8176.0f, 1114.5f, 0.00569237443f,
    -0.0121134827f, 2231.5f, 0.125711694f, 0.0345559902f, 10681.5f, 13229.0f,
    0.00140634237f, -0.045368392f, 1386.5f, 0.122750416f, 0.0119888969f, 52.5f,
    100.5f, -0.0120088598f, -0.105407946f, 94.5f, 0.0234145187f, -0.00219204929f,
    599.5f, 279.5f, -0.00303723477f, -0.064637728f, 887.5f, 0.0527122952f,
    -0.00129227631f, 10396.0f, 769.5f, -0.010876013f, 0.00624660077f, 2758.5f,
    0.0936839879f, -0.0266379714f, 1.00000002e-35f, 1502.5f, -0.0333781242f, 0.0277561508f,
    608.5f, 0.0153763378f, -0.00834556948f, 23.5f, 92.5f, -0.0297195818f,
    0.0758820996f, 325.5f, -0.0488128811f, 0.000456775946f, 70.5f, 146.5f,
    0.0296181645f, -0.108370155f, 190.5f, -0.0361777022f, 0.0010624578f, 2320.0f,
    563.5f, -0.00209157798f, 0.0310032088f, 2431.5f, -0.092011489f, -0.0197434351f,
    2.5f, 928.5f, 0.00561636966f, 0.0598595776f, 1.5f, -0.023611445f,
    0.00325312279f, 8176.0f, 53.5f, 0.00428491738f, -0.0150453895f, 2471.5f,
    0.121249668f, 0.0366821624f, 63.5f, 83.5f, 0.0035619773f, -0.0285722017f,
    322.5f, -0.0842880383f, 0.0313462019f, 454.5f, 121.5f, -0.00118782395f,
    0.0350764319f, 2707.5f, -0.0826168731f, 0.00976420846f, 6021.5f, 666.5f,
    -0.00209747418f, 0.0291875601f, 3948.5f, -0.0855121836f, -0.0244749319f, 2366.0f,
    165.5f, -0.00264836592f, 0.0289377645f, -0.100103654f, 10681.5f, 10029.5f,
    0.00245012669f, -0.0337551348f, 1883.5f, 0.104365215f, -0.00291375211f, 3283.5f,
    3958.5f, 6.17423211e-05f, -0.0447852612f, 1307.5f, 0.0743737444f, 0.00291885878f,
    154.5f, 12.5f, -0.000301808119f, -0.0751605704f, 720.5f, 0.0212106425f,
    -0.00364655768f, 10124.5f, 420.5f, 0.00182509306f, -0.0344499871f, 5292.5f,
    0.0785777569f, 0.00957183912f
};
static const uint16_t tap_swipe_v6_rank07_trial026_tree_roots[140] = {
    0u, 7u, 14u, 21u, 28u, 35u, 42u, 49u, 56u, 63u, 70u, 77u,
    84u, 91u, 98u, 105u, 112u, 119u, 126u, 133u, 140u, 147u, 154u, 161u,
    168u, 175u, 182u, 189u, 196u, 203u, 210u, 217u, 224u, 231u, 238u, 245u,
    252u, 259u, 266u, 273u, 280u, 287u, 294u, 301u, 308u, 315u, 322u, 329u,
    336u, 343u, 350u, 357u, 364u, 371u, 378u, 385u, 392u, 399u, 406u, 413u,
    420u, 427u, 432u, 439u, 446u, 453u, 460u, 467u, 474u, 481u, 488u, 495u,
    502u, 509u, 516u, 523u, 530u, 537u, 544u, 551u, 558u, 565u, 572u, 579u,
    586u, 593u, 600u, 607u, 614u, 621u, 628u, 635u, 642u, 649u, 656u, 663u,
    670u, 677u, 684u, 691u, 698u, 705u, 712u, 719u, 726u, 733u, 740u, 747u,
    754u, 761u, 768u, 775u, 782u, 789u, 796u, 803u, 808u, 815u, 822u, 829u,
    836u, 843u, 850u, 857u, 864u, 871u, 878u, 885u, 892u, 899u, 906u, 913u,
    920u, 927u, 934u, 941u, 946u, 953u, 960u, 967u
};
static const uint8_t tap_swipe_v6_rank07_trial026_node_child_flags[974] = {
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
    0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x00u,
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
    0x43u, 0x00u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u,
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
    0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x00u, 0x44u, 0x43u,
    0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u,
    0x44u, 0x43u, 0x00u, 0x00u, 0x46u, 0x00u, 0x00u, 0x44u, 0x43u, 0x00u, 0x00u, 0x46u,
    0x00u, 0x00u
};
static const uint8_t tap_swipe_v6_rank07_trial026_node_features[974] = {
    15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 34u,
    0xFFu, 0xFFu, 15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu,
    0xFFu, 24u, 0xFFu, 0xFFu, 15u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u,
    35u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 2u, 0xFFu,
    0xFFu, 30u, 2u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu,
    3u, 0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 22u,
    0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 2u, 30u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu,
    30u, 22u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu, 15u, 24u, 0xFFu, 0xFFu, 34u,
    0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu,
    0xFFu, 30u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 15u,
    35u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 3u, 30u, 0xFFu, 0xFFu, 30u, 0xFFu,
    0xFFu, 15u, 30u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu,
    3u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 15u, 35u,
    0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 15u, 35u, 0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu,
    30u, 22u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 22u,
    0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 24u, 34u, 0xFFu,
    0xFFu, 15u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u,
    15u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 30u, 28u, 0xFFu, 0xFFu, 18u, 0xFFu,
    0xFFu, 30u, 15u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 15u, 31u, 0xFFu, 0xFFu,
    5u, 0xFFu, 0xFFu, 18u, 34u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 24u, 22u,
    0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 33u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu,
    15u, 16u, 0xFFu, 0xFFu, 30u, 0xFFu, 0xFFu, 24u, 19u, 0xFFu, 0xFFu, 28u,
    0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 24u, 28u, 0xFFu,
    0xFFu, 28u, 0xFFu, 0xFFu, 30u, 16u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 5u,
    8u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu, 0xFFu, 2u, 0xFFu,
    0xFFu, 30u, 19u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 18u, 22u, 0xFFu, 0xFFu,
    18u, 0xFFu, 0xFFu, 15u, 22u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 30u, 18u,
    0xFFu, 0xFFu, 22u, 0xFFu, 0xFFu, 30u, 18u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu,
    18u, 28u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 30u, 19u, 0xFFu, 0xFFu, 3u,
    0xFFu, 0xFFu, 15u, 34u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 24u, 22u, 0xFFu,
    0xFFu, 3u, 0xFFu, 0xFFu, 12u, 2u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 30u,
    33u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 30u, 18u, 0xFFu, 0xFFu, 2u, 0xFFu,
    0xFFu, 24u, 20u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 15u, 16u, 0xFFu, 0xFFu,
    5u, 0xFFu, 0xFFu, 30u, 17u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 24u, 32u,
    0xFFu, 0xFFu, 15u, 0xFFu, 0xFFu, 30u, 19u, 0xFFu, 0xFFu, 19u, 0xFFu, 0xFFu,
    24u, 18u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 18u, 34u, 0xFFu, 0xFFu, 0xFFu,
    10u, 7u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 22u, 0xFFu, 0xFFu, 2u,
    0xFFu, 0xFFu, 30u, 15u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 12u, 8u, 0xFFu,
    0xFFu, 5u, 0xFFu, 0xFFu, 30u, 17u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 18u,
    19u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 34u, 0xFFu, 0xFFu, 27u, 0xFFu,
    0xFFu, 15u, 7u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 10u, 8u, 0xFFu, 0xFFu,
    5u, 0xFFu, 0xFFu, 24u, 18u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 24u, 28u,
    0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 30u, 18u, 0xFFu, 0xFFu, 20u, 0xFFu, 0xFFu,
    24u, 28u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 10u, 15u, 0xFFu, 0xFFu, 34u,
    0xFFu, 0xFFu, 30u, 16u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 5u, 7u, 0xFFu,
    0xFFu, 10u, 0xFFu, 0xFFu, 24u, 5u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 15u,
    22u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 12u, 10u, 0xFFu, 0xFFu, 10u, 0xFFu,
    0xFFu, 17u, 10u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 16u, 5u, 0xFFu, 0xFFu,
    17u, 0xFFu, 0xFFu, 7u, 5u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 3u, 24u,
    0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu, 15u, 29u, 0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu,
    18u, 28u, 0xFFu, 0xFFu, 18u, 0xFFu, 0xFFu, 30u, 17u, 0xFFu, 0xFFu, 15u,
    0xFFu, 0xFFu, 30u, 32u, 0xFFu, 0xFFu, 27u, 0xFFu, 0xFFu, 18u, 19u, 0xFFu,
    0xFFu, 34u, 0xFFu, 0xFFu, 30u, 18u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 30u,
    12u, 0xFFu, 0xFFu, 1u, 0xFFu, 0xFFu, 17u, 10u, 0xFFu, 0xFFu, 5u, 0xFFu,
    0xFFu, 24u, 28u, 0xFFu, 0xFFu, 3u, 0xFFu, 0xFFu, 12u, 10u, 0xFFu, 0xFFu,
    5u, 0xFFu, 0xFFu, 16u, 5u, 0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 3u, 30u,
    0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 17u, 8u, 0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu,
    8u, 30u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 10u, 4u, 0xFFu, 0xFFu, 12u,
    0xFFu, 0xFFu, 16u, 5u, 0xFFu, 0xFFu, 13u, 0xFFu, 0xFFu, 15u, 32u, 0xFFu,
    0xFFu, 16u, 0xFFu, 0xFFu, 5u, 34u, 0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu, 24u,
    32u, 0xFFu, 0xFFu, 27u, 0xFFu, 0xFFu, 30u, 12u, 0xFFu, 0xFFu, 19u, 0xFFu,
    0xFFu, 8u, 5u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 18u, 31u, 0xFFu, 0xFFu,
    28u, 0xFFu, 0xFFu, 30u, 19u, 0xFFu, 0xFFu, 7u, 0xFFu, 0xFFu, 24u, 18u,
    0xFFu, 0xFFu, 1u, 0xFFu, 0xFFu, 15u, 7u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu,
    24u, 22u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 10u, 7u, 0xFFu, 0xFFu, 28u,
    0xFFu, 0xFFu, 30u, 19u, 0xFFu, 0xFFu, 2u, 0xFFu, 0xFFu, 18u, 3u, 0xFFu,
    0xFFu, 18u, 0xFFu, 0xFFu, 17u, 24u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 24u,
    28u, 0xFFu, 0xFFu, 0xFFu, 30u, 19u, 0xFFu, 0xFFu, 7u, 0xFFu, 0xFFu, 10u,
    15u, 0xFFu, 0xFFu, 12u, 0xFFu, 0xFFu, 5u, 8u, 0xFFu, 0xFFu, 8u, 0xFFu,
    0xFFu, 15u, 7u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu, 17u, 10u, 0xFFu, 0xFFu,
    5u, 0xFFu, 0xFFu, 15u, 10u, 0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu, 16u, 5u,
    0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 13u, 10u, 0xFFu, 0xFFu, 5u, 0xFFu, 0xFFu,
    8u, 30u, 0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 17u, 15u, 0xFFu, 0xFFu, 15u,
    0xFFu, 0xFFu, 18u, 12u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu, 5u, 14u, 0xFFu,
    0xFFu, 16u, 0xFFu, 0xFFu, 5u, 34u, 0xFFu, 0xFFu, 7u, 0xFFu, 0xFFu, 24u,
    31u, 0xFFu, 0xFFu, 27u, 0xFFu, 0xFFu, 30u, 11u, 0xFFu, 0xFFu, 18u, 0xFFu,
    0xFFu, 15u, 24u, 0xFFu, 0xFFu, 16u, 0xFFu, 0xFFu, 19u, 30u, 0xFFu, 0xFFu,
    2u, 0xFFu, 0xFFu, 18u, 20u, 0xFFu, 0xFFu, 34u, 0xFFu, 0xFFu, 30u, 31u,
    0xFFu, 0xFFu, 28u, 0xFFu, 0xFFu, 18u, 28u, 0xFFu, 0xFFu, 0xFFu, 16u, 5u,
    0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 8u, 5u, 0xFFu, 0xFFu, 10u, 0xFFu, 0xFFu,
    1u, 31u, 0xFFu, 0xFFu, 17u, 0xFFu, 0xFFu, 2u, 30u, 0xFFu, 0xFFu, 6u,
    0xFFu, 0xFFu
};

/* Q20 probability LUT: one unit equals 0.05 probability. */
static const float tap_swipe_v6_rank07_trial026_q20_raw_thresholds[19] = {
    -2.94443893f, -2.19722462f, -1.73460102f, -1.38629436f, -1.09861231f,
    -0.847297847f, -0.619039237f, -0.405465096f, -0.200670689f, 0.0f,
    0.200670689f, 0.405465096f, 0.619039237f, 0.847297847f, 1.09861231f,
    1.38629436f, 1.73460102f, 2.19722462f, 2.94443893f
};
static const float tap_swipe_v6_rank07_trial026_q20_raw_midpoints[20] = {
    -3.66356158f, -2.51230574f, -1.9459101f, -1.55059743f, -1.23676264f,
    -0.969400585f, -0.730887532f, -0.510825634f, -0.302280873f, -0.100083455f,
    0.100083455f, 0.302280873f, 0.510825634f, 0.730887532f, 0.969400585f,
    1.23676264f, 1.55059743f, 1.9459101f, 2.51230574f, 3.66356158f
};

#define TAP_SWIPE_V6_RANK07_TRIAL026_EARLY_EXIT_GROUP_SIZE 4u
#define TAP_SWIPE_V6_RANK07_TRIAL026_EARLY_EXIT_GROUPS 35u
static const float tap_swipe_v6_rank07_trial026_remaining_min[36] = {
    -11.1003008f, -10.4437838f, -9.92456627f, -9.52615547f, -9.06925392f, -8.63884068f,
    -8.22408199f, -7.87501383f, -7.5292201f, -7.09597254f, -6.72580338f, -6.46357203f,
    -6.11539078f, -5.73289108f, -5.38806581f, -5.12739515f, -4.85447836f, -4.53679371f,
    -4.28249979f, -4.06264925f, -3.76645374f, -3.4945302f, -3.23839831f, -2.90760231f,
    -2.66133952f, -2.44688344f, -2.18873763f, -1.92826355f, -1.76006842f, -1.43672371f,
    -1.16662967f, -0.953193426f, -0.779719293f, -0.540675759f, -0.18815209f, 0.0f
};
static const float tap_swipe_v6_rank07_trial026_remaining_max[36] = {
    8.39125061f, 8.06309986f, 7.77687216f, 7.47636366f, 7.12494135f, 6.84327221f,
    6.58267832f, 6.3030858f, 6.08214855f, 5.871768f, 5.68168259f, 5.41785526f,
    5.26533365f, 5.10043192f, 4.95421934f, 4.75254583f, 4.56863546f, 4.39078283f,
    4.21704912f, 4.063797f, 3.76576519f, 3.41647315f, 3.14944863f, 2.98759866f,
    2.65235496f, 2.35276723f, 2.02871847f, 1.79829705f, 1.66315496f, 1.43327391f,
    1.26166201f, 0.894856274f, 0.644814551f, 0.403079212f, 0.278528571f, 0.0f
};

TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE float tap_swipe_v6_rank07_trial026_walk_lgbm_tree(
    uint16_t root_index,
    const t_core* p_core,
    float normalization_factor,
    tap_swipe_v6_rank07_trial026_feature_cache_t* cache)
{
    uint8_t node_index = 0u;
    for (;;) {
        const uint32_t absolute_index = (uint32_t)root_index + (uint32_t)node_index;
        const uint8_t feature_metadata = tap_swipe_v6_rank07_trial026_node_features[absolute_index];
        if (TAP_SWIPE_V6_RANK07_TRIAL026_UNLIKELY(feature_metadata == 0xFFu)) {
            return tap_swipe_v6_rank07_trial026_node_thresholds[absolute_index];
        }
        const uint8_t feature_index = feature_metadata;

        float feature_value =
            tap_swipe_v6_rank07_trial026_get_raw_model_feature(p_core, feature_index, cache) * normalization_factor;

        if (feature_value >= -1.0e-35f && feature_value <= 1.0e-35f) {
            feature_value = 0.0f;
        }
        const uint8_t child_flags = tap_swipe_v6_rank07_trial026_node_child_flags[absolute_index];
        const uint8_t right_child = (uint8_t)(child_flags & 0x3Fu);
        const uint8_t default_left = (uint8_t)((child_flags >> 6) & 0x01u);
        const uint8_t missing_is_zero = (uint8_t)(child_flags >> 7);

        const uint8_t is_missing = (uint8_t)(
            (feature_value != feature_value) || ((missing_is_zero != 0u) && feature_value == 0.0f));

        if (TAP_SWIPE_V6_RANK07_TRIAL026_UNLIKELY(is_missing != 0u)) {
            node_index = default_left != 0u
                ? (uint8_t)(node_index + 1u)
                : right_child;
        } else {
            node_index = feature_value <= tap_swipe_v6_rank07_trial026_node_thresholds[absolute_index]
                ? (uint8_t)(node_index + 1u)
                : right_child;
        }
    }
}

/** Return boosted-tree raw score before the logistic link. */
TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE float tap_swipe_v6_rank07_trial026_predict_raw_score(const t_core* p_core)
{
    tap_swipe_v6_rank07_trial026_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank07_trial026_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t tree_index = 0u; tree_index < TAP_SWIPE_V6_RANK07_TRIAL026_NUM_TREES; ++tree_index) {
        raw_score += tap_swipe_v6_rank07_trial026_walk_lgbm_tree(tap_swipe_v6_rank07_trial026_tree_roots[tree_index], p_core, normalization_factor, &cache);
    }
    return raw_score;
}

/** Convert a raw score to nearest 0.05 probability, returned as Q20 [0..20]. */
TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE uint8_t tap_swipe_v6_rank07_trial026_probability_q20_from_raw(float raw_score)
{
    uint8_t low = 0u;
    uint8_t high = 20u;
    while (low < high) {
        const uint8_t middle = (uint8_t)((low + high) >> 1u);
        if (raw_score < tap_swipe_v6_rank07_trial026_q20_raw_midpoints[middle]) {
            high = middle;
        } else {
            low = (uint8_t)(middle + 1u);
        }
    }
    return low;
}

TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE uint8_t tap_swipe_v6_rank07_trial026_probability_q20(const t_core* p_core)
{
    return tap_swipe_v6_rank07_trial026_probability_q20_from_raw(tap_swipe_v6_rank07_trial026_predict_raw_score(p_core));
}

/** Approximate probability in 0.05 steps; maximum rounding error is 0.025. */
TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE float tap_swipe_v6_rank07_trial026_predict_value(const t_core* p_core)
{
    return 0.05f * (float)tap_swipe_v6_rank07_trial026_probability_q20(p_core);
}

/** Runtime-selectable exact threshold for Q20 values 0.05..0.95; no expf/logf. */
TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE int8_t tap_swipe_v6_rank07_trial026_predict_q20(    const t_core* p_core, uint8_t threshold_q20)
{
    if (threshold_q20 == 0u) { return 1; }
    if (threshold_q20 >= 20u) { return 0; }
    const float raw_threshold = tap_swipe_v6_rank07_trial026_q20_raw_thresholds[threshold_q20 - 1u];
    tap_swipe_v6_rank07_trial026_feature_cache_t cache = {0};
    const float normalization_factor = tap_swipe_v6_rank07_trial026_runtime_normalization_factor(p_core, &cache);
    float raw_score = 0.0f;
    for (uint16_t group = 0u; group < TAP_SWIPE_V6_RANK07_TRIAL026_EARLY_EXIT_GROUPS; ++group) {
        const uint16_t first_tree = (uint16_t)(group * TAP_SWIPE_V6_RANK07_TRIAL026_EARLY_EXIT_GROUP_SIZE);
        uint16_t end_tree = (uint16_t)(first_tree + TAP_SWIPE_V6_RANK07_TRIAL026_EARLY_EXIT_GROUP_SIZE);
        if (end_tree > TAP_SWIPE_V6_RANK07_TRIAL026_NUM_TREES) { end_tree = TAP_SWIPE_V6_RANK07_TRIAL026_NUM_TREES; }
        for (uint16_t tree_index = first_tree; tree_index < end_tree; ++tree_index) {
            raw_score += tap_swipe_v6_rank07_trial026_walk_lgbm_tree(tap_swipe_v6_rank07_trial026_tree_roots[tree_index], p_core, normalization_factor, &cache);
        }
        if (raw_score + tap_swipe_v6_rank07_trial026_remaining_max[group + 1u] < raw_threshold) { return 0; }
        if (raw_score + tap_swipe_v6_rank07_trial026_remaining_min[group + 1u] >= raw_threshold) { return 1; }
    }
    return raw_score >= raw_threshold ? 1 : 0;
}

#define TAP_SWIPE_V6_RANK07_TRIAL026_DEFAULT_THRESHOLD_Q20 9u
TAP_SWIPE_V6_RANK07_TRIAL026_FORCE_INLINE int8_t tap_swipe_v6_rank07_trial026_predict(const t_core* p_core)
{
    return tap_swipe_v6_rank07_trial026_predict_q20(p_core, TAP_SWIPE_V6_RANK07_TRIAL026_DEFAULT_THRESHOLD_Q20);
}

#endif /* TAP_SWIPE_V6_RANK07_TRIAL026_MODEL_H */
