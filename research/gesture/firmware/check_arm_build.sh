#!/usr/bin/env bash
# Cross-compile firmware/lean_lite.h for a Cortex-M4F with the real CMSIS headers and check the machine code.
#
#   bash research/gesture/firmware/check_arm_build.sh
#
# Builds against CMSIS 5.9.0 (Core + its bundled DSP library) and against CMSIS-DSP v1.16.2 + CMSIS 5.9.0 Core,
# with clang (--target=thumbv7em-none-eabihf, -O2). Passes when, for both:
#   - it compiles with -Wall -Wextra -Werror,
#   - the handler uses the CLZ and SSAT instructions (__CLZ, __SSAT),
#   - it contains no floating-point instruction,
#   - its only calls are arm_min_q15 / arm_max_q15 (CMSIS-DSP, once per closed block).
# Needs clang, llvm-objdump and git (CMSIS is fetched once into $LEAN_LITE_ARM_WORK, default /tmp/lean_lite_arm).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${LEAN_LITE_ARM_WORK:-/tmp/lean_lite_arm}
CC=${CLANG:-clang}
OBJDUMP=${LLVM_OBJDUMP:-llvm-objdump}
command -v "$OBJDUMP" >/dev/null || OBJDUMP=$(ls /usr/bin/llvm-objdump-* 2>/dev/null | sort -V | tail -1)
mkdir -p "$WORK/stub"

fetch() {  # repo tag dir paths...
    local repo=$1 tag=$2 dir=$3; shift 3
    if [ ! -d "$WORK/$dir" ]; then
        git -c advice.detachedHead=false clone -q --depth 1 --branch "$tag" --filter=blob:none --sparse \
            "https://github.com/ARM-software/$repo.git" "$WORK/$dir"
        git -C "$WORK/$dir" sparse-checkout set "$@" >/dev/null 2>&1
    fi
}
fetch CMSIS_5 5.9.0 cmsis5 CMSIS/Core/Include CMSIS/DSP/Include CMSIS/DSP/PrivateInclude
fetch CMSIS-DSP v1.16.2 cmsisdsp Include PrivateInclude

# Freestanding target: the few libc headers arm_math.h includes.
cat > "$WORK/stub/string.h" <<'EOF'
#pragma once
typedef __SIZE_TYPE__ size_t;
void *memset(void *s, int c, size_t n);
void *memcpy(void *d, const void *s, size_t n);
EOF
cat > "$WORK/stub/math.h" <<'EOF'
#pragma once
float sqrtf(float); double sqrt(double); float fabsf(float); double fabs(double); float logf(float); float expf(float);
float powf(float, float); float sinf(float); float cosf(float); double log(double); double exp(double);
float floorf(float); float log2f(float); double log2(double); float atan2f(float, float);
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
#define isnan(x) __builtin_isnan(x)
#define isinf(x) __builtin_isinf(x)
EOF
cat > "$WORK/tu.c" <<'EOF'
#include "lean_lite.h"
lean_lite_state_t lite_state;
t_cload_lite lite_out;
void lite_step(const uint32_t *wl, const uint16_t *pm) { lean_lite_update(&lite_state, wl, pm, &lite_out); }
uint8_t lite_ready(void) { return lean_lite_ready(&lite_state); }
EOF

check() {  # name dsp_include
    local name=$1 dsp=$2 obj="$WORK/lean_lite_$1.o"
    "$CC" --target=thumbv7em-none-eabihf -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -ffreestanding \
        -std=c99 -Wall -Wextra -Werror -DARM_MATH_CM4 -D__FPU_PRESENT=1 \
        -I "$WORK/stub" -I "$HERE" -I "$dsp/Include" -I "$dsp/PrivateInclude" -I "$WORK/cmsis5/CMSIS/Core/Include" \
        -c "$WORK/tu.c" -o "$obj"
    local dis
    dis=$("$OBJDUMP" -dr --no-show-raw-insn "$obj")
    grep -qE '\sclz\s' <<<"$dis" || { echo "FAIL $name: no CLZ instruction"; exit 1; }
    grep -qE '\sssat\s' <<<"$dis" || { echo "FAIL $name: no SSAT instruction"; exit 1; }
    if grep -qE '\sv(add|sub|mul|div|sqrt|cvt|ldr|str|mov|cmp|abs|neg|fma)' <<<"$dis"; then
        echo "FAIL $name: floating-point instruction in the Lean-lite code"; exit 1
    fi
    local calls
    calls=$(grep -oE 'R_ARM_THM_(CALL|JUMP24)\s+\S+' <<<"$dis" | awk '{print $2}' | sort -u | tr '\n' ' ')
    for c in $calls; do
        case "$c" in arm_min_q15|arm_max_q15) ;; *) echo "FAIL $name: unexpected call $c"; exit 1 ;; esac
    done
    local bytes
    bytes=$(grep -cE '^\s+[0-9a-f]+:' <<<"$dis")
    echo "LEAN_LITE_ARM_BUILD_OK $name instructions=$bytes calls=[${calls% }]"
}
check cmsis5.9.0 "$WORK/cmsis5/CMSIS/DSP"
check cmsis-dsp1.16.2 "$WORK/cmsisdsp"

# tap_gestures.h: integer only, no calls
cat > "$WORK/tg.c" <<'EOF'
#include "tap_gestures.h"
tg_state_t g;
int tg_entry(unsigned char q, unsigned long t) { return (int) tg_update(&g, q, (uint32_t) t); }
EOF
for mode in "-DTG_REPORT_TAP_IMMEDIATELY=0" "-DTG_REPORT_TAP_IMMEDIATELY=1 -DTG_CHAIN_TAPS=1"; do
    # shellcheck disable=SC2086
    "$CC" --target=thumbv7em-none-eabihf -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -ffreestanding \
        -std=c99 -Wall -Wextra -Werror $mode -I "$HERE" -c "$WORK/tg.c" -o "$WORK/tg.o"
    tgdis=$("$OBJDUMP" -dr --no-show-raw-insn "$WORK/tg.o")
    if grep -qE '\sv(add|sub|mul|div|sqrt|cvt|ldr|str|mov|cmp|abs|neg|fma)' <<<"$tgdis"; then
        echo "FAIL tap_gestures ($mode): floating-point instruction"; exit 1
    fi
    if grep -qE 'R_ARM_THM_(CALL|JUMP24)' <<<"$tgdis"; then echo "FAIL tap_gestures ($mode): unexpected call"; exit 1; fi
    echo "TAP_GESTURES_ARM_BUILD_OK ($mode) instructions=$(grep -cE '^\s+[0-9a-f]+:' <<<"$tgdis")"
done
