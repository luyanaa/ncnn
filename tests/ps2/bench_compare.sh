#!/bin/bash
# Compare the generic PS2 build against the build with the PS2 kernels enabled.
#
# Builds benchncnn twice (NCNN_PS2_MMI=OFF / ON) and runs both under PCSX2,
# cropping the GS console screenshots so the numbers are readable.
#
# Usage: bench_compare.sh [seconds_per_run]
set -eu

SECS="${1:-150}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${TMPDIR:-/tmp}/ps2_bench"

export PS2DEV="${PS2DEV:-/usr/local/ps2dev}"
export PS2SDK="${PS2SDK:-$PS2DEV/ps2sdk}"

mkdir -p "$OUT"

configure_and_build() {
    local dir="$1" mmi="$2"
    cmake -S "$ROOT" -B "$ROOT/$dir" \
        -DCMAKE_TOOLCHAIN_FILE="$ROOT/toolchains/ps2-ee.toolchain.cmake" \
        -DCMAKE_BUILD_TYPE=Release \
        -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=ON \
        -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_VULKAN=OFF -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF \
        -DNCNN_PIXEL=OFF -DNCNN_PLATFORM_API=OFF -DNCNN_C_API=OFF \
        -DNCNN_DISABLE_PIC=ON -DNCNN_RUNTIME_CPU=OFF \
        -DNCNN_PS2_MMI="$mmi" > "$OUT/config-$mmi.log"
    cmake --build "$ROOT/$dir" -j8 --target benchncnn
}

run_one() {
    local tag="$1" elf="$2"
    "$ROOT/tests/ps2/run_on_pcsx2.sh" "$elf" "$SECS" "$OUT/$tag.png"
    if command -v convert >/dev/null 2>&1; then
        convert "$OUT/$tag.png" -crop 700x420+630+255 +repage "$OUT/$tag-crop.png"
    fi
}

configure_and_build build-ps2-base OFF
configure_and_build build-ps2-opt ON

run_one base "$ROOT/build-ps2-base/benchmark/benchncnn"
run_one opt  "$ROOT/build-ps2-opt/benchmark/benchncnn"

echo
echo "results:"
echo "  baseline (NCNN_PS2_MMI=OFF): $OUT/base-crop.png"
echo "  ps2 kernels (NCNN_PS2_MMI=ON): $OUT/opt-crop.png"
