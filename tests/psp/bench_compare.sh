#!/bin/bash
# Compare the scalar PSP build against the build with the VFPU kernels enabled.
#
# Builds benchncnn twice (NCNN_PSP_VFPU=OFF / ON) and runs both under PPSSPP.
# The model set is the PS2 one: squeezenet and mobilenet, fp32 and int8. A
# 32 MB PSP cannot hold more than a 32 MB PS2 can.
#
# Usage: bench_compare.sh [seconds_per_run]
set -eu

SECS="${1:-600}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# under $HOME, not /tmp: the PPSSPP snap cannot read or write /tmp
OUT="$HOME/ppsspp-ncnn/bench"

export PSPDEV="${PSPDEV:-/usr/local/pspdev}"
# exported, not just prepended: cmake re-checks the compiler from a clean
# environment, and psp-g++ is not on the default PATH
export PATH="$PSPDEV/bin:$PATH"

mkdir -p "$OUT"

configure_and_build() {
    local dir="$1" vfpu="$2"
    cmake -S "$ROOT" -B "$ROOT/$dir" \
        -DCMAKE_TOOLCHAIN_FILE="$ROOT/toolchains/psp.toolchain.cmake" \
        -DCMAKE_BUILD_TYPE=Release \
        -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=ON \
        -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_VULKAN=OFF -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF \
        -DNCNN_PIXEL=OFF -DNCNN_PLATFORM_API=OFF -DNCNN_C_API=OFF \
        -DNCNN_DISABLE_PIC=ON -DNCNN_SIMPLESTL=OFF \
        -DNCNN_PSP_VFPU="$vfpu" > "$OUT/config-$vfpu.log"
    cmake --build "$ROOT/$dir" -j8 --target benchncnn
}

configure_and_build build-psp-base OFF
configure_and_build build-psp-opt ON

"$ROOT/tests/psp/run_on_ppsspp.sh" "$ROOT/build-psp-base/benchmark/benchncnn" "$SECS" "$OUT/base.log"
"$ROOT/tests/psp/run_on_ppsspp.sh" "$ROOT/build-psp-opt/benchmark/benchncnn" "$SECS" "$OUT/opt.log"

echo
echo "results:"
python3 - "$OUT/base.log" "$OUT/opt.log" <<'PY'
import re, sys

def decode(path):
    text = open(path, errors="replace").read()
    rows = []
    # the runtime sends each result row as sceKernelPrintf("<text>%c"), which
    # PPSSPP logs at Error as "UNIMPL fmt (<text>%c, ...)". A supported format
    # would be Info and is dropped at the default log level.
    for name, vmin, vmax, vavg in re.findall(
            r'UNIMPL fmt \(([A-Za-z0-9_.+-]+)\s+min =\s*(-?\d+(?:\.\d+)?)\s+max =\s*(-?\d+(?:\.\d+)?)\s+avg =\s*(-?\d+(?:\.\d+)?)%c,',
            text):
        rows.append((name, float(vmin), float(vmax), float(vavg)))
    return rows

base = {r[0]: r for r in decode(sys.argv[1])}
opt = {r[0]: r for r in decode(sys.argv[2])}
names = list(dict.fromkeys(list(base) + list(opt)))
if not names:
    sys.exit("bench_compare: no result rows decoded")
print("%-20s %10s %10s %8s" % ("model", "base avg", "vfpu avg", "speedup"))
for name in names:
    b = base.get(name)
    o = opt.get(name)
    if b and o and o[3] > 0:
        print("%-20s %10.2f %10.2f %7.2fx" % (name, b[3], o[3], b[3] / o[3]))
    else:
        print("%-20s %10s" % (name, "missing"))
PY
