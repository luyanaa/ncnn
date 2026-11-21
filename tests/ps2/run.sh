#!/bin/sh
# Build and run the PS2 MMI / COP2 self-test under PCSX2, then save a cropped
# screenshot of the on-screen console.
set -e
cd "$(dirname "$0")"
SECS="${1:-15}"
make
./run_on_pcsx2.sh "$(pwd)/selftest.elf" "$SECS" "${TMPDIR:-/tmp}/ps2_selftest.png"
if command -v convert >/dev/null 2>&1; then
    convert "${TMPDIR:-/tmp}/ps2_selftest.png" -crop 700x540+630+250 +repage "${TMPDIR:-/tmp}/ps2_selftest_crop.png"
    echo "cropped screenshot: ${TMPDIR:-/tmp}/ps2_selftest_crop.png"
fi
