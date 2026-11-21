#!/bin/bash
# Run a PS2 EE ELF under PCSX2 and capture the GS console as a screenshot.
#
# The benchmark binaries print to ps2sdk's GS debug console (see
# benchmark/ps2/ps2_runtime.c), so a screenshot is the way to read results.
#
# Usage: run_on_pcsx2.sh <elf> [seconds] [out.png]
#
# Environment:
#   PCSX2_QT   path to pcsx2-qt            (default: pcsx2-qt from PATH)
#   PCSX2_DISPLAY  X display to use        (default: :0)
#   EGL_VENDOR_JSON mesa EGL vendor json   (needed on headless/software GL hosts)
set -u

ELF="${1:?usage: run_on_pcsx2.sh <elf> [seconds] [out.png]}"
SECS="${2:-30}"
OUT="${3:-/tmp/ps2screen.png}"

PCSX2_QT="${PCSX2_QT:-$(command -v pcsx2-qt || true)}"
if [ -z "$PCSX2_QT" ]; then
    echo "run_on_pcsx2.sh: pcsx2-qt not found, set PCSX2_QT" >&2
    exit 1
fi

export DISPLAY="${PCSX2_DISPLAY:-:0}"
if [ -n "${EGL_VENDOR_JSON:-}" ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON"
fi

# serialise concurrent users of the single emulator
LOCK="${TMPDIR:-/tmp}/ps2_runner.lock"
exec 9>"$LOCK"
if ! flock -w 1800 9; then
    echo "run_on_pcsx2.sh: timed out waiting for the emulator lock" >&2
    exit 1
fi

LOG="${TMPDIR:-/tmp}/ps2_run_$$.log"

# kill leftovers: the nix wrapper shows up as ".pcsx2-qt-wrapp" in comm
ps -eo pid,comm | awk '$2 ~ /^\.pcsx/ {print $1}' | xargs -r kill -9 2>/dev/null
ps -eo pid,comm | awk '$2 == "pcsx2-qt" {print $1}' | xargs -r kill -9 2>/dev/null
sleep 1

( timeout "$((SECS + 20))" "$PCSX2_QT" -nogui -fastboot -unlimited -elf "$ELF" \
    > "$LOG" 2>&1 ) &

for _ in $(seq 1 60); do
    grep -q "is executing" "$LOG" 2>/dev/null && break
    sleep 0.5
done
sleep "$SECS"

if command -v import >/dev/null 2>&1; then
    import -window root "$OUT" 2>/dev/null && echo "screenshot: $OUT"
fi

ps -eo pid,comm | awk '$2 ~ /^\.pcsx/ {print $1}' | xargs -r kill -9 2>/dev/null
ps -eo pid,comm | awk '$2 == "pcsx2-qt" {print $1}' | xargs -r kill -9 2>/dev/null

grep -aE "is executing|EE pc|Unknown device" "$LOG" | head -5
cp "$LOG" "${TMPDIR:-/tmp}/ps2_last.log" 2>/dev/null || true
