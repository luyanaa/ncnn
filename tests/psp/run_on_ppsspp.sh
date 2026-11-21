#!/bin/bash
# Run a PSP ELF under PPSSPP and collect the text it prints.
#
# The binaries replace newlib's _write so printf becomes sceKernelPrintf (see
# benchmark/psp/psp_runtime.c). PPSSPP high-level-emulates that call and writes
# the text to the file given with --log. The remote SIO port is not emulated.
#
# Usage: run_on_ppsspp.sh <elf> [seconds] [out.log]
#
# Environment:
#   PPSSPP        emulator command (default: the snap wrapper)
#   PPSSPP_MEM    accepted for compatibility; RAM size follows PSPModel, which
#                 this script pins to 0 (PSP-1000, 32 MB)
#   DISPLAY       X display (default: :0)
set -u

ELF="${1:?usage: run_on_ppsspp.sh <elf> [seconds] [out.log]}"
SECS="${2:-120}"
HOST_OUT="${3:-$HOME/ppsspp-ncnn/run.log}"

PPSSPP="${PPSSPP:-/snap/bin/ppsspp-emulator}"
if [ ! -x "$PPSSPP" ]; then
    echo "run_on_ppsspp.sh: PPSSPP not found at $PPSSPP, set PPSSPP" >&2
    exit 1
fi

# The snap cannot read or write /tmp. Keep the ELF, the log and the extra
# config under the home directory, which the home interface allows.
WORK="$HOME/ppsspp-ncnn"
mkdir -p "$WORK"

case "$HOST_OUT" in
    /tmp/*|/var/tmp/*)
        OUT="$WORK/$(basename "$HOST_OUT")"
        ;;
    *)
        OUT="$HOST_OUT"
        mkdir -p "$(dirname "$OUT")"
        ;;
esac

BASE="$(basename "$ELF")"
case "$BASE" in
    *.elf|*.prx) DEST="$WORK/$BASE" ;;
    *) DEST="$WORK/${BASE}.elf" ;;
esac
# copy even when the source is already in $WORK, so a rebuild is picked up
if [ "$(readlink -f "$ELF")" != "$(readlink -f "$DEST")" ]; then
    cp -f "$ELF" "$DEST"
fi

cat > "$WORK/extra.ini" <<EOF
[SystemParam]
PSPModel = 0
[General]
CheckForNewVersion = False
FirstRun = False
PauseOnLostFocus = False
EOF

# serialise concurrent users of the single emulator. The lock stays on the
# host; the snap never sees it.
LOCK="${TMPDIR:-/tmp}/ppsspp_runner.lock"
exec 9>"$LOCK"
if ! flock -w 1800 9; then
    echo "run_on_ppsspp.sh: timed out waiting for the emulator lock" >&2
    exit 1
fi

ps -eo pid,comm | awk 'tolower($2) ~ /^ppsspp/ || $2 ~ /PPSSPP/ {print $1}' | xargs -r kill -9 2>/dev/null
sleep 1

export DISPLAY="${DISPLAY:-:0}"
# A private config dir. Snap may override this with its own; either way it is
# not ~/.config/ppsspp.
export XDG_CONFIG_HOME="$WORK/xdg"
mkdir -p "$XDG_CONFIG_HOME"

rm -f "$OUT"
( cd "$WORK" && timeout "$((SECS + 30))" "$PPSSPP" --log="$OUT" --escape-exit \
    --windowed --xres 480 --yres 272 --graphics=software \
    --appendconfig="$WORK/extra.ini" "$DEST" > "$WORK/stdout.log" 2>&1 ) &
PID=$!

# the runtime sends every line as sceKernelPrintf("<text>%c"). %c is not
# implemented, so PPSSPP logs it at Error as "UNIMPL fmt (<text>%c, ...)",
# which the default level keeps; a supported format is Info and is dropped.
# The marker is the line "finished, parked", printed once main returns.
# A fault does not exit the process. A model can also compute for minutes
# without printing, so give up only after 15 minutes with nothing new.
fault=0
last_size=0
quiet=0
for _ in $(seq 1 "$((SECS * 2))"); do
    grep -q "finished, parked" "$OUT" 2>/dev/null && break
    if grep -q "Bad memory access" "$OUT" 2>/dev/null; then
        fault=1
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        break
    fi
    size=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
    if [ "$size" = "$last_size" ]; then
        quiet=$((quiet + 1))
        if [ "$quiet" -ge 1800 ] && [ "$size" -gt 0 ]; then
            break
        fi
    else
        quiet=0
        last_size=$size
    fi
    sleep 0.5
done

kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
ps -eo pid,comm | awk 'tolower($2) ~ /^ppsspp/ || $2 ~ /PPSSPP/ {print $1}' | xargs -r kill -9 2>/dev/null

if [ "$HOST_OUT" != "$OUT" ] && [ -s "$OUT" ]; then
    mkdir -p "$(dirname "$HOST_OUT")"
    cp -f "$OUT" "$HOST_OUT"
fi

if [ ! -s "$OUT" ]; then
    echo "run_on_ppsspp.sh: no output captured, emulator log follows" >&2
    cat "$WORK/stdout.log" >&2
    exit 1
fi

if [ "$fault" = 1 ]; then
    echo "run_on_ppsspp.sh: bad memory access, see $OUT" >&2
    exit 1
fi

if ! grep -q "finished, parked" "$OUT" 2>/dev/null; then
    echo "run_on_ppsspp.sh: no end marker, see $OUT" >&2
    exit 1
fi

echo "log: $OUT"
