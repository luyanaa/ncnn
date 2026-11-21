#!/bin/bash
# Link one PSP ELF, then group its import stubs by module.
#
# cmake -E cmake_link_script does not run a shell. A link rule written as
# "cmake -E remove -f <stage> && <compiler> ... && psp-group-imports.py" is
# one argv list, so remove -f deletes every later argument: the objects, the
# archive, the psp-gcc hard link, and this tree's copy of the regroup script.
# It still exits 0. Keep the post-link steps inside this one executable.
#
# The -o file is a fixed path outside the build tree. An absolute -o inside
# the build tree makes gcc LTO write its intermediate back under the same
# name, a ".tmp" suffix is the one LTO appends itself, and a leftover -o
# file is treated as another input. /tmp is a different filesystem from the
# build tree, so the result is copied, not renamed. Both bench builds share
# the path; do not link them at the same time.
#
# This install has no psp-g++, and psp-gcc does not link libstdc++. Operator
# new/delete and the cxxabi typeinfo live there, so every C++ link needs it.
set -eu
export PATH="/usr/local/pspdev/bin:${PATH:-}"

stage=/tmp/psplink_stage
rm -f "$stage"

out=
cmd=()
while [ $# -gt 0 ]; do
    case "$1" in
        -o)
            shift
            out=$1
            cmd+=(-o "$stage")
            ;;
        *)
            cmd+=("$1")
            ;;
    esac
    shift
done

if [ -z "$out" ]; then
    echo "psp-link.sh: no -o output" >&2
    exit 1
fi

has_std=0
for a in "${cmd[@]}"; do
    if [ "$a" = "-lstdc++" ]; then
        has_std=1
    fi
done
if [ "$has_std" = 0 ]; then
    cmd+=(-lstdc++)
fi

"${cmd[@]}"
cp -f "$stage" "$out"
rm -f "$stage"
python3 "$(dirname "$0")/psp-group-imports.py" "$out"
