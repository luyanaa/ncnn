# PS2 EE self-test for src/layer/mips/ps2_mmi.h and the VU0 macro (COP2) wrappers.
#
# Only the .elf is built: the ps2sdk hello Makefile's default target also builds
# an .erl, which this toolchain cannot link because it no longer accepts -mno-crt0.

PS2DEV ?= /usr/local/ps2dev
PS2SDK ?= $(PS2DEV)/ps2sdk
EE_TOOL_PREFIX = $(PS2DEV)/ee/bin/mips64r5900el-ps2-elf-

EE_BIN  = selftest.elf
EE_OBJS = selftest.o
EE_LIBS = -lkernel -ldebug

# allow the bit-reinterpret casts between the different vector shapes used by
# ps2_mmi.h (e.g. uint32x4_t <-> int16x8_t); they generate no code.
EE_CFLAGS := -flax-vector-conversions

# ps2_mmi.h lives in src/layer/mips; this Makefile is in tests/ps2.
EE_INCS := -I$(abspath ../..)/src/layer/mips

all: $(EE_BIN)

run: $(EE_BIN)
	/tmp/run_ps2.sh $(abspath $(EE_BIN)) 15 /tmp/ps2_selftest.png
	convert /tmp/ps2_selftest.png -crop 700x480+630+270 +repage /tmp/ps2_selftest_crop.png

clean:
	rm -f $(EE_OBJS) $(EE_BIN)

include $(PS2SDK)/samples/Makefile.pref
include $(PS2SDK)/samples/Makefile.eeglobal
