# PlayStation 2 Emotion Engine toolchain for ncnn
# Requires: PS2DEV and PS2SDK env vars (e.g. PS2DEV=/usr/local/ps2dev PS2SDK=/usr/local/ps2dev/ps2sdk)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR mips64r5900)

if(DEFINED ENV{PS2DEV})
    set(PS2DEV $ENV{PS2DEV})
else()
    message(FATAL_ERROR "PS2DEV environment variable is not set")
endif()
if(DEFINED ENV{PS2SDK})
    set(PS2SDK $ENV{PS2SDK})
else()
    message(FATAL_ERROR "PS2SDK environment variable is not set")
endif()

set(EE_BIN ${PS2DEV}/ee/bin)
set(CMAKE_C_COMPILER   ${EE_BIN}/mips64r5900el-ps2-elf-gcc)
set(CMAKE_CXX_COMPILER ${EE_BIN}/mips64r5900el-ps2-elf-g++)
set(CMAKE_ASM_COMPILER ${EE_BIN}/mips64r5900el-ps2-elf-gcc)
set(CMAKE_AR           ${EE_BIN}/mips64r5900el-ps2-elf-ar)
set(CMAKE_RANLIB       ${EE_BIN}/mips64r5900el-ps2-elf-ranlib)

# NCNN_PS2_EXTRA_FLAGS is for one-off measurements (for example
# -DNCNN_PS2_INT8_MMI).  It is empty unless the environment sets it, so a
# normal configure is unchanged.  Passing -DCMAKE_CXX_FLAGS on the command
# line is not an option: it replaces CMAKE_CXX_FLAGS_INIT and drops -D__PS2__.
if(DEFINED ENV{NCNN_PS2_EXTRA_FLAGS})
    set(NCNN_PS2_EXTRA_FLAGS $ENV{NCNN_PS2_EXTRA_FLAGS})
else()
    set(NCNN_PS2_EXTRA_FLAGS "")
endif()

set(EE_COMMON_FLAGS "-I${PS2SDK}/ee/include -I${PS2SDK}/common/include -D_EE -D__PS2__ -O2 -G0 -march=r5900 -mtune=r5900 -mno-check-zero-division -fno-builtin -fcommon ${NCNN_PS2_EXTRA_FLAGS}")
set(CMAKE_C_FLAGS_INIT   "${EE_COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${EE_COMMON_FLAGS} -fno-exceptions -fno-rtti")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-L${PS2SDK}/ee/lib -Wl,-zmax-page-size=128 -T${PS2SDK}/ee/startup/linkfile -ldebug -lkernel")

set(CMAKE_FIND_ROOT_PATH ${PS2DEV} ${PS2DEV}/ee ${PS2SDK})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(PS2 TRUE)
