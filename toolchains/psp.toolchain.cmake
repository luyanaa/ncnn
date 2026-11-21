# PlayStation Portable (Allegrex) toolchain for ncnn.
# Requires PSPDEV (e.g. PSPDEV=/usr/local/pspdev) and a built PSPSDK under it.
# psp-gcc's own spec links -lcglue/-lpthreadglue, which live in $PSPDEV/psp/sdk/lib
# and are not on the default search path, so the pspdev toolchain file is what
# actually makes the link succeed.

if(DEFINED ENV{PSPDEV})
    set(PSPDEV $ENV{PSPDEV})
else()
    message(FATAL_ERROR "PSPDEV environment variable is not set")
endif()

# pspdev.cmake names psp-g++. This install has neither psp-g++ nor psp-c++,
# only psp-gcc, which drives cc1plus for C++ sources itself. CMake validates the
# compiler before the toolchain file runs, so the variable has to be set before
# the include; otherwise the missing path is what gets checked.
if(NOT EXISTS "${PSPDEV}/bin/psp-g++")
    set(CMAKE_CXX_COMPILER "${PSPDEV}/bin/psp-gcc")
endif()

include(${PSPDEV}/psp/share/pspdev.cmake)

# repeated after the include: pspdev.cmake assigns the compiler again, and a
# plain set() there does not always lose to the one above
if(NOT EXISTS "${PSPDEV}/bin/psp-g++")
    set(CMAKE_CXX_COMPILER "${PSPDEV}/bin/psp-gcc" CACHE FILEPATH "" FORCE)
endif()

# pspdev.cmake reports the CPU as "mips". That selects ncnn's MSA / Loongson MMI
# probes, neither of which exists on the Allegrex. "psp" takes the branch that
# keeps NCNN_TARGET_ARCH=mips and adds NCNN_PSP_VFPU instead.
set(CMAKE_SYSTEM_PROCESSOR psp)

# -G0: the default small-data GP window is 64 KB and ncnn overflows it.
set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -O2 -G0")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -O2 -G0")
