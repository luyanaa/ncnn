// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// PlayStation Portable runtime support for benchmark binaries.
//
// newlib's printf goes through _write. On a PSP that has no host filesystem
// that write never reaches the machine running the emulator, so this file
// replaces _write and forwards each line to sceKernelPrintf.
//
// PPSSPP high-level-emulates that syscall and copies the text into the log
// given with --log. The remote SIO registers (pspDebugSioPutchar) are not
// emulated, so writing them produces nothing and can spin if the status bit
// reads back as "fifo full".
//
// The main thread must own the VFPU context. Without PSP_THREAD_ATTR_VFPU the
// first lv.q raises a coprocessor-unusable exception.

#include <pspsdk.h>
#include <pspkernel.h>
#include <pspmoduleinfo.h>
#include <pspsysmem.h>

#include <stdio.h>
#include <stdlib.h>

PSP_MODULE_INFO("ncnn", 0, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
/* newlib takes every free byte minus this threshold for its heap. benchncnn's
   weights and workspaces are all malloc'd, so leave only 256 KB outside. */
PSP_HEAP_THRESHOLD_SIZE_KB(256);

/* PPSSPP only records a sceKernelPrintf when the Printf channel is at Info,
   and a headless run is not, so a format it understands is dropped with no
   trace. A format it does not implement is logged at Error, which does show
   up, and that line quotes the format string. Every line is sent with a
   trailing %c, which is unimplemented, and lands in the log as
   "UNIMPL fmt (<text>%c, ...)". 512 covers one benchncnn result row. */
static char ncnn_psp_line[512];
static int ncnn_psp_line_len = 0;

static void ncnn_psp_emit(const char* s)
{
    char buf[512];
    int n = 0;

    /* any specifier the HLE understands would be printed at Info and vanish,
       so leave exactly one and make it one it rejects */
    while (s[n] && n < (int)sizeof(buf) - 3)
    {
        buf[n] = (s[n] == '%') ? '?' : s[n];
        n++;
    }
    buf[n++] = '%';
    buf[n++] = 'c';
    buf[n] = '\0';

    sceKernelPrintf(buf, 0, 0, 0);
}

static void ncnn_psp_flush_line(void)
{
    if (ncnn_psp_line_len <= 0)
        return;

    ncnn_psp_line[ncnn_psp_line_len] = '\0';
    ncnn_psp_emit(ncnn_psp_line);
    ncnn_psp_line_len = 0;
}

int _write(int fd, const void* buf, size_t nbytes)
{
    const char* p = (const char*)buf;
    size_t i;

    (void)fd;

    for (i = 0; i < nbytes; i++)
    {
        char c = p[i];

        if (c == '\n')
        {
            ncnn_psp_flush_line();
            continue;
        }
        if (c == '\r')
            continue;

        if (ncnn_psp_line_len >= (int)sizeof(ncnn_psp_line) - 1)
            ncnn_psp_flush_line();

        ncnn_psp_line[ncnn_psp_line_len++] = c;
    }

    return (int)nbytes;
}

static void ncnn_psp_on_exit(void)
{
    /* stdio flushes from its own atexit hook, which runs before this one
       because it is registered later. Flush again in case it did not, then
       emit any partial line. Parking keeps PPSSPP from closing the log before
       the runner reads it. */
    fflush(stdout);
    fflush(stderr);
    ncnn_psp_flush_line();
    ncnn_psp_emit("finished, parked");

    for (;;)
        sceKernelSleepThread();
}

__attribute__((constructor)) static void ncnn_psp_runtime_atexit(void)
{
    char buf[32];

    atexit(ncnn_psp_on_exit);
    /* how much user RAM is left before newlib claims the heap, so an
       empty-output failure is distinguishable from a crash before main */
    snprintf(buf, sizeof(buf), "free %08x", (unsigned)sceKernelMaxFreeMemSize());
    ncnn_psp_emit(buf);
}
