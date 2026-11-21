// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// PlayStation 2 EE runtime support for benchmark binaries.
//
// On PS2 there is no usable fileio / stdio device unless ps2link or an
// iomanX stack is present, so newlib's printf() would go nowhere.  We
// redirect the newlib write syscall to the GS debug console provided by
// ps2sdk's libdebug, and initialize that console from a constructor so it is
// available before main().

#include <stddef.h>
#include <stdlib.h>
#include <sys/types.h>

#include <debug.h>
#include <kernel.h>

static void ncnn_ps2_on_exit(void)
{
    /* Returning from main makes ps2sdk's crt0 call Exit(), which reboots the
       console into OSDSYS and wipes the console we just printed to.  Park the
       EE instead so the results stay on screen for the screenshot / the user. */
    scr_printf("--------------------------------\n");
    scr_printf("finished, parked (screen kept for inspection)\n");
    while (1)
        SleepThread();
}

__attribute__((constructor)) static void ncnn_ps2_runtime_init(void)
{
    init_scr();
    scr_printf("ncnn on PlayStation 2 (EE R5900)\n");
    scr_printf("--------------------------------\n");
    atexit(ncnn_ps2_on_exit);
}

int _write(int fd, const void* buf, size_t count)
{
    (void)fd;

    const char* p = (const char*)buf;
    for (size_t i = 0; i < count; i++)
    {
        if (p[i] == '\r')
            continue;
        scr_printf("%c", p[i]);
    }

    return (int)count;
}
