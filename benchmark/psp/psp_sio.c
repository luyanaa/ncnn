// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// The part of pspsdk's src/debug/sio.c that writes to the remote SIO port,
// copied rather than linked. sio.o also holds pspDebugSioInit, and that calls
// sceHprmEnd, which nothing else in a compute binary needs.
//
// PPSSPP mirrors UART4 into the log given with --log, so this is the whole
// output channel.

#include <stdint.h>

#define PSP_UART4_FIFO  0xBE500000
#define PSP_UART4_STAT  0xBE500018
#define PSP_UART_TXFULL 0x20

static inline uint32_t uart_load(uint32_t addr)
{
    uint32_t v;
    __asm__ __volatile__("lw %0, 0(%1)"
                         : "=r"(v)
                         : "r"(addr));
    return v;
}

static inline void uart_store(uint32_t addr, uint32_t v)
{
    __asm__ __volatile__("sw %0, 0(%1)"
                         :
                         : "r"(v), "r"(addr));
}

void pspDebugSioPutchar(int ch)
{
    while (uart_load(PSP_UART4_STAT) & PSP_UART_TXFULL)
        ;
    uart_store(PSP_UART4_FIFO, (uint32_t)ch);
}

int pspDebugSioPutText(const char* data, int len)
{
    int i;

    for (i = 0; i < len; i++)
    {
        if (data[i] == '\n')
        {
            if (((i > 0) && (data[i - 1] != '\r')) || (i == 0))
                pspDebugSioPutchar('\r');
        }

        pspDebugSioPutchar(data[i]);

        if ((i < (len - 1)) && (data[i] == '\r') && (data[i + 1] != '\n'))
            pspDebugSioPutchar('\n');
    }

    return len;
}

void pspDebugSioPuts(const char* str)
{
    while (*str)
        pspDebugSioPutchar(*str++);

    pspDebugSioPutchar('\r');
    pspDebugSioPutchar('\n');
}
