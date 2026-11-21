/* ps2_vu1_gemm.c - EE side driver for the VU1 microprogram GEMM engine.
 *
 * Upload flow (per call):
 *   1. Stage A (k-major, zero-padded), B rows and the K header in a
 *      16-byte-aligned EE buffer.
 *   2. VIF1 UNPACK V4-32: stage -> qwords 0 (A), 256 (B), 1023 (header).
 *   3. MSCAL(0) to start the microprogram; DMA completes when the VIF has
 *      consumed the packet, the microprogram runs on VU1 in parallel.
 *   4. Poll VIF1_STAT.VEW == 0 (E-bit reached), then read C straight from
 *      the IO-mapped VU1 data memory at 0x1100C000 (VU manual 3.6.1).
 */

#include <tamtypes.h>
#include <string.h>
#include <dma.h>
#include <packet2.h>
#include <packet2_utils.h>

#include "ps2_vu1_gemm.h"

extern u32 VU1Gemm_CodeStart __attribute__((section(".vudata")));
extern u32 VU1Gemm_CodeEnd __attribute__((section(".vudata")));

#define VU1_GEMM_MAX_K        256
#define VU1_GEMM_B_QWORD      256
#define VU1_GEMM_C_QWORD      512
#define VU1_GEMM_HEADER_QWORD 1023

/* VU1 data memory, IO-mapped on the EE bus. 16 qwords per float quad. */
#define VU1_DATA_MEM ((volatile u32 *)0x1100c000)
#define VIF1_STAT    (*(volatile u32 *)0x10003c00)
#define VIF1_STAT_VEW (1 << 2)

/* stage: A columns (K qwords) + B rows (K qwords) + header (1 qword) */
static u32 vu1_gemm_stage[VU1_GEMM_MAX_K * 4 * 2 + 4] __attribute__((aligned(16)));
static packet2_t *vu1_gemm_packet;
static int vu1_gemm_ready;

int ps2_vu1_gemm_init(void)
{
    u32 program_size;

    if (vu1_gemm_ready)
        return 0;

    dma_channel_initialize(DMA_CHANNEL_VIF1, NULL, 0);
    dma_channel_fast_waits(DMA_CHANNEL_VIF1);

    /* Packet large enough for the MPG program + 513 qwords of unpack data. */
    program_size = packet2_utils_get_packet_size_for_program(&VU1Gemm_CodeStart, &VU1Gemm_CodeEnd) + 1;
    vu1_gemm_packet = packet2_create(program_size + 560, P2_TYPE_NORMAL, P2_MODE_CHAIN, 1);
    if (!vu1_gemm_packet)
        return -1;

    packet2_vif_add_micro_program(vu1_gemm_packet, 0, &VU1Gemm_CodeStart, &VU1Gemm_CodeEnd);
    packet2_utils_vu_add_end_tag(vu1_gemm_packet);
    dma_channel_wait(DMA_CHANNEL_VIF1, 0);
    dma_channel_send_packet2(vu1_gemm_packet, DMA_CHANNEL_VIF1, 1);
    dma_channel_wait(DMA_CHANNEL_VIF1, 0);

    vu1_gemm_ready = 1;
    return 0;
}

int ps2_vu1_gemm_4xKx4(const float *A, const float *B, float *C, int K)
{
    int k, i, j, kp;
    volatile u32 *vu_mem = VU1_DATA_MEM;
    u32 guard;

    if (!vu1_gemm_ready)
        return -1;
    if (K < 1 || K > VU1_GEMM_MAX_K)
        return -2;

    kp = (K + 3) & ~3; /* zero-padded K, multiple of 4 */

    /* Stage A k-major: one qword per k, stage qword k = {A[0..3][k]} */
    for (k = 0; k < kp; k++)
        for (i = 0; i < 4; i++)
        {
            float v = (k < K) ? A[i * K + k] : 0.0f;
            memcpy(&vu1_gemm_stage[k * 4 + i], &v, 4);
        }

    /* Stage B rows right after A */
    for (k = 0; k < K; k++)
        for (j = 0; j < 4; j++)
            memcpy(&vu1_gemm_stage[(kp + k) * 4 + j], &B[k * 4 + j], 4);

    /* Header: K as plain integer in word 3 (w lane) of qword 1023 */
    vu1_gemm_stage[(kp + K) * 4 + 3] = (u32)K;

    packet2_reset(vu1_gemm_packet, 0);
    packet2_utils_vu_add_unpack_data(vu1_gemm_packet, 0, vu1_gemm_stage, kp, 0);
    packet2_utils_vu_add_unpack_data(vu1_gemm_packet, VU1_GEMM_B_QWORD,
                                     vu1_gemm_stage + kp * 4, K, 0);
    packet2_utils_vu_add_unpack_data(vu1_gemm_packet, VU1_GEMM_HEADER_QWORD,
                                     vu1_gemm_stage + (kp + K) * 4, 1, 0);
    packet2_utils_vu_add_start_program(vu1_gemm_packet, 0);
    packet2_utils_vu_add_end_tag(vu1_gemm_packet);

    dma_channel_wait(DMA_CHANNEL_VIF1, 0);
    dma_channel_send_packet2(vu1_gemm_packet, DMA_CHANNEL_VIF1, 1);
    dma_channel_wait(DMA_CHANNEL_VIF1, 0);

    /* Wait for the microprogram to hit its E-bit. */
    guard = 1 << 24;
    while ((VIF1_STAT & VIF1_STAT_VEW) && --guard)
        ;
    if (!guard)
        return -3;

    /* Read C[i][0..3] from VU1 data memory qwords 512..515. */
    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++)
            C[i * 4 + j] =
                *(const volatile float *)&vu_mem[((VU1_GEMM_C_QWORD + i) << 2) + j];

    return 0;
}

void ps2_vu1_gemm_shutdown(void)
{
    if (vu1_gemm_packet)
    {
        packet2_free(vu1_gemm_packet);
        vu1_gemm_packet = 0;
    }
    vu1_gemm_ready = 0;
}
