// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTION_MIPS_PS2_KERNEL_H
#define LAYER_CONVOLUTION_MIPS_PS2_KERNEL_H

#include <string.h>

#include "mat.h"

#include "fused_activation.h"

namespace ncnn {

// 4 output channels x 4 input channels per VU op pair
//
// bottom_packed is the padded input transposed to pixel-major layout
//   bottom_packed[pixel * inch4 + q]
// so that a single lqc2 fetches the 4 consecutive input channels q..q+3 of
// one pixel (channel-major Mat layout cannot do that, a plain lqc2 there
// would fetch 4 neighbouring pixels of the same channel).
// space_ofs_bytes[k] is the byte offset of kernel tap k inside one pixel row.
//
// SPR weight staging:
// The PS2 EE has a 16KB scratchpad (SPR) at physical address 0x70000000,
// accessible with plain load/store from the EE (it sits outside the data
// cache, so reads have stable low latency - that is exactly what makes it
// better than main memory for small hot data).  Before all output pixels of
// one 4-output-channel group are computed, the group's packed weights
// (maxk * inch4 * 4 channels * 4 bytes) are copied into SPR once; every
// output pixel of the group then reads its weights from SPR instead of
// re-reading them from main memory.  Groups whose weights do not fit into
// 16KB fall back to reading from main memory.
//
// NOTE on concurrency: SPR is a single global 16KB buffer shared by the
// whole EE.  ncnn on PS2 is built with NCNN_THREADS=OFF and runs with
// opt.num_threads == 1; use_spr must be false whenever opt.num_threads > 1
// (see convolution_mips_ps2.cpp) so that no two threads ever stage weights
// at the same time.
static void convolution_ps2_pack4(const Mat& bottom_packed, Mat& top_blob,
                                  const Mat& weight_data_ps2, const Mat& bias_data,
                                  int inch4, int outch, int maxk,
                                  int outw, int outh, int w,
                                  int stride_w, int stride_h,
                                  const int* space_ofs_bytes,
                                  bool use_spr,
                                  int activation_type, const Mat& activation_params)
{
    const bool bias_term = !bias_data.empty();
    const float* biasptr = bias_data;

    const int wch = maxk * inch4; // floats per output channel of weight_data_ps2

    // bytes needed to stage one 4-output-channel group in SPR
    const size_t group_bytes = (size_t)wch * 4 * 4;
    const bool spr_fits = use_spr && group_bytes <= 16384u;

    // physical SPR address, EE-visible in the uncached segment
    float* const spr_base = (float*)0x70000000;

    for (int p = 0; p < outch; p += 4)
    {
        const int n = outch - p < 4 ? outch - p : 4;

        float bv[4] = {0.f, 0.f, 0.f, 0.f};
        for (int c = 0; c < n; c++)
        {
            if (bias_term)
                bv[c] = biasptr[p + c];
        }

        float* outptr[4];
        for (int c = 0; c < n; c++)
            outptr[c] = top_blob.channel(p + c);

        const float* wb = (const float*)weight_data_ps2 + (size_t)p * wch;

        // stage this output group's weights in SPR: one memcpy, then every
        // output pixel below reads its weights from SPR.  The staged layout
        // is identical ([outch][maxk][inch4], 4 outchs), so no kernel-side
        // indexing change is needed.
        if (spr_fits)
        {
            memcpy(spr_base, wb, group_bytes);
            wb = spr_base;
        }

        for (int i = 0; i < outh; i++)
        {
            const float* xrow = (const float*)bottom_packed + (size_t)i * stride_h * w * inch4;

            float* orow[4];
            for (int c = 0; c < n; c++)
                orow[c] = outptr[c] + (size_t)i * outw;

            // eight neighbouring output pixels share one weight load; the
            // 4-pixel loop below handles the tail.  Pointwise weights do not
            // depend on the pixel, and reloading them dominates the shrinking
            // ("squeeze") layers.
            int j = 0;
            for (; j + 7 < outw; j += 8)
            {
                const char* xb = (const char*)(xrow + (size_t)j * stride_w * inch4);
                const int xstride = stride_w * inch4 * (int)sizeof(float);

                float accv[128] __attribute__((aligned(16)));

                __asm__ __volatile__(
                    "vsub.xyzw $vf4, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf5, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf6, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf7, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf8, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf9, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf10, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf11, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf18, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf19, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf20, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf21, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf22, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf23, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf24, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf25, $vf0, $vf0\n\t"
                    "move $8, %[xb]\n\t"
                    "move $9, %[sofs]\n\t"
                    "move $10, %[wb]\n\t"
                    "move $11, %[kmax]\n\t"
                    "1:\n\t"
                    "lw $2, 0($9)\n\t"
                    "addiu $9, $9, 4\n\t"
                    "addu $12, $8, $2\n\t"
                    "move $13, %[nq]\n\t"
                    "move $14, $10\n\t"
                    "addu $15, $14, %[wstep]\n\t"
                    "addu $16, $15, %[wstep]\n\t"
                    "addu $17, $16, %[wstep]\n\t"
                    "2:\n\t"
                    "lqc2 $vf13, 0($14)\n\t"
                    "lqc2 $vf14, 0($15)\n\t"
                    "lqc2 $vf15, 0($16)\n\t"
                    "lqc2 $vf16, 0($17)\n\t"
                    "lqc2 $vf1, 0($12)\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf1\n\t"
                    "vadd.xyzw $vf4, $vf4, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf1\n\t"
                    "vadd.xyzw $vf8, $vf8, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf1\n\t"
                    "vadd.xyzw $vf18, $vf18, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf1\n\t"
                    "vadd.xyzw $vf22, $vf22, $vf17\n\t"
                    "addu $12, $12, %[xs]\n\t"
                    "lqc2 $vf1, 0($12)\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf1\n\t"
                    "vadd.xyzw $vf5, $vf5, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf1\n\t"
                    "vadd.xyzw $vf9, $vf9, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf1\n\t"
                    "vadd.xyzw $vf19, $vf19, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf1\n\t"
                    "vadd.xyzw $vf23, $vf23, $vf17\n\t"
                    "addu $12, $12, %[xs]\n\t"
                    "lqc2 $vf1, 0($12)\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf1\n\t"
                    "vadd.xyzw $vf6, $vf6, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf1\n\t"
                    "vadd.xyzw $vf10, $vf10, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf1\n\t"
                    "vadd.xyzw $vf20, $vf20, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf1\n\t"
                    "vadd.xyzw $vf24, $vf24, $vf17\n\t"
                    "addu $12, $12, %[xs]\n\t"
                    "lqc2 $vf1, 0($12)\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf1\n\t"
                    "vadd.xyzw $vf7, $vf7, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf1\n\t"
                    "vadd.xyzw $vf11, $vf11, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf1\n\t"
                    "vadd.xyzw $vf21, $vf21, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf1\n\t"
                    "vadd.xyzw $vf25, $vf25, $vf17\n\t"
                    "subu $12, $12, %[xs3]\n\t"
                    "addiu $12, $12, 16\n\t"
                    "addiu $14, $14, 16\n\t"
                    "addiu $15, $15, 16\n\t"
                    "addiu $16, $16, 16\n\t"
                    "addiu $17, $17, 16\n\t"
                    "addiu $13, $13, -1\n\t"
                    "bgtz $13, 2b\n\t"
                    "nop\n\t"
                    "addu $10, $10, %[kstep]\n\t"
                    "addiu $11, $11, -1\n\t"
                    "bgtz $11, 1b\n\t"
                    "nop\n\t"
                    "vaddy.x $vf4, $vf4, $vf4\n\t" "vaddz.x $vf4, $vf4, $vf4\n\t" "vaddw.x $vf4, $vf4, $vf4\n\t"
                    "vaddy.x $vf5, $vf5, $vf5\n\t" "vaddz.x $vf5, $vf5, $vf5\n\t" "vaddw.x $vf5, $vf5, $vf5\n\t"
                    "vaddy.x $vf6, $vf6, $vf6\n\t" "vaddz.x $vf6, $vf6, $vf6\n\t" "vaddw.x $vf6, $vf6, $vf6\n\t"
                    "vaddy.x $vf7, $vf7, $vf7\n\t" "vaddz.x $vf7, $vf7, $vf7\n\t" "vaddw.x $vf7, $vf7, $vf7\n\t"
                    "vaddy.x $vf8, $vf8, $vf8\n\t" "vaddz.x $vf8, $vf8, $vf8\n\t" "vaddw.x $vf8, $vf8, $vf8\n\t"
                    "vaddy.x $vf9, $vf9, $vf9\n\t" "vaddz.x $vf9, $vf9, $vf9\n\t" "vaddw.x $vf9, $vf9, $vf9\n\t"
                    "vaddy.x $vf10, $vf10, $vf10\n\t" "vaddz.x $vf10, $vf10, $vf10\n\t" "vaddw.x $vf10, $vf10, $vf10\n\t"
                    "vaddy.x $vf11, $vf11, $vf11\n\t" "vaddz.x $vf11, $vf11, $vf11\n\t" "vaddw.x $vf11, $vf11, $vf11\n\t"
                    "vaddy.x $vf18, $vf18, $vf18\n\t" "vaddz.x $vf18, $vf18, $vf18\n\t" "vaddw.x $vf18, $vf18, $vf18\n\t"
                    "vaddy.x $vf19, $vf19, $vf19\n\t" "vaddz.x $vf19, $vf19, $vf19\n\t" "vaddw.x $vf19, $vf19, $vf19\n\t"
                    "vaddy.x $vf20, $vf20, $vf20\n\t" "vaddz.x $vf20, $vf20, $vf20\n\t" "vaddw.x $vf20, $vf20, $vf20\n\t"
                    "vaddy.x $vf21, $vf21, $vf21\n\t" "vaddz.x $vf21, $vf21, $vf21\n\t" "vaddw.x $vf21, $vf21, $vf21\n\t"
                    "vaddy.x $vf22, $vf22, $vf22\n\t" "vaddz.x $vf22, $vf22, $vf22\n\t" "vaddw.x $vf22, $vf22, $vf22\n\t"
                    "vaddy.x $vf23, $vf23, $vf23\n\t" "vaddz.x $vf23, $vf23, $vf23\n\t" "vaddw.x $vf23, $vf23, $vf23\n\t"
                    "vaddy.x $vf24, $vf24, $vf24\n\t" "vaddz.x $vf24, $vf24, $vf24\n\t" "vaddw.x $vf24, $vf24, $vf24\n\t"
                    "vaddy.x $vf25, $vf25, $vf25\n\t" "vaddz.x $vf25, $vf25, $vf25\n\t" "vaddw.x $vf25, $vf25, $vf25\n\t"
                    "nop\n\t" "nop\n\t" "nop\n\t" "nop\n\t"
                    "sqc2 $vf4, 0(%[ov])\n\t" "sqc2 $vf8, 16(%[ov])\n\t" "sqc2 $vf18, 32(%[ov])\n\t" "sqc2 $vf22, 48(%[ov])\n\t"
                    "sqc2 $vf5, 64(%[ov])\n\t" "sqc2 $vf9, 80(%[ov])\n\t" "sqc2 $vf19, 96(%[ov])\n\t" "sqc2 $vf23, 112(%[ov])\n\t"
                    "sqc2 $vf6, 128(%[ov])\n\t" "sqc2 $vf10, 144(%[ov])\n\t" "sqc2 $vf20, 160(%[ov])\n\t" "sqc2 $vf24, 176(%[ov])\n\t"
                    "sqc2 $vf7, 192(%[ov])\n\t" "sqc2 $vf11, 208(%[ov])\n\t" "sqc2 $vf21, 224(%[ov])\n\t" "sqc2 $vf25, 240(%[ov])\n\t"
                    :
                    : [xb] "r"(xb), [xs] "r"(xstride), [xs3] "r"(3 * xstride), [sofs] "r"(space_ofs_bytes), [wb] "r"(wb), [kmax] "r"(maxk), [nq] "r"(inch4 >> 2), [wstep] "r"(wch * 4), [kstep] "r"(inch4 * 4), [ov] "r"(accv)
                    : "$2", "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15", "$16", "$17", "memory");

                // only the first four of the eight pixels were accumulated
                // above; the second four need their own pass because VU0 has
                // only 32 vector registers.  Redo with the pixel base shifted.
                const char* xb2 = xb + 4 * xstride;
                __asm__ __volatile__(
                    "vsub.xyzw $vf4, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf5, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf6, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf7, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf8, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf9, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf10, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf11, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf18, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf19, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf20, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf21, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf22, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf23, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf24, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf25, $vf0, $vf0\n\t"
                    "move $8, %[xb]\n\t"
                    "move $9, %[sofs]\n\t"
                    "move $10, %[wb]\n\t"
                    "move $11, %[kmax]\n\t"
                    "1:\n\t"
                    "lw $2, 0($9)\n\t"
                    "addiu $9, $9, 4\n\t"
                    "addu $12, $8, $2\n\t"
                    "move $13, %[nq]\n\t"
                    "move $14, $10\n\t"
                    "addu $15, $14, %[wstep]\n\t"
                    "addu $16, $15, %[wstep]\n\t"
                    "addu $17, $16, %[wstep]\n\t"
                    "2:\n\t"
                    "lqc2 $vf13, 0($14)\n\t"
                    "lqc2 $vf14, 0($15)\n\t"
                    "lqc2 $vf15, 0($16)\n\t"
                    "lqc2 $vf16, 0($17)\n\t"
                    "lqc2 $vf1, 0($12)\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf1\n\t" "vadd.xyzw $vf4, $vf4, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf1\n\t" "vadd.xyzw $vf8, $vf8, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf1\n\t" "vadd.xyzw $vf18, $vf18, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf1\n\t" "vadd.xyzw $vf22, $vf22, $vf17\n\t"
                    "addu $12, $12, %[xs]\n\t"
                    "lqc2 $vf1, 0($12)\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf1\n\t" "vadd.xyzw $vf5, $vf5, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf1\n\t" "vadd.xyzw $vf9, $vf9, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf1\n\t" "vadd.xyzw $vf19, $vf19, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf1\n\t" "vadd.xyzw $vf23, $vf23, $vf17\n\t"
                    "addu $12, $12, %[xs]\n\t"
                    "lqc2 $vf1, 0($12)\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf1\n\t" "vadd.xyzw $vf6, $vf6, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf1\n\t" "vadd.xyzw $vf10, $vf10, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf1\n\t" "vadd.xyzw $vf20, $vf20, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf1\n\t" "vadd.xyzw $vf24, $vf24, $vf17\n\t"
                    "addu $12, $12, %[xs]\n\t"
                    "lqc2 $vf1, 0($12)\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf1\n\t" "vadd.xyzw $vf7, $vf7, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf1\n\t" "vadd.xyzw $vf11, $vf11, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf1\n\t" "vadd.xyzw $vf21, $vf21, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf1\n\t" "vadd.xyzw $vf25, $vf25, $vf17\n\t"
                    "subu $12, $12, %[xs3]\n\t"
                    "addiu $12, $12, 16\n\t"
                    "addiu $14, $14, 16\n\t" "addiu $15, $15, 16\n\t" "addiu $16, $16, 16\n\t" "addiu $17, $17, 16\n\t"
                    "addiu $13, $13, -1\n\t"
                    "bgtz $13, 2b\n\t"
                    "nop\n\t"
                    "addu $10, $10, %[kstep]\n\t"
                    "addiu $11, $11, -1\n\t"
                    "bgtz $11, 1b\n\t"
                    "nop\n\t"
                    "vaddy.x $vf4, $vf4, $vf4\n\t" "vaddz.x $vf4, $vf4, $vf4\n\t" "vaddw.x $vf4, $vf4, $vf4\n\t"
                    "vaddy.x $vf5, $vf5, $vf5\n\t" "vaddz.x $vf5, $vf5, $vf5\n\t" "vaddw.x $vf5, $vf5, $vf5\n\t"
                    "vaddy.x $vf6, $vf6, $vf6\n\t" "vaddz.x $vf6, $vf6, $vf6\n\t" "vaddw.x $vf6, $vf6, $vf6\n\t"
                    "vaddy.x $vf7, $vf7, $vf7\n\t" "vaddz.x $vf7, $vf7, $vf7\n\t" "vaddw.x $vf7, $vf7, $vf7\n\t"
                    "vaddy.x $vf8, $vf8, $vf8\n\t" "vaddz.x $vf8, $vf8, $vf8\n\t" "vaddw.x $vf8, $vf8, $vf8\n\t"
                    "vaddy.x $vf9, $vf9, $vf9\n\t" "vaddz.x $vf9, $vf9, $vf9\n\t" "vaddw.x $vf9, $vf9, $vf9\n\t"
                    "vaddy.x $vf10, $vf10, $vf10\n\t" "vaddz.x $vf10, $vf10, $vf10\n\t" "vaddw.x $vf10, $vf10, $vf10\n\t"
                    "vaddy.x $vf11, $vf11, $vf11\n\t" "vaddz.x $vf11, $vf11, $vf11\n\t" "vaddw.x $vf11, $vf11, $vf11\n\t"
                    "vaddy.x $vf18, $vf18, $vf18\n\t" "vaddz.x $vf18, $vf18, $vf18\n\t" "vaddw.x $vf18, $vf18, $vf18\n\t"
                    "vaddy.x $vf19, $vf19, $vf19\n\t" "vaddz.x $vf19, $vf19, $vf19\n\t" "vaddw.x $vf19, $vf19, $vf19\n\t"
                    "vaddy.x $vf20, $vf20, $vf20\n\t" "vaddz.x $vf20, $vf20, $vf20\n\t" "vaddw.x $vf20, $vf20, $vf20\n\t"
                    "vaddy.x $vf21, $vf21, $vf21\n\t" "vaddz.x $vf21, $vf21, $vf21\n\t" "vaddw.x $vf21, $vf21, $vf21\n\t"
                    "vaddy.x $vf22, $vf22, $vf22\n\t" "vaddz.x $vf22, $vf22, $vf22\n\t" "vaddw.x $vf22, $vf22, $vf22\n\t"
                    "vaddy.x $vf23, $vf23, $vf23\n\t" "vaddz.x $vf23, $vf23, $vf23\n\t" "vaddw.x $vf23, $vf23, $vf23\n\t"
                    "vaddy.x $vf24, $vf24, $vf24\n\t" "vaddz.x $vf24, $vf24, $vf24\n\t" "vaddw.x $vf24, $vf24, $vf24\n\t"
                    "vaddy.x $vf25, $vf25, $vf25\n\t" "vaddz.x $vf25, $vf25, $vf25\n\t" "vaddw.x $vf25, $vf25, $vf25\n\t"
                    "nop\n\t" "nop\n\t" "nop\n\t" "nop\n\t"
                    "sqc2 $vf4, 256(%[ov])\n\t" "sqc2 $vf8, 272(%[ov])\n\t" "sqc2 $vf18, 288(%[ov])\n\t" "sqc2 $vf22, 304(%[ov])\n\t"
                    "sqc2 $vf5, 320(%[ov])\n\t" "sqc2 $vf9, 336(%[ov])\n\t" "sqc2 $vf19, 352(%[ov])\n\t" "sqc2 $vf23, 368(%[ov])\n\t"
                    "sqc2 $vf6, 384(%[ov])\n\t" "sqc2 $vf10, 400(%[ov])\n\t" "sqc2 $vf20, 416(%[ov])\n\t" "sqc2 $vf24, 432(%[ov])\n\t"
                    "sqc2 $vf7, 448(%[ov])\n\t" "sqc2 $vf11, 464(%[ov])\n\t" "sqc2 $vf21, 480(%[ov])\n\t" "sqc2 $vf25, 496(%[ov])\n\t"
                    :
                    : [xb] "r"(xb2), [xs] "r"(xstride), [xs3] "r"(3 * xstride), [sofs] "r"(space_ofs_bytes), [wb] "r"(wb), [kmax] "r"(maxk), [nq] "r"(inch4 >> 2), [wstep] "r"(wch * 4), [kstep] "r"(inch4 * 4), [ov] "r"(accv)
                    : "$2", "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15", "$16", "$17", "memory");

                for (int t = 0; t < 8; t++)
                    for (int c = 0; c < n; c++)
                        orow[c][j + t] = activation_ss(accv[t * 16 + c * 4] + bv[c], activation_type, activation_params);
            }

            for (; j + 3 < outw; j += 4)
            {
                const char* xb = (const char*)(xrow + (size_t)j * stride_w * inch4);
                const int xstride = stride_w * inch4 * (int)sizeof(float);

                float accv[64] __attribute__((aligned(16)));

                __asm__ __volatile__(
                    "vsub.xyzw $vf4, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf5, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf6, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf7, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf8, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf9, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf10, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf11, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf18, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf19, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf20, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf21, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf22, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf23, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf24, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf25, $vf0, $vf0\n\t"
                    "move $8, %[xb]\n\t"
                    "move $9, %[sofs]\n\t"
                    "move $10, %[wb]\n\t"
                    "move $11, %[kmax]\n\t"
                    "1:\n\t"
                    "lw $2, 0($9)\n\t"
                    "addiu $9, $9, 4\n\t"
                    "addu $12, $8, $2\n\t"
                    "addu $3, $12, %[xs]\n\t"
                    "addu $24, $3, %[xs]\n\t"
                    "addu $25, $24, %[xs]\n\t"
                    "move $13, %[nq]\n\t"
                    "move $14, $10\n\t"
                    "addu $15, $14, %[wstep]\n\t"
                    "addu $16, $15, %[wstep]\n\t"
                    "addu $17, $16, %[wstep]\n\t"
                    "2:\n\t"
                    "lqc2 $vf1, 0($12)\n\t"
                    "lqc2 $vf2, 0($3)\n\t"
                    "lqc2 $vf3, 0($24)\n\t"
                    "lqc2 $vf26, 0($25)\n\t"
                    "lqc2 $vf13, 0($14)\n\t"
                    "lqc2 $vf14, 0($15)\n\t"
                    "lqc2 $vf15, 0($16)\n\t"
                    "lqc2 $vf16, 0($17)\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf1\n\t"
                    "vadd.xyzw $vf4, $vf4, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf2\n\t"
                    "vadd.xyzw $vf5, $vf5, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf3\n\t"
                    "vadd.xyzw $vf6, $vf6, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf26\n\t"
                    "vadd.xyzw $vf7, $vf7, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf1\n\t"
                    "vadd.xyzw $vf8, $vf8, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf2\n\t"
                    "vadd.xyzw $vf9, $vf9, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf3\n\t"
                    "vadd.xyzw $vf10, $vf10, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf26\n\t"
                    "vadd.xyzw $vf11, $vf11, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf1\n\t"
                    "vadd.xyzw $vf18, $vf18, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf2\n\t"
                    "vadd.xyzw $vf19, $vf19, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf3\n\t"
                    "vadd.xyzw $vf20, $vf20, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf26\n\t"
                    "vadd.xyzw $vf21, $vf21, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf1\n\t"
                    "vadd.xyzw $vf22, $vf22, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf2\n\t"
                    "vadd.xyzw $vf23, $vf23, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf3\n\t"
                    "vadd.xyzw $vf24, $vf24, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf26\n\t"
                    "vadd.xyzw $vf25, $vf25, $vf17\n\t"
                    "addiu $12, $12, 16\n\t"
                    "addiu $3, $3, 16\n\t"
                    "addiu $24, $24, 16\n\t"
                    "addiu $25, $25, 16\n\t"
                    "addiu $14, $14, 16\n\t"
                    "addiu $15, $15, 16\n\t"
                    "addiu $16, $16, 16\n\t"
                    "addiu $17, $17, 16\n\t"
                    "addiu $13, $13, -1\n\t"
                    "bgtz $13, 2b\n\t"
                    "nop\n\t"
                    "addu $10, $10, %[kstep]\n\t"
                    "addiu $11, $11, -1\n\t"
                    "bgtz $11, 1b\n\t"
                    "nop\n\t"
                    "vaddy.x $vf4, $vf4, $vf4\n\t"
                    "vaddz.x $vf4, $vf4, $vf4\n\t"
                    "vaddw.x $vf4, $vf4, $vf4\n\t"
                    "vaddy.x $vf5, $vf5, $vf5\n\t"
                    "vaddz.x $vf5, $vf5, $vf5\n\t"
                    "vaddw.x $vf5, $vf5, $vf5\n\t"
                    "vaddy.x $vf6, $vf6, $vf6\n\t"
                    "vaddz.x $vf6, $vf6, $vf6\n\t"
                    "vaddw.x $vf6, $vf6, $vf6\n\t"
                    "vaddy.x $vf7, $vf7, $vf7\n\t"
                    "vaddz.x $vf7, $vf7, $vf7\n\t"
                    "vaddw.x $vf7, $vf7, $vf7\n\t"
                    "vaddy.x $vf8, $vf8, $vf8\n\t"
                    "vaddz.x $vf8, $vf8, $vf8\n\t"
                    "vaddw.x $vf8, $vf8, $vf8\n\t"
                    "vaddy.x $vf9, $vf9, $vf9\n\t"
                    "vaddz.x $vf9, $vf9, $vf9\n\t"
                    "vaddw.x $vf9, $vf9, $vf9\n\t"
                    "vaddy.x $vf10, $vf10, $vf10\n\t"
                    "vaddz.x $vf10, $vf10, $vf10\n\t"
                    "vaddw.x $vf10, $vf10, $vf10\n\t"
                    "vaddy.x $vf11, $vf11, $vf11\n\t"
                    "vaddz.x $vf11, $vf11, $vf11\n\t"
                    "vaddw.x $vf11, $vf11, $vf11\n\t"
                    "vaddy.x $vf18, $vf18, $vf18\n\t"
                    "vaddz.x $vf18, $vf18, $vf18\n\t"
                    "vaddw.x $vf18, $vf18, $vf18\n\t"
                    "vaddy.x $vf19, $vf19, $vf19\n\t"
                    "vaddz.x $vf19, $vf19, $vf19\n\t"
                    "vaddw.x $vf19, $vf19, $vf19\n\t"
                    "vaddy.x $vf20, $vf20, $vf20\n\t"
                    "vaddz.x $vf20, $vf20, $vf20\n\t"
                    "vaddw.x $vf20, $vf20, $vf20\n\t"
                    "vaddy.x $vf21, $vf21, $vf21\n\t"
                    "vaddz.x $vf21, $vf21, $vf21\n\t"
                    "vaddw.x $vf21, $vf21, $vf21\n\t"
                    "vaddy.x $vf22, $vf22, $vf22\n\t"
                    "vaddz.x $vf22, $vf22, $vf22\n\t"
                    "vaddw.x $vf22, $vf22, $vf22\n\t"
                    "vaddy.x $vf23, $vf23, $vf23\n\t"
                    "vaddz.x $vf23, $vf23, $vf23\n\t"
                    "vaddw.x $vf23, $vf23, $vf23\n\t"
                    "vaddy.x $vf24, $vf24, $vf24\n\t"
                    "vaddz.x $vf24, $vf24, $vf24\n\t"
                    "vaddw.x $vf24, $vf24, $vf24\n\t"
                    "vaddy.x $vf25, $vf25, $vf25\n\t"
                    "vaddz.x $vf25, $vf25, $vf25\n\t"
                    "vaddw.x $vf25, $vf25, $vf25\n\t"
                    "nop\n\t"
                    "nop\n\t"
                    "nop\n\t"
                    "nop\n\t"
                    "sqc2 $vf4, 0(%[ov])\n\t"
                    "sqc2 $vf8, 16(%[ov])\n\t"
                    "sqc2 $vf18, 32(%[ov])\n\t"
                    "sqc2 $vf22, 48(%[ov])\n\t"
                    "sqc2 $vf5, 64(%[ov])\n\t"
                    "sqc2 $vf9, 80(%[ov])\n\t"
                    "sqc2 $vf19, 96(%[ov])\n\t"
                    "sqc2 $vf23, 112(%[ov])\n\t"
                    "sqc2 $vf6, 128(%[ov])\n\t"
                    "sqc2 $vf10, 144(%[ov])\n\t"
                    "sqc2 $vf20, 160(%[ov])\n\t"
                    "sqc2 $vf24, 176(%[ov])\n\t"
                    "sqc2 $vf7, 192(%[ov])\n\t"
                    "sqc2 $vf11, 208(%[ov])\n\t"
                    "sqc2 $vf21, 224(%[ov])\n\t"
                    "sqc2 $vf25, 240(%[ov])\n\t"
                    :
                    : [xb] "r"(xb), [xs] "r"(xstride), [sofs] "r"(space_ofs_bytes), [wb] "r"(wb), [kmax] "r"(maxk), [nq] "r"(inch4 >> 2), [wstep] "r"(wch * 4), [kstep] "r"(inch4 * 4), [ov] "r"(accv)
                    : "$2", "$3", "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15", "$16", "$17", "$24", "$25", "memory");

                for (int t = 0; t < 4; t++)
                    for (int c = 0; c < n; c++)
                        orow[c][j + t] = activation_ss(accv[t * 16 + c * 4] + bv[c], activation_type, activation_params);
            }

            for (; j < outw; j++)
            {
                const char* xbase = (const char*)(xrow + (size_t)j * stride_w * inch4);

                float accv[16] __attribute__((aligned(16)));

                __asm__ __volatile__(
                    "vsub.xyzw $vf8, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf9, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf10, $vf0, $vf0\n\t"
                    "vsub.xyzw $vf11, $vf0, $vf0\n\t"
                    "move $8, %[xb]\n\t"
                    "move $9, %[sofs]\n\t"
                    "move $10, %[wb]\n\t"
                    "move $11, %[kmax]\n\t"
                    "1:\n\t"
                    "lw $2, 0($9)\n\t"
                    "addiu $9, $9, 4\n\t"
                    "addu $12, $8, $2\n\t"
                    "move $13, %[nq]\n\t"
                    "move $14, $10\n\t"
                    "addu $15, $14, %[wstep]\n\t"
                    "addu $16, $15, %[wstep]\n\t"
                    "addu $17, $16, %[wstep]\n\t"
                    "2:\n\t"
                    "lqc2 $vf12, 0($12)\n\t"
                    "lqc2 $vf13, 0($14)\n\t"
                    "lqc2 $vf14, 0($15)\n\t"
                    "lqc2 $vf15, 0($16)\n\t"
                    "lqc2 $vf16, 0($17)\n\t"
                    "vmul.xyzw $vf17, $vf13, $vf12\n\t"
                    "vadd.xyzw $vf8, $vf8, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf14, $vf12\n\t"
                    "vadd.xyzw $vf9, $vf9, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf15, $vf12\n\t"
                    "vadd.xyzw $vf10, $vf10, $vf17\n\t"
                    "vmul.xyzw $vf17, $vf16, $vf12\n\t"
                    "vadd.xyzw $vf11, $vf11, $vf17\n\t"
                    "addiu $12, $12, 16\n\t"
                    "addiu $14, $14, 16\n\t"
                    "addiu $15, $15, 16\n\t"
                    "addiu $16, $16, 16\n\t"
                    "addiu $17, $17, 16\n\t"
                    "addiu $13, $13, -1\n\t"
                    "bgtz $13, 2b\n\t"
                    "nop\n\t"
                    "addu $10, $10, %[kstep]\n\t"
                    "addiu $11, $11, -1\n\t"
                    "bgtz $11, 1b\n\t"
                    "nop\n\t"
                    "vaddy.x $vf8, $vf8, $vf8\n\t"
                    "vaddz.x $vf8, $vf8, $vf8\n\t"
                    "vaddw.x $vf8, $vf8, $vf8\n\t"
                    "vaddy.x $vf9, $vf9, $vf9\n\t"
                    "vaddz.x $vf9, $vf9, $vf9\n\t"
                    "vaddw.x $vf9, $vf9, $vf9\n\t"
                    "vaddy.x $vf10, $vf10, $vf10\n\t"
                    "vaddz.x $vf10, $vf10, $vf10\n\t"
                    "vaddw.x $vf10, $vf10, $vf10\n\t"
                    "vaddy.x $vf11, $vf11, $vf11\n\t"
                    "vaddz.x $vf11, $vf11, $vf11\n\t"
                    "vaddw.x $vf11, $vf11, $vf11\n\t"
                    "nop\n\t"
                    "nop\n\t"
                    "nop\n\t"
                    "nop\n\t"
                    "sqc2 $vf8, 0(%[ov])\n\t"
                    "sqc2 $vf9, 16(%[ov])\n\t"
                    "sqc2 $vf10, 32(%[ov])\n\t"
                    "sqc2 $vf11, 48(%[ov])\n\t"
                    :
                    : [xb] "r"(xbase), [sofs] "r"(space_ofs_bytes), [wb] "r"(wb), [kmax] "r"(maxk), [nq] "r"(inch4 >> 2), [wstep] "r"(wch * 4), [kstep] "r"(inch4 * 4), [ov] "r"(accv)
                    : "$2", "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15", "$16", "$17", "memory");

                for (int c = 0; c < n; c++)
                    orow[c][j] = activation_ss(accv[c * 4] + bv[c], activation_type, activation_params);
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Weight-stationary COP2 kernel (stride_w == 1, any dilation / stride_h).
//
// The output-stationary kernel above reloads the 4 weight rows for every
// output pixel and needs a horizontal lane reduction per pixel.  This kernel
// instead fixes (output-channel group, tap, input channel) and sweeps output
// pixels: the weight is a scalar that is broadcast once and multiplied into a
// vector holding 4 neighbouring output pixels.
//
// Layout: the 4 VU lanes of one accumulator hold 4 output pixels, and the
// four accumulators $vf4..$vf7 / $vf8..$vf11 hold 4 output channels, so one
// inner step covers 4 pixels x 4 channels = 16 MACs with a single x load, a
// single weight load and no horizontal reduction (the lanes are already
// different pixels).
//
// Input is the channel-major bordered blob (plane q at bottom + q*cstep,
// cstep a multiple of 4 floats so every plane base stays quadword aligned).
// For output row i the pixel-group grid starts at j0 = -(R & 3) with
// R = i*stride_h*bw, which makes R + j0 a multiple of 4, so for every tap k
//
//   phase       m_k = space_ofs[k] & 3            (row/group independent)
//   window base a_k = floor4(R + j0 + space_ofs[k]) (16-byte aligned)
//
// With m_k != 0 the 4 wanted pixels straddle an aligned quadword; they are
// rebuilt from two aligned lqc2 loads plus lane rotations (vmr32), selected
// with write-masked vmul against an all-ones vector - the same trick the
// depthwise 3x3s1 fast path uses.
//
// Two output groups (8 pixels) are processed per tap so the weight load is
// amortized; the second group's "P" quadword equals the first group's "Q"
// quadword, so only one extra x load is needed per tap.
//
// All lqc2 loads are issued at the top of each inner step so their latency
// overlaps the rotation / MAC work.  $vf23 holds the all-ones vector, $vf22
// the weight quadword, $vf21 the broadcast-multiply temporary.
//
// Overread note: garbage lanes of the first/last group of a row may read up
// to 3 floats past the end of a channel plane (within the following plane,
// or past the blob for the last plane) - the same exposure the depthwise
// kernel has.  Those values only ever reach lanes whose outputs are
// discarded, so they cannot corrupt results.
// ---------------------------------------------------------------------------

// MULbc broadcasts the bc field of its third operand (ft), so the weight
// quadword goes last: fd = input * weight[bc].
#define WST_MAC4(X, T, A0, A1, A2, A3)                                                        \
    "vmulx.xyzw " T ", " X ", $vf22\n\t" "vadd.xyzw " A0 ", " A0 ", " T "\n\t"                \
    "vmuly.xyzw " T ", " X ", $vf22\n\t" "vadd.xyzw " A1 ", " A1 ", " T "\n\t"                \
    "vmulz.xyzw " T ", " X ", $vf22\n\t" "vadd.xyzw " A2 ", " A2 ", " T "\n\t"                \
    "vmulw.xyzw " T ", " X ", $vf22\n\t" "vadd.xyzw " A3 ", " A3 ", " T "\n\t"

#define WST_MAC4A(X) WST_MAC4(X, "$vf21", "$vf4", "$vf5", "$vf6", "$vf7")
#define WST_MAC4B(X) WST_MAC4(X, "$vf21", "$vf8", "$vf9", "$vf10", "$vf11")

#define WST_HEAD "beq $11, $0, 2f\n\t" "1:\n\t"
#define WST_TAIL "addu $8, $8, $14\n\t" "addu $10, $10, $13\n\t" "addiu $11, $11, -1\n\t" "bne $11, $0, 1b\n\t" "2:\n\t"

// rotation chains: L^m(P) lands in $vf14/15/16, L^m(Q) in $vf17/18/19
#define WST_CHAIN1 "vmr32 $vf14, $vf12\n\t" "vmr32 $vf17, $vf13\n\t"
#define WST_CHAIN2 WST_CHAIN1 "vmr32 $vf15, $vf14\n\t" "vmr32 $vf18, $vf17\n\t"
#define WST_CHAIN3 WST_CHAIN2 "vmr32 $vf16, $vf15\n\t" "vmr32 $vf19, $vf18\n\t"
// second group's Q chain reuses the freed P temp registers
#define WST_CHAINB1 "vmr32 $vf14, $vf25\n\t"
#define WST_CHAINB2 WST_CHAINB1 "vmr32 $vf15, $vf14\n\t"
#define WST_CHAINB3 WST_CHAINB2 "vmr32 $vf16, $vf15\n\t"

// X = L^m(P) on the low lanes, L^m(Q) on the high lanes (write-masked vmul
// against the all-ones vector $vf23).  m=1: lo xyz / hi w; m=2: xy / zw;
// m=3: x / yzw.
#define WST_COMB_A1 "vmul.xyz $vf20, $vf14, $vf23\n\t" "vmul.w $vf20, $vf17, $vf23\n\t"
#define WST_COMB_A2 "vmul.xy $vf20, $vf15, $vf23\n\t" "vmul.zw $vf20, $vf18, $vf23\n\t"
#define WST_COMB_A3 "vmul.x $vf20, $vf16, $vf23\n\t" "vmul.yzw $vf20, $vf19, $vf23\n\t"
// second group: its P quadword is the first group's Q (stages $vf17/18/19),
// its Q quadword was rotated into $vf14/15/16
#define WST_COMB_B1 "vmul.xyz $vf13, $vf17, $vf23\n\t" "vmul.w $vf13, $vf14, $vf23\n\t"
#define WST_COMB_B2 "vmul.xy $vf13, $vf18, $vf23\n\t" "vmul.zw $vf13, $vf15, $vf23\n\t"
#define WST_COMB_B3 "vmul.x $vf13, $vf19, $vf23\n\t" "vmul.yzw $vf13, $vf16, $vf23\n\t"

#define WST_TAP_FN(NAME, G2, M)                                                                \
    static inline void NAME(const float* xp_, const float* wp_, int nq_, int csb_, int wqb_)    \
    {                                                                                          \
        register const float* xp __asm__("$8") = xp_;                                          \
        register const float* wp __asm__("$10") = wp_;                                         \
        register int nq __asm__("$11") = nq_;                                                  \
        register int csb __asm__("$14") = csb_;                                                \
        register int wq __asm__("$13") = wqb_;                                                 \
        __asm__ __volatile__(                                                                  \
            WST_HEAD                                                                           \
            "lqc2 $vf12, 0($8)\n\t"                                                            \
            WST_LD_##G2##_##M                                                                  \
            "lqc2 $vf22, 0($10)\n\t"                                                           \
            WST_MID_##G2##_##M                                                                 \
            WST_TAIL                                                                           \
            : "+r"(xp), "+r"(wp), "+r"(nq)                                                     \
            : "r"(csb), "r"(wq)                                                                \
            : "memory");                                                                       \
    }

// x quadword loads: group A always at ($8); group B's P is A's Q (already
// loaded) for m!=0, so only its Q at ($8)+32 needs a load; for m==0 group B's
// P is at ($8)+16.
#define WST_LD_G2_0 "lqc2 $vf24, 16($8)\n\t"
#define WST_LD_G2_1 "lqc2 $vf13, 16($8)\n\t" "lqc2 $vf25, 32($8)\n\t"
#define WST_LD_G2_2 "lqc2 $vf13, 16($8)\n\t" "lqc2 $vf25, 32($8)\n\t"
#define WST_LD_G2_3 "lqc2 $vf13, 16($8)\n\t" "lqc2 $vf25, 32($8)\n\t"
#define WST_LD_G1_0
#define WST_LD_G1_1 "lqc2 $vf13, 16($8)\n\t"
#define WST_LD_G1_2 "lqc2 $vf13, 16($8)\n\t"
#define WST_LD_G1_3 "lqc2 $vf13, 16($8)\n\t"

#define WST_MID_G1_0 WST_MAC4A("$vf12")
#define WST_MID_G1_1 WST_CHAIN1 WST_COMB_A1 WST_MAC4A("$vf20")
#define WST_MID_G1_2 WST_CHAIN2 WST_COMB_A2 WST_MAC4A("$vf20")
#define WST_MID_G1_3 WST_CHAIN3 WST_COMB_A3 WST_MAC4A("$vf20")
#define WST_MID_G2_0 WST_MAC4A("$vf12") WST_MAC4B("$vf24")
#define WST_MID_G2_1 WST_CHAIN1 WST_COMB_A1 WST_MAC4A("$vf20") WST_CHAINB1 WST_COMB_B1 WST_MAC4B("$vf13")
#define WST_MID_G2_2 WST_CHAIN2 WST_COMB_A2 WST_MAC4A("$vf20") WST_CHAINB2 WST_COMB_B2 WST_MAC4B("$vf13")
#define WST_MID_G2_3 WST_CHAIN3 WST_COMB_A3 WST_MAC4A("$vf20") WST_CHAINB3 WST_COMB_B3 WST_MAC4B("$vf13")

WST_TAP_FN(wst_tap_g2_m0, G2, 0)
WST_TAP_FN(wst_tap_g2_m1, G2, 1)
WST_TAP_FN(wst_tap_g2_m2, G2, 2)
WST_TAP_FN(wst_tap_g2_m3, G2, 3)
WST_TAP_FN(wst_tap_g1_m0, G1, 0)
WST_TAP_FN(wst_tap_g1_m1, G1, 1)
WST_TAP_FN(wst_tap_g1_m2, G1, 2)
WST_TAP_FN(wst_tap_g1_m3, G1, 3)

// accumulator init: $vf23 = ones, $vf4..$vf7 (+ $vf8..$vf11 for two groups)
// = bias vectors (or zero).  pbv holds 4 bias vectors then the ones vector.
static inline void wst_pair_init(const float* pbv, bool bias_term, bool g2)
{
    if (bias_term)
    {
        if (g2)
        {
            __asm__ __volatile__(
                "lqc2 $vf23, 64(%0)\n\t"
                "lqc2 $vf4, 0(%0)\n\t"
                "lqc2 $vf5, 16(%0)\n\t"
                "lqc2 $vf6, 32(%0)\n\t"
                "lqc2 $vf7, 48(%0)\n\t"
                "lqc2 $vf8, 0(%0)\n\t"
                "lqc2 $vf9, 16(%0)\n\t"
                "lqc2 $vf10, 32(%0)\n\t"
                "lqc2 $vf11, 48(%0)\n\t"
                :
                : "r"(pbv)
                : "memory");
        }
        else
        {
            __asm__ __volatile__(
                "lqc2 $vf23, 64(%0)\n\t"
                "lqc2 $vf4, 0(%0)\n\t"
                "lqc2 $vf5, 16(%0)\n\t"
                "lqc2 $vf6, 32(%0)\n\t"
                "lqc2 $vf7, 48(%0)\n\t"
                :
                : "r"(pbv)
                : "memory");
        }
    }
    else
    {
        if (g2)
        {
            __asm__ __volatile__(
                "lqc2 $vf23, 64(%0)\n\t"
                "vsub.xyzw $vf4, $vf0, $vf0\n\t"
                "vsub.xyzw $vf5, $vf0, $vf0\n\t"
                "vsub.xyzw $vf6, $vf0, $vf0\n\t"
                "vsub.xyzw $vf7, $vf0, $vf0\n\t"
                "vsub.xyzw $vf8, $vf0, $vf0\n\t"
                "vsub.xyzw $vf9, $vf0, $vf0\n\t"
                "vsub.xyzw $vf10, $vf0, $vf0\n\t"
                "vsub.xyzw $vf11, $vf0, $vf0\n\t"
                :
                : "r"(pbv)
                : "memory");
        }
        else
        {
            __asm__ __volatile__(
                "lqc2 $vf23, 64(%0)\n\t"
                "vsub.xyzw $vf4, $vf0, $vf0\n\t"
                "vsub.xyzw $vf5, $vf0, $vf0\n\t"
                "vsub.xyzw $vf6, $vf0, $vf0\n\t"
                "vsub.xyzw $vf7, $vf0, $vf0\n\t"
                :
                : "r"(pbv)
                : "memory");
        }
    }
}

// drain the four (or eight) accumulators to an aligned scratch buffer
static inline void wst_pair_store(float* sc, bool g2)
{
    if (g2)
    {
        __asm__ __volatile__(
            "sqc2 $vf4, 0(%0)\n\t"
            "sqc2 $vf5, 16(%0)\n\t"
            "sqc2 $vf6, 32(%0)\n\t"
            "sqc2 $vf7, 48(%0)\n\t"
            "sqc2 $vf8, 64(%0)\n\t"
            "sqc2 $vf9, 80(%0)\n\t"
            "sqc2 $vf10, 96(%0)\n\t"
            "sqc2 $vf11, 112(%0)\n\t"
            :
            : "r"(sc)
            : "memory");
    }
    else
    {
        __asm__ __volatile__(
            "sqc2 $vf4, 0(%0)\n\t"
            "sqc2 $vf5, 16(%0)\n\t"
            "sqc2 $vf6, 32(%0)\n\t"
            "sqc2 $vf7, 48(%0)\n\t"
            :
            : "r"(sc)
            : "memory");
    }
}

static void convolution_ps2_wst(const float* bottom, int cstep,
                                float* top_data, int tstep,
                                const float* weight_wst, const float* biasptr,
                                int inch, int outch, int maxk,
                                int outw, int outh, int bw, int stride_h,
                                const int* space_ofs, bool use_spr,
                                int activation_type, const Mat& activation_params)
{
    float* const spr_base = (float*)0x70000000;
    const bool stage_ok = use_spr && (size_t)maxk * inch * 16 <= 16384u;

    const int outch4 = (outch + 3) / 4 * 4;
    const int cstepb = cstep * 4;
    const int wqb = outch4 * 4;
    const int nq = inch;

    float scr[32] __attribute__((aligned(16)));
    float pbv[20] __attribute__((aligned(16)));


    if (outw < 4 || outh <= 0)
    {
        // tiny rows: plain scalar loops (the group machinery needs outw >= 4)
        for (int p = 0; p < outch; p++)
        {
            float* outp = top_data + (size_t)p * tstep;
            const float b = biasptr ? biasptr[p] : 0.f;
            for (int i = 0; i < outh; i++)
            {
                const int R = i * stride_h * bw;
                for (int j = 0; j < outw; j++)
                {
                    float sum = b;
                    for (int k = 0; k < maxk; k++)
                    {
                        const float* xk = bottom + (size_t)space_ofs[k] + R + j;
                        const float* wk = weight_wst + ((size_t)k * inch) * outch4 + p;
                        for (int q = 0; q < inch; q++)
                            sum += wk[(size_t)q * outch4] * xk[(size_t)q * cstep];
                    }
                    outp[(size_t)i * outw + j] = activation_ss(sum, activation_type, activation_params);
                }
            }
        }
        return;
    }

    for (int p0 = 0; p0 < outch; p0 += 4)
    {
        const int n = outch - p0 < 4 ? outch - p0 : 4;

        for (int c = 0; c < 4; c++)
        {
            const float b = (biasptr && c < n) ? biasptr[p0 + c] : 0.f;
            pbv[c * 4 + 0] = b;
            pbv[c * 4 + 1] = b;
            pbv[c * 4 + 2] = b;
            pbv[c * 4 + 3] = b;
        }
        pbv[16] = pbv[17] = pbv[18] = pbv[19] = 1.f;

        float* outptr[4];
        for (int c = 0; c < n; c++)
            outptr[c] = top_data + (size_t)(p0 + c) * tstep;

        const bool staged = stage_ok && n == 4;
        const int wstride = staged ? 16 : wqb;
        if (staged)
        {
            for (int k = 0; k < maxk; k++)
                for (int q = 0; q < inch; q++)
                    memcpy(spr_base + ((size_t)k * inch + q) * 4,
                           weight_wst + ((size_t)k * inch + q) * outch4 + p0, 16);
        }
        const float* wbase = staged ? spr_base : weight_wst + p0;


        for (int i = 0; i < outh; i++)
        {
            const int R = i * stride_h * bw;
            const int rowfloor = R - (R & 3);
            const int jstart = -(R & 3);
            const float* xrow = bottom + rowfloor;

            int sofsf4[9];
            int mk[9];
            for (int k = 0; k < maxk; k++)
            {
                mk[k] = space_ofs[k] & 3;
                sofsf4[k] = space_ofs[k] - mk[k];
            }

            int j0 = jstart;
            while (j0 < outw)
            {
                const bool hasB = j0 + 4 < outw;

                wst_pair_init(pbv, biasptr != 0, hasB);

                for (int k = 0; k < maxk; k++)
                {
                    const float* xp = xrow + (j0 - jstart) + sofsf4[k];
                    const float* wp = wbase + (size_t)k * inch * (wstride / 4);
                    switch (mk[k])
                    {
                    case 0:
                        if (hasB) wst_tap_g2_m0(xp, wp, nq, cstepb, wstride);
                        else wst_tap_g1_m0(xp, wp, nq, cstepb, wstride);
                        break;
                    case 1:
                        if (hasB) wst_tap_g2_m1(xp, wp, nq, cstepb, wstride);
                        else wst_tap_g1_m1(xp, wp, nq, cstepb, wstride);
                        break;
                    case 2:
                        if (hasB) wst_tap_g2_m2(xp, wp, nq, cstepb, wstride);
                        else wst_tap_g1_m2(xp, wp, nq, cstepb, wstride);
                        break;
                    default:
                        if (hasB) wst_tap_g2_m3(xp, wp, nq, cstepb, wstride);
                        else wst_tap_g1_m3(xp, wp, nq, cstepb, wstride);
                        break;
                    }
                }

                wst_pair_store(scr, hasB);

                for (int g = 0; g < (hasB ? 2 : 1); g++)
                {
                    const int jg = j0 + 4 * g;
                    const int lo = jg < 0 ? -jg : 0;
                    const int hi = jg + 4 > outw ? outw - jg : 4;
                    for (int c = 0; c < n; c++)
                    {
                        float* orow = outptr[c] + (size_t)i * outw + jg;
                        const float* s = scr + g * 16 + c * 4;
                        for (int t = lo; t < hi; t++)
                            orow[t] = activation_ss(s[t], activation_type, activation_params);
                    }
                }

                j0 += hasB ? 8 : 4;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase-aligned weight-stationary kernel.
//
// The kernel above rebuilds every misaligned tap window with a vmr32 chain
// and a masked vmul, and it walks input channels one at a time, so the weight
// loads jump by outch4 floats between channels.  Both dominate its cost.
//
// This kernel removes both:
//
//  * The caller hands in four copies of the bordered input.  Copy m is
//    quadword aligned and holds the blob shifted down by m floats, i.e.
//    copy[u] == blob[u + m].  The row loop starts at j0 = -(R & 3), so
//    R + j0 is a multiple of 4; tap k then reads copy (space_ofs[k] & 3) at
//    base = R + j0 + space_ofs[k] - (space_ofs[k] & 3), which is a multiple of
//    4 and 16-byte aligned, and a plain lqc2 yields exactly the four pixels
//    the window wants.  No rotation, no mask.  (Three copies would suffice
//    only when the row stride is a multiple of 4; it is not, in general.)
//
//  * Weights are stored as [k][q][p/4], with q padded up to a multiple of 4:
//    four consecutive input channels of one tap are four consecutive
//    quadwords.  The q loop steps by 4, so those loads are sequential cache
//    lines, and each loaded weight is broadcast onto four output pixels.
//
// One inner step is 4 input channels x 4 output pixels x 4 output channels =
// 64 MACs, against the old kernel's 4 x 1 x 4 = 16, with a quarter of the
// weight loads and none of the alignment fixup.
//
// $vf1..$vf3 hold the four-pixel windows of channels q..q+2 (q+3 is loaded
// straight into the MAC), $vf4..$vf7 the four output-channel accumulators,
// $vf20 the broadcast-multiply temporary, $vf21..$vf24 the four weight lanes.
// ---------------------------------------------------------------------------

static inline void wst_pa_init(const float* pbv, bool bias_term)
{
    if (bias_term)
    {
        __asm__ __volatile__(
            "lqc2 $vf4, 0(%0)\n\t"
            "lqc2 $vf5, 16(%0)\n\t"
            "lqc2 $vf6, 32(%0)\n\t"
            "lqc2 $vf7, 48(%0)\n\t"
            :
            : "r"(pbv)
            : "memory");
    }
    else
    {
        __asm__ __volatile__(
            "vsub.xyzw $vf4, $vf0, $vf0\n\t"
            "vsub.xyzw $vf5, $vf0, $vf0\n\t"
            "vsub.xyzw $vf6, $vf0, $vf0\n\t"
            "vsub.xyzw $vf7, $vf0, $vf0\n\t"
            :
            :
            : "memory");
    }
}

static inline void wst_pa_store(float* sc)
{
    __asm__ __volatile__(
        "sqc2 $vf4, 0(%0)\n\t"
        "sqc2 $vf5, 16(%0)\n\t"
        "sqc2 $vf6, 32(%0)\n\t"
        "sqc2 $vf7, 48(%0)\n\t"
        :
        : "r"(sc)
        : "memory");
}

// one tap, q in groups of 4.  xp points at the phase-aligned window of
// channel q=0; wp at the first weight quadword of this tap.  wq is the byte
// distance between consecutive channels' weight quadwords.
static inline void wst_pa_tap4(const float* xp_, const float* wp_, int nq4_, int csb_, int wq_)
{
    register const float* xp __asm__("$8") = xp_;
    register const float* wp __asm__("$10") = wp_;
    register int nq4 __asm__("$11") = nq4_;
    register int csb __asm__("$14") = csb_;
    register int wq __asm__("$13") = wq_;

    __asm__ __volatile__(
        "beq $11, $0, 2f\n\t"
        "1:\n\t"
        "lqc2 $vf1, 0($8)\n\t"
        "addu $9, $8, $14\n\t"
        "lqc2 $vf2, 0($9)\n\t"
        "addu $9, $9, $14\n\t"
        "lqc2 $vf3, 0($9)\n\t"
        "addu $9, $9, $14\n\t"
        "lqc2 $vf21, 0($10)\n\t"
        "addu $12, $10, $13\n\t"
        "lqc2 $vf22, 0($12)\n\t"
        "addu $12, $12, $13\n\t"
        "lqc2 $vf23, 0($12)\n\t"
        "addu $12, $12, $13\n\t"
        "lqc2 $vf24, 0($12)\n\t"
        // MULbc broadcasts the bc field of its third operand, so the weight
        // quadword goes last: each output channel scales the four pixels
        "vmulx.xyzw $vf20, $vf1, $vf21\n\t"
        "vadd.xyzw $vf4, $vf4, $vf20\n\t"
        "vmuly.xyzw $vf20, $vf1, $vf21\n\t"
        "vadd.xyzw $vf5, $vf5, $vf20\n\t"
        "vmulz.xyzw $vf20, $vf1, $vf21\n\t"
        "vadd.xyzw $vf6, $vf6, $vf20\n\t"
        "vmulw.xyzw $vf20, $vf1, $vf21\n\t"
        "vadd.xyzw $vf7, $vf7, $vf20\n\t"
        "lqc2 $vf1, 0($9)\n\t"
        "vmulx.xyzw $vf20, $vf2, $vf22\n\t"
        "vadd.xyzw $vf4, $vf4, $vf20\n\t"
        "vmuly.xyzw $vf20, $vf2, $vf22\n\t"
        "vadd.xyzw $vf5, $vf5, $vf20\n\t"
        "vmulz.xyzw $vf20, $vf2, $vf22\n\t"
        "vadd.xyzw $vf6, $vf6, $vf20\n\t"
        "vmulw.xyzw $vf20, $vf2, $vf22\n\t"
        "vadd.xyzw $vf7, $vf7, $vf20\n\t"
        "vmulx.xyzw $vf20, $vf3, $vf23\n\t"
        "vadd.xyzw $vf4, $vf4, $vf20\n\t"
        "vmuly.xyzw $vf20, $vf3, $vf23\n\t"
        "vadd.xyzw $vf5, $vf5, $vf20\n\t"
        "vmulz.xyzw $vf20, $vf3, $vf23\n\t"
        "vadd.xyzw $vf6, $vf6, $vf20\n\t"
        "vmulw.xyzw $vf20, $vf3, $vf23\n\t"
        "vadd.xyzw $vf7, $vf7, $vf20\n\t"
        "vmulx.xyzw $vf20, $vf1, $vf24\n\t"
        "vadd.xyzw $vf4, $vf4, $vf20\n\t"
        "vmuly.xyzw $vf20, $vf1, $vf24\n\t"
        "vadd.xyzw $vf5, $vf5, $vf20\n\t"
        "vmulz.xyzw $vf20, $vf1, $vf24\n\t"
        "vadd.xyzw $vf6, $vf6, $vf20\n\t"
        "vmulw.xyzw $vf20, $vf1, $vf24\n\t"
        "vadd.xyzw $vf7, $vf7, $vf20\n\t"
        "addu $8, $9, $14\n\t"
        "addu $12, $12, $13\n\t"
        "addu $10, $12, $0\n\t"
        "addiu $11, $11, -1\n\t"
        "bne $11, $0, 1b\n\t"
        "2:\n\t"
        : "+r"(xp), "+r"(wp), "+r"(nq4)
        : "r"(csb), "r"(wq)
        : "$9", "$12", "memory");
}

// phase[m] (m = 0..3) is the bordered input shifted down by m floats:
// phase[m][u] == blob[u + m].  Every copy is quadword aligned and every
// channel plane is cstep floats, cstep a multiple of 4, so phase[m] + q*cstep
// stays aligned for every channel.  weight_pa is [k][q][p/4].
static void convolution_ps2_wst_pa(const float* const phase[4], int cstep,
                                   float* top_data, int tstep,
                                   const float* weight_pa, const float* biasptr,
                                   int inch, int outch, int maxk,
                                   int outw, int outh, int bw, int stride_h,
                                   const int* space_ofs,
                                   int activation_type, const Mat& activation_params)
{
    const int cstepb = cstep * 4;
    const int inch4 = (inch + 3) & ~3;
    const int nq4 = inch4 >> 2;
    const int outch4 = (outch + 3) & ~3;
    const int wqb = outch4 * 4; // bytes between consecutive channels' weights

    float scr[16] __attribute__((aligned(16)));
    float pbv[16] __attribute__((aligned(16)));

    for (int p0 = 0; p0 < outch; p0 += 4)
    {
        const int n = outch - p0 < 4 ? outch - p0 : 4;

        for (int c = 0; c < 4; c++)
        {
            const float b = (biasptr && c < n) ? biasptr[p0 + c] : 0.f;
            pbv[c * 4 + 0] = b;
            pbv[c * 4 + 1] = b;
            pbv[c * 4 + 2] = b;
            pbv[c * 4 + 3] = b;
        }

        float* outptr[4];
        for (int c = 0; c < n; c++)
            outptr[c] = top_data + (size_t)(p0 + c) * tstep;

        const float* wgrp = weight_pa + p0;

        for (int i = 0; i < outh; i++)
        {
            const int R = i * stride_h * bw;
            // the first group starts on the quadword that contains pixel 0, so
            // R + j0 is a multiple of 4 for every group and every tap window
            // below lands on an aligned address
            const int jstart = -(R & 3);

            for (int j0 = jstart; j0 < outw; j0 += 4)
            {
                wst_pa_init(pbv, biasptr != 0);

                for (int k = 0; k < maxk; k++)
                {
                    const int m = space_ofs[k] & 3;
                    const int base = R + j0 + space_ofs[k] - m;
                    const float* xp = phase[m] + base;
                    const float* wp = wgrp + (size_t)k * inch4 * outch4;

                    wst_pa_tap4(xp, wp, nq4, cstepb, wqb);
                }

                wst_pa_store(scr);

                const int lo = j0 < 0 ? -j0 : 0;
                const int hi = j0 + 4 > outw ? outw - j0 : 4;
                for (int c = 0; c < n; c++)
                {
                    float* orow = outptr[c] + (size_t)i * outw + j0;
                    const float* s = scr + c * 4;
                    for (int t = lo; t < hi; t++)
                        orow[t] = activation_ss(s[t], activation_type, activation_params);
                }
            }
        }
    }
}

} // namespace ncnn

#endif // LAYER_CONVOLUTION_MIPS_PS2_KERNEL_H
