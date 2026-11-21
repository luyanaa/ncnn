// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_PSP_VFPU_H
#define NCNN_PSP_VFPU_H

// Thin wrappers over the Allegrex VFPU (COP2). The register file is 8 matrices
// of 4x4 32-bit elements; one instruction addresses at most one column, so the
// widest load is 16 bytes and lv.q faults on anything less aligned.
//
// The wrappers use the column registers of matrix 0 (C000, C010, C020) and, for
// the three-operand forms, C030. Those are all below register 64, which is the
// range the vhtfm2/vhtfm3 write bug does not touch, and none of the arithmetic
// used here decomposes into a dot product, so destination/source aliasing is
// legal. Distinct registers are used anyway.
//
// vdiv is deliberately absent: it blocks the pipe for 14 cycles per element.

#include <stdint.h>

static inline bool psp_vfpu_aligned(const void* ptr)
{
    return ((uintptr_t)ptr & 15) == 0;
}

// out = op(a, b), one quad. op is a .q instruction such as "vadd.q".
#define PSP_VFPU_BINARY(op, a, b, out)                    \
    __asm__ __volatile__(                                 \
        "lv.q C000, 0(%0)\n\t"                            \
        "lv.q C010, 0(%1)\n\t" op " C020, C000, C010\n\t" \
        "sv.q C020, 0(%2)\n\t"                            \
        :                                                 \
        : "r"(a), "r"(b), "r"(out)                        \
        : "memory")

// out = max(a, 0) + min(a, 0) * slope. Shared by leaky relu and prelu.
#define PSP_VFPU_LEAKY(a, zero, slope, out)       \
    __asm__ __volatile__(                         \
        "lv.q C000, 0(%0)\n\t"                    \
        "lv.q C010, 0(%1)\n\t"                    \
        "lv.q C100, 0(%2)\n\t"                    \
        "vmax.q C020, C000, C010\n\t"             \
        "vmin.q C030, C000, C010\n\t"             \
        "vmul.q C030, C030, C100\n\t"             \
        "vadd.q C020, C020, C030\n\t"             \
        "sv.q C020, 0(%3)\n\t"                    \
        :                                         \
        : "r"(a), "r"(zero), "r"(slope), "r"(out) \
        : "memory")

// Hard sigmoid / hard swish building block: clamp(a * alpha + beta, 0, 1).
// alpha, beta, zero and one are pointers to aligned quads.
#define PSP_VFPU_HARDSIGMOID(a, alpha, beta, zero, one, out)           \
    __asm__ __volatile__(                                              \
        "lv.q C000, 0(%0)\n\t"                                         \
        "lv.q C010, 0(%1)\n\t"                                         \
        "lv.q C020, 0(%2)\n\t"                                         \
        "lv.q C030, 0(%3)\n\t"                                         \
        "lv.q C100, 0(%4)\n\t"                                         \
        "vmul.q C000, C000, C010\n\t"                                  \
        "vadd.q C000, C000, C020\n\t"                                  \
        "vmax.q C000, C000, C030\n\t"                                  \
        "vmin.q C000, C000, C100\n\t"                                  \
        "sv.q C000, 0(%5)\n\t"                                         \
        :                                                              \
        : "r"(a), "r"(alpha), "r"(beta), "r"(zero), "r"(one), "r"(out) \
        : "memory")

// As above, then multiplied by the original value. Hard swish.
#define PSP_VFPU_HARDSWISH(a, alpha, beta, zero, one, out)             \
    __asm__ __volatile__(                                              \
        "lv.q C000, 0(%0)\n\t"                                         \
        "lv.q C010, 0(%1)\n\t"                                         \
        "lv.q C020, 0(%2)\n\t"                                         \
        "lv.q C030, 0(%3)\n\t"                                         \
        "lv.q C100, 0(%4)\n\t"                                         \
        "vmul.q C110, C000, C010\n\t"                                  \
        "vadd.q C110, C110, C020\n\t"                                  \
        "vmax.q C110, C110, C030\n\t"                                  \
        "vmin.q C110, C110, C100\n\t"                                  \
        "vmul.q C110, C110, C000\n\t"                                  \
        "sv.q C110, 0(%5)\n\t"                                         \
        :                                                              \
        : "r"(a), "r"(alpha), "r"(beta), "r"(zero), "r"(one), "r"(out) \
        : "memory")

// out = |a|. vabs is a bit-manipulation op, latency 3.
#define PSP_VFPU_ABS(a, out)    \
    __asm__ __volatile__(       \
        "lv.q C000, 0(%0)\n\t"  \
        "vabs.q C010, C000\n\t" \
        "sv.q C010, 0(%1)\n\t"  \
        :                       \
        : "r"(a), "r"(out)      \
        : "memory")

// out = -a.
#define PSP_VFPU_NEG(a, out)    \
    __asm__ __volatile__(       \
        "lv.q C000, 0(%0)\n\t"  \
        "vneg.q C010, C000\n\t" \
        "sv.q C010, 0(%1)\n\t"  \
        :                       \
        : "r"(a), "r"(out)      \
        : "memory")

// Dot product of n floats, accumulated into *acc. The K loop of every fp32
// GEMM, inner product and convolution reduces to this: four-wide vdot.q into a
// column that is never a source, then the tail in scalar. vdot writes only the
// first lane of its destination, so the four partials stay in S020..S023 while
// the inputs stay in C000/C010, which do not overlap them.
//
// GCC treats the VFPU register file as clobbered between asm statements, so the
// running total cannot live in a register across the loop. Summing four dots
// before the one store keeps the spill to a quarter of the quads; one store
// per quad costs more than the dot saves.
//
// Both pointers must be 16-byte aligned for the vector part; lv.q faults on
// anything less. The caller checks, and an unaligned pair falls through to the
// scalar tail, which is the same arithmetic.
#define PSP_VFPU_DOT(a, b, n, acc)                                           \
    do                                                                       \
    {                                                                        \
        const float* psp_dot_a_ = (a);                                       \
        const float* psp_dot_b_ = (b);                                       \
        int psp_dot_n_ = (n);                                                \
        float psp_dot_sum_ = 0.f;                                            \
        if (psp_vfpu_aligned(psp_dot_a_) && psp_vfpu_aligned(psp_dot_b_))    \
        {                                                                    \
            float psp_dot_part_[4] __attribute__((aligned(16)));             \
            while (psp_dot_n_ >= 16)                                         \
            {                                                                \
                __asm__ __volatile__(                                        \
                    "lv.q C000, 0(%1)\n\t"                                   \
                    "lv.q C010, 0(%2)\n\t"                                   \
                    "vdot.q S020, C000, C010\n\t"                            \
                    "lv.q C000, 16(%1)\n\t"                                  \
                    "lv.q C010, 16(%2)\n\t"                                  \
                    "vdot.q S021, C000, C010\n\t"                            \
                    "lv.q C000, 32(%1)\n\t"                                  \
                    "lv.q C010, 32(%2)\n\t"                                  \
                    "vdot.q S022, C000, C010\n\t"                            \
                    "lv.q C000, 48(%1)\n\t"                                  \
                    "lv.q C010, 48(%2)\n\t"                                  \
                    "vdot.q S023, C000, C010\n\t"                            \
                    "sv.q C020, 0(%0)\n\t"                                   \
                    :                                                        \
                    : "r"(psp_dot_part_), "r"(psp_dot_a_), "r"(psp_dot_b_)   \
                    : "memory");                                             \
                psp_dot_sum_ += psp_dot_part_[0] + psp_dot_part_[1]          \
                                + psp_dot_part_[2] + psp_dot_part_[3];       \
                psp_dot_a_ += 16;                                            \
                psp_dot_b_ += 16;                                            \
                psp_dot_n_ -= 16;                                            \
            }                                                                \
            while (psp_dot_n_ >= 4)                                          \
            {                                                                \
                __asm__ __volatile__(                                        \
                    "lv.q C000, 0(%1)\n\t"                                   \
                    "lv.q C010, 0(%2)\n\t"                                   \
                    "vdot.q S020, C000, C010\n\t"                            \
                    "sv.s S020, 0(%0)\n\t"                                   \
                    :                                                        \
                    : "r"(psp_dot_part_), "r"(psp_dot_a_), "r"(psp_dot_b_)   \
                    : "memory");                                             \
                psp_dot_sum_ += psp_dot_part_[0];                            \
                psp_dot_a_ += 4;                                             \
                psp_dot_b_ += 4;                                             \
                psp_dot_n_ -= 4;                                             \
            }                                                                \
        }                                                                    \
        for (int psp_dot_i_ = 0; psp_dot_i_ < psp_dot_n_; psp_dot_i_++)      \
            psp_dot_sum_ += psp_dot_a_[psp_dot_i_] * psp_dot_b_[psp_dot_i_]; \
        *(acc) += psp_dot_sum_;                                              \
    } while (0)

// Transcendental unary ops. vexp2 and vlog2 are approximations (a few ulp),
// and vrcp is not used: reciprocal is a single scalar division on the FPU,
// which is exact where vrcp is not. The caller passes the mnemonic.
#define PSP_VFPU_UNARY(op, a, out)                  \
    __asm__ __volatile__(                           \
        "lv.q C000, 0(%0)\n\t" op " C010, C000\n\t" \
        "sv.q C010, 0(%1)\n\t"                      \
        :                                           \
        : "r"(a), "r"(out)                          \
        : "memory")

#endif // NCNN_PSP_VFPU_H
