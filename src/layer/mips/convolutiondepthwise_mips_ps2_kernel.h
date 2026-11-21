// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTIONDEPTHWISE_PS2_KERNEL_H
#define LAYER_CONVOLUTIONDEPTHWISE_PS2_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// PlayStation 2 EE (R5900) COP2 / VU0-macro-mode depth-wise convolution kernel.
//
// The kernel works on an already padded (bordered) single-channel-per-output
// float blob.  ncnn Mat is channel-major, so for stride_w == 1 the four
// neighbouring pixels needed by four neighbouring output pixels are contiguous
// in memory.  We therefore process four output pixels at a time with the four
// lanes of a VU vector.
//
// LQC2/SQC2 require 16-byte aligned addresses on real hardware (and on PCSX2
// the low address bits are simply ignored), so the straightforward
// "load four pixels at an arbitrary offset" scheme is not usable.  Instead the
// taps are classified by their alignment phase:
//
//   c = (-(row_base + space_ofs[k])) & 3
//
// and the partial sums of each class are accumulated into a separate array:
//
//   acc[c][4*t + lane]  contributes to output pixel (4*t + c + lane)
//
// Every LQC2 in the hot loop is then naturally aligned, and the different
// classes are merged with a cheap scalar pass at the end (they are only
// shifted copies of each other).  Only stride_w == 1 uses the vector path;
// other strides fall back to a simple scalar loop.

#if defined(_EE) || defined(__PS2__)
#define CONVDW_PS2_SIMD 1
#else
#define CONVDW_PS2_SIMD 0
#endif

#define CONVDW_PS2_MAXK 64
#define CONVDW_PS2_CHUNK 64

// accumulator registers: $vf4..$vf7, tap temporaries: $vf10/$vf11/$vf12
#if CONVDW_PS2_SIMD
#define CONVDW_PS2_TAP_LOOP(CLASS, ACC)                                                                 \
    {                                                                                                   \
        const int c_ = CLASS;                                                                           \
        for (int ii_ = 0; ii_ < cnt[c_]; ii_++)                                                         \
        {                                                                                               \
            int k_ = lst[c_][ii_];                                                                      \
            const float* xp_ = inrow + j0 + c_ + space_ofs[k_] + 4 * t;                                 \
            const float* wp_ = wbase + k_ * 4;                                                          \
            __asm__ __volatile__(                                                                       \
                "lqc2 $vf10, 0(%0)\n\t"                                                                 \
                "lqc2 $vf11, 0(%1)\n\t"                                                                 \
                "vmul.xyzw $vf12, $vf11, $vf10\n\t"                                                     \
                "vadd.xyzw " ACC ", " ACC ", $vf12\n\t"                                                 \
                :                                                                                       \
                : "r"(xp_), "r"(wp_));                                                                  \
        }                                                                                               \
    }
#endif

#if CONVDW_PS2_SIMD

// ---------------------------------------------------------------------------
// Fast path for kernel 3x3, stride_w == 1, dilation_w == 1 (any stride_h /
// dilation_h).  Four consecutive output pixels per 128-bit accumulator.
//
// For a kernel row (3 contiguous taps) only two aligned quadwords are needed:
//   P = x[j0+O-m0 .. j0+O-m0+3],  Q = P + 4
// where O is the row offset and m0 = (rowbase + j0 + O) & 3.  Tap s (phase
// m0+s) contributes  D[t] = x[j0+O+s+t] = L^(m0+s)(P or Q)[t]  where
// L^m is the "lane rotate left by m" (VMR32), so each tap is
//
//   ACC  = L^m(P) * (w * Mlo_m)
//   out += ACC  + L^m(Q) * (w * Mhi_m)        (VMULA + VMADD, 2 ops / tap)
//
// with constant per-phase masks Mlo/Mhi and the weight pre-folded into the
// masks by plain C code once per row.  No scratch buffer, no per-row tap
// classification, no scalar merge: the accumulator is stored directly to the
// (aligned) output row.  The first group of a row starts at j0 chosen so the
// SQC2 store is 16-byte aligned, which also keeps every LQC2 aligned; the
// remaining leading/trailing outputs go through a scalar tail.
// ---------------------------------------------------------------------------

// L^m masks: Mlo_m[t] = 1 for t < 4-m (taken from P), Mhi_m[t] = 1 for t >= 4-m
static const float convdw_ps2_mlo_tab[4][4] = {
    {1.f, 1.f, 1.f, 1.f},
    {1.f, 1.f, 1.f, 0.f},
    {1.f, 1.f, 0.f, 0.f},
    {1.f, 0.f, 0.f, 0.f},
};
static const float convdw_ps2_mhi_tab[4][4] = {
    {0.f, 0.f, 0.f, 0.f},
    {0.f, 0.f, 0.f, 1.f},
    {0.f, 0.f, 1.f, 1.f},
    {0.f, 1.f, 1.f, 1.f},
};

// rotation registers: L1..L3(P) = $vf7..$vf9, L1..L3(Q) = $vf10..$vf12
//
// Accumulation scheme: $ACC holds the running sum for the 4 output pixels
// (started as the bias vector), each tap contributes two VMADDA ops
// (ACC += L^m(P)*w*Mlo_m, ACC += L^m(Q)*w*Mhi_m), and the finished group is
// moved to $vf4 with a single VMADD (vf4 = ACC + zeros) before the SQC2.
#define CONVDW_ROT_P3 "vmr32 $vf7, $vf5\n\t" "vmr32 $vf8, $vf7\n\t" "vmr32 $vf9, $vf8\n\t"
#define CONVDW_ROT_Q3 "vmr32 $vf10, $vf6\n\t" "vmr32 $vf11, $vf10\n\t" "vmr32 $vf12, $vf11\n\t"
#define CONVDW_ROT_P2 "vmr32 $vf7, $vf5\n\t" "vmr32 $vf8, $vf7\n\t"
#define CONVDW_ROT_Q2 "vmr32 $vf10, $vf6\n\t" "vmr32 $vf11, $vf10\n\t"

// one kernel row = 3 taps with consecutive phases; WkL/WkH are the
// pre-weighted mask registers (WkL = w*mask_lo, WkH = w*mask_hi)
#define CONVDW_RUN0(PTR, W0L, W0H, W1L, W1H, W2L, W2H)   /* phases 0,1,2 */ \
    "lqc2 $vf5, 0(" PTR ")\n\t" \
    "lqc2 $vf6, 16(" PTR ")\n\t" \
    CONVDW_ROT_P2 CONVDW_ROT_Q2 \
    "vmadda.xyzw $ACC, $vf5, " W0L "\n\t" "vmadda.xyzw $ACC, $vf6, " W0H "\n\t" \
    "vmadda.xyzw $ACC, $vf7, " W1L "\n\t" "vmadda.xyzw $ACC, $vf10, " W1H "\n\t" \
    "vmadda.xyzw $ACC, $vf8, " W2L "\n\t" "vmadda.xyzw $ACC, $vf11, " W2H "\n\t"

#define CONVDW_RUN1(PTR, W0L, W0H, W1L, W1H, W2L, W2H)   /* phases 1,2,3 */ \
    "lqc2 $vf5, 0(" PTR ")\n\t" \
    "lqc2 $vf6, 16(" PTR ")\n\t" \
    CONVDW_ROT_P3 CONVDW_ROT_Q3 \
    "vmadda.xyzw $ACC, $vf7, " W0L "\n\t" "vmadda.xyzw $ACC, $vf10, " W0H "\n\t" \
    "vmadda.xyzw $ACC, $vf8, " W1L "\n\t" "vmadda.xyzw $ACC, $vf11, " W1H "\n\t" \
    "vmadda.xyzw $ACC, $vf9, " W2L "\n\t" "vmadda.xyzw $ACC, $vf12, " W2H "\n\t"

#define CONVDW_RUN2(PTR, W0L, W0H, W1L, W1H, W2L, W2H)   /* phases 2,3,0 */ \
    "lqc2 $vf5, 0(" PTR ")\n\t" \
    "lqc2 $vf6, 16(" PTR ")\n\t" \
    CONVDW_ROT_P3 CONVDW_ROT_Q3 \
    "vmadda.xyzw $ACC, $vf8, " W0L "\n\t" "vmadda.xyzw $ACC, $vf11, " W0H "\n\t" \
    "vmadda.xyzw $ACC, $vf9, " W1L "\n\t" "vmadda.xyzw $ACC, $vf12, " W1H "\n\t" \
    "vmadda.xyzw $ACC, $vf6, " W2L "\n\t" "vmadda.xyzw $ACC, $vf5, " W2H "\n\t"

/* phases 3,0,1: the last tap wraps past the second quadword, so a third
   aligned quad (Q2) is loaded and the phase-1 tap reads L(Q)/L(Q2) */
#define CONVDW_RUN3(PTR, W0L, W0H, W1L, W1H, W2L, W2H)   /* phases 3,0,1 */ \
    "lqc2 $vf5, 0(" PTR ")\n\t" \
    "lqc2 $vf6, 16(" PTR ")\n\t" \
    "lqc2 $vf7, 32(" PTR ")\n\t" \
    "vmr32 $vf8, $vf5\n\t" \
    "vmr32 $vf9, $vf8\n\t" \
    "vmr32 $vf10, $vf9\n\t" \
    "vmr32 $vf11, $vf6\n\t" \
    "vmr32 $vf12, $vf11\n\t" \
    "vmr32 $vf1, $vf12\n\t" \
    "vmr32 $vf12, $vf7\n\t" \
    "vmadda.xyzw $ACC, $vf10, " W0L "\n\t" "vmadda.xyzw $ACC, $vf1, " W0H "\n\t" \
    "vmadda.xyzw $ACC, $vf6, " W1L "\n\t" "vmadda.xyzw $ACC, $vf5, " W1H "\n\t" \
    "vmadda.xyzw $ACC, $vf11, " W2L "\n\t" "vmadda.xyzw $ACC, $vf12, " W2H "\n\t"

// one output quadword: ACC = bias + all taps, then ACC -> $vf4, store
#define CONVDW_GROUP(R0, R1, R2) \
    "vmula.xyzw $ACC, $vf13, $vf2\n\t" \
    CONVDW_RUN##R0("%[pp0]", "$vf14", "$vf15", "$vf16", "$vf17", "$vf18", "$vf19") \
    CONVDW_RUN##R1("%[pp1]", "$vf20", "$vf21", "$vf22", "$vf23", "$vf24", "$vf25") \
    CONVDW_RUN##R2("%[pp2]", "$vf26", "$vf27", "$vf28", "$vf29", "$vf30", "$vf31") \
    "vmadd.xyzw $vf4, $vf3, $vf3\n\t" \
    "sqc2 $vf4, 0(%[dst])\n\t"

#define CONVDW_ROWFN(NAME, R0, R1, R2)                                                     \
    static inline void NAME(const float* pp0_, const float* pp1_, const float* pp2_,        \
                            float* dst_, int ngroups_, const float* mt_, const float* ht_,  \
                            const float* bt_)                                               \
    {                                                                                       \
        register const float* pp0 __asm__("$16") = pp0_;                                    \
        register const float* pp1 __asm__("$17") = pp1_;                                    \
        register const float* pp2 __asm__("$18") = pp2_;                                    \
        register float* dst __asm__("$19") = dst_;                                          \
        register int ng __asm__("$20") = ngroups_;                                          \
        __asm__ __volatile__(                                                               \
            "lqc2 $vf13, 0(%[bt])\n\t"                                                      \
            "lqc2 $vf2, 16(%[bt])\n\t"                                                      \
            "lqc2 $vf3, 32(%[bt])\n\t"                                                      \
            "lqc2 $vf14, 0(%[mt])\n\t"  "lqc2 $vf15, 0(%[ht])\n\t"                           \
            "lqc2 $vf16, 16(%[mt])\n\t" "lqc2 $vf17, 16(%[ht])\n\t"                          \
            "lqc2 $vf18, 32(%[mt])\n\t" "lqc2 $vf19, 32(%[ht])\n\t"                          \
            "lqc2 $vf20, 48(%[mt])\n\t" "lqc2 $vf21, 48(%[ht])\n\t"                          \
            "lqc2 $vf22, 64(%[mt])\n\t" "lqc2 $vf23, 64(%[ht])\n\t"                          \
            "lqc2 $vf24, 80(%[mt])\n\t" "lqc2 $vf25, 80(%[ht])\n\t"                          \
            "lqc2 $vf26, 96(%[mt])\n\t" "lqc2 $vf27, 96(%[ht])\n\t"                          \
            "lqc2 $vf28, 112(%[mt])\n\t" "lqc2 $vf29, 112(%[ht])\n\t"                        \
            "lqc2 $vf30, 128(%[mt])\n\t" "lqc2 $vf31, 128(%[ht])\n\t"                        \
            "beq %[ng], $0, 2f\n\t"                                                          \
            "1:\n\t"                                                                         \
            CONVDW_GROUP(R0, R1, R2)                                                         \
            "addiu %[pp0], %[pp0], 16\n\t"                                                   \
            "addiu %[pp1], %[pp1], 16\n\t"                                                   \
            "addiu %[pp2], %[pp2], 16\n\t"                                                   \
            "addiu %[dst], %[dst], 16\n\t"                                                   \
            "addiu %[ng], %[ng], -1\n\t"                                                     \
            "bne %[ng], $0, 1b\n\t"                                                          \
            "2:\n\t"                                                                         \
            : [pp0] "+r"(pp0), [pp1] "+r"(pp1), [pp2] "+r"(pp2), [dst] "+r"(dst),             \
              [ng] "+r"(ng)                                                                  \
            : [mt] "r"(mt_), [ht] "r"(ht_), [bt] "r"(bt_)                                     \
            : "memory");                                                                      \
    }

// row variant = (phase of run 0, phase step between runs); runs use
// phases (m0, m0+d, m0+2d) mod 4
CONVDW_ROWFN(cdw_ps2_row_0_0, 0, 0, 0)
CONVDW_ROWFN(cdw_ps2_row_0_1, 0, 1, 2)
CONVDW_ROWFN(cdw_ps2_row_0_2, 0, 2, 0)
CONVDW_ROWFN(cdw_ps2_row_0_3, 0, 3, 2)
CONVDW_ROWFN(cdw_ps2_row_1_0, 1, 1, 1)
CONVDW_ROWFN(cdw_ps2_row_1_1, 1, 2, 3)
CONVDW_ROWFN(cdw_ps2_row_1_2, 1, 3, 1)
CONVDW_ROWFN(cdw_ps2_row_1_3, 1, 0, 3)
CONVDW_ROWFN(cdw_ps2_row_2_0, 2, 2, 2)
CONVDW_ROWFN(cdw_ps2_row_2_1, 2, 3, 0)
CONVDW_ROWFN(cdw_ps2_row_2_2, 2, 0, 2)
CONVDW_ROWFN(cdw_ps2_row_2_3, 2, 1, 0)
CONVDW_ROWFN(cdw_ps2_row_3_0, 3, 3, 3)
CONVDW_ROWFN(cdw_ps2_row_3_1, 3, 0, 1)
CONVDW_ROWFN(cdw_ps2_row_3_2, 3, 1, 3)
CONVDW_ROWFN(cdw_ps2_row_3_3, 3, 2, 1)

static inline void convdw_ps2_3x3s1_row(const float* pp0, const float* pp1, const float* pp2,
                                        float* dst, int ngroups, const float* mt, const float* ht,
                                        const float* bt, int m0, int d)
{
    switch (m0 * 4 + d)
    {
    case 0: cdw_ps2_row_0_0(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 1: cdw_ps2_row_0_1(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 2: cdw_ps2_row_0_2(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 3: cdw_ps2_row_0_3(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 4: cdw_ps2_row_1_0(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 5: cdw_ps2_row_1_1(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 6: cdw_ps2_row_1_2(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 7: cdw_ps2_row_1_3(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 8: cdw_ps2_row_2_0(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 9: cdw_ps2_row_2_1(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 10: cdw_ps2_row_2_2(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 11: cdw_ps2_row_2_3(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 12: cdw_ps2_row_3_0(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 13: cdw_ps2_row_3_1(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 14: cdw_ps2_row_3_2(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    case 15: cdw_ps2_row_3_3(pp0, pp1, pp2, dst, ngroups, mt, ht, bt); break;
    default: break;
    }
}

static inline void convdw_ps2_kernel_3x3s1(
    const float* bottom,
    float* top,
    int w,
    int outw,
    int outh,
    int channels,
    int incstep,
    int outcstep,
    const float* weight_tm,
    const float* bias,
    int stride_h,
    int dilation_h)
{
    const int dw = dilation_h * w;
    const int koff[9] = {0, 1, 2, dw, dw + 1, dw + 2, 2 * dw, 2 * dw + 1, 2 * dw + 2};

    for (int p = 0; p < channels; p++)
    {
        const float* inptr = bottom + (size_t)p * incstep;
        float* outptr = top + (size_t)p * outcstep;
        const float* wbase = weight_tm + (size_t)p * 36;
        const float b = bias ? bias[p] : 0.f;

        for (int i = 0; i < outh; i++)
        {
            const int R = i * stride_h * w;
            const float* inrow = inptr + R;
            float* orow = outptr + (size_t)i * outw;
            const int uphase = (int)(((size_t)p * outcstep + (size_t)i * outw) & 3);
            const int js = (4 - uphase) & 3; // first vector group: aligned SQC2 store
            const int ngroups = outw - js >= 4 ? (outw - js) >> 2 : 0;

            if (ngroups > 0)
            {
                int m0[3];
                int off[3];
                float wmt[36] __attribute__((aligned(16)));
                float wht[36] __attribute__((aligned(16)));
                float bvec[12] __attribute__((aligned(16)));

                bvec[0] = bvec[1] = bvec[2] = bvec[3] = b;
                bvec[4] = bvec[5] = bvec[6] = bvec[7] = 1.f; // ones
                bvec[8] = bvec[9] = bvec[10] = bvec[11] = 0.f; // zeros

                for (int r = 0; r < 3; r++)
                {
                    m0[r] = (R + js + r * dw) & 3;
                    off[r] = r * dw - m0[r];
                }

                for (int k = 0; k < 9; k++)
                {
                    const int m = (m0[k / 3] + k % 3) & 3;
                    const float wv = wbase[k * 4];
                    for (int t = 0; t < 4; t++)
                    {
                        wmt[k * 4 + t] = convdw_ps2_mlo_tab[m][t] * wv;
                        wht[k * 4 + t] = convdw_ps2_mhi_tab[m][t] * wv;
                    }
                }

                convdw_ps2_3x3s1_row(inrow + js + off[0], inrow + js + off[1],
                                     inrow + js + off[2], orow + js, ngroups, wmt, wht, bvec,
                                     m0[0], (m0[1] - m0[0]) & 3);
            }

            // scalar tail: outputs before/after the aligned vector groups
            for (int j = 0; j < js; j++)
            {
                float sum = b;
                const float* xp = inrow + j;
                for (int k = 0; k < 9; k++)
                    sum += wbase[k * 4] * xp[koff[k]];
                orow[j] = sum;
            }
            for (int j = js + 4 * ngroups; j < outw; j++)
            {
                float sum = b;
                const float* xp = inrow + j;
                for (int k = 0; k < 9; k++)
                    sum += wbase[k * 4] * xp[koff[k]];
                orow[j] = sum;
            }
        }
    }
}

#endif // CONVDW_PS2_SIMD

static inline void convdw_ps2_kernel(

    const float* bottom,
    float* top,
    int w,
    int h,
    int channels,
    int outw,
    int outh,
    int incstep,
    int outcstep,
    const float* weight_tm,
    const float* bias,
    int kernel_w,
    int kernel_h,
    int stride_w,
    int stride_h,
    int dilation_w,
    int dilation_h)
{
    const int maxk = kernel_w * kernel_h;

#if CONVDW_PS2_SIMD
    // specialized vector path: 3x3, stride_w == 1, dilation_w == 1
    if (stride_w == 1 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && maxk == 9
        && outw > 0 && outh > 0
        && ((uintptr_t)bottom & 15) == 0 && ((uintptr_t)weight_tm & 15) == 0
        && ((uintptr_t)top & 15) == 0 && (incstep & 3) == 0 && (outcstep & 3) == 0)
    {
        convdw_ps2_kernel_3x3s1(bottom, top, w, outw, outh, channels, incstep, outcstep,
                                weight_tm, bias, stride_h, dilation_h);
        return;
    }
#endif

    // kernel offsets (same convention as the reference implementation)

    int space_ofs[CONVDW_PS2_MAXK];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int r = 0; r < kernel_h; r++)
        {
            for (int s = 0; s < kernel_w; s++)
            {
                space_ofs[p1++] = p2;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    if (maxk <= 0 || outw <= 0 || outh <= 0)
        return;

    if (stride_w != 1)
    {
        // ------------------------------------------------------------------
        // generic scalar path (stride != 1)
        // ------------------------------------------------------------------
        for (int p = 0; p < channels; p++)
        {
            const float* inptr = bottom + (size_t)p * incstep;
            float* outptr = top + (size_t)p * outcstep;
            const float* wbase = weight_tm + (size_t)p * maxk * 4;
            const float b = bias ? bias[p] : 0.f;

            for (int i = 0; i < outh; i++)
            {
                const float* inrow = inptr + (size_t)i * stride_h * w;

                for (int j = 0; j < outw; j++)
                {
                    float sum = b;
                    const float* sptr = inrow + (size_t)j * stride_w;
                    for (int k = 0; k < maxk; k++)
                    {
                        sum += wbase[k * 4] * sptr[space_ofs[k]];
                    }
                    outptr[j] = sum;
                }
                outptr += outw;
            }
        }
        return;
    }

#if CONVDW_PS2_SIMD
    // ----------------------------------------------------------------------
    // four-pixel-at-a-time VU0 macro mode path (stride_w == 1)
    // ----------------------------------------------------------------------
    {
        // 4 phase accumulators, each with 4 leading slots for the negative
        // (prefix) indices.  Row length is a multiple of 16 bytes.
        float scratch[4][4 + CONVDW_PS2_CHUNK + 8] __attribute__((aligned(16)));
        float* base[4];
        for (int c = 0; c < 4; c++)
            base[c] = &scratch[c][4];

        for (int p = 0; p < channels; p++)
        {
            const float* inptr = bottom + (size_t)p * incstep;
            float* outptr = top + (size_t)p * outcstep;
            const float* wbase = weight_tm + (size_t)p * maxk * 4;
            const float b = bias ? bias[p] : 0.f;

            for (int i = 0; i < outh; i++)
            {
                const int R = i * stride_h * w;
                const float* inrow = inptr + R;

                // group taps by alignment class for this row
                int lst[4][CONVDW_PS2_MAXK];
                int cnt[4] = {0, 0, 0, 0};
                for (int k = 0; k < maxk; k++)
                {
                    int c = (-(R + space_ofs[k])) & 3;
                    lst[c][cnt[c]++] = k;
                }

                for (int j0 = 0; j0 < outw; j0 += CONVDW_PS2_CHUNK)
                {
                    const int chunk = outw - j0 < CONVDW_PS2_CHUNK ? outw - j0 : CONVDW_PS2_CHUNK;
                    const int ct = (chunk + 3) / 4;

                    // clear the prefix (negative index) slots
                    for (int c = 0; c < 4; c++)
                    {
                        for (int jj = 0; jj < c; jj++)
                            base[c][jj - c] = 0.f;
                    }

                    // ---- vector pass ----
                    for (int t = 0; t < ct; t++)
                    {
                        __asm__ __volatile__(
                            "vsub.xyzw $vf4, $vf4, $vf4\n\t"
                            "vsub.xyzw $vf5, $vf5, $vf5\n\t"
                            "vsub.xyzw $vf6, $vf6, $vf6\n\t"
                            "vsub.xyzw $vf7, $vf7, $vf7\n\t");

                        CONVDW_PS2_TAP_LOOP(0, "$vf4")
                        CONVDW_PS2_TAP_LOOP(1, "$vf5")
                        CONVDW_PS2_TAP_LOOP(2, "$vf6")
                        CONVDW_PS2_TAP_LOOP(3, "$vf7")

                        float* s0 = base[0] + 4 * t;
                        float* s1 = base[1] + 4 * t;
                        float* s2 = base[2] + 4 * t;
                        float* s3 = base[3] + 4 * t;
                        __asm__ __volatile__(
                            "sqc2 $vf4, 0(%0)\n\t"
                            "sqc2 $vf5, 0(%1)\n\t"
                            "sqc2 $vf6, 0(%2)\n\t"
                            "sqc2 $vf7, 0(%3)\n\t"
                            :
                            : "r"(s0), "r"(s1), "r"(s2), "r"(s3)
                            : "memory");
                    }

                    // ---- prefix contributions (outputs 0..c-1 of the chunk) ----
                    for (int c = 0; c < 4; c++)
                    {
                        const int npr = c < chunk ? c : chunk;
                        for (int ii = 0; ii < cnt[c]; ii++)
                        {
                            int k = lst[c][ii];
                            const float wv = wbase[k * 4];
                            const int d = space_ofs[k];
                            float* dst = base[c] - c;
                            const float* src = inrow + j0 + d;
                            for (int jj = 0; jj < npr; jj++)
                                dst[jj] += wv * src[jj];
                        }
                    }

                    // ---- merge classes and write ----
                    float* o = outptr + j0;
                    for (int jj = 0; jj < chunk; jj++)
                    {
                        float sum = b;
                        sum += base[0][jj];
                        sum += base[1][jj - 1];
                        sum += base[2][jj - 2];
                        sum += base[3][jj - 3];
                        o[jj] = sum;
                    }
                }

                outptr += outw;
            }
        }
    }
#else
    // non-EE host build: plain scalar fallback
    for (int p = 0; p < channels; p++)
    {
        const float* inptr = bottom + (size_t)p * incstep;
        float* outptr = top + (size_t)p * outcstep;
        const float* wbase = weight_tm + (size_t)p * maxk * 4;
        const float b = bias ? bias[p] : 0.f;

        for (int i = 0; i < outh; i++)
        {
            const float* inrow = inptr + (size_t)i * stride_h * w;
            for (int j = 0; j < outw; j++)
            {
                float sum = b;
                const float* sptr = inrow + (size_t)j * stride_w;
                for (int k = 0; k < maxk; k++)
                    sum += wbase[k * 4] * sptr[space_ofs[k]];
                outptr[j] = sum;
            }
            outptr += outw;
        }
    }
#endif
}

#endif // LAYER_CONVOLUTIONDEPTHWISE_PS2_KERNEL_H
