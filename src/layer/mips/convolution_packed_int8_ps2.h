// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// EE MMI int8 convolution.
//
// The R5900 has no byte multiply.  PHMADH sums four adjacent pairs of signed
// 16x16 products, so each byte is widened to a halfword first: PEXTLB with a
// zeroed operand zero-extends eight bytes into eight halfwords (verified on
// PCSX2: pextlb zero, src puts src's bytes in the low half of each halfword),
// and PSRAH 8 sign-extends them.  One PHMADH then covers eight input channels
// for one output channel.
//
// The weight layout consumed here is the non-MSA packed layout built by
// convolution_transform_kernel_packed_int8(): for a group of two output
// channels the bytes run [k][q/2] as k0[0],k0[1],k1[0],k1[1] repeated, which
// is exactly the interleave PEXTLB wants, so the weights are not repacked.
//
// Only the shapes the widening pays for are handled: elempack == 1 and both
// channel counts a multiple of 8.  Anything else returns false and the caller
// keeps the scalar convolution_packed_int8 path.  The int8 dot product itself
// measured slower than that scalar path, so this stays off unless
// NCNN_PS2_INT8_MMI is set.

#if defined(_EE) || defined(__PS2__)

#include "ps2_mmi.h"

static bool convolution_packed_int8_ps2(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
#if !defined(NCNN_PS2_INT8_MMI)
    (void)bottom_blob;
    (void)top_blob;
    (void)weight_data_tm;
    (void)kernel_w;
    (void)kernel_h;
    (void)dilation_w;
    (void)dilation_h;
    (void)stride_w;
    (void)stride_h;
    (void)opt;
    return false;
#else
    if (bottom_blob.elempack != 1 || top_blob.elempack != 1)
        return false;

    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    if (inch < 8 || (inch & 7) || outch < 2 || (outch & 1))
        return false;

    const int maxk = kernel_w * kernel_h;

    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    const size_t N = bottom_blob.cstep;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < outch / 2; pp++)
    {
        const int p = pp * 2;

        const int outw = top_blob.w;
        const int outh = top_blob.h;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                const signed char* kptr = (const signed char*)weight_data_tm.channel(p / 2);

                int sum0 = 0;
                int sum1 = 0;

                for (int q = 0; q < inch; q += 8)
                {
                    const signed char* sbase = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        // one tap, eight input channels, two output channels.
                        // The input bytes of one tap sit at the same column of
                        // eight consecutive channel planes, so two lq plus two
                        // por gather all eight.  The weight bytes are already
                        // interleaved per output channel, eight per channel.
                        //
                        // PHMADH writes one 32-bit sum per adjacent pair and
                        // only the low 64 bits of its result are addressable
                        // as a C scalar, so the four pair-sums are added here,
                        // inside the asm, and a single 32-bit total comes out.
                        int d0, d1;
                        __asm__ __volatile__(
                            "lq $8, 0(%4)\n\t"
                            "lq $9, 0(%5)\n\t"
                            "pcpyld $10, %6, %6\n\t"
                            "addu $11, %6, %7\n\t"
                            "pcpyld $11, $11, $11\n\t"
                            "lq $12, 0(%8)\n\t"
                            "por $8, $8, $10\n\t"
                            "por $9, $9, $11\n\t"
                            "pxor $13, $13, $13\n\t"
                            "pextlb $14, $13, $8\n\t"
                            "pextlb $15, $13, $12\n\t"
                            "psrah $14, $14, 8\n\t"
                            "psrah $15, $15, 8\n\t"
                            "phmadh $12, $14, $15\n\t"
                            "pextub $15, $13, $12\n\t"
                            "psrah $15, $15, 8\n\t"
                            "phmadh $13, $14, $15\n\t"
                            "pcpyud $14, $12, $12\n\t"
                            "paddw $12, $12, $14\n\t"
                            "pcpyud $14, $13, $13\n\t"
                            "paddw $13, $13, $14\n\t"
                            "pcpyld $14, $12, $12\n\t"
                            "paddw %0, $12, $14\n\t"
                            "pcpyld $14, $13, $13\n\t"
                            "paddw %1, $13, $14\n\t"
                            : "=&r"(d0), "=&r"(d1)
                            : "m"(*(const int*)sbase), "m"(*(const int*)kptr),
                            "r"(sbase), "r"(sbase + N), "r"(space_ofs[k]), "r"(w), "r"(kptr)
                            : "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15", "$hi", "$lo", "memory");

                        // paddw summed both halves, so each word holds the
                        // same total; the low 32 bits are the dot product.
                        sum0 += d0;
                        sum1 += d1;

                        kptr += 16;
                    }
                }

                outptr0[j] = sum0;
                outptr1[j] = sum1;
            }

            outptr0 += outw;
            outptr1 += outw;
        }
    }

    return true;
#endif
}

#else

static bool convolution_packed_int8_ps2(const Mat&, Mat&, const Mat&, int, int, int, int, int, int, const Option&)
{
    return false;
}

#endif
