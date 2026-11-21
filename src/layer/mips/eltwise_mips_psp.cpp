// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eltwise_mips_psp.h"

#include <algorithm>

#include "psp_vfpu.h"

namespace ncnn {

Eltwise_mips_psp::Eltwise_mips_psp()
{
    support_packing = true;
    support_any_packing = true;
}

// op is a .q mnemonic. Both pointers must be 16-byte aligned, which ncnn's
// allocator gives every channel; the caller checks anyway.
enum EltwiseOp
{
    ELT_ADD,
    ELT_MUL,
    ELT_MAX
};

static void eltwise_channel(const float* a, const float* b, float* out, int size, EltwiseOp op)
{
    int i = 0;
    if (psp_vfpu_aligned(a) && psp_vfpu_aligned(b) && psp_vfpu_aligned(out))
    {
        for (; i + 3 < size; i += 4)
        {
            if (op == ELT_ADD)
                PSP_VFPU_BINARY("vadd.q", a, b, out);
            else if (op == ELT_MUL)
                PSP_VFPU_BINARY("vmul.q", a, b, out);
            else
                PSP_VFPU_BINARY("vmax.q", a, b, out);

            a += 4;
            b += 4;
            out += 4;
        }
    }

    for (; i < size; i++)
    {
        if (op == ELT_ADD)
            *out = *a + *b;
        else if (op == ELT_MUL)
            *out = *a * *b;
        else
            *out = std::max(*a, *b);

        a++;
        b++;
        out++;
    }
}

// Weighted sum: out = a * c0 + b * c1. c0 and c1 are aligned quads.
static void eltwise_weighted(const float* a, const float* b, float* out, int size, const float* c0, const float* c1)
{
    int i = 0;
    if (psp_vfpu_aligned(a) && psp_vfpu_aligned(b) && psp_vfpu_aligned(out))
    {
        for (; i + 3 < size; i += 4)
        {
            __asm__ __volatile__(
                "lv.q C000, 0(%0)\n\t"
                "lv.q C010, 0(%1)\n\t"
                "lv.q C020, 0(%2)\n\t"
                "lv.q C030, 0(%3)\n\t"
                "vmul.q C000, C000, C020\n\t"
                "vmul.q C010, C010, C030\n\t"
                "vadd.q C000, C000, C010\n\t"
                "sv.q C000, 0(%4)\n\t"
                :
                : "r"(a), "r"(b), "r"(c0), "r"(c1), "r"(out)
                : "memory");
            a += 4;
            b += 4;
            out += 4;
        }
    }

    for (; i < size; i++)
    {
        *out = *a * c0[0] + *b * c1[0];
        a++;
        b++;
        out++;
    }
}

int Eltwise_mips_psp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int size = w * h * d * bottom_blob.elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const EltwiseOp op = op_type == Operation_PROD ? ELT_MUL : op_type == Operation_MAX ? ELT_MAX : ELT_ADD;

    if (op_type == Operation_SUM && coeffs.w != 0)
    {
        const float c0[4] __attribute__((aligned(16))) = {coeffs[0], coeffs[0], coeffs[0], coeffs[0]};
        const float c1[4] __attribute__((aligned(16))) = {coeffs[1], coeffs[1], coeffs[1], coeffs[1]};

        for (int q = 0; q < channels; q++)
            eltwise_weighted(bottom_blob.channel(q), bottom_blobs[1].channel(q), top_blob.channel(q), size, c0, c1);

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const float c[4] __attribute__((aligned(16))) = {coeffs[b], coeffs[b], coeffs[b], coeffs[b]};
            static const float one[4] __attribute__((aligned(16))) = {1.f, 1.f, 1.f, 1.f};

            for (int q = 0; q < channels; q++)
                eltwise_weighted(top_blob.channel(q), bottom_blobs[b].channel(q), top_blob.channel(q), size, one, c);
        }

        return 0;
    }

    for (int q = 0; q < channels; q++)
        eltwise_channel(bottom_blob.channel(q), bottom_blobs[1].channel(q), top_blob.channel(q), size, op);

    for (size_t b = 2; b < bottom_blobs.size(); b++)
        for (int q = 0; q < channels; q++)
            eltwise_channel(top_blob.channel(q), bottom_blobs[b].channel(q), top_blob.channel(q), size, op);

    return 0;
}

} // namespace ncnn
