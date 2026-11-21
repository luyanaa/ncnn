// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_mips_psp.h"

#include "psp_vfpu.h"

namespace ncnn {

Scale_mips_psp::Scale_mips_psp()
{
    support_packing = true;
    support_any_packing = true;
}

// scale4 and bias4 are four aligned floats. bias4 is null when there is no
// bias term. The tail indexes lanes with i % 4, which is only reached when a
// pointer is unaligned; aligned channels have a multiple-of-four length.
static void scale_row(float* ptr, int size, const float* scale4, const float* bias4)
{
    int i = 0;
    if (psp_vfpu_aligned(ptr))
    {
        if (bias4)
        {
            for (; i + 3 < size; i += 4)
            {
                __asm__ __volatile__(
                    "lv.q C000, 0(%0)\n\t"
                    "lv.q C010, 0(%1)\n\t"
                    "lv.q C020, 0(%2)\n\t"
                    "vmul.q C000, C000, C010\n\t"
                    "vadd.q C000, C000, C020\n\t"
                    "sv.q C000, 0(%0)\n\t"
                    :
                    : "r"(ptr), "r"(scale4), "r"(bias4)
                    : "memory");
                ptr += 4;
            }
        }
        else
        {
            for (; i + 3 < size; i += 4)
            {
                PSP_VFPU_BINARY("vmul.q", ptr, scale4, ptr);
                ptr += 4;
            }
        }
    }

    for (; i < size; i++)
    {
        *ptr *= scale4[i % 4];
        if (bias4)
            *ptr += bias4[i % 4];
        ptr++;
    }
}

// elempack 4 carries four independent scale values, packed and aligned.
// Anything else broadcasts the single value for this group.
static void scale_group(float* ptr, int size, int elempack, const float* scale, const float* bias)
{
    if (elempack == 4 && psp_vfpu_aligned(scale) && (!bias || psp_vfpu_aligned(bias)))
    {
        scale_row(ptr, size, scale, bias);
        return;
    }

    float scale4[4] __attribute__((aligned(16)));
    float bias4buf[4] __attribute__((aligned(16)));
    for (int k = 0; k < 4; k++)
    {
        scale4[k] = scale[elempack == 4 ? k : 0];
        if (bias)
            bias4buf[k] = bias[elempack == 4 ? k : 0];
    }
    scale_row(ptr, size, scale4, bias ? bias4buf : 0);
}

int Scale_mips_psp::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& /*opt*/) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const float* scale = scale_blob;
    const float* bias = bias_term ? (const float*)bias_data : 0;

    if (dims == 1)
    {
        const int size = bottom_top_blob.w * elempack;
        scale_group(bottom_top_blob, size, elempack, scale, bias);
    }

    if (dims == 2)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int size = w * elempack;

        for (int i = 0; i < h; i++)
            scale_group(bottom_top_blob.row(i), size, elempack, scale + i * elempack, bias ? bias + i * elempack : 0);
    }

    if (dims == 3 || dims == 4)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int d = bottom_top_blob.d;
        const int channels = bottom_top_blob.c;
        const int size = w * h * d * elempack;

        for (int q = 0; q < channels; q++)
            scale_group(bottom_top_blob.channel(q), size, elempack, scale + q * elempack, bias ? bias + q * elempack : 0);
    }

    return 0;
}

} // namespace ncnn
