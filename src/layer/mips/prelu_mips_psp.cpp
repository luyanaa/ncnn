// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prelu_mips_psp.h"

#include "psp_vfpu.h"

namespace ncnn {

PReLU_mips_psp::PReLU_mips_psp()
{
    support_packing = true;
    support_any_packing = true;
}

// slope4 points at four aligned floats. elempack 4 with per-element slopes
// uses the packed slope directly; everything else broadcasts one slope.
static void prelu_channel(float* ptr, int size, const float* slope4)
{
    static const float zero4[4] __attribute__((aligned(16))) = {0.f, 0.f, 0.f, 0.f};

    int i = 0;
    if (psp_vfpu_aligned(ptr))
    {
        for (; i + 3 < size; i += 4)
        {
            PSP_VFPU_LEAKY(ptr, zero4, slope4, ptr);
            ptr += 4;
        }
    }

    for (; i < size; i++)
    {
        if (*ptr < 0.f)
            *ptr *= slope4[i % 4];
        ptr++;
    }
}

int PReLU_mips_psp::forward_inplace(Mat& bottom_top_blob, const Option& /*opt*/) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const float* slope = slope_data;

    if (dims == 1)
    {
        const int w = bottom_top_blob.w * elempack;
        float* ptr = bottom_top_blob;

        if (num_slope > 1 && elempack == 4 && psp_vfpu_aligned(slope))
        {
            prelu_channel(ptr, w, slope);
        }
        else if (num_slope > 1)
        {
            for (int i = 0; i < w; i++)
                if (ptr[i] < 0.f)
                    ptr[i] *= slope[i];
        }
        else
        {
            const float s4[4] __attribute__((aligned(16))) = {slope[0], slope[0], slope[0], slope[0]};
            prelu_channel(ptr, w, s4);
        }
    }

    if (dims == 2)
    {
        const int w = bottom_top_blob.w * elempack;
        const int h = bottom_top_blob.h;

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            if (num_slope > 1 && elempack == 4 && psp_vfpu_aligned(slope + i * 4))
            {
                prelu_channel(ptr, w, slope + i * 4);
            }
            else
            {
                const float s = num_slope > 1 ? slope[i] : slope[0];
                const float s4[4] __attribute__((aligned(16))) = {s, s, s, s};
                prelu_channel(ptr, w, s4);
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int d = bottom_top_blob.d;
        const int channels = bottom_top_blob.c;
        const int size = w * h * d * elempack;

        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            if (num_slope > 1 && elempack == 4 && psp_vfpu_aligned(slope + q * 4))
            {
                prelu_channel(ptr, size, slope + q * 4);
            }
            else
            {
                const float s = num_slope > 1 ? slope[q] : slope[0];
                const float s4[4] __attribute__((aligned(16))) = {s, s, s, s};
                prelu_channel(ptr, size, s4);
            }
        }
    }

    return 0;
}

} // namespace ncnn
