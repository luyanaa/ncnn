// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "batchnorm_mips_psp.h"

#include "psp_vfpu.h"

namespace ncnn {

BatchNorm_mips_psp::BatchNorm_mips_psp()
{
    support_packing = true;
    support_any_packing = true;
}

// out = b * in + a. a4 and b4 are four aligned floats.
static void batchnorm_row(float* ptr, int size, const float* a4, const float* b4)
{
    int i = 0;
    if (psp_vfpu_aligned(ptr))
    {
        for (; i + 3 < size; i += 4)
        {
            __asm__ __volatile__(
                "lv.q C000, 0(%0)\n\t"
                "lv.q C010, 0(%1)\n\t"
                "lv.q C020, 0(%2)\n\t"
                "vmul.q C000, C000, C020\n\t"
                "vadd.q C000, C000, C010\n\t"
                "sv.q C000, 0(%0)\n\t"
                :
                : "r"(ptr), "r"(a4), "r"(b4)
                : "memory");
            ptr += 4;
        }
    }

    for (; i < size; i++)
    {
        *ptr = b4[i % 4] * *ptr + a4[i % 4];
        ptr++;
    }
}

static void batchnorm_group(float* ptr, int size, int elempack, const float* a, const float* b)
{
    if (elempack == 4 && psp_vfpu_aligned(a) && psp_vfpu_aligned(b))
    {
        batchnorm_row(ptr, size, a, b);
        return;
    }

    float a4[4] __attribute__((aligned(16)));
    float b4[4] __attribute__((aligned(16)));
    for (int k = 0; k < 4; k++)
    {
        a4[k] = a[elempack == 4 ? k : 0];
        b4[k] = b[elempack == 4 ? k : 0];
    }
    batchnorm_row(ptr, size, a4, b4);
}

int BatchNorm_mips_psp::forward_inplace(Mat& bottom_top_blob, const Option& /*opt*/) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const float* a = a_data;
    const float* b = b_data;

    if (dims == 1)
    {
        const int w = bottom_top_blob.w * elempack;
        batchnorm_group(bottom_top_blob, w, elempack, a, b);
    }

    if (dims == 2)
    {
        const int w = bottom_top_blob.w * elempack;
        const int h = bottom_top_blob.h;

        for (int i = 0; i < h; i++)
            batchnorm_group(bottom_top_blob.row(i), w, elempack, a + i * elempack, b + i * elempack);
    }

    if (dims == 3 || dims == 4)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int d = bottom_top_blob.d;
        const int channels = bottom_top_blob.c;
        const int size = w * h * d * elempack;

        for (int q = 0; q < channels; q++)
            batchnorm_group(bottom_top_blob.channel(q), size, elempack, a + q * elempack, b + q * elempack);
    }

    return 0;
}

} // namespace ncnn
