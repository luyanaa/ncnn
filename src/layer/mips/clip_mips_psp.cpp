// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "clip_mips_psp.h"

#include "psp_vfpu.h"

namespace ncnn {

Clip_mips_psp::Clip_mips_psp()
{
    support_packing = true;
    support_any_packing = true;
}

int Clip_mips_psp::forward_inplace(Mat& bottom_top_blob, const Option& /*opt*/) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int size = w * h * d * bottom_top_blob.elempack;

    const float min4[4] __attribute__((aligned(16))) = {min, min, min, min};
    const float max4[4] __attribute__((aligned(16))) = {max, max, max, max};

    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
        if (psp_vfpu_aligned(ptr))
        {
            for (; i + 3 < size; i += 4)
            {
                // clamp to [min, max]. vmax and vmin are plain vector ops.
                __asm__ __volatile__(
                    "lv.q C000, 0(%0)\n\t"
                    "lv.q C010, 0(%1)\n\t"
                    "lv.q C020, 0(%2)\n\t"
                    "vmax.q C000, C000, C010\n\t"
                    "vmin.q C000, C000, C020\n\t"
                    "sv.q C000, 0(%0)\n\t"
                    :
                    : "r"(ptr), "r"(min4), "r"(max4)
                    : "memory");
                ptr += 4;
            }
        }

        for (; i < size; i++)
        {
            if (*ptr < min)
                *ptr = min;
            if (*ptr > max)
                *ptr = max;
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
