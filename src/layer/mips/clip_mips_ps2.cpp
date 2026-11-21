// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "clip_mips_ps2.h"

namespace ncnn {

Clip_mips_ps2::Clip_mips_ps2()
{
    support_packing = true;
    support_any_packing = true;
}

int Clip_mips_ps2::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    const float min4buf[4] __attribute__((aligned(16))) = {min, min, min, min};
    const float max4buf[4] __attribute__((aligned(16))) = {max, max, max, max};

    // no omp: NCNN_OPENMP is off on PS2
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;

        if (elempack == 4)
        {
            // clamp: vmax with lower bound, then vmini with upper bound
            for (; i + 3 < size; i += 4)
            {
                __asm__ __volatile__(
                    "lqc2 $vf4, 0(%0)\n\t"
                    "lqc2 $vf5, 0(%1)\n\t"
                    "lqc2 $vf6, 0(%2)\n\t"
                    "vmax.xyzw $vf7, $vf4, $vf5\n\t"
                    "vmini.xyzw $vf7, $vf7, $vf6\n\t"
                    "sqc2 $vf7, 0(%0)\n\t"
                    :
                    : "r"(ptr), "r"(min4buf), "r"(max4buf)
                    : "memory");
                ptr += 4;
            }
        }

        // scalar remainder (and the whole scalar path when elempack != 4)
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
