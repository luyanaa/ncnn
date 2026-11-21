// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "relu_mips_ps2.h"

namespace ncnn {

ReLU_mips_ps2::ReLU_mips_ps2()
{
    support_packing = true;
    support_any_packing = true;
}

int ReLU_mips_ps2::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    static const float zero4[4] __attribute__((aligned(16))) = {0.f, 0.f, 0.f, 0.f};
    const float slope4buf[4] __attribute__((aligned(16))) = {slope, slope, slope, slope};

    // no omp: NCNN_OPENMP is off on PS2
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;

        if (elempack == 4)
        {
            if (slope == 0.f)
            {
                // relu: vmax(p, 0)
                for (; i + 3 < size; i += 4)
                {
                    __asm__ __volatile__(
                        "lqc2 $vf4, 0(%0)\n\t"
                        "lqc2 $vf5, 0(%1)\n\t"
                        "vmax.xyzw $vf6, $vf4, $vf5\n\t"
                        "sqc2 $vf6, 0(%2)\n\t"
                        :
                        : "r"(ptr), "r"(zero4), "r"(ptr)
                        : "memory");
                    ptr += 4;
                }
            }
            else
            {
                // leaky relu: pos = vmax(p, 0), neg = vmini(p, 0)
                // out = pos + slope * neg
                for (; i + 3 < size; i += 4)
                {
                    __asm__ __volatile__(
                        "lqc2 $vf4, 0(%0)\n\t"
                        "lqc2 $vf5, 0(%1)\n\t"
                        "lqc2 $vf8, 0(%3)\n\t"
                        "vmax.xyzw $vf6, $vf4, $vf5\n\t" // vf6 = max(p, 0)
                        "vmini.xyzw $vf7, $vf4, $vf5\n\t" // vf7 = min(p, 0)
                        "vmul.xyzw $vf7, $vf7, $vf8\n\t" // vf7 = slope * neg
                        "vadd.xyzw $vf6, $vf6, $vf7\n\t" // vf6 = pos + slope * neg
                        "sqc2 $vf6, 0(%2)\n\t"
                        :
                        : "r"(ptr), "r"(zero4), "r"(ptr), "r"(slope4buf)
                        : "memory");
                    ptr += 4;
                }
            }
        }

        // scalar remainder (and the whole scalar path when elempack != 4)
        if (slope == 0.f)
        {
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr = 0;
                ptr++;
            }
        }
        else
        {
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
