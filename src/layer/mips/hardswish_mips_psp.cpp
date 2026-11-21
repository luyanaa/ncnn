// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "hardswish_mips_psp.h"

#include "psp_vfpu.h"

namespace ncnn {

HardSwish_mips_psp::HardSwish_mips_psp()
{
    support_packing = true;
    support_any_packing = true;
}

int HardSwish_mips_psp::forward_inplace(Mat& bottom_top_blob, const Option& /*opt*/) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int size = w * h * d * bottom_top_blob.elempack;

    const float alpha4[4] __attribute__((aligned(16))) = {alpha, alpha, alpha, alpha};
    const float beta4[4] __attribute__((aligned(16))) = {beta, beta, beta, beta};
    static const float zero4[4] __attribute__((aligned(16))) = {0.f, 0.f, 0.f, 0.f};
    static const float one4[4] __attribute__((aligned(16))) = {1.f, 1.f, 1.f, 1.f};

    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
        if (psp_vfpu_aligned(ptr))
        {
            for (; i + 3 < size; i += 4)
            {
                PSP_VFPU_HARDSWISH(ptr, alpha4, beta4, zero4, one4, ptr);
                ptr += 4;
            }
        }

        for (; i < size; i++)
        {
            if (*ptr < lower)
                *ptr = 0.f;
            else if (*ptr > upper)
                ;
            else
                *ptr = *ptr * (*ptr * alpha + beta);
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
