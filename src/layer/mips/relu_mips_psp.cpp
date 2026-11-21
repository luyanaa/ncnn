// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "relu_mips_psp.h"

#include "psp_vfpu.h"

namespace ncnn {

ReLU_mips_psp::ReLU_mips_psp()
{
    support_packing = true;
    support_any_packing = true;
}

int ReLU_mips_psp::forward_inplace(Mat& bottom_top_blob, const Option& /*opt*/) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    // float32 only. support_bf16_storage stays off, so the framework casts
    // back to fp32 before calling this.

    static const float zero4[4] __attribute__((aligned(16))) = {0.f, 0.f, 0.f, 0.f};
    const float slope4[4] __attribute__((aligned(16))) = {slope, slope, slope, slope};

    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;

        if (psp_vfpu_aligned(ptr))
        {
            if (slope == 0.f)
            {
                for (; i + 3 < size; i += 4)
                {
                    PSP_VFPU_BINARY("vmax.q", ptr, zero4, ptr);
                    ptr += 4;
                }
            }
            else
            {
                // leaky relu: pos + slope * neg
                for (; i + 3 < size; i += 4)
                {
                    PSP_VFPU_LEAKY(ptr, zero4, slope4, ptr);
                    ptr += 4;
                }
            }
        }

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
