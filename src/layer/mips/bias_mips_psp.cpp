// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "bias_mips_psp.h"

#include "psp_vfpu.h"

namespace ncnn {

Bias_mips_psp::Bias_mips_psp()
{
    support_packing = true;
    support_any_packing = true;
}

int Bias_mips_psp::forward_inplace(Mat& bottom_top_blob, const Option& /*opt*/) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    const float* bias_ptr = bias_data;

    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        // elempack 4 stores the four biases packed and aligned; otherwise one
        // bias is broadcast across the channel.
        float bias4buf[4] __attribute__((aligned(16)));
        const float* bias4 = bias4buf;
        if (elempack == 4 && psp_vfpu_aligned(bias_ptr + q * 4))
        {
            bias4 = bias_ptr + q * 4;
        }
        else
        {
            const float b = bias_ptr[q];
            bias4buf[0] = bias4buf[1] = bias4buf[2] = bias4buf[3] = b;
        }

        int i = 0;
        if (psp_vfpu_aligned(ptr))
        {
            for (; i + 3 < size; i += 4)
            {
                PSP_VFPU_BINARY("vadd.q", ptr, bias4, ptr);
                ptr += 4;
            }
        }

        for (; i < size; i++)
        {
            *ptr += bias4[i % 4];
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
