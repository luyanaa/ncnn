// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dropout_mips_psp.h"

#include "psp_vfpu.h"

namespace ncnn {

Dropout_mips_psp::Dropout_mips_psp()
{
    support_packing = true;
    support_any_packing = true;
}

int Dropout_mips_psp::forward_inplace(Mat& bottom_top_blob, const Option& /*opt*/) const
{
    if (scale == 1.f)
        return 0;

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int size = w * h * d * bottom_top_blob.elempack;

    const float scale4[4] __attribute__((aligned(16))) = {scale, scale, scale, scale};

    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
        if (psp_vfpu_aligned(ptr))
        {
            for (; i + 3 < size; i += 4)
            {
                PSP_VFPU_BINARY("vmul.q", ptr, scale4, ptr);
                ptr += 4;
            }
        }

        for (; i < size; i++)
        {
            *ptr *= scale;
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
