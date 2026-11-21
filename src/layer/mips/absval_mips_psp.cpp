// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "absval_mips_psp.h"

#include "psp_vfpu.h"

namespace ncnn {

AbsVal_mips_psp::AbsVal_mips_psp()
{
    support_packing = true;
    support_any_packing = true;
}

int AbsVal_mips_psp::forward_inplace(Mat& bottom_top_blob, const Option& /*opt*/) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int size = w * h * d * bottom_top_blob.elempack;

    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
        if (psp_vfpu_aligned(ptr))
        {
            for (; i + 3 < size; i += 4)
            {
                PSP_VFPU_ABS(ptr, ptr);
                ptr += 4;
            }
        }

        for (; i < size; i++)
        {
            *ptr = *ptr > 0.f ? *ptr : -*ptr;
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
