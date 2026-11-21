// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CLIP_MIPS_PSP_H
#define LAYER_CLIP_MIPS_PSP_H

#include "clip.h"

namespace ncnn {

class Clip_mips_psp : public Clip
{
public:
    Clip_mips_psp();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CLIP_MIPS_PSP_H
