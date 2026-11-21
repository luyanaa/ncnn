// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SCALE_MIPS_PSP_H
#define LAYER_SCALE_MIPS_PSP_H

#include "scale.h"

namespace ncnn {

class Scale_mips_psp : public Scale
{
public:
    Scale_mips_psp();

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SCALE_MIPS_PSP_H
