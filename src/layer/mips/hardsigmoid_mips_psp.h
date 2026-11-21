// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_HARDSIGMOID_MIPS_PSP_H
#define LAYER_HARDSIGMOID_MIPS_PSP_H

#include "hardsigmoid.h"

namespace ncnn {

class HardSigmoid_mips_psp : public HardSigmoid
{
public:
    HardSigmoid_mips_psp();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_HARDSIGMOID_MIPS_PSP_H
