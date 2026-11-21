// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RELU_MIPS_PS2_H
#define LAYER_RELU_MIPS_PS2_H

#include "relu.h"

namespace ncnn {

class ReLU_mips_ps2 : public ReLU
{
public:
    ReLU_mips_ps2();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_RELU_MIPS_PS2_H
