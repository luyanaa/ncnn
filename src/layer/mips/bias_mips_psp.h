// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BIAS_MIPS_PSP_H
#define LAYER_BIAS_MIPS_PSP_H

#include "bias.h"

namespace ncnn {

class Bias_mips_psp : public Bias
{
public:
    Bias_mips_psp();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_BIAS_MIPS_PSP_H
