// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PRELU_MIPS_PSP_H
#define LAYER_PRELU_MIPS_PSP_H

#include "prelu.h"

namespace ncnn {

class PReLU_mips_psp : public PReLU
{
public:
    PReLU_mips_psp();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PRELU_MIPS_PSP_H
