// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DROPOUT_MIPS_PSP_H
#define LAYER_DROPOUT_MIPS_PSP_H

#include "dropout.h"

namespace ncnn {

class Dropout_mips_psp : public Dropout
{
public:
    Dropout_mips_psp();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DROPOUT_MIPS_PSP_H
