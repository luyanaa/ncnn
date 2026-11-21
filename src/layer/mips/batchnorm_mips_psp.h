// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BATCHNORM_MIPS_PSP_H
#define LAYER_BATCHNORM_MIPS_PSP_H

#include "batchnorm.h"

namespace ncnn {

class BatchNorm_mips_psp : public BatchNorm
{
public:
    BatchNorm_mips_psp();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_BATCHNORM_MIPS_PSP_H
