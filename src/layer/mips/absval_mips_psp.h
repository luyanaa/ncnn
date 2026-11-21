// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ABSVAL_MIPS_PSP_H
#define LAYER_ABSVAL_MIPS_PSP_H

#include "absval.h"

namespace ncnn {

class AbsVal_mips_psp : public AbsVal
{
public:
    AbsVal_mips_psp();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ABSVAL_MIPS_PSP_H
