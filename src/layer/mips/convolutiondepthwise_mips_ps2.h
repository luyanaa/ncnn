// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTIONDEPTHWISE_MIPS_PS2_H
#define LAYER_CONVOLUTIONDEPTHWISE_MIPS_PS2_H

#include "convolutiondepthwise_mips.h"

namespace ncnn {

class ConvolutionDepthWise_mips_ps2 : public ConvolutionDepthWise_mips
{
public:
    ConvolutionDepthWise_mips_ps2();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    // [num_output][maxk][4], each tap broadcast to four lanes for LQC2
    Mat weight_data_ps2;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE_MIPS_PS2_H
