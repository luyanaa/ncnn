// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTION_MIPS_PS2_H
#define LAYER_CONVOLUTION_MIPS_PS2_H

#include "convolution_mips.h"

namespace ncnn {

class Convolution_mips_ps2 : public Convolution_mips
{
public:
    Convolution_mips_ps2();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    // transformed kernel, layout [outch][maxk][inch4]
    // inch4 is inch rounded up to a multiple of 4 so that every lqc2/sqc2 on
    // a row of this buffer stays quadword aligned.  Trailing lanes are zero.
    // Used by the output-stationary kernel (stride_w != 1 cases).
    Mat weight_data_ps2;

    // transformed kernel for the weight-stationary kernel, layout
    // [maxk][inch][outch4]: for a fixed (tap k, input channel q) the four
    // output-channel weights p..p+3 are contiguous so one lqc2 fetches them
    // and they can be broadcast into the pixel-vector accumulators.
    // Output channels p >= num_output are zero.
    // Only built when the weight-stationary kernel is usable
    // (gated shape and stride_w == 1); otherwise empty.
    Mat weight_data_ps2_wst;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_MIPS_PS2_H
