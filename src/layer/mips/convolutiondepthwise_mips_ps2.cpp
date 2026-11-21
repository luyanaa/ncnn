// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolutiondepthwise_mips_ps2.h"

#include "convolutiondepthwise_mips_ps2_kernel.h"
#include "fused_activation.h"

namespace ncnn {

ConvolutionDepthWise_mips_ps2::ConvolutionDepthWise_mips_ps2()
{
    support_packing = false;
    support_bf16_storage = false;
}

int ConvolutionDepthWise_mips_ps2::create_pipeline(const Option& opt)
{
    // anything we do not transform keeps the ConvolutionDepthWise_mips
    // implementation (it has the int8 packed / bf16 pipelines)
    if (dynamic_weight || int8_scale_term || group != num_output)
        return ConvolutionDepthWise_mips::create_pipeline(opt);

    const int maxk = kernel_w * kernel_h;
    if (maxk <= 0 || maxk > CONVDW_PS2_MAXK)
        return ConvolutionDepthWise_mips::create_pipeline(opt);

    if (weight_data_size != num_output * maxk)
        return ConvolutionDepthWise_mips::create_pipeline(opt);

    if (weight_data.elemsize != (size_t)4u)
        return ConvolutionDepthWise_mips::create_pipeline(opt);

    weight_data_ps2.create(maxk * 4, num_output, (size_t)4u, opt.blob_allocator);
    if (weight_data_ps2.empty())
        return -100;

    for (int p = 0; p < num_output; p++)
    {
        const float* w = (const float*)weight_data + (size_t)p * maxk;
        float* wt = weight_data_ps2.row(p);
        for (int k = 0; k < maxk; k++)
        {
            wt[k * 4] = w[k];
            wt[k * 4 + 1] = w[k];
            wt[k * 4 + 2] = w[k];
            wt[k * 4 + 3] = w[k];
        }
    }

    // PS2 has 32MB of RAM; the broadcast-expanded copy is up to 4x the
    // original, so drop the original weights now that forward() owns the op.
    weight_data.release();

    return 0;
}

int ConvolutionDepthWise_mips_ps2::destroy_pipeline(const Option& opt)
{
    weight_data_ps2.release();
    return ConvolutionDepthWise_mips::destroy_pipeline(opt);
}

int ConvolutionDepthWise_mips_ps2::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // nothing was transformed in create_pipeline (int8 / dynamic weight /
    // grouped conv / unsupported size): the original weights are still there
    if (weight_data_ps2.empty())
        return ConvolutionDepthWise_mips::forward(bottom_blob, top_blob, opt);

    // from here on weight_data has been released, this layer owns the op
    if (bottom_blob.elempack != 1 || bottom_blob.elemsize != (size_t)4u)
        return -1;

    if (bottom_blob.c != num_output)
        return -1;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, (size_t)4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    convdw_ps2_kernel((const float*)bottom_blob_bordered, (float*)top_blob,
                      w, h, num_output, outw, outh,
                      (int)bottom_blob_bordered.cstep, (int)top_blob.cstep,
                      (const float*)weight_data_ps2,
                      bias_term ? (const float*)bias_data : 0,
                      kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h);

    if (activation_type != 0)
    {
        const int size = outw * outh;
        for (int p = 0; p < num_output; p++)
        {
            float* ptr = top_blob.channel(p);
            for (int i = 0; i < size; i++)
                ptr[i] = activation_ss(ptr[i], activation_type, activation_params);
        }
    }

    return 0;
}

} // namespace ncnn
