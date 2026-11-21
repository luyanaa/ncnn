// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_mips_ps2.h"

#include <string.h>
#include <vector>

#include "convolution_mips_ps2_kernel.h"



#include "fused_activation.h"

namespace ncnn {

Convolution_mips_ps2::Convolution_mips_ps2()
{
    support_packing = false;
    support_bf16_storage = false;
}

int Convolution_mips_ps2::create_pipeline(const Option& opt)
{
    (void)opt;

    // int8 quantized and dynamic-weight convolution keep the Convolution_mips
    // implementation: it builds the packed / winograd int8 weights that the
    // plain Convolution int8 path does not have (3x faster on int8 models).
    if (int8_scale_term || dynamic_weight)
        return Convolution_mips::create_pipeline(opt);

    if (weight_data.elemsize != (size_t)4u || weight_data.empty())
        return Convolution_mips::create_pipeline(opt);

    const int maxk = kernel_w * kernel_h;

    if (maxk <= 0 || num_output <= 0 || weight_data_size <= 0)
        return Convolution_mips::create_pipeline(opt);

    const int inch = weight_data_size / num_output / maxk;
    if (inch <= 0 || inch * maxk * num_output != weight_data_size)
        return Convolution_mips::create_pipeline(opt);

    // Which shapes the COP2 kernels beat the generic loop on was
    // measured end to end on PCSX2, one change at a time (see
    // docs/how-to-build/ps2.md):
    //   * pointwise (maxk == 1, inch >= 4): the output-stationary kernel.
    //     It wins on every mobilenet pointwise layer (1197 -> 502 ms) but
    //     loses on squeezenet's fire "squeeze" 1x1, and no output/input
    //     channel-ratio threshold separates the two without also dropping a
    //     winning mobilenet layer.  The kernel reuses each weight load across
    //     8 output pixels, which recovers part of that loss (4154 -> 3670 ms).
    //   * 3x3 stride 1: the phase-aligned weight-stationary kernel.  Still
    //     under evaluation; gated by NCNN_PS2_WST_PA until it measures faster
    //     than Convolution_mips.
    //   * everything else stays on Convolution_mips.
    if (maxk == 1)
    {
        if (inch < 4)
            return Convolution_mips::create_pipeline(opt);
    }
#if defined(NCNN_PS2_WST_PA)
    else if (maxk == 9 && stride_w == 1 && dilation_w == 1 && dilation_h == 1 && inch >= 4)
    {
    }
#endif
    else
    {
        return Convolution_mips::create_pipeline(opt);
    }

    const int outch4 = (num_output + 3) / 4 * 4;

#if defined(NCNN_PS2_WST_PA)
    // phase-aligned weight-stationary kernel: weights packed as
    // [k][q][p/4].  Channel q is padded up to a multiple of 4 (the padding is
    // zero), so four consecutive channels of one tap are four consecutive
    // quadwords and the unrolled q loop loads them as sequential cache lines.
    if (maxk == 9 && stride_w == 1 && dilation_w == 1 && dilation_h == 1)
    {
        const int inch4 = (inch + 3) / 4 * 4;

        weight_data_ps2_wst.create((size_t)maxk * inch4 * outch4, (size_t)4u, 1);
        if (weight_data_ps2_wst.empty())
            return -100;

        const float* kptr = weight_data;
        float* wtm = weight_data_ps2_wst;
        memset(wtm, 0, (size_t)maxk * inch4 * outch4 * sizeof(float));

        // src = outch-inch-maxk (weight_data[p][q][k])
        for (int k = 0; k < maxk; k++)
            for (int q = 0; q < inch; q++)
                for (int p = 0; p < num_output; p++)
                    wtm[((size_t)k * inch4 + q) * outch4 + p] = kptr[(size_t)p * inch * maxk + (size_t)q * maxk + k];

        weight_data.release();

        return 0;
    }
#endif

    // The weight-stationary kernel (maxk == 9, stride_w == 1) is kept
    // here but is currently unreachable: the gate above sends every non-1x1
    // layer to Convolution_mips, because the kernel measured slower than the
    // generic loop on every 3x3 shape.  It stays compiled so it can be
    // re-enabled per shape once it beats that loop.
    if (maxk == 9 && stride_w == 1)
    {
        weight_data_ps2_wst.create((size_t)maxk * inch * outch4, (size_t)4u, 1);
        if (weight_data_ps2_wst.empty())
            return -100;

        const float* kptr = weight_data;
        float* wtm = weight_data_ps2_wst;

        // src = outch-inch-maxk  (weight_data[p][q][k])
        // dst = maxk-inch-outch4 (weight_data_ps2_wst[k][q][p])
        memset(wtm, 0, (size_t)maxk * inch * outch4 * sizeof(float));
        for (int k = 0; k < maxk; k++)
        {
            for (int q = 0; q < inch; q++)
            {
                float* w = wtm + ((size_t)k * inch + q) * outch4;
                for (int p = 0; p < num_output; p++)
                    w[p] = kptr[(size_t)p * inch * maxk + (size_t)q * maxk + k];
            }
        }

        // The transform is a full-size copy of the weights.  PS2 has only
        // 32MB of RAM, so the original blob has to go once it is no longer
        // needed.  From here on forward() never calls the generic
        // implementation.
        weight_data.release();

        return 0;
    }

    // lqc2/sqc2 require quadword alignment, so the channel axis is padded to
    // a multiple of 4 and the padding lanes are zeroed.  The same padding is
    // applied to the interleaved input in forward().
    const int inch4 = (inch + 3) / 4 * 4;

    // persistent weights always live in the default allocator
    weight_data_ps2.create(inch4, maxk, outch4, (size_t)4u, 1);
    if (weight_data_ps2.empty())
        return -100;

    const float* kptr = weight_data;

    // src = outch-inch-maxk  (weight_data[p][q][k])
    // dst = outch-maxk-inch4 (weight_data_ps2[p][k][q])
    for (int p = 0; p < outch4; p++)
    {
        float* g = weight_data_ps2.channel(p);

        if (p >= num_output)
        {
            memset(g, 0, (size_t)maxk * inch4 * sizeof(float));
            continue;
        }

        for (int k = 0; k < maxk; k++)
        {
            float* gk = g + (size_t)k * inch4;
            const float* kk = kptr + (size_t)p * inch * maxk + k;

            for (int q = 0; q < inch; q++)
                gk[q] = kk[(size_t)q * maxk];

            for (int q = inch; q < inch4; q++)
                gk[q] = 0.f;
        }
    }

    // The transform is a full-size copy of the weights.  PS2 has only 32MB of
    // RAM, so the original blob has to go once it is no longer needed.
    // From here on forward() never calls the generic implementation.
    weight_data.release();

    return 0;
}

int Convolution_mips_ps2::destroy_pipeline(const Option& opt)
{
    weight_data_ps2.release();
    weight_data_ps2_wst.release();
    return Convolution_mips::destroy_pipeline(opt);
}

int Convolution_mips_ps2::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // dynamic weight, int8, or nothing transformed in create_pipeline: the
    // original weight blob is still around and the generic path handles it
    if (weight_data_ps2.empty() && weight_data_ps2_wst.empty())
        return Convolution_mips::forward(bottom_blob, top_blob, opt);



    const bool wst = !weight_data_ps2_wst.empty();

    // From here on weight_data has been released in create_pipeline, so this
    // layer must own the computation.  support_packing / bf16 / int8 are all
    // disabled, so a packed or non-fp32 input cannot reach this layer through
    // a normal ncnn graph.

    // flattened blob with 1x1 kernel: the deprecated InnerProduct compatibility
    // path.  Run it directly instead of falling back.
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        const int inch = bottom_blob.w * bottom_blob.elempack;
        if (bottom_blob.elempack != 1 || inch != weight_data_size / num_output)
            return -1;

        const int outch = num_output;
        const int outch4 = (outch + 3) / 4 * 4;

        top_blob.create(outch, (size_t)4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* x = bottom_blob;

        if (wst)
        {
            // weight_data_ps2_wst is [1][inch][outch4]: weight of output
            // channel p for input channel q is at wtm[q * outch4 + p]
            const float* wtm = weight_data_ps2_wst;

            for (int p = 0; p < outch; p++)
            {
                float sum = bias_data.empty() ? 0.f : ((const float*)bias_data)[p];

                int q = 0;
                for (; q + 3 < inch; q += 4)
                    sum += wtm[(size_t)q * outch4 + p] * x[q]
                           + wtm[(size_t)(q + 1) * outch4 + p] * x[q + 1]
                           + wtm[(size_t)(q + 2) * outch4 + p] * x[q + 2]
                           + wtm[(size_t)(q + 3) * outch4 + p] * x[q + 3];
                for (; q < inch; q++)
                    sum += wtm[(size_t)q * outch4 + p] * x[q];

                top_blob[p] = activation_ss(sum, activation_type, activation_params);
            }

            return 0;
        }

        // weight_data_ps2 is [outch][1][inch4], which is already the
        // InnerProduct weight layout.
        const int wch = (inch + 3) / 4 * 4;

        const float* wtm = weight_data_ps2;

        for (int p = 0; p < outch; p++)
        {
            const float* w = wtm + (size_t)p * wch;
            float sum = bias_data.empty() ? 0.f : ((const float*)bias_data)[p];

            int q = 0;
            for (; q + 3 < inch; q += 4)
                sum += w[q] * x[q] + w[q + 1] * x[q + 1] + w[q + 2] * x[q + 2] + w[q + 3] * x[q + 3];
            for (; q < inch; q++)
                sum += w[q] * x[q];

            top_blob[p] = activation_ss(sum, activation_type, activation_params);
        }

        return 0;
    }

    // unsupported layouts / configurations
    if (bottom_blob.dims != 3)
        return -1;
    if (bottom_blob.elempack != 1 || bottom_blob.elemsize != (size_t)4u)
        return -1;

    const int maxk = kernel_w * kernel_h;
    const int inch = bottom_blob.c;
    if (maxk <= 0 || num_output <= 0 || inch != weight_data_size / num_output / maxk)
        return -100;

    const int outch = num_output;
    const int inch4 = (inch + 3) / 4 * 4;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int bw = bottom_blob_bordered.w;
    const int bh = bottom_blob_bordered.h;

    const int outw = (bw - kernel_extent_w) / stride_w + 1;
    const int outh = (bh - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, outch, (size_t)4u, 1, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // kernel offsets inside one channel plane, and the same offsets scaled to
    // bytes of a whole interleaved pixel row
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = bw * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    const bool bias_term = !bias_data.empty();

#if defined(NCNN_PS2_WST_PA)
    if (wst)
    {
        // four phase-shifted copies of the bordered input.  Copy m is
        // quadword aligned and holds the blob shifted down by m floats
        // (copy[u] = blob[u + m]), so a tap whose flat offset is m mod 4 can be
        // read with a plain aligned lqc2.  cstep is a multiple of 4 (ncnn
        // aligns planes to 16 bytes), which keeps every channel base aligned.
        // The tail covers the few floats a window can read past a plane, and
        // inch4 rather than inch so the channel tail the kernel pads to is
        // backed by zeros.
        const int cstep = (int)bottom_blob_bordered.cstep;
        const int tail = 8;

        Mat shifted[4];
        const float* phase[4];
        for (int m = 0; m < 4; m++)
        {
            shifted[m].create(cstep * inch4 + tail, (size_t)4u, 1, opt.workspace_allocator);
            if (shifted[m].empty())
                return -100;

            float* dst = shifted[m];
            memset(dst, 0, (size_t)(cstep * inch4 + tail) * sizeof(float));
            for (int q = 0; q < inch; q++)
                memcpy(dst + (size_t)q * cstep,
                       (const float*)bottom_blob_bordered.channel(q) + m,
                       (size_t)(cstep - m) * sizeof(float));

            phase[m] = dst;
        }

        const float* biasptr = bias_term ? (const float*)bias_data : (const float*)0;

        convolution_ps2_wst_pa(phase, cstep,
                               (float*)top_blob, (int)top_blob.cstep,
                               weight_data_ps2_wst, biasptr,
                               inch, outch, maxk, outw, outh, bw, stride_h,
                               space_ofs,
                               activation_type, activation_params);

        return 0;
    }
#else
    if (wst)
    {
        // weight-stationary kernel: channel-major bordered input, no
        // pixel-major transpose needed
        const float* biasptr = bias_term ? (const float*)bias_data : (const float*)0;

        convolution_ps2_wst((const float*)bottom_blob_bordered, (int)bottom_blob_bordered.cstep,
                             (float*)top_blob, (int)top_blob.cstep,
                             weight_data_ps2_wst, biasptr,
                             inch, outch, maxk, outw, outh, bw, stride_h,
                             space_ofs, false,
                             activation_type, activation_params);

        return 0;
    }
#endif

    std::vector<int> _space_ofs_bytes(maxk);
    for (int k = 0; k < maxk; k++)
        _space_ofs_bytes[k] = space_ofs[k] * inch4 * (int)sizeof(float);

    // interleave the padded input to pixel-major [pixel][inch4]
    const int size = bw * bh;

    Mat bottom_packed;
    bottom_packed.create(inch4, size, (size_t)4u, 1, opt.workspace_allocator);
    if (bottom_packed.empty())
        return -100;

    {
        float* pk = bottom_packed;

        for (int q = 0; q < inch; q++)
        {
            const float* sptr = bottom_blob_bordered.channel(q);
            float* dptr = pk + q;
            for (int t = 0; t < size; t++)
            {
                *dptr = sptr[t];
                dptr += inch4;
            }
        }

        for (int q = inch; q < inch4; q++)
        {
            float* dptr = pk + q;
            for (int t = 0; t < size; t++)
            {
                *dptr = 0.f;
                dptr += inch4;
            }
        }
    }

    // SPR is a single global buffer shared by the whole EE: it can only be
    // used when no other thread runs the same kernel concurrently.  ncnn on
    // PS2 is built with NCNN_THREADS=OFF (opt.num_threads == 1); for safety
    // fall back to the main-memory path whenever more threads are configured.
    // Groups whose weights exceed 16KB fall back per group inside the kernel.
    const bool use_spr = opt.num_threads <= 1;

    convolution_ps2_pack4(bottom_packed, top_blob, weight_data_ps2, bias_data,
                          inch4, outch, maxk, outw, outh, bw,
                          stride_w, stride_h, &_space_ofs_bytes[0],
                          use_spr,
                          activation_type, activation_params);

    return 0;
}

} // namespace ncnn
