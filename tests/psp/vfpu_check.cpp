// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// Numeric check for the PSP VFPU fp32 dot product, run under PPSSPP.
//
// The reference is the same accumulation the scalar kernels do, computed here
// in plain C, so a mismatch is the VFPU path and not a difference of
// algorithms. Values are printed rather than only counted: the sceKernelPrintf
// log is the only output channel.

#include "net.h"

#include "cpu.h"
#include "layer/gemm.h"
#include "layer/mips/innerproduct_mips.h"
#include "layer/mips/convolution_mips.h"

#include <math.h>
#include <stdio.h>

static int g_fail = 0;

static void fill_pattern(float* p, int n, float scale)
{
    // mixed signs and magnitudes, including values that do not divide powers of
    // two, so the reduction order shows up in the low bits
    for (int i = 0; i < n; i++)
        p[i] = scale * ((i % 7) - 3) * 0.5f + ((i % 5) - 2) * 0.125f;
}

static void report(const char* name, const float* got, const float* ref, int n)
{
    // An empty Mat converts to a null pointer. Printing that is a failed
    // check, not a reason to fault the emulator.
    if (!got || !ref)
    {
        g_fail++;
        fprintf(stderr, "%-14s FAIL  n=%d empty output\n", name, n);
        return;
    }

    float max_abs = 0.f;
    int max_at = 0;
    for (int i = 0; i < n; i++)
    {
        float d = got[i] - ref[i];
        if (d < 0.f)
            d = -d;
        if (d > max_abs)
        {
            max_abs = d;
            max_at = i;
        }
    }

    // vdot is an exact sum of four products; the only extra rounding against
    // the scalar loop is the tree of partial sums, so anything past a few ulps
    // is a wrong answer rather than a reduction-order difference
    const int pass = max_abs < 1e-4f;
    if (!pass)
        g_fail++;

    fprintf(stderr, "%-14s %s  n=%d max_abs=%.3e at %d (got %.6f ref %.6f)\n",
            name, pass ? "PASS" : "FAIL", n, max_abs, max_at, got[max_at], ref[max_at]);
}

static void reference_gemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float sum = 0.f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[j * K + k];
            C[i * N + j] = sum;
        }
}

static void check_gemm(int M, int N, int K)
{
    ncnn::Mat A(K, M);
    ncnn::Mat B(K, N);
    fill_pattern(A, M * K, 0.25f);
    fill_pattern(B, N * K, -0.5f);

    std::vector<ncnn::Mat> bottom(2);
    bottom[0] = A;
    bottom[1] = B;
    std::vector<ncnn::Mat> top(1);

    // These members are only set by load_param. A bare constructor leaves them
    // uninitialised, and an alpha of 0 zeroes the whole product.
    ncnn::Gemm gemm;
    gemm.alpha = 1.f;
    gemm.beta = 1.f;
    gemm.transA = 0;
    gemm.transB = 1;
    gemm.constantA = 0;
    gemm.constantB = 0;
    gemm.constantC = 0;
    gemm.output_transpose = 0;
    gemm.output_N1M = 0;
    gemm.quantize_term = 0;

    ncnn::Option opt;
    gemm.forward(bottom, top, opt);

    ncnn::Mat ref(N, M);
    reference_gemm(A, B, ref, M, N, K);

    char name[32];
    snprintf(name, sizeof(name), "gemm %dx%dx%d", M, N, K);
    report(name, top[0], ref, M * N);
}

static void reference_innerproduct(const float* in, const float* w, const float* bias, float* out, int num_input, int num_output)
{
    for (int p = 0; p < num_output; p++)
    {
        float sum = bias ? bias[p] : 0.f;
        for (int i = 0; i < num_input; i++)
            sum += in[i] * w[p * num_input + i];
        out[p] = sum > 0.f ? sum : 0.f;
    }
}

static void check_innerproduct(int num_input, int num_output)
{
    ncnn::Mat in(num_input);
    fill_pattern(in, num_input, 0.5f);

    const int wsize = num_input * num_output;
    ncnn::Mat weight(wsize);
    ncnn::Mat bias(num_output);
    fill_pattern(weight, wsize, -0.25f);
    fill_pattern(bias, num_output, 0.125f);

    // load_param is what normally zeroes these. A bare constructor does not,
    // and a nonzero int8_scale_term with the default use_int8_inference sends
    // the layer down the quant path, which has no scales and returns nothing.
    ncnn::InnerProduct_mips layer;
    layer.num_output = num_output;
    layer.bias_term = 1;
    layer.weight_data_size = wsize;
    layer.int8_scale_term = 0;
    layer.activation_type = 1;
    layer.weight_data = weight;
    layer.bias_data = bias;

    ncnn::Option opt;
    opt.use_int8_inference = false;
    layer.create_pipeline(opt);

    ncnn::Mat out;
    layer.forward(in, out, opt);

    ncnn::Mat ref(num_output);
    reference_innerproduct(in, weight, bias, ref, num_input, num_output);

    char name[32];
    snprintf(name, sizeof(name), "innerproduct %d->%d", num_input, num_output);
    report(name, out, ref, num_output);

    layer.destroy_pipeline(opt);
}

static void reference_convolution(const float* in, const float* w, const float* bias, float* out, int inw, int inh, int inch, int outch, int kw, int kh)
{
    const int outw = inw - kw + 1;
    const int outh = inh - kh + 1;
    const int maxk = kw * kh;

    for (int oc = 0; oc < outch; oc++)
        for (int oy = 0; oy < outh; oy++)
            for (int ox = 0; ox < outw; ox++)
            {
                float sum = bias ? bias[oc] : 0.f;
                for (int ic = 0; ic < inch; ic++)
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++)
                        {
                            float v = in[ic * inw * inh + (oy + ky) * inw + (ox + kx)];
                            float wt = w[((oc * inch + ic) * kh + ky) * kw + kx];
                            sum += v * wt;
                        }
                out[(oc * outh + oy) * outw + ox] = sum > 0.f ? sum : 0.f;
            }
    (void)maxk;
}

static void check_convolution(int inw, int inh, int inch, int outch, int kw, int kh)
{
    ncnn::Mat in(inw, inh, inch);
    fill_pattern(in, inw * inh * inch, 0.25f);

    const int maxk = kw * kh;
    const int wsize = inch * outch * maxk;
    ncnn::Mat weight(wsize);
    ncnn::Mat bias(outch);
    fill_pattern(weight, wsize, -0.125f);
    fill_pattern(bias, outch, 0.0625f);

    // Same as Gemm: these members are only written by load_param. Garbage
    // padding allocates a border the heap cannot hold, and a garbage
    // int8_scale_term or dynamic_weight skips the fp32 kernel entirely.
    ncnn::Convolution_mips layer;
    layer.num_output = outch;
    layer.kernel_w = kw;
    layer.kernel_h = kh;
    layer.dilation_w = 1;
    layer.dilation_h = 1;
    layer.stride_w = 1;
    layer.stride_h = 1;
    layer.pad_left = 0;
    layer.pad_right = 0;
    layer.pad_top = 0;
    layer.pad_bottom = 0;
    layer.pad_value = 0.f;
    layer.bias_term = 1;
    layer.weight_data_size = wsize;
    layer.int8_scale_term = 0;
    layer.dynamic_weight = 0;
    layer.activation_type = 1;
    layer.weight_data = weight;
    layer.bias_data = bias;

    ncnn::Option opt;
    opt.use_winograd_convolution = false;
    opt.use_winograd23_convolution = false;
    opt.use_winograd43_convolution = false;
    opt.use_winograd63_convolution = false;
    opt.use_sgemm_convolution = false;
    opt.use_int8_inference = false;
    opt.use_packing_layout = false;
    opt.num_threads = 1;
    layer.create_pipeline(opt);

    ncnn::Mat out;
    layer.forward(in, out, opt);

    const int outw = inw - kw + 1;
    const int outh = inh - kh + 1;
    ncnn::Mat ref(outw, outh, outch);
    reference_convolution(in, weight, bias, ref, inw, inh, inch, outch, kw, kh);

    char name[40];
    snprintf(name, sizeof(name), "conv %dx%d i%d o%d", kw, kh, inch, outch);
    report(name, out, ref, outw * outh * outch);

    layer.destroy_pipeline(opt);
}

int main()
{
    fprintf(stderr, "ncnn PSP VFPU numeric check\n");

    // K below, at and above the 16-wide group, plus a non-multiple tail
    check_gemm(3, 5, 1);
    check_gemm(4, 4, 4);
    check_gemm(3, 7, 15);
    check_gemm(5, 6, 17);
    check_gemm(4, 4, 64);

    // both the 2-D gemm form and the flattened remainder, with tails
    check_innerproduct(7, 5);
    check_innerproduct(16, 3);
    check_innerproduct(63, 8);

    // outch < 2 is the VFPU kernel; the 3x3 one also has a tail channel
    check_convolution(8, 8, 4, 1, 1, 1);
    check_convolution(8, 8, 7, 1, 3, 3);
    check_convolution(6, 6, 16, 1, 1, 1);

    fprintf(stderr, "%s\n", g_fail ? "RESULT FAIL" : "RESULT PASS");
    return g_fail ? 1 : 0;
}
