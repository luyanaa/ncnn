// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "unaryop_mips_psp.h"

#include <math.h>

#include "psp_vfpu.h"

namespace ncnn {

UnaryOp_mips_psp::UnaryOp_mips_psp()
{
    support_packing = true;
    support_any_packing = true;
}

// vlog2 is an approximation, about 3e-5 absolute. log(x) = log2(x) * ln(2),
// and ln(2) is taken from the same constant table the scalar libm uses, so the
// error stays the VFPU's and is not added to by the scale factor.
static const float psp_ln2_4[4] __attribute__((aligned(16))) = {0.69314718f, 0.69314718f, 0.69314718f, 0.69314718f};

// vexp2 takes a base-2 exponent, so exp(x) = exp2(x / ln(2)).
static const float psp_inv_ln2_4[4] __attribute__((aligned(16))) = {1.44269504f, 1.44269504f, 1.44269504f, 1.44269504f};

static void unary_quad(float* ptr, int op_type)
{
    switch (op_type)
    {
    case UnaryOp::Operation_ABS:
        PSP_VFPU_ABS(ptr, ptr);
        break;
    case UnaryOp::Operation_NEG:
        PSP_VFPU_NEG(ptr, ptr);
        break;
    case UnaryOp::Operation_SQRT:
        PSP_VFPU_UNARY("vsqrt.q", ptr, ptr);
        break;
    case UnaryOp::Operation_EXP:
        PSP_VFPU_BINARY("vmul.q", ptr, psp_inv_ln2_4, ptr);
        PSP_VFPU_UNARY("vexp2.q", ptr, ptr);
        break;
    case UnaryOp::Operation_LOG:
        PSP_VFPU_UNARY("vlog2.q", ptr, ptr);
        PSP_VFPU_BINARY("vmul.q", ptr, psp_ln2_4, ptr);
        break;
    default:
        break;
    }
}

static void unary_run(float* ptr, int size, int op_type)
{
    int i = 0;
    if (psp_vfpu_aligned(ptr))
    {
        for (; i + 3 < size; i += 4)
        {
            unary_quad(ptr, op_type);
            ptr += 4;
        }
    }

    for (; i < size; i++)
    {
        switch (op_type)
        {
        case UnaryOp::Operation_ABS:
            *ptr = fabsf(*ptr);
            break;
        case UnaryOp::Operation_NEG:
            *ptr = -*ptr;
            break;
        case UnaryOp::Operation_SQRT:
            *ptr = sqrtf(*ptr);
            break;
        case UnaryOp::Operation_EXP:
            *ptr = expf(*ptr);
            break;
        case UnaryOp::Operation_LOG:
            *ptr = logf(*ptr);
            break;
        default:
            break;
        }
        ptr++;
    }
}

int UnaryOp_mips_psp::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    switch (op_type)
    {
    case Operation_ABS:
    case Operation_NEG:
    case Operation_SQRT:
    case Operation_EXP:
    case Operation_LOG:
        break;
    default:
        // floor, ceil, trig and the rest have no exact VFPU form, or only an
        // approximation too coarse to match the scalar reference.
        return UnaryOp::forward_inplace(bottom_top_blob, opt);
    }

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int size = w * h * d * bottom_top_blob.elempack;

    for (int q = 0; q < channels; q++)
        unary_run(bottom_top_blob.channel(q), size, op_type);

    return 0;
}

} // namespace ncnn
