// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "binaryop_mips_ps2.h"

#include <algorithm>
#include <float.h>
#include <math.h>

#include "cpu.h"

namespace ncnn {

static int get_reverse_op_type(int _op_type)
{
    if (_op_type == BinaryOp::Operation_SUB) return BinaryOp::Operation_RSUB;
    if (_op_type == BinaryOp::Operation_DIV) return BinaryOp::Operation_RDIV;
    if (_op_type == BinaryOp::Operation_POW) return BinaryOp::Operation_RPOW;
    if (_op_type == BinaryOp::Operation_RSUB) return BinaryOp::Operation_SUB;
    if (_op_type == BinaryOp::Operation_RDIV) return BinaryOp::Operation_DIV;
    if (_op_type == BinaryOp::Operation_RPOW) return BinaryOp::Operation_POW;
    if (_op_type == BinaryOp::Operation_ATAN2) return BinaryOp::Operation_RATAN2;
    if (_op_type == BinaryOp::Operation_RATAN2) return BinaryOp::Operation_ATAN2;
    if (_op_type == BinaryOp::Operation_FMOD) return BinaryOp::Operation_RFMOD;
    if (_op_type == BinaryOp::Operation_RFMOD) return BinaryOp::Operation_FMOD;
    if (_op_type == BinaryOp::Operation_FLOOR_DIVIDE) return BinaryOp::Operation_RFLOOR_DIVIDE;
    if (_op_type == BinaryOp::Operation_RFLOOR_DIVIDE) return BinaryOp::Operation_FLOOR_DIVIDE;
    if (_op_type == BinaryOp::Operation_REMAINDER) return BinaryOp::Operation_RREMAINDER;
    if (_op_type == BinaryOp::Operation_RREMAINDER) return BinaryOp::Operation_REMAINDER;
    return _op_type;
}

BinaryOp_mips_ps2::BinaryOp_mips_ps2()
{
    support_packing = true;
}

static float binary_op_scalar(int op_type, float x, float y)
{
    switch (op_type)
    {
    case BinaryOp::Operation_ADD:
        return x + y;
    case BinaryOp::Operation_SUB:
        return x - y;
    case BinaryOp::Operation_MUL:
        return x * y;
    case BinaryOp::Operation_DIV:
        return x / y;
    case BinaryOp::Operation_MAX:
        return std::max(x, y);
    case BinaryOp::Operation_MIN:
        return std::min(x, y);
    case BinaryOp::Operation_POW:
        return (float)powf(x, y);
    case BinaryOp::Operation_RSUB:
        return y - x;
    case BinaryOp::Operation_RDIV:
        return y / x;
    case BinaryOp::Operation_RPOW:
        return (float)powf(y, x);
    case BinaryOp::Operation_ATAN2:
        return (float)atan2f(x, y);
    case BinaryOp::Operation_RATAN2:
        return (float)atan2f(y, x);
    case BinaryOp::Operation_FMOD:
        return (float)fmodf(x, y);
    case BinaryOp::Operation_RFMOD:
        return (float)fmodf(y, x);
    case BinaryOp::Operation_LOGADDEXP:
        return (float)(std::max(x, y) + log1pf(expf(std::min(x, y) - std::max(x, y))));
    case BinaryOp::Operation_FLOOR_DIVIDE:
        return (float)floorf(x / y);
    case BinaryOp::Operation_RFLOOR_DIVIDE:
        return (float)floorf(y / x);
    case BinaryOp::Operation_REMAINDER:
        return (float)remainderf(x, y);
    case BinaryOp::Operation_RREMAINDER:
        return (float)remainderf(y, x);
    default:
        return 0.f;
    }
}

// ops that map directly onto COP2 vector instructions
static bool binary_op_cop2_supported(int op_type)
{
    return op_type == BinaryOp::Operation_ADD
           || op_type == BinaryOp::Operation_SUB
           || op_type == BinaryOp::Operation_MUL
           || op_type == BinaryOp::Operation_MAX
           || op_type == BinaryOp::Operation_MIN;
}

#define COP2_BINARY_OP(op, pa, pb, out)  \
    __asm__ __volatile__(                \
        "lqc2 $vf4, 0(%0)\n\t"           \
        "lqc2 $vf5, 0(%1)\n\t" op "\n\t" \
        "sqc2 $vf6, 0(%2)\n\t"           \
        :                                \
        : "r"(pa), "r"(pb), "r"(out)     \
        : "memory")

static void binary_op_vector_cop2(const float* pa, const float* pb, float* out, int size, int op_type)
{
    // size is a multiple of 4 when elempack == 4, but guard the tail anyway
    int i = 0;
    for (; i + 3 < size; i += 4)
    {
        switch (op_type)
        {
        case BinaryOp::Operation_ADD:
            COP2_BINARY_OP("vadd.xyzw $vf6, $vf4, $vf5", pa, pb, out);
            break;
        case BinaryOp::Operation_SUB:
            COP2_BINARY_OP("vsub.xyzw $vf6, $vf4, $vf5", pa, pb, out);
            break;
        case BinaryOp::Operation_MUL:
            COP2_BINARY_OP("vmul.xyzw $vf6, $vf4, $vf5", pa, pb, out);
            break;
        case BinaryOp::Operation_MAX:
            COP2_BINARY_OP("vmax.xyzw $vf6, $vf4, $vf5", pa, pb, out);
            break;
        case BinaryOp::Operation_MIN:
            COP2_BINARY_OP("vmini.xyzw $vf6, $vf4, $vf5", pa, pb, out);
            break;
        default:
            break;
        }

        pa += 4;
        pb += 4;
        out += 4;
    }

    for (; i < size; i++)
    {
        *out = binary_op_scalar(op_type, *pa, *pb);
        pa += 1;
        pb += 1;
        out += 1;
    }
}

static void binary_op_vector_scalar(const float* pa, const float* pb, float* out, int size, int op_type)
{
    int i = 0;
    for (; i < size; i++)
    {
        *out = binary_op_scalar(op_type, *pa, *pb);
        pa += 1;
        pb += 1;
        out += 1;
    }
}

static void binary_op_no_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    (void)opt;
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;
    const int elempack = a.elempack;

    for (int q = 0; q < channels; q++)
    {
        const float* pa = a.channel(q);
        const float* pb = b.channel(q);
        float* out = c.channel(q);

        // COP2 requires 16-byte aligned addresses; ncnn allocator guarantees
        // channel pointers are aligned, and size % 4 == 0 when elempack == 4
        if (elempack == 4 && binary_op_cop2_supported(op_type))
        {
            binary_op_vector_cop2(pa, pb, out, size, op_type);
        }
        else
        {
            binary_op_vector_scalar(pa, pb, out, size, op_type);
        }
    }
}

static void binary_op_scalar_broadcast(const Mat& a, float b, Mat& c, int op_type, const Option& opt)
{
    (void)opt;
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;
    const int elempack = a.elempack;

    // broadcast a scalar into a 16-byte aligned quad for COP2
    const float b4buf[4] __attribute__((aligned(16))) = {b, b, b, b};

    for (int q = 0; q < channels; q++)
    {
        const float* pa = a.channel(q);
        float* out = c.channel(q);

        if (elempack == 4 && binary_op_cop2_supported(op_type))
        {
            binary_op_vector_cop2(pa, b4buf, out, size, op_type);
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                *out = binary_op_scalar(op_type, *pa, b);
                pa += 1;
                out += 1;
            }
        }
    }
}

static void binary_op_broadcast_generic(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    (void)opt;
    const int dims = c.dims;

    if (dims == 2)
    {
        const int h = c.h;

        for (int y = 0; y < h; y++)
        {
            const int y0 = std::min(y, a.h - 1);
            const int y1 = std::min(y, b.h - 1);

            const float* pa = a.row<const float>(y0);
            const float* pb = b.row<const float>(y1);
            float* out = c.row<float>(y);

            for (int i = 0; i < c.w * c.elempack; i++)
            {
                *out = binary_op_scalar(op_type, *pa, *pb);
                pa += 1;
                pb += 1;
                out += 1;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int channels = c.c;

        for (int q = 0; q < channels; q++)
        {
            const int q0 = std::min(q, a.c - 1);
            const int q1 = std::min(q, b.c - 1);

            if (b.d * b.h * b.w == 1)
            {
                const float* pa = a.channel(q0);
                const float* pb = b.channel(q1);
                float* out = c.channel(q);

                for (int i = 0; i < a.w * a.h * a.d * a.elempack; i++)
                {
                    *out = binary_op_scalar(op_type, *pa, *pb);
                    pa += 1;
                    pb += 1;
                    out += 1;
                }
                continue;
            }

            if (b.h * b.w == 1)
            {
                for (int z = 0; z < c.d; z++)
                {
                    const int z0 = std::min(z, a.d - 1);
                    const int z1 = std::min(z, b.d - 1);

                    const float* pa = a.channel(q0).depth(z0);
                    const float* pb = b.channel(q1).depth(z1);
                    float* out = c.channel(q).depth(z);

                    for (int i = 0; i < a.w * a.h * a.elempack; i++)
                    {
                        *out = binary_op_scalar(op_type, *pa, *pb);
                        pa += 1;
                        pb += 1;
                        out += 1;
                    }
                }
                continue;
            }

            for (int z = 0; z < c.d; z++)
            {
                const int z0 = std::min(z, a.d - 1);
                const int z1 = std::min(z, b.d - 1);

                for (int y = 0; y < c.h; y++)
                {
                    const int y0 = std::min(y, a.h - 1);
                    const int y1 = std::min(y, b.h - 1);

                    const float* pa = a.channel(q0).depth(z0).row<const float>(y0);
                    const float* pb = b.channel(q1).depth(z1).row<const float>(y1);
                    float* out = c.channel(q).depth(z).row<float>(y);

                    for (int i = 0; i < a.w * a.elempack; i++)
                    {
                        *out = binary_op_scalar(op_type, *pa, *pb);
                        pa += 1;
                        pb += 1;
                        out += 1;
                    }
                }
            }
        }
    }
}

int BinaryOp_mips_ps2::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A = bottom_blobs[0];
    const Mat& B = bottom_blobs[1];
    const int outdims = std::max(A.dims, B.dims);

    Mat A2 = A;
    Mat B2 = B;
    if (A.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (A.w * A.elempack == B.h * B.elempack)
                A2 = A.reshape(1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 2;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A.cstep * A.elempack;
            }
        }
        if (outdims == 3 && A.dims == 1)
        {
            if (A.w * A.elempack == B.c * B.elempack)
                A2 = A.reshape(1, 1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 3;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A.cstep * A.elempack;
            }
        }
        if (outdims == 3 && A.dims == 2)
            A2 = A.reshape(1, A.w, A.h, opt.workspace_allocator);
        if (outdims == 4 && A.dims == 1)
        {
            if (A.w * A.elempack == B.c * B.elempack)
                A2 = A.reshape(1, 1, 1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 4;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A.cstep * A.elempack;
            }
        }
        if (outdims == 4 && A.dims == 2)
            A2 = A.reshape(1, 1, A.w, A.h, opt.workspace_allocator);
        if (outdims == 4 && A.dims == 3)
            A2 = A.reshape(1, A.w, A.h, A.c, opt.workspace_allocator);
    }
    if (B.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (B.w * B.elempack == A.h * A.elempack)
                B2 = B.reshape(1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 2;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B.cstep * B.elempack;
            }
        }
        if (outdims == 3 && B.dims == 1)
        {
            if (B.w * B.elempack == A.c * A.elempack)
                B2 = B.reshape(1, 1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 3;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B.cstep * B.elempack;
            }
        }
        if (outdims == 3 && B.dims == 2)
            B2 = B.reshape(1, B.w, B.h, opt.workspace_allocator);
        if (outdims == 4 && B.dims == 1)
        {
            if (B.w * B.elempack == A.c * A.elempack)
                B2 = B.reshape(1, 1, 1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 4;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B.cstep * B.elempack;
            }
        }
        if (outdims == 4 && B.dims == 2)
            B2 = B.reshape(1, 1, B.w, B.h, opt.workspace_allocator);
        if (outdims == 4 && B.dims == 3)
            B2 = B.reshape(1, B.w, B.h, B.c, opt.workspace_allocator);
    }

    const int outw = std::max(A2.w, B2.w);
    const int outh = std::max(A2.h, B2.h);
    const int outd = std::max(A2.d, B2.d);
    const int outc = std::max(A2.c, B2.c);
    const size_t out_elemsize = std::max(A2.elemsize, B2.elemsize);
    const int out_elempack = std::max(A2.elempack, B2.elempack);

    Mat& top_blob = top_blobs[0];
    if (outdims == 1)
    {
        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 2)
    {
        top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 4)
    {
        top_blob.create(outw, outh, outd, outc, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    const bool a_pack_is_lower = A2.elempack < B2.elempack;
    const bool a_pack_is_equal = A2.elempack == B2.elempack;
    const bool a_size_is_lower = A2.w * A2.h * A2.d * A2.c * A2.elempack < B2.w * B2.h * B2.d * B2.c * B2.elempack;

    if (a_pack_is_lower || (a_pack_is_equal && a_size_is_lower))
    {
        // B is larger, apply the reverse op
        if (B2.w * B2.h * B2.d * B2.c * B2.elempack == 1)
            binary_op_scalar_broadcast(A2, ((const float*)B2)[0], top_blob, get_reverse_op_type(op_type), opt);
        else if (A2.dims == B2.dims && A2.w == B2.w && A2.h == B2.h && A2.d == B2.d && A2.c == B2.c && A2.elempack == B2.elempack)
            binary_op_no_broadcast(A2, B2, top_blob, get_reverse_op_type(op_type), opt);
        else
            binary_op_broadcast_generic(B2, A2, top_blob, get_reverse_op_type(op_type), opt);
    }
    else
    {
        if (B2.w * B2.h * B2.d * B2.c * B2.elempack == 1)
            binary_op_scalar_broadcast(A2, ((const float*)B2)[0], top_blob, op_type, opt);
        else if (A2.dims == B2.dims && A2.w == B2.w && A2.h == B2.h && A2.d == B2.d && A2.c == B2.c && A2.elempack == B2.elempack)
            binary_op_no_broadcast(A2, B2, top_blob, op_type, opt);
        else
            binary_op_broadcast_generic(A2, B2, top_blob, op_type, opt);
    }

    return 0;
}

int BinaryOp_mips_ps2::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // inplace form: bottom_top_blob op= b (with_scalar param case)
    binary_op_scalar_broadcast(bottom_top_blob, b, bottom_top_blob, op_type, opt);

    return 0;
}

} // namespace ncnn
