// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef _PS2_MMI_H
#define _PS2_MMI_H

// PlayStation 2 Emotion Engine (Toshiba R5900) EE MMI extension
//
// The EE core has 32 general purpose registers that are 128-bit wide.
// Standard MIPS IV ld/sd only move 64-bit halves; the proprietary lq/sq
// instructions move full 128-bit values. The MMI multimedia instructions
// operate directly on the 128-bit GPRs.
//
// GCC (r5900 toolchain, e.g. GCC 15.2 with -march=r5900) allocates
// vector_size(16) variables to a pair of 64-bit GPRs, so plain "r"
// inline-asm constraints bind correctly.
//
// NOTE on mnemonics: the assembler (GNU binutils) uses short names for the
// compare and saturating arithmetic instructions compared to the EE manual:
//   EE manual          assembler mnemonic
//   PCMPEQB/H/W        PCEQB/H/W
//   PCMPGTB/H/W        PCGTB/H/W
//   PADDUSB/USH/UW     PADDUB/UH/UW
//   PSUBUSB/USH/UW     PSUBUB/UH/UW
// There is no PMADDHW / PACKSSWH / PACKSSH B / PACKUSHB / PANDI / PORI /
// PABSD on the R5900 (those names belong to Loongson MMI / are unencodable);
// narrowing packs are done with PPACB/PPACH/PPACW instead.
//
// R5900 "dual-lane" multiply semantics: PMULTW/PMULTUW/PMULTH write the
// products of the lower half of each operand into rd and the products of the
// upper half into the HI/LO accumulator. The accumulate variants PMADDW/
// PMADDH additionally add their upper products into HI/LO. Wrappers here
// return the rd part; use __mmi_pmfhi/__mmi_pmflo to read the accumulator.

// IMPORTANT -- how to use these wrappers
// -------------------------------------
// This GCC models a `vector_size(16)` value as *two* 64-bit GPRs, while every
// MMI instruction consumes and produces *one* 128-bit GPR.  Two consequences:
//
//  * A wrapper return value only survives if it goes straight back to memory.
//    Passing 128-bit values around in ordinary C variables loses the upper
//    64 bits (the second register of the pair is not tracked by the MMI op).
//    Verified on PCSX2: `c = __mmi_paddw(a, b)` with plain locals gives a
//    correct low half and stale high half.
//  * Therefore use the lq -> op -> sq idiom, keeping everything inside one
//    expression so the compiler never materialises the pair:
//
//        __mmi_sq(dst, __mmi_paddw(__mmi_lq(a), __mmi_lq(b)));
//
//    For anything larger than one operation write a single asm block that
//    does its own lq/ops/sq (this is what the PS2 kernels do).
//
// All 128-bit outputs are declared early-clobber ("=&r") because with a plain
// "=r" GCC may reuse an input register and insert a pair `move`, dropping the
// upper half.

#if defined(_EE) || defined(__PS2__)

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <string.h>

typedef uint8_t uint8x16_t __attribute__((vector_size(16)));
typedef uint16_t uint16x8_t __attribute__((vector_size(16)));
typedef uint32_t uint32x4_t __attribute__((vector_size(16)));

typedef int8_t int8x16_t __attribute__((vector_size(16)));
typedef int16_t int16x8_t __attribute__((vector_size(16)));
typedef int32_t int32x4_t __attribute__((vector_size(16)));

// ----------------------------- load / store -----------------------------
// lq/sq require 16-byte alignment. __mmi_lq/__mmi_sq assume it.
// __mmi_lq_u/__mmi_sq_u fall back to memcpy for potentially unaligned data.

static inline __attribute__((__always_inline__)) uint32x4_t __mmi_lq(const void* p)
{
    uint32x4_t r;
    __asm__ __volatile__("lq %0, 0(%1)"
            : "=&r"(r)
            : "r"(p), "m"(*(const uint32x4_t*)p)
            : "memory");
    return r;
}

static inline __attribute__((__always_inline__)) void __mmi_sq(void* p, uint32x4_t v)
{
    __asm__ __volatile__("sq %1, 0(%2)"
            : "+m"(*(uint32x4_t*)p)
            : "r"(v), "r"(p)
            : "memory");
}

static inline __attribute__((__always_inline__)) uint32x4_t __mmi_lq_u(const void* p)
{
    uint32x4_t r;
    memcpy(&r, p, 16);
    return r;
}

static inline __attribute__((__always_inline__)) void __mmi_sq_u(void* p, uint32x4_t v)
{
    memcpy(p, &v, 16);
}

// ----------------------------- add / sub (plain) -----------------------------
#define __PS2_MMI_BINOP(name, asm_op)                                        \
    static inline __attribute__((__always_inline__)) uint32x4_t name##_u(uint32x4_t a, uint32x4_t b) \
    {                                                                        \
        uint32x4_t r;                                                        \
        __asm__(asm_op " %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));              \
        return r;                                                            \
    }                                                                        \
    static inline __attribute__((__always_inline__)) int32x4_t name##_s(int32x4_t a, int32x4_t b) \
    {                                                                        \
        int32x4_t r;                                                         \
        __asm__(asm_op " %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));              \
        return r;                                                            \
    }

__PS2_MMI_BINOP(__mmi_paddb, "paddb")
__PS2_MMI_BINOP(__mmi_paddh, "paddh")
__PS2_MMI_BINOP(__mmi_paddw, "paddw")
__PS2_MMI_BINOP(__mmi_psubb, "psubb")
__PS2_MMI_BINOP(__mmi_psubh, "psubh")
__PS2_MMI_BINOP(__mmi_psubw, "psubw")

// ----------------------------- saturating add / sub -----------------------------
// assembler names: paddsb/paddsh/paddsw (signed saturating),
// paddub/padduh/padduw (unsigned saturating), likewise for psub
#define __PS2_MMI_SATOP(name, asm_op, T)                                     \
    static inline __attribute__((__always_inline__)) T name(T a, T b)        \
    {                                                                        \
        T r;                                                                 \
        __asm__(asm_op " %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));              \
        return r;                                                            \
    }

__PS2_MMI_SATOP(__mmi_paddsb_s, "paddsb", int32x4_t)
__PS2_MMI_SATOP(__mmi_paddsh_s, "paddsh", int32x4_t)
__PS2_MMI_SATOP(__mmi_paddsw_s, "paddsw", int32x4_t)
__PS2_MMI_SATOP(__mmi_psubsb_s, "psubsb", int32x4_t)
__PS2_MMI_SATOP(__mmi_psubsh_s, "psubsh", int32x4_t)
__PS2_MMI_SATOP(__mmi_psubsw_s, "psubsw", int32x4_t)

__PS2_MMI_SATOP(__mmi_paddub_u, "paddub", uint32x4_t)
__PS2_MMI_SATOP(__mmi_padduh_u, "padduh", uint32x4_t)
__PS2_MMI_SATOP(__mmi_padduw_u, "padduw", uint32x4_t)
__PS2_MMI_SATOP(__mmi_psubub_u, "psubub", uint32x4_t)
__PS2_MMI_SATOP(__mmi_psubuh_u, "psubuh", uint32x4_t)
__PS2_MMI_SATOP(__mmi_psubuw_u, "psubuw", uint32x4_t)

// ----------------------------- multiply / multiply-add -----------------------------
// dual-lane products; see file header comment.
// PMULTW: rd = {rs.w[0]*rt.w[0], rs.w[2]*rt.w[2]}; HI=rs.w[3]*rt.w[3], LO=rs.w[1]*rt.w[1]
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pmultw(int32x4_t a, int32x4_t b)
{
    int32x4_t r;
    __asm__("pmultw %0,%1,%2" : "=r"(r) : "r"(a), "r"(b) : "$hi", "$lo");
    return r;
}

// PMULTUW: same lane layout, unsigned 32x32
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pmultuw(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    __asm__("pmultuw %0,%1,%2" : "=r"(r) : "r"(a), "r"(b) : "$hi", "$lo");
    return r;
}

// PMULTH: rd = 16x16 products of lower halfwords; HI/LO hold upper-half products
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pmulth(int16x8_t a, int16x8_t b)
{
    int32x4_t r;
    __asm__("pmulth %0,%1,%2" : "=r"(r) : "r"(a), "r"(b) : "$hi", "$lo");
    return r;
}

// PMADDW: like PMULTW but upper products accumulate into HI/LO
// PMADDW is an accumulating form: rd = rd + (rs.w[0]*rt.w[0], -, rs.w[2]*rt.w[2], -)
// so the destination has to be an input as well.  The upper halves of each
// product are accumulated into LO/HI (see __mmi_pmfhi/__mmi_pmflo).
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pmaddw(int32x4_t acc, int32x4_t a, int32x4_t b)
{
    __asm__("pmaddw %0,%1,%2" : "+r"(acc) : "r"(a), "r"(b) : "$hi", "$lo");
    return acc;
}

// PMADDH: like PMULTH but upper products accumulate into HI/LO
// PMADDH accumulates into rd and LO/HI per lane (rd takes the even lanes of
// the updated LO/HI sums), so the destination is also an input.
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pmaddh(int32x4_t acc, int16x8_t a, int16x8_t b)
{
    __asm__("pmaddh %0,%1,%2" : "+r"(acc) : "r"(a), "r"(b) : "$hi", "$lo");
    return acc;
}

// PMADDUW: unsigned word multiply-add into HI/LO, rd gets lower product
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pmadduw(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    __asm__("pmadduw %0,%1,%2" : "=&r"(r) : "r"(a), "r"(b) : "$hi", "$lo");
    return r;
}

// accumulator access: PMFHI/PMFLO move the whole 128-bit accumulator into a
// GPR pair (the EE manual specifies 128 bits; reading only 64 loses half of it)
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pmfhi()
{
    uint32x4_t r;
    __asm__("pmfhi %0" : "=&r"(r));
    return r;
}
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pmflo()
{
    uint32x4_t r;
    __asm__("pmflo %0" : "=&r"(r));
    return r;
}

// ----------------------------- pack / unpack / interleave -----------------------------
// PEXTLB/LH/LW: interleave expansion of the low halves of the two source
// operands into the full 128-bit register (im2col / widening primitive).
// PEXTUB/UH/UW: same for the high halves.
// (exact source-lane order: consult the EE Core Instruction Manual)
#define __PS2_MMI_EXT2(name, asm_op, Td, Ta, Tb)                             \
    static inline __attribute__((__always_inline__)) Td name(Ta a, Tb b)     \
    {                                                                        \
        Td r;                                                                \
        __asm__(asm_op " %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));              \
        return r;                                                            \
    }

__PS2_MMI_EXT2(__mmi_pextlb_s, "pextlb", int16x8_t, int8x16_t, int8x16_t)
__PS2_MMI_EXT2(__mmi_pextlh_s, "pextlh", int32x4_t, int16x8_t, int16x8_t)
__PS2_MMI_EXT2(__mmi_pextlw_s, "pextlw", int32x4_t, int32x4_t, int32x4_t)
__PS2_MMI_EXT2(__mmi_pextlb_u, "pextlb", uint16x8_t, uint8x16_t, uint8x16_t)
__PS2_MMI_EXT2(__mmi_pextlh_u, "pextlh", uint32x4_t, uint16x8_t, uint16x8_t)
__PS2_MMI_EXT2(__mmi_pextlw_u, "pextlw", uint32x4_t, uint32x4_t, uint32x4_t)
__PS2_MMI_EXT2(__mmi_pextub_s, "pextub", int16x8_t, int8x16_t, int8x16_t)
__PS2_MMI_EXT2(__mmi_pextuh_s, "pextuh", int32x4_t, int16x8_t, int16x8_t)
__PS2_MMI_EXT2(__mmi_pextuw_s, "pextuw", int32x4_t, int32x4_t, int32x4_t)
__PS2_MMI_EXT2(__mmi_pextub_u, "pextub", uint16x8_t, uint8x16_t, uint8x16_t)
__PS2_MMI_EXT2(__mmi_pextuh_u, "pextuh", uint32x4_t, uint16x8_t, uint16x8_t)
__PS2_MMI_EXT2(__mmi_pextuw_u, "pextuw", uint32x4_t, uint32x4_t, uint32x4_t)

// PCPYLD: combine two 64-bit low halves of the source operands
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pcpyld(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    __asm__("pcpyld %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// PCPYUD: combine two 64-bit high halves of the source operands
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pcpyud(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    __asm__("pcpyud %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// PINTH: interleave the halfword halves of the two operands
static inline __attribute__((__always_inline__)) int16x8_t __mmi_pinth(int16x8_t a, int16x8_t b)
{
    int16x8_t r;
    __asm__("pinth %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// PREVH: reverse all 8 halfword lanes
static inline __attribute__((__always_inline__)) int16x8_t __mmi_prevh(int16x8_t a)
{
    int16x8_t r;
    __asm__("prevh %0,%1" : "=r"(r) : "r"(a));
    return r;
}

// PROT3W: rotate word lanes (rs.w[1] -> rd.w[0], rs.w[2] -> rd.w[1], rs.w[3] -> rd.w[2], rs.w[0] -> rd.w[3])
static inline __attribute__((__always_inline__)) int32x4_t __mmi_prot3w(int32x4_t a)
{
    int32x4_t r;
    __asm__("prot3w %0,%1" : "=r"(r) : "r"(a));
    return r;
}

// QFSRV: quad funneled (funnel) shift right variable, second operand supplies
// the 64 bits shifted in from the left
// QFSRV is a quadword funnel shift whose shift amount lives in the
// instruction's sa field; this binutils only accepts the 3-operand form and
// always encodes sa = 0, so no variable shift is expressible.  Kept as a
// 2-operand helper (sa = 0) rather than silently ignoring an argument.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_qfsrv(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    __asm__("qfsrv %0,%1,%2" : "=&r"(r) : "r"(a), "r"(b));
    return r;
}

// ----------------------------- compare (assembler: pceq*/pcgt*) -----------------------------
#define __PS2_MMI_CMPOP(name, asm_op)                                        \
    static inline __attribute__((__always_inline__)) uint32x4_t name##_u(uint32x4_t a, uint32x4_t b) \
    {                                                                        \
        uint32x4_t r;                                                        \
        __asm__(asm_op " %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));              \
        return r;                                                            \
    }                                                                        \
    static inline __attribute__((__always_inline__)) int32x4_t name##_s(int32x4_t a, int32x4_t b) \
    {                                                                        \
        int32x4_t r;                                                         \
        __asm__(asm_op " %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));              \
        return r;                                                            \
    }

__PS2_MMI_CMPOP(__mmi_pcmpeqb, "pceqb")
__PS2_MMI_CMPOP(__mmi_pcmpeqh, "pceqh")
__PS2_MMI_CMPOP(__mmi_pcmpeqw, "pceqw")
__PS2_MMI_CMPOP(__mmi_pcmpgtb, "pcgtb")
__PS2_MMI_CMPOP(__mmi_pcmpgth, "pcgth")
__PS2_MMI_CMPOP(__mmi_pcmpgtw, "pcgtw")

// ----------------------------- logic -----------------------------
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pand(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    __asm__("pand %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_por(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    __asm__("por %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pxor(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    __asm__("pxor %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pnor(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    __asm__("pnor %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// ----------------------------- shifts -----------------------------
// immediate shifts (0..31 for w, 0..15 for h); statement-expression macros so
// the immediate reaches the asm directly (works without -O)
#define __PS2_MMI_SHIFT(asm_op, T, a, n)                                     \
    __extension__ ({                                                         \
        T r;                                                                 \
        __asm__(asm_op " %0,%1,%2" : "=r"(r) : "r"(a), "i"(n));              \
        r;                                                                   \
    })
#define __mmi_psllh_u(a, n) __PS2_MMI_SHIFT("psllh", uint32x4_t, a, n)
#define __mmi_psllw_u(a, n) __PS2_MMI_SHIFT("psllw", uint32x4_t, a, n)
#define __mmi_psrlh_u(a, n) __PS2_MMI_SHIFT("psrlh", uint32x4_t, a, n)
#define __mmi_psrlw_u(a, n) __PS2_MMI_SHIFT("psrlw", uint32x4_t, a, n)
#define __mmi_psrah_s(a, n) __PS2_MMI_SHIFT("psrah", int32x4_t, a, n)
#define __mmi_psraw_s(a, n) __PS2_MMI_SHIFT("psraw", int32x4_t, a, n)



// variable quadword shifts (shift amount taken from bits[5:0] of the register)
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_psllvw(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    __asm__("psllvw %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_psrlvw(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    __asm__("psrlvw %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
static inline __attribute__((__always_inline__)) int32x4_t __mmi_psravw(int32x4_t a, int32x4_t b)
{
    int32x4_t r;
    __asm__("psravw %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// ----------------------------- abs / min / max -----------------------------
// PABSH: per 16-bit lane |x| (saturates for -32768)
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pabsh(int16x8_t a)
{
    int32x4_t r;
    __asm__("pabsh %0,%1" : "=r"(r) : "r"(a));
    return r;
}

// PABSW: per 32-bit lane |x| (saturates for -2147483648)
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pabsw(int32x4_t a)
{
    int32x4_t r;
    __asm__("pabsw %0,%1" : "=r"(r) : "r"(a));
    return r;
}

// PMAXH/PMINH: signed 16-bit per-lane min/max
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pmaxh(int16x8_t a, int16x8_t b)
{
    int32x4_t r;
    __asm__("pmaxh %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pminh(int16x8_t a, int16x8_t b)
{
    int32x4_t r;
    __asm__("pminh %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// PMAXW/PMINW: signed 32-bit per-lane min/max
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pmaxw(int32x4_t a, int32x4_t b)
{
    int32x4_t r;
    __asm__("pmaxw %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pminw(int32x4_t a, int32x4_t b)
{
    int32x4_t r;
    __asm__("pminw %0,%1,%2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// ----------------------------- horizontal multiply, pack, divide, permute -----------------------------
// These are R5900 EE MMI instructions from the EE Core Instruction Set
// manual.  They are unrelated to the Loongson instructions that happen to
// share a name: there is no PMADDHW here, PHMADH adds adjacent 16x16
// products horizontally instead of widening a dot product.
//
// Every one of these still has to be used as lq -> op -> sq inside a single
// expression.  See the note at the top of this file.

// PHMADH: four sums of adjacent signed 16x16 products.
// rd.w[i] = a.h[2*i]*b.h[2*i] + a.h[2*i+1]*b.h[2*i+1], i = 0..3.
// Does not accumulate.  LO.w[0]/HI.w[0]/LO.w[2]/HI.w[2] get copies of the
// four sums; the other four HI/LO words are undefined.
static inline __attribute__((__always_inline__)) int32x4_t __mmi_phmadh(int16x8_t a, int16x8_t b)
{
    int32x4_t r;
    __asm__("phmadh %0,%1,%2" : "=r"(r) : "r"(a), "r"(b) : "$hi", "$lo");
    return r;
}

// PHMSBH: odd lane minus even lane of each adjacent pair.
// rd.w[i] = a.h[2*i+1]*b.h[2*i+1] - a.h[2*i]*b.h[2*i].
static inline __attribute__((__always_inline__)) int32x4_t __mmi_phmsbh(int16x8_t a, int16x8_t b)
{
    int32x4_t r;
    __asm__("phmsbh %0,%1,%2" : "=r"(r) : "r"(a), "r"(b) : "$hi", "$lo");
    return r;
}

// PMSUBH subtracts eight signed 16x16 products from the HI/LO accumulator.
// This wrapper copies acc into both HI and LO first, so the four words that
// come back in rd are the even halfword products, taken from the even
// accumulator words:
//   rd.w[0] = acc.w[0] - a.h[0]*b.h[0]
//   rd.w[1] = acc.w[0] - a.h[2]*b.h[2]
//   rd.w[2] = acc.w[2] - a.h[4]*b.h[4]
//   rd.w[3] = acc.w[2] - a.h[6]*b.h[6]
// The odd-halfword differences stay in LO.w[1], HI.w[1], LO.w[3], HI.w[3].
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pmsubh(int32x4_t acc, int16x8_t a, int16x8_t b)
{
    int32x4_t r;
    __asm__("pmthi %3\n\tpmtlo %3\n\tpmsubh %0,%1,%2" : "=&r"(r) : "r"(a), "r"(b), "r"(acc) : "$hi", "$lo");
    return r;
}

// PMSUBW: signed 64-bit multiply-subtract of words 0 and 2 only.  The two
// operands must be sign-extended 32-bit values, which an int32x4_t is.  With
// HI and LO both seeded from acc the two differences are
//   (acc.w[0]<<32 | acc.w[0]) - (int64)a.w[0]*b.w[0]
//   (acc.w[2]<<32 | acc.w[2]) - (int64)a.w[2]*b.w[2]
// and rd receives both 32-bit halves of each.  When the product exceeds
// acc.w[i] the subtraction borrows, so the high word is acc.w[i] - 1.
static inline __attribute__((__always_inline__)) int32x4_t __mmi_pmsubw(int32x4_t acc, int32x4_t a, int32x4_t b)
{
    int32x4_t r;
    __asm__("pmthi %3\n\tpmtlo %3\n\tpmsubw %0,%1,%2" : "=&r"(r) : "r"(a), "r"(b), "r"(acc) : "$hi", "$lo");
    return r;
}

// PPACB: low byte of every halfword.  rt packs into the low 64 bits,
// rs into the high 64 bits.  rd.b[i] = rt.b[2*i], rd.b[8+i] = rs.b[2*i].
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_ppacb(uint32x4_t rs, uint32x4_t rt)
{
    uint32x4_t r;
    __asm__("ppacb %0,%1,%2" : "=r"(r) : "r"(rs), "r"(rt));
    return r;
}

// PPACH: low halfword of every word.  rt packs into the low 64 bits,
// rs into the high 64 bits.  rd.h[i] = rt.h[2*i], rd.h[4+i] = rs.h[2*i].
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_ppach(uint32x4_t rs, uint32x4_t rt)
{
    uint32x4_t r;
    __asm__("ppach %0,%1,%2" : "=r"(r) : "r"(rs), "r"(rt));
    return r;
}

// PPACW: low word of every doubleword.  rd.w = {rt.w[0], rt.w[2], rs.w[0], rs.w[2]}.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_ppacw(uint32x4_t rs, uint32x4_t rt)
{
    uint32x4_t r;
    __asm__("ppacw %0,%1,%2" : "=r"(r) : "r"(rs), "r"(rt));
    return r;
}

// PPAC5: pack four 8-8-8-8 words into four 1-5-5-5 halfwords, one per word,
// in the low halfword.  The high halfword of each word becomes 0.
// rd.h[2*i] = b3[7]<<15 | b2[7:3]<<10 | b1[7:3]<<5 | b0[7:3].
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_ppac5(uint32x4_t rt)
{
    uint32x4_t r;
    __asm__("ppac5 %0,%1" : "=r"(r) : "r"(rt));
    return r;
}

// PEXT5: expand the low 16 bits of each word from 1-5-5-5 to 8-8-8-8.
// Each 5-bit field shifts up by 3 and is zero-filled; the top bit shifts up
// by 7.  The high 16 bits of each input word are ignored.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pext5(uint32x4_t rt)
{
    uint32x4_t r;
    __asm__("pext5 %0,%1" : "=r"(r) : "r"(rt));
    return r;
}

// PDIVW: signed divide of words 0 and 2 only.  No rd; the sign-extended
// quotients land in LO and the sign-extended remainders in HI.  Read them
// back with __mmi_pmflo/__mmi_pmfhi.  Dividing by zero raises nothing.
static inline __attribute__((__always_inline__)) void __mmi_pdivw(int32x4_t rs, int32x4_t rt)
{
    __asm__("pdivw %0,%1" : : "r"(rs), "r"(rt) : "$hi", "$lo");
}

// PDIVUW: unsigned divide of words 0 and 2.  The quotient and the remainder
// are sign-extended into LO and HI even though the division is unsigned.
static inline __attribute__((__always_inline__)) void __mmi_pdivuw(uint32x4_t rs, uint32x4_t rt)
{
    __asm__("pdivuw %0,%1" : : "r"(rs), "r"(rt) : "$hi", "$lo");
}

// PDIVBW: divide each of the four signed words of rs by the one signed
// halfword rt.h[0].  Four quotients in LO, four remainders zero-extended to
// 32 bits in HI.  No exception on divide-by-zero; 0x80000000 / -1 returns
// quotient 0x80000000 and remainder 0.
static inline __attribute__((__always_inline__)) void __mmi_pdivbw(int32x4_t rs, int32x4_t rt)
{
    __asm__("pdivbw %0,%1" : : "r"(rs), "r"(rt) : "$hi", "$lo");
}

// PLZCW: count of leading bits equal to the top bit, minus one.  Only the
// low two words are defined: rd.w[0] from rs.w[0], rd.w[1] from rs.w[1].
// rd.w[2] and rd.w[3] are undefined.  lzc(0) and lzc(0xFFFFFFFF) both store 31.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_plzcw(uint32x4_t rs)
{
    uint32x4_t r;
    __asm__("plzcw %0,%1" : "=r"(r) : "r"(rs));
    return r;
}

// PEXEH: per 64-bit half, rd.h = {h2, h1, h0, h3} and {h6, h5, h4, h7}.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pexeh(uint32x4_t rt)
{
    uint32x4_t r;
    __asm__("pexeh %0,%1" : "=r"(r) : "r"(rt));
    return r;
}

// PEXCH: per 64-bit half, rd.h = {h0, h2, h1, h3} and {h4, h6, h5, h7}.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pexch(uint32x4_t rt)
{
    uint32x4_t r;
    __asm__("pexch %0,%1" : "=r"(r) : "r"(rt));
    return r;
}

// PEXEW: rd.w = {w2, w1, w0, w3}.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pexew(uint32x4_t rt)
{
    uint32x4_t r;
    __asm__("pexew %0,%1" : "=r"(r) : "r"(rt));
    return r;
}

// PEXCW: rd.w = {w0, w2, w1, w3}.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pexcw(uint32x4_t rt)
{
    uint32x4_t r;
    __asm__("pexcw %0,%1" : "=r"(r) : "r"(rt));
    return r;
}

// PINTEH: interleave the low halfword of each word.
// rd.h = {rt.h0, rs.h0, rt.h2, rs.h2, rt.h4, rs.h4, rt.h6, rs.h6}.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pinteh(uint32x4_t rs, uint32x4_t rt)
{
    uint32x4_t r;
    __asm__("pinteh %0,%1,%2" : "=r"(r) : "r"(rs), "r"(rt));
    return r;
}

// PCPYH: broadcast the lowest halfword of each 64-bit half.
// rd.h[0..3] = rt.h[0], rd.h[4..7] = rt.h[4].
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pcpyh(uint32x4_t rt)
{
    uint32x4_t r;
    __asm__("pcpyh %0,%1" : "=r"(r) : "r"(rt));
    return r;
}

// PADSBH: subtract the low four halfword pairs, add the high four.
// rd.h[i] = rs.h[i] - rt.h[i] for i = 0..3, rs.h[i] + rt.h[i] for i = 4..7.
// Wraps, does not saturate.
static inline __attribute__((__always_inline__)) int16x8_t __mmi_padsbh(int16x8_t rs, int16x8_t rt)
{
    int16x8_t r;
    __asm__("padsbh %0,%1,%2" : "=r"(r) : "r"(rs), "r"(rt));
    return r;
}

// PMFHL.LH: rd.h = {LO.h0, LO.h2, HI.h0, HI.h2, LO.h4, LO.h6, HI.h4, HI.h6}.
// The even halfword of every HI/LO word; the odd halfwords are dropped.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pmfhl_lh()
{
    uint32x4_t r;
    __asm__ __volatile__("pmfhl.lh %0" : "=&r"(r) : : "$hi", "$lo");
    return r;
}

// PMFHL.LW: rd.w = {LO.w0, HI.w0, LO.w2, HI.w2}.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pmfhl_lw()
{
    uint32x4_t r;
    __asm__ __volatile__("pmfhl.lw %0" : "=&r"(r) : : "$hi", "$lo");
    return r;
}

// PMFHL.SH: every HI/LO word saturated to signed int16, clamped to
// [-32768, 32767], in the same lane order as PMFHL.LH:
// rd.h = {sat(LO.w0), sat(LO.w1), sat(HI.w0), sat(HI.w1),
//         sat(LO.w2), sat(LO.w3), sat(HI.w2), sat(HI.w3)}.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pmfhl_sh()
{
    uint32x4_t r;
    __asm__ __volatile__("pmfhl.sh %0" : "=&r"(r) : : "$hi", "$lo");
    return r;
}

// PMFHL.SLW: combine each HI||LO word pair into a signed 64-bit value,
// saturate to signed int32, and sign-extend back into the 64-bit half.
// low 64 = clamp((int64)HI.w0<<32 | LO.w0) to [-2147483648, 2147483647].
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pmfhl_slw()
{
    uint32x4_t r;
    __asm__ __volatile__("pmfhl.slw %0" : "=&r"(r) : : "$hi", "$lo");
    return r;
}

// PMFHL.UW: rd.w = {LO.w1, HI.w1, LO.w3, HI.w3}.
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pmfhl_uw()
{
    uint32x4_t r;
    __asm__ __volatile__("pmfhl.uw %0" : "=&r"(r) : : "$hi", "$lo");
    return r;
}

// PMTHL.LW: rs.w[0] -> LO.w[0], rs.w[1] -> HI.w[0], rs.w[2] -> LO.w[2],
// rs.w[3] -> HI.w[2].  The other four HI/LO words are left unchanged.
static inline __attribute__((__always_inline__)) void __mmi_pmthl_lw(uint32x4_t rs)
{
    __asm__("pmthl.lw %0" : : "r"(rs) : "$hi", "$lo");
}

// PMTHI / PMTLO: copy a whole 128-bit GPR into HI or LO.
static inline __attribute__((__always_inline__)) void __mmi_pmthi(uint32x4_t rs)
{
    __asm__("pmthi %0" : : "r"(rs) : "$hi", "$lo");
}
static inline __attribute__((__always_inline__)) void __mmi_pmtlo(uint32x4_t rs)
{
    __asm__("pmtlo %0" : : "r"(rs) : "$hi", "$lo");
}

// ----------------------------- misc helpers -----------------------------
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pzerow_u()
{
    return (uint32x4_t) {
        0, 0, 0, 0
    };
}
static inline __attribute__((__always_inline__)) uint16x8_t __mmi_pzeroh_u()
{
    return (uint16x8_t) {
        0, 0, 0, 0, 0, 0, 0, 0
    };
}
static inline __attribute__((__always_inline__)) uint8x16_t __mmi_pzerob_u()
{
    return (uint8x16_t) {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
}

static inline __attribute__((__always_inline__)) int32x4_t __mmi_pzerow_s()
{
    return (int32x4_t) {
        0, 0, 0, 0
    };
}
static inline __attribute__((__always_inline__)) int16x8_t __mmi_pzeroh_s()
{
    return (int16x8_t) {
        0, 0, 0, 0, 0, 0, 0, 0
    };
}
static inline __attribute__((__always_inline__)) int8x16_t __mmi_pzerob_s()
{
    return (int8x16_t) {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
}

static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pfillw_u(uint32_t v)
{
    return (uint32x4_t) {
        v, v, v, v
    };
}
static inline __attribute__((__always_inline__)) uint16x8_t __mmi_pfillh_u(uint16_t v)
{
    return (uint16x8_t) {
        v, v, v, v, v, v, v, v
    };
}
static inline __attribute__((__always_inline__)) uint8x16_t __mmi_pfillb_u(uint8_t v)
{
    return (uint8x16_t) {
        v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v
    };
}

static inline __attribute__((__always_inline__)) int32x4_t __mmi_pfillw_s(int32_t v)
{
    return (int32x4_t) {
        v, v, v, v
    };
}
static inline __attribute__((__always_inline__)) int16x8_t __mmi_pfillh_s(int16_t v)
{
    return (int16x8_t) {
        v, v, v, v, v, v, v, v
    };
}
static inline __attribute__((__always_inline__)) int8x16_t __mmi_pfillb_s(int8_t v)
{
    return (int8x16_t) {
        v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v
    };
}

// aliasing loads / stores (compiler keeps the alignment requirement)
static inline __attribute__((__always_inline__)) uint32x4_t __mmi_pldw_u(const void* p)
{
    return *(const uint32x4_t*)p;
}
static inline __attribute__((__always_inline__)) uint16x8_t __mmi_pldh_u(const void* p)
{
    return *(const uint16x8_t*)p;
}
static inline __attribute__((__always_inline__)) uint8x16_t __mmi_pldb_u(const void* p)
{
    return *(const uint8x16_t*)p;
}

static inline __attribute__((__always_inline__)) int32x4_t __mmi_pldw_s(const void* p)
{
    return *(const int32x4_t*)p;
}
static inline __attribute__((__always_inline__)) int16x8_t __mmi_pldh_s(const void* p)
{
    return *(const int16x8_t*)p;
}
static inline __attribute__((__always_inline__)) int8x16_t __mmi_pldb_s(const void* p)
{
    return *(const int8x16_t*)p;
}

static inline __attribute__((__always_inline__)) void __mmi_pstw_u(void* p, uint32x4_t v)
{
    *(uint32x4_t*)p = v;
}
static inline __attribute__((__always_inline__)) void __mmi_psth_u(void* p, uint16x8_t v)
{
    *(uint16x8_t*)p = v;
}
static inline __attribute__((__always_inline__)) void __mmi_pstb_u(void* p, uint8x16_t v)
{
    *(uint8x16_t*)p = v;
}

static inline __attribute__((__always_inline__)) void __mmi_pstw_s(void* p, int32x4_t v)
{
    *(int32x4_t*)p = v;
}
static inline __attribute__((__always_inline__)) void __mmi_psth_s(void* p, int16x8_t v)
{
    *(int16x8_t*)p = v;
}
static inline __attribute__((__always_inline__)) void __mmi_pstb_s(void* p, int8x16_t v)
{
    *(int8x16_t*)p = v;
}

#ifdef __cplusplus
}
#endif

#endif // defined(_EE) || defined(__PS2__)

#endif // _PS2_MMI_H
