# ncnn on PlayStation Portable (research notes)

The port is started, not finished. `toolchains/psp.toolchain.cmake` builds
ncnn as a mips sub-variant, and `ReLU` has a VFPU implementation
(`src/layer/mips/relu_mips_psp.cpp`, `vmax.q` / `vmin.q`). Everything else still
runs the scalar mips kernels. This note records what the hardware actually offers
so the next kernels are written against the chip and not against the PS2 port.

Sources, all community reverse engineering rather than a Sony manual (Sony never
published one):

- Allegrex overview: <https://www.ppsspp.org/docs/psp-hardware/cpu/allegrex-overview/>
- VFPU reference, generated from a tested YAML description:
  <https://pspdev.github.io/vfpu-docs/>, source `pspdev/vfpu-docs`
  (`docs/introduction.md`, `inst-vfpu-desc.yaml`, `docs/bugs.md`).

The VFPU docs are the trustworthy ones. Every instruction's encoding, prefixes,
register-overlap rule and pseudocode has a test meant to run on real hardware, so
the functional descriptions below are taken from that pseudocode, not guessed.

## Hardware summary

| unit | what it is | useful for NN inference |
|---|---|---|
| **Allegrex** | 32-bit MIPS II derivative, 222 MHz (later firmware 333 MHz), little-endian, no MMU/TLB | the whole control plane; scalar int8 with `seb`/`min`/`max` |
| **FPU (COP1)** | single precision only, 32-bit FPRs | scalar fp32 fallback |
| **VFPU (COP2)** | 128 × 32-bit registers, vector/matrix float and integer, shares the CPU instruction fetcher | fp32 convolution / GEMM, and the int8 pack/unpack around it |
| **Media Engine** | second Allegrex, VFPU removed, plus a fixed codec accelerator | not programmable, ignore |
| **GE** | fixed-function GPU, reads VRAM not the CPU cache | not a GPGPU; same conclusion as the PS2 GS, not studied here |

Two things make this simpler than the PS2:

1. There is one SIMD unit and it shares the instruction stream with the CPU, so
   there is no EE/VU0/VU1 split, no DMA microprogram, and no "does this overlap
   the CPU" question. Issue width is one.
2. The registers are 32-bit and addressed as ordinary vectors. A `vdot.q` reads
   two quads and writes one scalar. There is no 128-bit result that only the low
   64 bits of are addressable from C, which is the thing that made the EE
   `phmadh` kernel awkward.

## Allegrex integer ISA

MIPS II plus a small set of later and custom instructions. The ones that matter
for ncnn, with their encodings so nobody has to rediscover which `madd` is which:

| insn | effect | encoding |
|---|---|---|
| `seb` | sign-extend byte to word | MIPS32 `0x7c000420` |
| `seh` | sign-extend halfword to word | MIPS32 `0x7c000620` |
| `ext` / `ins` | bitfield extract / insert | MIPS32 |
| `wsbh` | swap bytes within each halfword | MIPS32 |
| `wsbw` | swap bytes in a word | Allegrex-only, `BSHFL` |
| `bitrev` | reverse bits in a word | Allegrex-only |
| `clz` / `clo` | count leading zeros / ones | MIPS32 semantics, `SPECIAL` encoding |
| `rotr` / `rotrv` | rotate right | MIPS32 |
| `madd` / `maddu` / `msub` / `msubu` | 32×32 → HI/LO accumulate | `SPECIAL`, the R4010-style encoding `0x0000001c`, **not** the MIPS32 `0x70000000` |
| `min` / `max` | signed minimum / maximum, three operands | Allegrex-only, `0x0000002d` / `0x0000002c` |

`min`/`max` and `seb` are the useful ones: ReLU, Clip and int8 sign extension
collapse to a single instruction with no branch. They are scalar, so they cover
the tail and the pack/unpack glue, not the inner dot product.

## VFPU register file

128 registers of 32 bits, arranged as **8 matrices of 4×4**. A vector is a row or
a column of one matrix and cannot cross a matrix boundary. That is the real
constraint: the widest register you can name in one instruction is 4 elements
(128 bits), and the widest matrix is one 4×4 block.

Naming, where `X` is the matrix 0..7:

| form | meaning |
|---|---|
| `S000` | one 32-bit element |
| `C000` / `R000` | a column / a row |
| `.s` `.p` `.t` `.q` | 1, 2, 3, 4 elements |
| `M000` | 4×4 in row-major order |
| `E000` | the same 4×4 read transposed (column-major) |

Not every pattern exists. Pair and triple vectors have 128 addressable patterns,
quads only 64; 2×2 and 3×3 matrices have 64, 4×4 only 16. In practice stick to the
edge-aligned ones (`C000`, `R000`, `M000`) and treat the shifted forms (`C010`,
`R001`) as something to avoid — see the `vhtfm` bug below.

`M` versus `E` is the sharp edge. Loading and storing is row-major, so data goes
in and out through `M`/`R`/`C`. `vtfm` and `vhtfm` treat their matrix operand as
column-major, so they want `E`. `vmmul` is worse: the assembler flips the `M`/`E`
bit of the `rs` operand (and the multiply order) so that `vmmul.q M000, M100, M200`
reads as natural column-major multiplication. The pseudocode is the part that is
tested; the assembler spelling is the part that lies. Write `vmmul` with the
`.word` encoding, or check the disassembly, before trusting it in a kernel.

Eight control registers sit above the 128: `VFPU_PFXS/PFXT/PFXD` (129..131) hold
the prefix state, `VFPU_CC` (131) the compare flags, `VFPU_REV` (135) is
read-only, and 136..147 are the RNG state. Saving a thread context means saving
all of these, not just the 128.

## Prefixes

Every arithmetic instruction can be decorated. `vpfxs` / `vpfxt` rewrite the two
inputs, `vpfxd` rewrites the output, and the assembler emits them as separate
instructions in front of the real one:

```
vmul.q R000, R100[x,y,x,y], R200[-x,-y,z,w]
    →  vpfxs [x,y,x,y] / vpfxt [-x,-y,z,w] / vmul.q R000, R100, R200
```

Inputs can be negated, forced absolute, swizzled, or replaced by a constant.
Constants are only `0, ±1, ±2, ±3, ±1/2, ±1/3, ±1/4, ±1/6`. The output can be clamped
to `[0,1]` or `[-1,1]`, or have lanes masked off.

A prefix costs one issue cycle and adds **no** latency. That makes `[|x|]` a free
Abs and `[[0:1]]` a free ReLU for values already in `0..1`, but only when the
decorated instruction was going to issue anyway. For throughput it is often
cheaper to hoist `vabs` / `vsat0` once and reuse the register than to pay the
prefix on every instruction of a tight loop.

## Pipeline and overlap

| class | throughput | latency | examples |
|---|---|---|---|
| bit manipulation | 1 | 3 | `vmov` `vabs` `vneg` `vsat0` `vsat1` `vi2c` `vc2i` |
| arithmetic | 1 | 5 | `vadd` `vsub` `vmul` `vi2f` `vf2iz` |
| reduction | 1 | 7 | `vdot` `vfad` `vavg` |
| transform | N | N+6 | `vtfm2/3/4` — 8 / 9 / 10 cycles for `.p` / `.t` / `.q` |
| matrix multiply | N² | N²+6 | `vmmul` — **22 cycles** for a 4×4 |
| division | 14N | 14N+3 | `vdiv.q` = 56 cycles, avoid |
| transcendental | N | N+6 | `vsin` `vexp2` … approximate, see below |
| prefix | 1 | 1 | consumed by the very next instruction |

N is the element count of the flavour (1/2/3/4). Latency hiding means
interleaving independent quads, exactly as on any other in-order SIMD.

`vmmul`, `vtfm2/3/4`, `vhtfm2/3/4`, `vqmul` and `vcrsp` forbid **any** overlap
between inputs and output. They are internally a run of dot products that write
the destination a piece at a time, so an input that aliases the output is
overwritten mid-instruction and the result is wrong. `vmscl` and `vmmov` allow
overlap only when the element count and the row/column direction match.

Two hazards that are easy to miss:

- `mfvc` reading `VFPU_CC` needs a `vnop` (or any other VFPU instruction) between
  it and the `vcmp` that produced the flags.
- `vhtfm2.p` and `vhtfm3.t` write the result shifted by one element whenever the
  destination register number is 64 or above, i.e. any register that is not
  left/top aligned (`R001`, `C010`, …). All PSP models. `vhtfm4.q` cannot hit it
  because a quad has no such register. Stay on `R0`/`C0`.

## Floating point

IEEE-754 single precision with two breaks from the standard, both verified by
the `ieee754-fun.c` test in vfpu-docs:

- Rounding mode is hardwired to round-to-nearest. There is no way to select
  round-toward-zero or round-to-even-differently.
- No denormals. A denormal input is flushed to zero, and a result that would be
  denormal is flushed to zero.

`vrcp`, `vrsq`, `vsin`, `vcos`, `vexp2`, `vlog2` are approximations: roughly the
bottom 3 to 3.5 mantissa bits are wrong (relative error about 7e-7). Fine for
activations if the reference is compared loosely, not fine for anything that
must match the scalar libm bit for bit. Use the scalar FPU for those.

## Memory

| insn | alignment | notes |
|---|---|---|
| `lv.s` / `sv.s` | 4 bytes | one element, fault otherwise |
| `lv.q` / `sv.q` | **16 bytes** | one quad, fault otherwise |
| `lvl.q` / `lvr.q` | 4 bytes (low 2 bits ignored) | unaligned quad, like `lwl`/`lwr` |

`lvl.q` and `lvr.q` (and the `ulv.q` macro that expands to them) corrupt an FPU
register on the PSP-1000. The low 5 bits of the VFPU destination select which
`$f` register dies: `C000` clobbers `$f0`, `C010` clobbers `$f1`. Fixed on the
2000 and later, but anything claiming to run on a 1000 has to either avoid the
instructions or list the FPU register as clobbered so gcc saves it. Aligned
`lv.q` has no such bug, so the practical rule is the same as everywhere else in
this project: pack to 16 bytes and use the aligned load.

The GE cannot read the CPU cache, so anything handed to it must be written back
explicitly. Irrelevant while the GE is unused for compute, worth remembering the
moment someone proposes it.

## What this means for int8

The VFPU has no integer multiply. Integer support is pack, unpack and convert,
and the multiply itself is fp32. So an int8 dot product is:

```
vc2i.s  C000, S000      # 4 packed bytes  -> 4 ints, byte placed at bits 31..24
vi2f.q  C000, C000, 24  # divide by 2^24   -> the original signed byte as a float
vdot.q  S200, C000, C100
vf2iz.s S200, S200, 0   # trunc toward zero back to int
```

The conversions, from the tested pseudocode:

| insn | what it really does |
|---|---|
| `vc2i.s` | one word of 4 packed bytes → a quad of ints, each byte moved to bits 31..24, low 24 bits zero |
| `vuc2ifs.s` | one word of 4 bytes → a quad, each byte replicated into all four byte lanes |
| `vs2i` / `vus2i` | packed signed / unsigned shorts → ints, value in the top 16 bits |
| `vi2f rd, rs, n` | int → float, then divide by `2^n` |
| `vf2iz` / `vf2in` / `vf2iu` / `vf2id` | float → int with a `2^n` scale: trunc / nearest / ceil / floor |
| `vi2c.q` | 4 ints → 4 packed bytes, taking bits 31..24 of each |
| `vi2uc.q` | same, but negative inputs clamp to 0 |
| `vf2h` / `vh2f` | fp32 ↔ fp16, two per word. `vf2h` discards the top mantissa bits and can turn some NaNs into Inf |

Note the direction of the shift. `vc2i` does **not** zero-extend a byte into the
low 8 bits the way a normal SIMD unpack does; it puts the byte at the top and
leaves the bottom clear. `vi2f` with scale 24 undoes that for signed values,
because the low 24 bits are zero and dividing by `2^24` lands exactly on the
byte. Unsigned needs `vuc2ifs` or a different scale, and the two are not
interchangeable.

The cost is real. Four bytes become one `vdot.q`, which has latency 7, and each
side needs a `vc2i` plus a `vi2f` (latency 3 and 5) before it. Against the EE,
where `pextlb` + `psrah` + `phmadh` dots eight bytes per instruction and stays in
integer, this is a lot of instruction for eight MACs. It only wins where the
weights are reused enough to hoist the `vi2f` out of the loop, i.e. the same
output-stationary 1×1 case the PS2 kernel already special-cases. A depthwise 3×3
with tiny `K` will not pay for it.

fp32 is the straightforward win: `lv.q` / `vmul.q` / `vdot.q` / `sv.s` is a
normal SIMD kernel, and `vmmul.q` + `vtfm4.q` give a 4×4 tile in 22 and 10
cycles with no microcode.

## How it plugs into ncnn

`CMAKE_SYSTEM_PROCESSOR MATCHES "^(mips)"` is the existing mips path, split by
`NCNN_MSA`, `NCNN_MMI` and `NCNN_PS2_MMI`. The PSP toolchain file includes
`pspdev.cmake` and then overrides the processor to `psp`, because pspdev itself
reports `mips` and that would run the MSA / MMI probes. `psp` does not match
`^(mips)`, so it has its own branch ahead of that one.

The branch still sets `NCNN_TARGET_ARCH=mips`. Allegrex is MIPS II plus a
handful of instructions, so the scalar mips kernels are the baseline, and
`NCNN_PSP_VFPU` (default ON) swaps in `src/layer/mips/<layer>_mips_psp.cpp`
through `cmake/ncnn_add_layer.cmake`, the same way `NCNN_PS2_MMI` does.
`NCNN_MSA` and `NCNN_MMI` are forced off.

The VFPU wrappers live in `src/layer/mips/psp_vfpu.h`. `lv.q` faults below
16-byte alignment, so every kernel checks the pointer and finishes the tail in
scalar code. Kernels implemented:

| layer | VFPU form |
|---|---|
| ReLU, PReLU | `vmax.q` against zero; leaky is `vmax` + `vmin` + `vmul` + `vadd` |
| Clip, HardSigmoid | `vmax.q` then `vmin.q`, HardSigmoid after `vmul` + `vadd` |
| HardSwish | the HardSigmoid result multiplied back by the input |
| AbsVal, UnaryOp ABS/NEG | `vabs.q`, `vneg.q` |
| UnaryOp SQRT/EXP/LOG | `vsqrt.q`, `vexp2.q`, `vlog2.q`. The last two are approximations (a few ulp); EXP and LOG are scaled by `1/ln2` and `ln2` because the VFPU only has the base-2 forms. Reciprocal stays on the scalar FPU: `vrcp` is approximate and `vdiv` costs 14 cycles per element |
| Dropout, Bias | `vmul.q`, `vadd.q` |
| Scale, BatchNorm | `vmul.q` + `vadd.q`, per-channel or packed-4 |
| Eltwise | `vadd.q`, `vmul.q`, `vmax.q`, weighted sum as two multiplies and an add |
| GEMM | `vdot.q` over the K axis of `gemm_transB`, the fp32 path. The packed tiled kernel stays scalar: its layout is built for MSA lane shuffles the VFPU has no equivalent of |
| InnerProduct | `vdot.q` on the `elempack == 1` fp32 dots, both the 2-D gemm form and the flattened remainder |
| Convolution | `vdot.q` on the fp32 remainder kernel (`outch < 2`). The input channels are regrouped by four at transform time so each tap is one aligned quad; the `outch >= 2` kernel and the int8 path are unchanged |

None of them advertise bf16. The int8 dot product stays off the VFPU: there is
no integer multiply, and the `vc2i`/`vi2f` detour is fp32 arithmetic on data
that was already int8 (see below). Pooling has no vector form that beats the
scalar compare. `min` / `max` / `seb` need no option; gcc emits them from
ordinary C at `-march=allegrex`.

Do not reuse the PS2 files. `ps2_mmi.h` is R5900 MMI, the register model is
different, and the `vector_size(16)` 64-bit-GPR trick has no meaning on a
32-bit CPU.

## Building

`psp-gcc` is not on `PATH`. Bare `psp-gcc` also fails to link: its spec pulls
in `-lcglue` and `-lpthreadglue`, which live in `$PSPDEV/psp/sdk/lib` and are
only added by `pspdev.cmake`. The ncnn toolchain file includes that.

```bash
export PSPDEV=/usr/local/pspdev
export PATH="$PSPDEV/bin:$PATH"

cmake -B build-psp -DCMAKE_TOOLCHAIN_FILE=toolchains/psp.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_VULKAN=OFF -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF \
    -DNCNN_PIXEL=OFF -DNCNN_PLATFORM_API=OFF -DNCNN_C_API=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_DISABLE_PIC=ON
cmake --build build-psp -j8
```

`-G0` is in the toolchain. Do not pass `-DCMAKE_CXX_FLAGS` on the command
line: it replaces `CMAKE_CXX_FLAGS_INIT` and drops the SDK include path and
`-DPSP`.

`tests/psp/run_on_ppsspp.sh` runs a built ELF under the PPSSPP snap and
`tests/psp/bench_compare.sh` builds `benchncnn` twice, `NCNN_PSP_VFPU` off and
on, and prints the speedup. The model set is the PS2 one (squeezenet and
mobilenet, fp32 and int8): a 32 MB PSP cannot hold more than a 32 MB PS2 can.

PPSSPP does not emulate the remote SIO registers, so `pspDebugSioPutchar`
produces nothing. `benchmark/psp/psp_runtime.c` replaces newlib's `_write` and
forwards each line to `sceKernelPrintf`, which the emulator copies into the
file given with `--log`. A format the HLE understands is logged at Info, and a
headless run does not have the Printf channel that high, so the line is
dropped with no trace. `%c` is not implemented and is logged at Error, which
does show up, and that line quotes the format string. Every line is therefore
sent with a trailing `%c` and appears as `UNIMPL fmt (<text>%c, ...)`. The
runner waits for the `finished, parked` marker and then parks the thread so the
log stays open until it has been read.

## Measured results

PPSSPP 1.20.4, PSP-1000 (32 MB), `loop_count=1`, one thread, the two builds run
back to back. "baseline" is the same tree with `-DNCNN_PSP_VFPU=OFF`, so the
builds differ only in the VFPU kernels. `tests/psp/bench_compare.sh` reproduces
this.

| model | baseline | VFPU kernels | ratio |
|---|---|---|---|
| squeezenet | 667.89 ms | 651.76 ms | 1.02x |
| squeezenet_int8 | 906.34 ms | 890.36 ms | 1.02x |
| mobilenet | 272.49 ms | 263.66 ms | 1.03x |
| mobilenet_int8 | 13483.01 ms | 13478.90 ms | 1.00x |

The int8 models do not use the VFPU: there is no integer multiply, so their dot
product stays scalar (see above). The fp32 gain is small because most of both
networks is convolution, and a convolution with `num_input > 16` or
`num_output > 16` takes the im2col plus sgemm path. That sgemm is the generic
scalar one. The packed tiled kernel needs the lane shuffles MSA has and the
VFPU does not, so only the `outch < 2` remainder and the `elempack == 1`
inner-product dots are on the VFPU, and a 1x1 convolution is intercepted by
`convolution_im2col_gemm` before it ever reaches the VFPU kernel.

`mobilenet_int8` prints `layer create_pipeline 27 conv6/sep failed` and still
reports a time. `conv6/sep` is a 1024x1024 1x1 convolution, and transforming its
kernel does not fit in the memory left after the preceding layers' weights. The
transform returns -100 instead of writing through the null pointer `malloc`
hands back, so the run continues on the untransformed weight. The same line
appears in both builds, and the reported time covers the layers that did
build.
