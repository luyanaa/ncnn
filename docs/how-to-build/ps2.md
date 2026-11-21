# ncnn on PlayStation 2

This document describes the PlayStation 2 (Emotion Engine) port of ncnn: how to
build it, how to run it, which parts of the hardware are used for inference, and
the pitfalls discovered along the way.

## Hardware summary

| unit | what it is | useful for NN inference |
|---|---|---|
| **EE** (Emotion Engine) | MIPS r5900, ~295 MHz, 32 MB RDRAM, 16 KB scratchpad | the control plane and scalar work |
| **EE MMI** | 128-bit integer SIMD on the EE GPRs (`paddw`, `pmulth`, `phmadh`, …) | int8 / int16 kernels |
| **VU0** | 128-bit 4×fp32 MAC unit, 4 KB code + 4 KB data, usable as EE's COP2 | fp32 convolution / GEMM inner loops |
| **VU1** | same as VU0 but 16 KB + 16 KB, DMA-fed through VIF1 | offloading matrix tiles, runs in parallel with the EE |
| **SPR** | 16 KB scratchpad at `0x70000000`, reachable by every DMA channel | staging buffer |
| **IOP** | R3000A @ 37 MHz, 2 MB, separate memory | I/O only, not useful for compute |
| **GS** | fixed-function rasteriser, no shaders, 16/32-bit formats | not usable as a GPGPU |

The port therefore targets **EE MMI + VU0 (COP2) + VU1**, and leaves the IOP and
the GS alone.

## Building

Requirements: [ps2dev](https://github.com/ps2dev/ps2dev) toolchain with
`PS2DEV` and `PS2SDK` exported.

```bash
export PS2DEV=/usr/local/ps2dev
export PS2SDK=$PS2DEV/ps2sdk

cmake -B build-ps2 -DCMAKE_TOOLCHAIN_FILE=toolchains/ps2-ee.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_VULKAN=OFF -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF \
    -DNCNN_PIXEL=OFF -DNCNN_PLATFORM_API=OFF -DNCNN_C_API=OFF \
    -DNCNN_DISABLE_PIC=ON
cmake --build build-ps2 -j8
```

`CMAKE_SYSTEM_PROCESSOR` is `mips64r5900`, which lands in the normal `mips`
target.  PS2 is a **mips sub-variant**, handled the same way as the other mips
boards: `NCNN_TARGET_ARCH` stays `mips`, and `NCNN_PS2_MMI` selects the EE
implementations.  Kernels live in `src/layer/mips/` as `<layer>_mips_ps2.cpp`
and replace the generic `_mips` implementation on PS2 builds
(`cmake/ncnn_add_layer.cmake`).

Footprint notes for the 32 MB machine:

* `NCNN_SIMPLESTL`, `NCNN_THREADS=OFF`, `NCNN_OPENMP=OFF`,
  `NCNN_PIXEL*=OFF`, `NCNN_C_API=OFF` are all worth setting.
* Arch layers that build a transformed weight blob **must release the original
  weights** in `create_pipeline()`; otherwise a model like MobileNet exhausts
  the 32 MB (the symptom is a `layer load_model ... ret:failed` on an unrelated
  later layer).

## Running

No file I/O stack exists unless ps2link / iomanX is present, so `printf()` goes
nowhere by default.  `benchmark/ps2/ps2_runtime.c` fixes that: a constructor
calls `init_scr()` and `_write()` is redirected to `scr_printf()`, so ordinary
`printf`/`fprintf(stderr, …)` appear on the GS screen.

Under PCSX2 2.x:

```bash
pcsx2-qt -nogui -fastboot -unlimited -elf build-ps2/benchmark/benchncnn
```

PCSX2 needs a PS2 BIOS and, on a headless/software-GL host, a working EGL
vendor configuration (Mesa's `libEGL_mesa.so.0` via
`__EGL_VENDOR_LIBRARY_FILENAMES`).  The GS output can be captured with a
screenshot; `tests/ps2/run.sh` automates build + run + crop.

## What is implemented

### `src/layer/mips/ps2_mmi.h`

Wrappers for ~97 R5900 MMI instructions: `lq`/`sq`, byte/half/word add &
subtract (incl. saturating), `pmulth`/`pmultw`/`pmultuw`/`pmaddh`/`pmaddw`/
`pmadduw`/`phmadh`, `pextl*`/`pextu*`/`pcpyld`/`pcpyud`, compares, logical ops,
immediate shifts, `pmin`/`pmax`/`pabs`, `pmfhi`/`pmflo`.

Important: **`pmaddhw`, `pcmpeqw`, `paddusb`, `packsswh`, `pandi`, `pabsd` do
not exist on the R5900.**  Those are Loongson MMI names.  The R5900 equivalents
are `phmadh`, `pceqw`, `paddub`, `ppacb`/`ppach`, and there is no `pabsd`.

### `src/layer/mips/*_mips_ps2.cpp`

| layer | approach |
|---|---|
| `Convolution` | fp32, VU0 COP2, 4 output channels × 4 input channels per step, weights transformed to `[outch][maxk][inch4]`, input interleaved to pixel-major so one `lqc2` fetches four input channels |
| `ConvolutionDepthWise` | fp32, VU0 COP2, four output pixels in the four lanes, alignment-phase accumulation to satisfy the `lqc2` quadword requirement |
| `ReLU` / `Clip` / `BinaryOp` | fp32, VU0 COP2, elementwise 4-wide |

Non-fp32 / packed / grouped / dynamic-weight cases fall back to the generic
implementation where the original weights are still available; the layers
report `-1` instead of silently producing wrong results when the caller would
need data that was released.

### VU1 GEMM engine (`ps2_vu1_gemm.{vsm,h,c}`)

A VU1 microprogram computing `4×K · K×4 → 4×4` fp32, plus an EE-side VIF1 DMA
driver.  Validated for K = 1, 3, 4, 8, 17, 64, 256 (max relative error 1.3e-6).

Two facts that make this much simpler than the usual examples:

* Results do **not** need `xgkick`; VU1 data memory is mapped into the EE
  address space at `0x1100C000`, so the EE can `lq` the result directly after
  the microprogram finishes.
* `xtop`/`xgkick`/`ilw`-style GIF plumbing is irrelevant here — this is a plain
  compute kernel, not a graphics path.

The driver is the blocking version (no double buffering yet).

## Measured results

PCSX2 2.8.2, `loop_count=1`, one thread, PCSX2 and the comparison build run
back to back on the same host.  "baseline" is the same tree built with
`-DNCNN_PS2_MMI=OFF` (generic C++ layers), so the two builds differ only in the
PS2 kernels.  `tests/ps2/bench_compare.sh` reproduces this.

| model | baseline | PS2 kernels | ratio |
|---|---|---|---|
| mobilenet | 1197 ms | **502 ms** | **2.39x** |
| mobilenet_int8 | 12253 ms | 12253 ms | 1.00x |
| squeezenet | 1732 ms | 3670 ms | 0.47x (**slower**) |
| squeezenet_int8 | 6325 ms | 6325 ms | 1.00x |

### Where the numbers come from

Measured by enabling one thing at a time:

| configuration | squeezenet | mobilenet |
|---|---|---|
| baseline (no PS2 kernels) | 1732 | 1197 |
| dense conv fully disabled, depthwise + elementwise on | **1732** | 1233 |
| 1x1 conv (output-stationary) + depthwise | 3488 | **502** |
| 3x3 conv (weight-stationary) + 1x1 (output-stationary) + depthwise | 3488 | 502 |
| everything weight-stationary | 2867 | 762 |

Two conclusions from that table:

1. **The PS2 build itself costs nothing.**  With the dense convolution disabled,
   squeezenet is 1732 ms - identical to the baseline.  So the whole regression
   is the dense-convolution kernel, not some general overhead of the port.
2. **MobileNet's 2.39x comes from the 1x1 kernel.**  Disabling it costs 730 ms
   (502 -> 1233), even though depthwise stays enabled.

### Why the dense kernel is a regression on squeezenet

The pointwise kernel first transposes the input into a pixel-major layout so
that one `lqc2` fetches four input channels of one pixel.  That transpose is
O(inch * size) and is **independent of the output channel count**.  Squeezenet's
fire "squeeze" layers are `inch 512 -> outch 64`, so the transform rewrites 512
channels to produce 64 outputs - the transpose dominates and the layer ends up
slower than the generic loop.  MobileNet's pointwise layers are "expand"
(`outch >= inch`), where the same transform is amortised properly.

Gating the pointwise kernel on `outch * 2 >= inch` was tried and made
squeezenet *worse* (4183 ms), so the transpose is not the only effect; the
kernel's per-group re-read of the input also matters.  The gate that is shipped
is therefore the simple measured one (`maxk == 1 && inch >= 4`,
`maxk == 9 && inch >= 16`), with the 3x3 layers using the weight-stationary
kernel and the 1x1 layers the output-stationary one.

The weight-stationary kernel does what it was written for - squeezenet improves
from 3894 to 2867 ms (+26%) and it removes the horizontal reduction - but it is
still slower than the generic loop on every 3x3 shape, and it costs mobilenet
260 ms (502 -> 762).  With the 3x3 layers left on the generic loop entirely,
squeezenet is 4154 ms, i.e. the 1x1 kernel alone accounts for the whole
regression.

### Pointwise kernel: sharing one weight load across pixels

The pointwise kernel reloads the whole weight block for every output pixel,
and that reload dominates the shrinking ("squeeze") layers where there is
little output to amortise it over.  The weights do not depend on the pixel, so
the kernel now applies each loaded weight to several neighbouring pixels at
once.  Measured on squeezenet, 3x3 still generic:

| pixels per weight load | squeezenet | mobilenet |
|---|---|---|
| 1 | 4154 ms | 502 ms |
| 2 | 3796 ms | 502 ms |
| 4 | 3665 ms | 502 ms |
| 8 | 3670 ms | 502 ms |

It saturates at 4: past that the arithmetic, not the weight traffic, is the
limit.  MobileNet is unaffected throughout, so the extra register pressure
costs the expand layers nothing.  The shipped kernel uses 8, which is no slower
than 4 and keeps the 4-pixel loop as its tail.

Things that were tried and did **not** help on PCSX2:

* staging a group's weights in the 16 KB EE scratchpad (SPR): no measurable
  change (3708 vs 3689 ms on squeezenet, identical on mobilenet).  PCSX2 models
  SPR reads and cache-hit main-memory reads with similar timing.
* gating the 3x3 kernel by per-group weight volume: made both models slower.
* gating the pointwise kernel by the output/input channel ratio.  Every
  threshold either drops squeezenet's squeeze layers and *also* drops a
  mobilenet layer that was winning (mobilenet 502 -> 892 ms) or keeps the
  squeeze layers and stays at 4154 ms.  The ratio does not separate the two.
* a per-layer timing harness that runs the generic loop and the PS2 kernel on
  the same input inside one forward.  Constructing a `Convolution` object on
  the stack faults on the EE (the constructor writes through a null pointer,
  seen as a TLB miss storm at pc 0x12xxxx), and an inline reference loop was
  too slow to finish a model inside the run window.  Abandoned; the numbers
  above all come from whole-model runs.

int8 is deliberately untouched: the int8 layers delegate to
`Convolution_mips`, which has the packed / winograd int8 pipelines.  Falling
back to the plain `Convolution` int8 path instead costs 3.3x (measured:
40.9 s vs 12.2 s on mobilenet_int8).

## How the three compute units cooperate

Full measurements: [`ps2-vu-cooperation.md`](ps2-vu-cooperation.md).  The
short version, all measured on PCSX2 with CP0 Count:

* **VU0 (COP2) does not overlap with the EE.**  Interleaving COP2 with MMI/FPU
  instructions gives no speed-up at all (24576 ops: 24576 vs 24611 ticks), and
  interleaving EE im2col with VU0 dot products is 26% *slower* than doing them
  in phases.  The EE issues one instruction per tick, so the instruction port
  is the bottleneck, not the arithmetic unit.  VU0 is worth using only because
  one `vmul.xyzw` is 4 MACs.
* **VU1 does overlap, and it is the one real source of parallelism.**  A
  double-buffered VU1 pipeline (weights resident in VU memory, tiles streamed
  through VIF1, command mailbox instead of repeated `MSCAL`) is
  **1.73x faster at K=64 and 1.84x at K=256** than the blocking driver.
* **Cost per MAC**: EE scalar C loop 12.05 ticks, VU0 ACC-chain dot product
  2.76 ticks, VU1 double-buffered 12.9 ticks, VU1 blocking 23.8 ticks.
  A VU1 tile needs roughly `K >= 128` before the ~7k-tick VIF1 round trip is
  amortised; below that VU1 loses.
* **Suggested split**: large-K im2col+GEMM tiles -> VU1 double-buffered while
  the EE builds the next tile; small tiles and depthwise -> the existing VU0
  kernels; int8 and data movement -> EE MMI.  Do not try to interleave COP2
  with MMI on the EE.

## Pitfalls worth remembering

1. **`vector_size(16)` is not a 128-bit register.**  This GCC models a 16-byte
   vector as *two* 64-bit GPRs, while an MMI instruction consumes one 128-bit
   GPR.  A value that passes through an ordinary C variable loses its upper
   half.  Use the `lq → op → sq` idiom in one expression, or write one asm block
   per tile.  This is documented at the top of `ps2_mmi.h`.
2. **`vmadd fd, fs, ft` means `fd = ACC + fs * ft`.**  It does not accumulate
   into `fd`.  For a loop accumulator use `vmul` + `vadd`, or the classic
   `vmulax`/`vmadday`/`vmaddaz`/`vmaddw` dot-product chain inside one block.
3. **`lqc2`/`sqc2` require quadword alignment.**  On PCSX2 an unaligned address
   is silently truncated (reads the aligned quadword); on hardware it raises an
   address error.  Convolution therefore pads the channel axis and the
   depthwise kernel accumulates per alignment phase.
4. **`vdiv Q, fs, ft` computes `fs / ft`**; `vrsqrt Q, fs, ft` computes
   `fs / sqrt(ft)`.  Any instruction that reads Q must be preceded by
   `vwaitq` — the hardware does not check the dependency for you.
5. **`qmfc2`/`qmtc2` need no `nop` padding** between macro-mode instructions;
   only reading the clip flags after `vclipw` and the `.I` variants around
   microprogram calls need explicit synchronisation.
6. **VU0 macro mode cannot use** `xgkick`, `xtop`, `ilw`/`isw`, the EFU
   instructions (`esadd`, `eatan`, …) or microprogram branches.
7. **EE FPU division/sqrt are approximate** (12-bit); use the VU's
   `vdiv`/`vrsqrt` when precision matters.
8. **Never issue `MSCAL` while VU1 is still running** — VIF1 hangs.  Either
   wait for the E-bit or use a command mailbox in VU data memory and have the
   microprogram poll it.
9. **A single `UNPACK` of 256 or more quadwords crashes PCSX2**; split the DMA
   into chunks of 128 qwords or less.
10. **Do not merge an UNPACK with the VIF code that sets the "flush" flag into
    one command** — the flag lands in the wrong slot.  Send them separately.
11. **binutils for r5900 rejects the paired-single FPU instructions**, and
    `qfsrv` only assembles in its 3-operand form (shift always 0), so variable
    quadword shifts have to be built from `vmr32` chains instead.
12. **`_write` redirection must handle `\r`**; printing it verbatim to
    `scr_printf` corrupts the console and crashes on startup.

## Verification and benchmarking

`tests/ps2/` contains:

* `selftest.c` — exercises the MMI and COP2 primitives against reference values
  derived from the EE Core Instruction Set manual.  34 groups pass, plus one
  group that documents the known `vector_size(16)` limitation.  `run.sh` builds
  it, runs it under PCSX2 and crops the screenshot.
* `run_on_pcsx2.sh` — runs any ELF under PCSX2 and captures the GS console.
  Serialised with `flock` so several people/jobs can share one machine.
* `bench_compare.sh` — builds `benchncnn` twice (`NCNN_PS2_MMI=OFF` / `ON`) and
  runs both, so the two configurations are always compared back to back.

Always compare back to back on an idle host.  An early measurement of this
port reported an 8x speed-up that turned out to be an artefact of several
leftover emulator processes saturating the machine.

## Future work

* **Dense convolution, second pass - measured, not a win.**  Both ideas were
  built and measured (`NCNN_PS2_WST_PA`, off by default).  The kernel keeps four
  phase-shifted copies of the channel-major input (three are not enough: the row
  stride is not a multiple of 4, so all four phases occur) and unrolls `q` by 4,
  so every tap is a plain aligned `lqc2` and the weight quadwords load as
  sequential cache lines.  It is numerically correct (max relative error ~4e-3
  against a scalar reference, which is the VU multiplier's documented 1-bit
  rounding accumulated over the taps), but it is slower than leaving the 3x3
  layers on `Convolution_mips`:

  | | squeezenet | mobilenet |
  |---|---|---|
  | 3x3 on `Convolution_mips` (current) | 3669 ms | 502 ms |
  | 3x3 on the phase-aligned kernel | 2904 ms | 803 ms |
  | both models fully generic (baseline) | 1732 ms | 1197 ms |

  Squeezenet improves (3669 -> 2904) but stays well behind the generic loop
  (1732), and mobilenet regresses by 300 ms, so the kernel stays compiled and
  gated off.  The four full-frame copies plus the still instruction-bound inner
  loop (one ACC, so every MAC is `vmul` + `vadd`) cost more than the `vmr32`
  and mask work they removed.  The remaining dense-conv lever is the
  "squeeze" 1x1 case (`outch << inch`), where the pointwise kernel's
  O(inch*size) pixel-major transpose does not pay off.
* **VU1 pipeline.**  `ps2_vu1_gemm.{vsm,c,h}` is a working 4xKx4 fp32 engine
  (K = 1..256 validated) but blocking.  The research prototype of the
  double-buffered version (weights resident in VU memory, tiles streamed,
  command mailbox instead of repeated `MSCAL`) measured **1.73x at K=64 and
  1.84x at K=256** over the blocking driver - see
  [`ps2-vu-cooperation.md`](ps2-vu-cooperation.md).  Promoting it into
  `src/layer/mips/` and wiring it into the 1x1 convolution is the next step;
  it only pays off for `K >= 128`.
* **int8 kernels.**  `squeezenet_int8` / `mobilenet_int8` are unchanged.  The
  int8 layers currently get their speed from `Convolution_mips`' packed /
  winograd pipelines.  An EE MMI dot product is now wired in
  (`convolution_packed_int8_ps2.h`: `pextlb` widens eight bytes to halfwords,
  `psrah` sign-extends them, one `phmadh` covers eight input channels) but it
  is off by default.  A plain int8 kernel measured about 3.3x slower than the
  scalar packed path, so `convolution_packed_int8_ps2` returns false unless
  built with `-DNCNN_PS2_INT8_MMI` (set `NCNN_PS2_EXTRA_FLAGS`, do not pass
  `-DCMAKE_CXX_FLAGS`), and even then it only handles `elempack == 1` with both
  channel counts a multiple of 8.  The widening has to live in one asm block:
  `phmadh` writes 128 bits and only the low 64 are addressable as a C scalar,
  so the four pair-sums are reduced inside the block.
* **SPR on real hardware.**  Staging weights in the scratchpad showed no gain
  under PCSX2 but may help on a real console; worth re-measuring there.
* **GS, for int8 pointwise work, not for GEMM.**  The GS is fixed-function,
  but the pixel pipes are a usable ALU if the data is stored as pixels.  A
  plane of int8 lives in the 4 MB eDRAM as a PSMCT32 / PSMCT8 image.  A
  fullscreen sprite sent over GIF walks every pixel on up to 16 pipelines, so
  the loop itself is the draw.  Two hardware paths cover the arithmetic:

  * Alpha blend is `(((A - B) * C) >> 7) + D`.  A, B and D each come from the
    source colour, the framebuffer colour, or zero.  C is the source alpha, the
    framebuffer alpha, or the 8-bit FIX register.  `COLCLAMP = 1` saturates to
    0..255; `COLCLAMP = 0` keeps the low 8 bits.  The texture function has the
    same `>> 7` (modulate is `(texture * fragment) >> 7`, highlight adds the
    fragment alpha on top).  There is no multiply of two colour channels, and
    every product discards its low 7 bits.  That rules out GEMM and the int8
    convolution dot product: those need the full `int8 * int8` and a wide
    accumulator, which is what `phmadh` on the EE is for.
  * An 8-bit texture (PSMCT8) indexes a 256-entry CLUT and writes up to 32 bits
    back.  Any function of a single int8 is exact, because the table has one
    slot per input bit pattern.  Signed and unsigned are the same 256 entries
    read differently.

  The int8 ops that fit, and the ones that do not:

  | op | mechanism | exact? |
  |---|---|---|
  | ReLU, Clip, ReLU6, Abs, threshold | one PSMCT8 draw, the function baked into the CLUT | yes |
  | per-tensor requantize of a value that is already 8-bit (`clip(round(x * scale) + bias)`) | same CLUT.  The int32 accumulator has to be narrowed on the EE first; it does not fit a colour channel | yes, once the input is 8-bit |
  | per-channel requantize | one CLUT holds one scale, so one draw per channel, reloading 1 KB of table.  Pays only when the plane is much larger than 256 pixels | yes, setup-heavy |
  | Sigmoid, Swish, HSigmoid and the other int8 LUTs | the CLUT is the LUT ncnn already uses | yes, at the table's own rounding |
  | saturating or wrapping add/sub of two unsigned planes | blend with `FIX = 0x80`, so `* C >> 7` is the identity, and `COLCLAMP` picks saturate or wrap.  Sub is `A = framebuffer, B = source` | yes for unsigned.  Signed does not, because a biased `a+128` plus `b+128` overflows 255 and the clamp destroys it |
  | per-tensor bias | highlight texture function, fragment colour 128 and the bias in the fragment alpha, one value for the whole sprite | yes |
  | elementwise max | store the value in Z (PSMZ32) and draw the second plane with `ZTST = GREATER`.  Signed needs a +128 bias so the unsigned compare matches | yes |
  | spatial max pool | one draw per window offset, shifted so the texels land on the same pixel, same Z test.  A 2x2 pool is four sprites.  There is no `LESS` test, so min is max of `255 - x` with an invert pass on each side | yes |
  | elementwise mul, int8 conv, GEMM | the only multiply is `>> 7`.  Usable for a Q7 scale (128 means 1.0), not for a product whose low bits a later requantize shift still needs | no |

  Multipass is free in the sense that the framebuffer of one sprite is the
  texture of the next (ping-pong between two eDRAM buffers), so a max pool or
  a stack of blends is several draws and one transfer back.  The transfer is
  the constraint: local-to-host crosses GIF, and it is only worth paying for
  when the GS did more than copy the bytes.  A CLUT ReLU on a tiny blob loses
  to the EE; the same ReLU on a full-resolution map, or a 2x2 max pool that
  finishes before the readback, is the case to measure.  PCSX2 does not model
  the pixel pipes or the bus, so this cannot be scored the way the MMI
  selftest was.

  Textures are limited to 1024 on a side, so a plane bigger than that has to
  be tiled.  Host-to-local copies are still the right tool for the pure moves
  (Unfold, Padding's `copy_make_border_image` once `src.w >= 12`, Flatten,
  ShuffleChannel, Packing, Crop); `Mat::create` only allocates.  None of the
  GS path is implemented.
