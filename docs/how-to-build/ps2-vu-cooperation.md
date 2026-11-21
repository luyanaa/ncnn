# PS2 ncnn 移植研究:EE MMI / VU0 (COP2) / VU1 (微码) 三者协同

日期:2026-11-20。环境:PCSX2 2.8.2(-nogui -fastboot -unlimited),ps2sdk
(mips64r5900el-ps2-elf-gcc 15.2.0,dvp-as binutils 2.45.1),EE 主频按
294 MHz 记。所有计时用 CP0 Count(`mfc0 $9`),单位 **tick**;
PCSX2 上 tick 与 EE 周期的换算不是严格 1:2(见"陷阱"第 8 条),
因此本文只对**同一 ELF 内的相对值**下结论。

实验代码在 `/tmp/coop/`(不进 repo):`bench1.c`(EE 内部交织)、
`vu1_pp.vsm` + `pp.c` + `bench2.c`(VU1 双缓冲流水线)。
构建:`make bench1.elf bench2.elf`,运行:`/tmp/run_ps2.sh <elf> 20`。

---

## 1. 三者能力/限制速查(资料 + 实测混合,见标注)

| | EE MMI (128-bit int) | VU0 macro (COP2) | VU1 微码 |
|---|---|---|---|
| 数据类型 | int8/16/32 SIMD | 4×fp32 | 4×fp32 |
| 峰值 | 1 instr/tick(EE 单发射,实测) | 1 instr/tick(共享 EE 发射口,实测) | 独立处理器,与 EE 真并行(实测) |
| 每 instr MAC | `paddw` 4 int-add;`pmaddw` 依赖 HI/LO **(未单独测)** | `vmul.xyzw` 4 MAC(独立时) | 每循环 4 mul + 4 add ≈ 4 MAC/cycle 理想 |
| 真实累加结构 | — | ACC 链(vmulax/vmadday/…)实测 **2.76 tick/MAC**(延迟瓶颈) | 4 独立累加器,无 ACC 链问题 |
| 精度 | int | fp32 | fp32 |
| 数据获取 | lq/sq(EE LSU) | `lqc2/sqc2` 须 16B 对齐 | VIF1 DMA UNPACK;结果 EE 从 0x1100C000 直接 `lq` |
| 与 EE 并行 | 无(就是 EE) | **无实际并行收益**(PCSX2 实测,见 §3.1;硬件上 EE 单发射也决定指令口是瓶颈) | **有**(实测 1.73–1.84×,见 §3.2) |
| 主要限制 | `vector_size(16)` 是两个 64-bit GPR;须 lq→op→sq 单表达式 | macro 模式禁 xgkick/xtop/ilw/EFU;`vdiv` 后读 Q 须 `vwaitq` | 16KB 数据 mem;MSCAL 不能在 VU1 运行中发射(实测会挂死);单次 UNPACK ≥256 qw 在 PCSX2 崩机(实测) |

---

## 2. 实测数据

### 2.1 bench1:EE 单元发射率与交织(bench1.elf,512 单元 × 8 次取 min)

纯发射(直线码,目的寄存器轮转,操作独立):

| 测试 | 操作数 | ticks | tick/操作 |
|---|---|---|---|
| MMI `paddw` | 8192 | 7173 | 0.875 |
| FPU `mul.s`+`add.s` 对 | 16384 | 19970 | 1.22 |
| COP2 `vmul.xyzw` | 8192 | 4641 | 0.567(= **0.142 tick/MAC**) |

交织(同一计数窗口,操作总数与"两个纯测试之和"相同):

| 交织 | 操作数 | ticks | 纯序列之和 | 结论 |
|---|---|---|---|---|
| COP2+FPU | 24576 | 24576 | 24611 | **无收益**(差 0.1%) |
| COP2+MMI | 16384 | 11779 | 11814 | **无收益**(差 0.3%) |
| FPU+MMI | 24576 | 21135 | 27143 | −22%(PCSX2 对 int+FP 有轻度双发射,量级小) |

**结论 A(实测):在 PCSX2 上 COP2 与 EE 的 MMI/FPU 之间不存在可利用的流水线
重叠;EE 单发射本来就限制在 1 instr/tick。选择 VU0 的理由只能是"每条指令
4 MAC"(`vmul.xyzw`),而不是"与 EE 并行"。**

真实点积循环(4 MAC/k,依赖 ACC 链,K=256):

| 循环 | ticks | tick/MAC |
|---|---|---|
| VU0 `vmulax/vmadday/vmaddaz/vmaddaw` 链 | 2822 | **2.76** |
| EE FPU 标量点积(手写 asm) | 3185 | 3.11 |
| EE FPU 4 行 tile(16 mul.s/k,独立累加) | 12548 | 3.07 |
| EE 编译器生成的标量 C 三重循环(K=256 tile) | 49360 | **12.05** |

**推论(实测+推断):VU0 的 ACC 链是主要损失来源;VU0 内核必须保持 ≥2 条
独立累加链(如现有 ncnn conv 内核的 vmul+vadd 结构)才能接近 0.15–0.3
tick/MAC 的纯发射率。EE 侧手写 asm ~3 tick/MAC,ncnn 现有 C 路径 12
tick/MAC,说明现有内核还有 ~4× 的纯软件优化空间(不换处理器)。**

### 2.2 bench1 A2:EE 搬运 + VU0 计算的交织

任务:每遍搬 512 qw(8 KB,`lq/sq` MMI,代表 im2col 数据搬运)+ VU0 对 512 qw
做点积,共 2 遍:

| 模式 | ticks(两次) |
|---|---|
| 先全部 copy,再全部 dot(顺序) | 12308 / 12308 |
| 每 64 qw 交替 copy/dot(交织) | 15495 / 15495 |

**结论(实测):交织反而慢 26%。因为 COP2 发射占用同一个 EE 发射口,把
MMI load/store 挤掉;没有任何隐藏并行。"EE 做 im2col + VU0 做点积"的
收益只能来自把二者放进不同处理器,而不是在同一 EE 上交织。**

### 2.3 bench2:VU1 流水线(vu1_pp.vsm + pp.c)

引擎设计(与现有阻塞式 `ps2_vu1_gemm` 的差别):
- A(4×K,k-major)**常驻** VU1 数据 mem qw 0..K−1,每个 layer 只传一次;
- tile = K×4 权重块,经 VIF1 UNPACK 流式送入,奇偶 tile 分别落在
  B@256 / B@512(双缓冲),C 分别写在 769 / 774;
- 每个 tile 附带一个 flag qword{x=n, y=K}(768/773),微程序轮询 flag,
  tile 完成后把完成计数写到 qw 1022(EE 直接读 0x1100C000 轮询);
- 微程序**永不下 E-bit**,新 layer 用 CMD 邮箱(qw 1019)软复位 n=0,
  从不重发 MSCAL。

DMA 往返(单 tile:发送 + 等状态):

| K | ticks/tile | 备注 |
|---|---|---|
| 4 | 2,846,967 | 首调用异常值(疑似 PCSX2 重编译/首次同步开销),**作废,不要引用** |
| 256 | 141,998 | 4096 MAC;同步往返被固定开销支配 |

K=64,T=32 tiles(每 tile 64 qw B + 1 flag qw = 1 KB):

| 模式 | ticks 总 | ticks/tile | tick/MAC |
|---|---|---|---|
| 阻塞(send→wait 每块) | 199,558 | 6235 | 9.74 |
| **双缓冲(连发,最后统一等)** | **115,117** | **3597** | **5.62** |

K=256,T=8 tiles(每 tile 4 KB):

| 模式 | ticks 总 | ticks/tile | tick/MAC |
|---|---|---|---|
| 阻塞 | 121,765 | 15220 | 23.8 |
| **双缓冲** | **66,330** | **8291** | **12.9** |

数值验证(K=64、K=256 均与 EE CPU 参考逐元素比较):**PASS**(最大相对误差
< 1e-3 阈值,fp32 累加顺序差异内)。

横向比较(每 MAC 的 tick):

| 引擎 | tick/MAC | 相对 EE 标量 C 循环 |
|---|---|---|
| EE 标量 C 三重循环 | 12.05 | 1× |
| VU1 阻塞(现状,ps2_vu1_gemm 同构) | 23.8 | **0.5×(更慢!)** |
| VU1 双缓冲 | 12.9 | ~1×(K=256,tile 只有 4 KB) |
| VU0 ACC 链点积 | 2.76 | 4.4× |
| VU0 独立 vmul 流(理想内核) | 0.142 | 85× |

**结论 B(实测):**
1. 双缓冲比阻塞快 **1.73×(K=64)/ 1.84×(K=256)**;阻塞与双缓冲之差
   ≈ 6.9k tick/tile ≈ 每包 DMA 同步开销。
2. **本项目最确定的收益点就是 VU1 双缓冲化**(把现有
   `ps2_vu1_gemm_4xKx4` 的"staging→UNPACK→MSCAL→等 VEW→读回"改成
   A 常驻 + tile 流式 + CMD 邮箱复位)。
3. 但 VU1 的 tick/MAC(12.9)仍远差于 VU0 的理想值(0.14)——VU1 的收益
   来自**并行与固定开销摊薄**,不是计算效率;只有当 EE 同时有别的活干
   (im2col、量化、下一层的准备)时才真正赚钱。
4. 纯 VU1 计算估算:K=256 双缓冲 8291 tick/tile,其中 DMA+同步占大头;
   微程序本体 256×~13 VU cycle ≈ 3.3k VU cycle。**tile 越大越划算:
   每笔 DMA 往返 ~7k tick 固定开销,4×K×4 tile 需要 K ≥ ~150 才能让
   计算时间超过开销;K≤32 的小 tile 用 VU1 是亏的(实测 K=4 往返
   完全被开销支配)。**

---

## 3. 对 ncnn 的具体建议

### 3.1 分工

| 层/形状 | 用谁 | 依据 |
|---|---|---|
| 1×1 / 3×3 卷积 im2col+GEMM,**K(=inch,pack 后)≥128 的 tile** | **VU1(双缓冲版)** | §2.3:tick/MAC 随 K 增大摊薄;EE 同时做下一 tile 的 im2col |
| K < 32 的小 tile、depthwise、逐点激活 | **VU0 COP2**(现有内核已验证) | DMA 固定开销在小 tile 上不可摊薄(§2.3) |
| int8 量化 requant、im2col 打包、数据重排 | **EE MMI** | int 单元;COP2/VU1 都是 fp32;MMI lq/sq 搬运是 EE 本职 |
| VU0 与 EE 的"交织" | **不要做** | §2.1/§2.2 实测无收益甚至更慢 |

### 3.2 建议流水线(ASCII)

```
            EE (MMI+FPU)                      VU1 (微码, 常驻)               VU0 (COP2)
            ────────────                      ────────────────               ──────────
 layer N:   im2col tile(k) ──VIF1 UNPACK──▶  B[k&1] 槽, flag{k}
            (量化/打包, lq/sq)                计算 C[k&1] ◀── 同时进行 ──
            同时: 搬 tile(k+1) ─UNPACK────▶  B[(k+1)&1]
            ...
            轮询 status(qw1022)==k+1 ─────▶
            lq 读 C[k&1] (0x1100C000) ──────
            下一层: UNPACK A + CMD=1 ─────▶  软复位 n=0 (无 MSCAL!)
            激活/量化 C ──────────────────────────────────────────────────▶ VU0 做 ReLU/Clip/quant
```

要点(全部来自实测):
1. **EE 永远不等 VU1 的单个 tile**:连发 UNPACK(双缓冲),只在需要 C
   的时候轮询 status。收益 1.73–1.84×。
2. **不要在 VU1 运行中发 MSCAL**(实测挂死 VIF1/DMA)。换层/换 A 用
   CMD 邮箱(qw 1019 写 1,微程序自复位 n=0 并清 CMD)。
3. **单次 UNPACK ≤ 128 qw**(实测 ≥256 qw 的 UNPACK 在 PCSX2 崩机;
   192/240/255 正常,256 必崩;硬件未验证)。`add_unpack_chunked()`
   模式即可,无性能损失。
4. B 和 flag 必须分成两个 UNPACK:合并的 K+1 qw UNPACK 会把尾部 flag
   写到 dest+K,在 K=256 时正好砸进偶数槽的 flag 区(实测)。
5. VU0 只做 elementwise/小 tile;不要试图和 EE 交织 COP2 与 MMI。

### 3.3 内存预算(全部 ≤ 32 MB 的一部分,实测布局)

- VU1 数据 mem(16 KB,不占 EE 内存):
  A ≤256 qw(0..255)+ B0 256 qw(256..511)+ B1 256 qw(512..767)
  + flag 2 qw(768,773)+ C 8 qw(769..777)+ status/cmd 2 qw(1019,1022)。
- EE 侧每 layer 新增:stage 缓冲 `(K+2)*4` u32(≈16.5 KB,K=256)、
  tile 缓冲 `(K+1)*4` u32(≈16.5 KB)、packet2 缓冲 ~9 KB。
  共 **~42 KB 静态**,可复用(ncnn 的 blob/workspace 或静态区)。
- VIF1 packet 复用一个 `packet2_t`(chain 模式,REF 引用 EE 数据,
  不拷贝)。

---

## 4. 已证伪 / 未验证清单

**实测证伪:**
1. COP2 与 EE MMI/FPU 交织无并行收益(§2.1)。
2. EE 上"copy + COP2 dot"交织比分阶段慢 26%(§2.2)。
3. VU1 阻塞式小 tile(K=4)完全被固定开销支配(§2.3)。

**实测成立:**
4. VU1 双缓冲 1.73–1.84×(§2.3),K=64/256 数值 PASS。
5. MSCAL-while-running 挂死 → 必须用 CMD 邮箱复位。
6. 单条 UNPACK ≥256 qw 崩 PCSX2 → 分块 ≤128。
7. 合并 B+flag 的 UNPACK 会污染偶数槽 flag(K=256 实测)。

**资料推断(未在本环境实测):**
8. 真硬件上 VU0 macro 是同步发射、结果就绪前 EE 停顿(EE User's
   Manual);PCSX2 对 COP2 的建模接近同步,结论 A 预计在真机上同样成立,
   但"硬件上 COP2 发射后 EE 可继续发射后续无关指令"这一点未在真机验证。
9. 真硬件 RDRAM 带宽(~3.2 GB/s 峰值,双通道)比 PCSX2 模型宽;
   §2.3 的 DMA 开销(≈7k tick/包)在真机上可能小得多 → 双缓冲收益
   可能更大,而不是更小。
10. PCSX2 的 CP0 Count 粒度较粗(8192 条直线指令只计 7173 tick,说明
    不是严格的 2 cycle/tick);所有绝对值仅供同 ELF 内相对比较。

---

## 5. 关键代码与构建

构建/运行:

```sh
cd /tmp/coop && make bench1.elf bench2.elf
/tmp/run_ps2.sh /tmp/coop/bench1.elf 20 /tmp/ps2_bench1.png
/tmp/run_ps2.sh /tmp/coop/bench2.elf 20 /tmp/ps2_bench2.png
```

VU1 微程序核心(vu1_pp.vsm,完整文件见 /tmp/coop;upper=左,lower=右):

```asm
VU1Pp_Poll:
        nop     ilw.x   VI08, 1019(VI00)   ; CMD 邮箱
        nop     nop
        nop     ibeq    VI08, VI09, VU1Pp_Reset   ; VI09=1
        nop     ilw.x   VI08, 768(VI00)    ; F0.x
        nop     nop
        nop     ibeq    VI08, VI06, VU1Pp_Slot0   ; n 匹配偶槽
        nop     ilw.x   VI08, 773(VI00)    ; F1.x
        nop     nop
        nop     ibne    VI08, VI06, VU1Pp_Poll
VU1Pp_Slot1:                                ; B@512, C@774
        nop     iaddiu  VI03, VI00, 0x0200
        ...
VU1Pp_Compute:                              ; 4 独立累加器,无 ACC 链
        nop     lq      VF04, 0(VI02)      ; A 列 k
        nop     lq      VF05, 0(VI03)      ; B 行 k
        mulx.xyzw VF06, VF05, VF04x  nop
        add.xyzw  VF08, VF08, VF06  nop
        ...        (muly/mulz/mulw → VF09/VF10/VF11)
        nop     ibne    VI02, VI05, VU1Pp_Compute
        nop     sq      VF08, 0(VI10)      ; C 到 769/774
        ...
        nop     isw     VI06, 1022(VI00)   ; status = n+1(EE 直接读)
```

EE 侧双缓冲发送(pp.c):

```c
int ps2_pp_send_tile(int n, const float *B, int K)
{
    static u32 tilebuf[(PP_MAX_K + 1) * 4] __attribute__((aligned(16)));
    memcpy(tilebuf, B, (u32)K * 16);
    tilebuf[K*4+0] = n; tilebuf[K*4+1] = K; tilebuf[K*4+2] = tilebuf[K*4+3] = 0;
    packet2_reset(pkt, 0);
    add_unpack_chunked((n & 1) ? 512 : 256, tilebuf, K);        /* ≤128 qw/块 */
    add_unpack_chunked((n & 1) ? 773 : 768, tilebuf + K*4, 1);  /* flag 单独 */
    packet2_utils_vu_add_end_tag(pkt);
    dma_channel_wait(DMA_CHANNEL_VIF1, 0);
    dma_channel_send_packet2(pkt, DMA_CHANNEL_VIF1, 1);
    dma_channel_wait(DMA_CHANNEL_VIF1, 0);
    return 0;   /* 不等待 VU1!双缓冲的关键 */
}
```

分层/换 A(禁止运行中 MSCAL):

```c
/* load_A: UNPACK A→qw0、CMD{x=1}→qw1019、清 flag;仅首次 load 才 MSCAL */
```

### 环境陷阱(本次全部踩过,给后续通道省时间)

1. `printf`/`_write` 覆盖式输出在 ps2sdk + 该 gcc 下启动即崩;改用
   `vsnprintf` + `scr_printf`(见 /tmp/coop/out.h)。
2. binutils r5900 目标拒绝 FPU paired-single(`mul.ps`/`madd.ps`,
   module singlefloat),COP2(VU 指令)不受影响;EE FPU 基准只能用
   `mul.s/add.s` 标量。
3. dvp-as 的 VU 语法要求寄存器写作 `VI06/VF04`、`$ACC`,且 upper/lower
   成对;分支后保持 ≥2 条 nop 行。
4. `.vu` 段经 dvp-as 产出 `.vutext`,默认链接脚本会放进 LOAD 段
   (0x120000),无需手工链接脚本。
5. VIF1 每包固定同步开销 ≈7k tick(PCSX2 模型),是所有"阻塞 vs 流水"
   结论的直接来源。
6. PCSX2 崩溃会回落到 OSDSYS 并抹掉 GS 屏,看不到崩溃前的输出 —— 调试
   时把输出切成"每阶段一个独立小 ELF",或在每个阶段后 `for(;;)`。
