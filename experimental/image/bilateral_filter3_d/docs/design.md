# BilateralFilter3D 设计说明

## 1. 数学契约

对连续 ND Tensor `x`、必选属性 `kernel_size`(k)、`sigma_space`、`sigma_color`，输出 shape、dtype
和布局与输入一致。支持 Rank 3/4/5：最后三维为 D/H/W，前导维乘积记作 `nVolume`，每个体数据独立处理。
语义**参考** MONAI 的双边滤波实现，但边界处理不同（本算子为有效邻域截断 + 重归一化，
MONAI 为逐维 replicate 寻址 + 全窗口分母），窗口来源与通道语义亦有差异，
见第 1.1 节。

```text
r = k / 2
y[z,y,x] = sum(w * v) / sum(w),  (dz,dy,dx) in [-r,r]^3 且 (z+dz, y+dy, x+dx) 在体内
w        = exp(-( (dz^2+dy^2+dx^2) * inv2ss + (v - c)^2 * inv2sc ))
inv2ss   = 0.5 / sigma_space^2      (Host 侧饱和到 1e30)
inv2sc   = 0.5 / sigma_color^2      (同上)
```

边界为**有效邻域截断 + 重归一化**：越界邻域既不进分子也不进分母。中心体素自身的偏移为 0、色差为 0，
权重恒为 1，因此分母恒 ≥ 1，除法不会遇到 0 分母。

所有累加在 FP32 UB 内完成；FP16 输入在搬入后 Cast 到 FP32，输出时以 `CAST_RINT` 写回。

## 1.1 与 MONAI 的差异（重要）

本算子的语义**参考** MONAI 的双边滤波实现，但**不是**其等价实现。MONAI 侧对应的是
`monai.networks.layers.filtering.BilateralFilter`（`torch.autograd.Function`，需编译 `monai._C`
扩展），**不存在** `BilateralFilter(spatial_dims=3)` 这一调用形式：MONAI 没有 `spatial_dims`
参数（维数由 `input.dim() - 2` 推断），参数名为 `spatial_sigma` / `color_sigma`。已知差异共三处。

**① 边界处理不同（边界邻域非均匀时结果不同）**

MONAI 的精确路径（`fast_approx=False`）逐维把邻域索引 clamp 到边界，并对全部 `windowSize^3`
个偏移**无条件**累加权重，即越界偏移折叠到边界体素被重复计入
（`monai/csrc/filtering/bilateral/bilateralfilter_cpu.cpp`）：

```cpp
int neighbourIndexClamped = std::min(desc.sizes[i] - 1, std::max(0, neighbourIndex));
...
weightSum += totalWeight;   // 循环体内无条件累加, 无 bounds check
```

本算子（见第 3 节）用掩码把越界邻域的**权重清零**，越界点分子分母都不进，即
「有效邻域截断 + 重归一化」。

一维可手算反例：`W=3`、`x=[0,1,0]`、`kernel_size=3`（`r=1`）、`σs=σc=1`，取左边界体素 `x[0]=0`。
空间权重 `exp(-0.5)=0.6065307`，中心权重 `1`；色差为 0 时色权重为 1，色差为 1 时色权重
`exp(-0.5)=0.6065307`：

| 语义 | 分子 | 分母 | 输出 |
|---|---|---|---|
| 本算子（截断 + 重归一化） | `1 × 0.3678794` | `1 + 0.3678794` | **0.2689414** |
| MONAI（replicate + 全窗口） | `1 × 0.3678794` | `0.6065307 + 1 + 0.3678794` | **0.1863237** |

两者相差约 31%（以本算子为基准），差异**全部**来自左越界邻域是否计入分母。

**② 窗口来源不同**

MONAI 没有 `kernel_size` 参数，窗口由 `spatial_sigma` 推导：
`windowSize = ceil(5 * spatial_sigma) | 1`（强制奇数），半径 `floor(windowSize/2)`。
本算子的窗口由必选属性 `kernel_size` 给定。两者的空间权重函数形式一致
（`exp(-d^2/(2*sigma_space^2))`，中心权重恒为 1），但**支撑集不同**：除非取
`kernel_size = ceil(5 * sigma_space) | 1`，否则结果在全域（不只是边界）都不同。

| sigma_space | MONAI 窗口 | 说明 |
|---|---|---|
| 0.8 / 1.0 | 5 | 取 `kernel_size=5` 可对齐支撑集 |
| 1.2 | 7 | 取 `kernel_size=5` 则支撑集不一致 |
| 2.0 | 11 | 超出本算子 `kernel_size <= 9` 的上限，无法对齐 |

**③ 通道语义不同**

MONAI 对多通道做**联合**值域加权：色差在全部通道上求和（`colorDistanceSquared += diff * diff`
跨 `channelCount` 循环），且一套 `totalWeight` 共享给所有通道。本算子把前导维视为**独立体数据
批量**处理，各体数据之间不耦合。故 rank >= 4 且前导维乘积 > 1 时两者结果不同；只有单通道
（前导维乘积为 1）时在该维度上等价。

> **注：MONAI 的默认路径不是上述精确实现。** `BilateralFilter.forward(ctx, input,
> spatial_sigma=5, color_sigma=0.5, fast_approx=True)` 的默认 `fast_approx=True` 走
> **permutohedral lattice 近似**（`bilateralfilter_{cpu,cuda}_phl.*`），其边界行为与本算子
> **未做对比**。上面第 ① 条的对照仅适用于 `fast_approx=False` 的精确路径。

## 2. 并行划分与数据流

划分单位是**行**：一行 = W 个连续体素，`totalRows = nVolume * D * H`。核间按 grid-stride 领行：

```text
for (row = GetBlockIdx(); row < totalRows; row += usedCoreNum) ProcessRow(row);
```

> **注意**：步长必须用 Tiling 下发的 `usedCoreNum`，不能用 `GetBlockNum()`。本算子的启动方式下
> `GetBlockNum()` 返回 1，会导致所有核以步长 1 重叠遍历全部行、互相覆盖输出（真机实测踩坑）。
> 该行为已由 host UT 的 `FewerRowsThanCoresClampsBlockDim` 用例在 Tiling 侧锁定。

每行再按 `chunkLen`（8 的倍数，由 Tiling 按 UB 预算推导，上限 1024）分块，块内全部是整块全宽
向量指令：

```text
对每个块 (cb, cl = min(chunkLen, W - cb)):
  GM 中心块 --(DataCopyPad)--> cCh[0..cl)          (FP16 则 Cast 到 FP32)
  若该块触边: 为每个 dx 构造掩码(见下)
  accD = accN = 0
  对每个有效 (dz,dy,dx):
    GM 邻域 --(DataCopyPad, 源地址 = nbRowBase + cb + dx)--> nbCh[0..cl)
    tmp = exp(-(nbCh - cCh)^2 * inv2sc) * sw[dz,dy,dx]
    若该块触边: tmp *= mask[dx]
    accD += tmp;  accN += tmp * nbCh
  div = accN / accD  --> outQue --> GM y[cb]
```

**关键手法：邻域的移位在 GM 地址里完成。** `DataCopyPad` 的 GM 源地址可以任意偏移，所以从
`nbRowBase + cb + dx` 处加载一整块落到缓冲的 lane 0 后，`nbCh[j]` 天然就是输出 lane `j`
（即输出列 `cb + j`）所需的邻域值。UB 侧因此永远是零偏移的完整张量，不触碰第 3 节任何约束。

空间权重表 `sw[k^3]` 与 `ramp[j] = j` 与数据无关，每核只在 `Init` 阶段算一次。

## 3. 910B 向量指令的对齐约束

这组约束来自 910B 真机实测，是本实现所有「绕路」写法的唯一理由。**改动 kernel 前务必先读本节**，
否则极易踩回去——这些失败模式多数是静默的（不报错、只是结果错或读到零）。

| 约束 | 失败表现 |
|---|---|
| 向量指令的 UB 操作数（含 `LocalTensor` 偏移视图）必须 32 字节对齐 | 内核直接中止，输出全部丢失 |
| 经**非零偏移视图**做向量写（即使该偏移已 32B 对齐） | 静默失效，写入不生效 |
| 整行 `DataCopyPad` 复用 8 元素的瓦片队列 | 越界写坏相邻缓冲，表现为散布的数据污染 |
| 输出复用 VECIN 队列（而非 VECOUT） | 队列同步事件错乱，后续 MTE2 加载静默返回零 |
| 手工 `FetchEventID` + `SetFlag/WaitFlag` 做 S↔V 同步 | 与 TQue 内部事件冲突，破坏 MTE2→V 同步，加载静默返回零 |
| 用 `PipeBarrier<PIPE_V>` 充当标量↔向量的同步 | **不同步跨管道依赖**，偶发读到旧值；详见下文 |

**`PipeBarrier<PIPE_V>` 只同步 V↔V。** 标量单元读 V 管道刚写入的 UB（`GetValue`）或标量写入后
立刻被向量指令读取（`SetValue`），都是跨管道依赖，`PIPE_V` 屏障对它们无效。这类失败是**非确定性**
的：真机上同一形状重复运行时而通过时而失败，而且只有标量循环**最后写入**的那几个 lane 出错
（落地时间最短）。排查时的决定性证据是同一条命令里相同形状跑出 `OK` 和 `MISMATCH` 两种结果。
正确做法是 `PipeBarrier<PIPE_ALL>()`——不要自行分配 event，那会撞上上一行的坑。

由此导出的实现规则：

1. 块宽 `chunkLen` 恒为 8 的倍数（32 字节对齐），块基址恒为缓冲的 lane 0；
2. 邻域元素**从 GM 预移位加载**，移位在 GM 地址里完成，UB 侧永远是零偏移的完整张量；
3. 越界 lane 由掩码在**权重**上清零，而不是去清数据（清数据需要标量写，与后续 V 管道读存在
   S→V 竞争）；
4. 所有向量写的 dst 都是完整张量，不使用偏移视图。**掩码表因此拆成 `2r+1` 个独立 `TBuf`**，
   而不是一个 `(2r+1) × chunkLen` 的大 buffer 按行索引——往第 `dx+r` 行写就是偏移写，会静默失效；
5. 掩码用纯向量算术从 `ramp[j] = j` 构造，不用标量逐元素写（后者每行每块要 `(2r+1) × chunkLen`
   次 `SetValue`，比它要省下的向量指令还贵）：

   ```text
   a    = clamp(ramp - lo + 1, 0, 1)   -> j >= lo 时为 1
   b    = clamp(hi - ramp,     0, 1)   -> j <  hi 时为 1
   mask = a * b                        -> lo <= j < hi 时为 1
   ```

   `clamp` 用 `Min`/`Max` 配预置的全 1 / 全 0 常量张量，不依赖 `Mins`/`Maxs` 标量变体。

**免掩码的快路径及其判据。** 只有触边的块需要掩码，内部块可省掉那一条 `Mul`。判据必须是

```text
needMask = !(cb >= r && cb + cl + r <= W)
```

而**不能**简化成 `cb > 0 && cb + cl < W`。反例：`cb + cl = W - 1`、`dx = 2` 时块内末尾 lane 的
邻域已越界，按后者会被判为无需掩码，于是把越界邻域当有效值累加——静默算错。该判据在
`k=9`（r=4）配 `chunkLen=8` 的最紧情形下有专门的 UT 覆盖。

**唯一的标量退化路径**是 GM 源地址为负，即 `nbRowBase + cb + dx < 0`。这要求 `nbRowBase < r`，
只有全局首行且 `dz=dy=0` 时成立（W 极小时另有紧邻张量起始的几行）。此时连续加载无法落位，
退化为标量逐元素组装。它在整个 kernel 调用中最多发生 `r` 次，对吞吐无影响。

启动时 UB 初始位型任意（可能是 NaN），而加载长度不足一整块时尾部 lane 的残留会以
`NaN * 0 = NaN` 穿透掩码，因此队列槽位与中心/邻域缓冲在首次使用前统一清零。

## 4. Tiling

Host 侧从平台读取 AIV core 数与 UB 容量：

- `totalRows = nVolume * D * H`，`usedCoreNum = min(totalRows, aivCoreNum)`；
- `inv2ss`/`inv2sc` = `0.5/sigma^2`，在 double 下计算后饱和到 `1e30` 再降为 FP32：
  `exp(-x)` 在 `x > ~88` 时已下溢为 0，所以饱和值仍等价于「权重恒为零」，同时避免零距离偏移上的
  `0*inf` NaN；
- Tiling key 只编码输入 dtype（0 = FP16，1 = FP32）；
- `workspace[0]` 取平台的 `GetLibApiWorkSpaceSize()`，kernel 不使用用户 workspace；
- `chunkLen` 由 UB 预算反解，取 `min(align8(W), 1024, 预算允许的最大 8 倍数)`。

每个 chunk 元素的 UB 占用：

```text
perChunkElem = (13 + (2r+1)) * 4 + 4 * sizeof(dtype)   字节
               ^^              ^^^^^^^^   ^
               13 份 fp32 工作缓冲        输入队列 2 槽 + 输出队列 2 槽
               (2r+1) 份 fp32 掩码缓冲
```

13 份工作缓冲为：中心块、邻域块、两个累加器、临时、除法结果、`ramp`、掩码构造的 A/B、
负偏移修复的 head/fix、常量全 1、全 0。再加与块宽无关的 `k^3` 个 fp32 空间权重表和 2 KB 框架余量。
`k=9`、`chunkLen=1024`、FP32 时约 109 KB，在 910B 可用 UB 内。

与上一版不同，这个预算检查**不再是死代码**：它真正决定 `chunkLen`。改动缓冲布局时必须同步这里的
系数，否则设备侧会超出 UB。`BF3D_CHUNK_MAX` 在 host 与 device 两侧各有一份定义，必须一致。

`BilateralFilter3DTilingData` 在 host 与 device 两侧各有一份定义（`op_host/..._tiling.h` 与
`op_kernel/..._tiling_def.h`），**字段顺序与类型必须严格一致**——device 侧按 `uint32_t` 字长逐字
拷贝，顺序错位不会报错，只会读到错误的数值。

## 5. 校验与边界

Host/Tiling 拒绝：dtype 非 FP16/FP32、rank 非 3/4/5、存在为 0 的维度、`D > 1000`、`H/W > 4096`、
前导维乘积 `> 12800`、`kernel_size` 非 3/5/7/9、`sigma_space`/`sigma_color` 非有限正数。

InferShape 对**负数维度（未知 shape 占位符）放行**，只拒绝显式的 0。该算子声明了
`DynamicShapeSupportFlag(true)` 并提供 `[-2]` 的 binary 配置，若 InferShape 连 `-1` 一起拒绝，
图模式下无法完成 shape 推导。

## 6. 测试覆盖

| 层次 | 规模 | 内容 |
|---|---|---|
| InferShape UT | 16 例 | rank 3/4/5 透传、全/半动态 shape 透传、rank 2/6 与 0 维拒绝、属性非法值拒绝 |
| Tiling UT | 22 例 | 全部规格上界与属性校验分支；tiling key 的 dtype 选择；`usedCoreNum` 的行数钳制 |
| Kernel UT（CPU 孪生） | 16 例 | W 对齐/残尾/小于一块；radius 超出 D/H 的双侧截断；多块（含免掩码内部块、残尾末块、`r` 为块宽一半的最紧判据）；多核 grid-stride；多 volume 边界；sigma 饱和与极大 sigma |
| 真机 E2E | 10 例 | 910B 上对比 double 精度 host golden，FP32 容差 5e-4、FP16 容差 2e-3 |

Kernel UT 的 golden 是独立的 double 精度实现，不复用 kernel 的任何算术。

> Kernel UT 受构建框架限制：它按 `<repo>/image/<op>` 的固定深度推算 kernel 源码位置，
> `experimental/` 多出的一层会使路径错位、kernel 源码不参与编译、链接报
> `undefined reference to 'bilateral_filter3_d'`。需复制到扁平布局的仓库副本中执行
> （同 `gamma_correction`，与算子代码无关）。

## 7. 性能

本算子是**指令发射受限**，不是访存受限：耗时精确正比于 `k^3`，与搬运量无关。

初版实现受第 3 节的对齐约束所限，把向量指令长度压到 8 个 FP32（硬件单条可处理 64 个），且每条
向量指令后跟一个 `PipeBarrier<PIPE_V>`。以下为初版瓦片实现的历史实测基线（910B，FP32），
用于与当前分块实现对照：

| 规模 | k | median | 等效带宽 |
|---|---|---|---|
| 64³ | 5 | 21.9 ms | 0.10 GB/s |
| 128³ | 5 | 175.2 ms | 0.10 GB/s |
| 128³ | 3 | 42.1 ms | 0.40 GB/s |

按条数核算：`128³ k=5` 每行 16 个瓦片 × 125 邻域 × 约 10 条 = 2 万条，全张量 3.28 亿条，摊到
48 核约 683 万条/核；175 ms 对应每条约 25.7 ns ≈ 38 cycle，而 8 元素向量操作本身只需 1~2 cycle，
其余全是流水停顿。

当前实现改为行内分块（第 2 节），每个邻域在整块上只发 8~9 条向量指令，指令与 barrier 条数同比
下降。`W = 128` 时整行是一块，每行从 2 万条降到约 1125 条，按条数估算约 **18 倍**。

当前分块实现的已有实测结果如下，与 [README 性能表](../README.md#性能) 一致：Ascend 910B、
FP32，`sigma_space = sigma_color = 1.0`，使用 `--bench` 预热 3 次后测量 20 次并取中位数。
计时使用设备侧 `aclrtEvent` 包围 `RunOnce`，不包含输入准备与输出回传。

| 规模 | k | 分块版 median |
|---|---|---|
| 64³ | 5 | 2.254 ms |
| 128³ | 5 | 9.029 ms |
| 128³ | 3 | 2.123 ms |

按历史基线的已展示精度计算，`128³ k=5` 的加速比为 `175.2 / 9.029 ≈ 19.4`。
前述约 18 倍是指令条数估算，19.4 倍为该场景的实测时延比。

基准命令（依次对应上述三组场景）：

```bash
./test_aclnn_bilateral_filter3_d --bench 64 64 64 0 5 1.0 1.0 20
./test_aclnn_bilateral_filter3_d --bench 128 128 128 0 5 1.0 1.0 20
./test_aclnn_bilateral_filter3_d --bench 128 128 128 0 3 1.0 1.0 20
```

参数为 `D H W dtype(0=fp32,1=fp16) k σs σc 迭代次数`。

仍未做的下一步：削减冗余的 `PipeBarrier<PIPE_V>`。同一 PIPE_V 内的指令本身顺序发射，这些 barrier
大概率多余，但第 3 节列出的失败模式正源于同步相关的改动，需单独验证后再动。
