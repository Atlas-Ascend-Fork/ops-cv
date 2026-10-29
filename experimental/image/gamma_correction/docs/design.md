# GammaCorrection 设计说明

## 1. 数学契约

对连续 ND Tensor `x`、必选属性 `gamma`、可选属性 `correctionMode`（默认 0）与 `maxValue`
（默认 0，按 dtype 推导），输出 shape、dtype 和布局与输入一致。支持 Rank 2/3/4，Kernel 将
所有维度展平后按一维连续元素处理。所有中间计算在 FP32 UB 内完成。

值域上界 `M`：`maxValue>0` 时取该值；否则浮点取 1.0，uint8/int8 取 255.0，uint16/int16 取 65535.0。

```text
Mode 0 (纯 pow):    y = pow(x, gamma)                      # 浮点与 torch.pow 等价
Mode 1 (归一化 gamma): t = clamp(x,0,M); y = clamp(pow(t/M, gamma)*M, 0, M)   # NaN/Inf->0
Mode 2 (sRGB 编码):  t = clamp(x,0,M)/M
                     u = (t<=0.0031308) ? 12.92*t : 1.055*t^(1/2.4) - 0.055
                     y = clamp(u*M, 0, M)
```

整数 dtype 在所有模式下统一经 `WriteBack` 写回：NaN 置 0（Mode 0 的负底数非整数指数），
再饱和到 dtype 值域，最后 `CAST_RINT` 窄化。

## 2. 融合数据流

通用浮点路径（Mode 0）为：

```text
GM x -> UB x -> FP32 -> Log -> gamma * value -> Exp -> output dtype -> GM y
```

Mode 1 在其外层包裹 `Clamp(x,0,M) -> Muls(1/M) -> [pow] -> NaN/Inf 清零 -> Muls(M) -> Clamp(0,M)`；
Mode 2 替换 pow 段为 sRGB 双分支（线性段 `12.92*x` + 幂段 `1.055*x^(1/2.4)-0.055`），用
`Compare` 生成阈值谓词、`Select` 混合两分支。所有阶段在同一个 kernel 内完成。与拆分为
Cast/Log/Muls/Exp/Cast 多算子相比，输入只从 GM 搬入一次，输出只写回一次，不创建中间 Tensor，
也没有额外用户 workspace。这就是本工程的算子融合边界；`op_graph/fusion_pass` 不做图模式替换。

参考官方 `Pows` 的标量复用、dtype 分支、多核连续切分、尾块搬运和特殊指数路径，
同时补充任意正 gamma 的 `Log/Exp` 通用路径与图像 gamma 校正的归一化/clamp/sRGB 语义。

## 3. gamma 分支

| gamma | 实现 | 适用模式 |
|---|---|---|
| `1.0` | FP32 透传 | 仅 Mode 0 |
| `2.0` | `Mul(x, x)` | 仅 Mode 0 |
| `0.5` | `Sqrt(x)` | 仅 Mode 0 |
| 其他正整数且不超过 `uint32_t` | UB 内二进制整数幂 | 仅 Mode 0 |
| 其他正实数 | `Exp(gamma * Log(x))` | 全部模式（Mode 1/2 恒走此路径） |

Mode 1/2 始终走通用 `exp(g*ln)` 路径，使归一化后的结果落在 `[0,1]`，与 gamma 精确值无关。
整数幂直接使用原始底数，因此负底数的奇偶符号正确（Mode 0）。Mode 1/2 因输入已先 clamp 到
`[0,M]`，`Log` 定义域非负；Mode 1 额外用 `Compare(EQ)`（NaN!=NaN）+ `Select` 把 NaN/Inf 清零，
符合图像像素非负语义。

## 4. Tiling

Host 侧从平台读取 AIV core 数与 UB 容量：

- `totalNum` 为所有维度乘积，使用 `uint64_t` 并检查溢出；
- 每核至少分配约 4 KiB 输入，`blockFactor` 再按 512 元素向上对齐；
- `usedCoreNum` 不超过平台 AIV core 数；
- UB 只使用平台容量的 80%，保留数学高阶 API 的内部开销；
- 输入队列双缓冲，输出队列单缓冲，另分配 3 份 FP32 计算 Buffer；
- `ubFactor` 按 256 字节向下对齐；
- 最后一核和最后一个 UB tile 均使用真实有效元素数，搬入搬出统一采用 `DataCopyPad`。

UB 上界按每元素 `3 * sizeof(input) + 6 * sizeof(float) + sizeof(float)` 计算，其中 3 份
输入 dtype 空间对应双缓冲输入队列与单缓冲输出队列；6 份 FP32 计算缓冲覆盖 sRGB 最重路径
（xFloat/result/base/norm/lin/pow）；末项为 uint8 谓词缓冲（按 float 槽保守对齐）。

## 5. 校验与边界

Host/Tiling 拒绝非 FP16/BF16/FP32/UINT8/INT8/UINT16/INT16、rank 非 2/3/4、非正维度、
超过规格的维度、非有限或非正 gamma、`correctionMode` 非 0/1/2，以及元素数溢出。InferShape
支持带未知维度的编译期 shape 透传，运行时 Tiling 只接受已具体化的正维度。整数 dtype 的
`maxValue` 由位宽推导；三种模式都经 `WriteBack` 做 clamp-before-cast，保证 FP32→整数的窄化
不依赖隐式饱和（Mode 0 另把 NaN 置 0）。

UINT16 在 dav_c220 上没有直接的 Cast 通路，经 int16 位型重解释中转：读入时对 int16 视角下
的负值加 65536，写回时对 RINT 后 ≥32768 的值先减 65536 再转 int16，两步修正都用 FP32
min/max 算术完成（整数在 2^24 内精确），覆盖完整 `[0, 65535]` 值域。

测试覆盖 7 种 dtype、Mode 0 的三种快速路径、Mode 1（归一化+clamp+NaN 清零）、Mode 2（sRGB 分段）、
整数负底数、非整数负底数、NaN/Inf、非 32 字节对齐尾块和多核切分。
