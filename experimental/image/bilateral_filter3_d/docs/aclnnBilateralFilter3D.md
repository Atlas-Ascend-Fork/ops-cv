# aclnnBilateralFilter3D

## 功能说明

对 Rank 3/4/5 的 ND Tensor 执行三维双边滤波，
语义**参考** MONAI 的双边滤波实现，但边界处理不同（本算子为有效邻域截断 + 重归一化，
MONAI 为逐维 replicate 寻址 + 全窗口分母），窗口来源与通道语义亦有差异，
见 `design.md` 第 1.1 节。

输入的最后三维视为 D/H/W，其余前导维按独立体数据批量处理。输出 shape、dtype 和 ND 布局与输入一致。

每个体素在 `kernel_size^3` 邻域内做加权平均，权重同时由空间距离与值域差决定：

```text
y = sum(w * v) / sum(w)
w = exp(-( |off|^2 / (2*sigma_space^2) + (v - c)^2 / (2*sigma_color^2) ))
```

其中 `c` 为中心体素值、`v` 为邻域体素值、`off` 为三维偏移。边界采用**有效邻域截断 + 重归一化**：
越界邻域不参与累加，分母只累加实际参与的权重，因此边界体素不会被零值拉偏。

## 接口原型

```cpp
aclnnStatus aclnnBilateralFilter3DGetWorkspaceSize(
    const aclTensor* x,
    int64_t kernelSize,
    double sigmaSpace,
    double sigmaColor,
    const aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnBilateralFilter3D(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

参数顺序为输入、属性、输出。当前 kernel 为纯向量实现、不申请用户 workspace，但调用方仍必须使用
第一段接口返回的 `workspaceSize`（其中含数学高阶 API 的框架内部开销），不可硬编码为 0。

## 参数

| 参数 | 说明 |
|---|---|
| x | Rank 3/4/5，FLOAT16/FLOAT32，ND；最后三维为 D/H/W |
| kernelSize | 必选奇数，取值 3/5/7/9 |
| sigmaSpace | 必选有限正数 |
| sigmaColor | 必选有限正数 |
| y | shape/dtype 与 x 相同 |

## 规格约束

| 约束 | 上界 |
|---|---|
| D | 1000 |
| H、W | 4096 |
| 前导维乘积（N\*C） | 12800 |

不支持空张量（任一维为 0 会被拒绝）。InferShape 支持编译期未知维度（负数占位符）透传，
运行时 Tiling 只接受已具体化的正维度。

`sigma` 极小时 `0.5/sigma^2` 会在 FP32 下溢出，Host 侧将该倒数饱和到 `1e30`：此时非零距离/非零
色差的权重 `exp(-x)` 恒为 0，输出收敛到中心体素值，而不会产生 `0*inf` 的 NaN。

## 错误码

| 返回 | 触发条件 |
|---|---|
| `ACLNN_ERR_PARAM_NULLPTR` | `x`、`y` 或出参指针为空 |
| `ACLNN_ERR_PARAM_INVALID` | dtype 非 FLOAT16/FLOAT32；rank 非 3/4/5；存在为 0 的维度；维度超出上表规格；`kernelSize` 非 3/5/7/9；`sigmaSpace` 或 `sigmaColor` 非有限正数；`y` 与 `x` 的 shape/dtype 不一致 |

调用样例见 [`examples/test_aclnn_bilateral_filter3_d.cpp`](../examples/test_aclnn_bilateral_filter3_d.cpp)。
