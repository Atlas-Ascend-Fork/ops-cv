# aclnnGammaCorrection

## 功能说明

对 Rank 2/3/4 图像 Tensor 执行逐元素 gamma 校正。按 `correctionMode` 选择纯 `pow(x, gamma)`、
归一化+pow+clamp、或 sRGB 编码曲线三种流水线；输出 shape、dtype 和 ND 布局与输入一致。

## 接口原型

以下接口由 `op_host/gamma_correction_def.cpp` 的算子定义在构建时自动生成（`ACLNNTYPE aclnn`），
不在本目录中手写；头文件 `aclnn_gamma_correction.h` 随算子包安装到
`opp/vendors/<vendor>/op_api/include/`。

```cpp
aclnnStatus aclnnGammaCorrectionGetWorkspaceSize(
    const aclTensor* x,
    double gamma,
    int64_t correctionMode,
    double maxValue,
    const aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnGammaCorrection(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

参数顺序为输入、属性、输出。当前 kernel 不申请用户 workspace，但调用方仍必须使用第一段
接口返回的 `workspaceSize`，不可硬编码为 0。`correctionMode` 与 `maxValue` 为新增属性，
向后兼容：传入 `correctionMode=0` 且 `maxValue=0.0` 即等价于纯 `pow(x, gamma)`。

## 参数

| 参数 | 说明 |
|---|---|
| x | Rank 2/3/4，FLOAT16/BFLOAT16/FLOAT32/UINT8/INT8/UINT16/INT16，ND |
| gamma | 必选有限正数 |
| correctionMode | 0=纯 pow，1=归一化+pow+clamp，2=sRGB 编码（默认 0） |
| maxValue | 值域上界；<=0 时按 dtype 推导（浮点 1.0，8-bit 255.0，16-bit 65535.0） |
| y | shape/dtype 与 x 相同 |

## 约束说明

- 输入 `x` 的 rank 必须为 2、3 或 4；各维度上界按位置从末维往前计算：

| Rank | 维度顺序 | 各维上界 |
|---|---|---|
| 4 | [N, C, H, W] | N ≤ 200，C ≤ 1000，H ≤ 10000，W ≤ 10000 |
| 3 | [d0, d1, d2] | d0 ≤ 1000，d1 ≤ 10000，d2 ≤ 10000 |
| 2 | [d0, d1] | d0 ≤ 10000，d1 ≤ 10000 |

- 任一维度 ≤ 0 会被拒绝。
- 上述校验发生在 tiling 阶段（即 `aclnnGammaCorrectionGetWorkspaceSize` 调用期间），
  越界时报错文案形如 `invalid x dimension <i>: <dim>, allowed range is [1, <limit>]`，
  其中 `<i>` 为维度序号（从 0 开始），`<limit>` 为该维上界。

调用样例见 [`examples/test_aclnn_gamma_correction.cpp`](../examples/test_aclnn_gamma_correction.cpp)。
