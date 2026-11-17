# aclnnGuidedFilter

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| <term>Ascend 910B 系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

引导滤波（Guided Image Filtering）。以引导图 `guide` 的结构信息为约束，对待滤波图 `src` 执行保边平滑：在
局部窗口内假设输出是引导图的线性变换，逐像素求解 `alpha` / `beta` 后，再做一次均值滤波重构。算子语义与
OpenCV `cv::ximgproc::guidedFilter` 对齐。

`guide` 与 `src` 采用通道后置（NHWC）的 ND 布局，`guide` 支持 `[H, W]`、`[H, W, C]`、`[N, H, W, C]` 三种
rank；`src` 的 rank 与空间尺寸必须与 `guide` 一致，通道数可以不同。输出 shape 与 `src` 完全一致。

## 函数原型

### aclnnGuidedFilterGetWorkspaceSize

```cpp
aclnnStatus aclnnGuidedFilterGetWorkspaceSize(
    const aclTensor* guide,
    const aclTensor* src,
    int64_t radius,
    double eps,
    int64_t dDepth,
    double scale,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

### aclnnGuidedFilter

```cpp
aclnnStatus aclnnGuidedFilter(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| ------ | --------- | ---- | -------- | -------- |
| `guide` | 输入 | 引导图，shape 为 `[H, W]` / `[H, W, C]` / `[N, H, W, C]` | UINT8、FLOAT16、FLOAT32 | ND |
| `src` | 输入 | 待滤波图，rank 与空间尺寸须与 `guide` 一致，通道数可不同 | UINT8、FLOAT16、FLOAT32 | ND |
| `radius` | 输入 | 窗口半径，取值 `[1, 512]` | INT64 | - |
| `eps` | 输入 | 正则项，控制平滑强度，须为有限值且 `>= 0` | DOUBLE | - |
| `dDepth` | 输入 | 输出位深：`-1` 跟随 `src`、`0`=CV_8U、`5`=CV_32F、`7`=CV_16F | INT64 | - |
| `scale` | 输入 | 快速模式降采样比例，当前仅支持 `1.0` | DOUBLE | - |
| `out` | 输出 | 输出张量，shape 与 `src` 一致，dtype 由 `dDepth` 决定 | UINT8、FLOAT16、FLOAT32 | ND |
| `workspaceSize` | 输出 | 执行所需 workspace 字节数 | UINT64 | - |
| `executor` | 输出 | ACLNN 执行器 | `aclOpExecutor*` | - |
| `workspace` | 输入 | workspace 地址；大小为 0 时可传空指针 | `void*` | - |
| `stream` | 输入 | ACL runtime stream | `aclrtStream` | - |

`eps` 不随输入 dtype 缩放（与 OpenCV 一致）。`dDepth` 决定 `out` 的位深而不改变取值范围：UINT8 输入按
`0~255` 原值参与运算、不做归一化，输出按四舍五入回写。

## 返回值

| 返回值 | 描述 |
| ------ | ---- |
| `ACLNN_SUCCESS` | 接口执行成功 |
| `ACLNN_ERR_PARAM_NULLPTR` | 必选指针参数为空 |
| `ACLNN_ERR_PARAM_INVALID` | shape、数据类型、`radius` / `eps` / `dDepth` / `scale` 或单行字节数不受支持 |

## 约束说明

- `guide` 与 `src` 的 rank 必须相同，空间尺寸 `(H, W)` 必须完全一致，通道数可以不同。
- `guide` 通道数必须在 `[1, 3]`（超出直接报错，不做自动降维）；`src` 通道数必须在 `[1, 64]`。
- rank 必须为 2、3 或 4。rank 为 4 时，`guide` 与 `src` 的 batch 维 `N` 必须相同。
- `guide` 与 `src` 的数据类型必须一致，支持 UINT8、FLOAT16、FLOAT32；输出 dtype 由 `dDepth` 决定，取值
  只能是 `-1`、`0`、`5`、`7`。
- 单行字节数 `W * max(guideC, srcC) * elemSize` 不得超过 65535。
- `N`、`H`、`W` 均不得超过 `INT32_MAX`，且 `H * W` 不得超过 `2^30`。
- 不支持空 Tensor（`N` / `H` / `W` 为 0 时报错）。
- `scale` 当前仅支持 `1.0`，其余取值显式报错（快速模式尚未实现）。
- 不支持原地计算或输入输出共享存储。

## 调用示例

完整示例见 [`../examples/test_aclnn_guided_filter.cpp`](../examples/test_aclnn_guided_filter.cpp)，其中包含 CPU
参考实现与逐元素对拍。调用顺序如下：

1. 创建 `guide`、`src`、`out` 三个 Tensor，按 `dDepth` 选择输出 dtype。
2. 调用 `aclnnGuidedFilterGetWorkspaceSize` 获取 workspace 大小和 executor。
3. 按需申请 workspace，调用 `aclnnGuidedFilter`。
4. 同步 stream 后读取输出。
