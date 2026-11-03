# aclnnGaussianBlur

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

对 FLOAT32 图像执行二维高斯模糊。输入采用 `[H, W]` 或 `[H, W, C]` 的 ND 布局，输出 shape、数据类型和
数据格式与输入一致。算子语义与 OpenCV `GaussianBlur` 对齐，内部采用融合的可分离卷积实现。

## 函数原型

### aclnnGaussianBlurGetWorkspaceSize

```cpp
aclnnStatus aclnnGaussianBlurGetWorkspaceSize(
    const aclTensor* src,
    const aclIntArray* ksize,
    double sigmaX,
    double sigmaY,
    int64_t borderType,
    const aclTensor* dst,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

### aclnnGaussianBlur

```cpp
aclnnStatus aclnnGaussianBlur(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    const aclTensor* src,
    const aclIntArray* ksize,
    double sigmaX,
    double sigmaY,
    int64_t borderType,
    aclTensor* dst,
    aclrtStream stream);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| ------ | --------- | ---- | -------- | -------- |
| `src` | 输入 | 输入图像，shape 为 `[H, W]` 或 `[H, W, C]` | FLOAT32 | ND |
| `ksize` | 输入 | 高斯核尺寸 `[kernelWidth, kernelHeight]` | INT64 数组 | - |
| `sigmaX` | 输入 | 水平方向标准差 | DOUBLE | - |
| `sigmaY` | 输入 | 垂直方向标准差；小于等于 0 时使用 `sigmaX` | DOUBLE | - |
| `borderType` | 输入 | 边界扩展模式 | INT64 | - |
| `dst` | 输出 | 输出图像，shape、数据类型和格式与 `src` 相同 | FLOAT32 | ND |
| `workspaceSize` | 输出 | 执行所需 workspace 字节数 | UINT64 | - |
| `executor` | 输出 | ACLNN 执行器 | `aclOpExecutor*` | - |
| `workspace` | 输入 | workspace 地址；大小为 0 时可传空指针 | `void*` | - |
| `stream` | 输入 | ACL runtime stream | `aclrtStream` | - |

`borderType` 支持 `0`（constant）、`1`（replicate）、`2`（reflect）和 `4`（reflect 101）。

## 返回值

| 返回值 | 描述 |
| ------ | ---- |
| `ACLNN_SUCCESS` | 接口执行成功 |
| `ACLNN_ERR_PARAM_NULLPTR` | 必选指针参数为空 |
| `ACLNN_ERR_PARAM_INVALID` | shape、数据类型、核尺寸、标准差或边界模式不受支持 |

## 约束说明

- `src` 和 `dst` 仅支持 FLOAT32、ND 格式，并且必须具有相同的 shape、数据类型和格式。
- 输入 rank 必须为 2 或 3，所有维度必须大于 0。
- 不支持原地计算或输入输出共享存储。
- `ksize` 必须包含两个元素。显式尺寸必须为正奇数；尺寸为 0 时根据对应标准差推导。
- 规范化后的核尺寸支持 `1/3/5/7/9/11/15/21/31`。
- `sigmaX` 必须大于等于 0，两个标准差都必须为有限值。
- 不支持 `BORDER_WRAP`、`BORDER_ISOLATED` 及其他未列出的边界模式。

## 调用示例

完整示例见 [`../examples/test_aclnn_gaussian_blur.cpp`](../examples/test_aclnn_gaussian_blur.cpp)。调用顺序如下：

1. 创建输入、输出 Tensor 和 `ksize`。
2. 调用 `aclnnGaussianBlurGetWorkspaceSize` 获取 workspace 大小和 executor。
3. 按需申请 workspace，调用 `aclnnGaussianBlur`。
4. 同步 stream 后读取输出。
