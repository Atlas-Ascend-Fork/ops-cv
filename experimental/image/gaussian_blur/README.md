# GaussianBlur

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| Ascend 950PR&950DT系列产品 | √ |

## 功能说明

GaussianBlur 对输入图像执行二维高斯模糊。算子采用可分离卷积实现，先进行水平方向卷积，再进行
垂直方向卷积，输出尺寸和数据类型与输入保持一致。

对于坐标 `(x, y)`，计算过程为：

```text
tmp(x, y, c) = sum(src(x + i, y, c) * kernelX(i))
dst(x, y, c) = sum(tmp(x, y + j, c) * kernelY(j))
```

输入支持以下两种 ND 图像布局：

- `[H, W]`：单通道图像。
- `[H, W, C]`：多通道图像，通道维位于最后一维。

## 接口说明

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

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------ | ------------- | ---- | -------- | -------- |
| `src` | 输入 | 输入图像，shape 为 `[H, W]` 或 `[H, W, C]` | FLOAT32 | ND |
| `ksize` | 属性 | 高斯核尺寸 `[kernelWidth, kernelHeight]` | INT64 数组 | - |
| `sigmaX` | 属性 | 水平方向高斯标准差 | DOUBLE | - |
| `sigmaY` | 属性 | 垂直方向高斯标准差；小于等于 0 时使用 `sigmaX` | DOUBLE | - |
| `borderType` | 属性 | 边界扩展模式 | INT64 | - |
| `dst` | 输出 | 输出图像，shape、数据类型和格式与 `src` 相同 | FLOAT32 | ND |

`borderType` 支持：

| 值 | 模式 | 说明 |
| -- | ---- | ---- |
| `0` | `BORDER_CONSTANT` | 边界外填充 0 |
| `1` | `BORDER_REPLICATE` | 复制最邻近边界像素 |
| `2` | `BORDER_REFLECT` | 镜像边界，包含边界像素 |
| `4` | `BORDER_REFLECT_101` | 镜像边界，不重复边界像素 |

## 约束说明

- 仅支持 Ascend 950PR&950DT系列产品。
- `src` 和 `dst` 仅支持 FLOAT32、ND 格式。
- `src` rank 必须为 2 或 3，所有维度必须大于 0。
- `dst` 必须与 `src` 具有相同的 shape、数据类型和格式。
- 不支持原地计算，`src` 和 `dst` 不能使用同一存储地址。
- `ksize` 必须包含两个元素。显式核尺寸必须为正奇数；设置为 0 时根据对应的 `sigma` 推导。
- 规范化后的核尺寸支持 `1/3/5/7/9/11/15/21/31`。
- `sigmaX` 必须大于等于 0，`sigmaX` 和 `sigmaY` 必须为有限值。
- 当核尺寸设置为 0 时，对应的标准差必须大于 0。
- 不支持 `BORDER_WRAP`、`BORDER_ISOLATED` 和其他未列出的边界模式。

## 调用说明

| 调用方式 | 接口声明 | 说明 |
| -------- | -------- | ---- |
| ACLNN 调用 | [`docs/aclnnGaussianBlur.md`](./docs/aclnnGaussianBlur.md) | 接口约束、参数和返回值说明。 |
| ACLNN 示例 | [`examples/test_aclnn_gaussian_blur.cpp`](./examples/test_aclnn_gaussian_blur.cpp) | 完整的 Tensor 创建、workspace 申请和算子执行示例。 |
