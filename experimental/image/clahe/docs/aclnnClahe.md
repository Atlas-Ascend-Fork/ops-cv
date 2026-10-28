# aclnnClahe

## 产品支持情况

- Ascend 950PR/Ascend 950DT：×
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：√
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：√
- Atlas 200I/500 A2 推理产品：×
- Atlas 推理系列产品：×
- Atlas 训练系列产品：×

## 功能说明

- 接口功能：

  aclnnClahe 实现单通道图像的限制对比度自适应直方图均衡化。

- 计算逻辑：

  对每幅图像划分局部网格并统计直方图，限制局部直方图计数、重新分配超额计数，再将累计分布映射通过相邻网格中心的双线性插值融合到输出图像。有限非正 clipLimit 关闭裁剪。

## 函数原型

```cpp
#include "aclnn_clahe.h"

aclnnStatus aclnnClaheGetWorkspaceSize(
    const aclTensor *x,
    double clipLimit,
    const aclIntArray *tileGridSizeOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnClahe(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnClaheGetWorkspaceSize 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x | 输入 | 单通道图像，支持 `[H,W]`、`[N,H,W]`、`[N,1,H,W]` | UINT8、UINT16 | ND |
| clipLimit | 输入 | 对比度裁剪参数，默认属性值为 40；C 接口接收 double，内部转换为 float32 | DOUBLE | - |
| tileGridSizeOptional | 输入 | 横向、纵向网格数量；nullptr 使用 `[8,8]` | INT64 数组 | - |
| out | 输出 | CLAHE 处理结果，shape 和数据类型与 x 一致；对应算子输出 y | UINT8、UINT16 | ND |
| workspaceSize | 输出 | 所需 workspace 字节数，包含系统保留区 | UINT64 | - |
| executor | 输出 | 算子执行器 | aclOpExecutor | - |

## aclnnClahe 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| workspace | 输入 | 按 GetWorkspaceSize 返回大小分配的 Device 侧 workspace 首地址 |
| workspaceSize | 输入 | workspace 大小，单位为字节 |
| executor | 输入 | GetWorkspaceSize 返回的执行器 |
| stream | 输入 | 指定执行 stream |

## 返回值

aclnnStatus：返回状态码，具体参见 aclnn 返回码定义。

## 约束说明

- x 支持 rank 2、3、4；rank 4 时通道数必须为 1。N 为 1～256，H、W 必须为正数。
- out 的 shape、数据类型与 x 一致，输入、输出均为 ND；输出应独立分配。
- clipLimit 转换后的 float32 值必须有限，不支持 NaN 或 Inf。有限非正值关闭裁剪，正值按 tile 面积和 bin 数归一化，裁剪计数下界为 1。
- tileGridSizeOptional 非空时必须包含两个正整数，依次为横向和纵向网格数量；每个数值及网格乘积须处于 int32 正数范围。
- 扩展后的尺寸及 tile 面积须处于 int32 正数范围；总元素数和 workspace 大小须处于 int64 范围。实际可分配规模取决于设备内存。
- 网格不能整除图像时，按 OpenCV CLAHE 的右侧、底部 REFLECT_101 规则扩展；若任一轴需要扩展，另一轴也进入扩展分支。
- uint8、uint16 分别使用完整的 256、65536 个 bin，不进行灰度位移；CDF 映射和双线性插值使用 fp32，整数输出采用最近偶数舍入。
- workspace 按返回大小分配，并将分配得到的首地址传给 aclnnClahe，不应手工跳过系统保留区。

## 调用示例

完整调用示例：

[test_aclnn_clahe.cpp](../examples/test_aclnn_clahe.cpp)
