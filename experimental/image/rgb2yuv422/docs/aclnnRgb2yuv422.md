<!-- npu="950" id7 -->
# aclnnRgb2yuv422

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

将 RGB 图像转换为 YUV422（YUYV 打包格式）色彩空间。基于 ITU-R BT.601 标准矩阵进行 RGB → YUV444 转换，再进行水平 2:1 色度子采样（YUV422），最终按 YUYV 格式交替打包输出。

## 计算公式

**float16/float32 输入（归一化公式）：**

$$
Y  =  0.29900 \cdot R + 0.58700 \cdot G + 0.11400 \cdot B
$$
$$
U  = -0.16874 \cdot R - 0.33126 \cdot G + 0.50000 \cdot B
$$
$$
V  =  0.50000 \cdot R - 0.41869 \cdot G - 0.08131 \cdot B
$$

**uint8 输入（含 +128 偏移公式）：**

$$
Y  =  0.29900 \cdot R + 0.58700 \cdot G + 0.11400 \cdot B
$$
$$
U  = -0.16874 \cdot R - 0.33126 \cdot G + 0.50000 \cdot B + 128
$$
$$
V  =  0.50000 \cdot R - 0.41869 \cdot G - 0.08131 \cdot B + 128
$$

## 函数原型

### GetWorkspaceSize 接口

```cpp
aclnnStatus aclnnRgb2yuv422GetWorkspaceSize(
    const aclTensor* x,
    char *dataFormatOptional,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor
);
```

### 执行接口

```cpp
aclnnStatus aclnnRgb2yuv422(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream
);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
|--------|----------|------|----------|----------|----------|------|-------------|
| x | 输入 | RGB 图像张量 | NHWC: [..., H, W, 3]; NCHW: [..., 3, H, W] | uint8, float16, float32 | ND | ≥3 | 支持 |
| dataFormatOptional | 可选输入 | 数据格式 | "NHWC" 或 "NCHW"(默认"NHWC") | char* | - | - | - |
| out | 输出 | YUV422 输出张量 | NHWC: [..., H, W, 2]; NCHW: [..., 2, H, W] | 与输入一致 | ND | ≥3 | - |
| workspaceSize | 输出 | workspace 大小 | 返回0 | uint64_t* | - | - | - |
| executor | 输出 | op 执行器 | - | - | - | - | - |

## 返回值错误码表

| 错误码 | 触发条件 |
|--------|---------|
| ACLNN_SUCCESS | 正常执行 |
| ACLNN_ERR_PARAM_NULLPTR | x 或 out 为空指针 |
| ACLNN_ERR_PARAM_INVALID | dtype 不在支持列表中、data_format 不为 NHWC/NCHW、通道维不等于3、rank < 3 |

## 约束说明

- 输入必须至少为 3 维
- 通道维大小必须为 3
- data_format 必须为 "NHWC" 或 "NCHW"
- 输入输出 dtype 必须一致

## 调用示例

```cpp
#include "aclnnop/aclnn_rgb2yuv422.h"

int64_t xShape[] = {4, 8, 3};
aclDataType dataType = ACL_UINT8;
aclFormat format = ACL_FORMAT_ND;

uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
char dataFormatOptional[] = "NHWC";
aclnnRgb2yuv422GetWorkspaceSize(x, dataFormatOptional, out, &workspaceSize, &executor);

void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

aclnnRgb2yuv422(workspaceAddr, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```
<!-- end id7 -->
