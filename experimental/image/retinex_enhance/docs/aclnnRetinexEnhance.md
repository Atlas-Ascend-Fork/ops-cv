# aclnnRetinexEnhance

[查看源码](../op_kernel/retinex_enhance.cpp)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

对单张HWC图像执行固定参数的多尺度Retinex增强。算子使用三个尺度的Gaussian环绕函数估计照度分量，
在对数域完成多尺度Retinex响应、颜色恢复和输出范围处理。输入和输出Shape相同，输出值位于`[0, 255]`。

令`x=clip(image,0,255)+1`，`MSR`为三个尺度的`ln(x)-ln(Gaussian(x))`的平均值。
灰度输出为`clip(30*(MSR+6),0,255)`；RGB输出为`clip(30*(CRF*MSR+6),0,255)`，
其中`CRF_i=ln(125*x_i)-ln(x_R+x_G+x_B)`。RGBA仅增强RGB，Alpha为`clip(image_alpha,0,255)`。
增益30、偏置6为固定参数，不执行逐图最小最大值归一化。

## 函数原型

每次调用必须先执行[两段式接口](../../../../docs/zh/context/two_phase_api.md)的第一段接口，获取本次计算所需的
workspace大小和执行器，再调用第二段接口执行计算。

### GetWorkspaceSize接口

```cpp
aclnnStatus aclnnRetinexEnhanceGetWorkspaceSize(
    const aclTensor* image,
    const aclTensor* enhanced,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

### 执行接口

```cpp
aclnnStatus aclnnRetinexEnhance(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## aclnnRetinexEnhanceGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（Shape） | 非连续Tensor |
| ------ | --------- | ---- | -------- | -------- | -------- | ------------- | ------------ |
| `image` | 输入 | 待增强的灰度、RGB或RGBA图像。 | HWC单图；有限输入在计算前截断到`[0, 255]`。 | FLOAT32 | ND | 3维，`[H,W,C]`，`H>0`、`W>0`、`C∈{1,3,4}` | 不支持 |
| `enhanced` | 输出 | 增强后的图像。 | Shape、数据类型和数据格式必须与`image`一致，且不得与`image`共享存储。 | FLOAT32 | ND | 与`image`一致 | 不支持 |
| `workspaceSize` | 输出 | 返回执行阶段需要申请的workspace字节数。 | 不得为空指针。 | `uint64_t*` | - | - | - |
| `executor` | 输出 | 返回包含本次算子计算流程的执行器。 | 不得为空指针。 | `aclOpExecutor**` | - | - | - |

通道语义如下：

- `C=1`：执行灰度多尺度Retinex，不执行RGB颜色恢复。
- `C=3`：执行RGB多尺度Retinex和颜色恢复。
- `C=4`：前三通道执行RGB路径；Alpha通道截断到`[0, 255]`后保留。

## aclnnRetinexEnhance

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 |
| ------ | --------- | ---- | -------- |
| `workspace` | 输入 | Device侧workspace地址。 | 当`workspaceSize`大于0时不得为空指针，空间大小不得小于第一段接口返回值。 |
| `workspaceSize` | 输入 | workspace字节数。 | 必须使用第一段接口返回的大小。 |
| `executor` | 输入 | 算子执行器。 | 必须使用同一次第一段接口调用返回的执行器，不得为空指针。 |
| `stream` | 输入 | 执行算子的ACL Runtime Stream。 | 使用有效的Stream，以ACL Runtime接口约束为准。 |

## 返回值错误码表

| 错误码 | 触发条件 |
| ------ | -------- |
| `ACLNN_SUCCESS` | 接口执行成功。 |
| `ACLNN_ERR_PARAM_NULLPTR` | 必选指针参数为空。 |
| `ACLNN_ERR_PARAM_INVALID` | 输入或输出的Shape、数据类型、数据格式不受支持，或输入输出描述不一致。 |

## 约束说明

- 仅支持Atlas A2训练系列产品/Atlas A2推理系列产品。
- `image`和`enhanced`仅支持FLOAT32、ND格式的连续Tensor。
- 输入Rank必须为3，Shape为`[H,W,C]`；`H`和`W`必须大于0，`C`仅支持1、3或4。
- 支持运行期动态`H/W`；每次调用根据实际Shape重新计算Tiling和workspace。
- NaN或Inf输入的数值结果不作承诺。
- Gaussian尺度固定为15、80和250，边界采用重复边缘的偶对称扩展。
- 总元素数`H*W*C`不得超过`2^30`，保证FP32 Gather的uint32字节偏移可表示。
- Shape还受平台UB容量限制，UB需容纳完整输入行、halo、纵向索引及至少一行累加缓冲。
  大宽度或大高度可能导致第一段接口在Tiling阶段返回失败，此时不得执行第二段接口；
  H/W没有独立于平台和通道数的固定上限。例如UB为256 KiB时，`[1,65536,4]`无法容纳完整输入行。

## 调用示例

完整示例见[`../examples/test_aclnn_retinex_enhance.cpp`](../examples/test_aclnn_retinex_enhance.cpp)。基本调用顺序如下：

1. 创建输入和输出`aclTensor`。
2. 调用`aclnnRetinexEnhanceGetWorkspaceSize`获取workspace大小和executor。
3. 根据返回值在Device侧申请workspace。
4. 调用`aclnnRetinexEnhance`并同步stream。
5. 释放workspace、Tensor和ACL Runtime资源。
