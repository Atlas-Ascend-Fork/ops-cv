# aclnnHitOrMiss
## 产品支持情况
- Ascend 950PR/Ascend 950DT：×
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：√
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：√
- Atlas 200I/500 A2 推理产品：×
- Atlas 推理系列产品：×
- Atlas 训练系列产品：×
## 功能说明
- 接口功能：

aclnnHitOrMiss 实现二值形态学 Hit-or-Miss 运算。

根据输入二值图像 input 和二维结构元素 kernel，判断输入区域是否同时满足前景约束和背景约束。

- 计算逻辑：

kernel 元素仅支持 -1、0、1，其中 1 表示前景约束，-1 表示背景约束，0 表示忽略。
当 kernel 中至少存在一个 1 或 -1 约束时，所有有效约束同时满足则输出 True，否则输出 False。
当 kernel 全部为 0 时，输出等于 input 的二值化结果（input != 0）。
当约束映射到 input 范围外时，该约束的匹配结果直接取 borderValue；kernel 为 1 或 -1 时规则相同，kernel 为 -1 时不对 borderValue 取反。
## 函数原型
#include "aclnn_hit_or_miss.h"

aclnnStatus aclnnHitOrMissGetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *kernel,
    const aclIntArray *axesOptional,
    const aclIntArray *anchorOptional,
    bool borderValue,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnHitOrMiss(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
## aclnnHitOrMissGetWorkspaceSize 参数说明
| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| input | 输入 | 输入二值图像 | BOOL、UINT8 | ND |
| kernel | 输入 | 二维结构元素 | INT8 | ND |
| axesOptional | 输入 | 参与计算的两个维度 | INT | - |
| anchorOptional | 输入 | kernel 锚点位置 | INT | - |
| borderValue | 输入 | 越界位置的约束匹配值 | BOOL | - |
| out | 输出 | Hit-or-Miss 计算结果 | BOOL | ND |
| workspaceSize | 输出 | workspace 大小 | - | - |
| executor | 输出 | 算子执行器 | - | - |
## aclnnHitOrMiss 参数说明
| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| workspace | 输入 | Device 侧 workspace 地址 |
| workspaceSize | 输入 | workspace 大小 |
| executor | 输入 | GetWorkspaceSize 返回的执行器 |
| stream | 输入 | 指定执行 stream |
## 返回值
aclnnStatus：返回状态码，具体参见 aclnn 返回码定义。
## 约束说明
- 输入 input 支持 rank 2~8。
- 输入 kernel 必须为二维 Tensor。
- kernel 高度和宽度范围为 1~63。
- kernel 元素值仅支持 -1、0、1，其他取值不在支持范围内。
- 输出 out 的 shape 与 input 保持一致。
- 输出数据类型固定为 BOOL。
- axesOptional 必须包含两个不同维度。
- anchorOptional 支持默认值 -1，由算子自动计算中心位置。
## 调用示例
完整调用示例：
test_aclnn_hit_or_miss.cpp

## 接口生成与数值约束说明

`aclnn_hit_or_miss.h` 及对应实现由仓库根据 `ACLNNTYPE aclnn` 在构建阶段自动生成。调用方需先构建并安装本算子，再使用同一安装包中的头文件和接口库，具体说明见 [README](../README.md)。

kernel 元素仅支持 -1、0、1。Host 阶段不读取 kernel 元素内容，不进行元素值域校验；当前实现将非 1、非 -1 的元素作为忽略项处理，不会因此返回值域错误。调用方应保证传入值符合约束。
