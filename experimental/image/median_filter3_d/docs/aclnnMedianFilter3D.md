# aclnnMedianFilter3D

## 产品支持情况

- Ascend 950PR/Ascend 950DT：×
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：√
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：√
- Atlas 200I/500 A2 推理产品：×
- Atlas 推理系列产品：×
- Atlas 训练系列产品：×

## 功能说明

对输入最后三维的完整窗口取升序索引 `n//2` 的元素。边界模式、类型、窗口范围和精度约束见 [README](../README.md)。

## 函数原型

该接口采用两段式调用。

```cpp
aclnnStatus aclnnMedianFilter3DGetWorkspaceSize(
    const aclTensor* x, const aclIntArray* kernelSizeOptional, char* modeOptional,
    double cval, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnMedianFilter3D(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

## aclnnMedianFilter3DGetWorkspaceSize 参数说明

| 参数 | 说明 |
| --- | --- |
| x | 输入，rank 3/4/5，uint8/float32/float16，ND |
| kernelSizeOptional | 三个正整数 [Kd,Kh,Kw]，乘积 <=1024；属性默认 [3,3,3] |
| modeOptional | REFLECT 或 CONSTANT；属性默认 REFLECT |
| cval | CONSTANT 填充值；属性默认 0，包装层转换为 float32 |
| out | 输出，与输入形状和 dtype 相同，须独立存储 |
| workspaceSize | 返回所需系统及用户 workspace 总字节数 |
| executor | 执行计划，供第二段使用 |

## aclnnMedianFilter3D 参数说明

| 参数 | 说明 |
| --- | --- |
| workspace | Device 侧 workspace 地址 |
| workspaceSize | 返回所需系统及用户 workspace 总字节数 |
| executor | 执行计划，供第二段使用 |
| stream | 执行流 |

## 返回值

成功返回 ACLNN_SUCCESS；非法参数由接口或 tiling 阶段拒绝，调用方须检查两段返回码及同步结果。

## 约束说明

窗口、类型和边界约束见 [README](../README.md)。

## 调用示例

[示例源码](../examples/test_aclnn_median_filter3_d.cpp) 验证 uint8 的 1x1x5 体数据和 [1,1,3] REFLECT 窗口，预期输出 [10,10,80,20,20]。
