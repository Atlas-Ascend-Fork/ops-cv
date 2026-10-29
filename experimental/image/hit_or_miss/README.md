# HitOrMiss

## 产品支持情况

| 产品 | 是否支持 |
|---|---|
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |


## 功能说明

- 算子功能：

  实现二值图像形态学 Hit-or-Miss 运算。

- 计算逻辑：

  kernel 元素仅支持 -1、0、1：

  - 1 表示当前位置必须匹配前景
  - -1 表示当前位置必须匹配背景
  - 0 表示忽略该位置

  当 kernel 中至少存在一个 1 或 -1 约束时，所有有效约束同时满足则输出 True，否则输出 False。
  当 kernel 全部为 0 时，输出等于 input 的二值化结果（input != 0）。
  当约束映射到 input 范围外时，该约束的匹配结果直接取 border_value；kernel 为 1 或 -1 时规则相同，kernel 为 -1 时不对 border_value 取反。


## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 |
|-|-|-|-|
| input | 输入 | 输入二值图像 | BOOL、UINT8 |
| kernel | 输入 | 二维结构元素 | INT8 |
| axes | 属性 | 运算维度 | INT |
| anchor | 属性 | kernel锚点 | INT |
| border_value | 属性 | 越界位置的约束匹配值 | BOOL |
| output | 输出 | 匹配结果 | BOOL |


## 约束说明

- input 支持 rank 2~8
- kernel 必须二维
- kernel 高宽范围 1~63
- kernel 元素值仅支持 -1、0、1，其他取值不在支持范围内
- output shape 与 input 一致


## 调用说明

通过 aclnn 接口调用：

[test_aclnn_hit_or_miss](examples/test_aclnn_hit_or_miss.cpp)

## ACLNN 接口生成与调用

本算子使用 `ACLNNTYPE aclnn`，由仓库的 `cmake/opbuild.cmake` 在构建阶段根据 OpDef 自动生成 `aclnn_hit_or_miss.h` 和 `aclnn_hit_or_miss.cpp`，并将生成的实现编入算子接口库。源码目录不单独维护手写的同名 ACLNN 实现。

运行 examples 中的示例前，需先完成本算子的构建和安装。示例使用安装包 `op_api/include` 下的头文件，并链接同一安装包 `op_api/lib` 下的接口库。

## kernel 元素值约束

kernel 的合法元素值为 -1、0、1。Host 阶段不读取 kernel 元素内容，不进行元素值域校验；当前 Kernel 仅识别 1 和 -1，其余元素作为忽略项处理，不会因此返回值域错误。调用方应保证 kernel 元素满足取值约束。
