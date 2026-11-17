# MedianFilter3D

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

对 ND 张量的最后三个轴 D/H/W 进行三维中值滤波，N/C 独立。输出形状和类型保持不变。
对完整矩形窗口排序后选择索引 `floor(Kd*Kh*Kw/2)`；偶数窗口选择较大的中间元素。
窗口偏移为 `[-floor(K/2), K-1-floor(K/2)]`，对应 SciPy 的 `origin=0`。

## 参数说明

| 参数 | 说明 |
| --- | --- |
| x / y | uint8、float32、float16；ND；DHW / NDHW / NCDHW |
| kernel_size | 长度 3 的正整数列表，默认 [3,3,3]；乘积不超过 1024，允许偶数及各向异性 |
| mode | REFLECT（默认）或 CONSTANT，区分大小写 |
| cval | float 属性，默认 0；CONSTANT 时用于填充，REFLECT 时忽略 |

## 约束说明

所有运行时维度为正，元素总数不超过 INT32_MAX；不支持空张量。图推导允许 -1 未知维。
输入由生成的 ACLNN 包装层转为连续布局。输出不得与输入别名。
REFLECT 是半采样对称：`d c b a | a b c d | d c b a`，支持多次反射及长度 1 的轴。
CONSTANT 的 cval 以 float32 参与排序，选中后才转换为输出类型；uint8 的 cval 须在 [0,255]，转换截断。
浮点输入允许正负无穷，NaN 行为不作保证；CONSTANT cval 不允许 NaN。
float16 的有限值精确提升到 float32 排序，输出转回 float16；选中的 cval 按转换规则舍入。
浮点零值不保证符号位保持。

## 调用说明

通过 [aclnnMedianFilter3D](docs/aclnnMedianFilter3D.md) 调用。参见[调用示例](examples/test_aclnn_median_filter3_d.cpp)。
