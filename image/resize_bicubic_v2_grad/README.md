# ResizeBicubicV2Grad

## 产品支持情况

| 产品 | 是否支持 |
| :---- | :----: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |

## 功能说明

- 算子功能：计算二维双三次插值（ResizeBicubicV2）的反向梯度。正向插值按双三次插值权重将原图 `original_image` 缩放为 `grads` 的尺寸；本算子将上游梯度 `grads` 按同样的插值权重散射回原图尺寸，得到 `y`。
- 计算公式：

  双三次插值权重函数 $W(x)$（其中常数 $a=-0.75$）：

  $$
  W(x) = \begin{cases}
  (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{for } |x|≤1 \\
  a|x|^3 -5a|x|^2 + 8a|x| - 4a  & \text{for } 1<|x|<2 \\
  0 & \text{otherwise} \\
  \end{cases}
  $$

  $$
  \frac{\partial L}{\partial X_{i,j}} = \sum_{i'} \sum_{j'} \frac{\partial L}{\partial Y_{i',j'}} \times W(i' - i) \times W(j' - j)
  $$

- 公式变量与算子参数的映射：

  | 公式变量 | 算子参数 | 说明 |
  | :---- | :---- | :---- |
  | grads | 输入 grads | 上游梯度，正向 ResizeBicubicV2 输出尺寸的张量（公式中的 gradOut） |
  | y | 输出 y | 原图梯度（公式中的 gradInput），shape 与 original_image 一致 |
  | original_image | 输入 original_image | 正向 ResizeBicubicV2 的输入原图，本算子仅使用其形状（N、C、H、W） |
  | i'、j'、i、j | - | H/W 轴上的元素坐标 |
  | scales | 属性 scales | 缩放系数，生效口径见下文与参数说明 |
  | align_corners | 属性 align_corners | 角点对齐开关，决定源坐标计算方式 |

- 缩放系数 scales 的生效条件（与实现及 [aclnnUpsampleBicubic2dBackward](../upsample_bicubic2d_grad/docs/aclnnUpsampleBicubic2dBackward.md) 公开文档一致）：
  - align_corners=true 时，scales 属性被忽略，缩放系数由输入/输出 H、W 轴长度推导（对应轴长度大于 1 时为 (H_original-1)/(H_grads-1)、(W_original-1)/(W_grads-1)）；
  - align_corners=false 且 scales 对应分量大于 0 时，该分量生效，有效缩放系数为 1/scales；
  - align_corners=false 且 scales 对应分量不大于 0 时，该分量被忽略，缩放系数由对应轴长度推导（H_original/H_grads、W_original/W_grads）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 300px">
    <col style="width: 250px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>grads</td>
      <td>输入</td>
      <td>反向传播的上游梯度，为正向ResizeBicubicV2输出尺寸的张量，对应公式中的gradOut。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>original_image</td>
      <td>输入</td>
      <td>正向ResizeBicubicV2的输入原图张量，本算子仅使用其形状（N、C、H、W），对应公式中的原图尺寸。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>original_image的梯度，对应公式中的gradInput，shape与original_image一致。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>属性</td>
      <td>可选，BOOL类型，默认false。角点对齐开关：true表示输入输出张量按角像素中心点对齐；false表示按半像素（half pixel）中心计算插值坐标。同时决定scales属性的生效口径（见功能说明）。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>属性</td>
      <td>可选，ListFloat类型，默认{0.0, 0.0}，长度须为2。两个元素分别表示grads与y的H轴、W轴下标比例（grads_H/y_H、grads_W/y_W，即正向输出与输入的空间尺寸比）。仅在align_corners=false且对应分量大于0时生效，有效缩放系数为1/scales；align_corners=true时被忽略，由输入/输出H、W轴长度推导（见功能说明）。</td>
      <td>LIST_FLOAT</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- grads、original_image、y为4维张量；三者的数据类型与N、C维需保持一致；y的shape与original_image一致。
- 各维度取值需大于0（不支持空Tensor，零长度轴在tiling阶段被拒绝）。
- scales属性长度须为2。
- Kernel直调（GE图模式构图）按连续张量实现与验证，非连续输入的语义未在公开契约中定义（aclnn API通路内部会对非连续输入做连续化处理）。

## 调用说明

| 调用方式 | 样例代码 | 说明                                                                                                                                |
| ---- | ---- |-----------------------------------------------------------------------------------------------------------------------------------|
| aclnn API | [test_aclnn_resize_bicubic_v2_grad](examples/test_aclnn_resize_bicubic_v2_grad.cpp) | 通过[aclnnUpsampleBicubic2dBackward](../upsample_bicubic2d_grad/docs/aclnnUpsampleBicubic2dBackward.md)接口方式调用ResizeBicubicV2Grad算子。 |
| GE图模式 | -  | 通过[算子IR](op_graph/resize_bicubic_v2_grad_proto.h)构图方式调用ResizeBicubicV2Grad算子。         |

- examples/ 目录下的 [repro_timeout_bicubic_grad.cpp](examples/repro_timeout_bicubic_grad.cpp) 为split-K修复的历史问题复现脚本：构造8.6GB梯度输入（grads (1,1,2147483649,1) → y (1,1,2,1)）验证极端上采样反向不再触发矢量核看门狗超时，属交付件，非通用调用样例，需在安装修复后算子包的环境按需编译运行。
