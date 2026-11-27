# ResizeBicubicV2

## 产品支持情况

| 产品 | 是否支持 |
| :---- | :----: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |

## 功能说明

- 算子功能：使用双三次插值调整图像大小到指定的大小。
- 计算公式：

  输出图像中坐标为$(X,Y)$（第$Y$行、第$X$列）的像素，按坐标映射规则（由属性`align_corners`决定，与PyTorch `upsample_bicubic2d`一致）映射到输入图像的源坐标$(h,w)$。以$(h,w)$为参考点，取输入图像中相邻的4×4（共16个）像素：第$i$行第$j$列的邻域像素坐标为$h_i = \lfloor h \rfloor - 1 + i$、$w_j = \lfloor w \rfloor - 1 + j$（$i, j \in \{0,1,2,3\}$），其像素值记为$a_{ij}$。源坐标到邻域像素的距离定义为$d_i = |h - h_i|$、$d_j = |w - w_j|$（即W函数的自变量，取值范围[0,2)）。

  三次插值权重函数（系数取$a = -0.75$）：

  $$
  W(x) = \begin{cases}
  (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{for } |x|≤1 \\
  a|x|^3 -5a|x|^2 + 8a|x| - 4a  & \text{for } 1<|x|<2 \\
  0 & \text{otherwise} \\
  \end{cases}
  $$

  输出像素值（对16个邻域像素按行、列距离加权求和）：
  $$
  B(X,Y) = \sum_{i=0}^3 \sum_{j=0}^3a_{ij} \times W(d_i) \times W(d_j)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1000px">
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
      <td>x</td>
      <td>输入</td>
      <td>输入图像的四维Tensor，对应公式中x。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>输出图像的高和宽。1维Tensor，元素数为2，各元素均大于0；取值须与输出y按格式推断的H、W一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>双三次插值调整后图像。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>属性</td>
      <td>是否对齐输入与输出张量四个角像素的中心以保留角点像素值，决定源坐标的映射规则。默认值：false。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>属性</td>
      <td>空间维度（高、宽）的缩放系数，元素数为2，分别为高、宽方向的输出与输入尺寸之比。默认值：{0.0, 0.0}（表示不指定，此时缩放系数由输入、输出尺寸推导）。</td>
      <td>ListFloat</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入张量 `x` 和输出张量 `y` 均须为 4 维；输入 `x` 的 `N`、`C` 维须大于 0，且 `y` 的 `N`、`C` 维须分别与 `x` 相等。
- 输入和输出的 `H`、`W` 维均须大于 0。
- `size` 须为一维 INT32 张量，包含 2 个元素，分别指定输出 `H`、`W`；两个取值均须大于 0。

## 调用说明

| 调用方式 | 样例代码 | 说明                                                                                                       |
| ---- | ---- |----------------------------------------------------------------------------------------------------------|
|aclnn接口|[test_aclnn_bicubic_v2](examples/test_aclnn_bicubic_v2.cpp)| 通过[aclnnUpsampleBicubic2d](../upsample_bicubic2d/docs/aclnnUpsampleBicubic2d.md)接口方式调用ResizeBicubicV2算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/resize_bicubic_v2_proto.h)构图方式调用ResizeBicubicV2算子。         |
