# UpsampleBilinear2dAA

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：对由多个输入通道组成的输入信号应用2D双线性抗锯齿采样。
- 计算公式：对于一个二维插值点$(N, C, H, W)$，插值$I(N, C, H, W)$可以表示为：

  $$
  {I(N, C, H, W)} = \sum_{i=0}^{kH}\sum_{j=0}^{kW}{w(i) * w(j)} * {f(h_i, w_j)}/\sum_{i=0}^{kH}w(i)/\sum_{j=0}^{kW}w(j)
  $$

  $$
  scaleH =\begin{cases}
  (input.dim(2)-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  input.dim(2) / outputSize[0] & otherwise
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  (input.dim(3)-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  input.dim(3) / outputSize[1] & otherwise
  \end{cases}
  $$

  其中：
  - alignCorners为true，表示输入和输出张量的角像素点对齐；alignCorners为false，表示输入和输出张量的边像素点对齐。
  - $kW$、$kH$分别表示W方向和H方向影响插值点大小的点的数量
  - 如果$scaleH >= 1$，则$kH = floor(scaleH) * 2 + 1$，否则$kH = 3$
  - 如果$scaleW >= 1$，则$kW = floor(scaleW) * 2 + 1$，否则$kW = 3$
  - $f(h_i, w_j)$是原图像在$(h_i, w_j)$的像素值
  - $w(i)$、$w(j)$是双线性抗锯齿插值的W方向和H方向权重，计算公式为：

    $$
      w(i) = \begin{cases}
      1 - |h_i - h| & |h_i -h| < 1 \\
      0 & otherwise
      \end{cases}
    $$

    $$
      w(j) = \begin{cases}
      1 - |w_j - w| & |w_j -w| < 1 \\
      0 & otherwise
      \end{cases}
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>表示进行采样的输入张量，对应公式中的`input`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>指定输出空间大小，对应公式中的`outputSize`。size为2，且各元素均大于0。表示指定`output`在H和W维度上的空间大小。</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td><ul><li>决定是否对齐角像素点，对应公式中的`alignCorners`。如果设置为`true`，则输入和输出张量按其角像素的中心点对齐，保留角像素处的值。如果设置为`false`，则输入和输出张量通过其角像素的角点对齐，并使用边缘值对边界外的值进行填充。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>可选属性</td>
      <td><ul><li>指定空间大小的height维度乘数，对应公式中的`scalesH`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>可选属性</td>
      <td><ul><li>指定空间大小的width维度乘数，对应公式中的`scalesW`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式中的`I`。数据类型和数据格式与入参`input`的数据类型和数据格式保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_bilinear2d_aa](examples/test_aclnn_upsample_bilinear2d_aa.cpp) | 通过[aclnnUpsampleBilinear2dAA](docs/aclnnUpsampleBilinear2dAA.md)接口方式调用UpsampleBilinear2dAA算子。 |
