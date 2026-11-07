# UpsampleBicubic2dAA

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

- 算子功能：对由多个输入通道组成的输入信号应用双三次抗锯齿算法进行上采样。如果输入Tensor x的shape为(N, C, H, W)，则输出Tensor out的shape为(N, C, outputSize[0], outputSize[1])。
- 计算公式：对于一个二维插值点$(N, C, h, w)$，插值$out(N, C, h, w)$可以表示为：

  $$
  {out(N, C, h, w)}=\sum_{i=0}^{kW}\sum_{j=0}^{kH}{W(i, j)}*{f(h_i, w_j)}
  $$

  $$
  scaleH =\begin{cases}
  (x.dim(2)-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  x.dim(2) / outputSize[0] & otherwise
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  (x.dim(3)-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  x.dim(3) / outputSize[1] & otherwise
  \end{cases}
  $$

  其中：
  - alignCorners为true，表示输入和输出张量的角像素点对齐；alignCorners为false，表示输入和输出张量的边像素点对齐。
  - i和j是$W(i, j)$的索引变量。
  - 如果$scaleH >= 1$，则$kH = 1/scaleH$，否则$kH = 4$
  - 如果$scaleW >= 1$，则$kW = 1/scaleW$，否则$kW = 4$
  - $h_i = |h| + i$
  - $w_j = |w| + j$
  - $f(h_i, w_j)$是原图像在$(h_i, w_j)$的像素值
  - $W(i, j)$是双三次抗锯齿插值的权重，定义为：

    $$
    W(d) =\begin{cases}
    (a+2)|d|^3-(a+3)|d|^2+1 & |d|\leq1 \\
    a|d|^3-5a|d|^2+8a|d|-4a & 1<|d|<2 \\
    0 & otherwise
    \end{cases}
    $$

    其中：
    - 抗锯齿场景$a=-0.5$。
    - $d = |(h, w) - (h_i, w_j)|$

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
      <td>x</td>
      <td>输入</td>
      <td>表示进行上采样的输入张量，对应公式中的`x`。数据类型与出参`y`的数据类型一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>表示指定`y`在H和W维度上的空间大小，对应公式中的`outputSize`。size为2。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td><ul><li>决定是否对齐角像素点，对应公式中的`alignCorners`。align_corners为true，则输入和输出张量的角像素点会被对齐，否则不对齐。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>可选属性</td>
      <td><ul><li>指定空间大小的height维度乘数，对应公式中的`scalesH`。</li><li>默认值为0.0。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>可选属性</td>
      <td><ul><li>指定空间大小的width维度乘数，对应公式中的`scalesW`。</li><li>默认值为0.0。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式中的`out`。数据类型与入参`x`的数据类型一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

- 形状约束：
  - `x`与`y`均为4维，按NCHW语义解释。
  - `y`的N轴、C轴与`x`保持一致；`y`的H轴、W轴与`output_size`保持一致。
  - `x`的C轴、H轴、W轴取值均需大于0。
- 格式约束：`x`与`y`的数据格式必须一致（同为`ND`或同为`NCHW`），否则算子校验失败。
- 空Tensor：通过aclnn接口调用时，支持N轴为0的空Tensor，此时第一段接口返回成功且不执行计算；`x`的C轴、H轴、W轴任一为0时返回161002。
- 以下上界由调用者保证，算子不做校验，超出时行为未定义：
  - `x`、`y`每个维度的取值小于等于2^20。
  - N \* C \* x_H < 2^31。
  - 内存占用小于60GB，计算公式为：

    $$
    (x\_H * x\_W + y\_H * y\_W + x\_H * y\_W) * N * C * sizeof(dtype) < 60 * 1024 * 1024 * 1024
    $$

- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：输入数据缩放场景的缩小倍数必须小于等于50，即`x`的H轴/`output_size[0]` <= 50且`x`的W轴/`output_size[1]` <= 50。
- 属性约束：
  - `output_size`的size为2，各元素取值范围为[1, 2^20]（上界由调用者保证，算子不做校验）。
  - 当`scales_h`大于0时，满足`output_size[0]` = floor(x_H \* `scales_h`)；当`scales_w`大于0时，满足`output_size[1]` = floor(x_W \* `scales_w`)。`scales_h`、`scales_w`取默认值0.0时该约束不适用，此时缩放比由`x`与`output_size`的比值决定。
- 动态shape：
  - aclnn接口的shape由aclTensor携带，必须为确定的4维取值，不支持-1/-2动态占位。
  - 本算子在`Ascend 950PR&950DT系列产品`、`Kirin X90处理器系列产品`、`Kirin 9030处理器系列产品`上声明支持动态shape与动态rank；GE图模式当前不可用，-1/-2动态占位场景不支持。
- 数值约束：输入含±inf、NaN或接近数据类型表示上限（如FLOAT32的±3.4e38、FLOAT16的±65504）的极值时，抗锯齿加权累加可能溢出为±inf或产生NaN，输出不保证与CPU参考逐元素一致，此类输入不在精度保证范围内。
- 确定性计算：本算子默认确定性实现。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_upsample_bicubic2d_aa](examples/test_aclnn_upsample_bicubic2d_aa.cpp) | 通过[aclnnUpsampleBicubic2dAA](docs/aclnnUpsampleBicubic2dAA.md)接口方式调用UpsampleBicubic2dAA算子。 |

通路支持现状：

- aclnn直调：支持，见上表。
- GE图模式：当前不支持，算子未注册图模式的shape与dtype推导。
- PyTorch图模式：torchair无`aten._upsample_bicubic2d_aa.default`的转换器，const/dynamic图模式不可执行，仅aclgraph模式可执行。
- TensorFlow：本算子未提供TensorFlow接口与绑定。
