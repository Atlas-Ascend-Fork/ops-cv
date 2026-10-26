# Crop

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：从输入张量`x`中裁剪出一个子区域，子区域的大小由参考张量`size`的shape决定，裁剪的起始位置由`axis`和`offsets`属性控制。`axis`指定裁剪的起始维度，在该维度之前的维度保持与`x`相同，从`axis`维开始的各维度按`offsets`偏移裁剪到`size`对应维度的大小。兼容主流深度学习框架的Crop层，主要用于SSD等目标检测网络中进行多尺度特征图的裁剪对齐。

- 计算公式：

$$
y[i_0, i_1, \ldots, i_{n-1}] = x[j_0, j_1, \ldots, j_{n-1}]
$$

其中：

$$
j_k = \begin{cases} i_k, & k < \text{axis} \\ i_k + \text{offsets}[k - \text{axis}], & k \geq \text{axis} \end{cases}
$$

输出shape：

$$
\text{y.shape}[k] = \begin{cases} \text{x.shape}[k], & k < \text{axis} \\ \text{size.shape}[k], & k \geq \text{axis} \end{cases}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>待裁剪的输入张量。</td>
      <td>FLOAT16、FLOAT、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>参考裁剪张量，其shape决定输出shape的axis及之后维度。每个维度Si不能超过x对应维度Di。rank必须与x一致。</td>
      <td>与x相同</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>裁剪后的输出张量，dtype与x/size一致。</td>
      <td>与x相同</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td>裁剪起始维度，缺省值为2。该维度之前的维度保持与x相同。取值范围[-rank(x), rank(x)-1]，负值会转换为正值。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>属性</td>
      <td>各维度裁剪偏移量。长度为1时所有裁剪维度使用同一偏移；长度大于1时必须等于rank(x)-axis。各维度需满足0 <= offsets[i] <= x.shape[i] - size.shape[i]。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入`x`和`size`的维度数（rank）必须一致，支持1D~8D。
- `x`、`size`、`y`三者的数据类型必须一致。
- `axis`取值范围为[-rank(x), rank(x)-1]，若为负值会自动转换为正值（axis += rank(x)）。
- `offsets`长度为1时，从`axis`开始的所有维度使用同一偏移值；`offsets`长度大于1时，必须等于rank(x) - axis，分别对应各维度。
- 各维度偏移需满足`offsets[i] + size.shape[i] <= x.shape[i]`且`offsets[i] >= 0`，否则报错。
- `axis`之前的维度，输出shape继承`x`的对应维度；`size`在`axis`之前的维度不参与裁剪。
- 输出shape在`axis`之前的维度继承`x`的对应维度，`axis`及之后的维度取`size`的对应维度。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>样例代码</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE图模式</td>
    <td><a href="./examples/test_geir_crop.cpp">test_geir_crop</a></td>
    <td>通过<a href="./op_graph/crop_proto.h">算子IR</a>构图方式调用Crop算子，参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用指南</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
