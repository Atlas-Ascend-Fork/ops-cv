# UpsampleNearestExact3d

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。
- 计算公式：

  $$
  scaleD =\begin{cases}
  scalesD & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[0]/D &  Otherwise\\
  \end{cases}
  $$

  $$
  scaleH =\begin{cases}
  scalesH & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[1]/H &  Otherwise\\
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  scalesW & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[2]/W &  Otherwise\\
  \end{cases}
  $$

  $$
  d_{src} = min(floor((d_{dst} + 0.5) / scaleD),  D - 1)
  $$

  $$
  h_{src} = min(floor((h_{dst} + 0.5) / scaleH),  H - 1)
  $$

  $$
  w_{src} = min(floor((w_{dst} + 0.5) / scaleW),  W - 1)
  $$

  $$
  out(N, C, d_{dst},h_{dst}, w_{dst}) = self(N, C, d_{src},h_{src}, w_{src})
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
      <td>x</td>
      <td>输入</td>
      <td>表示进行上采样的输入张量，对应公式中的`self`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>指定输出`out`的Tensor大小，对应公式中的`output_size`。size为3，各元素均大于零。表示输出`out`在D、H和W维度上的空间大小。</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_d</td>
      <td>可选属性</td>
      <td><ul><li>指定D维采样缩放系数，对应公式中的scalesD。</li><li>默认值为0.0；仅当scale_d、scale_h和scale_w均大于0时，才使用这三个系数；否则根据输入shape和output_size推导缩放系数。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_h</td>
      <td>可选属性</td>
      <td><ul><li>指定H维采样缩放系数，对应公式中的scalesH。</li><li>默认值为0.0；仅当scale_d、scale_h和scale_w均大于0时，才使用这三个系数；否则根据输入shape和output_size推导缩放系数。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_w</td>
      <td>可选属性</td>
      <td><ul><li>指定W维采样缩放系数，对应公式中的scalesW。</li><li>默认值为0.0；仅当scale_d、scale_h和scale_w均大于0时，才使用这三个系数；否则根据输入shape和output_size推导缩放系数。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式描述中的`out`。数据类型和数据格式需与入参`self`的数据类型和数据格式保持一致，shape由输入x的N/C维和output_size决定；scale_d、scale_h和scale_w仅影响采样坐标。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
  </tbody></table>

- <term>Atlas推理系列产品</term>：输入和输出的数据类型不支持BFLOAT16。
- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_nearest_exact3d](examples/test_aclnn_upsample_nearest_exact3d.cpp) | 通过[aclnnUpsampleNearestExact3d](docs/aclnnUpsampleNearestExact3d.md)接口方式调用UpsampleNearestExact3d算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/upsample_nearest_exact3d_proto.h)构图方式调用UpsampleNearestExact3d算子。         |
