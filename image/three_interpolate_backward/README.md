# ThreeInterpolateBackward

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

- 算子功能：根据grad_x, idx, weight进行三点插值计算梯度得到grad_y。
- 计算公式：

  $$
  grad\_y[b,c,idx[b,n,i]] =
  grad\_y[b,c,idx[b,n,i]] + grad\_x[b,c,n]*weight[b,n,i]\\ i\in[0,2]\ b\in[0,B) \ c\in[0,C) \ n\in[0,N)
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
      <td>grad_x</td>
      <td>输入</td>
      <td>网络反向传播前一步的梯度值，对应公式中的`grad_x`。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>5HD（NC1HWC0）、ND</td>
    </tr>
    <tr>
      <td>idx</td>
      <td>输入</td>
      <td>目标特征三个相邻点的特征索引，对应公式中的`idx`。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>目标特征三个相邻点的特征权重，对应公式中的`weight`。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>m</td>
      <td>属性</td>
      <td>输入特征最低维度的大小，用于指导infershape时推导`grad_y`的形状。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_y</td>
      <td>输出</td>
      <td>梯度计算结果，对应公式中的`grad_y`。数据类型和数据格式需要与`grad_x`的数据类型和数据格式一致。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>5HD（NC1HWC0）、ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `idx`中的取值范围为[0, m)（非负且小于`m`），越界取值（负数或大于等于`m`）不保证计算结果的正确性。
- `grad_x` shape为(B, C, N)，`idx`/`weight` shape为(B, N, 3)，`grad_y` shape为(B, C, M)；且`idx`/`weight`的B/N维与`grad_x`一致，`grad_y`的B/C维与`grad_x`一致、M等于属性`m`，shape不一致时接口校验失败。
- 属性`m`必须大于0。
- `grad_x`/`weight`/`grad_y`为FLOAT32或FLOAT16，`idx`为INT32或INT64。
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`grad_x`、`grad_y`的数据格式为5HD（即NC1HWC0，与算子IR定义`FORMAT_NC1HWC0`一致）。
- <term>Ascend 950PR&950DT系列产品</term>：`grad_x`、`grad_y`的数据格式为ND。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_three_interpolate_backward](examples/test_aclnn_three_interpolate_backward.cpp) | 通过[aclnnThreeInterpolateBackward](docs/aclnnThreeInterpolateBackward.md)接口方式调用ThreeInterpolateBackward算子。 |
<!--
| 图模式 | [test_geir_three_interpolate_backward](examples/test_geir_three_interpolate_backward.cpp)  | 通过[算子IR](op_graph/three_interpolate_backward_proto.h)构图方式调用ThreeInterpolateBackward算子。         |
-->
