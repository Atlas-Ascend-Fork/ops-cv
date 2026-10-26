# ResizeLinear

## 产品支持情况

| 产品 | 是否支持 |
| :---- | :----: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |

## 功能说明

- 算子功能：使用单线性插值调整图像大小到指定的大小。
- 计算公式：

  对一维数据使用周围两个点进行加权插值
  $$
  y = \frac{x_1 - x}{x_1 - x_0}y_0 + \frac{x - x_0}{x_1 - x_0}y_1
  $$

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
      <td>x</td>
      <td>输入</td>
      <td>输入图像的三维Tensor，形状为(N, C, L)，对应公式中x。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>长度为1的一维INT32 Tensor，给出输出L维的大小。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>单线性插值调整后图像，对应公式中y。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 无

## 调用说明

| 调用方式 | 样例代码 | 说明                                                                                                 |
| ---- | ---- |----------------------------------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_resize_linear](examples/test_aclnn_resize_linear.cpp) | 通过[aclnnUpsampleLinear1d](../upsample_linear1d/docs/aclnnUpsampleLinear1d.md)接口方式调用ResizeLinear算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/resize_linear_proto.h)构图方式调用ResizeLinear算子。         |
