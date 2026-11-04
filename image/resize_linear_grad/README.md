# ResizeLinearGrad

## 产品支持情况

| 产品 | 是否支持 |
| :---- | :----: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：[ResizeLinear](../resize_linear/README.md) 的反向传播。
- 计算公式：

  设 $\lambda(x_i)$ 为正向插值输出位置 $x$ 对原始输入位置 $x_i$ 的插值权重，则反向传播按下式将输出梯度累加到输入梯度：

  $$
  \mathrm{gradInput}(x_i) \mathrel{+}= \mathrm{gradOut}(x) \cdot \lambda(x_i)
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
      <td>grads</td>
      <td>输入</td>
      <td>三维Tensor（逻辑形状为[N,C,L_out]），正向ResizeLinear输出的梯度，对应公式中gradOut。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>original_image</td>
      <td>输入</td>
      <td>三维Tensor（逻辑形状为[N,C,L_in]），用于提供正向ResizeLinear输入的形状；N、C维与grads相同。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>三维Tensor（逻辑形状为[N,C,L_in]），对正向ResizeLinear输入的梯度，形状与original_image相同。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 无

## 调用说明

| 调用方式 | 样例代码 | 说明                                                                                                                            |
| ---- | ---- |-------------------------------------------------------------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_resize_linear_grad](examples/test_aclnn_resize_linear_grad.cpp) | 通过[aclnnUpsampleLinear1dBackward](../upsample_bilinear2d_grad/docs/aclnnUpsampleLinear1dBackward.md)接口方式调用ResizeLinearGrad算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/resize_linear_grad_proto.h)构图方式调用ResizeLinearGrad算子。         |
