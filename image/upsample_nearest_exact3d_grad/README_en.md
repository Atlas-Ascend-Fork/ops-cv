# UpsampleNearestExact3dGrad

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: backward computation of [UpsampleNearestExact3d](../upsample_nearest_exact3d/README_en.md).
- Calculation formula:

  For the input gradOut(N, C, d, h, w) and any point (N, C, D, H, W) on the output gradInput, the following relationship applies:

  $$
  gradInput(N, C, D, H, W) = \sum_{d = srcD}^{srcDUp - 1}\sum_{h = srcH}^{srcHUp - 1}\sum_{w = srcW}^{srcWUp - 1}gradOut(N, C, d, h, w)
  $$

  In the formula:

  $$
  scaleD =\begin{cases}
  scalesD & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[0]/inputSize[2] &  Otherwise\\
  \end{cases}
  $$

  $$
  scaleH =\begin{cases}
  scalesH & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[1]/inputSize[3] &  Otherwise\\
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  scalesW & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[2]/inputSize[4] &  Otherwise\\
  \end{cases}
  $$

  $$
  srcD = Min(ceil(scaleD * D - 0.5), outputSize[0])
  $$

  $$
  srcDUp = Min(ceil(scaleD * (D + 1) - 0.5), outputSize[0])
  $$

  $$
  srcH = Min(ceil(scaleH * H - 0.5), outputSize[1])
  $$

  $$
  srcHUp = Min(ceil(scaleH * (H + 1) - 0.5), outputSize[1])
  $$

  $$
  srcW = Min(ceil(scaleW * W - 0.5), outputSize[2])
  $$

  $$
  srcWUp = Min(ceil(scaleW * (W + 1) - 0.5), outputSize[2])
  $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attr</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>grad_output</td>
      <td>Input</td>
      <td>Represents the gradient tensor for the backward computation, corresponding to the input `gradOutput` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>input_size</td>
      <td>Attr</td>
      <td>Indicates the spatial sizes of output `y` in the N, C, D, H, and W dimensions respectively. The size is 5, and all elements are greater than 0. The following requirements must be met: input_size[0] == grad_output_tensor_size[0]; input_size[1] == grad_output_tensor_size[1].</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Optional attr</td>
      <td><ul><li>Indicates the spatial sizes of input `grad_output` in the D, H, and W dimensions. The size is 3, and all elements are greater than 0. The following requirements must be met: grad_output_tensor_size[2] == floor(input_size[2] * scales[0]) == output_size[0]; grad_output_tensor_size[3] == floor(input_size[3] * scales[1]) == output_size[1]; grad_output_tensor_size[4] == floor(input_size[4] * scales[2]) == output_size[2].</li><li>The default value is {0, 0, 0}.</li></ul></td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the scaling array along each dimension. It contains three elements: scales_depth, scales_height, and scales_width. It corresponds to `scalesD`, `scalesH`, and `scalesW` in the formula.</li><li>The default value is {0.0f, 0.0f, 0.0f}.</li></ul></td>
      <td>LISTFLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor of the backward computation, corresponding to `gradInput` in the formula. The data type and data format are the same as the data type and data format of the input parameter `grad_output`. The shape depends on the input `input_size`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_nearest_exact3d_backward](examples/test_aclnn_upsample_nearest_exact3d_backward.cpp) | Call the UpsampleNearestExact3dGrad operator through the [aclnnUpsampleNearestExact3dBackward](docs/aclnnUpsampleNearestExact3dBackward_en.md) interface. |
| Graph mode | - | Call the UpsampleNearestExact3dGrad operator by constructing a graph in the [operator IR](op_graph/upsample_nearest_exact3d_grad_proto.h) mode. |
