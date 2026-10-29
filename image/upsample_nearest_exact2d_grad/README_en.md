# UpsampleNearestExact2dGrad

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

- Operator function: backward propagation of [UpsampleNearest](../upsample_nearest/README_en.md) when exact_mode is true.
- Calculation formula:

  For the input gradOut(N, C, h, w) and any point (N, C, H, W) on the output gradInput, the following relationship applies:

  $$
  gradInput(N, C, H, W) = \sum_{h = srcH}^{srcHUp - 1}\sum_{w = srcW}^{srcWUp - 1}gradOut(N, C, h, w)
  $$

  In the formula:

  $$
  scalesH = outputSize[0]/inputSize[2]
  $$

  $$
  scalesW = outputSize[1]/inputSize[3]
  $$

  $$
  srcH = Min(ceil(scalesH * H - 0.5), outputSize[0])
  $$

  $$
  srcHUp = Min(ceil(scalesH * (H + 1) - 0.5), outputSize[0])
  $$

  $$
  srcW = Min(ceil(scalesW * W - 0.5), outputSize[1])
  $$

  $$
  srcWUp = Min(ceil(scalesW * (W + 1) - 0.5), outputSize[1])
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
      <td>Represents the gradient tensor for the backward computation, corresponding to `grad_output` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>Indicates the spatial sizes of input `grad_output` in the H and W dimensions. The size is 2, and all elements are greater than 0.</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_size</td>
      <td>Attr</td>
      <td>Indicates the spatial sizes of output `grad_input` in the N, C, H, and W dimensions respectively. The size is 4, and all elements are greater than 0.</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the height dimension of output `grad_input`, corresponding to `scales_h` in the formula. Negative values cannot be passed in.</li><li>The default value is empty.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the width dimension of output `grad_input`, corresponding to `scales_w` in the formula. Negative values cannot be passed in.</li><li>The default value is empty.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>Output</td>
      <td>Represents the output tensor of the backward computation, corresponding to `grad_input` in the formula. The data type and data format are the same as the data type and data format of the input parameter `grad_output`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_nearest_exact1d_backward](examples/test_aclnn_upsample_nearest_exact1d_backward.cpp) | Call the UpsampleNearestExact2dGrad operator through the [aclnnUpsampleNearestExact1dBackward](docs/aclnnUpsampleNearestExact1dBackward_en.md) interface. |
| aclnn interface  | [test_aclnn_upsample_nearest_exact2d_backward](examples/test_aclnn_upsample_nearest_exact2d_backward.cpp) | Call the UpsampleNearestExact2dGrad operator through the [aclnnUpsampleNearestExact2dBackward](docs/aclnnUpsampleNearestExact2dBackward_en.md) interface. |
