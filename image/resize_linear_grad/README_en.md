# ResizeLinearGrad

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: computes the gradient of the input image based on bilinear interpolation.
- Calculation formula:

  $$
  \frac{\alpha L}{\alpha X_i} = \frac{\alpha L}{\alpha Y_i} \times \frac{y_1 - y_0}{x_1 - x_0}
  $$

## Parameter Description

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
      <th>Parameter name</th>
      <th>Input/Output/Attr</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>grads</td>
      <td>Input</td>
      <td>Image adjusted by the forward single linear interpolation, corresponding to Y in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>original_image</td>
      <td>Input</td>
      <td>Height and width of the original image.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Input gradient of the forward Resize.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- None

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---- | ---- |----|
| aclnn interface  | [test_aclnn_resize_linear_grad](examples/test_aclnn_resize_linear_grad.cpp) | Call the ResizeLinearGrad operator through the [aclnnUpsampleLinear1dBackward](../upsample_bilinear2d_grad/docs/aclnnUpsampleLinear1dBackward_en.md) interface. |
| Graph mode | -  | Call the ResizeLinearGrad operator by building a graph through [operator IR](op_graph/resize_linear_grad_proto.h). |
