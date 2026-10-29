# ResizeLinear

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |

## Function Description

- Operator function: resizes an image to the specified size using single linear interpolation.
- Calculation formula:

  For 1D data, the two surrounding points are used for weighted interpolation
  $$
  y = \frac{x_1 - x}{x_1 - x_0}y_0 + \frac{x - x_0}{x_1 - x_0}y_1
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
      <td>x</td>
      <td>Input</td>
      <td>3D tensor of the input image, with the shape (N, C, L), corresponding to x in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>1D INT32 tensor of length 1, giving the size of the output L dimension.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Image resized by single linear interpolation, corresponding to y in the formula.</td>
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
| aclnn interface  | [test_aclnn_resize_linear](examples/test_aclnn_resize_linear.cpp) | Call the ResizeLinear operator through the [aclnnUpsampleLinear1d](../upsample_linear1d/docs/aclnnUpsampleLinear1d_en.md) interface. |
| Graph mode | -  | Call the ResizeLinear operator by building a graph through [operator IR](op_graph/resize_linear_proto.h). |
