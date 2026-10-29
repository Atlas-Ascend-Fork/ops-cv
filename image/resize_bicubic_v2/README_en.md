# ResizeBicubicV2

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |

## Function Description

- Operator function: resizes an image to the specified size using bicubic interpolation.
- Calculation formula:

  Pixel positions of the surrounding 16 points:

  $$
  W(x) = \begin{cases}
  (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{for } |x|≤1 \\
  a|x|^3 -5a|x|^2 + 8a|x| - 4a  & \text{for } 1<|x|<2 \\
  0 & \text{otherwise} \\
  \end{cases}
  $$

  Pixel values:
  $$
  B(X,Y) = \sum_{i=0}^3 \sum_{j=0}^3a_{ij} \times W(i) \times W(j)
  $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1000px">
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
      <td>4D tensor of the input image, corresponding to x in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>Height and width of the output image.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Image resized by bicubic interpolation.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Attr</td>
      <td>Specifies whether to align the centers of the four corner pixels of the input and output tensors to preserve the corner pixel values. Default value: false.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Attr</td>
      <td>Scaling coefficients of the spatial dimensions (height and width). Default value: {0.0, 0.0}.</td>
      <td>ListFloat</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

None

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---- | ---- |----|
| aclnn interface | [test_aclnn_resize_bicubic_v2](examples/test_aclnn_bicubic_v2.cpp) | Call the ResizeBicubicV2 operator through the [aclnnUpsampleBicubic2d](../upsample_bicubic2d/docs/aclnnUpsampleBicubic2d_en.md) interface. |
| Graph mode | -  | Call the ResizeBicubicV2 operator by building a graph through [operator IR](op_graph/resize_bicubic_v2_proto.h). |
