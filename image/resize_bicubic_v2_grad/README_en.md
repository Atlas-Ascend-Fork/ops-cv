# ResizeBicubicV2Grad

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |

## Function Description

- Operator function: computes the gradient of the input image based on bicubic interpolation.
- Calculation formula:

  $$
  W(x) = \begin{cases}
  (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{for } |x|≤1 \\
  a|x|^3 -5a|x|^2 + 8a|x| - 4a  & \text{for } 1<|x|<2 \\
  0 & \text{otherwise} \\
  \end{cases}
  $$

  $$
  \frac{\partial L}{\partial X_{i,j}} = \sum_{i'} \sum_{j'} \frac{\partial L}{\partial Y_{i',j'}} \times W(i' - i) \times W(j' - j)
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
      <td>Image adjusted by the forward bicubic interpolation, corresponding to Y in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>original_image</td>
      <td>Input</td>
      <td>Height and width of the original image.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Input gradient of the forward Resize.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
  </tbody>
</table>

## Constraints

- None

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---- | ---- |----|
| aclnn interface  | [test_aclnn_resize_bicubic_v2_grad](examples/test_aclnn_resize_bicubic_v2_grad.cpp) | Call the ResizeBicubicV2Grad operator through the [aclnnUpsampleBicubic2dBackward](../upsample_bicubic2d_grad/docs/aclnnUpsampleBicubic2dBackward_en.md) interface. |
| Graph mode | -  | Call the ResizeBicubicV2Grad operator by building a graph through [operator IR](op_graph/resize_bicubic_v2_grad_proto.h). |
