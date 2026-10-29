# ResizeBilinearV2Grad

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function Description

- Operator function: backpropagation of [ResizeBilinearV2](../resize_bilinear_v2/README_en.md).

- Calculation formula:

  Assume that the four points $Q_{11}(x_1, y_1), Q_{12}(x_1, y_2), Q_{21}(x_2, y_1), Q_{22}(x_2, y_2)$ in $grads$ are known.

  Assume that the position of $grads$ in the output image is $(h', w')$, and its floating-point coordinate mapped back to the original image is $(pos\_h, pos\_w)$. The offsets are defined as follows:

  $$
  d_h = pos\_h - x_1
  $$

  $$
  d_w = pos\_w - y_1
  $$

  The corresponding gradient accumulation formulas are as follows:

  Top-left point $Q_{11}$:

  $$
  y(N, C, x_1, y_1) += grads(N, C, h', w') \cdot (1 - d_h) \cdot (1 - d_w)
  $$

  Top-right point $Q_{12}$:

  $$
  y(N, C, x_1, y_2) += grads(N, C, h', w') \cdot (1 - d_h) \cdot d_w
  $$

  Bottom-left point $Q_{21}$:

  $$
  y(N, C, x_2, y_1) += grads(N, C, h', w') \cdot d_h \cdot (1 - d_w)
  $$

  Bottom-right point $Q_{22}$:

  $$
  y(N, C, x_2, y_2) += grads(N, C, h', w') \cdot d_h \cdot d_w
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
      <td>grads</td>
      <td>Input</td>
      <td>Gradient tensor of the output of the forward resize, corresponding to grads in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>original_image</td>
      <td>Input</td>
      <td>Input tensor of the forward resize.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Gradient tensor of the input of the forward resize, corresponding to y in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_bilinear_2d_backward](examples/test_aclnn_upsample_bilinear_2d_backward.cpp) | Call the ResizeBilinearV2Grad operator through the [aclnnUpsampleBilinear2dBackward](docs/aclnnUpsampleBilinear2dBackward_en.md) interface. |
| Graph mode | -  | Call the ResizeBilinearV2Grad operator by building a graph through [operator IR](op_graph/resize_bilinear_v2_grad_proto.h). |
