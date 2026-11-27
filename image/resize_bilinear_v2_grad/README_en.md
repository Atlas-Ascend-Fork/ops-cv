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
      <td>Gradient tensor of the output of the forward resize, corresponding to grads in the formula. A 4-D tensor whose data format is the same as original_image.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>original_image</td>
      <td>Input</td>
      <td>Input tensor of the forward resize; only its shape and data type take part in the computation. A 4-D tensor whose N and C dimensions must be the same as grads. When grads is FLOAT16 or BFLOAT16, its data type must be the same as grads; when grads is FLOAT32, its data type can be FLOAT32, FLOAT16, or BFLOAT16.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td>When true, aligns the centers of the four corner pixels of the input and output images; defaults to false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>half_pixel_centers</td>
      <td>Optional attr</td>
      <td>When true, samples using half-pixel center coordinates; cannot be true at the same time as align_corners. Defaults to false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Optional attr</td>
      <td>Contains two scaling factors for H and W, representing the ratio of the corresponding spatial axis subscripts of grads to y; takes effect only when align_corners is false and the corresponding value is greater than 0. Defaults to [0.0, 0.0].</td>
      <td>LIST_FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Gradient tensor of the input of the forward resize, corresponding to y in the formula. Its shape, data type, and data format are the same as original_image.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
  </tbody></table>

## Constraints

- `align_corners` and `half_pixel_centers` cannot both be true.
- When `scales` is set, provide two scaling factors for H and W; the default value is `[0.0, 0.0]`.
- Data format restriction: the data formats of grads, original_image, and y must be consistent, and only NCHW and NHWC are supported; ND and other formats are not supported.
- Empty tensor restriction: empty tensors are not supported. The N, H, and W extents of grads and y must all be greater than 0; a non-positive N, H, or W extent is rejected during tiling with a shape error.
- Dynamic shape and dynamic rank: the operator declares support for dynamic shape and dynamic rank at registration (DynamicShapeSupportFlag, DynamicRankSupportFlag); that is, -1 (unknown dimension) and -2 (unknown rank) placeholders are allowed in shapes at compile time. At execution time, the real shape must be a determined 4-D shape satisfying the above restrictions, and the placeholders must be resolved before execution.
- Data type combination: the data types of grads and y can differ. When grads is FLOAT16 or BFLOAT16, the data type of y must be the same as grads; when grads is FLOAT32, y can be FLOAT32, FLOAT16, or BFLOAT16.
- scales effective condition: scales takes effect only when align_corners is false and the corresponding scales element is greater than 0; when align_corners is true, scales is ignored and the scaling ratio is derived from the H and W extents of grads and y.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_bilinear_2d_backward](examples/test_aclnn_upsample_bilinear_2d_backward.cpp) | Call the ResizeBilinearV2Grad operator through the [aclnnUpsampleBilinear2dBackward](docs/aclnnUpsampleBilinear2dBackward_en.md) interface. |
| Graph mode | -  | Call the ResizeBilinearV2Grad operator by building a graph through [operator IR](op_graph/resize_bilinear_v2_grad_proto.h). |
