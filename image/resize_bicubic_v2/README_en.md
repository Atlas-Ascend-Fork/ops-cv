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

  For the output pixel at $(X,Y)$ (row $Y$, column $X$), the coordinate mapping rule (determined by the `align_corners` attribute, consistent with PyTorch `upsample_bicubic2d`) maps it to the source coordinate $(h,w)$ in the input image. Taking $(h,w)$ as the reference point, the 4×4 (16 in total) neighboring pixels of the input image are selected: the neighbor at row $i$, column $j$ has coordinates $h_i = \lfloor h \rfloor - 1 + i$ and $w_j = \lfloor w \rfloor - 1 + j$ ($i, j \in \{0,1,2,3\}$), whose pixel value is denoted as $a_{ij}$. The distances from the source coordinate to the neighboring pixels are defined as $d_i = |h - h_i|$ and $d_j = |w - w_j|$ (i.e. the arguments of the W function, in the range [0, 2)).

  The cubic interpolation weight function (coefficient $a = -0.75$):

  $$
  W(x) = \begin{cases}
  (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{for } |x|≤1 \\
  a|x|^3 -5a|x|^2 + 8a|x| - 4a  & \text{for } 1<|x|<2 \\
  0 & \text{otherwise} \\
  \end{cases}
  $$

  Output pixel value (weighted sum of the 16 neighboring pixels by row/column distance):
  $$
  B(X,Y) = \sum_{i=0}^3 \sum_{j=0}^3a_{ij} \times W(d_i) \times W(d_j)
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
      <td>Height and width of the output image. A 1D tensor with 2 elements, each greater than 0; the values must be consistent with the H and W of the output y inferred by the format.</td>
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
      <td>Specifies whether to align the centers of the four corner pixels of the input and output tensors to preserve the corner pixel values, which determines the source coordinate mapping rule. Default value: false.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Attr</td>
      <td>Scaling coefficients of the spatial dimensions (height and width), with 2 elements: the output-to-input size ratio along the height and width respectively. Default value: {0.0, 0.0} (not specified; in this case the scaling coefficients are derived from the input and output sizes).</td>
      <td>ListFloat</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

- The input tensor `x` and output tensor `y` must both be 4D. The `N` and `C` dimensions of `x` must be greater than 0, and the `N` and `C` dimensions of `y` must match those of `x`, respectively.
- The `H` and `W` dimensions of the input and output must both be greater than 0.
- `size` must be a 1D INT32 tensor containing 2 elements that specify the output `H` and `W`; both values must be greater than 0.

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---- | ---- |----|
| aclnn interface | [test_aclnn_bicubic_v2](examples/test_aclnn_bicubic_v2.cpp) | Call the ResizeBicubicV2 operator through the [aclnnUpsampleBicubic2d](../upsample_bicubic2d/docs/aclnnUpsampleBicubic2d_en.md) interface. |
| Graph mode | -  | Call the ResizeBicubicV2 operator by building a graph through [operator IR](op_graph/resize_bicubic_v2_proto.h). |
