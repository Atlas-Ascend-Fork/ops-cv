# AnchorResponseFlags

English | [简体中文](./README.md)

## Product Support

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: generates the response flags of anchors in an object detection network. Based on the center point positions of the ground truth bounding boxes, the operator determines which anchor grid positions are responsible for detecting objects and generates the corresponding flag bits.

- Calculation formula:

$$
cx_i = (x1_i + x2_i) \times 0.5
$$

$$
cy_i = (y1_i + y2_i) \times 0.5
$$

$$
grid\_x_i = \lfloor cx_i / stride\_w \rfloor
$$

$$
grid\_y_i = \lfloor cy_i / stride\_h \rfloor
$$

$$
grid\_idx_i = grid\_y_i \times feat\_w + grid\_x_i
$$

$$
responsible\_grid[grid\_idx_i] = 1
$$

$$
output = repeat(responsible\_grid, num\_base\_anchors)
$$

- Evaluation precision: The divisions in the above formulas are evaluated in float32 precision. Both the kernel and the reference implementation perform the division in float32 before rounding down. When stride exceeds 2^24 (the maximum integer that float32 can represent exactly), the divisor participates in the computation after being rounded to its float32 value.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>gt_bboxes</td>
      <td>Input</td>
      <td>Ground truth bounding box coordinates. The shape is [N, 4], and the format is [x1, y1, x2, y2].</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>flags</td>
      <td>Output</td>
      <td>Anchor response flags. The shape is [feat_h * feat_w * num_base_anchors], and each value is 0 or 1.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>featmap_size</td>
      <td>Attr</td>
      <td>Feature map size. It is a list of length not less than 2: [feat_h, feat_w]. Only the first two elements are used in the computation; any additional elements are ignored.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attr</td>
      <td>Strides. It is a list of length not less than 2: [stride_w, stride_h], and the values must be positive integers. The stride in the x direction comes first, and the stride in the y direction comes second. Only the first two elements are used in the computation; any additional elements are ignored.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_base_anchors</td>
      <td>Attr</td>
      <td>Number of anchors at each grid position.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

1. The input gt_bboxes must be a 2D tensor, and the second dimension must be 4. The number of rows N must not exceed INT32_MAX (2^31-1).
2. The lengths of both featmap_size and strides must be not less than 2. Only the first two elements are used in the computation (featmap_size uses [feat_h, feat_w], strides uses [stride_w, stride_h]); any additional elements are ignored.
3. The values in featmap_size must be positive integers (excluding 0) and must not exceed INT32_MAX.
4. The values in strides must be positive integers (excluding 0) and must not exceed INT32_MAX.
5. num_base_anchors must be in the range [0, INT32_MAX] (non-negative).
6. The output dtype is fixed to uint8 and does not change with the input dtype.
7. If gt_bboxes is empty (N=0), the output is an all-zero tensor.
8. If a mapped grid index exceeds the feature map range, the index is clipped to the valid range: a negative index is clipped to 0 (that is, when the center point lies outside the left/top boundary of the feature map, it maps to the first row/column grid), and a positive out-of-range index is clipped to feat_size - 1.
9. When a center point coordinate is NaN or Inf (including the case where the center point sum overflows to Inf), that center point maps to grid index 0.
10. gt_bboxes supports dynamic shape: the 0th dimension (the number of rows N) can be set to -1 (unknown dim) at graph construction time, and unknown rank (shape [-2]) is also supported. In this case, InferShape skips validation of the input dimensions, and the output shape [feat_h * feat_w * num_base_anchors] is derived only from the attributes, so it can be used with dynamic batch graphs; N is materialized from the actual input before graph execution.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_anchor_response_flags](examples/arch35/test_geir_anchor_response_flags.cpp) | Call the AnchorResponseFlags operator by constructing a graph in the [operator IR](op_graph/anchor_response_flags_proto.h) mode. |
