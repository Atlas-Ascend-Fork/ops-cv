# ToAbsoluteBBox

English | [简体中文](./README.md)

## Product Support

| Product | Support |
| :----------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: Converts normalized bounding box coordinates (relative coordinates, value range [0, 1]) into absolute pixel coordinates according to the image height and width. This operator is often used in the post-processing phase of inference of object detection models (such as Faster R-CNN, SSD, and YOLO) to restore the normalized bboxes predicted by the model to the absolute coordinate space of the image.

  The four coordinate components of a bbox are in the order of `[ymin, xmin, ymax, xmax]`: ymin/ymax is multiplied by the image height H, and xmin/xmax is multiplied by the image width W. H comes from `shape_hw[1]`, and W comes from `shape_hw[2]` (the layout of `shape_hw` is `[batch, H, W, C]`). The attribute `reversed_box` controls the bbox data layout and thereby determines the position of the coordinate dimension (the axis of length 4):

  - When `reversed_box = false` (default): The layout is `(batch, num, 4)`, the coordinate dimension is the last axis, and the scaling factors in coordinate order are `[H, W, H, W]`.
  - When `reversed_box = true`: The layout is `(batch, 4, num)`, the coordinate dimension is the second axis, and the scaling factors in coordinate order are `[H, W, H, W]`.

- Computation formula:

  $$
  y_{min}^{abs} = y_{min}^{norm} \times H,\quad x_{min}^{abs} = x_{min}^{norm} \times W
  $$

  $$
  y_{max}^{abs} = y_{max}^{norm} \times H,\quad x_{max}^{abs} = x_{max}^{norm} \times W
  $$

  In the formula, H is the image height and W is the image width.

## Parameter Description

<table style="table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 540px">
  <col style="width: 200px">
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
      <td>normalized_boxes</td>
      <td>Input</td>
      <td>Normalized bounding box coordinates. They correspond to the normalized coordinates in the formula, with a value range of [0, 1], and the coordinate components are in the order of [ymin, xmin, ymax, xmax]. When reversed_box=false, the layout is (batch, num, 4) and the coordinate dimension is the last axis; when reversed_box=true, the layout is (batch, 4, num) and the coordinate dimension is the second axis. 3D tensor.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape_hw</td>
      <td>Input</td>
      <td>Image size information, with the layout [batch, H, W, C]. It corresponds to H=shape_hw[1] and W=shape_hw[2] in the formula. The shape is fixed to (4,).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reversed_box</td>
      <td>Optional Attr</td>
      <td><ul><li>Controls the bbox data layout and the position of the coordinate dimension. false: The layout is (batch, num, 4) and the coordinate dimension is the last axis. true: The layout is (batch, 4, num) and the coordinate dimension is the second axis.</li><li>Default value: false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Bounding box with absolute coordinates. It corresponds to the absolute coordinates in the formula. The data type and shape are the same as those of normalized_boxes.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- normalized_boxes and y must have the same data type and shape, and only FLOAT16 and FLOAT32 are supported.
- shape_hw is fixed to INT32 with shape (4,), and the layout is [batch, H, W, C]. H is shape_hw[1], and W is shape_hw[2].
- normalized_boxes is 3D, and the position of the coordinate dimension (the axis of length 4) is determined by reversed_box: It is the last axis when the value is false, and the second axis when the value is true.
- All tensors support only the ND format, and empty tensors are not supported.
- In the FLOAT16 scenario, the operator completes the coordinate scaling directly at the FLOAT16 precision, without FLOAT32 intermediate conversion.
- Determinism: This operator performs element-wise multiplication (no reduce and no accumulation). The same input produces the same output (bitwise reproducible).

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | ----------------------------------------------------------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_to_absolute_b_box](./examples/arch35/test_geir_to_absolute_b_box.cpp) | Invokes the ToAbsoluteBBox operator by constructing a graph through the [operator IR](./op_graph/to_absolute_b_box_proto.h). |
