# RoiAlignRotatedGrad

English | [简体中文](./README.md)

## Product Support

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×   |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: Passes gradients back to the corresponding positions through the coordinates of each point of the rotated boxes.

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
      <td>x_grad</td>
      <td>Input</td>
      <td>Gradient of the input feature map.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>Input</td>
      <td>ROI bounding boxes.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>Attr</td>
      <td>Scaling ratio of the ROI bounding boxes.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sampling_ratio</td>
      <td>Attr</td>
      <td>Sampling ratio used when computing RoiAlign.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y_grad_shape</td>
      <td>Attr</td>
      <td>Shape of the output.</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_h</td>
      <td>Attr</td>
      <td>Height of the pooled feature map in the RoiAlign output.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_w</td>
      <td>Attr</td>
      <td>Width of the pooled feature map in the RoiAlign output.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>aligned</td>
      <td>Attr</td>
      <td>Whether to perform quantization. When the value is true, the operator subtracts 0.5 from the center coordinates of the target box.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clockwise</td>
      <td>Attr</td>
      <td>Clock direction. true indicates the clockwise direction.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y_grad</td>
      <td>Output</td>
      <td>Output feature map.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The input x_grad is 4D (N, pooled_h, pooled_w, C). The channel count C is in the range [0, 1024], and pooled_h and pooled_w are in the range [0, 1024].
- The input rois is 2D (6, N). Dimension 0 is fixed to 6 (the number of fields), and dimension 1 N is the number of ROIs and must equal x_grad.dim(0).
- x_grad.dim(1) must equal the attribute pooled_h, and x_grad.dim(2) must equal the attribute pooled_w.
- x_grad.dim(3) must equal y_grad_shape[3] (the channel counts are the same).
- The length of the attribute y_grad_shape must be 4, each element must be a non-negative integer, and the semantics are [B, H, W, C].
- The attributes pooled_h and pooled_w must be greater than 0, and sampling_ratio must be greater than or equal to 0.
- batch_ind in row 0 of rois must be an integer value within [0, y_grad_shape[0]). The kernel interprets the value by truncated rounding (rounding toward zero), and an out-of-range value is undefined behavior. angle in row 5 is a value in radians.
- The output y_grad is the accumulated gradient, which can be positive or negative, without value range clipping. Positions not covered by ROIs are 0.
- This operator is precision-sensitive. Lower-precision processing by the framework is prohibited, and only the single float32 data type combination is supported.

## Invocation

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_roi_align_rotated_grad](examples/test_geir_roi_align_rotated_grad.cpp) | Invokes the RoiAlignRotatedGrad operator in graph mode. |
