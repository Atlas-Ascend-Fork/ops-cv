# Iou3D

English | [简体中文](./README.md)

## Product Support

| Product | Support |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: Computes the 3D IoU matrix between two groups of 3D rotated boxes (7-DoF: `[x, y, z, w, h, d, theta]`).
- Computation principle:
  - BEV projection: Each box rotates by `theta` in the XY plane, producing four vertices: `P = center ± 0.5*w*(cos, sin) ± 0.5*h*(-sin, cos)`.
  - Intersection area: The operator performs vertex containment tests and edge intersections to collect the vertices of the intersection polygon. With 0 or 3 vertices, the operator directly computes the triangle area; with more than 3 vertices, it uses centroid decomposition plus polar-angle sorting to compute the sum.
  - Z-axis overlap: `real_d = max(min(z1max, z2max) - max(z1min, z2min), 0)`, where `z_min = z - 0.5*d` and `z_max = z + 0.5*d`.
- Computation formula:

  $$
  \text{IoU}_{3D} = \frac{\text{bev\_area} \times \text{real\_d}}{V_A + V_B - V_{inter} + \varepsilon}, \quad \varepsilon = 10^{-6}
  $$

  In the formula, `V = w * h * d` is the box volume, and `V_inter = bev_area * real_d` is the intersection volume.
- Three guards for numerical stability: `epsilon = 1e-6` is added to the denominator to prevent division by zero; the Z-axis overlap is clamped to a non-negative value; the area of a degenerate polygon (with fewer than 3 vertices) is set to 0.

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
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
    <td>bboxes</td>
    <td>Input</td>
    <td>Predicted boxes, with shape <code>[B, 7, N]</code>. The second dimension is the 7-DoF <code>[x, y, z, w, h, d, theta]</code>.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gtboxes</td>
    <td>Input</td>
    <td>Ground-truth boxes, with shape <code>[B, 7, K]</code>. The second dimension is the 7-DoF <code>[x, y, z, w, h, d, theta]</code>.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>iou</td>
    <td>Output</td>
    <td>3D IoU matrix, with shape <code>[B, N, K]</code> and a value range of <code>[0, 1]</code>. <code>iou[b, n, k]</code> is the 3D IoU between the n-th predicted box and the k-th ground-truth box in batch b.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

- 7-DoF semantics: `x, y, z` are the center coordinates, `w, h, d` are the box sizes in three directions, and `theta` is the rotation angle around the Z-axis (in radians).
- The batch dimension `B` of the three tensors must be the same, and `bboxes.shape[1] == gtboxes.shape[1] == 7`.

## Constraints

- Data types: `bboxes`, `gtboxes`, and `iou` must all be float32.
- Fixed channel of 7: `bboxes.shape[1]` and `gtboxes.shape[1]` must equal 7 (7-DoF). Otherwise, the operator rejects the input.
- Upper limit of K:
  - <term>Ascend 950PR&950DT products</term>: The new implementation no longer limits `K <= 2000`. However, B, N, and K must not exceed the `uint32_t` range, and the number of pairs processed by each core cannot exceed the `uint32_t` range.
  - Other supported products use the historical implementation, which requires `K <= 2000`.
- Invalid box ordering: Within the same batch, invalid boxes (`w * h * d = 0`) cannot be placed before valid boxes.
- Broadcast is not supported. The two inputs must be independent rank-3 ND tensors, and only the batch dimension B must be the same.
- <term>Ascend 950PR&950DT products</term>: Graph mode supports dynamic shapes (-1 indicates an unknown dimension) and dynamic ranks (-2 indicates an unknown rank).

## Invocation

| Invocation Method | Sample | Description |
| ------------ | ------------ | ------------ |
| GE graph mode | [test_geir_iou3d](./examples/test_geir_iou3d.cpp) | Invokes the Iou3D operator by constructing a graph through the [operator IR](./op_graph/iou3d_proto.h) |
