# RotatedBoxDecode

English | [简体中文](./README.md)

## Product Support

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

RotatedBoxDecode is a box regression decoding operator in rotated object detection. It adds the offsets (deltas) predicted by the network to the preset anchor boxes (anchor_box) to restore the final rotated detection boxes.

The input is in the corner-point format `[lx, ly, rx, ry, angle]` (the unit of the angle is degree). The computation consists of six steps:

1. **Convert anchor box corner points to the center**: Computes `(cx, cy, w, h)` from `(lx, ly, rx, ry)`, with the lower limits of the width and height clamped to 1
2. **Normalize deltas**: The operator divides each of the five channels of deltas by weight
3. **Solve for the center**: `t_cx = a_cx + Δx·a_w` and `t_cy = a_cy + Δy·a_h`
4. **Solve for the width and height**: `t_w = exp(Δw)·a_w` and `t_h = exp(Δh)·a_h`
5. **Solve for the angle**: `θ_t = atan(tan(θ_a) + Δt)` (degree↔radian conversion, and the addition in tan space resists the 180° periodicity)
6. **Convert the center to corner points**: Restores `(lx', ly', rx', ry')` from `(cx, cy, w, h)`, and passes the angle through

The computation formulas are as follows. In the formula, `Δ = deltas / weight` (weight is a 5-tuple `[wx, wy, ww, wh, wt]`, divided channel by channel), the angles are added in tan space and then computed with atan, the principal value range of the output is `[-90°, 90°)`, and `max(·, 1)` clamps the lower limits of the width and height to prevent degeneration:

$$
\begin{aligned}
a_w &= \max(rx - lx,\ 1),\quad a_h = \max(ry - ly,\ 1) \\
a_{cx} &= lx + a_w / 2,\quad a_{cy} = ly + a_h / 2 \\
\Delta &= \text{deltas} / \text{weight} \\
t_{cx} &= a_{cx} + \Delta x \cdot a_w \\
t_{cy} &= a_{cy} + \Delta y \cdot a_h \\
t_w    &= e^{\Delta w} \cdot a_w \\
t_h    &= e^{\Delta h} \cdot a_h \\
\theta_t &= \arctan\big(\tan(\theta_a) + \Delta t\big) \\
lx' &= t_{cx} - t_w/2,\quad rx' = t_{cx} + t_w/2 \\
ly' &= t_{cy} - t_h/2,\quad ry' = t_{cy} + t_h/2
\end{aligned}
$$

## Parameter Description

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
<td>anchor_box</td>
<td>Input</td>
<td>Anchor box, 3D (B, 5, N), where 5 = [lx, ly, rx, ry, angle], and the unit of the angle is degree. It corresponds to anchor_box in the formula.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>deltas</td>
<td>Input</td>
<td>Network regression offsets, 3D (B, 5, N), where 5 = [dx, dy, dw, dh, dt]. They correspond to deltas in the formula.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>Output</td>
<td>Decoded rotated boxes, 3D (B, 5, N), where 5 = [lx', ly', rx', ry', θ_t]. The shape/dtype is the same as that of anchor_box.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>weight</td>
<td>Optional Attr</td>
<td>5-tuple weights [wx, wy, ww, wh, wt]. The operator divides each channel of deltas by the corresponding weight for normalization. Default value: [1, 1, 1, 1, 1].</td>
<td>ListFloat</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- anchor_box, deltas, and y have exactly the same shape and the same dtype.
- The tensors must be 3D, shape[1] must be 5, shape[0] ≥ 0, and shape[2] ≥ 0.
- The length of weight must be 5, and no element can be 0 (otherwise inf/NaN is generated; the kernel does not check this).
- The data layout is ND, and other layouts are not supported.
- Tiling is computed at compilation time, and dynamic shapes are not supported.
- During FLOAT16 computation, the operator promotes the data to FLOAT32 and then casts the result back to FLOAT16.
- Empty tensors (B=0 or N=0) are supported.

## Invocation

| Invocation Method | Sample Code | Description |
| --- | --- | --- |
| GE graph mode | [test_geir_rotated_box_decode.cpp](examples/arch35/test_geir_rotated_box_decode.cpp) | Invokes the operator by constructing a graph through the [operator IR](op_graph/rotated_box_decode_proto.h). |
