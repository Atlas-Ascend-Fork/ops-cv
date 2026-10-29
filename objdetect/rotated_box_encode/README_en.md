# RotatedBoxEncode

English | [简体中文](./README.md)

## Product Support

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

The operator encodes the geometric deviations between the reference rotated box (anchor_box) and the corresponding ground-truth rotated box (gt_box) into the 5-channel regression delta targets `(dx, dy, dw, dh, dθ)` for regression supervision in the training phase of rotated object detection. Both the input and output are 3D tensors `(B, 5, N)`. The five channels are organized in the corner form `(x0, y0, x1, y1, θ_deg)`, where `θ_deg` is in degrees. Let `weight = (wx, wy, ww, wh, wa)`. The channel-wise encoding formulas are:

$$
\begin{aligned}
w_a &= \max(x1_a - x0_a,\ 1.0),\quad h_a = \max(y1_a - y0_a,\ 1.0) \\
cx_a &= x0_a + w_a / 2,\quad cy_a = y0_a + h_a / 2 \\
w_g &= \max(x1_g - x0_g,\ 1.0),\quad h_g = \max(y1_g - y0_g,\ 1.0) \\
cx_g &= x0_g + w_g / 2,\quad cy_g = y0_g + h_g / 2 \\
dx &= \frac{cx_g - cx_a}{w_a} \cdot wx \\
dy &= \frac{cy_g - cy_a}{h_a} \cdot wy \\
dw &= \ln\frac{w_g}{w_a} \cdot ww \\
dh &= \ln\frac{h_g}{h_a} \cdot wh \\
d\theta &= \bigl(\tan(\theta_g \cdot \pi/180) - \tan(\theta_a \cdot \pi/180)\bigr) \cdot wa
\end{aligned}
$$

In the formula, `max(·, 1.0)` prevents division by zero and `log(0)` caused by degenerate boxes, and propagates NaN according to the `np.maximum` semantics. For fp16 inputs, the operator internally promotes the precision to fp32, completes the logarithm, trigonometric, and division computations, and then writes the results back in fp16.

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
<td>Reference anchor rotated box. It corresponds to anchor_box in the formula.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>gt_box</td>
<td>Input</td>
<td>Ground-truth rotated box. It corresponds to gt_box in the formula.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>Output</td>
<td>Encoded 5-channel delta. It corresponds to (dx,dy,dw,dh,dθ) in the formula.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>weight</td>
<td>Optional Attr</td>
<td>5-channel encoding weights [wx,wy,ww,wh,wa]. Default value: [1.0,1.0,1.0,1.0,1.0].</td>
<td>ListFloat</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- anchor_box, gt_box, and y must have the same data type, and only FLOAT16 or FLOAT is supported.
- The shapes of anchor_box and gt_box must be exactly the same (no broadcast), the rank must be 3, and dimension 1 must be 5.
- If weight is provided, its length must be 5. If it is not provided, the default value `[1.0, 1.0, 1.0, 1.0, 1.0]` is used.
- For an empty tensor (B=0 or N=0), the operator short-circuits and directly returns an empty output without reporting an error.

## Invocation

| Invocation Method | Sample Code | Description |
| :--- | :--- | :--- |
| Graph mode | [test_geir_rotated_box_encode](./examples/arch35/test_geir_rotated_box_encode.cpp) | Invokes the RotatedBoxEncode operator by constructing a graph through the [operator IR](./op_graph/rotated_box_encode_proto.h). |
