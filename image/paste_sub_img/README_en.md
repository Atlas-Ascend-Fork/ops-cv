# PasteSubImg

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

PasteSubImg is an area-accumulation operator for image or feature map patch stitching, and its computing paradigm is layout_transform. The operator extracts pixels from the rectangular sub-region of the source image `patch_img` specified by `core_area_coord`. It maps the pixels to the corresponding rectangular position on the target canvas `combine_img` using the `scale` coordinate scaling and the translation offsets given by `patch_coord`. The operator performs element-wise accumulation on each pixel within the target region (`combine_img[dst] += patch_img[src]`) and keeps pixels outside the target region unchanged. `combine_img` serves as both input and output in an in-place manner. The operator applies to scenarios that require accumulation over overlapping regions, such as image stitching, super-resolution backfilling, and sliding-window inference feature map aggregation.

The calculation formula is as follows, where `patch_coord = [px1, py1, px2, py2]` (only px1 and py1 participate in the calculation), `core_area_coord = [cx1, cy1, cx2, cy2]` (the right/bottom boundaries are exclusive), and scale is the coordinate scaling factor:

$$
s_{cy1}=\lfloor cy1 \cdot scale \rfloor,\ s_{cy2}=\lfloor cy2 \cdot scale \rfloor,\ s_{cx1}=\lfloor cx1 \cdot scale \rfloor,\ s_{cx2}=\lfloor cx2 \cdot scale \rfloor
$$

$$
d_{cy1}=\lfloor (cy1+py1) \cdot scale \rfloor,\ d_{cx1}=\lfloor (cx1+px1) \cdot scale \rfloor
$$

$$
combine\_img\_out[h, w, c] =
\begin{cases}
combine\_img[h, w, c] + patch\_img[s_{cy1}+(h-d_{cy1}),\ s_{cx1}+(w-d_{cx1}),\ c] & (h,w) \in [d_{cy1},\ d_{cy1}+\Delta h) \times [d_{cx1},\ d_{cx1}+\Delta w) \\
combine\_img[h, w, c] & \text{otherwise}
\end{cases}
$$

In the formula, $\Delta h = s_{cy2} - s_{cy1}$ and $\Delta w = s_{cx2} - s_{cx1}$. The C dimension is fully aligned for element-wise addition. The output has the same name as the input (in-place write-back), and the accumulation result is denoted as combine_img_out in the formula.

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
<td>patch_img</td>
<td>Input</td>
<td>Source image or feature map patch, 3D (H, W, C), HWC row-major; corresponds to patch_img in the formula.</td>
<td>UINT8, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>patch_coord</td>
<td>Input</td>
<td>Paste position coordinates [px1, py1, px2, py2] on the target canvas, 1D (4,); only px1 (column offset) and py1 (row offset) participate in the calculation, corresponding to patch_coord in the formula.</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>core_area_coord</td>
<td>Input</td>
<td>Source extraction region [cx1, cy1, cx2, cy2], 1D (4,); the right/bottom boundaries are exclusive, corresponding to core_area_coord in the formula.</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>combine_img</td>
<td>Input</td>
<td>Target canvas, 3D (H_out, W_out, C), HWC row-major; serves as both input and output in an in-place manner, and the accumulation result is directly written back to this tensor.</td>
<td>UINT8, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>combine_img</td>
<td>Output</td>
<td>Canvas after accumulation, with the same name, shape, and data type as the input combine_img, written back in place; denoted as combine_img_out in the formula.</td>
<td>UINT8, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>scale</td>
<td>Attr</td>
<td>Coordinate scaling factor, which is multiplied to all coordinate values to map the coordinate space to the pixel space; required attribute, no default value.</td>
<td>FLOAT</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- The data types of patch_img and combine_img must be the same, and their C dimensions must be the same.
- The shapes of both patch_coord and core_area_coord are [4]; core_area_coord must satisfy cx2 > cx1 and cy2 > cy1.
- The source region must be completely inside `patch_img`: `0 ≤ s_cy1 < s_cy2 ≤ H` and `0 ≤ s_cx1 < s_cx2 ≤ W`, where H and W are the height and width of `patch_img`.
- The target region must be completely inside `combine_img`: `d_cy1 ≥ 0`, `d_cy1 + Δh ≤ H_out`, `d_cx1 ≥ 0`, and `d_cx1 + Δw ≤ W_out`, where H_out and W_out are the height and width of `combine_img`.
- The value range of scale is [0.0, 256.0], and non-integer results are truncated to int indexes.
- The data layout is HWC row-major ND, and other layouts such as NCHW are not supported.
- combine_img is written back in place, so ensure that it uses contiguous memory before the call.
- Empty tensors are not supported (the length of each dimension must be ≥ 1).
- The uint8 path completes the addition through float16 transit, and overflow is handled as saturation (values greater than 255 are truncated to 255), which differs from the wrap-around semantics of some frameworks.

## Invocation Description

| Invocation Method | Sample Code | Description |
| --- | --- | --- |
| GE graph mode | - | Build a graph and call the operator through [operator IR](op_graph/paste_sub_img_proto.h). |
