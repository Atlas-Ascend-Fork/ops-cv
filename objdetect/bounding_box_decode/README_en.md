# BoundingBoxDecode

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

The operator decodes the offsets `deltas` predicted by the regression branch of object detection together with the regions of interest `rois` into absolute boxes `bboxes` in the image coordinate space. The boxes are used for subsequent NMS, box drawing, or COCO mAP evaluation. This operator is the inverse operation of `bounding_box_encode`. The two operators share the same `(means, stds)` normalization parameters and the same `(x1, y1, x2, y2)` anchor box coordinate convention. They must be used together to ensure consistent encoding between training and inference.

Assume that the input `rois=(x1, y1, x2, y2)`, the offsets `deltas=(dx', dy', dw', dh')`, `means=(m0,m1,m2,m3)`, `stds=(s0,s1,s2,s3)`, and `max_shape=(H, W)`. The decoding steps are as follows:

$$
pw = x2 - x1 + 1,\quad pcx = (x1 + x2) \times 0.5
$$

$$
dx = dx' \times s0 + m0,\quad dw = dw' \times s2 + m2
$$

$$
gw = pw \times \exp(dw),\quad gx = pcx + pw \times dx
$$

$$
x1_{out} = \mathrm{clip}(gx - gw \times 0.5 + 0.5,\ 0,\ W - 1)
$$

The y dimension (y1, y2) follows the same rule, and the clipping upper bound is H-1. The clipping upper bound is the last valid pixel `W-1`/`H-1`, and the output coordinate domain is `[0, W-1]`/`[0, H-1]`. For fp16 input, the intermediate `exp` and multiply-accumulate computations are performed in the fp32 domain and then cast back to fp16.

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
<td>rois</td>
<td>Input</td>
<td>Region of interest coordinates (x1,y1,x2,y2), corresponding to rois in the formula.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>deltas</td>
<td>Input</td>
<td>Regression offsets (dx',dy',dw',dh'), corresponding to deltas in the formula.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>bboxes</td>
<td>Output</td>
<td>Decoded absolute boxes (x1,y1,x2,y2), corresponding to bboxes in the formula.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>means</td>
<td>Optional attr</td>
<td>Means for denormalizing deltas. The length is 4, and the default value is [0.0, 0.0, 0.0, 0.0].</td>
<td>FLOAT</td>
<td>-</td>
</tr>
<tr>
<td>stds</td>
<td>Optional attr</td>
<td>Standard deviations for denormalizing deltas. The length is 4, all elements are not 0, and the default value is [1.0, 1.0, 1.0, 1.0].</td>
<td>FLOAT</td>
<td>-</td>
</tr>
<tr>
<td>max_shape</td>
<td>Attr</td>
<td>Clipping upper limit (H, W) of the decoded boxes. The length is 2.</td>
<td>INT64</td>
<td>-</td>
</tr>
<tr>
<td>wh_ratio_clip</td>
<td>Optional attr</td>
<td>Width-height ratio clipping threshold. The default value is 0.016.</td>
<td>FLOAT</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- The shapes of rois and deltas must be exactly (N, 4), without broadcasting.
- The dtypes of rois and deltas must be the same.
- The lengths of means and stds must be 4, and no element of stds can be 0.
- The length of max_shape must be 2.
- wh_ratio_clip must be greater than 0. It does not participate in the core decoding formula currently, and is used only for input parameter validation.
- Only the ND data format is supported for the input.

## Invocation Description

<table style="table-layout: fixed; width: 1000px">
<colgroup>
<col style="width: 180px">
<col style="width: 200px">
<col style="width: 620px">
</colgroup>
<thead>
<tr>
<th>Invocation Method</th>
<th>Sample Code</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>GE graph mode</td>
<td>-</td>
<td><a href="examples/arch35/test_geir_bounding_box_decode.cpp">test_geir_bounding_box_decode.cpp</a></td>
</tr>
</tbody>
</table>
