# DecodeBboxV2

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

The operator decodes the offsets `boxes` predicted by the regression branch of object detection together with the anchor boxes `anchors` into absolute boxes `y` in the image coordinate space. The operator supports two layouts, `(N,4)` and `(4,N)`, which are determined by the `reversed_box` attribute.

Assume that the anchor boxes `anchors=(ymin, xmin, ymax, xmax)`, the offsets `boxes=(ty, tx, th, tw)`, and `scales=(sy, sx, sh, sw)`. The decoding steps are as follows:

$$
ah = ymax - ymin,\quad aw = xmax - xmin
$$

$$
tys = ty / sy,\quad ths = th / sh
$$

$$
h = \exp(\min(ths, decode\_clip)) \times ah \quad (decode\_clip > 0)
$$

$$
cy = tys \times ah + ymin + ah \times 0.5
$$

$$
ymin_{out} = cy - h \times 0.5,\quad ymax_{out} = cy + h \times 0.5
$$

The x dimension follows the same rule. For fp16 input, the intermediate `exp` and multiply-accumulate computations are performed in the fp32 domain and then cast back to fp16.

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
<td>boxes</td>
<td>Input</td>
<td>Regression offsets (ty,tx,th,tw). The shape is determined by reversed_box: (N,4) when the value is false, and (4,N) when the value is true.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>anchors</td>
<td>Input</td>
<td>Anchor box coordinates (ymin,xmin,ymax,xmax). The shape and dtype are the same as those of boxes, both (N,4) or (4,N).</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>Output</td>
<td>Decoded absolute boxes (ymin,xmin,ymax,xmax). The shape and dtype are the same as those of boxes, both (N,4) or (4,N).</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>scales</td>
<td>Optional attr</td>
<td>Scaling factors. The length is 4, and all elements must be not 0. The default value is [1.0, 1.0, 1.0, 1.0].</td>
<td>FLOAT</td>
<td>-</td>
</tr>
<tr>
<td>decode_clip</td>
<td>Optional attr</td>
<td>Clipping threshold for exp. 0.0 indicates no clipping. The default value is 0.0.</td>
<td>FLOAT</td>
<td>-</td>
</tr>
<tr>
<td>reversed_box</td>
<td>Optional attr</td>
<td>Layout flag. false indicates (N,4), true indicates (4,N), and the default value is false.</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- The shapes and dtypes of boxes, anchors, and y must be exactly the same.
- The length of scales must be 4, and all elements must be not 0.
- The value range of decode_clip is [0.0, 10.0].
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
<td><a href="examples/arch35/test_geir_decode_bbox_v2.cpp">test_geir_decode_bbox_v2.cpp</a></td>
<td>Call the DecodeBboxV2 operator by constructing a graph in the <a href="op_graph/decode_bbox_v2_proto.h">operator IR</a> mode</td>
</tr>
</tbody>
</table>
