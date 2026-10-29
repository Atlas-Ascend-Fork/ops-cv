# CheckValid

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

The operator checks whether candidate boxes are within the valid image range. Given the candidate boxes `bbox_tensor=(x0, y0, x1, y1)` and the image metadata `img_metas=(H, W, r)`, the operator determines whether each box satisfies:

$$
\text{valid} = (x_0 \geq 0) \wedge (y_0 \geq 0) \wedge (x_1 \leq W \cdot r - 1) \wedge (y_1 \leq H \cdot r - 1)
$$

The output `valid_tensor` is of the int8 type. 1 indicates valid, and 0 indicates invalid.

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
<td>bbox_tensor</td>
<td>Input</td>
<td>Candidate box coordinates (x0,y0,x1,y1). The shape is (N,4).</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>img_metas</td>
<td>Input</td>
<td>Image metadata (H,W,r). It contains at least three elements, and the first three are valid.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>valid_tensor</td>
<td>Output</td>
<td>Validity mask. 1 indicates valid and 0 indicates invalid. The shape is (N,1).</td>
<td>INT8</td>
<td>ND</td>
</tr>
</tbody>
</table>

## Constraints

- The shape of bbox_tensor must be (N, 4), and the last dimension is fixed to 4.
- The dtypes of bbox_tensor and img_metas must be the same. float16/float32 is supported.
- The number of elements in img_metas must be ≥3 (the first three elements are H, W, and r).
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
<td><a href="examples/arch35/test_geir_check_valid.cpp">test_geir_check_valid.cpp</a></td>
<td>Call the CheckValid operator by constructing a graph in the <a href="op_graph/check_valid_proto.h">operator IR</a> mode</td>
</tr>
</tbody>
</table>
