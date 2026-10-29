# RotatedOverlaps

English | [简体中文](./README.md)

## Product Support

| Product | Support |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor series products</term> | × |
| <term>Kirin 9030 processor series products</term> | × |

## Function Description

- Operator function: Computes the overlap area matrix between two groups of 2D rotated boxes and outputs the overlap area, without computing the IoU.
- Computation principle: The operator converts each rotated box into four vertices, collects the points where the vertices of one group of boxes are located inside the other group of boxes and the edge intersection points, and computes the intersection polygon area in the vertex order.
- Coordinate format: When `trans=false`, the rectangular box format is `[x, y, w, h, theta]`. When `trans=true`, the rectangular box format is `[x1, y1, x2, y2, theta]`.
- The unit of `theta` is degree, and the output is the overlap area instead of the IoU.

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
    <td>boxes</td>
    <td>Input</td>
    <td>First group of 2D rotated boxes, with shape <code>[B, 5, N]</code>. The second dimension is 5, indicating [x, y, w, h, theta] or [x1, y1, x2, y2, theta].</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>query_boxes</td>
    <td>Input</td>
    <td>Second group of 2D rotated boxes, with shape <code>[B, 5, K]</code>. The second dimension is 5, indicating [x, y, w, h, theta] or [x1, y1, x2, y2, theta].</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>overlaps</td>
    <td>Output</td>
    <td>Overlap area matrix of the two groups of rotated boxes, with shape <code>[B, N, K]</code>.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>trans</td>
    <td>Attr</td>
    <td>Whether to use the [x1, y1, x2, y2, theta] coordinate format. Default value: false.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- The batch dimension `B` of the two inputs must be the same.
- The first implementation requires `B`, `N`, and `K` to be positive, and `query_boxes.shape[2]` (K) must not exceed 2000.
- When the input boxes contain non-finite coordinates or degenerate rectangles, the pairing outputs involving such rectangles are 0.
- GE IR dynamic shapes are supported, and runtime Tiling uses the instantiated concrete shape.

## Invocation

| Invocation Method | Sample | Description |
| ------------ | ------------ | ------------ |
| GE graph mode | [test_geir_rotated_overlaps](./examples/test_geir_rotated_overlaps.cpp) | Invokes the RotatedOverlaps operator by constructing a graph through the [operator IR](./op_graph/rotated_overlaps_proto.h). |
