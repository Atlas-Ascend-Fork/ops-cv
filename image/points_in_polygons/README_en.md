# PointsInPolygons

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

PointsInPolygons is an operator that determines the membership between points and quadrilaterals. Given N 2D points and M quadrilaterals (each quadrilateral has 4 vertices expanded in order into 8 scalars), the operator determines whether each point falls inside each quadrilateral and outputs an N×M binary floating-point matrix, where `1.0` indicates inside and `0.0` indicates outside.

  $$
  x\_intersect_k = x_k + \frac{(py - y_k) \times (x_{k-1} - x_k)}{y_{k-1} - y_k}
  $$

  $$
  c = \sum_{k=0}^{3} ((y_k < py \land y_{k-1} \ge py) \lor (y_k \ge py \land y_{k-1} < py)) \land (x\_intersect_k > px)
  $$

  $$
  output_{i,j} = \begin{cases} 1.0, & c \text{ is odd} \\ 0.0, & c \text{ is even} \end{cases}
  $$

In the formula, px and py are the coordinates of point i, and x_k and y_k are the coordinates of the k-th vertex of polygon j.

Vertices are visited in the order k=0,1,2,3, with k-1 taken modulo 4. An exact vertex hit, or an exact match between px and a crossing edge's intersection x coordinate, ends traversal for that pair. The output uses the parity accumulated before the hit. A boundary point can therefore produce `0.0` or `1.0`, depending on vertex ordering and traversal order.

Intersections use float32 multiplication before division. Only crossing edges participate; horizontal edges are skipped. Comparisons use exact equality without an epsilon, so near-boundary results can differ from a float64 geometric reference. Division is correctly rounded in float32, including the subnormal path.

## Parameter Description

- **Parameter description**:

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
  <td>points</td>
  <td>Input</td>
  <td>Represents the set of 2D points to be determined, each row (x,y) is a 2D point coordinate, corresponding to the source of px and py in the formula.</td>
  <td>FLOAT</td>
  <td>ND</td>
  </tr>
  <tr>
  <td>polygons</td>
  <td>Input</td>
  <td>Represents the set of quadrilaterals to be determined, the shape is (8,M): the 0th dimension is fixed to 8 and stores the coordinate components of the 4 vertices in the (x0,y0,x1,y1,x2,y2,x3,y3) order; the 1st dimension is the M polygons, corresponding to the source of x_k and y_k in the formula.</td>
  <td>FLOAT</td>
  <td>ND</td>
  </tr>
  <tr>
  <td>output</td>
  <td>Output</td>
  <td>Represents the point-quadrilateral membership determination result, an element is 1.0 (inside) or 0.0 (outside), corresponding to output in the formula.</td>
  <td>FLOAT</td>
  <td>ND</td>
  </tr>
  </tbody>
  </table>

## Constraints

- Inputs and output support FLOAT (float32) in ND format only.
- points and polygons must both be rank 2. All actual input dimensions must be nonnegative.
- output.shape=(N,M), where N=points.shape[0] and M=polygons.shape[1]. A dynamic declaration containing -1 produces (-1,-1); tiling validates the actual shapes.
- When both N and M are nonzero, points.shape[1] must be 2 and polygons.shape[0] must be 8. When N=0 or M=0, return an empty output and skip these two dimension checks.
- For membership computation, polygons must use contiguous (8,M) layout, with one quadrilateral per column. Transpose row-wise (M,8) data and make it contiguous before invocation.

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_points_in_polygons](examples/arch35/test_geir_points_in_polygons.cpp) | Call the PointsInPolygons operator by building a graph through [operator IR](op_graph/points_in_polygons_proto.h). |
