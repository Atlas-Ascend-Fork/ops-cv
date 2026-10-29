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

PointsInPolygons is an operator that determines the membership between points and quadrilaterals. Given N 2D points and M quadrilaterals (each quadrilateral has 4 vertices expanded in order into 8 scalars), the operator determines whether each point falls inside each quadrilateral (a point on an edge or a vertex is regarded as outside) and outputs an N×M binary floating-point matrix, where `1.0` indicates inside and `0.0` indicates outside.

  $$
  x\_intersect_k = x_k + \frac{py - y_k}{y_{k+1} - y_k} \times (x_{k+1} - x_k)
  $$

  $$
  c = \sum_{k=0}^{3} ((y_k > py) \neq (y_{k+1} > py)) \land (x\_intersect_k > px)
  $$

  $$
  output_{i,j} = \begin{cases} 1.0, & c \text{ is odd} \\ 0.0, & c \text{ is even} \end{cases}
  $$

In the formula, px and py are the coordinates of point i, and x_k and y_k are the coordinates of the k-th vertex of polygon j.

Two supplementary conventions apply to the formula above:

- Boundary priority: if a point falls on any edge or any vertex of a quadrilateral, the point is on the boundary, the output is directly set to `0.0`, and the parity of the intersection count is no longer used for the determination.
- Crossing condition: the operator computes `x_intersect_k` and counts an intersection only for an edge whose two y coordinates cross py (that is, an edge satisfying `(y_k > py) ≠ (y_{k+1} > py)`). A horizontal edge (`y_{k+1} = y_k`) does not satisfy the crossing condition, so the operator skips the intersection calculation and division by zero never occurs.

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

- The data types of points and polygons must both be FLOAT, and other data types are not supported.
- The data formats of points, polygons, and output must all be ND.
- The ranks of points and polygons must both be 2; for non-empty tensors, the 1st dimension of points must be fixed to 2, and the 0th dimension of polygons must be fixed to 8.
- The shape of output must be (N,M), where N = points.shape[0] and M = polygons.shape[1]; N and M are independent and have no broadcast relationship.
- Empty tensors are supported: when N = 0 or M = 0, output is an empty tensor (the number of elements is 0), the interface does not report an error and returns directly.

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_points_in_polygons](examples/arch35/test_geir_points_in_polygons.cpp) | Call the PointsInPolygons operator by building a graph through [operator IR](op_graph/points_in_polygons_proto.h). |
