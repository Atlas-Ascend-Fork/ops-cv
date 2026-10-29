# RoiAlignRotated

English | [简体中文](./README.md)

## Product Support

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×   |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

Note: The data format is ND. The dimension order varies by chip: On <term>Ascend 950PR&950DT products</term>, x uses the NCHW dimension order (N,C,H,W). On other supported chips, x uses the NHWC dimension order (N,H,W,C). For details, see "Constraints".

## Function Description

- Operator function: Rotated RoI align pooling. The operator divides each rotated rectangular RoI into bins by pooled_h×pooled_w, maps the sampling points in each bin to the feature map coordinate system through the rotation matrix, performs bilinear interpolation, takes the average, and outputs fixed-size RoI features.

- Computation formula:

  For the r-th RoI, rois[r]=(batch_idx, roi_cx, roi_cy, roi_w, roi_h, θ), and θ is negative when clockwise=true. The operator performs preprocessing first (scaling before offsetting, without rounding). The width and height are also multiplied by spatial_scale first, and the scaled values are denoted as w and h. The subsequent bin sizes and sampling coordinates are all based on w and h:

  $$
  center_x = roi_cx \times spatial\_scale - offset,\quad center_y = roi_cy \times spatial\_scale - offset
  $$

  $$
  w = roi_w \times spatial\_scale,\quad h = roi_h \times spatial\_scale
  $$

  $$
  offset = \begin{cases} 0.5, & aligned=true \\ 0, & aligned=false \end{cases}
  $$

  Bin size and sampling count (the same applies to grid_w):

  $$
  bin_h = \frac{h}{pooled_h},\quad grid_h = \begin{cases} sampling\_ratio, & sampling\_ratio > 0 \\ \lceil bin_h \rceil, & sampling\_ratio = 0 \end{cases},\quad count = \max(grid_h \times grid_w,\ 1)
  $$

  Coordinates of the sampling points in a bin in the local RoI coordinate system (iy∈[0,grid_h), ix∈[0,grid_w), and the same applies to xx):

  $$
  yy = -\frac{h}{2} + ph \cdot bin_h + \frac{(iy+0.5) \times bin_h}{grid_h}
  $$

  Rotation to the feature map coordinate system:

  $$
  \begin{bmatrix} sx \\ sy \end{bmatrix} =
  \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}
  \begin{bmatrix} xx \\ yy \end{bmatrix} +
  \begin{bmatrix} center_x \\ center_y \end{bmatrix}
  $$

  The operator performs bilinear interpolation on the feature map (out-of-boundary sampling points contribute 0) and takes the average within each bin:

  $$
  y[r,c,ph,pw] = \frac{1}{count} \sum_{iy=0}^{grid_h-1} \sum_{ix=0}^{grid_w-1} f(sx,\ sy)
  $$

  Let $x_l=\lfloor sx \rfloor$, $y_l=\lfloor sy \rfloor$, $lx=sx-x_l$, and $ly=sy-y_l$:

  $$
  f(sx,\ sy) = (1-ly)(1-lx)\,x[n,c,y_l,x_l] + (1-ly)\,lx\,x[n,c,y_l,x_l+1] + ly\,(1-lx)\,x[n,c,y_l+1,x_l] + ly\,lx\,x[n,c,y_l+1,x_l+1]
  $$

  Supplementary notes:
  - Out-of-boundary (sx<-1, sx>W, sy<-1, or sy>H) sampling points contribute 0. Negative coordinates are clamped to 0. When $x_l \ge W-1$ or $y_l \ge H-1$, the computation degrades to single-point sampling.
  - When aligned=false, w and h are first floored to 1 after scaling, and then the operator computes the bin sizes and sampling coordinates.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 150px">
  <col style="width: 130px">
  <col style="width: 480px">
  <col style="width: 130px">
  <col style="width: 115px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input feature map, 4D, with data format ND. On <term>Ascend 950PR&950DT products</term>, x uses the NCHW dimension order, with shape (N,C,H,W). On other supported chips, x uses the NHWC dimension order, with shape (N,H,W,C).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>Input</td>
      <td>Rotated RoI list, 2D. Each RoI is (batch_idx, center_x, center_y, w, h, angle). The coordinates are in the original image coordinate system, and the unit of angle is radian. On <term>Ascend 950PR&950DT products</term>, the row-major shape is (R,6). On other supported chips, the transposed shape is (6,R).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output feature map, with data format ND, the same as x. On <term>Ascend 950PR&950DT products</term>, the shape is (R,C,pooled_h,pooled_w). On other supported chips, the shape is (R,pooled_h,pooled_w,C).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pooled_h</td>
      <td>Attr</td>
      <td>Pooling size of the output in the H dimension. The value must be greater than 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_w</td>
      <td>Attr</td>
      <td>Pooling size of the output in the W dimension. The value must be greater than 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>Attr</td>
      <td>Scaling factor from RoI coordinates (original image coordinates) to input feature map coordinates. The value must be greater than 0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sampling_ratio</td>
      <td>Optional Attr</td>
      <td>Number of sampling points per dimension in each bin. 0 indicates dense sampling by ceil(bin_size). The value must be greater than or equal to 0. Default value: 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>aligned</td>
      <td>Optional Attr</td>
      <td>When the value is true, the RoI center is offset by 0.5 pixel. When the value is false, the offset is 0 and the lower limit of roi_w/roi_h is 1. Default value: true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clockwise</td>
      <td>Optional Attr</td>
      <td>When the value is true, the RoI angle is interpreted in the clockwise direction (θ is negative). Default value: false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The data formats of x, rois, and y of this operator are ND, and the dimension order is determined by the chip:
  - <term>Ascend 950PR&950DT products</term>: x uses the NCHW dimension order (N,C,H,W), rois is (R,6), and y is (R,C,pooled_h,pooled_w).
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Kirin X90 processor series products</term>, and <term>Kirin 9030 processor series products</term>: x uses the NHWC dimension order (N,H,W,C), rois is (6,R), and y is (R,pooled_h,pooled_w,C).
- The shape of the input x must be 4D.
- The shape of the input rois must be 2D. On <term>Ascend 950PR&950DT products</term>, when R=0, the operator outputs an empty tensor (0,C,pooled_h,pooled_w). Other supported chips require R≥1.
- RoI value constraints (the operator does not check them, and the caller must ensure them): batch_idx∈[0,N), where N is the size of dimension 0 of the input x; angle∈[0,π), in radians; and no element can be NaN/Inf.
- Attribute constraints: pooled_h>0 and pooled_w>0 (checked by the operator); spatial_scale>0 and sampling_ratio>=0 (not checked by the operator, and the caller must ensure them).
- Only the FLOAT32 data type is supported, and the data types of x, rois, and y must be the same. Only the continuous layout is supported.

## Invocation

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_roi_align_rotated](examples/arch35/test_geir_roi_align_rotated.cpp) | Invokes the RoiAlignRotated operator through the [operator IR](./op_graph/roi_align_rotated_proto.h) API (<term>Ascend 950PR&950DT products</term>: ND, x uses the NCHW dimension order, and the sample includes negative test cases for the format gate). |
| Graph mode | [test_geir_roi_align_rotated](examples/test_geir_roi_align_rotated.cpp) | Invokes the RoiAlignRotated operator through the [operator IR](./op_graph/roi_align_rotated_proto.h) API (<term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Kirin X90 processor series products</term>, and <term>Kirin 9030 processor series products</term>: ND, x uses the NHWC dimension order, and rois is transposed to (6,R)). |
