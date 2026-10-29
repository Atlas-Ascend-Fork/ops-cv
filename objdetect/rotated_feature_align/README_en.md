# RotatedFeatureAlign

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: Performs bilinear interpolation on the input feature map x at the sampling points specified by the rotated box parameter bboxes, and accumulates the interpolation results back to the original feature map. This implements feature alignment between proposals and the feature map in rotated object detection networks (such as RotatedRetinaNet and RotatedFasterRCNN).

- Computation formula:

$$
y[n,c,h,w] = x[n,c,h,w] + \sum_{i=0}^{points-1} \mathrm{bilinear}(x[n,:,:,:], py_i, px_i)
$$

In the formula, the five planes of bboxes are in the order of (y_ctr,x_ctr,w,h,angle), and the values of the planes (y_ctr,x_ctr,w,h) are all multiplied by spatial_scale, while angle is not multiplied. When points=1, only the center point is sampled. When points=5, the center point and four rotated corner points are sampled. In bilinear interpolation, when a sampling point is out of bounds (y<-1.0 or y>H or x<-1.0 or x>W, strict inequalities), the contribution of that sampling point is 0.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>Feature map, with shape (N,C,H,W). It is x in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bboxes</td>
      <td>Input</td>
      <td>Rotated box parameters, with shape (N,5,H,W). The five planes are (y_ctr,x_ctr,w,h,angle) in order. It is bboxes in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Feature alignment result, with the same shape as x. It is y in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>Attr</td>
      <td>Scale factor from the feature map to the original image. The typical value range is (0,1], for example 0.125.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>points</td>
      <td>Attr</td>
      <td>Number of sampling points. 1 indicates center point sampling, and 5 indicates sampling of the center point plus the four corner points. Default value: 1.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- x must be 4D (N,C,H,W).
- bboxes must be 4D (N,5,H,W), and dim1 is fixed to 5.
- x.dim0 must equal bboxes.dim0 (the same N), x.dim2 must equal bboxes.dim2 (the same H), and x.dim3 must equal bboxes.dim3 (the same W).
- The data types of x, bboxes, and y support only float32.
- spatial_scale is the scale factor from the feature map to the original image, and the typical value range is (0,1], for example 0.125.
- points only allows 1 or 5, and the default value is 1.
- In bboxes, w and h should be non-negative values (the width and height of the rotated box), and angle is a value in radians.
- When the sampling coordinates are in the interval [-1,0], the operator treats them as 0. When y is greater than or equal to H-1, both y_low and y_high are set to H-1 (similarly, both are set to W-1 in the x direction). When a sampling point is out of bounds (y<-1.0 or y>H or x<-1.0 or x>W, strict inequalities), the contribution of that sampling point is 0.

## Invocation

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/arch35/test_geir_rotated_feature_align.cpp">test_geir_rotated_feature_align</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
