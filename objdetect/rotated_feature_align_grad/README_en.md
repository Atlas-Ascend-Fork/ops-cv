# RotatedFeatureAlignGrad

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

## Function Description

- Operator function: This is the backpropagation operator of RotatedFeatureAlign (rotated box feature alignment). Based on the gradient dy of the output feature and the rotated box information bboxes at each pixel position, the operator accumulates the gradient contributions of the sampling point bilinear interpolation back to the corresponding positions of the input feature, obtaining the gradient dx of the input feature.
- Computation formula:

  $$
  dx_{n,y,x,c} = dy_{n,y,x,c} + \sum_{h=0}^{H-1}\sum_{w=0}^{W-1}\sum_{p \in P} dy_{n,h,w,c} \cdot w_p(n,h,w;y,x)
  $$

  In the formula, $P$ is the set of sampling points of the rotated box at pixel $(n,h,w)$: When `points=1`, the set contains only the box center point; when `points=5`, the set contains the box center point and four rotated corner points. $w_p$ is the weight of the bilinear interpolation of the sampling point $p$ falling on pixel $(y,x)$, and the weight is 0 when the sampling point is out of bounds. In bboxes, y/x/w/h are multiplied by `spatial_scale` before participating in the sampling point computation, and angle is not scaled.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
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
      <td>dy</td>
      <td>Input</td>
      <td>Gradient of the output feature. The shape must be a 4D tensor [N, H, W, C].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bboxes</td>
      <td>Input</td>
      <td>Rotated box information at each pixel position. The shape must be a 4D tensor [N, H, W, 5]. Each box consists of [y, x, w, h, angle], which are the box center coordinates, width, height, and rotation angle respectively.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>Attr</td>
      <td>Scaling ratio between the feature map and the original image. The y/x/w/h coordinates in bboxes are multiplied by this value before participating in the sampling point computation.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>points</td>
      <td>Optional Attr</td>
      <td>Number of sampling points for each rotated box. 1 indicates sampling the box center point, and 5 indicates sampling the box center point and four rotated corner points. Default value: 1.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>Output</td>
      <td>Gradient of the input feature.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The shape of dy must be a 4D tensor [N, H, W, C], and the shape of bboxes must be a 4D tensor [N, H, W, 5]. In addition, the N, H, and W dimensions of the two must be the same.
- The value of the points attribute supports only 1 or 5.

## Invocation

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_rotated_feature_align_grad](examples/test_geir_rotated_feature_align_grad.cpp) | Invokes the RotatedFeatureAlignGrad operator through the [operator IR](./op_graph/rotated_feature_align_grad_proto.h) API. |
