# YoloxBoundingBoxDecode

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

- Operator function: Decodes the normalized offsets bboxes predicted by the model together with the prior boxes priors into the final bounding box coordinates (top-left and bottom-right corners) according to the YOLOX decoding formulas.

- Computation formula:

$$
xys = bboxes[..., 0:2] \times priors[:, 2:4] + priors[:, 0:2]
$$

$$
whs = \exp(bboxes[..., 2:4]) \times priors[:, 2:4] \times 0.5
$$

$$
decoded\_bboxes[..., 0:2] = xys - whs
$$

$$
decoded\_bboxes[..., 2:4] = xys + whs
$$

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
      <td>priors</td>
      <td>Input</td>
      <td>Prior box information, with shape (N, 4). Each row is [center_x, center_y, width, height].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bboxes</td>
      <td>Input</td>
      <td>Offsets predicted by the model, with shape (B, N, 4). Each row is [dx, dy, dw, dh].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>decoded_bboxes</td>
      <td>Output</td>
      <td>Decoded bounding boxes, with shape (B, N, 4). Each row is [top-left x, top-left y, bottom-right x, bottom-right y].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The input priors supports only 2D, with shape (N, 4), where N is the number of prior boxes on the feature map.
- The input bboxes supports only 3D, with shape (B, N, 4), where N must be the same as the N of priors.
- The shape of the output decoded_bboxes is the same as that of bboxes, that is, (B, N, 4).
- The input dtype supports only FLOAT16 and FLOAT, and the dtypes of the two inputs must be the same.
- Overflow/underflow of the exp operation propagates according to the IEEE 754 rules, without clamp processing.
- Empty tensor scenarios (B=0 or N=0) are supported, and the output is empty in this case.

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
    <td><a href="./examples/test_geir_yolox_bounding_box_decode.cpp">test_geir_yolox_bounding_box_decode</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
