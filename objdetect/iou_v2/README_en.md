# IouV2

English | [简体中文](./README.md)

## Product Support

|Product             |  Support  |
|:-------------------------|:----------:|
|  Ascend 950PR&950DT products   |     √    |
|  Atlas A3 products   |     √    |
|  Atlas A2 products     |     √    |
|  Atlas 200I/500 A2 inference products    |     ×    |
|  Atlas inference products    |     √   |
|  Atlas training products    |     ×    |
|  Kirin X90 processor series products | √ |
|  Kirin 9030 processor series products | √ |

## Function Description

- Operator function: Computes the ratio of the overlap area of two rectangles to the total area of the two rectangles. Assume that the top-left corner of the predicted box is (X1, Y1) and the bottom-right corner is (X2, Y2), and that the top-left corner of the ground-truth box is (X3, Y3) and the bottom-right corner is (X4, Y4).

- Computation formula:

  $$
  IOU = \frac {Area_3} {Area_1 + Area_2 - Area_3} \\
  IOF = \frac {Area_3} {Area_2}
  $$

  In the formula, Area_1 is the area of bBox, Area_2 is the area of gtBox, and Area_3 is the area of the overlapping part of the two. For the definitions of x and y, see Parameter Description.

  $$
  Area_1 = (X_2 - X_1)(Y_2 - Y_1) \\
  Area_2 = (X_4 - X_3)(Y_4 - Y_3) \\
  Area_3 = max( min(X_2, X_4) - max(X_1, X_3), 0 ) * max( min(Y_2, Y_4) - max(Y_1, Y_3), 0 )
  $$

  IOF is defined as the intersection-over-foreground ratio. The computation formula is:

  $$
  IOF=\frac {Area_3} {Foreground Area}
  $$

  For the operator in this repository, the foreground area is Area2. For the competitor mmcv, the foreground area is Area1.

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
      <td>bboxes</td>
      <td>Input</td>
      <td>Predicted rectangular boxes.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gtboxes</td>
      <td>Input</td>
      <td>Ground-truth rectangular boxes.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attr</td>
      <td>Prevents division by zero. When computing areas, the operator adds eps to both the length and the width.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>aligned</td>
      <td>Attr</td>
      <td>Identifies whether the two inputs have the same shape.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Attr</td>
      <td>Selects the computation mode, "iou" or "iof".</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>overlap</td>
      <td>Output</td>
      <td>The IoU or IoF value computed from the two inputs.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- Input shape limit: The input is a 2D tensor with shape (N, 4). The four values (X1, Y1, X2, Y2) in the second dimension must satisfy X1 < X2 and Y1 < Y2.
- `bboxes` and `gtboxes` must have the same data type, and the output `overlap` has the same data type as `bboxes`.

## Invocation

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_iou_v2](examples/test_aclnn_iou_v2.cpp) | Invokes the IouV2 operator through the [aclnnIou](docs/aclnnIou_en.md) API. |
