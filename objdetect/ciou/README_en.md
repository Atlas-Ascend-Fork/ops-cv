# CIoU

English | [简体中文](./README.md)

## Product Support

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: serves as a loss function for bounding box regression. Based on IoU, it also considers the center point distance, width-height ratio, and overlap area, to more comprehensively measure the difference between predicted boxes and ground truth boxes.
- Calculation formula:

$$
CIoU = IoU - \frac{\rho^2(b^p, b^g)}{c^2} - \alpha v \\
v = \frac{4}{\pi^2}(arctan(\frac{w^g}{h^g}) - arctan(\frac{w^p}{h^p}))^2 \\
\alpha = \frac{v}{1 - IoU + v} \\
$$

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
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gtboxes</td>
      <td>Input</td>
      <td>Ground truth rectangular boxes.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>trans</td>
      <td>Attr</td>
      <td>Specifies the format of the rectangular boxes.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_cross</td>
      <td>Attr</td>
      <td>Specifies whether to perform the cross operation between bBoxes and gtBoxes.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Attr</td>
      <td>Selects the calculation mode "iou" or "iof".</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>atan_sub_flag</td>
      <td>Attr</td>
      <td>Specifies whether to output atan_sub.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>overlap</td>
      <td>Output</td>
      <td>The intersection over union (IoU) or intersection over foreground (IoF) calculated from the two inputs.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>atan_sub</td>
      <td>Output</td>
      <td>Difference between the two arctan values in the calculation process.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- If the input format is [x0, y0, x1, y1], (x0, y0) and (x1, y1) represent the top-left and bottom-right corners of the rectangular box respectively. x1 > x0 and y1 > y0 must be satisfied.
- M and N must be the same.
- `is_cross` currently supports only `false`.
- `atan_sub_flag` currently supports only `true`.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_ciou](examples/test_aclnn_ciou.cpp) | Call the CIoU operator through the [aclnnCIoU](docs/aclnnCIoU_en.md) interface. |
