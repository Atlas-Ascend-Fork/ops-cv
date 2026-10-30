# BoundingBoxEncode

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×   |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: calculates the encoding offsets between the anchor box and the ground truth box, and generates regression targets for object detection.

- Calculation formula:

  First, convert the input coordinates $(x_1, y_1, x_2, y_2)$ to the center point plus width and height format:

  $$
  cx = (x_1 + x_2) / 2, \quad cy = (y_1 + y_2) / 2, \quad w = x_2 - x_1 + 1, \quad h = y_2 - y_1 + 1
  $$

  Then calculate the encoding offsets:

  $$
  dx = \frac{g_{cx} - p_{cx}}{p_w}, \quad dy = \frac{g_{cy} - p_{cy}}{p_h}, \quad dw = \ln\left(\frac{g_w}{p_w}\right), \quad dh = \ln\left(\frac{g_h}{p_h}\right)
  $$

  Finally, perform the mean normalization:

  $$
  \delta_i = \frac{raw_i - means_i}{stds_i}, \quad i \in \{0,1,2,3\}
  $$

  In the formula, $p$ is the corresponding value of anchor_box, and $g$ is the corresponding value of ground_truth_box.

## Parameter Description

<table style="table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 350px">
  <col style="width: 200px">
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
      <td>anchor_box</td>
      <td>Input</td>
      <td>Anchor box coordinate tensor. The coordinate format is (x1, y1, x2, y2). The data type must be the same as that of ground_truth_box.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ground_truth_box</td>
      <td>Input</td>
      <td>Ground truth box coordinate tensor. The coordinate format is (x1, y1, x2, y2). The data type and shape must be the same as those of anchor_box.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>means</td>
      <td>Attr</td>
      <td>Mean offsets for encoding. The length is 4. The default value is [0.0, 0.0, 0.0, 0.0].</td>
      <td>ListFloat</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stds</td>
      <td>Attr</td>
      <td>Standard deviation scaling values for encoding. The length is 4, and no element can be 0. The default value is [1.0, 1.0, 1.0, 1.0].</td>
      <td>ListFloat</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deltas</td>
      <td>Output</td>
      <td>Output tensor of encoding offsets. The data type is the same as that of anchor_box, and the shape is the same as that of anchor_box.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The data types of anchor_box and ground_truth_box must be the same. float16 and float32 are supported.
- The shapes of anchor_box and ground_truth_box must be exactly (N, 4).
- The lengths of means and stds must be 4, and no element of stds can be 0.
- The coordinate format is the standard (x1, y1, x2, y2) format, that is, the top-left and bottom-right coordinates.
- The width and height calculations in the formula include a +1 offset (w = x2 - x1 + 1, h = y2 - y1 + 1). This ensures that the width and height are at least 1 and prevents division by zero.
- Empty tensors are supported (an empty output is returned when N=0).

## Invocation Description

| Invocation Method   | Sample Code                                                                        | Description                                         |
| ---------------- |-----------------------------------------------------------------------------| --------------------------------------------------- |
| Graph mode | [test_geir_bounding_box_encode](./examples/arch35/test_geir_bounding_box_encode.cpp) | Call the BoundingBoxEncode operator by constructing a graph in the [operator IR](./op_graph/bounding_box_encode_proto.h) mode. |
