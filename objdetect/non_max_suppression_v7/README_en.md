# NonMaxSuppressionV7

English | [简体中文](./README.md)

## Product Support

|Product|Support|
|:---|:---:|
|<term>Ascend 950PR&950DT products</term>|√|
|<term>Atlas A3 products</term>|√|
|<term>Atlas A2 products</term>|√|
|<term>Atlas 200I/500 A2 inference products</term>|×|
|<term>Atlas inference products</term>|×|
|<term>Atlas training products</term>|×|
|<term>Kirin X90 processor series products</term>|×|
|<term>Kirin 9030 processor series products</term>|×|

## Function Description

- Operator function: Performs greedy non-maximum suppression on candidate boxes by batch and class, and outputs the index triples (batch_index, class_index, box_index) of the selected boxes.

- Computation formula:

  $$
  IoU = \frac {Area_{inter}} {Area_{current} + Area_{next} - Area_{inter}}
  $$

  The operator selects candidate boxes in descending order of score. It processes only the candidate boxes whose scores are greater than score_threshold. When the IoU between a candidate box and a selected box is greater than iou_threshold, the operator suppresses the candidate box.

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
      <td>boxes</td>
      <td>Input</td>
      <td>Candidate boxes, with shape (B, N, 4). When center_point_box is 0, the coordinate format is (y1, x1, y2, x2); when center_point_box is 1, the coordinate format is (x_center, y_center, width, height).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scores</td>
      <td>Input</td>
      <td>Scores of the candidate boxes, with shape (B, C, N).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_output_size</td>
      <td>Optional Input</td>
      <td>Maximum number of candidate boxes to output for each batch and class. The input is a scalar or a tensor with shape (1,). When the value is less than or equal to 0, the operator does not output the class.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>iou_threshold</td>
      <td>Optional Input</td>
      <td>IoU threshold for determining whether a candidate box needs to be suppressed. The input is a scalar or a tensor with shape (1,). Default value: 0.0.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>score_threshold</td>
      <td>Optional Input</td>
      <td>Score threshold for filtering candidate boxes. The input is a scalar or a tensor with shape (1,). Default value: 0.0.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index_id</td>
      <td>Optional Input</td>
      <td>Index mapping of the candidate boxes, with shape (B, C, N, 3) or (B, C, N, 4).</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>center_point_box</td>
      <td>Attr</td>
      <td>Coordinate format of the candidate boxes. The value is 0 or 1. Default value: 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_boxes_size</td>
      <td>Attr</td>
      <td>Size of the first dimension of the output selected_indices. The value range is [0, INT32_MAX]. Default value: 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selected_indices</td>
      <td>Output</td>
      <td>Indices of the selected candidate boxes, with shape (max_boxes_size, 3). The insufficient part is filled with (-1, -1, -1).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- The data types of boxes and scores can be selected independently.

## Constraints

- boxes and scores must have static positive shapes, and boxes.shape[0] must equal scores.shape[0], and boxes.shape[1] must equal scores.shape[2].

- center_point_box supports only 0 and 1, and the value range of max_boxes_size is [0, INT32_MAX].

- max_output_size, iou_threshold, and score_threshold are scalars or tensors with shape (1,). index_id is a tensor with shape (B, C, N, 3) or (B, C, N, 4).

- The GE IR dynamic rank scenario (input shape declared as `[-2]`) is currently not supported.

## Invocation

|Invocation Method|Sample|Description|
|:---|:---|:---|
|Graph-mode invocation|[test_geir_non_max_suppression_v7](./examples/arch35/test_geir_non_max_suppression_v7.cpp)|Invokes the NonMaxSuppressionV7 operator by constructing a graph through the [operator IR](./op_graph/non_max_suppression_v7_proto.h).|
