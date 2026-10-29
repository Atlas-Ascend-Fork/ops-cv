# SortedNMS

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
|  <term>Kirin X90 processor series products</term> | × |
|  <term>Kirin 9030 processor series products</term> | × |

## Function Description

- Operator function: On the candidate box sequence already sorted in descending order of score, the operator greedily selects non-suppressed boxes according to the IoU threshold, and outputs the indices of the selected boxes in the original boxes.

- Computation formula:

  $$
  IoU = \frac {Area_{inter}} {Area_{current} + Area_{next} - Area_{inter}}
  $$

  In the formula, Area_current is the area of the currently selected box, Area_next is the area of the candidate box, Area_inter is the overlap area of the two boxes, and offset is the offset for coordinate computation.

  $$
  Area_i = max(X_{2i} - X_{1i} + offset, 0) * max(Y_{2i} - Y_{1i} + offset, 0) \\
  Area_{inter} = max(min(X_{2c}, X_{2n}) - max(X_{1c}, X_{1n}) + offset, 0) * max(min(Y_{2c}, Y_{2n}) - max(Y_{1c}, Y_{1n}) + offset, 0)
  $$

  The operator traverses the candidate boxes in the non-increasing order of sorted_scores. When the score of a candidate box is greater than score_threshold and the box is not suppressed, the operator adds the corresponding index in input_indices to the output. When the IoU between a candidate box and the currently selected box is greater than iou_threshold, the operator suppresses the candidate box.

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
      <td>Candidate rectangular boxes, with shape (N, 4). The coordinate format is (X1, Y1, X2, Y2).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_scores</td>
      <td>Input</td>
      <td>Scores of the candidate rectangular boxes, with shape (N,). They must be sorted in non-increasing order.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_indices</td>
      <td>Input</td>
      <td>Candidate box indices corresponding to sorted_scores, with shape (N). The value range is [0, N).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_output_size</td>
      <td>Input</td>
      <td>Maximum number of candidate boxes to output. The input is a scalar or a tensor with shape (1,).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>iou_threshold</td>
      <td>Input</td>
      <td>IoU threshold for determining whether a candidate box needs to be suppressed. The input is a scalar or a tensor with shape (1,).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>score_threshold</td>
      <td>Input</td>
      <td>Score threshold for filtering candidate boxes. The input is a scalar or a tensor with shape (1,).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Attr</td>
      <td>Offset used when computing coordinate differences. The value is 0 or 1. Default value: 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selected_indices</td>
      <td>Output</td>
      <td>Indices of the selected candidate boxes in the original boxes, with shape (M). M is the runtime computation result and M is not greater than min(max_output_size, N).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- boxes and iou_threshold must have the same data type, and sorted_scores and score_threshold must have the same data type. The data types of the two groups can be different.

## Constraints

- Input shape limit: boxes is a 2D tensor (N, 4), sorted_scores and input_indices are 1D tensors (N,), and max_output_size, iou_threshold, and score_threshold are scalars or tensors with shape (1,).

- sorted_scores must be sorted in non-increasing order, and input_indices must be valid candidate box indices.

## Invocation

| Invocation Method | Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_sorted_nms](./examples/test_geir_sorted_nms.cpp) | Invokes the SortedNMS operator by constructing a graph through the [operator IR](./op_graph/sorted_nms_proto.h). |
