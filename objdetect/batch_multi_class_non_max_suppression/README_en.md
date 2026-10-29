# BatchMultiClassNonMaxSuppression

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: performs greedy non-maximum suppression (NMS) on candidate boxes of each batch and each class, and then selects at most `max_total_size` detection boxes by score from the retained results of all classes.
- The coordinate format of boxes is `[y_min, x_min, y_max, x_max]`. When `q=1`, all classes share boxes; when `q=C`, each class uses its own boxes.
- When `clip_window` is provided, the operator performs cropping first. When `change_coordinate_frame=true`, the operator then performs normalization with the top-left corner of the window as the origin and the width and height of the window as the scale.

For a candidate box `a` and a selected box `b`, the IoU is calculated as follows:

$$
IoU(a,b)=\frac{Area(a\cap b)}{max(Area(a)+Area(b)-Area(a\cap b), 1e-12)}
$$

A score is retained only when `score > score_threshold`. A box is suppressed when the IoU is strictly greater than `iou_threshold`. At most `max_size_per_class` boxes are retained for each class. If the final result contains fewer than `max_total_size` boxes, the result is padded with 0.

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
      <td>Candidate boxes. The coordinate format is [y_min, x_min, y_max, x_max], and the shape is [B,N,q,4]; q is 1 or C.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scores</td>
      <td>Input</td>
      <td>Scores of each candidate box and each class. The shape is [B,N,C], and the data type must be the same as that of boxes.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clip_window</td>
      <td>Optional input</td>
      <td>Cropping window. The coordinate format is yxyx, and the shape is [B,4]. A null pointer can be passed in, and the data type must be the same as that of boxes.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_valid_boxes</td>
      <td>Optional input</td>
      <td>Number of valid candidate boxes in each batch. The shape is [B]. At runtime, each value is limited to [0,N]: a value less than 0 is processed as 0, and a value greater than the number N of candidate boxes is processed as N. A null pointer can be passed in.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>score_threshold</td>
      <td>Attr</td>
      <td>Score threshold. Only candidate boxes with scores strictly greater than this value are retained. The value is not limited to a finite one: -inf disables the threshold filtering on finite scores, and +inf or NaN filters out all candidate boxes.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>iou_threshold</td>
      <td>Attr</td>
      <td>IoU suppression threshold. A box is suppressed when the IoU is strictly greater than this value. The value is not limited to a finite one or [0,1]: a value greater than or equal to 1, +inf, or NaN does not trigger suppression, and a value less than 0 or -inf suppresses the candidate boxes after the highest-score box in each class.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_size_per_class</td>
      <td>Attr</td>
      <td>Maximum number of candidate boxes retained for each class. The value range is [1,1000].</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_total_size</td>
      <td>Attr</td>
      <td>Maximum number of candidate boxes retained for each batch. The value range is [1,1000].</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>change_coordinate_frame</td>
      <td>Attr</td>
      <td>Determines whether to normalize the cropped coordinates to the window coordinate system. The default value is false; clip_window must be provided when the value is true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transpose_box</td>
      <td>Attr</td>
      <td>Determines whether to insert Transpose before the operator. Currently only false is supported, and the default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nmsed_boxes</td>
      <td>Output</td>
      <td>Detection boxes after NMS. The shape is [B,M,4], where M is max_total_size.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nmsed_scores</td>
      <td>Output</td>
      <td>Detection scores after NMS. The shape is [B,M].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nmsed_classes</td>
      <td>Output</td>
      <td>Class indices after NMS. The shape is [B,M], and the values are represented in floating-point type.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nmsed_num</td>
      <td>Output</td>
      <td>Number of valid outputs in each batch. The shape is [B].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Only ND format and float16/float32 boxes and scores are supported; the data types of the three floating-point outputs are the same as the dtype of boxes.
- boxes is 4D and scores is 3D. B, N, C, and q must be positive; q must be 1 or C.
- The public interface of `transpose_box` currently supports only `false`, consistent with the public interface constraint of <term>Atlas A2 products</term>; internal nodes generated by graph fusion still use the `true` layout.
- The shape of `clip_window` must be `[B,4]`, and the shape of `num_valid_boxes` must be `[B]`.
- GE IR dynamic shape is supported; the runtime tiling receives the instantiated concrete positive shape.
- `image_size` is a compatibility attribute of the graph mode prototype. The current general NMS path does not read this attribute; the `norm_class` semantics dedicated to old platforms are not within the scope of this implementation.
- The relative order of equal scores is not a public interface commitment; the caller should use `nmsed_num` to determine the valid output range.

## Invocation Description

| Invocation Method | Sample Code | Description |
| :--- | :--- | :--- |
| Graph mode | [test_geir_batch_multi_class_non_max_suppression](examples/arch35/test_geir_batch_multi_class_non_max_suppression.cpp) | Call the BatchMultiClassNonMaxSuppression operator by constructing a graph in the [operator IR](./op_graph/batch_multi_class_non_max_suppression_proto.h) mode. |
