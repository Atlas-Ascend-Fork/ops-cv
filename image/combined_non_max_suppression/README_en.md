# CombinedNonMaxSuppression

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
|:--|:--:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: This operator performs greedy non-maximum suppression (NMS) independently for each batch and each class. It then merges the candidate boxes of all classes in descending order of confidence.
- Input and output sizes:

  - `boxes`: The shape is (batch, num_boxes, q, 4), where q is 1 or num_classes.
  - `scores`: The shape is (batch, num_boxes, num_classes).
  - `nmsed_boxes`: The shape is (batch, output_size, 4).
  - `nmsed_scores` and `nmsed_classes`: The shape is (batch, output_size).
  - `valid_detections`: The shape is (batch).

## Parameter Description

| Parameter name | Input/Output/Attr | Description | Data type | Data format |
|:--|:--|:--|:--|:--|
| boxes | Input | Candidate box coordinates. The shape is (batch, num_boxes, q, 4). The last dimension indicates y1, x1, y2, and x2 in order. | FLOAT16, FLOAT32 | ND |
| scores | Input | Candidate box scores. The shape is (batch, num_boxes, num_classes). | FLOAT16, FLOAT32 | ND |
| max_output_size_per_class | Input | Maximum number of candidate boxes retained for each class. Value range: [1, 1000]. | INT32 | ND |
| max_total_size | Input | Maximum number of candidate boxes output for each batch. Value range: [1, 1000]. | INT32 | ND |
| iou_threshold | Input | IoU suppression threshold. The Ascend 950PR&950DT products support 0 and positive numbers. | FLOAT32 | ND |
| score_threshold | Input | Score threshold. Only candidate boxes whose scores are strictly greater than this value are retained. | FLOAT32 | ND |
| pad_per_class | Optional attr | Whether to pad the output by class. If this parameter is true, output_size is min(max_total_size, max_output_size_per_class * num_classes). The default value is false. | BOOL | - |
| clip_boxes | Optional attr | Whether to clip the output candidate box coordinates to [0, 1]. The default value is true. | BOOL | - |
| nmsed_boxes | Output | Filtered candidate boxes. | FLOAT16, FLOAT32 | ND |
| nmsed_scores | Output | Filtered candidate box scores. | FLOAT16, FLOAT32 | ND |
| nmsed_classes | Output | Filtered class indexes, expressed as floating-point numbers. | FLOAT16, FLOAT32 | ND |
| valid_detections | Output | Number of valid candidate boxes in each batch. | INT32 | ND |

## Constraints

- The data types of `boxes`, `scores`, and the first three outputs must be the same. The two thresholds are always FLOAT32.
- `boxes` must be a four-dimensional tensor, and `scores` must be a three-dimensional tensor. The batch and num_boxes dimensions of the two tensors must be the same.
- The last dimension of `boxes` must be 4, and the q dimension must be 1 or num_classes.
- batch, num_boxes, and num_classes must be greater than 0. num_classes cannot exceed 200.
- num_boxes cannot exceed `INT32_MAX`. The product of batch and num_classes cannot exceed `INT32_MAX`. The total number of elements in boxes cannot exceed `INT64_MAX`.
- `max_output_size_per_class` and `max_total_size` must be scalars. The value range of both parameters is [1, 1000].
- `iou_threshold` and `score_threshold` must be scalars.
- <term>Ascend 950PR&950DT products</term>:
  - `iou_threshold` supports 0 and positive numbers. It does not support negative numbers, NaN, or infinity.
  - If `iou_threshold > 1`, IoU suppression is not performed. However, score filtering by `score_threshold` and the output quantity limits of `max_output_size_per_class` and `max_total_size` still take effect.
- `max_output_size_per_class`, `max_total_size`, `iou_threshold`, and `score_threshold` are value-dependent inputs. The input values must be known at compilation time. The values of the first two parameters are used for output shape inference, and the values of the last two parameters are used for NMS computation.
- The candidate box coordinates are in the order of (y1, x1, y2, x2). Reversed coordinates are supported. When computing IoU, the operator takes the minimum and maximum values of the coordinate endpoints respectively.
- Currently, GE IR dynamic shape and dynamic rank scenarios are not supported (the input shape declarations are `[-1]` and `[-2]` respectively).

## Invocation Description

| Invocation Method | Sample Code | Description |
|:--|:--|:--|
| Graph mode | [test_geir_combined_non_max_suppression](examples/arch35/test_geir_combined_non_max_suppression.cpp) | Calls the CombinedNonMaxSuppression operator by constructing a graph in the [operator IR](op_graph/combined_non_max_suppression_proto.h) mode. For compilation and verification, refer to [Operator Invocation](../../docs/zh/invocation/quick_op_invocation.md). |
