# NMSWithMask

English | [简体中文](./README.md)

## Product Support

|Product | Supported |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: This operator performs non-maximum suppression (NMS) on bounding boxes, and outputs the boxes, indexes, and masks selected after NMS. It is used for post-processing in object detection to remove duplicated detected boxes.

## Parameter Description

| Parameter name | Input/Output/Attr | Description | Data type | Data format |
|:-------------------------|:----------:|:-----|:-----|:----:|
| box_scores | Input | A two-dimensional Tensor. Its shape is (num_boxes, 5), where 5 indicates [y1, x1, y2, x2, score]. num_boxes does not exceed 39936. | FLOAT16, FLOAT, BF16 | ND |
| iou_threshold | Attr | A floating-point number. It indicates the threshold used to determine whether candidate boxes overlap too much in IoU (intersection over union). The default value is 0.5. | FLOAT | - |
| selected_boxes | Output | A two-dimensional Tensor. Its shape is (num_boxes, 5), the same as the original input box_scores. | FLOAT16, FLOAT, BF16 | ND |
| selected_idx | Output | A one-dimensional Tensor. Its shape is (num_boxes), indicating the sequence numbers from 0 to num_boxes - 1. | INT32 | ND |
| selected_mask | Output | A one-dimensional Tensor. Its shape is (num_boxes), indicating the mask status of the target boxes. | UINT8 | ND |

## Constraints

- The last dimension of the input box_scores must be 5, indicating [y1, x1, y2, x2, score] in order.
- The batch dimensions of the outputs selected_boxes, selected_idx, and selected_mask are the same as the batch dimension of the input box_scores.
- In FLOAT16 or BF16 scenarios, the sorting and computation of the operator may introduce computation errors compared with the benchmark.

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | - | Calls the NMSWithMask operator by constructing a graph in the [operator IR](op_graph/nms_with_mask_proto.h) mode. |
