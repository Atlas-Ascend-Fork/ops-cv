# NonMaxSuppressionV3

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: This operator selects a subset of candidate boxes (bounding boxes) in descending order of scores using a greedy strategy.

## Parameter Description

| Parameter name | Input/Output/Attr | Description | Data type | Data format |
|-------------------|----------|-----------------------------------------------------------------------------------|---------------|------|
| boxes | Input | A two-dimensional Tensor. Its shape is (num_boxes, 4). The input format is (x1, y1, x2, y2), and x1 < x2 and y1 < y2 are required. | FLOAT16, FLOAT | ND |
| scores | Input | A one-dimensional Tensor. Its shape is (num_boxes). It indicates the scores corresponding to the candidate boxes (one-to-one correspondence with each row of boxes). | FLOAT16, FLOAT | ND |
| max_output_size | Input | A scalar integer Tensor. It indicates the maximum number of candidate boxes selected by non-maximum suppression (Non-Maximum Suppression). | INT32 | ND |
| iou_threshold | Input | A scalar floating-point Tensor. It indicates the threshold used to determine whether candidate boxes overlap too much in IoU (intersection over union). | FLOAT16, FLOAT | ND |
| score_threshold | Input | A scalar floating-point Tensor. It indicates the threshold used to remove candidate boxes based on scores. | FLOAT16, FLOAT | ND |
| offset | Optional attr | • An optional integer.<br>• The default value is 0. | INT | - |
| selected_indices | Output | A one-dimensional integer Tensor with the shape (M). It indicates the indexes selected from the input boxes Tensor, where M <= max_output_size. | INT32 | ND |

## Constraints

- boxes and scores must use the same data type, which is either FLOAT16 or FLOAT.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| Graph mode invocation | [test_geir_non_max_suppression_v3](./examples/test_geir_non_max_suppression_v3.cpp) | Calls the NonMaxSuppressionV3 operator by constructing a graph in the [operator IR](./op_graph/non_max_suppression_v3_proto.h) mode. |
