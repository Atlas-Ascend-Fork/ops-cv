# GridUnnormal

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: GridUnnormal is the coordinate denormalization operator in the GridSample pipeline. It takes the normalized coordinates `grid` and the size auxiliary tensor `assist` with the same shape as inputs, and outputs the fractional offset `diff` and the integer sampling position `position`.

- Computation formula: Each element is computed independently:

  ```text
  t        = (grid + 1) * 0.5
  pos_base = align_corners ? t * (assist - 1) : t * assist - 0.5
  position = floor(pos_base)
  diff     = pos_base - floor(pos_base)
  ```

`floor` rounds toward negative infinity. It cannot be replaced by truncation toward zero.

## Parameter Description

| Parameter name | Input/Output/Attr | Description | Data type | Data format |
|---|---|---|---|---|
| `grid` | Input | Normalized sampling coordinates. It supports 4D static shapes, dynamic shapes, and an unknown rank at compilation time. When the rank is determined, it must be 4 and the last dimension must be 2. Empty Tensors with a total of 0 elements are supported. | `FLOAT16`, `FLOAT` | ND |
| `assist` | Input | Input size auxiliary value corresponding to each element. It must be exactly the same as `grid` in shape and dtype. Broadcasting is not supported. When the rank is determined, it must be 4 and the last dimension must be 2. | `FLOAT16`, `FLOAT` | ND |
| `diff` | Output | Fractional part of `pos_base`. Its shape and dtype are the same as those of `grid`. | `FLOAT16`, `FLOAT` | ND |
| `position` | Output | Floor result of `pos_base`. Its shape is the same as that of `grid`. | `INT32` | ND |
| `align_corners` | Attr | Optional attribute. The default value is `false`. If the value is `true`, the computation uses `t * (assist - 1)`. If the value is `false`, the computation uses `t * assist - 0.5`. | `BOOL` | - |

## Constraints

- The shape and dtype of `grid` and `assist` must be the same.
- When the rank is determined, `grid` and `assist` must be 4D Tensors with the shape `[batch, height, width, 2]`.
- Only the ND format is supported. Non-contiguous Tensors are not involved as user-visible APIs.
- The dtype of `diff` follows `grid`. `position` is fixed to int32.
- Intermediate computation uses fp32. fp16 inputs are promoted to fp32 for computation, and `diff` is written back as fp16.
- Empty Tensors with a total of 0 elements support empty input and empty output, and the device side does not access data.
- Non-finite inputs and `floor(pos_base)` values beyond the int32 representable range are outside the supported scope of this operator.
- This operator is an internal GE graph operator. It does not provide external APIs for aclnn, torch, TensorFlow, ONNX, or Caffe.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|---|---|---|
| Graph mode invocation | [test_geir_grid_unnormal](examples/arch35/test_geir_grid_unnormal.cpp) | Calls the GridUnnormal operator by constructing a graph in the [operator prototype](op_graph/grid_unnormal_proto.h) mode of this directory. |
