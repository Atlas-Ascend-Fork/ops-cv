# IMGWarpOffsets

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

- Operator function: This operator performs an offset transformation on the input tensor images based on the tensor offsets, and generates the warp_images tensor.

## Parameter Description

| Parameter name | Input/Output/Attr | Description | Data type | Data format |
|---------------------|----------|----------------------------------------------------------------------------|---------------------|------|
| images | Input | A four-dimensional Tensor. Its shape is (batch, image_height, image_width, 3). | UINT8, FLOAT16, FLOAT | ND |
| offsets | Input | A four-dimensional Tensor. Its shape is (batch, 4, new_height, new_width). | FLOAT, INT32 | ND |
| warp_images | Output | A five-dimensional Tensor. Its shape is (batch, 4, new_height, new_width, 3), and its data type is the same as that of the input images. | UINT8, FLOAT16, FLOAT | ND |

## Constraints

- When the data type of offsets is INT32, images must be FLOAT16.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| Graph mode invocation | [test_geir_image_warp_offsets](./examples/test_geir_image_warp_offsets.cpp) | Calls the IMGWarpOffsets operator by constructing a graph in the [operator IR](./op_graph/image_warp_offsets_proto.h) mode. |
