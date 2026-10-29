# ScaleAndTranslate

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: performs 2D resampling on the input image according to the given output size, scaling factors, and translation values.

## Parameter Description

| Parameter name | Input/Output/Attr | Description | Data type | Data format |
|-----|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|------|
| images   | Input       | The input is a 4D tensor, with the shape (batch, input_height, input_width, channels). | INT8, UINT8, INT16, UINT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE | NHWC |
| size   | Input       | 1D tensor containing 2 elements, representing the output height and output width [output_height, output_width] in order, and both values must be positive. | INT32                                                   | ND   |
| scale   | Input       | 1D tensor containing 2 elements, representing the scaling factors in the row and column directions [row_scale, col_scale] in order, and both values must be greater than 0. | FLOAT                                                   | ND   |
| translation   | Input       | 1D tensor containing 2 elements, representing the translation values in the row and column directions [row_translation, col_translation] in order. | FLOAT                                                   | ND   |
| kernel_type   | Optional attr     | • Specifies the resampling kernel type. Supports "lanczos1", "lanczos3", "lanczos5", "gaussian", "box", "triangle", "keyscubic", and "mitchellcubic".<br>• The default value is "lanczos3". | STRING                                                  | -    |
| antialias   | Optional attr     | • Specifies whether to generate sampling weights with a larger effective kernel radius in downscaling scenarios to reduce aliasing.<br>• The default value is true. | BOOL                                                    | -    |
| y   | Output       | Returns a 4D FLOAT tensor, with the shape (batch, output_height, output_width, channels). | FLOAT                                                   | NHWC |

## Constraints

- images must be a 4D tensor, and input_height, input_width, and channels must be greater than 0.
- size must be a 1D INT32 tensor containing 2 elements, and output_height and output_width must be positive.
- scale reads the first 2 FLOAT elements according to the source code implementation, representing row_scale and col_scale respectively, and both values must be greater than 0.
- translation reads the first 2 FLOAT elements according to the source code implementation, representing row_translation and col_translation respectively.
- The data type of y is fixed to FLOAT, and the shape is jointly determined by the batch and channels of images and the output height and width specified by size.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| Graph mode invocation | [test_geir_scale_and_translate](./examples/test_geir_scale_and_translate.cpp) | Call the ScaleAndTranslate operator by building a graph through [operator IR](./op_graph/scale_and_translate_proto.h). |
