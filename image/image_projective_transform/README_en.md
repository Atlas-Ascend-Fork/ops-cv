# ImageProjectiveTransform

English | [简体中文](./README.md)

## Product Support

|Product | Supported |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: This operator applies a projective transform to the input image. Based on the transformation matrix, it maps each pixel of the output image back to the corresponding coordinate in the input image, and then computes the output pixel values through interpolation.

- Computation formula:

  For each pixel $(x_{out}, y_{out})$ in the output image, the corresponding input image coordinate $(x_{in}, y_{in})$ is computed by the projective transformation matrix:

  $$
  x_{in} = \frac{a_0 \cdot x_{out} + a_1 \cdot y_{out} + a_2}{a_6 \cdot x_{out} + a_7 \cdot y_{out} + 1}
  $$

  $$
  y_{in} = \frac{a_3 \cdot x_{out} + a_4 \cdot y_{out} + a_5}{a_6 \cdot x_{out} + a_7 \cdot y_{out} + 1}
  $$

  where $a_0, a_1, \ldots, a_7$ are the 8 parameters of the transformation matrix, provided by the input `transforms` tensor.

  The output pixel values are computed according to the interpolation mode:

  - **BILINEAR (bilinear interpolation)**:

    $$
    x_{floor} = \lfloor x_{in} \rfloor, \quad x_{ceil} = x_{floor} + 1
    $$

    $$
    y_{floor} = \lfloor y_{in} \rfloor, \quad y_{ceil} = y_{floor} + 1
    $$

    $$
    v_{yfloor} = (x_{ceil} - x_{in}) \cdot p(x_{floor}, y_{floor}) + (x_{in} - x_{floor}) \cdot p(x_{ceil}, y_{floor})
    $$

    $$
    v_{yceil} = (x_{ceil} - x_{in}) \cdot p(x_{floor}, y_{ceil}) + (x_{in} - x_{floor}) \cdot p(x_{ceil}, y_{ceil})
    $$

    $$
    result = (y_{ceil} - y_{in}) \cdot v_{yfloor} + (y_{in} - y_{floor}) \cdot v_{yceil}
    $$

    When the source coordinates exceed the input image boundaries, the fill value (the default is 0) is used to replace the out-of-bounds pixel values. When the denominator of the transformation matrix is 0 or the result is NaN/Inf, floating-point types output NaN, and integer types output INT_MIN.

  - **NEAREST (nearest-neighbor interpolation)**:

    $$
    x_i = \text{round}(x_{in}), \quad y_i = \text{round}(y_{in})
    $$

    When the source coordinates exceed the input image boundaries, the fill value (the default is 0) is used.

## Parameter Description

- **Parameter description**:

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
      <td>images</td>
      <td>Input</td>
      <td>Input image tensor. The shape is (N, H, W, C), where N is the batch size, H is the image height, W is the image width, and C is the number of channels.</td>
      <td>FLOAT16 / FLOAT / UINT8 / INT32</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>transforms</td>
      <td>Input</td>
      <td>Projective transformation matrix parameters. The shape is (N, 8) or (1, 8). The latter broadcasts the same transform to all images. Each row contains 8 transform parameters [a0, a1, a2, a3, a4, a5, a6, a7].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_shape</td>
      <td>Input</td>
      <td>Spatial size of the output image. The shape is (2,) and contains [height, width].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>interpolation</td>
      <td>Attr (Required)</td>
      <td>Interpolation mode. The value is "BILINEAR" (bilinear interpolation) or "NEAREST" (nearest-neighbor interpolation).</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fill_mode</td>
      <td>Attr (Optional)</td>
      <td>Padding mode. "CONSTANT", "NEAREST", "REFLECT", and "WRAP" are supported. The default value is "CONSTANT".</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transformed_images</td>
      <td>Output</td>
      <td>Transformed image tensor. The shape is (N, HOut, WOut, C), and the data type is the same as that of images.</td>
      <td>FLOAT16 / FLOAT / UINT8 / INT32</td>
      <td>NHWC</td>
    </tr>
  </tbody></table>

## Constraints

- The input images must be four-dimensional (NHWC format).
- The input transforms must be two-dimensional with the shape (M, 8). M must be 1 or the batch size N of images, and the data type is FLOAT. When N is 0, M can be 0 or 1.
- The input output_shape must be one-dimensional with the shape (2,), and the data type is INT32.
- The two elements [height, width] of output_shape determine the middle two dimensions (HOut, WOut) of the output transformed_images, that is, transformed_images.shape[1]=output_shape[0] and transformed_images.shape[2]=output_shape[1]. When an element of output_shape is 0, the output is an empty tensor. The operator returns directly and does not perform computation.
- The known length of output_shape must be exactly 2. The readable constants height and width must be non-negative. 0 is reserved for an empty output, and negative values cause an error.
- fill_mode is case-sensitive. "CONSTANT" pads out-of-bounds positions with 0, "NEAREST" clamps to the edge coordinates, "REFLECT" mirrors repeatedly at the edges, and "WRAP" remaps periodically. An invalid fill_mode or interpolation causes an error, and an empty output does not bypass the check.
- InferShape allows an unknown rank and partial dynamic dimensions, and retains the known N/C and constant H/W as much as possible. When no readable output_shape constant exists, the output H/W is -1. InferShape infers the NCHW/NHWC dimension positions according to the origin format of images. The actual AI Core kernel still supports only NHWC.
- Tiling preferentially uses the readable output_shape data, and then uses the non-negative output storage H/W (including 0). When both are unavailable, Tiling falls back to the input H/W. An error is reported when the batch is less than the dynamic sentinel -1 or the spatial product exceeds the int64 range.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| Graph mode invocation | [test_geir_image_projective_transform](./examples/arch35/test_geir_image_projective_transform.cpp) | Calls the ImageProjectiveTransform operator by constructing a graph in the [operator IR](./op_graph/image_projective_transform_proto.h) mode. |
