# ResizeNearestNeighborV2

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function Description

- Operator function: applies the nearest neighbor interpolation algorithm to the input signal composed of multiple input channels for upsampling. If the input shape is (N, C, H, W), the output shape is (N, C, size[0], size[1]).
- Calculation formula:

  $$
  y(N, C, H, W) = x(N, C, min(floor(H * scaleH),  H-1), min(floor(W * scaleW),  W-1))
  $$

  $$
  scaleH = x\_H / size[0]
  $$

  $$
  scaleW = x\_W / size[1]
  $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 223px">
  <col style="width: 101px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>4D tensor of the input image, corresponding to `x` in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>Height and width of the output image.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Image after nearest neighbor interpolation, corresponding to `y` in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
  </tbody>
  </table>

  **Attribute description:**

  | Attribute | Type | Default value | Description |
  | :---- | :---- | :---- | :---- |
  | align_corners | Bool | false | Determines whether to align the corner pixels of the input and output images. When set to true, corner pixels are aligned and the source coordinate is mapped by $(H_{in}-1)/(size[0]-1)$; when set to false, scaling is proportional and corner pixels are not strictly aligned. |
  | half_pixel_centers | Bool | false | Determines the pixel center position during interpolation. When set to true, the pixel center is at half-pixel coordinates and the source coordinate is mapped by $(h+0.5) \times scaleH$; when set to false, the pixel center is at integer coordinates. |
  | scales | ListFloat | {0.0f, 0.0f} | Spatial size multiplier, an optional parameter. Only the default value {0.0f, 0.0f} is supported (meaning the scaling factor is derived from `size` and the input spatial size); non-default values are not supported. |

## Constraints

- Only 4D tensors are supported, with the data format of NCHW or NHWC.
- `align_corners` and `half_pixel_centers` cannot both be true: this combination has no corresponding TensorFlow ResizeNearestNeighbor semantics and is rejected in the Tiling phase. The valid combinations are (false, false), (false, true), and (true, false).
- `size` must be a 1D tensor containing exactly 2 INT32 elements (the output height and width respectively). This input is a value-dependent input: when `size` is a constant, it participates in InferShape to derive the output size.
- Neither the input `x` nor the output `y` supports empty tensors: when either has zero elements, the Tiling phase returns a failure (the error message is `Input x and output y do not support empty tensors`).
- `scales` only supports the default value {0.0f, 0.0f}; non-default values are not supported.
- The N and C axes of the output `y` must be consistent with the input `x`, and the data type and data format must also be consistent with `x`.
- In dynamic shape scenarios, the input shape supports dynamic marking. InferShape propagates as follows:
  - When `size` is a non-constant (its value is unknown), the H and W dimensions of the output `y` are derived as -1 (unknown dimension), while the N and C dimensions still follow the corresponding values of `x`.
  - When the shape of the input `x` is -2 (unknown rank), the input is accepted and the output is derived as \[-1, -1, size\[0\], size\[1\]\]; if `size` is also a non-constant, the output H and W are also -1.
- Two-layer data type scope: FLOAT16, FLOAT32, and BFLOAT16 listed in the table above are the data types supported by the AI Core implementation of this operator (see the operator registration in op_host), and are also the available range of the ACLNN interface on <term>Ascend 950PR&950DT products</term>; the data types declared in the [operator IR](op_graph/resize_nearest_neighbor_v2_proto.h) are a larger set aligned with TensorFlow ResizeNearestNeighbor (including INT8, UINT8, INT16, UINT16, INT32, INT64, and DOUBLE). In graph mode, these types do not go through AI Core and are carried by the AICPU implementation. When invoking based on this README, refer to the table above.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                                                      |
| -------------- | --------------------------- |-------------------------------------------------------------------------|
| aclnn  | [test_aclnn_upsample_nearest1d.cpp](examples/test_aclnn_upsample_nearest1d.cpp) | Call the ResizeNearestNeighborV2 operator through the [aclnnUpsampleNearest1d](docs/aclnnUpsampleNearest1d_en.md) interface to perform nearest neighbor upsampling on the L dimension of a 3D (NCL) input. |
| aclnn  | [test_aclnn_upsample_nearest2d.cpp](examples/test_aclnn_upsample_nearest2d.cpp) | Call the ResizeNearestNeighborV2 operator through the [aclnnUpsampleNearest2d](docs/aclnnUpsampleNearest2d_en.md) interface. |
| Graph mode | -  | Call the ResizeNearestNeighborV2 operator by building a graph through [operator IR](op_graph/resize_nearest_neighbor_v2_proto.h). |
