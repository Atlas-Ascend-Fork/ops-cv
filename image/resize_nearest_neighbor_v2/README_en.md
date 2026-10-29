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

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                                                      |
| -------------- | --------------------------- |-------------------------------------------------------------------------|
| aclnn  | [test_aclnn_upsample_nearest2d.cpp](examples/test_aclnn_upsample_nearest2d.cpp) | Call the ResizeNearestNeighborV2 operator through the [aclnnUpsampleNearest2d](docs/aclnnUpsampleNearest2d_en.md) interface. |
| Graph mode | -  | Call the ResizeNearestNeighborV2 operator by building a graph through [operator IR](op_graph/resize_nearest_neighbor_v2_proto.h). |
