# UpsampleNearest3d

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |    √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: applies the nearest neighbor interpolation algorithm to perform upsampling on the input signal composed of multiple input channels.
- Calculation formula:
  - Core algorithm logic:
    1. Map each point of the target image back to the original image to obtain a floating-point coordinate.
    2. Based on this floating-point coordinate, calculate the closest point in the original image, and directly copy the value of the latter to the former.
  - Detailed calculation logic:

    For a point p(x,y,z) in a direction of the output out, the point mapped back to the original image is denoted as q(x',y',z'). The relationship is as follows:

    $$
    x' = \min(\lfloor x * scales\_depth \rfloor, self\_D - 1), \quad scales\_depth = self\_D / outputSize[0]
    $$

    $$
    y' = \min(\lfloor y * scales\_height \rfloor, self\_H - 1), \quad scales\_height = self\_H / outputSize[1]
    $$

    $$
    z' = \min(\lfloor z * scales\_width \rfloor, self\_W - 1), \quad scales\_width = self\_W / outputSize[2]
    $$

    The following formula applies:

    $$
    {V(p_{x,y,z})} = {V(q_{x',y',z'})}
    $$

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
      <td>x</td>
      <td>Input</td>
      <td>Represents the input tensor for upsampling, corresponding to `self` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>Specifies the output spatial size, corresponding to `outputSize` in the formula. The size is 3, and all elements are greater than 0. It indicates the spatial sizes of output `y` in the D, H, and W dimensions.</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the scaling array along each dimension. It contains three elements: scale_depth, scale_height, and scale_width.</li><li>The default value is empty.</li></ul></td>
      <td>LISTFLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor after sampling, corresponding to `out` in the formula. The data type and data format must be the same as the data type and data format of the input parameter `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
  </tbody></table>

- For <term>Atlas inference products</term>, the BFLOAT16 data type of the input and output is not supported.
- For <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>, BFLOAT16 is not supported.

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_nearest1d_v2](examples/test_aclnn_upsample_nearest1d_v2.cpp) | Call the UpsampleNearest3d operator through the [aclnnUpsampleNearest1dV2](docs/aclnnUpsampleNearest1dV2_en.md) interface. |
| aclnn interface  | [test_aclnn_upsample_nearest2d_v2](examples/test_aclnn_upsample_nearest2d_v2.cpp) | Call the UpsampleNearest3d operator through the [aclnnUpsampleNearest2dV2](docs/aclnnUpsampleNearest2dV2_en.md) interface. |
| aclnn interface  | [test_aclnn_upsample_nearest3d](examples/test_aclnn_upsample_nearest3d.cpp) | Call the UpsampleNearest3d operator through the [aclnnUpsampleNearest3d](docs/aclnnUpsampleNearest3d_en.md) interface. |
| Graph mode | - | Call the UpsampleNearest3d operator by constructing a graph in the [operator IR](op_graph/upsample_nearest3d_proto.h) mode. |
