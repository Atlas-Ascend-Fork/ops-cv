# Lut3D

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

- Operator function: 3D LUT (Look-Up Table) trilinear interpolation color transformation. This operator maps the colors of the input image through a 3D color look-up table. It supports trilinear interpolation to achieve smooth color transitions.
- Inputs:
  - `img`: Input image. It supports the ND or NHWC format, and the last dimension is 3 (B/G/R channels).
  - `lut_table`: 3D LUT table. The shape is [N, N, N, 3], where N is the LUT side length (maximum 20).
- Output:
  - `lut_img`: Image after the 3D LUT transformation. The output dtype is always float32.

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
      <td>img</td>
      <td>Input</td>
      <td>Input image. The last dimension is 3 (B/G/R channels).</td>
      <td>UINT8, FLOAT</td>
      <td>ND, NHWC</td>
    </tr>
    <tr>
      <td>lut_table</td>
      <td>Input</td>
      <td>3D LUT table. The shape is [N, N, N, 3].</td>
      <td>UINT8, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lut_img</td>
      <td>Output</td>
      <td>Image after the 3D LUT transformation. The dtype is always float32.</td>
      <td>FLOAT</td>
      <td>ND, NHWC</td>
    </tr>
  </tbody></table>

## Constraints

- The input image must be 3D or 4D, and the last dimension must be 3.
- The LUT table must be 4D. The first three dimensions are equal (N <= 20), and the last dimension is 3.
- The element value ranges of both UINT8 and FLOAT inputs are [0, 255].
- Supported dtype combinations:
  - uint8 × uint8 → float32
  - float32 × float32 → float32

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---- | ---- | ---- |
| Graph mode | [test_geir_lut3_d](examples/arch35/test_geir_lut3_d.cpp) | Calls the Lut3D operator by constructing a graph in the [operator IR](op_graph/lut3_d_proto.h) mode. For compilation and verification, refer to [Operator Invocation](../../docs/zh/invocation/quick_op_invocation.md). |
