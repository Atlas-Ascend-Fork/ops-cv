# RGB2YUV422

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |     √    |
| <term>Atlas A3 products</term>      |    √     |
| <term>Atlas A2 products</term>      |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                       |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: converts an RGB image to the YUV422 (YUYV packed format) color space.

- Calculation formula:

uint8 input (BT.601/256 fixed-point coefficients, with a +128 offset):
$$
Y  = (77 \cdot R + 150 \cdot G + 29 \cdot B) / 256
$$
$$
U  = (-43 \cdot R - 85 \cdot G + 128 \cdot B) / 256 + 128
$$
$$
V  = (128 \cdot R - 107 \cdot G - 21 \cdot B) / 256 + 128
$$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>rgb</td>
      <td>Input</td>
      <td>RGB image tensor, with the shape [..., H, W, 3].</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>yuv</td>
      <td>Output</td>
      <td>YUV422 (YUYV packed) tensor, with the shape [..., H, W, 2].</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

The input must have at least 3 dimensions, and the size of the channel dimension must be 3. Only uint8 input and output and the ND format are supported.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_rgb2_yuv422](examples/test_geir_rgb2_yuv422.cpp)  | Call the RGB2YUV422 operator by building a graph through [operator IR](op_graph/rgb2_yuv422_proto.h). |
