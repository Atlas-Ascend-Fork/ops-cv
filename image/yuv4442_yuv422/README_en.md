# YUV4442YUV422

English | [简体中文](./README.md)

## Product Support

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: converts YUV444 format image data to YUV422 format. Each pixel position of the input contains four channels [Y, U, V, pad], where the fourth channel (index 3) is a padding position and is ignored. The operator performs 2:1 subsampling (direct selection, cosited) on the chroma components of horizontally adjacent pixels, and outputs two-channel data in YUV422 format.

- Calculation formula:

$$
y[i, j, 0] = \text{clip\_uint8}(Y_0)
$$
$$
y[i, j, 1] = \text{clip\_uint8}(U_0)
$$
$$
y[i, j+1, 0] = \text{clip\_uint8}(Y'_0)
$$
$$
y[i, j+1, 1] = \text{clip\_uint8}(V_0)
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
      <td>x</td>
      <td>Input</td>
      <td>YUV444 input image data. The shape is (h, w, 4), and each pixel has four channels [Y, U, V, pad]; the fourth channel (index 3) is ignored.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>YUV422 output image data. The shape is (h, w, 2), and each pixel has two channels (Y, UV).</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The input must be a 3D tensor, and the third dimension is fixed to 4 (the channel order is [Y, U, V, pad]; the fourth channel (index 3) is ignored).
- The output is a 3D tensor. The first two dimensions are the same as those of the input, and the third dimension is fixed to 2 (the two-channel packed format of YUV422).
- Only float16 input and uint8 output are supported.
- The input and output require contiguous storage (ND layout).

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_yuv4442_yuv422](examples/test_geir_yuv4442_yuv422.cpp) | Call the YUV4442YUV422 operator by constructing a graph in the [operator IR](op_graph/yuv4442_yuv422_proto.h) mode. |
