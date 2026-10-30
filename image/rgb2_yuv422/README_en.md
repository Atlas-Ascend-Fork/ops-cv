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

- Numerical behavior: the preceding floating-point intermediate results are first saturating-clamped to [0, 255] and then converted to the uint8 output by rounding to nearest even.

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

<table style="table-layout: fixed; width: 980px"><colgroup>
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

- The input is a uint8 tensor in ND format, with the shape [..., H, W, 3]; the output shape is [..., H, W, 2] (YUYV packed).
- Dimension constraint: the rank of the input is in [3, 8] (the FORMAT_ND platform limit is 8). Inputs with rank < 3 or rank > 8 are rejected; the size of the channel dimension (the last dimension) must be 3.
- Dynamic shape semantics: during graph construction, each dimension (including the channel dimension) can be -1 (unknown dimension), and the corresponding output dimension remains -1; for an input with unknown rank (-2), the output also has an unknown rank. After the shape is determined at runtime, the preceding dimension and channel constraints must still be met.
- Empty tensors: an empty input with H=0 or W=0 is a valid input, and the output is the corresponding empty tensor.
- Odd W: supported. The output is still [..., H, W, 2], and the pixels in the last column write only the [Y, U] components (no paired pixels, and no out-of-bounds reads or writes occur).
- Memory layout: the input and output must be contiguous ND tensors, and the kernel accesses them by linear address.
- Large-size inputs: when any of H, W, or outerDims exceeds INT32_MAX, the 64-bit index mode (TilingKey schMode=1) is used automatically, and the computation results remain correct.
- Exception rejection: inputs whose rank exceeds the limit, whose channel dimension is not 3 (and not -1), or whose dtype is not uint8 are rejected in the InferShape/tiling stage with an error log; the kernel API does not define an independent error code, and the rejection is reflected as a compilation or execution failure.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_rgb2_yuv422](examples/test_geir_rgb2_yuv422.cpp)  | Call the RGB2YUV422 operator by building a graph through [operator IR](op_graph/rgb2_yuv422_proto.h). |
