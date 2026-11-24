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

- Operator function: converts YUV444 format image data to YUV422 format. Each pixel position of the input contains four channels [Y, U, V, pad], where the fourth channel (index 3) is padding and is ignored. Each adjacent pixel pair is emitted in YUYV byte order: the even column is [Y_even, U_even], and the odd column is [Y_odd, V_odd]; the output shape is (h, w, 2). U and V come from different input columns, so this is not co-sited chroma sampling.

- Calculation formula:

$$
y[i, 2j, 0] = \text{clip\_uint8}(x[i, 2j, 0])
$$
$$
y[i, 2j, 1] = \text{clip\_uint8}(x[i, 2j, 1])
$$
$$
y[i, 2j+1, 0] = \text{clip\_uint8}(x[i, 2j+1, 0])
$$
$$
y[i, 2j+1, 1] = \text{clip\_uint8}(x[i, 2j+1, 2])
$$

The byte order for each adjacent pixel pair in contiguous memory is [Y0, U0, Y1, V1]. When w is odd, the unpaired last column outputs [Y_last, U_last].

- Numerical conversion behavior (float16 → uint8): the value is first rounded to the nearest integer (half away from zero, e.g., 2.5 → 3) and then clamped to [0, 255], that is, clip_uint8(x) = min(max(round(x), 0), 255). Special values: NaN outputs 0; +Inf outputs 255; -Inf and values less than 0 output 0; values greater than 255 output 255.

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
      <td>YUV422 output image data. The shape is (h, w, 2), packed as YUYV: even columns are [Y, U], and odd columns are [Y, V].</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The input must be a 3D tensor, and the third dimension is fixed to 4 (the channel order is [Y, U, V, pad]; the fourth channel (index 3) is ignored).
- The output is a 3D tensor. The first two dimensions are the same as those of the input, and the third dimension is fixed to 2; contiguous memory uses YUYV byte order.
- Odd W: supported. The output is still (h, w, 2), and the pixels in the last column write only the [Y, U] components (no paired pixels, and no out-of-bounds reads or writes occur).
- Only float16 input and uint8 output are supported.
- The input and output require contiguous storage (ND layout).

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_yuv4442_yuv422](examples/test_geir_yuv4442_yuv422.cpp) | Call the YUV4442YUV422 operator by constructing a graph in the [operator IR](op_graph/yuv4442_yuv422_proto.h) mode. |
