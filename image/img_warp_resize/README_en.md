# IMGWarpResize

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: This is an image bilinear interpolation sampling operator. It is used for image warp resizing in OCR scenarios. It receives the preprocessed four-corner pixel values and floating-point coordinates, and computes the output pixel values through bilinear interpolation.

- Computation formula:

For each pixel position (n, c, h, w):

$$
x_{lerp} = x_{float} - \lfloor x_{float} \rfloor
$$

$$
y_{lerp} = y_{float} - \lfloor y_{float} \rfloor
$$

$$
top = top_{left} + (top_{right} - top_{left}) \times x_{lerp}
$$

$$
bottom = bottom_{left} + (bottom_{right} - bottom_{left}) \times x_{lerp}
$$

$$
warp_{img} = top + (bottom - top) \times y_{lerp}
$$

Here $x_{float}$ and $y_{float}$ come from warp_index, and the four-corner pixel values come from dimension 1 of img (in the order of top_left, top_right, bottom_left, bottom_right).

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 310px">
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
      <td>img</td>
      <td>Input</td>
      <td>Four-corner pixel values. The shape is [N, 4, C, H, W], and the 4 in dimension 1 = [top_left, top_right, bottom_left, bottom_right].</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>warp_index</td>
      <td>Input</td>
      <td>Floating-point coordinates. The shape is [N, 2, H, W], and the 2 in dimension 1 = [x_float, y_float].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>warp_img</td>
      <td>Output</td>
      <td>Bilinear interpolation result. The shape is [N, C, H, W], and the dtype is the same as that of img.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- img must be a 5D tensor [N, 4, C, H, W], and dimension 1 must be 4.
- warp_index must be a 4D tensor [N, 2, H, W], and dimension 1 must be 2.
- The N, H, and W dimensions of img and warp_index must be the same.
- Empty tensors are not supported (none of N, C, H, or W can be 0).
- Non-contiguous tensors are not supported (AutoContiguous is automatically called internally).
- warp_index is always of the float32 type.
- All intermediate computations are performed in float32 precision. float16 inputs are first converted to float32 for computation and then converted back to float16 for output.
- The total number of output elements N*C*H*W does not exceed INT32_MAX.

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_img_warp_resize](examples/arch35/test_geir_img_warp_resize.cpp) | Calls the IMGWarpResize operator by constructing a graph in the [operator IR](op_graph/img_warp_resize_proto.h) mode. |
