# ResizeBilinearV2

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

- Operator function: resizes an image to the specified size using bilinear interpolation.

- Calculation formula:

  Assume that the values of four known points $Q_{11}(x_1, y_1), Q_{12}(x_1, y_2), Q_{21}(x_2, y_1), Q_{22}(x_2, y_2)$ in x are given. The following formulas interpolate the data to obtain the value at the point coordinate $P(a, b)$ in y.

  $$
  f(R_1) = \frac{x_2 - a}{x_2 - x_1}f(Q_{11}) + \frac{a - x_1}{x_2 - x_1}f(Q_{21})
  $$

  $$
  f(R_2) = \frac{x_2 - a}{x_2 - x_1}f(Q_{12}) + \frac{a - x_1}{x_2 - x_1}f(Q_{22})
  $$

  $$
  f(P) = \frac{y_2 - b}{y_2 - y_1}f(R_{1}) + \frac{b - y_1}{y_2 - y_1}f(R_{2})
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
      <td>4D tensor of the input image, corresponding to x in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>Height and width of the output image.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td>If true, align the centers of the four corner pixels of the input and output images; default value: false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>half_pixel_centers</td>
      <td>Optional attr</td>
      <td>If true, sample based on the half-pixel center coordinates; this attribute cannot be true together with align_corners, default value: false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Optional attr</td>
      <td>Specifies the data type of the output y, supporting FLOAT32, FLOAT16, and BFLOAT16; default value: FLOAT32.</td>
      <td>TYPE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Optional attr</td>
      <td>Specifies the scaling coefficients in the H and W directions, the default value is [0.0, 0.0], indicating that the scaling ratios are calculated based on size.</td>
      <td>LIST_FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Image resized by bilinear interpolation, corresponding to y in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- `align_corners` and `half_pixel_centers` cannot be true at the same time.
- When `scales` is set, provide the two scaling coefficients for H and W; the default value is `[0.0, 0.0]`.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_resize](examples/test_aclnn_resize.cpp) | Call the ResizeBilinearV2 operator through the [aclnnResize](docs/aclnnResize_en.md) interface. |
| Graph mode | -  | Call the ResizeBilinearV2 operator by building a graph through [operator IR](op_graph/resize_bilinear_v2_proto.h). |
