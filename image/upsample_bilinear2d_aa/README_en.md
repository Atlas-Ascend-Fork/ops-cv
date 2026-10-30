# UpsampleBilinear2dAA

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: applies 2D bilinear antialiasing sampling to the input signal composed of multiple input channels.
- Calculation formula: for a 2D interpolation point $(N, C, H, W)$, the interpolation $I(N, C, H, W)$ can be expressed as:

  $$
  {I(N, C, H, W)} = \sum_{i=0}^{kH}\sum_{j=0}^{kW}{w(i) * w(j)} * {f(h_i, w_j)}/\sum_{i=0}^{kH}w(i)/\sum_{j=0}^{kW}w(j)
  $$

  $$
  scaleH =\begin{cases}
  (input.dim(2)-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  input.dim(2) / outputSize[0] & otherwise
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  (input.dim(3)-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  input.dim(3) / outputSize[1] & otherwise
  \end{cases}
  $$

  In the formula:
  - If alignCorners is true, the corner pixels of the input and output tensors are aligned; if alignCorners is false, the edge pixels of the input and output tensors are aligned.
  - $kW$ and $kH$ denote the numbers of points that affect the size of the interpolation point in the W and H directions respectively
  - If $scaleH >= 1$, then $kH = floor(scaleH) * 2 + 1$; otherwise, $kH = 3$
  - If $scaleW >= 1$, then $kW = floor(scaleW) * 2 + 1$; otherwise, $kW = 3$
  - $f(h_i, w_j)$ is the pixel value of the original image at $(h_i, w_j)$
  - $w(i)$ and $w(j)$ are the weights in the W and H directions of the bilinear antialiasing interpolation, and the calculation formulas are as follows:

    $$
      w(i) = \begin{cases}
      1 - |h_i - h| & |h_i -h| < 1 \\
      0 & otherwise
      \end{cases}
    $$

    $$
      w(j) = \begin{cases}
      1 - |w_j - w| & |w_j -w| < 1 \\
      0 & otherwise
      \end{cases}
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
      <td>input</td>
      <td>Input</td>
      <td>Represents the input tensor for sampling, corresponding to `input` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>Specifies the output spatial size, corresponding to `outputSize` in the formula. The size is 2, and all elements are greater than 0. It indicates the spatial size of `output` in the H and W dimensions.</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td><ul><li>Determines whether to align the corner pixels, corresponding to `alignCorners` in the formula. If it is set to `true`, the input and output tensors are aligned by the centers of their corner pixels, preserving the values at the corner pixels. If it is set to `false`, the input and output tensors are aligned by the corners of their corner pixels, and the values outside the boundary are padded with the edge values.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the height dimension of the spatial size, corresponding to `scalesH` in the formula.</li><li>The default value is empty.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the width dimension of the spatial size, corresponding to `scalesW` in the formula.</li><li>The default value is empty.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Represents the output tensor after sampling, corresponding to `I` in the formula. The data type and data format are the same as the data type and data format of the input parameter `input`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_bilinear2d_aa](examples/test_aclnn_upsample_bilinear2d_aa.cpp) | Call the UpsampleBilinear2dAA operator through the [aclnnUpsampleBilinear2dAA](docs/aclnnUpsampleBilinear2dAA_en.md) interface. |
