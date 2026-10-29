# UpsampleBicubic2d

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: applies 2D bicubic upsampling to the input signal composed of multiple input channels. If the shape of the input tensor x is (N, C, H, W), the shape of the output tensor out is (N, C, outputSize[0], outputSize[1]).
- Calculation formula: for a 2D interpolation point $(N, C, h, w)$, the interpolation $out(N, C, h, w)$ can be expressed as:

  $$
  {out(N, C, h, w)}=\sum_{i=0}^{3}\sum_{j=0}^{3}{W(i, j)}*{f(h_i, w_j)}
  $$

  $$
  scaleH =\begin{cases}
  (self.dim(2)-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  self.dim(2) / outputSize[0] & otherwise
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  (self.dim(3)-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  self.dim(3) / outputSize[1] & otherwise
  \end{cases}
  $$

  In the formula:
  - If alignCorners is true, the corner pixels of the input and output tensors are aligned; if alignCorners is false, the edge pixels of the input and output tensors are aligned.
  - i and j are the index variables of $W(i, j)$.
  - $f(h_i, w_j)$ is the pixel value of the original image at $(h_i, w_j)$.
  - $W(i, j)$ is the weight of the bicubic antialiasing interpolation, defined as:

    $$
    W(d) =\begin{cases}
    (a+2)|d|^3-(a+3)|d|^2+1 & |d|\leq1 \\
    a|d|^3-5a|d|^2+8a|d|-4a & 1<|d|<2 \\
    0 & otherwise
    \end{cases}
    $$

    In the formula:
    - $a=-0.75$
    - $d = |(h, w) - (h_i, w_j)|$

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
      <td>Represents the input tensor for upsampling, corresponding to `self` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>Represents the spatial size of `output` in the H and W dimensions, corresponding to `outputSize` in the formula. The size is 2, and all elements are greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td><ul><li>Determines whether to align the corner pixels, corresponding to `alignCorners` in the formula. If align_corners is true, the corner pixels of the input and output tensors are aligned; otherwise, they are not aligned.</li><li>The default value is false.</li></ul></td>
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
      <td>Represents the output tensor after sampling, corresponding to `out` in the formula. The data type and data format must be the same as the data type and data format of `self`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas inference products</term>: the data types of the input and output do not support BFLOAT16.
  - <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_bicubic2d](examples/test_aclnn_upsample_bicubic2d.cpp) | Call the UpsampleBicubic2d operator through the [aclnnUpsampleBicubic2d](docs/aclnnUpsampleBicubic2d_en.md) interface. |
