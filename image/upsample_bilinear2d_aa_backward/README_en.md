# UpsampleBilinear2dAABackward

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

## Function Description

- Operator function: backward propagation of [UpsampleBilinear2dAA](../upsample_bilinear2d_aa/README_en.md).

- Calculation formula: for a 2D interpolation point $(N, C, H, W)$, the interpolation $I(N, C, H, W)$ can be expressed as:

  $$
  {I(N, C, H, W)} = \sum_{i=0}^{kW}\sum_{j=0}^{kH}{w(i) * w(j)} * {f(h_i, w_j)}/\sum_{i=0}^{kW}w(i)/\sum_{j=0}^{kH}w(j)
  $$

  $$
  scaleH =\begin{cases}
  (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  inputSize[2] / outputSize[0] & otherwise
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  inputSize[3] / outputSize[1] & otherwise
  \end{cases}
  $$

  In the formula:
  - If alignCorners is true, the corner pixels of the input and output tensors are aligned; if alignCorners is false, the edge pixels of the input and output tensors are aligned.
  - $kW$ and $kH$ denote the numbers of points that affect the size of the interpolation point in the W and H directions respectively
  - If $scaleH >= 1$, then $kH = floor(scaleH) * 2 + 1$; otherwise, $kH = 3$
  - If $scaleW >= 1$, then $kW = floor(scaleW) * 2 + 1$; otherwise, $kW = 3$
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

  - Assume that the output image out $(h, w)$ of the forward interpolation is affected by the original image input $(h_i, w_j)$. Then:

    $$
    gradInput(h_i,w_j) += gradOutput(h,w) * w(i) * w(j)
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
      <td>grad_output</td>
      <td>Input</td>
      <td>Represents the gradient tensor for the backward computation, corresponding to `gradOutput` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the input spatial size, corresponding to `outputSize` in the formula. The size is 2, and all elements are greater than 0. It indicates the spatial size of `grad_output` in the H and W dimensions.</li><li>The default value is empty.</li></ul></td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_size</td>
      <td>Attr</td>
      <td>Specifies the output spatial size, corresponding to `inputSize` in the formula. The size is 4, and all elements are greater than 0. It indicates the spatial sizes of output `grad_input` in the N, C, H, and W dimensions respectively.</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td><ul><li>Determines whether to align the corner pixels, corresponding to `alignCorners` in the formula. If align_corners is true, the corner pixels of the input and output tensors are aligned. Otherwise, the top-left vertex and two edges of the input and output tensors are aligned.</li><li>The default value is false.</li></ul></td>
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
      <td>grad_input</td>
      <td>Output</td>
      <td>Represents the output tensor of the backward computation, corresponding to `gradInput` in the formula. The data type and data format are the same as the data type and data format of the input parameter `grad_output`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_bilinear2d_aa_backward](examples/test_aclnn_upsample_bilinear2d_aa_backward.cpp) | Call the UpsampleBilinear2dAABackward operator through the [aclnnUpsampleBilinear2dAABackward](docs/aclnnUpsampleBilinear2dAABackward_en.md) interface. |
