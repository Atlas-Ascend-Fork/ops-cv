# UpsampleBicubic2dAAGrad

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

- Operator function: backpropagation of [UpsampleBicubic2dAA](../upsample_bicubic2d_aa/README_en.md). If the shape of the input tensor is (N, C, H, W), the shape of the output tensor is (N, C, inputSize[2], inputSize[3]).

- Calculation formula: for a 2D interpolation point $(N, C, h, w)$, the interpolation $gradInput(N, C, h, w)$ can be expressed as:

  $$
  {gradInput(N, C, h, w)}=\sum_{r: h\in R_H(r)}\sum_{s: w\in R_W(s)}
  {gradOutput(N, C, r, s)} W_H(h|r) W_W(w|s)
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
  - $r$ and $s$ are the H and W coordinates of gradOutput; $h$ and $w$ are the H and W coordinates of gradInput. $gradInput_H$ and $gradInput_W$ denote the H and W lengths of gradInput respectively.
  - For the direction $D\in\{H,W\}$ and the gradOutput coordinate $q$, the center coordinate is:

    $$
    c_D(q)=\begin{cases}
    (q+0.5)scale_D & scale_D>0 \\
    0.5 & otherwise
    \end{cases}
    $$

  - The valid gradInput coordinate range is calculated based on support and clipped to the gradInput boundary:

    $$
    R_D(q)=\left[\max(0,\lfloor c_D(q)-support_D+0.5\rfloor),\,
    \min(gradInput_D,\lfloor c_D(q)+support_D+0.5\rfloor)\right)
    $$

    In the formula, $support_D=2scale_D$ (when $scale_D\geq1$); otherwise, $support_D=2$. Therefore, when the scale is not less than 1, the window participating in the calculation expands with the scale.

  - The weight distance scaling coefficient is $invScale_D=1/scale_D$ (when $scale_D\geq1$); otherwise, $invScale_D=1$. The normalized weight of the position $x\in R_D(q)$ in the direction $D$ is:

    $$
    \widetilde W_D(x|q)=CubicFilterAA((x-c_D(q)+0.5)invScale_D),\qquad
    W_D(x|q)=\frac{\widetilde W_D(x|q)}{\sum_{t\in R_D(q)}\widetilde W_D(t|q)}
    $$

  - $CubicFilterAA(x)$ is the bicubic kernel, defined as:

    $$
    CubicFilterAA(x)=\begin{cases}
    (a+2)|x|^3-(a+3)|x|^2+1 & |x|\leq1 \\
    a|x|^3-5a|x|^2+8a|x|-4a & 1<|x|<2 \\
    0 & otherwise
    \end{cases}
    $$

    In the antialiasing scenario, $a=-0.5$.

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
      <td>Represents the gradient tensor of the backward calculation, corresponding to `gradOutput` in the formula description. The data type is the same as the data type of the output `grad_input`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>Represents the spatial size of the input `grad_output` in the H and W dimensions, corresponding to `outputSize` in the formula. The size is 2, and all elements are greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_size</td>
      <td>Attr</td>
      <td>Represents the spatial size of the output `grad_input` in the N, C, H, and W dimensions respectively, corresponding to `inputSize` in the formula. The size is 4, and all elements are greater than zero.</td>
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
      <td><ul><li>Represents the multiplier of the height dimension of the output `grad_input`, corresponding to `scalesH` in the formula.</li><li>The default value is empty.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>Optional attr</td>
      <td><ul><li>Represents the multiplier of the width dimension of the output `grad_input`, corresponding to `scalesW` in the formula.</li><li>The default value is empty.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>Output</td>
      <td>Represents the output tensor after sampling, corresponding to `gradInput` in the formula. The data type is the same as the data type of the input `grad_output`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_bicubic2d_aa_grad](examples/test_aclnn_upsample_bicubic2d_aa_grad.cpp) | Call the UpsampleBicubic2dAAGrad operator through the [aclnnUpsampleBicubic2dAAGrad](docs/aclnnUpsampleBicubic2dAAGrad_en.md) interface. |
