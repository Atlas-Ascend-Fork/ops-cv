# UpsampleLinear1d

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: applies the linear interpolation algorithm to perform upsampling on the input signal composed of multiple input channels. If the input shape is (N, C, L), the output shape is (N, C, outputSize).

- Calculation formula:
  - Core algorithm logic:
    1. Map each point of the target image back to the original image to obtain a floating-point coordinate.
    2. Based on this floating-point coordinate, calculate the adjacent points in the original image.
    3. Calculate the weights from the adjacent points to the corresponding target point respectively, and multiply and accumulate the values based on the weights to obtain the target point value.
  - Detailed calculation logic:
    The scaling mode is classified into corner alignment and edge alignment. Corner alignment (alignCorners is true) indicates alignment by the center of the top-left pixel of the original image. Edge alignment (alignCorners is false) indicates alignment by the top-left vertex and two edges of the original image. Differences exist in the calculation of the scaling coefficients and coordinate positions. The following formula applies:

    $$
    scale =\begin{cases}
    (self.dim[2]-1) / (outputSize[0]-1) & alignCorners=true\&outputSize[0]>1 \\
    0 & alignCorners=true\&outputSize[0]=1 \\
    1 / scales & alignCorners=false\&scales>0\\
    self.dim[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    Therefore, for a point p(x) in a direction of the output, the point mapped back to the original image is denoted as q(x'). The relationship is as follows:

    $$
    x' =\begin{cases}
    x * scale & alignCorners=true \\
    MAX(0,{(x+0.5)*scale-0.5}) & alignCorners=false
    \end{cases}
    $$

    - Define:

      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$

    - The following formula applies:

      $$
      {V(p_{x})} = {V(p_{x0})} * {lambda_{0}}  + {V(p_{x1})} * {lambda_{1}}
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
      <td>Represents the input tensor for upsampling, corresponding to `self` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>Specifies the output spatial size, corresponding to `outputSize` in the formula. The size is 1, and the value is greater than 0. It indicates the spatial size of `y` in the H and W dimensions.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td><ul><li>Determines whether to align the corner pixels, corresponding to `alignCorners` in the formula. If it is set to `true`, the input and output tensors are aligned by the centers of their corner pixels, preserving the values at the corner pixels. If it is set to `false`, the input and output tensors are aligned by the corners of their corner pixels, and the interpolation uses the edge values as padding for values outside the boundary.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the L dimension of output `y`, corresponding to `scales` in the formula.</li><li>There is no default value.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor after sampling, corresponding to `output` in the formula description. The data type and data format are the same as the data type and data format of the input parameter `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_linear1d](examples/test_aclnn_upsample_linear1d.cpp) | Call the UpsampleLinear1d operator through the [aclnnUpsampleLinear1d](docs/aclnnUpsampleLinear1d_en.md) interface. |
