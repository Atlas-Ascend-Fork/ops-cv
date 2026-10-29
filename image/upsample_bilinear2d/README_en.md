# UpsampleBilinear2d

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
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: applies 2D bilinear upsampling to the input signal composed of multiple input channels.
  - For the input shape: if the input shape is (N, C, H, W), the output shape is (N, C, outputSize[0], outputSize[1]).
  - For the center alignment option: pixels are generally regarded as a grid. When alignCorners = True, pixels are regarded as the points at the top-left corner of the grid, the pixels at the output corners are aligned with the centers of the corner pixels of the original image, and the points in the same direction are equidistant. When alignCorners = False, pixels are regarded as the points on the crossing lines of the grid, the pixels at the output corners are still the corner pixels of the original image, but the points in the same direction are not equidistant.
- Calculation formula:
  - Core algorithm logic:
    1. Map each point of the target image back to the original image to obtain a coordinate with a decimal point.
    2. Based on this floating-point coordinate, calculate the adjacent points before and after in the original image.
    3. Calculate the weights of the adjacent points to the corresponding target point respectively, and obtain the target point value by multiplying and accumulating according to the weights.
  - Specific calculation logic:
    The scaling methods are divided into corner alignment and edge alignment. Corner alignment (alignCorners is true) means alignment with the center of the top-left pixel of the original image, and edge alignment (alignCorners is false) means alignment with the top-left vertex and two edges of the original image. Differences exist when calculating the scaling coefficients and coordinate positions. For a 2D interpolation point $(N, C, H, W)$, the following formulas apply:

    $$
    scale\_h =\begin{cases}
    (self.dim[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scales\_h & alignCorners=false\&scales\_h>0\\
    self.dim[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_w =\begin{cases}
    (self.dim[3]-1) / (outputSize[1]-1) & alignCorners=true \\
    1 / scales\_w & alignCorners=false\&scales\_w>0\\
    self.dim[3] / outputSize[1] & alignCorners=false
    \end{cases}
    $$

    Therefore, for a point p(x,y) in a certain direction of the output, the point mapped back to the original image is denoted as q(x',y'), and the relationship is as follows:

    $$
    x' =\begin{cases}
    x * scale\_h & alignCorners=true \\
    MAX(0,{(x+0.5)*scale\_h-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    y' =\begin{cases}
    y * scale\_w & alignCorners=true \\
    MAX(0,{(y+0.5)*scale\_w-0.5}) & alignCorners=false
    \end{cases}
    $$

    - Denote:

      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$

      $$
      y_{0} =int(y'),y_{1} =int(y')+1, lambdb_{0} = y_{1}-y', lambdb_{1} =   1-lambdb_{0}
      $$

    - The following formula applies:

      $$
      {V(p_{x, y})} = {V(p_{x0, y0})} * {lambda_{0}} * {lambdb_{0}} + {V(p_{x0, y1})} * {lambda_{0}} * {lambdb_{1}} + {V(p_{x1, y0})} * {lambda_{1}} * {lambdb_{0}} + {V(p_{x1, y1})} * {lambda_{1}} * {lambdb_{1}}
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
      <td>Represents the input tensor for upsampling, corresponding to `self` in the formula description.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>Specifies the output spatial size, corresponding to `outputSize` in the formula. The size is 2, indicating the spatial size of `y` in the H and W dimensions.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td><ul><li>Determines whether to align the corner pixels, corresponding to `alignCorners` in the formula. If align_corners is true, the corner pixels of the input and output tensors are aligned; otherwise, the top-left vertex and two edges of the input and output tensors are aligned.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multipliers of the height and width dimensions of the spatial size, corresponding to the array [scales_h, scales_w] composed of `scales_h` and `scales_w` in the formula.</li><li>The default value is empty.</li></ul></td>
      <td>LISTFLOAT</td>
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

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_bilinear2d](examples/test_aclnn_upsample_bilinear2d.cpp) | Call the UpsampleBilinear2d operator through the [aclnnUpsampleBilinear2d](docs/aclnnUpsampleBilinear2d_en.md) interface. |
