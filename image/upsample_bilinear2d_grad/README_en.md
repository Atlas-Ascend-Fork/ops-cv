# UpsampleBilinear2dGrad

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

## Function Description

- Operator function: applies backward propagation of 2D bilinear upsampling to the input signal composed of multiple input channels.
- Calculation formula:
  - Core algorithm logic of the forward process:
    1. Map each point of the target image back to the original image to obtain a floating-point coordinate.
    2. Based on this floating-point coordinate, calculate the adjacent points in the original image.
    3. Calculate the weights from the adjacent points to the corresponding target point respectively, and multiply and accumulate the values based on the weights to obtain the target point value.
  - Detailed calculation logic:
    The scaling mode is classified into corner alignment and edge alignment. Corner alignment (alignCorners is true) indicates alignment by the center of the top-left pixel of the original image. Edge alignment (alignCorners is false) indicates alignment by the top-left vertex and two edges of the original image. Differences exist in the calculation of the scaling coefficients and coordinate positions. For a 2D interpolation point $(N, C, H, W)$, the following formulas apply:

    $$
    scale\_h =\begin{cases}
    (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scales\_h & alignCorners=false\&scales\_h>0\\
    inputSize[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_w =\begin{cases}
    (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
    1 / scales\_w & alignCorners=false\&scales\_w>0\\
    inputSize[3] / outputSize[1] & alignCorners=false
    \end{cases}
    $$

    Therefore, consider a point p(x,y) in a the output image, where x and y represent the horizontal (W direction) and vertical (H direction) coordinates, respectivvely. Its corresponding point mapped back to the original image is denoted as q(x',y'), defined by the following relationship:

    $$
    x' =\begin{cases}
    x * scale\_w & alignCorners=true \\
    MAX(0,{(x+0.5)*scale\_w-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    y' =\begin{cases}
    y * scale\_h & alignCorners=true \\
    MAX(0,{(y+0.5)*scale\_h-0.5}) & alignCorners=false
    \end{cases}
    $$

    - Define:

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

    - Assume that the output image out $(x, y)$ of the forward interpolation is affected by the original image input $(x_i, y_j)$. Then:

      $$
      gradInput(x_i,y_j) += gradOutput(x,y) * lambda(x_i,y_j)* lambdb(x_i,y_j)
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
      <td>Attr</td>
      <td>Specifies the input spatial size, corresponding to `outputSize` in the formula. The size is 2, and all elements are greater than 0. It indicates the spatial size of `grad_output` in the H and W dimensions.</td>
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
      <td><ul><li>Determines whether to align the corner pixels, corresponding to `alignCorners` in the formula. If it is set to `true`, the input and output tensors are aligned by the centers of their corner pixels, preserving the values at the corner pixels. If it is set to `false`, the input and output tensors are aligned by the corners of their corner pixels, and the interpolation uses the edge values as padding for values outside the boundary.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the height dimension of the spatial size, corresponding to `scales_h` in the formula.</li><li>The default value is empty.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the width dimension of the spatial size, corresponding to `scales_w` in the formula.</li><li>The default value is empty.</li></ul></td>
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

For <term>Atlas inference products</term> and <term>Atlas 200I/500 A2 inference products</term>, the BFLOAT16 data type of the input and output is not supported.

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_bilinear2d_backward_v2](examples/test_aclnn_upsample_bilinear2d_backward_v2.cpp) | Call the UpsampleBilinear2dGrad operator through the [aclnnUpsampleBilinear2dBackwardV2](docs/aclnnUpsampleBilinear2dBackwardV2_en.md) interface. |
| aclnn interface  | [test_aclnn_upsample_linear1d_backward](examples/test_aclnn_upsample_linear1d_backward.cpp) | Call the UpsampleBilinear2dGrad operator through the [aclnnUpsampleLinear1dBackward](docs/aclnnUpsampleLinear1dBackward_en.md) interface. |
