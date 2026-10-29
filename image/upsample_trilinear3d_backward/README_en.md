# UpsampleTrilinear3dBackward

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

- Operator function: backward computation of [ResizeUpsampleTrilinear](../resize_upsample_trilinear/README_en.md).

- Calculation formula:
  - Core algorithm logic of the forward process:
    1. Map each point of the target image back to the original image to obtain a floating-point coordinate.
    2. Based on this floating-point coordinate, calculate the adjacent points in the original image.
    3. Calculate the weights from the adjacent points to the corresponding target point respectively, and multiply and accumulate the values based on the weights to obtain the target point value.

  - Detailed calculation logic:
    The scaling mode is classified into corner alignment and edge alignment. Corner alignment (alignCorners is true) indicates alignment by the center of the top-left pixel of the original image. Edge alignment (alignCorners is false) indicates alignment by the top-left vertex and two edges of the original image. Differences exist in the calculation of the scaling coefficients and coordinate positions. For a 3D interpolation point $(N, C, D, H, W)$, the following formulas apply:

    $$
    scale\_d =\begin{cases}
    (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scales\_d & alignCorners=false\&scales\_d>0\\
    inputSize[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_h =\begin{cases}
    (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
    1 / scales\_h & alignCorners=false\&scales\_h>0\\
    inputSize[3] / outputSize[1] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_w =\begin{cases}
    (inputSize[4]-1) / (outputSize[2]-1) & alignCorners=true \\
    1 / scales\_w & alignCorners=false\&scales\_w>0\\
    inputSize[4] / outputSize[2] & alignCorners=false
    \end{cases}
    $$

    Therefore, for a point p(x,y,z) in a direction of the output, the point mapped back to the original image is denoted as q(x',y',z'). The relationship is as follows:

    $$
    x' =\begin{cases}
    x * scale\_d & alignCorners=true \\
    MAX(0,{(x+0.5)*scale\_d-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    y' =\begin{cases}
    y * scale\_h & alignCorners=true \\
    MAX(0,{(y+0.5)*scale\_h-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    z' =\begin{cases}
    z * scale\_w & alignCorners=true \\
    MAX(0,{(z+0.5)*scale\_w-0.5}) & alignCorners=false
    \end{cases}
    $$

    - Define:

      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$

      $$
      y_{0} =int(y'),y_{1} =int(y')+1, lambdb_{0} = y_{1}-y', lambdb_{1} =   1-lambdb_{0}
      $$

      $$
      z_{0} =int(z'),z_{1} =int(z')+1, lambdc_{0} = z_{1}-z', lambdc_{1} =   1-lambdc_{0}
      $$

    - The following formula applies:

      $$
      {V(p_{x, y, z})} = {V(p_{x0, y0, z0})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x0, y0, z1})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x0, y1, z0})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x0, y1, z1})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{1}} + {V(p_{x1, y0, z0})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x1, y0, z1})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x1, y1, z0})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x1, y1, z1})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{1}}
      $$

    - Assume that the output image out $(x, y, z)$ of the forward interpolation is affected by the original image input $(x_i, y_j, z_k)$. Then:

      $$
      gradInput(x_i,y_j,z_k) += gradOutput(x,y,z) * lambda(x_i,y_j,z_k)* lambdb(x_i,y_j,z_k)* lambdc(x_i,y_j,z_k)
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
      <td>FLOAT32, FLOAT16, DOUBLE</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>input_size</td>
      <td>Attr</td>
      <td>Indicates the spatial sizes of output `gradInput` in the N, C, D, H, and W dimensions respectively, corresponding to `inputSize` in the formula. The size is 5, and all elements are greater than 0.</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>Indicates the spatial sizes of input `grad_output` in the D, H, and W dimensions, corresponding to `outputSize` in the formula. The size is 3, and all elements are greater than 0.</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_d</td>
      <td>Optional attr</td>
      <td>Specifies the scaling coefficient of the depth dimension, corresponding to `scales_d` in the formula.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_h</td>
      <td>Optional attr</td>
      <td>Specifies the scaling coefficient of the height dimension, corresponding to `scales_h` in the formula.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_w</td>
      <td>Optional attr</td>
      <td>Specifies the scaling coefficient of the width dimension, corresponding to `scales_w` in the formula.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td><ul><li>Determines whether to align the corner pixels, corresponding to `alignCorners` in the formula. If align_corners is true, the corner pixels of the input and output tensors are aligned. Otherwise, the corner pixels are not aligned.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor of the backward computation, corresponding to `gradInput` in the formula. The data type and data format are the same as the data type and data format of the input parameter `grad_output`; the shape is specified by `input_size`.</td>
      <td>FLOAT32, FLOAT16, DOUBLE</td>
      <td>NCDHW</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_trilinear3d_backward](examples/test_aclnn_upsample_trilinear3d_backward.cpp) | Call the UpsampleTrilinear3dBackward operator through the [aclnnUpsampleTrilinear3dBackward](docs/aclnnUpsampleTrilinear3dBackward_en.md) interface. |
