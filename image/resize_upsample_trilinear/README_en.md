# ResizeUpsampleTrilinear

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: applies the trilinear interpolation algorithm to the input signal composed of multiple input channels for upsampling.
- Calculation formula:
  - Core algorithm logic:
    1. Map each point of the target image back to the original image to obtain a coordinate with a decimal point.
    2. Based on this floating-point coordinate, calculate the adjacent points before and after in the original image.
    3. Calculate the weights of the adjacent points to the corresponding target point respectively, and obtain the target point value by multiplying and accumulating according to the weights.
  - Specific calculation logic:
    The scaling methods are divided into corner alignment and edge alignment. Corner alignment (alignCorners is true) means alignment with the center of the top-left pixel of the original image, and edge alignment (alignCorners is false) means alignment with the top-left vertex and two edges of the original image. Differences exist when calculating the scaling coefficients and coordinate positions. For a 3D interpolation point $(N, C, D, H, W)$, the following formulas apply:

    $$
    scale\_d =\begin{cases}
    1 & outputSize[0] = self.dim[2] \\
    (self.dim[2]-1) / (outputSize[0]-1) & alignCorners=true\&outputSize[0]>1 \\
    0 & alignCorners=true\&outputSize[0]=1 \\
    1 / scales\_d & alignCorners=false\&scales\_d>0\\
    self.dim[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_h =\begin{cases}
    1 & outputSize[1] = self.dim[3] \\
    (self.dim[3]-1) / (outputSize[1]-1) & alignCorners=true\&outputSize[1]>1 \\
    0 & alignCorners=true\&outputSize[1]=1 \\
    1 / scales\_h & alignCorners=false\&scales\_h>0\\
    self.dim[3] / outputSize[1] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_w =\begin{cases}
    1 & outputSize[2] = self.dim[4] \\
    (self.dim[4]-1) / (outputSize[2]-1) & alignCorners=true\&outputSize[2]>1 \\
    0 & alignCorners=true\&outputSize[2]=1 \\
    1 / scales\_w & alignCorners=false\&scales\_w>0\\
    self.dim[4] / outputSize[2] & alignCorners=false
    \end{cases}
    $$

    Therefore, for a point p(x,y,z) in a certain direction of the output, the point mapped back to the original image is denoted as q(x',y',z'), and the relationship is as follows:

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

    - Denote:

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
      <td>Optional attr</td>
      <td><ul><li>Represents the spatial size of `output` in the D, H, and W dimensions, corresponding to `outputSize` in the formula. The size is 3, and all elements are greater than zero.</li><li>The default value is empty.</li></ul></td>
      <td>LISTINT</td>
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
      <td>scales_d</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the depth dimension of the spatial size, corresponding to `scales_d` in the formula.</li><li>The default value is empty.</li></ul></td>
      <td>FLOAT32</td>
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
      <td>output</td>
      <td>Output</td>
      <td>Represents the output tensor after sampling, corresponding to `output` in the formula description. The data type and data format are the same as the data type and data format of the input parameter `input`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

<term>Atlas inference products</term>:

- The data types of the input and output do not support BFLOAT16.
- The input parameter `input` does not support the input of inf or -inf.

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_trilinear3d](examples/test_aclnn_upsample_trilinear3d.cpp) | Call the ResizeUpsampleTrilinear operator through the [aclnnUpsampleTrilinear3d](docs/aclnnUpsampleTrilinear3d_en.md) interface. |
