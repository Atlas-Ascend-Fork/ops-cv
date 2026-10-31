# GridSample

English | [简体中文](./README.md)

## Product Support

|Product | Supported |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | √ |

## Function Description

- Operator function: This operator takes an input tensor and a corresponding grid. According to the coordinate information provided at each position in grid, the operator fills the pixel values at the corresponding positions in input into the positions specified by the grid, and obtains the final output.
- Computation formula:

  - Computation process:
    1. Computes the coordinates mapped to input based on the (x, y) or (x, y, z) values stored in grid. These coordinates depend on align_corners and padding_mode.
    2. Computes the output values by selecting different interpolation modes (bilinear, nearest, or bicubic) based on the input interpolation_mode.

  - where:
    - 3D scenario:

      The sizes of the input input, the grid, and the output output are as follows:

      $$
      input: (N, C, D_{in}, H_{in}, W_{in})\\
      grid: (N, D_{out}, H_{out}, W_{out}, 3)\\
      output: (N, C, D_{out}, H_{out}, W_{out})
      $$

      Here N is the same in input, grid, and output. C is the same in input and output. $D_{out}$, $H_{out}$, and $W_{out}$ are the same in grid and output. The size of the last dimension of grid is 3, indicating that the input pixel position information is (x, y, z), and the value ranges of x, y, and z are normalized to [-1, 1].

      - Denormalization computation formula:
        - align_corners=true indicates that the feature values are located at the pixel centers.

         $$
         x = (grid_x + 1) / 2 * (W_{in} - 1)
         $$

         $$
         y = (grid_y +1) / 2 * (H_{in} - 1)
         $$

         $$
         z = (grid_z +1) / 2 * (D_{in} - 1)
         $$

        - align_corners=false indicates that the feature values are located at the pixel corners.

         $$
         x = ((grid_x +1) * W_{in} - 1) / 2
         $$

         $$
         y = ((grid_y +1) * H_{in} - 1) / 2
         $$

         $$
         z = ((grid_z +1) * D_{in} - 1) / 2
         $$

      - Out-of-range coordinates are processed differently according to paddingMode:
        - paddingMode=0 indicates that out-of-bounds positions are padded with 0.
        - paddingMode=1 indicates that out-of-bounds positions are padded with the boundary values.
        - paddingMode=2 indicates that out-of-bounds positions are padded with the symmetric values of the boundary values.
      - When sampling input, different processing is performed according to interpolationMode:
        - interpolationMode="bilinear" indicates that the weighted average of the eight coordinates around (x, y, z) in input is taken.

          $$
          {output(grad_z, grad_y, grad_x)} = \sum_{i=0}^{1}\sum_{j=0}^{1}\sum_{k=0}^{1}{w(i, j, k)} * {f(x_i, y_j, z_k)}
          $$

          where:
          - $f(x, y, z)$ is the pixel value of the original image at $(x, y, z)$, and $(x, y, z)$ refers to the eight coordinates around (x, y, z).
          - $w(i, j, k)$ is the weight of the 8 surrounding points in bilinear interpolation. The formulas are:

            $$
              w(i) = \begin{cases}
              1 - |x_i - x| & |x_i - x| < 1 \\
              0 & otherwise
              \end{cases}
            $$

            $$
              w(j) = \begin{cases}
              1 - |y_j - y| & |y_j - y| < 1 \\
              0 & otherwise
              \end{cases}
            $$

            $$
              w(k) = \begin{cases}
              1 - |z_k - z| & |z_k - z| < 1 \\
              0 & otherwise
              \end{cases}
            $$

            $$
            w(i, j, k)= w(i) * w(j) * w(k)
            $$

        - interpolationMode="nearest" indicates that the coordinate value in input closest to (x, y, z) is taken.

          $$
          output(grad_z, grad_y, grad_x) = input(z_{in}, y_{in}, x_{in})
          $$

          where:

          $$
          x_{in} = min(round(x),  W_{in} - 1)\\
          x_{in} = max(x_{in},  0)
          $$

          $$
          y_{in} = min(round(y),  H_{in} - 1)\\
          y_{in} = max(y_{in},  0)
          $$

          $$
          z_{in} = min(round(z),  D_{in} - 1)\\
          z_{in} = max(z_{in},  0)
          $$

    - 2D scenario:

      The sizes of the input input, the grid, and the output output are as follows:

      $$
      input: (N, C, H_{in}, W_{in})\\
      grid: (N, H_{out}, W_{out}, 2)\\
      output: (N, C, H_{out}, W_{out})
      $$

      Here N is the same in input, grid, and output. C is the same in input and output. $H_{out}$ and $W_{out}$ are the same in grid and output. The size of the last dimension of grid is 2, indicating that the input pixel position information is (x, y), and the value ranges of x and y are normalized to [-1, 1]. (-1, -1) indicates the top-left coordinate, and (1, 1) indicates the bottom-right coordinate.

      - Denormalization computation formula:
        - align_corners=true indicates that the feature values are located at the pixel centers.

          $$
          x = (grid_x + 1) / 2 * (W_{in} - 1)
          $$

          $$
          y = (grid_y +1) / 2 * (H_{in} - 1)
          $$

        - align_corners=false indicates that the feature values are located at the pixel corners.

          $$
          x = ((grid_x +1) * W_{in} - 1) / 2
          $$

          $$
          y = ((grid_y +1) * H_{in} - 1) / 2
          $$

      - Out-of-range coordinates are processed differently according to paddingMode:

        - paddingMode=0 indicates that out-of-bounds positions are padded with 0.
        - paddingMode=1 indicates that out-of-bounds positions are padded with the boundary values.
        - paddingMode=2 indicates that out-of-bounds positions are padded with the symmetric values of the boundary values.

      - When sampling input, different processing is performed according to interpolationMode:

        - interpolationMode=0 indicates that the weighted average of the four coordinates around (x, y) is taken.

          $$
          output(grad_y, grad_x) = \sum_{i=0}^{1}\sum_{j=0}^{1}{w(i, j)} * {f(x_i, y_j)}
          $$

          where:
          - $f(x_i, y_j)$ is the pixel value of the original image at $(x_i, y_j)$, and $(x_i, y_j)$ refers to the four coordinates around (x, y).
          - $w(i, j)$ is the weight of the 4 surrounding points in bilinear interpolation. The formulas are:

            $$
              w(i) = \begin{cases}
              1 - |x_i - x| & |x_i - x| < 1 \\
              0 & otherwise
              \end{cases}
            $$

            $$
              w(j) = \begin{cases}
              1 - |y_j - y| & |y_j - y| < 1 \\
              0 & otherwise
              \end{cases}
            $$

            $$
            w(i, j)= w(i) * w(j)
            $$

        - interpolationMode=1 indicates that the coordinate value in input closest to (x, y) is taken.

          $$
          output(grad_y, grad_x) = input(y_{in}, x_{in})
          $$

          where:

          $$
          x_{in} = min(round(x),  W_{in} - 1)\\
          x_{in} = max(x_{in},  0)
          $$

          $$
          y_{in} = min(round(y),  H_{in} - 1)\\
          y_{in} = max(y_{in},  0)
          $$

        - interpolationMode=2 indicates that the weighted average of the sixteen coordinates around (x, y) is taken.

          $$
          {output(grad_y, grad_x)}=\sum_{i=0}^{3}\sum_{j=0}^{3}{W(i, j)}*{f(x_i, y_j)}
          $$

          where:
          - $f(x_i, y_j)$ is the pixel value of the original image at $(x, y)$, and $(x_i, y_j)$ refers to the sixteen coordinates around (x, y).
          - $W(i, j)$ is the weight of bicubic interpolation, defined as:

            $$
            W(d) =\begin{cases}
            (a+2)|d|^3-(a+3)|d|^2+1 & |d|\leq1 \\
            a|d|^3-5a|d|^2+8a|d|-4a & 1<|d|<2 \\
            0 & otherwise
            \end{cases}
            $$

            where:
            - $a=-0.75$
            - $d = |(x_i, y_j) - (x, y)|$

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
      <td>Input tensor for the interpolation computation. It corresponds to `input` described in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grid</td>
      <td>Input</td>
      <td>Grid for sampling. It corresponds to `grid` described in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>interpolation_mode</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the interpolation mode. It corresponds to `interpolation_mode` in the formula description. bilinear (bilinear interpolation), nearest (nearest-neighbor interpolation), and bicubic (bicubic interpolation) are supported.</li><li>The default value is "bilinear".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the padding mode. It corresponds to `padding_mode` in the formula description. Three modes are supported: zeros(0), border(1), and reflection(2).</li><li>The default value is "zeros".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the mapping between feature map coordinates and feature values. When this parameter is set to true, the feature values are located at the pixel centers. When this parameter is set to false, the feature values are located at the pixel corners.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>channel_last</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies whether the C axis of the input x is the last axis. If the value is true, the shape of x is `[batch, height, width, channels]`; otherwise, the shape is `[batch, channels, height, width]`.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scheduler_mode</td>
      <td>Optional attr</td>
      <td><ul><li>Scheduling mode, which specifies the execution mode. 0: general. 1: sliding window. The value 1 takes effect only when the value of channel_last is true.</li><li>The default value is 1.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Final output of the interpolation computation. It corresponds to `output` described in the formula.
      </td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas inference products</term>:

  - In the 2D scenario, all the following conditions must be met:
    - The data types of the input and output support only FLOAT32.
    - The attribute `interpolation_mode` must be set to `bilinear`.
    - The attribute `padding_mode` must be set to `zeros`.
    - The shape of parameter `x` must satisfy C=32 or C \* H \* W < 20480.
  - The 3D scenario is not supported.

- <term>Kirin X90 processor products</term> and <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Constraints

- In the 3D scenario:
  - The attribute `interpolation_mode` does not support the "bicubic" configuration.
  - For input `x`, (D-axis size \* H-axis size \* W-axis size) < maximum value of INT32.
- In the 2D scenario:
  - For input `x`, (H-axis size * W-axis size) < maximum value of INT32.
  - For input `x`, precision deviations may occur if the value of (C_size \* (H_size \* W_size - 1) \* sizeof(T)) exceeds INT32_MAX.
- For the shapes of `x` and `grid`, all dimensions must be greater than 0.
- When the input value of `grid` multiplied by the image length or width exceeds the 24-bit binary number (16777216), sampling points may have errors and precision may deviate.
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: If grid contains data beyond the range [-1, 1], errors may occur when computing small-range data with bicubic interpolation, and the precision may deviate.

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_grid_sampler2_d](examples/test_aclnn_grid_sampler2_d.cpp) | Calls the GridSample operator through the [aclnnGridSampler2D](docs/aclnnGridSampler2D_en.md) API. |
| aclnn API | [test_aclnn_grid_sampler3_d](examples/test_aclnn_grid_sampler3_d.cpp) | Calls the GridSample operator through the [aclnnGridSampler3D](docs/aclnnGridSampler3D_en.md) API. |
| Graph mode | - | Calls the GridSample operator by constructing a graph in the [operator IR](op_graph/grid_sample_proto.h) mode. |
