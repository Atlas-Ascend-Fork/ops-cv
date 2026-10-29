# GridSampler2DGrad

English | [简体中文](./README.md)

## Product Support

|Product | Supported |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: This is the backpropagation of the 2D scenario in [GridSampler](../grid_sample/README_en.md). It computes the gradients of the input tensor and the grid tensor.
- Computation process:

    1. Computes the coordinates mapped to input based on the (x, y) values stored in grid. These coordinates depend on align_corners and padding_mode.
    2. Computes the weight values assigned to the gradients of the points around this coordinate by selecting different interpolation modes (bilinear, nearest, or bicubic) based on the input interpolationMode.
    3. Computes the final dx and dgrid results by multiplying the gradient values stored in grad by the weight values of the corresponding points.

- Computation formula:

  The sizes of grad, input, grid, dx, and dgrid are as follows:
  $$
  grad: (N, C, H_{out}, W_{out})\\
  input: (N, C, H_{in}, W_{in})\\
  grid: (N, H_{out}, W_{out}, 2)\\
  dx: (N, C, H_{in}, W_{in})\\
  dgrid: (N, H_{out}, W_{out}, 2)
  $$

  Here N is the same in grad, input, grid, dx, and dgrid. C is the same in grad, input, and dx. $H_{in}$ and $W_{in}$ are the same in input and dx. $H_{out}$ and $W_{out}$ are the same in grad, grid, and dgrid. The size of the last dimension of grid is 2, indicating that the input pixel position information is (x, y). The value ranges of x and y are normalized to [-1, 1]. (-1, -1) indicates the top-left coordinate, and (1, 1) indicates the bottom-right coordinate.

  1. Coordinate denormalization:
     The (x, y) in grid must first be denormalized to the input pixel coordinates (ix, iy). At the same time, the gradient multipliers `gix_mult` and `giy_mult` (used for subsequent dgrid computation) are computed:
     - align_corners = true:

       $$
       ix = \frac{(x+1)}{2} \cdot (W_{in} - 1), \quad gix\_mult = \frac{W_{in} - 1}{2} \\
       iy = \frac{(y+1)}{2} \cdot (H_{in} - 1), \quad giy\_mult = \frac{H_{in} - 1}{2}
       $$

     - align_corners = false:

       $$
       ix = \frac{(x+1) \cdot W_{in} - 1}{2}, \quad gix\_mult = \frac{W_{in}}{2} \\
       iy = \frac{(y+1) \cdot H_{in} - 1}{2}, \quad giy\_mult = \frac{H_{in}}{2}
       $$

  2. Impact of padding_mode on the gradient multipliers:
     - padding_mode="zeros": `gix_mult` remains unchanged
     - padding_mode="border": $gix\_mult = gix\_mult × grad\_clip$ (grad_clip=0 when the coordinate is outside the boundary; otherwise, grad_clip=1)
     - padding_mode="reflection": $gix\_mult = gix\_mult × grad\_refl × grad\_clip$ (grad_refl is the derivative of the reflection coordinate transformation function with respect to the input coordinate. It indicates the direction and rate at which the output coordinate changes with the input coordinate after reflection. The value is -1, 0, or 1.)

  3. Gradient formulas of the interpolation modes:
     - Bilinear (bilinear interpolation)

       The coordinates and weights of the four corner points are:

       | Corner point | Coordinates $(iy_p, ix_p)$ | Weight $w_p$ |
       |:------:|:------:|:----------:|
       | nw (northwest) | $(iy_{nw}, ix_{nw})$ | $(ix_{se} - ix) × (iy_{se} - iy)$ |
       | ne (northeast) | $(iy_{ne}, ix_{ne})$ | $(ix - ix_{sw}) × (iy_{sw} - iy)$ |
       | sw (southwest) | $(iy_{sw}, ix_{sw})$ | $(ix_{ne} - ix) × (iy - iy_{ne})$ |
       | se (southeast) | $(iy_{se}, ix_{se})$ | $(ix - ix_{nw}) × (iy - iy_{nw})$ |

       where:

       $$
       ix_{nw} = floor(ix) \\
       iy_{nw} = floor(iy) \\
       ix_{ne} = floor(ix) + 1 \\
       iy_{ne} = floor(iy)\\
       ix_{sw} = floor(ix) \\
       iy_{sw} = floor(iy) + 1 \\
       ix_{se} = floor(ix) + 1 \\
       iy_{se} = floor(iy) + 1
       $$

       - dx (input gradient): scatters the upstream gradients by weight to the corresponding positions in input

         $$
         dx(N, C, iy_p, ix_p) \mathrel{+}= w_p \cdot grad(N, C, H_{out}, W_{out})
         $$

         That is, for each output pixel (h, w), the operator multiplies its gradient by the bilinear weight and accumulates the result to the four adjacent pixel positions of input (out-of-bounds positions are not accumulated).
       - dgrid (grid gradient): partial derivatives with respect to (ix, iy)

         $$
         gix = \sum_{c} \left[ -V_{nw} \cdot (iy_{se} - iy) + V_{ne} \cdot   (iy_{sw} - iy) - V_{sw} \cdot (iy - iy_{ne}) + V_{se} \cdot (iy - iy_{nw}) \right] \cdot grad(N, C, H_{out}, W_{out})
         $$

         $$
         giy = \sum_{c} \left[ -V_{nw} \cdot (ix_{se} - ix) - V_{ne} \cdot   (ix - ix_{sw}) + V_{sw} \cdot (ix_{ne} - ix) + V_{se} \cdot (ix - ix_{nw}) \right] \cdot grad(N, C, H_{out}, W_{out})
         $$

         where $V_p = input(N, C, iy_p, ix_p)$ (only corner points within the boundary participate in the computation).
       - Final:

         $$
         dgrid(N, H_{out}, W_{out}, 0) = gix\_mult \cdot gix
         $$

         $$
         dgrid(N, H_{out}, W_{out}, 1) = giy\_mult \cdot giy
         $$

     - Nearest (nearest-neighbor interpolation)
       - dx: directly accumulates the upstream gradient to the nearest-neighbor position

          $$
          dx(N, C, \text{round}(iy), \text{round}(ix)) \mathrel{+}= grad(N, C, H_{out}, W_{out})
          $$

       - dgrid: nearest-neighbor interpolation is not differentiable with respect to the coordinates. Therefore, **dgrid = 0**.

     - Bicubic (bicubic interpolation)
       - dx:

          $$
          dx(N, C, iy', ix') \mathrel{+}= grad(N, C, H_{out}, W_{out}) \cdot x\_coeffs[i] \cdot y\_coeffs[j]
          $$

          where:

          $(ix', iy') = (ix_{nw}-1+i, iy_{nw}-1+j)$, $i,j \in \{0,1,2,3\}$. Out-of-bounds positions are processed according to padding_mode.

          $$
          tx = ix - floor(ix) \\
          ty = iy - floor(iy) \\
          $$

          $$
          A = -0.75 \\
          x_0 = tx + 1.0 \\
          x\_coeffs[0] = ((A * x_0 - 5* A) * x_0 + 8 * A) * x_0 - 4 * A
          $$

          $$
          x_1 = tx \\
          x\_coeffs[1] = ((A + 2)* x_1 - (A + 3)) * x_1 * x_1 + 1
          $$

          $$
          x_2 = 1 - tx \\
          x\_coeffs[2] = ((A + 2)* x_2 - (A + 3)) * x_2 * x_2 + 1
          $$

          $$
          x_3 = 2 - tx \\
          x\_coeffs[3] = ((A * x_3 - 5* A) * x_3 + 8 * A) * x_3 - 4 * A
          $$

          $$
          y_0 = ty + 1.0 \\
          y\_coeffs[0] = ((A * y_0 - 5* A) * y_0 + 8 * A) * y_0 - 4 * A
          $$

          $$
          y_1 = ty \\
          y\_coeffs[1] = ((A + 2)* y_1 - (A + 3)) * y_1 * y_1 + 1
          $$

          $$
          y_2 = 1 - ty \\
          y\_coeffs[2] = ((A + 2)* y_2 - (A + 3)) * y_2 * y_2 + 1
          $$

          $$
          y_3 = 2 - ty \\
          y\_coeffs[3] = ((A * y_3 - 5* A) * y_3 + 8 * A) * y_3 - 4 * A
          $$

       - dgrid:

          $$
          gix = -\sum_{C}\sum_{i=0}^{3}\sum_{j=0}^{3} V_{ij} \cdot x\_coeffs\_grad[i] \cdot y\_coeffs[j] \cdot grad(N, C, H_{out}, W_{out})
          $$

          $$
          giy = -\sum_{C}\sum_{i=0}^{3}\sum_{j=0}^{3} V_{ij} \cdot y\_coeffs\_grad[j] \cdot x\_coeffs[i] \cdot grad(N, C, H_{out}, W_{out})
          $$

          where:

          $V_{ij} = get\_value\_bounded(input(N, C, H_{in}, W_{in}), ix_{nw}-1+i, iy_{nw}-1  +j)$. `x_coeffs_grad` and `y_coeffs_grad` are the derivatives of the cubic interpolation coefficients with respect to tx/ty:

          $$
          x\_coeffs\_grad[0] = (-3A \cdot x - 10A) \cdot x - 8A \\
          \quad x = |-1 - tx|
          $$

          $$
          x\_coeffs\_grad[1] = (-3(A+2) \cdot x - 2(A+3)) \cdot x \\
          \quad x = | 0 - tx|
          $$

          $$
          x\_coeffs\_grad[2] = (3(A+2) \cdot x - 2(A+3)) \cdot x \\
          \quad x = |1 - tx|
          $$

          $$
          x\_coeffs\_grad[3] = (3A \cdot x - 10A) \cdot x + 8A \\
          \quad x = |2 - tx|
          $$

          $$
          y\_coeffs\_grad[0] = (-3A \cdot y - 10A) \cdot y - 8A \\
          \quad y = |-1 - ty|
          $$

          $$
          y\_coeffs\_grad[1] = (-3(A+2) \cdot y - 2(A+3)) \cdot y \\
          \quad y = | 0 - ty|
          $$

          $$
          y\_coeffs\_grad[2] = (3(A+2) \cdot y - 2(A+3)) \cdot y \\
          \quad y = |1 - ty|
          $$

          $$
          y\_coeffs\_grad[3] = (3A \cdot y - 10A) \cdot y + 8A \\
          \quad y = |2 - ty|
          $$

          Final:

          $$
          dgrid(N, H_{out}, W_{out}, 0) = gix\_mult \cdot gix
          $$

          $$
          dgrid(N, H_{out}, W_{out}, 1) = giy\_mult \cdot giy
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
      <td>grad</td>
      <td>Input</td>
      <td>Output gradient of the upper layer in the backpropagation process. It corresponds to `grad` in the formula description.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor of the backpropagation. It corresponds to `input` in the formula description. The shape supports only four dimensions. The N axis and C axis values of `x` and `grad` must be the same, and the dimension values of the last two dimensions of `x` cannot be 0.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grid</td>
      <td>Input</td>
      <td>Tensor of the sampled pixel positions. It corresponds to `grid` in the formula description. The shape supports only four dimensions. The N axis, H axis, and W axis values of `grid` and `grad` must be the same, and the value of the last dimension of `grid` equals 2.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>interpolation_mode</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the interpolation mode. It corresponds to `interpolation_mode` in the formula description. "bilinear" (bilinear interpolation), "nearest" (nearest-neighbor interpolation), and "bicubic" (bicubic interpolation) are supported.</li><li>The default value is "bilinear".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the padding mode. It corresponds to `padding_mode` in the formula description. Three modes are supported: "zeros", "border", and "reflection".</li><li>The default value is "zeros".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the mapping between feature map coordinates and feature values. It corresponds to `align_corners` in the formula description. When this parameter is set to true, the feature values are located at the pixel centers. When this parameter is set to false, the feature values are located at the pixel corners.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>Output</td>
      <td>Output gradient of the backpropagation. It corresponds to `dx` in the formula description. Its data type, data format, and shape are the same as the data type, data format, and shape of `x`.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgrid</td>
      <td>Output</td>
      <td>Gradient of grid. It corresponds to `dgrid` in the formula description. Its data type, data format, and shape are the same as the data type, data format, and shape of `grid`.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

<term>Atlas training products</term>: The data types of the input and output parameters do not support DOUBLE or BFLOAT16.

## Constraints

None

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_grid_sampler2_d_backward](examples/test_aclnn_grid_sampler2_d_backward.cpp) | Calls the GridSampler2DGrad operator through the [aclnnGridSampler2DBackward](docs/aclnnGridSampler2DBackward_en.md) API. |
| Graph mode | - | Calls the GridSampler2DGrad operator by constructing a graph in the [operator IR](op_graph/grid_sampler2_d_grad_proto.h) mode. |
