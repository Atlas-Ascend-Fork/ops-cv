# GridSampler3DGrad

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

- Operator function: This is the backpropagation of the 3D scenario in [GridSampler](../grid_sample/README_en.md). It computes the gradients of the input tensor and the grid tensor.

- Computation process:
  1. Computes the coordinates mapped to input based on the (x, y, z) values stored in grid. These coordinates depend on align_corners and padding_mode.
  2. Computes the output values by selecting different interpolation modes (bilinear or nearest) based on the input interpolation_mode.
  3. Computes the final dx and dgrid results by multiplying the gradient values stored in grad by the weight values of the corresponding points.

- Computation formula:

  The sizes of grad, input, grid, dx, and dgrid are as follows:

  $$
  grad: (N, C, D_{out}, H_{out}, W_{out})\\
  input: (N, C, D_{in}, H_{in}, W_{in})\\
  grid: (N, D_{out}, H_{out}, W_{out}, 3)\\
  dx: (N, C, D_{in}, H_{in}, W_{in})\\
  dgrid: (N, D_{out}, H_{out}, W_{out}, 3)
  $$

  Here N is the same in grad, input, grid, dx, and dgrid. C is the same in grad, input, and dx. $D_{in}$, $H_{in}$, and $W_{in}$ are the same in input and dx. $D_{out}$, $H_{out}$, and $W_{out}$ are the same in grad, grid, and dgrid. The size of the last dimension of grid is 3, indicating that the input pixel position information is (x, y, z), and the value ranges of x, y, and z are normalized to [-1, 1].

  1. Coordinate denormalization:

     The (x, y, z) in grid must first be denormalized to the input pixel coordinates (ix, iy, iz). At the same time, the gradient multipliers `gix_mult`, `giy_mult`, and `giz_mult` (used for subsequent dgrid computation) are computed:
     - align_corners = true:

       $$
       ix = \frac{(x+1)}{2} \cdot (W_{in} - 1), \quad gix\_mult = \frac{W_{in} - 1}{2} \\
       iy = \frac{(y+1)}{2} \cdot (H_{in} - 1), \quad giy\_mult = \frac{H_{in} - 1}{2} \\
       iz = \frac{(z+1)}{2} \cdot (D_{in} - 1), \quad giz\_mult = \frac{D_{in} - 1}{2}
       $$

     - align_corners = false:

       $$
       ix = \frac{(x+1) \cdot W_{in} - 1}{2}, \quad gix\_mult = \frac{W_{in}}{2} \\
       iy = \frac{(y+1) \cdot H_{in} - 1}{2}, \quad giy\_mult = \frac{H_{in}}{2} \\
       iz = \frac{(z+1) \cdot D_{in} - 1}{2}, \quad giz\_mult = \frac{D_{in}}{2}
       $$

  2. Impact of padding_mode on the gradient multipliers:
     - padding_mode="zeros": `gix_mult` remains unchanged
     - padding_mode="border": $gix\_mult = gix\_mult × grad\_clip$ (grad_clip=0 when the coordinate is outside the boundary; otherwise, grad_clip=1)
     - padding_mode="reflection": $gix\_mult = gix\_mult × grad\_refl × grad\_clip$ (grad_refl is the derivative of the reflection coordinate transformation function with respect to the input coordinate. It indicates the direction and rate at which the output coordinate changes with the input coordinate after reflection. The value is -1, 0, or 1.)

  3. Gradient formulas of the interpolation modes:
     - Bilinear (trilinear interpolation, Trilinear)

       The coordinates and weights of the four corner points are:

       | Corner point | Coordinates $(iz_p, iy_p, ix_p)$ | Weight $w_p$ |
       |:------:|:------:|:----------:|
       | tnw (top-north-west) | $(⌊iz⌋, ⌊iy⌋, ⌊ix⌋)$ | $(ix_{bse} - ix) × (iy_{bse} - iy) × (iz_{bse} - iz)$ |
       | tne (top-north-east) | $(⌊iz⌋, ⌊iy⌋, ⌊ix⌋+1)$ | $(ix - ix_{bsw}) × (iy_{bsw} - iy) × (iz_{bsw} -  iz)$ |
       | tsw (top-south-west) | $(⌊iz⌋, ⌊iy⌋+1, ⌊ix⌋)$ | $(ix_{bne} - ix) × (iy - iy_{bne}) × (iz_{bne} -  iz)$ |
       | tse (top-south-east) | $(⌊iz⌋, ⌊iy⌋+1, ⌊ix⌋+1)$ | $(ix - ix_{bnw}) × (iy - iy_{bnw}) × (iz_{bnw} -  iz)$ |
       | bnw (bottom-north-west) | $(⌊iz⌋+1, ⌊iy⌋, ⌊ix⌋)$ | $(ix_{tse} - ix) × (iy_{tse} - iy) × (iz -  iz_{tse})$|
       | bne (bottom-north-east) | $(⌊iz⌋+1, ⌊iy⌋, ⌊ix⌋+1)$ | $(ix - ix_{tsw}) × (iy_{tsw} - iy) × (iz -  iz_{tsw})$ |
       | bsw (bottom-south-west) | $(⌊iz⌋+1, ⌊iy⌋+1, ⌊ix⌋)$ | $(ix_{tne} - ix) × (iy - iy_{tne}) × (iz -  iz_{tne})$ |
       | bse (bottom-south-east) | $(⌊iz⌋+1, ⌊iy⌋+1, ⌊ix⌋+1)$ | $(ix - ix_{tnw}) × (iy - iy_{tnw}) × (iz -  iz_{tnw})$ |

       where:

       $$
       ix_{tnw} = floor(ix) \\
       iy_{tnw} = floor(iy) \\
       iz_{tnw} = floor(iz) \\
       $$

       $$
       ix_{tne} = ix_{tnw} + 1 \\
       iy_{tne} = iy_{tnw} \\
       iz_{tne} = iz_{tnw}\\
       $$

       $$
       ix_{tsw} = ix_{tnw} \\
       iy_{tsw} = iy_{tnw} + 1 \\
       iz_{tsw} = iz_{tnw}\\
       $$

       $$
       ix_{tse} = ix_{tnw} + 1 \\
       iy_{tse} = iy_{tnw} + 1 \\
       iz_{tse} = iz_{tnw} \\
       $$

       $$
       ix_{bnw} = ix_{tnw}  \\
       iy_{bnw} = iy_{tnw}  \\
       iz_{bnw} = iz_{tnw} + 1 \\
       $$

       $$
       ix_{bne} = ix_{tnw} + 1  \\
       iy_{bne} = iy_{tnw}  \\
       iz_{bne} = iz_{tnw} + 1 \\
       $$

       $$
       ix_{bsw} = ix_{tnw}  \\
       iy_{bsw} = iy_{tnw} + 1  \\
       iz_{bsw} = iz_{tnw} + 1 \\
       $$

       $$
       ix_{bse} = ix_{tnw} + 1 \\
       iy_{bse} = iy_{tnw} + 1  \\
       iz_{bse} = iz_{tnw} + 1 \\
       $$

       - dx (input gradient): scatters the upstream gradients by the trilinear weights to the corresponding positions in input

         $$
         dx(N, C, iz_p, iy_p, ix_p) \mathrel{+}= w_p \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         That is, for each output pixel (d, h, w), the operator multiplies its gradient by the trilinear weight and accumulates the result to the 8 adjacent voxel positions of input (out-of-bounds positions are not accumulated).
       - dgrid (grid gradient): partial derivatives with respect to (ix, iy, iz)

         $$
         gix = \sum_{c} \left[ -V_{tnw} \cdot (iy_{bse}-iy)(iz_{bse}-iz) + V_{tne} \cdot (iy_{bsw}-iy)(iz_{bsw}-iz) - V_{tsw} \cdot (iy-iy_{bne})(iz_{bne}-iz) + V_{tse} \cdot (iy-iy_{bnw})(iz_{bnw}-iz) - V_{bnw} \cdot (iy_{tse}-iy)(iz-iz_{tse}) + V_{bne} \cdot (iy_{tsw}-iy)(iz-iz_{tsw}) - V_{bsw} \cdot (iy-iy_{tne})(iz-iz_{tne}) + V_{bse} \cdot (iy-iy_{tnw})(iz-iz_{tnw}) \right] \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         $$
         giy = \sum_{c} \left[ -V_{tnw} \cdot (ix_{bse}-ix)(iz_{bse}-iz) - V_{tne} \cdot (ix-ix_{bsw})(iz_{bsw}-iz) + V_{tsw} \cdot (ix_{bne}-ix)(iz_{bne}-iz) + V_{tse} \cdot (ix-ix_{bnw})(iz_{bnw}-iz) - V_{bnw} \cdot (ix_{tse}-ix)(iz-iz_{tse}) - V_{bne} \cdot (ix-ix_{tsw})(iz-iz_{tsw}) + V_{bsw} \cdot (ix_{tne}-ix)(iz-iz_{tne}) + V_{bse} \cdot (ix-ix_{tnw})(iz-iz_{tnw}) \right] \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         $$
         giz = \sum_{c} \left[ -V_{tnw} \cdot (ix_{bse}-ix)(iy_{bse}-iy) - V_{tne} \cdot (ix-ix_{bsw})(iy_{bsw}-iy) - V_{tsw} \cdot (ix_{bne}-ix)(iy-iy_{bne}) - V_{tse} \cdot (ix-ix_{bnw})(iy-iy_{bnw}) + V_{bnw} \cdot (ix_{tse}-ix)(iy_{tse}-iy) + V_{bne} \cdot (ix-ix_{tsw})(iy_{tsw}-iy) + V_{bsw} \cdot (ix_{tne}-ix)(iy-iy_{tne}) + V_{bse} \cdot (ix-ix_{tnw})(iy-iy_{tnw}) \right] \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         where $V_p = input(N, C, iz_p, iy_p, ix_p)$ (only corner points within the boundary participate in the computation).
       - Final:

         $$
         dgrid(N, D_{out}, H_{out}, W_{out}, 0) = gix\_mult \cdot gix
         $$

         $$
         dgrid(N, D_{out}, H_{out}, W_{out}, 1) = giy\_mult \cdot giy
         $$

         $$
         dgrid(N, D_{out}, H_{out}, W_{out}, 2) = giz\_mult \cdot giz
         $$

       - Nearest (nearest-neighbor interpolation)
         - dx: directly accumulates the upstream gradient to the nearest-neighbor position

          $$
          dx(N, C, \text{round}(iz), \text{round}(iy), \text{round}(ix)) \mathrel{+}= grad(N, C, D_{out}, H_{out}, W_{out})
          $$

         - dgrid: nearest-neighbor interpolation is not differentiable with respect to the coordinates. Therefore, **dgrid = 0**.

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
      <td>Output gradient of the upper layer in the backpropagation process. It corresponds to `grad` in the formula description. Its data type is the same as that of `x`. When the data type is DOUBLE, the data format does not support NDHWC.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>NCDHW, NDHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor of the backpropagation. It corresponds to `input` in the formula description. The shape supports only five dimensions. The N axis and C axis values of `x` and `grad` must be the same, and the D, H, and W values of x cannot be 0.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>NCDHW, NDHWC</td>
    </tr>
    <tr>
      <td>grid</td>
      <td>Input</td>
      <td>Tensor of the sampled pixel positions. It corresponds to `grid` in the formula description. The shape supports only five dimensions. The N axis, D axis, H axis, and W axis values of `grid` and `grad` must be the same, and the value of the last dimension equals 3.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>interpolation_mode</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the interpolation mode. It corresponds to `interpolation_mode` in the formula description. bilinear (0: bilinear interpolation) and nearest (1: nearest-neighbor interpolation) are supported.</li><li>The default value is "bilinear".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the padding mode. It corresponds to `padding_mode` in the formula description. Three modes are supported: 0 ('zeros'), 1 ('border'), and 2 ('reflection').</li><li>The default value is "zeros".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the mapping between feature map coordinates and feature values. It corresponds to `align_corners` in the formula description. If the value is true, the extreme values -1 and 1 are regarded as the centers of the corner pixels of the reference input. If the value is false, they are regarded as the corners of the corner pixels of the reference input.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>Output</td>
      <td>Output gradient of the backpropagation. It corresponds to `dx` in the formula description. Its data type, data format, and shape are the same as the data type, data format, and shape of `x`.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>NCDHW, NDHWC</td>
    </tr>
    <tr>
      <td>dgrid</td>
      <td>Output</td>
      <td>Gradient of `grid`. It corresponds to `dgrid` in the formula description. Its data type, data format, and shape are the same as the data type, data format, and shape of `grid`.</td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>NDHWC</td>
    </tr>
  </tbody></table>

<term>Atlas training products</term>: The data types of the input and output parameters do not support DOUBLE or BFLOAT16.

## Constraints

None

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_grid_sampler3_d_backward](examples/test_aclnn_grid_sampler3_d_backward.cpp) | Calls the GridSampler3DGrad operator through the [aclnnGridSampler3DBackward](docs/aclnnGridSampler3DBackward_en.md) API. |
| Graph mode | - | Calls the GridSampler3DGrad operator by constructing a graph in the [operator IR](op_graph/grid_sampler3_d_grad_proto.h) mode. |
