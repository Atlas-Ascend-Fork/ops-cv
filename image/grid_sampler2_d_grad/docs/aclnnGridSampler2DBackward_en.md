# aclnnGridSampler2DBackward

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Backward propagation of [aclnnGridSampler2D](../../grid_sample/docs/aclnnGridSampler2D.md), completing the gradient computation of tensor input and tensor grid.
- Computation flow:

    1. Calculate the coordinates mapped onto input based on the (x, y) values stored in grid. These coordinates are related to alignCorners and paddingMode.
    2. Based on the input interpolationMode, select bilinear, nearest, or bicubic interpolation mode to compute the weight values assigned to the gradient at the points around the coordinate.
    3. Multiply the gradient value stored in grad by the weight value of the corresponding point to compute the final dx and dgrid results.

- Formula:

  The dimensions of grad, input, grid, dx, and dgrid are as follows:
  $$
  grad: (N, C, H_{out}, W_{out})\\\\
  input: (N, C, H_{in}, W_{in})\\\\
  grid: (N, H_{out}, W_{out}, 2)\\\\
  dx: (N, C, H_{in}, W_{in})\\\\
  dgrid: (N, H_{out}, W_{out}, 2)
  $$

Where N in grad, input, grid, dx, and dgrid are all the same; C in grad, input, and dx are the same; $H_{in}$ and $W_{in}$ in input and dx are the same; $H_{out}$ and $W_{out}$ in grad, grid, and dgrid are the same; and the last dimension of grid has a size of 2, indicating that the input pixel position information is (x, y). The value ranges of x and y are normalized to [-1, 1], where (-1, 1) represents the top-left corner coordinate and (1, -1) represents the bottom-right corner coordinate.

  1. Coordinate de-normalization:
     The (x, y) in grid must first be de-normalized to the input pixel coordinates (ix, iy), and the gradient multipliers `gix_mult` and `giy_mult` are computed (used for subsequent dgrid computation):
     - alignCorners = true:

       $$
       ix = \frac{(x+1)}{2} \cdot (W_{in} - 1), \quad gix\_mult = \frac{W_{in} - 1}{2} \\
       iy = \frac{(y+1)}{2} \cdot (H_{in} - 1), \quad giy\_mult = \frac{H_{in} - 1}{2}
       $$

     - alignCorners = false：

       $$
       ix = \frac{(x+1) \cdot W_{in} - 1}{2}, \quad gix\_mult = \frac{W_{in}}{2} \\
       iy = \frac{(y+1) \cdot H_{in} - 1}{2}, \quad giy\_mult = \frac{H_{in}}{2}
       $$

  2. Effect of paddingMode on gradient multipliers:
     - paddingMode="zeros", `gix_mult` remains unchanged
     - paddingMode="border", $gix\_mult = gix\_mult × grad\_clip$ (grad_clip=0 when the coordinate is outside the boundary, otherwise =1)
     - paddingMode="reflection", $gix\_mult = gix\_mult × grad\_refl × grad\_clip$ (grad_refl is the derivative of the reflection coordinate transformation function with respect to the input coordinate, indicating the direction and rate at which the reflected output coordinate changes with the input coordinate. The value is -1, 0, or 1).

  3. Gradient formulas for each interpolation mode:
     - Bilinear

       The coordinates and weights of the four corner points are:

       | Corner | Coordinate $(iy_p, ix_p)$ | Weight $w_p$ |
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

       - dx (input gradient): Scatter the upstream gradient to the corresponding position of input according to the weight

         $$
         dx(N, C, iy_p, ix_p) \mathrel{+}= w_p \cdot grad(N, C, H_{out}, W_{out})
         $$

         That is, for each output pixel (h, w), multiply its gradient by the bilinear weight and accumulate it to the four adjacent pixel positions of input (positions out of bounds are not accumulated).
       - dgrid (grid gradient): Partial derivative with respect to (ix, iy)

         $$
         gix = \sum_{c} \left[ -V_{nw} \cdot (iy_{se} - iy) + V_{ne} \cdot   (iy_{sw} - iy) - V_{sw} \cdot (iy - iy_{ne}) + V_{se} \cdot (iy - iy_{nw}) \right] \cdot grad(N, C, H_{out}, W_{out})
         $$

         $$
         giy = \sum_{c} \left[ -V_{nw} \cdot (ix_{se} - ix) - V_{ne} \cdot   (ix - ix_{sw}) + V_{sw} \cdot (ix_{ne} - ix) + V_{se} \cdot (ix - ix_{nw}) \right] \cdot grad(N, C, H_{out}, W_{out})
         $$

         where $V_p = input(N, C, iy_p, ix_p)$ (participates in computation only when the corner point is within the boundary).

       - Final:

         $$
         dgrid(N, H_{out}, W_{out}, 0) = gix\_mult \cdot gix
         $$

         $$
         dgrid(N, H_{out}, W_{out}, 1) = giy\_mult \cdot giy
         $$

       - Nearest

         - dx: Accumulate the upstream gradient directly to the nearest neighbor position

           $$
           dx(N, C, \text{round}(iy), \text{round}(ix)) \mathrel{+}= grad(N, C, H_{out}, W_{out})
           $$

         - dgrid: Nearest interpolation is not differentiable with respect to coordinates, so **dgrid = 0**.

       - Bicubic
         - dx:

           $$
           dx(N, C, iy', ix') \mathrel{+}= grad(N, C, H_{out}, W_{out}) \cdot x\_coeffs[i] \cdot y\_coeffs[j]
           $$

           where:

           $(ix', iy') = (ix_{nw}-1+i, iy_{nw}-1+j)$, $i,j \in \{0,1,2,3\}$, positions out of bounds are handled according to paddingMode.

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

         - dgrid：

            $$
            gix = -\sum_{C}\sum_{i=0}^{3}\sum_{j=0}^{3} V_{ij} \cdot x\_coeffs\_grad[i] \cdot y\_coeffs[j] \cdot grad(N, C, H_{out}, W_{out})
            $$

            $$
            giy = -\sum_{C}\sum_{i=0}^{3}\sum_{j=0}^{3} V_{ij} \cdot y\_coeffs\_grad[j] \cdot x\_coeffs[i] \cdot grad(N, C, H_{out}, W_{out})
            $$

            where:

            $V_{ij} = get\_value\_bounded(input(N, C, H_{in}, W_{in}), ix_{nw}-1+i, iy_{nw}-1  +j)$, `x_coeffs_grad` and `y_coeffs_grad` are the derivatives of the cubic interpolation coefficients with respect to tx/ty:

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

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnGridSampler2DBackwardGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnGridSampler2DBackward API to perform the computation.

```Cpp
aclnnStatus aclnnGridSampler2DBackwardGetWorkspaceSize(
  const aclTensor*     gradOutput,
  const aclTensor*     input,
  const aclTensor*     grid,
  int64_t              interpolationMode,
  int64_t              paddingMode,
  bool                 alignCorners,
  const aclBoolArray*  outputMask,
  aclTensor*           inputGrad,
  aclTensor*           gridGrad,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnGridSampler2DBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnGridSampler2DBackwardGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimensions (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the output gradient of the previous layer during backward propagation, corresponding to `grad` in the formula description.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of `input`.</li><li>shape (N, C, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>out</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>out</em>) is supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input tensor of backward propagation, corresponding to `input` in the formula description.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The N and C axes of `input` and `gradOutput` must be the same, and the dimension values of the last two dimensions of `input` must not be 0.</li><li>shape (N, C, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>in</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>in</em>) is supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grid(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the tensor that takes pixel positions, corresponding to `grid` in the formula description.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of `input`.</li><li>The N, H, and W axes of `grid` and `gradOutput` must be the same, and the last dimension of `grid` equals 2.</li><li>shape (N, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>out</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>out</em>, 2) is supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>interpolationMode(int64_t)</td>
      <td>Input</td>
      <td>Indicates the interpolation mode, corresponding to `interpolationMode` in the formula description.</td>
      <td>Three modes are supported: 0: bilinear, 1: nearest, 2: bicubic.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddingMode(int64_t)</td>
      <td>Input</td>
      <td>Indicates the padding mode, that is, when the value of (x, y) exceeds the sampling range of the input feature map, a specific value is returned. Corresponds to `paddingMode` in the formula description.</td>
      <td>Three modes are supported: 0: zeros, 1: border, 2: reflection.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners(bool)</td>
      <td>Input</td>
      <td>Indicates how to set the correspondence between feature map coordinates and feature values, corresponding to `alignCorners` in the formula description.</td>
      <td>When set to true, the feature value is located at the pixel center. When set to false, the feature value is located at the pixel corner.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask(aclBoolArray*)</td>
      <td>Input</td>
      <td>Indicates the output mask.</td>
      <td>outputMask[0] is true/false, indicating whether to obtain the output `inputGrad`; outputMask[1] is true/false, indicating whether to obtain the output `gridGrad`.</td>
      <td>BOOLARRAY</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputGrad(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output gradient of backward propagation, corresponding to `dx` in the formula description.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of `input`.</li><li>The shape is the same as that of `input`.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gridGrad(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the grid gradient, corresponding to `dgrid` in the formula description.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of `input`.</li><li>The shape is the same as that of `grid`.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>:

    The data types of parameters `gradOutput`, `input`, `grid`, `inputGrad`, and `gridGrad` do not support BFLOAT16 or DOUBLE.
  <!-- end id7 -->

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and reports an error in the following scenarios:

<table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed gradOutput, input, grid, inputGrad, or gridGrad is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The data types of gradOutput, input, grid, inputGrad, and gridGrad are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape dimension values of gradOutput, input, and grid are not four-dimensional.</td>
    </tr>
    <tr>
      <td>The value of interpolationMode or paddingMode is not within the supported range.</td></tr>
    <tr>
      <td>The first dimension values of input, grid, and gradOutput are not the same.</td>
    </tr>
    <tr>
      <td>The second dimension value of input and the second dimension value of gradOutput are not the same.</td>
    </tr>
    <tr>
      <td>The second dimension value of grid and the third dimension value of gradOutput are not the same.</td>
    </tr>
    <tr>
      <td>The third dimension value of grid and the fourth dimension value of gradOutput are not the same.</td>
    </tr>
    <tr>
      <td>The dimension value of the last two dimensions of input is 0.</td>
    </tr>
    <tr>
      <td>The value of the last dimension of grid is not equal to 2.</td>
    </tr>
    <tr>
      <td>The shape of input is inconsistent with that of inputGrad, or the shape of grid is inconsistent with that of gridGrad.</td>
    </tr>
  </tbody></table>

## aclnnGridSampler2DBackward

- **Parameters**

<table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>The workspace memory address applied for on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device side, obtained through the first-phase API aclnnGridSampler2DBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream on which the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGridSampler2DBackward defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt. The deterministic implementation must meet all the following conditions:
    - The data types of the input and output are not DOUBLE.
    - The C-axis values of the input and output are less than or equal to 2048.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grid_sampler2d_backward.h"

namespace {
constexpr uint64_t kNum2 = 2U;
}

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate; resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from host to device-side memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of a contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) device/stream initialization; refer to the acl API manual
    // Fill in deviceId according to your actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
    int64_t interpolationMode = 0;
    int64_t paddingMode = 0;
    bool alignCorners = false;
    aclBoolArray* outputMask = nullptr;
    std::vector<int64_t> gradOutputShape = {1, 1, 3, 3};
    std::vector<int64_t> inputShape = {1, 1, 5, 8};
    std::vector<int64_t> gridShape = {1, 3, 3, 2};
    std::vector<int64_t> inputGradShape = {1, 1, 5, 8};
    std::vector<int64_t> gridGradShape = {1, 3, 3, 2};
    void* gradOutputDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* gridDeviceAddr = nullptr;
    void* inputGradDeviceAddr = nullptr;
    void* gridGradDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* input = nullptr;
    aclTensor* grid = nullptr;
    aclTensor* inputGrad = nullptr;
    aclTensor* gridGrad = nullptr;

    std::vector<float> gradOutputHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> inputHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    std::vector<float> gridHostData = {-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1};
    std::vector<float> inputGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gridGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    bool maskValue[2] = {true, true};
    outputMask = aclCreateBoolArray(&(maskValue[0]), kNum2);

    // Create gradOutput aclTensor
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT,
                          &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create grid aclTensor
    ret = CreateAclTensor(gridHostData, gridShape, &gridDeviceAddr, aclDataType::ACL_FLOAT, &grid);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create inputGrad aclTensor
    ret = CreateAclTensor(inputGradHostData, inputGradShape, &inputGradDeviceAddr, aclDataType::ACL_FLOAT, &inputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gridGrad aclTensor
    ret = CreateAclTensor(gridGradHostData, gridGradShape, &gridGradDeviceAddr, aclDataType::ACL_FLOAT, &gridGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API; change it to the specific Api name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnGridSampler2DBackward
    ret = aclnnGridSampler2DBackwardGetWorkspaceSize(gradOutput, input, grid, interpolationMode, paddingMode,
                                                     alignCorners, outputMask, inputGrad, gridGrad, &workspaceSize,
                                                     &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2DBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGridSampler2DBackward
    ret = aclnnGridSampler2DBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2DBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
    auto inputGradSize = GetShapeSize(inputGradShape);
    std::vector<float> inputGradResultData(inputGradSize, 0);
    ret = aclrtMemcpy(inputGradResultData.data(), inputGradResultData.size() * sizeof(inputGradResultData[0]),
                      inputGradDeviceAddr, inputGradSize * sizeof(inputGradResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy inputGradResultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < inputGradSize; i++) {
        LOG_PRINT("inputGradResultData[%ld] is: %f\n", i, inputGradResultData[i]);
    }

    auto gridGradSize = GetShapeSize(gridGradShape);
    std::vector<float> gridGradResultData(gridGradSize, 0);
    ret = aclrtMemcpy(gridGradResultData.data(), gridGradResultData.size() * sizeof(gridGradResultData[0]),
                      gridGradDeviceAddr, gridGradSize * sizeof(gridGradResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gridGradResultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < gridGradSize; i++) {
        LOG_PRINT("gridGradResultData[%ld] is: %f\n", i, gridGradResultData[i]);
    }

    // 6. Release aclTensor and aclBoolArray; modify according to the specific API definition.
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(input);
    aclDestroyTensor(grid);
    aclDestroyTensor(inputGrad);
    aclDestroyTensor(gridGrad);
    aclDestroyBoolArray(outputMask);

    // 7. Release device resources; modify according to the specific API definition.
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(gridDeviceAddr);
    aclrtFree(inputGradDeviceAddr);
    aclrtFree(gridGradDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
