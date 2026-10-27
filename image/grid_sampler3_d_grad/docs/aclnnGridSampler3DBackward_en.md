# aclnnGridSampler3DBackward

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

- Description: Backward propagation of [aclnnGridSampler3D](../../grid_sample/docs/aclnnGridSampler3D.md), completing the gradient computation of tensor input and tensor grid.

- Computation flow:
  1. Calculate the coordinates mapped onto input based on the (x, y, z) values stored in grid. These coordinates are related to alignCorners and paddingMode.
  2. Based on the input interpolationMode, select bilinear or nearest interpolation mode to compute the output value.
  3. Multiply the gradient value stored in grad by the weight value of the corresponding point to compute the final dx and dgrid results.

- Formula:

The dimensions of grad, input, grid, dx, and dgrid are as follows:

  $$
  grad: (N, C, D_{out}, H_{out}, W_{out})\\
  input: (N, C, D_{in}, H_{in}, W_{in})\\
  grid: (N, D_{out}, H_{out}, W_{out}, 3)\\
  dx: (N, C, D_{in}, H_{in}, W_{in})\\
  dgrid: (N, D_{out}, H_{out}, W_{out}, 3)
  $$

Here, N is consistent across grad, input, grid, dx, and dgrid; C is consistent across grad, input, and dx; $D_{in}$, $H_{in}$, $W_{in}$ are consistent across input and dx; $D_{out}$, $H_{out}$, $W_{out}$ are consistent across grad, grid, and dgrid; the last dimension of grid has a size of 3, indicating that the input pixel position information is (x, y, z), and the value ranges of x, y, and z are normalized to [-1, 1].

  1. Coordinate de-normalization:

     The (x, y, z) in grid must first be de-normalized to the input pixel coordinates (ix, iy, iz), and the gradient multipliers `gix_mult`, `giy_mult`, and `giz_mult` are computed (used for subsequent dgrid computation):
     - alignCorners = true:

       $$
       ix = \frac{(x+1)}{2} \cdot (W_{in} - 1), \quad gix\_mult = \frac{W_{in} - 1}{2} \\
       iy = \frac{(y+1)}{2} \cdot (H_{in} - 1), \quad giy\_mult = \frac{H_{in} - 1}{2} \\
       iz = \frac{(z+1)}{2} \cdot (D_{in} - 1), \quad giz\_mult = \frac{D_{in} - 1}{2}
       $$

     - alignCorners = false：

       $$
       ix = \frac{(x+1) \cdot W_{in} - 1}{2}, \quad gix\_mult = \frac{W_{in}}{2} \\
       iy = \frac{(y+1) \cdot H_{in} - 1}{2}, \quad giy\_mult = \frac{H_{in}}{2} \\
       iz = \frac{(z+1) \cdot D_{in} - 1}{2}, \quad giz\_mult = \frac{D_{in}}{2}
       $$

  2. Effect of paddingMode on gradient multipliers:
     - paddingMode="zeros", `gix_mult` remains unchanged
     - paddingMode="border", $gix\_mult = gix\_mult × grad\_clip$ (grad_clip=0 when the coordinate is outside the boundary, otherwise =1)
     - paddingMode="reflection", $gix\_mult = gix\_mult × grad\_refl × grad\_clip$ (grad_refl is the derivative of the reflection coordinate transformation function with respect to the input coordinate, indicating the direction and rate at which the reflected output coordinate changes with the input coordinate. The value is -1, 0, or 1).

  3. Gradient formulas for each interpolation mode:
     - Bilinear

       The coordinates and weights of the four corner points are:

       | Corner | Coordinate $(iz_p, iy_p, ix_p)$ | Weight $w_p$ |
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

       - dx (input gradient): Scatter the upstream gradient to the corresponding position of input according to the trilinear weight

         $$
         dx(N, C, iz_p, iy_p, ix_p) \mathrel{+}= w_p \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         That is, for each output pixel (d, h, w), multiply its gradient by the trilinear weight and accumulate it to the eight adjacent voxel positions of input (positions out of bounds are not accumulated).
       - dgrid (grid gradient): Partial derivative with respect to (ix, iy, iz)

         $$
         gix = \sum_{c} \left[ -V_{tnw} \cdot (iy_{bse}-iy)(iz_{bse}-iz) + V_{tne} \cdot (iy_{bsw}-iy)(iz_{bsw}-iz) - V_{tsw} \cdot (iy-iy_{bne})(iz_{bne}-iz) + V_{tse} \cdot (iy-iy_{bnw})(iz_{bnw}-iz) - V_{bnw} \cdot (iy_{tse}-iy)(iz-iz_{tse}) + V_{bne} \cdot (iy_{tsw}-iy)(iz-iz_{tsw}) - V_{bsw} \cdot (iy-iy_{tne})(iz-iz_{tne}) + V_{bse} \cdot (iy-iy_{tnw})(iz-iz_{tnw}) \right] \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         $$
         giy = \sum_{c} \left[ -V_{tnw} \cdot (ix_{bse}-ix)(iz_{bse}-iz) - V_{tne} \cdot (ix-ix_{bsw})(iz_{bsw}-iz) + V_{tsw} \cdot (ix_{bne}-ix)(iz_{bne}-iz) + V_{tse} \cdot (ix-ix_{bnw})(iz_{bnw}-iz) - V_{bnw} \cdot (ix_{tse}-ix)(iz-iz_{tse}) - V_{bne} \cdot (ix-ix_{tsw})(iz-iz_{tsw}) + V_{bsw} \cdot (ix_{tne}-ix)(iz-iz_{tne}) + V_{bse} \cdot (ix-ix_{tnw})(iz-iz_{tnw}) \right] \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         $$
         giz = \sum_{c} \left[ -V_{tnw} \cdot (ix_{bse}-ix)(iy_{bse}-iy) - V_{tne} \cdot (ix-ix_{bsw})(iy_{bsw}-iy) - V_{tsw} \cdot (ix_{bne}-ix)(iy-iy_{bne}) - V_{tse} \cdot (ix-ix_{bnw})(iy-iy_{bnw}) + V_{bnw} \cdot (ix_{tse}-ix)(iy_{tse}-iy) + V_{bne} \cdot (ix-ix_{tsw})(iy_{tsw}-iy) + V_{bsw} \cdot (ix_{tne}-ix)(iy-iy_{tne}) + V_{bse} \cdot (ix-ix_{tnw})(iy-iy_{tnw}) \right] \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         where $V_p = input(N, C, iz_p, iy_p, ix_p)$ (participates in computation only when the corner point is within the boundary).
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

     - Nearest
       - dx: Accumulate the upstream gradient directly to the nearest neighbor position

          $$
          dx(N, C, \text{round}(iz), \text{round}(iy), \text{round}(ix)) \mathrel{+}= grad(N, C, D_{out}, H_{out}, W_{out})
          $$

       - dgrid: Nearest interpolation is not differentiable with respect to coordinates, so **dgrid = 0**.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnGridSampler3DBackwardGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnGridSampler3DBackward API to perform the computation.

```Cpp
aclnnStatus aclnnGridSampler3DBackwardGetWorkspaceSize(
  const aclTensor*    gradOutput,
  const aclTensor*    input,
  const aclTensor*    grid,
  int64_t             interpolationMode,
  int64_t             paddingMode,
  bool                alignCorners,
  const aclBoolArray* outputMask,
  aclTensor*          inputGrad,
  aclTensor*          gridGrad,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```Cpp
aclnnStatus aclnnGridSampler3DBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnGridSampler3DBackwardGetWorkspaceSize

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
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of `input`.</li><li>When the data type is DOUBLE, the data format does not support NDHWC.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>NCDHW, NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input tensor of backward propagation, corresponding to `input` in the formula description.</td>
      <td><ul><li>Empty tensors are supported.</li><li>When the data type is DOUBLE, the data format does not support NDHWC.</li><li>The N and C axes of `input` and `gradOutput` must be the same, and the D, H, W values of `input` must not be 0.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>NCDHW, NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grid(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the tensor that samples pixel positions, corresponding to `grid` in the formula description.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of `input`.</li><li>The N, D, H, and W axes of `grid` and `gradOutput` must be the same, and the C axis value must be 3.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>interpolationMode(int64_t)</td>
      <td>Input</td>
      <td>Indicates the interpolation mode, corresponding to `interpolationMode` in the formula description.</td>
      <td>Two modes are supported: 0: bilinear, 1: nearest.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddingMode(int64_t)</td>
      <td>Input</td>
      <td>Indicates the padding mode, that is, when grid has values exceeding the [-1, 1] range, the corresponding output is processed according to paddingMode. Corresponds to `paddingMode` in the formula description.</td>
      <td>Three modes are supported: 0: zeros, 1: border, 2: reflection.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners(bool)</td>
      <td>Input</td>
      <td>Indicates how to set the correspondence between feature map coordinates and feature values.</td>
      <td>If True, the extreme values -1 and 1 are treated as the center points of the corner pixels of the reference input. If False, they are treated as the corner points of the corner pixels of the reference input.</td>
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
      <td>NCDHW, NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gridGrad(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the grid gradient, corresponding to `dgrid` in the formula description.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of `input`.</li><li>The shape is the same as that of `grid`.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
      <td>NCDHW, NDHWC</td>
      <td>5</td>
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

    The data types of parameters `gradOutput`, `input`, `grid`, `inputGrad`, and `gridGrad` do not support BFLOAT16.
  <!-- end id7 -->

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

The first segment of the API performs input parameter validation and reports an error in the following scenarios:

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
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of gradOutput, input, grid, inputGrad, or gridGrad is not within the supported range, or the data types are inconsistent.</td>
    </tr>
    <tr>
      <td>The values of interpolationMode and paddingMode are not within the supported range.</td>
    </tr>
    <tr>
      <td>The dimensional relationship among gradOutput, input, grid, inputGrad, and gridGrad does not match.</td>
    </tr>
  </tbody></table>

## aclnnGridSampler3DBackward

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
      <td>The workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device side, obtained through the first-phase API aclnnGridSampler3DBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which contains the operator computation process.</td>
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
  - aclnnGridSampler3DBackward defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grid_sampler3d_backward.h"

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
    std::vector<int64_t> gradOutputShape = {1, 1, 1, 2, 2};
    std::vector<int64_t> inputShape = {1, 1, 1, 3, 3};
    std::vector<int64_t> gridShape = {1, 1, 2, 2, 3};
    std::vector<int64_t> inputGradShape = {1, 1, 1, 3, 3};
    std::vector<int64_t> gridGradShape = {1, 1, 2, 2, 3};
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

    std::vector<float> gradOutputHostData = {1, 1, 1, 1};
    std::vector<float> inputHostData = {
        1, 2, 3, 4, 5, 6, 7, 8, 9,
    };
    std::vector<float> gridHostData = {-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0};
    std::vector<float> inputGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gridGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    bool maskValue[2] = {true, true};
    const uint64_t kNum = 2U;
    outputMask = aclCreateBoolArray(&(maskValue[0]), kNum);

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
    // Call the first-phase API of aclnnGridSampler3DBackward
    ret = aclnnGridSampler3DBackwardGetWorkspaceSize(gradOutput, input, grid, interpolationMode, paddingMode,
                                                     alignCorners, outputMask, inputGrad, gridGrad, &workspaceSize,
                                                     &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler3DBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGridSampler3DBackward
    ret = aclnnGridSampler3DBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler3DBackward failed. ERROR: %d\n", ret); return ret);

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
