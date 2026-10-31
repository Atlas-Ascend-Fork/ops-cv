# aclnnGridSampler3D

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

- Description: Provide an input tensor input and a corresponding grid. Based on the coordinate information at each position in the grid, fill the pixel values at the corresponding positions in input into the positions specified by grid to obtain the final output.
- Formula:

The dimensions of input, grid, and output are as follows:

  $$
  input: (N, C, D_{in}, H_{in}, W_{in})\\
  grid: (N, D_{out}, H_{out}, W_{out}, 3)\\
  output: (N, C, D_{out}, H_{out}, W_{out})
  $$

Where N in input, grid, and output is consistent, C in input and output is consistent, $D_{out}$, $H_{out}$, $W_{out}$ in grid and output are consistent, and the last dimension of grid has a size of 3, indicating that the input pixel position information is (x, y, z), with the value ranges of x, y, and z normalized to [-1, 1].

  - Formula for de-normalization:
    - alignCorners=true means the feature value is located at the pixel center.

      $$
      x' = (grid\_x + 1) / 2 * (W_{in} - 1)
      $$

      $$
      y' = (grid\_y +1) / 2 * (H_{in} - 1)
      $$

      $$
      z' = (grid\_z +1) / 2 * (D_{in} - 1)
      $$

    - alignCorners=false means the feature value is located at the pixel corner.

      $$
      x' = ((grid\_x +1) * W_{in} - 1) / 2
      $$

      $$
      y' = ((grid\_y +1) * H_{in} - 1) / 2
      $$

      $$
      z' = ((grid\_z +1) * D_{in} - 1) / 2
      $$

  - Coordinates out of range are handled differently according to paddingMode:
    - paddingMode=0 means out-of-range positions are filled with 0.
    - paddingMode=1 means out-of-range positions are filled with the boundary value.
    - paddingMode=2 means out-of-range positions are filled with the symmetric value of the boundary value.
  - When sampling input, different processing is applied according to interpolationMode:
    - interpolationMode="bilinear" means taking the weighted average of the eight coordinates around (x, y, z) in input.

      $$
      {output(N, C, D_{out}, H_{out}, W_{out})} = \sum_{i=0}^{1}\sum_{j=0}^{1}\sum_{k=0}^{1}{w(i, j, k)} * {f(x', y', z')}
      $$

      where:
      - $f(x', y', z')$ is the pixel value of the original image at $(x', y', z')$.
      - $w(i, j, k)$ is the weight of the eight neighboring points for bilinear interpolation, calculated as:

        $$
          w(i) = \begin{cases}
          1 - |x'_i - x_i| & |x'_i - x_i| < 1 \\
          0 & otherwise
          \end{cases}
        $$

        $$
          w(j) = \begin{cases}
          1 - |y'_j - y_j| & |y'_j - y_j| < 1 \\
          0 & otherwise
          \end{cases}
        $$

        $$
          w(k) = \begin{cases}
          1 - |z'_k - z_k| & |z'_k - z_k| < 1 \\
          0 & otherwise
          \end{cases}
        $$

        $$
        w(i, j, k)= w(i) * w(j) * w(k)
        $$

    - interpolationMode="nearest" means taking the coordinate value closest to (x, y, z) in input.

      $$
      output(N, C, D_{out}, H_{out}, W_{out}) = input(N, C, D_{in}, H_{in}, W_{in})
      $$

      where:

      $$
      W_{out} = min(round(x'),  W_{in} - 1)\\
      W_{out} = max(round(x'),  0)
      $$

      $$
      H_{out} = min(round(y'),  H_{in} - 1)\\
      H_{out} = max(round(y'),  0)
      $$

      $$
      D_{out} = min(round(z'),  D_{in} - 1)\\
      D_{out} = max(round(z'),  0)
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnGridSampler3DGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnGridSampler3D API to perform the computation.

```Cpp
aclnnStatus aclnnGridSampler3DGetWorkspaceSize(
  const aclTensor *input,
  const aclTensor *grid,
  int64_t          interpolationMode,
  int64_t          paddingMode,
  bool             alignCorners,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGridSampler3D(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGridSampler3DGetWorkspaceSize

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
      <td>input(aclTensor*)</td>
      <td>Input</td>
      <td>The input tensor for interpolation, corresponding to `input` described in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>When the data format is ND, it is processed in NCDHW format.</li></ul></td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>NCDHW, NDHWC, ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grid(aclTensor*)</td>
      <td>Input</td>
      <td>The sampling grid, corresponding to `grid` described in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of input `input`.</li><li>shape (N, D<sub>out</sub>, H<sub>out</sub>, W<sub>out</sub>, 3) is supported, and N must be the same as N in the shape of input `input`.</li></ul></td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>ND</td>
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
      <td>Indicates the padding mode outside the grid range. That is, when grid has values exceeding the [-1, 1] range, the corresponding output is processed according to paddingMode. Corresponds to `paddingMode` in the formula description.</td>
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
      <td>out(aclTensor*)</td>
      <td>Output</td>
      <td>The final output result of the interpolation, corresponding to `output` described in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data format and data type are the same as those of input.</li><li>The N and C axes of the shape are the same as those of input, and the D, H, and W axes of the shape are the same as those of grid.</li></ul></td>
      <td>FLOAT16, FLOAT32, DOUBLE, BFLOAT16</td>
      <td>NCDHW, NDHWC, ND</td>
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

    The data types of parameters `input`, `grid`, and `out` do not support BFLOAT16.
  <!-- end id7 -->

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API completes input parameter validation and reports an error in the following scenarios:

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
      <td>The input, grid, or out passed in is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of input, grid, or out is not within the supported range, or the data types are inconsistent.</td>
    </tr>
    <tr>
      <td>The data format of input, grid, or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The value of interpolationMode or paddingMode is not within the supported range.</td></tr>
    <tr>
      <td>The dimension relationship among input, grid, and out does not match.</td>
    </tr>
  </tbody></table>

## aclnnGridSampler3D

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
      <td>The workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnGridSampler3DGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which includes the operator computation flow.</td>
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

- For input `input`, (D-axis size \* H-axis size \* W-axis size) < maximum value of INT32.
- Deterministic computation:
  - aclnnGridSampler3D defaults to a deterministic implementation.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grid_sampler3d.h"

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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
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
    std::vector<int64_t> inputShape = {1, 1, 1, 3, 3};
    std::vector<int64_t> gridShape = {1, 1, 2, 2, 3};
    std::vector<int64_t> outShape = {1, 1, 1, 2, 2};
    void* inputDeviceAddr = nullptr;
    void* gridDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* grid = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> inputHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> gridHostData = {-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0};
    std::vector<float> outHostData = {0, 0, 0, 0};

    // Create input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create grid aclTensor
    ret = CreateAclTensor(gridHostData, gridShape, &gridDeviceAddr, aclDataType::ACL_FLOAT, &grid);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API; change it to the specific Api name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnGridSampler3D
    ret = aclnnGridSampler3DGetWorkspaceSize(input, grid, interpolationMode, paddingMode, alignCorners, out,
                                             &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler3DGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGridSampler3D
    ret = aclnnGridSampler3D(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler3D failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor; modify according to the specific API definition.
    aclDestroyTensor(input);
    aclDestroyTensor(grid);
    aclDestroyTensor(out);

    // 7. Release device resources; modify according to the specific API definition.
    aclrtFree(inputDeviceAddr);
    aclrtFree(gridDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
