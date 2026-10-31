# aclnnGridSampler2D

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
- <term>Atlas 200I/500 A2 inference products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Provide an input tensor and a corresponding grid. Then, based on the coordinate information at each position in the grid, fill the pixel values at the corresponding positions in input into the positions specified by the grid to obtain the final output.
- Formula:

The dimensions of the input, grid, and output are as follows:

  $$
  input: (N, C, H_{in}, W_{in})\\
  grid: (N, H_{out}, W_{out}, 2)\\
  output: (N, C, H_{out}, W_{out})
  $$

  Here, N of input, grid, and output are consistent; C of input and output are consistent; $H_{out}$ and $W_{out}$ of grid and output are consistent; the last dimension of grid is 2, indicating that the input pixel position information is (x, y), and the value ranges of x and y are normalized to [-1, 1], where (-1, -1) indicates the top-left corner coordinate and (1, 1) indicates the bottom-right corner coordinate.

  - Formula for de-normalization:
    - alignCorners=true means the feature value is located at the pixel center.

      $$
      x' = (grid\_x + 1) / 2 * (W_{in} - 1)
      $$

      $$
      y' = (grid\_y +1) / 2 * (H_{in} - 1)
      $$

    - alignCorners=false means the feature value is located at the pixel corner.

      $$
      x' = ((grid\_x +1) * W_{in} - 1) / 2
      $$

      $$
      y' = ((grid\_y +1) * H_{in} - 1) / 2
      $$

  - Corner sampling example: when input has shape (1, 1, 2, 2), its only channel contains `[[1, 2], [3, 4]]`, and alignCorners=true, the normalized coordinates and nearest-neighbor sampling results are as follows:

    | Grid coordinate (x, y) | Image position | Sampled value |
    | --- | --- | --- |
    | (-1, -1) | Top-left | 1 |
    | (1, -1) | Top-right | 2 |
    | (-1, 1) | Bottom-left | 3 |
    | (1, 1) | Bottom-right | 4 |

    When alignCorners=true, -1 and 1 map to the centers of the first and last pixels. When alignCorners=false, they map to the outer image boundaries: -0.5 and W<sub>in</sub>-0.5 for x, and -0.5 and H<sub>in</sub>-0.5 for y. The sampled values then also depend on interpolationMode and paddingMode.

  - Coordinates out of range are handled differently according to paddingMode:

    - paddingMode=0 means out-of-range positions are filled with 0.
    - paddingMode=1 means out-of-range positions are filled with the boundary value.
    - paddingMode=2 means out-of-range positions are filled with the symmetric value of the boundary value.

  - When sampling input, different processing is applied according to interpolationMode:

    - interpolationMode=0 means taking the weighted average of the four coordinates around (x, y).

      $$
      output(N, C, H_{out}, W_{out}) = \sum_{i=0}^{1}\sum_{j=0}^{1}{w(i, j)} * {f(x', y')}
      $$

      where:
      - $f(x', y')$ is the pixel value of the original image at $(x', y')$.
      - $w(i, j)$ is the weight of the four neighboring points for bilinear interpolation, calculated as:

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
        w(i, j)= w(i) * w(j)
        $$

    - interpolationMode=1 means taking the coordinate value closest to (x, y) in input.

      $$
      output(N, C, H_{out}, W_{out}) = input(N, C, H_{in}, W_{in})
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

    - interpolationMode=2 means taking the weighted average of the sixteen coordinates around (x, y).

      $$
      {output(N, C, H_{out}, W_{out})}=\sum_{i=0}^{3}\sum_{j=0}^{3}{W(i, j)}*{f(x', y')}
      $$

      where:
      - i and j are the index variables of $W(i, j)$.
      - $f(x', y')$ is the pixel value of the original image at $(x', y')$.
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
        - $d = |(x_i, y_j) - (x'_i, y'_j)|$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnGridSampler2DGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnGridSampler2D API to perform the computation.

```Cpp
aclnnStatus aclnnGridSampler2DGetWorkspaceSize(
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
aclnnStatus aclnnGridSampler2D(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGridSampler2DGetWorkspaceSize

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
      <td><ul><li>Empty tensors are supported.</li><li>shape (N, C, H<sub>in</sub>, W<sub>in</sub>) is supported. H*W is less than the maximum value of INT32. The dimension values of the last two dimensions of `input`'s shape must not be 0.</li></ul></td>
      <td>FLOAT32, FLOAT16, DOUBLE, BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grid(aclTensor*)</td>
      <td>Input</td>
      <td>The sampling grid, corresponding to `grid` described in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of input `input`.</li><li>shape (N, H<sub>out</sub>, W<sub>out</sub>, 2) is supported, and N must be the same as N in the shape of input `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, DOUBLE, BFLOAT16</td>
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
      <td>Indicates how to set the correspondence between feature map coordinates and feature values. When set to true, the feature value is located at the pixel center; when set to false, the feature value is located at the pixel corner.</td>
      <td>When set to true, the feature value is located at the pixel center. When set to false, the feature value is located at the pixel corner.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out(aclTensor*)</td>
      <td>Output</td>
      <td>The final output result of the interpolation, corresponding to `output` described in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of input.</li><li>shape (N, C, H<sub>out</sub>, W<sub>out</sub>) is supported, and N, C must be the same as N, C in the shape of input, and H<sub>out</sub>, W<sub>out</sub> must be the same as H<sub>out</sub>, W<sub>out</sub> in the shape of grid.</li></ul></td>
      <td>FLOAT32, FLOAT16, DOUBLE, BFLOAT16</td>
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

  DOUBLE data is processed on the AI CPU and supports bilinear and nearest interpolation, but not bicubic interpolation. The table lists the data types supported by the API as a whole. Product-specific restrictions are listed below.

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>:
    - The parameter `interpolationMode` does not support interpolation mode 2: bicubic.
    - The parameters `input`, `grid`, and `out` do not support the BFLOAT16 data type.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    For input parameter `interpolationMode`, the 2: bicubic mode is supported only when the input data type is FLOAT32, FLOAT16, or BFLOAT16.
  <!-- end id8 -->
  <!-- npu="310b" id9 -->
  - <term>Atlas 200I/500 A2 inference products</term>:

    When the API runs on the AI Core, the following conditions must be met:
    - The input parameter `interpolationMode` is bilinear.
    - The input parameter `paddingMode` is zeros.
    - The data types of parameters `input`, `grid`, and `out` are FLOAT16.
    - The shape of parameter `input` must have a C dimension value of 32.
  <!-- end id9 -->
  <!-- npu="310p" id10 -->
  - <term>Atlas inference products</term>:

    The parameters `input`, `grid`, and `out` do not support the BFLOAT16 data type.

    When the API runs on the AI Core, the following conditions must be met:
    - The input parameter `interpolationMode` is bilinear.
    - The input parameter `paddingMode` is zeros.
    - The data types of parameters `input`, `grid`, and `out` are FLOAT32 or FLOAT16, with the corresponding shape requirements below.
    - FLOAT32: the shape of parameter `input` must satisfy C=32 or C*H*W < 20480.
    - FLOAT16: the shape of parameter `input` must satisfy C*H*W < 20480. input and grid are converted to FLOAT32 for computation, and the result is converted back to FLOAT16. C=32 alone is insufficient to use this AI Core path.
    - The data format of parameter `input` does not support NHWC.
  <!-- end id10 -->

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
      <td>The input, grid, or out passed in is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of input, grid, or out is not within the supported range, or the data types are inconsistent.</td>
    </tr>
    <tr>
      <td>The value of interpolationMode or paddingMode is not within the supported range.</td></tr>
    <tr>
      <td>When interpolationMode is bicubic, the current product does not support this interpolation mode, or the data type of input, grid, or out is not FLOAT32, FLOAT16, or BFLOAT16.</td>
    </tr>
    <tr>
      <td>The dimension relationship among input, grid, and output does not match.</td>
    </tr>
    <tr>
      <td>The last two dimensions of input are empty.</td>
    </tr>
  </tbody></table>

## aclnnGridSampler2D

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
      <td>The workspace size allocated on the Device side, obtained through the first-phase API aclnnGridSampler2DGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>The operator executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The data format of parameters `input`, `grid`, and `out` only supports (N, C, H, W). When other data formats are input, they are processed as (N, C, H, W) by default.
- For input `input`, (H-axis size * W-axis size) < maximum value of INT32.
- When the input value of grid multiplied by the image length or width exceeds the 24-bit binary number (16777216), sampling points may have errors and precision may deviate.
- If grid contains a large amount of data exceeding the [-1, 1] range, using the zeros or border padding strategy will cause many duplicate values in the result.

<!-- npu="A3,910b" id11 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - If grid contains values outside the range [-1, 1], using bicubic interpolation may introduce errors in small-value computations and cause precision deviations.
  - When bilinear or bicubic interpolation is used, the FLOAT16 data type requires workspace memory.
<!-- end id11 -->
<!-- npu="910" id12 -->
- <term>Atlas training products</term>: When bilinear interpolation is used, the FLOAT16 data type requires workspace memory.
<!-- end id12 -->
- Deterministic computation:
  - aclnnGridSampler2D defaults to a deterministic implementation.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grid_sampler2d.h"

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
    std::vector<int64_t> inputShape = {1, 1, 5, 8};
    std::vector<int64_t> gridShape = {1, 3, 3, 2};
    std::vector<int64_t> outShape = {1, 1, 3, 3};
    void* inputDeviceAddr = nullptr;
    void* gridDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* grid = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> inputHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    std::vector<float> gridHostData = {-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

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
    // Call the first-phase API of aclnnGridSampler2D
    ret = aclnnGridSampler2DGetWorkspaceSize(input, grid, interpolationMode, paddingMode, alignCorners, out,
                                             &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2DGetWorkspaceSize failed. ERROR: %d.\n[ERROR msg]%s", ret,
                                            aclGetRecentErrMsg());
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg());
                  return ret);
    }
    // Call the second-phase API of aclnnGridSampler2D
    ret = aclnnGridSampler2D(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGridSampler2D failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg());
              return ret);

    // 4. (Boilerplate) Synchronize and wait for task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg());
              return ret);

    // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d.\n[ERROR msg]%s",
                                            ret, aclGetRecentErrMsg());
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor; modify according to the specific API definition.
    aclDestroyTensor(input);
    aclDestroyTensor(grid);
    aclDestroyTensor(out);

    // 7. Release Device resources; modify according to the specific API definition.
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
