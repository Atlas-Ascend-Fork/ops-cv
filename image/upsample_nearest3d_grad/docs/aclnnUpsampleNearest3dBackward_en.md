# aclnnUpsampleNearest3dBackward

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

- Description: Backward computation of [aclnnUpsampleNearest3d](../../upsample_nearest3d/docs/aclnnUpsampleNearest3d.md).
- Formula:

  For input gradOut(N, C, d, h, w), for any point (N, C, D, H, W) on output gradInput, the following applies:

  $$
  gradInput(N, C, D, H, W) = \sum_{d = srcD}^{srcDUp - 1}\sum_{h = srcH}^{srcHUp - 1}\sum_{w = srcW}^{srcWUp - 1}gradOut(N, C, d, h, w)
  $$

  where:

  $$
  scaleD =\begin{cases}
  scalesD & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[0]/inputSize[2] &  Otherwise\\
  \end{cases}
  $$

  $$
  scaleH =\begin{cases}
  scalesH & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[1]/inputSize[3] &  Otherwise\\
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  scalesW & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[2]/inputSize[4] &  Otherwise\\
  \end{cases}
  $$

  $$
  srcD = Min(ceil(scaleD * D), outputSize[0])
  $$

  $$
  srcDUp = Min(ceil(scaleD * (D + 1)), outputSize[0])
  $$

  $$
  srcH = Min(ceil(scaleH * H), outputSize[1])
  $$

  $$
  srcHUp = Min(ceil(scaleH * (H + 1)), outputSize[1])
  $$

  $$
  srcW = Min(ceil(scaleW * W), outputSize[2])
  $$

  $$
  srcWUp = Min(ceil(scaleW * (W + 1)), outputSize[2])
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnUpsampleNearest3dBackwardGetWorkspaceSize` API to obtain the input parameters and compute the required workspace size according to the computation flow, then call the `aclnnUpsampleNearest3dBackward` API to perform the computation.

```Cpp
aclnnStatus aclnnUpsampleNearest3dBackwardGetWorkspaceSize(
  const aclTensor   *gradOut,
  const aclIntArray *outputSize,
  const aclIntArray *inputSize,
  double             scalesD,
  double             scalesH,
  double             scalesW,
  aclTensor         *gradInput,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnUpsampleNearest3dBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnUpsampleNearest3dBackwardGetWorkspaceSize

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
      <td>gradOut(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the gradient tensor for backward computation, corresponding to `gradOut` in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT32, FLOAT16, DOUBLE, BFLOAT16</td>
      <td>NCDHW, NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize(aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the spatial size of input `gradOut` in the D, H, and W dimensions.</td>
      <td>size is 3, and all elements are greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize(aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the spatial size of output `gradInput`.</td>
      <td><ul><li>size is 5, and the last two elements are greater than zero.</li><li>When the data format of input `gradOut` is NCDHW, it indicates the spatial size of output `gradInput` in the N, C, D, H, and W dimensions respectively; when the data format of input `gradOut` is NDHWC, it indicates the spatial size of output `gradInput` in the N, D, H, W, and C dimensions respectively.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesD(double)</td>
      <td>Input</td>
      <td>Indicates the depth-dimension multiplier of output `gradInput`, corresponding to `scalesD` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH(double)</td>
      <td>Input</td>
      <td>Indicates the height-dimension multiplier of output `gradInput`, corresponding to `scalesH` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW(double)</td>
      <td>Input</td>
      <td>Indicates the width-dimension multiplier of output `gradInput`, corresponding to `scalesW` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output tensor of backward computation, corresponding to `gradInput` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type and data format are the same as those of input parameter `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, DOUBLE, BFLOAT16</td>
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
      <td>If the input parameter is a mandatory input, output, or mandatory attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data type of gradOut is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of gradOut and gradInput are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension of gradOut is not 5.</td>
    </tr>
    <tr>
      <td>The size of outputSize is not equal to 3.</td>
    </tr>
    <tr>
      <td>An element value of outputSize is not greater than 0.</td>
    </tr>
    <tr>
      <td>The size of inputSize is not equal to 5.</td>
    </tr>
    <tr>
      <td>An element value of inputSize is not greater than 0.</td>
    </tr>
    <tr>
      <td>The size of gradOut in the D, H, and W dimensions is inconsistent with outputSize[0], outputSize[1], and outputSize[2].</td>
    </tr>
    <tr>
      <td>The size of gradInput in the N and C dimensions is inconsistent with inputSize[0] and inputSize[1].</td>
    </tr>
    <tr>
      <td>The size of gradInput in the D, H, and W dimensions is inconsistent with inputSize[2], inputSize[3], and inputSize[4].</td>
    </tr>
  </tbody></table>

## aclnnUpsampleNearest3dBackward

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained through the first-phase API aclnnUpsampleNearest3dBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Shape constraints for parameters `gradOut` and `gradInput`:
  - The value of each dimension is less than or equal to 2^20.
  - The N and C axes of parameter `gradInput` are the same as those of `gradOut`.
  - The memory usage must be less than 60 GB. The formula for memory usage is as follows:

    $$
    N * C * (gradOut\_D * gradOut\_H * gradOut\_W + gradInput\_D * gradInput\_H * gradInput\_W + gradOut\_D * gradOut\_H * gradInput\_W + gradOut\_D * gradInput\_H * gradInput\_W) * sizeof(dtype) < 60 * 1024 * 1024 * 1024
    $$

    where:
    - N represents the N axis of the input and output.
    - C represents the C axis of the input and output.
    - dtype represents the data type of the input tensor.
  - N * C * gradOut_D * gradOut_H < 2^31
  - gradInput_W * gradInput_H < 2^31
- When the data format of parameters gradOut and gradInput is neither NCDHW nor NDHWC, other input data formats are processed as NCDHW by default.
- When the values of scalesD, scalesH, and scalesW are all greater than 0, parameters inputSize, outputSize, scalesD, scalesH, and scalesW must satisfy the following constraints:

  $$
  outputSize\_D = floor(inputSize\_D * scalesD)
  $$

  $$
  outputSize\_H = floor(inputSize\_H * scalesH)
  $$

  $$
  outputSize\_W = floor(inputSize\_W * scalesW)
  $$

- Deterministic computation:
  - aclnnUpsampleNearest3dBackward defaults to a deterministic implementation.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_nearest_3d_backward.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    // check handle according to your own needs
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
    std::vector<int64_t> gradOutShape = {2, 2, 2, 2, 2};
    std::vector<int64_t> gradInputShape = {2, 2, 1, 1, 1};
    void* gradOutDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<float> gradOutHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> gradInputHostData = {2.0, 2, 2, 2};
    std::vector<int64_t> outputSizeData = {2, 2, 2};
    std::vector<int64_t> inputSizeData = {2, 2, 1, 1, 1};
    double scalesD = 0.0;
    double scalesH = 0.0;
    double scalesW = 0.0;
    // Create gradOut aclTensor
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gradInput aclTensor
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const aclIntArray* outputSize = aclCreateIntArray(outputSizeData.data(), outputSizeData.size());
    CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    const aclIntArray* inputSize = aclCreateIntArray(inputSizeData.data(), inputSizeData.size());
    CHECK_RET(inputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the CANN operator library API; change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnUpsampleNearest3dBackward
    ret = aclnnUpsampleNearest3dBackwardGetWorkspaceSize(gradOut, outputSize, inputSize, scalesD, scalesH, scalesW,
                                                         gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearest3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnUpsampleNearest3dBackward
    ret = aclnnUpsampleNearest3dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearest3dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar; modify according to the specific API definition.
    aclDestroyTensor(gradOut);
    aclDestroyTensor(gradInput);

    // 7. Release device resources; modify according to the specific API definition.
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
