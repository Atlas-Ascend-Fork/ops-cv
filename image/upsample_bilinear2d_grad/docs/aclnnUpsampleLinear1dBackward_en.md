# aclnnUpsampleLinear1dBackward

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

- Description: Backward propagation of [aclnnUpsampleLinear1d](../../upsample_linear1d/docs/aclnnUpsampleLinear1d.md).

- Formula:
  - Core algorithm logic of the forward pass:
    1. Map each point of the target image back to the original image to obtain a coordinate with a fractional part.
    2. Based on this floating-point coordinate, compute the adjacent original image points before and after.
    3. Compute the weights of the adjacent points to the corresponding target point respectively, and multiply and accumulate them according to the weights to obtain the target point value.
  - Specific computation logic:
    The scaling method is divided into corner alignment and edge alignment. Corner alignment (alignCorners = true) means aligning according to the pixel center point at the top-left corner of the original image; edge alignment (alignCorners = false) means aligning according to the top-left vertex and two edges of the original image. There are differences in computing the scaling coefficient and coordinate position. Then the following formula applies:

    $$
    scale =\begin{cases}
    (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scales & alignCorners=false\&scales>0\\
    inputSize[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    Therefore, for a point p(x) in a certain direction of output, the point mapped back to the original image is denoted as q(x'), and the relationship is:

    $$
    x' =\begin{cases}
    x * scale & alignCorners=true \\
    MAX(0,{(x+0.5)*scale-0.5}) & alignCorners=false
    \end{cases}
    $$

    - Notation:

      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$

    - Then the following formula applies:

      $$
      {V(p_{x})} = {V(p_{x0})} * {lambda_{0}} + {V(p_{x1})} * {lambda_{1}}
      $$

    - Assumption: The output image out $(x)$ of forward interpolation is affected by the original image input $(x_i)$. Then:

      $$
      gradInput(x_i) += gradOut(x) * lambda(x_i)
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnUpsampleLinear1dBackwardGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnUpsampleLinear1dBackward API to perform the computation.

```Cpp
aclnnStatus aclnnUpsampleLinear1dBackwardGetWorkspaceSize(
  const aclTensor   * gradOut,
  const aclIntArray * outputSize,
  const aclIntArray * inputSize,
  bool                alignCorners,
  double              scales,
  aclTensor         * out,
  uint64_t          * workspaceSize,
  aclOpExecutor    ** executor)
```

```Cpp
aclnnStatus aclnnUpsampleLinear1dBackward(
  void          * workspace,
  uint64_t        workspaceSize,
  aclOpExecutor * executor,
  aclrtStream     stream)
```

## aclnnUpsampleLinear1dBackwardGetWorkspaceSize

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
      <td>gradOut(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the gradient tensor for backward computation, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>When the data format is ND, it is processed in NCL format by default.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCL</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize(aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the spatial size of input `gradOut` in the L dimension, corresponding to `outputSize` in the formula.</td>
      <td>size is 1, and the value is greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize(aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the spatial size of output out in the N, C, and L dimensions respectively, corresponding to `inputSize` in the formula</td>
      <td>size is 3, and all elements are greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners(bool)</td>
      <td>Input</td>
      <td>A BOOL-type parameter, corresponding to `alignCorners` in the formula.</td>
      <td><ul><li>If set to True, the input and output tensors are aligned by the center point of their corner pixels, preserving the value at the corner pixel.</li><li>If set to False, the input and output tensors are aligned by the corner point of their corner pixels, and the interpolation uses the edge value to fill out-of-boundary values.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales(double)</td>
      <td>Input</td>
      <td>Indicates the L-dimension multiplier of output out, corresponding to `scales` in the formula.</td>
      <td>The value is not greater than 500.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output tensor of backward computation, corresponding to `gradInput` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The output dimension must be 3D. The data type and data format are the same as those of input `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCL</td>
      <td>3</td>
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

    The data types of input parameter `gradOut` and output parameter `out` only support FLOAT32 and FLOAT16.
  <!-- end id7 -->

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    Input parameter `gradOut`: When the value of the corresponding axis of gradOut's shape is different from that of inputSize's corresponding axis, the data type only supports FLOAT32 and FLOAT16.
  <!-- end id8 -->

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
      <td>A required input, output, or required attribute is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="14">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="14">161002</td>
    </tr>
    <tr>
      <td>The data type of gradOut is not within the supported range.</td>
    </tr>
    <tr><td>The data types of gradOut and out are inconsistent.</td>
    </tr>
    <tr><td>The dimension of gradOut is not 3.</td>
    </tr>
    <tr><td>The size of outputSize is not equal to 1.</td>
    </tr>
    <tr><td>An element value of outputSize is less than 1.</td>
    </tr>
    <tr><td>The size of inputSize is not equal to 3.</td>
    </tr>
    <tr><td>A certain element value of inputSize is less than 1.</td>
    </tr>
    <tr><td>The size of gradOut in the L dimension differs from outputSize[0].</td>
    </tr>
    <tr><td>The dimension sizes of the N/C axes of gradOut and out are not equal.</td>
    </tr>
    <tr><td>The size of each dimension of the shape of out differs from the corresponding element value in inputSize.</td>
    </tr>
    <tr><td>The value of scales does not meet the constraints.</td>
    </tr>
  </tbody></table>

## aclnnUpsampleLinear1dBackward

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
      <td>The workspace size applied on the Device side, obtained through the first-phase API aclnnUpsampleLinear1dBackwardGetWorkspaceSize.</td>
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

- Shape constraints for parameters `gradOut` and `out`:
  - The value of each dimension is less than or equal to 2^20.
  - The N and C axes of parameter `out` are the same as those of `gradOut`.
  - The memory usage must be less than 60 GB. The formula for memory usage is as follows:

    $$
    (gradOut\_L + out\_L + out\_L) * N * C  * sizeof(dtype) < 60 * 1024 * 1024 * 1024
    $$

    where:
    - N represents the N axis of the input and output.
    - C represents the C axis of the input and output.
    - dtype represents the data type of the input tensor.
  - N * C < 2^31
- When the data format of input parameter `gradOut` and output parameter `out` is neither NCL nor ND, other input data formats are processed as NCL by default.

<!-- npu="A3,910b" id9 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: For the backward API, the downscaling factor of the input data must be less than or equal to 500, that is:

  $$
  outputSize[0] / Output shape length L <= 500
  $$
<!-- end id9 -->

- Parameters inputSize, outputSize, and scales must satisfy the following constraints:

  $$
  outputSize = floor(inputSize\_L * scales)
  $$

- Deterministic computation:

  <!-- npu="A3,910b,910" id10 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas training products</term>: aclnnUpsampleLinear1dBackward defaults to a deterministic implementation.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnUpsampleLinear1dBackward defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt.
  <!-- end id11 -->

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_linear_1d_backward.h"

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
    std::vector<int64_t> selfShape = {1, 1, 3};
    std::vector<int64_t> outShape = {1, 1, 6};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};

    // Create self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> outArraySize = {3};
    const aclIntArray* outputSize = aclCreateIntArray(outArraySize.data(), outArraySize.size());
    CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    std::vector<int64_t> inputArraySize = {1, 1, 6};
    const aclIntArray* inputSize = aclCreateIntArray(inputArraySize.data(), inputArraySize.size());
    CHECK_RET(inputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the CANN operator library API; change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnUpsampleLinear1dBackward
    ret = aclnnUpsampleLinear1dBackwardGetWorkspaceSize(self, outputSize, inputSize, true, 0.5, out, &workspaceSize,
                                                        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleLinear1dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnUpsampleLinear1dBackward
    ret = aclnnUpsampleLinear1dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleLinear1dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar; modify according to the specific API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclDestroyIntArray(outputSize);
    aclDestroyIntArray(inputSize);

    // 7. Release device resources; modify according to the specific API definition.
    aclrtFree(selfDeviceAddr);
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
