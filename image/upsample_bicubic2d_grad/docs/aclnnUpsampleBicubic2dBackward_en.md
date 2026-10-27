# aclnnUpsampleBicubic2dBackward

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

- Description: Backward propagation of [aclnnUpsampleBicubic2d](../../upsample_bicubic2d/docs/aclnnUpsampleBicubic2d.md). If the shape of the input tensor is (N, C, H, W), the shape of the output tensor is (N, C, inputSize[2], inputSize[3]).

- Formula: For a 2D interpolation point $(N, C, h, w)$, the interpolation $gradInput(N, C, h, w)$ can be expressed as:

  $$
  {gradInput(N, C, h, w)}=\sum_{i=0}^{3}\sum_{j=0}^{3}{W(i, j)}*{f(h_i, w_j)}
  $$

  $$
  scaleH =\begin{cases}
  (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  inputSize[2] / outputSize[0] & otherwise
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  inputSize[3] / outputSize[1] & otherwise
  \end{cases}
  $$

  where:
  - alignCorners = true means the corner pixels of the input and output tensors are aligned; alignCorners = false means the edge pixels of the input and output tensors are aligned.
  - i and j are the index variables of $W(i, j)$.
  - $f(h_i, w_j)$ is the pixel value of gradOut at $(h_i, w_j)$.
  - $W(i, j)$ is the weight of bicubic anti-aliasing interpolation, defined as:

    $$
    W(d) =\begin{cases}
    (a+2)|d|^3-(a+3)|d|^2+1 & |d|\leq1 \\
    a|d|^3-5a|d|^2+8a|d|-4a & 1<|d|<2 \\
    0 & otherwise
    \end{cases}
    $$

    where:
    - $a=-0.75$
    - $d = |(h, w) - (h_i, w_j)|$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnUpsampleBicubic2dBackwardGetWorkspaceSize" API to obtain the input parameters and compute the required workspace size according to the computation flow, then call the "aclnnUpsampleBicubic2dBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnUpsampleBicubic2dBackwardGetWorkspaceSize(
  const aclTensor*   gradOut,
  const aclIntArray* outputSize,
  const aclIntArray* inputSize,
  const bool         alignCorners,
  double             scalesH,
  double             scalesW,
  aclTensor*         gradInput,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnUpsampleBicubic2dBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnUpsampleBicubic2dBackwardGetWorkspaceSize

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
      <td>Indicates the gradient tensor for backward computation, corresponding to `gradOut` in the formula description.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type is the same as that of `gradInput`.</li><li>When the data format is ND, it is processed in NCHW format by default.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW, ND, NHWC</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize(aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the spatial size of input `gradOut` in the H and W dimensions. Corresponds to `outputSize` in the formula.</td>
      <td>size is 2, and all elements are greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize(aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the spatial size of output `gradInput` in the N, C, H, W or N, H, W, C dimensions. Corresponds to `inputSize` in the formula.</td>
      <td>size is 4, and all elements are greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners(bool)</td>
      <td>Input</td>
      <td>Determines whether to align corner pixels, corresponding to `alignCorners` in the formula.</td>
      <td>If True, the corner pixels of the input and output tensors are aligned; otherwise, they are not aligned.</td>
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
      <td><ul><li>Empty tensors are not supported.</li><li>The data type and data format are the same as those of `gradOut`.</li><li>The N and C axes of the shape are the same as those of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW, ND, NHWC</td>
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
    - The parameters `gradOut` and `gradInput` do not support the BFLOAT16 data type.
    - The parameters `gradOut` and `gradInput` do not support the NHWC data format.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    The data format of parameters `gradOut` and `gradInput` does not support NHWC.
  <!-- end id8 -->

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
      <td>The passed gradOut, outputSize, inputSize, or gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data type or data format of gradOut is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of gradOut and gradInput are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension of gradOut is not 4.</td>
    </tr>
    <tr>
      <td>The size of outputSize is not equal to 2.</td>
    </tr>
    <tr>
      <td>An element value of outputSize is less than 1.</td>
    </tr>
    <tr>
      <td>The size of inputSize is not equal to 4.</td>
    </tr>
    <tr>
      <td>An element value of inputSize is less than 1.</td>
    </tr>
    <tr>
      <td>The sizes of gradOut and inputSize differ in the N and C dimensions.</td>
    </tr>
    <tr>
      <td>The sizes of gradOut in the H and W dimensions are inconsistent with outputSize[0] and outputSize[1].</td>
    </tr>
    <tr>
      <td>The dimension sizes of the N and C axes of gradOut and gradInput are not equal.</td>
    </tr>
    <tr>
      <td>The data formats of gradOut and gradInput are not within the supported range.</td>
    </tr>
  </tbody></table>

## aclnnUpsampleBicubic2dBackward

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
      <td>The workspace size applied on the Device side, obtained by the first-phase API aclnnUpsampleBicubic2dBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which contains the operator computation flow.</td>
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

<!-- npu="A3,910b" id9 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:

  - All dimension values of gradOut and gradInput are less than or equal to (2^31-1).
  - $N * C * outputSize\_H$ <= 2^31-1
    where:
      - N represents the N axis of the input and output.
      - C represents the C axis of the input and output.

<!-- end id9 -->
- Parameters inputSize, outputSize, scalesH, and scalesW must satisfy the following constraints:

  $$
  outputSize\_H = floor(inputSize\_H * scalesH)
  $$

  $$
  outputSize\_W = floor(inputSize\_W * scalesW)
  $$

- Deterministic computation:

  aclnnUpsampleBicubic2dBackward defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt. Deterministic computation is not supported when the following conditions are met:
  - $inputSize\_W$ > 130000
  - $inputSize\_H$/$outputSize\_H$ >=50
  - $inputSize\_W$/$outputSize\_W$ >=50 && $inputSize\_N$ * $inputSize\_C$ * $inputSize\_H$ > $inputSize\_W$ * 0.5
  - $inputSize\_H$/$outputSize\_H$ < 0.02 && $inputSize\_W$/$outputSize\_W$ < 0.02 && $inputSize\_N$ * $inputSize\_C$ * $inputSize\_H$ > $inputSize\_W$ * 10000

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_bicubic_2d_backward.h"

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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
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
    std::vector<int64_t> selfShape = {1, 1, 2, 2};
    std::vector<int64_t> outShape = {1, 1, 3, 3};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4.1};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Create self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> outArraySize = {2, 2};
    const aclIntArray* outputSize = aclCreateIntArray(outArraySize.data(), outArraySize.size());
    CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    std::vector<int64_t> inputArraySize = {1, 1, 3, 3};
    const aclIntArray* inputSize = aclCreateIntArray(inputArraySize.data(), inputArraySize.size());
    CHECK_RET(inputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the CANN operator library API; change it to the specific Api name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    const double scalesH = 1.1;
    const double scalesW = 1.1;
    // Call the first-phase API of aclnnUpsampleBicubic2dBackward
    ret = aclnnUpsampleBicubic2dBackwardGetWorkspaceSize(self, outputSize, inputSize, 1, scalesH, scalesW, out, &workspaceSize,
                                                         &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBicubic2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnUpsampleBicubic2dBackward
    ret = aclnnUpsampleBicubic2dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBicubic2dBackward failed. ERROR: %d\n", ret); return ret);

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

    // 6. Release aclTensor; modify according to the specific API definition.
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
