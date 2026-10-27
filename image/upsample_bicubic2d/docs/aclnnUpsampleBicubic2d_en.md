# aclnnUpsampleBicubic2d

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

- Description: Apply 2D bicubic upsampling to an input signal composed of multiple input channels. If the shape of input tensor x is (N, C, H, W), the shape of output tensor out is (N, C, outputSize[0], outputSize[1]).
- Formula: For a 2D interpolation point $(N, C, h, w)$, the interpolation $out(N, C, h, w)$ can be expressed as:

  $$
  {out(N, C, h, w)}=\sum_{i=0}^{3}\sum_{j=0}^{3}{W(i, j)}*{f(h_i, w_j)}
  $$

  $$
  scaleH =\begin{cases}
  (self.dim(2)-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  self.dim(2) / outputSize[0] & otherwise
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  (self.dim(3)-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  self.dim(3) / outputSize[1] & otherwise
  \end{cases}
  $$

  where:
  - alignCorners = true means the corner pixels of the input and output tensors are aligned; alignCorners = false means the edge pixels of the input and output tensors are aligned.
  - i and j are the index variables of $W(i, j)$.
  - $f(h_i, w_j)$ is the pixel value of the original image at $(h_i, w_j)$.
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

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnUpsampleBicubic2dGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnUpsampleBicubic2d API to perform the computation.

```Cpp
aclnnStatus aclnnUpsampleBicubic2dGetWorkspaceSize(
  const aclTensor*   self,
  const aclIntArray* outputSize,
  const bool         alignCorners,
  const double       scalesH,
  const double       scalesW,
  aclTensor*         out,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnUpsampleBicubic2d(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnUpsampleBicubic2dGetWorkspaceSize

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
      <th>Parameter Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the input tensor for upsampling, corresponding to `self` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>When the data format is ND, it is processed in NCHW format by default.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW, ND, NHWC</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize (aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the spatial size of `out` in the H and W dimensions. Corresponds to `outputSize` in the corresponding formula.</td>
      <td>size is 2, and each element is greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners (bool)</td>
      <td>Input</td>
      <td>Determines whether to align corner pixels, corresponding to `alignCorners` in the corresponding formula.</td>
      <td>If alignCorners is True, the corner pixels of the input and output tensors are aligned; otherwise, the edge pixels of the input and output tensors are aligned.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH(double)</td>
      <td>Input</td>
      <td>The multiplier for the height dimension of the specified spatial size, corresponding to `scalesH` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW (double)</td>
      <td>Input</td>
      <td>Specifies the width dimensional multiplier of the spatial size, corresponding to `scalesW` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the output tensor after upsampling, corresponding to `out` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type and data format must be consistent with `self`.</li><li>The shape supports 4 dimensions: (batch, channel, height, width), where batch and channel are derived from the 0th and 1st dimensions of the input `self`, respectively, and height and width are derived from the first and second values of `outputSize`, respectively.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW, ND, NHWC</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910,310p,310b" id7 -->
  - <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>:
    - Data type: The parameters `self` and `out` do not support BFLOAT16.
    - Data format: The parameters `self` and `out` do not support NHWC.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    The data format of parameters `self` and `out` does not support NHWC.
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
      <td>The passed self, outputSize, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data type of self is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The data format of self is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data formats of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of self and out are not 4-dimensional.</td>
    </tr>
    <tr>
      <td>The size of outputSize is not equal to 2.</td>
    </tr>
    <tr>
      <td>The size of self in the C, H, and W dimensions is not greater than 0.</td>
    </tr>
    <tr>
      <td>An element value of outputSize is not greater than 0.</td>
    </tr>
    <tr>
      <td>The size of out in the N and C dimensions is not equal to the size of self in the N and C dimensions.</td>
    </tr>
    <tr>
      <td>The size of out in the H and W dimensions is not equal to the corresponding element value in outputSize.</td>
    </tr>
    <tr>
      <td>self, outputSize, scalesH, and scalesW do not meet the constraints.</td>
    </tr>
  </tbody></table>

## aclnnUpsampleBicubic2d

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
      <td>Side-applied workspace memory address on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace side-applied on the Device, obtained by the first-phase API aclnnUpsampleBicubic2dGetWorkspaceSize.</td>
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

- Shape constraints for parameters `self` and `out`:
  - The value of each dimension is less than or equal to 2^20.
  - The N and C axes of parameter `out` are the same as those of `self`.
  - The memory usage must be less than 60 GB. The formula for memory usage is as follows:

    $$
    (self\_H * self\_W + out\_H * out\_W + self\_H * out\_W) * N * C  * sizeof(dtype) < 60 * 1024 * 1024 * 1024
    $$

    where:
    - N represents the N axis of the input and output.
    - C represents the C axis of the input and output.
    - dtype represents the data type of the input tensor.
  - N * C * self_H < 2^31
- When the value of scalesH is greater than 0, parameters self, outputSize, and scalesH must satisfy the following constraints:

  $$
  outputSize\_H = floor(self\_H * scalesH)
  $$

- When the value of scalesW is greater than 0, parameters self, outputSize, and scalesW must satisfy the following constraints:

  $$
  outputSize\_W = floor(self\_W * scalesW)
  $$

- Deterministic computation:
  - aclnnUpsampleBicubic2d defaults to a deterministic implementation.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_bicubic_2d.h"

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
    constexpr int64_t INPUT_H = 3;
    constexpr int64_t INPUT_W = 3;
    constexpr int64_t OUTPUT_H = 5;
    constexpr int64_t OUTPUT_W = 5;
    std::vector<int64_t> selfShape = {1, 1, INPUT_H, INPUT_W};
    std::vector<int64_t> outShape = {1, 1, OUTPUT_H, OUTPUT_W};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> outHostData(OUTPUT_H * OUTPUT_W, 0);
    std::vector<int64_t> outputSizeData = {OUTPUT_H, OUTPUT_W};
    bool alignCorners = false;
    double scalesH = 0.0;
    double scalesW = 0.0;
    // Create self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create outputSize aclIntArray
    const aclIntArray* outputSize = aclCreateIntArray(outputSizeData.data(), outputSizeData.size());
    CHECK_RET(outputSize != nullptr, return ret);

    // 3. Call the CANN operator library API; change it to the specific Api name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnUpsampleBicubic2d
    ret = aclnnUpsampleBicubic2dGetWorkspaceSize(self, outputSize, alignCorners, scalesH, scalesW, out, &workspaceSize,
                                                 &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBicubic2dGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnUpsampleBicubic2d
    ret = aclnnUpsampleBicubic2d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBicubic2d failed. ERROR: %d\n", ret); return ret);

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

    // 6. Release aclTensor and aclIntArray; modify according to the specific API definition.
    aclDestroyTensor(self);
    aclDestroyIntArray(outputSize);
    aclDestroyTensor(out);

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
