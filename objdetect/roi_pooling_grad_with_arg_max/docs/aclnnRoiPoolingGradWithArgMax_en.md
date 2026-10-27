# aclnnRoiPoolingGradWithArgMax

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Implement the backward of RoiPoolingWithArgMax. Traverse the pooling result of each ROI, accumulate the backward gradient contribution on the feature map coordinates, that is, complete the backward computation over the entire image.
- Formula:

  $$
  \frac{\partial L}{\partial x_i} = \sum_{r}\sum_{j}[i = i^*(r,j)]\frac{\partial L}{\partial y_{rj}}
  $$

Where,

  $$
  [i = i^*(r,j)]  = \begin{cases} 1, & i^*(r,j) \geq 1 \\ 0, & otherwise \end{cases}
  $$

The decision function `[i = i^*(r,j)]` indicates whether node i is selected as the maximum output by the j-th output node of candidate region r.

## Function Prototype

The operator execution API is a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize" API to obtain the input parameters and compute the required workspace size according to the computation flow, then call the "aclnnRoiPoolingGradWithArgMax" API to perform the computation.

```cpp
aclnnStatus aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize(
    const aclTensor*      gradOutput,
    const aclTensor*      gradInputRef,
    const aclTensor*      rois,
    const aclTensor*      argmax,
    int64_t               pooledH,
    int64_t               pooledW,
    double                spatialScale,
    uint64_t*             workspaceSize,
    aclOpExecutor**       executor);
```

```cpp
aclnnStatus aclnnRoiPoolingGradWithArgMax(
  void*                   workspace,
  uint64_t                workspaceSize,
  aclOpExecutor*          executor,
  aclrtStream             stream)
```

## aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize

- **Parameters**

<table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
    <col style="width: 200px">
    <col style="width: 120px">
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 212px">
    <col style="width: 120px">
    <col style="width: 250px">
    <col style="width: 145px">
    </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Note</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-Contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput (aclTensor*)</td>
      <td>Input</td>
      <td>Gradient input.</td>
      <td>-</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4-dimensional, with shape (roisN, C, pooledH, pooledW)</td>
      <td>√</td>
    </tr>
      <tr>
      <td>gradInputRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Output result.</td>
      <td>-</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>4D, shape (N, C, H, W)</td>
      <td>√</td>
    </tr>
      <tr>
      <td>rois (aclTensor*)</td>
      <td>Input</td>
      <td>ROI region.</td>
      <td>-</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2D, shape is (roisN, 5)<br>5 representing (batchId, x1, x2, y1, y2)</td>
      <td>√</td>
    </tr>
      <tr>
      <td>argmax (aclTensor*)</td>
      <td>Input</td>
      <td>Specifies the index of the target gradient.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>4-dimensional, with shape (roisN, C, pooledH, pooledW)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pooledH (int64_t)</td>
      <td>Attribute</td>
      <td>Pooling height.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>pooledW (int64_t)</td>
      <td>Attribute</td>
      <td>Pooling width.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatialScale (double)</td>
      <td>Attribute</td>
      <td>Scaling ratio for mapping input coordinates to ROI coordinates.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that the user needs to allocate on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the operator executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  Returns the aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and reports an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1124px"><colgroup>
  <col style="width: 284px">
  <col style="width: 124px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A null pointer is passed for gradOutput, rois, argmax, or gradInputRef.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data types of gradOutput, rois, argmax, and gradInputRef are not within the supported range.</td>
    </tr>
    <tr>
      <td>gradOutput, argmax, and gradInputRef have the same data type.</td>
    </tr>
    <tr>
      <td>The shape size of gradOutput, argmax, and gradInputRef is 4, and the shape size of rois is 2.</td>
    </tr>
    <tr>
      <td>gradOutput, argmax, and rois have equal shape[0].</td>
    </tr>
    <tr>
      <td>gradOutput and argmax have equal shape[1].</td>
    </tr>
    <tr>
      <td>The shape[2] of gradOutput and argmax equals pooledH, and shape[3] equals pooledW.</td>
    </tr>
    <tr>
      <td>The values of rois are greater than or equal to 0.</td>
    </tr>
    <tr>
      <td>pooledH and pooledW are greater than 0.</td>
    </tr>
    <tr>
      <td>rois[:, 1] is less than rois[:, 2] and rois[:, 3] is less than rois[:, 4].</td>
    </tr>
    <tr>
      <td>rois.shape[0] and gradOutput.shape[0] are less than or equal to 1024.</td>
    </tr>
    <tr>
      <td>gradInputRef.shape[1] equals gradOutput.shape[1].</td>
    </tr>
  </tbody>
  </table>

## aclnnRoiPoolingGradWithArgMax

* **Parameters**:

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
      <col style="width: 150px">
      <col style="width: 100px">
      <col style="width: 650px">
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
          <td>Size of the workspace requested on the Device side, obtained from the first-phase API aclnnRoiPoolingGradWithArgMax.</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Input</td>
          <td>Operator executor, which encapsulates the operator computation process.</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>Input</td>
          <td>Specifies the stream for task execution.</td>
        </tr>
      </tbody></table>

* **Return Value**:

  Returns the aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The data types of gradOutput, rois, argmax, and gradInputRef are within the supported range.
- gradOutput, argmax, and gradInputRef have the same data type.
- The shape size of gradOutput, argmax, and gradInputRef is 4, and the shape size of rois is 2.
- gradOutput, argmax, and rois have equal shape[0].
- gradOutput and argmax have equal shape[1].
- The shape[2] of gradOutput and argmax equals pooledH and shape[3] equals pooledW.
- The values of rois are greater than or equal to 0.
- pooledH and pooledW are greater than 0.
- rois[:, 1] is less than rois[:, 2] and rois[:, 3] is less than rois[:, 4].
- rois.shape[0] and gradOutput.shape[0] are less than or equal to 1024.
- Deterministic computation:

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 training products/Atlas A2 inference products, Atlas A3 training products/Atlas A3 inference products</term>: aclnnRoiPoolingGradWithArgMax defaults to a deterministic implementation.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnRoiPoolingGradWithArgMax defaults to a non-deterministic implementation. Deterministic computation cannot be enabled.
  <!-- end id8 -->

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_pooling_grad_with_arg_max.h"
#include <iostream>
using namespace std;


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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate; initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 2. Allocate device-side memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 3. Call aclrtMemcpy to copy data from host to device-side memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of a contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. Call acl to perform device/stream initialization
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
    aclTensor* gradOutput = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    std::vector<int64_t> gradOutputShape = {1, 32, 2, 2};
    std::vector<float> gradOutputHostData(128, 1.0); // 128: creates a vector with 1*32*2*2=128 elements
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* gradInputRef = nullptr;
    void* gradInputRefDeviceAddr = nullptr;
    std::vector<int64_t> gradInputRefShape = {4, 32, 3, 3};
    std::vector<float> gradInputRefHostData(1152, 1.0);
    ret = CreateAclTensor(gradInputRefHostData, gradInputRefShape, &gradInputRefDeviceAddr, aclDataType::ACL_FLOAT, &gradInputRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* rois = nullptr;
    void* roisDeviceAddr = nullptr;
    std::vector<int64_t> roisShape = {1, 5};
    std::vector<float> roisHostData = {0.0, 0.0, 1.0, 0.0, 1.0};
    ret = CreateAclTensor(roisHostData, roisShape, &roisDeviceAddr, aclDataType::ACL_FLOAT, &rois);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* argmax = nullptr;
    void* argmaxDeviceAddr = nullptr;
    std::vector<int64_t> argmaxShape = {1, 32, 2, 2};
    std::vector<int32_t> argmaxHostData(128, 3.0);
    ret = CreateAclTensor(argmaxHostData, argmaxShape, &argmaxDeviceAddr, aclDataType::ACL_INT32, &argmax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int32_t pooledH = 2;
    int32_t pooledW = 2;
    double spatialScale = 1.0;

    // 3. Call the CANN operator library API; change it to the specific Api name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 4. Call the first-phase API of aclnnRoiPoolingGradWithArgMax
    ret = aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize(gradOutput, gradInputRef, rois, argmax, pooledH, pooledW, spatialScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. Call the second-phase API of aclnnRoiPoolingGradWithArgMax
    ret = aclnnRoiPoolingGradWithArgMax(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingGradWithArgMax failed. ERROR: %d\n", ret); return ret);

    // 6. (Boilerplate) Synchronize and wait for task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnRoiPoolingGradWithArgMax run success.\n");

    // 7. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
    PrintOutResult(gradInputRefShape, &gradInputRefDeviceAddr);

    // 8. Release aclTensor; modify according to the specific API definition.
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(gradInputRef);
    aclDestroyTensor(rois);
    aclDestroyTensor(argmax);

    // 9. Release device resources
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(gradInputRefDeviceAddr);
    aclrtFree(roisDeviceAddr);
    aclrtFree(argmaxDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 10. acl deinitialization
    aclFinalize();

    return 0;
}
```
