# aclnnRoiAlignV2Backward

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

Backward propagation of [aclnnRoiAlignV2](../../roi_align/docs/aclnnRoiAlignV2.md). RoiAlign is a pooling layer that handles feature maps of non-uniform input sizes and outputs a fixed-size feature map.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnRoiAlignV2BackwardGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnRoiAlignV2Backward API to perform the computation.

```Cpp
aclnnStatus aclnnRoiAlignV2BackwardGetWorkspaceSize(
  const aclTensor*        gradOutput,
  const aclTensor*        boxes,
  const aclIntArray*      inputShape,
  int64_t                 pooledHeight,
  int64_t                 pooledWidth,
  float                   spatialScale,
  int64_t                 samplingRatio,
  bool                    aligned,
  aclTensor*              gradInput,
  uint64_t*               workspaceSize,
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnRoiAlignV2Backward(
  void*                   workspace,
  uint64_t                workspaceSize,
  aclOpExecutor*          executor,
  aclrtStream             stream)
```

## aclnnRoiAlignV2BackwardGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1570px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 298px">
  <col style="width: 184px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-5agr">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimensions (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">gradOutput(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input of backward propagation.</td>
      <td class="tg-0pky">Must have the same data type as boxes and gradInput.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">NCHW</td>
      <td class="tg-0pky">4-dimensional, shape (K, C, pooledHeight, pooledWidth)<br>Indicates that the input gradient tensor of backward propagation has K elements in one batch, each element has C feature maps of size pooledHeight * pooledWidth.<br>K must be consistent with the 0th dimension of boxes.</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">boxes(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">ROI box coordinates.</td>
      <td class="tg-0pky">Must have the same data type as gradOutput and gradInput.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2-dimensional, shape (K, 5)<br>5 represents the box-related information (image_id, x1, y1, x2, y2).<br>image_id value range [0, B), rounded down to the image id, where B is the first value of inputShape.<br>Coordinates satisfy 0 <= x1 <= x2 <= inputWidth/spatialScale and 0 <= y1 <= y2 <= inputHeight/spatialScale.</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">inputShape(aclIntArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">The shape of the forward input, used to specify the output shape of backward propagation.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT32, INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">size is 4, with values (B, C, inputHeight, inputWidth)<br>Indicates that the input tensor of forward RoiAlign has B images in one batch, each image has C feature maps of size inputHeight * inputWidth.</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">pooledHeight(int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The height of the output image after forward RoiAlign pooling.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">pooledWidth(int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The width of the output image after forward RoiAlign pooling.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">spatialScale(float)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">A multiplicative spatial scale factor that converts ROI coordinates from their input spatial scale to the scale used for pooling, that is, the spatial scale of the input feature map X relative to the input image.</td>
      <td class="tg-0pky">Must be greater than 0.</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">samplingRatio(int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">In RoiAlign, the sampling frequency used to compute each output element in the H and W directions.</td>
      <td class="tg-0pky">Must be greater than or equal to 0.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">aligned(bool)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">If false, aligns with the <a href="../../roi_align/docs/aclnnRoiAlign.md">aclnnRoiAlign</a> version implementation.<br>If true, the box coordinate pixels are offset by -0.5 to better align adjacent pixel indices.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">BOOL</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">gradInput(aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output of backward propagation.</td>
      <td class="tg-0pky">Must have the same data type as gradOutput and boxes.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">NCHW</td>
      <td class="tg-0pky">4-dimensional, shape (B, C, inputHeight, inputWidth)</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize(uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size that needs to be allocated on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor(aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which contains the operator computation flow.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Error Code:**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and reports an error in the following scenarios:

<table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 290px">
  <col style="width: 134px">
  <col style="width: 844px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed gradOutput, boxes, inputShape, and gradInput are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and data formats of gradOutput and gradInput are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput, boxes, inputShape, and gradInput do not meet the constraint restrictions.</td>
    </tr>
    <tr>
      <td>spatialScale must be greater than 0, and samplingRatio must be greater than or equal to 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnRoiAlignV2Backward

- **Parameters**

<table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 170px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>Size of the workspace applied for on the Device, obtained through the first-phase API aclnnRoiAlignV2BackwardGetWorkspaceSize.</td>
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

- **Error Code:**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRoiAlignV2Backward defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_align_v2_backward.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device-side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from Host-side memory to Device-side memory
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

template <typename T>
int CreateAclNchTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device-side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from Host-side memory to Device-side memory
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

int main() {
  // 1. (Boilerplate) device/stream initialization; refer to the acl API manual
  // Fill in deviceId according to your actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
  std::vector<int64_t> gradOutputShape = {1, 1, 3, 3};
  std::vector<int64_t> boxesShape = {1, 5};
  std::vector<int64_t> inputShape = {1, 1, 6, 6};

  void* gradOutputDeviceAddr = nullptr;
  void* boxesDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* boxes = nullptr;
  aclTensor* gradInput = nullptr;

  std::vector<float> gradOutputHostData = {4.5, 6.5, 8.5, 16.5, 18.5, 20.5, 28.5, 30.5, 32.5};
  std::vector<float> boxesHostData = {0.0, -2.0, -2.0, 22.0, 22.0};
  std::vector<float> gradInputHostData = {1.125, 1.125, 1.625, 1.625, 2.125, 2.125, 1.125, 1.125, 1.625, 1.625, 2.125, 2.125,
                                    4.125, 4.125, 4.625, 4.625, 5.125, 5.125, 4.125, 4.125, 4.625, 4.625, 5.125, 5.125,
                                    7.125, 7.125, 7.625, 7.625, 8.125, 8.125, 7.125, 7.125, 7.625, 7.625, 8.125, 8.125};

  // Create gradOutput aclTensor
  ret = CreateAclNchTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create boxes aclTensor
  ret = CreateAclTensor(boxesHostData, boxesShape, &boxesDeviceAddr, aclDataType::ACL_FLOAT, &boxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create inputShape aclIntArray
  const aclIntArray *inputShapeArray = aclCreateIntArray(inputShape.data(), inputShape.size());
  CHECK_RET(inputShapeArray != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  // Create gradInput aclTensor
  ret = CreateAclNchTensor(gradInputHostData, inputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int64_t pooledHeight = 3;
  int64_t pooledWidth = 3;
  int64_t samplingRatio = 2;
  float spatialScale = 0.25f;
  bool aligned = false;

  // 3. Call the CANN operator library API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRoiAlignV2Backward
  ret = aclnnRoiAlignV2BackwardGetWorkspaceSize(gradOutput, boxes, inputShapeArray, pooledHeight, pooledWidth, spatialScale,
                                              samplingRatio, aligned, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlignV2BackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRoiAlignV2Backward
  ret = aclnnRoiAlignV2Backward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlignV2Backward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
  auto size = GetShapeSize(inputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    gradInputDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor; modify according to the specific API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(boxes);
  aclDestroyIntArray(inputShapeArray);
  aclDestroyTensor(gradInput);

  // 7. Release device resources; modify according to the specific API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(boxesDeviceAddr);
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
