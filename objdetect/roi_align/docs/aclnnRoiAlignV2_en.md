# aclnnRoiAlignV2

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

RoiAlign is a pooling layer used for feature maps with non-uniform input sizes, and outputs feature maps of fixed size. [aclnnRoiAlign](./aclnnRoiAlign.md) aligns with the ONNX opset 10 operator prototype, while aclnnRoiAlignV2 aligns with the torchvision operator prototype. aclnnRoiAlignV2 uses boxes to replace the rois and batch_indices of aclnnRoiAlign, adds the aligned input parameter, and removes the mode input parameter, defaulting to the mode="avg" scenario.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnRoiAlignV2GetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnRoiAlignV2 API to perform the computation.

```Cpp
aclnnStatus aclnnRoiAlignV2GetWorkspaceSize(
  const aclTensor*        self,
  const aclTensor*        boxes,
  int64_t                 pooledHeight,
  int64_t                 pooledWidth,
  float                   spatialScale,
  int64_t                 samplingRatio,
  bool                    aligned,
  aclTensor*              out,
  uint64_t*               workspaceSize,
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnRoiAlignV2(
  void*                   workspace,
  uint64_t                workspaceSize,
  aclOpExecutor*          executor,
  aclrtStream             stream)
```

## aclnnRoiAlignV2GetWorkspaceSize

- **Parameters**

<table class="tg" style="undefined;table-layout: fixed; width: 1540px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 268px">
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
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Image feature map input.</td>
      <td class="tg-0pky">Must be the same as the data types of boxes and out.</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">NCHW</td>
      <td class="tg-0pky">4D, with shape (B, C, inputHeight, inputWidth)<br>It indicates that there are B images in a batch of the input tensor, and each image has C feature maps whose size is inputHeight x inputWidth.<br>**B**, **inputHeight**, and **inputWidth** cannot be 0D.</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">boxes (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Coordinates of the region of interest (ROI) box.</td>
      <td class="tg-0pky">Must be the same as the data types of self and out.</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2 dimension: (K, 5)<br>5 indicates the box information (image_id, x1, y1, x2, y2).<br>K must be consistent with the 0th dimension of the output.<br>The value of image_id is in the range of [0, B), which is rounded down to the image ID. B indicates the size of the 0th dimension of self. The coordinates must meet the following conditions: 0 &lt;= x1 &lt;= x2 &lt;= inputWidth/spatialScale, 0 <= y1 <= y2 <= inputHeight/spatialScale.</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">pooledHeight (int64_t)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Height of the output image after pooling.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">pooledWidth (int64_t)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Width of the output image after pooling.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">spatialScale (float)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Multiplicative spatial scale factor, which converts the ROI coordinates from the input spatial scale to the scale used for pooling. That is, the spatial scale of the input feature map X relative to the input image.</td>
      <td class="tg-0lax">Must be greater than 0.</td>
      <td class="tg-0lax">FLOAT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">samplingRatio (int64_t)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Sampling frequency used to calculate each output element in the H and W directions.</td>
      <td class="tg-0lax">Must be greater than or equal to 0.</td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">aligned (bool)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">If the value is false, the implementation of the <a href="./aclnnRoiAlign.md">aclnnRoiAlign</a> version is aligned.<br>If the value is true, the pixel offset of the box coordinate is -0.5 to better align the adjacent pixel indexes.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">BOOL</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">out (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Output after pooling.</td>
      <td class="tg-0lax">Must be consistent with the data types of self and boxes.</td>
      <td class="tg-0lax">FLOAT, FLOAT16</td>
      <td class="tg-0lax">NCHW</td>
      <td class="tg-0lax">4D, with shape (K, C, pooledHeight, pooledWidth)<br>Indicates that there are K elements in a batch of the output tensor, and each element has C feature maps whose size is pooledHeight x pooledWidth.</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the size of the workspace to be applied for on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the operator executor, including the operator computation process.</td>
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
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input self, boxes, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type and data format of self, boxes, and out are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self, boxes, and out do not meet the constraint restrictions.</td>
    </tr>
    <tr>
      <td>spatialScale must be greater than 0, and samplingRatio must be greater than or equal to 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnRoiAlignV2

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size to be applied for on the Device side, obtained through the first-phase API aclnnRoiAlignV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which encapsulates the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream on which the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Error Code:**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRoiAlignV2 defaults to a deterministic implementation.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_align_v2.h"

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
  std::vector<int64_t> selfShape = {1, 1, 6, 6};
  std::vector<int64_t> boxesShape = {1, 5};
  std::vector<int64_t> outShape = {1, 1, 3, 3};

  void* selfDeviceAddr = nullptr;
  void* boxesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* boxes = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
  std::vector<float> boxesHostData = {0.0, -2.0, -2.0, 22.0, 22.0};
  std::vector<float> outHostData = {4.5, 6.5, 8.5, 16.5, 18.5, 20.5, 28.5, 30.5, 32.5};

  // Create self aclTensor
  ret = CreateAclNchTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create boxes aclTensor
  ret = CreateAclTensor(boxesHostData, boxesShape, &boxesDeviceAddr, aclDataType::ACL_FLOAT, &boxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor
  ret = CreateAclNchTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int64_t pooledHeight = 3;
  int64_t pooledWidth = 3;
  int64_t samplingRatio = 2;
  float spatialScale = 0.25f;
  bool aligned = false;

  // 3. Call the CANN operator library API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRoiAlignV2
  ret = aclnnRoiAlignV2GetWorkspaceSize(self, boxes, pooledHeight, pooledWidth, spatialScale,
                                              samplingRatio, aligned, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlignV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnRoiAlignV2
  ret = aclnnRoiAlignV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiAlignV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor; modify according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(boxes);
  aclDestroyTensor(out);

  // 7. Release device resources; modify according to the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(boxesDeviceAddr);
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
