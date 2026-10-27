# aclnnCIoU

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

* Description: A loss function used for bounding box regression. Based on IoU, it also accounts for the center-point distance, aspect ratio, and overlap area, providing a more comprehensive measure of the difference between the predicted and ground-truth boxes.
* Formula:

$$
CIoU = IoU - \frac{\rho^2(b^p, b^g)}{c^2} - \alpha v \\
v = \frac{4}{\pi^2}(arctan(\frac{w^g}{h^g}) - arctan(\frac{w^p}{h^p}))^2 \\
\alpha = \frac{v}{1 - IoU + v} \\
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnCIoUGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnCIoU API to perform the computation.

```cpp
aclnnStatus aclnnCIoUGetWorkspaceSize(
  const aclTensor   *bBoxes,
  const aclTensor   *gtBoxes,
  bool               trans,
  bool               isCross,
  const char        *mode,
  aclTensor         *overlap,
  aclTensor         *atanSub,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor);
```

```cpp
aclnnStatus aclnnCIoU(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  aclrtStream        stream);
```

## aclnnCIoUGetWorkspaceSize

- **Parameters**

<table style="undefined;table-layout: fixed; width: 1533px"><colgroup>
  <col style="width: 161px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 290px">
  <col style="width: 252px">
  <col style="width: 128px">
  <col style="width: 149px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-Contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>bBoxes</td>
      <td>Input</td>
      <td>Predicted bounding boxes.</td>
      <td>A 2D tensor with shape [4, M].</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gtBoxes</td>
      <td>Input</td>
      <td>Ground truth bounding box.</td>
      <td>2D tensor of shape [4, N].</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>trans</td>
      <td>Input</td>
      <td>Specifies the format of the bounding box.</td>
      <td>true: Specifies the input format as [x, y, w, h].<br>false: Specifies the input format as [x0, y0, x1, y1].</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isCross</td>
      <td>Input</td>
      <td>Specifies whether to perform cross operation between bBoxes and gtBoxes.</td>
      <td>true: the output shape is [M, N].<br>false: the output shape is [1, N].</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Input</td>
      <td>Used to select the calculation mode: "iou" or "iof".</td>
      <td>-</td>
      <td>CHAR*</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>overlap</td>
      <td>Output</td>
      <td>The intersection over union or foreground intersection ratio computed from the two inputs.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>atanSub</td>
      <td>Output</td>
      <td>Difference between the two arctan values during computation.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the operator executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and reports an error in the following scenarios:

<table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
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
      <td>bBoxes, gtBoxes, overlap, or atanSub is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>bBoxes, gtBoxes, overlap, or atanSub is not two-dimensional.</td>
    </tr>
    <tr>
      <td>The data types of bBoxes, gtBoxes, overlap, and atanSub are inconsistent.</td>
    </tr>
    <tr>
      <td>The data type or data format of bBoxes, gtBoxes, overlap, or atanSub is not within the supported range.</td>
    </tr>
    <tr>
      <td>The first dimension of bBoxes or gtBoxes is not 4.</td>
    </tr>
    <tr>
      <td>The second dimension of bBoxes or gtBoxes is not a multiple of 1024.</td>
    </tr>
    <tr>
      <td>The first dimension of overlap or atanSub is not 1.</td>
    </tr>
    <tr>
      <td>The second dimensions of bBoxes, gtBoxes, overlap, and atanSub are not equal.</td>
    </tr>
    <tr>
      <td>isCross is not false.</td>
    </tr>
    <tr>
      <td>mode is neither "iou" nor "iof".</td>
    </tr>
  </tbody>
  </table>

## aclnnCIoU

- **Parameters**

<table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 319px">
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
      <td>The workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained through the first-phase API aclnnCIoUGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream on which the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic Computation:
  - aclnnCIoU defaults to a deterministic implementation.
- If the input format is [x0, y0, x1, y1], (x0, y0) and (x1, y1) indicate the top-left and bottom-right corners of the bounding box, respectively, and must satisfy x1 > x0 and y1 > y0.
- M and N must be consistent.
- `isCross` currently supports only `false`.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ciou.h"

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
  // Boilerplate; AscendCL initialization
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

int main() {
  // 1. (Boilerplate) device/stream initialization; refer to the AscendCL external interface list
  // Fill in deviceId according to your actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
  // input
  std::vector<float> bBoxesHostData(4096, 1);
  std::vector<float> gtBoxesHostData(4096, 2);
  std::vector<float> overlapHostData(1024, 0);
  std::vector<float> atanSubHostData(1024, 0);
  std::vector<int64_t> bBoxesShape = {4, 1024};
  std::vector<int64_t> gtBoxesShape = {4, 1024};
  std::vector<int64_t> overlapShape = {1, 1024};
  std::vector<int64_t> atanSubShape = {1, 1024};
  void* bBoxesDeviceAddr = nullptr;
  void* gtBoxesDeviceAddr = nullptr;
  void* overlapDeviceAddr = nullptr;
  void* atanSubDeviceAddr = nullptr;
  aclTensor* bBoxes = nullptr;
  aclTensor* gtBoxes = nullptr;
  aclTensor* overlap = nullptr;
  aclTensor* atanSub = nullptr;

  ret = CreateAclTensor(bBoxesHostData, bBoxesShape, &bBoxesDeviceAddr, aclDataType::ACL_FLOAT, &bBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gtBoxesHostData, gtBoxesShape, &gtBoxesDeviceAddr, aclDataType::ACL_FLOAT, &gtBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(overlapHostData, overlapShape, &overlapDeviceAddr, aclDataType::ACL_FLOAT, &overlap);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(atanSubHostData, atanSubShape, &atanSubDeviceAddr, aclDataType::ACL_FLOAT, &atanSub);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // attr
  bool trans = false;
  bool isCross = false;
  const char* mode = "iou";

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. Call the CANN operator library API; change it to the specific Api name.
  ret = aclnnCIoUGetWorkspaceSize(bBoxes, gtBoxes, trans, isCross, mode, overlap, atanSub, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnCIoUGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // aclnnCIoU
  ret = aclnnCIoU(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnCIoU failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
  auto overlapSize = GetShapeSize(overlapShape);
  std::vector<float> overlapData(overlapSize, 0);
  ret = aclrtMemcpy(overlapData.data(), overlapData.size() * sizeof(overlapData[0]), overlapDeviceAddr,
                    overlapSize * sizeof(overlapData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy overlapData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < overlapSize; i++) {
    LOG_PRINT("overlap[%ld] is: %f\n", i, overlapData[i]);
  }

  auto atanSubsize = GetShapeSize(atanSubShape);
  std::vector<float> atanSubData(atanSubsize, 0);
  ret = aclrtMemcpy(atanSubData.data(), atanSubData.size() * sizeof(atanSubData[0]), atanSubDeviceAddr,
                    atanSubsize * sizeof(atanSubData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy atanSubData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < atanSubsize; i++) {
    LOG_PRINT("atanSub[%ld] is: %f\n", i, atanSubData[i]);
  }

  // 6. Release aclTensor and aclScalar; modify according to the specific API definition.
  aclDestroyTensor(bBoxes);
  aclDestroyTensor(gtBoxes);
  aclDestroyTensor(overlap);
  aclDestroyTensor(atanSub);

  // 7. Release device resources
  aclrtFree(bBoxesDeviceAddr);
  aclrtFree(gtBoxesDeviceAddr);
  aclrtFree(overlapDeviceAddr);
  aclrtFree(atanSubDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
