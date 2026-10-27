# aclnnNonMaxSuppression

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: For the detection boxes of each category, first sort them in descending order by confidence score, then sequentially select the box with the highest score as the selected box, and suppress candidate boxes whose IoU with the selected box exceeds `iouThreshold`, while removing output boxes whose score does not exceed `scoreThreshold`.

- Formula:
  - IoU formula: For two bounding boxes A and B, IoU is defined as the ratio of the intersection area to the union area:

    $$\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$$

  - Threshold comparison rule

    IoU suppression uses a **strictly greater than** comparison, which is equivalent to the following algebraic transformation:

    $$\text{IoU} > \text{iouThreshold} \quad \Leftrightarrow \quad \text{Area}(A \cap B) > \bigl(\text{Area}(A) + \text{Area}(B)\bigr) \times \frac{\text{iouThreshold}}{1 + \text{iouThreshold}}$$

     That is, when the intersection area exceeds the threshold on the right, the candidate box is suppressed.

    - **iouThreshold** (`aclFloatArray*`): Value range `[0, 1]`, controls the NMS suppression strength. A larger value retains more boxes.
    - **scoreThreshold** (`aclFloatArray*`): Value range `[0, 1]`, filters out low-score boxes in the final output; boxes with score ≤ `scoreThreshold` are filled with `-1` to indicate invalid.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnNonMaxSuppressionGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnNonMaxSuppression API to perform the computation.

```Cpp
aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(
  const aclTensor*        boxes,
  const aclTensor*        scores,
  aclIntArray*            maxOutputBoxesPerClass,
  aclFloatArray*          iouThreshold,
  aclFloatArray*          scoreThreshold,
  int32_t                 centerPointBox,
  aclTensor*              selectedIndices,
  uint64_t*               workspaceSize,
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnNonMaxSuppression(
  void*                   workspace,
  uint64_t                workspaceSize,
  aclOpExecutor*          executor,
  aclrtStream             stream)
```

## aclnnNonMaxSuppressionGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 238px">
  <col style="width: 300px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 224px">
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
      <td class="tg-0pky">boxes(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor.</td>
      <td class="tg-0pky">The maximum number of boxes per batch (spatial_dimension) is 50000.</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">[num_batches, spatial_dimension, 4]</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">scores(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor.</td>
      <td class="tg-0pky">The maximum number of boxes per batch (spatial_dimension) is 50000.</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">[num_batches, num_classes, spatial_dimension]</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">maxOutputBoxesPerClass(aclIntArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Indicates the maximum number of boxes selected per batch per category.</td>
      <td class="tg-0lax">The value must be greater than 0, with an upper limit of 700.</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">iouThreshold(aclFloatArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Indicates the threshold for determining whether a box overlaps too much relative to IoU.</td>
      <td class="tg-0lax">Value range [0, 1].</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">scoreThreshold(aclFloatArray*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Indicates the threshold for deciding when to remove a box based on the score.</td>
      <td class="tg-0lax">Value range [0, 1].</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">centerPointBox(int)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Used to determine the bounding box format.</td>
      <td class="tg-0lax"><ul><li>Value range [0, 1].</li><li>When equal to 0, mainly used for TensorFlow models; data is provided in the form (y1, x1, y2, x2), where (y1, x1) and (y2, x2) are the diagonal corner coordinates of the box, and the user must ensure x1 < x2 and y1 < y2.</li><li>When equal to 1, mainly used for PyTorch models; data is provided in the form (x_center, y_center, width, height).</li></ul></td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">selectedIndices(aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Output tensor.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">[num_selected_indices, 3]<br>Data is provided in the form [batch_index, class_index, box_index].</td>
      <td class="tg-0lax">√</td>
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

  The first-phase API completes input parameter validation and reports an error in the following cases:
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
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current product is not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A null pointer is passed for boxes, scores, or selectedIndices.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data types of boxes, scores, and maxOutputBoxesPerClass are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data formats of boxes, scores, and selectedIndices are not within the supported range.</td>
    </tr>
    <tr>
      <td>boxes and scores must be 3-dimensional.</td>
    </tr>
    <tr>
      <td>The 0th dimension of boxes must be equal to the 0th dimension of scores.</td>
    </tr>
    <tr>
      <td>The 1st dimension of boxes must be equal to the 2nd dimension of scores.</td>
    </tr>
    <tr>
      <td>The 2nd dimension of boxes must be equal to 4.</td>
    </tr>
    <tr>
      <td>The number of boxes per batch (spatial_dimension) exceeds 50000.</td>
    </tr>
    <tr>
      <td>The values of iouThreshold, scoreThreshold, centerPointBox, and maxOutputBoxesPerClass are not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnNonMaxSuppression

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
      <td>Size of the workspace applied on the Device side, obtained through the first-phase API aclnnNonMaxSuppressionGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which encapsulates the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream on which the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Error Code:**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The parameter maxOutputBoxesPerClass must be greater than 0, with an upper limit of 700. The data types of input parameters boxes and scores must be the same.
- The number of boxes per batch (spatial_dimension of boxes / spatial_dimension of scores) has an upper limit of 50000.
- In the FLOAT16 scenario, sorting and comparison benchmarking by the operator may introduce computation errors.
- Deterministic computation:
  - aclnnNonMaxSuppression defaults to a deterministic implementation.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_non_max_suppression.h"

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

template <typename T>
int64_t GetShapeSize(const std::vector<T>& shape) {
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
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray) {
  auto size = GetShapeSize(hostData) * sizeof(T);
  // Call aclrtMalloc to allocate device-side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from Host-side memory to Device-side memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Call the aclCreateIntArray API to create aclIntArray
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
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
  std::vector<int64_t> boxesShape = {1, 7, 4};
  std::vector<int64_t> scoresShape = {1, 1, 7};
  std::vector<int64_t> maxSizePerClassShape = {3};
  std::vector<int64_t> selectedIndicesShape = {3, 3};

  void* boxesDeviceAddr = nullptr;
  void* scoresDeviceAddr = nullptr;
  void* maxSizePerClassDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* boxes = nullptr;
  aclTensor* scores = nullptr;
  aclIntArray* maxOutputBoxesPerClass = nullptr;
  aclFloatArray* iouThd = nullptr;
  aclFloatArray* scoresThd = nullptr;
  aclTensor* selectedIndices = nullptr;

  std::vector<float> boxesHostData = {
    49.1, 32.4, 51.0, 35.9,
    49.3, 32.9, 51.0, 35.3,
    49.2, 31.8, 51.0, 35.4,
    35.1, 11.5, 39.1, 15.7,
    35.6, 11.8, 39.3, 14.2,
    35.3, 11.5, 39.9, 14.5,
    35.2, 11.7, 39.7, 15.7,
  };
  std::vector<float> scoresHostData = {0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3};
  std::vector<int64_t> maxOutputBoxesPerClassHostData = {3};
  std::vector<float> iouThresholdHostData = {0.6};
  std::vector<float> scoreThresholdHostData = {0};
  std::vector<int32_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create aclTensor: boxes
  ret = CreateAclTensor(boxesHostData, boxesShape, &boxesDeviceAddr, aclDataType::ACL_FLOAT, &boxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create aclTensor: scores
  ret = CreateAclTensor(scoresHostData, scoresShape, &scoresDeviceAddr, aclDataType::ACL_FLOAT, &scores);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create AclIntArray: maxOutputBoxesPerClass
  ret = CreateAclIntArray(maxOutputBoxesPerClassHostData, &maxSizePerClassDeviceAddr, &maxOutputBoxesPerClass);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create AclFloatArray: iouThreshold
  iouThd = aclCreateFloatArray(iouThresholdHostData.data(), iouThresholdHostData.size());
  CHECK_RET(iouThd != nullptr, return 0);

  // Create AclFloatArray: scoresThreshold
  scoresThd = aclCreateFloatArray(scoreThresholdHostData.data(), scoreThresholdHostData.size());
  CHECK_RET(scoresThd != nullptr, return 0);

  // Create aclTensor: selectedIndices
  ret = CreateAclTensor(outHostData, selectedIndicesShape, &outDeviceAddr, aclDataType::ACL_INT32, &selectedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create attr int: centerPointBox
  int64_t centerPointBox = 0;

  // 3. Call the CANN operator library API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNonMaxSuppression
  ret = aclnnNonMaxSuppressionGetWorkspaceSize(boxes, scores, maxOutputBoxesPerClass, iouThd, scoresThd, centerPointBox, selectedIndices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNonMaxSuppressionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnNonMaxSuppression
  ret = aclnnNonMaxSuppression(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNonMaxSuppression failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
  auto size = GetShapeSize(selectedIndicesShape);
  std::vector<int32_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. Release aclTensor; modify according to the specific API definition.
  aclDestroyTensor(boxes);
  aclDestroyTensor(scores);
  aclDestroyIntArray(maxOutputBoxesPerClass);
  aclDestroyFloatArray(iouThd);
  aclDestroyFloatArray(scoresThd);
  aclDestroyTensor(selectedIndices);

  // 7. Release device resources; modify according to the specific API definition.
  aclrtFree(boxesDeviceAddr);
  aclrtFree(scoresDeviceAddr);
  aclrtFree(maxSizePerClassDeviceAddr);
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
