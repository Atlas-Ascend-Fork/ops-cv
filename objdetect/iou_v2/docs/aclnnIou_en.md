# aclnnIou

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Operator function: For two sets of input rectangle boxes, compute the Intersection over Union (IoU) or Intersection over Foreground (IoF) to evaluate the overlap between the predicted box (bBox) and the ground-truth box (gtBox).
- Formula:

  $$
  IOU = \frac {Area_3} {Area_1 + Area_2 - Area_3} \\
  IOF = \frac {Area_3} {Area_2}
  $$

Where Area_1 is the area of bBox, Area_2 is the area of gtBox, and Area_3 is the area of their overlapping region. For definitions of x and y, see the Parameters.

  $$
  Area_1 = (X_1 - X_0)(Y_1 - Y_0) \\
  Area_2 = (X_3 - X_2)(Y_3 - Y_2) \\
  Area_3 = max( min(X_1, X_3) - max(X_0, X_2), 0 ) * max( min(Y_1, Y_3) - max(Y_0, Y_2), 0 )
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnIouGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnIou API to perform the computation.

```Cpp
aclnnStatus aclnnIouGetWorkspaceSize(
  const aclTensor*        bBoxes,
  const aclTensor*        gtBoxes,
  const char*             mode,
  float                   eps,
  bool                    aligned,
  aclTensor*              overlap,
  uint64_t*               workspaceSize,
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnIou(
  void*                   workspace,
  uint64_t                workspaceSize,
  aclOpExecutor*          executor,
  aclrtStream             stream)
```

## aclnnIouGetWorkspaceSize

- **Parameters**

<table class="tg" style="undefined;table-layout: fixed; width: 1575px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 200px">
  <col style="width: 400px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 120px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-5agr">Parameter Name</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (Shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">bBoxes (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Predicted bounding boxes.</td>
      <td class="tg-0pky"><ul><li>m in shape indicates the number of bounding boxes.</li><li>4 in shape refers to [x0, y0, x1, y1], where (x0, y0) and (x1, y1) represent the top-left and bottom-right corners of the bounding box, respectively, and must satisfy x1 > x0, y1 > y0.</li></ul></td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">is (m, 4).</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">gtBoxes (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Ground truth bounding boxes.</td>
      <td class="tg-0pky"><ul><li>n is the number of bounding boxes.</li><li>4 refers to [x2, y2, x3, y3], where (x2, y2) and (x3, y3) represent the top-left and bottom-right corners of the bounding box, respectively, and must satisfy x3 > x2, y3 > y2.</li><li>The data type must be consistent with bBoxes.</li></ul></td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">(n, 4).</td>
      <td class="tg-0pky">✓</td>
    </tr>
    <tr>
      <td class="tg-0lax">mode (char*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Selects the calculation mode: "iou" or "iof".</td>
      <td class="tg-0lax"><ul><li>"iou": Computes the intersection over union.</li><li>"iof": Computes the intersection over foreground.</li></ul></td>
      <td class="tg-0lax">String</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">eps (float)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">To prevent division by zero, eps is added to both the length and width when calculating the area.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">aligned (bool)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Indicates whether the shapes of the two inputs are the same.</td>
      <td class="tg-0lax"><ul><li>True: bBoxes and gtBoxes have the same shape, both being (m, 4), and the output shape is (m, 1).</li><li>False: bBoxes and gtBoxes have different shapes, being (m, 4) and (n, 4) respectively, and the output shape is (m, n).</li></ul></td>
      <td class="tg-0lax">BOOL</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">overlap (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">The IOU/IOF computed from the two inputs.</td>
      <td class="tg-0lax">The data type must be consistent with bBoxes.</td>
      <td class="tg-0lax">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">Shape: (m, n) or (m, 1).</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be requested on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which includes the operator computation flow.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: The BFLOAT16 data type is not supported.
  <!-- end id7 -->

- **Return Value**

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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input bBoxes, gtBoxes, and output overlap are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>bBoxes, gtBoxes, or overlap is not two-dimensional.</td>
    </tr>
    <tr>
      <td>The data types of bBoxes, gtBoxes, and overlap are inconsistent.</td>
    </tr>
    <tr>
      <td>The data type or format of bBoxes, gtBoxes, or overlap is not within the supported range.</td>
    </tr>
    <tr>
      <td>The second dimension of bBoxes or gtBoxes is not 4.</td>
    </tr>
    <tr>
      <td>When aligned is true, the first dimension of bBoxes and gtBoxes is different.</td>
    </tr>
    <tr>
      <td>When aligned is true, the second dimension of overlap is not 1.</td>
    </tr>
    <tr>
      <td>mode is neither "iou" nor "iof".</td>
    </tr>
    <tr>
      <td>eps is less than 0.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>API internal validation error, typically caused by input data or attribute specifications that are outside the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurred when the API called the npu runtime API, for example, SocVersion is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnIou

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
      <td>The workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size to be applied for on the Device side, obtained through the first-phase API aclnnIouGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>The operator executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream on which the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnIou defaults to a deterministic implementation.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/level2/aclnn_iou.h"

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

int main() {
  // 1. (Boilerplate) device/stream initialization; refer to the acl API manual
  // Fill in deviceId according to your actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
  // input
  std::vector<float> bBoxesHostData = {1.0, 1.0, 5.0, 3.0, 1.0, 1.0, 5.0, 3.0};
  std::vector<float> gtBoxesHostData = {4.0, 2.0, 9.0, 5.0, 4.0, 2.0, 9.0, 5.0};
  std::vector<float> overlapHostData = {0.045455, 0.045455};
  std::vector<int64_t> bBoxesShape = {2, 4};
  std::vector<int64_t> gtBoxesShape = {2, 4};
  std::vector<int64_t> overlapShape = {2, 1};
  void* bBoxesDeviceAddr = nullptr;
  void* gtBoxesDeviceAddr = nullptr;
  void* overlapDeviceAddr = nullptr;
  aclTensor* bBoxes = nullptr;
  aclTensor* gtBoxes = nullptr;
  aclTensor* overlap = nullptr;

  ret = CreateAclTensor(bBoxesHostData, bBoxesShape, &bBoxesDeviceAddr, aclDataType::ACL_FLOAT, &bBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gtBoxesHostData, gtBoxesShape, &gtBoxesDeviceAddr, aclDataType::ACL_FLOAT, &gtBoxes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(overlapHostData, overlapShape, &overlapDeviceAddr, aclDataType::ACL_FLOAT, &overlap);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // attr
  const char* mode = "iou";
  float eps = 0.0f;
  bool aligned = true;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. Call the CANN operator library API; change it to the specific Api name.
  ret = aclnnIouGetWorkspaceSize(bBoxes, gtBoxes, mode, eps, aligned, overlap, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnIouGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // aclnnIou
  ret = aclnnIou(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnIou failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
  auto size = GetShapeSize(overlapShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), overlapDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar; modify according to the specific API definition.
  aclDestroyTensor(bBoxes);
  aclDestroyTensor(gtBoxes);
  aclDestroyTensor(overlap);

  // 7. Release device resources
  aclrtFree(bBoxesDeviceAddr);
  aclrtFree(gtBoxesDeviceAddr);
  aclrtFree(overlapDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
