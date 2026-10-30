# aclnnThreeInterpolateBackward

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

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnThreeInterpolateBackwardGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnThreeInterpolateBackward API to perform the computation.

```Cpp
aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(
  const aclTensor *grad_x,
  const aclTensor *idx,
  const aclTensor *weight,
  int              m,
  aclTensor       *grad_y,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnThreeInterpolateBackward(
  void          *workspace,
  uint64_t       workspace_size,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## Function

- Operator function: Compute the gradient based on grad_x, idx, and weight through three-point interpolation to obtain grad_y.

- Formula:

  $$
  grad\_y[b,c,idx[b,n,i]] =
  grad\_y[b,c,idx[b,n,i]] + grad\_x[b,c,n]*weight[b,n,i]\\ i\in[0,2]\ b\in[0,B) \ c\in[0,C) \ n\in[0,N)
  $$

## aclnnThreeInterpolateBackwardGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1556px"><colgroup>
  <col style="width: 171px">
  <col style="width: 121px">
  <col style="width: 271px">
  <col style="width: 331px">
  <col style="width: 224px">
  <col style="width: 103px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0lax">Instruction</th>
      <th class="tg-0lax">Data Type</th>
      <th class="tg-0lax">Data Format</th>
      <th class="tg-0lax">Dimensions (shape)</th>
      <th class="tg-0lax">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">grad_x(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The gradient value of the previous step before network backward propagation.</td>
      <td class="tg-0lax">shape (b, c, n) is supported.</td>
      <td class="tg-0lax">FLOAT, FLOAT16</td>
      <td class="tg-0lax">NCHW, ND</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">idx(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The three nearest-neighbor feature indices of the target feature.</td>
      <td class="tg-0lax">shape (b, n, 3) is supported.</td>
      <td class="tg-0lax">INT32, INT64</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">weight(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The three nearest-neighbor feature weights of the target feature.</td>
      <td class="tg-0lax">shape (b, n, 3) is supported.</td>
      <td class="tg-0lax">FLOAT, FLOAT16</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">m(int)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">The size of the lowest dimension of output grad_y (that is, the number of points of the input feature of the forward three_interpolate), used to infer the shape of grad_y during infershape.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT32</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">grad_y(aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Gradient computation result.</td>
      <td class="tg-0lax">shape (b, c, m) is supported.</td>
      <td class="tg-0lax">FLOAT, FLOAT16</td>
      <td class="tg-0lax">NCHW, ND</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize(uint64_t*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the workspace size that needs to be allocated on the Device side.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor(aclOpExecutor**)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the op executor, which contains the operator computation flow.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API completes input parameter validation and reports an error in the following scenarios:

<table class="tg" style="undefined;table-layout: fixed; width: 859px"><colgroup>
  <col style="width: 302px">
  <col style="width: 142px">
  <col style="width: 415px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Return Value</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">The input grad_x, idx, weight, or grad_y is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="2">161002</td>
      <td class="tg-0pky">The data type or data format of grad_x, idx, weight, m, or grad_y is not within the supported range.</td>
    </tr>
    <tr>
      <td class="tg-0lax">When grad_x, idx, weight, or grad_y is not a null pointer, its data type, data format, or shape is not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnThreeInterpolateBackward

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device side, obtained through the first-phase API aclnnThreeInterpolateBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which contains the operator computation process.</td>
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

- The values in idx are within the range [0, m) (non-negative and less than m). The correctness of computation results is not guaranteed for out-of-range values (negative values or values greater than or equal to m).
- grad_x, idx, weight, and grad_y must satisfy the dimension correspondence: idx and weight have the same shape of (b, n, 3), and their b and n dimensions are consistent with the b and n dimensions of grad_x; the b and c dimensions of grad_y are consistent with those of grad_x, and its m dimension equals the attribute m. If the shapes are inconsistent, the API returns ACLNN_ERR_PARAM_INVALID.
- Deterministic computation:
  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: aclnnThreeInterpolateBackward defaults to a non-deterministic implementation and does not support enabling via configuration.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnThreeInterpolateBackward defaults to a deterministic implementation.
  <!-- end id8 -->
<!-- npu="A3,910b" id9 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: grad_x and grad_y are passed in NCHW format with 4D shape, namely (b, c, n, 1) and (b, c, m, 1) respectively.
<!-- end id9 -->
<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>: grad_x and grad_y are passed in ND format with 3D shape, namely (b, c, n) and (b, c, m) respectively.
<!-- end id10 -->

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="A3,910b" id11 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

  > Note:
  >
  > In this example, the input and output shapes are 4D and the data format is NCHW.

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The second-phase API of aclnnThreeInterpolateBackward, which performs the computation.
*/
aclnnStatus aclnnThreeInterpolateBackward(void *workspace, uint64_t workspaceSize,
                                          aclOpExecutor *executor, aclrtStream stream);

/**
 * @brief The first-phase API of aclnnThreeInterpolateBackward, which computes the workspace size according to the specific computation flow.
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(const aclTensor *grad_x, const aclTensor *idx,
                                                          const aclTensor *weight, int m, aclTensor *grad_y,
                                                          uint64_t *workspaceSize, aclOpExecutor **executor);
#ifdef __cplusplus
}
#endif

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
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
                    aclDataType dataType, aclTensor** tensor, aclFormat format) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) device/stream initialization; refer to the AscendCL external interface list
  // Fill in deviceId according to your actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  uint32_t bs = 1;
  uint32_t cs = 2;
  uint32_t ms = 6;
  uint32_t ns = 5;
  uint32_t c0 = 16;

  auto c1 = (cs + c0 - 1) / c0;

  // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
  std::vector<int64_t> gradXShape = {bs, cs, ns, 1}; // nchw
  std::vector<int64_t> idxShape = {bs, ns, 3}; // nd
  std::vector<int64_t> weightShape = {bs, ns, 3}; // nd
  std::vector<int64_t> gradYShape = {bs, cs, ms, 1}; // nchw

  void* gradXDeviceAddr = nullptr;
  void* idxDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* gradYDeviceAddr = nullptr;

  aclTensor* gradX = nullptr;
  aclTensor* idx = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* gradY = nullptr;

  std::vector<float> gradXHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint32_t> idxHostData = {0, 3, 2, 3, 5, 1, 0, 3, 0, 0, 3, 5, 0, 4, 1};
  std::vector<float> weightHostData = {3.2081969,-0.86573875,-1.0929844,5.6045847,5.328887,-2.0821328,0.98572457,9.612394,2.216394,
                                       1.8722068,4.2976365,-7.019285,-8.4070425,1.0159919,-0.78814566};
  std::vector<float> gradYHostData = {0};

  ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(idxHostData, idxShape, &idxDeviceAddr, aclDataType::ACL_INT32, &idx, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API; change it to the specific Api name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnThreeInterpolateBackward
  ret = aclnnThreeInterpolateBackwardGetWorkspaceSize(gradX, idx, weight, ms, gradY, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThreeInterpolateBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnThreeInterpolateBackward
  ret = aclnnThreeInterpolateBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThreeInterpolateBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
  PrintOutResult(gradYShape, &gradYDeviceAddr);

  // 6. Release aclTensor and aclScalar; modify according to the specific API definition.
  aclDestroyTensor(gradX);
  aclDestroyTensor(idx);
  aclDestroyTensor(weight);
  aclDestroyTensor(gradY);

  // 7. Release device resources; modify according to the specific API definition.
  aclrtFree(gradXDeviceAddr);
  aclrtFree(idxDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(gradYDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

<!-- end id11 -->

<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT products</term>：

  > Note:
  >
  > In this example, the input and output shapes are 3D and the data format is ND.

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The second-phase API of aclnnThreeInterpolateBackward, which performs the computation.
*/
aclnnStatus aclnnThreeInterpolateBackward(void *workspace, uint64_t workspaceSize,
                                          aclOpExecutor *executor, aclrtStream stream);

/**
 * @brief The first-phase API of aclnnThreeInterpolateBackward, which computes the workspace size according to the specific computation flow.
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(const aclTensor *grad_x, const aclTensor *idx,
                                                          const aclTensor *weight, int m, aclTensor *grad_y,
                                                          uint64_t *workspaceSize, aclOpExecutor **executor);
#ifdef __cplusplus
}
#endif

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
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
                    aclDataType dataType, aclTensor** tensor, aclFormat format) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) device/stream initialization; refer to the AscendCL external interface list
  // Fill in deviceId according to your actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  uint32_t bs = 1;
  uint32_t cs = 2;
  uint32_t ms = 6;
  uint32_t ns = 5;
  uint32_t c0 = 16;

  auto c1 = (cs + c0 - 1) / c0;

  // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
  std::vector<int64_t> gradXShape = {bs, cs, ns}; // nd
  std::vector<int64_t> idxShape = {bs, ns, 3}; // nd
  std::vector<int64_t> weightShape = {bs, ns, 3}; // nd
  std::vector<int64_t> gradYShape = {bs, cs, ms}; // nd

  void* gradXDeviceAddr = nullptr;
  void* idxDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* gradYDeviceAddr = nullptr;

  aclTensor* gradX = nullptr;
  aclTensor* idx = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* gradY = nullptr;

  std::vector<float> gradXHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint32_t> idxHostData = {0, 3, 2, 3, 5, 1, 0, 3, 0, 0, 3, 5, 0, 4, 1};
  std::vector<float> weightHostData = {3.2081969,-0.86573875,-1.0929844,5.6045847,5.328887,-2.0821328,0.98572457,9.612394,2.216394,
                                       1.8722068,4.2976365,-7.019285,-8.4070425,1.0159919,-0.78814566};
  std::vector<float> gradYHostData(bs * cs * ms, 0);

  ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(idxHostData, idxShape, &idxDeviceAddr, aclDataType::ACL_INT32, &idx, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY, aclFormat::ACL_FORMAT_ND);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API; change it to the specific Api name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnThreeInterpolateBackward
  ret = aclnnThreeInterpolateBackwardGetWorkspaceSize(gradX, idx, weight, ms, gradY, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThreeInterpolateBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnThreeInterpolateBackward
  ret = aclnnThreeInterpolateBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThreeInterpolateBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
  PrintOutResult(gradYShape, &gradYDeviceAddr);

  // 6. Release aclTensor and aclScalar; modify according to the specific API definition.
  aclDestroyTensor(gradX);
  aclDestroyTensor(idx);
  aclDestroyTensor(weight);
  aclDestroyTensor(gradY);

  // 7. Release device resources; modify according to the specific API definition.
  aclrtFree(gradXDeviceAddr);
  aclrtFree(idxDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(gradYDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

<!-- end id12 -->
