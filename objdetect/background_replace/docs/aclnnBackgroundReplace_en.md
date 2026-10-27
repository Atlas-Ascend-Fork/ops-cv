# aclnnBackgroundReplace

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

- Description:
Blend the input new background image with the existing image, and replace the background with the new background through masking.

- Formula:

  $$
  out = bkg * (1 - mask) + src * mask
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnBackgroundReplaceGetWorkspaceSize" API to obtain the input parameters and compute the required workspace size according to the flow, then call the "aclnnBackgroundReplace" API to perform the computation.

```Cpp
aclnnStatus aclnnBackgroundReplaceGetWorkspaceSize(
  const aclTensor*    bkg,
  const aclTensor*    src,
  const aclTensor*    mask,
  const aclTensor*    out,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```Cpp
aclnnStatus aclnnBackgroundReplace(
  void*               workspace,
  uint64_t            workspaceSize,
  aclOpExecutor*      executor,
  aclrtStream         stream)
```

## aclnnBackgroundReplaceGetWorkspaceSize

- **Parameters**

<table class="tg" style="undefined;table-layout: fixed; width: 1409px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 238px">
  <col style="width: 184px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 199px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-5agr">Parameter Name</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Usage Note</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">bkg (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">UINT8, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">HWC (C=1, 3).</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">src (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">UINT8, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">HWC (C=1, 3).</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">mask (aclTensor*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Input tensor.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">FLOAT16</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">HWC (C=1).</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output tensor.</td>
      <td class="tg-0pky">Data type and shape are consistent with the input background image bkg.</td>
      <td class="tg-0pky">UINT8, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">HWC (C=1, 3).</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be allocated on the device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which includes the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Return Value**

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
      <td>The passed bkg, src, mask, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of bkg, src, mask, or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data format of bkg, src, mask, or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of bkg, src, or out are inconsistent.</td>
    </tr>
    <tr>
      <td>The data formats of bkg, src, mask, or out are inconsistent.</td>
    </tr>
  </tbody>
  </table>

## aclnnBackgroundReplace

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
      <td>Address of the workspace memory applied for on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the device side, obtained by the first-phase API aclnnBackgroundReplaceGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which contains the operator computation process.</td>
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

- Deterministic computation:
  - aclnnBackgroundReplace defaults to a deterministic implementation

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_background_replace.h"

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
  // Call aclrtMalloc to allocate Device-side memory
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
  // check handle according to your own needs
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
  std::vector<int64_t> bkgShape = {4, 2, 1};
  std::vector<int64_t> srcShape = {4, 2, 1};
  std::vector<int64_t> maskShape = {4, 2, 1};
  std::vector<int64_t> outShape = {4, 2, 1};
  void* bkgDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  void* maskDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* bkg = nullptr;
  aclTensor* src = nullptr;
  aclTensor* mask = nullptr;
  aclTensor* out = nullptr;
  std::vector<uint8_t> bkgHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint8_t> srcHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> maskHostData = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // Create bkg aclTensor
  ret = CreateAclTensor(bkgHostData, bkgShape, &bkgDeviceAddr, aclDataType::ACL_UINT8, &bkg);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create src aclTensor
  ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_UINT8, &src);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create mask aclTensor
  ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_FLOAT16, &mask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBackgroundReplace API call example
  // 3. Call the CANN operator library API; change it to the specific API name.
  // Call the first-phase API of aclnnBackgroundReplace
  ret = aclnnBackgroundReplaceGetWorkspaceSize(bkg, src, mask, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBackgroundReplaceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBackgroundReplace
  ret = aclnnBackgroundReplace(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBackgroundReplace failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Get the output values and copy the results from Device-side memory to Host-side; modify according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<uint8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar; modify according to the specific API definition.
  aclDestroyTensor(bkg);
  aclDestroyTensor(src);
  aclDestroyTensor(mask);
  aclDestroyTensor(out);

  // 7. Release Device resources; modify according to the specific API definition.
  aclrtFree(bkgDeviceAddr);
  aclrtFree(srcDeviceAddr);
  aclrtFree(maskDeviceAddr);
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
