# aclnnMrgbaCustom

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

- Description: Complete the alpha multiplication computation of tensor rgb and tensor alpha

- Formula:

  $$
  out = rgb * ((broadcast)alpha/255)
  $$

## Example
  Assume rgb is a three-channel color image and alpha is its corresponding transparency (single channel). After using this operator, the image can be generated as a three-channel image with transparency.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMrgbaCustomGetWorkspaceSize" API to obtain the required workspace size and the executor that contains the operator computation flow, then call the "aclnnMrgbaCustom" API to perform the computation.

```Cpp
aclnnStatus aclnnMrgbaCustomGetWorkspaceSize(
  const aclTensor*      rgb,
  const aclTensor*      alpha,
  const aclTensor*      out,
  uint64_t*             workspaceSize,
  aclOpExecutor**       executor)
```

```Cpp
aclnnStatus aclnnMrgbaCustom(
  void*                 workspace,
  uint64_t              workspaceSize,
  aclOpExecutor*        executor,
  aclrtStream           stream)
```

## aclnnMrgbaCustomGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1409px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 238px">
  <col style="width: 184px">
  <col style="width: 127px">
  <col style="width: 120px">
  <col style="width: 239px">
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
      <td class="tg-0pky">rgb(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The rgb in the formula.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">UINT8</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">HWC(C=3), satisfies <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with alpha.</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">alpha(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The alpha in the formula.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">UINT8</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">HWC(C=1), satisfies <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with rgb.</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">out(aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output tensor.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">UINT8</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">HWC(C=3), same as the shape of rgb.</td>
      <td class="tg-0pky">-</td>
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
      <th>return value</th>
      <th>error code</th>
      <th>description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>rgb, alpha, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of rgb and alpha are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of rgb and alpha do not meet the HWC (C=3) and HWC (C=1) requirements.</td>
    </tr>
  </tbody>
  </table>

## aclnnMrgbaCustom

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
      <td>The workspace size allocated on the Device side, obtained through the first-phase API aclnnMrgbaCustomGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which contains the operator computation flow.</td>
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

- Deterministic computation:
  - aclnnMrgbaCustom defaults to a deterministic implementation

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mrgba_custom.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                  \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    printf(message, ##__VA_ARGS__);  \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i: shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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

template<typename T>
int CreateAclTensor(const std::vector <T> &hostData, const std::vector <int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
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
    // 1. (Boilerplate) device/stream initialization; refer to the acl external interface list
    // Fill in deviceId according to your actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check handle according to your own needs
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2.Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
    std::vector<int64_t> rgbShape = {4, 3};
    std::vector<int64_t> alphaShape = {4, 1};
    std::vector<int64_t> dstShape = {4, 3};
    void *rgbDeviceAddr = nullptr;
    void *alphaDeviceAddr = nullptr;
    void *dstDeviceAddr = nullptr;
    aclTensor *rgb = nullptr;
    aclTensor *alpha = nullptr;
    aclTensor *dst = nullptr;
    std::vector<uint8_t> rgbHostData = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    std::vector<uint8_t> alphaHostData = {255, 255, 255, 255};
    std::vector<uint8_t> dstHostData = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // Create rgb aclTensor
    ret = CreateAclTensor(rgbHostData, rgbShape, &rgbDeviceAddr, aclDataType::ACL_UINT8, &rgb);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create alpha aclTensor
    ret = CreateAclTensor(alphaHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_UINT8, &alpha);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create dst aclTensor
    ret = CreateAclTensor(dstHostData, dstShape, &dstDeviceAddr, aclDataType::ACL_UINT8, &dst);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnMrgba
    ret = aclnnMrgbaCustomGetWorkspaceSize(rgb, alpha, dst, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMrgbaCustomGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMrgba
    ret = aclnnMrgbaCustom(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMrgbaCustom failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
    auto size = GetShapeSize(dstShape);
    std::vector<uint8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dstDeviceAddr,
                      size * sizeof(uint8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
    }

    // 6. Release aclTensor
    aclDestroyTensor(rgb);
    aclDestroyTensor(alpha);
    aclDestroyTensor(dst);

    // 7. Release device resources; modify according to the specific API definition.
    aclrtFree(rgbDeviceAddr);
    aclrtFree(alphaDeviceAddr);
    aclrtFree(dstDeviceAddr);
    if(workspaceSize > 0){
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```
