# aclnnUpsampleTrilinear3dBackward

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

- Description: Backward computation of [aclnnUpsampleTrilinear3d.md](../../resize_upsample_trilinear/docs/aclnnUpsampleTrilinear3d.md).

- Formula:
  - Core algorithm logic of the forward pass:
    1. Map each point of the target image back to the original image to obtain a coordinate with a fractional part.
    2. Based on this floating-point coordinate, compute the adjacent original image points before and after.
    3. Compute the weights of the adjacent points to the corresponding target point respectively, and multiply and accumulate them according to the weights to obtain the target point value.

  - Specific computation logic:
    The scaling method is divided into corner alignment and edge alignment. Corner alignment (alignCorners = true) means aligning according to the pixel center point at the top-left corner of the original image; edge alignment (alignCorners = false) means aligning according to the top-left vertex and two edges of the original image. There are differences in computing the scaling coefficient and coordinate position. For a 3D interpolation point $(N, C, D, H, W)$, the following formula applies:

    $$
    scale\_d =\begin{cases}
    (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scales\_d & alignCorners=false\&scales\_d>0\\
    inputSize[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_h =\begin{cases}
    (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
    1 / scales\_h & alignCorners=false\&scales\_h>0\\
    inputSize[3] / outputSize[1] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_w =\begin{cases}
    (inputSize[4]-1) / (outputSize[2]-1) & alignCorners=true \\
    1 / scales\_w & alignCorners=false\&scales\_w>0\\
    inputSize[4] / outputSize[2] & alignCorners=false
    \end{cases}
    $$

Therefore, for a point p(x,y,z) in a certain direction of the output, the point mapped back to the original image is denoted as q(x',y',z'), and the relationship is as follows:

    $$
    x' =\begin{cases}
    x * scale\_d & alignCorners=true \\
    MAX(0,{(x+0.5)*scale\_d-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    y' =\begin{cases}
    y * scale\_h & alignCorners=true \\
    MAX(0,{(y+0.5)*scale\_h-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    z' =\begin{cases}
    z * scale\_w & alignCorners=true \\
    MAX(0,{(z+0.5)*scale\_w-0.5}) & alignCorners=false
    \end{cases}
    $$

    - Notation:

      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$

      $$
      y_{0} =int(y'),y_{1} =int(y')+1, lambdb_{0} = y_{1}-y', lambdb_{1} =   1-lambdb_{0}
      $$

      $$
      z_{0} =int(z'),z_{1} =int(z')+1, lambdc_{0} = z_{1}-z', lambdc_{1} =   1-lambdc_{0}
      $$

    - Then the following formula applies:

      $$
      {V(p_{x, y, z})} = {V(p_{x0, y0, z0})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x0, y0, z1})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x0, y1, z0})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x0, y1, z1})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{1}} + {V(p_{x1, y0, z0})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x1, y0, z1})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x1, y1, z0})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x1, y1, z1})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{1}}
      $$

    - Assumption: The output image out $(x, y, z)$ of forward interpolation is affected by the original image input $(x_i, y_j, z_k)$. Then:

      $$
      gradInput(x_i,y_j,z_k) += gradOut(x,y,z) * lambda(x_i,y_j,z_k)* lambdb(x_i,y_j,z_k)* lambdc(x_i,y_j,z_k)
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnUpsampleTrilinear3dBackward API to perform the computation.

```Cpp
aclnnStatus aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize(
  const aclTensor   *gradOut,
  const aclIntArray *outputSize,
  const aclIntArray *inputSize,
  bool               alignCorners,
  double             scalesD,
  double             scalesH,
  double             scalesW,
  aclTensor         *gradInput,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnUpsampleTrilinear3dBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize

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
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-Contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOut (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the gradient tensor for backward computation, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>When the data format is ND, it is processed in NCDHW format by default.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, DOUBLE</td>
      <td>NCDHW, NDHWC, ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize (aclIntArray*)</td>
      <td>Input</td>
      <td>Represents the spatial size of the input `gradOut` in the D, H, and W dimensions, corresponding to `outputSize` in the formula.</td>
      <td>The size is 3, and each element is greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize (aclIntArray*)</td>
      <td>Input</td>
      <td>Represents the spatial size of the output `gradInput` in the N, C, D, H, and W dimensions, corresponding to `inputSize` in the corresponding formula.</td>
      <td>The size is 5, and the last three elements are all greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners (bool)</td>
      <td>Input</td>
      <td>Indicates whether to align corner pixels, corresponding to `alignCorners` in the formula.</td>
      <td>If true, the corner pixels of the input and output tensors are aligned; otherwise, they are not aligned.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesD (double)</td>
      <td>Input</td>
      <td>Represents the depth dimension multiplier of the output `gradInput`, corresponding to `scales_d` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH (double)</td>
      <td>Input</td>
      <td>Represents the height dimension multiplier of the output `gradInput`, corresponding to `scales_h` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW (double)</td>
      <td>Input</td>
      <td>Represents the width dimension multiplier of the output `gradInput`, corresponding to `scales_w` in the corresponding formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the output tensor of the backward computation, corresponding to `gradInput` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The sizes of the N, C, D, H, and W dimensions in the shape must be consistent with the N, C, D, H, and W dimensional spatial sizes specified in `inputSize`.</li><li>The data type, data format, and shape must be consistent with the input parameter `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, DOUBLE</td>
      <td>NCDHW, NDHWC, ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be side-applied on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the operator executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>:

    The data types of parameters `gradOut` and `gradInput` do not support BFLOAT16.
  <!-- end id7 -->

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and reports an error in the following scenarios:

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
      <td>The passed gradOut, outputSize, inputSize, or gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The data type and data format of gradOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of gradOut and gradInput are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension of gradOut is not 5.</td>
    </tr>
    <tr>
      <td>The size of outputSize is not equal to 3.</td>
    </tr>
    <tr>
      <td>An element value in outputSize is not greater than 0.</td>
    </tr>
    <tr>
      <td>The size of inputSize is not equal to 5.</td>
    </tr>
    <tr>
      <td>An element value in the last three elements of inputSize is not greater than 0.</td>
    </tr>
    <tr>
      <td>The sizes of gradOut and inputSize differ in the N and C dimensions.</td>
    </tr>
    <tr>
      <td>The size of gradOut in the D, H, and W dimensions is inconsistent with outputSize[0], outputSize[1], and outputSize[2].</td>
    </tr>
    <tr>
      <td>The shape of gradInput is inconsistent with inputSize[0], inputSize[1], inputSize[2], inputSize[3], and inputSize[4].</td>
    </tr>
  </tbody></table>

## aclnnUpsampleTrilinear3dBackward

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
      <td>The workspace memory address applied on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained through the first-phase API aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize.</td>
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

- Shape constraints for parameters `gradOut` and `gradInput`:
  - The value of each dimension is less than or equal to 2^20.
  - The N and C axes of parameter `gradInput` are the same as those of `gradOut`.
  - The memory usage must be less than 60 GB. The formula for memory usage is as follows:

    $$
    N * C * (gradOut\_D * gradOut\_H * gradOut\_W + gradInput\_D * gradInput\_H * gradInput\_W + gradOut\_D * gradOut\_H * gradInput\_W + gradOut\_D * gradInput\_H * gradInput\_W) * sizeof(dtype) < 60 * 1024 * 1024 * 1024
    $$

    where:
    - N represents the N axis of the input and output.
    - C represents the C axis of the input and output.
    - dtype represents the data type of the input tensor.
  - N * C * gradOut_D * gradOut_H < 2^31
  - gradInput_W * gradInput_H < 2^31
- When the values of scalesD, scalesH, and scalesW are all greater than 0, parameters inputSize, outputSize, scalesD, scalesH, and scalesW must satisfy the following constraints:

  $$
  outputSize\_D = floor(inputSize\_D * scalesD)
  $$

  $$
  outputSize\_H = floor(inputSize\_H * scalesH)
  $$

  $$
  outputSize\_W = floor(inputSize\_W * scalesW)
  $$

- Deterministic computation:
  - aclnnUpsampleTrilinear3dBackward defaults to a deterministic implementation.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_trilinear_3d_backward.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
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
    // check handle according to your own needs
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
    std::vector<int64_t> gradOutShape = {2, 2, 2, 2, 2};
    std::vector<int64_t> gradInputShape = {2, 2, 1, 1, 1};
    void* gradOutDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<float> gradOutHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> gradInputHostData = {2.0, 2, 2, 2};
    std::vector<int64_t> outputSizeData = {2, 2, 2};
    std::vector<int64_t> inputSizeData = {2, 2, 1, 1, 1};
    bool alignCorners = false;
    double scalesD = 0.0;
    double scalesH = 0.0;
    double scalesW = 0.0;
    // Create gradOut aclTensor
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gradInput aclTensor
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const aclIntArray* outputSize = aclCreateIntArray(outputSizeData.data(), outputSizeData.size());
    CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    const aclIntArray* inputSize = aclCreateIntArray(inputSizeData.data(), inputSizeData.size());
    CHECK_RET(inputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the CANN operator library API; change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnUpsampleTrilinear3dBackward
    ret = aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize(gradOut, outputSize, inputSize, alignCorners, scalesD,
                                                           scalesH, scalesW, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnUpsampleTrilinear3dBackward
    ret = aclnnUpsampleTrilinear3dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleTrilinear3dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar; modify according to the specific API definition.
    aclDestroyTensor(gradOut);
    aclDestroyTensor(gradInput);

    // 7. Release device resources; modify according to the specific API definition.
    aclrtFree(gradOutDeviceAddr);
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
