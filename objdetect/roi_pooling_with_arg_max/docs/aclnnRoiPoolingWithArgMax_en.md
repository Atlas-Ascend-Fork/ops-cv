# aclnnRoiPoolingWithArgMax

[📄 View Source Code](https://gitcode.com/cann/ops-cv/tree/master/objdetect/roi_pooling_with_arg_max)

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products/Atlas A3 inference products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products/Atlas A2 inference products</term>: Not supported
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

- Description: Pool the input feature map by ROI (region of interest). Within each ROI, spatially divide it into `pooled_h × pooled_w` bins, perform max pooling on each bin, and output the pooling result and the one-dimensional index (argmax) of the maximum value within the channel.

- Formula:

  The shape of the input feature map $x$ is $(N, C, H, W)$, and the shape of the ROI tensor $\text{rois}$ is $(\text{num\_rois}, 5)$, where each row represents $(b_n, x_1, y_1, x_2, y_2)$. The scalar parameters are $s_h$, $s_w$ (spatial_scale), and $\text{pooled\_h}$, $\text{pooled\_w}$. The subscript $n$ indicates the ROI index, $c$ indicates the channel, and $(\text{ph}, \text{pw})$ indicates the pooling bin.

  - **ROI mapped to feature map**: Multiply the ROI coordinates by spatial_scale to obtain the floating-point interval on the feature map:

    $$
    \tilde{x}_1 = x_1 s_w,\quad \tilde{y}_1 = y_1 s_h,\quad \tilde{x}_2 = (x_2+1)s_w,\quad \tilde{y}_2 = (y_2+1)s_h
    $$

    $$
    W_{\text{roi}} = \tilde{x}_2 - \tilde{x}_1,\qquad H_{\text{roi}} = \tilde{y}_2 - \tilde{y}_1
    $$

    If $W_{\text{roi}} \le 0$ or $H_{\text{roi}} \le 0$, the $y$ of this ROI is all 0, and $\text{argmax}$ is all -1.

  - **Bin stride and interval**: Each pooling bin (ph, pw) corresponds to a bin inside the ROI. The stride and floating-point interval are:

    $$
    \Delta w = \frac{W_{\text{roi}}}{\text{pooled\_w}},\qquad \Delta h = \frac{H_{\text{roi}}}{\text{pooled\_h}}
    $$

    $$
    \tilde{w}_1 = \text{pw} \cdot \Delta w + \tilde{x}_1,\quad \tilde{w}_2 = (\text{pw}+1) \cdot \Delta w + \tilde{x}_1
    $$

    $$
    \tilde{h}_1 = \text{ph} \cdot \Delta h + \tilde{y}_1,\quad \tilde{h}_2 = (\text{ph}+1) \cdot \Delta h + \tilde{y}_1
    $$

Round and clip to $[0,W) \times [0,H)$:

    $$
    w_1 = \text{clip}(\lfloor\tilde{w}_1\rfloor,\, 0,\, W),\quad w_2 = \text{clip}(\lceil\tilde{w}_2\rceil,\, 0,\, W)
    $$

    $$
    h_1 = \text{clip}(\lfloor\tilde{h}_1\rfloor,\, 0,\, H),\quad h_2 = \text{clip}(\lceil\tilde{h}_2\rceil,\, 0,\, H)
    $$

    where $\text{clip}(a,l,u) = \min(\max(a,l), u)$. If $w_2 \le w_1$ or $h_2 \le h_1$, this bin is empty: $y=0$, $\text{argmax}=-1$.

  - **Pooling output and Argmax**: Let $b = \text{rois}[n,0]$, and the bin region $R = \{(h,w) : h_1 \le h < h_2,\, w_1 \le w < w_2\}$, then

    $$
    y[n,c,\text{ph},\text{pw}] = \max_{(h,w) \in R} x[b,c,h,w]
    $$

    (0 when $R$ is empty)

    $$
    \text{argmax}[n,c,\text{ph},\text{pw}] = h^* W + w^*
    $$

    $(h^*, w^*)$ is the position of the maximum value in the bin (the first one is taken if there are multiple solutions); empty $R$ is -1.

  - **Output Shape**:

    | Output | Shape | Data Type |
    |------|--------|----------|
    | $y$ | $(\text{num\_rois},\, C,\, \text{pooled\_h},\, \text{pooled\_w})$ | Same as $x$ |
    | $\text{argmax}$ | Same as above | INT32 |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnRoiPoolingWithArgMaxGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnRoiPoolingWithArgMax API to perform the computation.

```Cpp
aclnnStatus aclnnRoiPoolingWithArgMaxGetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *rois,
  int64_t            pooled_h,
  int64_t            pooled_w,
  float              spatial_scale_h,
  float              spatial_scale_w,
  aclTensor         *y,
  aclTensor         *argmax,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnRoiPoolingWithArgMax(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRoiPoolingWithArgMaxGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1560px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 270px">
  <col style="width: 330px">
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 190px">
  <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimensions (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x(aclTensor*)</td>
      <td>Input</td>
      <td>Input feature map, in NCHW format, (N, C, H, W).</td>
      <td><ul><li>The input must be 4-dimensional.</li><li>N must be a multiple of 16.</li><li>N is less than or equal to 1024.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rois(aclTensor*)</td>
      <td>Input</td>
      <td>ROI boxes, 5 elements per row: batch_idx, x1, y1, x2, y2.</td>
      <td><ul><li>shape (num_rois, 5).</li><li>The 0th dimension is less than or equal to 1024.</li><li>x1, y1, x2, y2 are greater than or equal to 0.0.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pooled_h(int64_t)</td>
      <td>Input</td>
      <td>Pooling output height.</td>
      <td>Must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_w(int64_t)</td>
      <td>Input</td>
      <td>Pooling output width.</td>
      <td>Must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_h(float)</td>
      <td>Input</td>
      <td>The scaling ratio in the height direction when ROI coordinates are mapped to the feature map.</td>
      <td>Must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_w(float)</td>
      <td>Input</td>
      <td>The scaling ratio in the width direction when ROI coordinates are mapped to the feature map.</td>
      <td>Must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y(aclTensor*)</td>
      <td>Output</td>
      <td>Pooling result, shape (num_rois, C, pooled_h, pooled_w).</td>
      <td><ul><li>The data type is the same as that of x.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>argmax(aclTensor*)</td>
      <td>Output</td>
      <td>The linear offset index of the maximum value within the channel for each pooling bin.</td>
      <td><ul><li>The shape is the same as that of y.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and reports an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead>
    <tr>
      <th>Status Code Name</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If the passed parameter is a required input, output, or required attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
    </tr>
    <tr>
      <td>The data type or format of x, rois, y, or argmax is not within the supported range.</td>
    </tr>
    <tr><td>The shape of x is not 4-dimensional (NCHW).</td>
    </tr>
    <tr><td>The second dimension of rois's shape is not 5.</td>
    </tr>
    <tr><td>pooled_h, pooled_w, spatial_scale_h, and spatial_scale_w are not greater than 0.</td>
    </tr>
  </tbody></table>

## aclnnRoiPoolingWithArgMax

- **Parameters**

<table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>The workspace size to be applied on the Device side, obtained through the first-phase API aclnnRoiPoolingWithArgMaxGetWorkspaceSize.</td>
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
  - aclnnRoiPoolingWithArgMax defaults to a deterministic implementation.
- The shape[0] of x, argmax, and rois are equal

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_pooling_with_arg_max.h"

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
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorOutput(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                          aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> xShape = {2, 16, 25, 42};
  std::vector<int64_t> roisShape = {2, 5};
  std::vector<int64_t> yShape = {2, 16, 3, 3};
  std::vector<int64_t> argmaxShape = {2, 16, 3, 3};

  void* xDeviceAddr = nullptr;
  void* roisDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* argmaxDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* rois = nullptr;
  aclTensor* y = nullptr;
  aclTensor* argmax = nullptr;

  int64_t xSize = GetShapeSize(xShape);
  int64_t roisSize = GetShapeSize(roisShape);
  std::vector<float> xHostData(xSize, 1.0f);
  std::vector<float> roisHostData(roisSize, 0.0f);
  roisHostData[0] = 0.0f;
  roisHostData[1] = 0.0f;
  roisHostData[2] = 0.0f;
  roisHostData[3] = 24.0f;
  roisHostData[4] = 41.0f;
  roisHostData[5] = 1.0f;
  roisHostData[6] = 0.0f;
  roisHostData[7] = 0.0f;
  roisHostData[8] = 24.0f;
  roisHostData[9] = 41.0f;

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(roisHostData, roisShape, &roisDeviceAddr, aclDataType::ACL_FLOAT, &rois);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorOutput<float>(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorOutput<int32_t>(argmaxShape, &argmaxDeviceAddr, aclDataType::ACL_INT32, &argmax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t pooledH = 3;
  int64_t pooledW = 3;
  float spatialScaleH = 1.0f;
  float spatialScaleW = 1.0f;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnRoiPoolingWithArgMaxGetWorkspaceSize(x, rois, pooledH, pooledW, spatialScaleH, spatialScaleW,
                                                  y, argmax, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingWithArgMaxGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnRoiPoolingWithArgMax(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingWithArgMax failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  int64_t yElem = GetShapeSize(yShape);
  std::vector<float> yResult(yElem, 0.0f);
  ret = aclrtMemcpy(yResult.data(), yElem * sizeof(float), yDeviceAddr, yElem * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);

  int64_t argmaxElem = GetShapeSize(argmaxShape);
  std::vector<int32_t> argmaxResult(argmaxElem, 0);
  ret = aclrtMemcpy(argmaxResult.data(), argmaxElem * sizeof(int32_t), argmaxDeviceAddr,
                    argmaxElem * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy argmax from device to host failed. ERROR: %d\n", ret); return ret);

  aclDestroyTensor(x);
  aclDestroyTensor(rois);
  aclDestroyTensor(y);
  aclDestroyTensor(argmax);
  aclrtFree(xDeviceAddr);
  aclrtFree(roisDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(argmaxDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
