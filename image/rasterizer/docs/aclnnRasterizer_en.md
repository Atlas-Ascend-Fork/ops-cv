# aclnnRasterizer

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Implement rasterization. Given the points and faces in 3D space, obtain the minimum depth of each pixel on the screen and its corresponding face index, and compute the perspective-corrected interpolation of the barycentric coordinates of that face.

- Formula:

  $findices$ records the face index corresponding to the minimum depth of each pixel, and $barycentric$ records the perspective-corrected interpolation of the barycentric coordinates of each vertex relative to the face recorded in $findices$.
  The Z-Buffer used in the computation records the minimum depth $z_{\min}(x, y)$ of each pixel $(x, y)$ and the triangle face index $\text{face\_idx}(x, y)$ corresponding to that depth.

The computation process is as follows:
  For each triangle face $f$ in space:

  1. Convert the three vertex coordinates $(v_0, v_1, v_2)$ of $f$ to screen coordinates $(v_{s0}, v_{s1}, v_{s2})$
  2. Compute the rectangular range surrounding $f$ based on $(v_{s0}, v_{s1}, v_{s2})$
  3. For each pixel $v_i = (x_i, y_i)$ inside the rectangle, perform the following operations:

     a. Compute the pixel center coordinate $v_c$
     b. Compute the barycentric coordinates $(\alpha, \beta, \gamma)$ of $v_c$ relative to triangle $f$
     c. Determine whether $v_c$ is inside the triangle based on $(\alpha, \beta, \gamma)$. If $v_c$ is not inside the triangle, process the next pixel in the rectangle; otherwise, perform the following steps
     d. Obtain the depth value of the current pixel using $(\alpha, \beta, \gamma)$ and $(v_{s0}, v_{s1}, v_{s2})$
     e. If depth prior is enabled, use the depth prior map to compute the depth threshold depth_thres; otherwise, directly proceed to the next step "Z-Buffer Update"

        - Use the depth prior map to compute the depth threshold depth_thres
        - If depth < depth_thres, process the next pixel in the rectangle; otherwise, perform the following steps

f. Z-Buffer update:

       - If $depth < z_{\min}(x_i, y_i)$:

         $$
         \quad z_{\min}(x_i, y_i) \gets \text{depth} \\
         \quad \text{face\_idx}(x_i, y_i) \gets f
         $$

       - If $depth = z_{\min}(x_i, y_i)$:

         $$
         \quad \text{face\_idx}(x_i, y_i) \gets \min(\text{face\_idx}(x_i, y_i),\ f)
         $$

After processing all triangular faces in space according to the above steps, for each pixel $v_i = (x_i, y_i)$ on the screen of size $height * width$:

  1. Take the face index $f_{idx}$ corresponding to $v_i$ from the Z-Buffer, $findices (x_i, y_i) \gets f_{idx}$
  2. Convert the three vertex coordinates $(v_0, v_1, v_2)$ of $f$ to screen coordinates $(v_{s0}, v_{s1}, v_{s2})$
  3. Compute the center coordinate $v_c$ of $v_i$
  4. Compute the barycentric coordinates $(\alpha, \beta, \gamma)$ of $v_c$ relative to triangle $f$
  5. Compute the perspective-corrected interpolation $(\tilde{\alpha}, \tilde{\beta}, \tilde{\gamma})$ using $(\alpha, \beta, \gamma)$
  6. $barycentric(x_i, y_i) \gets (\tilde{\alpha}, \tilde{\beta}, \tilde{\gamma})$

The specific computation methods involved are as follows:

  - Convert vertex $v = (x, y, z, w)$ to screen coordinate $v_s = (x_s, y_s, z_s)$

    $$
    x_s = (x / w * 0.5 + 0.5) * (width - 1) + 0.5\\
    y_s = (0.5 + 0.5 * y / w) * (height - 1) + 0.5\\
    z_s = z / w * 0.49999 + 0.5
    $$

  - Barycentric coordinates $(\alpha, \beta, \gamma)$ of point $v$ relative to triangle $(v_0, v_1, v_2)$

    1. Compute the signed areas $area$, $beta\_tri$, and $gamma\_tri$ of triangles $(v_0, v_1, v_2)$, $(v_0, v, v_2)$, and $(v_0, v_1, v)$ respectively
    2. If $area$ is 0, then $\alpha = \beta = \gamma = -1$; otherwise

      $$
      \beta = beta\_tri / area\\
      \gamma = gamma\_tri / area\\
      \alpha = 1 - \beta - \gamma
      $$

  - The signed area of the triangle formed by vertices $v_0 = (x_0, y_0, z_0)$, $v_1 = (x_1, y_1, z_1)$, and $v_2 = (x_2, y_2, z_2)$

    $$
    area = (x_2 - x_0) * (y_1 - y_0) - (x_1 - x_0) * (y_2 - y_0)
    $$

  - Compute the depth $depth$ of pixel $v = (x, y)$ by combining the barycentric coordinates $(\alpha, \beta, \gamma)$ and the triangle screen coordinates $(v_0 = (x_0, y_0, z_0), v_1 = (x_1, y_1, z_1), v_2 = (x_2, y_2, z_2))$

    $$
    depth = \alpha * z_0 + \beta * z_1 + \gamma * z_2
    $$

  - Compute the depth threshold $depth\_thres$ of point $v = (x, y)$ by combining the depth map $d$ and the occlusion truncation $occlusion\_truncation$

    $$
    depth\_thres = d(x, y) * 0.49999 + 0.5 + occlusion\_truncation
    $$

  - Determine whether a vertex is inside the triangle based on the barycentric coordinates $(\alpha, \beta, \gamma)$
    If $\alpha >= 0$ and $\beta >= 0$ and $\gamma >= 0$, the point is inside the triangle (including on the triangle edge); otherwise, the point is not inside the triangle.
  - Compute the perspective-corrected interpolation $(\lambda_0^{corrected}, \lambda_1^{corrected}, \lambda_2^{corrected})$ by combining the barycentric coordinates $(\lambda_0, \lambda_1, \lambda_2)$ and the three triangle vertex coordinates $v_0 = (x_0, y_0, z_0, w_0)$, $v_1 = (x_1, y_1, z_1, w_1)$, and $v_2 = (x_2, y_2, z_2, w_2)$

    $$
    \lambda_i^{corrected} = \frac{\lambda_i / w_i} { \sum (\lambda_j / w_j)}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnRasterizerGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the aclnnRasterizer API to perform the computation.

```Cpp
aclnnStatus aclnnRasterizerGetWorkspaceSize(
    const aclTensor *v,
    const aclTensor *f,
    const aclTensor *dOptional,
    int64_t          width,
    int64_t          height,
    double           occlusionTruncation,
    int64_t          useDepthPrior,
    const aclTensor *findicesOut,
    const aclTensor *barycentricOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```Cpp
aclnnStatus aclnnRasterizer(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnRasterizerGetWorkspaceSize

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
      <td>v(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input tensor of vertex coordinates in space.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>shape (numVertices, 4), where numVertices indicates the number of vertices, a positive integer. Each vertex coordinate is represented as (x, y, z, w).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>f(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input tensor of faces in space.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>shape (numFaces, 3), where numFaces indicates the number of faces in space, a positive integer. Each face is a triangle, and each vertex of the triangle is represented as the index of the vertex in v. Therefore, the value of elements in f should be a valid index of elements in v, that is, the value range is [0, numVertices-1]. The caller must ensure that the elements in f are valid.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input tensor of the depth map, used to compute the depth threshold.</td>
      <td><ul><li>Optional input; empty tensors are supported.</li><li>This parameter does not take effect.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>width(int64_t)</td>
      <td>Input</td>
      <td>Screen width.</td>
      <td>Value range [1, 4096].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>height(int64_t)</td>
      <td>Input</td>
      <td>Screen height.</td>
      <td>Value range [1, 4096].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>occlusionTruncation(double)</td>
      <td>Input</td>
      <td>Occlusion truncation, used to compute the depth threshold.</td>
      <td>This parameter does not take effect.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>useDepthPrior(int64_t)</td>
      <td>Input</td>
      <td>Indicates whether to apply the depth prior.</td>
      <td><ul><li>Value is 0 or 1. 1 means applying the depth prior, 0 means not applying the depth prior.</li><li>The current operator does not support applying the depth prior, so the value is fixed at 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>findicesOut(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the index of the face corresponding to the minimum depth of each pixel on the screen.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>shape (height, width), where height is the screen height and width is the screen width.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>barycentricOut(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output tensor of the perspective-corrected interpolation of the barycentric coordinates of each pixel on the screen relative to the face corresponding to the minimum depth.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>shape (height, width, 3), where height is the screen height and width is the screen width.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
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
        <td>The input v, f, findicesOut, or barycentricOut is a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data type of v, f, findicesOut, or barycentricOut is not within the supported range.</td>
      </tr>
      <tr>
        <td rowspan="11">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="11">561002</td>
        <td>The shape of v, f, findicesOut, or barycentricOut is not within the supported range.</td>
      </tr>
      <tr>
        <td>The value of useDepthPrior, height, or width is not within the supported range.</td>
      </tr>
    </tbody></table>

## aclnnRasterizer

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
      <td>The workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device side, obtained through the first-phase API aclnnRasterizerGetWorkspaceSize.</td>
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

  aclnnStatus: returns a status code. For details, see [aclnn_return_code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Only the input case where useDepthPrior is 0 is supported. Parameters dOptional, occlusionTruncation, and useDepthPrior do not take effect in actual computation.

- Deterministic computation:
  - aclnnRasterizer defaults to a deterministic implementation.

## Example

The sample code is for reference only. For the specific compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rasterizer.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs, which need to be custom-constructed according to the API definition.
    std::vector<int64_t> vShape = {3, 4};
    std::vector<int64_t> fShape = {1, 3};
    std::vector<int64_t> dShape = {10, 10};
    std::vector<int64_t> findicesShape = {10, 10};
    std::vector<int64_t> baryShape = {10, 10, 3};
    int64_t height = 10;
    int64_t width = 10;
    float occlusionTruncation = 0.0f;
    int64_t useDepthPrior = 0;
    std::vector<float> vData = {6.0f,       4.0f,       1.0f,      6.9f,       7.0928106f, 0.3491799f,
                                3.0046327f, 6.6574745f, 7.308903f, 7.6934705f, 0.1315008f, 3.9899914f};
    std::vector<int32_t> fData = {2, 1, 0};
    std::vector<float> dData(100, 0.0f);
    std::vector<int32_t> findicesData(100, 0);
    std::vector<float> baryData(10 * 10 * 3, 0.0f);

    void* vDeviceAddr = nullptr;
    void* fDeviceAddr = nullptr;
    void* dDeviceAddr = nullptr;
    void* findicesDeviceAddr = nullptr;
    void* baryDeviceAddr = nullptr;

    aclTensor* v = nullptr;
    aclTensor* f = nullptr;
    aclTensor* d = nullptr;
    aclTensor* findices = nullptr;
    aclTensor* barycentric = nullptr;

    ret = CreateAclTensor(vData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(fData, fShape, &fDeviceAddr, aclDataType::ACL_INT32, &f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(dData, dShape, &dDeviceAddr, aclDataType::ACL_FLOAT, &d);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(findicesData, findicesShape, &findicesDeviceAddr, aclDataType::ACL_INT32, &findices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(baryData, baryShape, &baryDeviceAddr, aclDataType::ACL_FLOAT, &barycentric);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API; change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnRasterizer
    ret = aclnnRasterizerGetWorkspaceSize(v, f, d, width, height, occlusionTruncation, useDepthPrior, findices,
                                          barycentric, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRasterizerGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnRasterizer
    ret = aclnnRasterizer(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRasterizer failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Get the output values and copy the results from device-side memory to host; modify according to the specific API definition.
    auto findicesSize = GetShapeSize(findicesShape);
    std::vector<int32_t> findicesOutData(findicesSize, 0);
    ret = aclrtMemcpy(findicesOutData.data(), findicesSize * sizeof(findicesOutData[0]), findicesDeviceAddr,
                      findicesSize * sizeof(findicesOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    auto barycentricSize = GetShapeSize(baryShape);
    std::vector<float> baryOutData(barycentricSize, 0);
    ret = aclrtMemcpy(baryOutData.data(), barycentricSize * sizeof(baryOutData[0]), baryDeviceAddr,
                      barycentricSize * sizeof(baryOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < findicesSize; i++) {
        LOG_PRINT("findices[%ld] is: %d\n", i, findicesOutData[i]);
    }
    for (int64_t i = 0; i < barycentricSize; i++) {
        LOG_PRINT("barycentric[%ld] is: %f\n", i, baryOutData[i]);
    }

    // 6. Release aclTensor; modify according to the specific API definition.
    aclDestroyTensor(v);
    aclDestroyTensor(f);
    aclDestroyTensor(d);
    aclDestroyTensor(findices);
    aclDestroyTensor(barycentric);

    // 7. Release device resources; modify according to the specific API definition.
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
