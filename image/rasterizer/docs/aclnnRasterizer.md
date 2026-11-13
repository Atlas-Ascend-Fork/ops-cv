# aclnnRasterizer

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：实现光栅化计算。根据给定的三维空间中的点和面，获取屏幕中每个像素点的最小深度及其对应的面片索引，并计算该面片的重心坐标透视矫正插值。

- 计算公式：

  $findices$记录每个像素点最小深度对应的面片索引，$barycentric$记录每个顶点相对于$findices$中记录的面的重心坐标透视矫正插值。
  计算过程中使用的Z-Buffer记录每个像素点$(x, y)$的最小深度$z_{\min}(x, y)$以及该深度对应的三角形面片索引$\text{face\_idx}(x, y)$。

  计算过程如下：
  对空间中的每个三角形面片$f$：

  1. 将$f$的三个顶点坐标$(v_0, v_1, v_2)$转换为屏幕坐标$(v_{s0}, v_{s1}, v_{s2})$
  2. 根据$(v_{s0}, v_{s1}, v_{s2})$计算包围$f$的矩形范围
  3. 对矩形内每个像素点$v_i = (x_i, y_i)$，执行以下操作：

     a. 计算像素中心坐标$v_c$
     b. 计算$v_c$相对于三角形$f$的重心坐标$(\alpha, \beta, \gamma)$
     c. 根据$(\alpha, \beta, \gamma)$判断$v_c$是否在三角形内部。若$v_c$不在三角形内部，则处理矩形内下个像素点，否则执行下述步骤
     d. 使用$(\alpha, \beta, \gamma)$和$(v_{s0}, v_{s1}, v_{s2})$得到当前像素的深度值depth
     e. 若启用了深度先验，则使用深度图计算深度阈值depth_thres；否则，直接执行下一步“Z-Buffer更新”

        - 使用深度图计算深度阈值depth_thres
        - 如果depth < depth_thres，处理矩形内下个像素点，否则执行下述步骤

     f. Z-Buffer更新：

       - 若$depth < z_{\min}(x_i, y_i)$：

         $$
         \quad z_{\min}(x_i, y_i) \gets \text{depth} \\
         \quad \text{face\_idx}(x_i, y_i) \gets f
         $$

       - 若$depth = z_{\min}(x_i, y_i)$：

         $$
         \quad \text{face\_idx}(x_i, y_i) \gets \min(\text{face\_idx}(x_i, y_i),\ f)
         $$

  按上述步骤对空间中所有的三角形面片进行处理后，对大小为$height * width$的屏幕上每个像素点$v_i = (x_i, y_i)$：

  1. 取Z-Buffer中$v_i$对应的面片索引$f_{idx}$，$findices (x_i, y_i) \gets f_{idx}$
  2. 将$f$的三个顶点坐标$(v_0, v_1, v_2)$转换为屏幕坐标$(v_{s0}, v_{s1}, v_{s2})$
  3. 计算$v_i$的中心点坐标$v_c$
  4. 计算$v_c$相对于三角形$f$的重心坐标$(\alpha, \beta, \gamma)$
  5. 使用$(\alpha, \beta, \gamma)$计算透视矫正插值$(\tilde{\alpha}, \tilde{\beta}, \tilde{\gamma})$
  6. $barycentric(x_i, y_i) \gets (\tilde{\alpha}, \tilde{\beta}, \tilde{\gamma})$

  以下是涉及的各种具体计算方法：

  - 顶点$v = (x, y, z, w)$转换为屏幕坐标$v_s = (x_s, y_s, z_s)$

    $$
    x_s = (x / w * 0.5 + 0.5) * (width - 1) + 0.5\\
    y_s = (0.5 + 0.5 * y / w) * (height - 1) + 0.5\\
    z_s = z / w * 0.49999 + 0.5
    $$

  - 点$v$相对于三角形$(v_0, v_1, v_2)$的重心坐标$(\alpha, \beta, \gamma)$

    1. 分别计算三角形$(v_0, v_1, v_2)$、$(v_0, v, v_2)$和$(v_0, v_1, v)$的有向面积$area$、$beta\_tri$和$gamma\_tri$
    2. 若$area$为0，则$\alpha = \beta = \gamma = -1$，否则

      $$
      \beta = beta\_tri / area\\
      \gamma = gamma\_tri / area\\
      \alpha = 1 - \beta - \gamma
      $$

  - 由顶点$v_0 = (x_0, y_0, z_0)$、$v_1 = (x_1, y_1, z_1)$和$v_2 = (x_2, y_2, z_2)$组成的三角形的有向面积

    $$
    area = (x_2 - x_0) * (y_1 - y_0) - (x_1 - x_0) * (y_2 - y_0)
    $$

  - 结合重心坐标$(\alpha, \beta, \gamma)$和三角形屏幕坐标$(v_0 = (x_0, y_0, z_0), v_1 = (x_1, y_1, z_1), v_2 = (x_2, y_2, z_2))$计算像素点$v = (x, y)$的深度$depth$

    $$
    depth = \alpha * z_0 + \beta * z_1 + \gamma * z_2
    $$

  - 结合深度图$d$，遮挡截断$occlusion\_truncation$计算点$v = (x, y)$的深度阈值$depth\_thres$

    $$
    depth\_thres = d(x, y) * 0.49999 + 0.5 + occlusion\_truncation
    $$

  - 根据重心坐标$(\alpha, \beta, \gamma)$判断点是否在三角形内：
    如果$\alpha >= 0$且$\beta >= 0$且$\gamma >= 0$则点在三角形内（包括在三角形边上），否则点不在三角形内。
  - 结合重心坐标$(\lambda_0, \lambda_1, \lambda_2)$以及三角形的三个顶点坐标$v_0 = (x_0, y_0, z_0, w_0)$、$v_1 = (x_1, y_1, z_1, w_1)$和$v_2 = (x_2, y_2, z_2, w_2)$计算透视矫正插值$(\lambda_0^{corrected}, \lambda_1^{corrected}, \lambda_2^{corrected})$

    $$
    \lambda_i^{corrected} = \frac{\lambda_i / w_i} { \sum (\lambda_j / w_j)}
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnRasterizerGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRasterizer”接口执行计算。

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

- **参数说明**

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
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>v(aclTensor*)</td>
      <td>输入</td>
      <td>表示空间中顶点坐标的输入张量。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为(numVertices, 4)，其中numVertices表示顶点数量，为正整数。每个顶点坐标表示为(x, y, z, w)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>f(aclTensor*)</td>
      <td>输入</td>
      <td>表示空间中的面的输入张量。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为(numFaces, 3)，其中numFaces表示空间中面的数量，为正整数。每个面是一个三角形，三角形每个顶点表示为顶点在v中的索引，因此f中元素取值应当是对v中元素的合法索引，即取值范围为[0, numVertices-1]。由调用者保证f中元素合法。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dOptional(aclTensor*)</td>
      <td>输入</td>
      <td>表示深度图的输入张量，用于计算深度阈值。</td>
      <td><ul><li>可选输入，支持空Tensor。</li><li>非空Tensor时，shape为(height, width)。</li><li>此参数不生效。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>width(int64_t)</td>
      <td>输入</td>
      <td>屏幕宽度。</td>
      <td>取值范围[1, 4096]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>height(int64_t)</td>
      <td>输入</td>
      <td>屏幕高度。</td>
      <td>取值范围[1, 4096]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>occlusionTruncation(double)</td>
      <td>输入</td>
      <td>遮挡截断，用于计算深度阈值。</td>
      <td>此参数不生效。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  <tr>
      <td>useDepthPrior(int64_t)</td>
      <td>输入</td>
      <td>表示是否应用深度先验。</td>
      <td><ul><li>值为0或1。1表示应用深度先验，0表示不应用深度先验。</li><li>当前算子不支持应用深度先验，因此值固定为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>findicesOut(aclTensor*)</td>
      <td>输出</td>
      <td>表示屏幕中每个像素点最小深度对应的面的索引。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为(height, width)，其中height是屏幕高度，width是屏幕宽度。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
  <tr>
      <td>barycentricOut(aclTensor*)</td>
      <td>输出</td>
      <td>表示屏幕中每个像素点相对于最小深度对应的面的重心坐标透视矫正插值的输出张量。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为(height, width, 3)，其中height是屏幕高度，width是屏幕宽度。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor**)</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
    <col style="width: 268px">
    <col style="width: 140px">
    <col style="width: 762px">
    </colgroup>
    <thead>
      <tr>
        <th>返回码</th>
        <th>错误码</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>传入的v、f、findicesOut或barycentricOut是空指针。</td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>v、f、findicesOut或barycentricOut的数据类型不在支持范围之内。</td>
      </tr>
      <tr>
        <td>v、f、findicesOut或barycentricOut的shape不在支持范围之内。</td>
      </tr>
      <tr>
        <td>useDepthPrior、height或width取值不在支持范围之内。</td>
      </tr>
    </tbody></table>

## aclnnRasterizer

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRasterizerGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>
- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 仅支持useDepthPrior为0输入场景，参数dOptional、occlusionTruncation、useDepthPrior在实际计算中不生效。

- 确定性计算：
  - aclnnRasterizer默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

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
    // 固定写法，资源初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
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

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnRasterizer第一段接口
    ret = aclnnRasterizerGetWorkspaceSize(v, f, d, width, height, occlusionTruncation, useDepthPrior, findices,
                                          barycentric, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRasterizerGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnRasterizer第二段接口
    ret = aclnnRasterizer(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRasterizer failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
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

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(v);
    aclDestroyTensor(f);
    aclDestroyTensor(d);
    aclDestroyTensor(findices);
    aclDestroyTensor(barycentric);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
