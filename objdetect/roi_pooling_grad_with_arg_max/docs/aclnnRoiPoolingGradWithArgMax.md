# aclnnRoiPoolingGradWithArgMax

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
- <term>Atlas推理系列产品 </term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：实现RoiPoolingWithArgMax的反向。遍历每个ROI的池化结果，将feature map坐标上的反向梯度贡献累加，即完成整张图上的反向计算。
- 计算公式：

  $$
  \frac{\partial L}{\partial x_i} = \sum_{r}\sum_{j}[i = i^*(r,j)]\frac{\partial L}{\partial y_{rj}}
  $$

  其中，

  $$
  [i = i^*(r,j)]  = \begin{cases} 1, & i = i^*(r,j) \\ 0, & otherwise \end{cases}
  $$

  判决函数`[i = i^*(r,j)]`表示i节点是否被候选区域r的第j个输出节点选为最大值输出

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnRoiPoolingGradWithArgMax”接口执行计算。

```cpp
aclnnStatus aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize(
    const aclTensor*      gradOutput,
    const aclTensor*      gradInputRef,
    const aclTensor*      rois,
    const aclTensor*      argmax,
    int64_t               pooledH,
    int64_t               pooledW,
    double                spatialScale,
    uint64_t*             workspaceSize,
    aclOpExecutor**       executor);
```

```cpp
aclnnStatus aclnnRoiPoolingGradWithArgMax(
  void*                   workspace,
  uint64_t                workspaceSize,
  aclOpExecutor*          executor,
  aclrtStream             stream)
```

## aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
    <col style="width: 200px">
    <col style="width: 120px">
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 212px">
    <col style="width: 120px">
    <col style="width: 250px">
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
      <td>gradOutput（aclTensor*）</td>
      <td>输入</td>
      <td>梯度输入。</td>
      <td>-</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4维，shape为(roisN, C, pooledH, pooledW)</td>
      <td>√</td>
    </tr>
      <tr>
      <td>gradInputRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>输出结果。</td>
      <td>-</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4维，shape为(N, C, H, W)</td>
      <td>√</td>
    </tr>
      <tr>
      <td>rois（aclTensor*）</td>
      <td>输入</td>
      <td>ROI区域。</td>
      <td>-</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2维，shape为(roisN, 5)<br>5指(batchId, x1, x2, y1, y2)</td>
      <td>√</td>
    </tr>
      <tr>
      <td>argmax（aclTensor*）</td>
      <td>输入</td>
      <td>指定目标梯度的索引。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>4维，shape为(roisN, C, pooledH, pooledW)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pooledH（int64_t）</td>
      <td>属性</td>
      <td>池化高度。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>pooledW（int64_t）</td>
      <td>属性</td>
      <td>池化宽度。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatialScale（double）</td>
      <td>属性</td>
      <td>输入坐标映射到ROI坐标的缩放比例。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1124px"><colgroup>
  <col style="width: 284px">
  <col style="width: 124px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的gradOutput、rois、argmax、gradInputRef是空指针。</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>gradOutput、rois、argmax、gradInputRef的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradOutput、argmax与gradInputRef具有相同的数据类型。</td>
    </tr>
    <tr>
      <td>gradOutput、argmax、gradInputRef的shape大小为4，rois的shape大小为2。</td>
    </tr>
    <tr>
      <td>gradOutput、argmax、rois的shape[0]相等。</td>
    </tr>
    <tr>
      <td>gradOutput、argmax的shape[1]相等。</td>
    </tr>
    <tr>
      <td>gradOutput、argmax的shape[2]等于pooledH，shape[3]等于pooledW。</td>
    </tr>
    <tr>
      <td>rois的值大于等于0。</td>
    </tr>
    <tr>
      <td>pooledH、pooledW大于0。</td>
    </tr>
    <tr>
      <td>rois[:, 1] 小于rois[:, 2] 且rois[:, 3] 小于rois[:, 4]。</td>
    </tr>
    <tr>
      <td>rois.shape[0]、gradOutput.shape[0]小于等于1024。</td>
    </tr>
    <tr>
      <td>gradInputRef.shape[1]等于gradOutput.shape[1]。</td>
    </tr>
  </tbody>
  </table>

## aclnnRoiPoolingGradWithArgMax

* ​**参数说明**​：

  <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 900px"><colgroup>
      <col style="width: 150px">
      <col style="width: 100px">
      <col style="width: 650px">
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
          <td>在Device侧申请的workspace大小，由第一段接口aclnnRoiPoolingGradWithArgMax获取。</td>
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
      </tbody></table>
      </div>

* ​**返回值**​：

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- gradOutput、rois、argmax、gradInputRef的数据类型在支持的范围之内。
- gradOutput、argmax与gradInputRef具有相同的数据类型。
- gradOutput、argmax、gradInputRef的shape大小为4，rois的shape大小为2。
- gradOutput、argmax、rois的shape[0]相等。
- gradOutput、argmax的shape[1]相等。
- gradOutput、argmax的shape[2]等于pooledH和shape[3]等于pooledW。
- rois的值大于等于0。
- pooledH、pooledW大于0。
- rois[:, 1] 小于rois[:, 2] 且rois[:, 3] 小于rois[:, 4]。
- rois.shape[0]、gradOutput.shape[0]小于等于1024。
- 确定性计算：

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：aclnnRoiPoolingGradWithArgMax默认确定性实现。
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT系列产品</term>：aclnnRoiPoolingGradWithArgMax默认非确定性实现，不支持开启。
  <!-- end id8 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_pooling_grad_with_arg_max.h"
#include <iostream>
using namespace std;


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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 2. 申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 3. 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. 调用acl进行device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    aclTensor* gradOutput = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    std::vector<int64_t> gradOutputShape = {1, 32, 2, 2};
    std::vector<float> gradOutputHostData(128, 1.0); // 128：创建包含1*32*2*2=128个元素的向量
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* gradInputRef = nullptr;
    void* gradInputRefDeviceAddr = nullptr;
    std::vector<int64_t> gradInputRefShape = {4, 32, 3, 3};
    std::vector<float> gradInputRefHostData(1152, 1.0);
    ret = CreateAclTensor(gradInputRefHostData, gradInputRefShape, &gradInputRefDeviceAddr, aclDataType::ACL_FLOAT, &gradInputRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* rois = nullptr;
    void* roisDeviceAddr = nullptr;
    std::vector<int64_t> roisShape = {1, 5};
    std::vector<float> roisHostData = {0.0, 0.0, 1.0, 0.0, 1.0};
    ret = CreateAclTensor(roisHostData, roisShape, &roisDeviceAddr, aclDataType::ACL_FLOAT, &rois);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* argmax = nullptr;
    void* argmaxDeviceAddr = nullptr;
    std::vector<int64_t> argmaxShape = {1, 32, 2, 2};
    std::vector<int32_t> argmaxHostData(128, 3.0);
    ret = CreateAclTensor(argmaxHostData, argmaxShape, &argmaxDeviceAddr, aclDataType::ACL_INT32, &argmax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int32_t pooledH = 2;
    int32_t pooledW = 2;
    double spatialScale = 1.0;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 4. 调用aclnnRoiPoolingGradWithArgMax第一段接口
    ret = aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize(gradOutput, gradInputRef, rois, argmax, pooledH, pooledW, spatialScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用aclnnRoiPoolingGradWithArgMax第二段接口
    ret = aclnnRoiPoolingGradWithArgMax(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingGradWithArgMax failed. ERROR: %d\n", ret); return ret);

    // 6. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnRoiPoolingGradWithArgMax run success.\n");

    // 7. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(gradInputRefShape, &gradInputRefDeviceAddr);

    // 8. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(gradInputRef);
    aclDestroyTensor(rois);
    aclDestroyTensor(argmax);

    // 9. 释放device资源
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(gradInputRefDeviceAddr);
    aclrtFree(roisDeviceAddr);
    aclrtFree(argmaxDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 10. acl去初始化
    aclFinalize();

    return 0;
}
```
