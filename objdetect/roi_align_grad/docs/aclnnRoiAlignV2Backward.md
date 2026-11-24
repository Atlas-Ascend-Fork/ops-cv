# aclnnRoiAlignV2Backward

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
- <term>Atlas训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

[aclnnRoiAlignV2](../../roi_align/docs/aclnnRoiAlignV2.md)的反向传播，RoiAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnRoiAlignV2BackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRoiAlignV2Backward”接口执行计算。

```Cpp
aclnnStatus aclnnRoiAlignV2BackwardGetWorkspaceSize(
  const aclTensor*        gradOutput,
  const aclTensor*        boxes,
  const aclIntArray*      inputShape,
  int64_t                 pooledHeight,
  int64_t                 pooledWidth,
  float                   spatialScale,
  int64_t                 samplingRatio,
  bool                    aligned,
  aclTensor*              gradInput,
  uint64_t*               workspaceSize,
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnRoiAlignV2Backward(
  void*                   workspace,
  uint64_t                workspaceSize,
  aclOpExecutor*          executor,
  aclrtStream             stream)
```

## aclnnRoiAlignV2BackwardGetWorkspaceSize

- **参数说明**

  <table class="tg" style="undefined;table-layout: fixed; width: 1570px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 298px">
  <col style="width: 184px">
  <col style="width: 167px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-5agr">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">gradOutput（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">反向传播的输入。</td>
      <td class="tg-0pky">必须与boxes、gradInput数据类型一致。</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">NCHW</td>
      <td class="tg-0pky">4维，shape为（K，C，pooledHeight，pooledWidth）<br>表示反向传播的输入梯度张量一个batch内有K个元素，每个元素有C个尺寸为pooledHeight * pooledWidth的特征图。<br>K需要与boxes第0维保持一致。</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">boxes（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">感兴趣区域box坐标。</td>
      <td class="tg-0pky">必须与gradOutput、gradInput数据类型一致。</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2维，shape为（K，5）<br>5代表box相关信息（image_id，x1，y1，x2，y2）。<br>image_id取值范围[0, B)，向下取整到图像id，B为inputShape第一个值。<br>坐标需满足0 &lt;= x1 &lt;= x2 &lt;= inputWidth/spatialScale、0 &lt;= y1 &lt;= y2 &lt;= inputHeight/spatialScale，由调用方保证。</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">inputShape（aclIntArray*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">正向输入的shape，用来指定反向传播的输出shape。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT32、INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">size大小为4，值为（B，C，inputHeight，inputWidth）<br>表示正向RoiAlign的输入张量一个batch内有B张图像，每个图像有C个尺寸为inputHeight * inputWidth的特征图。<br>B、C、inputHeight、inputWidth均需大于0。</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">pooledHeight（int64_t）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">正向RoiAlign池化后输出图像的高度。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">pooledWidth（int64_t）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">正向RoiAlign池化后输出图像的宽度。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">spatialScale（float）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">乘法空间尺度因子，将ROI坐标从其输入空间尺度转换为池化时使用的尺度，即输入特征图X相对于输入图像的空间尺度。</td>
      <td class="tg-0pky">需大于0。</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">samplingRatio（int64_t）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">RoiAlign中用于计算每个输出元素在H和W方向上的采样频率。</td>
      <td class="tg-0pky">取值非负整数（上限INT32_MAX）；0表示自适应采样（取ceil(roi_size / pooled_size)）；超出范围按参数非法处理。</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">aligned（bool）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">如果为false，则box坐标不做偏移（无对齐）。<br>如果为true，则box坐标像素偏移-0.5来使相邻像素索引更好对齐。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">BOOL</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">gradInput（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">反向传播的输出。</td>
      <td class="tg-0pky">必须与gradOutput、boxes数据类型一致。</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">NCHW</td>
      <td class="tg-0pky">4维，shape为（B，C，inputHeight，inputWidth）</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize（uint64_t*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor（aclOpExecutor**）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回op执行器，包括了算子计算流程。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **返回码：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 290px">
  <col style="width: 134px">
  <col style="width: 844px">
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
      <td>传入的gradOutput、boxes、inputShape、gradInput是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>gradOutput和gradInput的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>gradOutput、boxes、inputShape和gradInput的shape不满足约束限制；inputShape各元素需大于0且不超过INT32_MAX。</td>
    </tr>
    <tr>
      <td>spatialScale需为有限值且大于0，samplingRatio需为非负整数（上限INT32_MAX）；pooledHeight/pooledWidth需大于0且不超过INT32_MAX。</td>
    </tr>
  </tbody>
  </table>

## aclnnRoiAlignV2Backward

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 170px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRoiAlignV2BackwardGetWorkspaceSize获取。</td>
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

- **返回码：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：

  <!-- npu="910" id7 -->
  - <term>Atlas训练系列产品</term>：默认采用原子累加（非确定性）。
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT系列产品</term>：确定性实现。
  <!-- end id8 -->
- inputShape：B、C、inputHeight、inputWidth均需大于0且不超过INT32_MAX；pooledHeight/pooledWidth需大于0且不超过INT32_MAX。
- boxes：坐标需满足 `0 <= x1 <= x2 <= inputWidth/spatialScale`、`0 <= y1 <= y2 <= inputHeight/spatialScale`，由调用方保证；坐标超出inputWidth/spatialScale、inputHeight/spatialScale上界时算子不报错，换算到特征图尺度后落在采样窗口`[-1, inputWidth]×[-1, inputHeight]`之外的采样点不产生梯度（梯度按窗口外采样点的占比相应减少）；反转框（`x1 > x2`或`y1 > y2`）属于违反上述约束的输入，算子不报错，不保证贡献为0。

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：反转框（`x1 > x2`或`y1 > y2`）在aligned=false时ROI宽或高按不小于1.0计算，aligned=true时按负的宽或高参与采样；不保证贡献为0，请勿依赖。
  <!-- end id9 -->
- spatialScale：需为有限值且大于0。
- sample_num/samplingRatio取非负整数，上限INT32_MAX，超出上限按非法参数处理；取0表示自适应采样（ceil(roi_size / pooled_size)）。
- 空tensor：rois数量为0时返回全0；pooledHeight/pooledWidth为0属非法输入。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_align_v2_backward.h"

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
                    aclDataType dataType, aclTensor** tensor) {
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

template <typename T>
int CreateAclNchTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    void* boxesDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* boxes = nullptr;
    aclTensor* gradInput = nullptr;
    const aclIntArray* inputShapeArray = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 统一的清理入口：任一失败路径都释放已申请的 device 内存、tensor、stream 与上下文。
    auto cleanup = [&]() {
        if (gradOutput != nullptr) {
            aclDestroyTensor(gradOutput);
        }
        if (boxes != nullptr) {
            aclDestroyTensor(boxes);
        }
        if (gradInput != nullptr) {
            aclDestroyTensor(gradInput);
        }
        if (inputShapeArray != nullptr) {
            aclDestroyIntArray(inputShapeArray);
        }
        if (gradOutputDeviceAddr != nullptr) {
            aclrtFree(gradOutputDeviceAddr);
        }
        if (boxesDeviceAddr != nullptr) {
            aclrtFree(boxesDeviceAddr);
        }
        if (gradInputDeviceAddr != nullptr) {
            aclrtFree(gradInputDeviceAddr);
        }
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        aclrtResetDevice(deviceId);
        aclFinalize();
    };

    int ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        cleanup();
        return ret;
    }

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradOutputShape = {1, 1, 3, 3};
    std::vector<int64_t> boxesShape = {1, 5};
    std::vector<int64_t> inputShape = {1, 1, 6, 6};

    std::vector<float> gradOutputHostData = {4.5, 6.5, 8.5, 16.5, 18.5, 20.5, 28.5, 30.5, 32.5};
    std::vector<float> boxesHostData = {0.0, -2.0, -2.0, 22.0, 22.0};
    std::vector<float> gradInputHostData = {1.125, 1.125, 1.625, 1.625, 2.125, 2.125, 1.125, 1.125, 1.625,
                                            1.625, 2.125, 2.125, 4.125, 4.125, 4.625, 4.625, 5.125, 5.125,
                                            4.125, 4.125, 4.625, 4.625, 5.125, 5.125, 7.125, 7.125, 7.625,
                                            7.625, 8.125, 8.125, 7.125, 7.125, 7.625, 7.625, 8.125, 8.125};

    ret = CreateAclNchTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT,
                             &gradOutput);
    if (ret != ACL_SUCCESS) {
        cleanup();
        return ret;
    }
    ret = CreateAclTensor(boxesHostData, boxesShape, &boxesDeviceAddr, aclDataType::ACL_FLOAT, &boxes);
    if (ret != ACL_SUCCESS) {
        cleanup();
        return ret;
    }
    inputShapeArray = aclCreateIntArray(inputShape.data(), inputShape.size());
    if (inputShapeArray == nullptr) {
        cleanup();
        return ACL_ERROR_INTERNAL_ERROR;
    }
    ret = CreateAclNchTensor(gradInputHostData, inputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    if (ret != ACL_SUCCESS) {
        cleanup();
        return ret;
    }

    int64_t pooledHeight = 3;
    int64_t pooledWidth = 3;
    int64_t samplingRatio = 2;
    float spatialScale = 0.25f;
    bool aligned = false;

    // 3. 调用CANN算子库API
    ret = aclnnRoiAlignV2BackwardGetWorkspaceSize(gradOutput, boxes, inputShapeArray, pooledHeight, pooledWidth,
                                                  spatialScale, samplingRatio, aligned, gradInput, &workspaceSize,
                                                  &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnRoiAlignV2BackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
        cleanup();
        return ret;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            cleanup();
            return ret;
        }
    }
    ret = aclnnRoiAlignV2Backward(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnRoiAlignV2Backward failed. ERROR: %d\n", ret);
        cleanup();
        return ret;
    }

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        cleanup();
        return ret;
    }

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(inputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
        cleanup();
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    cleanup();
    return 0;
}
```
