# aclnnDistanceTransform

[📄 查看算子定义](../op_host/distance_transform_def.cpp)

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | × |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 接口功能：计算二值图像中每个前景像素到最近背景像素的距离，输出与输入shape相同的FLOAT32距离图。输入值为0表示背景，非0值表示前景；背景像素的输出为0。
- 支持L1、L2和CHESSBOARD三种距离类型。Rank 3和Rank 4输入中的各幅图像独立计算，图像外部不作为背景。
- 功能语义参考OpenCV `cv2.distanceTransform`。全前景图像没有真实背景，按本算子的约定返回有限距离，具体见[约束说明](#约束说明)。

设像素坐标为$p=(p_h,p_w)$，图像内背景像素集合为$B$。当$B$非空时，三种精确距离的定义如下：

$$
\begin{aligned}
y_{L1}(p) &= \min_{q\in B}\left(|p_h-q_h|+|p_w-q_w|\right), \\
y_{L2}(p) &= \min_{q\in B}\sqrt{(p_h-q_h)^2+(p_w-q_w)^2}, \\
y_{CHESSBOARD}(p) &= \min_{q\in B}\max\left(|p_h-q_h|,|p_w-q_w|\right).
\end{aligned}
$$

`distanceType`与`maskSize`的关系如下：

| distanceType | maskSize | 计算语义 |
| :--- | :--- | :--- |
| `L1` | 0、3、5 | maskSize仅作合法性校验，不参与算法选择，实际固定按3×3邻域计算L1距离。 |
| `L2` | 0 | 精确欧氏距离，对应`cv2.DIST_MASK_PRECISE`。 |
| `L2` | 3 | 使用3×3 chamfer掩码计算近似L2距离。 |
| `L2` | 5 | 使用5×5 chamfer掩码计算近似L2距离。 |
| `CHESSBOARD` | 0、3、5 | maskSize仅作合法性校验，不参与算法选择，实际固定按3×3邻域计算棋盘距离。 |

## 函数原型

本接口采用[两段式接口](../../../../docs/zh/context/two_phase_api.md)：先调用`aclnnDistanceTransformGetWorkspaceSize`获取workspace大小和执行器，再调用`aclnnDistanceTransform`执行计算。

```Cpp
aclnnStatus aclnnDistanceTransformGetWorkspaceSize(
    const aclTensor* x,
    char*            distanceType,
    int64_t          maskSize,
    const aclTensor* y,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor);
```

```Cpp
aclnnStatus aclnnDistanceTransform(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream);
```

接口及头文件由[构建配置](../op_host/CMakeLists.txt)中的`add_modules_sources(OPTYPE distance_transform ACLNNTYPE aclnn)`根据[算子定义](../op_host/distance_transform_def.cpp)自动生成，源码目录不单独维护`op_api`实现。构建并安装自定义算子包后，使用生成的`aclnn_distance_transform.h`，并链接同一安装包的`libcust_opapi.so`。

`distanceType`对应算子属性`distance_type`，属性默认值为`"L2"`；`maskSize`对应属性`mask_size`，属性默认值为0。上述C接口没有省略这两个实参的调用形式，需要显式传入属性值。

## aclnnDistanceTransformGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述及使用说明 | 数据类型 | 数据格式 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| x（const aclTensor*） | 输入 | 二值图像Tensor。0表示背景，非0值表示前景，推荐使用0或1。不支持空Tensor。 | UINT8（`ACL_UINT8`） | ND | `(H, W)`、`(N, H, W)`或`(N, 1, H, W)` |
| distanceType（char*） | 输入 | 以`\0`结尾的距离类型字符串，支持`"L1"`、`"L2"`、`"CHESSBOARD"`，区分大小写。 | STRING | - | - |
| maskSize（int64_t） | 输入 | 取值为0、3或5。L2模式按照实际值选择精确或近似距离；L1和CHESSBOARD模式下，maskSize仅作合法性校验，不参与算法选择，实际固定按3×3邻域计算。 | INT64 | - | - |
| y（const aclTensor*） | 输出 | 调用者预先创建的距离图Tensor，shape与x相同。不支持空Tensor。指针的const修饰限定Tensor描述对象，不表示输出数据只读。 | FLOAT32（`ACL_FLOAT`） | ND | 与x相同 |
| workspaceSize（uint64_t*） | 输出 | 返回执行所需的Device侧workspace大小，单位为字节。应传入有效的主机侧地址。 | UINT64 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回包含本次算子计算流程的执行器。应传入有效的主机侧地址。 | aclOpExecutor | - | - |

### 返回值

返回类型为`aclnnStatus`，0表示成功。通用返回码见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

| 返回码 | 错误码 | 说明 |
| :--- | :--- | :--- |
| ACLNN_SUCCESS | 0 | 成功获取workspace大小和执行器。 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 通用参数校验检测到非法空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 通用参数校验检测到不合法的参数，例如不支持的数据类型。 |
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | Tiling阶段校验或处理失败，例如Shape、属性或精确L2的UB容量不满足要求。 |

非法参数可能在通用校验或Tiling阶段被发现，具体错误码以实际返回值为准；可通过`aclGetRecentErrMsg`获取详细错误信息。

## aclnnDistanceTransform

### 参数说明

| 参数名 | 输入/输出 | 描述及使用说明 |
| :--- | :--- | :--- |
| workspace（void*） | 输入 | Device侧workspace地址。按第一段接口返回的workspaceSize申请；workspaceSize大于0时必须提供有效地址。 |
| workspaceSize（uint64_t） | 输入 | workspace的字节数，使用第一段接口返回的值。 |
| executor（aclOpExecutor*） | 输入 | 第一段接口为本次调用生成的执行器。 |
| stream（aclrtStream） | 输入 | 执行算子的Stream，与输入、输出及workspace所属Device一致。 |

### 返回值

返回类型为`aclnnStatus`，0表示任务提交成功，其他状态码见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。接口按Stream异步执行；在读取输出或释放Device内存前，应先同步Stream并检查同步返回值。

不要重复调用第二段接口来复用已执行的executor。再次计算时，按照[两段式接口](../../../../docs/zh/context/two_phase_api.md)重新获取执行器。

## 约束说明

- 输入、输出使用连续ND Tensor，保持原有维度顺序，不引入布局转换。非连续Tensor应先转换为连续Tensor再调用。
- 输入rank必须为2、3或4；Rank 4输入的通道数必须为1，输出shape与输入完全一致。
- `N`、`H`和`W`必须为正数。`H`不超过2147483647（INT32_MAX），`W`不超过1073741824（2^30），Tensor总元素数不超过4294967295（UINT32_MAX）。实际可处理规模还受可用内存限制。
- 精确L2模式需要在UB中保存列方向中间状态，过大的`H`可能因UB容量不足而在Tiling阶段被拒绝；上述整数范围不是所有Shape均可执行的保证。
- 输入和输出的Device内存不能重叠。输入、输出及workspace应保持有效，直到对应Stream上的任务完成。
- `distanceType`必须使用受支持的大小写形式，`maskSize`只能为0、3或5。L1和CHESSBOARD模式也不接受其他maskSize值。
- 全背景图像的输出全部为0。全前景图像按下表返回有限哨兵值，不保证与OpenCV在无背景输入上的返回值相同。

| distanceType | maskSize | 全前景图像的返回值 |
| :--- | :--- | :--- |
| `L1` | 0、3、5 | `H + W` |
| `L2` | 0 | `sqrt(H² + W²)` |
| `L2` | 3、5 | `H + W` |
| `CHESSBOARD` | 0、3、5 | `max(H, W)` |

## 调用示例

完整的输入构造、Device和Stream初始化、三种距离类型调用、输出校验及资源释放，请参考[test_aclnn_distance_transform.cpp](../examples/test_aclnn_distance_transform.cpp)。通用构建流程见[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)；运行前需先构建并安装包含DistanceTransform的自定义算子包，确保使用同一安装包的头文件和动态库。

以下代码展示核心调用流程，假设x和y已经按上述规格创建、Device已初始化且stream有效：

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_distance_transform.h"

int RunDistanceTransform(const aclTensor* x, const aclTensor* y, aclrtStream stream)
{
    char distanceType[] = "L2";
    const int64_t maskSize = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnDistanceTransformGetWorkspaceSize(
        x, distanceType, maskSize, y, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }
    ret = aclnnDistanceTransform(workspace, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}
```
