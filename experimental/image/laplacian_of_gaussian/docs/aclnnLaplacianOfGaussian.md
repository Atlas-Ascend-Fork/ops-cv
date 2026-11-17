# aclnnLaplacianOfGaussian

[📄 查看算子定义](../op_host/laplacian_of_gaussian_def.cpp)

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：计算单通道灰度图像的高斯拉普拉斯（Laplacian of Gaussian，LoG）二阶响应，输出与输入 shape 相同的 FLOAT32 带符号响应。
- 实现为可分离二维 FIR：先沿 W 方向、再沿 H 方向各做一次一维卷积，两个方向分别使用高斯平滑系数 `g` 与高斯二阶导系数 `d`。
- 功能语义参考 OpenCV 的 `cv::GaussianBlur` + `cv::Laplacian` 组合与 SciPy 的 `scipy.ndimage.gaussian_laplace`。

记 σ = `sigma`、r = `radius`、一维核长 k = 2r+1：

$$
g[i] = \frac{\exp\left(-\frac{(i-r)^2}{2\sigma^2}\right)}{\sum_{j=0}^{k-1}\exp\left(-\frac{(j-r)^2}{2\sigma^2}\right)},\qquad
d[i] = \left(\frac{(i-r)^2}{\sigma^2} - 1\right)\frac{g[i]}{\sigma^2}
$$

$$
y = (d \otimes g + g \otimes d) \otimes x
$$

其中 $\otimes$ 表示一维卷积，`g`、`d` 均为偶对称核。

## 函数原型

本接口采用[两段式接口](../../../../docs/zh/context/two_phase_api.md)：先调用 `aclnnLaplacianOfGaussianGetWorkspaceSize` 获取 workspace 大小和执行器，再调用 `aclnnLaplacianOfGaussian` 执行计算。

```Cpp
aclnnStatus aclnnLaplacianOfGaussianGetWorkspaceSize(
    const aclTensor* x,
    double           sigma,
    int64_t          radius,
    double           truncate,
    char*            borderTypeOptional,
    const aclTensor* out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor);
```

```Cpp
aclnnStatus aclnnLaplacianOfGaussian(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream);
```

接口及头文件由[构建配置](../op_host/CMakeLists.txt)中的 `add_modules_sources(OPTYPE laplacian_of_gaussian ACLNNTYPE aclnn)` 根据[算子定义](../op_host/laplacian_of_gaussian_def.cpp)自动生成，源码目录不单独维护 `op_api` 实现。构建并安装自定义算子包后，使用生成的 `aclnn_laplacian_of_gaussian.h`，并链接同一安装包的 `libcust_opapi.so`。

## aclnnLaplacianOfGaussianGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述及使用说明 | 数据类型 | 数据格式 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| x（const aclTensor*） | 输入 | 单通道灰度图像。不支持空 Tensor。 | UINT8（`ACL_UINT8`）、FLOAT16（`ACL_FLOAT16`）、FLOAT32（`ACL_FLOAT`） | ND | `(H, W)`、`(N, H, W)` 或 `(N, 1, H, W)` |
| sigma（double） | 输入 | 高斯标准差，必须为有限正数。 | DOUBLE | - | - |
| radius（int64_t） | 输入 | 卷积半径。取 -1 表示按 `floor(float(sigma) * float(truncate) + 0.5)` 自动计算；非负值表示显式指定。 | INT64 | - | - |
| truncate（double） | 输入 | 截断系数，参与自动半径计算，必须为有限正数；即使显式指定 radius 也需有效。 | DOUBLE | - | - |
| borderTypeOptional（char*） | 输入 | 以 `\0` 结尾的边界模式字符串，支持 `"REFLECT"`、`"REFLECT_101"`、`"REPLICATE"`、`"CONSTANT"`，区分大小写。 | STRING | - | - |
| out（const aclTensor*） | 输出 | 调用者预先创建的响应 Tensor，shape 与 x 相同。指针的 const 修饰限定 Tensor 描述对象，不表示输出数据只读。 | FLOAT32（`ACL_FLOAT`） | ND | 与 x 相同 |
| workspaceSize（uint64_t*） | 输出 | 返回执行所需的 Device 侧 workspace 大小，单位为字节。 | UINT64 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回包含本次算子计算流程的执行器。 | aclOpExecutor | - | - |

### 返回值

返回类型为 `aclnnStatus`，0 表示成功。通用返回码见 [aclnn 返回码](../../../../docs/zh/context/aclnn_return_code.md)。

| 返回码 | 错误码 | 说明 |
| :--- | :--- | :--- |
| ACLNN_SUCCESS | 0 | 成功获取 workspace 大小和执行器。 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 通用参数校验检测到非法空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 通用参数校验检测到不合法参数，例如不支持的数据类型、非有限 sigma、非法 border_type。 |
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | Tiling 阶段校验失败，例如 rank/通道不合法、核长超出 2048 字节 tiling 容量、UB 预算不足。 |

非法参数可能在通用校验或 Tiling 阶段被发现，具体错误码以实际返回值为准；可通过 `aclGetRecentErrMsg` 获取详细错误信息。

## aclnnLaplacianOfGaussian

### 参数说明

| 参数名 | 输入/输出 | 描述及使用说明 |
| :--- | :--- | :--- |
| workspace（void*） | 输入 | Device 侧 workspace 地址。按第一段接口返回的 workspaceSize 申请；workspaceSize 大于 0 时必须提供有效地址。 |
| workspaceSize（uint64_t） | 输入 | workspace 的字节数，使用第一段接口返回的值。 |
| executor（aclOpExecutor*） | 输入 | 第一段接口为本次调用生成的执行器。 |
| stream（aclrtStream） | 输入 | 执行算子的 Stream，与输入、输出及 workspace 所属 Device 一致。 |

### 返回值

返回类型为 `aclnnStatus`，0 表示任务提交成功，其他状态码见 [aclnn 返回码](../../../../docs/zh/context/aclnn_return_code.md)。接口按 Stream 异步执行；在读取输出或释放 Device 内存前，应先同步 Stream 并检查同步返回值。

不要重复调用第二段接口来复用已执行的 executor；再次计算时按[两段式接口](../../../../docs/zh/context/two_phase_api.md)重新获取执行器。

## 约束说明

- 输入、输出使用连续 ND Tensor。算子描述中已声明 `.AutoContiguous()`，框架会在必要时将非连续输入转换为连续布局；调用者仍应优先传入连续 Tensor。
- 输入 rank 必须为 2、3 或 4；rank 4 时通道数必须为 1；输出 shape 与输入完全一致。
- `N`、`H`、`W` 必须为正数。
- `sigma`、`truncate` 必须为有限正数；`radius >= -1`。`radius = -1` 时按 `floor(float(sigma) * float(truncate) + 0.5)` 推导。
- 核长 `k = 2*radius+1` 的权重（`g`、`d` 各 k 项）需放入 2048 字节 tiling 存储的尾部，超限时接口返回失败，不做静默降级。
- 边界模式语义：`REFLECT` 重复端点，`REFLECT_101` 不重复端点，`REPLICATE` 复制端点，`CONSTANT` 补 0。
- 输入和输出的 Device 内存不能重叠；输入、输出及 workspace 应保持有效，直到对应 Stream 上的任务完成。
- 输入 dtype 支持 UINT8 / FLOAT16 / FLOAT32，输出固定 FLOAT32；不保证与 OpenCV 在极小 sigma 等退化输入上的数值一致。

## 调用示例

完整的输入构造、Device 和 Stream 初始化、输出校验及资源释放，请参考 [test_aclnn_laplacian_of_gaussian.cpp](../examples/test_aclnn_laplacian_of_gaussian.cpp)。通用构建流程见[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)；运行前需先构建并安装包含本算子的自定义算子包，确保使用同一安装包的头文件和动态库。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_laplacian_of_gaussian.h"

int RunLaplacianOfGaussian(const aclTensor* x, const aclTensor* y, aclrtStream stream)
{
    double sigma = 1.0;
    int64_t radius = -1;   // -1: 按 sigma 与 truncate 自动计算
    double truncate = 4.0;
    char borderTypeOptional[] = "REFLECT";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnLaplacianOfGaussianGetWorkspaceSize(
        x, sigma, radius, truncate, borderTypeOptional, y, &workspaceSize, &executor);
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
    ret = aclnnLaplacianOfGaussian(workspace, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}
```
