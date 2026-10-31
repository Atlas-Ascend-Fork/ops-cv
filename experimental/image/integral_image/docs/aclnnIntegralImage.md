# aclnnIntegralImage

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |

算子注册 SoC 为 `ascend910b`，功能与性能验证平台为 Ascend910B4。

## 功能说明

计算图像的积分图（Summed-Area Table，简称 SAT），使任意矩形区域内的像素和可以在 O(1) 时间内完成。
提供普通积分图 `sat`、平方和积分图 `sqsum` 及倾斜三角区域累加结果 `tilted`。

输入 `image` 支持 $(H,W)$ 与 $(H,W,C)$ 两种 Shape，采用 ND 格式，3D 输入按 HWC 布局组织，各通道独立计算。
输出积分图的 Shape 为 $(H+1,W+1)$ 或 $(H+1,W+1,C)$，即 $H$、$W$ 两维各加 1，通道维保持不变。`sat`、`sqsum` 的第 0 行、第 0 列为零；`tilted` 采用下述坐标与边界定义。

计算公式：

$$sat(y,x) = \sum_{i=0}^{y-1}\sum_{j=0}^{x-1} image(i,j)$$

$$sqsum(y,x) = \sum_{i=0}^{y-1}\sum_{j=0}^{x-1} image(i,j)^2$$

对 $0 \le y < H$、$0 \le x < W$，`tilted` 计算以当前坐标为顶点、向上按 45° 扩展的三角区域：

$$tilted(y,x) = \sum_{i=0}^{y}\ \sum_{j=\max(0,\ x-(y-i))}^{\min(W-1,\ x+(y-i))} image(i,j)$$

边界定义为 $tilted(H,x)=0$、$tilted(y,W)=0$，第 0 行为输入第 0 行。

`sqsum` 与 `tilted` 为可选输出，常用组合如下：

| 形态 | 传入的输出 | 计算内容 |
| :--- | :--- | :--- |
| 形态 1 | `satOut` | 普通积分图 |
| 形态 2 | `satOut` + `sqsumOutOptional` | 普通积分图与平方和积分图 |
| 形态 3 | `satOut` + `sqsumOutOptional` + `tiltedOutOptional` | 三路输出 |

可选输出传 `nullptr` 时，Kernel 跳过对应分支计算。

## 函数原型

### aclnnIntegralImageGetWorkspaceSize

```cpp
aclnnStatus aclnnIntegralImageGetWorkspaceSize(
    const aclTensor *image,
    int64_t sdepth,
    const aclTensor *satOut,
    const aclTensor *sqsumOutOptional,
    const aclTensor *tiltedOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

### aclnnIntegralImage

```cpp
aclnnStatus aclnnIntegralImage(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### 调用时序

1. 调用 `aclnnIntegralImageGetWorkspaceSize`，按本次实际传入的输入/输出组合获取所需 workspace 大小并生成执行器。
2. 按需申请 workspace，调用 `aclnnIntegralImage` 执行计算。
3. 同步 stream 后读取输出。

三种形态需要分别调用 `aclnnIntegralImageGetWorkspaceSize` 获取各自的执行器，不能共用一个执行器。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `image` | 输入 | 输入图像，Shape 为 $(H,W)$ 或 $(H,W,C)$ | UINT8、FLOAT16、FLOAT32 | ND |
| `sdepth` | 输入 | 输出数据类型选择，取值 -1/0/1 | INT64 | - |
| `satOut` | 输出（必选） | 积分图，Shape 为 $(H+1,W+1)$ 或 $(H+1,W+1,C)$ | INT32、FLOAT32 | ND |
| `sqsumOutOptional` | 输出（可选） | 平方和积分图，Shape 同 `satOut` | FLOAT32 | ND |
| `tiltedOutOptional` | 输出（可选） | 倾斜三角区域累加，坐标定义见功能说明，Shape 同 `satOut` | 同 `satOut` | ND |
| `workspaceSize` | 输出 | 本次执行所需的 workspace 字节数 | UINT64 | - |
| `executor` | 输出 | ACLNN 执行器 | `aclOpExecutor*` | - |
| `workspace` | 输入 | workspace 地址；`workspaceSize` 为 0 时可传空指针 | `void*` | - |
| `stream` | 输入 | ACL runtime stream | `aclrtStream` | - |

`sdepth` 取值说明：

| 取值 | 含义 |
| :--- | :--- |
| `-1` | 自动推导：UINT8 输入对应 INT32，FLOAT16/FLOAT32 输入对应 FLOAT32（默认值） |
| `0` | INT32 输出，仅对 UINT8 输入有效、FLOAT16/FLOAT32 输入传入会报错 |
| `1` | FLOAT32 输出，任意合法输入类型均可 |

调用方需按上述规则申请输出，`GetWorkspaceSize` 会检查 `satOut` 和已启用可选输出的完整 Shape 及数据类型。

## 返回值

| 返回值 | 描述 |
| :--- | :--- |
| `ACLNN_SUCCESS` | 接口执行成功 |
| 非零错误码 | 参数校验、Tiling 或执行失败；具体原因见 CANN 运行日志 |

## 约束说明

- `image` 必须为 ND 格式，Rank 必须为 2 或 3，各维必须大于 0。
- `image` 支持 UINT8、FLOAT16、FLOAT32；`sqsumOutOptional` 固定为 FLOAT32。
- `satOut` 与 `tiltedOutOptional` 的数据类型必须与 `sdepth` 推导结果一致。
- 调用方预先分配所有启用输出；每个输出的 Rank 和各维尺寸必须与输入推导结果一致。
- H+1、W+1 必须能用 INT64 表示；HWC 分块配置还需满足平台 UB 容量和 AIV 核数约束。
- 3D 输入按 HWC 布局解释，通道数 $C$ 不要求 32 元素对齐。
- $H$、$W$ 不要求 32 元素对齐，支持 $H=1$、$W=1$、单行、单列等退化尺寸。
- INT32 结果采用 32 位整数回绕语义。$255 \times H \times W \le 2^{31}-1$ 可保证所有 UINT8 输入均不溢出；需要更大数值范围时可使用 `sdepth=1`。
- `sqsum` 输出为 FLOAT32；FLOAT16 输入在 Kernel 内以 FLOAT32 累加，浮点精度受累加量级与累加顺序影响。
- 不支持原地计算，输入与输出不得共享存储。
- 未列出的 `sdepth` 取值（如 `2`）会被 Host 侧校验拒绝。

## 调用示例

完整示例见 [`../examples/test_aclnn_integral_image.cpp`](../examples/test_aclnn_integral_image.cpp)，
该示例覆盖三种输出形态、UINT8/FLOAT16/FLOAT32 输入、非对齐尺寸与 3D 多通道场景。
