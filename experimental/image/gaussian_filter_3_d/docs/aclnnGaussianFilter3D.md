# aclnnGaussianFilter3D

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品 </term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

三维高斯滤波算子，对输入体数据 `x` 沿 D/H/W 三个空间轴依次执行一维高斯卷积（可分离实现），输出与输入同 shape、同 dtype 的滤波结果 `y`，语义对齐 `scipy.ndimage.gaussian_filter`。

三维高斯核可分离，等价于沿 W、H、D 三个轴各做一次一维高斯卷积：

$$
w[i] = \frac{\exp(-0.5\,i^2/\sigma^2)}{\sum_{j=-r}^{r}\exp(-0.5\,j^2/\sigma^2)},\quad i \in [-r, r]
$$

$$
y = \mathrm{conv1d}_{D}\big(\mathrm{conv1d}_{H}(\mathrm{conv1d}_{W}(x, w), w), w\big)
$$

其中截断系数 `truncate` 固定为 4.0，核半径 `r = int(4 * sigma + 0.5)`（round-half-up），核大小为 `2r+1`。权重在 Host 侧按 fp64 生成并归一化后转 fp32 随 TilingData 下发，Device 侧不做 exp 计算，三轴共用同一组权重。

边界处理模式 `mode` 与 `scipy.ndimage` 一一对应：

| mode | 语义 | 等效 numpy.pad |
|---|---|---|
| REFLECT（默认） | 重复边缘像素 | symmetric |
| CONSTANT | 越界补 0 | constant(cval=0) |
| NEAREST | 边缘值 clamp | replicate（edge） |
| MIRROR | 不重复边缘 | reflect |
| WRAP | 循环平铺 | wrap |

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用 `aclnnGaussianFilter3DGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含算子计算流程的执行器，再调用 `aclnnGaussianFilter3D` 接口执行计算。

```Cpp
aclnnStatus aclnnGaussianFilter3DGetWorkspaceSize(
  const aclTensor *x,
  double sigma,
  const char *mode,
  const aclTensor *y,
  uint64_t *workspaceSize,
  aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnGaussianFilter3D(
  void *workspace,
  uint64_t workspaceSize,
  aclOpExecutor *executor,
  const aclrtStream stream)
```

## aclnnGaussianFilter3DGetWorkspaceSize

- **参数说明**

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) | 非连续tensor |
|:--|:--|:--|:--|:--|:--|:--|
| x | 输入 | 待滤波的三维体数据，NCDHW 布局，前导维合并为 batch。 | FLOAT32、FLOAT16、BFLOAT16 | ND | 支持 3/4/5 维 | √ |
| sigma | 输入 | 高斯核标准差，三轴共用（各向同性），取值范围 [0, 100]；为 0 时直通输出 y=x。 | double | - | 标量 | - |
| mode | 输入 | 边界填充模式字符串，取值 "REFLECT"、"CONSTANT"、"NEAREST"、"MIRROR"、"WRAP"。 | const char* | - | 字符串 | - |
| y | 输出 | 滤波输出，与 x 完全同 shape、同 dtype。 | FLOAT32、FLOAT16、BFLOAT16 | ND | 与 x 一致 | √ |
| workspaceSize | 输出 | 返回需要在 device 侧申请的 workspace 大小。 | - | - | - | - |
| executor | 输出 | 返回 op 执行器，包含算子计算流程。 | - | - | - | - |

- **返回值**：返回 `aclnnStatus`，成功时返回 `ACL_SUCCESS`（0），其它值为失败。

## 约束与限制

- 输入 `x` 支持 rank 3/4/5，即 `[D,H,W]`、`[N,D,H,W]`、`[N,C,D,H,W]`，统一按 NCDHW 处理，前导维（N、C）合并为 batch。
- 输入 `x` 支持 FLOAT32、FLOAT16、BFLOAT16，输出 `y` 与输入同 dtype；内部统一使用 fp32 累加，fp16/bf16 输入升 fp32 计算，输出经饱和 clamp 后 Cast 回输入 dtype。
- `sigma` 取值范围为 [0, 100]，核半径 `r = int(4*sigma + 0.5)` 不超过 400，单轴最多 801 个 tap。
- `mode` 仅支持 REFLECT、CONSTANT、NEAREST、MIRROR、WRAP 五种，三轴共用；CONSTANT 模式越界补 0。
- 输出 `y` 与输入 `x` 形状、数据类型必须完全一致。
- 总元素数 N·C·D·H·W 不超过 2^31-1。
- 非连续 Tensor 会在内部转为连续后再计算。

## aclnnGaussianFilter3D

| 参数名 | 输入/输出 | 描述 |
|:--|:--|:--|
| workspace | 输入 | 在 device 侧申请的 workspace 内存起址。 |
| workspaceSize | 输入 | 在 device 侧申请的 workspace 大小，由 `aclnnGaussianFilter3DGetWorkspaceSize` 获取。 |
| executor | 输入 | op 执行器，包含算子计算流程，由 `aclnnGaussianFilter3DGetWorkspaceSize` 获取。 |
| stream | 输入 | acl stream 流。 |

- **返回值**：返回 `aclnnStatus`，成功时返回 `ACL_SUCCESS`（0），其它值为失败。

## 支持的型号

- Ascend 910B
