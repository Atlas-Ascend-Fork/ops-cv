# aclnnSsim

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 910B 系列产品</term>   |    √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |    ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品（910B 系列）</term>    |     √    |

## 功能说明

- 接口功能：Ssim 算子（ACLNN 接口 aclnnSsim）计算两幅图像的结构相似度指标 MSSIM（Mean Structural Similarity）：接收输入 x1、x2，使用可分离的 11x11 高斯窗（sigma=1.5）在图像内部滑窗（窗口不做任何边界填充，只覆盖完全落在输入内部的像素）分别估计局部均值 mu1、mu2，局部方差 sig1^2、sig2^2 与协方差 sig12，按公式 SSIM = (2*mu1*mu2 + C1)*(2*sig12 + C2) / ((mu1^2 + mu2^2 + C1)*(sig1^2 + sig2^2 + C2)) 逐像素合成相似度，其中 C1=(0.01*L)^2、C2=(0.03*L)^2，L 按 dtype 推断（uint8 取 255，float32 取 1.0）；输出只覆盖有效区域 (H-10)x(W-10)，对该区域内全部像素与全部通道取均值，得到 MSSIM 标量 yOut（shape [1]，float32）。输入支持 rank 2~8：rank2 为 [H, W] 灰度图，rank>=3 为 [..., H, W, C]，彩色图像按通道分别计算后取平均，H/W 要求不小于窗口边长 11；dtype 支持 uint8 与 float32。
- 计算公式：

  $\mathrm{SSIM}(x_1,x_2)=\frac{(2\mu_1\mu_2+C_1)(2\sigma_{12}+C_2)}{(\mu_1^2+\mu_2^2+C_1)(\sigma_1^2+\sigma_2^2+C_2)}$，$yOut=\mathrm{mean}(\mathrm{SSIM})$，其中 $C_1=(0.01L)^2$、$C_2=(0.03L)^2$，uint8 取 $L=255$，float32 取 $L=1.0$。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnSsimGetWorkspaceSize`接口获取入参并根据计算流程计算所需workspace大小，再调用`aclnnSsim`接口执行计算。

```Cpp
aclnnStatus aclnnSsimGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  aclTensor       *yOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSsim(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnSsimGetWorkspaceSize

|参数名|输入/输出|描述|数据类型|数据格式|形状(shape)|
|:--|:--|:--|:--|:--|:--|
|x1|输入|参考图像，输入 Tensor，H/W 不小于 11|UINT8、FLOAT32|ND|2-8|
|x2|输入|待比较图像，shape 与 dtype 需与 x1 一致|UINT8、FLOAT32|ND|2-8|
|yOut|输出|MSSIM 计算结果，shape 为 [1]|FLOAT32|ND|1|
|workspaceSize|输出|返回所需 device 侧 workspace 大小|-|-|-|
|executor|输出|返回 op 执行所需的 executor 指针|-|-|-|

- **返回值**：`aclnnStatus` 类型，返回 `ACL_SUCCESS`（0）表示成功；其他返回码表示失败。

## 约束说明

- 输入 rank 为 2 ~ 8；rank2 为 [H, W] 灰度图，rank>=3 为 [..., H, W, C]，彩色图像按通道分别计算后取平均。
- H/W 要求不小于窗口边长 11。
- 边界不做填充：只有完全落在输入内部的 11x11 窗口参与计算，因此输出对应 (H-10)x(W-10) 的有效区域。
- float32 输入按归一化到 [0, 1] 的图像使用（L=1.0）；当输入幅值远大于 1 时，mu^2、uxx 等中间量会超出 float32 表示范围并产生 NaN/Inf，调用方需先自行归一化。
- float32 输出与 fp64 参考实现误差 <= 1e-5。

## aclnnSsim

|参数名|输入/输出|描述|
|:--|:--|:--|
|workspace|输入|device 侧 workspace 内存地址|
|workspaceSize|输入|device 侧 workspace 内存大小，通过 `aclnnSsimGetWorkspaceSize` 获取|
|executor|输入|op 执行器，通过 `aclnnSsimGetWorkspaceSize` 获取|
|stream|输入|acl stream|

- **返回值**：`aclnnStatus` 类型，返回 `ACL_SUCCESS`（0）表示成功；其他返回码表示失败。

## 支持的型号

- Ascend 910B
