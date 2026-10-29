# Ssim

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 910B 系列产品</term>   |    √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |    ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品（910B 系列）</term>    |     √    |

## 功能说明

- 算子功能：Ssim 算子（ACLNN 接口 aclnnSsim）计算两幅图像的结构相似度指标 MSSIM（Mean Structural Similarity），接收输入 x1、x2，输出 MSSIM 标量 yOut（shape [1]，float32）。
- 计算公式：
  - 接收输入 x1、x2，使用可分离的 11x11 高斯窗（sigma=1.5，零填充边界）分别估计局部均值 mu1、mu2，局部方差 sig1^2、sig2^2 与协方差 sig12，按公式逐像素合成相似度：

    $\mathrm{SSIM}(x_1,x_2)=\frac{(2\mu_1\mu_2+C_1)(2\sigma_{12}+C_2)}{(\mu_1^2+\mu_2^2+C_1)(\sigma_1^2+\sigma_2^2+C_2)}$，$yOut=\mathrm{mean}(\mathrm{SSIM})$

  - 其中 $C_1=(0.01L)^2$、$C_2=(0.03L)^2$，L 按 dtype 推断：uint8 取 $L=255$，float32 取 $L=1.0$；最后对裁剪边带 (winSize-1)/2 = 5 后的内部有效区域取均值。
- 输入支持 rank 2~8：rank2 为 [H, W] 灰度图，rank>=3 为 [..., H, W, C]，彩色图像按通道分别计算后取平均；H/W 要求不小于窗口边长 11；dtype 支持 uint8 与 float32。

## 参数说明

|参数名|输入/输出|描述|数据类型|数据格式|
|:--|:--|:--|:--|:--|
|x1|输入|参考图像。shape 为 [H, W]（rank2）或 [..., H, W, C]（rank>=3），H/W >= 11|UINT8、FLOAT32|ND|
|x2|输入|待比较图像，shape 与 dtype 需与 x1 一致|UINT8、FLOAT32|ND|
|yOut|输出|MSSIM 计算结果，shape 为 [1]|FLOAT32|ND|

## 约束说明

- 输入 rank 为 2 ~ 8；rank2 为 [H, W] 灰度图，rank>=3 为 [..., H, W, C]，彩色图像按通道分别计算后取平均。
- H/W 要求不小于窗口边长 11。
- float32 输出与 fp64 参考实现误差 <= 1e-5。

## 支持的型号

- Ascend 910B
