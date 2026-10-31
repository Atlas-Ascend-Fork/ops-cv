# AdaptiveThreshold

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：对灰度图像进行自适应阈值二值化。
- 与固定阈值不同，每个像素的阈值由其局部邻域计算得到，从而在光照不均匀的场景下仍能较好地分离前景与背景。
- 邻域阈值支持两种计算方法：
  - `MEAN`：取 `block_size × block_size` 邻域内像素的算术平均值。
  - `GAUSSIAN`：取邻域内像素的高斯加权平均值。
- 像素阈值 `T(x, y) = 邻域均值（或高斯加权均值） - C`。
- 二值化规则：

  ```text
  THRESH_BINARY:     dst(x, y) = src(x, y) > T(x, y) ? max_value : 0
  THRESH_BINARY_INV: dst(x, y) = src(x, y) > T(x, y) ? 0 : max_value
  ```

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------ | -------------- | ---- | -------- | -------- |
| x | 输入 | 输入的灰度图像，支持 2D/3D/4D（4D 时通道维度必须为 1）。 | UINT8 / FLOAT16 / FLOAT32 | ND |
| max_value | 属性 | 二值化后非零像素的取值。 | FLOAT | - |
| adaptive_method | 属性 | 邻域阈值的计算方法，取值 `MEAN` 或 `GAUSSIAN`。 | STRING | - |
| threshold_type | 属性 | 二值化类型，取值 `THRESH_BINARY` 或 `THRESH_BINARY_INV`。 | STRING | - |
| block_size | 属性 | 邻域窗口大小，取值 3~255 的奇数。 | INT | - |
| C | 属性 | 从邻域均值（或加权均值）中减去的常数。 | FLOAT | - |
| y | 输出 | 二值化结果，形状与输入一致，数据类型与输入一致。 | UINT8 / FLOAT16 / FLOAT32 | ND |

## 约束说明

- `x` 与 `y` 的形状保持一致，数据类型保持一致。
- `x` 支持 `UINT8`、`FLOAT16`、`FLOAT32` 三种数据类型，且为单通道灰度图。
- `x` 支持 `[H, W]`、`[N, H, W]`、`[N, 1, H, W]` 三种形状，其中 `N` 为图像数量、`H` 为高、`W` 为宽。
- `adaptive_method` 取值必须为 `MEAN` 或 `GAUSSIAN`。
- `threshold_type` 取值必须为 `THRESH_BINARY` 或 `THRESH_BINARY_INV`。
- `block_size` 必须为奇数，且满足 `3 <= block_size <= 255`。
- 当前仅支持 `ND` 格式。
- 非空 `FLOAT32` 输入暂不支持 `block_size=255`；该组合的空图可正常返回。
- `max_value` 与 `C` 的取值语义与输入数据类型相关：
  - `UINT8` 输入：`max_value` 会饱和取整到 `[0, 255]`；`C` 按 `threshold_type` 取整（`THRESH_BINARY` 向上取整、`THRESH_BINARY_INV` 向下取整）。
  - `FLOAT16` / `FLOAT32` 输入：`max_value` 与 `C` 保留原值（不取整），邻域均值按浮点计算。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| ACLNN 调用 | [test_aclnn_adaptive_threshold.cpp](./examples/test_aclnn_adaptive_threshold.cpp) | UINT8 灰度图像的 Tensor 创建、两阶段调用、结果获取及资源释放示例。 |
