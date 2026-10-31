# IntegralImage

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |

算子注册 SoC 为 `ascend910b`，功能与性能验证平台为 Ascend910B4。

## 功能说明

- 算子功能：计算图像的积分图（Summed-Area Table，简称 SAT），使任意矩形区域内的像素和可以在 O(1) 时间内完成。可作为 Haar 特征提取、自适应阈值、分水岭分割、目标边界分析和图像形态学处理的前置算子。

- 常用 ACLNN 输出组合：
  - 形态 1：输出普通积分图 `sat`；
  - 形态 2：输出 `sat` 和平方和积分图 `sqsum`；
  - 形态 3：输出 `sat`、`sqsum` 和倾斜区域累加结果 `tilted`。

  ACLNN 接口的 `sqsum`、`tilted` 为可选输出；传 `nullptr` 时跳过对应分支计算。

- 计算定义：输入 `image` 的空间尺寸为 $H \times W$，各输出的空间尺寸为 $(H+1) \times (W+1)$。HWC 输入按通道独立计算。

普通积分图及平方和积分图采用左闭右开的前缀区域，第 0 行、第 0 列为零：

$$sat(y,x) = \sum_{i=0}^{y-1}\sum_{j=0}^{x-1} image(i,j)$$

$$sqsum(y,x) = \sum_{i=0}^{y-1}\sum_{j=0}^{x-1} image(i,j)^2$$

`tilted` 采用以 $(y,x)$ 为顶点、向上按 45° 扩展的三角区域累加，坐标范围为 $0 \le y < H$、$0 \le x < W$：

$$tilted(y,x) = \sum_{i=0}^{y}\ \sum_{j=\max(0,\ x-(y-i))}^{\min(W-1,\ x+(y-i))} image(i,j)$$

其边界定义为 $tilted(H,x)=0$、$tilted(y,W)=0$；第 0 行为输入第 0 行。

例如，输入为 `[[1,2],[3,4]]` 时：

```text
sat    = [[0,0,0], [0,1,3], [0,4,10]]
sqsum  = [[0,0,0], [0,1,5], [0,10,30]]
tilted = [[1,2,0], [6,7,0], [0,0,0]]
```

## 参数说明

### 输入

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| image | 输入 | 待计算积分图的图像，shape 为 $(H,W)$ 或 $(H,W,C)$ | UINT8、FLOAT16、FLOAT32 | ND |

### 输出

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| sat | 输出（必选） | 积分图，shape 为 $(H+1,W+1)$ 或 $(H+1,W+1,C)$ | INT32（UINT8 输入且 sdepth 取 -1/0）、FLOAT32（FLOAT16/FLOAT32 输入，或 sdepth=1） | ND |
| sqsum | 输出（可选） | 平方和积分图，shape 同 `sat` | FLOAT32 | ND |
| tilted | 输出（可选） | 倾斜三角区域累加，坐标定义见功能说明，shape 同 `sat` | 同 `sat` | ND |

### 属性

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 取值范围 |
| :--- | :--- | :--- | :--- | :--- |
| sdepth | 属性 | 输出数据类型选择，可选，默认 -1 | INT64 | -1（自动）、0（INT32，仅 UINT8 输入）、1（FLOAT32） |

`sdepth=-1` 时按输入类型自动推导：UINT8 → INT32，FLOAT16/FLOAT32 → FLOAT32。

## 约束说明

- 形状：输入 `image` 支持 $(H,W)$ 与 $(H,W,C)$ 两种排布；输出 `sat`、`sqsum`、`tilted` 对 H、W 两维各加 1，通道维保持不变，即 $(H+1,W+1)$ 或 $(H+1,W+1,C)$。
- 输出申请：调用方预先申请 `sat` 和已启用可选输出；每个输出的 Rank、H、W、C 及数据类型必须符合接口定义，`GetWorkspaceSize` 阶段检查这些参数。
- 输入维度：H、W、C 必须为正，H+1、W+1 必须能用 INT64 表示。
- 通道布局：3D 输入按 HWC 组织，各通道独立计算；通道数 $C$ 不要求 32 对齐。
- 非对齐：$H$、$W$ 无需按 32 元素对齐，Kernel 通过 `DataCopyPad` 处理尾块，支持 $H=1$、$W=1$、单行、单列等退化尺寸。
- 数据类型：输入支持 UINT8、FLOAT16、FLOAT32；`sqsum` 输出固定为 FLOAT32。
- 整数累加：INT32 结果采用 32 位整数回绕语义。对 UINT8 输入，$255 \times H \times W \le 2^{31}-1$ 是保证所有可能输入均不溢出的充分条件；需要更大数值范围时可使用 `sdepth=1`。
- 浮点累加：FLOAT16 输入在 Kernel 内以 FLOAT32 累加；FLOAT32 的舍入精度受累加量级与累加顺序影响。
- 分块配置：块宽根据 UB 容量、输入类型和通道数确定；legacy 路径要求列块总数不超过可用 AIV 核数。

## 计算流程

| 条件 | 路径 | 计算步骤 |
| :--- | :--- | :--- |
| 单通道且 W≥128 | two-phase | 多核按行生成前缀并写 workspace；同步后按列块纵向累加，生成 `sat` 和已启用的 `sqsum`。 |
| 单通道 W<128，或多通道 HWC | legacy | 按列分块，各通道独立扫描；补上当前块左侧的行前缀，再进行纵向累加。 |
| 提供 `tilted` | 对角前缀分解 | 计算或复用行前缀，分别扫描两组对角前缀并合并；每通道执行三次 `SyncAll`。 |

`tilted` 按通道复用 workspace；未请求的可选输出跳过相应计算分支。workspace 大小以本次 `GetWorkspaceSize` 返回值为准。

## 调用说明

`examples/test_aclnn_integral_image.cpp` 演示 ACLNN 两段式调用（`aclnnIntegralImageGetWorkspaceSize` 获取 Workspace 与执行器，再调用 `aclnnIntegralImage` 执行），覆盖三种输出形态、UINT8/FLOAT16/FLOAT32 输入、非对齐尺寸与 3D 多通道场景。

| 运行方式 | 用途 |
| :--- | :--- |
| `./test_aclnn_integral_image` | 完整功能验证，包含参数校验反例。 |
| `./test_aclnn_integral_image u8_f32` | UINT8 输入、FLOAT32 输出专项验证。 |
| `./test_aclnn_integral_image validation` | 输出 Shape、类型与 `sdepth` 参数校验；只调用 `GetWorkspaceSize`。 |
| `./test_aclnn_integral_image verify` | 六组大尺寸 `sat` 逐元素验证。 |
| `./test_aclnn_integral_image bench 20` | 预热三次、统计二十次，分别报告 Host 墙钟和设备事件计时的中位数。 |

## 验证与参考资料

### UINT8 输入、FLOAT32 输出验证

在 ACLNN 示例可执行文件所在目录运行：

```bash
./test_aclnn_integral_image u8_f32
```

该入口使用 `sdepth=1`，覆盖窄图、two-phase 非对齐尾块、单像素、单列及 HWC 输入；
验证三种输出组合、可选输出传 `nullptr`，并逐元素比较 CPU 参考结果。
默认功能测试入口也包含这组用例。

### 接口与测试资料

| 文档 | 说明 |
| :--- | :--- |
| [`docs/aclnnIntegralImage.md`](./docs/aclnnIntegralImage.md) | ACLNN 接口原型、参数说明、返回值与约束说明。 |
| [`docs/host_ut_validation_20261028.md`](./docs/host_ut_validation_20261028.md) | Host UT 覆盖范围、执行结果与结果摘录。 |
| [`docs/u8_f32_validation_20261029.md`](./docs/u8_f32_validation_20261029.md) | UINT8→FLOAT32 类型注册、ACLNN 真机及单元测试结果。 |
