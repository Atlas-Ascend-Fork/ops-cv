# GammaCorrection

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

`GammaCorrection` 对图像张量执行逐元素幂律亮度校正，按 `correctionMode` 选择三种流水线：

- **Mode 0（纯 pow，默认）**：`y[i] = pow(x[i], gamma)`，与 `torch.pow` 等价，
  负底数非整数指数按 pow 语义产生 NaN，0/Inf/NaN 自然传播。
- **Mode 1（归一化 gamma）**：
  `t = clamp(x, 0, M); y = clamp(pow(t/M, gamma) * M, 0, M)`，pow 内部的 NaN/Inf
  清零。先把输入钳到 `[0, M]` 保证 `Log` 定义域，输出再钳回，避免越界。
- **Mode 2（sRGB 编码曲线）**：
  `t = clamp(x, 0, M)/M; u = (t<=0.0031308) ? 12.92·t : 1.055·t^(1/2.4) - 0.055; y = clamp(u·M, 0, M)`。
  此模式 gamma 属性被忽略。

`M` 为值域上界：`maxValue` 给定（>0）时取该值，否则按 dtype 推导
（浮点 1.0；uint8/int8 255.0；uint16/int16 65535.0）。

通用路径在一个 AI Core kernel 内融合
`Cast(FP32) -> [归一化] -> Log/Muls/Exp 或 sRGB 分段 -> [NaN 清零] -> [反归一化] -> Clamp -> Cast(output dtype)`，
中间结果仅驻留 UB，不生成中间 GM Tensor。Mode 0 的 `gamma=1.0/2.0/0.5` 分别走透传、平方和
平方根快速路径；其他可表示为 `uint32_t` 的正整数使用整数幂路径，以保留负底数的 `pow` 语义。

## 参数说明

| 参数名 | 输入/输出/属性 | shape | 数据类型 | 格式 |
|---|---|---|---|---|
| x | 输入 | Rank 2/3/4 | FLOAT16、BFLOAT16、FLOAT32、UINT8、INT8、UINT16、INT16 | ND |
| gamma | 必选属性 | 正有限标量 | FLOAT | - |
| correctionMode | 可选属性 | 0/1/2（默认 0） | INT | - |
| maxValue | 可选属性 | 值域上界，<=0 时按 dtype 推导 | FLOAT | - |
| y | 输出 | 与 x 相同 | 与 x 相同 | ND |

## 约束说明

- 输入 rank 必须为 2、3 或 4，各维度必须为正数。
- 按最后一维向前计数，维度上限依次为 `10000, 10000, 1000, 200`。
- `gamma` 必须为有限正数；`correctionMode` 必须为 0/1/2；`maxValue` 给定时必须 > 0。
- 输入和输出 dtype 必须一致；BF16 仅在具备原生支持的平台启用。
- Mode 0：`x < 0` 且 `gamma` 非整数时输出 NaN，整数 gamma 保留正负号幂语义。
- 整数 dtype（所有模式）：先把 NaN 置 0、再饱和到 dtype 值域，最后按 RINT 取整写回，
  窄化转换不依赖硬件隐式饱和；UINT16 覆盖完整 `[0, 65535]` 值域。
- Mode 1/2：负底数与 NaN/Inf 在图像语义下被钳为 0。
- 当前注册 `ascend910b`。

## 工程结构
 ops 仓库中的单算子模块：

```text
├── CMakeLists.txt
├── docs
├── examples                # aclnn two-stage call sample
├── op_graph
├── op_host                 # op def / infershape / tiling
│   └── config/ascend910b/gamma_correction_binary.json
├── op_kernel
└── tests/ut
```

本算子不手写 `op_api`：aclnn 两段式接口 `aclnnGammaCorrectionGetWorkspaceSize` /
`aclnnGammaCorrection` 由 `op_host/gamma_correction_def.cpp` 的算子定义在构建时自动生成
（`op_host/CMakeLists.txt` 中 `ACLNNTYPE aclnn`），头文件 `aclnn_gamma_correction.h` 随算子包
安装到 `opp/vendors/<vendor>/op_api/include/`。

接口调用见 [`docs/aclnnGammaCorrection.md`](docs/aclnnGammaCorrection.md)，实现与
Tiling 细节见 [`docs/design.md`](docs/design.md)。
