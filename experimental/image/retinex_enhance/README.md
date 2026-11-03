# RetinexEnhance

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

`RetinexEnhance` 对单张 HWC 图像执行固定参数的多尺度 Retinex 增强，支持灰度、RGB 和 RGBA。

## 参数与数据格式

| 参数名 | 输入/输出 | Shape | 布局/颜色顺序 | 数据类型 | 数据格式 | 数值范围 |
| --- | --- | --- | --- | --- | --- | --- |
| image | 输入 | `[H,W,C]`，`C∈{1,3,4}`，`H/W` 支持运行期动态变化 | HWC；灰度/RGB/RGBA；连续内存 | FLOAT32 | ND | 任意有限 FP32 值；计算前截断到 `[0,255]` |
| enhanced | 输出 | 与 `image` 相同的 `[H,W,C]` | HWC；灰度/RGB/RGBA；独立连续内存 | FLOAT32 | ND | `[0,255]` |

## 动态 Shape 支持

- 编译期支持 `[-1,-1,1]`、`[-1,-1,3]`、`[-1,-1,4]` 以及 `[-1,-1,-1]` 等 Rank 3 Unknown Shape，输出 Shape 原样继承输入。
- 运行期 `H` 和 `W` 可独立变化且必须大于 0，通道数必须为 1、3 或 4；Tiling 会按每次实际 Shape 重新计算条带、block 数和 workspace。
- 已覆盖 `1x1`、小于最大 Gaussian 半径、非 32B 对齐、240p、480p、720p 和 1080p 等多组 Shape。

## 约束说明

- `H` 和 `W` 必须大于 0，最后一维仅支持 1、3 或 4。
- `C=1` 执行单通道 Retinex；`C=3` 执行 RGB 色彩恢复；`C=4` 仅增强 RGB，Alpha 按输入范围规范化后原样保留。
- 仅支持单图 HWC 输入，不支持 Batch、CHW、BGR 或输入输出内存别名。
- NaN/Inf 输入的数值结果不作承诺。
- 需要一份 `H*W*C*sizeof(float)` 的算子用户 workspace，另加框架系统 workspace。
- Gaussian 尺度固定为 15、80、250，边界采用重复边缘的偶对称扩展。
- 总元素数 `H*W*C` 不得超过 `2^30`，以保证 FP32 Gather 的 uint32 字节偏移可表示。
- 动态 Shape 仍受平台 UB 容量限制：UB 需要容纳完整输入行、halo、纵向索引和至少一行累加缓冲。大宽度或大高度可能导致 Tiling 返回失败，Kernel 不会启动；不存在与平台、通道数无关的固定 H/W 上限。例如 UB 为 256 KiB 时，`[1,65536,4]` 无法容纳完整输入行。

## 固定输出映射

令 `x=clip(image,0,255)+1`，`MSR` 为三个尺度的 `ln(x)-ln(Gaussian(x))` 的平均值。
灰度输出为 `clip(30*(MSR+6),0,255)`；RGB 输出为 `clip(30*(CRF*MSR+6),0,255)`，
其中 `CRF_i=ln(125*x_i)-ln(x_R+x_G+x_B)`。RGBA 的 RGB 使用相同公式，Alpha 为 `clip(image_alpha,0,255)`。
增益 30 和偏置 6 是固定参数，不执行逐图最小最大值归一化。

## 调用说明

| 调用方式 | 接口声明 | 说明 |
| --- | --- | --- |
| ACLNN 调用 | [`docs/aclnnRetinexEnhance.md`](./docs/aclnnRetinexEnhance.md) | 两段式接口、参数、返回值和约束说明。 |
| ACLNN 示例 | [`examples/test_aclnn_retinex_enhance.cpp`](./examples/test_aclnn_retinex_enhance.cpp) | Tensor 创建、workspace 申请和算子执行示例。 |

## 单元测试

在已配置 CANN 和 CPU 仿真测试依赖的环境中执行：

```bash
bash build.sh --ophost -u --ops=retinex_enhance --experimental
bash build.sh --opkernel -u --ops=retinex_enhance --experimental --soc=ascend910b
```

Kernel UT 构建所用 Python 3 需要 NumPy。构建期间使用独立 NumPy float64 实现生成 golden 头文件，
仅写入构建目录；测试直接运行生产 Kernel 类，覆盖 C=1/3/4、重复边缘、小于半径的图像、
跨条带尾块、超范围有限输入、Alpha 保留和重复执行一致性。
比较门限为最大绝对误差不超过 0.5、P99 绝对误差不超过 0.1、平均绝对误差不超过 0.02。
