# BilateralFilter3D

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

`BilateralFilter3D` 对三维体数据执行双边滤波，语义**参考** MONAI 的双边滤波实现，
但边界处理不同（本算子为有效邻域截断 + 重归一化，MONAI 为逐维 replicate 寻址 + 全窗口分母），
窗口来源与通道语义亦有差异，详见 `docs/design.md` 第 1.1 节。
输入的最后三维视为 D/H/W，其余前导维按独立体数据批量处理；输出 shape、dtype 与 ND 布局
均与输入一致。

双边滤波是**保边平滑**：权重同时由空间距离和值域差决定，因此在抑制噪声的同时不会像高斯模糊那样
糊掉边缘。每个体素在 `kernel_size^3` 邻域内做加权平均：

```text
r = kernel_size / 2
y[z,h,w] = Σ(v * weight) / Σ(weight),  (dz,dy,dx) ∈ [-r,r]^3 且邻域点在体内
weight   = exp(-( (dz²+dy²+dx²) / (2·sigma_space²) + (v - c)² / (2·sigma_color²) ))
```

其中 `c` 为中心体素值、`v` 为邻域体素值。`sigma_space` 越大空间平滑范围越广，`sigma_color`
越大则对值域差越不敏感（趋近于纯空间加权平均）。

边界采用**有效邻域截断 + 重归一化**：越界邻域既不计入分子也不计入分母，因此边界体素不会被零值
拉偏。中心体素自身的空间偏移与值域差均为 0、权重恒为 1，故分母恒 ≥ 1，不存在除零。

实现为纯向量（AIV）kernel，所有累加在 FP32 UB 内完成，FP16 输入搬入后 Cast 到 FP32 计算、
输出时以 `CAST_RINT` 写回。不申请用户 workspace，不生成中间 GM Tensor。

## 参数说明

| 参数名 | 输入/输出/属性 | shape | 数据类型 | 格式 |
|---|---|---|---|---|
| x | 输入 | Rank 3/4/5，最后三维为 D/H/W | FLOAT16、FLOAT32 | ND |
| kernel_size | 必选属性 | 奇数，取值 3/5/7/9 | INT | - |
| sigma_space | 必选属性 | 有限正数 | FLOAT | - |
| sigma_color | 必选属性 | 有限正数 | FLOAT | - |
| y | 输出 | 与 x 相同 | 与 x 相同 | ND |

## 约束说明

- 输入 rank 必须为 3、4 或 5，各维度必须为正数，不支持空张量。
- 维度上限：`D ≤ 1000`，`H ≤ 4096`，`W ≤ 4096`，前导维乘积（N·C）`≤ 12800`。
- `kernel_size` 必须为 3/5/7/9 之一；`sigma_space` 与 `sigma_color` 必须为有限正数。
- 输入与输出的 shape、dtype 必须一致。
- `sigma` 极小时 `0.5/sigma²` 在 FP32 下溢出，Host 侧将该倒数饱和到 `1e30`。此时非零距离或
  非零色差的权重 `exp(-x)` 恒为 0，输出收敛到中心体素值，而不会产生 `0*inf` 的 NaN。
- InferShape 对未知维度（负数占位符）透传以支持图模式，运行时 Tiling 只接受已具体化的正维度。
- 当前注册 `ascend910b`。

## 性能

Ascend 910B 实测（FP32，20 次迭代取中位数）：

| 规模 | kernel_size | median |
|---|---|---|
| 64×64×64 | 5 | 2.254 ms |
| 128×128×128 | 5 | 9.029 ms |
| 128×128×128 | 3 | 2.123 ms |

本算子为**指令发射受限**而非访存受限，耗时正比于 `kernel_size³`。实现要点与仍存的优化空间见
[`docs/design.md`](docs/design.md) 第 7 节。

## 工程结构

ops 仓库中的单算子模块：

```text
├── CMakeLists.txt
├── docs                    # 接口文档与设计说明
├── examples                # aclnn 两段式调用样例, 含 --bench/--diag 模式
├── op_graph
├── op_host
│   └── config/ascend910b/bilateral_filter3_d_binary.json
├── op_kernel
└── tests/ut                # infershape / tiling / kernel golden UT
```

aclnn 接口由框架按 `ACLNNTYPE aclnn` 自动生成，无需手写 `op_api`。

接口调用见 [`docs/aclnnBilateralFilter3D.md`](docs/aclnnBilateralFilter3D.md)，实现、Tiling
策略与 910B 向量对齐约束见 [`docs/design.md`](docs/design.md)。
