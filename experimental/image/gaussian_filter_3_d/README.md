# aclnnGaussianFilter3D

## 算子简介

对 5D 体数据 `x[N, C, D, H, W]` 沿 D/H/W 三个空间轴执行三维高斯滤波（可分离实现），输出与输入同 shape 同 dtype：

```text
y = G_{sigma} * x
```

语义对齐 `scipy.ndimage.gaussian_filter`（truncate 固定 4.0，不开放为 API）：

- 核半径：`r = int(4 * sigma + 0.5)`（round-half-up，非 ceil），核大小 `2r+1`；
- 核权重：`w[i] = exp(-0.5*i^2/sigma^2) / sum(exp(...))`，i ∈ [-r, r]，host 侧 fp64 计算后转 fp32，三轴共用；
- `sigma = 0` 时 r=0，恒等直通（y=x）；
- 边界模式 `mode` 五种，三轴共用，语义与 scipy.ndimage 一一对应：

| mode | 语义 | 等效 np.pad |
|---|---|---|
| REFLECT（默认） | 重复边缘像素 | symmetric |
| CONSTANT | 越界补 0 | constant(cval=0) |
| NEAREST | clamp | replicate（edge） |
| MIRROR | 不重复边缘 | reflect |
| WRAP | 循环 | wrap |

- 累加精度：一律 fp32 累加；fp16/bf16 输入升 fp32 计算，输出饱和转换（clamp 后 Cast）回输入 dtype；
- rank 3/4/5 输入（[D,H,W] / [N,D,H,W] / [N,C,D,H,W]）统一按 NCDHW 处理，前导维合并为 batch。

## 目录结构

```text
experimental/image/gaussian_filter_3_d/
├── op_host/
│   ├── gaussian_filter_3_d_def.cpp          # 算子原型注册
│   ├── gaussian_filter_3_d_infershape.cpp   # shape/dtype推导
│   ├── gaussian_filter_3_d_tiling.cpp       # tiling：权重生成(fp64)、tile自适应、多核切分、workspace
│   ├── gaussian_filter_3_d_tiling.h         # TilingData 定义
│   ├── config/ascend910b/gaussian_filter_3_d_binary.json
│   └── op_api/                             # aclnnGaussianFilter3D 二段式接口 + l0op
├── op_kernel/
│   └── gaussian_filter_3_d.cpp              # AscendC kernel（Ascend 910B AIV）
└── tests/ut/                               # op_host infershape / op_api / op_kernel 三件套
```

## 实现要点

- 单次 kernel launch 内顺序执行三个 pass（W→H→D），pass 之间用 `SyncAll` 跨核同步；
  pass W/H 的 fp32 中间结果写在 workspace 的 ping/pong 两块缓冲中，pass D 直接写输出；
- 无论沿哪个轴滤波，乘加永远沿 W（地址连续）方向向量化（Axpy），与 DALI
  "W×C 融合"思想一致；H/D pass 的 tap 表现为整行/整面偏移，仍是连续段运算；
- 每输出行按 `tileLen`（按 UB 容量自适应、8 元素对齐）分 chunk，`task = line * chunk`
  线性编号均匀摊到所有 AIV；
- pass W 的 tap 沿行内偏移：对齐中段（8 元素对齐）向量化，行边缘（含越界折叠映射、
  CONSTANT 补 0）走标量路径；fp16/bf16 路径以标量补充对齐边界；
- 权重（最多 2*400+1=801 个 fp32）经 TilingData 下发，kernel 内不做 exp 计算。

## 支持的型号

- Atlas A2系列产品

## 约束说明

- x：rank 3/4/5，FLOAT32 / FLOAT16 / BFLOAT16，ND 连续；
- y：与 x 完全同 shape 同 dtype；
- sigma ∈ [0, 100]，核半径 r = int(4σ+0.5) ≤ 400（最多 801 个 tap）；
- mode ∈ {REFLECT, CONSTANT, NEAREST, MIRROR, WRAP}；
- 总元素数 N·C·D·H·W ≤ 2^31-1；
- r 可以大于对应维度长度（折叠映射天然支持，CONSTANT 直接补 0）。

## 测试

- op_host infershape UT：`tests/ut/op_host/test_GaussianFilter3D_infershape.cpp`
- op_api UT：`tests/ut/op_host/op_api/test_aclnn_gaussian_filter_3_d.cpp`
- op_kernel UT：`tests/ut/op_kernel/test_gaussian_filter_3_d.cpp`（与 CPU fp64 可分离高斯参考比对，
  覆盖 5 种 mode、r>dim、sigma=0、fp16/bf16）
