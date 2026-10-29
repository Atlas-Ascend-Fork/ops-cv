# HOGDescriptor

在 Ascend 910B 上计算图像的方向梯度直方图描述子。支持 float16、float32 输入，输出始终为 float32。

## 输入与输出

输入 `x` 使用 ND 格式，支持 `[H,W]`、`[H,W,C]`、`[N,H,W]` 和 `[N,H,W,C]`，C 为 1 或 3。
三维输入的最后一维为 1 或 3 时，按 HWC 解释；其余三维输入按 NHW 解释。
例如，批量灰度图的宽度为 1 或 3 时，需要用四维 NHWC 表达以消除歧义。

N 的范围为 1..200，H/W 的范围为 1..10000。当前实现额外要求填充后的宽度不超过 1024，且至少一个块行的直方图及全部工作缓冲能放入设备 UB；不满足时返回失败。
不支持空张量、未知维度及 uint8 输入。gamma 开启时，图像像素应为非负有限值。

设 cell 大小为 `(pH,pW)`，block 大小为 `(bH,bW)`，以 cell 为单位的步长为 `(sH,sW)`：

```text
cellsH = ceil(H / pH)
cellsW = ceil(W / pW)
blocksH = floor((cellsH - bH) / sH) + 1
blocksW = floor((cellsW - bW) / sW) + 1
D = orientations * bH * bW * blocksH * blocksW
```

cell 网格必须容纳完整 block。单图描述子最多为 1 GiB。
N=1 时输出 `[D]`，N>1 时输出 `[N,D]`；显式 N=1 的四维输入也输出一维。

## 属性

以下属性均为必填；“样例值”用于说明推荐调用参数，不表示接口有可省略的默认参数。

| 属性 | 类型 | 范围 / 含义 | 样例值 |
|---|---|---|---|
| pixels_per_cell | ListInt | 两个整数，每项 1..256 | [8,8] |
| cells_per_block | ListInt | 两个整数，每项 1..8 | [2,2] |
| block_stride | ListInt | 两个整数，每项 1..8，以 cell 为单位 | [1,1] |
| orientations | Int | 方向 bin 数，2..36 | 9 |
| block_norm | String | L1、L1-sqrt、L2、L2-Hys | L2-Hys |
| l2_hys_threshold | Float | 0..1，仅用于 L2-Hys 的截断步骤 | 0.2 |
| gamma_correction | Bool | 是否先对像素取平方根 | true |
| signed_gradient | Bool | false 使用 [0,π)，true 使用 [0,2π) | false |
| derivative_aperture | Int | 仅支持 1，即中心差分 [-1,0,1] | 1 |
| eps | Float | 正有限数，归一化保护项 | 1e-6 |

## 计算语义

1. 向右、向下复制边缘像素，补齐整个 cell 网格；可选对像素取平方根。
2. 使用中心差分计算梯度，在填充后图像的边缘复制取值。RGB 图像逐像素选取梯度幅值最大的通道，相等时取首个通道。
3. 方向坐标为 `theta * orientations / range - 0.5`，在相邻两个 bin 间插值，bin 索引循环回绕。
4. 以 cell 中心为格点，在相邻四个 cell 间做空间插值；超出图像边缘的投票折回边缘 cell。
5. 每个 block 做独立归一化，按 block 行列、cell 行列、bin 递增顺序输出。

L1 使用 `v / (sum(abs(v)) + eps)`，L1-sqrt 在此基础上取平方根。
L2 使用 `v / sqrt(sum(v*v) + eps*eps)`。
L2-Hys 先做 L2，按阈值截断，再按相同公式做一次 L2。
本算子采用上述确定语义；不声明与 OpenCV 或 skimage 的所有配置完全等价。

当 `signed_gradient=true` 时，方向计算将 `abs(gx)<1e-30` 的 gx 明确置零，再执行原有分母保护及象限修正。
因此用于 gx 的零值/负值指示量只取 0 或 1；gy 的正值指示量由精确比较产生，避免非正规数导致部分饱和。
幅值在该处理前计算，保留原始梯度的投票权重；unsigned 路径沿用原有计算。
该近零处理是一项数值边界约定：当 gy 有有效幅值时方向接近竖直；若两个分量都极小，原有 FP32 平方幅值会下溢为零。

## 构建与验证

先加载当前 CANN 环境，在完整 ops-cv 仓库根目录执行：

```bash
export CCACHE_DISABLE=1
bash build.sh --pkg --experimental --soc=ascend910b \
    --ops=hog_descriptor --vendor_name=hogd910 -j16
```

安装本次生成的 `.run` 包，并使 `ASCEND_CUSTOM_OPP_PATH` 指向该安装中的 `vendors/hogd910_cv` 目录；
将其 `op_api/lib` 加入 `LD_LIBRARY_PATH` 后，重新编译并运行样例：

```bash
bash build.sh --run_example hog_descriptor eager cust \
    --vendor_name=hogd910 --experimental --soc=ascend910b
bash build.sh --ophost_test --experimental --ops=hog_descriptor --soc=ascend910b -j16
```

样例保留原 23 个场景，并增加 gamma 关闭的 FP32 signed 极小 gx、极小 gy 两个定向场景，共 25 个正确性场景。CPU golden 保持双精度 atan2，不采用上述清零规则作为参考答案。
Host UT 覆盖 shape、属性非法值、dtype 路由与 UB 预算边界。

不依赖 CANN 的数值回归可以单独运行：

```bash
g++ -std=c++17 -O2 experimental/image/hog_descriptor/tests/test_hog_descriptor_helpers.cpp \
    -o /tmp/test_hog_descriptor_helpers
/tmp/test_hog_descriptor_helpers
```

样例支持 `--bench H W C dtype iters`，dtype=1 表示 float16，dtype=2 表示 float32。
`--bench-signed H W C dtype iters` 使用相同计时方式测试 signed 路径。
预热 3 次，报告多次 stream event 区间的中位数。
区间包围完整 ACLNN 调用，含 workspace 准备和同步造成的影响，不作为纯 Kernel 时延。
`logical_io` 仅为输入加输出的字节数，不能代表 Kernel 实际搬运量或硬件带宽。
