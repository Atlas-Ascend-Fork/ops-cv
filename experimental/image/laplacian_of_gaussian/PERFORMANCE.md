# LaplacianOfGaussian 性能实测（提交版本）

更新日期：2026-11-08。本文数据对应提交版本（分支 `laplacianofgaussian`，提交 `b1fc3824` 之后的重测），替换此前基于 2026-10-28 早期实现的旧记录。

结论：34 例对比 `torch_npu` 组合基线（full2d / separable），几何平均 **1.33x**；大图优势明显（4K 4.29x、1080p 3.26~3.32x、512² 1.53~1.65x），128×129 及更小尺寸受 ACLNN 启动开销主导，与基线持平或略慢（0.76~1.03x）。

## 环境

| 项 | 值 |
| --- | --- |
| 设备 | Ascend 910B4（physical 4，运行时 physical=4/user=0/logic=4/back=0） |
| 容器 / CANN | `CANNBot_lxlf` / CANN 9.1.0 |
| 其他 | Python 3.11、NumPy 1.26.4、SciPy 1.13.1；基线实现由 `torch_npu` 提供 |

采集前确认目标卡健康且无其他进程；未干预其他卡。

## 方法

- **计时口径**：设备事件区间（`aclrtSynchronizeStream` 前后），包含完整公开调用；预热 2 次、测量 5 次，每侧取 5 次运行的平均值。同一进程内先测被测实现、再测基线，两侧串行。
- **基线**：`torch_npu` 的两种 LoG 实现——直接二维 `conv2d`（full2d）与「两次一维卷积」可分离实现（separable）。逐例保留精度合格且均值更小的那一种，选择结果记入 CSV 的 `baseline_impl` 列。
- **精度**：每个用例先做逐像素比对（`abs(err) <= 2^-16 + 2^-10*abs(golden)`、`max_abs <= 0.01`、RMSE 界），基线与被测实现均需通过，未通过的实现不参与比较。
- **输入**：固定随机种子，sigma=1、显式 radius=4、truncate=4；边界按用例表给定；非法/极端属性不在本表范围。

逐例原始数据见 `tests/results/`（运行后生成）或 `support_docs_LaplacianOfGaussian/performance/性能对比_demo8_20261106.csv`。

## 结果

| 规模 | 用例数 | 相对基线加速比 |
| --- | ---: | --- |
| 4K（2160×3840，FP32） | 1 | 4.29x |
| 1080p（1080×1920，三 dtype） | 3 | 3.26 ~ 3.32x |
| 512²（三 dtype） | 3 | 1.53 ~ 1.65x |
| 512² sigma 扫描（0.5 / 1 / 3 / 8） | 9 | 1.51 ~ 1.64x |
| 128×129 边界扫描（四种边界 × 三 dtype） | 9 | 1.03 ~ 1.27x |
| 128×129 | 3 | 0.83 ~ 1.03x |
| 64×64 | 3 | 1.25 ~ 1.48x |
| 256² | 3 | 0.86 ~ 0.98x |
| 17×31 | 3 | 0.76 ~ 0.89x |

几何平均 1.33x。

**读法**：4K/1080p 上被测实现把逐 tap 搬运改为 GM 侧 `DataCopyPad` 直读（见 PR 描述 Kernel 侧设计），优势随图像尺寸增长；128×129 及更小尺寸的单次事件区间约 0.22~0.30 ms，其中 ACLNN 启动与 kernel Init 占主要部分，耗时与像素数几乎无关，因此与基线基本持平或略慢；256² 恰在基线由「开销主导」转向「算力主导」的交叉点。比值不代表单例显著性，不外推到其他 dtype/尺寸。

## 版本演进

6 例代表集（U/H/F × small/medium）的设备事件平均耗时：

| 版本 | 平均 ms | 相对初版 | 1080p（ms） | 关键改动 |
| --- | ---: | ---: | ---: | --- |
| 初版 | 0.6867 | 1.00x | 24.4 | 逐 tap 标量搬运 |
| demo1 | 0.5136 | 1.34x | — | 行缓存分块 |
| demo2 | 0.5016 | 1.37x | — | 纵向成对折叠（利用 g/d 偶对称） |
| demo3 | 0.4406 | 1.56x | 24.4 | 横向取消补偿累加 |
| demo4 | 0.4341 | 1.58x | 23.9 | 纵向取消补偿累加 |
| demo5 | 0.2605 | 2.65x | 1.27 | GM 侧移位：内部列段逐 tap 从 GM 直读，规避 UB 32B 对齐约束 |
| demo6 | 0.2591 | 2.82x | 0.60 | 环形缓存 4 组 → 2 组、UB 余量 8KB → 16KB |
| demo7 | 0.2441 | 3.01x | 0.50 | 向量指令间改用 `PipeBarrier<PIPE_V>` |
| **demo8（提交版本）** | **0.2544** | **2.81x** | **0.50** | 恢复纵向 TwoSum 补偿，修复稳定性用例 |

demo4 / demo7 更快但未通过稳定性用例（128×8193 翻转输入的单像素抵消超差），仅作过程记录。

## 已否决的优化方向

| 尝试 | 结果 |
| --- | --- |
| UB 内移位（`Muls` 读偏移视图 / `Gather` / `DataCopy` 偏移） | 设备报 UB 地址未对齐；`Gather` 在 count>64 时结果错误 |
| halo 落 GM workspace 再按 tap 回读 | 输出非法（workspace 被框架占用） |
| Kahan 求和替代 TwoSum（13 → 5 条向量指令） | 稳定性 168/168 通过，但 `L-uint8-512x512-s1` 精度失败 |
| 仅对 float32 补偿（按 dtype 条件编译） | float16 的 flip 用例随即失败 |
| 强制拆分列宽换取更大行分块（tile 上限 128/64） | 256²、512² 均无改善，per-row 固定开销反而增加 |

## 复现

```bash
# 1. 构建并产出安装包
bash build.sh --pkg --experimental --soc=ascend910b --ops=laplacian_of_gaussian \
  --vendor_name=custom --cann_3rd_lib_path=$PWD/third_party -j8

# 2. 功能与数值验证（smoke / full / extended / large / stability）
source $CANN_ENV
python3 experimental/image/laplacian_of_gaussian/tests/launch.py \
  --suite full --package build_out/cann-ops-cv-custom-linux.aarch64.run

# 3. 性能对比（含 torch_npu 基线）
python3 experimental/image/laplacian_of_gaussian/tests/launch.py --mode benchmark \
  --suite perf \
  --package build_out/cann-ops-cv-custom-linux.aarch64.run \
  --baseline experimental/image/laplacian_of_gaussian/tests/npu_baseline.py \
  --warmup 2 --repeats 5
```

测试资产说明见 [tests/README.md](tests/README.md)。

## 限制与后续

- 未做 kernel 级资源安全 UT：当前 `tiling fits()` 的 UB 预算已与 `LoGChunk::Init` 的 `InitBuffer` 逐项对齐（`(10+elem)*padded + (16+2k)*4*width + 2*stride + kUbSlack`），并已由 host UT 的预算边界用例覆盖，但缺少 kernel 级对拍用例，计划作为后续补充。
- 5 次为受控摸底，非生产稳态大样本；未覆盖全部属性组合与超大尺寸。
- 小尺寸（≤ 256²）受启动开销主导，若要进一步压缩需从 Init 与 block 调度入手，属后续优化项。
