# LaplacianOfGaussian 测试资产

本目录提供仓内可复核的数值验证与性能对比资产：

| 文件 | 作用 |
| --- | --- |
| `launch.py` | 入口：解包算子包、编译 `runner.cpp` / `benchmark.cpp`、准备环境后调用 `cases.py` |
| `cases.py` | 用例定义（按 `--suite` 生成）与逐例比对、基准计时 |
| `golden.py` | 独立的 FP64 逐 tap 二维卷积 golden（与 SciPy 结果互为复核） |
| `npu_baseline.py` | `torch_npu` 基线（full2d / separable），供 `--baseline` 使用 |
| `preflight.py` | 采集前检查目标卡健康、空闲与进程占用 |
| `profile_tasks.py` | 可选的原生 msprof 任务时长采集（`--profile`） |
| `runner.cpp` | ACLNN 两段式调用 + 输入只读/输出逐位重复/设备映射校验 |
| `benchmark.cpp` | 常驻进程内的事件区间计时驱动 |
| `ut/` | Host UT（tiling、infershape），由 `build.sh -u --ophost` 驱动 |

## 环境要求

- 已 `source` CANN 环境（`CANN_ENV` 或 `ASCEND_INSTALL_PATH`），可访问目标 Device；
- Python 3.11，`numpy==1.26.4`、`scipy==1.13.1`（版本在 `launch.py` 中强校验，请使用独立 venv 或 `PYTHONPATH` 指向私有的固定版本，不要覆盖共享环境）；
- `torch` 与 `torch_npu` 仅在跑性能对比（`--baseline`）时需要；
- 通过 `ASCEND_RT_VISIBLE_DEVICES` 指定目标卡（默认脚本内为 4，可按环境调整）。

## 用例套件

| `--suite` | 用例数 | 覆盖 |
| --- | ---: | --- |
| `smoke` | 3 | 三种 dtype 各一，快速冒烟 |
| `full` | 29 | 20 个基础代表例 + 9 个 batch 变体，覆盖 rank 2/3/4、四种边界、自动与显式 radius |
| `extended` | 432 | 36 组 shape/radius × 3 dtype × 4 边界，含 128×8193 长图、r=123 大半径与 1×1 退化尺寸 |
| `large` | 15 | 512² / 1080p / 4K，以及 sigma=8、sigma=0.3、radius=0 三个极端例 |
| `stability` | 168 | 120 个基础例 + 48 个 original/flip/roll/newseed 重放变体 |
| `perf` | 6 | 2 尺寸 × 3 dtype，`--mode benchmark` 使用 |
| `perf2` | 37 | 6 尺寸 × 3 dtype + 边界扫描 + sigma 扫描 + 4K，性能对比主表 |

## 使用

```bash
# 1. 构建并产出安装包
bash build.sh --pkg --experimental --soc=ascend910b --ops=laplacian_of_gaussian \
  --vendor_name=custom --cann_3rd_lib_path=$PWD/third_party -j8

# 2. 数值验证（示例：代表集）
source $CANN_ENV
python3 experimental/image/laplacian_of_gaussian/tests/launch.py \
  --suite full --package build_out/cann-ops-cv-custom-linux.aarch64.run

# 3. 性能对比（含 torch_npu 基线，warmup 2 / repeats 5）
python3 experimental/image/laplacian_of_gaussian/tests/launch.py --mode benchmark \
  --suite perf2 \
  --package build_out/cann-ops-cv-custom-linux.aarch64.run \
  --baseline experimental/image/laplacian_of_gaussian/tests/npu_baseline.py \
  --warmup 2 --repeats 5
```

`--output` 必须指定一个不存在的目录；不指定时在临时目录下创建。结果目录保留逐例输入、输出、golden、JSON 与日志，便于复核。

## 判定口径

- 逐像素要求 `abs(error) <= 2^-16 + 2^-10 * abs(golden)`、`max_abs <= 0.01`、RMSE <= 混合容差界；
- 每例同时与 SciPy FP64 golden 和独立 FP64 逐 tap 卷积 golden 比对，两者需互相一致；
- 额外校验输入只读、同一输入两次调用输出逐位相同、`physical/user/logic` 设备映射；
- 性能比较仅采用精度合格的基线实现；计时为设备事件区间，预热 2 次、测量 5 次取平均。

## 说明

- 完整数据与结论见 [PERFORMANCE.md](../PERFORMANCE.md)。
- kernel 级资源安全 UT（`tiling fits()` 预算与 `LoGChunk::Init` 分配逐项对拍的 kernel UT）尚未交付，计划随后续 PR 补充；当前 UB 预算由 host UT 的边界用例覆盖。
