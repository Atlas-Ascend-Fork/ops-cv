# GoodFeaturesToTrack × OpenCV 交叉验证

本目录提供 GoodFeaturesToTrack 算子与 OpenCV `cv::goodFeaturesToTrack` 的**可复现**对拍脚本与原始数据，
用于支撑 PR 中"内部格点与 OpenCV 逐个精确相等"的结论。

## 文件说明

| 文件 | 说明 |
| :--- | :--- |
| `run_crosscheck.py` | 对拍脚本：numpy 逐位重建输入 → 调用 OpenCV → 与算子输出逐点比对；`--import-dump` 从 examples 日志导入算子输出 |
| `golden/ours_corners.json` | **算子真机输出角点**（examples 的 `CORNERS` 行导入，含 `n` 与 `(x, y)` 列表） |
| `golden/opencv_golden.json` | OpenCV 参考角点 + 复现环境记录（OpenCV / numpy / python 版本与全部参数） |

## 用例与参数口径

两个用例的输入由 numpy 按 examples 的 `GenSyntheticImage` / `GenCheckerboard` 规则**逐位重建**，
参数与 examples 中同名用例完全一致（`qualityLevel=0.01`、`minDistance=1.0`、`blockSize=3`、
`useHarrisDetector=false`、`k=0.04`）：

| 用例 | shape | maxCorners | 几何真值 |
| :--- | :--- | ---: | :--- |
| `fp32 方块 blockSize=3` | 64×64，4 个 8×8 亮方块 | 1000 | 4 个方块各自 4 个外角，共 16 |
| `fp32 棋盘格 256x256` | 256×256 棋盘格 cell=8 | 20000 | 内部格点 $(x,y)=(8k-1,\,8l-1),\ k,l\in[1,31]$，共 961 |

## 如何复现

```bash
# 1) 构建算子包并运行 examples（设 GFT_DUMP_CORNERS=1 会额外打印 CORNERS 行）
cd <repo_root>
source <cann>/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=good_features_to_track -j16
export ASCEND_CUSTOM_OPP_PATH=$PWD/build_out/_CPack_Packages/Linux/External/cann-ops-cv-custom-linux.aarch64.run/packages/vendors/custom_cv
GFT_DUMP_CORNERS=1 bash build.sh --run_example good_features_to_track eager cust \
    --experimental --vendor_name=custom --soc=ascend910b | tee /tmp/gft_ex.log

# 2) 从日志导入算子输出，并与 OpenCV 对拍（需要 opencv-python；无 OpenCV 时加 --no-opencv）
python3 tests/opencv_crosscheck/run_crosscheck.py --import-dump /tmp/gft_ex.log
python3 tests/opencv_crosscheck/run_crosscheck.py

# 3) 换 OpenCV 版本复现时刷新 golden
python3 tests/opencv_crosscheck/run_crosscheck.py --write-golden
```

> OpenCV 版本会影响结果（不同版本的阈值/局部极大判据有差异）。已提交的 golden 由
> **OpenCV 5.0.0 / numpy 2.2.6 / python 3.10.12** 生成，完整信息见 `golden/opencv_golden.json` 的 `env` 字段。

## 判定口径

1. **几何真值必须被双方精确命中**：上表"几何真值"列的点，在算子输出与 OpenCV 输出中都必须逐点存在（0px 容差）。
2. **不允许漏检**：OpenCV 找到的每个角点，算子输出中都必须存在 1px（Chebyshev）内的对应点。
3. 真值以外的额外角点只做**报告**（不硬判）：阈值、局部极大判据、并列响应取舍不同所致。

## 实测结果（2026-10-15，Ascend 910B4 + OpenCV 5.0.0）

```
=== fp32 方块 blockSize=3  (h=64, w=64, maxCorners=1000) ===
  角点数：算子 = 16   OpenCV = 16   几何真值 = 16
  真值覆盖[算子]：精确命中 16/16
  真值覆盖[OpenCV]：精确命中 16/16
  OpenCV 角点在算子输出 1px 内的覆盖：16/16
  一对一匹配@0px：共同 16/16，算子独有 0，OpenCV 独有 0
  算子每个点到最近 OpenCV 点的距离：最大=0.0  中位=0.0

=== fp32 棋盘格 256x256  (h=256, w=256, maxCorners=20000) ===
  角点数：算子 = 1087   OpenCV = 3844   几何真值 = 961
  真值覆盖[算子]：精确命中 961/961
  真值覆盖[OpenCV]：精确命中 961/961
  OpenCV 角点在算子输出 1px 内的覆盖：3844/3844
  真值外角点[算子]：126 个，其中边界窄带内 126 个（y 范围 1..254）
  真值外角点[OpenCV]：2883 个，其中边界窄带内 0 个（y 范围 7..248）
  一对一匹配@0px：共同 961/1087，算子独有 126，OpenCV 独有 0（@1px）
  算子每个点到最近 OpenCV 点的距离：最大=6.0  中位=0.0

结果：PASS
```

结论：

- **方块图**：两边角点集合**完全一致**（16/16，0px，无任何多余/缺失）。
- **棋盘格**：961 个内部格点两边**逐点精确相等**（这是 PR 结论的原始依据）；OpenCV 找到的全部 3844 个角点，
  在算子输出 1px 内**全部**有对应点，即算子没有漏掉 OpenCV 的任何角点位置。
- **数量差异的来源**（已在算子 README「与 OpenCV 的行为差异」中披露）：
  - 算子多出的 126 个点**全部**落在距边界 2px 的窄带内，来自 Sobel 边界补 0（OpenCV 用 `BORDER_REFLECT_101`）。
  - OpenCV 多出的 2883 个点分布在棋盘格**格边台阶**处（y 7..248，与边界无关），是阈值 / 局部极大判据
    与并列响应取舍不同的结果，不属边界效应。
