# Scharr

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

- 算子功能：对输入图像（单个 HxW 平面或前置批次维的 ND 张量）计算 Scharr 一阶差分，
  与 OpenCV `cv2.Scharr(..., ddepth=CV_32F)` 在浮点（fp32）域**逐位一致**。
- 计算公式（X 方向，`direction="X"`）：

  ```text
  y(r,c) = 3*x(r-1,c-1) - 3*x(r-1,c+1)
         + 10*x(r,c-1)  - 10*x(r,c+1)
         + 3*x(r+1,c-1) - 3*x(r+1,c+1)
  ```

  Y 方向（`direction="Y"`）为上式沿主对角转置的版本。
- `border_type` 控制越界采样：`REFLECT_101`（反射，默认）、`REPLICATE`（边缘复制）、
  `CONSTANT`（常量 0）。
- uint8 / float16 输入先升为 float32 再参与计算（与 OpenCV 参考路径一致）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 120px">
  <col style="width: 360px">
  <col style="width: 180px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入图像。支持前置批次维，最后两维为 H、W。</td>
      <td>FLOAT32、FLOAT16、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>border_type</td>
      <td>属性</td>
      <td>边界模式："REFLECT_101" / "REPLICATE" / "CONSTANT"，默认 "REFLECT_101"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>direction</td>
      <td>属性</td>
      <td>差分方向："X" 或 "Y"，默认 "X"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出图像，shape 与输入一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入 rank ≥ 2，最后两维为 H、W，其余前置维视为批量；所有维度需大于 0。
- 输出固定为 FLOAT32，shape 与输入一致。
- 精度口径：与 OpenCV（cv2 5.x，`ddepth=CV_32F`）在浮点域逐位一致；
  uint8 输入等价于 `cv2.Scharr(u8.astype(np.float32))`（OpenCV 原生调用已验证与其位级一致）。
- 当前仅支持 ND 格式。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| aclnn调用 | examples/test_aclnn_scharr.cpp | aclnn 调用示例。 |

---

## 测试

- Host UT（Tiling / InferShape）：**7/7** 通过（Tiling 冻结基线 3 例、InferShape 4 例）；
- Kernel UT（CPU 仿真）：**13/13** 通过（uint8 6 例、float32 5 例、float16 2 例，与 numpy golden 逐位对拍）；
- ACLNN 示例（真机）：uint8 / float16 / float32 三轮与 golden 逐位一致。

---

## 性能测试

本目录提供与 ops-cv 合仓规范一致的性能评测组件：

| 组件 | 路径 | 说明 |
|---|---|---|
| ACLNN 性能基准 | `examples/benchmark_aclnn_scharr.cpp` | 42 例矩阵（3 dtype × 2 方向 × 3 边界 × 分辨率/非对齐/批量），逐用例输出 host/device min/p50/p95/mean 与 workspace 字节数（CSV → stdout） |
| torch_npu 参考基准 | `tests/performance/benchmark_torch_npu_scharr.py` | 同 case 表、同 CSV schema（cat 拼边界 + 切片乘加） |
| 汇总与门槛判定 | `tests/performance/summarize_scharr_performance.py` | 三轮 P50 中位数，gate 默认 1.05（Host+Device 双指标），输出 `summary.csv` / `summary.json` |
| 性能剖面 | `tests/performance/run_profile_case.sh`、`summarize_scharr_profiles.py` | 单用例采集（Task Duration、Block Dim、核间均衡、Vec/MTE2/MTE3/Scalar 占比、主存与 UB 带宽、L2 命中率） |

**对标口径**：基线为同卡 `torch_npu` 等价组合计算链；`speedup = torch latency / custom latency`，验收门槛 Host 与 Device 均需满足 `torch/custom >= 1.05x`；每用例 warmup 后采集 repeats 次取 P50，正式结果取三轮 P50 的中位数；初始化、输入构造、Executor/Workspace 申请等一次性开销不计入正式结果。

构建与运行（在已安装本算子包并 source CANN 环境后）：

```bash
cd examples
mkdir -p build_bench && cd build_bench
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# 三轮自定义算子 + 三轮 torch 参考（输出 custom_run*.csv / torch_run*.csv）
for r in 1 2 3; do ./bin/benchmark_aclnn_scharr > custom_run$r.csv; done
for r in 1 2 3; do python3 ../../tests/performance/benchmark_torch_npu_scharr.py > torch_run$r.csv; done

# 汇总与门槛判定
python3 ../../tests/performance/summarize_scharr_performance.py --input .
```

结果摘要（Ascend 910B4 / CANN 9.1.0 / torch_npu 2.9.0）：全量 **42/42** 通过 Host/Device ≥1.05x 门槛；计算主导的大图（2048×2048）场景 Device 加速比 **1.94x~4.83x（中位 3.7x）**，可作为保守参考；逐例数据见 `summary.csv` / `summary.json`。

**性能结构说明**（与等价 Torch 组合链对比）：

| 指标 | 等价 Torch 组合链 | 本算子 |
|---|---:|---:|
| 单轮算子下发次数 | ~13 次 | 1 次 |
| 全尺寸中间张量 | 5~6 个 | 0 个 |
| 全尺寸访存遍数 | ~10 遍 | 1 遍 |

## 对拍验证（cv2）

`tests/golden/compare_with_cv2.py` 提供与 OpenCV 的可复现对拍：构建安装算子包后运行示例，
示例会在当前目录 dump 输入/输出 bin，再用脚本以 cv2 生成参考结果并逐位比对。

1. 构建并安装算子包：

   ```bash
   bash build.sh --pkg --experimental --ops=scharr --soc=ascend910b
   bash build_out/cann-ops-cv-custom-linux.aarch64.run
   ```

2. 编译并运行示例（固定用例：uint8/float32/float16 × X/Y × 边界模式）：

   ```bash
   VENDOR=${ASCEND_HOME_PATH}/opp/vendors/custom_cv
   g++ -std=c++17 -O2 examples/test_aclnn_scharr.cpp \
     -I${ASCEND_HOME_PATH}/include -I${VENDOR}/op_api/include \
     -L${ASCEND_HOME_PATH}/lib64 -L${VENDOR}/op_api/lib \
     -lascendcl -lnnopbase -lcust_opapi -o test_aclnn_scharr
   LD_LIBRARY_PATH=${VENDOR}/op_api/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH} ./test_aclnn_scharr
   ```

3. 执行对拍（`--shape` 与示例中的 H,W 一致）：

   ```bash
   python3 tests/golden/compare_with_cv2.py --dump-dir . --shape 65,64
   ```

参考结果（Ascend 910B4 / CANN 9.1.0 / OpenCV 5.0.0 aarch64）：3/3 逐位一致，`max_abs_diff=0`。
