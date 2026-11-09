# LbpFeature

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A2系列产品 | √ |

## 功能说明

- 算子功能：对输入图像（单个 H×W 平面或前置批次维的 ND 张量）计算**圆形邻域局部二值模式（LBP）特征编码**，
  与 opencv_contrib `face` 模块 `cv::elbp` 的实现语义一致（float32 计算域）。
- 计算过程：对每个中心像素 `c`，取圆周上均匀分布的 `neighbors` 个邻居
  （第 k 个邻居角度为 `2πk/neighbors`，圆周半径 `radius`），
  邻居值 `t_k` 由该处四个整数角点双线性插值得到，逐位生成二进制编码并累加为整数：

  ```text
  t_k   = w1*p1 + w2*p2 + w3*p3 + w4*p4          （邻居处双线性插值，w 为四角点权重）
  bit_k = (t_k > c) 或 |t_k - c| < FLT_EPS ? 1 : 0
  y     = Σ_{k=0..neighbors-1} (bit_k << k)     （int32 累加）
  ```

- 边界处理：输出图像比输入每边缩小 `radius` 个像素（不计算边界像素），输出 shape 为
  输入前导维不变、H/W 各减 `2×radius`。

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
      <td>FLOAT32、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>radius</td>
      <td>属性</td>
      <td>邻域半径，需 ≥ 0，默认 1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>neighbors</td>
      <td>属性</td>
      <td>圆周邻居个数，范围 [1, 31]，默认 8。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>LBP 编码图，shape 为输入前导维不变、H/W 各减 2×radius。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入 rank ≥ 2，最后两维为 H、W，其余前置维视为批量；所有维度需大于 0。
- `radius` ≥ 0；`1 ≤ neighbors ≤ 31`（编码位宽限制）。
- H、W 均需大于 `2×radius`（否则输出尺寸非正，算子返回失败）。
- 输出固定为 INT32，shape 按上述规则推导。
- 精度口径：与 opencv_contrib `face` 模块 `cv::elbp`（float32 计算域）逐位一致；
  插值运算顺序、比较语义（含非有限值传播）均保持一致，不做零权重角点剪枝。
- 当前仅支持 ND 格式。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| aclnn调用 | examples/test_aclnn_lbp_feature.cpp | aclnn 调用示例（uint8 / float32 两轮，与 UT 同源数据对拍）。 |

---

## 测试

- Host UT（Tiling / InferShape）：**21/21** 通过（Tiling 冻结基线 12 例、InferShape 9 例）；
- Kernel UT（CPU 仿真）：**13/13** 通过（uint8 2 例、float32 11 例，与 numpy golden 逐位对拍）；
- ACLNN 示例（真机）：uint8 / float32 双轮与同源 golden 逐位一致。

---

## 性能测试

本目录提供与 ops-cv 合仓规范一致的性能评测组件：

| 组件 | 路径 | 说明 |
|---|---|---|
| ACLNN 性能基准 | `examples/benchmark_aclnn_lbp_feature.cpp` | 40 例矩阵（2 dtype × 形状 × (radius, neighbors)），逐用例输出 host/device min/p50/p95/mean 与 workspace 字节数（CSV → stdout） |
| torch_npu 参考基准 | `tests/performance/benchmark_torch_npu_lbp_feature.py` | 同 case 表、同 CSV schema；逐邻居 `grid_sample` 版与向量化 gather 版两种实现，**每例取更快者**作为对标基线（最强合理参照） |
| 汇总与门槛判定 | `tests/performance/summarize_lbp_feature_performance.py` | 三轮 P50 中位数，gate 默认 1.05（Host+Device 双指标），输出 `summary.csv` / `summary.json` |

构建与运行（在已安装本算子包并 source CANN 环境后）：

```bash
cd examples
mkdir -p build_bench && cd build_bench
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# 三轮自定义算子 + 三轮 torch 参考（输出 custom_run*.csv / torch_run*.csv）
for r in 1 2 3; do ./bin/benchmark_aclnn_lbp_feature > custom_run$r.csv; done
for r in 1 2 3; do python3 ../../tests/performance/benchmark_torch_npu_lbp_feature.py > torch_run$r.csv; done

# 汇总与门槛判定
python3 ../../tests/performance/summarize_lbp_feature_performance.py --input .
```

结果摘要（Ascend 910B4 / CANN 9.1.0 / torch_npu 2.9.0）：全量 **40/40** 通过 Host/Device ≥1.05x 门槛（三轮 P50 中位数）；逐例数据见 `summary.csv` / `summary.json`。
