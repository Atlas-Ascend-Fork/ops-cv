# DeformableRoiPool

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：可变形 ROI 池化。对每个 ROI，在特征图上按空间分 bin 进行可变形采样（可选偏移量），对每个 bin 内的采样点做双线性插值后取平均，得到池化输出。

- 计算公式：

$$
y(n, c, ph, pw) = \frac{1}{N_{bin}} \sum_{iy=0}^{H_{bin}-1} \sum_{ix=0}^{W_{bin}-1} \sum_{q} w_q \cdot x(idx_n, c, y_q, x_q)
$$

其中：

- $n$ 为 ROI 索引，$c$ 为通道索引，$(ph, pw)$ 为输出空间位置。
- $idx_n$ 为第 $n$ 个 ROI 的 batch 索引（$rois[n, 0]$）。
- $H_{bin}$ 和 $W_{bin}$ 为每个 bin 内的采样点数（由 $sampling\_ratio$ 或 bin 大小自适应确定）。
- $w_q$ 为双线性插值权重，$(y_q, x_q)$ 为四角点坐标。
- 采样坐标经过 offset 修正：$start_h = roi\_start_h + ph \cdot bin\_size_h + offset(n, 1, ph, pw) \cdot \gamma \cdot roi\_height$，
  $start_w = roi\_start_w + pw \cdot bin\_size_w + offset(n, 0, ph, pw) \cdot \gamma \cdot roi\_width$。

核心步骤：

1. ROI 坐标变换：$roi\_start = roi \cdot spatial\_scale - 0.5$。
2. Bin 大小：$bin\_size_h = roi\_height / pooled\_h$，$bin\_size_w = roi\_width / pooled\_w$。
3. 采样点数：$sampling\_ratio > 0$ 时取该值，否则自适应为 $\lceil bin\_size \rceil$。
4. 应用 offset 偏移量（可选）。
5. 双线性插值：对每个 bin 内的每个采样点，取四角点值加权累加。
6. 平均池化：bin 内所有采样点累加值除以采样点数。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 130px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>特征图。shape 为 <code>(N, C, H, W)</code> 的 NCHW 张量，公式中的 x。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>ROI 位置坐标。shape 为 <code>(num_rois, 5)</code>，每行格式为 <code>(batch_idx, x1, y1, x2, y2)</code>。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>可选输入</td>
      <td>偏移量场。shape 为 <code>(num_rois, 2, pooled_h, pooled_w)</code>，第 1 维分别对应宽度和高度方向的偏移。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>池化输出。shape 为 <code>(num_rois, C, pooled_h, pooled_w)</code>。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性</td>
      <td>特征图到原始图像的空间缩放因子，默认值1.0。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>输出空间大小，格式为 <code>[pooled_height, pooled_width]</code>。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sampling_ratio</td>
      <td>属性</td>
      <td>每个 bin 在高度和宽度方向的采样点数，取值范围为 <code>[0, 46340]</code>。0 表示根据 bin 大小自适应，大于 0 时使用指定值，默认值为 0。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>属性</td>
      <td>偏移量缩放因子，偏移量乘以gamma后再与ROI宽度/高度相乘得到实际偏移，默认值0.1。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入 `x` 必须为 4 维 NCHW 张量，shape 为 `(N, C, H, W)`；`N` 可以为 0，`C`、`H`、`W` 必须为正整数。
- 输入 `rois` 必须为 2 维张量，shape 为 `(num_rois, 5)`，第 1 维固定为 5。
- 输入 `offset` 为可选参数，若不传入则不应用偏移量，等效于 `offset` 全零。若传入，shape 必须为
  `(num_rois, 2, pooled_h, pooled_w)`，其中 `pooled_h` 和 `pooled_w` 与 `output_size` 属性一致。
- `num_rois` 由 `rois.shape[0]` 决定，`output_size` 必须包含 2 个 `[1, INT32_MAX]` 范围内的整数。
- 输入 `x` 的元素总数（`N * C * H * W`）不得超过 500000000；输出 `y` 的元素总数
  （`num_rois * C * pooled_h * pooled_w`）不得超过 200000000。超过上述上限时算子返回 `GRAPH_FAILED` 拒绝执行。
- `x` 的数据类型决定输出 `y` 的数据类型，`rois` 和 `offset` 的数据类型必须与 `x` 一致。
- 仅支持 FLOAT16 和 FLOAT 数据类型，不支持 INT 类型。
- `sampling_ratio` 必须在 `[0, 46340]` 范围内；`spatial_scale` 必须为大于 0 的有限值，`gamma` 必须为有限值。
- 当 `rois.shape[0]` 为 0 时，输出 shape 为 `(0, C, pooled_h, pooled_w)`；当 `N` 为 0 且 `num_rois` 大于 0 时，输出保持 `(num_rois, C, pooled_h, pooled_w)`，所有元素为 0。
- 支持动态 Shape 与动态 Rank：算子注册开启 DynamicShapeSupportFlag 与 DynamicRankSupportFlag，`x` 的维度可标记为 `-1`（未知维）或 `-2`（未知秩）。InferShape 对未知维/未知秩放行；维度被具体化后再由 Tiling 按上述 `[1, INT32_MAX]` 等既定范围校验，未知维（`-1`）若未具体化则无法通过 Tiling 校验。
- ROI 的 batch 索引会截断到 `[0, N - 1]`。ROI/offset 中的 NaN 或 Inf 按 0 处理；落在 `[-1, H] × [-1, W]` 之外的采样点不参与累加。
- 自适应采样单方向网格超过 46340 时，该 ROI 不产生有效采样点，输出为 0。
- 运行时要求算子仓库根目录下的 `CMakeLists.txt` 已完成编译配置。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| GE图模式 | [test_geir_deformable_roi_pool](examples/test_geir_deformable_roi_pool.cpp) | 通过[算子IR](op_graph/deformable_roi_pool_proto.h)构图方式调用DeformableRoiPool算子。 |
