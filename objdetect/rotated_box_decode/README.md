# RotatedBoxDecode

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

RotatedBoxDecode 是旋转目标检测中的框回归解码算子，将网络预测的偏移量（deltas）叠加到预设锚框（anchor_box）上，还原出最终的旋转检测框。

输入为角点格式 `[lx, ly, rx, ry, angle]`（角度单位为度），计算分六步：

1. **锚框角点转中心**：由 `(lx, ly, rx, ry)` 算 `(cx, cy, w, h)`，宽高夹下限 1
2. **delta 归一化**：deltas 五通道分别除以 weight
3. **解中心**：`t_cx = a_cx + Δx·a_w`，`t_cy = a_cy + Δy·a_h`
4. **解宽高**：`t_w = exp(Δw)·a_w`，`t_h = exp(Δh)·a_h`
5. **解角度**：`θ_t = atan(tan(θ_a) + Δt)`（度↔弧度转换，tan 空间加法抗 180° 周期）
6. **中心转角点**：由 `(cx, cy, w, h)` 还原 `(lx', ly', rx', ry')`，角度透传

计算公式如下，其中 `Δ = deltas / weight`（weight 五元 `[wx, wy, ww, wh, wt]`，按通道对应相除），角度在 tan 空间相加再 atan，输出主值区间 `[-90°, 90°)`，宽高 `max(·, 1)` 夹下限防止退化：

$$
\begin{aligned}
a_w &= \max(rx - lx,\ 1),\quad a_h = \max(ry - ly,\ 1) \\
a_{cx} &= lx + a_w / 2,\quad a_{cy} = ly + a_h / 2 \\
\Delta &= \text{deltas} / \text{weight} \\
t_{cx} &= a_{cx} + \Delta x \cdot a_w \\
t_{cy} &= a_{cy} + \Delta y \cdot a_h \\
t_w    &= e^{\Delta w} \cdot a_w \\
t_h    &= e^{\Delta h} \cdot a_h \\
\theta_t &= \arctan\big(\tan(\theta_a) + \Delta t\big) \\
lx' &= t_{cx} - t_w/2,\quad rx' = t_{cx} + t_w/2 \\
ly' &= t_{cy} - t_h/2,\quad ry' = t_{cy} + t_h/2
\end{aligned}
$$

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
<td>anchor_box</td>
<td>输入</td>
<td>锚框，3D (B, 5, N)，5 = [lx, ly, rx, ry, angle]，角度单位为度；对应公式中 anchor_box。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>deltas</td>
<td>输入</td>
<td>网络回归偏移，3D (B, 5, N)，5 = [dx, dy, dw, dh, dt]；对应公式中 deltas。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>解码后旋转框，3D (B, 5, N)，5 = [lx', ly', rx', ry', θ_t]；shape/dtype 与 anchor_box 一致。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>weight</td>
<td>可选属性</td>
<td>五元权重 [wx, wy, ww, wh, wt]，deltas 各通道分别除以对应权重进行归一化；默认 [1, 1, 1, 1, 1]。</td>
<td>ListFloat</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- anchor_box、deltas、y 三者 shape 完全相同，dtype 相同。
- 必须 3D，shape[1] 必须为 5，shape[0] ≥ 0，shape[2] ≥ 0。
- weight 长度必须为 5，各元素不能为 0（否则产生 inf/NaN，kernel 不校验）。
- 数据布局为 ND，不支持其他布局。
- 支持动态 Shape 与动态 Rank：算子注册开启 DynamicShapeSupportFlag 与 DynamicRankSupportFlag，anchor_box/deltas 的维度可标记为 -1（未知维），整体 shape 可标记为 -2（未知秩）。InferShape 对未知维/未知秩放行；维度被具体化后再由 Tiling 校验：Tiling 仅校验 rank=3 与 shape[1]=5，因此未知秩（{-2}）或 shape[1]=-1 会被 Tiling 拒绝；shape[0]/shape[2] 为 -1 时 Tiling 不做拒绝，必须在 Tiling 前具体化，否则 tiling 参数非法、结果未定义。
- 支持空 Tensor（B=0 或 N=0）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_rotated_box_decode.cpp](examples/arch35/test_geir_rotated_box_decode.cpp) | 通过 [算子IR](op_graph/rotated_box_decode_proto.h) 构图调用。 |
