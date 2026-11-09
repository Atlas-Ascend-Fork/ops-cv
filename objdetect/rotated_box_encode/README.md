# RotatedBoxEncode

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

将参考旋转框（anchor_box）与对应的 ground-truth 旋转框（gt_box）之间的几何偏差编码为 5 通道回归 delta 目标 `(dx, dy, dw, dh, dθ)`，用于旋转目标检测训练阶段的回归监督。输入与输出均为 3D 张量 `(B, 5, N)`，5 通道按角点形式 corner form `(x0, y0, x1, y1, θ_deg)` 组织，`θ_deg` 为角度制。设 `weight = (wx, wy, ww, wh, wa)`，逐通道编码公式为：

$$
\begin{aligned}
w_a &= \max(x1_a - x0_a,\ 1.0),\quad h_a = \max(y1_a - y0_a,\ 1.0) \\
cx_a &= x0_a + w_a / 2,\quad cy_a = y0_a + h_a / 2 \\
w_g &= \max(x1_g - x0_g,\ 1.0),\quad h_g = \max(y1_g - y0_g,\ 1.0) \\
cx_g &= x0_g + w_g / 2,\quad cy_g = y0_g + h_g / 2 \\
dx &= \frac{cx_g - cx_a}{w_a} \cdot wx \\
dy &= \frac{cy_g - cy_a}{h_a} \cdot wy \\
dw &= \ln\frac{w_g}{w_a} \cdot ww \\
dh &= \ln\frac{h_g}{h_a} \cdot wh \\
d\theta &= \bigl(\tan(\theta_g \cdot \pi/180) - \tan(\theta_a \cdot \pi/180)\bigr) \cdot wa
\end{aligned}
$$

其中 `max(·, 1.0)` 防止退化框导致除零与 `log(0)`，并按 `np.maximum` 语义传播 NaN。fp16 输入在内部升精度到 fp32 完成对数、三角与除法计算后再回写 fp16。

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
<td>表示anchor参考旋转框，对应公式中anchor_box。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>gt_box</td>
<td>输入</td>
<td>表示ground-truth旋转框，对应公式中gt_box。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>表示编码后的5通道delta，对应公式中(dx,dy,dw,dh,dθ)。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>weight</td>
<td>可选属性</td>
<td>5通道编码权重[wx,wy,ww,wh,wa]，默认[1.0,1.0,1.0,1.0,1.0]。</td>
<td>ListFloat</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- anchor_box、gt_box、y 三者数据类型必须一致，且仅支持 FLOAT16 或 FLOAT。
- anchor_box 与 gt_box 的 shape 必须完全一致（无广播），秩为 3，且第 1 维必须为 5。5 通道顺序固定为 `(x0, y0, x1, y1, θ_deg)`。
- `-1` 表示动态维、`-2` 表示动态秩，仅用于构图期；InferShape 会传播未知维/未知秩。上板执行时所有维度必须是具体的非负值，负维会被 Tiling 拒绝。
- weight 若提供，长度必须为 5，元素应为有限值；非有限 weight 的行为未定义。不提供时使用默认值 `[1.0, 1.0, 1.0, 1.0, 1.0]`。
- 角度 `θ_deg` 的有效计算范围受硬件三角函数限制（约 ±65504 rad，即约 ±3.75e6 度）。角度超出该范围时，`dθ` 可能输出 NaN；调用方应保证输入角度在有效范围内。
- 空Tensor（B=0 或 N=0）直接短路返回空输出，不报错。
- 数据类型、格式、秩、维度、两输入同形或 weight 长度不满足上述约束时，构图或 Tiling 阶段返回失败。具体错误码由框架统一映射，本文档不单独定义错误码表。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| 图模式 | [test_geir_rotated_box_encode](./examples/arch35/test_geir_rotated_box_encode.cpp) | 通过[算子IR](./op_graph/rotated_box_encode_proto.h)构图方式调用RotatedBoxEncode算子。 |

图模式样例需显式指定 SoC，例如：

```bash
bash build.sh --run_example rotated_box_encode graph --soc=ascend950
```
