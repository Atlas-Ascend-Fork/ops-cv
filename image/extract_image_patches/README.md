# ExtractImagePatches

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |     √    |
| <term>Atlas A2系列产品</term>    |     √    |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：从 4D 输入图像中按滑动窗口提取图像块（patch），并将每个 patch 展平拼接到通道维。兼容 TensorFlow 的 `ExtractImagePatches`，常用于卷积替代、局部特征聚合等场景。

- 计算公式：

对于 NHWC 格式输入 `x` 形状为 `[N, H, W, C]`，输出 `y` 形状为 `[N, out_h, out_w, C * kH * kW]`：

$$
y[n, i, j, (p \cdot kW + q) \cdot C + c] = x[n, i \cdot stride_h + p \cdot rate_h - pad_h, j \cdot stride_w + q \cdot rate_w - pad_w, c]
$$

其中：

- `kH = ksizes[H]`，`kW = ksizes[W]`：patch 高/宽。
- 展平顺序为 patch 行、patch 列、通道（`p`、`q`、`c`），其中 `0 <= c < C`。
- `stride_h = strides[H]`，`stride_w = strides[W]`：滑动步长。
- `rate_h = rates[H]`，`rate_w = rates[W]`：扩张率（dilation）。
- `pad_h/pad_w`：由padding模式（SAME/VALID）决定的padding量。
- 采样点越界时（VALID 模式）对应输出位置不存在；SAME模式补零。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>4D 图像输入，支持NHWC与NCHW两种origin format。公式中的x。</td>
      <td>FLOAT16、FLOAT、BF16、INT8、UINT8</td>
      <td>NHWC/NCHW</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>patch展平到通道维的输出，format与输入origin format一致。公式中的y。</td>
      <td>FLOAT16、FLOAT、BF16、INT8、UINT8</td>
      <td>NHWC/NCHW</td>
    </tr>
    <tr>
      <td>ksizes</td>
      <td>属性</td>
      <td>patch大小，长度 4 的ListInt，空间 H/W 维必须大于 0，N/C 维必须为 1。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>滑动步长，长度 4 的ListInt，N/C维必须为 1。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rates</td>
      <td>属性</td>
      <td>扩张率（dilation），长度 4 的ListInt，N/C维必须为 1。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性</td>
      <td>padding模式，取值为"SAME"或"VALID"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入必须为4D张量（NHWC或NCHW格式）。
- ksizes/strides/rates长度必须为4，且N/C维必须为 1。
- ksizes和strides的H/W维必须大于 0，rates的H/W维必须大于等于 1。
- padding取值仅支持 "SAME" 或 "VALID"。
- 输入输出dtype相同，无类型提升。
- 纯数据搬运算子，所有特殊值（NaN/Inf/+0/-0）原样透传，SAME padding越界补零值为+0.0。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/arch35/test_geir_extract_image_patches.cpp">test_geir_extract_image_patches</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
