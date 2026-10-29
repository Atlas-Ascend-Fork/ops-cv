# PixelShuffle

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 910B 系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：PixelShuffle（亚像素卷积）将通道维重排到空间维，实现特征图空间上采样。4D 输入 `[N, C·r², H, W]` 输出 `[N, C, H·r, W·r]`；5D 输入 `[N, C·r³, D, H, W]` 输出 `[N, C, D·r, H·r, W·r]`。广泛应用于图像/视频超分辨率、生成网络末端上采样，以及 3D 医学影像（MRI/CT）体素超分等任务。

- 计算公式：

4D 置换关系：
$$
y[n,\,c,\,h\cdot r+\mathrm{hi},\,w\cdot r+\mathrm{wi}] = x[n,\,c\cdot r^{2}+\mathrm{hi}\cdot r+\mathrm{wi},\,h,\,w]
$$

5D 置换关系：
$$
y[n,\,c,\,d\cdot r+\mathrm{di},\,h\cdot r+\mathrm{hi},\,w\cdot r+\mathrm{wi}] = x[n,\,c\cdot r^{3}+\mathrm{di}\cdot r^{2}+\mathrm{hi}\cdot r+\mathrm{wi},\,d,\,h,\,w]
$$

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
      <td>输入张量。4D: shape [N, C·r², H, W]；5D: shape [N, C·r³, D, H, W]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量。4D: shape [N, C, H·r, W·r]；5D: shape [N, C, D·r, H·r, W·r]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale_factor</td>
      <td>属性</td>
      <td>上采样因子 r，取值范围 r ≥ 2。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

输入 rank 必须为 4 或 5（4D: [N, C, H, W]；5D: [N, C, D, H, W]），通道维固定为 index 1。4D 要求 C 可被 r² 整除，5D 要求 C 可被 r³ 整除，r 需满足 2 ≤ r ≤ 2²⁰（超过上界时 Tiling/InferShape 阶段返回失败）。输入 dtype 支持 float16 和 float32，输出 dtype 与输入一致，格式均为 ND。算子注册的 AICore 配置为 ascend910b 与 ascend910_93。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn接口 | [test_aclnn_pixel_shuffle](examples/test_aclnn_pixel_shuffle.cpp) | 通过 aclnnPixelShuffle 接口方式调用 PixelShuffle 算子。 |
| 图模式 | - | 通过[算子IR](op_graph/pixel_shuffle_proto.h)构图方式调用PixelShuffle算子。 |
