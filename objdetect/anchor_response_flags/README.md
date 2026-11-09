# AnchorResponseFlags

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

- 算子功能：在目标检测网络中生成锚框（anchor）的响应标志。根据真值框（ground truth bounding boxes）的中心点位置，确定哪些锚框网格位置负责检测目标，并生成对应的标志位。

- 计算公式：

$$
cx_i = (x1_i + x2_i) \times 0.5
$$

$$
cy_i = (y1_i + y2_i) \times 0.5
$$

$$
grid\_x_i = \lfloor cx_i / stride\_w \rfloor
$$

$$
grid\_y_i = \lfloor cy_i / stride\_h \rfloor
$$

$$
grid\_idx_i = grid\_y_i \times feat\_w + grid\_x_i
$$

$$
responsible\_grid[grid\_idx_i] = 1
$$

$$
output = repeat(responsible\_grid, num\_base\_anchors)
$$

- 求值精度：上述公式中的除法以 float32 精度求值，kernel 与参考实现均先做 float32 除法再向下取整；当 stride 超过 2^24（float32 可精确表示的整数上限）时，除数按 float32 舍入后的值参与运算。

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
      <td>gt_bboxes</td>
      <td>输入</td>
      <td>真值框坐标，shape为[N, 4]，格式为[x1, y1, x2, y2]。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>flags</td>
      <td>输出</td>
      <td>锚框响应标志，shape为[feat_h * feat_w * num_base_anchors]，值为0或1。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>featmap_size</td>
      <td>属性</td>
      <td>特征图大小，长度不少于 2 的列表 [feat_h, feat_w]，仅前两个元素参与计算，多余元素忽略。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>步长，长度不少于 2 的列表 [stride_w, stride_h]，值必须为正整数。x 方向步长在前，y 方向步长在后，仅前两个元素参与计算，多余元素忽略。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_base_anchors</td>
      <td>属性</td>
      <td>每个网格位置的锚框数量。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

1. 输入 gt_bboxes 必须为 2D tensor，第二维必须为 4，行数 N 不超过 INT32_MAX（2^31-1）。
2. featmap_size 与 strides 的长度均必须不少于 2，仅前两个元素参与计算（featmap_size 取 [feat_h, feat_w]，strides 取 [stride_w, stride_h]），多余元素忽略。
3. featmap_size 中的值必须为正整数（不包含 0），且不超过 INT32_MAX。
4. strides 中的值必须为正整数（不包含 0），且不超过 INT32_MAX。
5. num_base_anchors 取值范围为 [0, INT32_MAX]（不含负数）。
6. 输出 dtype 固定为 uint8，不随输入 dtype 变化。
7. 当 gt_bboxes 为空（N=0）时，输出全零 tensor。
8. 中心点映射后的 grid 索引超出特征图范围时，索引被裁剪到有效范围内：负索引裁剪到 0（即中心点位于特征图左/上边界之外时映射到首行/首列网格），正向越界裁剪到 feat_size - 1。
9. 当中心点坐标值为 NaN 或 Inf（包括中心点求和溢出为 Inf 的情况）时，该中心点映射到 grid 索引 0。
10. gt_bboxes 支持动态 Shape：第 0 维（行数 N）在构图时可置 -1（unknown dim），也支持 unknown rank（shape 为 [-2]）。此时 InferShape 跳过对输入维度的校验，输出 shape [feat_h * feat_w * num_base_anchors] 仅由属性推导，可与动态 batch 图配合使用；N 在图执行前由实际输入具体化。

## 调用说明

| 调用方式 | 调用样例     | 说明           |
|---------|---------------------------|--------------------------------|
| GE图模式 | [test_geir_anchor_response_flags](examples/arch35/test_geir_anchor_response_flags.cpp)  | 通过[算子IR](op_graph/anchor_response_flags_proto.h)构图方式调用AnchorResponseFlags算子。   |
