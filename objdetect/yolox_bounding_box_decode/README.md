# YoloxBoundingBoxDecode

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

- 算子功能：根据YOLOX解码公式，将模型预测的归一化偏移量bboxes与先验框priors解码为最终的边界框坐标（左上角、右下角）。

- 计算公式：

$$
xys = bboxes[..., 0:2] \times priors[:, 2:4] + priors[:, 0:2]
$$

$$
whs = \exp(bboxes[..., 2:4]) \times priors[:, 2:4] \times 0.5
$$

$$
decoded\_bboxes[..., 0:2] = xys - whs
$$

$$
decoded\_bboxes[..., 2:4] = xys + whs
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
      <td>priors</td>
      <td>输入</td>
      <td>先验框信息，shape为(N, 4)，每行为[offset_x, offset_y, scale_x, scale_y]。前两列为中心偏移，后两列为缩放因子，用于缩放dx/dy及exp(dw/dh)。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bboxes</td>
      <td>输入</td>
      <td>模型预测的偏移量，shape为(B, N, 4)，每行为[dx, dy, dw, dh]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>decoded_bboxes</td>
      <td>输出</td>
      <td>解码后的边界框，shape为(B, N, 4)，每行为[左上角x, 左上角y, 右下角x, 右下角y]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入priors仅支持2维，shape为(N, 4)，N为特征图上的先验框数量。
- 输入bboxes仅支持3维，shape为(B, N, 4)，其中N必须与priors的N一致。
- 输出decoded_bboxes与bboxes的shape一致，为(B, N, 4)。
- 输入dtype仅支持FLOAT16、FLOAT，两个输入的dtype需保持一致。
- exp运算溢出/下溢按IEEE 754规则传播，不做clamp处理。
- 支持空tensor场景（B=0或N=0），此时输出为空。

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
    <td><a href="./examples/test_geir_yolox_bounding_box_decode.cpp">test_geir_yolox_bounding_box_decode</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
