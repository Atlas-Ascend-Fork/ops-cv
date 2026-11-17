# BlendFaceBgPartTwo

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：Alpha混合归一化的累积人脸与背景图像（BlendFaceBg流程的第二部分）。
- 计算公式：

  给定累积人脸图像acc_face、累积掩码acc_mask、最大掩码max_mask和背景图像bg_img，计算融合图像：

  $$
  fusion\_face = \frac{acc\_face}{acc\_mask + \epsilon}
  $$

  $$
  bg\_weight = 1 - max\_mask
  $$

  $$
  fused\_img = fusion\_face \times max\_mask + bg\_img \times bg\_weight
  $$

  其中：
  - acc_face：累积的人脸图像，来自BlendFaceBg流程的第一部分
  - acc_mask：累积掩码，用于归一化分母
  - max_mask：最大掩码，作为Alpha混合权重，取值范围[0,1]
  - bg_img：背景图像，支持float32或uint8类型（uint8会在算子内部转换为float32）
  - epsilon：防除零的小量，默认值1e-12

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>acc_face</td>
      <td>输入</td>
      <td>累积人脸图像，来自BlendFaceBg流程的第一部分，shape为(H,W,C)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>acc_mask</td>
      <td>输入</td>
      <td>累积掩码，用于归一化分母，shape为(H,W,C)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_mask</td>
      <td>输入</td>
      <td>最大掩码，作为Alpha混合权重，取值范围[0,1]，shape为(H,W,C)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bg_img</td>
      <td>输入</td>
      <td>背景图像，shape为(H,W,C)。当类型为uint8时，算子内部会转换为float32进行计算。</td>
      <td>UINT8、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>属性</td>
      <td>防除零小量，应用于acc_mask。默认值为1e-12。</td>
      <td>FLOAT</td>
      <td>标量</td>
    </tr>
    <tr>
      <td>fused_img</td>
      <td>输出</td>
      <td>Alpha混合后的融合图像，shape与acc_face相同。数据类型固定为FLOAT。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 所有输入张量的shape必须相同，且rank必须为3（H,W,C）。
- acc_face、acc_mask、max_mask的数据类型必须为FLOAT。
- bg_img的数据类型可以是FLOAT或UINT8。
- 输出fused_img的数据类型固定为FLOAT。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [examples/arch35/test_geir_blend_face_bg_part_two.cpp](examples/arch35/test_geir_blend_face_bg_part_two.cpp)（固定shape） | 通过[算子IR](op_graph/blend_face_bg_part_two_proto.h)构图方式调用BlendFaceBgPartTwo算子。 |
