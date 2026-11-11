# Lut3D

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：3D LUT（Look-Up Table）三线性插值颜色变换。将输入图像通过3D颜色查找表进行颜色映射，支持三线性插值以获得平滑的颜色过渡效果。
- 输入：
  - `img`：输入图像，支持ND或NHWC格式，最后一个维度为3（B/G/R通道）
  - `lut_table`：3D LUT表，形状为[N, N, N, 3]，N为LUT边长（最大20）
- 输出：
  - `lut_img`：经过3D LUT变换后的图像，输出dtype始终为float32

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
      <td>img</td>
      <td>输入</td>
      <td>输入图像，最后一个维度为3（B/G/R通道）。</td>
      <td>UINT8、FLOAT</td>
      <td>ND、NHWC</td>
    </tr>
    <tr>
      <td>lut_table</td>
      <td>输入</td>
      <td>3D LUT表，形状为[N, N, N, 3]。</td>
      <td>UINT8、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lut_img</td>
      <td>输出</td>
      <td>经过3D LUT变换后的图像，dtype始终为float32。</td>
      <td>FLOAT</td>
      <td>ND、NHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入图像维度必须为3D或4D，且最后一个维度为3
- LUT表必须为4D，且前三个维度相等（N <= 20），最后一个维度为3
- UINT8和FLOAT输入的元素值域均为[0, 255]
- 支持的dtype组合：
  - uint8 × uint8 → float32
  - float32 × float32 → float32

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---- | ---- | ---- |
| GE图模式 | [test_geir_lut3_d](examples/arch35/test_geir_lut3_d.cpp) | 通过[算子IR](op_graph/lut3_d_proto.h)构图方式调用Lut3D算子，参见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)完成编译和验证。 |
