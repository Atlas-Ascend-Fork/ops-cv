# RGB2YUV422

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |     √    |
| <term>Atlas A3系列产品</term>      |    √     |
| <term>Atlas A2系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                       |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：将 RGB 图像转换为 YUV422 (YUYV 打包格式) 色彩空间。

- 计算公式：

uint8输入（BT.601/256定点系数，含+128偏移）：
$$
Y  = (77 \cdot R + 150 \cdot G + 29 \cdot B) / 256
$$
$$
U  = (-43 \cdot R - 85 \cdot G + 128 \cdot B) / 256 + 128
$$
$$
V  = (128 \cdot R - 107 \cdot G - 21 \cdot B) / 256 + 128
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
      <td>rgb</td>
      <td>输入</td>
      <td>RGB 图像张量，shape [..., H, W, 3]。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>yuv</td>
      <td>输出</td>
      <td>YUV422 (YUYV 打包) 张量，shape [..., H, W, 2]。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

输入必须至少为 3 维，通道维大小必须为 3。仅支持 uint8 输入输出，ND 格式。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_rgb2_yuv422](examples/test_geir_rgb2_yuv422.cpp)  | 通过[算子IR](op_graph/rgb2_yuv422_proto.h)构图方式调用RGB2YUV422算子。         |
