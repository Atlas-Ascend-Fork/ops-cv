# SpatialTransformer

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

算子功能：Spatial Transformer Network (STN) 算子用于对输入张量进行仿射变换。该算子通过变换矩阵theta对输入图像x进行空间变换，输出变换后的图像y。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>x</td>
      <td>输入</td>
      <td>输入张量。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64<br>
          FLOAT16、FLOAT、DOUBLE</td>
      <td>NCHW、NC1HWC0</td>
    </tr>
    <tr>
      <td>theta</td>
      <td>可选输入</td>
      <td>变换矩阵，包含仿射变换参数；当use_default_theta的6项均为1且default_theta至少提供6个系数时可省略。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64<br>
          FLOAT16、FLOAT、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>变换后的输出张量。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64<br>
          FLOAT16、FLOAT、DOUBLE</td>
      <td>NCHW、NC1HWC0</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>指定输出的高度和宽度，包含2个整数。默认为 [-1, -1]，表示使用输入尺寸。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>default_theta</td>
      <td>属性</td>
      <td>默认的仿射变换参数，当use_default_theta为true时使用。省略theta时至少提供6个系数。默认为空列表。</td>
      <td>ListFloat</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>属性</td>
      <td>如果为true，则输入和输出张量的4个角像素中心对齐，保留角像素的值。默认为false。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_default_theta</td>
      <td>属性</td>
      <td>指定哪些theta参数从default_theta使用。1表示使用默认值，0表示使用输入theta。默认为空列表。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入张量x的格式必须为NCHW或NC1HWC0。
- 输出张量y的格式必须与输入张量x的格式一致。
- 变换矩阵theta的形状必须为 [batch, 2, 3] 或 [2, 3]。
- 当use_default_theta为空列表时，使用输入theta进行变换。
- 当use_default_theta不为空时，对应位置为1的参数使用default_theta中的值，为0的参数使用输入theta中的值。
- 当use_default_theta的6项均为1且default_theta至少提供6个系数时，无需提供theta输入。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_spatial_transformer](./examples/test_geir_spatial_transformer.cpp)   | 通过[算子IR](./op_graph/spatial_transformer_proto.h)构图方式调用SpatialTransformer算子。 |
