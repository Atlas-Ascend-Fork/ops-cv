# IouV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  Ascend 950PR&950DT系列产品   |     √    |
|  Atlas A3系列产品   |     √    |
|  Atlas A2系列产品     |     √    |
|  Atlas 200I/500 A2推理产品    |     ×    |
|  Atlas推理系列产品    |     √   |
|  Atlas训练系列产品    |     ×    |
|  Kirin X90处理器系列产品 | √ |
|  Kirin 9030处理器系列产品 | √ |

## 功能说明

- 算子功能：计算两个矩形的重叠面积占两个矩阵总面积的比例，设预测框的左上角坐标为（X1，Y1），右下角坐标为（X2，Y2），真实框的左上角坐标为（X3，Y3），右下角坐标为（X4，Y4）。

- 计算公式：

  $$
  IOU = \frac {Area_3} {Area_1 + Area_2 - Area_3} \\
  IOF = \frac {Area_3} {Area_2}
  $$

  其中，Area_1为bBox的面积，Area_2为gtBox的面积，Area_3为两者重叠部分面积，x和y的定义见参数说明。

  $$
  Area_1 = (X_2 - X_1)(Y_2 - Y_1) \\
  Area_2 = (X_4 - X_3)(Y_4 - Y_3) \\
  Area_3 = max( min(X_2, X_4) - max(X_1, X_3), 0 ) * max( min(Y_2, Y_4) - max(Y_1, Y_3), 0 )
  $$

  IOF定义为前景交叉比，计算公式：

  $$
  IOF=\frac {Area_3} {前景面积}
  $$

  对于库上算子，前景面积为Area2；对于竞品mmcv，前景面积为Area1。

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
      <td>bboxes</td>
      <td>输入</td>
      <td>预测矩形框。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gtboxes</td>
      <td>输入</td>
      <td>真值矩形框。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>防止除零，计算面积时，长和宽都会加上eps。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>aligned</td>
      <td>属性</td>
      <td>用于标识两个输入的shape是否相同。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>属性</td>
      <td>用于选择计算方式"iou"或"iof"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>overlap</td>
      <td>输出</td>
      <td>根据两个输入计算得到的交并比/前景交叉比。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

- 输入shape限制：输入shape为(N, 4)的二维张量，第二维的四个值(X1, Y1, X2, Y2)需满足X1 < X2, Y1 < Y2
- `bboxes`与`gtboxes`的数据类型必须相同，输出`overlap`的数据类型与`bboxes`相同。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_iou_v2](examples/test_aclnn_iou_v2.cpp) | 通过[aclnnIou](docs/aclnnIou.md)接口方式调用IouV2算子。 |
