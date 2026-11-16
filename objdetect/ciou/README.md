# CIoU

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    ×     |
| <term>Atlas A2系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：用于边界框回归的损失函数，在IoU的基础上同时考虑了中心点距离、宽高比和重叠面积，以更全面地衡量预测框与真实框之间的差异。
- 计算公式：

$$
CIoU = IoU - \frac{\rho^2(b^p, b^g)}{c^2} - \alpha v \\
v = \frac{4}{\pi^2}(arctan(\frac{w^g}{h^g}) - arctan(\frac{w^p}{h^p}))^2 \\
\alpha = \frac{v}{1 - IoU + v} \\
$$

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
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gtboxes</td>
      <td>输入</td>
      <td>真值矩形框。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>trans</td>
      <td>属性</td>
      <td>用于指定矩形框的格式。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_cross</td>
      <td>属性</td>
      <td>用于指定bBoxes与gtBoxes之间是否进行交叉运算。</td>
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
      <td>atan_sub_flag</td>
      <td>属性</td>
      <td>用于指定是否输出atan_sub。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>overlap</td>
      <td>输出</td>
      <td>根据两个输入计算得到的CIoU值，即mode为"iou"时的交并比或mode为"iof"时的前景交叉比，减去中心点距离与宽高比两项惩罚项后的结果，可能小于0。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>atan_sub</td>
      <td>输出</td>
      <td>计算过程中两个arctan的差值。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 若输入格式为[x0, y0, x1, y1]，(x0, y0)和(x1, y1)分别表示矩形框的左上角和右下角，需满足x1 > x0, y1 > y0。
- M和N需要一致。
- `is_cross`目前仅支持`false`。
- `atan_sub_flag`目前仅支持`true`。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_ciou](examples/test_aclnn_ciou.cpp) | 通过[aclnnCIoU](docs/aclnnCIoU.md)接口方式调用CIoU算子。 |
