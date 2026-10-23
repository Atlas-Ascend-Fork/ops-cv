# BoundingBoxDecode

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

将目标检测回归分支预测的偏移量`deltas`结合感兴趣区域`rois`解码为图像坐标空间的绝对框`bboxes`，用于后续NMS、画框或COCO mAP评估。该算子是`bounding_box_encode`的逆运算，两者共享同一套`(means, stds)`标准化参数与`(x1, y1, x2, y2)`锚框坐标约定，配套使用才能保证训练与推理编码一致。

设输入`rois=(x1, y1, x2, y2)`、偏移量`deltas=(dx', dy', dw', dh')`、`means=(m0,m1,m2,m3)`、`stds=(s0,s1,s2,s3)`、`max_shape=(H, W)`，解码步骤为：

$$
pw = x2 - x1 + 1,\quad pcx = (x1 + x2) \times 0.5
$$

$$
dx = dx' \times s0 + m0,\quad dw = dw' \times s2 + m2
$$

$$
gw = pw \times \exp(dw),\quad gx = pcx + pw \times dx
$$

$$
x1_{out} = \mathrm{clip}(gx - gw \times 0.5 + 0.5,\ 0,\ W - 1)
$$

y维度（y1、y2）同理，裁剪上界为H-1。裁剪上界取最后有效像素`W-1`/`H-1`，输出坐标域为`[0, W-1]`/`[0, H-1]`。fp16输入时中间`exp`及乘加计算在fp32域进行后回cast到fp16。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td>rois</td>
<td>输入</td>
<td>感兴趣区域坐标(x1,y1,x2,y2)，对应公式中的rois。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>deltas</td>
<td>输入</td>
<td>回归偏移量(dx',dy',dw',dh')，对应公式中deltas。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>bboxes</td>
<td>输出</td>
<td>解码后的绝对框(x1,y1,x2,y2)，对应公式中的bboxes。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>means</td>
<td>可选属性</td>
<td>deltas反标准化均值，长度4，默认[0.0, 0.0, 0.0, 0.0]。</td>
<td>FLOAT</td>
<td>-</td>
</tr>
<tr>
<td>stds</td>
<td>可选属性</td>
<td>deltas反标准化标准差，长度4且各元素非0，默认[1.0, 1.0, 1.0, 1.0]。</td>
<td>FLOAT</td>
<td>-</td>
</tr>
<tr>
<td>max_shape</td>
<td>属性</td>
<td>解码框裁剪上限(H, W)，长度2。</td>
<td>INT64</td>
<td>-</td>
</tr>
<tr>
<td>wh_ratio_clip</td>
<td>可选属性</td>
<td>宽高比裁剪阈值，默认0.016。</td>
<td>FLOAT</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- rois与deltas的shape必须完全一致(N, 4)，无广播。
- rois与deltas的dtype必须一致。
- means与stds长度必须为4，stds各元素不能为0。
- max_shape长度必须为2。
- wh_ratio_clip必须大于0，当前不参与核心解码公式，仅做入参校验。
- 输入仅支持ND数据格式。

## 调用说明

<table style="table-layout: fixed; width: 1000px">
<colgroup>
<col style="width: 180px">
<col style="width: 200px">
<col style="width: 620px">
</colgroup>
<thead>
<tr>
<th>调用方式</th>
<th>样例代码</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>GE图模式</td>
<td>-</td>
<td><a href="examples/arch35/test_geir_bounding_box_decode.cpp">test_geir_bounding_box_decode.cpp</a></td>
</tr>
</tbody>
</table>
