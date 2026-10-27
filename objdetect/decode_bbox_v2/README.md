# DecodeBboxV2

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

将目标检测回归分支预测的偏移量`boxes`结合锚框`anchors`解码为图像坐标空间的绝对框`y`。该算子支持`(N,4)`和`(4,N)`两种布局，由`reversed_box`属性决定。

设锚框`anchors=(ymin, xmin, ymax, xmax)`、偏移量`boxes=(ty, tx, th, tw)`、`scales=(sy, sx, sh, sw)`，解码步骤为：

$$
ah = ymax - ymin,\quad aw = xmax - xmin
$$

$$
tys = ty / sy,\quad ths = th / sh
$$

$$
h = \exp(\min(ths, decode\_clip)) \times ah \quad (decode\_clip > 0)
$$

$$
cy = tys \times ah + ymin + ah \times 0.5
$$

$$
ymin_{out} = cy - h \times 0.5,\quad ymax_{out} = cy + h \times 0.5
$$

x维度同理。fp16输入时中间`exp`及乘加计算在fp32域进行后回cast到fp16。

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
<td>boxes</td>
<td>输入</td>
<td>回归偏移量(ty,tx,th,tw)。shape由reversed_box决定：false时为(N,4)，true时为(4,N)。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>anchors</td>
<td>输入</td>
<td>锚框坐标(ymin,xmin,ymax,xmax)。shape与dtype均与boxes相同，同为(N,4)或(4,N)。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>解码后的绝对框(ymin,xmin,ymax,xmax)。shape与dtype均与boxes相同，同为(N,4)或(4,N)。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>scales</td>
<td>可选属性</td>
<td>缩放因子，长度4，各元素必须非零，默认[1.0, 1.0, 1.0, 1.0]。</td>
<td>FLOAT</td>
<td>-</td>
</tr>
<tr>
<td>decode_clip</td>
<td>可选属性</td>
<td>exp裁剪阈值，0.0表示不裁剪，默认0.0。</td>
<td>FLOAT</td>
<td>-</td>
</tr>
<tr>
<td>reversed_box</td>
<td>可选属性</td>
<td>布局标志，false=(N,4)，true=(4,N)，默认false。</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- boxes, anchors, y的shape以及dtype必须完全一致。
- scales长度必须为4，且各元素必须非零。
- decode_clip范围为[0.0, 10.0]。
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
<td><a href="examples/arch35/test_geir_decode_bbox_v2.cpp">test_geir_decode_bbox_v2.cpp</a></td>
<td>通过<a href="op_graph/decode_bbox_v2_proto.h">算子IR</a>构图方式调用DecodeBboxV2算子</td>
</tr>
</tbody>
</table>
