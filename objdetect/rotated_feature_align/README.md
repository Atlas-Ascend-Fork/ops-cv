# RotatedFeatureAlign

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

- 算子功能：对输入特征图x按旋转框参数bboxes指定的采样点做双线性插值，并将插值结果累加回原特征图，实现旋转目标检测网络（如RotatedRetinaNet、RotatedFasterRCNN）中proposal与特征图的特征对齐。

- 计算公式：

$$
y[n,c,h,w] = x[n,c,h,w] + \sum_{i=0}^{points-1} \mathrm{bilinear}(x[n,:,:,:], py_i, px_i)
$$

其中bboxes的5个平面顺序为(y_ctr,x_ctr,w,h,angle)，平面(y_ctr,x_ctr,w,h)的值均乘以spatial_scale，angle不乘。points=1时仅采样中心点；points=5时采样中心点及4个旋转角点。双线性插值在采样点越界（y<-1.0或y>H或x<-1.0或x>W，严格不等式）时该采样点贡献为0。

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
      <td>x</td>
      <td>输入</td>
      <td>特征图，形状为(N,C,H,W)，公式中的x。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bboxes</td>
      <td>输入</td>
      <td>旋转框参数，形状为(N,5,H,W)，5个平面依次为(y_ctr,x_ctr,w,h,angle)，公式中的bboxes。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>特征对齐结果，与x同形状，公式中的y。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性</td>
      <td>特征图到原图的尺度因子，典型取值(0,1]，如0.125。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>points</td>
      <td>属性</td>
      <td>采样点数，1表示中心点采样，5表示中心点加4角点采样，默认值为1。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- x必须为4维(N,C,H,W)。
- bboxes必须为4维(N,5,H,W)，dim1固定为5。
- x.dim0与bboxes.dim0相等（N一致），x.dim2与bboxes.dim2相等（H一致），x.dim3与bboxes.dim3相等（W一致）。
- x、bboxes、y的数据类型均仅支持float32。
- spatial_scale为特征图到原图的尺度因子，典型取值(0,1]，如0.125。
- points仅允许取1或5，默认值为1。
- bboxes中w、h应为非负值（旋转框宽高），angle为弧度值。
- 采样坐标位于[-1,0]区间时按0处理；y大于等于H-1时y_low与y_high均取H-1（x方向同理取W-1）；采样点越界（y<-1.0或y>H或x<-1.0或x>W，严格不等式）时该采样点贡献为0。

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
    <td><a href="./examples/arch35/test_geir_rotated_feature_align.cpp">test_geir_rotated_feature_align</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
