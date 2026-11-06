# RotatedFeatureAlignGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×   |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：RotatedFeatureAlign（旋转框特征对齐）的反向传播算子，根据输出特征的梯度 dy 和各像素位置的旋转框信息 bboxes，将采样点双线性插值的梯度贡献累加回输入特征的对应位置，得到输入特征的梯度 dx。
- 计算公式：

  $$
  dx_{n,y,x,c} = dy_{n,y,x,c} + \sum_{h=0}^{H-1}\sum_{w=0}^{W-1}\sum_{p \in P} dy_{n,h,w,c} \cdot w_p(n,h,w;y,x)
  $$

  其中 $P$ 为像素 $(n,h,w)$ 处旋转框的采样点集合：`points=1` 时仅含框中心点，`points=5` 时含框中心点与4个旋转角点；$w_p$ 为采样点 $p$ 双线性插值落在像素 $(y,x)$ 上的权重，采样点越界时权重为0。bboxes 中 y/x/w/h 乘以 `spatial_scale` 后参与采样点计算，angle 不缩放。

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
      <td>dy</td>
      <td>输入</td>
      <td>输出特征的梯度。shape必须为[N, H, W, C]的4维Tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bboxes</td>
      <td>输入</td>
      <td>各像素位置的旋转框信息，shape必须为[N, H, W, 5]的4维Tensor。每个框组成为[y, x, w, h, angle]，分别为框中心坐标、宽、高和旋转角度。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>输入特征的梯度。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性</td>
      <td>特征图与原始图像之间的缩放比例，bboxes中y/x/w/h坐标乘以该值后参与采样点计算。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>points</td>
      <td>可选属性</td>
      <td>每个旋转框的采样点数，1表示采样框中心点，5表示采样框中心点及4个旋转角点。默认值为1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- dy的shape必须为[N, H, W, C]的4维Tensor，bboxes的shape必须为[N, H, W, 5]的4维Tensor，且两者的N、H、W维度必须一致。
- dy与bboxes的每一维取值范围为[0, 2147483647]，且元素总数N·H·W·C与N·H·W·5均需在int64表示范围内；超出上述范围的输入视为非法，算子将拒绝执行。
- points属性取值仅支持1或5。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE图模式 | [test_geir_rotated_feature_align_grad](examples/test_geir_rotated_feature_align_grad.cpp) | 通过[算子IR](./op_graph/rotated_feature_align_grad_proto.h)接口方式调用RotatedFeatureAlignGrad算子。 |
