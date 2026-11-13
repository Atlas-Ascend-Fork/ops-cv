# RoiAlignRotated

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×   |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

注：算子注册的数据格式为ND。维度序随芯片不同：<term>Ascend 950PR&950DT系列产品</term>上x为NCHW维度序(N,C,H,W)，图模式下x也支持显式NCHW格式，拒绝NHWC格式；其余支持的芯片上x为NHWC维度序(N,H,W,C)。详见"约束说明"。

## 功能说明

- 算子功能：旋转RoI对齐池化。对每个旋转矩形RoI按pooled_h×pooled_w划分bin，bin内采样点经旋转矩阵映射到特征图坐标系后做双线性插值，再取平均，输出固定尺寸的RoI特征。

- 计算公式：

  对第r个RoI，rois[r]=(batch_idx, roi_cx, roi_cy, roi_w, roi_h, θ)，clockwise=true时θ取负。先做预处理（先缩放后偏移，不做取整），宽高同样先乘spatial_scale、缩放后记为w、h，后续的bin尺寸、采样坐标均基于w、h：

  $$
  center_x = roi_cx \times spatial\_scale - offset,\quad center_y = roi_cy \times spatial\_scale - offset
  $$

  $$
  w = roi_w \times spatial\_scale,\quad h = roi_h \times spatial\_scale
  $$

  $$
  offset = \begin{cases} 0.5, & aligned=true \\ 0, & aligned=false \end{cases}
  $$

  bin尺寸与采样数（grid_w同理）：

  $$
  bin_h = \frac{h}{pooled_h},\quad grid_h = \begin{cases} sampling\_ratio, & sampling\_ratio > 0 \\ \lceil bin_h \rceil, & sampling\_ratio = 0 \end{cases},\quad count = \max(grid_h \times grid_w,\ 1)
  $$

  bin内采样点在RoI局部坐标系中的坐标（iy∈[0,grid_h)，ix∈[0,grid_w)，xx同理）：

  $$
  yy = -\frac{h}{2} + ph \cdot bin_h + \frac{(iy+0.5) \times bin_h}{grid_h}
  $$

  旋转到特征图坐标系：

  $$
  \begin{bmatrix} sx \\ sy \end{bmatrix} =
  \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}
  \begin{bmatrix} xx \\ yy \end{bmatrix} +
  \begin{bmatrix} center_x \\ center_y \end{bmatrix}
  $$

  在特征图上做双线性插值（越界采样点贡献为0），bin内取平均：

  $$
  y[r,c,ph,pw] = \frac{1}{count} \sum_{iy=0}^{grid_h-1} \sum_{ix=0}^{grid_w-1} f(sx,\ sy)
  $$

  以下为内部采样点的插值公式，边界采样点按补充说明处理。设$x_l=\lfloor sx \rfloor$，$y_l=\lfloor sy \rfloor$，$lx=sx-x_l$，$ly=sy-y_l$：

  $$
  f(sx,\ sy) = (1-ly)(1-lx)\,x[n,c,y_l,x_l] + (1-ly)\,lx\,x[n,c,y_l,x_l+1] + ly\,(1-lx)\,x[n,c,y_l+1,x_l] + ly\,lx\,x[n,c,y_l+1,x_l+1]
  $$

  补充说明：
  - 先检查采样坐标：越界（sx<-1、sx>W、sy<-1、sy>H）采样点贡献为0；其余采样点的负坐标clamp到0。
  - 两个坐标轴分别处理上边界：$x_l \ge W-1$时，x轴的低、高索引均取W-1，sx重置为W-1；y轴同理。仅一个轴触发边界处理时，另一个轴仍按其插值权重计算；两个轴均触发时，退化为单点采样。
  - aligned=false时，w、h在缩放后先取下限1，再计算bin尺寸与采样坐标。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 150px">
  <col style="width: 130px">
  <col style="width: 480px">
  <col style="width: 130px">
  <col style="width: 115px">
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
      <td>输入特征图，4维。<term>Ascend 950PR&950DT系列产品</term>上图模式支持ND或NCHW格式，均按NCHW维度序解释，shape为(N,C,H,W)；其余支持的芯片上为ND格式、NHWC维度序，shape为(N,H,W,C)。</td>
      <td>FLOAT32</td>
      <td>ND；950图模式另支持NCHW</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>旋转RoI列表，2维，每个RoI为(batch_idx, center_x, center_y, w, h, angle)，坐标为原图坐标系，angle单位为弧度。<term>Ascend 950PR&950DT系列产品</term>上行主序shape为(R,6)；其余支持的芯片上转置shape为(6,R)。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出特征图，注册的数据格式为ND。<term>Ascend 950PR&950DT系列产品</term>上shape为(R,C,pooled_h,pooled_w)；其余支持的芯片上shape为(R,pooled_h,pooled_w,C)。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pooled_h</td>
      <td>属性</td>
      <td>输出在H维度上的池化尺寸，必须大于0；<term>Ascend 950PR&950DT系列产品</term>上还须不超过INT32_MAX。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_w</td>
      <td>属性</td>
      <td>输出在W维度上的池化尺寸，必须大于0；<term>Ascend 950PR&950DT系列产品</term>上还须不超过INT32_MAX。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性</td>
      <td>RoI坐标（原图坐标）到输入特征图坐标的缩放因子，必须大于0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sampling_ratio</td>
      <td>可选属性</td>
      <td>每个bin内每个维度的采样点数，0表示按ceil(bin_size)密集采样，必须大于等于0。默认值为0；<term>Ascend 950PR&950DT系列产品</term>上还须不超过INT32_MAX。具体校验规则见"约束说明"。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>aligned</td>
      <td>可选属性</td>
      <td>true时RoI中心偏移0.5像素；false时offset为0且roi_w/roi_h下限为1。默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clockwise</td>
      <td>可选属性</td>
      <td>true时RoI角度按顺时针方向解释（θ取负）。默认值为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 本算子x、rois、y注册的数据格式为ND，维度序由芯片决定：
  - <term>Ascend 950PR&950DT系列产品</term>：图模式下x支持ND或显式NCHW格式，均按NCHW维度序(N,C,H,W)解释；x为NHWC或其他格式时拒绝执行。rois为ND格式、shape为(R,6)，y的shape为(R,C,pooled_h,pooled_w)。
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：x为NHWC维度序(N,H,W,C)，rois为(6,R)，y为(R,pooled_h,pooled_w,C)。
- 输入x的shape必须为4维；<term>Ascend 950PR&950DT系列产品</term>上N、C、H、W均不得超过INT32_MAX。
- 输入rois的shape必须为2维；<term>Ascend 950PR&950DT系列产品</term>上R=0时输出空tensor(0,C,pooled_h,pooled_w)，其余支持的芯片要求R≥1。
- rois取值约束：batch_idx∈[0,N)，N为输入x的第0维大小；angle∈[0,π)，单位为弧度；各元素不允许为NaN/Inf。其中angle等取值由调用方保证；batch_idx越界时算子不报错，该RoI按空RoI处理输出0（不进行越界读取）。
- 属性约束：pooled_h>0、pooled_w>0，由算子校验；spatial_scale>0、sampling_ratio>=0由调用方保证，算子不检查这两项取值条件。
- <term>Ascend 950PR&950DT系列产品</term>上还会校验整数范围：pooled_h、pooled_w不超过INT32_MAX，sampling_ratio位于[INT32_MIN,INT32_MAX]；超出范围时拒绝执行，避免写入32位参数时截断。这不替代调用方对sampling_ratio>=0的保证。INT32_MIN=-2147483648，INT32_MAX=2147483647。
- 仅支持FLOAT32数据类型，x、rois、y的数据类型必须一致；仅支持连续布局。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_roi_align_rotated](examples/arch35/test_geir_roi_align_rotated.cpp) | 通过[算子IR](./op_graph/roi_align_rotated_proto.h)接口方式调用RoiAlignRotated算子（<term>Ascend 950PR&950DT系列产品</term>：x支持ND或NCHW格式，按NCHW维度序解释，含NHWC格式拒绝负向用例）。 |
| 图模式 | [test_geir_roi_align_rotated](examples/test_geir_roi_align_rotated.cpp) | 通过[算子IR](./op_graph/roi_align_rotated_proto.h)接口方式调用RoiAlignRotated算子（<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：ND，x为NHWC维度序，rois为(6,R)转置）。 |
