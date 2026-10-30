# ResizeBilinearV2Grad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：[ResizeBilinearV2](../resize_bilinear_v2/README.md)的反向传播。

- 计算公式：

  假设$grads$中已知四个点$Q_{11}(x_1, y_1), Q_{12}(x_1, y_2), Q_{21}(x_2, y_1), Q_{22}(x_2, y_2)$。

  假设$grads$在输出图中的位置是 $(h', w')$，它映射回原图的浮点坐标为 $(pos\_h, pos\_w)$。则偏移量定义为：

  $$
  d_h = pos\_h - x_1
  $$

  $$
  d_w = pos\_w - y_1
  $$

  对应的梯度累加公式如下：

  左上点$Q_{11}$：

  $$
  y(N, C, x_1, y_1) += grads(N, C, h', w') \cdot (1 - d_h) \cdot (1 - d_w)
  $$

  右上点$Q_{12}$：

  $$
  y(N, C, x_1, y_2) += grads(N, C, h', w') \cdot (1 - d_h) \cdot d_w
  $$

  左下点$Q_{21}$：

  $$
  y(N, C, x_2, y_1) += grads(N, C, h', w') \cdot d_h \cdot (1 - d_w)
  $$

  右下点$Q_{22}$：

  $$
  y(N, C, x_2, y_2) += grads(N, C, h', w') \cdot d_h \cdot d_w
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
      <td>grads</td>
      <td>输入</td>
      <td>正向resize的输出的梯度Tensor，对应公式中grads。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>original_image</td>
      <td>输入</td>
      <td>正向resize的输入Tensor。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td>为true时对齐输入和输出图像四角像素的中心；默认为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>half_pixel_centers</td>
      <td>可选属性</td>
      <td>为true时按半像素中心坐标进行采样；不能与align_corners同时为true，默认为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>可选属性</td>
      <td>包含H、W两个缩放系数，表示grads与y对应空间轴下标的比例；仅在align_corners为false且对应值大于0时生效，默认值为[0.0, 0.0]。</td>
      <td>LIST_FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>正向resize的输入的梯度Tensor，对应公式中y。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `align_corners`和`half_pixel_centers`不能同时为true。
- `scales`设置时应提供H、W两个缩放系数；默认值为`[0.0, 0.0]`。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_bilinear_2d_backward](examples/test_aclnn_upsample_bilinear_2d_backward.cpp) | 通过[aclnnUpsampleBilinear2dBackward](docs/aclnnUpsampleBilinear2dBackward.md)接口方式调用ResizeBilinearV2Grad算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/resize_bilinear_v2_grad_proto.h)构图方式调用ResizeBilinearV2Grad算子。         |
