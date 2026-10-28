# ResizeNearestNeighborV2Grad

## 产品支持情况

| 产品 | 是否支持 |
| :---- | :----: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：[ResizeNearestNeighborV2](../resize_nearest_neighbor_v2/README.md)的反向传播，根据最近邻插值的映射关系，将输出梯度散射回输入空间并累加。
- 计算公式：

  正向ResizeNearestNeighborV2中，每个输出像素值复制于某个输入像素。反向传播时，输入像素的梯度为所有映射到该位置的输出像素梯度的累加和：

  $$y(N, C, h, w) += \text{grads}(N, C, h', w')$$

  其中 $(h', w')$ 为输出坐标，通过最近邻映射回输入坐标：

  $$h = \min(\lfloor h' \times \text{scaleH} \rfloor, H_{in} - 1)$$

  $$w = \min(\lfloor w' \times \text{scaleW} \rfloor, W_{in} - 1)$$

  $$\text{scaleH} = \frac{H_{in}}{H_{out}}, \quad \text{scaleW} = \frac{W_{in}}{W_{out}}$$

  当属性取值不同时，映射公式变化如下：

  - align\_corners=True，half\_pixel\_centers=False：$h = \min(\text{round}(h' \times \text{scaleH}), H_{in} - 1)$
  - align\_corners=False，half\_pixel\_centers=True：$h = \min(\lfloor (h' + 0.5) \times \text{scaleH} \rfloor, H_{in} - 1)$
  - align\_corners=False，half\_pixel\_centers=False：$h = \min(\lfloor h' \times \text{scaleH} \rfloor, H_{in} - 1)$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 223px">
  <col style="width: 101px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>grads</td>
      <td>输入</td>
      <td>正向ResizeNearestNeighborV2输出的梯度Tensor，对应公式中`grads`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>原始图像的高和宽，即正向输入的空间尺寸，用于确定输出梯度Tensor的H和W维度。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>正向ResizeNearestNeighborV2输入的梯度Tensor，对应公式中`y`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
  </tbody>
  </table>

  **属性说明：**

  | 属性名 | 类型 | 默认值 | 描述 |
  | :---- | :---- | :---- | :---- |
  | align_corners | Bool | false | 决定是否对齐输入和输出图像的角点像素。为True时角点像素对齐。 |
  | half_pixel_centers | Bool | false | 决定插值时像素中心位置。为True时像素中心在半像素坐标处。 |
  | scales | ListFloat | {0.0f, 0.0f} | 空间尺寸乘数，可选参数。 |

## 约束说明

- 仅支持4维Tensor（NCHW或NHWC格式）

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :---- | :---- | :---- |
| aclnn接口 | [test_aclnn_upsample_nearest2d_backward](examples/test_aclnn_upsample_nearest2d_backward.cpp) | 通过[aclnnUpsampleNearest2dBackward](../upsample_nearest2d_grad/docs/aclnnUpsampleNearest2dBackward.md)接口方式调用ResizeNearestNeighborV2Grad算子。 |
| aclnn接口 | [test_aclnn_upsample_nearest1d_backward](examples/test_aclnn_upsample_nearest1d_backward.cpp) | 通过[aclnnUpsampleNearest1dBackward](../upsample_nearest2d_grad/docs/aclnnUpsampleNearest1dBackward.md)接口方式调用ResizeNearestNeighborV2Grad算子。 |
| 图模式 | [test_geir_resize_nearest_neighbor_v2_grad](examples/test_geir_resize_nearest_neighbor_v2_grad.cpp) | 通过[算子IR](op_graph/resize_nearest_neighbor_v2_grad_proto.h)构图方式调用ResizeNearestNeighborV2Grad算子，验证图模式下输出dtype跟随输入grads。 |
