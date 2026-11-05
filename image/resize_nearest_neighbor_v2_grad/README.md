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

  其中 $H_{in}$、$W_{in}$ 为原始图像的高和宽（由输入`size`给定，即输出`y`的空间尺寸），$H_{out}$、$W_{out}$ 为输入`grads`的高和宽。有效缩放因子 scaleH、scaleW 由属性`align_corners`与`scales`共同决定：

  - align\_corners=False 且 scales 的对应分量大于0时，取该分量的倒数：

    $$\text{scaleH} = \frac{1}{\text{scales}[0]}, \quad \text{scaleW} = \frac{1}{\text{scales}[1]}$$

  - align\_corners=False 且 scales 的对应分量小于等于0（缺省）时，由空间尺寸推导：

    $$\text{scaleH} = \frac{H_{in}}{H_{out}}, \quad \text{scaleW} = \frac{W_{in}}{W_{out}}$$

  - align\_corners=True 时忽略`scales`，按角点对齐推导（分母为1时取0）：

    $$\text{scaleH} = \frac{H_{in} - 1}{H_{out} - 1}, \quad \text{scaleW} = \frac{W_{in} - 1}{W_{out} - 1}$$

  当属性取值不同时，映射公式变化如下（各分支使用上述对应的 scaleH、scaleW）：

  - align\_corners=True，half\_pixel\_centers=False：$h = \min(\text{round}(h' \times \text{scaleH}), H_{in} - 1)$
  - align\_corners=False，half\_pixel\_centers=True：$h = \min(\lfloor (h' + 0.5) \times \text{scaleH} \rfloor, H_{in} - 1)$
  - align\_corners=False，half\_pixel\_centers=False：$h = \min(\lfloor h' \times \text{scaleH} \rfloor, H_{in} - 1)$
  - align\_corners=True，half\_pixel\_centers=True：非法组合，不支持，详见[约束说明](#约束说明)。

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
      <td>原始图像的高和宽，即正向输入的空间尺寸，用于确定输出梯度Tensor的H和W维度。为1维Tensor，固定2个元素，依次为orig_height、orig_width。</td>
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
  | align_corners | Bool | false | 决定是否对齐输入和输出图像的角点像素。为True时角点像素对齐，有效缩放因子按$(H_{in}-1)/(H_{out}-1)$推导并忽略`scales`。不可与half_pixel_centers同时为True。 |
  | half_pixel_centers | Bool | false | 决定插值时像素中心位置。为True时像素中心在半像素坐标处。不可与align_corners同时为True。 |
  | scales | ListFloat | {0.0f, 0.0f} | 空间尺寸乘数，可选参数，固定2个元素，依次对应H、W方向。元素取值大于0时，该方向的有效缩放因子取其倒数（即$1/\text{scales}[i]$）；小于等于0时该方向由空间尺寸推导。align_corners为True时本属性被忽略。 |

## 约束说明

- 仅支持4维Tensor（NCHW或NHWC格式），且`grads`与`y`的数据类型、数据格式必须一致。
- `align_corners`与`half_pixel_centers`不可同时为True，该组合为非法组合，在Host校验阶段直接返回失败。
- `size`固定为1维、2元素的INT32 Tensor，其两个元素值须与输出`y`的H、W维度一致；`grads`与`y`的N、C维度大小必须相等。
- 动态Shape与动态Rank：形状推导（InferShape）支持`size`为非常量的场景，此时输出`y`的H、W置为`-1`；`grads`为未知Rank（以`-2`表示）时，输出`y`为4个`-1`。Tiling消费的是运行期已具体化的shape，未具体化的`-1`/`-2`维/秩不进入Tiling校验，相应约束在shape具体化后才生效。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :---- | :---- | :---- |
| aclnn接口 | [test_aclnn_upsample_nearest2d_backward](examples/test_aclnn_upsample_nearest2d_backward.cpp) | 通过[aclnnUpsampleNearest2dBackward](../upsample_nearest2d_grad/docs/aclnnUpsampleNearest2dBackward.md)接口方式调用ResizeNearestNeighborV2Grad算子。 |
| aclnn接口 | [test_aclnn_upsample_nearest1d_backward](examples/test_aclnn_upsample_nearest1d_backward.cpp) | 通过[aclnnUpsampleNearest1dBackward](../upsample_nearest2d_grad/docs/aclnnUpsampleNearest1dBackward.md)接口方式调用ResizeNearestNeighborV2Grad算子。 |
| 图模式 | [test_geir_resize_nearest_neighbor_v2_grad](examples/test_geir_resize_nearest_neighbor_v2_grad.cpp) | 通过[算子IR](op_graph/resize_nearest_neighbor_v2_grad_proto.h)构图方式调用ResizeNearestNeighborV2Grad算子，验证图模式下输出dtype跟随输入grads。 |

> [!NOTE]
>
> 本算子不交付自有的aclnn接口：[CMakeLists.txt](CMakeLists.txt)中声明`ACLNNTYPE aclnn_exclude`，安装包的op_api头文件为空且`libcust_opapi.so`中无aclnn导出符号。
> 上表中`aclnnUpsampleNearest2dBackward`、`aclnnUpsampleNearest1dBackward`为CANN内置接口，其Host侧实现位于[upsample_nearest2d_grad](../upsample_nearest2d_grad)；在<term>Ascend 950PR&950DT系列产品</term>上，该接口下发后由本算子承接Kernel执行。
