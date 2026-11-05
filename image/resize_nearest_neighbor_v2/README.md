# ResizeNearestNeighborV2

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

- 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。如果输入shape为(N, C, H, W)，则输出shape为(N, C, size[0], size[1])。
- 计算公式：

  $$
  y(N, C, H, W) = x(N, C, min(floor(H * scaleH),  H-1), min(floor(W * scaleW),  W-1))
  $$

  $$
  scaleH = x\_H / size[0]
  $$

  $$
  scaleW = x\_W / size[1]
  $$

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
      <td>x</td>
      <td>输入</td>
      <td>输入图像的四维Tensor，对应公式中`x`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>输出图像的高和宽。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>最近邻插值后的图像，对应公式中的`y`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
  </tbody>
  </table>

  **属性说明：**

  | 属性名 | 类型 | 默认值 | 描述 |
  | :---- | :---- | :---- | :---- |
  | align_corners | Bool | false | 决定是否对齐输入和输出图像的角点像素。为True时角点像素对齐，源坐标按$(H_{in}-1)/(size[0]-1)$映射；为False时按比例缩放，不严格对齐角点。 |
  | half_pixel_centers | Bool | false | 决定插值时的像素中心位置。为True时像素中心位于半像素坐标处，源坐标按$(h+0.5) \times scaleH$映射；为False时像素中心位于整数坐标点。 |
  | scales | ListFloat | {0.0f, 0.0f} | 空间尺寸乘数，可选参数。仅支持默认值{0.0f, 0.0f}（表示由`size`与输入空间尺寸推导缩放比例），非默认取值不受支持。 |

## 约束说明

- 仅支持4维Tensor，数据格式为NCHW或NHWC。
- `align_corners`与`half_pixel_centers`不支持同时为true：该组合无对应的TensorFlow ResizeNearestNeighbor语义，Tiling阶段会拒绝并返回失败。合法组合为(false, false)、(false, true)、(true, false)。
- `size`必须为1维且恰含2个INT32元素（分别为输出的高、宽）。该输入为值依赖（ValueDepend）输入：当`size`为常量时参与InferShape推导输出尺寸。
- 输入`x`与输出`y`均不支持空Tensor：任一元素个数为0时Tiling阶段返回失败（报错信息为`Input x and output y do not support empty tensors`）。
- `scales`仅支持默认值{0.0f, 0.0f}，非默认取值不受支持。
- 输出`y`的N轴、C轴必须与输入`x`保持一致，数据类型和数据格式也须与`x`一致。
- 动态Shape场景下输入Shape支持动态标记，InferShape按如下规则传播：
  - `size`为非常量（取值未知）时，输出`y`的H、W维度推导为-1（未知维度），N、C维度仍沿用`x`的对应取值。
  - 输入`x`的Shape为-2（未知Rank）时，该输入被接受，输出推导为\[-1, -1, size\[0\], size\[1\]\]；若`size`同时为非常量，则输出H、W亦为-1。
- 数据类型的两层口径：上表所列FLOAT16、FLOAT32、BFLOAT16为本算子AI Core实现（见op_host算子注册）所支持的数据类型，也是ACLNN接口在<term>Ascend 950PR&950DT系列产品</term>上的可用范围；[算子IR](op_graph/resize_nearest_neighbor_v2_proto.h)声明的数据类型为与TensorFlow ResizeNearestNeighbor对齐的更大集合（含INT8、UINT8、INT16、UINT16、INT32、INT64、DOUBLE），图模式下这些类型不走AI Core，由AICPU实现承载。按本README调用时请以上表为准。

## 调用说明

| 调用方式   | 样例代码           | 说明                                                                      |
| -------------- | --------------------------- |-------------------------------------------------------------------------|
| aclnn  | [test_aclnn_upsample_nearest1d.cpp](examples/test_aclnn_upsample_nearest1d.cpp) | 通过[aclnnUpsampleNearest1d](docs/aclnnUpsampleNearest1d.md)接口方式调用ResizeNearestNeighborV2算子，对3维(NCL)输入在L维度做最近邻上采样。 |
| aclnn  | [test_aclnn_upsample_nearest2d.cpp](examples/test_aclnn_upsample_nearest2d.cpp) | 通过[aclnnUpsampleNearest2d](docs/aclnnUpsampleNearest2d.md)接口方式调用ResizeNearestNeighborV2算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/resize_nearest_neighbor_v2_proto.h)构图方式调用ResizeNearestNeighborV2算子。         |
