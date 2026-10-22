# CropAndResize

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：从输入图像中提取多个裁剪区域，并将它们统一调整为指定大小，支持双线性插值和最近邻插值。
- 计算公式：

  对于输入x的shape为(batch, image_height, image_width, depth)，第$i$个框（$i \in [0, num\_boxes)$），令$b = box\_index[i]$，$(y_1, x_1, y_2, x_2) = boxes[i]$，裁剪后图像的长宽crop_size的shape为(crop_height, crop_width)，输出$y$在位置$(i, p_y, p_x, d)$的值按以下公式计算。

  - method = bilinear时：

    1. 计算缩放比例$height\_scale$和$width\_scale$：

       $$
       height\_scale = \begin{cases} (y_2 - y_1) \times (image\_height - 1) / (crop\_height - 1), & crop\_height > 1 \\ 0, & crop\_height = 1 \end{cases}
       $$

       $$
       width\_scale = \begin{cases} (x_2 - x_1) \times (image\_width - 1) / (crop\_width - 1), & crop\_width > 1 \\ 0, & crop\_width = 1 \end{cases}
       $$

    2. 将输出坐标$(p_y, p_x)$映射到输入图像坐标$(in\_y, in\_x)$：

       $$
       in\_y = \begin{cases} y_1 \times (image\_height - 1) + p_y \times height\_scale, & crop\_height > 1 \\ 0.5 \times (y_1 + y_2) \times (image\_height - 1), &     crop\_height = 1 \end{cases}
       $$

       $$
       in\_x = \begin{cases} x_1 \times (image\_width - 1) + p_x \times width\_scale, & crop\_width > 1 \\ 0.5 \times (x_1 + x_2) \times (image\_width - 1), & crop\_width =     1 \end{cases}
       $$

       若$in\_y < 0$或$in\_y > image\_height - 1$或$in\_x < 0$ 或 $in\_x > image\_width - 1$，则：

       $$
       y(i, p_y, p_x, d) = extrapolation\_value
       $$

       否则，进行双线性插值。令$top = \lfloor in\_y \rfloor$，$bottom = \lceil in\_y \rceil$，$left = \lfloor in\_x \rfloor$，$right = \lceil in\_x \rceil$，$y\_ratio = in\_y - top$，$x\_ratio = in\_x - left$，则：

       $$
       \begin{aligned}
       y(i, p_y, p_x, d) =\; & (1 - y\_ratio)(1 - x\_ratio) \cdot x(b, top, left, d) \\
       +\; & (1 - y\_ratio) \cdot x\_ratio \cdot x(b, top, right, d) \\
       +\; & y\_ratio \cdot (1 - x\_ratio) \cdot x(b, bottom, left, d) \\
       +\; & y\_ratio \cdot x\_ratio \cdot x(b, bottom, right, d)
       \end{aligned}
       $$

  - method = nearest时：

    $height\_scale$、$width\_scale$、$in\_y$、$in\_x$ 的计算方式与bilinear方法相同。

    若$in\_y$或$in\_x$越界（条件同bilinear），则：

    $$
    y(i, p_y, p_x, d) = extrapolation\_value
    $$

    否则，取最近邻像素：

    $$
    y(i, p_y, p_x, d) = x(b, round(in\_y), round(in\_x), d)
    $$

    其中$round(\cdot)$为四舍五入取整。

  - method = bilinear_v2时：

    1. 首先将归一化框坐标映射为输入图像上的像素坐标，计算裁剪窗口。令$map(c, n) = \lfloor c \times n \rfloor$（当$c = 1$时取$n - 1$），$y_{1o} = map(y_1, image\_height)$，$x_{1o} = map(x_1, image\_width)$，$y_{2o} = map(y_2, image\_height)$，$x_{2o} = map(x_2, image\_width)$，则裁剪窗口的高$h$和宽$w$为：

       $$
       h = \max(y_{2o} - y_{1o} + 1,\; 1), \quad w = \max(x_{2o} - x_{1o} + 1,\; 1)
       $$

    2. 对于输出位置$(p_y, p_x)$，计算其在裁剪窗口中的浮点索引$r_y$和$r_x$：

       $$
       r_y = (p_y + 0.5) \times h / crop\_height - 0.5
       $$

       $$
       r_x = (p_x + 0.5) \times w / crop\_width - 0.5
       $$

    3. 计算插值下标和权重。令$clamp(v, lo, hi) = \max(\min(v, hi), lo)$，则：

       $$
       lower\_y = clamp(\lfloor r_y \rfloor, 0, h - 1), \quad upper\_y = clamp(\lceil r_y \rceil, 0, h - 1)
       $$

       $$
       lower\_x = clamp(\lfloor r_x \rfloor, 0, w - 1), \quad upper\_x = clamp(\lceil r_x \rceil, 0, w - 1)
       $$

       $$
       y\_ratio = r_y - lower\_y, \quad x\_ratio = r_x - lower\_x
       $$

       当归一化框坐标超出`[0, 1]`且对应采样位置超出输入图像边界时，输出使用`extrapolation_value`；框坐标在`[0, 1]`内时，窗口边缘的采样下标按上述公式钳制。

    4. 双线性插值：

       $$
       \begin{aligned}
       y(i, p_y, p_x, d) =\; & (1 - y\_ratio)(1 - x\_ratio) \cdot x(b, y_{1o} + lower\_y, x_{1o} + lower\_x, d) \\
       +\; & (1 - y\_ratio) \cdot x\_ratio \cdot x(b, y_{1o} + lower\_y, x_{1o} + upper\_x, d) \\
       +\; & y\_ratio \cdot (1 - x\_ratio) \cdot x(b, y_{1o} + upper\_y, x_{1o} + lower\_x, d) \\
       +\; & y\_ratio \cdot x\_ratio \cdot x(b, y_{1o} + upper\_y, x_{1o} + upper\_x, d)
       \end{aligned}
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
      <td>x</td>
      <td>输入</td>
      <td>输入为四维的Tensor，其shape为(batch, image_height, image_width, depth)，其中image_height和image_width必须是正数。对应计算公式描述中的`输入x`。</td>
      <td>UINT8、UINT16、INT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>boxes</td>
      <td>输入</td>
      <td>二维Tensor，其shape为(num_boxes, 4)。第i行指定了box_index[i]在图像中一个框的位置，并且使用归一化坐标[y1, x1, y2, x2]进行表示。对于bilinear和nearest方法，归一化y坐标通过y*(image_height - 1)映射到图像坐标系；bilinear_v2按其专属公式映射。宽度维度的处理方式相似，坐标超出[0, 1]范围的对应输出使用extrapolation_value。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>box_index</td>
      <td>输入</td>
      <td>boxes的索引，其shape为(num_boxes)的一维tensor，取值范围[0, batch)，box_index[i]的值指定了第i个框所对应的图像。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>crop_size</td>
      <td>输入</td>
      <td>包含2个元素的1-D的Tensor，shape为(crop_height, crop_width)，所有裁剪后的图像区域都会被调整为此大小。图像内容的长宽比不会被保留。crop_height和crop_width必须是正数。对应计算公式描述中的`crop_size`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>extrapolation_value</td>
      <td>可选属性</td>
      <td><ul><li>可选外插值。</li><li>默认值为0.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>method</td>
      <td>可选属性</td>
      <td><ul><li>指定调整大小时的采样方法，为可选字符串。对应计算公式中的`method`。</li><li>提供的方法有：“bilinear”，“nearest”，“bilinear_v2”。</li><li>默认值为“bilinear”。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>返回一个四维Tensor，shape大小为(num_boxes, crop_height, crop_width, depth)。对应计算公式描述中的`输出y`。</td>
      <td>FLOAT、FLOAT16</td>
      <td>NHWC</td>
    </tr>
  </tbody></table>

## 约束说明

本算子提供 **AiCore** 与 **AiCpu** 两种实现。图模式下由 GE 引擎分配自动路由：输入满足 AiCore 约束时优先由 AiCore 执行；不满足 AiCore 约束时自动 fallback 到 AiCpu 执行。

**AiCore 实现约束：**

- 数据类型：x 仅支持 FLOAT16/FLOAT；boxes 支持 FLOAT/FLOAT16（y 数据类型与 boxes 一致）。
- 输入 shape 需为静态；动态 shape 由 AiCpu 执行。
- method 仅支持 bilinear。
- crop_height、crop_width 均不超过 16，且 crop_height × crop_width ≤ 32765。
- num_boxes（boxes.shape[0]）必须满足 50 < num_boxes ≤ 4000。
- depth（通道数，x.shape[3]）必须满足 256 ≤ depth ≤ 2048。
- image_height × image_width ≤ 65530（x 为 FLOAT16 时）/ 32765（x 为 FLOAT 时）。

**AiCpu 实现能力：**

- 数据类型：x 支持 UINT8/INT8/UINT16/INT16/INT32/INT64/FLOAT16/FLOAT/DOUBLE；boxes 仅支持 FLOAT；y 为 FLOAT。
- method 支持 bilinear/nearest/bilinear_v2。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_crop_and_resize](./examples/test_geir_crop_and_resize.cpp)   | 通过[算子IR](./op_graph/crop_and_resize_proto.h)构图方式调用CropAndResize算子。 |
