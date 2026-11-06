# ImageProjectiveTransform

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对输入图像施加射影变换（Projective Transform），根据变换矩阵将输出图像中的每个像素映射回输入图像中对应的坐标，再通过插值计算输出像素值。

- 计算公式：

  对于输出图像中的每个像素 $(x_{out}, y_{out})$，其对应的输入图像坐标 $(x_{in}, y_{in})$ 通过射影变换矩阵计算：

  $$
  x_{in} = \frac{a_0 \cdot x_{out} + a_1 \cdot y_{out} + a_2}{a_6 \cdot x_{out} + a_7 \cdot y_{out} + 1}
  $$

  $$
  y_{in} = \frac{a_3 \cdot x_{out} + a_4 \cdot y_{out} + a_5}{a_6 \cdot x_{out} + a_7 \cdot y_{out} + 1}
  $$

  其中 $a_0, a_1, \ldots, a_7$ 为变换矩阵的8个参数，由输入 `transforms` 张量提供。

  根据插值模式计算输出像素值：

  - **BILINEAR（双线性插值）**：

    $$
    x_{floor} = \lfloor x_{in} \rfloor, \quad x_{ceil} = x_{floor} + 1
    $$

    $$
    y_{floor} = \lfloor y_{in} \rfloor, \quad y_{ceil} = y_{floor} + 1
    $$

    $$
    v_{yfloor} = (x_{ceil} - x_{in}) \cdot p(x_{floor}, y_{floor}) + (x_{in} - x_{floor}) \cdot p(x_{ceil}, y_{floor})
    $$

    $$
    v_{yceil} = (x_{ceil} - x_{in}) \cdot p(x_{floor}, y_{ceil}) + (x_{in} - x_{floor}) \cdot p(x_{ceil}, y_{ceil})
    $$

    $$
    result = (y_{ceil} - y_{in}) \cdot v_{yfloor} + (y_{in} - y_{floor}) \cdot v_{yceil}
    $$

    当源坐标超出输入图像边界时，使用填充值（默认为0）代替越界像素值。当变换矩阵分母（$a_6 \cdot x_{out} + a_7 \cdot y_{out} + 1$）为0时，该像素输出填充值0（浮点与整数类型一致）。当分母非0、但映射得到的坐标 $x_{in}$ 或 $y_{in}$ 为NaN/Inf时：BILINEAR模式下浮点类型输出NaN、INT32输出INT_MIN、UINT8输出0；NEAREST模式下输出填充值0。

  - **NEAREST（最近邻插值）**：

    $$
    x_i = \text{round}(x_{in}), \quad y_i = \text{round}(y_{in})
    $$

    当源坐标超出输入图像边界时，使用填充值（默认为0）。

## 参数说明

- **参数说明**：

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
      <td>images</td>
      <td>输入</td>
      <td>输入图像张量，shape为(N, H, W, C)，其中N为批次数，H为图像高度，W为图像宽度，C为通道数。</td>
      <td>FLOAT16、FLOAT、UINT8、INT32</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>transforms</td>
      <td>输入</td>
      <td>射影变换矩阵参数，shape为(N, 8)或(1, 8)；后一种将同一变换广播至所有图像。每行包含8个变换参数[a0, a1, a2, a3, a4, a5, a6, a7]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_shape</td>
      <td>输入</td>
      <td>输出图像的空间尺寸，shape为(2,)，包含[height, width]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>interpolation</td>
      <td>属性</td>
      <td>插值模式，取值为"BILINEAR"（双线性插值）或"NEAREST"（最近邻插值）。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fill_mode</td>
      <td>可选属性</td>
      <td>填充模式，支持"CONSTANT"、"NEAREST"、"REFLECT"和"WRAP"，默认值为"CONSTANT"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transformed_images</td>
      <td>输出</td>
      <td>变换后的图像张量，shape为(N, HOut, WOut, C)，数据类型与images一致。</td>
      <td>FLOAT16、FLOAT、UINT8、INT32</td>
      <td>NHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- images输入必须为4维（NHWC格式）。
- transforms输入必须为2维，shape为(M, 8)，M必须为1或images的批次数N，数据类型为FLOAT。N为0时允许M为0或1。
- output_shape输入必须为1维，shape为(2,)，数据类型为INT32。
- output_shape的两个元素[height, width]决定了输出transformed_images的中间两维(HOut, WOut)，即transformed_images.shape[1]=output_shape[0]，transformed_images.shape[2]=output_shape[1]。当output_shape某元素为0时，表示输出为空tensor，算子直接返回，不进行计算。
- output_shape的已知长度必须恰好为2，可读常量height和width必须非负；0保留为空输出，负值报错。
- fill_mode区分大小写："CONSTANT"越界填0，"NEAREST"限制到边缘坐标，"REFLECT"按边缘重复镜像，"WRAP"按TensorFlow参考实现的环绕语义重映射（以维长减1为除数取单步商、按维长回绕并夹取到[0, 维长-1]，非数学周期取模）。非法fill_mode或interpolation报错，空输出也不绕过检查。
- 浮点插值结果转换为整型输出时按向零截断：UINT8结果夹取到[0, 255]，INT32结果在NaN/Inf或越界（|值| ≥ 2^31）时输出INT_MIN。
- InferShape允许未知rank和部分动态维，尽量保留已知的N/C及常量H/W；无可读output_shape常量时输出H/W为-1。InferShape按images的origin format推导NCHW/NHWC维位置；实际AI Core kernel仍只支持NHWC。
- Tiling入口要求images四个维度均已具体化并落在[0, INT32_MAX]：未具体化的动态哨兵-1、其他负值或超出INT32_MAX的维度报错；空间乘积超出int64范围时报错。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| GE图模式 | [test_geir_image_projective_transform](./examples/arch35/test_geir_image_projective_transform.cpp) | 通过[算子IR](./op_graph/image_projective_transform_proto.h)构图方式调用ImageProjectiveTransform算子，参见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)完成编译和验证。样例位于`examples/arch35`，编译运行须显式指定`--soc=ascend950`（如`bash build.sh --run_example image_projective_transform graph --soc=ascend950`）。 |

- 本算子以GE图模式注册；自定义包按`ACLNNTYPE aclnn_exclude`构建，包内不生成`aclnnImageProjectiveTransform`接口。TensorFlow适配源码位于`framework/image_projective_transform_tf_plugin.cpp`，把公开TensorFlow入口`ImageProjectiveTransformV2`映射到本算子（该入口无`fill_value`参数，填充语义按本算子CONSTANT模式处理）；TF插件库由仓库TF插件构建机制在相应包型中生成，本自定义run包不包含该插件。
