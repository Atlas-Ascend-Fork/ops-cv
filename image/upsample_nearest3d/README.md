# UpsampleNearest3d

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |    √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

- <term>Ascend 950PR/Ascend 950DT（A5）</term>上不支持 GEIR。

## 功能说明

- 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。
- 计算公式：
  - 核心算法逻辑：
    1. 将目标图像的每一个点映射回原图，得到一个带小数点的坐标。
    2. 根据这个浮点数坐标，计算距离最近的原始图像的点，后者的值直接复制给前者。
  - 具体计算逻辑：

    对于out的某个方向上的点p(x,y,z)，映射回原始图像中的点记为q(x',y',z')，则有关系：

    $$
    x' = \min(\lfloor x * scales\_depth \rfloor, self\_D - 1), \quad scales\_depth = self\_D / outputSize[0]
    $$

    $$
    y' = \min(\lfloor y * scales\_height \rfloor, self\_H - 1), \quad scales\_height = self\_H / outputSize[1]
    $$

    $$
    z' = \min(\lfloor z * scales\_width \rfloor, self\_W - 1), \quad scales\_width = self\_W / outputSize[2]
    $$

    则有以下公式：

    $$
    {V(p_{x,y,z})} = {V(q_{x',y',z'})}
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
      <td>表示进行上采样的输入张量，对应公式中的`self`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>指定输出空间大小，对应公式中的`outputSize`。size为3，各元素均大于零。表示输出`y`在D、H和W维度上的空间大小。</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>可选属性</td>
      <td><ul><li>指定沿每个维度的缩放数组，包含3个元素：scale_depth, scale_height, scale_width。</li><li>默认为空。</li></ul></td>
      <td>LISTFLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式中的`out`。数据类型和数据格式需与入参`x`的数据类型和数据格式保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
  </tbody></table>

- <term>Atlas推理系列产品</term>：输入和输出的数据类型不支持BFLOAT16。
- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

- <term>Ascend 950PR/Ascend 950DT（A5）</term>上不支持 GEIR，不支持通过算子 IR（[op_graph/upsample_nearest3d_proto.h](op_graph/upsample_nearest3d_proto.h)）构图方式调用本算子。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_nearest1d_v2](examples/test_aclnn_upsample_nearest1d_v2.cpp) | 通过[aclnnUpsampleNearest1dV2](docs/aclnnUpsampleNearest1dV2.md)接口方式调用UpsampleNearest3d算子。 |
| aclnn接口  | [test_aclnn_upsample_nearest2d_v2](examples/test_aclnn_upsample_nearest2d_v2.cpp) | 通过[aclnnUpsampleNearest2dV2](docs/aclnnUpsampleNearest2dV2.md)接口方式调用UpsampleNearest3d算子。 |
| aclnn接口  | [test_aclnn_upsample_nearest3d](examples/test_aclnn_upsample_nearest3d.cpp) | 通过[aclnnUpsampleNearest3d](docs/aclnnUpsampleNearest3d.md)接口方式调用UpsampleNearest3d算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/upsample_nearest3d_proto.h)构图方式调用UpsampleNearest3d算子。<term>Ascend 950PR/Ascend 950DT（A5）</term>上不支持 GEIR，该调用方式不适用。         |
