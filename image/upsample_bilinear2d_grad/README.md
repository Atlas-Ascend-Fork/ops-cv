# UpsampleBilinear2dGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     ×    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     √    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对由多个输入通道组成的输入信号应用2D双线性上采样的反向传播。
- 计算公式：
  - 正向的核心算法逻辑：
    1. 将目标图像的每一个点映射回原图，得到一个带小数点的坐标。
    2. 根据这个浮点数坐标，计算前后相邻的原始图像的点。
    3. 分别计算相邻点到对应目标点的权重，按照权重相乘累加即可得到目标点值。
  - 具体计算逻辑：
    缩放方式分为角对齐和边对齐，角对齐（alignCorners为true）表示按照原始图片左上角像素中心点对齐，边对齐（alignCorners为false）表示按照原始图片左上角顶点及两条边对齐，在计算缩放系数和坐标位置时存在差异。对于一个二维插值点$(N, C, H, W)$，则有以下公式：

    $$
    scale\_h =\begin{cases}
    (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scales\_h & alignCorners=false\&scales\_h>0\\
    inputSize[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_w =\begin{cases}
    (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
    1 / scales\_w & alignCorners=false\&scales\_w>0\\
    inputSize[3] / outputSize[1] & alignCorners=false
    \end{cases}
    $$

    因此，对于output中横向（W方向）坐标为x、纵向（H方向）坐标为y的点p(x,y)，映射回原始图像中的点记为q(x',y')，则有关系：

    $$
    x' =\begin{cases}
    x * scale\_w & alignCorners=true \\
    MAX(0,{(x+0.5)*scale\_w-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    y' =\begin{cases}
    y * scale\_h & alignCorners=true \\
    MAX(0,{(y+0.5)*scale\_h-0.5}) & alignCorners=false
    \end{cases}
    $$

    - 记：

      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$

      $$
      y_{0} =int(y'),y_{1} =int(y')+1, lambdb_{0} = y_{1}-y', lambdb_{1} =   1-lambdb_{0}
      $$

    - 则有以下公式：

      $$
      {V(p_{x, y})} = {V(p_{x0, y0})} * {lambda_{0}} * {lambdb_{0}} + {V(p_{x0, y1})} * {lambda_{0}} * {lambdb_{1}} + {V(p_{x1, y0})} * {lambda_{1}} * {lambdb_{0}} + {V(p_{x1, y1})} * {lambda_{1}} * {lambdb_{1}}
      $$

    - 假设：正向插值的输出图像out $(x, y)$受原图像input $(x_i, y_j)$影响，则有：

      $$
      gradInput(x_i,y_j) += gradOutput(x,y) * lambda(x_i,y_j)* lambdb(x_i,y_j)
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
      <td>grad_output</td>
      <td>输入</td>
      <td>表示反向计算的梯度Tensor，对应公式中的`gradOutput`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>指定输入空间大小，对应公式中的`outputSize`。size为2，且各元素均大于0。表示指定`grad_output`在H和W维度上的空间大小。</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_size</td>
      <td>属性</td>
      <td>指定输出空间大小，对应公式中的`inputSize`。size为4，且各元素均大于零。表示输出`grad_input`分别在N、C、H和W维度上的空间大小。</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td><ul><li>决定是否对齐角像素点，对应公式中的`alignCorners`。如果设置为true，则输入和输出张量按其角像素的中心点对齐，保留角像素处的值；如果设置为false，则输入和输出张量通过其角像素的角点对齐，并且插值使用边缘值填充用于外界边值。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>可选属性</td>
      <td><ul><li>指定空间大小的height维度乘数，对应公式中的`scales_h`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>可选属性</td>
      <td><ul><li>指定空间大小的width维度乘数，对应公式中的`scales_w`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>表示反向计算的输出张量，对应公式中的`gradInput`。数据类型和数据格式与入参`grad_output`的数据类型和数据格式保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

<term>Atlas推理系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>：输入和输出的数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_bilinear2d_backward_v2](examples/test_aclnn_upsample_bilinear2d_backward_v2.cpp) | 通过[aclnnUpsampleBilinear2dBackwardV2](docs/aclnnUpsampleBilinear2dBackwardV2.md)接口方式调用UpsampleBilinear2dGrad算子。 |
| aclnn接口  | [test_aclnn_upsample_linear1d_backward](examples/test_aclnn_upsample_linear1d_backward.cpp) | 通过[aclnnUpsampleLinear1dBackward](docs/aclnnUpsampleLinear1dBackward.md)接口方式调用UpsampleBilinear2dGrad算子。 |
