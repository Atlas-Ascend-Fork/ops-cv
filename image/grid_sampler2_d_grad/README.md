# GridSampler2DGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：[GridSampler](../grid_sample/README.md)中2D场景的反向传播，完成张量input与张量grid的梯度计算。
- 计算流程：

    1. 根据grid存储的(x, y)值，计算出映射到input上的坐标，这些坐标和align_corners、padding_mode有关。
    2. 根据输入的interpolationMode，选择使用bilinear、nearest、bicubic不同插值模式计算该坐标周围点分配到梯度的权重值。
    3. 根据grad存储的梯度值乘上对应点的权重值，计算出最终dx、dgrid的结果。

- 计算公式：

  grad、input、grid、dx、dgrid的尺寸如下：
  $$
  grad: (N, C, H_{out}, W_{out})\\
  input: (N, C, H_{in}, W_{in})\\
  grid: (N, H_{out}, W_{out}, 2)\\
  dx: (N, C, H_{in}, W_{in})\\
  dgrid: (N, H_{out}, W_{out}, 2)
  $$

  其中grad、input、grid、dx、dgrid中的N均相同，grad、input和dx中的C相同，input和dx中的$H_{in}$、$W_{in}$相同，grad、grid和dgrid中的$H_{out}$、$W_{out}$相同，grid最后一维大小为2，表示input像素位置信息为(x, y)。x和y的取值范围归一化到[-1, 1]，(-1, -1)表示左上角坐标，(1, 1)表示右下角坐标。

  1. 坐标反归一化：
     grid中的(x, y)需要先反归一化到input像素坐标(ix, iy)，同时计算梯度乘子`gix_mult`、`giy_mult`（用于后续dgrid计算）：
     - align_corners = true：

       $$
       ix = \frac{(x+1)}{2} \cdot (W_{in} - 1), \quad gix\_mult = \frac{W_{in} - 1}{2} \\
       iy = \frac{(y+1)}{2} \cdot (H_{in} - 1), \quad giy\_mult = \frac{H_{in} - 1}{2}
       $$

     - align_corners = false：

       $$
       ix = \frac{(x+1) \cdot W_{in} - 1}{2}, \quad gix\_mult = \frac{W_{in}}{2} \\
       iy = \frac{(y+1) \cdot H_{in} - 1}{2}, \quad giy\_mult = \frac{H_{in}}{2}
       $$

  2. padding_mode对梯度乘子的影响：
     - padding_mode="zeros"，`gix_mult`不变
     - padding_mode="border"，$gix\_mult = gix\_mult × grad\_clip$（反归一化后的像素坐标不满足$0 < coord < (size - 1)$时grad_clip=0，否则为1）
     - padding_mode="reflection"，$gix\_mult = gix\_mult × grad\_refl × grad\_clip$（grad_refl是反射坐标变换函数对输入坐标的导数，表示反射后输出坐标随输入坐标变化的方向和速率。取值为-1，0，1）。

  3. 各插值模式的梯度公式：
     - Bilinear（双线性插值）

       四个角点坐标和权重为：

       | 角点 | 坐标$(iy_p, ix_p)$ | 权重$w_p$ |
       |:------:|:------:|:----------:|
       | nw（西北） | $(iy_{nw}, ix_{nw})$ | $(ix_{se} - ix) × (iy_{se} - iy)$ |
       | ne（东北） | $(iy_{ne}, ix_{ne})$ | $(ix - ix_{sw}) × (iy_{sw} - iy)$ |
       | sw（西南） | $(iy_{sw}, ix_{sw})$ | $(ix_{ne} - ix) × (iy - iy_{ne})$ |
       | se（东南） | $(iy_{se}, ix_{se})$ | $(ix - ix_{nw}) × (iy - iy_{nw})$ |

       其中：

       $$
       ix_{nw} = floor(ix) \\
       iy_{nw} = floor(iy) \\
       ix_{ne} = floor(ix) + 1 \\
       iy_{ne} = floor(iy)\\
       ix_{sw} = floor(ix) \\
       iy_{sw} = floor(iy) + 1 \\
       ix_{se} = floor(ix) + 1 \\
       iy_{se} = floor(iy) + 1
       $$

       - dx（input梯度）：将上游梯度按权重散射到input对应位置

         $$
         dx(N, C, iy_p, ix_p) \mathrel{+}= w_p \cdot grad(N, C, H_{out}, W_{out})
         $$

         即对每个输出像素(h, w)，将其梯度乘以双线性权重，累加到input的四个相邻像素位置（越界位置不累加）。
       - dgrid（grid梯度）：对(ix, iy)的偏导

         $$
         gix = \sum_{c} \left[ -V_{nw} \cdot (iy_{se} - iy) + V_{ne} \cdot   (iy_{sw} - iy) - V_{sw} \cdot (iy - iy_{ne}) + V_{se} \cdot (iy - iy_{nw}) \right] \cdot grad(N, C, H_{out}, W_{out})
         $$

         $$
         giy = \sum_{c} \left[ -V_{nw} \cdot (ix_{se} - ix) - V_{ne} \cdot   (ix - ix_{sw}) + V_{sw} \cdot (ix_{ne} - ix) + V_{se} \cdot (ix - ix_{nw}) \right] \cdot grad(N, C, H_{out}, W_{out})
         $$

         其中 $V_p = input(N, C, iy_p, ix_p)$（仅当角点在边界内时参与计算）。
       - 最终：

         $$
         dgrid(N, H_{out}, W_{out}, 0) = gix\_mult \cdot gix
         $$

         $$
         dgrid(N, H_{out}, W_{out}, 1) = giy\_mult \cdot giy
         $$

     - Nearest（最邻近插值）
       - dx：将上游梯度直接累加到最近邻位置

          $$
          dx(N, C, \text{round}(iy), \text{round}(ix)) \mathrel{+}= grad(N, C, H_{out}, W_{out})
          $$

       - dgrid：最邻近插值对坐标不可导，因此 **dgrid = 0**。

     - Bicubic（双三次插值）
       - dx：

          $$
          dx(N, C, iy', ix') \mathrel{+}= grad(N, C, H_{out}, W_{out}) \cdot x\_coeffs[i] \cdot y\_coeffs[j]
          $$

          其中：

          $(ix', iy') = (ix_{nw}-1+i, iy_{nw}-1+j)$，$i,j \in \{0,1,2,3\}$，越界位置根据padding_mode处理。

          $$
          tx = ix - floor(ix) \\
          ty = iy - floor(iy) \\
          $$

          $$
          A = -0.75 \\
          x_0 = tx + 1.0 \\
          x\_coeffs[0] = ((A * x_0 - 5* A) * x_0 + 8 * A) * x_0 - 4 * A
          $$

          $$
          x_1 = tx \\
          x\_coeffs[1] = ((A + 2)* x_1 - (A + 3)) * x_1 * x_1 + 1
          $$

          $$
          x_2 = 1 - tx \\
          x\_coeffs[2] = ((A + 2)* x_2 - (A + 3)) * x_2 * x_2 + 1
          $$

          $$
          x_3 = 2 - tx \\
          x\_coeffs[3] = ((A * x_3 - 5* A) * x_3 + 8 * A) * x_3 - 4 * A
          $$

          $$
          y_0 = ty + 1.0 \\
          y\_coeffs[0] = ((A * y_0 - 5* A) * y_0 + 8 * A) * y_0 - 4 * A
          $$

          $$
          y_1 = ty \\
          y\_coeffs[1] = ((A + 2)* y_1 - (A + 3)) * y_1 * y_1 + 1
          $$

          $$
          y_2 = 1 - ty \\
          y\_coeffs[2] = ((A + 2)* y_2 - (A + 3)) * y_2 * y_2 + 1
          $$

          $$
          y_3 = 2 - ty \\
          y\_coeffs[3] = ((A * y_3 - 5* A) * y_3 + 8 * A) * y_3 - 4 * A
          $$

       - dgrid：

          $$
          gix = -\sum_{C}\sum_{i=0}^{3}\sum_{j=0}^{3} V_{ij} \cdot x\_coeffs\_grad[i] \cdot y\_coeffs[j] \cdot grad(N, C, H_{out}, W_{out})
          $$

          $$
          giy = -\sum_{C}\sum_{i=0}^{3}\sum_{j=0}^{3} V_{ij} \cdot y\_coeffs\_grad[j] \cdot x\_coeffs[i] \cdot grad(N, C, H_{out}, W_{out})
          $$

          其中：

          $V_{ij} = get\_value\_bounded(input(N, C, H_{in}, W_{in}), ix_{nw}-1+i, iy_{nw}-1  +j)$，`x_coeffs_grad`和`y_coeffs_grad`是三次插值系数对tx/ty的导数：

          $$
          x\_coeffs\_grad[0] = (-3A \cdot x - 10A) \cdot x - 8A \\
          \quad x = |-1 - tx|
          $$

          $$
          x\_coeffs\_grad[1] = (-3(A+2) \cdot x - 2(A+3)) \cdot x \\
          \quad x = | 0 - tx|
          $$

          $$
          x\_coeffs\_grad[2] = (3(A+2) \cdot x - 2(A+3)) \cdot x \\
          \quad x = |1 - tx|
          $$

          $$
          x\_coeffs\_grad[3] = (3A \cdot x - 10A) \cdot x + 8A \\
          \quad x = |2 - tx|
          $$

          $$
          y\_coeffs\_grad[0] = (-3A \cdot y - 10A) \cdot y - 8A \\
          \quad y = |-1 - ty|
          $$

          $$
          y\_coeffs\_grad[1] = (-3(A+2) \cdot y - 2(A+3)) \cdot y \\
          \quad y = | 0 - ty|
          $$

          $$
          y\_coeffs\_grad[2] = (3(A+2) \cdot y - 2(A+3)) \cdot y \\
          \quad y = |1 - ty|
          $$

          $$
          y\_coeffs\_grad[3] = (3A \cdot y - 10A) \cdot y + 8A \\
          \quad y = |2 - ty|
          $$

          最终：

          $$
          dgrid(N, H_{out}, W_{out}, 0) = gix\_mult \cdot gix
          $$

          $$
          dgrid(N, H_{out}, W_{out}, 1) = giy\_mult \cdot giy
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
      <td>grad</td>
      <td>输入</td>
      <td>表示反向传播过程中上一层的输出梯度，对应公式描述中的`grad`。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示反向传播的输入张量，对应公式描述中的`input`。shape仅支持四维，且需满足`x`和`grad`的N轴和C轴的值保持一致，`x`最后两维的维度值不可为0。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grid</td>
      <td>输入</td>
      <td>表示采样像素位置的张量，对应公式描述中的`grid`。shape仅支持四维，且需满足`grid`和`grad`的N轴、H轴、W轴的值保持一致，`grid`最后一维的值等于2。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>interpolation_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示插值模式，对应公式描述中的`interpolation_mode`。支持"bilinear"（双线性插值）、"nearest"（最邻近插值）、"bicubic"（双三次插值）。</li><li>默认值为"bilinear"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>可选属性</td>
      <td><ul><li>用于表示填充模式，对应公式描述中的`padding_mode`。支持"zeros"、"border"、"reflection"三种模式。</li><li>默认值为"zeros"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td><ul><li>表示设定特征图坐标与特征值的对应方式，对应公式描述中的`align_corners`。设定为true时，特征值位于像素中心；设定为false时，特征值位于像素的角点。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>表示反向传播的输出梯度，对应公式描述中的`dx`。数据类型、数据格式和shape与`x`的数据类型、数据格式和shape保持一致。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgrid</td>
      <td>输出</td>
      <td>表示grid梯度，对应公式描述中的`dgrid`。数据类型、数据格式和shape与`grid`的数据类型、数据格式和shape保持一致。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

<term>Atlas训练系列产品</term>：输入参数和输出参数的数据类型不支持DOUBLE、BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_grid_sampler2_d_backward](examples/test_aclnn_grid_sampler2_d_backward.cpp) | 通过[aclnnGridSampler2DBackward](docs/aclnnGridSampler2DBackward.md)接口方式调用GridSampler2DGrad算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/grid_sampler2_d_grad_proto.h)构图方式调用GridSampler2DGrad算子。         |
