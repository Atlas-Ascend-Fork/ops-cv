# GridSampler3DGrad

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

- 算子功能：[GridSampler](../grid_sample/README.md)中3D场景的反向传播，完成张量input与张量grid的梯度计算。

- 计算流程：
  1. 根据grid存储的(x, y, z)值，计算出映射到input上的坐标，这些坐标和align_corners、padding_mode有关。
  2. 坐标根据输入的interpolation_mode，选择使用bilinear、nearest不同插值模式计算输出值。
  3. 根据grad存储的梯度值乘上对应点的权重值，计算出最终dx、dgrid的结果。

- 计算公式：

  grad、input、grid、dx、dgrid的尺寸如下：

  $$
  grad: (N, C, D_{out}, H_{out}, W_{out})\\
  input: (N, C, D_{in}, H_{in}, W_{in})\\
  grid: (N, D_{out}, H_{out}, W_{out}, 3)\\
  dx: (N, C, D_{in}, H_{in}, W_{in})\\
  dgrid: (N, D_{out}, H_{out}, W_{out}, 3)
  $$

  其中grad、input、grid、dx、dgrid中的N是一致的，grad、input和dx中的C是一致的，input和dx中的$D_{in}$、$H_{in}$、$W_{in}$是一致的，grad、grid和dgrid中的$D_{out}$、$H_{out}$、$W_{out}$是一致的，grid最后一维大小为3，表示input像素位置信息为(x, y, z)，会将x、y、z的取值范围归一化到[-1, 1]之间。

  1. 坐标反归一化：

     grid中的(x, y, z)需要先反归一化到input像素坐标(ix, iy, iz)，同时计算梯度乘子`gix_mult`、`giy_mult`、`giz_mult`（用于后续dgrid计算）：
     - align_corners = true：

       $$
       ix = \frac{(x+1)}{2} \cdot (W_{in} - 1), \quad gix\_mult = \frac{W_{in} - 1}{2} \\
       iy = \frac{(y+1)}{2} \cdot (H_{in} - 1), \quad giy\_mult = \frac{H_{in} - 1}{2} \\
       iz = \frac{(z+1)}{2} \cdot (D_{in} - 1), \quad giz\_mult = \frac{D_{in} - 1}{2}
       $$

     - align_corners = false：

       $$
       ix = \frac{(x+1) \cdot W_{in} - 1}{2}, \quad gix\_mult = \frac{W_{in}}{2} \\
       iy = \frac{(y+1) \cdot H_{in} - 1}{2}, \quad giy\_mult = \frac{H_{in}}{2} \\
       iz = \frac{(z+1) \cdot D_{in} - 1}{2}, \quad giz\_mult = \frac{D_{in}}{2}
       $$

  2. padding_mode对梯度乘子的影响：
     - padding_mode="zeros"，`gix_mult`不变
     - padding_mode="border"，$gix\_mult = gix\_mult × grad\_clip$（反归一化后的像素坐标不满足$0 < coord < (size - 1)$时grad_clip=0，否则为1）
     - padding_mode="reflection"，$gix\_mult = gix\_mult × grad\_refl × grad\_clip$（grad_refl是反射坐标变换函数对输入坐标的导数，表示反射后输出坐标随输入坐标变化的方向和速率。取值为-1，0，1）。

  3. 各插值模式的梯度公式：
     - Bilinear（三线性插值，Trilinear）

       四个角点坐标和权重为：

       | 角点 | 坐标$(iz_p, iy_p, ix_p)$ | 权重$w_p$ |
       |:------:|:------:|:----------:|
       | tnw（顶-北-西） | $(⌊iz⌋, ⌊iy⌋, ⌊ix⌋)$ | $(ix_{bse} - ix) × (iy_{bse} - iy) × (iz_{bse} - iz)$ |
       | tne（顶-北-东） | $(⌊iz⌋, ⌊iy⌋, ⌊ix⌋+1)$ | $(ix - ix_{bsw}) × (iy_{bsw} - iy) × (iz_{bsw} -  iz)$ |
       | tsw（顶-南-西） | $(⌊iz⌋, ⌊iy⌋+1, ⌊ix⌋)$ | $(ix_{bne} - ix) × (iy - iy_{bne}) × (iz_{bne} -  iz)$ |
       | tse（顶-南-东） | $(⌊iz⌋, ⌊iy⌋+1, ⌊ix⌋+1)$ | $(ix - ix_{bnw}) × (iy - iy_{bnw}) × (iz_{bnw} -  iz)$ |
       | bnw（底-北-西） | $(⌊iz⌋+1, ⌊iy⌋, ⌊ix⌋)$ | $(ix_{tse} - ix) × (iy_{tse} - iy) × (iz -  iz_{tse})$|
       | bne（底-北-东） | $(⌊iz⌋+1, ⌊iy⌋, ⌊ix⌋+1)$ | $(ix - ix_{tsw}) × (iy_{tsw} - iy) × (iz -  iz_{tsw})$ |
       | bsw（底-南-西） | $(⌊iz⌋+1, ⌊iy⌋+1, ⌊ix⌋)$ | $(ix_{tne} - ix) × (iy - iy_{tne}) × (iz -  iz_{tne})$ |
       | bse（底-南-东） | $(⌊iz⌋+1, ⌊iy⌋+1, ⌊ix⌋+1)$ | $(ix - ix_{tnw}) × (iy - iy_{tnw}) × (iz -  iz_{tnw})$ |

       其中：

       $$
       ix_{tnw} = floor(ix) \\
       iy_{tnw} = floor(iy) \\
       iz_{tnw} = floor(iz) \\
       $$

       $$
       ix_{tne} = ix_{tnw} + 1 \\
       iy_{tne} = iy_{tnw} \\
       iz_{tne} = iz_{tnw}\\
       $$

       $$
       ix_{tsw} = ix_{tnw} \\
       iy_{tsw} = iy_{tnw} + 1 \\
       iz_{tsw} = iz_{tnw}\\
       $$

       $$
       ix_{tse} = ix_{tnw} + 1 \\
       iy_{tse} = iy_{tnw} + 1 \\
       iz_{tse} = iz_{tnw} \\
       $$

       $$
       ix_{bnw} = ix_{tnw}  \\
       iy_{bnw} = iy_{tnw}  \\
       iz_{bnw} = iz_{tnw} + 1 \\
       $$

       $$
       ix_{bne} = ix_{tnw} + 1  \\
       iy_{bne} = iy_{tnw}  \\
       iz_{bne} = iz_{tnw} + 1 \\
       $$

       $$
       ix_{bsw} = ix_{tnw}  \\
       iy_{bsw} = iy_{tnw} + 1  \\
       iz_{bsw} = iz_{tnw} + 1 \\
       $$

       $$
       ix_{bse} = ix_{tnw} + 1 \\
       iy_{bse} = iy_{tnw} + 1  \\
       iz_{bse} = iz_{tnw} + 1 \\
       $$

       - dx（input梯度）：将上游梯度按三线性权重散射到input对应位置

         $$
         dx(N, C, iz_p, iy_p, ix_p) \mathrel{+}= w_p \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         即对每个输出像素(d, h, w)，将其梯度乘以三线性权重，累加到input的8个相邻体素位置（越界位置不累加）。
       - dgrid（grid梯度）：对(ix, iy, iz)的偏导

         $$
         gix = \sum_{c} \left[ -V_{tnw} \cdot (iy_{bse}-iy)(iz_{bse}-iz) + V_{tne} \cdot (iy_{bsw}-iy)(iz_{bsw}-iz) - V_{tsw} \cdot (iy-iy_{bne})(iz_{bne}-iz) + V_{tse} \cdot (iy-iy_{bnw})(iz_{bnw}-iz) - V_{bnw} \cdot (iy_{tse}-iy)(iz-iz_{tse}) + V_{bne} \cdot (iy_{tsw}-iy)(iz-iz_{tsw}) - V_{bsw} \cdot (iy-iy_{tne})(iz-iz_{tne}) + V_{bse} \cdot (iy-iy_{tnw})(iz-iz_{tnw}) \right] \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         $$
         giy = \sum_{c} \left[ -V_{tnw} \cdot (ix_{bse}-ix)(iz_{bse}-iz) - V_{tne} \cdot (ix-ix_{bsw})(iz_{bsw}-iz) + V_{tsw} \cdot (ix_{bne}-ix)(iz_{bne}-iz) + V_{tse} \cdot (ix-ix_{bnw})(iz_{bnw}-iz) - V_{bnw} \cdot (ix_{tse}-ix)(iz-iz_{tse}) - V_{bne} \cdot (ix-ix_{tsw})(iz-iz_{tsw}) + V_{bsw} \cdot (ix_{tne}-ix)(iz-iz_{tne}) + V_{bse} \cdot (ix-ix_{tnw})(iz-iz_{tnw}) \right] \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         $$
         giz = \sum_{c} \left[ -V_{tnw} \cdot (ix_{bse}-ix)(iy_{bse}-iy) - V_{tne} \cdot (ix-ix_{bsw})(iy_{bsw}-iy) - V_{tsw} \cdot (ix_{bne}-ix)(iy-iy_{bne}) - V_{tse} \cdot (ix-ix_{bnw})(iy-iy_{bnw}) + V_{bnw} \cdot (ix_{tse}-ix)(iy_{tse}-iy) + V_{bne} \cdot (ix-ix_{tsw})(iy_{tsw}-iy) + V_{bsw} \cdot (ix_{tne}-ix)(iy-iy_{tne}) + V_{bse} \cdot (ix-ix_{tnw})(iy-iy_{tnw}) \right] \cdot grad(N, C, D_{out}, H_{out}, W_{out})
         $$

         其中 $V_p = input(N, C, iz_p, iy_p, ix_p)$（仅当角点在边界内时参与计算）。
       - 最终：

         $$
         dgrid(N, D_{out}, H_{out}, W_{out}, 0) = gix\_mult \cdot gix
         $$

         $$
         dgrid(N, D_{out}, H_{out}, W_{out}, 1) = giy\_mult \cdot giy
         $$

         $$
         dgrid(N, D_{out}, H_{out}, W_{out}, 2) = giz\_mult \cdot giz
         $$

       - Nearest（最邻近插值）
         - dx：将上游梯度直接累加到最近邻位置

          $$
          dx(N, C, \text{round}(iz), \text{round}(iy), \text{round}(ix)) \mathrel{+}= grad(N, C, D_{out}, H_{out}, W_{out})
          $$

         - dgrid：最邻近插值对坐标不可导，因此 **dgrid = 0**。

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
      <td>表示反向传播过程中上一层的输出梯度，对应公式描述中的`grad`。数据类型与`x`的数据类型一致。当数据类型DOUBLE时，数据格式不支持NDHWC。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>NCDHW、NDHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示反向传播的输入张量，对应公式描述中的`input`。shape仅支持五维，且需满足`x`和`grad`的N轴和C轴的值保持一致，x的D，H，W值不可为0。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>NCDHW、NDHWC</td>
    </tr>
    <tr>
      <td>grid</td>
      <td>输入</td>
      <td>表示采样像素位置的张量，对应公式描述中的`grid`。shape仅支持五维，且需满足`grid`和`grad`的N轴、D轴、H轴、W轴的值保持一致，最后一维的值等于3。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>interpolation_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示插值模式，对应公式描述中的`interpolation_mode`。支持bilinear（0：双线性插值）和nearest（1：最邻近插值）。</li><li>默认值为"bilinear"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>可选属性</td>
      <td><ul><li>用于表示填充模式，对应公式描述中的`padding_mode`。支持0（'zeros'）、1（'border'）、2（'reflection'）三种模式。</li><li>默认值为"zeros"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td><ul><li>表示设定特征图坐标与特征值的对应方式，对应公式描述中的`align_corners`。如果为true，则将极值-1和1视为参考输入的角像素点的中心点；如果为false，则视为参考输入的角像素点的角点。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>表示反向传播的输出梯度，对应公式描述中的`dx`。数据类型、数据格式和shape与`x`的数据类型、数据格式和shape保持一致。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>NCDHW、NDHWC</td>
    </tr>
    <tr>
      <td>dgrid</td>
      <td>输出</td>
      <td>表示`grid`梯度，对应公式描述中的`dgrid`。数据类型、数据格式和shape与`grid`的数据类型、数据格式和shape保持一致。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、BFLOAT16</td>
      <td>NDHWC</td>
    </tr>
  </tbody></table>

<term>Atlas训练系列产品</term>：输入参数和输出参数的数据类型不支持DOUBLE、BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_grid_sampler3_d_backward](examples/test_aclnn_grid_sampler3_d_backward.cpp) | 通过[aclnnGridSampler3DBackward](docs/aclnnGridSampler3DBackward.md)接口方式调用GridSampler3DGrad算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/grid_sampler3_d_grad_proto.h)构图方式调用GridSampler3DGrad算子。         |
