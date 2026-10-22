
# aclnnGridSampler2DBackward

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

- 接口功能：[aclnnGridSampler2D](../../grid_sample/docs/aclnnGridSampler2D.md)的反向传播，完成张量input与张量grid的梯度计算。
- 计算流程：

    1. 根据grid存储的(x, y)值，计算出映射到input上的坐标，这些坐标和alignCorners、paddingMode有关。
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
     - alignCorners = true：

       $$
       ix = \frac{(x+1)}{2} \cdot (W_{in} - 1), \quad gix\_mult = \frac{W_{in} - 1}{2} \\
       iy = \frac{(y+1)}{2} \cdot (H_{in} - 1), \quad giy\_mult = \frac{H_{in} - 1}{2}
       $$

     - alignCorners = false：

       $$
       ix = \frac{(x+1) \cdot W_{in} - 1}{2}, \quad gix\_mult = \frac{W_{in}}{2} \\
       iy = \frac{(y+1) \cdot H_{in} - 1}{2}, \quad giy\_mult = \frac{H_{in}}{2}
       $$

  2. paddingMode对梯度乘子的影响：
     - paddingMode="zeros"，`gix_mult`不变
     - paddingMode="border"，$gix\_mult = gix\_mult × grad\_clip$（坐标在边界外时grad_clip=0，否则=1）
     - paddingMode="reflection"，$gix\_mult = gix\_mult × grad\_refl × grad\_clip$（grad_refl是反射坐标变换函数对输入坐标的导数，表示反射后输出坐标随输入坐标变化的方向和速率。取值为-1，0，1）。

  3. 各插值模式的梯度公式：
     - Bilinear（双线性插值）

       四个角点坐标和权重为：

       | 角点 | 坐标$(iy_p, ix_p)$ | 权重$w_p$ |
       |:------:|:------:|:----------:|
       | nw (西北) | $(iy_{nw}, ix_{nw})$ | $(ix_{se} - ix) × (iy_{se} - iy)$ |
       | ne (东北) | $(iy_{ne}, ix_{ne})$ | $(ix - ix_{sw}) × (iy_{sw} - iy)$ |
       | sw (西南) | $(iy_{sw}, ix_{sw})$ | $(ix_{ne} - ix) × (iy - iy_{ne})$ |
       | se (东南) | $(iy_{se}, ix_{se})$ | $(ix - ix_{nw}) × (iy - iy_{nw})$ |

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

           $(ix', iy') = (ix_{nw}-1+i, iy_{nw}-1+j)$，$i,j \in \{0,1,2,3\}$，越界位置根据paddingMode处理。

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

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnGridSampler2DBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGridSampler2DBackward”接口执行计算。

```Cpp
aclnnStatus aclnnGridSampler2DBackwardGetWorkspaceSize(
  const aclTensor*     gradOutput,
  const aclTensor*     input,
  const aclTensor*     grid,
  int64_t              interpolationMode,
  int64_t              paddingMode,
  bool                 alignCorners,
  const aclBoolArray*  outputMask,
  aclTensor*           inputGrad,
  aclTensor*           gridGrad,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnGridSampler2DBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnGridSampler2DBackwardGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput（aclTensor*）</td>
      <td>输入</td>
      <td>表示反向传播过程中上一层的输出梯度，对应公式描述中的`grad`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`input`保持一致。</li><li>支持shape为(N, C, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>out</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>out</em>)。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input（aclTensor*）</td>
      <td>输入</td>
      <td>表示反向传播的输入张量，对应公式描述中的`input`。</td>
      <td><ul><li>支持空Tensor。</li><li>`input`和`gradOutput`的N轴和C轴的值保持一致，且`input`最后两维的维度值不可为0。</li><li>支持shape为(N, C, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>in</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>in</em>)。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grid（aclTensor*）</td>
      <td>输入</td>
      <td>表示采样像素位置的张量，对应公式描述中的`grid`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`input`保持一致。</li><li>`grid`和`gradOutput`的N轴、H轴、W轴的值保持一致，`grid`最后一维的值等于2。</li><li>支持shape为(N, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>out</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>out</em>, 2)。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>interpolationMode（int64_t）</td>
      <td>输入</td>
      <td>表示插值模式，对应公式描述中的`interpolationMode`。</td>
      <td>支持0：bilinear（双线性插值）、1：nearest（最邻近插值）、2：bicubic（双三次插值）三种模式。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddingMode（int64_t）</td>
      <td>输入</td>
      <td>表示填充模式，即当(x, y)取值超过输入特征图采样范围时，返回一个特定值。对应公式描述中的`paddingMode`。</td>
      <td>支持0：zeros、1：border、2：reflection三种模式。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners（bool）</td>
      <td>输入</td>
      <td>表示设定特征图坐标与特征值的对应方式，对应公式描述中的`alignCorners`。</td>
      <td>设定为true时，特征值位于像素中心。设定为false时，特征值位于像素的角点。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask（aclBoolArray*）</td>
      <td>输入</td>
      <td>用于表示输出的掩码。</td>
      <td>outputMask[0]为true/false，表示是/否获取输出`inputGrad`；outputMask[1]为true/false，表示是/否获取输出`gridGrad`。</td>
      <td>BOOLARRAY</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputGrad（aclTensor*）</td>
      <td>输出</td>
      <td>表示反向传播的输出梯度，对应公式描述中的`dx`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`input`的数据类型一致。</li><li>shape与`input`保持一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gridGrad（aclTensor*）</td>
      <td>输出</td>
      <td>表示grid梯度，对应公式描述中的`dgrid`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`input`的数据类型一致。</li><li>shape与`grid`保持一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910" id7 -->
  - <term>Atlas训练系列产品</term>：

    参数`gradOutput`、`input`、`grid`、`inputGrad`、`gridGrad`的数据类型不支持BFLOAT16、DOUBLE。
  <!-- end id7 -->

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的gradOutput、input、grid、inputGrad、gridGrad是空指针。</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>gradOutput、input、grid、inputGrad、gridGrad的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradOutput、input、grid的shape维度值不为四维。</td>
    </tr>
    <tr>
      <td>interpolationMode或paddingMode的值不在支持范围内。</td></tr>
    <tr>
      <td>input的第一个维度和grid、gradOutput的第一个维度值不相同。</td>
    </tr>
    <tr>
      <td>input的第二个维度和gradOutput的第二个维度值不相同。</td>
    </tr>
    <tr>
      <td>grid的第二个维度和gradOutput的第三个维度值不相同。</td>
    </tr>
    <tr>
      <td>grid的第三个维度和gradOutput的第四个维度值不相同。</td>
    </tr>
    <tr>
      <td>input最后两维的维度值为0。</td>
    </tr>
    <tr>
      <td>grid最后一维的值不等于2。</td>
    </tr>
    <tr>
      <td>input的shape和inputGrad的shape不一致，或grid的shape和gridGrad的shape不一致。</td>
    </tr>
  </tbody></table>

## aclnnGridSampler2DBackward

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGridSampler2DBackwardGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnGridSampler2DBackward默认非确定性实现，支持通过aclrtSetSysParamOpt开启确定性。确定性实现需同时满足如下条件：
    - 输入和输出的数据类型不为DOUBLE。
    - 输入和输出的C轴取值小于等于2048。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grid_sampler2d_backward.h"

namespace {
constexpr uint64_t kNum2 = 2U;
}

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t interpolationMode = 0;
    int64_t paddingMode = 0;
    bool alignCorners = false;
    aclBoolArray* outputMask = nullptr;
    std::vector<int64_t> gradOutputShape = {1, 1, 3, 3};
    std::vector<int64_t> inputShape = {1, 1, 5, 8};
    std::vector<int64_t> gridShape = {1, 3, 3, 2};
    std::vector<int64_t> inputGradShape = {1, 1, 5, 8};
    std::vector<int64_t> gridGradShape = {1, 3, 3, 2};
    void* gradOutputDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* gridDeviceAddr = nullptr;
    void* inputGradDeviceAddr = nullptr;
    void* gridGradDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* input = nullptr;
    aclTensor* grid = nullptr;
    aclTensor* inputGrad = nullptr;
    aclTensor* gridGrad = nullptr;

    std::vector<float> gradOutputHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> inputHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    std::vector<float> gridHostData = {-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1};
    std::vector<float> inputGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gridGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    bool maskValue[2] = {true, true};
    outputMask = aclCreateBoolArray(&(maskValue[0]), kNum2);

    // 创建gradOutput aclTensor
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT,
                          &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建grid aclTensor
    ret = CreateAclTensor(gridHostData, gridShape, &gridDeviceAddr, aclDataType::ACL_FLOAT, &grid);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建inputGrad aclTensor
    ret = CreateAclTensor(inputGradHostData, inputGradShape, &inputGradDeviceAddr, aclDataType::ACL_FLOAT, &inputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gridGrad aclTensor
    ret = CreateAclTensor(gridGradHostData, gridGradShape, &gridGradDeviceAddr, aclDataType::ACL_FLOAT, &gridGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnGridSampler2DBackward第一段接口
    ret = aclnnGridSampler2DBackwardGetWorkspaceSize(gradOutput, input, grid, interpolationMode, paddingMode,
                                                     alignCorners, outputMask, inputGrad, gridGrad, &workspaceSize,
                                                     &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2DBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGridSampler2DBackward第二段接口
    ret = aclnnGridSampler2DBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2DBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto inputGradSize = GetShapeSize(inputGradShape);
    std::vector<float> inputGradResultData(inputGradSize, 0);
    ret = aclrtMemcpy(inputGradResultData.data(), inputGradResultData.size() * sizeof(inputGradResultData[0]),
                      inputGradDeviceAddr, inputGradSize * sizeof(inputGradResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy inputGradResultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < inputGradSize; i++) {
        LOG_PRINT("inputGradResultData[%ld] is: %f\n", i, inputGradResultData[i]);
    }

    auto gridGradSize = GetShapeSize(gridGradShape);
    std::vector<float> gridGradResultData(gridGradSize, 0);
    ret = aclrtMemcpy(gridGradResultData.data(), gridGradResultData.size() * sizeof(gridGradResultData[0]),
                      gridGradDeviceAddr, gridGradSize * sizeof(gridGradResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gridGradResultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < gridGradSize; i++) {
        LOG_PRINT("gridGradResultData[%ld] is: %f\n", i, gridGradResultData[i]);
    }

    // 6. 释放aclTensor和aclBoolArray，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(input);
    aclDestroyTensor(grid);
    aclDestroyTensor(inputGrad);
    aclDestroyTensor(gridGrad);
    aclDestroyBoolArray(outputMask);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(gridDeviceAddr);
    aclrtFree(inputGradDeviceAddr);
    aclrtFree(gridGradDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
