# Dilation2DBackpropFilter

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：Dilation2DBackpropFilter 是形态学膨胀（Morphological Dilation）2D 操作的反向传播中计算 filter 梯度的算子。对 out_backprop 的每个位置，重新计算前向 Dilation2D 找到 argmax 的 filter 窗口位置，然后将梯度原子累加到对应的 filter 位置。

- 计算公式：

对于每个输出位置 (b, h_out, w_out, d)（以下以 NHWC 为例，NCHW 时维度索引相应调整）：

$$
h_{beg} = h_{out} \times stride_h - pad_{top}
$$

$$
w_{beg} = w_{out} \times stride_w - pad_{left}
$$

$$
cur\_val = \min(T), \quad h_{max} = 0, \quad w_{max} = 0
$$

$$
\text{for } h \in [0, filter_h), w \in [0, filter_w):
$$

$$
\quad h_{in} = h_{beg} + h \times rate_h, \quad w_{in} = w_{beg} + w \times rate_w
$$

$$
\quad \text{if } 0 \le h_{in} < H_{pad} \text{ and } 0 \le w_{in} < W_{pad}:
$$

$$
\quad \quad x_{val} = \begin{cases} x[b, h_{in}, w_{in}, d], & h_{in} < H_{in} \text{ and } w_{in} < W_{in} \\ 0, & \text{其他（padding 区域）} \end{cases}
$$

$$
\quad \quad val = x_{val} + filter[h, w, d]
$$

$$
\quad \quad \text{if } val > cur\_val: \quad cur\_val = val, \quad h_{max} = h, \quad w_{max} = w
$$

$$
y[h_{max}, w_{max}, d] \mathrel{+}= out\_backprop[b, h_{out}, w_{out}, d]
$$

其中 $H_{pad}$、$W_{pad}$ 为窗口位置的参与边界：CALCULATED 模式下 $H_{pad} = H_{in} + pad_{top} + pad_{bottom}$、$W_{pad} = W_{in} + pad_{left} + pad_{right}$，落在原输入外的下/右 padding 区域位置以 $x = 0$ 参与 argmax；SAME/VALID 模式下 $H_{pad} = H_{in}$、$W_{pad} = W_{in}$，越界位置不参与。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>输入特征图，4D张量。NHWC格式为(N, H_in, W_in, C)，NCHW格式为(N, C, H_in, W_in)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>输入</td>
      <td>膨胀滤波器，3D张量。NHWC格式为(filter_h, filter_w, C)，NCHW格式为(C, filter_h, filter_w)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_backprop</td>
      <td>输入</td>
      <td>前向输出的梯度，4D张量。NHWC格式为(N, H_out, W_out, C)，NCHW格式为(N, C, H_out, W_out)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>filter的梯度，3D张量。NHWC格式为(filter_h, filter_w, C)，NCHW格式为(C, filter_h, filter_w)，shape与filter相同。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>滑动窗口步长，4元素列表。NHWC格式为[1, stride_h, stride_w, 1]，NCHW格式为[1, 1, stride_h, stride_w]，stride_h/stride_w >= 1。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rates</td>
      <td>属性</td>
      <td>膨胀率，4元素列表。NHWC格式为[1, rate_h, rate_w, 1]，NCHW格式为[1, 1, rate_h, rate_w]，rate_h/rate_w >= 1。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>属性</td>
      <td>填充模式："SAME" / "VALID" / "CALCULATED"。默认"SAME"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>显式padding值[pad_top, pad_bottom, pad_left, pad_right]，仅CALCULATED时生效。默认{0,0,0,0}。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>属性</td>
      <td>是否使用ceil函数计算输出尺寸。默认false。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>数据格式："NHWC"或"NCHW"。默认"NHWC"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入 x 必须为4维张量，filter 必须为3维张量 (filter_h, filter_w, C)，out_backprop 必须为4维张量。NHWC格式下 x 为 (N, H_in, W_in, C)、out_backprop 为 (N, H_out, W_out, C)；NCHW格式下 x 为 (N, C, H_in, W_in)、out_backprop 为 (N, C, H_out, W_out)。
- 三个输入的 dtype 必须相同，输出 y 的 dtype 与输入一致。
- x 的 C 维必须与 filter 的 C 维以及 out_backprop 的 C 维一致。
- out_backprop 的 N 维必须与 x 的 N 维一致（两维均已知且不相等时拒绝）。
- x 与 filter 的空间维静态已知时，out_backprop 的 H/W 必须等于由 x、filter、strides、rates、padding_mode 推导的前向输出尺寸（动态维度 -1 不受影响）。
- strides[0]=1 且 strides[3]=1（仅空间维度步进，NHWC格式）；NCHW格式下要求 strides[0]=1 且 strides[1]=1。
- rates[0]=1 且 rates[3]=1（仅空间维度膨胀，NHWC格式）；NCHW格式下要求 rates[0]=1 且 rates[1]=1。
- 支持 NHWC 和 NCHW 两种数据格式，默认使用 NHWC。
- 支持空 tensor（shape 含0维），输出为对应的全零 tensor。
- 支持动态 Shape（DynamicCompileStatic / DynamicRank / DynamicShape）。
- 默认为非确定模式（多位置梯度经原子累加，浮点加法顺序不定）；仅在开启确定性开关时为确定性（per-thread buffer + 确定性归约，无 atomicAdd）。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>样例代码</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE图模式</td>
    <td><a href="./examples/test_geir_dilation2_d_backprop_filter.cpp">test_geir_dilation2_d_backprop_filter</a></td>
    <td>通过算子IR构图方式调用，IR定义参见<a href="./op_graph/dilation2_d_backprop_filter_proto.h">dilation2_d_backprop_filter_proto.h</a>；参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
