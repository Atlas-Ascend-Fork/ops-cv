# Dilation2DBackpropInput

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

- 算子功能：形态学膨胀操作Dilation2D的反向传播（输入梯度）算子。给定前向输入`x`、滤波器`filter`和输出梯度`out_backprop`，计算输入梯度`in_backprop`。

- 计算公式：

前向Dilation2D公式：

$$
output[b, h_{out}, w_{out}, d] = \max_{fh, fw} \left( x[b, h_{out} \cdot stride_h + fh \cdot rate_h, w_{out} \cdot stride_w + fw \cdot rate_w, d] + filter[fh, fw, d] \right)
$$

反向BackpropInput公式：

$$
in\_backprop[b, *, *, d] = 0
$$

$$
in\_backprop[b, h_{in\_max}, w_{in\_max}, d] \mathrel{+}= out\_backprop[b, h_{out}, w_{out}, d]
$$

其中$(h_{in\_max}, w_{in\_max})$为前向计算时窗口内`x + filter`取最大值的位置（使用`>`严格大于，选第一个最大值），越界位置不参与argmax。

## 参数说明

<table><thead>
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
    <td>前向输入张量，4D，公式中的x。NHWC格式时shape为(N, H, W, C)，NCHW格式时shape为(N, C, H, W)。</td>
    <td>float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>filter</td>
    <td>输入</td>
    <td>滤波器，3D，公式中的filter。NHWC格式时shape为(Hf, Wf, C)，NCHW格式时shape为(C, Hf, Wf)。</td>
    <td>float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out_backprop</td>
    <td>输入</td>
    <td>输出梯度，4D，shape与前向输出一致。NHWC格式时shape为(N, Ho, Wo, C)，NCHW格式时shape为(N, C, Ho, Wo)。</td>
    <td>float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>输入梯度，4D，shape和dtype与x相同，公式中的in_backprop。</td>
    <td>float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>strides</td>
    <td>属性</td>
    <td>滑动窗口步长，长度为4的列表。N/C维度必须为1。</td>
    <td>ListInt</td>
    <td>-</td>
  </tr>
  <tr>
    <td>rates</td>
    <td>属性</td>
    <td>膨胀率，长度为4的列表。N/C维度必须为1。</td>
    <td>ListInt</td>
    <td>-</td>
  </tr>
  <tr>
    <td>padding_mode</td>
    <td>属性</td>
    <td>padding模式，取值为"SAME"、"VALID"或"CALCULATED"。默认值为"SAME"。</td>
    <td>String</td>
    <td>-</td>
  </tr>
  <tr>
    <td>pads</td>
    <td>属性</td>
    <td>显式padding值[pad_top, pad_bottom, pad_left, pad_right]，仅CALCULATED模式生效。默认值为{0, 0, 0, 0}。</td>
    <td>ListInt</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ceil_mode</td>
    <td>属性</td>
    <td>CALCULATED模式下是否用ceil计算输出尺寸，true为ceil，false为floor。默认值为false。</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>data_format</td>
    <td>属性</td>
    <td>数据格式，取值为"NHWC"或"NCHW"。默认值为"NHWC"。</td>
    <td>String</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- **数据类型约束**：x、filter、out_backprop、y四者dtype相同，仅支持float32。
- **维度约束**：x为4D（rank=4），filter为3D（rank=3），out_backprop为4D（rank=4）。
- **shape约束**：
  - x的C维度必须等于filter的depth维度（NHWC: `x.dim(3) == filter.dim(2)`；NCHW: `x.dim(1) == filter.dim(0)`）。
  - out_backprop的N/C维度与x一致，Ho/Wo由x的H/W、strides、rates、padding_mode推导。
  - 输出y的shape与x完全相同。
- **属性约束**：
  - strides长度必须为4，且N/C维度必须为1（stride_n=1, stride_c=1）。
  - rates长度必须为4，且N/C维度必须为1（rate_n=1, rate_c=1）。
  - padding_mode取值必须为{"SAME", "VALID", "CALCULATED"}之一。
  - pads长度必须为4，CALCULATED模式下pads[i]需小于window_h/w（window_h = (filter_h - 1) * rate_h + 1）。
  - data_format取值必须为{"NHWC", "NCHW"}之一。
- **format约束**：x、filter、out_backprop、y的GE内存布局格式均为ND；data_format仅定义4D shape的维度顺序，NHWC为(N,H,W,C)，NCHW为(N,C,H,W)。
- **值域约束**：argmax比较使用`>`严格大于，选择第一个最大值对应的位置进行梯度累加，相同值不累加到多个位置。CALCULATED模式下padding区域使用极小值填充，确保不参与argmax。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_dilation2_d_backprop_input.cpp">test_geir_dilation2_d_backprop_input</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
