# ImgRawDecodePostHandle

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

- 算子功能：将YUV格式的4通道图像数据转换为RAW Bayer格式输出，同时对每个通道进行伽马校正（Gamma Correction）和黑电平补偿（Black Level Correction）。支持`binning`和`quad`两种Bayer排列模式。

- 计算公式：

$$
v_{f32} = \frac{v_{u16}}{64.0}
$$

$$
mask = clamp(v_{f32}, 55.0, 56.0) - 55.0
$$

$$
d = \frac{v_{f32} - 56.0}{967.0} \times mask
$$

$$
g =
\begin{cases}
\exp(\ln(\max(d, 10^{-38})) \times \gamma), & d > 0, \\
\exp(-\infty \times \gamma), & d \le 0\; (\gamma > 0\text{ 时该项为 }0).
\end{cases}
$$

$$
d = g \times 967.0 + 56.0
$$

$$
d = d \times mask
$$

$$
result = v_{f32} \times (1 - mask) + d
$$

$$
raw\_img = uint16(round(result))
$$

4通道按Bayer pattern交织为2D RAW图像输出。

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
    <td>img_channel_0</td>
    <td>输入</td>
    <td>通道0 YUV图像数据，shape为(h_gm, w_gm)，公式中的v_u16。</td>
    <td>UINT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>img_channel_1</td>
    <td>输入</td>
    <td>通道1 YUV图像数据，shape为(h_gm, w_gm)，公式中的v_u16。</td>
    <td>UINT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>img_channel_2</td>
    <td>输入</td>
    <td>通道2 YUV图像数据，shape为(h_gm, w_gm)，公式中的v_u16。</td>
    <td>UINT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>img_channel_3</td>
    <td>输入</td>
    <td>通道3 YUV图像数据，shape为(h_gm, w_gm)，公式中的v_u16。</td>
    <td>UINT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>img_size</td>
    <td>输入</td>
    <td>输出图像尺寸，shape为(2,)，值为[h_out, w_out]。输出shape由此tensor的元素值决定。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma</td>
    <td>输入</td>
    <td>4通道伽马校正值，shape为(4,)。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bayer_pattern</td>
    <td>属性</td>
    <td>Bayer排列模式，取值为"binning"或"quad"，默认值为"binning"。决定通道交织方式和输出尺寸约束。</td>
    <td>String</td>
    <td>-</td>
  </tr>
  <tr>
    <td>raw_img</td>
    <td>输出</td>
    <td>RAW Bayer图像，shape为(h_out, w_out)。</td>
    <td>UINT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- img_channel_0、img_channel_1、img_channel_2、img_channel_3的rank必须为2，且4个通道shape必须一致。
- img_size的shape必须为(2,)，值为[h_out, w_out]，且实际数据必须包含2个int32元素，否则inferShape报错。
- img_size的值必须为正整数，即h_out和w_out必须大于0。
- gamma的shape必须为(4,)。
- binning模式下h_out和w_out必须为2的倍数。
- quad模式下h_out和w_out必须为4的倍数。
- 输出宽度w_out必须大于等于16。
- h_gm必须大于等于h_out/2，w_gm必须大于等于w_out/2。
- 不支持空tensor。
- 非连续tensor会自动转换为连续tensor后处理。
- 该算子输出为uint16整数类型，gamma校正路径涉及ln/exp浮点运算，允许atol=2的整数容差比对。

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
    <td><a href="./examples/test_geir_img_raw_decode_post_handle.cpp">test_geir_img_raw_decode_post_handle</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
