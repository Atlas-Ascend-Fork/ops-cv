# ResizeGrad

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

- 算子功能：计算Resize正向算子的反向梯度，将resize后的梯度grads按linear（1D）或cubic（2D）插值权重累加回原始分辨率的每个采样位置，输出与正向输入同shape的梯度y。

- 计算公式（cubic模式，输出点$[h,w]$反向聚合覆盖它的输入窗口）：

$$
y[n,c,h,w] = \sum_{(o_h,o_w)\in S(h,w)} grads[n,c,o_h,o_w] \cdot c_h(o_h) \cdot c_w(o_w)
$$

  其中$c_h$、$c_w$为三次卷积插值系数（系数a固定为-0.75），$S(h,w)$为覆盖输出点$[h,w]$的输入采样窗口；linear模式为1D线性插值的两点加权累加。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 190px">
  <col style="width: 110px">
  <col style="width: 300px">
  <col style="width: 230px">
  <col style="width: 150px">
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
      <td>grads</td>
      <td>输入</td>
      <td>resize后的梯度，4维Tensor。cubic模式NCHW布局为(N,C,H_out,W_out)，HWNC布局为(H_out,W_out,N,C)；linear模式为(N,C,1,W_out)。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>roi</td>
      <td>可选输入</td>
      <td>张量，仅在coordinate_transformation_mode为"tf_crop_and_resize"时生效，本次不支持该模式，不使用。</td>
      <td>FLOAT16、FLOAT32、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>可选输入</td>
      <td>1维float Tensor，各维度缩放系数，cubic模式2元（1元时广播到H/W）、linear模式1元。有限正数生效（缩放系数为1/scales[i]）；缺省、非有限值（NaN/±Inf）或元素不大于0时按尺寸推导。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>original_size</td>
      <td>输入</td>
      <td>1维int32/int64 Tensor，输出Tensor的尺寸。cubic模式4元[N,C,H_in,W_in]（NCHW语义），linear模式3元[N,C,W_in]。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>coordinate_transformation_mode</td>
      <td>属性</td>
      <td>resize后坐标到原始坐标的变换方式，本次仅支持"half_pixel"和"align_corners"，默认"half_pixel"。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubic_coeff_a</td>
      <td>属性</td>
      <td>cubic插值系数，仅cubic模式使用；linear模式忽略；cubic模式下固定-0.75且必须为有限值（NaN/±Inf会被拒绝）。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>exclude_outside</td>
      <td>属性</td>
      <td>保留属性，默认0，本次不支持，不参与计算。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>extrapolation_value</td>
      <td>属性</td>
      <td>保留属性，默认0.0，仅"tf_crop_and_resize"模式生效，本次不支持，不参与计算。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>属性</td>
      <td>插值模式，本次仅支持"linear"和"cubic"，默认"nearest"但不支持（传入即报错）。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nearest_mode</td>
      <td>属性</td>
      <td>保留属性，默认"round_prefer_floor"，仅nearest模式使用，本次不支持，不参与计算。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>计算时输入数据的布局，支持"NCHW"和"HWNC"，默认"NCHW"；linear模式仅支持"NCHW"。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>正向输入的梯度，shape由original_size推导，dtype与grads相同。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- grads：仅支持FLOAT16、FLOAT32，必须为4维。cubic模式NCHW布局为(N,C,H_out,W_out)，HWNC布局为(H_out,W_out,N,C)；linear模式为(N,C,1,W_out)且第3维（dim(2)）必须为1。H_out或W_out允许为0，此时输出按0处理。grads中的非有限输入（±Inf、NaN）不在数值保证范围内。
- y：dtype与grads一致，shape由original_size推导；data_format为"NCHW"时y为(N,C,H_in,W_in)，为"HWNC"时y为(H_in,W_in,N,C)。
- original_size：值依赖输入，各元素必须大于0，前2维（N、C）须与grads一致；cubic模式必须为4元[N,C,H_in,W_in]，linear模式必须为3元[N,C,W_in]。
- scales：可选；cubic模式2元（1元时广播到H/W）、linear模式1元；有限正数生效（缩放系数为1/scales[i]）；极小的有限正数可能被拒绝；缺省、非有限值（NaN/±Inf）或元素不大于0时按输入输出尺寸推导；linear模式下scales[0]须约等于grads.W/original_size.W（容差1e-4）。
- mode：仅支持"linear"和"cubic"；默认值"nearest"及其他取值不支持，必须显式传入"linear"或"cubic"。
- data_format：支持"NCHW"和"HWNC"，默认"NCHW"；linear模式仅支持"NCHW"。
- coordinate_transformation_mode：仅支持"half_pixel"和"align_corners"，默认"half_pixel"。
- cubic_coeff_a：仅cubic模式校验；cubic模式下固定-0.75且必须为有限值（NaN/±Inf会被拒绝）；linear模式忽略该属性。
- roi、exclude_outside、extrapolation_value、nearest_mode：保留参数，本次不参与计算。
- 动态Shape：支持-1未知维和-2未知秩；输出形状未知时y的shape为未知；original_size必须为常量输入才能推导具体输出形状，否则y的shape为未知。

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
    <td><a href="./examples/test_geir_resize_grad.cpp">test_geir_resize_grad</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证；算子IR定义见<a href="./op_graph/resize_grad_proto.h">resize_grad_proto.h</a>。</td>
  </tr>
</tbody>
</table>
