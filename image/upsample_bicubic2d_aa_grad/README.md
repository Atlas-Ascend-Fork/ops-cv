# UpsampleBicubic2dAAGrad

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

- 算子功能：[UpsampleBicubic2dAA](../upsample_bicubic2d_aa/README.md)的反向传播。如果输入张量的shape为(N, C, H, W)，则输出张量的shape为(N, C, inputSize[2], inputSize[3])。

- 计算公式：对于一个二维插值点$(N, C, h, w)$，插值$gradInput(N, C, h, w)$可以表示为：

  $$
  {gradInput(N, C, h, w)}=\sum_{r: h\in R_H(r)}\sum_{s: w\in R_W(s)}
  {gradOutput(N, C, r, s)} W_H(h|r) W_W(w|s)
  $$

  $$
  scaleH =\begin{cases}
  (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  inputSize[2] / outputSize[0] & otherwise
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  inputSize[3] / outputSize[1] & otherwise
  \end{cases}
  $$

  其中：
  - alignCorners为true，表示输入和输出张量的角像素点对齐；alignCorners为false，表示输入和输出张量的边像素点对齐。
  - $r$和$s$是gradOutput的H、W坐标；$h$和$w$是gradInput的H、W坐标。$gradInput_H$和$gradInput_W$分别表示gradInput的H、W长度。
  - 对方向$D\in\{H,W\}$、gradOutput坐标$q$，中心坐标为：

    $$
    c_D(q)=\begin{cases}
    (q+0.5)scale_D & scale_D>0 \\
    0.5 & otherwise
    \end{cases}
    $$

  - 有效gradInput坐标范围按support计算并裁剪到gradInput边界：

    $$
    R_D(q)=\left[\max(0,\lfloor c_D(q)-support_D+0.5\rfloor),\,
    \min(gradInput_D,\lfloor c_D(q)+support_D+0.5\rfloor)\right)
    $$

    其中$support_D=2scale_D$（$scale_D\geq1$），否则$support_D=2$。因此scale不小于1时，参与计算的窗口会随scale扩大。

  - 权重距离缩放系数为$invScale_D=1/scale_D$（$scale_D\geq1$），否则$invScale_D=1$。方向$D$上位置$x\in R_D(q)$的归一化权重为：

    $$
    \widetilde W_D(x|q)=CubicFilterAA((x-c_D(q)+0.5)invScale_D),\qquad
    W_D(x|q)=\frac{\widetilde W_D(x|q)}{\sum_{t\in R_D(q)}\widetilde W_D(t|q)}
    $$

  - $CubicFilterAA(x)$是双三次核，定义为：

    $$
    CubicFilterAA(x)=\begin{cases}
    (a+2)|x|^3-(a+3)|x|^2+1 & |x|\leq1 \\
    a|x|^3-5a|x|^2+8a|x|-4a & 1<|x|<2 \\
    0 & otherwise
    \end{cases}
    $$

    其中抗锯齿场景$a=-0.5$。

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
      <td>表示反向计算的梯度Tensor，对应公式描述中的`gradOutput`。数据类型与输出`grad_input`的数据类型一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>表示输入`grad_output`在H和W维度上的空间大小，对应公式中的`outputSize`。size为2，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_size</td>
      <td>属性</td>
      <td>表示输出`grad_input`分别在N、C、H和W维度上的空间大小，对应公式中的`inputSize`。size为4，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td><ul><li>决定是否对齐角像素点，对应公式中的`alignCorners`。align_corners为true，则输入和输出张量的角像素点会被对齐，否则不对齐。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>可选属性</td>
      <td><ul><li>表示输出`grad_input`的height维度乘数，对应公式中的`scalesH`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>可选属性</td>
      <td><ul><li>表示输出`grad_input`的width维度乘数，对应公式中的`scalesW`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式中的`gradInput`。数据类型与输入`grad_output`的数据类型一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_bicubic2d_aa_grad](examples/test_aclnn_upsample_bicubic2d_aa_grad.cpp) | 通过[aclnnUpsampleBicubic2dAAGrad](docs/aclnnUpsampleBicubic2dAAGrad.md)接口方式调用UpsampleBicubic2dAAGrad算子。 |
