# IMGWarpResize

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

- 算子功能：图像双线性插值采样算子，用于OCR场景中的图像变形缩放。接收预处理好的四角像素值和浮点坐标，通过双线性插值计算输出像素值。

- 计算公式：

对每个像素位置 (n, c, h, w)：

$$
x_{lerp} = x_{float} - \lfloor x_{float} \rfloor
$$

$$
y_{lerp} = y_{float} - \lfloor y_{float} \rfloor
$$

$$
top = top_{left} + (top_{right} - top_{left}) \times x_{lerp}
$$

$$
bottom = bottom_{left} + (bottom_{right} - bottom_{left}) \times x_{lerp}
$$

$$
warp_{img} = top + (bottom - top) \times y_{lerp}
$$

其中 $x_{float}$、$y_{float}$ 来自warp_index，四角像素值来自img的维度1(顺序为top_left, top_right, bottom_left, bottom_right)。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 310px">
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
      <td>img</td>
      <td>输入</td>
      <td>四角像素值，shape为[N, 4, C, H, W]，维度1的4=[top_left, top_right, bottom_left, bottom_right]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>warp_index</td>
      <td>输入</td>
      <td>浮点坐标，shape为[N, 2, H, W]，维度1的2=[x_float, y_float]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>warp_img</td>
      <td>输出</td>
      <td>双线性插值结果，shape为[N, C, H, W]，dtype与img一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- img必须为5D张量[N, 4, C, H, W]，维度1必须为4。
- warp_index必须为4D张量[N, 2, H, W]，维度1必须为2。
- img与warp_index的N、H、W维度必须一致。
- 不支持空tensor(N、C、H、W均不能为0)。
- 不支持非连续Tensor输入。
- warp_index始终为float32类型。
- 所有中间计算在float32 精度下进行，float16输入先转换为float32计算再转回float16输出。
- 输出总元素数N*C*H*W不超过INT32_MAX。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_img_warp_resize](examples/arch35/test_geir_img_warp_resize.cpp)  | 通过[算子IR](op_graph/img_warp_resize_proto.h)构图方式调用IMGWarpResize算子。         |
