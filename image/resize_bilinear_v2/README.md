# ResizeBilinearV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：使用双线性插值调整图像大小到指定的大小。

- 计算公式：

  假设x中已知四个点$Q_{11}(x_1, y_1), Q_{12}(x_1, y_2), Q_{21}(x_2, y_1), Q_{22}(x_2, y_2)$的值，下面对数据进行插值，得到y中的一个点坐标$P(a, b)$中的值。

  $$
  f(R_1) = \frac{x_2 - a}{x_2 - x_1}f(Q_{11}) + \frac{a - x_1}{x_2 - x_1}f(Q_{21})
  $$

  $$
  f(R_2) = \frac{x_2 - a}{x_2 - x_1}f(Q_{12}) + \frac{a - x_1}{x_2 - x_1}f(Q_{22})
  $$

  $$
  f(P) = \frac{y_2 - b}{y_2 - y_1}f(R_{1}) + \frac{b - y_1}{y_2 - y_1}f(R_{2})
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
      <td>x</td>
      <td>输入</td>
      <td>输入图像的四维Tensor，对应公式中x。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>输出图像的高和宽。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td>为true时对齐输入和输出图像四角像素的中心；默认为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>half_pixel_centers</td>
      <td>可选属性</td>
      <td>为true时按半像素中心坐标进行采样；不能与align_corners同时为true，默认为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>可选属性</td>
      <td>指定输出y的数据类型，支持FLOAT32、FLOAT16、BFLOAT16；默认为FLOAT32。</td>
      <td>TYPE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>可选属性</td>
      <td>指定H、W方向的缩放系数，默认值为[0.0, 0.0]，表示根据size计算缩放比例。</td>
      <td>LIST_FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>双线性插值调整后图像，对应公式中y。数据格式与x一致。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、NHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- `align_corners`和`half_pixel_centers`不能同时为true。
- `scales`语义与合法域：必须恰为2个元素，依次为H、W方向的缩放系数（输出长度与输入长度之比），默认值`[0.0, 0.0]`表示不覆盖。它**不改变输出shape**——`y`的H、W恒由`size`经shape推导得出，`size`不会被忽略；`scales`只覆盖坐标映射所用的缩放系数：某一元素大于0.0时该方向按`1/scales`映射，否则回退为`H_in/H_out`（W方向同理）。`align_corners`为true时该覆盖不生效，缩放系数固定为`(H_in-1)/(H_out-1)`。元素个数不等于2时Tiling校验失败。
- 固定shape约束：`x`与`y`均必须为4维Tensor，且`y`的N、C维与`x`相同，数据格式须一致（同为NCHW或同为NHWC）。
- 空Tensor约束：`x`与`y`均不支持空Tensor。当任一元素数为0时，Tiling阶段校验失败并返回`GRAPH_FAILED`。需注意aclnn直调通路（aclnnResize）的入参校验**不含**专门的空Tensor分支，因此不会返回特定的L2错误码，空Tensor最终在Tiling阶段被拒绝、表现为执行期失败。
- `size`为值依赖输入，必须是1维、2元素的INT32 Tensor，依次为输出的高（new_height）与宽（new_width），且两元素均须为正。
- 非连续Tensor：GE图模式下`x`、`y`要求为连续Tensor；aclnn直调通路（aclnnResize）支持非连续输入输出，接口内部会先做连续化转换，详见[aclnnResize](docs/aclnnResize.md)。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_resize](examples/test_aclnn_resize.cpp) | 通过[aclnnResize](docs/aclnnResize.md)接口方式调用ResizeBilinearV2算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/resize_bilinear_v2_proto.h)构图方式调用ResizeBilinearV2算子。         |
