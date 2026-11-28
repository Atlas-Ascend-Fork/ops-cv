# Col2im

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

- 算子功能：实现Im2col的反向映射（Col2Im），将按滑窗展开的gradOutput重排为inputSize指定的梯度张量。多个滑窗映射到同一输出位置时，对相应元素求和。
- 计算公式：

  考虑一个形状为$(C×\prod(kernel_size),L)$或$(N,C×\prod(kernel_size),L)$的gradOutput张量，其中$N$是批处理维度，$C$是通道维度。

  此操作将gradOutput重排为形状为$(C,spatial\_size[0],spatial\_size[1])$或$(N,C,spatial\_size[0],spatial\_size[1])$的out张量。

  其中：
  - $C×\prod(kernel_size)$是每个滑窗块内的值的数量，而$L$是这些块的总数：

    $$
    L=\prod_d⌊{\frac{spatial\_size[d]+2×padding[d]−dilation[d]×(kernel\_size[d]−1)−1}{stride[d]}+1}⌋
    $$

  - spatial_size由inputSize构成，而$d$覆盖所有空间维度。
  因此，gradOutput的最后一个维度为列维度，该维度上的每个索引对应一个滑窗块；这些滑窗块映射到out时，重叠位置的元素进行累加。

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
      <td>gradOutput</td>
      <td>输入</td>
      <td>反向输入，对应公式中的output。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputSize</td>
      <td>属性</td>
      <td>输入张量的形状，对应公式中的spatial_size。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>属性</td>
      <td>卷积核的大小，对应公式中的kernel_size。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>属性</td>
      <td>膨胀参数，对应公式中的dilation。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性</td>
      <td>卷积的填充大小，对应公式中的padding。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>属性</td>
      <td>卷积的步长，对应公式中的stride。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>梯度计算结果，对应公式中的input。数据类型和数据格式需要与`gradOutput`的数据类型和数据格式一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_im2col_backward](examples/test_aclnn_im2col_backward.cpp) | 通过[aclnnIm2colBackward](docs/aclnnIm2colBackward.md)接口方式调用Col2im算子。 |
