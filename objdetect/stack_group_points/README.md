# StackGroupPoints

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×   |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：根据特征点所属的组，重组点云的特征。
- 计算公式：点云数据`features`形状为$(N1 + N2 ..., C)$，根据`features_batch_cnt`形状$(N1, N2, ...)$将其分组，同样地，`indices`形状为$(M1 + M2 ..., nsample)$，根据`indices_batch_cnt`形状$(M1, M2, ...)$将其分组。算子根据`indices`分组从对应的`features`组中抽取特征成输出`out`，实现如下功能：
  $$
  out[m:c:np] = features[indices[m:np]:c]
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>features</td>
      <td>输入</td>
      <td>待重组的点云特征。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>features_batch_cnt</td>
      <td>输入</td>
      <td>每个批次中包含的点云特征数量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>重组indices。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices_batch_cnt</td>
      <td>输入</td>
      <td>每个批次中包含的重组indices数量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出点云特征，对应公式中的out。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `features` 和 `indices` 必须为二维 Tensor；`features_batch_cnt` 和 `indices_batch_cnt` 必须为一维 Tensor。
- `features` 的通道维 C、`indices` 的 nsample 维以及 batch 计数长度 B 必须为正数。
- `features_batch_cnt` 与 `indices_batch_cnt` 的长度必须相等，且分别描述同一批次的特征数量和索引数量。
- 每个批次的 `indices` 值必须位于对应批次 `features` 数量范围内；越界索引的输出为零。

## 调用说明

| 调用方式   | 样例代码                                                                        | 说明                                         |
| ---------------- |-----------------------------------------------------------------------------| --------------------------------------------------- |
| 图模式 | [test_geir_stack_group_points](./examples/test_geir_stack_group_points.cpp) | 通过[算子IR](./op_graph/stack_group_points_proto.h)构图方式调用StackGroupPoints算子。         |
