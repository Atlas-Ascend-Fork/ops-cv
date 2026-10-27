# PointsInPolygons

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

PointsInPolygons是一个点-四边形从属关系判定算子：给定N个二维点与M个四边形（每个四边形由4个顶点按顺序展开为8个标量），对每个(点，四边形)组合判定该点是否落在该四边形内部（落在边或顶点上视为在外），输出N×M的二值浮点矩阵，`1.0`表示在内、`0.0`表示在外。

  $$
  x\_intersect_k = x_k + \frac{py - y_k}{y_{k+1} - y_k} \times (x_{k+1} - x_k)
  $$

  $$
  c = \sum_{k=0}^{3} ((y_k > py) \neq (y_{k+1} > py)) \land (x\_intersect_k > px)
  $$

  $$
  output_{i,j} = \begin{cases} 1.0, & c \text{ 为奇数} \\ 0.0, & c \text{ 为偶数} \end{cases}
  $$

其中px、py为点i的坐标，x_k、y_k为多边形j的第k个顶点坐标。

上述公式有两点补充约定：

- 边界优先：点落在四边形任一边或任一顶点上时视为在边界上，输出直接取 `0.0`，不再按交点数的奇偶判定。
- 跨越条件：仅对两端 y 坐标跨过 py 的边（即满足 `(y_k > py) ≠ (y_{k+1} > py)` 的边）计算 `x_intersect_k` 并计入交点数；水平边（`y_{k+1} = y_k`）不满足跨越条件，跳过交点计算，不会出现除以零。

## 参数说明

- **参数说明**：

  <table style="table-layout: fixed; width: 1576px">
  <colgroup>
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
  </tr>
  </thead>
  <tbody>
  <tr>
  <td>points</td>
  <td>输入</td>
  <td>表示待判定的二维点集，每行(x,y)为一个二维点坐标，对应公式中px、py的来源。</td>
  <td>FLOAT</td>
  <td>ND</td>
  </tr>
  <tr>
  <td>polygons</td>
  <td>输入</td>
  <td>表示待判定的四边形集合，shape为(8,M)：第0维固定8，按(x0,y0,x1,y1,x2,y2,x3,y3)顺序存放4个顶点的坐标分量；第1维为M个多边形。对应公式中x_k、y_k的来源。</td>
  <td>FLOAT</td>
  <td>ND</td>
  </tr>
  <tr>
  <td>output</td>
  <td>输出</td>
  <td>表示点-四边形从属判定结果，元素为1.0（在内）或0.0（在外），对应公式中output。</td>
  <td>FLOAT</td>
  <td>ND</td>
  </tr>
  </tbody>
  </table>

## 约束说明

- points与polygons的数据类型必须均为FLOAT，不支持其他数据类型。
- points、polygons与output的数据格式必须均为ND。
- points与polygons的rank必须均为2；非空Tensor时points第1维必须固定为2，polygons第0维必须固定为8。
- output的shape必须为(N,M)，其中N=points.shape[0]、M=polygons.shape[1]，N与M独立，无broadcast关系。
- 支持空Tensor：当N=0或M=0时，output为空Tensor（元素数为0），接口不报错，直接返回。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_points_in_polygons](examples/arch35/test_geir_points_in_polygons.cpp)  | 通过[算子IR](op_graph/points_in_polygons_proto.h)构图方式调用PointsInPolygons算子。         |
