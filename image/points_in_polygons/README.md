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

PointsInPolygons是一个点-四边形从属关系判定算子：给定N个二维点与M个四边形（每个四边形由4个顶点按顺序展开为8个标量），对每个(点，四边形)组合判定该点是否落在该四边形内部，输出N×M的二值浮点矩阵，`1.0`表示在内、`0.0`表示在外。

  $$
  x\_intersect_k = x_k + \frac{(py - y_k) \times (x_{k-1} - x_k)}{y_{k-1} - y_k}
  $$

  $$
  c = \sum_{k=0}^{3} ((y_k < py \land y_{k-1} \ge py) \lor (y_k \ge py \land y_{k-1} < py)) \land (x\_intersect_k > px)
  $$

  $$
  output_{i,j} = \begin{cases} 1.0, & c \text{ 为奇数} \\ 0.0, & c \text{ 为偶数} \end{cases}
  $$

其中px、py为点i的坐标，x_k、y_k为多边形j的第k个顶点坐标。

按k=0、1、2、3遍历顶点，k-1按模4取值。命中顶点，或跨越边的交点横坐标与px精确相等时，立即结束该组合的遍历，按此前累计交点数的奇偶输出。因此边界点可能输出`0.0`或`1.0`，与顶点排列及遍历顺序有关。

交点按float32先乘后除计算，仅跨越边参与计算，水平边跳过。比较采用精确相等判断，不引入epsilon；近边界结果可能与float64几何参考不同。除法使用正确舍入的float32实现，包括次正规路径。

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

- 输入输出仅支持FLOAT（float32），格式均为ND。
- points与polygons均须为二维Tensor，实际输入各维度必须非负。
- output.shape=(N,M)，其中N=points.shape[0]、M=polygons.shape[1]。动态声明含-1时推导为(-1,-1)，实际shape由tiling校验。
- N与M均非0时，points.shape[1]必须为2、polygons.shape[0]必须为8；N=0或M=0时返回空输出，并豁免这两个固定轴检查。
- 参与计算的polygons须为连续(8,M)布局，每列一个四边形；(M,8)布局须先转置并连续化。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE图模式 | [test_geir_points_in_polygons](examples/arch35/test_geir_points_in_polygons.cpp)  | 通过[算子IR](op_graph/points_in_polygons_proto.h)构图方式调用PointsInPolygons算子。         |
