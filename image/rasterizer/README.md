# Rasterizer

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     ×    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：实现光栅化计算。根据给定的三维空间中的点和面，获取屏幕中每个像素点的最小深度及其对应的面片索引，并计算该面片的重心坐标透视矫正插值。

- 计算公式：

  $findices$记录每个像素点最小深度对应的面索引，$barycentric$记录每个顶点相对于$findices$中记录的面的重心坐标透视矫正插值。
  计算过程中使用的Z-Buffer记录每个像素点$(x, y)$的最小深度$z_{\min}(x, y)$以及该深度对应的三角形面片索引$\text{face\_idx}(x, y)$。

  计算过程如下：
  对空间中的每个三角形面片$f$：

  1. 将$f$的三个顶点坐标$(v_0, v_1, v_2)$转换为屏幕坐标$(v_{s0}, v_{s1}, v_{s2})$
  2. 根据$(v_{s0}, v_{s1}, v_{s2})$计算包围$f$的矩形范围
  3. 对矩形内每个像素点$v_i = (x_i, y_i)$，执行以下操作：

     a. 计算像素中心坐标$v_c$
     b. 计算$v_c$相对于三角形$f$的重心坐标$(\alpha, \beta, \gamma)$
     c. 根据$(\alpha, \beta, \gamma)$判断$v_c$是否在三角形内部。若$v_c$不在三角形内部，则处理矩形内下个像素点，否则执行下述步骤
     d. 使用$(\alpha, \beta, \gamma)$和$(v_{s0}, v_{s1}, v_{s2})$得到当前像素的深度值depth
     e. 若启用了深度先验，则使用深度先验图计算深度阈值depth_thres；否则，直接执行下一步“Z-Buffer更新”

        - 使用深度先验图计算深度阈值depth_thres
        - 如果depth < depth_thres，处理矩形内下个像素点，否则执行下述步骤

     f. Z-Buffer更新：

        - 若$depth < z_{\min}(x_i, y_i)$：

          $$
          \quad z_{\min}(x_i, y_i) \gets \text{depth} \\
          \quad \text{face\_idx}(x_i, y_i) \gets f
          $$

        - 若$depth = z_{\min}(x_i, y_i)$：

          $$
          \quad \text{face\_idx}(x_i, y_i) \gets \min(\text{face\_idx}(x_i, y_i),\ f)
          $$

  按上述步骤对空间中所有的三角形面片进行处理后，对大小为$height * width$的屏幕上每个像素点$v_i = (x_i, y_i)$：

  1. 取Z-Buffer中$v_i$对应的面片索引$f_{idx}$，$findices (x_i, y_i) \gets f_{idx}$
  2. 将$f$的三个顶点坐标$(v_0, v_1, v_2)$转换为屏幕坐标$(v_{s0}, v_{s1}, v_{s2})$
  3. 计算$v_i$的中心点坐标$v_c$
  4. 计算$v_c$相对于三角形$f$的重心坐标$(\alpha, \beta, \gamma)$
  5. 使用$(\alpha, \beta, \gamma)$计算透视矫正插值$(\tilde{\alpha}, \tilde{\beta}, \tilde{\gamma})$
  6. $barycentric(x_i, y_i) \gets (\tilde{\alpha}, \tilde{\beta}, \tilde{\gamma})$

  以下是涉及的各种具体计算方法：

  - 顶点$v = (x, y, z, w)$转换为屏幕坐标$v_s = (x_s, y_s, z_s)$

    $$
    x_s = (x / w * 0.5 + 0.5) * (width - 1) + 0.5\\
    y_s = (0.5 + 0.5 * y / w) * (height - 1) + 0.5\\
    z_s = z / w * 0.49999 + 0.5
    $$

  - 点$v$相对于三角形 $(v_0, v_1, v_2)$的重心坐标$(\alpha, \beta, \gamma)$

    1. 分别计算三角形$(v_0, v_1, v_2)$、$(v_0, v, v_2)$和$(v_0, v_1, v)$的有向面积$area$、$beta\_tri$和$gamma\_tri$
    2. 若$area$为0，则$\alpha = \beta = \gamma = -1$，否则

      $$
      \beta = beta\_tri / area\\
      \gamma = gamma\_tri / area\\
      \alpha = 1 - \beta - \gamma
      $$

  - 由顶点$v_0 = (x_0, y_0, z_0)$，$v_1 = (x_1, y_1, z_1)$和$v_2 = (x_2, y_2, z_2)$组成的三角形的有向面积

    $$
    area = (x_2 - x_0) * (y_1 - y_0) - (x_1 - x_0) * (y_2 - y_0)
    $$

  - 结合重心坐标$(\alpha, \beta, \gamma)$和三角形屏幕坐标$(v_0 = (x_0, y_0, z_0), v_1 = (x_1, y_1, z_1), v_2 = (x_2, y_2, z_2))$计算像素点$v = (x, y)$ 的深度$depth$

    $$
    depth = \alpha * z_0 + \beta * z_1 + \gamma * z_2
    $$

  - 结合深度图$d$，遮挡截断$occlusion\_truncation$计算点$v = (x, y)$的深度阈值$depth\_thres$

    $$
    depth\_thres = d(x, y) * 0.49999 + 0.5 + occlusion\_truncation
    $$

  - 根据重心坐标$(\alpha, \beta, \gamma)$判断顶点是否在三角形内
    如果$\alpha >= 0$且$\beta >= 0$且$\gamma >= 0$则点在三角形内（包括在三角形边上），否则点不在三角形内。
  - 结合重心坐标$(\lambda_0, \lambda_1, \lambda_2)$以及三角形的三个顶点坐标$v_0 = (x_0, y_0, z_0, w_0)$，$v_1 = (x_1, y_1, z_1, w_1)$和$v_2 = (x_2, y_2, z_2, w_2)$计算透视矫正插值$(\lambda_0^{corrected}, \lambda_1^{corrected}, \lambda_2^{corrected})$

    $$
    \lambda_i^{corrected} = \frac{\lambda_i / w_i} { \sum (\lambda_j / w_j)}
    $$

## 参数说明

- **参数说明**：

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
      <td>v</td>
      <td>输入</td>
      <td>表示空间中顶点坐标的输入张量，对应公式描述中的`v`，size为2。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>f</td>
      <td>输入</td>
      <td>表示空间中面片的输入张量，对应公式描述中的`f`，size为2。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>d</td>
      <td>可选输入</td>
      <td><ul><li>表示深度图的输入张量，用于计算深度阈值，此参数不生效。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>height</td>
      <td>输入</td>
      <td><ul><li>表示屏幕高度。</li><li>取值范围为[1,4096]。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>width</td>
      <td>输入</td>
      <td><ul><li>表示屏幕宽度。</li><li>取值范围为[1,4096]。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>occlusion_truncation</td>
      <td>输入</td>
      <td><ul><li>遮挡截断，用于计算深度阈值，此参数不生效但需要传入。</li><li>默认值为0.0，当前仅支持取值为0.0。</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_depth_prior</td>
      <td>输入</td>
      <td><ul><li>表示是否应用深度先验，此参数不生效但需要传入。</li><li>默认值为0，当前仅支持取值为0。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>findices</td>
      <td>输出</td>
      <td>表示屏幕中每个像素点最小深度对应的面片索引，对应公式描述中的`findices`，size为2。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>barycentric</td>
      <td>输出</td>
      <td>表示屏幕中每个像素点基于最小深度的面片，求解得到的重心坐标透视矫正插值的输出张量，对应公式描述中的`barycentric`，shape为(height, width, 3)，最后一维对应三个顶点的透视矫正重心坐标。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持useDepthPrior为0的输入场景，参数d、occlusionTruncation、useDepthPrior在实际计算中不生效。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_rasterizer](examples/test_aclnn_rasterizer.cpp) | 通过[aclnnRasterizer](docs/aclnnRasterizer.md)接口方式调用Rasterizer算子。 |
