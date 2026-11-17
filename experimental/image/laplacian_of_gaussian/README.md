# LaplacianOfGaussian

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

- 算子功能：高斯拉普拉斯（Laplacian of Gaussian，LoG）二阶微分边缘检测。先用高斯核平滑抑制噪声，再取拉普拉斯提取边缘，输出带符号的二阶响应。实现为可分离二维 FIR：先沿 W 方向、再沿 H 方向各做一次一维卷积，两个方向分别使用高斯平滑系数 g 与高斯二阶导系数 d。
- 计算公式：

  - 记 σ = sigma，半径 r = radius，一维核长 k = 2r+1：

    $$g[i] = \frac{\exp\left(-\frac{(i-r)^2}{2\sigma^2}\right)}{\sum_{j=0}^{k-1}\exp\left(-\frac{(j-r)^2}{2\sigma^2}\right)},\quad
    d[i] = \left(\frac{(i-r)^2}{\sigma^2} - 1\right)\frac{g[i]}{\sigma^2}$$

  - 可分离卷积（⊗ 表示一维卷积）：

    $$y = (d \otimes g + g \otimes d) \otimes x$$

  - 输入节点：
    - x (shape[N,C,H,W]，C=1，UINT8 / FLOAT16 / FLOAT32) - 输入特征图
  - 属性节点：
    - sigma (FLOAT) - 高斯标准差，必须为正有限数
    - radius (INT，可选) - 卷积半径，-1 表示自动计算
    - truncate (FLOAT，可选) - 截断系数，默认 4.0
    - border_type (STRING，可选) - 边界模式，默认 REFLECT
  - 输出节点：
    - y (shape[N,C,H,W]，FLOAT32) - 带符号的 LoG 响应，与输入同形状

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 300px">
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
      <td>x</td>
      <td>输入</td>
      <td>输入特征图，支持 [H,W]、[N,H,W]、[N,1,H,W] 三种形态。</td>
      <td>UINT8、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>属性（必选）</td>
      <td>高斯标准差，正有限数。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>radius</td>
      <td>属性（可选）</td>
      <td>卷积半径，默认 -1 表示按 sigma 与 truncate 自动计算。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>truncate</td>
      <td>属性（可选）</td>
      <td>截断系数，默认 4.0，参与自动半径计算。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>border_type</td>
      <td>属性（可选）</td>
      <td>边界模式：REFLECT（默认）、REFLECT_101、REPLICATE、CONSTANT。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>带符号的 LoG 响应，形状与输入一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入支持 2~4 维；4 维时 C 必须为 1，N/H/W 必须为正数。
- 不支持原地覆盖，输入与输出不能指向同一块内存。
- radius 为 -1 时自动取 `floor(float(sigma) * float(truncate) + 0.5)`；显式指定 radius 时 truncate 仍需为有效正数。
- 边界模式语义：REFLECT 重复端点（dcba|abcd|cba），REFLECT_101 不重复端点（dcb|abcd|cba），REPLICATE 复制端点，CONSTANT 补 0。
- 当输入尺寸或半径导致片上资源不足时接口返回错误，不做静默降级。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn 调用 | [test_aclnn_laplacian_of_gaussian.cpp](./examples/test_aclnn_laplacian_of_gaussian.cpp) | 通过 aclnnLaplacianOfGaussian 接口调用 LaplacianOfGaussian 算子。 |
