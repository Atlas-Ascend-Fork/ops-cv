# YUV4442YUV422

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

- 算子功能：将 YUV444 格式图像数据转换为 YUV422 格式。输入每个像素位置包含 4 个通道 [Y, U, V, pad]，其中第 4 个通道（索引 3）为填充位并被忽略。算子对水平相邻像素的色度分量进行 2:1 子采样（直接选择，cosited），输出 YUV422 格式的 2 通道数据。

- 计算公式：

$$
y[i, j, 0] = \text{clip\_uint8}(Y_0)
$$
$$
y[i, j, 1] = \text{clip\_uint8}(U_0)
$$
$$
y[i, j+1, 0] = \text{clip\_uint8}(Y'_0)
$$
$$
y[i, j+1, 1] = \text{clip\_uint8}(V_0)
$$

- 数值转换行为（float16 → uint8）：先就近取整（半值向远离零方向进位，如 2.5 → 3），再截断到 [0, 255]，即 clip_uint8(x) = min(max(round(x), 0), 255)。特殊值处理：NaN 输出 0；+Inf 输出 255；-Inf 及小于 0 的值输出 0；大于 255 的值输出 255。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>YUV444 输入图像数据，shape 为 (h, w, 4)，每像素 4 通道 [Y, U, V, pad]；其中第 4 个通道（索引 3）会被忽略。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>YUV422 输出图像数据，shape 为 (h, w, 2)，每像素 2 通道 (Y, UV)。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入必须为 3 维张量，第三维固定为 4（通道顺序为 [Y, U, V, pad]，第 4 个通道（索引 3）会被忽略）。
- 输出为 3 维张量，前两个维度与输入相同，第三维固定为 2（YUV422 的 2 通道打包格式）。
- 奇数W：支持。输出仍为 (h, w, 2)，最后一列像素仅写入 [Y, U] 分量（无配对像素，不产生越界读写）。
- 仅支持 float16 输入、uint8 输出。
- 输入输出要求连续存储（ND layout）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE图模式 | [test_geir_yuv4442_yuv422](examples/test_geir_yuv4442_yuv422.cpp)  | 通过[算子IR](op_graph/yuv4442_yuv422_proto.h)构图方式调用YUV4442YUV422算子。         |
