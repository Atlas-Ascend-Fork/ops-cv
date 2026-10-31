# ExtractGlimpseV2

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

- 算子功能：从批量输入图像中提取指定位置和大小的子图像（glimpse）。根据给定的偏移坐标和裁剪尺寸，从每张输入图像中裁剪出一个子区域，输出为一批裁剪后的图像。

- 计算公式：

$$
\text{对于每个 batch item } i:
$$
$$
\text{1. 坐标变换: } offset\_y, offset\_x \text{ (根据 normalized/centered 属性)}
$$
$$
\text{2. 裁剪区域: } [start\_y, end\_y) \times [start\_x, end\_x) \text{ (clamp 到图像边界)}
$$
$$
\text{3. 数据拷贝: } glimpse[i][base\_y:base\_y+copy\_h][base\_x:base\_x+copy\_w][:] = input[i][start\_y:end\_y][start\_x:end\_x][:]
$$
$$
\text{4. 越界填充: } glimpse \text{ 中未拷贝区域填零 (noise="zero")}
$$

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
      <td>input</td>
      <td>输入</td>
      <td>输入图像批次，4D tensor，shape为[batch, height, width, channels]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>裁剪尺寸[size_h, size_w]，1D const tensor，shape为[2]，每个元素必须大于0。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>输入</td>
      <td>每个batch item的偏移坐标(y, x)，2D tensor，shape为[batch, 2]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>glimpse</td>
      <td>输出</td>
      <td>裁剪后的图像批次，4D tensor，shape为[batch, size_h, size_w, channels]。其中size_h, size_w为size的输入。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>centered</td>
      <td>属性</td>
      <td>偏移坐标是否相对于图像居中。默认值：true。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>normalized</td>
      <td>属性</td>
      <td>偏移坐标是否归一化到[-1, 1]。默认值：true。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>uniform_noise</td>
      <td>属性</td>
      <td>是否使用均匀分布噪声（当前实现仅支持false）。默认值：true。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>noise</td>
      <td>属性</td>
      <td>噪声类型（当前实现仅支持"zero"）。默认值："uniform"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- input必须为4D tensor。
- offsets必须为2D tensor，且第二维必须为2。
- size必须为1D const tensor，且长度为2，每个元素必须大于0。
- input和offsets的batch维度必须一致。
- 当前仅支持float32数据类型。
- uniform_noise仅支持false，noise仅支持"zero"。
- 各维度值不超过INT32_MAX（2147483647）；输出元素总数（batch×size_h×size_w×channels）不超过2^62。
- 支持零维张量输入（batch、height、width或channels为0），此时输出为空tensor且不执行设备计算。
- 支持动态shape与动态rank输入：未知维度以-1表示，未知rank以-2表示；输出H/W维始终取size张量的实际数值（size为值依赖输入）。

## 拒绝行为说明

| 拒绝条件 | 拒绝阶段 |
| --- | --- |
| input非4D / offsets非2D / offsets第二维非2 / size非1D且长度非2 / input与offsets的batch不一致 | InferShape |
| size张量实际元素数不足2 | InferShape |
| size元素存在负值 | InferShape |
| size元素存在0 | Tiling |
| 输入dtype非FLOAT（size非INT32、offsets非FLOAT） | Tiling |
| uniform_noise=true 或 noise非"zero" | Tiling |
| 输出元素总数溢出int64 | Tiling |

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE图模式 | [样例](examples/arch35/test_geir_extract_glimpse_v2.cpp)  | 通过[算子IR](op_graph/extract_glimpse_v2_proto.h)构图方式调用ExtractGlimpseV2算子。         |
