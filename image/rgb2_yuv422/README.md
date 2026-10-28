# RGB2YUV422

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |     √    |
| <term>Atlas A3系列产品</term>      |    √     |
| <term>Atlas A2系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                       |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：将RGB图像转换为YUV422 (YUYV打包格式)色彩空间。

- 计算公式：

uint8输入（BT.601/256定点系数，含+128偏移）：
$$
Y  = (77 \cdot R + 150 \cdot G + 29 \cdot B) / 256
$$
$$
U  = (-43 \cdot R - 85 \cdot G + 128 \cdot B) / 256 + 128
$$
$$
V  = (128 \cdot R - 107 \cdot G - 21 \cdot B) / 256 + 128
$$

- 数值行为：上述浮点中间结果先饱和截断（clamp）到[0, 255]，再按四舍五入（round-to-nearest-even）转换为uint8输出。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
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
      <td>rgb</td>
      <td>输入</td>
      <td>RGB图像张量，shape [..., H, W, 3]。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>yuv</td>
      <td>输出</td>
      <td>YUV422 (YUYV打包)张量，shape [..., H, W, 2]。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入为ND格式uint8张量，shape为 [..., H, W, 3]；输出shape为 [..., H, W, 2]（YUYV打包）。
- 维数约束：输入rank∈[3, 8]（FORMAT_ND平台维数上限为8），rank<3或rank>8会被拒绝；通道维（最后一维）大小必须为3。
- 动态shape语义：构图阶段各维度（含通道维）可为-1（未知维度），输出对应维度保持-1；未知rank（-2）输入的输出同为未知rank。运行期shape确定后仍须满足上述维数与通道约束。
- 空tensor：H=0或W=0的空输入为合法输入，输出为对应空张量。
- 奇数W：支持。输出仍为 [..., H, W, 2]，最后一列像素仅写入 [Y, U]分量（无配对像素，不产生越界读写）。
- 内存布局：输入与输出须为连续（contiguous）的ND张量，kernel按线性地址访问。
- 大尺寸输入：H / W / outerDims任一超过INT32_MAX时自动切换64位索引模式（TilingKey schMode=1）执行，计算结果保持正确。
- 异常拒绝：rank超限、通道维非3（且非-1）、dtype非uint8的输入在InferShape / tiling阶段被拒绝并输出错误日志；kernel接口不定义独立错误码，拒绝以编译/执行失败形式体现。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_rgb2_yuv422](examples/test_geir_rgb2_yuv422.cpp)  | 通过[算子IR](op_graph/rgb2_yuv422_proto.h)构图方式调用RGB2YUV422算子。         |
