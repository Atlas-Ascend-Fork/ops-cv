# Rgb2yuv422

## 贡献说明

| 贡献者      | 贡献算子 | 贡献时间       | 贡献内容     |
|----------|------|------------|----------|
| CANN-BOT SIMT | rgb2yuv422 | 2026/06/17 | 从 ops-math 迁移 rgb2yuv422 算子到 ops-cv |

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |    ×     |
| <term>Atlas A2系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：将 RGB 图像转换为 YUV422 (YUYV 打包格式) 色彩空间。

- 计算公式：

float16/float32 输入（归一化公式）：
$$
Y  =  0.29900 \cdot R + 0.58700 \cdot G + 0.11400 \cdot B
$$
$$
U  = -0.16874 \cdot R - 0.33126 \cdot G + 0.50000 \cdot B
$$
$$
V  =  0.50000 \cdot R - 0.41869 \cdot G - 0.08131 \cdot B
$$

uint8 输入（含 +128 偏移公式）：
$$
Y  =  0.29900 \cdot R + 0.58700 \cdot G + 0.11400 \cdot B
$$
$$
U  = -0.16874 \cdot R - 0.33126 \cdot G + 0.50000 \cdot B + 128
$$
$$
V  =  0.50000 \cdot R - 0.41869 \cdot G - 0.08131 \cdot B + 128
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
      <td>x</td>
      <td>输入</td>
      <td>RGB 图像张量。NHWC 格式: shape [..., H, W, 3]；NCHW 格式: shape [..., 3, H, W]。</td>
      <td>UINT8、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>YUV422 (YUYV 打包) 张量。NHWC 格式: shape [..., H, W, 2]；NCHW 格式: shape [..., 2, H, W]。</td>
      <td>UINT8、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>输入数据的通道排列格式。"NHWC" 表示通道在最后一维，"NCHW" 表示通道在倒数第三维（3D输入时在第一维）。默认值为 "NHWC"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

输入必须至少为 3 维，通道维大小必须为 3。输入 dtype 支持 uint8、float16 和 float32，输出 dtype 与输入 dtype 一致。data_format 必须为 "NHWC" 或 "NCHW"。仅支持 Ascend 950PR&950DT系列产品。

## aclnn 调用示例

`examples/test_aclnn_rgb2yuv422.cpp` 使用实际的 `aclnnRgb2yuv422GetWorkspaceSize` 和 `aclnnRgb2yuv422` 两段接口，将 ND 存储与 NHWC/NCHW 属性分别设置。两个场景都转换一张红色图像，核对有限输出值及 Y/U/V 结果；失败时返回非零状态。

示例需要 Ascend 950 系列设备，以及包含本算子的配套 CANN/自定义算子包。按仓内 aclnn 示例流程编译运行；Host 侧的编译或 golden 验证不代表已执行设备算子。
