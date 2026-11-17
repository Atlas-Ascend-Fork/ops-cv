# GuidedFilter

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 910B 系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：引导滤波（Guided Image Filtering），语义对齐 `cv::ximgproc::guidedFilter`。以 `guide`（引导图）的结构信息为约束，对 `src`（待滤波图）做保边平滑。支持灰度引导（C=1）与彩色引导（C=2/3），`src` 通道数与 `guide` 相互独立（1~64）。

- 计算公式（对每个输出通道 $s$、窗口 $\omega_k$ 为 $(2r+1)\times(2r+1)$）：

$$
\mu_c = \mathrm{box}(g_c),\quad
\Sigma_{cd} = \mathrm{box}(g_c g_d) - \mu_c \mu_d + \varepsilon\,\delta_{cd},\quad
\nu_s = \mathrm{box}(p_s),\quad
\omega_{cs} = \mathrm{box}(g_c p_s) - \mu_c \nu_s
$$

$$
\alpha_{cs} = \sum_{d}\left(\Sigma^{-1}\right)_{cd}\omega_{ds},\qquad
\beta_s = \nu_s - \sum_{c}\alpha_{cs}\,\mu_c
$$

$$
q_s = \overline{\mathrm{box}(\alpha_{cs})}\cdot g_c + \overline{\mathrm{box}(\beta_s)}
= \mathrm{box}(\beta_s) + \sum_{c}\mathrm{box}(\alpha_{cs})\cdot g_c
$$

其中 `box(·)` 为半径 `radius` 的均值滤波，边界按 **BORDER_REFLECT**（`period = 2n`）延拓；$\varepsilon$ 不随输入 dtype 缩放（与 OpenCV 一致）。

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
      <td>guide</td>
      <td>输入</td>
      <td>引导图。rank2 [H, W]、rank3 [H, W, C]、rank4 [N, H, W, C]（通道后置）；C ∈ [1, 3]。</td>
      <td>UINT8、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src</td>
      <td>输入</td>
      <td>待滤波图。rank 与空间尺寸须与 guide 一致；通道数 Cs ∈ [1, 64]，可与 guide 不同。</td>
      <td>UINT8、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst</td>
      <td>输出</td>
      <td>输出张量。shape 与 src 完全一致；dtype 由 dDepth 决定。</td>
      <td>UINT8、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>radius</td>
      <td>属性</td>
      <td>窗口半径 r。必填，默认 4，取值范围 [1, 512]。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>正则项，控制平滑强度。可选，默认 0.01，要求有限且 ≥ 0；不随 dtype 缩放。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dDepth</td>
      <td>属性</td>
      <td>输出位深。可选，默认 -1：-1 跟随 src；0=CV_8U、5=CV_32F、7=CV_16F。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>属性</td>
      <td>快速模式降采样比例。可选，默认 1.0，当前仅支持 1.0，其余取值显式报错。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

`guide` 与 `src` 的 rank 必须相同且空间尺寸 (H, W) 完全一致，通道数可以不同：`guide` 通道数 C 必须在 [1, 3]，超出直接返回失败（不做自动降维）；`src` 通道数 Cs 在 [1, 64]。两者 dtype 必须一致，支持 uint8、float16、float32；`dst` 的 dtype 由 `dDepth` 决定，取值只能是 -1（跟随 `src`）、0（CV_8U）、5（CV_32F）、7（CV_16F）。单行字节数 `W × max(C, Cs) × elemSize` 不得超过 65535（DataCopyPad 的 `blockLen` 为 uint16），超过时 Tiling 阶段直接拒绝。计算中间量统一用 float32；uint8 输入按 0~255 原值参与运算、不做归一化，输出按四舍五入（`ROUND_HALF_UP`）回写。不支持空 tensor（N/H/W 为 0 时返回失败）。算子注册的 AICore 配置为 ascend910b 与 ascend910_93。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn接口 | [test_aclnn_guided_filter](examples/test_aclnn_guided_filter.cpp) | 通过 aclnnGuidedFilter 接口方式调用 GuidedFilter 算子。 |
| 图模式 | - | 暂未提供算子 IR 构图方式。 |
