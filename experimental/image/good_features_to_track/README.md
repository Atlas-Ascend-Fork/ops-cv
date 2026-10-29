# GoodFeaturesToTrack

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 910B 系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

算子注册的 AICore 配置为 `ascend910b` 与 `ascend910_93`。

## 功能说明

- 算子功能：GoodFeaturesToTrack（Shi-Tomasi 角点检测，对齐 OpenCV `cv::goodFeaturesToTrack`）。输入单通道灰度图，输出响应最强的若干角点坐标与个数；支持可选 ROI 掩码、可选 Harris 响应、`quality_level` 阈值与 `min_distance` 非极大值抑制、`max_corners` 上限。常用于视觉里程计、光流跟踪的特征点提取，也可作为 SIFT/ORB 等描述子算法的前端。

- 计算公式：

1) Sobel 梯度（3×3），$g_x,\ g_y$；

2) 结构张量的 `block_size` × `block_size` 窗口和：

$$
S_{xx}=\sum g_x^2,\qquad S_{yy}=\sum g_y^2,\qquad S_{xy}=\sum g_x g_y
$$

3) Shi-Tomasi 响应取结构张量的最小特征值（闭式解）：

$$
\lambda_{\min}=\frac{S_{xx}+S_{yy}-\sqrt{(S_{xx}+S_{yy})^2-4\,(S_{xx}S_{yy}-S_{xy}^2)}}{2}
$$

`use_harris_detector=true` 时改用 Harris 响应：

$$
R=S_{xx}S_{yy}-S_{xy}^2-k\,(S_{xx}+S_{yy})^2
$$

4) 阈值与抑制：保留 $R>\text{quality\_level}\times R_{\max}$ 的像素，再按响应从大到小做贪心非极大值抑制——已选角点与新角点的欧氏距离小于 `min_distance` 时丢弃新角点；最多输出 `max_corners` 个。

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
      <td>image</td>
      <td>输入</td>
      <td>输入灰度图，shape [N, H, W]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>可选输入</td>
      <td>ROI 掩码，shape [N, H, W]，与 image 同尺寸；非 0 表示该像素参与检测。不传则全图参与。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>corners</td>
      <td>输出</td>
        <td>角点坐标，shape [N, max_corners, 2]，每行 <b>(x, y)</b>，与 OpenCV <code>cv::goodFeaturesToTrack</code> 的坐标序一致。有效个数由 <code>num_corners</code> 给出；未填充位置的哨兵值实测为 <b>(-1,-1)</b> 或 <b>(0,0)</b>（取决于容量/对齐路径，不保证只有一种），因此<b>不能靠哨兵值判断有效个数</b>。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_corners</td>
      <td>输出</td>
      <td>每张图实际输出的角点数，shape [N]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_corners</td>
      <td>属性</td>
      <td>最大角点数，默认 1000；取 0 表示不限制，输出容量按 H*W 计算。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quality_level</td>
      <td>属性</td>
      <td>质量阈值系数，默认 0.01；取值范围 (0, 1]。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>min_distance</td>
      <td>属性</td>
      <td>角点间最小欧氏距离，默认 1.0；取值范围 ≥ 0。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>属性</td>
      <td>结构张量的窗口边长，默认 3；需为 ≥ 1 的奇数。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_harris_detector</td>
      <td>属性</td>
      <td>是否使用 Harris 响应，默认 false（Shi-Tomasi 最小特征值）。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>k</td>
      <td>属性</td>
      <td>Harris 响应灵敏度系数，默认 0.04；取值范围 ≥ 0，仅在 use_harris_detector=true 时生效。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

  - `image` 必须为 rank 3 的 `[N, H, W]`，`N/H/W > 0`，且 `H`、`W`、`H*W` 均不超过 `UINT32_MAX`；dtype 仅支持 float16 与 float32。
  - `H*W` 需满足 `H*W <= 2^24`（16,777,216）：阶段 2b 的键直方图用 fp32 累加计数，fp32 只能精确表示 `<= 2^24` 的整数，超出后单桶计数与跨核求和会舍入、`b*/b**` 边界桶选取偏移（结果仍确定，但会偏离精确 Top-K）；超出时 Tiling 阶段返回失败。如需更大分辨率，需改用 int32/int64 计数路径。
  - `W` 需满足 `4*W <= 65535`（即 `W <= 16383`）：内核行搬运用 `DataCopyParams.blockLen`（字节单位、上限 65535），response 缓冲恒为 float、单行占 `4*W` 字节；超出时 blockLen 会被截断，且该宽度域会回退到无宽度防护的回退路径，因此 Tiling 阶段直接返回失败。
  - `mask` 传入时必须与 `image` 形状完全一致，dtype 为 uint8。
- 属性需满足：`quality_level ∈ (0, 1]`、`min_distance ≥ 0`、`block_size ≥ 1` 且为奇数、`k ≥ 0`；不满足时 Tiling 阶段返回失败。
  - 输出 `corners` 固定为 `[N, max_corners, 2]` 的上界形状（`max_corners=0` 时为 `[N, H*W, 2]`），有效个数以 `num_corners` 为准；未填充位置的哨兵值实测为 `(-1,-1)` 或 `(0,0)`（取决于容量/对齐路径），**不能靠哨兵值判断有效个数**。
  - **判定有效性必须依据 `num_corners`**：`x=0` 或 `y=0` 是合法角点坐标（图像边缘），不能以坐标是否为 0 判断，也不能只看单个分量是否为 -1。
  - 角点选取（阈值 + 非极大值抑制）由单个核串行完成，保证结果与调度无关、逐次可复现；并列响应的取舍同样是确定性的。
  - 退化输入（无纹理常量图等）下图像边界会产生少量边缘假响应，因为本实现的 Sobel 边界延拓方式与 OpenCV 不同；详见下文「与 OpenCV 的行为差异」。
  - `quality_level` 阈值**按图独立**：`R_max` 逐图统计（batch 内每张图用自己的最大响应计算 `threshold = quality_level × R_max`），与 `cv::goodFeaturesToTrack` 的逐图语义一致。

## 与 OpenCV 的行为差异

本算子语义对齐 `cv::goodFeaturesToTrack`（含输出坐标序 `(x, y)`），但以下三点存在差异，从 OpenCV 迁移使用时请注意：

1. **Sobel 边界延拓方式不同**：本算子在图像外按**补 0**（零填充）计算 3×3 Sobel 梯度，OpenCV 默认使用 `BORDER_REFLECT_101`（边缘镜像）。因此距边界约 2 个像素的窄带内响应与角点与 OpenCV 不同（实测 256×256 棋盘格：本算子比 OpenCV 多出的 126 个角点**全部**落在该窄带内）；距边界更远的图像内部，两边的几何角点逐点精确一致。

2. **无纹理（常量图）退化输入的回退不同**：响应恒 0 时阈值 `quality_level × R_max` 退化为 0；OpenCV 对这类输入返回 0 个角点，本实现因边界补 0 会在图像四角附近产生少量（实测 ≤ 4 个）边缘假角点。该差异只影响**常量/近常量图**这类退化输入，真实纹理图像不受影响；全 0 `mask` 输入下本算子返回 0 个角点。

3. **除边界外，阈值与并列响应取舍也不同**：本实现用稳定基数排序 + 栅格序贪心做阈值与非极大值抑制，结果**确定性可复现**（同一输入反复执行逐位一致、与核数/调度无关），但阈值比较与局部极大判据不保证与 OpenCV 相同，因此在并列响应密集的图案上两者给出的角点**数量**可能明显不同。实测 256×256 棋盘格（`maxCorners=20000`）：OpenCV 3844 个、本算子 1087 个，其中 **961 个内部格点两边逐点精确相等**，且 OpenCV 找到的每个角点在 1px 内都能在本算子输出中找到（即不漏检位置）。完整对拍脚本、原始数据与复现步骤见 [tests/opencv_crosscheck](tests/opencv_crosscheck/README.md)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn接口 | [test_aclnn_good_features_to_track](examples/test_aclnn_good_features_to_track.cpp) | 通过 aclnnGoodFeaturesToTrack 接口方式调用 GoodFeaturesToTrack 算子（含功能与精度用例）。 |
| 与 OpenCV 对拍 | [tests/opencv_crosscheck](tests/opencv_crosscheck/README.md) | 可复现的交叉验证：numpy 逐位重建输入 → OpenCV 参考 → 与算子输出 dump 逐点比对。 |
| 图模式 | - | 未提供图模式样例，可通过算子 IR 信息（op_host 内 OP_ADD 注册）构图调用。 |
