# RoiPoolingWithArgMax

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×   |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：对输入特征图按ROI（感兴趣区域）进行池化，在每个ROI内按空间划分为`pooled_h × pooled_w`个格子，对每个格子做最大池化，并输出池化结果及最大值在通道内的一维索引（argmax）。

- 计算公式：

  输入特征图 $x$ 的shape为 $(N, C, H, W)$，ROI张量 $\text{rois}$ 的shape为 $(\text{num\_rois}, 5)$，每行表示 $(b_n, x_1, y_1, x_2, y_2)$。标量参数为 $s_h$、$s_w$（spatial_scale）以及 $\text{pooled\_h}$、$\text{pooled\_w}$。下标 $n$ 表示ROI索引，$c$ 表示通道，$(\text{ph}, \text{pw})$ 表示池化格点。

  - **ROI映射到特征图**：将ROI坐标乘以spatial_scale得到特征图上的浮点区间：

    $$
    \tilde{x}_1 = x_1 s_w,\quad \tilde{y}_1 = y_1 s_h,\quad \tilde{x}_2 = (x_2+1)s_w,\quad \tilde{y}_2 = (y_2+1)s_h
    $$

    $$
    W_{\text{roi}} = \tilde{x}_2 - \tilde{x}_1,\qquad H_{\text{roi}} = \tilde{y}_2 - \tilde{y}_1
    $$

    若 $W_{\text{roi}} \le 0$ 或 $H_{\text{roi}} \le 0$，该ROI的 $y$ 全为0，$\text{argmax}$ 全为 -1。

  - **Bin步长与区间**：每个池化格(ph, pw)对应ROI内一个bin，步长与浮点区间为：

    $$
    \Delta w = \frac{W_{\text{roi}}}{\text{pooled\_w}},\qquad \Delta h = \frac{H_{\text{roi}}}{\text{pooled\_h}}
    $$

    $$
    \tilde{w}_1 = \text{pw} \cdot \Delta w + \tilde{x}_1,\quad \tilde{w}_2 = (\text{pw}+1) \cdot \Delta w + \tilde{x}_1
    $$

    $$
    \tilde{h}_1 = \text{ph} \cdot \Delta h + \tilde{y}_1,\quad \tilde{h}_2 = (\text{ph}+1) \cdot \Delta h + \tilde{y}_1
    $$

    取整并裁剪到 $[0,W) \times [0,H)$：

    $$
    w_1 = \text{clip}(\lfloor\tilde{w}_1\rfloor,\, 0,\, W),\quad w_2 = \text{clip}(\lceil\tilde{w}_2\rceil,\, 0,\, W)
    $$

    $$
    h_1 = \text{clip}(\lfloor\tilde{h}_1\rfloor,\, 0,\, H),\quad h_2 = \text{clip}(\lceil\tilde{h}_2\rceil,\, 0,\, H)
    $$

    其中 $\text{clip}(a,l,u) = \min(\max(a,l), u)$。若 $w_2 \le w_1$ 或 $h_2 \le h_1$，该bin为空：$y=0$，$\text{argmax}=-1$。

  - **池化输出与Argmax**：记 $b = \text{rois}[n,0]$，bin区域 $R = \{(h,w) : h_1 \le h < h_2,\, w_1 \le w < w_2\}$，则

    $$
    y[n,c,\text{ph},\text{pw}] = \max_{(h,w) \in R} x[b,c,h,w]
    $$

    （空 $R$ 时为0）

    $$
    \text{argmax}[n,c,\text{ph},\text{pw}] = h^* W + w^*
    $$

    $(h^*, w^*)$ 为bin内最大值位置（多解取第一个）；空 $R$ 为 -1。

  - **输出Shape**：

    | 输出 | Shape | 数据类型 |
    |------|--------|----------|
    | $y$ | $(\text{num\_rois},\, C,\, \text{pooled\_h},\, \text{pooled\_w})$ | 与 $x$ 一致 |
    | $\text{argmax}$ | 同上 | INT32 |

## 参数说明

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
      <td>x</td>
      <td>输入</td>
      <td>输入特征图。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>ROI边界框。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>roi_actual_num</td>
      <td>可选输入</td>
      <td>指定每个batch的ROI数量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pooled_h</td>
      <td>属性</td>
      <td>指定池化率的高。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_w</td>
      <td>属性</td>
      <td>指定池化率的宽。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_h</td>
      <td>属性</td>
      <td>将ROI坐标映射到输入特征图坐标时，高度方向的缩放比例。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_w</td>
      <td>属性</td>
      <td>将ROI坐标映射到输入特征图坐标时，宽度方向的缩放比例。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pool_channel</td>
      <td>属性</td>
      <td>指定池化的通道数。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>池化结果。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>输出</td>
      <td>每个池化格点最大值在通道内的线性偏移索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* x、rois、y、argmax的数据类型或格式在支持的范围之内。
* x的shape是4维（NCHW）。
* rois的shape第二维是5。
* pooled_h、pooled_w、spatial_scale_h、spatial_scale_w大于0。
* x.shape[0] 表示输入 batch 数；y、argmax 的 shape[0] 等于 rois.shape[0]（num_rois）。
* rois.shape[0]、x.shape[0]小于等于1024。
* x.shape[1]等于pool_channel。
* x.shape[1]为16的倍数。
* rois.shape[0]等于roi_actual_num。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_roi_pooling_with_arg_max](examples/arch35/test_aclnn_roi_pooling_with_arg_max.cpp) | 通过[aclnnRoiPoolingWithArgMax](docs/aclnnRoiPoolingWithArgMax.md)接口方式调用RoiPoolingWithArgMax算子。 |
