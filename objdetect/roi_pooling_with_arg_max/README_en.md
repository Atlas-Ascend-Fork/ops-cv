# RoiPoolingWithArgMax

English | [简体中文](./README.md)

## Product Support

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×   |
|  <term>Atlas training products</term>    |     √    |

## Function Description

- Operator function: Pools the input feature map by ROI (region of interest). Within each ROI, the operator divides the space into `pooled_h × pooled_w` bins, performs max pooling on each bin, and outputs the pooling results and the 1D index (argmax) of the maximum value within the channel.

- Computation formula:

  The input feature map $x$ has the shape $(N, C, H, W)$, and the ROI tensor $\text{rois}$ has the shape $(\text{num\_rois}, 5)$, where each row represents $(b_n, x_1, y_1, x_2, y_2)$. The scalar parameters are $s_h$, $s_w$ (spatial_scale), $\text{pooled\_h}$, and $\text{pooled\_w}$. The subscript $n$ denotes the ROI index, $c$ denotes the channel, and $(\text{ph}, \text{pw})$ denotes the pooling grid point.

  - **ROI mapping to the feature map**: The operator multiplies the ROI coordinates by spatial_scale to obtain the floating-point intervals on the feature map:

    $$
    \tilde{x}_1 = x_1 s_w,\quad \tilde{y}_1 = y_1 s_h,\quad \tilde{x}_2 = (x_2+1)s_w,\quad \tilde{y}_2 = (y_2+1)s_h
    $$

    $$
    W_{\text{roi}} = \tilde{x}_2 - \tilde{x}_1,\qquad H_{\text{roi}} = \tilde{y}_2 - \tilde{y}_1
    $$

    If $W_{\text{roi}} \le 0$ or $H_{\text{roi}} \le 0$, all values of $y$ for this ROI are 0, and all values of $\text{argmax}$ are -1.

  - **Bin step and interval**: Each pooling grid point (ph, pw) corresponds to a bin within the ROI. The steps and floating-point intervals are:

    $$
    \Delta w = \frac{W_{\text{roi}}}{\text{pooled\_w}},\qquad \Delta h = \frac{H_{\text{roi}}}{\text{pooled\_h}}
    $$

    $$
    \tilde{w}_1 = \text{pw} \cdot \Delta w + \tilde{x}_1,\quad \tilde{w}_2 = (\text{pw}+1) \cdot \Delta w + \tilde{x}_1
    $$

    $$
    \tilde{h}_1 = \text{ph} \cdot \Delta h + \tilde{y}_1,\quad \tilde{h}_2 = (\text{ph}+1) \cdot \Delta h + \tilde{y}_1
    $$

    The operator rounds and clips the values to $[0,W) \times [0,H)$:

    $$
    w_1 = \text{clip}(\lfloor\tilde{w}_1\rfloor,\, 0,\, W),\quad w_2 = \text{clip}(\lceil\tilde{w}_2\rceil,\, 0,\, W)
    $$

    $$
    h_1 = \text{clip}(\lfloor\tilde{h}_1\rfloor,\, 0,\, H),\quad h_2 = \text{clip}(\lceil\tilde{h}_2\rceil,\, 0,\, H)
    $$

    In the formula, $\text{clip}(a,l,u) = \min(\max(a,l), u)$. If $w_2 \le w_1$ or $h_2 \le h_1$, the bin is empty: $y=0$ and $\text{argmax}=-1$.

  - **Pooling output and argmax**: Let $b = \text{rois}[n,0]$, and the bin region $R = \{(h,w) : h_1 \le h < h_2,\, w_1 \le w < w_2\}$. Then

    $$
    y[n,c,\text{ph},\text{pw}] = \max_{(h,w) \in R} x[b,c,h,w]
    $$

    (0 when $R$ is empty)

    $$
    \text{argmax}[n,c,\text{ph},\text{pw}] = h^* W + w^*
    $$

    $(h^*, w^*)$ is the position of the maximum value in the bin (the first one is taken if multiple solutions exist). When $R$ is empty, the value is -1.

  - **Output shape**:

    | Output | Shape | Data type |
    |------|--------|----------|
    | $y$ | $(\text{num\_rois},\, C,\, \text{pooled\_h},\, \text{pooled\_w})$ | Same as $x$ |
    | $\text{argmax}$ | Same as above | INT32 |

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attr</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input feature map.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>Input</td>
      <td>ROI bounding boxes.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>roi_actual_num</td>
      <td>Optional Input</td>
      <td>Specifies the number of ROIs in each batch.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pooled_h</td>
      <td>Attr</td>
      <td>Specifies the height of the pooling ratio.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_w</td>
      <td>Attr</td>
      <td>Specifies the width of the pooling ratio.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_h</td>
      <td>Attr</td>
      <td>Scaling ratio in the height direction when mapping ROI coordinates to input feature map coordinates.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_w</td>
      <td>Attr</td>
      <td>Scaling ratio in the width direction when mapping ROI coordinates to input feature map coordinates.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pool_channel</td>
      <td>Attr</td>
      <td>Specifies the number of channels for pooling.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Pooling result.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>Output</td>
      <td>Linear offset index of the maximum value of each pooling grid point within the channel.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

* The data types or formats of x, rois, y, and argmax are within the supported range.
* The shape of x is 4D (NCHW).
* The second dimension of the shape of rois is 5.
* pooled_h, pooled_w, spatial_scale_h, and spatial_scale_w are greater than 0.
* x.shape[0] denotes the input batch count, and the shape[0] of y and argmax equals rois.shape[0] (num_rois).
* rois.shape[0] and x.shape[0] are less than or equal to 1024.
* x.shape[1] equals pool_channel.
* x.shape[1] is a multiple of 16.
* rois.shape[0] equals roi_actual_num.

## Invocation

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_roi_pooling_with_arg_max](examples/arch35/test_aclnn_roi_pooling_with_arg_max.cpp) | Invokes the RoiPoolingWithArgMax operator through the [aclnnRoiPoolingWithArgMax](docs/aclnnRoiPoolingWithArgMax_en.md) API. |
