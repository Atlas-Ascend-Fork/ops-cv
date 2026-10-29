# DeformableRoiPoolGrad

English | [简体中文](./README.md)

## Product Support

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: DeformableRoiPoolGrad is the backward gradient operator of DeformableRoiPool (deformable region of interest pooling) and is used in training scenarios. It receives the upstream gradient `grad`, and combines the forward inputs `x` (feature map), `rois`, and `offset` to calculate the gradient `grad_x` of `feature_map` and the gradient `grad_offset` of `offset`.

- Calculation formula:

**grad_input (gradient of feature_map)**: distributes the upstream gradient backward to the four neighborhood points based on the bilinear interpolation weights.

$$
\text{grad\_output\_this\_bin} = \text{grad\_output}[n,c,ph,pw] / \text{count}
$$

$$
\text{grad\_input}[\text{roi\_batch\_ind}, c, y_{low}, x_{low}] \mathrel{+}= \text{grad\_output\_this\_bin} \cdot w_1
$$

The bilinear interpolation weights are as follows:

$$
w_1 = (y_{high} - y) \cdot (x_{high} - x), \quad w_2 = (y_{high} - y) \cdot (x - x_{low})
$$

$$
w_3 = (y - y_{low}) \cdot (x_{high} - x), \quad w_4 = (y - y_{low}) \cdot (x - x_{low})
$$

**grad_offset (gradient of offset)** (calculated only when offset is not empty):

$$
\text{ogx} = \gamma \cdot \text{roi\_width} \cdot \text{grad\_output\_this\_bin} \cdot (\text{input}_{11} \cdot l_y + \text{input}_{10} \cdot (y_{high} - y_c) + \text{input}_{01} \cdot (-l_y) + \text{input}_{00} \cdot (y_c - y_{high}))
$$

$$
\text{ogy} = \gamma \cdot \text{roi\_height} \cdot \text{grad\_output\_this\_bin} \cdot (\text{input}_{11} \cdot l_x + \text{input}_{01} \cdot (x_{high} - x_c) + \text{input}_{10} \cdot (-l_x) + \text{input}_{00} \cdot (x_c - x_{high}))
$$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 280px">
  <col style="width: 150px">
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
      <td>grad</td>
      <td>Input</td>
      <td>Upstream gradient. The shape is (N, C, pH, pW), where N is the number of ROIs, C is the number of channels, and pH/pW are the pooling height and width.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>feature_map (forward input). The shape is (B, C, H, W), where B is the batch size, and H/W are the feature map height and width.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>Input</td>
      <td>ROI positions. The shape is (N, 5), and the format is [batch_ind, x1, y1, x2, y2].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input (optional)</td>
      <td>Offsets. The shape is (N, 2, pH, pW), where channel 0 is the w direction and channel 1 is the h direction. If the value is empty, grad_offset is not calculated.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>ListInt. The length is 2: [pooled_height, pooled_width], which is the size of the pooling output.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>Attr</td>
      <td>Scaling ratio of feature_map to the original image. The default value is 1.0.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sampling_ratio</td>
      <td>Attr</td>
      <td>Sampling ratio. The value range is <code>[0, 46340]</code>; 0 indicates the adaptive value <code>ceil(roi_size/pooled_size)</code>. The default value is 0.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Attr</td>
      <td>Scaling factor of offset. The default value is 0.1.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>Output</td>
      <td>Gradient of feature_map. The shape is the same as that of x.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_offset</td>
      <td>Output</td>
      <td>Gradient of offset. The shape is the same as that of offset. If offset is empty, a zero tensor is output.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, and <term>Atlas A2 products</term> are supported; <term>Atlas 200I/500 A2 inference products</term> are not within the support scope. <term>Atlas A3 products</term> and <term>Atlas A2 products</term> are provided by the existing TBE implementation in the CANN compatible delivery, and the newly added SIMT Kernel in this directory is only for <term>Ascend 950PR&950DT products</term>.
- The registered format of all tensors is ND; NCHW in this document only describes the logical axis order.
- The data types of the inputs `grad`, `x`, `rois`, non-empty `offset`, and the outputs must be the same, and only FLOAT is supported.
- `grad` and `x` must be 4D tensors, and `rois` must be `(N, 5)`. The following requirements must be met: `grad.N==rois.N` and `grad.C==x.C`.
- All Kernel dimensions must be representable by INT32, and `x.H*x.W` must not exceed INT32_MAX. `output_size` must contain exactly two elements within the range `[1, INT32_MAX]`, and the values must be the same as the `pH/pW` of `grad`.
- A non-empty `offset` must be `(N, 2, pH, pW)`; no input, a 0-dimension placeholder, or a rank-1 `{0}` placeholder is all processed as an empty `offset`.
- `spatial_scale` must be a finite positive number; the support range of `sampling_ratio` is `[0, 46340]`; `gamma` must be a finite number.
- The format of `rois` is `[batch_ind, x1, y1, x2, y2]`, and the `batch_ind` of valid inputs should be an integer within `[0, B)`. In the defense path of <term>Ascend 950PR&950DT products</term>, when `B>0`, a non-finite batch index is treated as 0, an out-of-range value is clipped to `[0, B-1]`, and a decimal is truncated toward zero. A non-finite ROI coordinate participates in the calculation as 0. When `B=0`, both gradient outputs are zero/empty tensors.
- A non-finite value in a non-empty `offset` determines the sampling position as 0, and the `grad_offset` in the corresponding direction is written as 0. This defense semantics is consistent with the forward <term>Ascend 950PR&950DT products</term> implementation.
- When `sampling_ratio=0`, the adaptive number of sampling points is `ceil(roi_size/pooled_size)`. If the result is non-finite, not positive, or exceeds 46340 in a single direction, the ROI does not generate valid sampling.
- If a sampling coordinate is non-finite or exceeds `[-1, H] × [-1, W]`, the sampling point is skipped.
- `grad_x` uses a single FLOAT workspace, and the upper limit of the user workspace is 2GiB. If the limit is exceeded, Tiling returns a failure.
- The inputs must be contiguous.
- The deterministic owner scheme is used: each output position is accumulated by a unique thread in a fixed order, avoiding cross-thread write conflicts.

## Invocation Description

| Invocation Method | Sample Code | Description |
|----------|----------|------|
| Graph mode | [test_geir_deformable_roi_pool_grad](examples/test_geir_deformable_roi_pool_grad.cpp) | Call the DeformableRoiPoolGrad operator by constructing a graph in the [operator IR](op_graph/deformable_roi_pool_grad_proto.h) mode. |
