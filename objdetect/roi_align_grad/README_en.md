# RoiAlignGrad

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    √     |

## Function Description

- Operator function: ROIAlignGrad is the backpropagation operator of the ROIAlign operator. ROIAlign is a pooling operation that extracts fixed-size ROI (Region of Interest) features from feature maps of non-uniform sizes. The backpropagation distributes the output gradients back to the input feature map according to the bilinear interpolation weights used in the forward propagation.

- Computation formula:

  For each sampling point $(y, x)$ in each bin of each ROI, the operator computes the interpolation weights of its four neighbor pixels $(y_{low}, x_{low})$, $(y_{low}, x_{high})$, $(y_{high}, x_{low})$, and $(y_{high}, x_{high})$ through bilinear interpolation:

  $$
  l_y = y - y_{low}, \quad l_x = x - x_{low}, \quad h_y = 1 - l_y, \quad h_x = 1 - l_x
  $$

  $$
  w_1 = h_y \cdot h_x, \quad w_2 = h_y \cdot l_x, \quad w_3 = l_y \cdot h_x, \quad w_4 = l_y \cdot l_x
  $$

  The operator accumulates the gradient $\frac{\partial L}{\partial y}$ of each sampling point to the four neighbor pixels according to the weights:

  $$
  \text{xdiff}[b, c, y_{low}, x_{low}] += \frac{1}{count} \cdot \frac{\partial L}{\partial y} \cdot w_1
  $$

  $$
  \text{xdiff}[b, c, y_{low}, x_{high}] += \frac{1}{count} \cdot \frac{\partial L}{\partial y} \cdot w_2
  $$

  $$
  \text{xdiff}[b, c, y_{high}, x_{low}] += \frac{1}{count} \cdot \frac{\partial L}{\partial y} \cdot w_3
  $$

  $$
  \text{xdiff}[b, c, y_{high}, x_{high}] += \frac{1}{count} \cdot \frac{\partial L}{\partial y} \cdot w_4
  $$

  In the formula, $b$ is the batch index, $c$ is the channel index, and $count = roi\_bin\_grid\_h \times roi\_bin\_grid\_w$ is the total number of sampling points in the bin.

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
      <td>ydiff</td>
      <td>Input</td>
      <td>Backpropagation gradient input, with shape (N, C, pooled_height, pooled_width).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>Input</td>
      <td>ROI coordinates, with shape (N, 5). Each row is in the format [batch_idx, roi_x1, roi_y1, roi_x2, roi_y2].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois_n</td>
      <td>Input (Optional)</td>
      <td>Number of valid ROIs in each batch, with shape (B,). The current computation does not use this parameter.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>xdiff_shape</td>
      <td>Attr (Required)</td>
      <td>Shape of the output feature map, that is, the shape of the forward input feature map, in the format [B, C, H, W]. The value must be a list of four positive integers.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_height</td>
      <td>Attr (Required)</td>
      <td>Height of the ROI output feature map.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_width</td>
      <td>Attr (Required)</td>
      <td>Width of the ROI output feature map.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>Attr (Required)</td>
      <td>Scaling ratio of the feature map to the original image.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sample_num</td>
      <td>Attr (Optional)</td>
      <td>Number of sampling times for each bin. Default value: 2. 0 indicates automatic computation (ceil(roi_size / pooled_size)).</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roi_end_mode</td>
      <td>Attr (Optional)</td>
      <td>
        <ul style="margin-top: 0; margin-bottom: 0; padding-left: 2ch;">
          <li>0: No alignment.</li>
          <li>1: TF offset mode (x2/y2 += spatial_scale, and roi_w/roi_h is set to max(., 1.0)).</li>
          <li>2: PyTorch alignment mode (x1/y1/x2/y2 -= 0.5).</li>
          <li>3: MMDetection alignment mode (same as 2).</li>
        </ul>
        Default value: 1.
      </td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xdiff</td>
      <td>Output</td>
      <td>Gradient of the input feature map, with shape (B, C, H, W).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The operator supports only the float32 data type.
- ydiff and xdiff are 4D ND tensors, and rois is a 2D ND tensor.
- The xdiff_shape attribute must be a list of four positive integers.
- For an empty tensor (a shape containing a 0 dimension), the operator returns all-zero gradients.

## Invocation

| Invocation Method | Sample | Description |
|----------|----------|------|
| aclnn API invocation | [test_aclnn_roi_align_v2_backward_l2](examples/test_aclnn_roi_align_v2_backward.cpp) | Invokes the RoiAlignGrad operator through the [aclnnRoiAlignV2Backward](./docs/aclnnRoiAlignV2Backward_en.md) API. |
| Graph mode      | [test_geir_roi_align_grad](examples/test_geir_roi_align_grad.cpp)                                    | Invokes the RoiAlignGrad operator by constructing a graph through the [operator IR](op_graph/roi_align_grad_proto.h).                         |
