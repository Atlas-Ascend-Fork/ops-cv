# ROIAlign

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>    |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: ROI Align (Region of Interest Align) extracts the features of regions of interest (ROIs) from the feature map, and generates fixed-size output feature maps through bilinear interpolation and pooling.

- Computation formula:

$$
\text{output}[n, c, ph, pw] = \text{pool}\left(\text{bilinear\_interpolate}(\text{features}[\text{batch\_idx}, c], y, x)\right)
$$

In the formula, the coordinate mapping is controlled by the `roi_end_mode` attribute, and the pooling method is controlled by the `pool_mode` attribute (avg/max).

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>features</td>
      <td>Input</td>
      <td>Input feature map, with shape (N, C, H, W).</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>Input</td>
      <td>ROI coordinates, with shape (K, 5). Each row is [batch_idx, x1, y1, x2, y2].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois_n</td>
      <td>Input (Optional)</td>
      <td>Batch index corresponding to each ROI, with shape (K,). The i-th element specifies the batch index used by the i-th ROI. When provided, it overrides batch_idx in the i-th row of rois; otherwise, batch_idx in rois is used.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Pooled ROI features, with shape (K, C, pooled_height, pooled_width).</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>Attr (Required)</td>
      <td>Coordinate scaling factor, which maps ROI coordinates to the feature map scale.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_height</td>
      <td>Attr (Required)</td>
      <td>Output height (number of bins).</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_width</td>
      <td>Attr (Required)</td>
      <td>Output width (number of bins).</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sample_num</td>
      <td>Attr (Optional)</td>
      <td>A positive value indicates that each bin uses this number of sampling points in the H and W directions respectively. 0 indicates that the operator determines the number of sampling points by rounding up the bin size of each ROI in the H and W directions. Value range: [0, 46340] (floor(sqrt(INT32_MAX))). Default value: 2.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roi_end_mode</td>
      <td>Attr (Optional)</td>
      <td>Coordinate transformation mode: 0 means no offset; 1 means adding 1 before scaling; 2 means subtracting the 0.5 offset; 3 means subtracting the 0.5 offset (equivalent to 2). Only 0, 1, 2, and 3 are supported. Default value: 1.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pool_mode</td>
      <td>Attr (Optional)</td>
      <td>Pooling mode: "avg" (average pooling) or "max" (max pooling). Default value: "avg".</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- features must be a 4D tensor (N, C, H, W) in ND format.
- rois must be a 2D tensor (K, 5), and each row is in the format [batch_idx, x1, y1, x2, y2].
- batch_idx must be within [0, N), where N is the size of dimension 0 of input features. If it is out of range, the operator does not report an error: the ROI is treated as an empty ROI and outputs 0 (no out-of-bounds read is performed). 0 <= x1 <= x2 <= W / spatial_scale, and 0 <= y1 <= y2 <= H / spatial_scale (x1 == x2 or y1 == y2 is valid; for roi_end_mode 0/1 the ROI extent in that direction is clamped to a minimum of 1.0).
- features and rois must use the same dtype (float16 or float32).
- rois_n (optional) must be a 1D INT32 tensor with length K (the number of rois rows); the i-th element is the batch index of the i-th ROI and overrides batch_idx in the i-th row of rois.
- pooled_height and pooled_width must satisfy 1 <= value <= INT32_MAX (2147483647).
- sample_num must satisfy 0 <= value <= 46340 (floor(sqrt(INT32_MAX))).
- roi_end_mode only supports 0, 1, 2, and 3.
- pool_mode only supports "avg" and "max".
- spatial_scale must be finite and > 0.

## Invocation

| Invocation Method | Sample Code | Description |
|----------|----------|------|
| aclnn API | [test_aclnn_roi_align](examples/test_aclnn_roi_align.cpp) | Call the ROIAlign operator through the [aclnnRoiAlign](docs/aclnnRoiAlign_en.md) interface. |
| aclnn API | [test_aclnn_roi_align_v2](examples/test_aclnn_roi_align_v2.cpp) | Call the ROIAlign operator through the [aclnnRoiAlignV2](docs/aclnnRoiAlignV2_en.md) interface. |
| GE graph mode | [test_geir_roi_align](examples/arch35/test_geir_roi_align.cpp) | Call the ROIAlign operator by building a graph through [operator IR](op_graph/roi_align_proto.h). Refer to [Operator Invocation](../../docs/zh/invocation/quick_op_invocation.md) to complete operator compilation and verification. |
