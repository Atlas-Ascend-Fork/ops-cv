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
      <td>Batch quantity corresponding to each ROI, with shape (K,).</td>
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
      <td>A positive value indicates that each bin uses this number of sampling points in the H and W directions respectively. 0 indicates that the operator determines the number of sampling points by rounding up the bin size of each ROI in the H and W directions. Default value: 2.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roi_end_mode</td>
      <td>Attr (Optional)</td>
      <td>Coordinate transformation mode: 0 means no offset; 1 means adding 1 before scaling; >=2 means subtracting the 0.5 offset. Default value: 1.</td>
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

- features must be a 4D tensor (N, C, H, W).
- rois must be a 2D tensor (K, 5), and each row is in the format [batch_idx, x1, y1, x2, y2].
- 0 <= batch_idx <= N - 1, 0 <= x1 < x2, and 0 <= y1 < y2.
- features and rois must use the same dtype (float16 or float32).
- pooled_height and pooled_width must be >= 1.
- spatial_scale must be > 0.

## Invocation

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/arch35/test_geir_roi_align.cpp">test_geir_roi_align</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
