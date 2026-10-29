# DeformableRoiPool

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

- Operator function: deformable ROI pooling. For each ROI, the operator performs deformable sampling by spatially dividing bins on the feature map (offsets are optional), performs bilinear interpolation on the sampling points in each bin, and then takes the average to obtain the pooling output.

- Calculation formula:

$$
y(n, c, ph, pw) = \frac{1}{N_{bin}} \sum_{iy=0}^{H_{bin}-1} \sum_{ix=0}^{W_{bin}-1} \sum_{q} w_q \cdot x(idx_n, c, y_q, x_q)
$$

In the formula:

- $n$ is the ROI index, $c$ is the channel index, and $(ph, pw)$ is the output spatial position.
- $idx_n$ is the batch index of the $n$-th ROI ($rois[n, 0]$).
- $H_{bin}$ and $W_{bin}$ are the numbers of sampling points in each bin (determined adaptively by $sampling\_ratio$ or the bin size).
- $w_q$ is the bilinear interpolation weight, and $(y_q, x_q)$ are the coordinates of the four corner points.
- The sampling coordinates are corrected by the offset: $start_h = roi\_start_h + ph \cdot bin\_size_h + offset(n, 1, ph, pw) \cdot \gamma \cdot roi\_height$,
  $start_w = roi\_start_w + pw \cdot bin\_size_w + offset(n, 0, ph, pw) \cdot \gamma \cdot roi\_width$.

Core steps:

1. ROI coordinate transformation: $roi\_start = roi \cdot spatial\_scale - 0.5$.
2. Bin size: $bin\_size_h = roi\_height / pooled\_h$, and $bin\_size_w = roi\_width / pooled\_w$.
3. Number of sampling points: use the value of $sampling\_ratio$ when $sampling\_ratio > 0$; otherwise, adaptively use $\lceil bin\_size \rceil$.
4. Apply the offset (optional).
5. Bilinear interpolation: for each sampling point in each bin, perform weighted accumulation of the four corner point values.
6. Average pooling: divide the accumulated value of all sampling points in a bin by the number of sampling points.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 130px">
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
      <td>Feature map. It is an NCHW tensor with the shape <code>(N, C, H, W)</code>, corresponding to x in the formula.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>Input</td>
      <td>ROI position coordinates. The shape is <code>(num_rois, 5)</code>, and the format of each row is <code>(batch_idx, x1, y1, x2, y2)</code>.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Optional input</td>
      <td>Offset field. The shape is <code>(num_rois, 2, pooled_h, pooled_w)</code>, and the first dimension corresponds to the offsets in the width and height directions respectively.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Pooling output. The shape is <code>(num_rois, C, pooled_h, pooled_w)</code>.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>Attr</td>
      <td>Spatial scaling factor from the feature map to the original image. The default value is 1.0.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>Output spatial size. The format is <code>[pooled_height, pooled_width]</code>.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sampling_ratio</td>
      <td>Attr</td>
      <td>Number of sampling points of each bin in the height and width directions. The value range is <code>[0, 46340]</code>. 0 indicates adaptively determining the value based on the bin size, and a value greater than 0 indicates using the specified value. The default value is 0.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Attr</td>
      <td>Offset scaling factor. The offset is multiplied by gamma and then multiplied by the ROI width/height to obtain the actual offset. The default value is 0.1.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The input `x` must be a 4D NCHW tensor, and the shape is `(N, C, H, W)`; `N` can be 0, and `C`, `H`, and `W` must be positive integers.
- The input `rois` must be a 2D tensor, and the shape is `(num_rois, 5)`, where the second dimension is fixed to 5.
- The input `offset` is an optional parameter. If it is not passed in, no offset is applied, which is equivalent to an all-zero `offset`. If it is passed in, the shape must be
  (num_rois, 2, pooled_h, pooled_w), where `pooled_h` and `pooled_w` are the same as the `output_size` attribute.
- `num_rois` is determined by `rois.shape[0]`, and `output_size` must contain two integers within the range `[1, INT32_MAX]`.
- The total number of elements of the input `x` (`N * C * H * W`) must not exceed 500000000, and the total number of elements of the output `y`
  (`num_rois * C * pooled_h * pooled_w`) must not exceed 200000000. If the preceding limits are exceeded, the operator returns `GRAPH_FAILED` and refuses to run.
- The data type of `x` determines the data type of the output `y`, and the data types of `rois` and `offset` must be the same as that of `x`.
- Only the FLOAT16 and FLOAT data types are supported, and the INT type is not supported.
- `sampling_ratio` must be within the range `[0, 46340]`; `spatial_scale` must be a finite value greater than 0, and `gamma` must be a finite value.
- When `rois.shape[0]` is 0, the output shape is `(0, C, pooled_h, pooled_w)`. When `N` is 0 and `num_rois` is greater than 0, the output keeps the shape `(num_rois, C, pooled_h, pooled_w)`, and all elements are 0.
- Dynamic shape and dynamic rank are supported: the operator registration enables DynamicShapeSupportFlag and DynamicRankSupportFlag, and the dimensions of `x` can be marked as `-1` (unknown dimension) or `-2` (unknown rank). InferShape allows unknown dimensions and unknown ranks. After the dimensions are materialized, Tiling validates them against the specified ranges such as the preceding `[1, INT32_MAX]`. If the unknown dimension (`-1`) is not materialized, it cannot pass the Tiling validation.
- The batch index of an ROI is truncated to `[0, N - 1]`. NaN or Inf values in ROI/offset are processed as 0, and the sampling points outside `[-1, H] × [-1, W]` do not participate in the accumulation.
- If the grid of adaptive sampling exceeds 46340 in a single direction, the ROI does not generate valid sampling points, and the output is 0.
- At runtime, the `CMakeLists.txt` in the root directory of the operator repository must have completed the compilation configuration.

## Invocation Description

| Invocation Method | Sample Code | Description |
|----------|----------|------|
| GE graph mode | [test_geir_deformable_roi_pool](examples/test_geir_deformable_roi_pool.cpp) | Call the DeformableRoiPool operator by constructing a graph in the [operator IR](op_graph/deformable_roi_pool_proto.h) mode. |
