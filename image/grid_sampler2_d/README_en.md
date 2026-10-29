# GridSampler2D

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |


## Function Description

- Operator function: This operator performs 2D grid sampling on the four-dimensional input `x` based on the normalized coordinates provided by `grid`. It supports bilinear, nearest-neighbor, and bicubic interpolation.
- Input and output sizes:

  $$
  x: (N, C, H_{in}, W_{in})
  $$

  $$
  grid: (N, H_{out}, W_{out}, 2)
  $$

  $$
  y: (N, C, H_{out}, W_{out})
  $$

  The last dimension of `grid` stores the x and y coordinates in order. The coordinates are usually normalized to [-1, 1]. The actual input coordinates are determined by `align_corners`:

  - `align_corners=true`:

    $$
    x' = \frac{grid_x + 1}{2}(W_{in}-1), \quad y' = \frac{grid_y + 1}{2}(H_{in}-1)
    $$

  - `align_corners=false`:

    $$
    x' = \frac{(grid_x + 1)W_{in}-1}{2}, \quad y' = \frac{(grid_y + 1)H_{in}-1}{2}
    $$

  Out-of-bounds coordinates are processed in the zeros, border, or reflection mode specified by `padding_mode`. The output is then computed in the bilinear, nearest, or bicubic mode specified by `interpolation_mode`.

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input feature map. The shape is (N, C, H<sub>in</sub>, W<sub>in</sub>).</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NCHW</td>
    </tr>
    <tr>
      <td>grid</td>
      <td>Input</td>
      <td>Sampling grid. The shape is (N, H<sub>out</sub>, W<sub>out</sub>, 2), and the data type must be the same as that of x.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Sampling result. The shape is (N, C, H<sub>out</sub>, W<sub>out</sub>), and the data type is the same as that of x.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NCHW</td>
    </tr>
    <tr>
      <td>interpolation_mode</td>
      <td>Optional attr</td>
      <td>Interpolation mode. "bilinear", "nearest", and "bicubic" are supported. The default value is "bilinear".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>Optional attr</td>
      <td>Padding mode. "zeros", "border", and "reflection" are supported. The default value is "zeros".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td>Whether to align the centers of the corner pixels of the input and output. The default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

- Both `x` and `grid` are 4D tensors. Their data types are the same and support only `float16` or `float32`.
- The last dimension of `grid` must be 2, and the batch dimensions of `x` and `grid` must be the same.
- `H_in` and `W_in` of `x` must be greater than 0. Empty tensors are supported when `grid` produces an empty output.
- Each dimension size of `x` and `grid` cannot exceed `INT32_MAX`.
- The product of the height and width of `x` (`H_in * W_in`) cannot exceed `INT32_MAX`.
- The total number of output pixels (`N * H_out * W_out`) cannot exceed `INT32_MAX`.
- The total number of elements of `x` (`N * C * H_in * W_in`) must be within the representable range of a 64-bit signed integer (otherwise, the input is rejected).
- `interpolation_mode` supports only `bilinear`, `nearest`, and `bicubic`.
- `padding_mode` supports only `zeros`, `border`, and `reflection`.
- Shape inference (InferShape) supports dynamic shapes and dynamic ranks. `H_in`/`W_in` of `x` can use `-1` to indicate an unknown dimension (the output height and width are inferred from `grid`), and `-2` indicates an unknown rank when the entire tensor shape is unknown. Tiling consumes the shape materialized at runtime. Unmaterialized `-1`/`-2` dimensions/ranks do not enter Tiling validation. The batch dimension is consistent between `x` and `grid`.
- The operator registers only the logical formats `x`=NCHW, `grid`=ND, and `y`=NCHW. The acceptance/rejection behavior of other logical formats is not declared.
- The `grid` coordinates are normalized coordinates (usually in [-1, 1]). For non-finite coordinates (NaN/±Inf), the behavior is consistent with PyTorch `grid_sample` (`bilinear`/`bicubic` output NaN under `zeros`, and `nearest` maps to valid indexes). For degenerate reflection coordinates with extremely large magnitudes (about `|grid| ≳ 1e8`), this implementation does not guarantee bitwise consistency with PyTorch.
- Inputs that exceed the capacity upper bounds (any dimension greater than `INT32_MAX`, `H_in * W_in` greater than `INT32_MAX`, `N * H_out * W_out` greater than `INT32_MAX`, or `N * C * H_in * W_in` beyond the range of a 64-bit signed integer) are rejected. On the kernel path, this rejection does not expose an independent error code.

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---- | ---- | ---- |
| GE graph mode | [test_geir_grid_sampler2_d](examples/arch35/test_geir_grid_sampler2_d.cpp) | Calls the GridSampler2D operator by constructing a graph in the [operator IR](op_graph/grid_sampler2_d_proto.h) mode. For compilation and verification, refer to [Operator Invocation](../../docs/zh/invocation/quick_op_invocation.md). The sample is located in `examples/arch35`. For compilation and running, you must explicitly specify `--soc=ascend950` (for example, `bash build.sh --run_example grid_sampler2_d graph --soc=ascend950`). |

- This operator is registered in GE graph mode. The custom package is built with `ACLNNTYPE aclnn_exclude` and does not contain the TensorFlow plugin. Therefore, the package does not generate `aclnnGridSampler2D` or TensorFlow APIs. The ACLNN/TF numerical paths use the framework built-in APIs with the same names, and the `cust` routing of the custom package sample does not apply.
