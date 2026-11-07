# UpsampleBicubic2dAA

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: applies the bicubic antialiasing algorithm to the input signal composed of multiple input channels for upsampling. If the shape of the input tensor x is (N, C, H, W), the shape of the output tensor out is (N, C, outputSize[0], outputSize[1]).
- Calculation formula: for a 2D interpolation point $(N, C, h, w)$, the interpolation $out(N, C, h, w)$ can be expressed as:

  $$
  {out(N, C, h, w)}=\sum_{i=0}^{kW}\sum_{j=0}^{kH}{W(i, j)}*{f(h_i, w_j)}
  $$

  $$
  scaleH =\begin{cases}
  (x.dim(2)-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  x.dim(2) / outputSize[0] & otherwise
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  (x.dim(3)-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  x.dim(3) / outputSize[1] & otherwise
  \end{cases}
  $$

  In the formula:
  - If alignCorners is true, the corner pixels of the input and output tensors are aligned; if alignCorners is false, the edge pixels of the input and output tensors are aligned.
  - i and j are the index variables of $W(i, j)$.
  - If $scaleH >= 1$, then $kH = 1/scaleH$; otherwise, $kH = 4$
  - If $scaleW >= 1$, then $kW = 1/scaleW$; otherwise, $kW = 4$
  - $h_i = |h| + i$
  - $w_j = |w| + j$
  - $f(h_i, w_j)$ is the pixel value of the original image at $(h_i, w_j)$
  - $W(i, j)$ is the weight of the bicubic antialiasing interpolation, defined as:

    $$
    W(d) =\begin{cases}
    (a+2)|d|^3-(a+3)|d|^2+1 & |d|\leq1 \\
    a|d|^3-5a|d|^2+8a|d|-4a & 1<|d|<2 \\
    0 & otherwise
    \end{cases}
    $$

    In the formula:
    - In the antialiasing scenario, $a=-0.5$.
    - $d = |(h, w) - (h_i, w_j)|$

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
      <td>Represents the input tensor for upsampling, corresponding to `x` in the formula. The data type is the same as the data type of the output parameter `y`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>Represents the spatial size of `y` in the H and W dimensions, corresponding to `outputSize` in the formula. The size is 2.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attr</td>
      <td><ul><li>Determines whether to align the corner pixels, corresponding to `alignCorners` in the formula. If align_corners is true, the corner pixels of the input and output tensors are aligned; otherwise, they are not aligned.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the height dimension of the spatial size, corresponding to `scalesH` in the formula.</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the width dimension of the spatial size, corresponding to `scalesW` in the formula.</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor after sampling, corresponding to `out` in the formula. The data type is the same as the data type of the input parameter `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- Shape constraints:
  - Both `x` and `y` are 4-dimensional and are interpreted using NCHW semantics.
  - The N and C axes of `y` are the same as those of `x`; the H and W axes of `y` are the same as `output_size`.
  - The C, H, and W axes of `x` must all be greater than 0.
- Format constraint: The data formats of `x` and `y` must be the same (both `ND` or both `NCHW`); otherwise, the operator fails validation.
- Empty tensor: When invoked through the aclnn API, an empty tensor whose N axis is 0 is supported. In this case, the first-phase API returns success and no computation is performed. If any of the C, H, or W axes of `x` is 0, 161002 is returned.
- The following upper bounds must be guaranteed by the caller. The operator does not validate them, and the behavior is undefined when they are exceeded:
  - The value of each dimension of `x` and `y` is less than or equal to 2^20.
  - N \* C \* x_H < 2^31.
  - The memory usage must be less than 60 GB. The formula for memory usage is as follows:

    $$
    (x\_H * x\_W + y\_H * y\_W + x\_H * y\_W) * N * C * sizeof(dtype) < 60 * 1024 * 1024 * 1024
    $$

- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: In the input data scaling scenario, the downscaling factor must be less than or equal to 50, that is, the H axis of `x` / `output_size[0]` <= 50 and the W axis of `x` / `output_size[1]` <= 50.
- Attribute constraints:
  - The size of `output_size` is 2, and each element is in the range [1, 2^20] (the upper bound must be guaranteed by the caller and is not validated by the operator).
  - When `scales_h` is greater than 0, `output_size[0]` = floor(x_H \* `scales_h`) must hold; when `scales_w` is greater than 0, `output_size[1]` = floor(x_W \* `scales_w`) must hold. When `scales_h` and `scales_w` take the default value 0.0, the constraints above do not apply. In this case, the scaling ratio is determined by the ratio between `x` and `output_size`.
- Dynamic shape:
  - The shape of the aclnn API is carried by aclTensor and must be a concrete 4-dimensional value. The -1/-2 dynamic placeholders are not supported.
  - This operator declares dynamic shape and dynamic rank support on <term>Ascend 950PR&950DT products</term>, <term>Kirin X90 processor series products</term>, and <term>Kirin 9030 processor series products</term>. The GE graph mode is currently unavailable, so the -1/-2 dynamic placeholder scenarios are not supported.
- Numerical constraints: When the input contains ±inf, NaN, or extreme values close to the representable upper bound of the data type (for example, ±3.4e38 for FLOAT32 and ±65504 for FLOAT16), the anti-aliasing weighted accumulation may overflow to ±inf or produce NaN. The output is not guaranteed to match the CPU reference element-wise, and such inputs are outside the precision guarantee.
- Deterministic computation: This operator defaults to a deterministic implementation.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_bicubic2d_aa](examples/test_aclnn_upsample_bicubic2d_aa.cpp) | Call the UpsampleBicubic2dAA operator through the [aclnnUpsampleBicubic2dAA](docs/aclnnUpsampleBicubic2dAA_en.md) interface. |

Current pathway support:

- aclnn direct invocation: Supported. See the table above.
- GE graph mode: Currently not supported. The operator does not register shape and dtype inference for graph mode.
- PyTorch graph mode: torchair has no converter for `aten._upsample_bicubic2d_aa.default`; the const/dynamic graph modes cannot be executed, and only the aclgraph mode can be executed.
- TensorFlow: This operator does not provide a TensorFlow interface or binding.
