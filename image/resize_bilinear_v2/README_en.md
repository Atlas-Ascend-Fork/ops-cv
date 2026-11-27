# ResizeBilinearV2

## Applicable Products

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Resizes an image to a specified size using bilinear interpolation.

- Formula:

  Assume the values of four known points $Q_{11}(x_1, y_1), Q_{12}(x_1, y_2), Q_{21}(x_2, y_1), Q_{22}(x_2, y_2)$ in x. The data is interpolated below to obtain the value at a point coordinate $P(a, b)$ in y.

  $$
  f(R_1) = \frac{x_2 - a}{x_2 - x_1}f(Q_{11}) + \frac{a - x_1}{x_2 - x_1}f(Q_{21})
  $$

  $$
  f(R_2) = \frac{x_2 - a}{x_2 - x_1}f(Q_{12}) + \frac{a - x_1}{x_2 - x_1}f(Q_{22})
  $$

  $$
  f(P) = \frac{y_2 - b}{y_2 - y_1}f(R_{1}) + \frac{b - y_1}{y_2 - y_1}f(R_{2})
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input 4-D image tensor, corresponding to x in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>Height and width of the output image.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Optional attribute</td>
      <td>When true, aligns the centers of the four corner pixels of the input and output images; defaults to false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>half_pixel_centers</td>
      <td>Optional attribute</td>
      <td>When true, samples at half-pixel center coordinates; cannot be true together with align_corners; defaults to false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Optional attribute</td>
      <td>Specifies the data type of output y, supporting FLOAT32, FLOAT16, and BFLOAT16; defaults to FLOAT32.</td>
      <td>TYPE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Optional attribute</td>
      <td>Specifies the scaling factors in the H and W directions. The default value [0.0, 0.0] means the scaling ratio is computed from size.</td>
      <td>LIST_FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Image after bilinear interpolation, corresponding to y in the formula. The data format is the same as x.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
  </tbody></table>

## Constraints

- `align_corners` and `half_pixel_centers` cannot both be true.
- Semantics and valid domain of `scales`: it must have exactly 2 elements, which are the scaling factors in the H and W directions respectively (the ratio of output length to input length). The default value `[0.0, 0.0]` means no override. It **does not change the output shape** — the H and W of `y` are always derived from `size` by shape inference, and `size` is never ignored. `scales` only overrides the scaling factor used by the coordinate mapping: when an element is greater than 0.0, that direction is mapped by `1/scales`; otherwise it falls back to `H_in/H_out` (same for W). When `align_corners` is true the override does not take effect, and the scaling factor is fixed to `(H_in-1)/(H_out-1)`. When the number of elements is not 2, Tiling validation fails.
- Fixed-shape constraint: both `x` and `y` must be 4-D tensors, the N and C dimensions of `y` must be the same as `x`, and the data formats must be consistent (both NCHW or both NHWC).
- Empty-tensor constraint: neither `x` nor `y` supports empty tensors. When the element count of either is 0, Tiling validation fails and returns `GRAPH_FAILED`. Note that the input parameter validation of the aclnn direct-call path (aclnnResize) has **no** dedicated empty-tensor branch, so it does not return a specific L2 error code; an empty tensor is ultimately rejected at the Tiling stage, manifesting as an execution-time failure.
- `size` is a value-dependent input and must be a 1-D INT32 tensor of 2 elements, which are the output height (new_height) and width (new_width), both of which must be positive.
- Non-contiguous tensor: in GE graph mode, `x` and `y` must be contiguous tensors. The aclnn direct-call path (aclnnResize) supports non-contiguous inputs and outputs; the interface performs contiguity conversion internally. For details, see [aclnnResize](docs/aclnnResize.md).

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_resize](examples/test_aclnn_resize.cpp) | Calls the ResizeBilinearV2 operator through the [aclnnResize](docs/aclnnResize.md) interface. |
| Graph mode | -  | Calls the ResizeBilinearV2 operator through the [operator IR](op_graph/resize_bilinear_v2_proto.h) graph construction. |
