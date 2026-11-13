# ResizeGrad

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: computes the backward gradient of the forward Resize operator. The operator accumulates the gradient grads after resize back to each sampling position of the original resolution using linear (1D) or cubic (2D) interpolation weights, and outputs the gradient y with the same shape as the forward input.

- Calculation formula (in cubic mode, the output point $[h,w]$ aggregates in reverse the input window covering it):

$$
y[n,c,h,w] = \sum_{(o_h,o_w)\in S(h,w)} grads[n,c,o_h,o_w] \cdot c_h(o_h) \cdot c_w(o_w)
$$

  In the formula, $c_h$ and $c_w$ are the cubic convolution interpolation coefficients (the coefficient a is fixed to -0.75), and $S(h,w)$ is the input sampling window covering the output point $[h,w]$. The linear mode is the two-point weighted accumulation of 1D linear interpolation.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 190px">
  <col style="width: 110px">
  <col style="width: 300px">
  <col style="width: 230px">
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
      <td>grads</td>
      <td>Input</td>
      <td>Gradient after resize, a 4D tensor. In cubic mode, the NCHW layout is (N,C,H_out,W_out) and the HWNC layout is (H_out,W_out,N,C); in linear mode, it is (N,C,1,W_out).</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>roi</td>
      <td>Optional input</td>
      <td>Tensor, which takes effect only when coordinate_transformation_mode is "tf_crop_and_resize"; this mode is not supported this time, so the parameter is not used.</td>
      <td>FLOAT16, FLOAT32, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Optional input</td>
      <td>1D float tensor of the scaling coefficients of each dimension, with 2 elements in cubic mode (a single element is broadcast to H/W) and 1 element in linear mode. A finite positive value takes effect (scale = 1/scales[i]); when the parameter is omitted, an element is non-finite (NaN/+/-Inf), or an element is not greater than 0, the coefficients are derived from the sizes.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>original_size</td>
      <td>Input</td>
      <td>1D int32/int64 tensor of the size of the output tensor. In cubic mode, it has 4 elements [N,C,H_in,W_in] (NCHW semantics); in linear mode, it has 3 elements [N,C,W_in].</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>coordinate_transformation_mode</td>
      <td>Attr</td>
      <td>Transformation mode from resized coordinates to original coordinates; only "half_pixel" and "align_corners" are supported this time, default value: "half_pixel".</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubic_coeff_a</td>
      <td>Attr</td>
      <td>Cubic interpolation coefficient; used only in cubic mode; ignored in linear mode; in cubic mode it is fixed to -0.75 and must be finite (NaN/+/-Inf is rejected).</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>exclude_outside</td>
      <td>Attr</td>
      <td>Retained attribute, default value 0; not supported this time and does not participate in the calculation.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>extrapolation_value</td>
      <td>Attr</td>
      <td>Retained attribute, default value 0.0, takes effect only in "tf_crop_and_resize" mode; not supported this time and does not participate in the calculation.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Attr</td>
      <td>Interpolation mode; only "linear" and "cubic" are supported this time, the default value "nearest" is not supported (an error is reported when it is passed in).</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nearest_mode</td>
      <td>Attr</td>
      <td>Retained attribute, default value "round_prefer_floor", used only in nearest mode; not supported this time and does not participate in the calculation.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attr</td>
      <td>Layout of the input data during calculation, supporting "NCHW" and "HWNC", default value "NCHW"; the linear mode supports only "NCHW".</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Gradient of the forward input, whose shape is derived from original_size and whose dtype is the same as that of grads.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- grads: supports only FLOAT16 and FLOAT32, and must be 4D. In cubic mode, the NCHW layout is (N,C,H_out,W_out) and the HWNC layout is (H_out,W_out,N,C); in linear mode, it is (N,C,1,W_out) and the 3rd dimension (dim(2)) must be 1. H_out or W_out may be 0, in which case the output is treated as empty/zero. Non-finite inputs (plus/minus Inf, NaN) in grads are outside the numerical guarantee.
- y: dtype is the same as grads, and shape is derived from original_size; y is (N,C,H_in,W_in) when data_format is "NCHW" and (H_in,W_in,N,C) when it is "HWNC".
- original_size: value-dependent input; each element must be greater than 0, and the first 2 dimensions (N, C) must be consistent with those of grads; in cubic mode it must have 4 elements [N,C,H_in,W_in], and in linear mode 3 elements [N,C,W_in].
- scales: optional; 2 elements in cubic mode (a single element is broadcast to H/W) and 1 element in linear mode; a finite positive value takes effect (scale = 1/scales[i]); extremely small finite positive values may be rejected; when omitted, non-finite (NaN/+/-Inf), or not greater than 0, the coefficients are derived from the input and output sizes; in linear mode, scales[0] must be approximately equal to grads.W/original_size.W (tolerance 1e-4).
- mode: only "linear" and "cubic" are supported; the default value "nearest" and other values are not supported, so callers must explicitly pass "linear" or "cubic".
- data_format: supports "NCHW" and "HWNC", default "NCHW"; linear mode supports only "NCHW".
- coordinate_transformation_mode: only "half_pixel" and "align_corners" are supported, default "half_pixel".
- cubic_coeff_a: validated only in cubic mode; fixed to -0.75 and must be finite (NaN/Inf is rejected); ignored in linear mode.
- roi, exclude_outside, extrapolation_value, nearest_mode: reserved parameters and not used in this calculation.
- Dynamic shape: -1 unknown dims and -2 unknown rank are supported; when the output shape is unknown, the shape of y is unknown. original_size must be a const input to derive a concrete output shape; otherwise the shape of y is unknown.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph mode invocation</td>
    <td><a href="./examples/test_geir_resize_grad.cpp">test_geir_resize_grad</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">operator invocation</a> to complete operator compilation and verification; the operator IR definition is <a href="./op_graph/resize_grad_proto.h">resize_grad_proto.h</a>.</td>
  </tr>
</tbody>
</table>
