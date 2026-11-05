# ResizeBicubicV2Grad

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |

## Function Description

- Operator function: computes the backward gradient of the 2D bicubic interpolation (ResizeBicubicV2). The forward interpolation resizes the original image `original_image` to the size of `grads` using bicubic interpolation weights; this operator scatters the upstream gradient `grads` back to the original image size with the same interpolation weights, yielding `y`.
- Calculation formula:

  The bicubic interpolation weight function $W(x)$ (where the constant $a=-0.75$):

  $$
  W(x) = \begin{cases}
  (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{for } |x|≤1 \\
  a|x|^3 -5a|x|^2 + 8a|x| - 4a  & \text{for } 1<|x|<2 \\
  0 & \text{otherwise} \\
  \end{cases}
  $$

  $$
  \frac{\partial L}{\partial X_{i,j}} = \sum_{i'} \sum_{j'} \frac{\partial L}{\partial Y_{i',j'}} \times W(i' - i) \times W(j' - j)
  $$

- Mapping between formula variables and operator parameters:

  | Formula variable | Operator parameter | Description |
  | :---- | :---- | :---- |
  | grads | input grads | upstream gradient, a tensor of the forward ResizeBicubicV2 output size (gradOut in the formula) |
  | y | output y | gradient of the original image (gradInput in the formula), with the same shape as original_image |
  | original_image | input original_image | original input image of the forward ResizeBicubicV2; only its shape (N, C, H, W) is consumed |
  | i', j', i, j | - | element coordinates on the H/W axes |
  | scales | attribute scales | scaling coefficients; see below and Parameter Description for the effective conditions |
  | align_corners | attribute align_corners | corner alignment switch, deciding how the source coordinates are calculated |

- Effective conditions of the scaling coefficients `scales` (consistent with the implementation and the public [aclnnUpsampleBicubic2dBackward](../upsample_bicubic2d_grad/docs/aclnnUpsampleBicubic2dBackward_en.md) documentation):
  - When align_corners=true, the scales attribute is ignored and the scaling coefficients are derived from the input/output H and W axis lengths ((H_original-1)/(H_grads-1) and (W_original-1)/(W_grads-1) when the corresponding axis lengths are greater than 1);
  - When align_corners=false and the corresponding scales component is greater than 0, that component takes effect with the effective scaling coefficient 1/scales;
  - When align_corners=false and the corresponding scales component is not greater than 0, that component is ignored and the scaling coefficient is derived from the corresponding axis lengths (H_original/H_grads, W_original/W_grads).

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1005px">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 300px">
    <col style="width: 250px">
    <col style="width: 150px">
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
      <td>grads</td>
      <td>Input</td>
      <td>Upstream gradient of backward propagation, a tensor of the forward ResizeBicubicV2 output size, corresponding to gradOut in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>original_image</td>
      <td>Input</td>
      <td>Original input image tensor of the forward ResizeBicubicV2; only its shape (N, C, H, W) is consumed, corresponding to the original image size in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Gradient of original_image, corresponding to gradInput in the formula, with the same shape as original_image.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Attr</td>
      <td>Optional, Bool, default false. Corner alignment switch: true means the corner pixels of the input and output tensors are aligned; false means the interpolation coordinates are computed using half pixel centers. It also decides the effective conditions of the scales attribute (see Function Description).</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Attr</td>
      <td>Optional, ListFloat, default {0.0, 0.0}, length must be 2. The two elements represent the H-axis and W-axis subscript ratios of grads to y (grads_H/y_H and grads_W/y_W, i.e. the forward output-to-input spatial size ratio). A component takes effect only when align_corners=false and it is greater than 0, with the effective scaling coefficient 1/scales; ignored when align_corners=true, derived from the input/output H and W axis lengths (see Function Description).</td>
      <td>LIST_FLOAT</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

- grads, original_image and y are 4D tensors; the three must share the same data type and N, C dimensions; the shape of y is the same as original_image.
- Every dimension must be greater than 0 (empty tensors are not supported; zero-length axes are rejected at the tiling stage).
- The scales attribute must contain exactly 2 elements.
- The direct Kernel invocation (operator IR graph mode) is implemented and verified with contiguous tensors; the semantics of non-contiguous inputs are not defined in the public contract (the aclnn API path normalizes non-contiguous inputs internally).

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---- | ---- |----|
| aclnn interface | [test_aclnn_resize_bicubic_v2_grad](examples/test_aclnn_resize_bicubic_v2_grad.cpp) | Call the ResizeBicubicV2Grad operator through the [aclnnUpsampleBicubic2dBackward](../upsample_bicubic2d_grad/docs/aclnnUpsampleBicubic2dBackward_en.md) interface. |
| Graph mode | -  | Call the ResizeBicubicV2Grad operator by building a graph through [operator IR](op_graph/resize_bicubic_v2_grad_proto.h). |

- [repro_timeout_bicubic_grad.cpp](examples/repro_timeout_bicubic_grad.cpp) under examples/ is a reproduction script for the historical issue fixed by the split-K path: it constructs an 8.6GB gradient input (grads (1,1,2147483649,1) -> y (1,1,2,1)) to verify that the extreme upsample backward no longer triggers the vector-core watchdog timeout. It is a deliverable, not a general invocation sample, and should be compiled and run on demand in an environment with the fixed operator package installed.
