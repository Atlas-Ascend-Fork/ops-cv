# ResizeNearestNeighborV2Grad

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: backpropagation of [ResizeNearestNeighborV2](../resize_nearest_neighbor_v2/README_en.md). According to the mapping relationship of nearest neighbor interpolation, the operator scatters the output gradient back to the input space and accumulates it.
- Calculation formula:

  In the forward ResizeNearestNeighborV2, the value of each output pixel is copied from a certain input pixel. During backpropagation, the gradient of an input pixel is the accumulated sum of the gradients of all output pixels mapped to that position:

  $$y(N, C, h, w) += \text{grads}(N, C, h', w')$$

  In the formula, $(h', w')$ is the output coordinate, which is mapped back to the input coordinate through nearest neighbor mapping:

  $$h = \min(\lfloor h' \times \text{scaleH} \rfloor, H_{in} - 1)$$

  $$w = \min(\lfloor w' \times \text{scaleW} \rfloor, W_{in} - 1)$$

  $$\text{scaleH} = \frac{H_{in}}{H_{out}}, \quad \text{scaleW} = \frac{W_{in}}{W_{out}}$$

  When the attribute values are different, the mapping formulas change as follows:

  - align\_corners=True, half\_pixel\_centers=False: $h = \min(\text{round}(h' \times \text{scaleH}), H_{in} - 1)$
  - align\_corners=False, half\_pixel\_centers=True: $h = \min(\lfloor (h' + 0.5) \times \text{scaleH} \rfloor, H_{in} - 1)$
  - align\_corners=False, half\_pixel\_centers=False: $h = \min(\lfloor h' \times \text{scaleH} \rfloor, H_{in} - 1)$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 223px">
  <col style="width: 101px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>grads</td>
      <td>Input</td>
      <td>Gradient tensor of the output of the forward ResizeNearestNeighborV2, corresponding to `grads` in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>Height and width of the original image, that is, the spatial size of the forward input, used to determine the H and W dimensions of the output gradient tensor.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Gradient tensor of the input of the forward ResizeNearestNeighborV2, corresponding to `y` in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
  </tbody>
  </table>

  **Attribute description:**

  | Attribute name | Type | Default value | Description |
  | :---- | :---- | :---- | :---- |
  | align_corners | Bool | false | Determines whether to align the corner pixels of the input and output images. If True, the corner pixels are aligned. |
  | half_pixel_centers | Bool | false | Determines the pixel center position during interpolation. If True, the pixel center is at the half-pixel coordinate. |
  | scales | ListFloat | {0.0f, 0.0f} | Multiplier of the spatial size, an optional parameter. |

## Constraints

- Only 4D tensors (NCHW or NHWC format) are supported

## Invocation Description

| Invocation Method | Sample Code | Description |
| :---- | :---- | :---- |
| aclnn interface | [test_aclnn_upsample_nearest2d_backward](examples/test_aclnn_upsample_nearest2d_backward.cpp) | Call the ResizeNearestNeighborV2Grad operator through the [aclnnUpsampleNearest2dBackward](../upsample_nearest2d_grad/docs/aclnnUpsampleNearest2dBackward_en.md) interface. |
| aclnn interface | [test_aclnn_upsample_nearest1d_backward](examples/test_aclnn_upsample_nearest1d_backward.cpp) | Call the ResizeNearestNeighborV2Grad operator through the [aclnnUpsampleNearest1dBackward](../upsample_nearest2d_grad/docs/aclnnUpsampleNearest1dBackward_en.md) interface. |
| Graph mode | [test_geir_resize_nearest_neighbor_v2_grad](examples/test_geir_resize_nearest_neighbor_v2_grad.cpp) | Call the ResizeNearestNeighborV2Grad operator by building a graph through [operator IR](op_graph/resize_nearest_neighbor_v2_grad_proto.h), verifying that the output dtype follows the input grads in graph mode. |
