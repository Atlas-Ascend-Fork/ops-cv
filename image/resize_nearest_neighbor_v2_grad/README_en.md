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

  In the formula, $H_{in}$ and $W_{in}$ are the height and width of the original image (given by the input `size`, that is, the spatial size of the output `y`), and $H_{out}$ and $W_{out}$ are the height and width of the input `grads`. The effective scale factors scaleH and scaleW are jointly determined by the `align_corners` and `scales` attributes:

  - When align\_corners=False and the corresponding component of scales is greater than 0, the reciprocal of that component is used:

    $$\text{scaleH} = \frac{1}{\text{scales}[0]}, \quad \text{scaleW} = \frac{1}{\text{scales}[1]}$$

  - When align\_corners=False and the corresponding component of scales is less than or equal to 0 (default), the factor is derived from the spatial sizes:

    $$\text{scaleH} = \frac{H_{in}}{H_{out}}, \quad \text{scaleW} = \frac{W_{in}}{W_{out}}$$

  - When align\_corners=True, `scales` is ignored and the factor is derived by corner alignment (0 when the denominator is 1):

    $$\text{scaleH} = \frac{H_{in} - 1}{H_{out} - 1}, \quad \text{scaleW} = \frac{W_{in} - 1}{W_{out} - 1}$$

  When the attribute values are different, the mapping formulas change as follows (each branch uses the corresponding scaleH and scaleW above):

  - align\_corners=True, half\_pixel\_centers=False: $h = \min(\text{round}(h' \times \text{scaleH}), H_{in} - 1)$
  - align\_corners=False, half\_pixel\_centers=True: $h = \min(\lfloor (h' + 0.5) \times \text{scaleH} \rfloor, H_{in} - 1)$
  - align\_corners=False, half\_pixel\_centers=False: $h = \min(\lfloor h' \times \text{scaleH} \rfloor, H_{in} - 1)$
  - align\_corners=True, half\_pixel\_centers=True: illegal combination, not supported. For details, see [Constraints](#constraints).

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
      <td>Height and width of the original image, that is, the spatial size of the forward input, used to determine the H and W dimensions of the output gradient tensor. It is a 1D tensor with fixed 2 elements: orig_height and orig_width in sequence.</td>
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
  | align_corners | Bool | false | Determines whether to align the corner pixels of the input and output images. If True, the corner pixels are aligned, and the effective scale factor is derived as $(H_{in}-1)/(H_{out}-1)$ with `scales` ignored. It cannot be True at the same time as half_pixel_centers. |
  | half_pixel_centers | Bool | false | Determines the pixel center position during interpolation. If True, the pixel center is at the half-pixel coordinate. It cannot be True at the same time as align_corners. |
  | scales | ListFloat | {0.0f, 0.0f} | Multiplier of the spatial size, an optional parameter with fixed 2 elements corresponding to the H and W directions in sequence. When an element is greater than 0, the effective scale factor of that direction takes its reciprocal (that is, $1/\text{scales}[i]$); when it is less than or equal to 0, that direction is derived from the spatial sizes. This attribute is ignored when align_corners is True. |

## Constraints

- Only 4D tensors (NCHW or NHWC format) are supported, and the data type and data format of `grads` and `y` must be consistent.
- `align_corners` and `half_pixel_centers` cannot be True at the same time. This combination is illegal and returns failure directly at the Host validation stage.
- `size` is fixed to a 1D INT32 tensor with 2 elements, whose values must be consistent with the H and W dimensions of the output `y`; the N and C dimensions of `grads` and `y` must be equal.
- Dynamic shape and dynamic rank: shape inference (InferShape) supports the scenario where `size` is not a constant, in which case the H and W of the output `y` are set to `-1`; when `grads` has an unknown rank (represented by `-2`), the output `y` has four `-1` values. Tiling consumes the shape that has been materialized at runtime; the un-materialized `-1`/`-2` dimensions/rank do not enter Tiling validation, and the corresponding constraints take effect only after the shape is materialized.

## Invocation Description

| Invocation Method | Sample Code | Description |
| :---- | :---- | :---- |
| aclnn interface | [test_aclnn_upsample_nearest2d_backward](examples/test_aclnn_upsample_nearest2d_backward.cpp) | Call the ResizeNearestNeighborV2Grad operator through the [aclnnUpsampleNearest2dBackward](../upsample_nearest2d_grad/docs/aclnnUpsampleNearest2dBackward_en.md) interface. |
| aclnn interface | [test_aclnn_upsample_nearest1d_backward](examples/test_aclnn_upsample_nearest1d_backward.cpp) | Call the ResizeNearestNeighborV2Grad operator through the [aclnnUpsampleNearest1dBackward](../upsample_nearest2d_grad/docs/aclnnUpsampleNearest1dBackward_en.md) interface. |
| Graph mode | [test_geir_resize_nearest_neighbor_v2_grad](examples/test_geir_resize_nearest_neighbor_v2_grad.cpp) | Call the ResizeNearestNeighborV2Grad operator by building a graph through [operator IR](op_graph/resize_nearest_neighbor_v2_grad_proto.h), verifying that the output dtype follows the input grads in graph mode. |

> [!NOTE]
>
> This operator does not deliver its own aclnn interface: [CMakeLists.txt](CMakeLists.txt) declares `ACLNNTYPE aclnn_exclude`, the op_api header file of the installation package is empty, and `libcust_opapi.so` has no aclnn export symbols.
> `aclnnUpsampleNearest2dBackward` and `aclnnUpsampleNearest1dBackward` in the table above are built-in CANN interfaces, and their Host-side implementation is in [upsample_nearest2d_grad](../upsample_nearest2d_grad); on <term>Ascend 950PR&950DT products</term>, after the interface is dispatched, the Kernel is executed by this operator.
