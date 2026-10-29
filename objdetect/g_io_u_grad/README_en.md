# GIoUGrad

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: Computes the backpropagation gradients of the Generalized Intersection over Union (GIoU) loss function. Given the upstream gradient dy, the predicted boxes bboxes, and the ground-truth boxes gtboxes, the operator computes the gradients of the GIoU loss with respect to bboxes and gtboxes.

- Computation formula:

  $$
  GIoU = \frac{inter}{union} - \frac{enclose - union}{enclose}
  $$

  The backpropagation computes dinter, dunion, and denclose, and distributes the gradients to each coordinate dimension.

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
      <td>dy</td>
      <td>Input</td>
      <td>Upstream gradient, a 1D Tensor with shape=(N,), where N is the number of boxes.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bboxes</td>
      <td>Input</td>
      <td>Predicted box coordinates, a 2D Tensor with shape=(4, N). When trans=true, the format is [x,y,w,h]; when trans=false, the format is [x1,y1,x2,y2].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gtboxes</td>
      <td>Input</td>
      <td>Ground-truth box coordinates, a 2D Tensor with shape=(4, N). The format is the same as that of bboxes.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbboxes</td>
      <td>Output</td>
      <td>Gradient of bboxes, a 2D Tensor with shape=(4, N). The format is the same as that of the input bboxes.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgtboxes</td>
      <td>Output</td>
      <td>Gradient of gtboxes, a 2D Tensor with shape=(4, N). The format is the same as that of the input gtboxes.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>trans</td>
      <td>Attr</td>
      <td>Bool type. true indicates the xywh format, and false indicates the xyxy format. Default value: false.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_cross</td>
      <td>Attr</td>
      <td>Bool type. When the value is false, M=N (one-to-one); when the value is true, M≠N (cross computation). Default value: true. Currently, only is_cross=false is supported.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Attr</td>
      <td>String type. "iou" indicates Intersection over Union, and "iof" indicates Intersection over Foreground. Default value: "iou". Currently, only mode="iou" is supported.</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Currently, only mode="iou" is supported, and mode="iof" is not supported.
- Currently, only is_cross=false (M=N, one-to-one computation) is supported.
- Empty tensors are not supported (an error is reported when N=0).
- Continuous tensors are required (non-continuous tensors are automatically converted to continuous tensors).
- When the input is float16, the operator internally promotes it to float32 for computation to ensure precision.

## Invocation

| Invocation Method   | Sample Code | Description  |
| ------------ | ------------ | ------------ |
| Graph-mode invocation | [test_geir_g_io_u_grad.cpp](./examples/test_geir_g_io_u_grad.cpp) | Invokes the GIoUGrad operator by constructing a graph through the [operator IR](./op_graph/g_io_u_grad_proto.h) |
