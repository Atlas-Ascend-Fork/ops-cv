# DIoUGrad

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

- Operator function: calculates the backward gradient of the Distance-IoU (DIoU) metric. DIoUGrad is the backward propagation operator of the DIoU operator in object detection. It calculates the gradients of the upstream gradient dy with respect to the predicted boxes bboxes and the ground truth boxes gtboxes.

- Calculation formula:

  DIoU forward metric:

  $$
  DIoU = IoU - \frac{\rho^2}{c^2}
  $$

  In the formula:
  - $IoU = \frac{|B \cap B_{gt}|}{|B \cup B_{gt}|}$ is the intersection over union
  - $\rho^2 = (b_{cx} - b_{gt\_cx})^2 + (b_{cy} - b_{gt\_cy})^2$ is the square of the center point distance
  - $c^2 = c_w^2 + c_h^2$ is the square of the diagonal of the minimum enclosing rectangle

  Backward gradients:

  $$
  \frac{\partial (IoU - \rho^2/c^2)}{\partial B} = f(dy, IoU, \rho^2, c^2)
  $$

  $$
  \frac{\partial (IoU - \rho^2/c^2)}{\partial B_{gt}} = f(dy, IoU, \rho^2, c^2)
  $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 220px">
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
      <td>Upstream gradient, a 1D tensor, where N is the number of boxes</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bboxes</td>
      <td>Input</td>
      <td>Predicted bounding boxes, a 2D tensor with shape (4, N), where 4 refers to [x1,y1,x2,y2] or [x,y,w,h]</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gtboxes</td>
      <td>Input</td>
      <td>Ground truth bounding boxes, a 2D tensor with shape (4, M), where 4 refers to [x1,y1,x2,y2] or [x,y,w,h]</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbboxes</td>
      <td>Output</td>
      <td>Gradient of the predicted boxes, with shape (4, N)</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgtboxes</td>
      <td>Output</td>
      <td>Gradient of the ground truth boxes, with shape (4, M)</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>trans</td>
      <td>Attr</td>
      <td>true: supported, interpreted in the xywh format [x, y, w, h]; false: supported, interpreted in the xyxy format [x1, y1, x2, y2]</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_cross</td>
      <td>Attr</td>
      <td>false: M=N (one-to-one correspondence); true: M≠N (cross calculation). Currently only false is supported</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Attr</td>
      <td>"iou": inter/union; "iof": inter/bboxes_area. Currently only "iou" is supported</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The dtypes of all inputs must be the same (dy, bboxes, and gtboxes must be all float16 or all float32)
- dy must be a 1D tensor
- bboxes and gtboxes must be 2D tensors, and dim[0] must be 4
- The trans attribute supports two formats: true (xywh, that is, [x, y, w, h]) and false (xyxy, that is, [x1, y1, x2, y2])
- The is_cross attribute currently supports only false (M=N)
- The mode attribute currently supports only "iou"
- The shape of the output dbboxes is the same as that of bboxes, and the shape of dgtboxes is the same as that of gtboxes
- Empty tensors are supported (the output shape is (4, 0) when N=0), and the output is an empty gradient in this case
- The internal calculation uses float32 precision, and the fp16 input is internally promoted to fp32

## Invocation Description

| Invocation Method   | Sample Code | Description  |
| ------------ | ------------ | ------------ |
| Graph mode invocation | [test_geir_d_io_u_grad.cpp](./examples/test_geir_d_io_u_grad.cpp) | Call the DIoUGrad operator by constructing a graph in the [operator IR](./op_graph/d_io_u_grad_proto.h) mode |
