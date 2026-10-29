# Yolo

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

- Operator function: Reorganizes and activates the detection feature maps of the YOLOv2/v3 object detection networks, and converts the original convolution output into three outputs: detection box coordinates, object confidence, and class probabilities. These outputs are used by the subsequent Yolov3DetectionOutput operator.

- Computation formula:

The input x has the shape (N, boxes*(coords+1+classes), H, W), and the channel dimension C is arranged by (elem, box), that is, the x coordinates of all anchor boxes are arranged first, then the y coordinates of all anchor boxes, and so on.

Coordinate processing:

- The x and y coordinates are activated by sigmoid.
- The w and h coordinates are directly moved, and the positions of w and h are exchanged in the output. The output arrangement is (x, y, h, w).

$$
sigmoid(x) = \frac{1}{1 + exp(-x)}
$$

The object confidence and class probabilities are processed according to yolo_mode, which is determined by the three attributes yolo_version, softmax, and background:

- YOLO_MODE_1 (V3, or V2 with softmax=false and background=false): obj=sigmoid, classes=sigmoid
- YOLO_MODE_2 (V2 with softmax=true and background=false): obj=sigmoid, classes=softmax
- YOLO_MODE_3 (V2 with softmax=false and background=true): obj=move, classes=sigmoid
- YOLO_MODE_4 (V2 with softmax=true and background=true): obj and classes are processed by softmax together

## Parameter Description

<table><thead>
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
    <td>Convolution layer output feature map, with shape (N, boxes*(coords+1+classes), H, W). The channel dimension C=boxes*(coords+1+classes), arranged by (elem, box). The NCHW data arrangement is supported.</td>
    <td>float16, float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>coord_data</td>
    <td>Output</td>
    <td>Detection box coordinates, with shape (N, boxes*coords, CeilX(H*W*2+32,32)/2). The channel arrangement is (x, y, h, w). The last dimension of the output uses ceilx memory alignment, and the valid data is the first H*W elements.</td>
    <td>float16, float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>obj_prob</td>
    <td>Output</td>
    <td>Object confidence, with shape (N, CeilX(boxes*H*W*2+32,32)/2). The last dimension of the output uses ceilx memory alignment, and the valid data is the first boxes*H*W elements.</td>
    <td>float16, float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>classes_prob</td>
    <td>Output</td>
    <td>Class probabilities, with shape (N, classes, CeilX(boxes*H*W*2+32,32)/2). The last dimension of the output uses ceilx memory alignment, and the valid data is the first boxes*H*W elements.</td>
    <td>float16, float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>boxes</td>
    <td>Attr</td>
    <td>Number of anchor boxes. The value is greater than 0. Default value: 3 (5 by default for YOLOv2).</td>
    <td>Int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>coords</td>
    <td>Attr</td>
    <td>Number of coordinate parameters, fixed to 4, corresponding to x, y, w, and h. Default value: 4.</td>
    <td>Int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>classes</td>
    <td>Attr</td>
    <td>Number of predicted classes. The value range is [1, 1024]. Default value: 80.</td>
    <td>Int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>yolo_version</td>
    <td>Attr</td>
    <td>YOLO version. The value is "V2" or "V3". Default value: "V3".</td>
    <td>String</td>
    <td>-</td>
  </tr>
  <tr>
    <td>softmax</td>
    <td>Attr</td>
    <td>Whether to perform softmax on the class probabilities. It is valid only when yolo_version is "V2". Default value: false.</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>background</td>
    <td>Attr</td>
    <td>Controls the operation types of obj and classes. It is valid only when yolo_version is "V2". Default value: false.</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>softmaxtree</td>
    <td>Attr</td>
    <td>Fixed to false and not used. Default value: false.</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- The input x must be in the 4D NCHW format, with shape (N, boxes*(coords+1+classes), H, W).
- The channel dimension C must satisfy C = boxes*(coords+1+classes). Otherwise, the result is undefined.
- coords is fixed to 4, corresponding to the four coordinate values x, y, w, and h.
- boxes must be greater than 0.
- The value range of classes is [1, 1024].
- H and W must be greater than 0, that is, H*W > 0. Empty space dimensions are not supported.
- yolo_version supports only "V2" or "V3".
- The input x must be a continuous tensor.
- When the input is float16, intermediate computations are promoted to the float32 precision.
- The channel arrangement of the output coord_data is (x, y, h, w). Note that the positions of w and h are exchanged compared with the input.
- The softmax and background attributes are valid only when yolo_version is "V2".
- The last dimension of the output shape uses ceilx memory alignment, which is consistent with the 1.0 infershape of canndev:
  - Last dimension of coord_data: CeilX(H*W*2+32, 32)/2, the valid data is the first H*W elements, and the padding area is filled with zeros.
  - Last dimension of obj_prob: CeilX(boxes*H*W*2+32, 32)/2, the valid data is the first boxes*H*W elements, and the padding area is filled with zeros.
  - Last dimension of classes_prob: CeilX(boxes*H*W*2+32, 32)/2, the valid data is the first boxes*H*W elements, and the padding area is filled with zeros.
  - CeilX(size, align) = (size + align - 1) / align * align, where size is the number of elements multiplied by 2 (float16 occupies 2 bytes), and align is 32 (32-byte alignment).
- The operator is deterministic: sigmoid uses the standard formula 1/(1+exp(-x)), and softmax uses a numerically stable 3-pass implementation with max subtraction (each thread independently accumulates in sequence, without cross-thread reduction). The same input produces exactly the same result across multiple runs.
- When the input is float16, intermediate computations are promoted to the float32 precision, and the final result is converted back to float16. float16 subnormal values ([2^-24, 2^-14)) are preserved by manually constructing bit patterns, bypassing the hardware FTZ (Flush-To-Zero) mode.
- The operator uses multi-core multi-thread parallelism: Based on the global core index of GetBlockIdx/GetBlockNum, each core has 512 threads, and Grid-Stride loops distribute the workload.

## Invocation

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/arch35/test_geir_yolo.cpp">test_geir_yolo</a></td>
    <td>Invokes the Yolo operator by constructing a graph through the <a href="./op_graph/yolo_proto.h">operator IR</a>. Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
