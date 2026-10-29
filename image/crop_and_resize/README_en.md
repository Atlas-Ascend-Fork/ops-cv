# CropAndResize

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: This operator extracts multiple crop regions from the input image and resizes them to the specified size. It supports bilinear interpolation and nearest-neighbor interpolation.
- Computation formula:

  The shape of the input x is (batch, image_height, image_width, depth). For the $i$-th box ($i \in [0, num\_boxes)$), let $b = box\_index[i]$ and $(y_1, x_1, y_2, x_2) = boxes[i]$. The crop_size of the cropped image has the shape (crop_height, crop_width). The value of the output $y$ at position $(i, p_y, p_x, d)$ is computed by the following formulas.

  - When method = bilinear:

    1. Compute the scaling ratios $height\_scale$ and $width\_scale$:

       $$
       height\_scale = \begin{cases} (y_2 - y_1) \times (image\_height - 1) / (crop\_height - 1), & crop\_height > 1 \\ 0, & crop\_height = 1 \end{cases}
       $$

       $$
       width\_scale = \begin{cases} (x_2 - x_1) \times (image\_width - 1) / (crop\_width - 1), & crop\_width > 1 \\ 0, & crop\_width = 1 \end{cases}
       $$

    2. Map the output coordinates $(p_y, p_x)$ to the input image coordinates $(in\_y, in\_x)$:

       $$
       in\_y = \begin{cases} y_1 \times (image\_height - 1) + p_y \times height\_scale, & crop\_height > 1 \\ 0.5 \times (y_1 + y_2) \times (image\_height - 1), &     crop\_height = 1 \end{cases}
       $$

       $$
       in\_x = \begin{cases} x_1 \times (image\_width - 1) + p_x \times width\_scale, & crop\_width > 1 \\ 0.5 \times (x_1 + x_2) \times (image\_width - 1), & crop\_width =     1 \end{cases}
       $$

       If $in\_y < 0$, $in\_y > image\_height - 1$, $in\_x < 0$, or $in\_x > image\_width - 1$, then:

       $$
       y(i, p_y, p_x, d) = extrapolation\_value
       $$

       Otherwise, bilinear interpolation is performed. Let $top = \lfloor in\_y \rfloor$, $bottom = \lceil in\_y \rceil$, $left = \lfloor in\_x \rfloor$, $right = \lceil in\_x \rceil$, $y\_ratio = in\_y - top$, and $x\_ratio = in\_x - left$. Then:

       $$
       \begin{aligned}
       y(i, p_y, p_x, d) =\; & (1 - y\_ratio)(1 - x\_ratio) \cdot x(b, top, left, d) \\
       +\; & (1 - y\_ratio) \cdot x\_ratio \cdot x(b, top, right, d) \\
       +\; & y\_ratio \cdot (1 - x\_ratio) \cdot x(b, bottom, left, d) \\
       +\; & y\_ratio \cdot x\_ratio \cdot x(b, bottom, right, d)
       \end{aligned}
       $$

  - When method = nearest:

    The computations of $height\_scale$, $width\_scale$, $in\_y$, and $in\_x$ are the same as those in the bilinear method.

    If $in\_y$ or $in\_x$ is out of bounds (with the same conditions as bilinear), then:

    $$
    y(i, p_y, p_x, d) = extrapolation\_value
    $$

    Otherwise, the nearest-neighbor pixel is used:

    $$
    y(i, p_y, p_x, d) = x(b, round(in\_y), round(in\_x), d)
    $$

    where $round(\cdot)$ denotes rounding to the nearest integer.

  - When method = bilinear_v2:

    1. First maps the normalized box coordinates to pixel coordinates on the input image, and computes the crop window. Let $map(c, n) = \lfloor c \times n \rfloor$ ($n - 1$ is used when $c = 1$), $y_{1o} = map(y_1, image\_height)$, $x_{1o} = map(x_1, image\_width)$, $y_{2o} = map(y_2, image\_height)$, and $x_{2o} = map(x_2, image\_width)$. The height $h$ and width $w$ of the crop window are:

       $$
       h = |y_{2o} - y_{1o}| + 1, \quad w = |x_{2o} - x_{1o}| + 1
       $$

    Let $s_y=1$ when $y_{2o} \ge y_{1o}$, and $s_y=-1$ otherwise. $s_x$ is defined in the same way for the width direction. Sampling proceeds along the endpoint order, and a reversed box is flipped on the corresponding axis.

    2. For the output position $(p_y, p_x)$, compute the floating-point indexes $r_y$ and $r_x$ in the crop window:

       $$
       r_y = (p_y + 0.5) \times h / crop\_height - 0.5
       $$

       $$
       r_x = (p_x + 0.5) \times w / crop\_width - 0.5
       $$

    3. Compute the interpolation indexes and weights. Let $clamp(v, lo, hi) = \max(\min(v, hi), lo)$. Then:

       $$
       lower\_y = clamp(\lfloor r_y \rfloor, 0, h - 1), \quad upper\_y = clamp(\lceil r_y \rceil, 0, h - 1)
       $$

       $$
       lower\_x = clamp(\lfloor r_x \rfloor, 0, w - 1), \quad upper\_x = clamp(\lceil r_x \rceil, 0, w - 1)
       $$

       $$
       y\_ratio = r_y - lower\_y, \quad x\_ratio = r_x - lower\_x
       $$

    4. Check the global image coordinates. After multiplying the relative indexes in the window by the corresponding directions $s_y$ and $s_x$ and adding them to the window start point, if any sampling index is out of the image range, all channels of the current output use `extrapolation_value`, and out-of-bounds pixels are not read. When the box coordinates exceed `[0, 1]`, the floating-point image coordinates computed along the sampling direction also need to be checked, and values beyond the image boundary are also extrapolated. When the box coordinates are within `[0, 1]`, the window edges are clamped by the preceding formulas. Otherwise, bilinear interpolation is performed:

       $$
       \begin{aligned}
       y(i, p_y, p_x, d) =\; & (1 - y\_ratio)(1 - x\_ratio) \cdot x(b, y_{1o} + s_y \cdot lower\_y, x_{1o} + s_x \cdot lower\_x, d) \\
       +\; & (1 - y\_ratio) \cdot x\_ratio \cdot x(b, y_{1o} + s_y \cdot lower\_y, x_{1o} + s_x \cdot upper\_x, d) \\
       +\; & y\_ratio \cdot (1 - x\_ratio) \cdot x(b, y_{1o} + s_y \cdot upper\_y, x_{1o} + s_x \cdot lower\_x, d) \\
       +\; & y\_ratio \cdot x\_ratio \cdot x(b, y_{1o} + s_y \cdot upper\_y, x_{1o} + s_x \cdot upper\_x, d)
       \end{aligned}
       $$

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
      <td>A four-dimensional Tensor. Its shape is (batch, image_height, image_width, depth), where image_height and image_width must be positive numbers. It corresponds to the `input x` in the computation formula description.</td>
      <td>UINT8, UINT16, INT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>boxes</td>
      <td>Input</td>
      <td>A two-dimensional Tensor. Its shape is (num_boxes, 4). Row i specifies the position of a box in the image corresponding to box_index[i], represented by normalized coordinates [y1, x1, y2, x2]. bilinear/nearest maps the normalized y coordinate to the image coordinate through y*(image_height - 1). bilinear_v2 computes the window endpoints using the preceding map formula (the exact coordinate 1 maps to the last valid pixel). y1 > y2 is allowed. In this case, the sampled crop region is the vertically flipped version of the original image. The width dimension is processed in a similar way. When the normalized coordinate values are out of the range [0, 1], the extrapolation_value is used to fill in the extrapolated values.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>box_index</td>
      <td>Input</td>
      <td>Index of boxes. It is a one-dimensional tensor with the shape (num_boxes). Value range: [0, batch). The value of box_index[i] specifies the image corresponding to the i-th box.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>crop_size</td>
      <td>Input</td>
      <td>A 1-D Tensor that contains 2 elements. Its shape is (crop_height, crop_width). All cropped image regions are resized to this size. The aspect ratio of the image content is not preserved. crop_height and crop_width must be positive numbers. It corresponds to `crop_size` in the computation formula description.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>extrapolation_value</td>
      <td>Optional attr</td>
      <td><ul><li>Optional extrapolation value.</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>method</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the sampling method for resizing. It is an optional string. It corresponds to `method` in the computation formula.</li><li>The available methods are "bilinear", "nearest", and "bilinear_v2".</li><li>The default value is "bilinear".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Returns a four-dimensional Tensor with the shape (num_boxes, crop_height, crop_width, depth). It corresponds to the `output y` in the computation formula description.</td>
      <td>FLOAT, FLOAT16</td>
      <td>NHWC</td>
    </tr>
  </tbody></table>

## Constraints

This operator provides two implementations: **AiCore** and **AiCpu**. In graph mode, the GE engine performs automatic routing. When the inputs satisfy the AiCore constraints, AiCore executes the operator preferentially. When the inputs do not satisfy the AiCore constraints, execution automatically falls back to AiCpu.

**AiCore implementation constraints:**

- Data types: x supports only FLOAT16/FLOAT. boxes supports FLOAT/FLOAT16 (the data type of y is the same as that of boxes).
- The input shape must be static. Dynamic shapes are executed by AiCpu.
- method supports only bilinear.
- Neither crop_height nor crop_width exceeds 16, and crop_height × crop_width ≤ 32765.
- num_boxes (boxes.shape[0]) must satisfy 50 < num_boxes ≤ 4000.
- depth (the number of channels, x.shape[3]) must satisfy 256 ≤ depth ≤ 2048.
- image_height × image_width ≤ 65530 (when x is FLOAT16) / 32765 (when x is FLOAT).

**AiCpu implementation capabilities:**

- Data types: x supports UINT8/INT8/UINT16/INT16/INT32/INT64/FLOAT16/FLOAT/DOUBLE. boxes supports only FLOAT. y is FLOAT.
- method supports bilinear/nearest/bilinear_v2.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| Graph mode invocation | [test_geir_crop_and_resize](./examples/test_geir_crop_and_resize.cpp) | Calls the CropAndResize operator by constructing a graph in the [operator IR](./op_graph/crop_and_resize_proto.h) mode. |
