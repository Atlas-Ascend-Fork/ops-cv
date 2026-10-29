# Dilation2DBackpropInput

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: This is the backpropagation (input gradient) operator of the morphological dilation operation Dilation2D. Given the forward input `x`, the filter `filter`, and the output gradient `out_backprop`, the operator computes the input gradient `in_backprop`.

- Computation formula:

Forward Dilation2D formula:

$$
output[b, h_{out}, w_{out}, d] = \max_{fh, fw} \left( x[b, h_{out} \cdot stride_h + fh \cdot rate_h, w_{out} \cdot stride_w + fw \cdot rate_w, d] + filter[fh, fw, d] \right)
$$

Backward BackpropInput formula:

$$
in\_backprop[b, *, *, d] = 0
$$

$$
in\_backprop[b, h_{in\_max}, w_{in\_max}, d] \mathrel{+}= out\_backprop[b, h_{out}, w_{out}, d]
$$

Here $(h_{in\_max}, w_{in\_max})$ is the position where `x + filter` reaches the maximum value in the window during the forward computation (`>` strictly greater than is used, and the first maximum value is selected). Out-of-bounds positions do not participate in argmax.

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
    <td>Forward input tensor. It is 4D and corresponds to x in the formula. In NHWC format, the shape is (N, H, W, C). In NCHW format, the shape is (N, C, H, W).</td>
    <td>float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>filter</td>
    <td>Input</td>
    <td>Filter. It is 3D and corresponds to filter in the formula. In NHWC format, the shape is (Hf, Wf, C). In NCHW format, the shape is (C, Hf, Wf).</td>
    <td>float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out_backprop</td>
    <td>Input</td>
    <td>Output gradient. It is 4D, and its shape is the same as that of the forward output. In NHWC format, the shape is (N, Ho, Wo, C).</td>
    <td>float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Input gradient. It is 4D, its shape and dtype are the same as those of x, and it corresponds to in_backprop in the formula.</td>
    <td>float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>strides</td>
    <td>Attr</td>
    <td>Sliding window strides. It is a list of length 4. The N/C dimensions must be 1.</td>
    <td>ListInt</td>
    <td>-</td>
  </tr>
  <tr>
    <td>rates</td>
    <td>Attr</td>
    <td>Dilation rates. It is a list of length 4. The N/C dimensions must be 1.</td>
    <td>ListInt</td>
    <td>-</td>
  </tr>
  <tr>
    <td>padding_mode</td>
    <td>Attr</td>
    <td>Padding mode. The value is "SAME", "VALID", or "CALCULATED". The default value is "SAME".</td>
    <td>String</td>
    <td>-</td>
  </tr>
  <tr>
    <td>pads</td>
    <td>Attr</td>
    <td>Explicit padding values [pad_top, pad_bottom, pad_left, pad_right]. This parameter takes effect only in CALCULATED mode. The default value is {0, 0, 0, 0}.</td>
    <td>ListInt</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ceil_mode</td>
    <td>Attr</td>
    <td>Whether to use ceil to compute the output size in CALCULATED mode. true indicates ceil, and false indicates floor. The default value is false.</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>data_format</td>
    <td>Attr</td>
    <td>Data format. The value is "NHWC" or "NCHW". The default value is "NHWC".</td>
    <td>String</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- **Data type constraints**: The dtypes of x, filter, out_backprop, and y are the same. Only float32 is supported.
- **Dimension constraints**: x is 4D (rank=4), filter is 3D (rank=3), and out_backprop is 4D (rank=4).
- **Shape constraints**:
  - The C dimension of x must equal the depth dimension of filter (NHWC: `x.dim(3) == filter.dim(2)`; NCHW: `x.dim(1) == filter.dim(0)`).
  - The N/C dimensions of out_backprop are the same as those of x. Ho/Wo are derived from the H/W of x, strides, rates, and padding_mode.
  - The shape of the output y is exactly the same as that of x.
- **Attribute constraints**:
  - The length of strides must be 4, and the N/C dimensions must be 1 (stride_n=1, stride_c=1).
  - The length of rates must be 4, and the N/C dimensions must be 1 (rate_n=1, rate_c=1).
  - The value of padding_mode must be one of {"SAME", "VALID", "CALCULATED"}.
  - The length of pads must be 4. In CALCULATED mode, pads[i] must be less than window_h/w (window_h = (filter_h - 1) * rate_h + 1).
  - The value of data_format must be one of {"NHWC", "NCHW"}.
- **Format constraints**: The formats of x, out_backprop, and y are the same, and all are ND. The format must be consistent with data_format.
- **Value constraints**: The argmax comparison uses `>` (strictly greater than). The position corresponding to the first maximum value is selected for gradient accumulation, and equal values are not accumulated to multiple positions. In CALCULATED mode, the padding area is filled with an extremely small value to ensure that it does not participate in argmax.

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
    <td><a href="./examples/test_geir_dilation2_d_backprop_input.cpp">test_geir_dilation2_d_backprop_input</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
