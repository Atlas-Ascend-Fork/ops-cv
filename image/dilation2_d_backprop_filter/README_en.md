# Dilation2DBackpropFilter

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

- Operator function: Dilation2DBackpropFilter computes the filter gradients in the backpropagation of the 2D morphological dilation operation. For each position of out_backprop, the operator recomputes the filter window position where the forward Dilation2D finds the argmax, and then atomically accumulates the gradient to the corresponding filter position.

- Computation formula:

For each output position (b, h_out, w_out, d) (the following uses NHWC as an example, and the dimension indexes are adjusted accordingly for NCHW):

$$
h_{beg} = h_{out} \times stride_h - pad_{top}
$$

$$
w_{beg} = w_{out} \times stride_w - pad_{left}
$$

$$
cur\_val = \min(T), \quad h_{max} = 0, \quad w_{max} = 0
$$

$$
\text{for } h \in [0, filter_h), w \in [0, filter_w):
$$

$$
\quad h_{in} = h_{beg} + h \times rate_h, \quad w_{in} = w_{beg} + w \times rate_w
$$

$$
\quad \text{if } 0 \le h_{in} < H_{in} \text{ and } 0 \le w_{in} < W_{in}:
$$

$$
\quad \quad val = x[b, h_{in}, w_{in}, d] + filter[h, w, d]
$$

$$
\quad \quad \text{if } val > cur\_val: \quad cur\_val = val, \quad h_{max} = h, \quad w_{max} = w
$$

$$
y[h_{max}, w_{max}, d] \mathrel{+}= out\_backprop[b, h_{out}, w_{out}, d]
$$

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
      <td>x</td>
      <td>Input</td>
      <td>Input feature map. It is a 4D tensor. In NHWC format, the shape is (N, H_in, W_in, C). In NCHW format, the shape is (N, C, H_in, W_in).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>Input</td>
      <td>Dilation filter. It is a 3D tensor (filter_h, filter_w, C).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_backprop</td>
      <td>Input</td>
      <td>Gradient of the forward output. It is a 4D tensor. In NHWC format, the shape is (N, H_out, W_out, C). In NCHW format, the shape is (N, C, H_out, W_out).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Gradient of filter. It is a 3D tensor (filter_h, filter_w, C). Its shape is the same as that of filter.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attr</td>
      <td>Sliding window strides. It is a list of 4 elements [1, stride_h, stride_w, 1], where stride_h/stride_w >= 1.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rates</td>
      <td>Attr</td>
      <td>Dilation rates. It is a list of 4 elements [1, rate_h, rate_w, 1], where rate_h/rate_w >= 1.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>Attr</td>
      <td>Padding mode: "SAME" / "VALID" / "CALCULATED". The default value is "SAME".</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attr</td>
      <td>Explicit padding values [pad_top, pad_bottom, pad_left, pad_right]. This parameter takes effect only when the mode is CALCULATED. The default value is {0,0,0,0}.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Attr</td>
      <td>Whether to use the ceil function to compute the output size. The default value is false.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attr</td>
      <td>Data format: "NHWC" or "NCHW". The default value is "NHWC".</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The input x must be a 4D tensor. filter must be a 3D tensor (filter_h, filter_w, C). out_backprop must be a 4D tensor. In NHWC format, x is (N, H_in, W_in, C) and out_backprop is (N, H_out, W_out, C). In NCHW format, x is (N, C, H_in, W_in) and out_backprop is (N, C, H_out, W_out).
- The dtypes of the three inputs must be the same. The dtype of the output y is the same as that of the inputs.
- The C dimension of x must be the same as the C dimension of filter and the C dimension of out_backprop.
- strides[0]=1 and strides[3]=1 (only spatial dimensions stride, NHWC format). In NCHW format, strides[0]=1 and strides[1]=1 are required.
- rates[0]=1 and rates[3]=1 (only spatial dimensions dilate, NHWC format). In NCHW format, rates[0]=1 and rates[1]=1 are required.
- Both NHWC and NCHW data formats are supported. NHWC is used by default.
- Empty tensors (the shape contains a dimension of 0) are supported. The output is the corresponding all-zero tensor.
- Dynamic shapes (DynamicCompileStatic / DynamicRank / DynamicShape) are supported.
- The computation is deterministic (per-thread buffer + deterministic reduction, without atomicAdd).

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
    <td><a href="./examples/test_geir_dilation2_d_backprop_filter.cpp">test_geir_dilation2_d_backprop_filter</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
