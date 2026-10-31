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
\quad \text{if } 0 \le h_{in} < H_{pad} \text{ and } 0 \le w_{in} < W_{pad}:
$$

$$
\quad \quad x_{val} = \begin{cases} x[b, h_{in}, w_{in}, d], & h_{in} < H_{in} \text{ and } w_{in} < W_{in} \\ 0, & \text{otherwise (padding region)} \end{cases}
$$

$$
\quad \quad val = x_{val} + filter[h, w, d]
$$

$$
\quad \quad \text{if } val > cur\_val: \quad cur\_val = val, \quad h_{max} = h, \quad w_{max} = w
$$

$$
y[h_{max}, w_{max}, d] \mathrel{+}= out\_backprop[b, h_{out}, w_{out}, d]
$$

where $H_{pad}$ and $W_{pad}$ are the participation boundaries of window positions: in CALCULATED mode, $H_{pad} = H_{in} + pad_{top} + pad_{bottom}$ and $W_{pad} = W_{in} + pad_{left} + pad_{right}$, and positions in the bottom/right padding region outside the original input participate in the argmax with $x = 0$; in SAME/VALID mode, $H_{pad} = H_{in}$ and $W_{pad} = W_{in}$, and out-of-bounds positions do not participate.

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
      <td>Dilation filter. It is a 3D tensor. In NHWC format, the shape is (filter_h, filter_w, C). In NCHW format, the shape is (C, filter_h, filter_w).</td>
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
      <td>Gradient of filter. It is a 3D tensor. In NHWC format, the shape is (filter_h, filter_w, C). In NCHW format, the shape is (C, filter_h, filter_w). Its shape is the same as that of filter.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attr</td>
      <td>Sliding window strides. It is a list of 4 elements. In NHWC format, it is [1, stride_h, stride_w, 1]. In NCHW format, it is [1, 1, stride_h, stride_w], where stride_h/stride_w >= 1.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rates</td>
      <td>Attr</td>
      <td>Dilation rates. It is a list of 4 elements. In NHWC format, it is [1, rate_h, rate_w, 1]. In NCHW format, it is [1, 1, rate_h, rate_w], where rate_h/rate_w >= 1.</td>
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
- The N dimension of out_backprop must be the same as the N dimension of x (rejected when both dimensions are known and different).
- When the spatial dimensions of x and filter are statically known, the H/W of out_backprop must be equal to the forward output size derived from x, filter, strides, rates, and padding_mode (dynamic dimensions of -1 are not affected).
- strides[0]=1 and strides[3]=1 (only spatial dimensions stride, NHWC format). In NCHW format, strides[0]=1 and strides[1]=1 are required.
- rates[0]=1 and rates[3]=1 (only spatial dimensions dilate, NHWC format). In NCHW format, rates[0]=1 and rates[1]=1 are required.
- Both NHWC and NCHW data formats are supported. NHWC is used by default.
- Empty tensors (the shape contains a dimension of 0) are supported. The output is the corresponding all-zero tensor.
- Dynamic shapes (DynamicCompileStatic / DynamicRank / DynamicShape) are supported.
- By default, the computation is non-deterministic (gradients at multiple positions are atomically accumulated, so the floating-point addition order is not fixed). It is deterministic (per-thread buffer + deterministic reduction, without atomicAdd) only when the determinism switch is enabled.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Sample Code</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE graph mode</td>
    <td><a href="./examples/test_geir_dilation2_d_backprop_filter.cpp">test_geir_dilation2_d_backprop_filter</a></td>
    <td>The operator is invoked by constructing a graph with the operator IR. For the IR definition, see <a href="./op_graph/dilation2_d_backprop_filter_proto.h">dilation2_d_backprop_filter_proto.h</a>. Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
