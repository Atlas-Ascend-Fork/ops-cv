# Col2ImV2

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

- Operator function: This operator rearranges and combines sliding window local blocks (columns) into a batched image tensor. This is the col2im/fold computation. The output uses the output-centric sliding window mode. When multiple sliding windows cover the same output position, the operator accumulates the values. Output positions that no sliding window covers are 0.

- Computation formula:

$$
ho = \left\lfloor \frac{outH + 2 \times padH - dilH \times (kH - 1) - 1}{strideH} \right\rfloor + 1
$$

$$
y[n,c,h,w] = \sum_{h\_col,w\_col,h\_k,w\_k} x[n,\ (c \times kH + h\_k) \times kW + w\_k,\ h\_col \times wo + w\_col]
$$

The summation subscripts must satisfy $h = h\_col \times strideH - padH + h\_k \times dilH$, and $w$ follows the same rule. The formula for computing $wo$ is symmetric to that of $ho$.

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
      <td>Sliding window local blocks for the col2im computation. It corresponds to x in the formula. The shape is (N, C×kH×kW, L).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Input</td>
      <td>Output image size (outH, outW). It is a const tensor of length 2.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kernel_size</td>
      <td>Input</td>
      <td>Sliding window size (kH, kW). It is a const tensor of length 2.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Attr</td>
      <td>Dilation intervals of the sliding window (dilH, dilW). It is a ListInt of length 2. Each element must be greater than 0.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Attr</td>
      <td>Edge padding size (padH, padW). It is a ListInt of length 2. Each element must be greater than or equal to 0.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Attr</td>
      <td>Sliding window stride (strideH, strideW). It is a ListInt of length 2. Each element must be greater than 0.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>col2im computation result. It corresponds to y in the formula. The shape is (N, C, outH, outW), and the dtype is the same as that of x.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- x supports only three-dimensional tensors with the shape (N, C×kH×kW, L). y supports only four-dimensional tensors with the shape (N, C, outH, outW).
- x and y have the same dtype. Only float32 and float16 are supported.
- output_size and kernel_size support only one-dimensional const tensors. The length must be 2, and the dtype supports only int32. The values must be known at compilation time (value-dependent input).
- Each element of output_size must be greater than 0. Each element of kernel_size must be greater than 0.
- dilation has a length of 2 and its elements are greater than 0. padding has a length of 2 and its elements are greater than or equal to 0. stride has a length of 2 and its elements are greater than 0.
- Cross-parameter constraint: The first dimension of x must be divisible by kH×kW. The second dimension L of x must equal ho×wo, where ho=(outH+2×padH-dilH×(kH-1)-1)//strideH+1 and wo=(outW+2×padW-dilW×(kW-1)-1)//strideW+1.
- Output positions that no sliding window covers have the value 0. When multiple sliding windows cover the same output position, the values are accumulated.

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
    <td><a href="./examples/test_geir_col2_im_v2.cpp">test_geir_col2_im_v2</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
