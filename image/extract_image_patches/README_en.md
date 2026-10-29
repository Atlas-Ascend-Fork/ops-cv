# ExtractImagePatches

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

- Operator function: This operator extracts image patches from 4D input images by sliding windows, and flattens and concatenates each patch to the channel dimension. It is compatible with the TensorFlow `ExtractImagePatches` operator. It is commonly used in scenarios such as convolution replacement, local feature aggregation, and so on.

- Computation formula:

For an NHWC format input `x` with the shape `[N, H, W, C]`, the output `y` has the shape `[N, out_h, out_w, C * kH * kW]`:

$$
y[n, i, j, (p \cdot kW + q) \cdot C + c] = x[n, i \cdot stride_h + p \cdot rate_h - pad_h, j \cdot stride_w + q \cdot rate_w - pad_w, c]
$$

where:

- `kH = ksizes[H]` and `kW = ksizes[W]`: patch height/width.
- The flattening order is patch row, patch column, and channel (`p`, `q`, `c`), where `0 <= c < C`.
- `stride_h = strides[H]` and `stride_w = strides[W]`: sliding strides.
- `rate_h = rates[H]` and `rate_w = rates[W]`: dilation rates.
- `pad_h/pad_w`: padding amounts determined by the padding mode (SAME/VALID).
- When a sampling point is out of bounds (VALID mode), the corresponding output position does not exist. In SAME mode, zeros are padded.

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
      <td>4D image input. Both NHWC and NCHW origin formats are supported. It corresponds to x in the formula.</td>
      <td>FLOAT16, FLOAT, BF16, INT8, UINT8</td>
      <td>NHWC/NCHW</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output with patches flattened to the channel dimension. Its format is the same as the input origin format. It corresponds to y in the formula.</td>
      <td>FLOAT16, FLOAT, BF16, INT8, UINT8</td>
      <td>NHWC/NCHW</td>
    </tr>
    <tr>
      <td>ksizes</td>
      <td>Attr</td>
      <td>Patch size. It is a ListInt of length 4. The spatial H/W dimensions must be greater than 0, and the N/C dimensions must be 1.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attr</td>
      <td>Sliding strides. It is a ListInt of length 4. The N/C dimensions must be 1.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rates</td>
      <td>Attr</td>
      <td>Dilation rates. It is a ListInt of length 4. The N/C dimensions must be 1.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Attr</td>
      <td>Padding mode. The value is "SAME" or "VALID".</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The input must be a 4D tensor (NHWC or NCHW format).
- The lengths of ksizes/strides/rates must be 4, and the N/C dimensions must be 1.
- The H/W dimensions of ksizes and strides must be greater than 0, and the H/W dimensions of rates must be greater than or equal to 1.
- The value of padding supports only "SAME" or "VALID".
- The input and output dtypes are the same. No type promotion is performed.
- This is a pure data movement operator. All special values (NaN/Inf/+0/-0) are passed through as is. The out-of-bounds padded zero value in SAME padding is +0.0.

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
    <td><a href="./examples/arch35/test_geir_extract_image_patches.cpp">test_geir_extract_image_patches</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
