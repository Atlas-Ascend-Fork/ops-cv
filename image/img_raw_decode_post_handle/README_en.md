# ImgRawDecodePostHandle

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

- Operator function: This operator converts 4-channel image data in YUV format to RAW Bayer format output. It also performs gamma correction and black level correction on each channel. Two Bayer arrangement modes, `binning` and `quad`, are supported.

- Computation formula:

$$
v_{f32} = \frac{v_{u16}}{64.0}
$$

$$
mask = clamp(v_{f32}, 55.0, 56.0) - 55.0
$$

$$
d = \frac{v_{f32} - 56.0}{967.0} \times mask
$$

$$
g =
\begin{cases}
\exp(\ln(\max(d, 10^{-38})) \times \gamma), & d > 0, \\
\exp(-\infty \times \gamma), & d \le 0\; (\text{when }\gamma > 0\text{, this term is }0).
\end{cases}
$$

$$
d = g \times 967.0 + 56.0
$$

$$
d = d \times mask
$$

$$
result = v_{f32} \times (1 - mask) + d
$$

$$
raw\_img = uint16(round(result))
$$

The 4 channels are interleaved into a 2D RAW image output according to the Bayer pattern.

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
    <td>img_channel_0</td>
    <td>Input</td>
    <td>Channel 0 YUV image data. The shape is (h_gm, w_gm). It corresponds to v_u16 in the formula.</td>
    <td>UINT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>img_channel_1</td>
    <td>Input</td>
    <td>Channel 1 YUV image data. The shape is (h_gm, w_gm). It corresponds to v_u16 in the formula.</td>
    <td>UINT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>img_channel_2</td>
    <td>Input</td>
    <td>Channel 2 YUV image data. The shape is (h_gm, w_gm). It corresponds to v_u16 in the formula.</td>
    <td>UINT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>img_channel_3</td>
    <td>Input</td>
    <td>Channel 3 YUV image data. The shape is (h_gm, w_gm). It corresponds to v_u16 in the formula.</td>
    <td>UINT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>img_size</td>
    <td>Input</td>
    <td>Output image size. The shape is (2,) and the value is [h_out, w_out]. The output shape is determined by the element values of this tensor.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma</td>
    <td>Input</td>
    <td>Gamma correction values of the 4 channels. The shape is (4,).</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bayer_pattern</td>
    <td>Attr</td>
    <td>Bayer arrangement mode. The value is "binning" or "quad", and the default value is "binning". It determines the channel interleaving mode and the output size constraints.</td>
    <td>String</td>
    <td>-</td>
  </tr>
  <tr>
    <td>raw_img</td>
    <td>Output</td>
    <td>RAW Bayer image. The shape is (h_out, w_out).</td>
    <td>UINT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- The ranks of img_channel_0, img_channel_1, img_channel_2, and img_channel_3 must be 2, and the shapes of the 4 channels must be the same.
- The shape of img_size must be (2,) and the value is [h_out, w_out]. The actual data must contain 2 int32 elements. Otherwise, inferShape reports an error.
- The values of img_size must be positive integers, that is, h_out and w_out must be greater than 0.
- The shape of gamma must be (4,).
- In binning mode, h_out and w_out must be multiples of 2.
- In quad mode, h_out and w_out must be multiples of 4.
- The output width w_out must be greater than or equal to 16.
- h_gm must be greater than or equal to h_out/2, and w_gm must be greater than or equal to w_out/2.
- Empty tensors are not supported.
- Non-contiguous tensors are automatically converted to contiguous tensors before processing.
- The output of this operator is of the uint16 integer type. The gamma correction path involves ln/exp floating-point computation, and an integer tolerance comparison with atol=2 is allowed.

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
    <td><a href="./examples/test_geir_img_raw_decode_post_handle.cpp">test_geir_img_raw_decode_post_handle</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
