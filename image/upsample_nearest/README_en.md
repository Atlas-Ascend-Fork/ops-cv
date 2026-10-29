# UpsampleNearest

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function:

  Applies the nearest neighbor interpolation algorithm to perform upsampling on the input signal composed of multiple input channels.
  - If the input shape is (N, C, L), the output shape is (N, C, outputSize);
  - If the input shape is (N, C, H, W), the output shape is (N, C, outputSize[0], outputSize[1]).

- Calculation formula:

  - When exact_mode=true:

    $$
    h_{src} = min(floor((h_{dst} + 0.5) * scalesH),  H - 1)
    $$

    $$
    w_{src} = min(floor((w_{dst} + 0.5) * scalesW),  W - 1)
    $$

    $$
    out(N, C, h_{dst}, w_{dst}) = self(N, C, h_{src}, w_{src})
    $$

  - When exact_mode=false:

    $$
    h_{src} = min(floor(h_{dst} * scalesH),  H - 1)
    $$

    $$
    w_{src} = min(floor(w_{dst} * scalesW),  W - 1)
    $$

    $$
    out(N, C, h_{dst}, w_{dst}) = self(N, C, h_{src}, w_{src})
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
      <td>Represents the input tensor for upsampling, corresponding to `self` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>Specifies the output spatial size, corresponding to `outputSize` in the formula. The size is 2. It indicates the spatial size of `y` in the H and W dimensions.</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the height dimension of the spatial size, corresponding to `scalesH` in the formula.</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the multiplier of the width dimension of the spatial size, corresponding to `scalesW` in the formula.</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>exact_mode</td>
      <td>Optional attr</td>
      <td><ul><li>Determines whether to use the exact mode, corresponding to `exact_mode` in the formula description.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor after sampling, corresponding to `out` in the formula. The data type is the same as the data type of the input parameter `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- For <term>Atlas inference products</term>, the BFLOAT16 and UINT8 data types of the input and output are not supported.
- For <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>, BFLOAT16 and UINT8 are not supported.
- Only <term>Atlas A2 products</term> and <term>Atlas A3 products</term> support the UINT8 data type.

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_nearest](examples/test_aclnn_upsample_nearest.cpp) | Call the UpsampleNearest operator through the [aclnnUpsampleNearestExact1d](docs/aclnnUpsampleNearestExact1d_en.md) interface. |
| aclnn interface  | [test_aclnn_upsample_nearest_exact2d](examples/test_aclnn_upsample_nearest_exact2d.cpp) | Call the UpsampleNearest operator through the [aclnnUpsampleNearestExact2d](docs/aclnnUpsampleNearestExact2d_en.md) interface. |
