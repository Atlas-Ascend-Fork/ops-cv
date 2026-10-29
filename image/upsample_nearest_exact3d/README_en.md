# UpsampleNearestExact3d

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: applies the nearest neighbor interpolation algorithm to perform upsampling on the input signal composed of multiple input channels.
- Calculation formula:

  $$
  scaleD =\begin{cases}
  scalesD & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[0]/D &  Otherwise\\
  \end{cases}
  $$

  $$
  scaleH =\begin{cases}
  scalesH & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[1]/H &  Otherwise\\
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  scalesW & scalesD>0\&scalesH>0\&scalesW>0\\
  outputSize[2]/W &  Otherwise\\
  \end{cases}
  $$

  $$
  d_{src} = min(floor((d_{dst} + 0.5) / scaleD),  D - 1)
  $$

  $$
  h_{src} = min(floor((h_{dst} + 0.5) / scaleH),  H - 1)
  $$

  $$
  w_{src} = min(floor((w_{dst} + 0.5) / scaleW),  W - 1)
  $$

  $$
  out(N, C, d_{dst},h_{dst}, w_{dst}) = self(N, C, d_{src},h_{src}, w_{src})
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
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>Specifies the tensor size of output `out`, corresponding to `output_size` in the formula. The size is 3, and all elements are greater than 0. It indicates the spatial sizes of output `out` in the D, H, and W dimensions.</td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Optional attr</td>
      <td><ul><li>Specifies the scaling array along each dimension. It contains three elements: scale_depth, scale_height, and scale_width. It corresponds to `scalesD`, `scalesH`, and `scalesW` in the formula.</li><li>The default value is [0.0, 0.0, 0.0].</li></ul></td>
      <td>LISTFLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor after sampling, corresponding to `out` in the formula description. The data type and data format must be the same as the data type and data format of the input parameter `self`. The shape depends on the input x and output_size/scales.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
  </tbody></table>

- For <term>Atlas inference products</term>, the BFLOAT16 data type of the input and output is not supported.
- For <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>, BFLOAT16 is not supported.

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_upsample_nearest_exact3d](examples/test_aclnn_upsample_nearest_exact3d.cpp) | Call the UpsampleNearestExact3d operator through the [aclnnUpsampleNearestExact3d](docs/aclnnUpsampleNearestExact3d_en.md) interface. |
| Graph mode | - | Call the UpsampleNearestExact3d operator by constructing a graph in the [operator IR](op_graph/upsample_nearest_exact3d_proto.h) mode. |
