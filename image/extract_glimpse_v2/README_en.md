# ExtractGlimpseV2

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

- Operator function: This operator extracts sub-images (glimpses) of specified positions and sizes from a batch of input images. Based on the given offset coordinates and crop size, the operator crops one sub-region from each input image. The output is a batch of cropped images.

- Computation formula:

$$
\text{For each batch item } i:
$$
$$
\text{1. Coordinate transformation: } offset\_y, offset\_x \text{ (according to the normalized/centered attributes)}
$$
$$
\text{2. Crop region: } [start\_y, end\_y) \times [start\_x, end\_x) \text{ (clamped to the image boundaries)}
$$
$$
\text{3. Data copy: } glimpse[i][base\_y:base\_y+copy\_h][base\_x:base\_x+copy\_w][:] = input[i][start\_y:end\_y][start\_x:end\_x][:]
$$
$$
\text{4. Out-of-bounds padding: } glimpse \text{ fills the uncopied region with zeros (noise="zero")}
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
      <td>input</td>
      <td>Input</td>
      <td>Batch of input images. It is a 4D tensor with the shape [batch, height, width, channels].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>Crop size [size_h, size_w]. It is a 1D const tensor with the shape [2], and its values cannot be less than 0.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>Input</td>
      <td>Offset coordinates (y, x) of each batch item. It is a 2D tensor with the shape [batch, 2].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>glimpse</td>
      <td>Output</td>
      <td>Batch of cropped images. It is a 4D tensor with the shape [batch, size_h, size_w, channels]. Here size_h and size_w come from the size input.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>centered</td>
      <td>Attr</td>
      <td>Whether the offset coordinates are centered relative to the image. Default value: true.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>normalized</td>
      <td>Attr</td>
      <td>Whether the offset coordinates are normalized to [-1, 1]. Default value: true.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>uniform_noise</td>
      <td>Attr</td>
      <td>Whether to use uniform noise (the current implementation supports only false). Default value: true.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>noise</td>
      <td>Attr</td>
      <td>Noise type (the current implementation supports only "zero"). Default value: "uniform".</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- input must be a 4D tensor.
- offsets must be a 2D tensor, and the second dimension must be 2.
- size must be a 1D const tensor. Its length must be 2, and its values cannot be less than 0.
- The batch dimensions of input and offsets must be the same.
- Currently, only the float32 data type is supported.
- Currently, only the out-of-bounds padding mode with noise="zero" is supported.

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | - | Calls the ExtractGlimpseV2 operator by constructing a graph in the [operator IR](op_graph/extract_glimpse_v2_proto.h) mode. |
