# SpatialTransformer

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

Operator function: the Spatial Transformer Network (STN) operator performs an affine transformation on the input tensor. The operator spatially transforms the input image x through the transformation matrix theta and outputs the transformed image y.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>Input tensor.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64<br>
          FLOAT16, FLOAT, DOUBLE</td>
      <td>NCHW, NC1HWC0</td>
    </tr>
    <tr>
      <td>theta</td>
      <td>Input</td>
      <td>Transformation matrix, containing the affine transformation parameters.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64<br>
          FLOAT16, FLOAT, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Transformed output tensor.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64<br>
          FLOAT16, FLOAT, DOUBLE</td>
      <td>NCHW, NC1HWC0</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attr</td>
      <td>Specifies the height and width of the output, containing 2 integers. The default value is [-1, -1], indicating that the input size is used.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>default_theta</td>
      <td>Attr</td>
      <td>Default affine transformation parameters, used when use_default_theta is true. The default value is an empty list.</td>
      <td>ListFloat</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>Attr</td>
      <td>If true, the centers of the 4 corner pixels of the input and output tensors are aligned to preserve the values of the corner pixels. The default value is false.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_default_theta</td>
      <td>Attr</td>
      <td>Specifies which theta parameters are used from default_theta. 1 indicates that the default value is used, and 0 indicates that the input theta is used. The default value is an empty list.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The format of the input tensor x must be NCHW or NC1HWC0.
- The format of the output tensor y must be the same as the format of the input tensor x.
- The shape of the transformation matrix theta must be [batch, 2, 3] or [2, 3].
- When use_default_theta is an empty list, the input theta is used for the transformation.
- When use_default_theta is not empty, the parameter at a position with value 1 uses the value in default_theta, and the parameter with value 0 uses the value in the input theta.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode invocation | [test_geir_spatial_transformer](./examples/test_geir_spatial_transformer.cpp)   | Call the SpatialTransformer operator by building a graph through [operator IR](./op_graph/spatial_transformer_proto.h). |
