# ThreeInterpolateBackward

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: computes the gradient by performing three-point interpolation based on grad_x, idx, and weight to obtain grad_y.
- Calculation formula:

  $$
  grad\_y[b,c,idx[b,n,i]] =
  grad\_y[b,c,idx[b,n,i]] + grad\_x[b,c,n]*weight[b,n,i]\\ i\in[0,2]\ b\in[0,B) \ c\in[0,C) \ n\in[0,N)
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
      <td>grad_x</td>
      <td>Input</td>
      <td>Gradient value of the previous step of network backpropagation, corresponding to `grad_x` in the formula.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>5HD (NC1HWC0), ND</td>
    </tr>
    <tr>
      <td>idx</td>
      <td>Input</td>
      <td>Feature indexes of the three adjacent points of the target feature, corresponding to `idx` in the formula.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Feature weights of the three adjacent points of the target feature, corresponding to `weight` in the formula.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>m</td>
      <td>Attr</td>
      <td>Size of the lowest dimension of the input features, used to guide the inference of the shape of `grad_y` during infershape.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_y</td>
      <td>Output</td>
      <td>Gradient calculation result, corresponding to `grad_y` in the formula. The data type and data format must be the same as the data type and data format of `grad_x`.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>5HD (NC1HWC0), ND</td>
    </tr>
  </tbody></table>

## Constraints

- The values in `idx` must be less than `m`.
- The shape of `grad_x` is (B, C, N), the shape of `idx`/`weight` is (B, N, 3), and the shape of `grad_y` is (B, C, M).
- The attribute `m` must be greater than 0.
- `grad_x`/`weight`/`grad_y` are FLOAT32 or FLOAT16, and `idx` is INT32 or INT64.
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data format of `grad_x` and `grad_y` is 5HD (that is, NC1HWC0, consistent with `FORMAT_NC1HWC0` defined in the operator IR).
- <term>Ascend 950PR&950DT products</term>: the data format of `grad_x` and `grad_y` is ND.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_three_interpolate_backward](examples/test_aclnn_three_interpolate_backward.cpp) | Call the ThreeInterpolateBackward operator through the [aclnnThreeInterpolateBackward](docs/aclnnThreeInterpolateBackward_en.md) interface. |
<!--
| Graph mode | [test_geir_three_interpolate_backward](examples/test_geir_three_interpolate_backward.cpp)  | Call the ThreeInterpolateBackward operator by building a graph through [operator IR](op_graph/three_interpolate_backward_proto.h).         |
-->
