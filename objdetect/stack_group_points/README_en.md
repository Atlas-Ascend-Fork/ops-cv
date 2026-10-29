# StackGroupPoints

English | [简体中文](./README.md)

## Product Support

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×   |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: Reorganizes the features of the point cloud according to the groups to which the feature points belong.
- Computation formula: The point cloud data `features` has the shape $(N1 + N2 ..., C)$, and the operator groups it according to `features_batch_cnt` with the shape $(N1, N2, ...)$. Similarly, `indices` has the shape $(M1 + M2 ..., nsample)$, and the operator groups it according to `indices_batch_cnt` with the shape $(M1, M2, ...)$. The operator extracts features from the corresponding `features` group according to the `indices` grouping to form the output `out`, implementing the following function:
  $$
  out[m:c:np] = features[indices[m:np]:c]
  $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>features</td>
      <td>Input</td>
      <td>Point cloud features to reorganize.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>features_batch_cnt</td>
      <td>Input</td>
      <td>Number of point cloud features contained in each batch.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Reorganization indices.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices_batch_cnt</td>
      <td>Input</td>
      <td>Number of reorganization indices contained in each batch.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output point cloud features. It corresponds to out in the formula.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- `features` and `indices` must be 2D tensors, and `features_batch_cnt` and `indices_batch_cnt` must be 1D tensors.
- The channel dimension C of `features`, the nsample dimension of `indices`, and the batch count length B must be positive.
- The lengths of `features_batch_cnt` and `indices_batch_cnt` must be equal, and they describe the number of features and the number of indices of the same batch respectively.
- The `indices` values of each batch must be within the range of the number of `features` of the corresponding batch. The output of out-of-bound indices is zero.

## Invocation

| Invocation Method   | Sample Code                                                                        | Description                                         |
| ---------------- |-----------------------------------------------------------------------------| --------------------------------------------------- |
| Graph mode | [test_geir_stack_group_points](./examples/test_geir_stack_group_points.cpp) | Invokes the StackGroupPoints operator by constructing a graph through the [operator IR](./op_graph/stack_group_points_proto.h).         |
