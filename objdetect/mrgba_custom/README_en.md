# MrgbaCustom

English | [简体中文](./README.md)

## Product Support

|Product             |  Support  |
|:-------------------------|:----------:|
|  Ascend 950PR&950DT products   |     ×    |
|  Atlas A3 products   |     ×    |
|  Atlas A2 products     |     ×    |
|  Atlas 200I/500 A2 inference products    |     ×    |
|  Atlas inference products    |     √  |
|  Atlas training products    |     ×    |

## Function Description

- Operator function: Performs the transparency multiplication between tensor rgb and tensor alpha.
- Computation formula:

  $$
  out = rgb * \frac{alpha}{255}
  $$

  In the formula, alpha is broadcast to the same shape as rgb.

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
      <td>rgb</td>
      <td>Input</td>
      <td>aclTensor on the device, with shape HWC (C=3) supported. It satisfies the broadcast relationship with alpha, and only continuous tensors are supported.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>aclTensor on the device, with shape HWC (C=1) supported. It satisfies the broadcast relationship with rgb, and only continuous tensors are supported.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>aclTensor on the device, with shape HWC (C=3) supported. Its shape is the same as that of rgb, and only continuous tensors are supported.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

* Input shape limit: The shape of rgb is HWC (C=3), the shape of alpha is HWC (C=1), and the shape of out is the same as that of rgb. In addition, out and alpha satisfy the broadcast relationship.
* Deterministic computation: aclnnMrgbaCustom uses a deterministic implementation by default.

## Invocation

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_mrgba_custom](examples/test_aclnn_mrgba_custom.cpp) | Invokes the MrgbaCustom operator through the [aclnnMrgbaCustom](docs/aclnnMrgbaCustom_en.md) API. |
