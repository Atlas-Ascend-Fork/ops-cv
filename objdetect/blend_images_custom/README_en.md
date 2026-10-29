# BlendImagesCustom

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  Ascend 950PR&950DT products   |     ×    |
|  Atlas A3 products   |     ×    |
|  Atlas A2 products     |     ×    |
|  Atlas 200I/500 A2 inference products    |     ×    |
|  Atlas inference products    |     √   |
|  Atlas training products    |     ×    |

## Function Description

- Operator function: performs the alpha multiplication computation on the tensors rgb, frame, and alpha.

- Calculation formula:

  $$
  out = rgb * \frac{alpha}{255} + frame * (1 - \frac{alpha}{255})
  $$

  In the formula, alpha is broadcast to the same shape as rgb and frame.

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
      <td>An aclTensor on the device. The HWC (C=3) shape is supported, and it satisfies the broadcast relationship with alpha. Only contiguous tensors are supported.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>An aclTensor on the device. The HWC (C=1) shape is supported, and it satisfies the broadcast relationship with rgb. Only contiguous tensors are supported.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>frame</td>
      <td>Input</td>
      <td>An aclTensor on the device. The HWC (C=3) shape is supported, and it satisfies the broadcast relationship with alpha. Only contiguous tensors are supported.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>An aclTensor on the device. The HWC (C=3) shape is supported, and the shape is the same as that of frame. Only contiguous tensors are supported.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_blend_images_custom](examples/test_aclnn_blend_images_custom.cpp) | Call the BlendImagesCustom operator through the [aclnnBlendImagesCustom](docs/aclnnBlendImagesCustom_en.md) interface. |
