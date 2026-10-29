# BackgroundReplace

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

- Operator function: fuses the input new background image with the existing image, and replaces the background with the new background by using a mask.

- Calculation formula:

  $$
  out = bkg * (1 - mask) + src * mask
  $$

  In the formula, bkg is the background image, src is the source image, and mask is the mask.

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
      <td>bkg</td>
      <td>Input</td>
      <td>Background image, an aclTensor on the device. The HWC (C=1, 3) shape is supported. Only contiguous tensors are supported.</td>
      <td>UINT8, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src</td>
      <td>Input</td>
      <td>Source image, an aclTensor on the device. The HWC (C=1, 3) shape is supported. Only contiguous tensors are supported.</td>
      <td>UINT8, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>Mask, an aclTensor on the device. The HWC (C=1) shape is supported. Only contiguous tensors are supported.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output image, an aclTensor on the device. The HWC (C=1, 3) shape is supported. The data type and shape are the same as those of the input background image bkg. Only contiguous tensors are supported.</td>
      <td>UINT8, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

* Deterministic computation: aclnnBackgroundReplace uses a deterministic implementation by default.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_background_replace](examples/test_aclnn_background_replace.cpp) | Call the BackgroundReplace operator through the [aclnnBackgroundReplace](docs/aclnnBackgroundReplace_en.md) interface. |
