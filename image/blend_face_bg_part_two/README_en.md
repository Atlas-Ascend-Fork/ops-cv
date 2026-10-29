# BlendFaceBgPartTwo

English | [简体中文](./README.md)

## Product Support

|Product | Supported |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: This operator performs Alpha blending of the normalized accumulated face image and the background image (the second part of the BlendFaceBg process).
- Computation formula:

  Given the accumulated face image acc_face, the accumulated mask acc_mask, the maximum mask max_mask, and the background image bg_img, the operator computes the fused image:

  $$
  fusion\_face = \frac{acc\_face}{acc\_mask + \epsilon}
  $$

  $$
  bg\_weight = 1 - max\_mask
  $$

  $$
  fused\_img = fusion\_face \times max\_mask + bg\_img \times bg\_weight
  $$

  where:
  - acc_face: accumulated face image, which comes from the first part of the BlendFaceBg process
  - acc_mask: accumulated mask, which is used as the normalization denominator
  - max_mask: maximum mask, which serves as the Alpha blending weight. Value range: [0,1]
  - bg_img: background image, which supports the float32 or uint8 type (uint8 is converted to float32 inside the operator)
  - epsilon: a small value that prevents division by zero. The default value is 1e-12

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
      <td>acc_face</td>
      <td>Input</td>
      <td>Accumulated face image. It comes from the first part of the BlendFaceBg process. The shape is (H,W,C).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>acc_mask</td>
      <td>Input</td>
      <td>Accumulated mask. It is used as the normalization denominator. The shape is (H,W,C).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_mask</td>
      <td>Input</td>
      <td>Maximum mask. It serves as the Alpha blending weight. Value range: [0,1]. The shape is (H,W,C).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bg_img</td>
      <td>Input</td>
      <td>Background image. The shape is (H,W,C). If the type is uint8, the operator internally converts it to float32 for computation.</td>
      <td>UINT8, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Attr</td>
      <td>A small value that prevents division by zero. It is applied to acc_mask. The default value is 1e-12.</td>
      <td>FLOAT</td>
      <td>Scalar</td>
    </tr>
    <tr>
      <td>fused_img</td>
      <td>Output</td>
      <td>Fused image after Alpha blending. Its shape is the same as that of acc_face. The data type is fixed to FLOAT.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- All input tensors must have the same shape, and the rank must be 3 (H,W,C).
- The data types of acc_face, acc_mask, and max_mask must be FLOAT.
- The data type of bg_img can be FLOAT or UINT8.
- The data type of the output fused_img is fixed to FLOAT.

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode invocation | [examples/arch35/test_geir_blend_face_bg_part_two.cpp](examples/arch35/test_geir_blend_face_bg_part_two.cpp) (fixed shape)<br/>
