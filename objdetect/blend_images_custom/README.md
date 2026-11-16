# BlendImagesCustom

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  Ascend 950PR&950DT系列产品   |     ×    |
|  Atlas A3系列产品   |     ×    |
|  Atlas A2系列产品     |     ×    |
|  Atlas 200I/500 A2推理产品    |     ×    |
|  Atlas推理系列产品    |     √   |
|  Atlas训练系列产品    |     ×    |

## 功能说明

- 算子功能：完成张量rgb、frame和alpha的透明度乘法计算。

- 计算公式：

  $$
  out = rgb * \frac{alpha}{255} + frame * (1 - \frac{alpha}{255})
  $$

  其中，alpha会广播到与rgb和frame相同的shape。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>rgb</td>
      <td>输入</td>
      <td>Device侧的aclTensor，shape支持HWC(C=3)，与alpha满足broadcast关系。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>Device侧的aclTensor，shape支持HWC(C=1)，与rgb满足broadcast关系。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>frame</td>
      <td>输入</td>
      <td>Device侧的aclTensor，shape支持HWC(C=3)，与alpha满足broadcast关系。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>Device侧的aclTensor，shape支持HWC(C=3)，与frame的shape一致。只支持连续Tensor。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_blend_images_custom](examples/test_aclnn_blend_images_custom.cpp) | 通过[aclnnBlendImagesCustom](docs/aclnnBlendImagesCustom.md)接口方式调用BlendImagesCustom算子。 |
