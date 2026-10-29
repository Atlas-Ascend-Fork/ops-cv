# RoiPoolingGradWithArgMax

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

## Function Description

- Operator function: Implements the backward computation of RoiPoolingWithArgMax. The operator traverses the pooling results of each ROI and accumulates the backward gradient contributions on the feature map coordinates, thereby completing the backward computation on the entire image.
- Computation formula:

  $$
  \frac{\partial L}{\partial x_i} = \sum_{r}\sum_{j}[i = i^*(r,j)]\frac{\partial L}{\partial y_{rj}}
  $$

  In the formula,

  $$
  [i = i^*(r,j)]  = \begin{cases} 1, & i = i^*(r,j) \\ 0, & otherwise \end{cases}
  $$

  The decision function `[i = i^*(r,j)]` indicates whether node i is selected by the j-th output node of the candidate region r as the maximum value output.

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
      <td>grad</td>
      <td>Input</td>
      <td>Gradient of the input feature map.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input feature map.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>Input</td>
      <td>ROI bounding boxes.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>roi_actual_num</td>
      <td>Optional Input</td>
      <td>Specifies the number of ROIs in each batch.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>Input</td>
      <td>Specifies the index of the gradient.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pooled_h</td>
      <td>Attr</td>
      <td>Specifies the height of the pooling ratio.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_w</td>
      <td>Attr</td>
      <td>Specifies the width of the pooling ratio.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_h</td>
      <td>Attr</td>
      <td>Scaling ratio for mapping the height of the input coordinates to ROI coordinates.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_w</td>
      <td>Attr</td>
      <td>Scaling ratio for mapping the width of the input coordinates to ROI coordinates.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pool_channel</td>
      <td>Attr</td>
      <td>Specifies the number of channels for pooling.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output result.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

* The data types of gradOutput, rois, argmax, and gradInputRef are within the supported range.
* gradOutput, argmax, and gradInputRef have the same data type.
* gradOutput, argmax, and gradInputRef have a shape size of 4, and rois has a shape size of 2.
* gradOutput, argmax, and rois have the same shape[0].
* gradOutput and argmax have the same shape[1].
* The shape[2] of gradOutput and argmax equals pooledH, and their shape[3] equals pooledW.
* The values of rois are greater than or equal to 0.
* pooledH and pooledW are greater than 0.
* rois[:, 1] is less than rois[:, 2], and rois[:, 3] is less than rois[:, 4].
* rois.shape[0] and gradOutput.shape[0] are less than or equal to 1024.

## Invocation

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_roi_pooling_grad_with_arg_max](examples/arch35/test_aclnn_roi_pooling_grad_with_arg_max.cpp) | Invokes the RoiPoolingGradWithArgMax operator through the [aclnnRoiPoolingGradWithArgMax](docs/aclnnRoiPoolingGradWithArgMax_en.md) API. |
