# RoiPoolingGradWithArgMax

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×   |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：实现RoiPoolingWithArgMax的反向。遍历每个ROI的池化结果，将feature map坐标上的反向梯度贡献累加，即完成整张图上的反向计算。
- 计算公式：

  $$
  \frac{\partial L}{\partial x_i} = \sum_{r}\sum_{j}[i = i^*(r,j)]\frac{\partial L}{\partial y_{rj}}
  $$

  其中，

  $$
  [i = i^*(r,j)]  = \begin{cases} 1, & i = i^*(r,j) \\ 0, & otherwise \end{cases}
  $$

  判决函数`[i = i^*(r,j)]`表示i节点是否被候选区域r的第j个输出节点选为最大值输出

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
      <td>grad</td>
      <td>输入</td>
      <td>输入特征图的梯度。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入特征图。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>ROI边界框。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>roi_actual_num</td>
      <td>可选输入</td>
      <td>指定每个batch的ROI数量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>输入</td>
      <td>指定梯度的索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pooled_h</td>
      <td>属性</td>
      <td>指定池化率的高。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_w</td>
      <td>属性</td>
      <td>指定池化率的宽。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_h</td>
      <td>属性</td>
      <td>输入坐标的高映射到ROI坐标的缩放比例。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_w</td>
      <td>属性</td>
      <td>输入坐标的宽映射到ROI坐标的缩放比例。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pool_channel</td>
      <td>属性</td>
      <td>指定池化的通道数。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出结果。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* gradOutput、rois、argmax、gradInputRef的数据类型在支持的范围之内。
* gradOutput、argmax与gradInputRef具有相同的数据类型
* gradOutput、argmax、gradInputRef的shape大小为4，rois的shape大小为2
* gradOutput、argmax、rois的shape[0]相等
* gradOutput、argmax的shape[1]相等
* gradOutput、argmax的shape[2]等于pooledH和shape[3]等于pooledW
* rois的值大于等于0
* pooledH、pooledW大于0。
* rois[:, 1] 小于rois[:, 2] 且rois[:, 3] 小于rois[:, 4]
* rois.shape[0]、gradOutput.shape[0]小于等于1024

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_roi_pooling_grad_with_arg_max](examples/arch35/test_aclnn_roi_pooling_grad_with_arg_max.cpp) | 通过[aclnnRoiPoolingGradWithArgMax](docs/aclnnRoiPoolingGradWithArgMax.md)接口方式调用RoiPoolingGradWithArgMax算子。 |
