# ROIAlign

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |    ×     |
| <term>Atlas A2系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：ROI Align(Region of Interest Align)从特征图中提取感兴趣区域(ROI)的特征，通过双线性插值和池化生成固定大小的输出特征图。

- 计算公式：

$$
\text{output}[n, c, ph, pw] = \text{pool}\left(\text{bilinear\_interpolate}(\text{features}[\text{batch\_idx}, c], y, x)\right)
$$

其中坐标映射由`roi_end_mode`属性控制，池化方式由`pool_mode`属性控制（avg/max）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>features</td>
      <td>输入</td>
      <td>输入特征图，shape为(N, C, H, W)。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois</td>
      <td>输入</td>
      <td>ROI坐标，shape为(K, 5)，每行[batch_idx, x1, y1, x2, y2]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rois_n</td>
      <td>输入（可选）</td>
      <td>每个ROI对应的batch索引，shape为(K,)。第i个元素指定第i个ROI使用的batch索引；提供时覆盖rois第i行中的batch_idx，未提供时使用rois中的batch_idx。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>池化后的ROI特征，shape为(K, C, pooled_height, pooled_width)。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>spatial_scale</td>
      <td>属性（必选）</td>
      <td>坐标缩放因子，将ROI坐标映射到特征图尺度。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_height</td>
      <td>属性（必选）</td>
      <td>输出高度（bin数量）。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_width</td>
      <td>属性（必选）</td>
      <td>输出宽度（bin数量）。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sample_num</td>
      <td>属性（可选）</td>
      <td>正值表示每个bin在H、W方向分别使用该数量的采样点；0表示按ROI在H、W方向各自的bin尺寸向上取整确定采样点数。取值范围[0, 46340]（即floor(sqrt(INT32_MAX))）。默认值2。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roi_end_mode</td>
      <td>属性（可选）</td>
      <td>坐标变换模式：0=无偏移；1=加1后缩放；2=减0.5偏移；3=减0.5偏移（与2等价）。仅支持0、1、2、3。默认值1。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pool_mode</td>
      <td>属性（可选）</td>
      <td>池化模式："avg"（平均池化）或"max"（最大池化）。默认值"avg"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- features 必须为 4D 张量 (N, C, H, W)，数据格式为 ND。
- rois 必须为 2D 张量 (K, 5)，每行格式为 [batch_idx, x1, y1, x2, y2]。
- batch_idx 取值范围为 [0, N)，N 为输入 features 的第 0 维大小；越界时算子不报错，该 ROI 按空 ROI 处理输出 0（不进行越界读取）。0 <= x1 <= x2 <= W / spatial_scale, 0 <= y1 <= y2 <= H / spatial_scale（x1 == x2 或 y1 == y2 合法；roi_end_mode 为 0/1 时，ROI 在特征图对应方向的边长按最小 1.0 处理）。
- features 和 rois 必须使用相同 dtype（float16 或 float32）。
- rois_n（可选）必须为 1D INT32 张量，长度等于 K（rois 行数）；第 i 个元素为第 i 个 ROI 的 batch 索引并覆盖 rois 第 i 行的 batch_idx。
- pooled_height 和 pooled_width 必须满足 1 <= 值 <= INT32_MAX（2147483647）。
- sample_num 必须满足 0 <= 值 <= 46340（floor(sqrt(INT32_MAX))）。
- roi_end_mode 仅支持 0、1、2、3。
- pool_mode 仅支持 "avg" 和 "max"。
- spatial_scale 必须为有限值且 > 0。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| aclnn API | [test_aclnn_roi_align](examples/test_aclnn_roi_align.cpp) | 通过[aclnnRoiAlign](docs/aclnnRoiAlign.md)接口方式调用ROIAlign算子。 |
| aclnn API | [test_aclnn_roi_align_v2](examples/test_aclnn_roi_align_v2.cpp) | 通过[aclnnRoiAlignV2](docs/aclnnRoiAlignV2.md)接口方式调用ROIAlign算子。 |
| GE图模式 | [test_geir_roi_align](examples/arch35/test_geir_roi_align.cpp) | 通过[算子IR](op_graph/roi_align_proto.h)构图方式调用ROIAlign算子，参见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)完成算子编译和验证。 |
