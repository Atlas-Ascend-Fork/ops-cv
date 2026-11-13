# 算子接口（aclnn）

## 使用说明

为方便调用算子，提供一套基于C的API（以aclnn为前缀的API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

- **头文件/库文件**

  调用算子API时，需引用依赖的头文件和库文件，一般头文件默认在`${INSTALL_DIR}/include/aclnnop`，库文件默认在`${INSTALL_DIR}/lib64`，具体文件如下：

  - 头文件：方式1（推荐）：引用算子仓总头文件aclnn\_ops\_\$\{ops\_project\}.h。方式2：引用单个算子API的头文件aclnn\_\*.h。
  - 库文件：引用算子仓对应的库文件libopapi_${ops_project}.so。注意，原所有算子仓总库文件libopapi.so后续会废弃，不推荐使用，也不支持与单个算子库文件同时使用。

  ${INSTALL_DIR}表示CANN安装后文件路径；\$\{ops\_project\}表示算子仓名（如math、nn、cv、transformer），请改为实际算子仓名。

- **版本演进说明**

  请注意，部分API存在多个V版本，使用时选择最高V版本即可（高版本API已兼容低版本API的所有能力）。

## 接口列表

> [!NOTE]
>
> - 算子特性介绍：调用API前，请先学习算子相关基础知识，包括**确定性算法**、**Batch一致性**、**常见量化模式**等，具体介绍参见[算子基本概念](context/basic_concept.md)。
> - 符号说明：表格中“-”符号表示该接口暂不支持当前列产品。

算子接口列表如下：

| 接口名      | 说明     | 确定性说明（A2/A3） | 确定性说明（Ascend 950） |
| -------------- | --------------------------- | --------------------------- | --------------------------- |
| [aclnnMrgbaCustom](../../objdetect/mrgba_custom/docs/aclnnMrgbaCustom.md) | 完成张量rgb和张量alpha的透明度乘法计算。 | - | - |
| [aclnnBackgroundReplace](../../objdetect/background_replace/docs/aclnnBackgroundReplace.md) | 将输入的新背景图片与已有图片进行融合，通过掩码的方式将背景替换为新的背景。 | - |- |
| [aclnnBlendImagesCustom](../../objdetect/blend_images_custom/docs/aclnnBlendImagesCustom.md) | 完成张量rgb、frame和alpha的透明度乘法计算。 | - | - |
| [aclnnGridSampler2D](../../image/grid_sample/docs/aclnnGridSampler2D.md) | 根据网格定义的坐标，从输入张量中采样像素值并重映射到输出空间。 |默认确定性实现|默认确定性实现|
| [aclnnGridSampler3D](../../image/grid_sample/docs/aclnnGridSampler3D.md) | 根据网格定义的坐标，从输入张量中采样像素值并重映射到输出空间。 |默认确定性实现|默认确定性实现|
| [aclnnGridSampler2DBackward](../../image/grid_sampler2_d_grad/docs/aclnnGridSampler2DBackward.md) | [aclnnGridSampler2D](../../image/grid_sample/docs/aclnnGridSampler2D.md)的反向传播，完成张量input与张量grid的梯度计算。 |默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
| [aclnnGridSampler3DBackward](../../image/grid_sampler3_d_grad/docs/aclnnGridSampler3DBackward.md) | [aclnnGridSampler3D](../../image/grid_sample/docs/aclnnGridSampler3D.md)的反向传播，完成张量input与张量grid的梯度计算。 |默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
| [aclnnRasterizer](../../image/rasterizer/docs/aclnnRasterizer.md) | 实现光栅化计算，根据给定的三维空间中的点和面，获取屏幕中每个像素点的最小深度及其对应的面片索引，并计算该面片的重心坐标透视矫正插值。      |默认确定性实现|默认确定性实现|
| [aclnnResize](../../image/resize_bilinear_v2/docs/aclnnResize.md) | 根据scales调整输入张量的大小。                               |默认确定性实现|默认确定性实现|
| [aclnnThreeInterpolateBackward](../../image/three_interpolate_backward/docs/aclnnThreeInterpolateBackward.md) | 根据grad_x、idx、weight进行三点插值计算梯度得到grad_y。      |默认非确定性实现，不支持配置开启|默认确定性实现|
| [aclnnUpsampleNearest1d](../../image/resize_nearest_neighbor_v2/docs/aclnnUpsampleNearest1d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest2d](../../image/resize_nearest_neighbor_v2/docs/aclnnUpsampleNearest2d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleTrilinear3d](../../image/resize_upsample_trilinear/docs/aclnnUpsampleTrilinear3d.md) | 对由多个输入通道组成的输入信号应用三线性插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleBicubic2d](../../image/upsample_bicubic2d/docs/aclnnUpsampleBicubic2d.md) | 对由多个输入通道组成的输入信号应用2D双三次上采样。           |默认确定性实现|默认确定性实现|
| [aclnnUpsampleBicubic2dAA](../../image/upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md) | 对由多个输入通道组成的输入信号应用双三次抗锯齿算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleBicubic2dAAGrad](../../image/upsample_bicubic2d_aa_grad/docs/aclnnUpsampleBicubic2dAAGrad.md) | [aclnnUpsampleBicubic2dAA](../../image/upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md)的反向传播。 |默认确定性实现|默认非确定性实现，支持配置开启|
| [aclnnUpsampleBicubic2dBackward](../../image/upsample_bicubic2d_grad/docs/aclnnUpsampleBicubic2dBackward.md) | [aclnnUpsampleBicubic2d](../../image/upsample_bicubic2d/docs/aclnnUpsampleBicubic2d.md)的反向传播。 |默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
| [aclnnUpsampleBilinear2d](../../image/upsample_bilinear2d/docs/aclnnUpsampleBilinear2d.md) | 对由多个输入通道组成的输入信号应用2D双线性上采样。|默认确定性实现|默认确定性实现|
| [aclnnUpsampleBilinear2dAA](../../image/upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md) | 对由多个输入通道组成的输入信号应用2D双线性抗锯齿采样。|默认确定性实现|默认确定性实现|
| [aclnnUpsampleBilinear2dAABackward](../../image/upsample_bilinear2d_aa_backward/docs/aclnnUpsampleBilinear2dAABackward.md) | [aclnnUpsampleBilinear2dAA](../../image/upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md)的反向传播。 |默认确定性实现|默认非确定性实现，支持配置开启|
| [aclnnUpsampleBilinear2dBackward](../../image/resize_bilinear_v2_grad/docs/aclnnUpsampleBilinear2dBackward.md) | [aclnnUpsampleBilinear2d](../../image/upsample_bilinear2d/docs/aclnnUpsampleBilinear2d.md)的反向传播。 |默认确定性实现|默认非确定性实现，支持配置开启|
| [aclnnUpsampleBilinear2dBackwardV2](../../image/upsample_bilinear2d_grad/docs/aclnnUpsampleBilinear2dBackwardV2.md) | [aclnnUpsampleBilinear2d](../../image/upsample_bilinear2d/docs/aclnnUpsampleBilinear2d.md)的反向传播。 |默认确定性实现|默认非确定性实现，支持配置开启|
| [aclnnUpsampleLinear1d](../../image/upsample_linear1d/docs/aclnnUpsampleLinear1d.md) | 对由多个输入通道组成的输入信号应用线性插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleLinear1dBackward](../../image/upsample_bilinear2d_grad/docs/aclnnUpsampleLinear1dBackward.md) | [aclnnUpsampleLinear1d](../../image/upsample_linear1d/docs/aclnnUpsampleLinear1d.md)的反向传播。 |默认确定性实现|默认非确定性实现，支持配置开启|
| [aclnnUpsampleNearestExact1d](../../image/upsample_nearest/docs/aclnnUpsampleNearestExact1d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearestExact2d](../../image/upsample_nearest/docs/aclnnUpsampleNearestExact2d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest1dBackward](../../image/upsample_nearest2d_grad/docs/aclnnUpsampleNearest1dBackward.md) | [aclnnUpsampleNearest1dV2](../../image/upsample_nearest3d/docs/aclnnUpsampleNearest1dV2.md)的反向传播。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest2dBackward](../../image/upsample_nearest2d_grad/docs/aclnnUpsampleNearest2dBackward.md) | [aclnnUpsampleNearest2dV2](../../image/upsample_nearest3d/docs/aclnnUpsampleNearest2dV2.md)的反向传播。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest1dV2](../../image/upsample_nearest3d/docs/aclnnUpsampleNearest1dV2.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest2dV2](../../image/upsample_nearest3d/docs/aclnnUpsampleNearest2dV2.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest3d](../../image/upsample_nearest3d/docs/aclnnUpsampleNearest3d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest3dBackward](../../image/upsample_nearest3d_grad/docs/aclnnUpsampleNearest3dBackward.md) | [aclnnUpsampleNearest3d](../../image/upsample_nearest3d/docs/aclnnUpsampleNearest3d.md)的反向传播。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearestExact1dBackward](../../image/upsample_nearest_exact2d_grad/docs/aclnnUpsampleNearestExact1dBackward.md) | [aclnnUpsampleNearestExact1d](../../image/upsample_nearest/docs/aclnnUpsampleNearestExact1d.md)的反向传播。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearestExact2dBackward](../../image/upsample_nearest_exact2d_grad/docs/aclnnUpsampleNearestExact2dBackward.md) | [aclnnUpsampleNearestExact2d](../../image/upsample_nearest/docs/aclnnUpsampleNearestExact2d.md)的反向传播。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearestExact3d](../../image/upsample_nearest_exact3d/docs/aclnnUpsampleNearestExact3d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearestExact3dBackward](../../image/upsample_nearest_exact3d_grad/docs/aclnnUpsampleNearestExact3dBackward.md) | [aclnnUpsampleNearestExact3d](../../image/upsample_nearest_exact3d/docs/aclnnUpsampleNearestExact3d.md)的反向传播。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleTrilinear3dBackward](../../image/upsample_trilinear3d_backward/docs/aclnnUpsampleTrilinear3dBackward.md) | [aclnnUpsampleTrilinear3d](../../image/resize_upsample_trilinear/docs/aclnnUpsampleTrilinear3d.md)的反向传播。 |默认确定性实现| - |
| [aclnnCIoU](../../objdetect/ciou/docs/aclnnCIoU.md)          | 用于边界框回归的损失函数，在IoU的基础上同时考虑了中心点距离、宽高比和重叠面积，以更全面地衡量预测框与真实框之间的差异。 | - |默认确定性实现|
| [aclnnIou](../../objdetect/iou_v2/docs/aclnnIou.md)          | 计算两组矩形框（预测框bBox与真值框gtBox）的交并比（IoU）或前景交叉比（IOF），用于评估其重叠程度。 |默认确定性实现|默认确定性实现|
| [aclnnNonMaxSuppression](../../objdetect/non_max_suppression_v6/docs/aclnnNonMaxSuppression.md)          | 删除分数小于scoreThreshold的边界框，筛选出与之前被选中部分重叠较高（IoU较高）的框。 | - |- |
| [aclnnRoiAlign](../../objdetect/roi_align/docs/aclnnRoiAlign.md) | RoIAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。 |默认确定性实现|默认确定性实现|
| [aclnnRoiAlignV2](../../objdetect/roi_align/docs/aclnnRoiAlignV2.md) | RoIAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。 |默认确定性实现|- |
| [aclnnRoiAlignV2Backward](../../objdetect/roi_align_grad/docs/aclnnRoiAlignV2Backward.md) |[aclnnRoiAlignV2](../../objdetect/roi_align/docs/aclnnRoiAlignV2.md)的反向传播。 |默认非确定性实现，支持配置开启|默认确定性实现|
| [aclnnRoiPoolingWithArgMax](../../objdetect/roi_pooling_with_arg_max/docs/aclnnRoiPoolingWithArgMax.md) | 对输入特征图按ROI（感兴趣区域）进行池化，在每个ROI内按空间划分为pooled_h × pooled_w个格子，对每个格子做最大池化，并输出池化结果及最大值在通道内的一维索引（argmax）。| - |默认确定性实现|
| [aclnnRoiPoolingGradWithArgMax](../../objdetect/roi_pooling_grad_with_arg_max/docs/aclnnRoiPoolingGradWithArgMax.md) | [aclnnRoiPoolingWithArgMax](../../objdetect/roi_pooling_with_arg_max/docs/aclnnRoiPoolingWithArgMax.md)的反向传播。 | - |默认非确定性实现，不支持配置开启|
| [aclnnIm2colBackward](../../image/col2im/docs/aclnnIm2colBackward.md) | 将梯度张量还原为原始输入张量的梯度，完成反向传播计算。 |默认确定性实现|默认确定性实现|
