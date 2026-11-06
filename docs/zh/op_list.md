# 算子列表

> 说明：
>
>- **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](./install/dir_structure.md)。
>- **算子执行硬件单元**：大部分算子运行在AI Core，少部分算子运行在AI CPU。默认情况下，项目中提到的算子一般指AI Core算子。关于AI Core和AI CPU详细介绍参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。
>- **算子接口列表**：为方便调用算子，CANN提供一套C API执行算子，一般以aclnn为前缀，全量接口参见[aclnn列表](op_api_list.md)。
>- **V版本演进说明**：部分算子存在多个V版本，使用时选择最高V版本即可（高版本算子已兼容低版本算子的所有能力）。

项目提供的所有算子分类和算子列表如下：

<table><thead>
  <tr>
    <th rowspan="2">算子分类</th>
    <th rowspan="2">算子目录</th>
    <th colspan="2">算子实现</th>
    <th>aclnn调用</th>
    <th>图模式调用</th>
    <th rowspan="2">算子执行硬件单元</th>
    <th rowspan="2">说明</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>op_api</th>
    <th>op_graph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>image</td>
    <td><a href="../../image/aipp/README.md">aipp</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>AIPP（Artificial Intelligence Pre-Processing）人工智能预处理，用于在AI Core上完成数据预处理，包括改变图像尺寸、色域转换（转换图像格式）、减均值/乘系数（改变图像像素），数据预处理之后再进行真正的模型推理。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/adjust_saturation/README.md">adjust_saturation</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>对RGB图像的饱和度进行调整。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/col2_im_v2/README.md">col2_im_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将滑动局部块重排组合为批处理图像张量（Col2Im，kernel_size/output_size 为 const tensor 输入）。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/col2im/README.md">col2im</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>从批处理输入张量中提取滑动局部块，将滑动局部块数组合并为一个大张量。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/combined_non_max_suppression/README.md">combined_non_max_suppression</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对每个batch、每个类别独立执行贪心非极大值抑制，再按置信度从高到低合并各类别的候选框。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/crop/README.md">crop</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>从输入张量中按照指定轴和偏移量裁剪出指定大小的子张量，兼容Caffe框架的Crop层。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/crop_and_resize/README.md">crop_and_resize</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>从输入图像中提取多个裁剪区域，并将它们统一调整为指定大小，支持双线性插值和最近邻插值。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/dilation2_d_backprop_filter/README.md">dilation2_d_backprop_filter</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>形态学膨胀2D操作（Dilation2D）的反向传播，计算filter的梯度。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/dilation2_d_backprop_input/README.md">dilation2_d_backprop_input</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Dilation2D形态学膨胀操作的输入梯度反向传播算子。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/extract_glimpse_v2/README.md">extract_glimpse_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>从批量输入图像中提取指定位置和大小的子图像（glimpse）。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/extract_image_patches/README.md">extract_image_patches</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>从输入图像中按指定ksizes、strides、rates和padding方式提取滑动局部块（patch），并将每个patch展平到通道维输出。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_sample/README.md">grid_sample</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>提供一个输入tensor以及一个对应的grid网格，然后根据grid中每个位置提供的坐标信息，将input中对应位置的像素值填充到网格指定的位置，得到最终的输出。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_unnormal/README.md">grid_unnormal</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将归一化采样网格坐标转换为输入特征图的像素坐标。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_sampler2_d/README.md">grid_sampler2_d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>根据二维采样网格提供的归一化坐标，对输入特征图进行双线性、最近邻或双三次插值采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_sampler2_d_grad/README.md">grid_sampler2_d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>GridSampler中2D场景的反向传播，完成张量input与张量grid的梯度计算。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_sampler3_d_grad/README.md">grid_sampler3_d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>GridSampler中3D场景的反向传播，完成张量input与张量grid的梯度计算。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/image_warp_offsets/README.md">image_warp_offsets</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>根据偏移量选取图像并进行扭曲变换。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/image_projective_transform/README.md">image_projective_transform</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对输入图像施加射影变换，根据变换矩阵将输出图像中的每个像素映射回输入图像中对应的坐标，再通过插值计算输出像素值。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/img_warp_resize/README.md">img_warp_resize</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>图像双线性插值采样算子，用于 OCR 场景中的图像变形缩放。接收预处理好的四角像素值和浮点坐标，通过双线性插值计算输出像素值。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/lut3_d/README.md">lut3_d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>通过3D颜色查找表对输入图像进行三线性插值颜色变换。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/non_max_suppression_v3/README.md">non_max_suppression_v3</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>按照分数递减顺序，采用贪心策略选择候选框（bounding boxes）子集。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/paste_sub_img/README.md">paste_sub_img</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>面向图像或特征图 patch 拼接场景的区域累加算子，从源图像按指定矩形子区域提取像素，经坐标缩放与平移量映射到目标画布对应位置执行逐元素累加，适用于图像拼接、超分辨率回填、滑窗推理特征图聚合等场景。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/points_in_polygons/README.md">points_in_polygons</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>判断给定二维点是否落在给定四边形内部，输出N×M的二值浮点矩阵，1.0表示在内、0.0表示在外。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/rasterizer/README.md">rasterizer</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>实现光栅化计算。根据给定的三维空间中的点和面，获取屏幕中每个像素点的最小深度及其对应的面片索引，并计算该面片的重心坐标透视矫正插值。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/rgb2_yuv422/README.md">rgb2_yuv422</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将 RGB 图像转换为 YUV422 (YUYV 打包格式) 色彩空间。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bicubic_v2/README.md">resize_bicubic_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>使用双三次插值调整图像大小到指定的大小。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bicubic_v2_grad/README.md">resize_bicubic_v2_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>计算输入图像在双三次插值基础下的梯度。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bilinear_v2/README.md">resize_bilinear_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>使用双线性插值调整图像大小到指定的大小。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bilinear_v2_grad/README.md">resize_bilinear_v2_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>ResizeBilinearV2的反向传播。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_grad/README.md">resize_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>计算Resize正向算子的反向梯度，按linear或cubic插值权重累加回原始分辨率。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_linear/README.md">resize_linear</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>使用单线性插值调整图像大小到指定的大小。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_linear_grad/README.md">resize_linear_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>计算输入图像在单线性插值基础下的梯度。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_nearest_neighbor_v2/README.md">resize_nearest_neighbor_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_nearest_neighbor_v2_grad/README.md">resize_nearest_neighbor_v2_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>根据最近邻插值的映射关系，将输出梯度散射回输入空间并累加。</td>
  </tr>
    <tr>
    <td>image</td>
    <td><a href="../../image/resize_upsample_trilinear/README.md">resize_upsample_trilinear</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用三线性插值算法进行上采样。</td>
  </tr>
    <tr>
      <td>image</td>
      <td><a href="../../image/scale_and_translate/README.md">scale_and_translate</a></td>
      <td>&check;</td>
      <td>&check;</td>
      <td>&cross;</td>
      <td>&check;</td>
      <td>AI CPU</td>
      <td>按给定输出尺寸、缩放因子和平移量对输入图像执行二维重采样。</td>
    </tr>
   <tr>
    <td>image</td>
    <td><a href="../../image/three_interpolate/README.md">three_interpolate</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>根据features、idx、weight进行3个最近邻加权特征插值得到y。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/three_interpolate_backward/README.md">three_interpolate_backward</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>根据grad_x,idx,weight进行三点插值计算梯度得到grad_y。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d/README.md">upsample_bicubic2d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用2D双三次上采样。如果输入Tensor x的shape为(N,C,H,W)，则输出Tensor out的shape为(N,C,outputSize[0],outputSize[1])。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d_aa/README.md">upsample_bicubic2d_aa</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用双三次抗锯齿算法进行上采样。如果输入Tensor x的shape为(N,C,H,W)，则输出Tensor out的shape为(N,C,outputSize[0],outputSize[1])。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d_aa_grad/README.md">upsample_bicubic2d_aa_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>如果输入张量grad_output的shape为(N,C,H,W)，则输出张量grad_input的shape为(N,C,inputSize[2],inputSize[3])。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d_grad/README.md">upsample_bicubic2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>如果输入张量grad_output的shape为(N,C,H,W)，则输出张量grad_input的shape为(N,C,inputSize[2],inputSize[3])。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d/README.md">upsample_bilinear2d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用2D双线性上采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d_aa/README.md">upsample_bilinear2d_aa</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用2D双线性抗锯齿采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d_aa_backward/README.md">upsample_bilinear2d_aa_backward</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>UpsampleBilinear2dAA的反向传播。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d_grad/README.md">upsample_bilinear2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>UpsampleBilinear2d的反向传播。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_linear1d/README.md">upsample_linear1d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用线性插值算法进行上采样。如果输入shape为(N, C, L)，则输出shape为(N, C, outputSize)。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest/README.md">upsample_nearest</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest_exact2d_grad/README.md">upsample_nearest_exact2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td><a href="../../image/upsample_nearest/README.md">UpsampleNearest</a>在exact_mode为true时的反向传播。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest_exact3d/README.md">upsample_nearest_exact3d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest_exact3d_grad/README.md">upsample_nearest_exact3d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>UpsampleNearestExact3d的反向计算。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest2d_grad/README.md">upsample_nearest2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>UpsampleNearest在exact_mode为false时的反向传播。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest3d/README.md">upsample_nearest3d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest3d_grad/README.md">upsample_nearest3d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>UpsampleNearest3d的反向计算。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_trilinear3d_backward/README.md">upsample_trilinear3d_backward</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>ResizeUpsampleTrilinear的反向计算。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/spatial_transformer/README.md">spatial_transformer</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>用于对输入图像或特征图进行几何变换等操作。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/yuv4442_yuv422/README.md">yuv4442_yuv422</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将 YUV444 格式图像数据转换为 YUV422 格式。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/blend_face_bg_part_two/README.md">blend_face_bg_part_two</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>完成人脸融合背景第二阶段的归一化与Alpha合成计算，支持uint8/float32背景图输入。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/mrgba_custom/README.md">mrgba_custom</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>完成张量rgb和张量alpha的透明度乘法计算。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/blend_images_custom/README.md">blend_images_custom</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>完成张量rgb、frame和alpha的透明度乘法计算。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/background_replace/README.md">background_replace</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将输入的新的背景图片与已有图片进行融合，通过掩码的方式将背景替换为新的背景。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/bounding_box_encode/README.md">bounding_box_encode</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>计算锚框与真实边界框之间的编码偏移量，生成目标检测回归目标。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/bounding_box_decode/README.md">bounding_box_decode</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将模型输出的相对于先验框（或锚点）的偏移量与缩放参数，转换为原始图像中真实的绝对边界框坐标。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/rotated_box_decode/README.md">rotated_box_decode</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>旋转目标检测中的框回归解码算子，将网络预测的偏移量叠加到预设锚框上，还原出最终的旋转检测框。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/rotated_box_encode/README.md">rotated_box_encode</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将参考旋转框与ground-truth旋转框之间的几何偏差编码为5通道回归delta目标，用于旋转目标检测训练。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/rotated_feature_align/README.md">rotated_feature_align</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>旋转目标检测中的特征对齐算子，根据旋转框参数对输入特征图进行双线性插值采样，将旋转区域特征对齐并叠加到原特征上。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/batch_multi_class_non_max_suppression/README.md">batch_multi_class_non_max_suppression</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对每个batch、每个类别的候选框执行贪心非极大值抑制，再从所有类别的保留结果中按分数选择检测框。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/to_absolute_b_box/README.md">to_absolute_b_box</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将归一化边界框坐标按图像高宽转换为绝对像素坐标，用于目标检测推理后处理。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/ciou/README.md">ciou</a></td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>用于边界框回归的损失函数，在IoU的基础上同时考虑了中心点距离、宽高比和重叠面积，以更全面地衡量预测框与真实框之间的差异。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/d_io_u_grad/README.md">d_io_u_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>计算Distance-IoU (DIoU)损失函数的反向梯度。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/deformable_roi_pool/README.md">deformable_roi_pool</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>可变形感兴趣区域池化，从特征图中提取每个ROI位置的池化特征，支持通过offset对采样点进行可变形偏移。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/deformable_roi_pool_grad/README.md">deformable_roi_pool_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>计算可变形 ROI 池化对 feature_map 和 offset 的反向梯度。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/g_io_u_grad/README.md">g_io_u_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>计算Generalized-IoU (GIoU)损失函数的反向梯度。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/iou3d/README.md">iou3d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>3D旋转框IoU算子：对两组7-DoF旋转框，先在BEV（鸟瞰）平面求旋转矩形交集面积，乘以Z轴重叠高度得到交集体积，再除以并集体积。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/iou_v2/README.md">iou_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>计算两个矩阵的重叠面积占两个矩阵总面积的比例，设预测框的左上角坐标为（X1，Y1），右下角坐标为（X2，Y2），真实框的左上角坐标为（X3，Y3），右下角坐标为（X4，Y4）。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/non_max_suppression_v6/README.md">non_max_suppression_v6</a></td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/non_max_suppression_v7/README.md">non_max_suppression_v7</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>按batch和类别对候选框执行贪心非极大值抑制，输出被选中框的索引三元组。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align/README.md">roi_align</a></td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>用于从特征图中提取感兴趣区域(ROI)的特征，通过双线性插值和池化生成固定大小的输出特征图。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align_grad/README.md">roi_align_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>用于从非均匀尺寸的特征图中提取固定尺寸的ROI（Region of Interest）特征。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align_rotated/README.md">roi_align_rotated</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>用于旋转候选框的ROI对齐池化层。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align_rotated_grad/README.md">roi_align_rotated_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>通过旋转框各点坐标将梯度回传至对应位置。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_pooling_with_arg_max/README.md">roi_pooling_with_arg_max</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>对输入特征图按ROI（感兴趣区域）进行池化，在每个ROI内按空间划分格子，对每个格子做最大池化，并输出池化结果及最大值在通道内的一维索引。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_pooling_grad_with_arg_max/README.md">roi_pooling_grad_with_arg_max</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>遍历每个ROI的池化结果，将feature map坐标上的反向梯度贡献累加，即完成整张图上的反向计算。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/rotated_overlaps/README.md">rotated_overlaps</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>计算两组二维旋转矩形框之间的交叠面积。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/rotated_feature_align_grad/README.md">rotated_feature_align_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>RotatedFeatureAlign的反向传播，根据输出特征梯度与各像素旋转框信息，将采样点双线性插值梯度回传至输入特征。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/stack_group_points/README.md">stack_group_points</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>根据特征点所属的组，重组点云中的特征点。 </td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/sorted_nms/README.md">sorted_nms</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>在已按分数降序排列的候选框序列上，按照交并比阈值贪心选择非抑制框，输出被选中框在原始候选框中的索引。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/anchor_response_flags/README.md">anchor_response_flags</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>在目标检测网络中生成锚框的响应标志。根据真值框的中心点位置，确定哪些锚框网格位置负责检测目标，并生成对应的标志位。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/yolo/README.md">yolo</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>对YOLOv2/v3目标检测网络的检测特征图进行数据重组和激活处理，将原始卷积输出转换为检测框坐标、目标置信度和类别概率三个输出。</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/yolox_bounding_box_decode/README.md">yolox_bounding_box_decode</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>根据YOLOX解码公式，将模型预测的边界框偏移量与先验框解码为左上角和右下角坐标。</td>
  </tr>
   <tr>
    <td>image</td>
    <td><a href="../../image/nms_with_mask/README.md">nms_with_mask</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>完成带掩码非极大值抑制计算。</td>
   </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/img_raw_decode_post_handle/README.md">img_raw_decode_post_handle</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将YUV格式的4通道图像数据转换为RAW Bayer格式输出，同时进行伽马校正和黑电平补偿，支持binning和quad两种Bayer排列模式。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../objdetect/check_valid/README.md">check_valid</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>检查给定的边界框（Bounding Boxes）是否位于指定的原始图片有效边界内。</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../objdetect/decode_bbox_v2/README.md">decode_bbox_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>将目标检测回归偏移量（boxes）结合锚框（anchors）解码为绝对坐标框（ymin, xmin, ymax, xmax）。</td>
  </tr>
</tbody>
</table>
