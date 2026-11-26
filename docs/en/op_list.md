# Operator List

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-18T08:11:16.452Z pushedAt=2026-11-18T09:59:29.697Z nl=2 -->

> Description:
>
>- **Operator directory**: The directory name is the lowercase underscore form of the operator name. Each directory contains all deliverables of the operator, including the code implementation, examples, and documents. For details about the directory, see [Project Directory](./install/dir_structure.md).
>- **Operator execution hardware unit**: Most operators run on AI Core, and a few run on AI CPU. By default, operators mentioned in the project generally refer to AI Core operators. For details about AI Core and AI CPU, see "Concepts, Principles, and Terms > Hardware Architecture and Data Processing Principles" in [*Ascend C Operator Development*](https://hiascend.com/en/document/redirect/CannCommunityOpdevAscendC).
>- **Operator API list**: To facilitate operator calls, CANN provides a set of C APIs for executing operators, generally prefixed with aclnn. For the full list of APIs, see [aclnn List](op_api_list.md).
>- **V version evolution description**: Some operators have multiple V versions. Select the highest V version when using them (a higher-version operator is compatible with all capabilities of lower-version operators).

All operator categories and operators provided by the project are as follows:

<table><thead>
  <tr>
    <th rowspan="2">Classification</th>
    <th rowspan="2">Directory</th>
    <th colspan="2">Implementation</th>
    <th>aclnn Call</th>
    <th>Graph Mode Call</th>
    <th rowspan="2">Operator Execution Hardware Unit</th>
    <th rowspan="2">Description</th>
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
    <td>AIPP (Artificial Intelligence Pre-Processing) is used to complete data preprocessing on AI Core, including changing the image size, color space conversion (converting the image format), and mean subtraction/coefficient multiplication (changing image pixels). Model inference is then performed after data preprocessing.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/adjust_saturation/README.md">adjust_saturation</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>Adjusts the saturation of an RGB image.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/col2_im_v2/README.md">col2_im_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Rearranges and combines sliding local blocks into a batch image tensor (Col2Im, with kernel_size/output_size as const tensor inputs).</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/col2im/README.md">col2im</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Extracts sliding local blocks from the batch input tensor and merges the sliding local block arrays into a large tensor.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/combined_non_max_suppression/README.md">combined_non_max_suppression</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Performs greedy non-maximum suppression independently for each batch and each category, and then merges the candidate boxes of each category by confidence in descending order.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/crop/README.md">crop</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Crops a sub-tensor of the specified size from the input tensor according to the specified axis and offset. It is compatible with the Crop layer of the Caffe framework.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/crop_and_resize/README.md">crop_and_resize</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>Extracts multiple crop regions from the input image and resizes them to a specified size. Supports bilinear interpolation and nearest-neighbor interpolation.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/dilation2_d_backprop_filter/README.md">dilation2_d_backprop_filter</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Backward propagation of the 2D morphological dilation operation (Dilation2D). Computes the gradient of the filter.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/dilation2_d_backprop_input/README.md">dilation2_d_backprop_input</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Input gradient backward propagation operator of the Dilation2D morphological dilation operation.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/extract_glimpse_v2/README.md">extract_glimpse_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Extracts sub-images (glimpses) of specified position and size from the batch of input images.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/extract_image_patches/README.md">extract_image_patches</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Extracts sliding local blocks (patches) from the input image according to the specified ksizes, strides, rates, and padding. Flattens each patch into the channel dimension for output.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_sample/README.md">grid_sample</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Provides an input tensor and a corresponding grid. Fills the pixel values at the corresponding positions in the input into the positions specified by the grid according to the coordinate information provided by each position in the grid, to obtain the final output.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_unnormal/README.md">grid_unnormal</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Converts normalized sampling grid coordinates to pixel coordinates of the input feature map.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_sampler2_d/README.md">grid_sampler2_d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Performs bilinear, nearest-neighbor, or bicubic interpolation sampling on the input feature map according to the normalized coordinates provided by the two-dimensional sampling grid.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_sampler2_d_grad/README.md">grid_sampler2_d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Backward propagation of the 2D use case in GridSampler. Completes the gradient computation of the input tensor and the grid tensor.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/grid_sampler3_d_grad/README.md">grid_sampler3_d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Backward propagation of the 3D use case in GridSampler. Completes the gradient computation of the input tensor and the grid tensor.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/image_warp_offsets/README.md">image_warp_offsets</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>Selects an image according to the offset and performs warp transformation.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/image_projective_transform/README.md">image_projective_transform</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Applies a projective transformation to the input image. Maps each pixel in the output image back to the corresponding coordinates in the input image according to the transformation matrix, and then computes the output pixel value through interpolation.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/img_warp_resize/README.md">img_warp_resize</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Image bilinear interpolation sampling operator used for image deformation and scaling in OCR use cases. Receives the preprocessed four-corner pixel values and floating-point coordinates, and computes the output pixel value through bilinear interpolation.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/lut3_d/README.md">lut3_d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Performs trilinear interpolation color transformation on the input image through a 3D color lookup table.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/non_max_suppression_v3/README.md">non_max_suppression_v3</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>Selects a subset of candidate boxes (bounding boxes) in descending order of scores using a greedy strategy.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/paste_sub_img/README.md">paste_sub_img</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>A region accumulation operator for image or feature map patch stitching. It extracts pixels from the source image in a specified rectangular sub-region, maps them to corresponding positions on the target canvas through coordinate scaling and translation, and performs element-wise accumulation. It is applicable to image stitching, super-resolution backfill, sliding-window inference feature map aggregation, and other use cases.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/points_in_polygons/README.md">points_in_polygons</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Determines whether a given two-dimensional point falls inside a given quadrilateral, and outputs an N×M binary floating-point matrix, where 1.0 indicates inside and 0.0 indicates outside.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/rasterizer/README.md">rasterizer</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Implements rasterization computation. Based on given points and faces in three-dimensional space, it obtains the minimum depth of each pixel on the screen and its corresponding face index, and computes the perspective-corrected barycentric coordinate interpolation of that face.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/rgb2_yuv422/README.md">rgb2_yuv422</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Converts an RGB image to the YUV422 (YUYV packed format) color space.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bicubic_v2/README.md">resize_bicubic_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Adjusts the image size to a specified size using bicubic interpolation.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bicubic_v2_grad/README.md">resize_bicubic_v2_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Computes the gradient of the input image based on bicubic interpolation.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bilinear_v2/README.md">resize_bilinear_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Adjusts the image size to a specified size using bilinear interpolation.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_bilinear_v2_grad/README.md">resize_bilinear_v2_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Backward propagation of ResizeBilinearV2.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_grad/README.md">resize_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Computes the backward gradient of the Resize forward operator, accumulating back to the original resolution according to linear or cubic interpolation weights.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_linear/README.md">resize_linear</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Uses unilinear interpolation to resize an image to a specified size.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_linear_grad/README.md">resize_linear_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Computes the gradient of the input image on the basis of unilinear interpolation.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_nearest_neighbor_v2/README.md">resize_nearest_neighbor_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Applies the nearest-neighbor interpolation algorithm to upsample an input signal composed of multiple input channels.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/resize_nearest_neighbor_v2_grad/README.md">resize_nearest_neighbor_v2_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Scatters the output gradient back to the input space and accumulates it according to the nearest-neighbor interpolation mapping.</td>
  </tr>
    <tr>
    <td>image</td>
    <td><a href="../../image/resize_upsample_trilinear/README.md">resize_upsample_trilinear</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Applies the trilinear interpolation algorithm to upsample an input signal composed of multiple input channels.</td>
  </tr>
    <tr>
      <td>image</td>
      <td><a href="../../image/scale_and_translate/README.md">scale_and_translate</a></td>
      <td>&check;</td>
      <td>&check;</td>
      <td>&cross;</td>
      <td>&check;</td>
      <td>AI CPU</td>
      <td>Performs two-dimensional resampling on the input image according to the given output size, scaling factor, and translation amount.</td>
    </tr>
   <tr>
    <td>image</td>
    <td><a href="../../image/three_interpolate/README.md">three_interpolate</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Performs weighted feature interpolation of three nearest neighbors according to features, idx, and weight to obtain y.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/three_interpolate_backward/README.md">three_interpolate_backward</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Performs three-point interpolation computation according to grad_x, idx, and weight to obtain the gradient grad_y.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d/README.md">upsample_bicubic2d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Applies 2D bicubic upsampling to an input signal composed of multiple input channels. If the shape of the input tensor x is (N, C, H, W), the shape of the output tensor out is (N, C, outputSize[0], outputSize[1]).</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d_aa/README.md">upsample_bicubic2d_aa</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Applies a bicubic anti-aliasing algorithm to upsample an input signal composed of multiple input channels. If the shape of the input tensor x is (N, C, H, W), the shape of the output tensor out is (N, C, outputSize[0], outputSize[1]).</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d_aa_grad/README.md">upsample_bicubic2d_aa_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>If the shape of the input tensor grad_output is (N, C, H, W), the shape of the output tensor grad_input is (N, C, inputSize[2], inputSize[3]).</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bicubic2d_grad/README.md">upsample_bicubic2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>If the shape of the input tensor grad_output is (N, C, H, W), the shape of the output tensor grad_input is (N, C, inputSize[2], inputSize[3]).</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d/README.md">upsample_bilinear2d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Applies 2D bilinear upsampling to an input signal composed of multiple input channels.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d_aa/README.md">upsample_bilinear2d_aa</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Applies 2D bilinear anti-aliasing sampling to an input signal composed of multiple input channels.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d_aa_backward/README.md">upsample_bilinear2d_aa_backward</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Backward propagation of UpsampleBilinear2dAA.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_bilinear2d_grad/README.md">upsample_bilinear2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Backward propagation of UpsampleBilinear2d.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_linear1d/README.md">upsample_linear1d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Applies a linear interpolation algorithm to upsample an input signal composed of multiple input channels. If the input shape is (N, C, L), the output shape is (N, C, outputSize).</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest/README.md">upsample_nearest</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Applies a nearest-neighbor interpolation algorithm to upsample an input signal composed of multiple input channels.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest_exact2d_grad/README.md">upsample_nearest_exact2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Backward propagation of <a href="../../image/upsample_nearest/README.md">UpsampleNearest</a> when exact_mode is true.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest_exact3d/README.md">upsample_nearest_exact3d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Applies a nearest-neighbor interpolation algorithm to upsample an input signal composed of multiple input channels.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest_exact3d_grad/README.md">upsample_nearest_exact3d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Backward computation of UpsampleNearestExact3d.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest2d_grad/README.md">upsample_nearest2d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Backward propagation of UpsampleNearest when exact_mode is false.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest3d/README.md">upsample_nearest3d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Applies the nearest-neighbor interpolation algorithm to upsample an input signal composed of multiple input channels.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_nearest3d_grad/README.md">upsample_nearest3d_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Backward computation of UpsampleNearest3d.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/upsample_trilinear3d_backward/README.md">upsample_trilinear3d_backward</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Backward computation of ResizeUpsampleTrilinear.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/spatial_transformer/README.md">spatial_transformer</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI CPU</td>
    <td>Performs operations such as geometric transformation on an input image or feature map.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/yuv4442_yuv422/README.md">yuv4442_yuv422</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Converts YUV444 format image data to YUV422 format.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/blend_face_bg_part_two/README.md">blend_face_bg_part_two</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Completes the normalization and alpha compositing computation of the second stage of face fusion background, supporting uint8/float32 background image input.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/mrgba_custom/README.md">mrgba_custom</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Completes the transparency multiplication computation of tensor rgb and tensor alpha.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/blend_images_custom/README.md">blend_images_custom</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Completes the transparency multiplication computation of tensor rgb, frame, and alpha.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/background_replace/README.md">background_replace</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Fuses the input new background image with the existing image, replacing the background with the new one through a mask.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/bounding_box_encode/README.md">bounding_box_encode</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Computes the encoded offset between anchor boxes and ground truth bounding boxes, generating object detection regression targets.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/bounding_box_decode/README.md">bounding_box_decode</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Converts the offset and scaling parameters output by the model relative to prior boxes (or anchors) into real absolute bounding box coordinates in the original image.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/rotated_box_decode/README.md">rotated_box_decode</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>A box regression decode operator in rotated object detection that superimposes the offsets predicted by the network onto preset anchor boxes to restore the final rotated detection boxes.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/rotated_box_encode/README.md">rotated_box_encode</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Encodes the geometric deviation between reference rotated boxes and ground truth rotated boxes into 5-channel regression delta targets for rotated object detection training.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/rotated_feature_align/README.md">rotated_feature_align</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Feature alignment operator in rotated object detection. Performs bilinear interpolation sampling on the input feature map according to rotated box parameters, aligns the rotated region features, and superimposes them onto the original features.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/batch_multi_class_non_max_suppression/README.md">batch_multi_class_non_max_suppression</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Performs greedy non-maximum suppression on candidate boxes of each category for each batch, then selects detection boxes by score from the retained results of all categories.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/to_absolute_b_box/README.md">to_absolute_b_box</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Converts normalized bounding box coordinates to absolute pixel coordinates according to image height and width. Used for post-processing in object detection inference.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/ciou/README.md">ciou</a></td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Loss function for bounding box regression. Based on IoU, it also considers center point distance, aspect ratio, and overlap area to more comprehensively measure the difference between the predicted box and the ground truth box.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/d_io_u_grad/README.md">d_io_u_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Computes the backward gradient of the Distance-IoU (DIoU) loss function.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/deformable_roi_pool/README.md">deformable_roi_pool</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Deformable region of interest pooling. Extracts pooled features at each ROI position from the feature map, and supports deformable offset of sampling points through offset.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/deformable_roi_pool_grad/README.md">deformable_roi_pool_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Computes the backward gradient of deformable ROI pooling with respect to feature_map and offset.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/g_io_u_grad/README.md">g_io_u_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Computes the backward gradient of the Generalized-IoU (GIoU) loss function.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/iou3d/README.md">iou3d</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>3D rotated box IoU operator: for two groups of 7-DoF rotated boxes, first computes the rotated rectangle intersection area on the BEV (bird's-eye view) plane, multiplies it by the Z-axis overlap height to obtain the intersection volume, and then divides it by the union volume.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/iou_v2/README.md">iou_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Computes the ratio of the overlap area of two matrices to the total area of the two matrices. The top-left corner coordinates of the predicted box are (X1, Y1), the bottom-right corner coordinates are (X2, Y2), the top-left corner coordinates of the ground truth box are (X3, Y3), and the bottom-right corner coordinates are (X4, Y4).</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/non_max_suppression_v6/README.md">non_max_suppression_v6</a></td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>This operator has no Ascend C code implementation yet. Contributions are welcome. For details, see the <a href="../../CONTRIBUTING.md">Contribution Guide</a>.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/non_max_suppression_v7/README.md">non_max_suppression_v7</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Performs greedy non-maximum suppression on candidate boxes by batch and category, and outputs the index triples of the selected boxes.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align/README.md">roi_align</a></td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Extracts features of regions of interest (ROI) from a feature map, and generates a fixed-size output feature map through bilinear interpolation and pooling.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align_grad/README.md">roi_align_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Extracts fixed-size ROI (Region of Interest) features from a feature map of non-uniform size.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align_rotated/README.md">roi_align_rotated</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>ROI align pooling layer for rotated candidate boxes.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_align_rotated_grad/README.md">roi_align_rotated_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Propagates the gradient back to the corresponding position through the point coordinates of each rotated box.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_pooling_with_arg_max/README.md">roi_pooling_with_arg_max</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Pools the input feature map by ROI (region of interest), divides each ROI into spatial cells, performs max pooling on each cell, and outputs the pooling result and the one-dimensional index of the maximum value within the channel.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/roi_pooling_grad_with_arg_max/README.md">roi_pooling_grad_with_arg_max</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>AI Core</td>
    <td>Traverses the pooling result of each ROI and accumulates the backward gradient contributions on the feature map coordinates, completing the backward computation on the entire image.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/rotated_overlaps/README.md">rotated_overlaps</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Calculates the overlap area between two groups of two-dimensional rotated rectangles.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/rotated_feature_align_grad/README.md">rotated_feature_align_grad</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Backward propagation of RotatedFeatureAlign. Propagates the bilinear interpolation gradient of the sampling points back to the input features according to the output feature gradient and the rotated box information of each pixel.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/stack_group_points/README.md">stack_group_points</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Reorganizes the feature points in the point cloud according to the group to which each feature point belongs. </td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/sorted_nms/README.md">sorted_nms</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>On a candidate box sequence already sorted by score in descending order, greedily selects non-suppressed boxes according to the intersection-over-union threshold, and outputs the indices of the selected boxes in the original candidate boxes.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/anchor_response_flags/README.md">anchor_response_flags</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Generates the response flags of anchor boxes in an object detection network. Determines which anchor box grid positions are responsible for detecting the target according to the center point position of the ground truth box, and generates the corresponding flags.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/yolo/README.md">yolo</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Performs data rearrangement and activation processing on the detection feature map of the YOLOv2/v3 object detection network, and converts the original convolution output into three outputs: detection box coordinates, target confidence, and category probability.</td>
  </tr>
  <tr>
    <td>objdetect</td>
    <td><a href="../../objdetect/yolox_bounding_box_decode/README.md">yolox_bounding_box_decode</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Decodes the bounding box offset and prior box predicted by the model into top-left corner and bottom-right corner coordinates according to the YOLOX decoding formula.</td>
  </tr>
   <tr>
    <td>image</td>
    <td><a href="../../image/nms_with_mask/README.md">nms_with_mask</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Completes non-maximum suppression computation with masks.</td>
   </tr>
  <tr>
    <td>image</td>
    <td><a href="../../image/img_raw_decode_post_handle/README.md">img_raw_decode_post_handle</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Converts 4-channel YUV image data to RAW Bayer format output, and simultaneously performs gamma correction and black level compensation. Supports both binning and quad Bayer arrangement modes.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../objdetect/check_valid/README.md">check_valid</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Checks whether the given bounding boxes are within the specified valid boundary of the original image.</td>
  </tr>
  <tr>
    <td>image</td>
    <td><a href="../../objdetect/decode_bbox_v2/README.md">decode_bbox_v2</a></td>
    <td>&check;</td>
    <td>&check;</td>
    <td>&cross;</td>
    <td>&check;</td>
    <td>AI Core</td>
    <td>Decodes the object detection regression offset (boxes) combined with anchor boxes into absolute coordinate boxes (ymin, xmin, ymax, xmax).</td>
  </tr>
</tbody>
</table>
