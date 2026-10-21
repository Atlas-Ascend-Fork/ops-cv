/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file op_cv_proto_extend.h
 * \brief
 */
#ifndef OPS_OP_CV_PROTO_EXTEND_H_
#define OPS_OP_CV_PROTO_EXTEND_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief First calculate the minimum closure area of the two boxes, IoU,
* the proportion of the closed area that does not belong to the two boxes in the closure area,
* and finally subtract this proportion from IoU to get GIoU . \n

* @par Inputs:
* Two inputs, including:
* @li bboxes: Bounding boxes, a 2D Tensor of type float16 or float32 with
* shape (N, 4). "N" indicates the number of bounding boxes, and the value
* "4" refers to [x1, y1, x2, y2] or [x, y, w, h].
* @li gtboxes: Ground-truth boxes, a 2D Tensor of type float16 or float32
* with shape (M, 4). "M" indicates the number of ground truth boxes, and
* the value "4" refers to [x1, y1, x2, y2] or [x, y, w, h] . \n

* @par Attributes:
* @li trans: An optional bool, true for 'xywh', false for 'xyxy'.
* @li is_cross: An optional bool, control whether the output shape is [M, N] or [1, N]
* @li mode: Computation mode, a character string with the value range of [iou, iof] . \n

* @par Outputs:
* overlap: A 2D Tensor of type float16 or float32 with shape [M, N] or [1, N],
* specifying the IoU or IoF ratio . \n

* @attention Constraints:
* Only computation of float16 data is supported. To avoid overflow, the input
* length and width are scaled by 0.2 internally.
*/
REG_OP(GIoU)
    .INPUT(bboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gtboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(overlap, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(trans, Bool, false)
    .ATTR(is_cross, Bool, true)
    .ATTR(mode, String, "iou")
    .OP_END_FACTORY_REG(GIoU)

/**
*@brief RotatedIou . \n

* @par Inputs:
*@li boxes : data of grad increment, a 3D Tensor of type float32 with
* shape (B, 5, N). "N" indicates the number of boxes, and the value
* "5" refers to [x1, y1, x2, y2, theta] or [x, y, w, h, theta].
* @li query_boxes: Bounding boxes, a 3D Tensor of type float32 with
* shape (B, 5, K). "K" indicates the number of boxes, and the value
* "5" refers to [x1, y1, x2, y2, theta] or [x, y, w, h, theta].

* @par Attributes:
*@li trans: An optional attr, true for 'xyxyt', false for 'xywht'.
* @li mode: An optional attr, a character string with the value range of ['iou', 'iof'],
* only support 'iou' now.
*@li is_cross: Cross calculation when it is True, and one-to-one calculation when it is False.
*@li v_threshold: An optional attr, provide condition relaxation for intersection calculation.
* @li e_threshold: An optional attr, provide condition relaxation for intersection calculation.

*@par Outputs:
* iou: A 3D Tensor of float32 with shape [B, N, K].

*@attention Constraints:
* In each batch, the invalid box cannot appear before the valid box.
*/
REG_OP(RotatedIou)
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(query_boxes, TensorType({DT_FLOAT}))
    .OUTPUT(iou, TensorType({DT_FLOAT}))
    .ATTR(trans, Bool, false)
    .ATTR(mode, String, "iou")
    .ATTR(is_cross, Bool, true)
    .ATTR(v_threshold, Float, 0)
    .ATTR(e_threshold, Float, 0)
    .OP_END_FACTORY_REG(RotatedIou)

/**
* @brief Resize the input tensor. \n
currently, only support resize image tensor using nearest neighbor and linear interpolation.

* @par Inputs:
* Input x must be a 4-D tensor. Inputs include: \n
* @li x: A Tensor. Must be one of the following types: uint8, int8, int16, \n
int32, int64, float16, float, double. 4-D with shape [batch, height, width, channels] \n
or shape [batch, channels, height, width].
* @li roi: A 1-D float Tensor. Only takes effect when attr coordinate_transformation_mode \n
is "tf_crop_and_resize". Must be one of the following types: float16, float, double.
* @li scales: A 1-D float Tensor, the scale array along each dimension, Only one of \n
'scales' and 'sizes' can be specified. Must be float type.
* @li sizes: A 1-D int64 Tensor, The size of the output tensor. Only one of \n
'scales' and 'sizes' can be specified.  If 'size' is specified, then set scales \n
to empty data (zero shape) in this operator's input list. Must be one of \n
the following types: int32, int64.

* @par Attributes:
* @li coordinate_transformation_mode: An optional String. how to transform \n
the coordinate in the resized tensor to the coordinate in the original tensor. \n
options: pytorch_half_pixel, align_corners, asymmetric, \n
tf_crop_and_resize.
* @li cubic_coeff_a: An optional Float. Defaults to -0.75, only used in cubic interpolation. \n
other optional: -0.5
* @li exclude_outside: An optional Int. Defaults to 0, If set to 1, the weight of sampling \n
locations outside the tensor will be set to 0 and the weight will be renormalized \n
so that their sum is 1.0.
* @li extrapolation_value: An optional Float. Defaults to 0.0f. When coordinate_transformation_mode \n
is "tf_crop_and_resize" and x_original is outside the range [0, length_original - 1], \n
this value is used as the corresponding output value.
* @li mode: An optional String. Defaults to nearest. Three interpolation modes: nearest (default), \n
linear and cubic.
* @li nearest_mode: An optional String. Defaults to round_prefer_floor. Four modes: round_prefer_floor, \n
round_prefer_ceil, floor, ceil. Only used by nearest interpolation.

* @par Outputs:
* y: A Tensor. Has the same type as x.

* @attention Constraints: \n
* Input x must be a 4-D tensor.

* @par Third-party framework compatibility
* Compatible with tensorflow ResizeNearestNeighborV2 operator.
*/

#ifndef OPS_PROTO_DEF_RESIZE
#define OPS_PROTO_DEF_RESIZE
REG_OP(Resize)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OPTIONAL_INPUT(roi, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(sizes, TensorType({DT_INT64, DT_INT32}))
    .OUTPUT(y,
            TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(coordinate_transformation_mode, String, "half_pixel")
    .ATTR(cubic_coeff_a, Float, -0.75)
    .ATTR(exclude_outside, Int, 0)
    .ATTR(extrapolation_value, Float, 0.0)
    .ATTR(mode, String, "nearest")
    .ATTR(nearest_mode, String, "round_prefer_floor")
    .OP_END_FACTORY_REG(Resize)
#endif

/**
*@brief Greedily selects a subset of bounding boxes in descending order of
score . \n

*@par Inputs:
*Input boxes and  scores must be float type. Inputs include:
*@li boxes: A 2-D float tensor of shape [num_boxes, 4].
*@li scores: A 1-D float tensor of shape [num_boxes] representing a single
score corresponding to each box (each row of boxes).
*@li max_output_size: A scalar integer tensor representing the maximum number
of boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding
whether boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for
deciding when to remove boxes based on score . \n

*@par Attributes:
*pad_to_max_output_size: If true, the output selected_indices is padded
to be of length max_output_size. Defaults to false . \n

*@par Outputs:
*@li selected_indices: A 1-D integer tensor of shape [M] representing the
selected indices from the boxes tensor, where M <= max_output_size.
*@li valid_outputs: A 0-D integer tensor representing the number of valid
elements in selected_indices, with the valid elements appearing first . \n

*@attention Constraints:
*Input boxes and  scores must be float type . \n

*@par Third-party framework compatibility
*Compatible with tensorflow NonMaxSuppressionV4 operator.
*/

REG_OP(NonMaxSuppressionV4)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .OUTPUT(valid_outputs, TensorType({DT_INT32}))
    .ATTR(pad_to_max_output_size, Bool, false)
    .OP_END_FACTORY_REG(NonMaxSuppressionV4)

/**
* @brief First calculate the minimum closure area of the two boxes, IoU,
* The DIoU is obtained by combining the center distance and IoU. \n

* @par Inputs:
* Two inputs, including:
* @li bboxes: Bounding boxes, a 2D Tensor of type float16 or float32 with
* shape (4, N). "N" indicates the number of bounding boxes, and the value
* "4" refers to [x1, y1, x2, y2] or [x, y, w, h].
* @li gtboxes: Ground-truth boxes, a 2D Tensor of type float16 or float32
* with shape (4, M). "M" indicates the number of ground truth boxes, and
* the value "4" refers to [x1, y1, x2, y2] or [x, y, w, h] . \n

* @par Attributes:
* @li trans: An optional bool, true for 'xywh', false for 'xyxy', default value is false.
* @li is_cross: An optional bool, control whether the output shape is [N, M] or [1, N], default value is false.
* @li mode: An optional string, computation mode, a character string with the value range of [iou, iof],
* default value is 'iou'. \n

* @par Outputs:
* overlap: A 2D Tensor of type float16 or float32 with shape [N, M] or [1, N],
* specifying the IoU or IoF ratio . \n

* @attention Constraints:
* "is_cross" only support false.
*/
REG_OP(DIoU)
    .INPUT(bboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gtboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(overlap, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(trans, Bool, false)
    .ATTR(is_cross, Bool, true)
    .ATTR(mode, String, "iou")
    .OP_END_FACTORY_REG(DIoU)

/**
* @brief Performs Dilation2D on the input .

* @par Inputs:
* @li x: A 4D tensor, format support NHWC. Support following data types:
* float16, float, double, int32, int64, uint8, int16, int8, uint16.
* @li filter: A tensor of shape is 3d, the type is same with x,
* and the c dimension is same with x. \n

* @par Attributes:
* @li strides: A required list of 4 ints, specifying the stride of the
* sliding window. The strides of the N and C dimensions are 1.
* @li rates: A required list of 4 ints.
* The rates of the N and C dimensions are 1.
* @li padding_mode: A optional string.
* Defaults to "SAME", it support SAME and VALID.
* @li pads: An optional list of 4 ints. Defaults to {0, 0, 0, 0}.
* @li ceil_mode: An optional bool. Defaults to "false". Use ceil or floor
* to calculate the output size when padding_mode is "CALCULATED".
* @li data_format: An optional string, specifying the data format
* of "rates" and "strides", either "NCHW" or "NHWC" (default). \n

* @par Outputs:
* y: The output tensor. Has the same type and format as input "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Dilation2D.
*/
#ifndef OPS_PROTO_DEF_DILATION2D
#define OPS_PROTO_DEF_DILATION2D
REG_OP(Dilation2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .INPUT(filter,
           TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .OUTPUT(y,
            TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(rates, ListInt)
    .ATTR(padding_mode, String, "SAME")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(ceil_mode, Bool, false)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(Dilation2D)
#endif
/**
* @brief Select top 'pre_nms_topn' scoring boxes, decodes them with respect to anchors, applies non-maximal
suppression
* on overlapping boxes with higher than 'nms_threshold' intersection-over-union (IoU) value, discarding boxes where
* shorter side is less than 'min_size'. \n

* @par Inputs:
* scores: 4-D tensor with shape of [num_images, height, width, num_anchors], containing the scores of the boxes for
* given anchors, can be unsorted. Must be one of the following types: float16, float32 . \n
* bbox_deltas: 4-D tensor with shape of [num_images, height, width, 4 * num_anchors], encoding boxes with respect to
* each anchor. Coordinates are given in the form [dy, dx, dh, dw].
* Must be one of the following types: float16, float32.. \n
* image_info: 2-D tensor with shape of [num_images, 5], containing image information Height, Width, Scale. Must be
one
* of the following types: float16, float32. \n
* anchors: 3-D tensor with shape of [height, width, 4 * num_anchors], describing the anchor boxes. Boxes are
formatted
* in the form [y1, x1, y2, x2]. Must be one of the following types: float16, float32. \n
* nms_threshold: A scalar of type float16 or float32, non-maximal suppression threshold. \n
* pre_nms_topn: A scalar of type int32, number of top scoring boxes to be used as input. \n
* min_size: A scalar of type float32, Any boxes that has a smaller size than min_size will be discarded. \n

* @par Attributes:
* post_nms_topn: An optional int32. Maximum number of rois in the output. Defaults to be 300. \n

* @par Outputs:
* @li rois: 3-D tensor with shape of [num_images, post_nms_topn, 4], padded by 0 if less than 'post_nms_topn'. Must
be
* one of the following types: float16, float32 . \n
* @li rois_probabilities: 2-D tensor with shape of [num_images, post_nms_topn], probability of each roi in 'rois'
* padded by 0 if needed, sorted by scores. Must be one of the following types: float16, float32. \n

* @attention Constraints:+
* Only supports 2864 input boxes at one time.\n

*/
REG_OP(GenerateBoundingBoxProposals)
    .INPUT(scores, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(bbox_deltas, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(image_info, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(anchors, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(nms_threshold, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(pre_nms_topn, TensorType({DT_INT32}))
    .INPUT(min_size, TensorType({DT_FLOAT}))
    .OUTPUT(rois, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(rois_probabilities, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(post_nms_topn, Int, 300)
    .OP_END_FACTORY_REG(GenerateBoundingBoxProposals)

/**
* @brief image to transforms. \n

* @par Inputs:
* @li images: [batch, height, width, channels], 4-D tensor,
* type support uint8, int32, int64, float32, double.
* @li transforms: [batch, 8] or [1, 8] matrix, 2-D tensor, type support float32.
* @li outout_shape: [new_height, new_width], 1-D tensor, type support int32.
* @li fill_value: [scalar], 1-D tensor, type support uint8, int32, int64, float32,
* double, this argus is optional.

* @par Attributes:
* @li interpolation: Interpolation method, type is string,
* support "NEAREST" or "BILINEAR".
* @li fill_mode: Fill mode, defaults is "CONSTANT" also support
* "REFLECT", "WRAP", or "CONSTANT".

* @par Outputs
* transformed_images: 4-D tensor with shape[batch, new_height, new_width,
* channels], as images,support uint8, int32, int64, float32, double\n

* @par Third-party framework compatibility.
* Compatible with tensorflow ImageProjectiveTransformv2 operator.
*/
REG_OP(ImageProjectiveTransformV2)
    .INPUT(images, TensorType({DT_UINT8, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(transforms, TensorType({DT_FLOAT}))
    .INPUT(output_shape, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(fill_value, TensorType({DT_UINT8, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(interpolation, String)
    .ATTR(fill_mode, String, "CONSTANT")
    .OUTPUT(transformed_images, TensorType({DT_UINT8, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(ImageProjectiveTransformV2)

/**
*@brief Calculate the resize_d function. \n

*@par Inputs:
*One input, including:
* x: A 4D tensor, indicating the original size. Must be one of the following types:
* float16, float32. The format must be NCHW. \n

*@par Attributes:
*@li sizes: A required listInt.
* Size of the output parameter shape in the H and W dimensions.
*@li scales: An optional listFloat. Scaling factor of the output data.
* Defaults to none.
*@li roi: An optional listInt. Coordinates of the roi,
* which are normalized in the coordinate system of the input image.
* Defaults to none.
*@li coordinate_transformation_mode: An optional String.
* Alignment mode in Bicubic interpolation mode.
* Defaults to "half_pixel", other optional: align_corners.
*@li cubic_coeff_a: An optional float.
* Indicates the calculation weight coefficient of the interpolation.
* Defaults to -0.75.
*@li exclude_outside: An optional int.
* Whether input parameter excludes out-of-range points during interpolation.
* Defaults to 0.
*@li extrapolation_value: An optional float.
* The fill value used when the interpolation point is out of the data range.
* Defaults to 0.0.
*@li mode: An optional String. ResizeD interpolation mode.
* Defaults to "nearest", other optional: linear or cubic.
* Currently, this parameter cannot be set to nearest.
* When mode is set to linear, the H axis of the operator input tensor must be 1.
*@li nearest_mode: An optional String. When processing non-integer coordinates,
* the nearest neighbor interpolation selects the processing mode of the nearest pixel.
* Defaults to "round_prefer_floor",
* other optional: "round_prefer_ceil", "floor", "ceil".
*@li data_format: An optional String. Format of the input data converted during calculation.
* Defaults to "NCHW", other optional: "HWNC".

*@par Outputs:
*y: A 4D tensor, indicating the target size. Must have the same type and format as x.
* Shape depends on x and sizes.

*@attention Constraints:
*@li The operator will not be enhanced in the future.
*@li For Ascend 950PR&950DT products, replace it with ResizeLinear and ResizeBicubicV2 operators.
*/
REG_OP(ResizeD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(sizes, ListInt)
    .ATTR(scales, ListFloat, {})
    .ATTR(roi, ListInt, {})
    .ATTR(coordinate_transformation_mode, String, "half_pixel")
    .ATTR(cubic_coeff_a, Float, -0.75)
    .ATTR(exclude_outside, Int, 0)
    .ATTR(extrapolation_value, Float, 0.0)
    .ATTR(mode, String, "nearest")
    .ATTR(nearest_mode, String, "round_prefer_floor")
    .ATTR(data_format, String, "NCHW")
    .OP_END_FACTORY_REG(ResizeD)

/**
*@brief Calculate the resize_grad_d function. \n

*@par Inputs:
* One input, including:
* grads: A 4D tensor, indicating the original size. Must be one of the following types:
* float16, float32. The format must be NCHW. \n

*@par Attributes:
*@li original_size: A required listInt. Shape of original_image.
* - When mode is set to cubic, original_size supports four numbers.
* The first two numbers are the same as the first two numbers of the input shape,
* and the last two numbers are the same as the last two numbers of the output shape.
* - When mode is set to linear, the value of original_size can be three numbers.
* The first two numbers are the same as the first two numbers of the input shape,
* and the second number is the same as the last number of the output shape.
*@li roi: An optional listInt.
* Represents the coordinates that are normalized in the coordinate system of the input image.
* 1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X.
* It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize".
* Defaults to none.
*@li scales: An optional listFloat. Array of scaling factors for each dimension.
* It takes value greater than 0. If it's less than 1, it's sampling down, otherwise, it's upsampling.
* Defaults to none.
* - When mode is set to cubic, only one of 'scales' and 'original_size' can be specified.
* If 'original_size' is specified, then set the scales to empty data (zero shape) in the input list of this
operator.
* - When mode is set to linear, the value of scales is: the last dimension of grads divided by the last dimension of
original_size.
*@li coordinate_transformation_mode: An optional String.
* This attribute describes how to transform the coordinate in the resized tensor
* to the coordinate in the original tensor.
* Support "half_pixel" and "align_corners". Defaults to "half_pixel".
*@li cubic_coeff_a: An optional float.
* Indicates the calculation weight coefficient of the interpolation.
* Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch).
* This attribute is valid only if "mode" is "cubic".
* Defaults to -0.75.
*@li exclude_outside: An optional int.
* Whether input parameter excludes out-of-range points during interpolation.
* If set to 1, the weight of sampling locations outside the tensor will be set to 0
* and the weight will be renormalized so that their sum is 1.0.
* Defaults to 0.
*@li extrapolation_value: An optional float.
* The fill value used when the interpolation point is out of the data range.
* When coordinate_transformation_mode is "tf_crop_and_resize" and x_original is outside
* the range [0, length_original - 1], this value is used as the corresponding output value.
* Defaults to 0.0.
*@li mode: An optional String. ResizeGradD interpolation mode.
* Support "nearest", "linear" and "cubic". Defaults to "nearest".
* Currently, this parameter cannot be set to nearest.
* When mode is set to linear, the H axis of the operator input tensor must be 1.
*@li nearest_mode: An optional String. When processing non-integer coordinates,
* the nearest neighbor interpolation selects the processing mode of the nearest pixel.
* Only used by nearest interpolation.
* Support "round_prefer_floor" and "round_prefer_ceil". Defaults to "round_prefer_floor".
*@li data_format: An optional String. Format of the input data converted during calculation.
* Support "NCHW" and "HWNC". Defaults to "NCHW".

*@par Outputs:
* y: A 4D tensor, indicating the target size. Must have the same type and format as grads.
* Shape depends on grads and original_size. \n
*/
REG_OP(ResizeGradD)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(original_size, ListInt)
    .ATTR(roi, ListInt, {})
    .ATTR(scales, ListFloat, {})
    .ATTR(coordinate_transformation_mode, String, "half_pixel")
    .ATTR(cubic_coeff_a, Float, -0.75)
    .ATTR(exclude_outside, Int, 0)
    .ATTR(extrapolation_value, Float, 0.0)
    .ATTR(mode, String, "nearest")
    .ATTR(nearest_mode, String, "round_prefer_floor")
    .ATTR(data_format, String, "NCHW")
    .OP_END_FACTORY_REG(ResizeGradD)

/**
* @brief Resize images to size using trilinear interpolation . \n

* @par Inputs:
* Input images must be a 5-D tensor. Inputs include:
* @li x: A 5-D tensor, type must be float16, float32 or double. Must set the format, supported format list
is ["NCDHW, NDHWC"] .
* @li size: A 1-D tensor, type must be int32. Contains 3 elements: new_depth, new_height, new_width.
Set the shape of output y . \n

* @par Attributes:
* @li align_corners: If true, the centers of the 8 corner pixels of the input and
output tensors are aligned, preserving the values at the corner pixels.
Defaults to false .
* @li half_pixel_centers: An optional bool. Defaults to false . \n
* @par Outputs:
* y: 5-D with shape [batch, channels, new_depth, new_height, new_width] . \n

* @par Third-party framework compatibility
* Compatible with onnx Resize operator using trilinear interpolation.
*/
REG_OP(ResizeTrilinear)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .OP_END_FACTORY_REG(ResizeTrilinear)

/**
*@brief Obtains the ROI feature matrix from the feature map list. It is a customized fused operator for mmdetection.
\n

*@par Inputs:
* Two inputs, including:
*@li features: A 5HD Tensor list of type float32 or float16.
*@li rois: ROI position. A 2D Tensor of float32 or float16 with shape (N, 5). "N" indicates the number of ROIs,
* the value "5" indicates the indexes of images where the ROIs are located, "x0", "y0", "x1", and "y1".
*@li index: Optional input, provided by the operator inserted before fusion-pass under specific models.

*@par Attributes:
*@li finest_scale: A optional attribute of type int, specifying the scale of calculate levels of "rois".
*@li roi_scale_factor: A optional attribute of type float32, specifying the rescaling of "rois" coordinates.
*@li spatial_scale: A optional attribute of type list float32, specifying the scaling ratio of "features"
* to the original image.
*@li pooled_height: A optional attribute of type int32, specifying the H dimension.
*@li pooled_width: A optional attribute of type int32, specifying the W dimension.
*@li sample_num: An optional attribute of type int32, specifying the horizontal and vertical sampling frequency
* of each output. If this attribute is set to "0", the sampling frequency is equal to the rounded up value of
"rois",
* which is a floating point number. Defaults to "0".
*@li pool_mode: An optional attribute of type string to indicate pooling mode. Defaults to "avg", only supports
"avg". \n
*@li aligned: An optional attribute of type bool, specifying the align to corner. Defaults to true. \n

*@par Outputs:
* output: Outputs the feature sample of each ROI position. The format is 5HD Tensor of type float32 or float16.
* The axis N is the number of input ROIs. Axes H, W, and C are consistent with the values of "pooled_height",
* "pooled_width", and "features", respectively.

*@par Third-party framework compatibility
*Compatible with mmdetection SingleRoIExtractor operator.
*/
REG_OP(RoiExtractor)
    .DYNAMIC_INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(finest_scale, Int, 56)
    .ATTR(roi_scale_factor, Float, 0)
    .ATTR(spatial_scale, ListFloat, {1.f / 4, 1.f / 8, 1.f / 16, 1.f / 32})
    .ATTR(pooled_height, Int, 7)
    .ATTR(pooled_width, Int, 7)
    .ATTR(sample_num, Int, 0)
    .ATTR(pool_mode, String, "avg")
    .ATTR(aligned, Bool, true)
    .OP_END_FACTORY_REG(RoiExtractor)

/**
*@brief Computes Fastrcnn RpnProposals function . \n

*@par Inputs:
*Inputs include:
* @li rois: A Tensor. Must be float16. N-D with shape [N, 4].
* @li cls_bg_prob: A Tensor. Must be float16. N-D with shape [N, 1].
* @li img_size: A Tensor. Must be int32. shape [H, W] . \n

*@par Attributes:
* @li score_threshold: required, float, threahold of topk process.
* @li k: required, Int, threahold of topk process.
* @li min_size: required, float, threahold of nms process.
* @li nms_threshold: required, float, threahold of nms process.
* @li post_nms_num: required, float, threahold of nms process.
* @li score_filter: bool, mark of score_filter. Defaults to "true"
* @li box_filter: bool, mark of box_filter. Defaults to "true"
* @li score_sigmoid: bool, mark of score_sigmoid. Defaults to "false"

*@par Outputs:
* @li sorted_box: A Tensor. Must be float16. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Unpack.
*/
REG_OP(RpnProposals)
    .INPUT(rois, TensorType({DT_FLOAT16}))
    .INPUT(cls_bg_prob, TensorType({DT_FLOAT16}))
    .INPUT(img_size, TensorType({DT_INT32}))
    .REQUIRED_ATTR(score_threshold, Float)
    .REQUIRED_ATTR(k, Int)
    .REQUIRED_ATTR(min_size, Float)
    .REQUIRED_ATTR(nms_threshold, Float)
    .REQUIRED_ATTR(post_nms_num, Int)
    .ATTR(score_filter, Bool, true)
    .ATTR(box_filter, Bool, true)
    .ATTR(score_sigmoid, Bool, false)
    .OUTPUT(sorted_box, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(RpnProposals)

/**
*@brief HDRNet and ISP direct data conversion
returned tensor's dimension will correspond to input dimension [0, 3, 4, 2, 1],
convert tensor dtype float16 to int16 . \n

*@par Inputs:
*one inputs, including:
*@li x: A Tensor. Must be one of the following types: float16.

*@par Outputs:
*y: A Tensor. Must be one of the following types: int16. \n

*@par Third-party framework compatibility
*only for use by corresponding operators in HDRnet networks
*/
#ifndef OPS_PROTO_DEF_TRANSARGB
#define OPS_PROTO_DEF_TRANSARGB
REG_OP(TransArgb)
    .INPUT(x, "T1")
    .OUTPUT(y, "T2")
    .DATATYPE(T1, TensorType({DT_FLOAT16}))
    .DATATYPE(T2, TensorType({DT_INT16}))
    .OP_END_FACTORY_REG(TransArgb)
#endif

/**
*@brief Normalizes data. It is called Region on YOLO v2 and Yolo on YOLO v3 . \n

*@par Inputs:
*x: An NCHW tensor of type float16. The data is with shape (N, boxes*(coords+obj+classes), H, W),
where, "obj" indicates the confidence of an object, and only one confidence is supported. Boxes are arranged
as xx...xyy...yww...whh...hbb...bc0c0..c0c1c1...c1......cncn...cn . \n

*@par Attributes:
*@li boxes: A required int32, specifying the number of anchor boxes. Defaults to "5" for V2 or "3" for V3.
*@li coords: An int32, specifying the number of parameters required for locating an object. The value is fixed at
"4", corresponding to (x,y,w,h).
*@li classes: An int32, specifying the number of prediction classes. Defaults to "80". The value range is [1, 1024].
*@li yolo_version: A string, specifying the YOLO version, either "V2" or "V3".Defaults to "V3"
*@li softmax: A bool, specifying whether to perform softmax, valid only when "yolo_version = V2". Defaults to
"false".
*@li background: A bool, specifying the operation types of the obj and classes, used in conjunction with "softmax"
and valid only when "yolo_version = V2". Defaults to "false".
*@li softmaxtree: A bool, Fixed to False, defined in Lite, but not used. Defaults to "false" . \n

*@par Outputs:
*@li coord_data: A float16 with shape [N, boxes*coords, ceilx(height*width*2+32, 32)/2],
* where "ceil" indicates that a detected box is aligned upwards with the second parameter. Specifies the coordinates
of a detected box.
*@li obj_prob: A float16 with shape [N, ceilx(boxes*height*width *2+32, 32)/2],
* where "ceil" indicates that a detected box is aligned upwards with the second parameter. Specifies the confidence.
*@li classes_prob: A float16 with shape [N, classes, ceilx(boxes*height*width *2+32, 32)/2],
* where "ceil" indicates that a detected box is aligned upwards with the second parameter. Specifies the prediction
classes . \n

*@attention Constraints:
*@li This operator applies to YOLO v2,v3 and v5 networks.
*@li The succeeding layer of the Yolo operator must be operator Yolov5DetectionOutput.
*@par Third-party framework compatibility
* It is a custom operator.
*/
REG_OP(YoloPreDetection)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(coord_data, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(obj_prob, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(classes_prob, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(boxes, Int, 3)
    .ATTR(coords, Int, 4)
    .ATTR(classes, Int, 80)
    .ATTR(yolo_version, String, "V5")
    .ATTR(softmax, Bool, false)
    .ATTR(background, Bool, false)
    .ATTR(softmaxtree, Bool, false)
    .OP_END_FACTORY_REG(YoloPreDetection)

/**
*@brief Performs YOLO V5 detection . \n

*@par Inputs:
*@li x: Dynamic inputs, the outputs of operator Yolo at the preceding layer.
*@li img_info: A float16 or float32, describing the image information . \n

*@par Attributes:
*@li biases: A required float. "biases = Number of Yolo operators at the preceding layer x 2 x boxes"
*@li boxes: A required int32, specifying the number of anchor boxes predicted for each Yolo layer.
*@li coords: Specifies the number of coordinate parameters. Must be 4.
*@li classes: A required int32, specifying the number of classes to be predicted.
*@li relative: An optional bool. Defaults to and must be "true".
*@li obj_threshold: A required float, specifying the confidence threshold for box filtering.
*@li post_nms_topn: An optional int32. This attribute is reserved.
*@li score_threshold: A required float, specifying the class score threshold for box filtering.
*@li iou_threshold: A required float, specifying the intersection-over-union (IOU) threshold for box filtering.\n
*@li pre_nms_topn: An optional int, specifying the number of boxes for non-maximum suppression (NMS). Defaults to "512".
*@li N: An optional int. Defaults to "10".
*@li resize_origin_img_to_net: An optional bool. Defaults to "false".
*@li out_box_dim: An optional int. Defaults to "3".
*@li alpha: An optional float. Defaults to "2.0".

*@par Outputs:
*@li box_out: A tensor of type float16 or float32, describing the information of each output box.
*@li box_out_num: A tensor of type int32, specifying the number of output boxes . \n
*/
#ifndef OPS_PROTO_DEF_YOLOV5DETECTIONOUTPUT
#define OPS_PROTO_DEF_YOLOV5DETECTIONOUTPUT
REG_OP(YoloV5DetectionOutput)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(biases, ListFloat)
    .ATTR(boxes, Int, 3)
    .ATTR(coords, Int, 4)
    .ATTR(classes, Int, 80)
    .ATTR(relative, Bool, true)
    .ATTR(obj_threshold, Float, 0.5)
    .ATTR(post_nms_topn, Int, 512)
    .ATTR(score_threshold, Float, 0.5)
    .ATTR(iou_threshold, Float, 0.45f)
    .ATTR(pre_nms_topn, Int, 512)
    .ATTR(N, Int, 10)
    .ATTR(resize_origin_img_to_net, Bool, false)
    .ATTR(out_box_dim, Int, 3)
    .ATTR(alpha, Float, 2.0)
    .OUTPUT(box_out, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(box_out_num, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(YoloV5DetectionOutput)
#endif

/**
*@brief Performs YOLO V3 detection output V2 . \n

*@par Inputs:
*@li x: Dynamic inputs, the outputs of operator Yolo at the preceding layer . \n

*@par Attributes:
*@li biases: A float list. "biases = Number of Yolo operators at the preceding layer x 2 x boxes"
*@li boxes: A required int, specifying the number of anchor boxes predicted for each Yolo layer.
*@li coords: Specifies the number of coordinate parameters. Must be 4.
*@li classes: A required int, specifying the number of classes to be predicted.
*@li relative: An optional bool. Defaults to and must be "true".
*@li obj_threshold: A required float, specifying the confidence threshold for box filtering.
*@li post_nms_topn: An optional int. This attribute is reserved.
*@li score_threshold: A required float, specifying the class score threshold for box filtering.
*@li iou_threshold: A required float, specifying the intersection-over-union (IOU) threshold for box filtering.
*@li pre_nms_topn: An optional int, specifying the number of boxes for non-maximum suppression (NMS). Defaults to "512".
*@li N: An optional int. Defaults to "10".
*@li resize_origin_img_to_net: An optional bool. Defaults to "false".
*@li out_box_dim: An optional int. Defaults to "3".

*@par Outputs:
*@li box_out: A tensor of type float16 or float32, describing the information of each output box.
*@li box_out_num: A tensor of type int32, specifying the number of output boxes . \n
*/
#ifndef OPS_PROTO_DEF_YOLOV3DETECTIONOUTPUTV2
#define OPS_PROTO_DEF_YOLOV3DETECTIONOUTPUTV2
REG_OP(YoloV3DetectionOutputV2)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(biases, ListFloat)
    .ATTR(boxes, Int, 3)
    .ATTR(coords, Int, 4)
    .ATTR(classes, Int, 80)
    .ATTR(relative, Bool, true)
    .ATTR(obj_threshold, Float, 0.5)
    .ATTR(post_nms_topn, Int, 512)
    .ATTR(score_threshold, Float, 0.5)
    .ATTR(iou_threshold, Float, 0.45f)
    .ATTR(pre_nms_topn, Int, 512)
    .ATTR(N, Int, 10)
    .ATTR(resize_origin_img_to_net, Bool, false)
    .ATTR(out_box_dim, Int, 3)
    .OUTPUT(box_out, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(box_out_num, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(YoloV3DetectionOutputV2)
#endif

/**
*@brief Generates bounding boxes based on "anchor_boxes" and "gt_bboxes" . \n

*@par Inputs:
*@li anchor_boxes: anchor sample boxes of origin image
*@li gt_bboxes: ground truth boxes
*@li stride: stride . \n

*@par Attributes:
*performance_mode: select performance mode, "high_precision" or "high_performance".

*@par Outputs:
*encoded_bboxes: Bboxes generated based on "anchor_boxes" and "gt_bboxes". Have the
* same format and type as "anchor_boxes".
*/
#ifndef OPS_PROTO_DEF_YOLOBOXESENCODE
#define OPS_PROTO_DEF_YOLOBOXESENCODE
REG_OP(YoloBoxesEncode)
    .INPUT(anchor_boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gt_bboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(stride, TensorType({DT_INT32}))
    .ATTR(performance_mode, String, "high_precision")
    .OUTPUT(encoded_bboxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(YoloBoxesEncode)
#endif

/**
*@brief Performs Position Sensitive PS ROI Pooling Grad. \n

 *@par Inputs:
 *@li assigned_gt_inds: A tensor of type float16 or float32, shape (n, ).
 *  Indicates the assigned ground truth index for each predicted box.
 *@li overlaps: A tensor. Datatype is same as assigned_gt_inds, IOU(Intersection over Union, between[0,1]) between
gt_bboxes and bboxes, shape (k, n).
 *  Where k is the number of ground truths, n is the number of predicted boxes.
 *@li box_responsible_flags: A tensor. Support uint8, shape (n, ).
 *  Flag to indicate whether the box is responsible for being assigned to a ground truth.
 *@li max_overlaps: A tensor. Datatype is same as assigned_gt_inds, overlaps.max(axis=0), shape (n, ).
 *  Maximum IOU for each predicted box across all ground truths.
 *@li argmax_overlaps: A tensor. Support int32, overlaps.argmax(axis=0), shape (n, ).
 *  Index of the ground truth with maximum IOU for each predicted box.
 *@li gt_max_overlaps: A tensor. Datatype is same as assigned_gt_inds, overlaps.max(axis=1), shape (k, ).
 *  Maximum IOU for each ground truth across all predicted boxes.
 *@li gt_argmax_overlaps: A tensor. Support int32, overlaps.argmax(axis=1), shape (k, ).
 *  Index of the predicted box with maximum IOU for each ground truth.
 *@li num_gts: A tensor. Support int32, shape (1, ).
 *  Real number of ground truths \n

 *@par Attributes:
 *@li pos_iou_thr: float. IOU threshold for positive bboxes. Bboxes with IOU >= pos_iou_thr are considered positive.
 *@li min_pos_iou: float. Minimum IOU for a bbox to be considered as a positive bbox.
 *@li gt_max_assign_all: bool. Whether to assign all bboxes with the same highest overlap with some gt to that gt. \n

 *@par Outputs:
 * assigned_gt_inds_pos: A tensor. Support float16/float32, shape (n, ).
 *   The final assigned ground truth indices for positive bboxes, used for gradient computation. \n
 */
#ifndef OPS_PROTO_DEF_GRIDASSIGNPOSITIVE
#define OPS_PROTO_DEF_GRIDASSIGNPOSITIVE
REG_OP(GridAssignPositive)
    .INPUT(assigned_gt_inds, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(overlaps, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(box_responsible_flags, TensorType({DT_UINT8}))
    .INPUT(max_overlaps, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(argmax_overlaps, TensorType({DT_INT32}))
    .INPUT(gt_max_overlaps, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gt_argmax_overlaps, TensorType({DT_INT32}))
    .INPUT(num_gts, TensorType({DT_INT32}))
    .OUTPUT(assigned_gt_inds_pos, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(pos_iou_thr, Float)
    .REQUIRED_ATTR(min_pos_iou, Float)
    .REQUIRED_ATTR(gt_max_assign_all, Bool)
    .OP_END_FACTORY_REG(GridAssignPositive)
#endif

/**
* @brief Performs non-maximum suppression (NMS) on the rotated boxes according
* to their intersection-over-union (IoU). Rotated NMS interatively removes lower
* scoring rotated boxes which have an IoU greater than iou_threshold with
* another (higher scoring) rotated box.

* @par Inputs:
* Three inputs, including:
* @li boxes: A 2D Tensor of float16 or float32 with shape (N, 5). Rotated boxes to
* perform NMS on. They are expected to be in (x1, y1, x2, y2, angle_degress) format.
* @li scores: A 1D Tensor of float16 or float32 with shape (N). Scores for each one of
* the rotated boxes.
* @li labels: A 1D Tensor of int32 or int64 with shape (N). Labels for each one of
* the rotated boxes.

* @par Attributes:
* iou_threshold: A required float attribute. Discards all overlapping rotated
* boxes with IoU < iou_threshold.

* @par Outputs:
* Two outputs, including:
* @li selected_detections: A 2D Tensor of float16 or float32 with shape (N, 5).
* The selected boxes that kept by Rotated NMS, sorted in decreasing order of scores.
* @li keep_indices: A 1D Tensor of int32 or int64 with shape (N). The indices of
* selected_detections.

* @attention Constraints:
* Currently, the tensor type of input (boxes, scores) only support float.
* The tensor type of keep_indices only support int32.
*/
REG_OP(RotatedNMS)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(selected_detections, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(keep_indices, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(iou_threshold, Float)
    .ATTR(is_angle, Bool, true)
    .OP_END_FACTORY_REG(RotatedNMS)
} // namespace ge

#endif
