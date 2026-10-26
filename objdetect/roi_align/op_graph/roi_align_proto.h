/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_align_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ROI_ALIGN_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ROI_ALIGN_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Obtains the ROI feature matrix from the feature map. It is a customized FasterRcnn operator . \n

* @par Inputs:
* Three inputs, including:
* @li features: A 5HD Tensor of type float32 or float16.
* @li rois: ROI position. A 2D Tensor with the same type as "features" (float32 or float16) and shape (N, 5).
* "N" indicates the number of ROIs, and "5" indicates the indexes of images where the ROIs are located,
* "x0", "y0", "x1", and "y1".
* @li rois_n: Optional INT32 input with shape (N), where N equals the number of ROIs.
* Each element is the corresponding ROI batch index and overrides the index in `rois`. \n

* @par Attributes:
* @li spatial_scale: A required attribute of type float32, specifying the scaling ratio of "features" to the original
image.
* @li pooled_height: A required attribute of type int32, specifying the H dimension.
* @li pooled_width: A required attribute of type int32, specifying the W dimension.
* @li sample_num: An optional attribute of type int32, specifying the horizontal and vertical sampling frequency of each
output. If this attribute is set to "0",
* the sampling frequency is equal to the rounded up value of "rois", which is a floating point number. Defaults to "2".
* @li roi_end_mode: An optional attribute of type int32, specifying the align mode. Defaults to "1", supports 0/1/2/3.
\n
* "0" is compatible with align = False for all frameworks. \n
* "1" is compatible with align = True for TensorFlow. \n
* "2" is compatible with align = True for pyTorch. \n
* "3" is compatible with align = True for MmDetecion v0.6. \n
* @li pool_mode: An optional attribute of type string, specifying the pooling mode. Defaults to "avg", supports "avg"
and "max". \n

* @par Outputs:
* y: Outputs the feature sample of each ROI position. The format is 5HD Tensor of type float32 or float16.
* The axis N is the number of input ROIs. Axes H, W, and C are consistent
* with the values of "pooled_height",
* "pooled_width", and "features", respectively.
*/
REG_OP(ROIAlign)
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(rois_n, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(spatial_scale, Float)
    .REQUIRED_ATTR(pooled_height, Int)
    .REQUIRED_ATTR(pooled_width, Int)
    .ATTR(sample_num, Int, 2)
    .ATTR(roi_end_mode, Int, 1)
    .ATTR(pool_mode, String, "avg")
    .OP_END_FACTORY_REG(ROIAlign)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_ROI_ALIGN_OPS_H_
