/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_CV_OBJDETECT_SORTED_NMS_PROTO_H_
#define OPS_CV_OBJDETECT_SORTED_NMS_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Greedily selects non-suppressed boxes in the supplied descending score order.
 *
 * @par Inputs:
 * @li boxes: A 2-D tensor of shape [num_boxes, 4].
 * @li sorted_scores: A 1-D tensor of shape [num_boxes], sorted in descending order.
 * @li input_indices: A 1-D int32 tensor of shape [num_boxes], with values in [0, num_boxes).
 * @li max_output_size: A scalar (rank-0) or 1-D [1] int32 tensor giving the maximum number of selected boxes.
 * @li iou_threshold: A scalar (rank-0) or 1-D [1] float tensor in [0, 1].
 * @li score_threshold: A scalar (rank-0) or 1-D [1] float tensor.
 *
 * boxes and iou_threshold must have the same type (float16 or float32). sorted_scores and
 * score_threshold must have the same type (float16 or float32), independently of boxes.
 *
 * @par Outputs:
 * selected_indices: A 1-D int32 tensor of shape [M], where M is computed at runtime and
 * M <= min(max_output_size, num_boxes).
 */
REG_OP(SortedNMS)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(sorted_scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_indices, TensorType({DT_INT32}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .ATTR(offset, Int, 0)
    .OP_END_FACTORY_REG(SortedNMS)
} // namespace ge

#endif // OPS_CV_OBJDETECT_SORTED_NMS_PROTO_H_
