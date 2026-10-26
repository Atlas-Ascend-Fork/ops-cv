/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 * The code snippet comes from Huawei's open-source Mindspore project.
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef NON_MAX_SUPPRESSION_V3_PROTO_H
#define NON_MAX_SUPPRESSION_V3_PROTO_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Greedily selects a subset of bounding boxes in descending order of
* score . \n

* @par Inputs:
* Input boxes and  scores must be float type. Inputs include:
* @li boxes: A 2-D float tensor of shape [num_boxes, 4]. They are expected to be in (x1, y1, x2, y2)
* format with x1 < x2 and y1 < y2.
* @li scores: A 1-D float tensor of shape [num_boxes] representing a single
* score corresponding to each box (each row of boxes).
* @li max_output_size: A scalar integer tensor representing the maximum number
* of boxes to be selected by non max suppression.
* @li iou_threshold: A 0-D float tensor representing the threshold for deciding
* whether boxes overlap too much with respect to IOU.
* @li score_threshold: A 0-D float tensor representing the threshold for
* deciding when to remove boxes based on score . \n

* @par Attributes:
* offset: An optional int. Defaults to 0. \n

* @par Outputs:
* selected_indices: A 1-D integer tensor of shape [M] representing the selected
* indices from the boxes tensor, where M <= max_output_size . \n

* @attention Constraints:
* Input boxes and scores must have the same data type, either float16 or float . \n

* @par Third-party framework compatibility
* Compatible with tensorflow NonMaxSuppressionV3 operator.
*/

REG_OP(NonMaxSuppressionV3)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .ATTR(offset, Int, 0)
    .OP_END_FACTORY_REG(NonMaxSuppressionV3)
} // namespace ge

#endif
