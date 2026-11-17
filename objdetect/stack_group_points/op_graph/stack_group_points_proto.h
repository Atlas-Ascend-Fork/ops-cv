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
 * \file stack_group_points_proto.h
 * \brief
 */
#ifndef STACK_GROUP_POINTS_PROTO_H_
#define STACK_GROUP_POINTS_PROTO_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Group the points in the point cloud according to the group they belong to. \n

* @par Inputs:
* Four inputs, including:
* @li features:  Tensor of features to group, input shape is (N1 + N2 ..., C).
* @li features_batch_cnt:  Input features nums in each batch, just like (N1, N2, ...).
* @li indices: The indices of features to group with, input shape is (M1 + M2 ..., nsample).
* @li indices_batch_cnt: Input indices nums in each batch, just like (M1, M2, ...). \n

* @par Outputs:
* One outputs: Grouped features, the shape is (M1 + M2 ..., C, nsample).

* @par Third-party framework compatibility
* Compatible with the MMCV operator GroupPoints(StackGroupPoints branch).
*/
#ifndef OPS_PROTO_DEF_STACKGROUPPOINTS
#define OPS_PROTO_DEF_STACKGROUPPOINTS
REG_OP(StackGroupPoints)
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(features_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(indices_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(StackGroupPoints)
#endif
} // namespace ge
#endif // STACK_GROUP_POINTS_PROTO_H_
