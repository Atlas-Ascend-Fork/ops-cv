/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_UPSAMPLE_NEAREST_EXACT2D_H_
#define OP_API_INC_UPSAMPLE_NEAREST_EXACT2D_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleNearestExact2d的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：对由四个输入通道组成的输入信号应用最近邻精确插值算法进行上采样插值
 * 计算公式：
 * out(N, C, H, W) = self(N, C, min(floor((H + 0.5) ∗ scalesH),H−1), min(floor((W + 0.5) ∗ scalesW),W−1))
 * @domain aclnn_ops_train aclnn_ops_infer
 * 参数描述：
 * @param [in]   self
 * 输入Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16，UINT8。支持非连续Tensor，数据格式支持ND、NCHW。
 * UINT8仅Atlas A2/A3系列产品支持，Atlas推理系列和Ascend 950PR&950DT系列产品不支持。
 * @param [in]   outputSize
 * 输出的size大小，数据类型支持INT32、INT64。
 * @param [in]   scaleH
 * 输出的H方向缩放系数，数据类型支持DOUBLE。
 * @param [in]   scaleW
 * 输出的W方向缩放系数，数据类型支持DOUBLE。
 * @param [in]   out
 * 输出Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16，UINT8。支持非连续Tensor，数据格式支持ND、NCHW。
 * UINT8仅Atlas A2/A3系列产品支持，Atlas推理系列和Ascend 950PR&950DT系列产品不支持。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnUpsampleNearestExact2dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize,
                                                                  double scalesH, double scalesW, aclTensor* out,
                                                                  uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnUpsampleNearestExact2d的第二段接口，用于执行计算。
 *
 * 功能描述：对由四个输入通道组成的输入信号应用最近邻精确插值算法进行上采样插值。
 *
 * @domain aclnn_ops_train aclnn_ops_infer
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnUpsampleNearestExact2dGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUpsampleNearestExact2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_UPSAMPLE_NEAREST_EXACT1D_H_
