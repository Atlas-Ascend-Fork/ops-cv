/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_RESIZE_H_
#define OP_API_INC_RESIZE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnResize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：调整输入张量的大小
 * @param [in]   self 输入Tensor，数据类型支持FLOAT,  FLOAT16。支持非连续Tensor，数据格式支持NCHW.
 * @param [in]   scales 输入aclFloatArray，数据类型支持 FLOAT.
 * @param [in]   mode 输入属性，数据类型支持const char*
 * @param [out]  workspaceSize 返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnResizeGetWorkspaceSize(const aclTensor* self, const aclFloatArray* scales, const char* mode,
                                                  aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnResize的第二段接口，用于执行计算。
 * @domain aclnn_ops_infer
 *
 * 算子功能：调整输入张量的大小
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnResizeGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnResize(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_RESIZE_H_
