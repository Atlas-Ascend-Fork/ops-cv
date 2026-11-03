/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See License in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GAUSSIAN_FILTER_3D_H_
#define OP_API_INC_LEVEL2_ACLNN_GAUSSIAN_FILTER_3D_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGaussianFilter3D的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：三维高斯滤波（可分离实现，W→H→D三遍一维卷积）
 * 计算公式：
 * $$ y = G_{\sigma} * x $$
 * 语义对齐 scipy.ndimage.gaussian_filter（truncate 固定 4.0，核半径 r = int(4*sigma+0.5)）。
 *
 * @param [in] x: 待进行高斯滤波的入参，5D/4D/3D体数据（[N,C,D,H,W]/[N,D,H,W]/[D,H,W]）。
 * npu device侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，且数据类型必须和y一样，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] sigma: 高斯核标准差，取值范围[0, 100]，等于0时直通输出（y=x）。
 * @param [in] mode: 边界填充模式，取值为 "REFLECT"、"CONSTANT"、"NEAREST"、"MIRROR"、"WRAP"，
 * 语义对齐 scipy.ndimage，三轴共用。
 * @param [in] y: 高斯滤波的出参，与x完全同shape同dtype，npu device侧的aclTensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGaussianFilter3DGetWorkspaceSize(const aclTensor* x, double sigma, const char* mode,
                                                            const aclTensor* y, uint64_t* workspaceSize,
                                                            aclOpExecutor** executor);

/**
 * @brief aclnnGaussianFilter3D的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGaussianFilter3D(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_GAUSSIAN_FILTER_3D_H_
