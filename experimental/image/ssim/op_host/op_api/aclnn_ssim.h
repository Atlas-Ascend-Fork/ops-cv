/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_SSIM_H_
#define OP_API_INC_LEVEL2_SSIM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSsim的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：计算两幅图像的结构相似度指标MSSIM。
 * 计算公式：
 *     mu1 = gauss(x1); mu2 = gauss(x2)
 *     sigma1_sq = gauss(x1*x1) - mu1^2
 *     sigma2_sq = gauss(x2*x2) - mu2^2
 *     sigma12   = gauss(x1*x2) - mu1*mu2
 *     C1 = (0.01*L)^2, C2 = (0.03*L)^2, L 按 dtype 推断（uint8=255, float32=1.0）
 *     ssim_map = ((2*mu1*mu2 + C1)/(mu1^2 + mu2^2 + C1)) * ((2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
 *     yOut = mean(ssim_map)  # 对裁剪边带 (winSize-1)/2 = 5 后的内部区域取均值
 *
 * @param [in] x1: 输入参考图像，数据类型支持 uint8、float32，shape 为 [..., H, W] 或 [..., H, W, C]，H/W >= 11。
 * @param [in] x2: 输入待评估图像，shape 与 dtype 必须与 x1 一致。
 * @param [in] yOut: 输出 MSSIM 标量，dtype float32，shape [1]。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSsimGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, aclTensor* yOut,
                                                uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnSsim的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起始地址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSsim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_SSIM_H_
