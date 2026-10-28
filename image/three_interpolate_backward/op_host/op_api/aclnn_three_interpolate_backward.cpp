/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_three_interpolate_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "three_interpolate_backward.h"
#include "aclnn_kernels/reshape.h"
#include "level0/fill.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr uint8_t DIM_INDEX_B = 0;
constexpr uint8_t DIM_INDEX_C = 1;
constexpr uint8_t DIM_INDEX_N = 2;
constexpr uint8_t DIM_INDEX_M = 2;
constexpr uint8_t NEW_SHAPE_DIMS = 4;
constexpr uint8_t CONST_1 = 1;
constexpr uint8_t CONST_2 = 2;
constexpr uint8_t CONST_3 = 3;
} // namespace

static bool CheckNotNull(const aclTensor* grad_x, const aclTensor* idx, const aclTensor* weight,
                         const aclTensor* grad_y)
{
    OP_CHECK_NULL(grad_x, return false);
    OP_CHECK_NULL(idx, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(grad_y, return false);
    return true;
}

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                       op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> IDX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                           op::DataType::DT_INT64};

static bool CheckDtypeValid(const aclTensor* grad_x, const aclTensor* idx, const aclTensor* weight)
{
    // 检查gradOutput和self数据类型是否在ThresholdBackward算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(grad_x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(idx, IDX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, DTYPE_SUPPORT_LIST, return false);
    return true;
}

// 检查 grad_x/idx/weight/grad_y 之间的形状一致性（issue #1407）。
// 契约：grad_x 与 grad_y 同为 4D NCHW (b,c,n,1)/(b,c,m,1) 或同为 3D ND (b,c,n)/(b,c,m)；
// idx/weight 均为 (b,n,3)，其中 b 与 grad_x 的 b 维一致、n 与 grad_x 的 n 维一致；
// grad_y 的 b、c 维与 grad_x 一致且 m 维等于属性 m。
// 形状不一致时 kernel 会按自身读取到的维度静默截断计算并返回错误梯度，必须在 host 侧拦截。
static bool CheckShapeConsistent(const aclTensor* grad_x, const aclTensor* idx, const aclTensor* weight,
                                 const aclTensor* grad_y, int m)
{
    const auto& gradXShape = grad_x->GetViewShape();
    const auto& idxShape = idx->GetViewShape();
    const auto& weightShape = weight->GetViewShape();
    const auto& gradYShape = grad_y->GetViewShape();

    auto gradXDimNum = gradXShape.GetDimNum();
    auto gradYDimNum = gradYShape.GetDimNum();
    // grad_x 仅支持 4D NCHW (b,c,n,1) 或 3D ND (b,c,n)，且 grad_y rank 与之相同、4D 时末维为 1
    if (gradXDimNum == NEW_SHAPE_DIMS) {
        if (gradYDimNum != NEW_SHAPE_DIMS || gradXShape.GetDim(NEW_SHAPE_DIMS - CONST_1) != CONST_1 ||
            gradYShape.GetDim(NEW_SHAPE_DIMS - CONST_1) != CONST_1) {
            return false;
        }
    } else if (gradXDimNum == CONST_3) {
        if (gradYDimNum != CONST_3) {
            return false;
        }
    } else {
        return false;
    }

    auto dimB = gradXShape.GetDim(DIM_INDEX_B);
    auto dimC = gradXShape.GetDim(DIM_INDEX_C);
    auto dimN = gradXShape.GetDim(DIM_INDEX_N);

    // idx/weight 必须均为 (b, n, 3) 且互相一致
    if (idxShape.GetDimNum() != CONST_3 || weightShape.GetDimNum() != CONST_3) {
        return false;
    }
    if (idxShape.GetDim(DIM_INDEX_B) != weightShape.GetDim(DIM_INDEX_B) ||
        idxShape.GetDim(DIM_INDEX_C) != weightShape.GetDim(DIM_INDEX_C) ||
        idxShape.GetDim(DIM_INDEX_M) != weightShape.GetDim(DIM_INDEX_M)) {
        return false;
    }
    // idx 的 b 维与 grad_x 的 b 维一致，n 维与 grad_x 的 n 维一致，末维为 3
    if (idxShape.GetDim(DIM_INDEX_B) != dimB || idxShape.GetDim(DIM_INDEX_C) != dimN ||
        idxShape.GetDim(DIM_INDEX_M) != CONST_3) {
        return false;
    }
    // grad_y 的 b、c 维与 grad_x 一致，m 维等于属性 m
    if (gradYShape.GetDim(DIM_INDEX_B) != dimB || gradYShape.GetDim(DIM_INDEX_C) != dimC ||
        gradYShape.GetDim(DIM_INDEX_M) != static_cast<int64_t>(m)) {
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* grad_x, const aclTensor* idx, const aclTensor* weight,
                               const aclTensor* grad_y, int m)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(grad_x, idx, weight, grad_y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型是否正确
    CHECK_RET(CheckDtypeValid(grad_x, idx, weight), ACLNN_ERR_PARAM_INVALID);

    // 3. 属性 m 必须大于 0。910B/910_93 不在此处拦截时，m=0 会进入 TransData 并以内部错误返回。
    if (m <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "attr m must be > 0.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 4. 检查 grad_x/idx/weight/grad_y 之间的形状一致性（b/c/n/m 维）
    if (!CheckShapeConsistent(grad_x, idx, weight, grad_y, m)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "shapes of grad_x/idx/weight/grad_y must be consistent: grad_x and grad_y should both be "
                "(b,c,n,1)/(b,c,m,1) NCHW or (b,c,n)/(b,c,m) ND, idx and weight should be (b,n,3) with "
                "b/n matching grad_x, and grad_y's m dim must equal attr m.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

// n==0 时 kernel 会把非空 grad_y 清零。aclnn 早退不写输出会把未初始化内存当作梯度返回。
static const aclTensor* GetOutTensorWithValueZero(aclTensor* grad_y, aclOpExecutor* executor)
{
    aclScalar* scalar = executor->AllocScalar(0);
    auto valueTensor = executor->ConvertToTensor(scalar, grad_y->GetDataType());
    auto gradYDims = op::ToShapeVector(grad_y->GetViewShape());
    aclIntArray* dimArray = executor->AllocIntArray(gradYDims.data(), gradYDims.size());
    auto dimTensor = executor->ConvertToTensor(dimArray, op::DataType::DT_INT64);
    return l0op::Fill(dimTensor, valueTensor, dimArray, executor);
}

static void CreateGradXReshapeData(const aclTensor* grad_x, FVector<int64_t>& newShapeVector)
{
    const auto& shape = grad_x->GetViewShape();
    newShapeVector.push_back(shape.GetDim(DIM_INDEX_B));
    newShapeVector.push_back(shape.GetDim(DIM_INDEX_C));
    newShapeVector.push_back(shape.GetDim(DIM_INDEX_M));
}

aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(const aclTensor* grad_x, const aclTensor* idx,
                                                          const aclTensor* weight, int m, aclTensor* grad_y,
                                                          uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnThreeInterpolateBackward, DFX_IN(grad_x, idx, weight, m), DFX_OUT(grad_y));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(grad_x, idx, weight, grad_y, m);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (grad_x->IsEmpty() || idx->IsEmpty() || weight->IsEmpty()) {
        // 输出为空（如 b==0 或 c==0）时没有元素可写，直接成功。
        // 输出非空（n==0 且 b,c,m>0）时必须写成全 0，与 kernel 清零语义一致。
        if (grad_y->IsEmpty()) {
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
        auto fillOut = GetOutTensorWithValueZero(grad_y, uniqueExecutor.get());
        CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(fillOut, grad_y, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto grad_x_contiguous = l0op::Contiguous(grad_x, uniqueExecutor.get());
    CHECK_RET(grad_x_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto idx_contiguous = l0op::Contiguous(idx, uniqueExecutor.get());
    CHECK_RET(idx_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weight_contiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_RET(weight_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_3510) {
        // 950(arch35)：def.cpp 该芯片 format=ND，kernel 原生 ND，跳过 5HD 转换直连
        auto outNd = l0op::ThreeInterpolateBackwardNd(grad_x_contiguous, idx_contiguous, weight_contiguous, m,
                                                      uniqueExecutor.get());
        CHECK_RET(outNd != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(outNd, grad_y, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // grad_x:(b,c,n,1)(NCHW) -> (b,c1,n,1,c0)(5HD)
        auto grad_x_5d = l0op::TransDataSpecial(grad_x_contiguous, op::Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
        CHECK_RET(grad_x_5d != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // grad_x:(b,c1,n,1,c0)(5HD) -> grad_y:(b,c1,m,1,c0)(5HD)
        auto out_5d = l0op::ThreeInterpolateBackward(grad_x_5d, idx_contiguous, weight_contiguous, m,
                                                     uniqueExecutor.get());
        CHECK_RET(out_5d != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto out_4d = l0op::TransDataSpecial(out_5d, op::Format::FORMAT_NCHW, 0, uniqueExecutor.get());
        CHECK_RET(out_4d != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(out_4d, grad_y, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnThreeInterpolateBackward(void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnThreeInterpolateBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspace_size, executor, stream);
}

#ifdef __cplusplus
}
#endif
