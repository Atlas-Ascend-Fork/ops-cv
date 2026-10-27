/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_non_max_suppression.h"
#include "non_max_suppression_v6.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include <string>
#include <vector>
#include "base/err_msg.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                             op::DataType::DT_FLOAT};

static constexpr size_t DIM_TWO = 2;
static constexpr size_t NUM_FOUR = 4;
static constexpr int32_t MAX_VALID_OUTPUT = 700;
static constexpr int64_t MAX_BOXES_NUM_PER_BATCH = 50000;

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor* boxes, const aclTensor* scores, aclIntArray* maxOutputBoxesPerClass,
                         const aclFloatArray* iouThreshold, const aclFloatArray* scoreThreshold,
                         aclTensor* selectedIndices)
{
    OP_CHECK_NULL(boxes, return false);
    OP_CHECK_NULL(scores, return false);
    OP_CHECK_NULL(maxOutputBoxesPerClass, return false);
    OP_CHECK_NULL(iouThreshold, return false);
    OP_CHECK_NULL(scoreThreshold, return false);
    OP_CHECK_NULL(selectedIndices, return false);
    return true;
}

static bool CheckThresholdSize(const aclFloatArray* iouThreshold, const aclFloatArray* scoreThreshold)
{
    if (iouThreshold->Size() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "iouThreshold size [%ld] should be equal to 1.",
                static_cast<int64_t>(iouThreshold->Size()));
        return false;
    }
    if (scoreThreshold->Size() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scoreThreshold size [%ld] should be equal to 1.",
                static_cast<int64_t>(scoreThreshold->Size()));
        return false;
    }
    return true;
}

static bool CheckMaxOutputSize(const aclIntArray* maxOutputBoxesPerClass)
{
    if (maxOutputBoxesPerClass->Size() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maxOutputBoxesPerClass size [%ld] should be equal to 1.",
                static_cast<int64_t>(maxOutputBoxesPerClass->Size()));
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* boxes, const aclTensor* scores, const aclFloatArray* iouThreshold)
{
    // 检查输入的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(boxes, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scores, FLOAT_DTYPE_SUPPORT_LIST, return false);
    // 检查boxes和scores数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(boxes, scores, return false);
    if ((iouThreshold->operator[](0) > 1) || (iouThreshold->operator[](0) < 0)) {
        return false;
    }
    return true;
}

static bool CheckFormatValid(const aclTensor* boxes, const aclTensor* scores, aclTensor* selectedIndices)
{
    return boxes->GetStorageFormat() == op::Format::FORMAT_ND && scores->GetStorageFormat() == op::Format::FORMAT_ND &&
           selectedIndices->GetStorageFormat() == op::Format::FORMAT_ND;
}

static bool CheckShape(const aclTensor* boxes, const aclTensor* scores)
{
    OP_CHECK_WRONG_DIMENSION(boxes, 3, return false);
    OP_CHECK_WRONG_DIMENSION(scores, 3, return false);

    auto const& boxesShape = boxes->GetViewShape();
    auto const& scoresShape = scores->GetViewShape();
    if (boxesShape.GetDim(0) != scoresShape.GetDim(0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "boxes shape dim0 [%ld] and scores shape dim0 [%ld] should be same",
                boxesShape.GetDim(0), scoresShape.GetDim(0));
        return false;
    }
    if (boxesShape.GetDim(1) != scoresShape.GetDim(DIM_TWO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "boxes shape dim1 [%ld] and scores shape dim2 [%ld] should be same",
                boxesShape.GetDim(1), scoresShape.GetDim(DIM_TWO));
        return false;
    }
    if (boxesShape.GetDim(DIM_TWO) != NUM_FOUR) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "boxes shape dim1 [%ld] should be 4", boxesShape.GetDim(DIM_TWO));
        return false;
    }
    // 每个 batch 的框数量（boxes/scores 的 spatial_dimension）上限为 50000
    if (boxesShape.GetDim(1) > MAX_BOXES_NUM_PER_BATCH) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "boxes num per batch [%ld] should be less than or equal to %ld.",
                boxesShape.GetDim(1), MAX_BOXES_NUM_PER_BATCH);
        return false;
    }
    return true;
}

static bool CheckAttr(const int centerPointBox)
{
    if (centerPointBox != 0 && centerPointBox != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "centerPointBox [%d] should be equal to 0 or 1", centerPointBox);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* boxes, const aclTensor* scores, aclIntArray* maxOutputBoxesPerClass,
                               const aclFloatArray* iouThreshold, aclFloatArray* scoreThreshold,
                               aclTensor* selectedIndices, int centerPointBox)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, selectedIndices),
              ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查阈值数组长度
    CHECK_RET(CheckThresholdSize(iouThreshold, scoreThreshold), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查每类最大输出框数数组长度
    CHECK_RET(CheckMaxOutputSize(maxOutputBoxesPerClass), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入的数据类型是否在API支持的数据类型范围之内、且满足约束，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(boxes, scores, iouThreshold), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查输入的数据格式是否在API支持范围之内
    CHECK_RET(CheckFormatValid(boxes, scores, selectedIndices), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查shape是否支持
    CHECK_RET(CheckShape(boxes, scores), ACLNN_ERR_PARAM_INVALID);

    // 7. 检查属性数据是否合法
    CHECK_RET(CheckAttr(centerPointBox), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(const aclTensor* boxes, const aclTensor* scores,
                                                   aclIntArray* maxOutputBoxesPerClass,
                                                   const aclFloatArray* iouThreshold, aclFloatArray* scoreThreshold,
                                                   int centerPointBox, aclTensor* selectedIndices,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnNonMaxSuppression,
                   DFX_IN(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
                   DFX_OUT(selectedIndices));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, selectedIndices,
                           centerPointBox);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    int64_t maxOutputSize = 0;
    if (maxOutputBoxesPerClass->Size() > 0) {
        maxOutputSize = maxOutputBoxesPerClass->operator[](0);
    }
    if (maxOutputSize <= 0 || maxOutputSize > MAX_VALID_OUTPUT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "maxOutputBoxesPerClass[%ld] should be greater than 0 and less than or equal to %d.", maxOutputSize,
                MAX_VALID_OUTPUT);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch != NpuArch::DAV_2002) {
        const std::string soc = op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString();
        const std::string reason = "SoC " + soc +
                                   " is not supported. Supported SoC is Ascend310P. Kernel is NonMaxSuppressionV6, "
                                   "dtype is FLOAT or FLOAT16, format is ND.";
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_RUNTIME_ERROR, "%s", reason.c_str());
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {"aclnnNonMaxSuppression", reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ1009", keys, values);
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    int64_t maxBoxesSize = maxOutputSize * scores->GetViewShape().GetDim(0) * scores->GetViewShape().GetDim(1);
    auto boxesContigous = l0op::Contiguous(boxes, uniqueExecutor.get());
    auto scoresContigous = l0op::Contiguous(scores, uniqueExecutor.get());

    // // 调用non_max_suppression算子
    const aclTensor* nmsOut = l0op::NonMaxSuppressionV6(boxesContigous, scoresContigous, maxOutputBoxesPerClass,
                                                        iouThreshold, scoreThreshold, centerPointBox, maxBoxesSize,
                                                        selectedIndices, uniqueExecutor.get());

    // 固定写法，将计算结果拷贝到输出selectedIndices上，selectedIndices可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(nmsOut, selectedIndices, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNonMaxSuppression(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNonMaxSuppression);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
