/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"
#include "error_report_util.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "upsample_nearest_exact3d.h"
#include "op_api/aclnn_check.h"
#include "aclnn_upsample_nearest_exact3d.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
struct Scales {
    const double scalesD;
    const double scalesH;
    const double scalesW;
};

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> ASCEND310P_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                  op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const int64_t DIM_LIMIT = 5;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static constexpr size_t EXPECT_SIZE = 3;
static constexpr double EPSILON = 1e-5;

static bool CheckNotNull(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2002) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND310P_DTYPE_SUPPORT_LIST, return false);
    } else if (IsRegBase(curArch)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_REGBASE, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(self, DIM_LIMIT, return false);
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleNearestExact3d", "output_size",
                                          std::to_string(outputSizeNum).c_str(), "3"),
             return false);

    // out的shape应该等于推导后的shape
    op::Shape inputShape = self->GetViewShape();
    op::Shape expectShape;
    if (self->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        expectShape.AppendDim(inputShape.GetDim(DIM_ZERO));
        expectShape.AppendDim((*outputSize)[DIM_ZERO]);
        expectShape.AppendDim((*outputSize)[DIM_ONE]);
        expectShape.AppendDim((*outputSize)[DIM_TWO]);
        expectShape.AppendDim(inputShape.GetDim(DIM_FOUR));
    } else {
        expectShape.AppendDim(inputShape.GetDim(DIM_ZERO));
        expectShape.AppendDim(inputShape.GetDim(DIM_ONE));
        expectShape.AppendDim((*outputSize)[DIM_ZERO]);
        expectShape.AppendDim((*outputSize)[DIM_ONE]);
        expectShape.AppendDim((*outputSize)[DIM_TWO]);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectShape, return false);
    return true;
}

static bool CheckInputElement(const aclTensor* self, const Scales& scale, const aclIntArray* outputSize)
{
    auto selfShape = self->GetViewShape();
    int64_t inputN = selfShape.GetDim(DIM_ZERO);
    int64_t outC = selfShape.GetDim(DIM_ONE);
    int64_t inputD = selfShape.GetDim(DIM_TWO);
    int64_t inputH = selfShape.GetDim(DIM_THREE);
    int64_t inputW = selfShape.GetDim(DIM_FOUR);
    int64_t outD = (*outputSize)[DIM_ZERO];
    int64_t outH = (*outputSize)[DIM_ONE];
    int64_t outW = (*outputSize)[DIM_TWO];
    if (self->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        outC = selfShape.GetDim(DIM_FOUR);
        inputD = selfShape.GetDim(DIM_ONE);
        inputH = selfShape.GetDim(DIM_TWO);
        inputW = selfShape.GetDim(DIM_THREE);
    }

    OP_CHECK(
        inputD > 0 && inputH > 0 && inputW > 0 && outD > 0 && outH > 0 && outW > 0,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleNearestExact3d", "input", std::to_string(inputD).c_str(),
                                                 "all input and output dims must be greater than 0"),
        return false);

    OP_CHECK(inputN > 0 && outC > 0,
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("aclnnUpsampleNearestExact3d", "sizes",
                                                       op::ToString(self->GetViewShape()).GetString(),
                                                       "non-empty 5D data tensor expected"),
             return false);

    if (scale.scalesD > 0 && scale.scalesH > 0 && scale.scalesW > 0) {
        int64_t dstDimD = static_cast<int64_t>(static_cast<double>(inputD) * scale.scalesD);
        int64_t dstDimD_Max = static_cast<int64_t>(static_cast<double>(inputD) * (scale.scalesD + EPSILON));
        int64_t dstDimD_Min = static_cast<int64_t>(static_cast<double>(inputD) * (scale.scalesD - EPSILON));
        OP_CHECK((outD == dstDimD) || (outD >= dstDimD_Min && outD <= dstDimD_Max),
                 error_report::ReportInvalidArgument(
                     "aclnnUpsampleNearestExact3d", "scale",
                     "scale conflicts with outputSize. scale_D * input_D should be equal to outputSize_D"),
                 return false);
        int64_t dstDimH = static_cast<int64_t>(static_cast<double>(inputH) * scale.scalesH);
        int64_t dstDimH_Max = static_cast<int64_t>(static_cast<double>(inputH) * (scale.scalesH + EPSILON));
        int64_t dstDimH_Min = static_cast<int64_t>(static_cast<double>(inputH) * (scale.scalesH - EPSILON));
        OP_CHECK((outH == dstDimH) || (outH >= dstDimH_Min && outH <= dstDimH_Max),
                 error_report::ReportInvalidArgument(
                     "aclnnUpsampleNearestExact3d", "scale",
                     "scale conflicts with outputSize. scale_H * input_H should be equal to outputSize_H"),
                 return false);
        int64_t dstDimW = static_cast<int64_t>(static_cast<double>(inputW) * scale.scalesW);
        int64_t dstDimW_Max = static_cast<int64_t>(static_cast<double>(inputW) * (scale.scalesW + EPSILON));
        int64_t dstDimW_Min = static_cast<int64_t>(static_cast<double>(inputW) * (scale.scalesW - EPSILON));
        OP_CHECK((outW == dstDimW) || (outW >= dstDimW_Min && outW <= dstDimW_Max),
                 error_report::ReportInvalidArgument(
                     "aclnnUpsampleNearestExact3d", "scale",
                     "scale conflicts with outputSize. scale_W * input_W should be equal to outputSize_W"),
                 return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, Scales& scale, const aclIntArray* outputSize,
                               const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(self, scale, outputSize), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否支持
    CHECK_RET(CheckShape(self, outputSize, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
}; // namespace

const aclTensor* upsampleNearestExact3dCompute(const aclTensor* selfContiguous, const aclIntArray* outputSize,
                                               const aclFloatArray* scales, aclOpExecutor* executor)
{
    if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        const int64_t permuteNCDHWList[] = {DIM_ZERO, DIM_FOUR, DIM_ONE, DIM_TWO, DIM_THREE};
        auto permuteNCDHWArray = executor->AllocIntArray(permuteNCDHWList, DIM_LIMIT);
        CHECK_RET(permuteNCDHWArray != nullptr, nullptr);

        auto selfTranspose = l0op::Transpose(selfContiguous, permuteNCDHWArray, executor);
        CHECK_RET(selfTranspose != nullptr, nullptr);

        auto selfUpsampleNearestExact = l0op::UpsampleNearestExact3dNcdhw(selfTranspose, outputSize, scales, executor);
        CHECK_RET(selfUpsampleNearestExact != nullptr, nullptr);

        const int64_t permuteNDHWCList[] = {DIM_ZERO, DIM_TWO, DIM_THREE, DIM_FOUR, DIM_ONE};
        auto permuteNDHWCArray = executor->AllocIntArray(permuteNDHWCList, DIM_LIMIT);
        CHECK_RET(permuteNDHWCArray != nullptr, nullptr);

        return l0op::Transpose(selfUpsampleNearestExact, permuteNDHWCArray, executor);
    }
    // NCDHW
    return l0op::UpsampleNearestExact3dNcdhw(selfContiguous, outputSize, scales, executor);
}

aclnnStatus aclnnUpsampleNearestExact3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize,
                                                        double scalesD, double scalesH, double scalesW, aclTensor* out,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleNearestExact3d, DFX_IN(self, outputSize, scalesD, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    Scales scale = {scalesD, scalesH, scalesW};
    auto ret = CheckParams(self, scale, outputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 空tensor支持
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclFloatArray* scales;
    if (IsRegBase()) {
        vector<float> scalesList = {float(scalesD), float(scalesH), float(scalesW)};
        scales = uniqueExecutor->AllocFloatArray(scalesList.data(), scalesList.size());
    } else {
        vector<float> scalesList{};
        // 使用double类型计算1/scale，避免tiling中用float计算造成精度损失
        if (scalesW > 0 && scalesH > 0 && scalesD > 0) {
            scalesList.push_back(static_cast<float>(1.0 / scalesD));
            scalesList.push_back(static_cast<float>(1.0 / scalesH));
            scalesList.push_back(static_cast<float>(1.0 / scalesW));
        }
        scales = uniqueExecutor->AllocFloatArray(scalesList.data(), scalesList.size());
    }
    CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用upsampleNearestExact3dCompute计算
    auto result = upsampleNearestExact3dCompute(selfContiguous, outputSize, scales, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearestExact3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearestExact3d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
