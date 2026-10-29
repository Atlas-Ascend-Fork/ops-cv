/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "upsample_trilinear_3d.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_upsample_trilinear_3d.h"

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
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/reshape.h"
#include "op_api/level2_base_cv.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_DOUBLE, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> ASCEND310P_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                                  op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> A5_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static constexpr size_t EXPECT_SIZE = 3;
static constexpr float MAX_SUPPORT_SCALE = 50.0;

static bool CheckNotNull(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2002 || curArch == NpuArch::DAV_3002) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND310P_DTYPE_SUPPORT_LIST, return false);
    } else if (curArch == NpuArch::DAV_3510) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, A5_DTYPE_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckIsPlatform310p(const aclTensor* self)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2002 || curArch == NpuArch::DAV_3002) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND310P_DTYPE_SUPPORT_LIST, return false);
    } else {
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* outputSize)
{
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(self, UPSAMPLE_DIM_LIMIT, return false);
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleTrilinear3d", "output_size",
                                          std::to_string(outputSizeNum).c_str(), "3"),
             return false);
    const op::Format inputFormat = self->GetStorageFormat();
    OP_CHECK(inputFormat == op::Format::FORMAT_ND || inputFormat == op::Format::FORMAT_NCDHW ||
                 inputFormat == op::Format::FORMAT_NDHWC,
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnUpsampleTrilinear3d", "input",
                                                     op::ToString(inputFormat).GetString(),
                                                     "input storage format support NCDHW or NDHWC"),
             return false);
    return true;
}

static bool CheckInputElement(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    OP_CHECK(
        self->GetStorageFormat() == out->GetStorageFormat(),
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnUpsampleTrilinear3d", "input",
                                                (std::string(op::ToString(self->GetStorageFormat()).GetString()) +
                                                 ", " + std::string(op::ToString(out->GetStorageFormat()).GetString()))
                                                    .c_str(),
                                                "input storage format should be equal to output storage format"),
        return false);
    auto selfShape = self->GetViewShape();
    int64_t inputN = selfShape.GetDim(DIM_ZERO);
    int64_t outC = 0;
    int64_t inputD = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t outD = (*outputSize)[DIM_ZERO];
    int64_t outH = (*outputSize)[DIM_ONE];
    int64_t outW = (*outputSize)[DIM_TWO];
    if (self->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        outC = selfShape.GetDim(DIM_FOUR);
        inputD = selfShape.GetDim(DIM_ONE);
        inputH = selfShape.GetDim(DIM_TWO);
        inputW = selfShape.GetDim(DIM_THREE);
    } else {
        outC = selfShape.GetDim(DIM_ONE);
        inputD = selfShape.GetDim(DIM_TWO);
        inputH = selfShape.GetDim(DIM_THREE);
        inputW = selfShape.GetDim(DIM_FOUR);
    }

    OP_CHECK(
        inputD > 0 && inputH > 0 && inputW > 0 && outD > 0 && outH > 0 && outW > 0,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleTrilinear3d", "input", std::to_string(inputD).c_str(),
                                                 "all input and output dims must be greater than 0"),
        return false);
    OP_CHECK(outC > 0,
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("aclnnUpsampleTrilinear3d", "sizes",
                                                       op::ToString(self->GetViewShape()).GetString(),
                                                       "non-empty 5D data tensor expected"),
             return false);
    op::Shape expectShape = op::Shape{inputN, outC, outD, outH, outW};
    if (out->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        expectShape = op::Shape{inputN, outD, outH, outW, outC};
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectShape, return false);
    return true;
}

static float ComputeScales(float scale, uint32_t inputSize, uint32_t outputSize)
{
    auto zero = static_cast<float>(0.);
    if (scale > zero) {
        return scale;
    } else {
        if (outputSize != 0) {
            return (static_cast<float>(inputSize) / outputSize);
        } else {
            return static_cast<float>(0);
        }
    }
}

static float AsComputeScale(bool alignCorners, int64_t inputSize, int64_t outputSize, float scale)
{
    if (outputSize == inputSize) {
        return static_cast<float>(1);
    }
    if (alignCorners) {
        if ((outputSize > 1) && (outputSize != 1)) {
            return static_cast<float>(inputSize - 1) / (outputSize - 1);
        } else {
            return static_cast<float>(0);
        }
    } else {
        return ComputeScales(scale, inputSize, outputSize);
    }
}

static bool CheckUplimit(const aclTensor* self, const aclTensor* out)
{
    if (IsRegBase()) {
        return true;
    }
    int64_t inN = self->GetViewShape().GetDim(DIM_ZERO);
    int64_t inC = self->GetViewShape().GetDim(DIM_ONE);
    int64_t inD = self->GetViewShape().GetDim(DIM_TWO);
    int64_t inH = self->GetViewShape().GetDim(DIM_THREE);
    int64_t inW = self->GetViewShape().GetDim(DIM_FOUR);
    int64_t outN = out->GetViewShape().GetDim(DIM_ZERO);
    int64_t outC = out->GetViewShape().GetDim(DIM_ONE);
    int64_t outD = out->GetViewShape().GetDim(DIM_TWO);
    int64_t outH = out->GetViewShape().GetDim(DIM_THREE);
    int64_t outW = out->GetViewShape().GetDim(DIM_FOUR);

    OP_CHECK(
        inN <= INT32_MAX && inC <= INT32_MAX && inD <= INT32_MAX && inH <= INT32_MAX && inW <= INT32_MAX,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleTrilinear3d", "self", std::to_string(INT32_MAX).c_str(),
                                                 "all input and output dims must be greater than 0"),
        return false);
    OP_CHECK(
        outN <= INT32_MAX && outC <= INT32_MAX && outD <= INT32_MAX && outH <= INT32_MAX && outW <= INT32_MAX,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleTrilinear3d", "out", std::to_string(INT32_MAX).c_str(),
                                                 "all input and output dims must be greater than 0"),
        return false);
    return true;
}

static bool IsSameShape(const op::Shape& lhs, const op::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static bool CanDirectWriteOutput(const aclTensor* input, const aclTensor* out)
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        return false;
    }
    if (input->GetStorageFormat() != out->GetStorageFormat() || out->GetViewOffset() != 0) {
        return false;
    }
    return IsSameShape(out->GetViewShape(), out->GetStorageShape());
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(self, outputSize), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(self, outputSize, out), ACLNN_ERR_PARAM_INVALID);

    // 5. 校验上边界
    CHECK_RET(CheckUplimit(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

const aclTensor* upsampleTrilinear3dCompute(const aclTensor* selfContiguous, const aclIntArray* outputSize,
                                            bool alignCorners, const aclFloatArray* scales,
                                            const aclFloatArray* castScales, float scaleW, float scaleH, float scaleD,
                                            aclOpExecutor* executor, aclTensor* directOut)
{
    if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        const int64_t permuteNCDHWList[] = {DIM_ZERO, DIM_FOUR, DIM_ONE, DIM_TWO, DIM_THREE};
        auto permuteNCDHWArray = executor->AllocIntArray(permuteNCDHWList,
                                                         sizeof(permuteNCDHWList) / sizeof(permuteNCDHWList[0]));
        CHECK_RET(permuteNCDHWArray != nullptr, nullptr);

        auto selfTranspose = l0op::Transpose(selfContiguous, permuteNCDHWArray, executor);
        CHECK_RET(selfTranspose != nullptr, nullptr);

        auto selfUpsampleTrilinear = l0op::UpsampleTrilinear3dNcdhw(selfTranspose, outputSize, alignCorners, scales,
                                                                    castScales, scaleW, scaleH, scaleD, executor);
        CHECK_RET(selfUpsampleTrilinear != nullptr, nullptr);
        const int64_t permuteNDHWCList[] = {DIM_ZERO, DIM_TWO, DIM_THREE, DIM_FOUR, DIM_ONE};
        auto permuteNDHWCArray = executor->AllocIntArray(permuteNDHWCList,
                                                         sizeof(permuteNDHWCList) / sizeof(permuteNDHWCList[0]));
        CHECK_RET(permuteNDHWCArray != nullptr, nullptr);

        return l0op::Transpose(selfUpsampleTrilinear, permuteNDHWCArray, executor);
    } else {
        return l0op::UpsampleTrilinear3dNcdhw(selfContiguous, outputSize, alignCorners, scales, castScales, scaleW,
                                              scaleH, scaleD, executor, directOut);
    }
}

static bool CheckScales(float scaleW, float scaleH, float scaleD)
{
    return (scaleW <= MAX_SUPPORT_SCALE && scaleH <= MAX_SUPPORT_SCALE && scaleD <= MAX_SUPPORT_SCALE);
}

// 310P NDHWC path: transpose NDHWC→DHWNC, upsample, transpose back to NDHWC, ViewCopy to out.
static aclnnStatus Run310pNdhwc(const aclTensor* selfContiguous, const aclIntArray* outputSize, bool alignCorners,
                                const aclFloatArray* scales, const aclFloatArray* castScales, float scaleW,
                                float scaleH, float scaleD, aclTensor* out, aclOpExecutor* executor)
{
    const int64_t permuteDHWNCList[] = {DIM_ONE, DIM_TWO, DIM_THREE, DIM_ZERO, DIM_FOUR};
    auto permuteDHWNCArray = executor->AllocIntArray(permuteDHWNCList,
                                                     sizeof(permuteDHWNCList) / sizeof(permuteDHWNCList[0]));
    CHECK_RET(permuteDHWNCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfTranspose = l0op::Transpose(selfContiguous, permuteDHWNCArray, executor);
    CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfUpsampleTrilinear = l0op::UpsampleTrilinear3dNcdhw(selfTranspose, outputSize, alignCorners, scales,
                                                                castScales, scaleW, scaleH, scaleD, executor);
    CHECK_RET(selfUpsampleTrilinear != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const int64_t permuteNDHWList[] = {DIM_THREE, DIM_ZERO, DIM_ONE, DIM_TWO, DIM_FOUR};
    auto permuteNDHWCArray = executor->AllocIntArray(permuteNDHWList,
                                                     sizeof(permuteNDHWList) / sizeof(permuteNDHWList[0]));
    CHECK_RET(permuteNDHWCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto result = l0op::Transpose(selfUpsampleTrilinear, permuteNDHWCArray, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(result, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// 310P NCDHW path: transpose NCDHW→DHWNC, upsample, transpose back to NCDHW, ViewCopy to out.
static aclnnStatus Run310pNcdhw(const aclTensor* selfContiguous, const aclIntArray* outputSize, bool alignCorners,
                                const aclFloatArray* scales, const aclFloatArray* castScales, float scaleW,
                                float scaleH, float scaleD, aclTensor* out, aclOpExecutor* executor)
{
    const int64_t permuteHWNCList[] = {DIM_TWO, DIM_THREE, DIM_FOUR, DIM_ZERO, DIM_ONE};
    auto permuteHWNCArray = executor->AllocIntArray(permuteHWNCList,
                                                    sizeof(permuteHWNCList) / sizeof(permuteHWNCList[0]));
    CHECK_RET(permuteHWNCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfTranspose = l0op::Transpose(selfContiguous, permuteHWNCArray, executor);
    CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfUpsampleTrilinear = l0op::UpsampleTrilinear3dNcdhw(selfTranspose, outputSize, alignCorners, scales,
                                                                castScales, scaleW, scaleH, scaleD, executor);
    CHECK_RET(selfUpsampleTrilinear != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const int64_t permuteNDHWList[] = {DIM_THREE, DIM_FOUR, DIM_ZERO, DIM_ONE, DIM_TWO};
    auto permuteNCDHWArray = executor->AllocIntArray(permuteNDHWList,
                                                     sizeof(permuteNDHWList) / sizeof(permuteNDHWList[0]));
    CHECK_RET(permuteNCDHWArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto result = l0op::Transpose(selfUpsampleTrilinear, permuteNCDHWArray, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(result, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// Non-310P fallback path: direct compute via upsampleTrilinear3dCompute, ViewCopy to out if needed.
static aclnnStatus RunNon310pCompute(const aclTensor* selfContiguous, const aclIntArray* outputSize, bool alignCorners,
                                     const aclFloatArray* scales, const aclFloatArray* castScales, float scaleW,
                                     float scaleH, float scaleD, aclTensor* out, aclOpExecutor* executor)
{
    aclTensor* directOut = CanDirectWriteOutput(selfContiguous, out) ? out : nullptr;
    auto result = upsampleTrilinear3dCompute(selfContiguous, outputSize, alignCorners, scales, castScales, scaleW,
                                             scaleH, scaleD, executor, directOut);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (result != out) {
        auto viewCopyResult = l0op::ViewCopy(result, out, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

// Allocate scales (1/x) and castScales (x) float arrays. Both are empty when any
// scale is non-positive.
static aclnnStatus AllocScalesArrays(double scalesD, double scalesH, double scalesW, const aclFloatArray*& scales,
                                     const aclFloatArray*& castScales, aclOpExecutor* executor)
{
    vector<float> scalesList{};
    vector<float> scalesCastList{};
    if (scalesD > 0 && scalesH > 0 && scalesW > 0) {
        scalesList.push_back(scalesD);
        scalesList.push_back(scalesH);
        scalesList.push_back(scalesW);
        scalesCastList.push_back(static_cast<float>(1.0 / scalesD));
        scalesCastList.push_back(static_cast<float>(1.0 / scalesH));
        scalesCastList.push_back(static_cast<float>(1.0 / scalesW));
    }
    scales = executor->AllocFloatArray(scalesList.data(), scalesList.size());
    CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);
    castScales = executor->AllocFloatArray(scalesCastList.data(), scalesCastList.size());
    CHECK_RET(castScales != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// Resolve per-axis source scales from castScales and output shape; returns the
// alignCorners-adjusted W/H/D scales used by downstream compute.
struct ResolvedScales {
    float scaleW;
    float scaleH;
    float scaleD;
};

static ResolvedScales ResolveScales(bool alignCorners, const aclFloatArray* castScales, const aclTensor* self,
                                    const op::Shape& outShape)
{
    uint64_t size = 0;
    float scalesD1 = 0.0;
    float scalesH1 = 0.0;
    float scalesW1 = 0.0;
    if (aclGetFloatArraySize(castScales, &size) == ACLNN_SUCCESS && size == DIM_THREE) {
        scalesD1 = (*castScales)[DIM_ZERO];
        scalesH1 = (*castScales)[DIM_ONE];
        scalesW1 = (*castScales)[DIM_TWO];
    }
    auto inputShape = self->GetViewShape();
    ResolvedScales rs;
    rs.scaleW = AsComputeScale(alignCorners, inputShape.GetDim(DIM_FOUR), outShape.GetDim(DIM_FOUR), scalesW1);
    rs.scaleH = AsComputeScale(alignCorners, inputShape.GetDim(DIM_THREE), outShape.GetDim(DIM_THREE), scalesH1);
    rs.scaleD = AsComputeScale(alignCorners, inputShape.GetDim(DIM_TWO), outShape.GetDim(DIM_TWO), scalesD1);
    return rs;
}
} // namespace

aclnnStatus aclnnUpsampleTrilinear3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize,
                                                     bool alignCorners, double scalesD, double scalesH, double scalesW,
                                                     aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleTrilinear3d, DFX_IN(self, outputSize, alignCorners, scalesD, scalesH, scalesW),
                   DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto checkTrilinear3dRet = CheckParams(self, outputSize, out);
    CHECK_RET(checkTrilinear3dRet == ACLNN_SUCCESS, checkTrilinear3dRet);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclFloatArray* scales = nullptr;
    const aclFloatArray* castScales = nullptr;
    auto allocRet = AllocScalesArrays(scalesD, scalesH, scalesW, scales, castScales, uniqueExecutor.get());
    CHECK_RET(allocRet == ACLNN_SUCCESS, allocRet);

    auto selfShape = op::ToShapeVector(self->GetViewShape());
    selfShape[DIM_TWO] = (*outputSize)[DIM_ZERO];
    selfShape[DIM_THREE] = (*outputSize)[DIM_ONE];
    selfShape[DIM_FOUR] = (*outputSize)[DIM_TWO];
    op::Shape outShape;
    op::ToShape(selfShape.data(), selfShape.size(), outShape);

    auto rs = ResolveScales(alignCorners, castScales, self, outShape);
    float scaleW = rs.scaleW;
    float scaleH = rs.scaleH;
    float scaleD = rs.scaleD;
    if (CheckIsPlatform310p(self) && CheckScales(scaleW, scaleH, scaleD)) {
        if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
            auto ret = Run310pNdhwc(selfContiguous, outputSize, alignCorners, scales, castScales, scaleW, scaleH,
                                    scaleD, out, uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        } else {
            auto ret = Run310pNcdhw(selfContiguous, outputSize, alignCorners, scales, castScales, scaleW, scaleH,
                                    scaleD, out, uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }
    } else {
        auto ret = RunNon310pCompute(selfContiguous, outputSize, alignCorners, scales, castScales, scaleW, scaleH,
                                     scaleD, out, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleTrilinear3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleTrilinear3d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
