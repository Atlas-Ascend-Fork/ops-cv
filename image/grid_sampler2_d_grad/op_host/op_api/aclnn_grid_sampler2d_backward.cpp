/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_grid_sampler2d_backward.h"
#include "acl/acl_rt.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "grid_sampler2d_grad.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/cast.h"
#include "level0/fill.h"
#include "op_api/level2_base_cv.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t FIRST_DIM = 0;
static const size_t SECOND_DIM = 1;
static const size_t FOURTH_DIM = 3;
static const size_t THIRD_DIM = 2;

static const int64_t INTERPOLATION_MODE_MIN_VALUE = 0;
static const int64_t INTERPOLATION_MODE_MAX_VALUE = 2;
static const int64_t PADDING_MODE_MIN_VALUE = 0;
static const int64_t PADDING_MODE_MAX_VALUE = 2;
static const int64_t SPATIAL_GRID_LAST_DIM_SIZE = 2;
static const int64_t SPATIAL_DIM_NUM = 4;
static const int64_t GRAD_RESULT_SIZE = 2;
static const int64_t MAX_CHANNEL_SIZE = 2048;
static const int64_t OUTPUTMASK_MAX_SIZE = 2;
static const int64_t REGBASE_MAX_CHANNEL_NUM = 32;
static const int64_t REGBASE_BIG_CHANNEL_NUM = 128;
static const int64_t BILINEAR = 0;
static const int64_t BORDER = 1;
static const int64_t REFLECTION = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_DOUBLE};

static inline bool Is960SocVersion(const NpuArch curArch) { return curArch == NpuArch::DAV_9201; }

static inline bool CheckSocVersionGe910B(void)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return curArch == NpuArch::DAV_2201 || Is960SocVersion(curArch) || IsRegBase(curArch);
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    if (CheckSocVersionGe910B()) {
        return DTYPE_SUPPORT_LIST_910B;
    } else {
        return DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* input, const aclTensor* grid,
                            const aclTensor* inputGrad, const aclTensor* gridGrad)
{
    // 检查输入输出数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(grid, input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(inputGrad, input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(gridGrad, input->GetDataType(), return false);

    // 检查input的数据类型是否在gridsampler2dgrad算子的支持列表内
    auto dtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(input, dtypeSupportList, return false);
    return true;
}

static bool CheckAttrValid(int64_t interpolationModeVal, int64_t paddingModeVal)
{
    // 检查interpolationMode 、paddingMode是否在支持范围内
    if (interpolationModeVal < INTERPOLATION_MODE_MIN_VALUE || interpolationModeVal > INTERPOLATION_MODE_MAX_VALUE) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnGridSampler2DBackward", "interpolation_mode",
            (std::to_string(interpolationModeVal) + ", " + std::to_string(INTERPOLATION_MODE_MIN_VALUE) + ", " +
             std::to_string(INTERPOLATION_MODE_MAX_VALUE))
                .c_str(),
            "interpolationMode should be within the supported range");
        return false;
    }

    if (paddingModeVal < PADDING_MODE_MIN_VALUE || paddingModeVal > PADDING_MODE_MAX_VALUE) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnGridSampler2DBackward", "padding_mode",
            (std::to_string(paddingModeVal) + ", " + std::to_string(PADDING_MODE_MIN_VALUE) + ", " +
             std::to_string(PADDING_MODE_MAX_VALUE))
                .c_str(),
            "paddingMode should be within the supported range");
        return false;
    }
    return true;
}

// 检查tuple <values, indices>里的元素是否为非null, true表示为非null，false表示为null
static bool CheckTupleNullptr(std::tuple<aclTensor*, aclTensor*> tensorTuple)
{
    if (std::tuple_size<decltype(tensorTuple)>::value != GRAD_RESULT_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnGridSampler2DBackward", "grad_result_size", std::to_string(GRAD_RESULT_SIZE).c_str(),
            "the length of the tuple returned by GridSampler2DGrad does not match the "
            "expected internal contract size");
        return false;
    }
    return (std::get<0>(tensorTuple) != nullptr) && (std::get<1>(tensorTuple) != nullptr);
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* input, const aclTensor* grid,
                       const aclTensor* inputGrad, const aclTensor* gridGrad, const aclBoolArray* outputMask)
{
    const auto& gradOutputShape = gradOutput->GetViewShape();
    const auto& inputShape = input->GetViewShape();
    const auto& gridShape = grid->GetViewShape();
    OP_CHECK_WRONG_DIMENSION(input, SPATIAL_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(grid, SPATIAL_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(gradOutput, SPATIAL_DIM_NUM, return false);
    if (inputShape.GetDim(FIRST_DIM) != gridShape.GetDim(FIRST_DIM) ||
        inputShape.GetDim(FIRST_DIM) != gradOutputShape.GetDim(FIRST_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnGridSampler2DBackward", "input",
                                               (std::string(op::ToString(inputShape).GetString()) + ", " +
                                                std::string(op::ToString(gridShape).GetString()) + ", " +
                                                std::string(op::ToString(gradOutputShape).GetString()))
                                                   .c_str(),
                                               "expect input grid and gradOutput to have same batch size");
        return false;
    }
    if (inputShape.GetDim(SECOND_DIM) != gradOutputShape.GetDim(SECOND_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnGridSampler2DBackward", "input",
                                               (std::string(op::ToString(inputShape).GetString()) + ", " +
                                                std::string(op::ToString(gradOutputShape).GetString()))
                                                   .c_str(),
                                               "expect input and gradOutput to have same channel size");
        return false;
    }
    if (inputShape.GetDim(THIRD_DIM) == 0 || inputShape.GetDim(FOURTH_DIM) == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnGridSampler2DBackward", "input",
                                              op::ToString(inputShape).GetString(),
                                              "expect input to have non-empty spatial dimensions");
        return false;
    }
    if (gridShape.GetDim(SECOND_DIM) != gradOutputShape.GetDim(THIRD_DIM) ||
        gridShape.GetDim(THIRD_DIM) != gradOutputShape.GetDim(FOURTH_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnGridSampler2DBackward", "grid",
                                               (std::string(op::ToString(gridShape).GetString()) + ", " +
                                                std::string(op::ToString(gradOutputShape).GetString()))
                                                   .c_str(),
                                               "expect grid and gradOutput to have same H and W size");
        return false;
    }
    if (gridShape.GetDim(FOURTH_DIM) != SPATIAL_GRID_LAST_DIM_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "aclnnGridSampler2DBackward", "grid",
            (std::to_string(SPATIAL_GRID_LAST_DIM_SIZE) + ", " + std::string(op::ToString(gridShape).GetString()))
                .c_str(),
            "expect grid to have size in last dimension");
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL(inputGrad, input, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gridGrad, grid, return false);
    if (outputMask->Size() != OUTPUTMASK_MAX_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPESIZE("aclnnGridSampler2DBackward", "aclnn_grid_sampler2_d_backward",
                                      std::to_string(outputMask->Size()).c_str(),
                                      std::to_string(OUTPUTMASK_MAX_SIZE).c_str());
        return false;
    }
    return true;
}

static bool CheckDtypeAndChannelCanTranspose(const aclTensor* input, int64_t paddingMode)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    const auto& inputShape = input->GetViewShape();
    auto channel = input->GetStorageFormat() == op::Format::FORMAT_NHWC ? inputShape.GetDim(FOURTH_DIM) :
                                                                          inputShape.GetDim(SECOND_DIM);
    int64_t deterministicValue = 0;
    aclError retRts = aclrtCtxGetSysParamOpt(ACL_OPT_DETERMINISTIC, &deterministicValue);
    if (retRts != ACL_ERROR_NONE) {
        deterministicValue = 0;
        OP_LOGD("Unable to get system param deterministic = %ld.", deterministicValue);
    }
    bool regbaseTranspose = (curArch == NpuArch::DAV_3510 || Is960SocVersion(curArch)) &&
                            input->GetDataType() == op::DataType::DT_FLOAT && deterministicValue == 0 &&
                            (channel >= REGBASE_MAX_CHANNEL_NUM && paddingMode == BORDER ||
                             channel >= REGBASE_BIG_CHANNEL_NUM && paddingMode == REFLECTION);
    bool transposeArch = curArch != NpuArch::DAV_1001 && curArch != NpuArch::DAV_3510 && !Is960SocVersion(curArch);
    return (input->GetDataType() != op::DataType::DT_DOUBLE && transposeArch || regbaseTranspose) &&
           channel <= MAX_CHANNEL_SIZE;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* input, const aclTensor* grid,
                               int64_t interpolationMode, int64_t paddingMode, const aclTensor* inputGrad,
                               const aclTensor* gridGrad, const aclBoolArray* outputMask)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2In1Out(gradOutput, input, grid, inputGrad, gridGrad), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入、输出的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, input, grid, inputGrad, gridGrad), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查属性参数是否在支持范围内
    CHECK_RET(CheckAttrValid(interpolationMode, paddingMode), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入、输出的shape匹配关系
    CHECK_RET(CheckShape(gradOutput, input, grid, inputGrad, gridGrad, outputMask), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 空tensor场景kernel不会写输出，用Fill生成全0值tensor，再ViewCopy到输出上（参考roi_align_v2_backward同款处理）
static const aclTensor* GetOutTensorWithValueZero(aclTensor* output, aclOpExecutor* executor)
{
    aclScalar* scalar = executor->AllocScalar(0);
    CHECK_RET(scalar != nullptr, nullptr);
    auto valueTensor = executor->ConvertToTensor(scalar, output->GetDataType());
    auto outputDims = op::ToShapeVector(output->GetViewShape());
    aclIntArray* dimArray = executor->AllocIntArray(outputDims.data(), outputDims.size());
    CHECK_RET(dimArray != nullptr, nullptr);
    auto dimTensor = executor->ConvertToTensor(dimArray, op::DataType::DT_INT64);
    return l0op::Fill(dimTensor, valueTensor, dimArray, executor);
}

aclnnStatus aclnnGridSampler2DBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* input,
                                                       const aclTensor* grid, int64_t interpolationMode,
                                                       int64_t paddingMode, bool alignCorners,
                                                       const aclBoolArray* outputMask, aclTensor* inputGrad,
                                                       aclTensor* gridGrad, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnGridSampler2DBackward,
                   DFX_IN(gradOutput, input, grid, interpolationMode, paddingMode, alignCorners, outputMask),
                   DFX_OUT(inputGrad, gridGrad));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 输出掩码校验
    auto outputMaskRet = ACLNN_SUCCESS;
    OP_CHECK_NULL(outputMask, outputMaskRet = ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(outputMaskRet == ACLNN_SUCCESS, outputMaskRet);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, input, grid, interpolationMode, paddingMode, inputGrad, gridGrad, outputMask);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // gridsampler2d算子的空tensor场景kernel不会写任何输出（空grid时copyTimes=0），
    // 非空输出必须显式清零，否则调用方预分配的未初始化内存会被当作梯度返回（脏内存伪梯度）
    if (gradOutput->IsEmpty() || input->IsEmpty() || grid->IsEmpty() ||
        ((*outputMask)[0] == false && (*outputMask)[1] == false)) {
        // inputGrad非空时清零（对齐GPU语义：无采样点时dx=0）
        if ((*outputMask)[0] && !inputGrad->IsEmpty()) {
            auto fillDx = GetOutTensorWithValueZero(inputGrad, uniqueExecutor.get());
            CHECK_RET(fillDx != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto dxCopyResult = l0op::ViewCopy(fillDx, inputGrad, uniqueExecutor.get());
            CHECK_RET(dxCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // gridGrad非空时清零（如input的C=0时grid仍非空，GPU语义下dgrid=0）
        if ((*outputMask)[1] && !gridGrad->IsEmpty()) {
            auto fillDgrid = GetOutTensorWithValueZero(gridGrad, uniqueExecutor.get());
            CHECK_RET(fillDgrid != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto dgridCopyResult = l0op::ViewCopy(fillDgrid, gridGrad, uniqueExecutor.get());
            CHECK_RET(dgridCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入grid转换成连续的tensor
    auto gridContiguous = l0op::Contiguous(grid, uniqueExecutor.get());
    CHECK_RET(gridContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入input转换成连续的tensor
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    int64_t dimSize = static_cast<int64_t>(inputContiguous->GetViewShape().GetDimNum());
    bool transposeFlag = CheckDtypeAndChannelCanTranspose(input, paddingMode);
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    bool castFlag = input->GetDataType() == op::DataType::DT_FLOAT16 && curArch == NpuArch::DAV_1001;
    std::tuple<aclTensor*, aclTensor*> gridSampler2DBackwardOut;
    if (transposeFlag) {
        std::vector<int64_t> perm = {0, 2, 3, 1};
        auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm.data(), dimSize);
        aclTensor* inputTranspose = const_cast<aclTensor*>(
            l0op::Transpose(inputContiguous, valuePerm, uniqueExecutor.get()));
        CHECK_RET(inputTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
        aclTensor* gradOutputTranspose = const_cast<aclTensor*>(
            l0op::Transpose(gradOutputContiguous, valuePerm, uniqueExecutor.get()));
        CHECK_RET(gradOutputTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
        gradOutputTranspose->SetStorageFormat(op::Format::FORMAT_NHWC);
        inputTranspose->SetStorageFormat(op::Format::FORMAT_NHWC);
        // 调用AICROE算子
        gridSampler2DBackwardOut = l0op::GridSamplerGrad(gradOutputTranspose, inputTranspose, gridContiguous,
                                                         interpolationMode, paddingMode, alignCorners,
                                                         uniqueExecutor.get());
    } else {
        if (castFlag) {
            gradOutputContiguous = l0op::Cast(gradOutputContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
            inputContiguous = l0op::Cast(inputContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
            gridContiguous = l0op::Cast(gridContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(gridContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        if ((curArch == NpuArch::DAV_3510 || Is960SocVersion(curArch)) &&
            input->GetDataType() != op::DataType::DT_DOUBLE) {
            gridSampler2DBackwardOut = l0op::GridSamplerGrad(gradOutputContiguous, inputContiguous, gridContiguous,
                                                             interpolationMode, paddingMode, alignCorners,
                                                             uniqueExecutor.get());
        } else {
            // 进行计算
            gridSampler2DBackwardOut = l0op::GridSampler2DGrad(gradOutputContiguous, inputContiguous, gridContiguous,
                                                               interpolationMode, paddingMode, alignCorners,
                                                               uniqueExecutor.get());
        }
    }

    CHECK_RET(CheckTupleNullptr(gridSampler2DBackwardOut), ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出inputGrad上
    if ((*outputMask)[0]) {
        if (transposeFlag) {
            std::vector<int64_t> outPerm = {0, 3, 1, 2};
            auto outValuePerm = uniqueExecutor.get()->AllocIntArray(outPerm.data(), dimSize);
            aclTensor* transOutput0 = const_cast<aclTensor*>(
                l0op::Transpose(std::get<0>(gridSampler2DBackwardOut), outValuePerm, uniqueExecutor.get()));
            CHECK_RET(transOutput0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
            transOutput0->SetStorageFormat(inputGrad->GetStorageFormat());
            auto inputGradViewCopyResult = l0op::ViewCopy(transOutput0, inputGrad, uniqueExecutor.get());
            CHECK_RET(inputGradViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            if (castFlag) {
                // 固定写法，将计算结果转换成输出out的数据类型
                auto inputGradCal = std::get<0>(gridSampler2DBackwardOut);
                CHECK_RET(inputGradCal != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto inputGradCast = l0op::Cast(inputGradCal, input->GetDataType(), uniqueExecutor.get());
                CHECK_RET(inputGradCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto inputGradViewCopyResult = l0op::ViewCopy(inputGradCast, inputGrad, uniqueExecutor.get());
                CHECK_RET(inputGradViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            } else {
                auto inputGradViewCopyResult = l0op::ViewCopy(std::get<0>(gridSampler2DBackwardOut), inputGrad,
                                                              uniqueExecutor.get());
                CHECK_RET(inputGradViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }
    }
    // 固定写法，将计算结果拷贝到输出gridGrad上
    if ((*outputMask)[1]) {
        if (castFlag) {
            auto gridGradCal = std::get<1>(gridSampler2DBackwardOut);
            CHECK_RET(gridGradCal != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto gridGradCast = l0op::Cast(gridGradCal, grid->GetDataType(), uniqueExecutor.get());
            CHECK_RET(gridGradCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto gridGradViewCopyResult = l0op::ViewCopy(gridGradCast, gridGrad, uniqueExecutor.get());
            CHECK_RET(gridGradViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            auto gridGradViewCopyResult = l0op::ViewCopy(std::get<1>(gridSampler2DBackwardOut), gridGrad,
                                                         uniqueExecutor.get());
            CHECK_RET(gridGradViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGridSampler2DBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGridSampler2DBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
