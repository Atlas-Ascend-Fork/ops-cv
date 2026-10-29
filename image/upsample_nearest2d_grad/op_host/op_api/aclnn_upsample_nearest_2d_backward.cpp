/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

#include "image/resize_nearest_neighbor_v2_grad/op_api/resize_nearest_neighbor_v2_grad.h"
#include "image/upsample_nearest_exact2d_grad/op_host/op_api/upsample_nearest_exact2d_grad.h"
#include "aclnn_upsample_nearest_2d_backward.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

struct Nearest2dGradData {
    const aclTensor* gradOut;
    aclTensor* gradInput;
    const aclIntArray* outputSize;
    const aclIntArray* inputSize;
    double scalesH;
    double scalesW;
};

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                       op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static constexpr int64_t DIM_LIMIT = 4;
static constexpr int64_t EXPECT_SIZE = 2;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;

static bool CheckNotNull(const aclTensor* gradOutput, const aclIntArray* outputSize, const aclIntArray* inputSize,
                         const aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

bool CheckInputElements(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize)
{
    int64_t outH = (*outputSize)[DIM_ZERO];
    int64_t outW = (*outputSize)[DIM_ONE];
    int64_t batch = (*inputSize)[DIM_ZERO];
    int64_t channels = (*inputSize)[DIM_ONE];
    int64_t inputH = (*inputSize)[DIM_TWO];
    int64_t inputW = (*inputSize)[DIM_THREE];
    auto gradOutShape = gradOut->GetViewShape();
    size_t shapeDim = gradOutShape.GetDimNum();
    FVector<int64_t> fullOutSize = {batch, channels, outH, outW};

    if (gradOut->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        inputH = (*inputSize)[DIM_ONE];
        inputW = (*inputSize)[DIM_TWO];
        channels = (*inputSize)[DIM_THREE];
        fullOutSize[DIM_ONE] = outH;
        fullOutSize[DIM_TWO] = outW;
        fullOutSize[DIM_THREE] = channels;
    }

    OP_CHECK(inputH > 0 && inputW > 0 && outH > 0 && outW > 0,
             OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON("aclnnUpsampleNearest2dBackward", "input",
                                                        (std::to_string(inputH) + ", " + std::to_string(inputW) + ", " +
                                                         std::to_string(outH) + ", " + std::to_string(outW))
                                                            .c_str(),
                                                        "input and output sizes should be greater than 0"),
             return false);

    for (size_t i = 0; i < shapeDim; ++i) {
        if (gradOutShape.GetDim(i) != fullOutSize[i]) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "aclnnUpsampleNearest2dBackward", "grad_output",
                (std::to_string(i) + ", " + std::to_string(fullOutSize[i]) + ", " + std::to_string(i)).c_str(),
                "expected grad_output to have the same shape as output; output.size() =");
            return false;
        }
    }
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* gradOut, const aclTensor* gradInput)
{
    const auto& supportList = GetDtypeSupportList();
    // 检查gradOut的数据类型是否在ResizeNearestNeighborV2Grad算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, supportList, return false);
    // 检查gradInput的数据类型是否与gradOut一致
    OP_CHECK_DTYPE_NOT_MATCH(gradOut, gradInput->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize)
{
    size_t outputSizeCount = outputSize->Size();
    size_t inputSizeNum = inputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOut, DIM_LIMIT, return false);
    OP_CHECK(outputSizeCount == EXPECT_SIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleNearest2dBackward", "output_size",
                                          std::to_string(outputSizeCount).c_str(), "2"),
             return false);

    OP_CHECK(inputSizeNum == DIM_LIMIT,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleNearest2dBackward", "input_size",
                                          std::to_string(inputSizeNum).c_str(), "4"),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                               const aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOut, outputSize, inputSize, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(gradOut, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 4.检查gradOut和gradInput N/C轴的大小是否一致
    CHECK_RET(CheckNCDimValid(gradInput, gradOut), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查输入元素是否合法
    CHECK_RET(CheckInputElements(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool isAiCoreSupport(const aclTensor* gradOutContiguous, const aclIntArray* inputSize, double scalesH,
                            double scalesW)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_2201) {
        return false;
    }
    if (scalesH <= 0 || scalesW <= 0) {
        return false;
    }

    int64_t inputC = (*inputSize)[1];
    int64_t inputH = (*inputSize)[2];
    int64_t inputW = (*inputSize)[3];
    int64_t gradOutH = gradOutContiguous->GetViewShape().GetDim(2);
    int64_t gradOutW = gradOutContiguous->GetViewShape().GetDim(3);
    if (gradOutContiguous->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        inputC = (*inputSize)[DIM_THREE];
        inputH = (*inputSize)[DIM_ONE];
        inputW = (*inputSize)[DIM_TWO];
        gradOutH = gradOutContiguous->GetViewShape().GetDim(DIM_ONE);
        gradOutW = gradOutContiguous->GetViewShape().GetDim(DIM_TWO);
    }
    const int64_t MAX_C = 6;
    const int64_t MIN_H = 8;
    const int64_t MIN_W = 16;
    if (inputC > MAX_C * inputH * inputW || inputC > MAX_C * gradOutH * gradOutW) {
        return false;
    }
    if (inputH < MIN_H || gradOutH < MIN_H || inputW < MIN_W || gradOutW < MIN_W) {
        return false;
    }
    if (static_cast<int64_t>(inputH * scalesH) != gradOutH || static_cast<int64_t>(inputW * scalesW) != gradOutW) {
        return false;
    }
    return true;
}

static const aclTensor* GoUpsampleNearestGrad(Nearest2dGradData nearest2dGradData, aclOpExecutor* executor)
{
    auto dataType = nearest2dGradData.gradOut->GetDataType();
    float realScalesH = nearest2dGradData.scalesH > 0 ? static_cast<float>(nearest2dGradData.scalesH) : 0;
    float realScalesW = nearest2dGradData.scalesW > 0 ? static_cast<float>(nearest2dGradData.scalesW) : 0;
    if (nearest2dGradData.gradOut->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        const int64_t permuteNCHWList[] = {DIM_ZERO, DIM_THREE, DIM_ONE, DIM_TWO};
        auto permuteNCHWArray = executor->AllocIntArray(permuteNCHWList, DIM_LIMIT);
        CHECK_RET(permuteNCHWArray != nullptr, nullptr);

        auto gradOutTranspose = l0op::Transpose(nearest2dGradData.gradOut, permuteNCHWArray, executor);
        CHECK_RET(gradOutTranspose != nullptr, nullptr);

        vector<int64_t> inputSizeList{};
        inputSizeList.push_back((*nearest2dGradData.inputSize)[DIM_ZERO]);
        inputSizeList.push_back((*nearest2dGradData.inputSize)[DIM_THREE]);
        inputSizeList.push_back((*nearest2dGradData.inputSize)[DIM_ONE]);
        inputSizeList.push_back((*nearest2dGradData.inputSize)[DIM_TWO]);
        auto inputSizeNCHW = executor->AllocIntArray(inputSizeList.data(), inputSizeList.size());
        CHECK_RET(inputSizeNCHW != nullptr, nullptr);
        // 升精度计算
        if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
            gradOutTranspose = l0op::Cast(gradOutTranspose, op::DataType::DT_FLOAT, executor);
            CHECK_RET(gradOutTranspose != nullptr, nullptr);
        }

        auto nearestExactOut = l0op::UpsampleNearestExact2dGrad(gradOutTranspose, nearest2dGradData.outputSize,
                                                                inputSizeNCHW, nearest2dGradData.gradInput, realScalesH,
                                                                realScalesW, false, executor);
        CHECK_RET(nearestExactOut != nullptr, nullptr);
        // 转回初始精度
        if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
            nearestExactOut = l0op::Cast(nearestExactOut, dataType, executor);
            CHECK_RET(nearestExactOut != nullptr, nullptr);
        }

        const int64_t permuteNHWCList[] = {DIM_ZERO, DIM_TWO, DIM_THREE, DIM_ONE};
        auto permuteNHWCArray = executor->AllocIntArray(permuteNHWCList, DIM_LIMIT);
        CHECK_RET(permuteNHWCArray != nullptr, nullptr);

        return l0op::Transpose(nearestExactOut, permuteNHWCArray, executor);
    } else {
        auto gradOut = nearest2dGradData.gradOut;
        // 升精度计算
        if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
            gradOut = l0op::Cast(gradOut, op::DataType::DT_FLOAT, executor);
            CHECK_RET(gradOut != nullptr, nullptr);
        }
        auto nearestExactOut = l0op::UpsampleNearestExact2dGrad(
            gradOut, nearest2dGradData.outputSize, nearest2dGradData.inputSize, nearest2dGradData.gradInput,
            realScalesH, realScalesW, false, executor);
        // 转回初始精度
        if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
            nearestExactOut = l0op::Cast(nearestExactOut, dataType, executor);
            CHECK_RET(nearestExactOut != nullptr, nullptr);
        }
        return nearestExactOut;
    }
}

static const aclTensor* GoUpsampleNearestGradTranspose(Nearest2dGradData nearest2dGradData, aclOpExecutor* executor)
{
    float realScalesH = nearest2dGradData.scalesH > 0 ? static_cast<float>(nearest2dGradData.scalesH) : 0;
    float realScalesW = nearest2dGradData.scalesW > 0 ? static_cast<float>(nearest2dGradData.scalesW) : 0;
    if ((*nearest2dGradData.outputSize)[DIM_ZERO] == (*nearest2dGradData.inputSize)[DIM_TWO] &&
        (*nearest2dGradData.outputSize)[DIM_ONE] == (*nearest2dGradData.inputSize)[DIM_THREE]) {
        realScalesH = 1.0;
        realScalesW = 1.0;
    }
    // 将输入gradOut格式转换成NC1HWC0
    auto gradOutTransdata = l0op::TransDataSpecial(nearest2dGradData.gradOut, op::Format::FORMAT_NC1HWC0, 0, executor);
    CHECK_RET(gradOutTransdata != nullptr, nullptr);

    auto resizeNearestGradOut = l0op::UpsampleNearestExact2dGrad(
        gradOutTransdata, nearest2dGradData.outputSize, nearest2dGradData.inputSize, nearest2dGradData.gradInput,
        realScalesH, realScalesW, false, executor);
    CHECK_RET(resizeNearestGradOut != nullptr, nullptr);

    return l0op::TransData(resizeNearestGradOut, nearest2dGradData.gradInput->GetStorageFormat(), 0, executor);
}

static const aclTensor* GoResizeNearestGrad(Nearest2dGradData nearest2dGradData, aclOpExecutor* executor)
{
    // FLOAT16转FLOAT计算
    auto gradOutCast = nearest2dGradData.gradOut;
    auto gradOutDtype = nearest2dGradData.gradOut->GetDataType();
    if (gradOutDtype == op::DataType::DT_FLOAT16) {
        gradOutCast = l0op::Cast(nearest2dGradData.gradOut, op::DataType::DT_FLOAT, executor);
        CHECK_RET(gradOutCast != nullptr, nullptr);
    }

    // 将输入gradOut格式转换成NC1HWC0
    auto gradOutTransdata = l0op::TransDataSpecial(gradOutCast, op::Format::FORMAT_NC1HWC0, 0, executor);
    CHECK_RET(gradOutTransdata != nullptr, nullptr);

    // 调用ResizeNearestNeighborV2Grad算子kernel, inputSize对应[N, C, H, W]或者[N, H, W, C]
    bool alignCorners = false;
    bool halfPixelCenters = false;
    auto resizeNearestGradOut = l0op::ResizeNearestNeighborV2Grad5Hd(gradOutTransdata, nearest2dGradData.inputSize,
                                                                     alignCorners, halfPixelCenters, executor);
    CHECK_RET(resizeNearestGradOut != nullptr, nullptr);

    auto resizeNearestGradOutTransdata = l0op::TransData(resizeNearestGradOut,
                                                         nearest2dGradData.gradInput->GetStorageFormat(), 0, executor);
    CHECK_RET(resizeNearestGradOutTransdata != nullptr, nullptr);

    // FLOAT16转FLOAT计算后转回FLOAT
    if (gradOutDtype == op::DataType::DT_FLOAT16) {
        resizeNearestGradOutTransdata = l0op::Cast(resizeNearestGradOutTransdata, op::DataType::DT_FLOAT16, executor);
        CHECK_RET(resizeNearestGradOutTransdata != nullptr, nullptr);
    }
    return resizeNearestGradOutTransdata;
}

aclnnStatus aclnnUpsampleNearest2dBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclIntArray* outputSize,
                                                           const aclIntArray* inputSize, double scalesH, double scalesW,
                                                           aclTensor* gradInput, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnUpsampleNearest2dBackward, DFX_IN(gradOut, outputSize, inputSize, scalesH, scalesW),
                   DFX_OUT(gradInput));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto checkRet = CheckParams(gradOut, outputSize, inputSize, gradInput);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    // resize_nearest_neighbor_v2_grad算子的空tensor在kernel中支持
    if (gradOut->IsEmpty() || gradInput->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOut转换成连续的tensor
    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* resizeNearestGradOutCast = nullptr;
    if (IsRegBase()) {
        bool alignCorners = false;
        bool halfPixelCenters = false;
        vector<float> scalesList{};
        scalesList.push_back(scalesH);
        scalesList.push_back(scalesW);
        const aclFloatArray* scales = uniqueExecutor->AllocFloatArray(scalesList.data(), scalesList.size());
        CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto resizeNearestGradOut = l0op::ResizeNearestNeighborV2Grad(gradOutContiguous, inputSize, alignCorners,
                                                                      halfPixelCenters, scales, uniqueExecutor.get());
        CHECK_RET(resizeNearestGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(resizeNearestGradOut, gradInput, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    } else if (isAiCoreSupport(gradOutContiguous, inputSize, scalesH, scalesW)) {
        Nearest2dGradData nearest2dGradData = {gradOutContiguous, gradInput, outputSize, inputSize, scalesH, scalesW};
        resizeNearestGradOutCast = GoUpsampleNearestGrad(nearest2dGradData, uniqueExecutor.get());
        CHECK_RET(resizeNearestGradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 && (scalesH > 0 || scalesW > 0)) {
        Nearest2dGradData nearest2dGradData = {gradOutContiguous, gradInput, outputSize, inputSize, scalesH, scalesW};
        resizeNearestGradOutCast = GoUpsampleNearestGradTranspose(nearest2dGradData, uniqueExecutor.get());
        CHECK_RET(resizeNearestGradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        Nearest2dGradData nearest2dGradData = {gradOutContiguous, gradInput, outputSize, inputSize, scalesH, scalesW};
        resizeNearestGradOutCast = GoResizeNearestGrad(nearest2dGradData, uniqueExecutor.get());
        CHECK_RET(CheckReduceOutShape(resizeNearestGradOutCast, gradInput), ACLNN_ERR_PARAM_INVALID);
    }

    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(resizeNearestGradOutCast, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearest2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearest2dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
