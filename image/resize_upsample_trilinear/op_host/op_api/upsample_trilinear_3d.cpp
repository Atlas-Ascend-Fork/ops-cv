/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_trilinear_3d.cpp
 * \brief
 */
#include "upsample_trilinear_3d.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleTrilinear3d);
OP_TYPE_REGISTER(ResizeUpsampleTrilinear);

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static constexpr float MAX_SUPPORT_SCALE = 50.0;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_310P_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                             op::DataType::DT_FLOAT16};

bool CheckScales(float scaleW, float scaleH, float scaleD)
{
    return (scaleW <= MAX_SUPPORT_SCALE && scaleH <= MAX_SUPPORT_SCALE && scaleD <= MAX_SUPPORT_SCALE);
}

static const aclTensor* CastToFp32(const aclTensor* self, aclOpExecutor* executor)
{
    self = l0op::Cast(self, op::DataType::DT_FLOAT, executor);
    OP_CHECK(self != nullptr, OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER, "ResizeUpsampleTrilinearAiCore cast self failed"),
             return nullptr);
    return self;
}

static const aclTensor* CastBackFromFp32(const aclTensor* out, op::DataType dataType, aclOpExecutor* executor)
{
    if (dataType == op::DataType::DT_FLOAT16 || dataType == op::DataType::DT_BF16) {
        out = l0op::Cast(out, dataType, executor);
    }
    return out;
}

// A950 (DAV_3510) AICORE branch
static const aclTensor* UpsampleTrilinear3dA950AiCore(const aclTensor* self, const aclIntArray* outputSize,
                                                      bool alignCorners, float scalesD, float scalesH, float scalesW,
                                                      const op::Shape& outShape, aclOpExecutor* executor,
                                                      const aclTensor* directOut)
{
    auto dataType = self->GetDataType();
    const aclTensor* out = directOut == nullptr ? executor->AllocTensor(outShape, dataType, self->GetStorageFormat()) :
                                                  directOut;
    CHECK_RET(out != nullptr, nullptr);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ResizeUpsampleTrilinear, OP_INPUT(self), OP_OUTPUT(out),
                                           OP_ATTR(outputSize, alignCorners, scalesD, scalesH, scalesW));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER, "ResizeUpsampleTrilinearA950AiCore ADD_TO_LAUNCHER_LIST_AICORE "
                                                     "failed; propagate the first error reported by the failing stage"),
             return nullptr);
    return out;
}

// A2 (DAV_2201) AICORE branch: cast fp16/bf16 → fp32 → compute → cast back
static const aclTensor* UpsampleTrilinear3dA2AiCore(const aclTensor* self, const aclIntArray* outputSize,
                                                    bool alignCorners, float scalesD, float scalesH, float scalesW,
                                                    const op::Shape& outShape, aclOpExecutor* executor)
{
    auto dataType = self->GetDataType();
    if (dataType == op::DataType::DT_FLOAT16 || dataType == op::DataType::DT_BF16) {
        self = CastToFp32(self, executor);
        CHECK_RET(self != nullptr, nullptr);
    }
    const aclTensor* out = executor->AllocTensor(outShape, op::DataType::DT_FLOAT, self->GetStorageFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ResizeUpsampleTrilinear, OP_INPUT(self), OP_OUTPUT(out),
                                           OP_ATTR(outputSize, alignCorners, scalesD, scalesH, scalesW));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER, "ResizeUpsampleTrilinearAiCore ADD_TO_LAUNCHER_LIST_AICORE "
                                                     "failed; propagate the first error reported by the failing stage"),
             return nullptr);
    return CastBackFromFp32(out, dataType, executor);
}

// 310P (DAV_2002) AICORE branch: cast fp16 → fp32 → compute → cast back to fp16
static const aclTensor* UpsampleTrilinear3d310PAiCore(const aclTensor* self, const aclIntArray* outputSize,
                                                      bool alignCorners, float scalesD, float scalesH, float scalesW,
                                                      const op::Shape& outShape, aclOpExecutor* executor)
{
    auto dataType = self->GetDataType();
    if (dataType == op::DataType::DT_FLOAT16) {
        self = CastToFp32(self, executor);
        CHECK_RET(self != nullptr, nullptr);
    }
    const aclTensor* out = executor->AllocTensor(outShape, op::DataType::DT_FLOAT, self->GetStorageFormat());
    auto aicoreRet = ADD_TO_LAUNCHER_LIST_AICORE(ResizeUpsampleTrilinear, OP_INPUT(self), OP_OUTPUT(out),
                                                 OP_ATTR(outputSize, alignCorners, scalesD, scalesH, scalesW));
    OP_CHECK(aicoreRet == ACLNN_SUCCESS,
             OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER, "ResizeUpsampleTrilinear310PAiCore ADD_TO_LAUNCHER_LIST_AICORE "
                                                     "failed; propagate the first error reported by the failing stage"),
             return nullptr);
    if (dataType == op::DataType::DT_FLOAT16) {
        out = l0op::Cast(out, op::DataType::DT_FLOAT16, executor);
    }
    return out;
}

// AICPU fallback branch: bf16 → fp32 → compute → cast back to bf16
static const aclTensor* UpsampleTrilinear3dAiCpu(const aclTensor* self, const aclIntArray* outputSize,
                                                 bool alignCorners, const aclFloatArray* scales, const aclFloatArray*,
                                                 const op::Shape& outShape, aclOpExecutor* executor)
{
    auto dataType = self->GetDataType();
    if (dataType == op::DataType::DT_BF16) {
        self = CastToFp32(self, executor);
        CHECK_RET(self != nullptr, nullptr);
    }
    uint64_t size = 0;
    auto ret = aclGetFloatArraySize(scales, &size);
    if (ret == ACLNN_SUCCESS && size == DIM_THREE) {
        outputSize = executor->AllocIntArray({}, 0);
    }
    const aclTensor* out = executor->AllocTensor(outShape, self->GetDataType(), self->GetStorageFormat());
    CHECK_RET(out != nullptr, nullptr);
    static internal::AicpuTaskSpace space("UpsampleTrilinear3d");
    ret = ADD_TO_LAUNCHER_LIST_AICPU(UpsampleTrilinear3d, OP_ATTR_NAMES({"output_size", "scales", "align_corners"}),
                                     OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(outputSize, scales, alignCorners));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    if (dataType == op::DataType::DT_BF16) {
        out = l0op::Cast(out, op::DataType::DT_BF16, executor);
    }
    return out;
}

const aclTensor* UpsampleTrilinear3dNcdhw(const aclTensor* self, const aclIntArray* outputSize, bool alignCorners,
                                          const aclFloatArray* scales, const aclFloatArray* castScales,
                                          float checkScaleW, float checkScaleH, float checkScaleD,
                                          aclOpExecutor* executor, aclTensor* directOut)
{
    L0_DFX(UpsampleTrilinear3dNcdhw, self, outputSize, alignCorners, scales);

    auto selfShape = op::ToShapeVector(self->GetViewShape());
    float scalesD = 0.0;
    float scalesH = 0.0;
    float scalesW = 0.0;

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    auto dimTwo = DIM_TWO;
    auto dimThree = DIM_THREE;
    auto dimFour = DIM_FOUR;

    if ((curArch == NpuArch::DAV_2002) && CheckType(self->GetDataType(), AICORE_310P_SUPPORT_LIST) &&
        CheckScales(checkScaleW, checkScaleH, checkScaleD)) {
        dimTwo = DIM_ZERO;
        dimThree = DIM_ONE;
        dimFour = DIM_TWO;
    }

    selfShape[dimTwo] = (*outputSize)[DIM_ZERO];
    selfShape[dimThree] = (*outputSize)[DIM_ONE];
    selfShape[dimFour] = (*outputSize)[DIM_TWO];

    uint64_t size = 0;
    auto ret = aclGetFloatArraySize(castScales, &size);
    if (ret == ACLNN_SUCCESS && size == DIM_THREE) {
        scalesD = (*castScales)[DIM_ZERO];
        scalesH = (*castScales)[DIM_ONE];
        scalesW = (*castScales)[DIM_TWO];
    }
    op::Shape outShape;
    op::ToShape(selfShape.data(), selfShape.size(), outShape);

    bool scalesOk = CheckScales(checkScaleW, checkScaleH, checkScaleD);
    bool dtypeOkA5OrA2 = CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
    bool dtypeOk310P = CheckType(self->GetDataType(), AICORE_310P_SUPPORT_LIST);
    if (curArch == NpuArch::DAV_3510 && dtypeOkA5OrA2 && scalesOk) {
        return UpsampleTrilinear3dA950AiCore(self, outputSize, alignCorners, scalesD, scalesH, scalesW, outShape,
                                             executor, directOut);
    }
    if (curArch == NpuArch::DAV_2201 && dtypeOkA5OrA2 && scalesOk) {
        return UpsampleTrilinear3dA2AiCore(self, outputSize, alignCorners, scalesD, scalesH, scalesW, outShape,
                                           executor);
    }
    if (curArch == NpuArch::DAV_2002 && dtypeOk310P && scalesOk) {
        return UpsampleTrilinear3d310PAiCore(self, outputSize, alignCorners, scalesD, scalesH, scalesW, outShape,
                                             executor);
    }
    return UpsampleTrilinear3dAiCpu(self, outputSize, alignCorners, scales, castScales, outShape, executor);
}
} // namespace l0op
