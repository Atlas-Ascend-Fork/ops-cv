/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_nearest_neighbor_v2.cpp
 * \brief resize_nearest_neighbor_v2
 */

#include "resize_nearest_neighbor_v2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "op_api/aclnn_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ResizeNearestNeighborV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                              op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || IsRegBase(curArch)) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICPU算子kernel
static const aclTensor* ResizeNearestNeighborV2AICPU(const aclTensor* x, const aclTensor* size, bool alignCorners,
                                                     bool halfPixelCenters, const aclTensor* y, aclOpExecutor* executor)
{
    L0_DFX(ResizeNearestNeighborV2AICPU, x, size, y);

    static internal::AicpuTaskSpace space("ResizeNearestNeighbor", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ResizeNearestNeighborV2,
                                          OP_ATTR_NAMES({"align_corners", "half_pixel_centers"}), OP_INPUT(x, size),
                                          OP_OUTPUT(y), OP_ATTR(alignCorners, halfPixelCenters));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return y;
}

// AICORE算子kernel
static const aclTensor* ResizeNearestNeighborV2AICORE(const aclTensor* x, const aclTensor* size,
                                                      const aclFloatArray* scales, bool alignCorners,
                                                      bool halfPixelCenters, const aclTensor* y,
                                                      aclOpExecutor* executor)
{
    L0_DFX(ResizeNearestNeighborV2AICORE, x, size, y);
    auto ret = ACLNN_SUCCESS;
    if (scales == nullptr) {
        ret = ADD_TO_LAUNCHER_LIST_AICORE(ResizeNearestNeighborV2, OP_INPUT(x, size), OP_OUTPUT(y),
                                          OP_ATTR(alignCorners, halfPixelCenters));
    } else {
        ret = ADD_TO_LAUNCHER_LIST_AICORE(ResizeNearestNeighborV2, OP_INPUT(x, size), OP_OUTPUT(y),
                                          OP_ATTR(alignCorners, halfPixelCenters, scales));
    }

    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ResizeNearestNeighborV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return y;
}

const aclTensor* ResizeNearestNeighborV2(const aclTensor* x, const aclTensor* size, const aclFloatArray* scales,
                                         bool alignCorners, bool halfPixelCenters, const aclTensor* y,
                                         aclOpExecutor* executor)
{
    if (IsAiCoreSupport(x)) {
        return ResizeNearestNeighborV2AICORE(x, size, scales, alignCorners, halfPixelCenters, y, executor);
    } else {
        return ResizeNearestNeighborV2AICPU(x, size, alignCorners, halfPixelCenters, y, executor);
    }
}
} // namespace l0op
