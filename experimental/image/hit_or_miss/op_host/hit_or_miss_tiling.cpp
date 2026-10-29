/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "hit_or_miss_utils.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/hit_or_miss_tiling_data.h"
#include "../op_kernel/hit_or_miss_tiling_key.h"

namespace optiling {

struct HitOrMissCompileInfo {};
namespace {
constexpr int32_t INPUT_INDEX = 0;
constexpr int32_t KERNEL_INDEX = 1;
constexpr int32_t AXES_ATTR_INDEX = 0;
constexpr int32_t ANCHOR_ATTR_INDEX = 1;
constexpr int32_t BORDER_VALUE_ATTR_INDEX = 2;

// 每个核负责独立的连续区间，起始位置按 32 字节对齐，尾块由 DataCopyPad 处理。
constexpr int32_t DEFAULT_CORE_NUM = 8;
constexpr int64_t OUTPUT_ALIGN_BYTES = 32;

using hit_or_miss::SafeMultiply;

ge::graphStatus ReportFailure(const gert::TilingContext* context, const char* message)
{
    OP_LOGE(context == nullptr ? "HitOrMiss" : context->GetNodeName(), "%s", message);
    return ge::GRAPH_FAILED;
}
} // namespace

static ge::graphStatus HitOrMissTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("HitOrMiss", "context is null");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* inputStorageShape = context->GetInputShape(INPUT_INDEX);

    const gert::StorageShape* kernelStorageShape = context->GetInputShape(KERNEL_INDEX);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();

    HitOrMissTilingData* tilingData = context->GetTilingData<HitOrMissTilingData>();

    if (inputStorageShape == nullptr || kernelStorageShape == nullptr || attrs == nullptr || tilingData == nullptr) {
        return ReportFailure(context, "input shape, kernel shape, attrs, or tiling data is null");
    }

    const gert::Shape& inputShape = inputStorageShape->GetStorageShape();
    const gert::Shape& kernelShape = kernelStorageShape->GetStorageShape();
    const auto* axesAttr = attrs->GetListInt(AXES_ATTR_INDEX);
    const auto* anchorAttr = attrs->GetListInt(ANCHOR_ATTR_INDEX);
    const bool* borderValueAttr = attrs->GetAttrPointer<bool>(BORDER_VALUE_ATTR_INDEX);
    if (borderValueAttr == nullptr) {
        return ReportFailure(context, "border_value must be present");
    }

    hit_or_miss::CanonicalParams params;
    const char* error = hit_or_miss::CanonicalizeParams(
        inputShape, kernelShape, axesAttr == nullptr ? nullptr : axesAttr->GetData(),
        axesAttr == nullptr ? 0 : axesAttr->GetSize(), anchorAttr == nullptr ? nullptr : anchorAttr->GetData(),
        anchorAttr == nullptr ? 0 : anchorAttr->GetSize(), false, params);
    if (error != nullptr) {
        return ReportFailure(context, error);
    }

    const int32_t rank = static_cast<int32_t>(inputShape.GetDimNum());
    const int64_t axis0 = params.axis0;
    const int64_t axis1 = params.axis1;
    const int64_t kernelH64 = params.kernelH;
    const int64_t kernelW64 = params.kernelW;
    const int64_t anchorH = params.anchorH;
    const int64_t anchorW = params.anchorW;

    int64_t totalLength = 1;
    int64_t runningStride = 1;
    int64_t axis0Stride = 0;
    int64_t axis1Stride = 0;
    int64_t axis0Dim = 0;
    int64_t axis1Dim = 0;

    for (int32_t index = rank - 1; index >= 0; --index) {
        const int64_t dim = inputShape.GetDim(index);
        if (index == axis0) {
            axis0Stride = runningStride;
            axis0Dim = dim;
        }
        if (index == axis1) {
            axis1Stride = runningStride;
            axis1Dim = dim;
        }

        int64_t nextStride = 0;
        if (!SafeMultiply(runningStride, dim, nextStride)) {
            return ReportFailure(context, "input element count overflows int64");
        }
        runningStride = nextStride;
    }
    totalLength = runningStride;

    const int64_t alignedBlockCount = totalLength / OUTPUT_ALIGN_BYTES +
                                      ((totalLength % OUTPUT_ALIGN_BYTES) != 0 ? 1 : 0);
    const int32_t usedCoreNum = static_cast<int32_t>(std::min<int64_t>(DEFAULT_CORE_NUM, alignedBlockCount));
    const int64_t blocksPerCore = (alignedBlockCount + static_cast<int64_t>(usedCoreNum) - 1) /
                                  static_cast<int64_t>(usedCoreNum);

    int64_t blockLength = 0;
    if (!SafeMultiply(blocksPerCore, OUTPUT_ALIGN_BYTES, blockLength)) {
        return ReportFailure(context, "aligned block length overflows int64");
    }

    tilingData->totalLength = totalLength;
    tilingData->blockLength = blockLength;
    tilingData->axis0Stride = axis0Stride;
    tilingData->axis1Stride = axis1Stride;
    tilingData->axis0Dim = axis0Dim;
    tilingData->axis1Dim = axis1Dim;
    tilingData->kernelH = static_cast<int32_t>(kernelH64);
    tilingData->kernelW = static_cast<int32_t>(kernelW64);
    tilingData->anchorH = static_cast<int32_t>(anchorH);
    tilingData->anchorW = static_cast<int32_t>(anchorW);
    tilingData->borderValue = *borderValueAttr ? 1 : 0;
    tilingData->usedCoreNum = usedCoreNum;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(HIT_OR_MISS_TILING_KEY_DEFAULT);
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ReportFailure(context, "workspace size buffer is null");
    }
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHitOrMiss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HitOrMiss).Tiling(HitOrMissTilingFunc).TilingParse<HitOrMissCompileInfo>(TilingParseForHitOrMiss);
} // namespace optiling
