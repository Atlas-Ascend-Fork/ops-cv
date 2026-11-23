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
#include <limits>
#include <set>
#include <string>
#include <vector>
#include "op_api/error_report_util.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/sorted_nms_tiling_data.h"

namespace optiling {
namespace {
constexpr size_t WORKSPACE_NUM = 1;
constexpr int64_t INPUT_BOXES = 0;
constexpr int64_t INPUT_SORTED_SCORES = 1;
constexpr int64_t INPUT_INDICES = 2;
constexpr int64_t INPUT_MAX_OUTPUT_SIZE = 3;
constexpr int64_t INPUT_IOU_THRESHOLD = 4;
constexpr int64_t INPUT_SCORE_THRESHOLD = 5;
constexpr int64_t ATTR_OFFSET = 0;
constexpr int64_t BOX_RANK = 2;
constexpr int64_t BOX_COORDS = 4;
constexpr int64_t MASK_BITS = 32;
constexpr int64_t PAIR_MASK_WORDS_PER_CORE = 1024;
constexpr int64_t MULTI_CORE_MIN_BOXES = 1025;
constexpr int64_t MULTI_CORE_MAX_BOXES = 8192;
constexpr int32_t STRATEGY_SINGLE_CORE = 0;
constexpr int32_t STRATEGY_PAIRWISE_MASK = 1;
constexpr size_t WORK_CONTROL_NUM = 2U;
constexpr uint64_t UB_BLOCK_SIZE = 32U;
constexpr uint64_t SIMT_DATA_CACHE_RESERVE = 128U * 1024U;
constexpr int64_t MAX_BOXES_NUM = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
} // namespace

static uint64_t AlignUbBytes(uint64_t bytes) { return (bytes + UB_BLOCK_SIZE - 1U) / UB_BLOCK_SIZE * UB_BLOCK_SIZE; }

static bool IsScalarOrSingleElement(const gert::Shape& shape)
{
    return shape.IsScalar() || (shape.GetDimNum() == 1U && shape.GetDim(0) == 1);
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context, int64_t boxesNum, int32_t strategy,
                                    int64_t maskWordNum)
{
    size_t* workspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    OP_CHECK_IF(boxesNum < 0 || boxesNum > MAX_BOXES_NUM,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "boxes", std::to_string(boxesNum).c_str(),
                    ("the first dimension of boxes must be in [0, " + std::to_string(MAX_BOXES_NUM) + "]").c_str()),
                return ge::GRAPH_FAILED);
    constexpr size_t MAX_WORKSPACE_INT32S = std::numeric_limits<size_t>::max() / sizeof(int32_t);
    const size_t boxesNumSize = static_cast<size_t>(boxesNum);
    size_t userWorkspaceInt32s = 0;
    if (strategy == STRATEGY_PAIRWISE_MASK) {
        const size_t maskWordNumSize = static_cast<size_t>(maskWordNum);
        OP_CHECK_IF(
            maskWordNumSize != 0 && boxesNumSize > MAX_WORKSPACE_INT32S / maskWordNumSize,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "boxes", std::to_string(boxesNum).c_str(),
                                                  "the pairwise mask workspace overflows"),
            return ge::GRAPH_FAILED);
        const size_t pairMaskWords = boxesNumSize * maskWordNumSize;
        OP_CHECK_IF(
            maskWordNumSize > MAX_WORKSPACE_INT32S - WORK_CONTROL_NUM ||
                pairMaskWords > MAX_WORKSPACE_INT32S - WORK_CONTROL_NUM - maskWordNumSize,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "boxes", std::to_string(boxesNum).c_str(),
                                                  "the pairwise mask workspace overflows"),
            return ge::GRAPH_FAILED);
        userWorkspaceInt32s = WORK_CONTROL_NUM + maskWordNumSize + pairMaskWords;
    } else {
        OP_CHECK_IF(
            boxesNumSize > MAX_WORKSPACE_INT32S - WORK_CONTROL_NUM,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "boxes", std::to_string(boxesNum).c_str(),
                                                  "the workspace size overflows"),
            return ge::GRAPH_FAILED);
        userWorkspaceInt32s = WORK_CONTROL_NUM + boxesNumSize;
    }
    const size_t userWorkspaceSize = userWorkspaceInt32s * sizeof(int32_t);
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    OP_CHECK_IF(userWorkspaceSize > std::numeric_limits<size_t>::max() - systemWorkspaceSize,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "boxes", std::to_string(boxesNum).c_str(),
                                                      "the workspace size overflows"),
                return ge::GRAPH_FAILED);
    workspace[0] = systemWorkspaceSize + userWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDesc(gert::TilingContext* context)
{
    const std::set<ge::DataType> dataDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT};
    const auto* boxesDesc = context->GetInputDesc(INPUT_BOXES);
    const auto* scoresDesc = context->GetInputDesc(INPUT_SORTED_SCORES);
    const auto* indicesDesc = context->GetInputDesc(INPUT_INDICES);
    const auto* maxOutputDesc = context->GetInputDesc(INPUT_MAX_OUTPUT_SIZE);
    const auto* iouDesc = context->GetInputDesc(INPUT_IOU_THRESHOLD);
    const auto* scoreThrDesc = context->GetInputDesc(INPUT_SCORE_THRESHOLD);
    OP_CHECK_IF(boxesDesc == nullptr || scoresDesc == nullptr || indicesDesc == nullptr || maxOutputDesc == nullptr ||
                    iouDesc == nullptr || scoreThrDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(),
                    "boxes, sorted_scores, input_indices, max_output_size, iou_threshold and score_threshold",
                    "the input desc is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(dataDtypes.count(boxesDesc->GetDataType()) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "boxes",
                                          Ops::Base::ToString(boxesDesc->GetDataType()).c_str(), "float16 or float32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dataDtypes.count(scoresDesc->GetDataType()) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "sorted_scores",
                                          Ops::Base::ToString(scoresDesc->GetDataType()).c_str(), "float16 or float32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dataDtypes.count(iouDesc->GetDataType()) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "iou_threshold",
                                          Ops::Base::ToString(iouDesc->GetDataType()).c_str(), "float16 or float32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dataDtypes.count(scoreThrDesc->GetDataType()) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "score_threshold",
                                  Ops::Base::ToString(scoreThrDesc->GetDataType()).c_str(), "float16 or float32"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "input_indices",
                                          Ops::Base::ToString(indicesDesc->GetDataType()).c_str(), "int32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxOutputDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "max_output_size",
                                          Ops::Base::ToString(maxOutputDesc->GetDataType()).c_str(), "int32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(boxesDesc->GetDataType() != iouDesc->GetDataType(),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "boxes and iou_threshold",
                                                       (Ops::Base::ToString(boxesDesc->GetDataType()) + " and " +
                                                        Ops::Base::ToString(iouDesc->GetDataType()))
                                                           .c_str(),
                                                       "boxes and iou_threshold must share one dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(scoresDesc->GetDataType() != scoreThrDesc->GetDataType(),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "sorted_scores and score_threshold",
                                                       (Ops::Base::ToString(scoresDesc->GetDataType()) + " and " +
                                                        Ops::Base::ToString(scoreThrDesc->GetDataType()))
                                                           .c_str(),
                                                       "sorted_scores and score_threshold must share one dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetOffsetAttr(gert::TilingContext* context, int32_t* offset)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* offsetPtr = attrs->GetAttrPointer<int64_t>(ATTR_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetPtr);
    OP_CHECK_IF(
        *offsetPtr != 0 && *offsetPtr != 1,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "offset", std::to_string(*offsetPtr).c_str(), "0 or 1"),
        return ge::GRAPH_FAILED);
    *offset = static_cast<int32_t>(*offsetPtr);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SortedNMSTilingFunc(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter SortedNMSTilingFunc");
    OP_CHECK_IF(CheckInputDesc(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "check input desc failed"), return ge::GRAPH_FAILED);

    auto boxesShapePtr = context->GetInputShape(INPUT_BOXES);
    auto scoresShapePtr = context->GetInputShape(INPUT_SORTED_SCORES);
    auto indicesShapePtr = context->GetInputShape(INPUT_INDICES);
    auto maxOutputShapePtr = context->GetInputShape(INPUT_MAX_OUTPUT_SIZE);
    auto iouShapePtr = context->GetInputShape(INPUT_IOU_THRESHOLD);
    auto scoreThrShapePtr = context->GetInputShape(INPUT_SCORE_THRESHOLD);
    OP_CHECK_IF(boxesShapePtr == nullptr || scoresShapePtr == nullptr || indicesShapePtr == nullptr ||
                    maxOutputShapePtr == nullptr || iouShapePtr == nullptr || scoreThrShapePtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(),
                    "boxes, sorted_scores, input_indices, max_output_size, iou_threshold and score_threshold",
                    "the input shape is nullptr"),
                return ge::GRAPH_FAILED);

    auto boxesShape = boxesShapePtr->GetStorageShape();
    auto scoresShape = scoresShapePtr->GetStorageShape();
    auto indicesShape = indicesShapePtr->GetStorageShape();
    auto maxOutputShape = maxOutputShapePtr->GetStorageShape();
    auto iouShape = iouShapePtr->GetStorageShape();
    auto scoreThrShape = scoreThrShapePtr->GetStorageShape();
    OP_CHECK_IF(boxesShape.GetDimNum() != BOX_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "boxes",
                                             (std::to_string(boxesShape.GetDimNum()) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        boxesShape.GetDim(1) != ge::UNKNOWN_DIM && boxesShape.GetDim(1) != BOX_COORDS,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "boxes", Ops::Base::ToString(boxesShape).c_str(),
                                              "the second dimension of boxes must be 4"),
        return ge::GRAPH_FAILED);
    int64_t boxesNum = boxesShape.GetDim(0);
    OP_CHECK_IF(
        boxesNum < 0 || boxesNum > MAX_BOXES_NUM,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "boxes", std::to_string(boxesNum).c_str(),
            ("the first dimension of boxes must be in [0, " + std::to_string(MAX_BOXES_NUM) + "] for tiling").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(scoresShape.GetDimNum() != 1 || scoresShape.GetDim(0) != boxesNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "sorted_scores", Ops::Base::ToString(scoresShape).c_str(),
                    ("sorted_scores must have shape [boxes_num], got boxes_num " + std::to_string(boxesNum)).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesShape.GetDimNum() != 1 || indicesShape.GetDim(0) != boxesNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "input_indices", Ops::Base::ToString(indicesShape).c_str(),
                    ("input_indices must have shape [boxes_num], got boxes_num " + std::to_string(boxesNum)).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsScalarOrSingleElement(maxOutputShape),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "max_output_size",
                                                      Ops::Base::ToString(maxOutputShape).c_str(),
                                                      "the shape of max_output_size must be scalar or [1]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsScalarOrSingleElement(iouShape),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "iou_threshold",
                                                      Ops::Base::ToString(iouShape).c_str(),
                                                      "the shape of iou_threshold must be scalar or [1]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsScalarOrSingleElement(scoreThrShape),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "score_threshold",
                                                      Ops::Base::ToString(scoreThrShape).c_str(),
                                                      "the shape of score_threshold must be scalar or [1]"),
                return ge::GRAPH_FAILED);
    const int32_t strategy = boxesNum >= MULTI_CORE_MIN_BOXES && boxesNum <= MULTI_CORE_MAX_BOXES ?
                                 STRATEGY_PAIRWISE_MASK :
                                 STRATEGY_SINGLE_CORE;
    const int64_t maskWordNum = strategy == STRATEGY_PAIRWISE_MASK ? (boxesNum + MASK_BITS - 1) / MASK_BITS : 0;
    OP_CHECK_IF(SetWorkspace(context, boxesNum, strategy, maskWordNum) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "set workspace failed"), return ge::GRAPH_FAILED);

    SortedNMSTilingData* tiling = context->GetTilingData<SortedNMSTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SortedNMSTilingData), 0, sizeof(SortedNMSTilingData)) != EOK,
        error_report::ReportTilingStageError(context->GetNodeName(), "memset_s failed to initialize the tiling data"),
        return ge::GRAPH_FAILED);

    tiling->boxesNum = boxesNum;
    OP_CHECK_IF(GetOffsetAttr(context, &tiling->offset) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "get offset failed"), return ge::GRAPH_FAILED);
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t physicalCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(
        physicalCoreNum <= 0,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "PlatformAscendC::GetCoreNumAiv returned " + std::to_string(physicalCoreNum)),
        return ge::GRAPH_FAILED);
    if (strategy == STRATEGY_PAIRWISE_MASK) {
        const auto* boxesDesc = context->GetInputDesc(INPUT_BOXES);
        OP_CHECK_NULL_WITH_CONTEXT(context, boxesDesc);
        if (boxesDesc->GetDataType() == ge::DT_FLOAT16) {
            const uint64_t boxesBytes = AlignUbBytes(static_cast<uint64_t>(boxesNum) * BOX_COORDS * sizeof(uint16_t));
            const uint64_t areasBytes = AlignUbBytes(static_cast<uint64_t>(boxesNum) * sizeof(float));
            uint64_t ubSize = 0;
            ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
            OP_CHECK_IF(
                ubSize <= SIMT_DATA_CACHE_RESERVE,
                error_report::ReportTilingStageError(
                    context->GetNodeName(), "PlatformAscendC::GetCoreMemSize(UB) returned " + std::to_string(ubSize) +
                                                ", which cannot reserve the SIMT Data Cache of " +
                                                std::to_string(SIMT_DATA_CACHE_RESERVE)),
                return ge::GRAPH_FAILED);
            const uint64_t localMemorySize = ubSize - SIMT_DATA_CACHE_RESERVE;
            OP_CHECK_IF(localMemorySize > std::numeric_limits<uint32_t>::max(),
                        error_report::ReportTilingStageError(
                            context->GetNodeName(),
                            "the local memory size " + std::to_string(localMemorySize) + " exceeds the uint32 range"),
                        return ge::GRAPH_FAILED);
            tiling->useLocalBoxes = boxesBytes <= localMemorySize && areasBytes <= localMemorySize - boxesBytes ? 1 : 0;
            OP_CHECK_IF(context->SetLocalMemorySize(static_cast<uint32_t>(localMemorySize)) != ge::GRAPH_SUCCESS,
                        error_report::ReportTilingStageError(
                            context->GetNodeName(),
                            "SetLocalMemorySize failed for localMemorySize=" + std::to_string(localMemorySize)),
                        return ge::GRAPH_FAILED);
        }
    }
    int64_t requiredCoreNum = 1;
    if (strategy == STRATEGY_PAIRWISE_MASK) {
        const int64_t pairMaskWords = boxesNum * maskWordNum;
        requiredCoreNum = (pairMaskWords + PAIR_MASK_WORDS_PER_CORE - 1) / PAIR_MASK_WORDS_PER_CORE;
    }
    tiling->coreNum = static_cast<int32_t>(std::min(requiredCoreNum, physicalCoreNum));
    context->SetBlockDim(static_cast<uint32_t>(tiling->coreNum));
    if (strategy == STRATEGY_PAIRWISE_MASK) {
        context->SetScheduleMode(1);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSortedNMS([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SortedNMSCompileInfo {};

IMPL_OP_OPTILING(SortedNMS).Tiling(SortedNMSTilingFunc).TilingParse<SortedNMSCompileInfo>(TilingParseForSortedNMS);
} // namespace optiling
