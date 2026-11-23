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
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "op_api/error_report_util.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/combined_non_max_suppression_tiling_data.h"

namespace optiling {
namespace {
constexpr int32_t BOXES_INDEX = 0;
constexpr int32_t SCORES_INDEX = 1;
constexpr int32_t MAX_PER_CLASS_INDEX = 2;
constexpr int32_t MAX_TOTAL_INDEX = 3;
constexpr int32_t IOU_THRESHOLD_INDEX = 4;
constexpr int32_t SCORE_THRESHOLD_INDEX = 5;
constexpr int32_t ATTR_CLIP_BOXES_INDEX = 1;
constexpr int32_t MAX_OUTPUT_SIZE = 1000;
constexpr int32_t MAX_CLASSES = 200;
constexpr uint64_t ALIGN_BYTES = 32;
constexpr uint32_t WORKSPACE_COUNT = 1;
constexpr int32_t SCHEDULE_MODE = 1;
constexpr uint64_t SIMT_UB_RESERVE = 32 * 1024;
constexpr uint64_t FALLBACK_LOCAL_MEMORY_SIZE = 64 * 1024;
constexpr int32_t HOT_UB_MAX_BOXES = 8704;

struct CombinedNonMaxSuppressionCompileInfo {
    int32_t coreNum = 0;
    uint64_t sysWorkspaceSize = 0;
};

uint64_t AlignUp(uint64_t value) { return (value + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES; }

template <typename T>
ge::graphStatus ReadScalar(gert::TilingContext* context, int32_t index, const char* paramName, T& value)
{
    const gert::Tensor* tensor = context->GetInputTensor(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensor);
    const T* data = tensor->GetData<T>();
    OP_CHECK_IF(data == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), paramName, "null",
                                                      "scalar data of the input is null"),
                return ge::GRAPH_FAILED);
    value = data[0];
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateAndFill(gert::TilingContext* context, CombinedNonMaxSuppressionTilingData& tiling,
                                uint64_t& userWorkspaceSize)
{
    const gert::StorageShape* boxesStorage = context->GetInputShape(BOXES_INDEX);
    const gert::StorageShape* scoresStorage = context->GetInputShape(SCORES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresStorage);
    const gert::Shape& boxes = boxesStorage->GetStorageShape();
    const gert::Shape& scores = scoresStorage->GetStorageShape();
    OP_CHECK_IF(boxes.GetDimNum() != 4 || scores.GetDimNum() != 3,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "boxes and scores",
                    (std::to_string(boxes.GetDimNum()) + " and " + std::to_string(scores.GetDimNum())).c_str(),
                    "boxes must be 4D and scores must be 3D"),
                return ge::GRAPH_FAILED);

    const int64_t batch = boxes.GetDim(0);
    const int64_t numBoxes = boxes.GetDim(1);
    const int64_t boxClasses = boxes.GetDim(2);
    const int64_t classes = scores.GetDim(2);
    OP_CHECK_IF(batch <= 0 || numBoxes <= 0 || classes <= 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "boxes and scores",
                    (Ops::Base::ToString(boxes) + " and " + Ops::Base::ToString(scores)).c_str(),
                    "the batch, num_boxes and num_classes dimensions must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        classes > MAX_CLASSES,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "num_classes", std::to_string(classes).c_str(),
                                              ("num_classes must be <= " + std::to_string(MAX_CLASSES)).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(boxes.GetDim(3) != 4 || scores.GetDim(0) != batch || scores.GetDim(1) != numBoxes,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "boxes and scores",
                    (Ops::Base::ToString(boxes) + " and " + Ops::Base::ToString(scores)).c_str(),
                    "the last dimension of boxes must be 4 and the batch/num_boxes dimensions must match"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        boxClasses != 1 && boxClasses != classes,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "boxes", Ops::Base::ToString(boxes).c_str(),
                                              "the q dimension of boxes must be 1 or num_classes"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        numBoxes > std::numeric_limits<int32_t>::max(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "num_boxes", std::to_string(numBoxes).c_str(),
                                              "num_boxes exceeds the int32 range"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        numBoxes > std::numeric_limits<int64_t>::max() / batch / boxClasses / 4,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "num_boxes", std::to_string(numBoxes).c_str(),
                                              "the element count of boxes exceeds the int64 range"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(batch > std::numeric_limits<int32_t>::max(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "batch", std::to_string(batch).c_str(),
                                                      "batch exceeds the int32 range"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        batch > std::numeric_limits<int32_t>::max() / classes,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "batch and num_classes",
                                               (std::to_string(batch) + " and " + std::to_string(classes)).c_str(),
                                               "batch * num_classes exceeds the int32 range"),
        return ge::GRAPH_FAILED);

    int32_t maxPerClass = 0;
    int32_t maxTotal = 0;
    float iouThreshold = 0.0F;
    float scoreThreshold = 0.0F;
    OP_CHECK_IF(ReadScalar(context, MAX_PER_CLASS_INDEX, "max_output_size_per_class", maxPerClass) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "max_output_size_per_class", "unreadable",
                                                      "failed to read max_output_size_per_class"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadScalar(context, MAX_TOTAL_INDEX, "max_total_size", maxTotal) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "max_total_size", "unreadable",
                                                      "failed to read max_total_size"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadScalar(context, IOU_THRESHOLD_INDEX, "iou_threshold", iouThreshold) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "iou_threshold", "unreadable",
                                                      "failed to read iou_threshold"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadScalar(context, SCORE_THRESHOLD_INDEX, "score_threshold", scoreThreshold) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "score_threshold", "unreadable",
                                                      "failed to read score_threshold"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        maxPerClass <= 0 || maxPerClass > MAX_OUTPUT_SIZE || maxTotal <= 0 || maxTotal > MAX_OUTPUT_SIZE,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "max_output_size_per_class and max_total_size",
            (std::to_string(maxPerClass) + " and " + std::to_string(maxTotal)).c_str(),
            ("max_output_size_per_class and max_total_size must be in [1, " + std::to_string(MAX_OUTPUT_SIZE) + "]")
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!std::isfinite(iouThreshold) || iouThreshold < 0.0F,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "iou_threshold",
                                                      std::to_string(iouThreshold).c_str(),
                                                      "iou_threshold must be finite and nonnegative"),
                return ge::GRAPH_FAILED);

    const gert::StorageShape* outputStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputStorage);
    const gert::Shape& output = outputStorage->GetStorageShape();
    OP_CHECK_IF(
        output.GetDimNum() != 3 || output.GetDim(0) != batch || output.GetDim(2) != 4,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "nmsed_boxes", Ops::Base::ToString(output).c_str(),
            ("nmsed_boxes must have shape [batch, output_size, 4], got batch " + std::to_string(batch)).c_str()),
        return ge::GRAPH_FAILED);
    const int64_t outputSize = output.GetDim(1);
    OP_CHECK_IF(
        outputSize <= 0 || outputSize > maxTotal,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output_size", std::to_string(outputSize).c_str(),
                                              "output_size must be in [1, max_total_size]"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* scoresOutputStorage = context->GetOutputShape(1);
    const gert::StorageShape* classesOutputStorage = context->GetOutputShape(2);
    const gert::StorageShape* validDetectionsOutputStorage = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresOutputStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, classesOutputStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, validDetectionsOutputStorage);
    const gert::Shape& scoresOutput = scoresOutputStorage->GetStorageShape();
    const gert::Shape& classesOutput = classesOutputStorage->GetStorageShape();
    const gert::Shape& validDetectionsOutput = validDetectionsOutputStorage->GetStorageShape();
    OP_CHECK_IF(
        scoresOutput.GetDimNum() != 2 || scoresOutput.GetDim(0) != batch || scoresOutput.GetDim(1) != outputSize,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "nmsed_scores", Ops::Base::ToString(scoresOutput).c_str(),
            ("nmsed_scores must have shape [batch, output_size], got batch " + std::to_string(batch) +
             " and "
             "output_size " +
             std::to_string(outputSize))
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        classesOutput.GetDimNum() != 2 || classesOutput.GetDim(0) != batch || classesOutput.GetDim(1) != outputSize,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "nmsed_classes", Ops::Base::ToString(classesOutput).c_str(),
            ("nmsed_classes must have shape [batch, output_size], got batch " + std::to_string(batch) +
             " and "
             "output_size " +
             std::to_string(outputSize))
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(validDetectionsOutput.GetDimNum() != 1 || validDetectionsOutput.GetDim(0) != batch,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "valid_detections", Ops::Base::ToString(validDetectionsOutput).c_str(),
                    ("valid_detections must have shape [batch], got batch " + std::to_string(batch)).c_str()),
                return ge::GRAPH_FAILED);

    // A class cannot contribute more than outputSize candidates to the final merge.
    // Preserve its ordered prefix, including the existing deterministic tie break.
    maxPerClass = std::min(maxPerClass, static_cast<int32_t>(std::min(numBoxes, outputSize)));

    const auto* compileInfo = context->GetCompileInfo<CombinedNonMaxSuppressionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    const int64_t taskCount = batch * classes;
    const int32_t usedCoreNum = static_cast<int32_t>(std::min<int64_t>(compileInfo->coreNum, taskCount));
    OP_CHECK_IF(usedCoreNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "batch * num_classes",
                                                      std::to_string(taskCount).c_str(),
                                                      "the used core num derived from batch * num_classes must be "
                                                      "positive"),
                return ge::GRAPH_FAILED);

    const uint64_t selectedCount = static_cast<uint64_t>(taskCount) * static_cast<uint64_t>(maxPerClass);
    uint64_t offset = 0;
    tiling.selectedScoresOffset = offset;
    offset = AlignUp(offset + selectedCount * sizeof(float));
    tiling.selectedIndicesOffset = offset;
    offset = AlignUp(offset + selectedCount * sizeof(int32_t));
    tiling.selectedCountsOffset = offset;
    offset = AlignUp(offset + static_cast<uint64_t>(taskCount) * sizeof(int32_t));
    offset = (offset + CNMS_GM_ALIGN_BYTES - 1) / CNMS_GM_ALIGN_BYTES * CNMS_GM_ALIGN_BYTES;
    tiling.suppressedOffset = offset;
    const uint64_t suppressedStride = (static_cast<uint64_t>(numBoxes) + CNMS_GM_ALIGN_BYTES - 1) /
                                      CNMS_GM_ALIGN_BYTES * CNMS_GM_ALIGN_BYTES;
    offset += static_cast<uint64_t>(usedCoreNum) * suppressedStride;
    userWorkspaceSize = offset;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* clipBoxes = attrs->GetBool(ATTR_CLIP_BOXES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, clipBoxes);

    tiling.batchSize = static_cast<int32_t>(batch);
    tiling.numBoxes = static_cast<int32_t>(numBoxes);
    tiling.boxClasses = static_cast<int32_t>(boxClasses);
    tiling.numClasses = static_cast<int32_t>(classes);
    tiling.maxOutputPerClass = maxPerClass;
    tiling.maxTotalSize = maxTotal;
    tiling.outputSize = static_cast<int32_t>(outputSize);
    tiling.usedCoreNum = usedCoreNum;
    tiling.clipBoxes = *clipBoxes ? 1 : 0;
    tiling.iouThreshold = iouThreshold;
    tiling.scoreThreshold = scoreThreshold;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CombinedNonMaxSuppressionTiling(gert::TilingContext* context)
{
    CombinedNonMaxSuppressionTilingData tiling{};
    uint64_t userWorkspaceSize = 0;
    OP_CHECK_IF(ValidateAndFill(context, tiling, userWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "tiling validation failed"), return ge::GRAPH_FAILED);

    auto* tilingData = context->GetTilingData<CombinedNonMaxSuppressionTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    *tilingData = tiling;

    const auto* compileInfo = context->GetCompileInfo<CombinedNonMaxSuppressionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    size_t* workspaceSizes = context->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = static_cast<size_t>(compileInfo->sysWorkspaceSize + userWorkspaceSize);
    context->SetBlockDim(tiling.usedCoreNum);
    context->SetScheduleMode(SCHEDULE_MODE);

    // Keep at least 32 KB of UB outside the TPipe local-memory region for the
    // SIMT data cache, matching the Ascend950 Sort operator resource split.
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const platform_ascendc::PlatformAscendC platform(platformInfo);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize <= SIMT_UB_RESERVE,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "PlatformAscendC::GetCoreMemSize(UB) returned " + std::to_string(ubSize) +
                                        ", which cannot cover the SIMT reserve " + std::to_string(SIMT_UB_RESERVE)),
        return ge::GRAPH_FAILED);
    // The large-input fallback operates through GM and benefits directly from
    // a larger SIMT data cache. Its TPipe buffers need less than 64 KB, so keep
    // the rest of UB (192 KB on Ascend950) available to the cache.
    const uint64_t localMemorySize = tiling.numBoxes <= HOT_UB_MAX_BOXES ?
                                         ubSize - SIMT_UB_RESERVE :
                                         std::min(FALLBACK_LOCAL_MEMORY_SIZE, ubSize - SIMT_UB_RESERVE);
    const auto localMemoryStatus = context->SetLocalMemorySize(static_cast<uint32_t>(localMemorySize));
    OP_CHECK_IF(
        localMemoryStatus != ge::GRAPH_SUCCESS,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "SetLocalMemorySize failed for localMemorySize=" + std::to_string(localMemorySize)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CombinedNonMaxSuppressionTilingParse(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<CombinedNonMaxSuppressionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo->coreNum = platform.GetCoreNumAiv();
    compileInfo->sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(
        compileInfo->coreNum <= 0,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "PlatformAscendC::GetCoreNumAiv returned " + std::to_string(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(CombinedNonMaxSuppression)
    .TilingInputsDataDependency({MAX_PER_CLASS_INDEX, MAX_TOTAL_INDEX, IOU_THRESHOLD_INDEX, SCORE_THRESHOLD_INDEX})
    .Tiling(CombinedNonMaxSuppressionTiling)
    .TilingParse<CombinedNonMaxSuppressionCompileInfo>(CombinedNonMaxSuppressionTilingParse);

} // namespace optiling
