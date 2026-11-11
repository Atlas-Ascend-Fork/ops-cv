/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "../../op_kernel/arch35/decode_bbox_v2_tiling_struct.h"
#include "../../op_kernel/arch35/decode_bbox_v2_struct.h"
#include "decode_bbox_v2_tiling_arch35.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>

namespace optiling {

namespace {

const std::set<ge::DataType> kSupportedDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT};
const std::set<ge::Format> kSupportedFormats = {ge::FORMAT_ND};

constexpr int64_t kMaxRank = 2;
constexpr int64_t kScalesLen = 4;
constexpr float kDecodeClipMin = 0.0f;
constexpr float kDecodeClipMax = 10.0f;

constexpr int64_t kElemsPerBox = DECODE_BBOX_V2_ELEMS_PER_BOX;
constexpr int64_t kMinTilingBits = DECODE_BBOX_V2_MIN_TILING_BITS;
constexpr int64_t kElemAlignFactor = DECODE_BBOX_V2_ELEM_ALIGN_FACTOR;
constexpr int64_t kAlign256 = DECODE_BBOX_V2_ALIGN_256;
constexpr int64_t kReservedUb = DECODE_BBOX_V2_RESERVED_UB;
constexpr int64_t kNumIoBufs = 3;

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
inline int64_t CeilAlign(int64_t v, int64_t f) { return CeilDiv(v, f) * f; }

ge::graphStatus CheckInputs(gert::TilingContext* ctx, int64_t& dim0, bool& reversedBox, float scales[4],
                            float& decodeClip)
{
    const char* nodeName = (ctx != nullptr) ? ctx->GetNodeName() : "nil";

    // 1. dtype check
    auto boxesDesc = ctx->GetInputDesc(0);
    auto anchorsDesc = ctx->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, boxesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, anchorsDesc);
    ge::DataType dtBoxes = boxesDesc->GetDataType();
    ge::DataType dtAnchors = anchorsDesc->GetDataType();
    if (kSupportedDtypes.find(dtBoxes) == kSupportedDtypes.end() ||
        kSupportedDtypes.find(dtAnchors) == kSupportedDtypes.end()) {
        OP_LOGE(nodeName, "inputs dtype not in {FP16, FP32}: boxes=%s anchors=%s",
                ge::TypeUtils::DataTypeToSerialString(dtBoxes).c_str(),
                ge::TypeUtils::DataTypeToSerialString(dtAnchors).c_str());
        return ge::GRAPH_FAILED;
    }
    if (dtBoxes != dtAnchors) {
        OP_LOGE(nodeName, "boxes/anchors dtype mismatch: %s vs %s",
                ge::TypeUtils::DataTypeToSerialString(dtBoxes).c_str(),
                ge::TypeUtils::DataTypeToSerialString(dtAnchors).c_str());
        return ge::GRAPH_FAILED;
    }

    // 2. format check
    for (size_t i = 0; i < 2; i++) {
        auto desc = ctx->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, desc);
        ge::Format fmt = desc->GetStorageFormat();
        if (kSupportedFormats.find(fmt) == kSupportedFormats.end()) {
            OP_LOGE(nodeName, "input[%zu] format not in {ND}: %s", i, ge::TypeUtils::FormatToSerialString(fmt).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    auto outDesc = ctx->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, outDesc);
    ge::Format outFmt = outDesc->GetStorageFormat();
    if (kSupportedFormats.find(outFmt) == kSupportedFormats.end()) {
        OP_LOGE(nodeName, "output format not in {ND}: %s", ge::TypeUtils::FormatToSerialString(outFmt).c_str());
        return ge::GRAPH_FAILED;
    }

    // 3. rank check
    auto boxesShape = ctx->GetInputShape(0);
    auto anchorsShape = ctx->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, boxesShape);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, anchorsShape);
    if (boxesShape->GetStorageShape().GetDimNum() != static_cast<size_t>(kMaxRank) ||
        anchorsShape->GetStorageShape().GetDimNum() != static_cast<size_t>(kMaxRank)) {
        OP_LOGE(nodeName, "rank != 2: boxes rank=%zu anchors rank=%zu", boxesShape->GetStorageShape().GetDimNum(),
                anchorsShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    // 4. attr check
    auto attrs = ctx->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, attrs);

    const auto* scalesVec = attrs->GetListFloat(0);
    int64_t scalesLen = (scalesVec != nullptr) ? static_cast<int64_t>(scalesVec->GetSize()) : kScalesLen;
    if (scalesLen != kScalesLen) {
        OP_LOGE(nodeName, "scales length must be 4, got %ld", scalesLen);
        return ge::GRAPH_FAILED;
    }
    if (scalesVec != nullptr) {
        const float* data = scalesVec->GetData();
        for (int64_t i = 0; i < kScalesLen; i++) {
            if (data[i] == 0.0f) {
                OP_LOGE(nodeName, "scales[%ld] must be non-zero, got %f", i, data[i]);
                return ge::GRAPH_FAILED;
            }
            scales[i] = data[i];
        }
    } else {
        scales[0] = 1.0f;
        scales[1] = 1.0f;
        scales[2] = 1.0f;
        scales[3] = 1.0f;
    }

    const float* clipPtr = attrs->GetFloat(1);
    decodeClip = (clipPtr != nullptr) ? *clipPtr : 0.0f;
    if (std::isnan(decodeClip) || decodeClip < kDecodeClipMin || decodeClip > kDecodeClipMax) {
        OP_LOGE(nodeName, "decode_clip must be in [0.0, 10.0], got %f", decodeClip);
        return ge::GRAPH_FAILED;
    }

    const bool* rbPtr = attrs->GetBool(2);
    reversedBox = (rbPtr != nullptr) ? *rbPtr : false;

    // 5. shape check
    int64_t b0 = boxesShape->GetStorageShape().GetDim(0);
    int64_t b1 = boxesShape->GetStorageShape().GetDim(1);
    int64_t a0 = anchorsShape->GetStorageShape().GetDim(0);
    int64_t a1 = anchorsShape->GetStorageShape().GetDim(1);
    if (b0 != a0 || b1 != a1) {
        OP_LOGE(nodeName, "boxes/anchors shape mismatch: boxes=[%ld,%ld] anchors=[%ld,%ld]", b0, b1, a0, a1);
        return ge::GRAPH_FAILED;
    }
    if (reversedBox) {
        if (b0 != 4) {
            OP_LOGE(nodeName, "reversed_box=true requires shape[0]==4, got %ld", b0);
            return ge::GRAPH_FAILED;
        }
        dim0 = b1;
    } else {
        if (b1 != 4) {
            OP_LOGE(nodeName, "reversed_box=false requires shape[-1]==4, got %ld", b1);
            return ge::GRAPH_FAILED;
        }
        dim0 = b0;
    }
    return ge::GRAPH_SUCCESS;
}

struct UbSplit {
    int64_t ubFormer;
};

UbSplit ComputeUbSplit(int64_t totalUb, int64_t elemBytes, int64_t numCalcBufs)
{
    int64_t ubAvailable = totalUb / 2 - kReservedUb;
    int64_t perBoxBytes = kNumIoBufs * kElemsPerBox * elemBytes +
                          numCalcBufs * kElemsPerBox * static_cast<int64_t>(sizeof(float));
    int64_t alignFactor = kAlign256 / (kElemsPerBox * elemBytes);
    int64_t maxBoxNum = ubAvailable / perBoxBytes;
    int64_t ubFormer = (maxBoxNum / alignFactor) * alignFactor;
    ubFormer = std::max(ubFormer, alignFactor);
    return {ubFormer};
}

struct MultiCoreSplit {
    int32_t coreNum;
    int64_t blockFormer;
    int64_t blockNum;
};

MultiCoreSplit ComputeMultiCoreSplit(int64_t dim0, int64_t elemBytes, int64_t availableCoreNum)
{
    if (dim0 == 0) {
        return {/*coreNum=*/1, /*blockFormer=*/0, /*blockNum=*/0};
    }
    int64_t minDtypeBits = kElemsPerBox * elemBytes * 8;
    int64_t coreNum = CeilDiv(dim0 * minDtypeBits, kMinTilingBits);
    coreNum = std::min(coreNum, availableCoreNum);
    coreNum = std::max(coreNum, int64_t(1));
    int64_t blockFormer = CeilAlign(CeilDiv(dim0, coreNum), kElemAlignFactor);
    int64_t blockNum = CeilDiv(dim0, blockFormer);
    // 512 对齐可能使实际块数少于估算核数（fp32 尤其），clamp 到实际块数消除空转核；
    coreNum = std::min(coreNum, blockNum);
    return {static_cast<int32_t>(coreNum), blockFormer, blockNum};
}

void FillAndLogTilingData(DecodeBboxV2TilingData& td, int64_t dim0, bool reversedBox, const MultiCoreSplit& mc,
                          const UbSplit& ub, const float scales[4], float decodeClip)
{
    td.dim0 = dim0;
    td.coreNum = mc.coreNum;
    td.blockFormer = mc.blockFormer;
    td.blockNum = mc.blockNum;

    td.ubFormer = ub.ubFormer;
    if (ub.ubFormer == 0) {
        td.ubLoopOfFormerBlock = 0;
        td.ubTailOfFormerBlock = 0;
        td.ubLoopOfTailBlock = 0;
        td.ubTailOfTailBlock = 0;
    } else {
        td.ubLoopOfFormerBlock = (mc.blockFormer + ub.ubFormer - 1) / ub.ubFormer;
        td.ubTailOfFormerBlock = mc.blockFormer - (td.ubLoopOfFormerBlock - 1) * ub.ubFormer;
        int64_t blockTail = (mc.blockNum > 0) ? (dim0 - (mc.blockNum - 1) * mc.blockFormer) : 0;
        td.ubLoopOfTailBlock = (blockTail + ub.ubFormer - 1) / ub.ubFormer;
        td.ubTailOfTailBlock = blockTail - (td.ubLoopOfTailBlock - 1) * ub.ubFormer;
    }

    for (int i = 0; i < 4; i++) {
        td.scales[i] = scales[i];
        td.invScales[i] = (scales[i] != 0.0f) ? (1.0f / scales[i]) : 0.0f;
    }
    td.decodeClip = decodeClip;
    td.halfVal = 0.5f;

    OP_LOGI("[DecodeBboxV2]", "dim0=%ld, coreNum=%d, blockFormer=%ld, blockNum=%ld, ubFormer=%ld", td.dim0, td.coreNum,
            td.blockFormer, td.blockNum, td.ubFormer);
    OP_LOGI("[DecodeBboxV2]", "ubLoop: former=%ld/%ld, tail=%ld/%ld", td.ubLoopOfFormerBlock, td.ubTailOfFormerBlock,
            td.ubLoopOfTailBlock, td.ubTailOfTailBlock);
    OP_LOGI("[DecodeBboxV2]", "reversedBox=%d, decodeClip=%f, scales=[%f,%f,%f,%f]", int(reversedBox), td.decodeClip,
            td.scales[0], td.scales[1], td.scales[2], td.scales[3]);
}

} // namespace

ge::graphStatus TilingFuncDecodeBboxV2(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC plat(platformInfo);
    uint64_t totalUb = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUb);
    int64_t availableCoreNum = static_cast<int64_t>(plat.GetCoreNumAiv());

    if (totalUb == 0) {
        OP_LOGE(context->GetNodeName(), "invalid platform query: total UB size == 0");
        return ge::GRAPH_FAILED;
    }
    if (availableCoreNum <= 0) {
        OP_LOGE(context->GetNodeName(), "invalid platform query: available core num = %ld", availableCoreNum);
        return ge::GRAPH_FAILED;
    }

    int64_t dim0 = 0;
    bool reversedBox = false;
    float scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float decodeClip = 0.0f;
    if (CheckInputs(context, dim0, reversedBox, scales, decodeClip) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    int64_t elemBytes;
    int64_t numCalcBufs;
    if (dtype == ge::DT_FLOAT16) {
        elemBytes = 2;
        numCalcBufs = 3;
    } else {
        elemBytes = 4;
        numCalcBufs = 0;
    }

    MultiCoreSplit mc;
    UbSplit ub;
    if (dim0 == 0) {
        mc = {/*coreNum=*/1, /*blockFormer=*/0, /*blockNum=*/0};
        ub = {/*ubFormer=*/0};
    } else {
        ub = ComputeUbSplit(static_cast<int64_t>(totalUb), elemBytes, numCalcBufs);
        mc = ComputeMultiCoreSplit(dim0, elemBytes, availableCoreNum);
    }

    DecodeBboxV2TilingData* td = context->GetTilingData<DecodeBboxV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    *td = DecodeBboxV2TilingData{};
    FillAndLogTilingData(*td, dim0, reversedBox, mc, ub, scales, decodeClip);

    context->SetBlockDim(static_cast<uint32_t>(td->coreNum));
    uint64_t tilingKey = reversedBox ? DECODE_BBOX_V2_LAYOUT_F4N : DECODE_BBOX_V2_LAYOUT_N4;
    context->SetTilingKey(tilingKey);

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

// Empty TilingParse placeholder: occupies the TilingParse slot so the framework's
// DefaultImpl registration (AutoTilingParser, requires compileInfo["_pattern"])
static ge::graphStatus TilingParseForDecodeBboxV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct DecodeBboxV2CompileInfo {};

IMPL_OP_OPTILING(DecodeBboxV2)
    .Tiling(TilingFuncDecodeBboxV2)
    .TilingParse<DecodeBboxV2CompileInfo>(TilingParseForDecodeBboxV2);

} // namespace optiling
