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
 * \file paste_sub_img_tiling_arch35.cpp
 * \brief Tiling implementation for paste_sub_img operator on arch35
 */
#include "paste_sub_img_tiling_arch35.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "exe_graph/runtime/runtime_attrs.h"

namespace optiling {

static constexpr uint64_t IDX_PATCH_IMG = 0;
static constexpr uint64_t IDX_PATCH_COORD = 1;
static constexpr uint64_t IDX_CORE_AREA_COORD = 2;
static constexpr uint64_t IDX_COMBINE_IMG = 3;
static constexpr int64_t EXPECTED_RANK_3D = 3;
static constexpr int64_t EXPECTED_RANK_1D = 1;
static constexpr int64_t COORD_LEN = 4;
static constexpr int64_t DIM_H = 0;
static constexpr int64_t DIM_W = 1;
static constexpr int64_t DIM_C = 2;
static constexpr int64_t PHYS_NODES = 2;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr uint64_t MAX_BUFFER_SIZE = 64 * 1024;
static constexpr int64_t RANK_MERGED = 3;
static constexpr float SCALE_MIN = 0.0f;
static constexpr float SCALE_MAX = 256.0f;
static constexpr uint64_t ATTR_IDX_SCALE = 0;

struct CoordBudget {
    int64_t sCy1, sCx1, dCy1, dCx1;
    int64_t activeH, activeW, activeC;
    int64_t patchBaseOffset, combineBaseOffset;
    int64_t patchStrideH, patchStrideW;
    int64_t combineStrideH, combineStrideW;
};

struct BufferResult {
    uint32_t bufferSize;
    uint32_t bufferSizeElements;
    uint32_t ubBlockSize;
    uint32_t ubBlockElements;
    uint32_t cacheLineSize;
    uint32_t cacheLineElements;
};

struct SplitAxisResult {
    uint8_t ubAxis;
    uint32_t ubFactor;
    uint64_t totalCount;
    uint8_t tilingKey;
};

struct MultiCoreSplitResult {
    uint64_t perCoreCount;
    uint64_t realCoreNum;
};

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

static inline bool IsComputeDtype(ge::DataType d)
{
    return d == ge::DT_UINT8 || d == ge::DT_FLOAT16 || d == ge::DT_FLOAT;
}

static bool ComputeCoordBudget(int64_t px1, int64_t py1, int64_t cx1, int64_t cy1, int64_t cx2, int64_t cy2,
                               float scale, int64_t patchH, int64_t patchW, int64_t patchC, int64_t combineH,
                               int64_t combineW, CoordBudget& out)
{
    double scaleD = static_cast<double>(scale);
    int64_t sCy1 = static_cast<int64_t>(static_cast<double>(cy1) * scaleD);
    int64_t sCy2 = static_cast<int64_t>(static_cast<double>(cy2) * scaleD);
    int64_t sCx1 = static_cast<int64_t>(static_cast<double>(cx1) * scaleD);
    int64_t sCx2 = static_cast<int64_t>(static_cast<double>(cx2) * scaleD);
    int64_t dCy1 = static_cast<int64_t>(static_cast<double>(cy1 + py1) * scaleD);
    int64_t dCx1 = static_cast<int64_t>(static_cast<double>(cx1 + px1) * scaleD);

    int64_t activeH = sCy2 - sCy1;
    int64_t activeW = sCx2 - sCx1;
    int64_t activeC = patchC;

    int64_t patchStrideH = patchW * activeC;
    int64_t combineStrideH = combineW * activeC;

    out.sCy1 = sCy1;
    out.sCx1 = sCx1;
    out.dCy1 = dCy1;
    out.dCx1 = dCx1;
    out.activeH = activeH;
    out.activeW = activeW;
    out.activeC = activeC;
    out.patchBaseOffset = sCy1 * patchStrideH + sCx1 * activeC;
    out.combineBaseOffset = dCy1 * combineStrideH + dCx1 * activeC;
    out.patchStrideH = patchStrideH;
    out.patchStrideW = activeC;
    out.combineStrideH = combineStrideH;
    out.combineStrideW = activeC;
    return true;
}

static void ComputeBuffer(uint64_t ubSize, uint32_t ubBlockSize, uint32_t cacheLineSize, uint32_t dtypeBytes,
                          BufferResult& out)
{
    uint64_t raw = std::min(ubSize / static_cast<uint64_t>(PHYS_NODES * DOUBLE_BUFFER), MAX_BUFFER_SIZE);
    out.bufferSize = static_cast<uint32_t>(raw & ~31UL);
    out.bufferSizeElements = out.bufferSize / dtypeBytes;
    out.ubBlockSize = ubBlockSize;
    out.ubBlockElements = ubBlockSize / dtypeBytes;
    out.cacheLineSize = cacheLineSize;
    out.cacheLineElements = cacheLineSize / dtypeBytes;
}

static bool FindSplitAxis(int64_t activeH, int64_t activeW, int64_t activeC, const BufferResult& buf, uint32_t coreNum,
                          SplitAxisResult& result)
{
    int64_t outShape[RANK_MERGED] = {activeH, activeW, activeC};
    int64_t alignedLast = CeilAlign(outShape[RANK_MERGED - 1], static_cast<int64_t>(buf.ubBlockElements));

    int64_t startAxis = RANK_MERGED - 1, innerElemCount = 1;
    for (int64_t ax = RANK_MERGED - 1; ax >= 0; ax--) {
        int64_t dimSize = (ax == RANK_MERGED - 1) ? alignedLast : outShape[ax];
        if (dimSize * innerElemCount >= static_cast<int64_t>(buf.cacheLineElements)) {
            startAxis = ax;
            break;
        }
        innerElemCount *= dimSize;
    }

    int64_t step = (startAxis == RANK_MERGED - 1) ? static_cast<int64_t>(buf.ubBlockElements) : 1;
    int64_t minUbFactor = CeilAlign(
        std::max(step, CeilDiv(static_cast<int64_t>(buf.cacheLineElements), innerElemCount)), step);

    int64_t outerCount = 1;
    for (int64_t ax = 0; ax < startAxis; ax++)
        outerCount *= outShape[ax];

    int64_t cumInnerCount = innerElemCount;
    int64_t bestAxis = 0, bestFactor = 1, bestTotalCount = 1, bestRealCore = 0;
    for (int64_t ax = startAxis; ax >= 0; ax--) {
        int64_t dimSize = (ax == RANK_MERGED - 1) ? alignedLast : outShape[ax];
        int64_t axStep = (ax == RANK_MERGED - 1) ? static_cast<int64_t>(buf.ubBlockElements) : 1;
        for (int64_t factor = (ax == startAxis ? minUbFactor : axStep); factor <= dimSize; factor += axStep) {
            int64_t blockSize = factor * cumInnerCount;
            if (blockSize > static_cast<int64_t>(buf.bufferSizeElements))
                break;
            int64_t totalCount = outerCount * CeilDiv(dimSize, factor);
            int64_t realCoreNum = CeilDiv(totalCount, CeilDiv(totalCount, static_cast<int64_t>(coreNum)));
            if (realCoreNum * 10 >= static_cast<int64_t>(coreNum) * 8 || realCoreNum > bestRealCore) {
                bestAxis = ax;
                bestFactor = factor;
                bestTotalCount = totalCount;
                if (realCoreNum > bestRealCore)
                    bestRealCore = realCoreNum;
            }
        }
        if (ax > 0) {
            outerCount /= outShape[ax - 1];
            cumInnerCount *= dimSize;
        }
    }

    result.ubAxis = static_cast<uint8_t>(bestAxis);
    result.ubFactor = static_cast<uint32_t>(bestFactor);
    result.totalCount = static_cast<uint64_t>(bestTotalCount);
    result.tilingKey = static_cast<uint8_t>(RANK_MERGED - bestAxis);
    return true;
}

static void MultiCoreSplit(uint64_t totalCount, uint32_t coreNum, MultiCoreSplitResult& result)
{
    int64_t tc = static_cast<int64_t>(totalCount);
    int64_t cn = static_cast<int64_t>(coreNum);
    int64_t perCore = CeilDiv(tc, cn);
    int64_t realCore = CeilDiv(tc, perCore);
    result.perCoreCount = static_cast<uint64_t>(perCore);
    result.realCoreNum = static_cast<uint64_t>(realCore);
}

static ge::graphStatus ReadCoordValue(gert::TilingContext* context, uint64_t inputIdx, const char* paramName,
                                      int64_t* out, size_t count)
{
    auto tensor = context->GetInputTensor(inputIdx);
    OP_CHECK_IF(tensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), paramName),
                return ge::GRAPH_FAILED);
    const int32_t* data = tensor->GetData<int32_t>();
    if (data != nullptr) {
        for (size_t i = 0; i < count; i++)
            out[i] = static_cast<int64_t>(data[i]);
        return ge::GRAPH_SUCCESS;
    }
    REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                              (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                        (std::string("failed to read ") + paramName +
                                                         " tensor data at tiling stage; Tiling cannot continue")
                                                            .c_str()}));
    return ge::GRAPH_FAILED;
}

static ge::graphStatus PasteSubImgTilingFunc(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do PasteSubImgTilingFunc");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            "GetPlatformInfo is null; Tiling cannot continue"})),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        coreNum == 0,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            "GetCoreNumAiv returned 0; Tiling cannot continue"})),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize == 0,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            "GetCoreMemSize(UB) returned 0; Tiling cannot continue"})),
        return ge::GRAPH_FAILED);

    auto patchShape = context->GetInputShape(IDX_PATCH_IMG);
    OP_CHECK_IF(patchShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "patch_img"),
                return ge::GRAPH_FAILED);
    auto patchCoordShape = context->GetInputShape(IDX_PATCH_COORD);
    OP_CHECK_IF(patchCoordShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "patch_coord"),
                return ge::GRAPH_FAILED);
    auto coreAreaCoordShape = context->GetInputShape(IDX_CORE_AREA_COORD);
    OP_CHECK_IF(coreAreaCoordShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "core_area_coord"),
                return ge::GRAPH_FAILED);
    auto combineShape = context->GetInputShape(IDX_COMBINE_IMG);
    OP_CHECK_IF(combineShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "combine_img"),
                return ge::GRAPH_FAILED);

    auto patchSS = patchShape->GetStorageShape();
    auto patchCoordSS = patchCoordShape->GetStorageShape();
    auto coreAreaCoordSS = coreAreaCoordShape->GetStorageShape();
    auto combineSS = combineShape->GetStorageShape();

    OP_CHECK_IF(patchSS.GetDimNum() != EXPECTED_RANK_3D || combineSS.GetDimNum() != EXPECTED_RANK_3D,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "patch_img, combine_img",
                    (std::to_string(patchSS.GetDimNum()) + ", " + std::to_string(combineSS.GetDimNum())).c_str(),
                    "the dim num of patch_img and combine_img must be 3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        patchCoordSS.GetDimNum() != EXPECTED_RANK_1D || coreAreaCoordSS.GetDimNum() != EXPECTED_RANK_1D,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "patch_coord, core_area_coord",
            (std::to_string(patchCoordSS.GetDimNum()) + ", " + std::to_string(coreAreaCoordSS.GetDimNum())).c_str(),
            "the dim num of patch_coord and core_area_coord must be 1"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(patchCoordSS.GetDim(0) != COORD_LEN || coreAreaCoordSS.GetDim(0) != COORD_LEN,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "patch_coord, core_area_coord",
                    (std::to_string(patchCoordSS.GetDim(0)) + ", " + std::to_string(coreAreaCoordSS.GetDim(0))).c_str(),
                    "the length of patch_coord and core_area_coord must be 4"),
                return ge::GRAPH_FAILED);

    auto patchTensor = context->GetInputTensor(IDX_PATCH_IMG);
    OP_CHECK_IF(patchTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "patch_img"),
                return ge::GRAPH_FAILED);
    auto combineTensor = context->GetInputTensor(IDX_COMBINE_IMG);
    OP_CHECK_IF(combineTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "combine_img"),
                return ge::GRAPH_FAILED);
    ge::DataType patchDtype = patchTensor->GetDataType();
    ge::DataType combineDtype = combineTensor->GetDataType();
    OP_CHECK_IF(!IsComputeDtype(patchDtype) || !IsComputeDtype(combineDtype),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "patch_img, combine_img",
                    (Ops::Base::ToString(patchDtype) + ", " + Ops::Base::ToString(combineDtype)).c_str(),
                    "the dtype of patch_img and combine_img must be uint8, float16 or float"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(patchDtype != combineDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "patch_img, combine_img",
                    (Ops::Base::ToString(patchDtype) + ", " + Ops::Base::ToString(combineDtype)).c_str(),
                    "the dtype of patch_img and combine_img must be the same"),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "GetAttrs is null; Tiling cannot continue"})),
                return ge::GRAPH_FAILED);

    int64_t patchCoordArr[4] = {};
    int64_t coreAreaCoordArr[4] = {};
    OP_CHECK_IF(ReadCoordValue(context, IDX_PATCH_COORD, "patch_coord", patchCoordArr, 4) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context, "failed to read patch_coord tensor data"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ReadCoordValue(context, IDX_CORE_AREA_COORD, "core_area_coord", coreAreaCoordArr, 4) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context, "failed to read core_area_coord tensor data"), return ge::GRAPH_FAILED);

    const float* scalePtr = attrs->GetAttrPointer<float>(ATTR_IDX_SCALE);
    OP_CHECK_IF(scalePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "scale"),
                return ge::GRAPH_FAILED);
    float scale = *scalePtr;
    OP_CHECK_IF(!(scale >= SCALE_MIN && scale <= SCALE_MAX),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "scale", std::to_string(scale).c_str(),
                                                      "scale must be in [0.0, 256.0]"),
                return ge::GRAPH_FAILED);

    int64_t px1 = patchCoordArr[0], py1 = patchCoordArr[1];
    int64_t cx1 = coreAreaCoordArr[0], cy1 = coreAreaCoordArr[1], cx2 = coreAreaCoordArr[2], cy2 = coreAreaCoordArr[3];
    OP_CHECK_IF(cx2 <= cx1 || cy2 <= cy1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "core_area_coord",
                                                      (std::to_string(cx1) + ", " + std::to_string(cy1) + ", " +
                                                       std::to_string(cx2) + ", " + std::to_string(cy2))
                                                          .c_str(),
                                                      "core_area_coord must satisfy cx2>cx1 and cy2>cy1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(cx1 < 0 || cy1 < 0 || cx1 + px1 < 0 || cy1 + py1 < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "core_area_coord",
                                                      (std::to_string(cx1) + ", " + std::to_string(cy1) + ", " +
                                                       std::to_string(cx1 + px1) + ", " + std::to_string(cy1 + py1))
                                                          .c_str(),
                                                      "source and destination coordinates must be non-negative"),
                return ge::GRAPH_FAILED);

    int64_t patchH = patchSS.GetDim(DIM_H);
    int64_t patchW = patchSS.GetDim(DIM_W);
    int64_t patchC = patchSS.GetDim(DIM_C);
    int64_t combineH = combineSS.GetDim(DIM_H);
    int64_t combineW = combineSS.GetDim(DIM_W);
    int64_t combineC = combineSS.GetDim(DIM_C);
    OP_CHECK_IF(
        patchC != combineC,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "patch_img, combine_img",
                                               (std::to_string(patchC) + ", " + std::to_string(combineC)).c_str(),
                                               "the C dimension of patch_img and combine_img must be the same"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(patchH <= 0 || patchW <= 0 || patchC <= 0 || combineH <= 0 || combineW <= 0 || combineC <= 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "patch_img, combine_img",
                    (std::to_string(patchH) + "x" + std::to_string(patchW) + "x" + std::to_string(patchC) + ", " +
                     std::to_string(combineH) + "x" + std::to_string(combineW) + "x" + std::to_string(combineC))
                        .c_str(),
                    "the dimensions of patch_img and combine_img must be positive"),
                return ge::GRAPH_FAILED);

    CoordBudget coord{};
    ComputeCoordBudget(px1, py1, cx1, cy1, cx2, cy2, scale, patchH, patchW, patchC, combineH, combineW, coord);
    if (coord.activeH == 0 || coord.activeW == 0) {
        OP_LOGI(context->GetNodeName(), "active region is empty, skip as no-op");
        PasteSubImgTilingData* tdNoop = context->GetTilingData<PasteSubImgTilingData>();
        OP_CHECK_IF(
            tdNoop == nullptr,
            REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                      (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                "GetTilingData is null; Tiling cannot continue"})),
            return ge::GRAPH_FAILED);
        auto memsetNoopRet = memset_s(tdNoop, sizeof(PasteSubImgTilingData), 0, sizeof(PasteSubImgTilingData));
        OP_CHECK_IF(memsetNoopRet != EOK,
                    REPORT_PREDEFINED_ERR_MSG(
                        "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                        (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                  (std::string("Failed to memset tiling data, ret is ") +
                                                   std::to_string(memsetNoopRet) + "; Tiling cannot continue")
                                                      .c_str()})),
                    return ge::GRAPH_FAILED);
        tdNoop->rank = RANK_MERGED;
        tdNoop->bufferSize = 32;
        tdNoop->dtypeBytes = static_cast<uint8_t>(GetSizeByDataType(patchDtype));
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint64_t>(PASTE_SUB_IMG_KEY_UBAXIS_C)));
        size_t* workspacesNoop = context->GetWorkspaceSizes(1);
        OP_CHECK_IF(workspacesNoop == nullptr,
                    REPORT_PREDEFINED_ERR_MSG(
                        "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                        (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                  "GetWorkspaceSizes(1) is null; Tiling cannot continue"})),
                    return ge::GRAPH_FAILED);
        workspacesNoop[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    int64_t sCy2Pre = static_cast<int64_t>(static_cast<double>(cy2) * static_cast<double>(scale));
    int64_t sCx2Pre = static_cast<int64_t>(static_cast<double>(cx2) * static_cast<double>(scale));
    OP_CHECK_IF(
        sCy2Pre > patchH || sCx2Pre > patchW,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "core_area_coord",
                                              (std::to_string(sCy2Pre) + ", " + std::to_string(sCx2Pre)).c_str(),
                                              "coord*scale is out of patch_img bounds"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        coord.dCy1 + coord.activeH > combineH || coord.dCx1 + coord.activeW > combineW,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "core_area_coord",
            (std::to_string(coord.dCy1 + coord.activeH) + ", " + std::to_string(coord.dCx1 + coord.activeW)).c_str(),
            "dest region is out of combine_img bounds"),
        return ge::GRAPH_FAILED);

    uint32_t dtypeBytes = static_cast<uint32_t>(GetSizeByDataType(patchDtype));
    uint32_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    uint32_t cacheLineSize = Ops::Base::GetSectorCacheLineSize(context);
    BufferResult buf{};
    ComputeBuffer(ubSize, ubBlockSize, cacheLineSize, dtypeBytes, buf);

    SplitAxisResult split{};
    FindSplitAxis(coord.activeH, coord.activeW, coord.activeC, buf, coreNum, split);

    MultiCoreSplitResult mc{};
    MultiCoreSplit(split.totalCount, coreNum, mc);

    PasteSubImgTilingData* td = context->GetTilingData<PasteSubImgTilingData>();
    OP_CHECK_IF(td == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "GetTilingData is null; Tiling cannot continue"})),
                return ge::GRAPH_FAILED);
    auto memsetRet = memset_s(td, sizeof(PasteSubImgTilingData), 0, sizeof(PasteSubImgTilingData));
    OP_CHECK_IF(
        memsetRet != EOK,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            (std::string("Failed to memset tiling data, ret is ") +
                                                             std::to_string(memsetRet) + "; Tiling cannot continue")
                                                                .c_str()})),
        return ge::GRAPH_FAILED);

    td->rank = RANK_MERGED;
    td->inShape[0] = coord.activeH;
    td->inShape[1] = coord.activeW;
    td->inShape[2] = coord.activeC;
    td->outShape[0] = coord.activeH;
    td->outShape[1] = coord.activeW;
    td->outShape[2] = coord.activeC;
    td->totalCount = split.totalCount;
    td->perCoreCount = mc.perCoreCount;
    td->ubAxis = split.ubAxis;
    td->ubFactor = split.ubFactor;
    td->bufferSize = buf.bufferSize;
    td->patchBaseOffset = coord.patchBaseOffset;
    td->combineBaseOffset = coord.combineBaseOffset;
    td->patchStrideH = coord.patchStrideH;
    td->patchStrideW = coord.patchStrideW;
    td->combineStrideH = coord.combineStrideH;
    td->combineStrideW = coord.combineStrideW;
    td->activeH = coord.activeH;
    td->activeW = coord.activeW;
    td->activeC = coord.activeC;
    td->dtypeBytes = static_cast<uint8_t>(dtypeBytes);

    context->SetBlockDim(static_cast<uint32_t>(mc.realCoreNum));
    context->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint64_t>(split.tilingKey)));

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(
        workspaces == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            "GetWorkspaceSizes(1) is null; Tiling cannot continue"})),
        return ge::GRAPH_FAILED);
    workspaces[0] = 0;

    OP_LOGI(context->GetNodeName(),
            "PasteSubImg tiling: region=(%ld,%ld,%ld) ubAxis=%u ubFactor=%u key=%u "
            "totalCount=%lu perCoreCount=%lu realCoreNum=%lu bufferSize=%u",
            coord.activeH, coord.activeW, coord.activeC, split.ubAxis, split.ubFactor, split.tilingKey, td->totalCount,
            td->perCoreCount, mc.realCoreNum, buf.bufferSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForPasteSubImg(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<PasteSubImgCompileInfo>();
    OP_CHECK_IF(
        compileInfo == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            "GetCompiledInfo is null; TilingParse cannot continue"})),
        return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            "GetPlatformInfo is null; Tiling cannot continue"})),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PasteSubImg)
    .Tiling(PasteSubImgTilingFunc)
    .TilingParse<PasteSubImgCompileInfo>(TilingPrepareForPasteSubImg);

} // namespace optiling
