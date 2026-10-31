/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file integral_image_tiling.cpp
 * \brief IntegralImage host tiling: two-phase (per-row scan + per-column vertical add) or legacy column blocks
 */

#include "log/log.h"
#include <algorithm>
#include <limits>
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "integral_image_compile_info.h"
#include "../op_kernel/integral_image_tiling_data.h"
#include "../op_kernel/integral_image_tiling_key.h"

namespace optiling {

constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t INDEXONE = 1;
constexpr int32_t MAX_BLOCK_WIDTH = 1024;
constexpr int32_t MIN_BLOCK_WIDTH = 32;
constexpr int32_t TWO_PHASE_MIN_WIDTH = 128;
constexpr int32_t MAX_COL_TILE_WIDTH = 256;
constexpr int32_t MIN_COL_TILE_WIDTH = 32;
// Workspace requests include the 16 MiB runtime reservation and algorithm scratch space.
// The kernel partitions its scratch buffers relative to the workspace pointer it receives.
constexpr int64_t RESERVED_WORKSPACE_BYTES = 16LL * 1024 * 1024;

// -------------------------------------------------------------------------------------------
// UB footprint of the kernel's legacy path (2D small-W and 3D HWC share exactly the same TBuf
// set). Mirroring the InitBuffer sequence in op_kernel/integral_image.h lets tiling choose
// a blockWidth that fits the available UB, including HWC interleaved buffers.
// -------------------------------------------------------------------------------------------
constexpr int64_t UB_ALIGN_BYTES = 32;
// Reserve for the TPipe scaffolding (CumSum LCM temp, stack) so the check keeps a safety margin.
constexpr int64_t UB_RESERVE_BYTES = 8 * 1024;
// Mirrors kPhaseBBatchRows in op_kernel/integral_image.h.
constexpr int64_t UB_BATCH_ROWS = 8;

static inline int64_t AlignUpTo32Bytes(int64_t bytes)
{
    return (bytes + UB_ALIGN_BYTES - 1) / UB_ALIGN_BYTES * UB_ALIGN_BYTES;
}

static int64_t LegacyUbFootprint(int64_t blockWidth, int64_t channel, int64_t inBytes)
{
    const int64_t accBytes = 4; // sizeof(AccT)
    const int64_t bw = blockWidth;
    const int64_t ch = channel;
    const int64_t sharedElems = bw * ch; // satBuf_ / accBigBuf_ / accSqBuf_
    const int64_t colTileWidth = 0;      // the legacy tiling leaves colTileWidth at 0
    // "one row" buffers: max(colTileWidth + 8, channel) elements, 32B aligned
    const int64_t rowBufBytes = AlignUpTo32Bytes(std::max(colTileWidth + 8, ch) * accBytes);
    int64_t bytes = 0;
    bytes += bw * ch * inBytes;                             // imgBuf_
    bytes += sharedElems * accBytes;                        // satBuf_
    bytes += sharedElems * accBytes;                        // accBigBuf_
    bytes += AlignUpTo32Bytes(ch * accBytes);               // prefixBuf_
    bytes += AlignUpTo32Bytes(8 * ch * accBytes);           // zeroBigBuf_
    bytes += (bw + 8) * accBytes;                           // castBuf_
    bytes += bw * accBytes;                                 // zeroBuf_
    bytes += (bw + 8) * accBytes;                           // leftBuf_
    bytes += (bw + 8) * 2;                                  // halfBuf_ (sizeof(half))
    bytes += rowBufBytes;                                   // wsRowBuf_
    bytes += (bw + 8) * accBytes;                           // lastRowBuf_
    bytes += UB_BATCH_ROWS * (colTileWidth + 8) * accBytes; // wsBatchBuf_
    bytes += UB_BATCH_ROWS * (colTileWidth + 8) * accBytes; // outBuf_
    bytes += UB_BATCH_ROWS * (bw + 8) * inBytes;            // imgBatchBuf_
    if (inBytes == 2) {                                     // castBatchBuf_ (fp16 only)
        bytes += UB_BATCH_ROWS * (bw + 8) * accBytes;
    }
    bytes += UB_BATCH_ROWS * (bw + 8) * accBytes;                        // dstBatchBuf_
    bytes += std::max(UB_BATCH_ROWS * (bw + 8), sharedElems) * accBytes; // dstSqBatchBuf_
    if (inBytes != 1) {                                                  // sqSrcBatchBuf_ (fp16/fp32)
        bytes += UB_BATCH_ROWS * (bw + 8) * accBytes;
    }
    bytes += UB_BATCH_ROWS * (colTileWidth + 8) * accBytes;     // wsSqBatchBuf_
    bytes += UB_BATCH_ROWS * (colTileWidth + 8) * accBytes;     // outSqBuf_
    bytes += rowBufBytes;                                       // wsSqRowBuf_
    bytes += rowBufBytes;                                       // outSqRowBuf_
    bytes += sharedElems * accBytes;                            // accSqBuf_
    bytes += (bw + 8) * accBytes;                               // sqWorkBuf_
    bytes += AlignUpTo32Bytes(std::max(bw + 8, ch) * accBytes); // zeroSqBuf_: HWC left boundary needs C zeros
    return bytes;
}

// 读取平台能力：UB 容量（字节）与可用 AIV 核数。
// 两者任一取不到有效值都直接失败，避免后续按 0 计算 tiling。
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubBytes, int64_t& aivCoreNum)
{
    fe::PlatFormInfos* platformInfos = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfos);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfos);
    aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivCoreNum <= 0, OP_LOGE(context, "IntegralImage: AIV core num is invalid."), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    OP_CHECK_IF(ubBytes == 0, OP_LOGE(context, "IntegralImage: UB size is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutput(gert::TilingContext* context, uint32_t index, const gert::Shape& expectedShape,
                                   ge::DataType expectedType)
{
    const auto* shape = context->GetOutputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    const auto* desc = context->GetOutputDesc(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    const auto& actualShape = shape->GetStorageShape();
    if (actualShape.GetDimNum() != expectedShape.GetDimNum()) {
        OP_LOGE(context, "IntegralImage: output[%u] rank must be %zu, got %zu", index, expectedShape.GetDimNum(),
                actualShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    for (size_t dim = 0; dim < expectedShape.GetDimNum(); ++dim) {
        if (actualShape.GetDim(dim) != expectedShape.GetDim(dim)) {
            OP_LOGE(context, "IntegralImage: output[%u] dim[%zu] must be %ld, got %ld", index, dim,
                    expectedShape.GetDim(dim), actualShape.GetDim(dim));
            return ge::GRAPH_FAILED;
        }
    }
    if (desc->GetDataType() != expectedType) {
        OP_LOGE(context, "IntegralImage: output[%u] dtype must be %d, got %d", index,
                static_cast<int32_t>(expectedType), static_cast<int32_t>(desc->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus IntegralImageTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNumAiv;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNumAiv) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputShape(INDEXZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto inputShapeX = inputShape->GetStorageShape();
    auto dimNum = inputShapeX.GetDimNum();
    if (dimNum != 2 && dimNum != 3) {
        OP_LOGE(context, "IntegralImage requires rank 2 or 3, got %zu", dimNum);
        return ge::GRAPH_FAILED;
    }

    // HWC layout: rank2 = (H, W), rank3 = (H, W, C), C is the last dim.
    int64_t height = inputShapeX.GetDim(INDEXZERO);
    int64_t width = inputShapeX.GetDim(INDEXONE);
    int64_t channel = (dimNum == 3) ? inputShapeX.GetDim(dimNum - 1) : 1;
    if (height <= 0 || width <= 0 || channel <= 0) {
        OP_LOGE(context, "invalid shape H=%ld W=%ld C=%ld", height, width, channel);
        return ge::GRAPH_FAILED;
    }
    if (height == std::numeric_limits<int64_t>::max() || width == std::numeric_limits<int64_t>::max()) {
        OP_LOGE(context, "IntegralImage: spatial output dimensions must fit int64");
        return ge::GRAPH_FAILED;
    }
    const auto* inputDesc = context->GetInputDesc(INDEXZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inDtype = inputDesc->GetDataType();
    if (inDtype != ge::DT_UINT8 && inDtype != ge::DT_FLOAT16 && inDtype != ge::DT_FLOAT) {
        OP_LOGE(context, "IntegralImage: unsupported input dtype");
        return ge::GRAPH_FAILED;
    }
    // Optional outputs are disabled for nullptr, rank-0 tensors or the rank-1 [0] placeholder.
    int64_t sqsumEnabled = 1;
    auto sqsumShape = context->GetOutputShape(1);
    if (sqsumShape == nullptr) {
        sqsumEnabled = 0;
    } else {
        auto sq = sqsumShape->GetShape();
        if (sq.GetDimNum() == 0 || (sq.GetDimNum() == 1 && sq.GetDim(0) == 0)) {
            sqsumEnabled = 0;
        }
    }
    int64_t tiltedEnabled = 1;
    auto tiltedShape = context->GetOutputShape(2);
    if (tiltedShape == nullptr) {
        tiltedEnabled = 0;
    } else {
        auto shp = tiltedShape->GetShape();
        if (shp.GetDimNum() == 0 || (shp.GetDimNum() == 1 && shp.GetDim(0) == 0)) {
            tiltedEnabled = 0;
        }
    }
    // sdepth: -1 自动（u8->int32，fp16/fp32->float32）；0=int32（仅 u8）；1=float32。
    int64_t sdepth = -1;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto p = attrs->GetAttrPointer<int64_t>(0);
        if (p != nullptr) {
            sdepth = *p;
        }
    }
    if (sdepth != -1 && sdepth != 0 && sdepth != 1) {
        OP_LOGE(context, "IntegralImage: unsupported sdepth=%ld (expect -1/0/1; float64 not supported on 910B)",
                sdepth);
        return ge::GRAPH_FAILED;
    }
    if (sdepth == 0 && inDtype != ge::DT_UINT8) {
        OP_LOGE(context, "IntegralImage: sdepth=int32 is only valid for uint8 input");
        return ge::GRAPH_FAILED;
    }
    const bool outF32 = (sdepth == 1);
    const ge::DataType accType = (inDtype == ge::DT_UINT8 && !outF32) ? ge::DT_INT32 : ge::DT_FLOAT;
    gert::Shape expectedShape = inputShapeX;
    expectedShape.SetDim(INDEXZERO, height + 1);
    expectedShape.SetDim(INDEXONE, width + 1);
    if (CheckOutput(context, 0, expectedShape, accType) != ge::GRAPH_SUCCESS ||
        (sqsumEnabled != 0 && CheckOutput(context, 1, expectedShape, ge::DT_FLOAT) != ge::GRAPH_SUCCESS) ||
        (tiltedEnabled != 0 && CheckOutput(context, 2, expectedShape, accType) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t accBytes = 4;
    int64_t inBytes = (inDtype == ge::DT_UINT8) ? 1 : ((inDtype == ge::DT_FLOAT16) ? 2 : 4);

    IntegralImageTilingData* tiling = context->GetTilingData<IntegralImageTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(IntegralImageTilingData), 0, sizeof(IntegralImageTilingData)) != EOK,
                OP_LOGE(context, "memset tiling error"), return ge::GRAPH_FAILED);
    tiling->height = height;
    tiling->width = width;
    tiling->channel = channel;
    tiling->sqsumEnabled = sqsumEnabled;
    tiling->tiltedEnabled = tiltedEnabled;

    if (channel == 1 && width >= TWO_PHASE_MIN_WIDTH) {
        // ---- two-phase path: phase A per-row horizontal scan, phase B per-column vertical add ----
        // UB budget must also reserve the phase-A batch buffers and the CumSum LCM tmp
        // (TPipe allocates all UB positions from one pool; PopStackBuffer(LCM) takes the tail).
        // fixed UB = legacy buffers (~90KB with sqsum phase-B additions):
        // wsBatch/out/wsSqBatch/outSq 8*(colTileW+8)*4*4 + rows/cols/acc/zero/left/cast etc.
        const int64_t kFixedUb = 96 * 1024;
        const bool fpPath = (inDtype == ge::DT_FLOAT16 || inDtype == ge::DT_FLOAT);
        int64_t rowTileWidth = MAX_BLOCK_WIDTH;
        int64_t ubBudget = static_cast<int64_t>(ubSize) * 9 / 10;
        while (rowTileWidth > MIN_BLOCK_WIDTH) {
            int64_t batchBytes = 0;
            if (fpPath) {
                int64_t extraCast = (inDtype == ge::DT_FLOAT16) ? accBytes : 0;
                batchBytes = 8LL * (rowTileWidth + 8) * (inBytes + accBytes + extraCast) +
                             8LL * rowTileWidth * accBytes * 2 + // CumSum LCM tmp
                             8LL * rowTileWidth * accBytes * 2 + // sqsum: sqSrcBatch + dstSqBatch
                             rowTileWidth * accBytes;            // sqsum column accumulator (accSq)
            } else {
                // u8: imgBatch(InT) + dstBatch + dstSqBatch(float32) + sqsum accumulator
                batchBytes = 8LL * (rowTileWidth + 8) * (inBytes + 2 * accBytes) + rowTileWidth * accBytes;
            }
            if (kFixedUb + batchBytes <= ubBudget) {
                break;
            }
            rowTileWidth >>= 1;
        }
        // Phase B: one column tile per core for wide images. colTileWidth is chosen so that
        // colTileCount is close to (but not exceeding) the core count, keeping the vector
        // width >= 32 elements. Too many tiles would make the busiest core run 2+ tiles serially.
        int64_t perCore = (width + coreNumAiv - 1) / coreNumAiv;
        int64_t colTileWidth = ((perCore + 7) / 8) * 8; // 32B aligned for AccT=float
        if (colTileWidth < MIN_COL_TILE_WIDTH) {
            colTileWidth = MIN_COL_TILE_WIDTH;
        }
        if (colTileWidth > MAX_COL_TILE_WIDTH) {
            colTileWidth = MAX_COL_TILE_WIDTH;
        }
        // Phase B reuses the phase-A UB buffers (sized by rowTileWidth) with colTileWidth wide
        // vector ops, and satBuf_ additionally needs an 8 element zero gutter, so keep
        // colTileWidth <= rowTileWidth - 8. rowTileWidth is a multiple of 32 here, hence the
        // result stays a multiple of 8 (32B for 4B accumulators).
        if (colTileWidth > rowTileWidth - 8) {
            colTileWidth = rowTileWidth - 8;
        }
        if (colTileWidth > width) {
            colTileWidth = width;
        }
        int64_t colTileCount = (width + colTileWidth - 1) / colTileWidth;
        int64_t activeCoreNum = std::min(coreNumAiv, std::max(height, colTileCount));
        if (activeCoreNum < 1) {
            activeCoreNum = 1;
        }

        tiling->twoPhase = 1;
        tiling->activeCoreNum = activeCoreNum;
        tiling->rowTileWidth = rowTileWidth;
        tiling->colTileWidth = colTileWidth;
        tiling->colTileCount = colTileCount;
        tiling->blockWidth = rowTileWidth; // reused by kernel UB buffer sizing
        tiling->coreNum = activeCoreNum;

        context->SetBlockDim(static_cast<uint32_t>(activeCoreNum));

        size_t* ws = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, ws);
        // workspace layout:
        //   [0, H*W)        sum row prefixes (AccT)
        //   [H*W, 2*H*W)    sqsum row prefixes (float)
        //   [2*H*W, 4*H*W)  tilted acc1 / acc2 diagonal-prefix buffers (AccT)
        ws[0] = static_cast<size_t>(RESERVED_WORKSPACE_BYTES + height * width * accBytes * 4);
    } else {
        // ---- legacy path: per-column block multi-core (2D small W or 3D HWC) ----
        const int64_t ubBudget = static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES;
        int32_t blockWidth = MAX_BLOCK_WIDTH;
        while (blockWidth > MIN_BLOCK_WIDTH && LegacyUbFootprint(blockWidth, channel, inBytes) > ubBudget) {
            blockWidth >>= 1;
        }
        const int64_t ubNeed = LegacyUbFootprint(blockWidth, channel, inBytes);
        if (ubNeed > ubBudget) {
            OP_LOGE(context,
                    "IntegralImage: H=%ld W=%ld C=%ld inBytes=%ld requires %ld UB bytes (>%ld available) "
                    "even at blockWidth=%d; please reduce channel/image size",
                    height, width, channel, inBytes, ubNeed, ubBudget, MIN_BLOCK_WIDTH);
            return ge::GRAPH_FAILED;
        }
        int32_t coreNum = static_cast<int32_t>((width + blockWidth - 1) / blockWidth);
        if (coreNum > coreNumAiv) {
            OP_LOGE(context, "IntegralImage: W=%ld requires %d blocks > platform cores %ld", width, coreNum,
                    coreNumAiv);
            return ge::GRAPH_FAILED;
        }

        tiling->blockWidth = blockWidth;
        tiling->coreNum = coreNum;
        context->SetBlockDim(static_cast<uint32_t>(coreNum));
        // tilted multi-core needs workspace: [0,H*W) row prefixes, [H*W,2*H*W) acc1,
        // [2*H*W,3*H*W) acc2
        size_t* ws = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, ws);
        ws[0] = static_cast<size_t>(RESERVED_WORKSPACE_BYTES + height * width * accBytes * 3);
    }

    uint64_t tilingKey = 0;
    if (inDtype == ge::DT_UINT8) {
        tilingKey = outF32 ? GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_U8_F32) :
                             GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_U8_I32);
    } else if (inDtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_F16_F32);
    } else if (inDtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_F32_F32);
    } else {
        OP_LOGE(context, "unsupported input dtype");
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForIntegralImage([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IntegralImage)
    .Tiling(IntegralImageTilingFunc)
    .TilingParse<IntegralImageCompileInfo>(TilingParseForIntegralImage);

} // namespace optiling
