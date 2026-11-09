/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../op_kernel/arch35/rotated_box_encode_tiling_data.h"
#include "../../op_kernel/arch35/rotated_box_encode_struct.h"
#include "rotated_box_encode_tiling_arch35.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "graph/types.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <vector>

namespace optiling {

// ===========================================================================
// Integer helpers (DESIGN §2 / §9 formulas).
// ===========================================================================
static inline int64_t CeilDiv(int64_t a, int64_t b) { return a / b + ((a % b) != 0); }
static inline int64_t AlignDown(int64_t v, int64_t a) { return (v / a) * a; }

// Align up without overflowing int64.  Host tiling must never wrap on the
// aligned block size (F-02); a failure is reported to the framework instead.
static inline bool CheckedCeilAlign(int64_t v, int64_t a, int64_t& out)
{
    const int64_t q = CeilDiv(v, a);
    if (q > std::numeric_limits<int64_t>::max() / a) {
        return false;
    }
    out = q * a;
    return true;
}

// ceil(dim0 * minDtypeBits / MIN_TILING_BITS), capped at availableCores.
// dim0 can exceed INT64_MAX / minDtypeBits for pathological shapes, so the
// saturation point is detected by division before any multiply (F-02).
static inline int64_t ComputeCoreNum(int64_t dim0, int64_t minDtypeBits, int64_t availableCores)
{
    const int64_t capScaled = availableCores * MIN_TILING_BITS;
    if (dim0 > capScaled / minDtypeBits) {
        return availableCores;
    }
    int64_t coreNum = CeilDiv(dim0 * minDtypeBits, MIN_TILING_BITS);
    return (coreNum < availableCores) ? coreNum : availableCores;
}

// ===========================================================================
// ComputeBranch0Tiling: Branch-0 (fp16-upcast) tiling computation.
// DESIGN-BRANCH-0.md §2: perBoxBytes=70, alignFactor=128, minDtypeBits=80.
// ===========================================================================
ge::graphStatus ComputeBranch0Tiling(const RotatedBoxEncodeBranch0Inputs& in, RotatedBoxEncodeTilingData& out)
{
    out.dim0 = in.dim0;
    out.N = in.N;
    for (int i = 0; i < BRANCH0_BOX_CHANNELS; ++i) {
        out.weight[i] = in.weight[i];
    }

    if (in.dim0 == 0) {
        out.coreNum = 0;
        return ge::GRAPH_SUCCESS;
    }

    int64_t perBoxBytes = BRANCH0_PER_BOX_BYTES;
    int64_t elemStride = BRANCH0_ELEM_STRIDE;
    int64_t alignFactor = BRANCH0_ALIGN_256_BYTES / std::gcd(BRANCH0_ALIGN_256_BYTES, elemStride);
    int64_t maxBoxNum = static_cast<int64_t>(in.ubSize) / perBoxBytes;
    if (maxBoxNum < alignFactor) {
        return ge::GRAPH_FAILED;
    }
    int64_t ubFormer = AlignDown(maxBoxNum, alignFactor);
    out.ubFormer = ubFormer;

    int64_t minDtypeBits = BRANCH0_BOX_CHANNELS * BRANCH0_ELEM_BYTES * 8;
    int64_t availableCores = static_cast<int64_t>(in.coreNumAiv);
    int64_t coreNum = ComputeCoreNum(in.dim0, minDtypeBits, availableCores);
    coreNum = (coreNum < 1) ? 1 : coreNum;

    int64_t blockFormer = 0;
    if (!CheckedCeilAlign(CeilDiv(in.dim0, coreNum), BRANCH0_ELEM_ALIGN_FACTOR, blockFormer)) {
        return ge::GRAPH_FAILED;
    }
    int64_t blockNum = CeilDiv(in.dim0, blockFormer);
    out.coreNum = static_cast<int32_t>(coreNum);
    out.blockFormer = blockFormer;
    out.blockNum = blockNum;

    int64_t ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t blockTail = in.dim0 - (blockNum - 1) * blockFormer;
    int64_t ubLoopOfTailBlock = CeilDiv(blockTail, ubFormer);
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;
    out.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    out.ubTailOfFormerBlock = ubTailOfFormerBlock;
    out.ubLoopOfTailBlock = ubLoopOfTailBlock;
    out.ubTailOfTailBlock = ubTailOfTailBlock;

    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// ComputeBranch1Tiling: Branch-1 (fp32-direct) tiling computation.
// DESIGN-BRANCH-1.md §2: perBoxBytes=60, alignFactor=64, minDtypeBits=160.
// ===========================================================================
ge::graphStatus ComputeBranch1Tiling(const RotatedBoxEncodeBranch1Inputs& in, RotatedBoxEncodeTilingData& out)
{
    out.dim0 = in.dim0;
    out.N = in.N;
    for (int i = 0; i < BRANCH1_BOX_CHANNELS; ++i) {
        out.weight[i] = in.weight[i];
    }

    if (in.dim0 == 0) {
        out.coreNum = 0;
        return ge::GRAPH_SUCCESS;
    }

    int64_t perBoxBytes = BRANCH1_PER_BOX_BYTES;
    int64_t elemStride = BRANCH1_ELEM_STRIDE;
    int64_t alignFactor = BRANCH1_ALIGN_256_BYTES / std::gcd(BRANCH1_ALIGN_256_BYTES, elemStride);
    int64_t maxBoxNum = static_cast<int64_t>(in.ubSize) / perBoxBytes;
    if (maxBoxNum < alignFactor) {
        return ge::GRAPH_FAILED;
    }
    int64_t ubFormer = AlignDown(maxBoxNum, alignFactor);
    out.ubFormer = ubFormer;

    int64_t minDtypeBits = BRANCH1_BOX_CHANNELS * BRANCH1_ELEM_BYTES * 8;
    int64_t availableCores = static_cast<int64_t>(in.coreNumAiv);
    int64_t coreNum = ComputeCoreNum(in.dim0, minDtypeBits, availableCores);
    coreNum = (coreNum < 1) ? 1 : coreNum;

    int64_t blockFormer = 0;
    if (!CheckedCeilAlign(CeilDiv(in.dim0, coreNum), BRANCH1_ELEM_ALIGN_FACTOR, blockFormer)) {
        return ge::GRAPH_FAILED;
    }
    int64_t blockNum = CeilDiv(in.dim0, blockFormer);
    out.coreNum = static_cast<int32_t>(coreNum);
    out.blockFormer = blockFormer;
    out.blockNum = blockNum;

    int64_t ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t blockTail = in.dim0 - (blockNum - 1) * blockFormer;
    int64_t ubLoopOfTailBlock = CeilDiv(blockTail, ubFormer);
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;
    out.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    out.ubTailOfFormerBlock = ubTailOfFormerBlock;
    out.ubLoopOfTailBlock = ubLoopOfTailBlock;
    out.ubTailOfTailBlock = ubTailOfTailBlock;

    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// TilingRotatedBoxEncode — runtime tiling entry (DESIGN §9.9).
// ===========================================================================
ge::graphStatus TilingRotatedBoxEncode(gert::TilingContext* ctx)
{
    // Platform parameters are read at runtime.  Never reuse compile-time
    // values or silently fall back to a degenerate 1-core / 0-byte plan (F-03,
    // F-04): a missing or invalid platform description is a hard failure.
    auto* platInfo = ctx->GetPlatformInfo();
    if (platInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC plat(platInfo);
    uint32_t coreNumAiv = plat.GetCoreNumAiv();
    // The tiling data stores the core count in int32; the documented A5 range
    // is far below this bound.  Reject rather than truncate or overflow.
    if (coreNumAiv == 0 || coreNumAiv > 1024) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizeVal = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeVal);
    if (ubSizeVal == 0 || ubSizeVal > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    const auto* anchorShape = ctx->GetInputShape(0);
    if (anchorShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& originShape = anchorShape->GetOriginShape();
    int64_t rank = static_cast<int64_t>(originShape.GetDimNum());

    const auto* anchorTensor = ctx->GetInputTensor(0);
    const auto* gtTensor = ctx->GetInputTensor(1);
    if (anchorTensor == nullptr || gtTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType inDtype0 = anchorTensor->GetDataType();
    ge::DataType inDtype1 = gtTensor->GetDataType();

    static const std::set<ge::DataType> SUPPORTED_DTYPES = {ge::DT_FLOAT16, ge::DT_FLOAT};
    if (SUPPORTED_DTYPES.find(inDtype0) == SUPPORTED_DTYPES.end()) {
        return ge::GRAPH_FAILED;
    }
    if (inDtype0 != inDtype1) {
        return ge::GRAPH_FAILED;
    }

    ge::Format anchorFmt = anchorTensor->GetStorageFormat();
    if (anchorFmt != ge::FORMAT_ND) {
        return ge::GRAPH_FAILED;
    }

    if (rank != MAX_RANK || originShape.GetDim(1) != BOX_CHANNELS) {
        return ge::GRAPH_FAILED;
    }
    int64_t B = originShape.GetDim(0);
    int64_t N = originShape.GetDim(2);

    const auto* attrs = ctx->GetAttrs();
    const auto* wList = (attrs != nullptr) ? attrs->GetListFloat(0) : nullptr;
    int64_t wLen = (wList != nullptr) ? static_cast<int64_t>(wList->GetSize()) : BOX_CHANNELS;
    if (wLen != BOX_CHANNELS) {
        return ge::GRAPH_FAILED;
    }

    const auto* gtShapePtr = ctx->GetInputShape(1);
    if (gtShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& gtOriginShape = gtShapePtr->GetOriginShape();
    bool shapeMismatch = false;
    if (gtOriginShape.GetDimNum() != static_cast<size_t>(rank)) {
        shapeMismatch = true;
    } else {
        for (int64_t i = 0; i < rank; ++i) {
            if (originShape.GetDim(static_cast<size_t>(i)) != gtOriginShape.GetDim(static_cast<size_t>(i))) {
                shapeMismatch = true;
                break;
            }
        }
    }
    if (shapeMismatch) {
        return ge::GRAPH_FAILED;
    }

    // Negative dimensions are unknown-shape markers, not valid runtime shapes
    // for tiling.  Rejecting them here prevents a negative box count from
    // reaching CeilDiv and dividing by zero (F-01).
    if (B < 0 || N < 0) {
        return ge::GRAPH_FAILED;
    }
    if (B != 0 && N > std::numeric_limits<int64_t>::max() / B) {
        return ge::GRAPH_FAILED;
    }
    int64_t dim0 = B * N;

    float weight[BOX_CHANNELS] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    if (wList != nullptr) {
        const float* wData = wList->GetData();
        if (wData != nullptr) {
            for (int i = 0; i < BOX_CHANNELS; ++i) {
                weight[i] = wData[static_cast<size_t>(i)];
            }
        }
    }

    auto* td = ctx->GetTilingData<RotatedBoxEncodeTilingData>();
    if (td == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const bool isFp16 = (inDtype0 == ge::DT_FLOAT16);
    if (isFp16) {
        ctx->SetTilingKey(GET_TPL_TILING_KEY(ROTATED_BOX_ENCODE_DTYPE_FP16));
    } else {
        ctx->SetTilingKey(GET_TPL_TILING_KEY(ROTATED_BOX_ENCODE_DTYPE_FP32));
    }

    std::memset(td, 0, sizeof(*td));
    td->dim0 = dim0;
    td->N = N;
    for (int i = 0; i < BOX_CHANNELS; ++i) {
        td->weight[i] = weight[i];
    }

    if (dim0 == 0) {
        ctx->SetBlockDim(1);
        size_t* ws = ctx->GetWorkspaceSizes(1);
        if (ws != nullptr) {
            ws[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus branchRet = ge::GRAPH_FAILED;
    if (isFp16) {
        RotatedBoxEncodeBranch0Inputs bin{};
        bin.dim0 = dim0;
        bin.N = N;
        bin.coreNumAiv = coreNumAiv;
        bin.ubSize = ubSizeVal;
        for (int i = 0; i < BOX_CHANNELS; ++i) {
            bin.weight[i] = weight[i];
        }
        branchRet = ComputeBranch0Tiling(bin, *td);
    } else {
        RotatedBoxEncodeBranch1Inputs bin{};
        bin.dim0 = dim0;
        bin.N = N;
        bin.coreNumAiv = coreNumAiv;
        bin.ubSize = ubSizeVal;
        for (int i = 0; i < BOX_CHANNELS; ++i) {
            bin.weight[i] = weight[i];
        }
        branchRet = ComputeBranch1Tiling(bin, *td);
    }

    if (branchRet != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = (td->coreNum > 0) ? static_cast<uint32_t>(td->coreNum) : 1;
    ctx->SetBlockDim(blockDim);
    size_t* ws = ctx->GetWorkspaceSizes(1);
    if (ws != nullptr) {
        ws[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

struct RotatedBoxEncodeCompileInfo {};

static ge::graphStatus TilingParseForRotatedBoxEncode([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RotatedBoxEncode)
    .Tiling(TilingRotatedBoxEncode)
    .TilingParse<RotatedBoxEncodeCompileInfo>(TilingParseForRotatedBoxEncode);

} // namespace optiling
