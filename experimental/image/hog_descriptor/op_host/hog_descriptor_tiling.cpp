/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstring>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "hog_descriptor_tiling.h"
#include "hog_descriptor_ub.h"

namespace ops {

static constexpr size_t ATTR_PIXELS_PER_CELL = 0;
static constexpr size_t ATTR_CELLS_PER_BLOCK = 1;
static constexpr size_t ATTR_BLOCK_STRIDE = 2;
static constexpr size_t ATTR_ORIENTATIONS = 3;
static constexpr size_t ATTR_BLOCK_NORM = 4;
static constexpr size_t ATTR_L2_HYS = 5;
static constexpr size_t ATTR_GAMMA = 6;
static constexpr size_t ATTR_SIGNED = 7;
static constexpr size_t ATTR_APERTURE = 8;
static constexpr size_t ATTR_EPS = 9;

static uint64_t TilingKeyFor(ge::DataType dt)
{
    if (dt == ge::DT_FLOAT16) {
        return optiling::HOG_TILING_KEY_FP16;
    }
    return optiling::HOG_TILING_KEY_FP32;
}

static ge::graphStatus HOGDescriptorTilingFunc(gert::TilingContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0U, OP_LOGE(context, "invalid Ascend910B AIV platform information."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    const auto dtype = desc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT,
                OP_LOGE(context, "HOGDescriptor only supports float16 and float32 inputs."), return ge::GRAPH_FAILED);
    const uint32_t inputBytes = dtype == ge::DT_FLOAT16 ? 2U : 4U;

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto shape = inputShape->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    OP_CHECK_IF(rank < 2U || rank > 4U, OP_LOGE(context, "HOGDescriptor only supports rank 2/3/4."),
                return ge::GRAPH_FAILED);

    // Decode [H,W]/[H,W,C]/[N,H,W]/[N,H,W,C] (rank3 with last dim 1/3 is HWC).
    int64_t nDim = 1, hDim = 0, wDim = 0, cDim = 1;
    if (rank == 2U) {
        hDim = shape.GetDim(0);
        wDim = shape.GetDim(1);
    } else if (rank == 3U) {
        const int64_t last = shape.GetDim(2);
        if (last == 1 || last == 3) {
            hDim = shape.GetDim(0);
            wDim = shape.GetDim(1);
            cDim = last;
        } else {
            nDim = shape.GetDim(0);
            hDim = shape.GetDim(1);
            wDim = shape.GetDim(2);
        }
    } else {
        nDim = shape.GetDim(0);
        hDim = shape.GetDim(1);
        wDim = shape.GetDim(2);
        cDim = shape.GetDim(3);
    }
    OP_CHECK_IF(nDim <= 0 || hDim <= 0 || wDim <= 0 || cDim <= 0,
                OP_LOGE(context, "HOGDescriptor does not support empty tensors."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cDim != 1 && cDim != 3, OP_LOGE(context, "channel C must be 1 or 3."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(nDim > 200 || hDim > 10000 || wDim > 10000,
                OP_LOGE(context, "batch must be <= 200 and H/W must be <= 10000."), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // Null-tolerant reads with spec defaults (autogen tool may probe without attrs).
    const gert::TypedContinuousVector<int64_t>* ppc = attrs->GetListInt(ATTR_PIXELS_PER_CELL);
    const gert::TypedContinuousVector<int64_t>* cpb = attrs->GetListInt(ATTR_CELLS_PER_BLOCK);
    const gert::TypedContinuousVector<int64_t>* bst = attrs->GetListInt(ATTR_BLOCK_STRIDE);
    const int64_t* orientations = attrs->GetInt(ATTR_ORIENTATIONS);
    const int64_t* aperture = attrs->GetInt(ATTR_APERTURE);
    const float* l2Hys = attrs->GetFloat(ATTR_L2_HYS);
    const float* eps = attrs->GetFloat(ATTR_EPS);
    const bool* gamma = attrs->GetAttrPointer<bool>(ATTR_GAMMA);
    const bool* signedG = attrs->GetAttrPointer<bool>(ATTR_SIGNED);
    const char* normStr = attrs->GetAttrPointer<char>(ATTR_BLOCK_NORM);
    OP_CHECK_IF((ppc != nullptr && ppc->GetSize() != 2) || (cpb != nullptr && cpb->GetSize() != 2) ||
                    (bst != nullptr && bst->GetSize() != 2),
                OP_LOGE(context, "pixels_per_cell, cells_per_block and block_stride must each contain two integers."),
                return ge::GRAPH_FAILED);
    static const int64_t defPair[2] = {8, 8};
    static const int64_t defPairB[2] = {2, 2};
    static const int64_t defPairS[2] = {1, 1};
    static const char defNorm[] = "L2-Hys";
    const int64_t* ppcV = ppc != nullptr ? reinterpret_cast<const int64_t*>(ppc->GetData()) : defPair;
    const int64_t* cpbV = cpb != nullptr ? reinterpret_cast<const int64_t*>(cpb->GetData()) : defPairB;
    const int64_t* bstV = bst != nullptr ? reinterpret_cast<const int64_t*>(bst->GetData()) : defPairS;
    const int64_t pH = ppcV[0], pW = ppcV[1];
    const int64_t bH = cpbV[0], bW = cpbV[1];
    const int64_t sH = bstV[0], sW = bstV[1];
    const int64_t nBins = orientations != nullptr ? *orientations : 9;
    const int64_t apertureV = aperture != nullptr ? *aperture : 1;
    const float l2HysV = l2Hys != nullptr ? *l2Hys : 0.2f;
    const float epsV = eps != nullptr ? *eps : 1e-6f;
    const bool gammaV = gamma != nullptr ? *gamma : true;
    const bool signedV = signedG != nullptr ? *signedG : false;
    if (normStr == nullptr) {
        normStr = defNorm;
    }
    OP_CHECK_IF(apertureV != 1, OP_LOGE(context, "derivative_aperture != 1 is not implemented yet."),
                return ge::GRAPH_FAILED);
    // Validate before the cell/block divisions, even when shape inference was bypassed.
    OP_CHECK_IF(pH < 1 || pH > 256 || pW < 1 || pW > 256, OP_LOGE(context, "pixels_per_cell must be in [1,256]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(bH < 1 || bH > 8 || bW < 1 || bW > 8, OP_LOGE(context, "cells_per_block must be in [1,8]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sH < 1 || sH > 8 || sW < 1 || sW > 8, OP_LOGE(context, "block_stride must be in [1,8]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(nBins < 2 || nBins > 36, OP_LOGE(context, "orientations must be in [2,36]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(l2HysV >= 0.0f && l2HysV <= 1.0f), OP_LOGE(context, "l2_hys_threshold must be in [0,1]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(epsV > 0.0f) || !std::isfinite(epsV), OP_LOGE(context, "eps must be a positive finite float."),
                return ge::GRAPH_FAILED);

    uint32_t normMode = 3;
    if (std::strcmp(normStr, "L1") == 0) {
        normMode = 0;
    } else if (std::strcmp(normStr, "L1-sqrt") == 0) {
        normMode = 1;
    } else if (std::strcmp(normStr, "L2") == 0) {
        normMode = 2;
    } else if (std::strcmp(normStr, "L2-Hys") != 0) {
        OP_LOGE(context, "block_norm must be one of L1/L1-sqrt/L2/L2-Hys.");
        return ge::GRAPH_FAILED;
    }

    const int64_t nCellsH = (hDim + pH - 1) / pH;
    const int64_t nCellsW = (wDim + pW - 1) / pW;
    OP_CHECK_IF(nCellsH < bH || nCellsW < bW, OP_LOGE(context, "no complete block fits the cell grid."),
                return ge::GRAPH_FAILED);
    const int64_t nBlocksH = (nCellsH - bH) / sH + 1;
    const int64_t nBlocksW = (nCellsW - bW) / sW + 1;
    const int64_t padH = nCellsH * pH;
    const int64_t padW = nCellsW * pW;
    const int64_t dimD = nBins * bH * bW * nBlocksH * nBlocksW;
    OP_CHECK_IF(nBlocksH <= 0 || nBlocksW <= 0, OP_LOGE(context, "no complete block fits the cell grid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimD * 4 > (1LL << 30), OP_LOGE(context, "descriptor too large."), return ge::GRAPH_FAILED);

    // Account for every explicit InitBuffer allocation, including queue depth,
    // dtype/channel-dependent rows and block-dependent staging. Reserve API scratch.
    const uint32_t nBinsPad = static_cast<uint32_t>((nBins + 7) / 8 * 8);
    const uint64_t fixedBytes = optiling::HogFixedUbBytes(static_cast<uint32_t>(padW), static_cast<uint32_t>(cDim),
                                                          inputBytes, static_cast<uint32_t>(nCellsW), nBinsPad,
                                                          static_cast<uint32_t>(bH), static_cast<uint32_t>(bW));
    OP_CHECK_IF(
        padW > static_cast<int64_t>(optiling::HOG_V1_MAX_STRIP_PIX),
        OP_LOGE(context, "padded width %ld exceeds the current limit %u.", padW, optiling::HOG_V1_MAX_STRIP_PIX),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(fixedBytes >= ubSize, OP_LOGE(context, "fixed buffers and API scratch do not fit UB."),
                return ge::GRAPH_FAILED);
    const int64_t planeBudget = static_cast<int64_t>(ubSize - fixedBytes);
    const int64_t planeRowBytes = nCellsW * nBinsPad * 4;
    // Pick the largest band length that fits; at least 1 block row must fit.
    int64_t bandLen = nBlocksH;
    while (bandLen > 1 && ((bandLen - 1) * sH + bH) * planeRowBytes > planeBudget) {
        --bandLen;
    }
    OP_CHECK_IF(((bandLen - 1) * sH + bH) * planeRowBytes > planeBudget,
                OP_LOGE(context, "cell histogram plane does not fit UB (row bytes %ld, budget %ld).", planeRowBytes,
                        planeBudget),
                return ge::GRAPH_FAILED);

    const int64_t bandsPerImg = (nBlocksH + bandLen - 1) / bandLen;
    const int64_t taskCnt = nDim * bandsPerImg;
    const uint32_t usedCore = taskCnt < static_cast<int64_t>(coreNum) ? static_cast<uint32_t>(taskCnt) : coreNum;

    const bool signedGrad = signedV;
    const double range = signedGrad ? 2.0 * 3.14159265358979323846 : 3.14159265358979323846;

    optiling::HogDescriptorTilingData tiling;
    tiling.set_dimD(static_cast<uint64_t>(dimD));
    tiling.set_n(static_cast<uint32_t>(nDim));
    tiling.set_h(static_cast<uint32_t>(hDim));
    tiling.set_w(static_cast<uint32_t>(wDim));
    tiling.set_c(static_cast<uint32_t>(cDim));
    tiling.set_pH(static_cast<uint32_t>(pH));
    tiling.set_pW(static_cast<uint32_t>(pW));
    tiling.set_bH(static_cast<uint32_t>(bH));
    tiling.set_bW(static_cast<uint32_t>(bW));
    tiling.set_sH(static_cast<uint32_t>(sH));
    tiling.set_sW(static_cast<uint32_t>(sW));
    tiling.set_nBins(static_cast<uint32_t>(nBins));
    tiling.set_nCellsH(static_cast<uint32_t>(nCellsH));
    tiling.set_nCellsW(static_cast<uint32_t>(nCellsW));
    tiling.set_nBlocksH(static_cast<uint32_t>(nBlocksH));
    tiling.set_nBlocksW(static_cast<uint32_t>(nBlocksW));
    tiling.set_padH(static_cast<uint32_t>(padH));
    tiling.set_padW(static_cast<uint32_t>(padW));
    tiling.set_nBinsPad(nBinsPad);
    tiling.set_bandLen(static_cast<uint32_t>(bandLen));
    tiling.set_bandsPerImg(static_cast<uint32_t>(bandsPerImg));
    tiling.set_taskCnt(static_cast<uint32_t>(taskCnt));
    tiling.set_usedCoreNum(usedCore);
    tiling.set_normMode(normMode);
    tiling.set_gammaFlag(gammaV ? 1U : 0U);
    tiling.set_signedFlag(signedGrad ? 1U : 0U);
    tiling.set_binScale(static_cast<float>(static_cast<double>(nBins) / range));
    tiling.set_l2Hys(l2HysV);
    tiling.set_eps(epsV);

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());

    context->SetBlockDim(usedCore);
    context->SetTilingKey(TilingKeyFor(dtype));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseHOGDescriptor([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HOGDescriptor)
    .Tiling(HOGDescriptorTilingFunc)
    .TilingParse<optiling::HogDescriptorCompileInfo>(TilingParseHOGDescriptor);

} // namespace ops
