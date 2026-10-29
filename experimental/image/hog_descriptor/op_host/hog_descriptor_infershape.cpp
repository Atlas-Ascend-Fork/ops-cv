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

namespace ops {

// Attribute indices (must match proto declaration order).
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

static constexpr int64_t MAX_N = 200;
static constexpr int64_t MAX_HW = 10000;
static constexpr int64_t MAX_CELL = 256;
static constexpr int64_t MAX_BLOCK = 8;
static constexpr int64_t MAX_STRIDE = 8;
static constexpr int64_t MIN_BINS = 2;
static constexpr int64_t MAX_BINS = 36;
// Guard: keep per-image descriptor byte size within int32-friendly limits.
static constexpr int64_t MAX_DESCRIPTOR_BYTES = 1LL << 30;

struct HogParams {
    int64_t pH = 8, pW = 8, bH = 2, bW = 2, sH = 1, sW = 1;
    int64_t nBins = 9;
    int normMode = 3; // 0=L1 1=L1-sqrt 2=L2 3=L2-Hys
    double l2Hys = 0.2;
    bool gamma = true;
    bool signedGrad = false;
    int64_t aperture = 1;
    double eps = 1e-6;
    int64_t n = 1, h = 0, w = 0, c = 1;
};

static bool ReadHogParams(gert::InferShapeContext* context, const gert::Shape* inputShape, HogParams& p,
                          int64_t& nCellsH, int64_t& nCellsW, int64_t& nBlocksH, int64_t& nBlocksW, int64_t& dimD)
{
    const size_t rank = inputShape->GetDimNum();
    if (rank == 2U) {
        p.n = 1;
        p.h = inputShape->GetDim(0);
        p.w = inputShape->GetDim(1);
        p.c = 1;
    } else if (rank == 3U) {
        const int64_t last = inputShape->GetDim(2);
        if (last == 1 || last == 3) {
            p.n = 1;
            p.h = inputShape->GetDim(0);
            p.w = inputShape->GetDim(1);
            p.c = last;
        } else {
            p.n = inputShape->GetDim(0);
            p.h = inputShape->GetDim(1);
            p.w = inputShape->GetDim(2);
            p.c = 1;
        }
    } else if (rank == 4U) {
        p.n = inputShape->GetDim(0);
        p.h = inputShape->GetDim(1);
        p.w = inputShape->GetDim(2);
        p.c = inputShape->GetDim(3);
    } else {
        OP_LOGE(context, "HOGDescriptor only supports rank 2/3/4 ND tensors.");
        return false;
    }
    if (p.n <= 0 || p.h <= 0 || p.w <= 0 || p.c <= 0) {
        OP_LOGE(context, "HOGDescriptor does not support empty tensors.");
        return false;
    }
    if (p.n > MAX_N) {
        OP_LOGE(context, "batch N=%ld exceeds the limit %ld.", p.n, MAX_N);
        return false;
    }
    if (p.h > MAX_HW || p.w > MAX_HW) {
        OP_LOGE(context, "H/W=%ld/%ld exceeds the limit %ld.", p.h, p.w, MAX_HW);
        return false;
    }
    if (p.c != 1 && p.c != 3) {
        OP_LOGE(context, "channel C must be 1 or 3, got %ld.", p.c);
        return false;
    }

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context, "HOGDescriptor attribute storage is missing.");
        return false;
    }
    // Null-tolerant reads with spec defaults: the aclnn autogen tool may probe the op
    // without materialized attribute values.
    const gert::TypedContinuousVector<int64_t>* ppc = attrs->GetListInt(ATTR_PIXELS_PER_CELL);
    const gert::TypedContinuousVector<int64_t>* cpb = attrs->GetListInt(ATTR_CELLS_PER_BLOCK);
    const gert::TypedContinuousVector<int64_t>* bst = attrs->GetListInt(ATTR_BLOCK_STRIDE);
    const int64_t* orientations = attrs->GetInt(ATTR_ORIENTATIONS);
    const char* normStr = attrs->GetAttrPointer<char>(ATTR_BLOCK_NORM);
    const float* l2Hys = attrs->GetFloat(ATTR_L2_HYS);
    const bool* gamma = attrs->GetAttrPointer<bool>(ATTR_GAMMA);
    const bool* signedG = attrs->GetAttrPointer<bool>(ATTR_SIGNED);
    const int64_t* aperture = attrs->GetInt(ATTR_APERTURE);
    const float* eps = attrs->GetFloat(ATTR_EPS);
    static const int64_t defPair[2] = {8, 8};
    static const int64_t defPairB[2] = {2, 2};
    static const int64_t defPairS[2] = {1, 1};
    static const int64_t defBins = 9;
    static const char defNorm[] = "L2-Hys";
    static const float defL2Hys = 0.2f;
    static const bool defGamma = true;
    static const bool defSigned = false;
    static const int64_t defAperture = 1;
    static const float defEps = 1e-6f;
    if ((ppc != nullptr && ppc->GetSize() != 2) || (cpb != nullptr && cpb->GetSize() != 2) ||
        (bst != nullptr && bst->GetSize() != 2)) {
        OP_LOGE(context, "pixels_per_cell, cells_per_block and block_stride must each contain two integers.");
        return false;
    }
    const int64_t* ppcV = ppc != nullptr ? reinterpret_cast<const int64_t*>(ppc->GetData()) : defPair;
    const int64_t* cpbV = cpb != nullptr ? reinterpret_cast<const int64_t*>(cpb->GetData()) : defPairB;
    const int64_t* bstV = bst != nullptr ? reinterpret_cast<const int64_t*>(bst->GetData()) : defPairS;
    p.pH = ppcV[0];
    p.pW = ppcV[1];
    p.bH = cpbV[0];
    p.bW = cpbV[1];
    p.sH = bstV[0];
    p.sW = bstV[1];
    p.nBins = orientations != nullptr ? *orientations : defBins;
    p.l2Hys = l2Hys != nullptr ? static_cast<double>(*l2Hys) : static_cast<double>(defL2Hys);
    p.gamma = gamma != nullptr ? *gamma : defGamma;
    p.signedGrad = signedG != nullptr ? *signedG : defSigned;
    p.aperture = aperture != nullptr ? *aperture : defAperture;
    p.eps = eps != nullptr ? static_cast<double>(*eps) : static_cast<double>(defEps);
    if (normStr == nullptr) {
        normStr = defNorm;
    }
    if (p.pH < 1 || p.pH > MAX_CELL || p.pW < 1 || p.pW > MAX_CELL) {
        OP_LOGE(context, "pixels_per_cell must be in [1,%ld], got [%ld,%ld].", MAX_CELL, p.pH, p.pW);
        return false;
    }
    if (p.bH < 1 || p.bH > MAX_BLOCK || p.bW < 1 || p.bW > MAX_BLOCK) {
        OP_LOGE(context, "cells_per_block must be in [1,%ld], got [%ld,%ld].", MAX_BLOCK, p.bH, p.bW);
        return false;
    }
    if (p.sH < 1 || p.sH > MAX_STRIDE || p.sW < 1 || p.sW > MAX_STRIDE) {
        OP_LOGE(context, "block_stride must be in [1,%ld], got [%ld,%ld].", MAX_STRIDE, p.sH, p.sW);
        return false;
    }
    if (p.nBins < MIN_BINS || p.nBins > MAX_BINS) {
        OP_LOGE(context, "orientations must be in [%ld,%ld], got %ld.", MIN_BINS, MAX_BINS, p.nBins);
        return false;
    }
    if (std::strcmp(normStr, "L1") == 0) {
        p.normMode = 0;
    } else if (std::strcmp(normStr, "L1-sqrt") == 0) {
        p.normMode = 1;
    } else if (std::strcmp(normStr, "L2") == 0) {
        p.normMode = 2;
    } else if (std::strcmp(normStr, "L2-Hys") == 0) {
        p.normMode = 3;
    } else {
        OP_LOGE(context, "block_norm must be one of L1/L1-sqrt/L2/L2-Hys, got %s.", normStr);
        return false;
    }
    if (!(p.l2Hys >= 0.0 && p.l2Hys <= 1.0)) {
        OP_LOGE(context, "l2_hys_threshold must be in [0,1].");
        return false;
    }
    if (p.aperture != 1) {
        OP_LOGE(context, "derivative_aperture %ld is not implemented yet (only 1, i.e. [-1,0,1]).", p.aperture);
        return false;
    }
    if (!(p.eps > 0.0) || !std::isfinite(p.eps)) {
        OP_LOGE(context, "eps must be a positive finite float.");
        return false;
    }

    nCellsH = (p.h + p.pH - 1) / p.pH;
    nCellsW = (p.w + p.pW - 1) / p.pW;
    if (nCellsH < p.bH || nCellsW < p.bW) {
        OP_LOGE(context,
                "image %ldx%ld yields %ldx%ld cells, smaller than cells_per_block %ldx%ld (no complete block).", p.h,
                p.w, nCellsH, nCellsW, p.bH, p.bW);
        return false;
    }
    nBlocksH = (nCellsH - p.bH) / p.sH + 1;
    nBlocksW = (nCellsW - p.bW) / p.sW + 1;
    dimD = p.nBins * p.bH * p.bW * nBlocksH * nBlocksW;
    if (dimD * 4 > MAX_DESCRIPTOR_BYTES) {
        OP_LOGE(context, "descriptor D=%ld (%ld bytes) exceeds the %ld-byte limit.", dimD, dimD * 4,
                MAX_DESCRIPTOR_BYTES);
        return false;
    }
    return true;
}

static ge::graphStatus InferShapeHOGDescriptor(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    HogParams p;
    int64_t nCellsH = 0, nCellsW = 0, nBlocksH = 0, nBlocksW = 0, dimD = 0;
    if (!ReadHogParams(context, inputShape, p, nCellsH, nCellsW, nBlocksH, nBlocksW, dimD)) {
        return ge::GRAPH_FAILED;
    }
    if (p.n == 1) {
        *outputShape = gert::Shape({dimD});
    } else {
        *outputShape = gert::Shape({p.n, dimD});
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeHOGDescriptor(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HOGDescriptor).InferShape(InferShapeHOGDescriptor).InferDataType(InferDataTypeHOGDescriptor);

} // namespace ops
