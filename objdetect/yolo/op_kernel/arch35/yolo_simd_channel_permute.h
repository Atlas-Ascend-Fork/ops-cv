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
 * \file yolo_simd_channel_permute.h
 * \brief AscendC (AIV, DMA through UB) kernel for the Ascend950PR YOLO_MODE_1 float32 fast path
 *
 * The retained SIMT path walks every (n, coef) unit element-wise: one thread reads one float per
 * class with a boxes*HW*4 = 100,320 B stride and 128 B warp bursts, which measures 0.8905 TB/s
 * (55.7 % of HBM peak). For a fixed (n, coef) the boxes*HW = 25,080 floats are CONTIGUOUS in GM
 * (channel index = coef*boxes + b), so this path moves a unit as 8 contiguous 1D DMA sub-blocks
 * through UB, applies 1/(1+exp(-x)) with basic vector ops, and writes the 32 B aligned destination
 * tile contiguously. Coord units keep the real row permutation (w/h swap) with a DataCopyPad row
 * read (2,280 B, 8 B aligned source) and a 2,368 B row write.
 *
 * Only the frozen geometry (boxes*HW == 25,080, 16 <= padCls <= 32, 16 <= padCoord <= 32) is
 * handled here; every other geometry, every other dtype and every non-3510 target compiles and
 * runs the legacy NsYolo::Process<T, YOLO_MODE_1> path, so an accidental key dispatch can never
 * produce wrong output.
 */

#ifndef YOLO_SIMD_CHANNEL_PERMUTE_H_
#define YOLO_SIMD_CHANNEL_PERMUTE_H_

#include "kernel_operator.h"
#include "yolo_tiling_data.h"
#include "yolo_tiling_key.h"
#include "yolo_simt.h"

namespace NsYoloChannelPermute {

using namespace AscendC;

// ==================== Frozen geometry constants ====================
static constexpr uint32_t COORDS = 4;             // coords attribute (x, y, w, h)
static constexpr uint32_t SUB_FLOATS = 3136;      // 12,544 B per DMA sub-block
static constexpr uint32_t SUB_BLOCKS = 8;         // 7 x 3136 + 3128
static constexpr uint32_t UNIT_FLOATS = 25080;    // boxes * HW of the routed geometry
static constexpr uint32_t BLOCK_FLOATS = 3152;    // 3128 + 24 pad -> 12,608 B
static constexpr uint32_t ROW_STAGE_FLOATS = 640; // ceilHW (592) + aligned Duplicate room
static constexpr uint32_t TAIL_DUP = 24;          // padCls of the frozen geometry (96 B)
static constexpr uint32_t PAD_MIN = 16;
static constexpr uint32_t PAD_MAX = 32;

// ==================== Unit DMA + vector sigmoid ====================
class KernelChannelPermute {
public:
    __aicore__ inline KernelChannelPermute(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR coord, GM_ADDR obj, GM_ADDR cls, const YoloTilingData* tiling)
    {
        boxes_ = static_cast<uint32_t>(tiling->boxes);
        hw_ = static_cast<uint32_t>(tiling->HW);
        ceilHW_ = static_cast<uint32_t>(tiling->ceilHW);
        ceilBoxesHw_ = static_cast<uint32_t>(tiling->ceilBoxesHw);
        classes_ = static_cast<uint32_t>(tiling->classes);
        unitsPerImage_ = COORDS + 1 + classes_;
        numUnits_ = static_cast<uint32_t>(tiling->N) * unitsPerImage_;
        padCls_ = ceilBoxesHw_ - UNIT_FLOATS;
        padCoord_ = ceilHW_ - hw_;

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        coordGm_.SetGlobalBuffer((__gm__ float*)coord);
        objGm_.SetGlobalBuffer((__gm__ float*)obj);
        clsGm_.SetGlobalBuffer((__gm__ float*)cls);

        pipe_->InitBuffer(inQue_, 4, SUB_FLOATS * sizeof(float));          // 50,176 B
        pipe_->InitBuffer(outQue_, 4, BLOCK_FLOATS * sizeof(float));       // 50,432 B
        pipe_->InitBuffer(rowInQue_, 4, ROW_STAGE_FLOATS * sizeof(float)); // 10,240 B
        pipe_->InitBuffer(onesBuf_, SUB_FLOATS * sizeof(float));           // 12,544 B
        ones_ = onesBuf_.Get<float>();
        Duplicate(ones_, 1.0f, static_cast<int32_t>(SUB_FLOATS));
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();
        const uint32_t perCore = numUnits_ / coreNum;
        const uint32_t remainder = numUnits_ % coreNum;
        uint32_t u = core * perCore + (core < remainder ? core : remainder);
        const uint32_t count = perCore + (core < remainder ? 1u : 0u);
        for (uint32_t i = 0; i < count; ++i, ++u) {
            const uint32_t img = u / unitsPerImage_;
            const uint32_t coef = u % unitsPerImage_;
            if (coef < COORDS) {
                CoordUnit(img, coef);
            } else if (coef == COORDS) {
                BlockUnit(u, img, objGm_);
            } else {
                BlockUnit(u, img, clsGm_);
            }
        }
    }

private:
    __aicore__ inline void Sigmoid(LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t len)
    {
        const int32_t n = static_cast<int32_t>(len);
        Muls(dst, src, -1.0f, n); // -x
        Exp(dst, dst, n);         // exp(-x)
        Adds(dst, dst, 1.0f, n);  // 1 + exp(-x)
        Div(dst, ones_, dst, n);  // 1 / (1 + exp(-x)), true division
    }

    // One class or obj unit: 8 contiguous sub-blocks, identity for the tile base.
    __aicore__ inline void BlockUnit(uint32_t u, uint32_t img, GlobalTensor<float>& dst)
    {
        const uint32_t srcBase = u * UNIT_FLOATS;
        uint32_t dstBase;
        if (u % unitsPerImage_ == COORDS) {
            dstBase = img * ceilBoxesHw_;
        } else {
            const uint32_t k = (u % unitsPerImage_) - (COORDS + 1);
            dstBase = (img * classes_ + k) * ceilBoxesHw_;
        }
        for (uint32_t s = 0; s < SUB_BLOCKS; ++s) {
            const bool last = (s == SUB_BLOCKS - 1);
            const uint32_t len = last ? (UNIT_FLOATS - (SUB_BLOCKS - 1) * SUB_FLOATS) : SUB_FLOATS;
            const uint32_t outLen = last ? (len + padCls_) : len;

            LocalTensor<float> inBuf = inQue_.AllocTensor<float>();
            DataCopy(inBuf, xGm_[srcBase + s * SUB_FLOATS], len);
            inQue_.EnQue(inBuf);
            inBuf = inQue_.DeQue<float>();

            LocalTensor<float> outBuf = outQue_.AllocTensor<float>();
            Sigmoid(outBuf, inBuf, len);
            if (last) {
                Duplicate(outBuf[len], 0.0f, static_cast<int32_t>(TAIL_DUP));
            }
            outQue_.EnQue(outBuf);
            inQue_.FreeTensor(inBuf);

            outBuf = outQue_.DeQue<float>();
            DataCopy(dst[dstBase + s * SUB_FLOATS], outBuf, outLen);
            outQue_.FreeTensor(outBuf);
        }
    }

    // One coord unit: boxes destination rows of ceilHW floats, source row = b*HW of the unit.
    // coordIdx: coef 0 -> 0 sigmoid, coef 1 -> 1 sigmoid, coef 2 (w) -> 3 move, coef 3 (h) -> 2 move.
    __aicore__ inline void CoordUnit(uint32_t img, uint32_t coef)
    {
        uint32_t coordIdx;
        if (coef == 0) {
            coordIdx = 0;
        } else if (coef == 1) {
            coordIdx = 1;
        } else if (coef == 2) {
            coordIdx = 3;
        } else {
            coordIdx = 2;
        }
        const bool isSigmoid = (coef < 2);
        const uint32_t srcCenter = (img * unitsPerImage_ + coef) * UNIT_FLOATS;
        const uint32_t dstCenter = (img * COORDS * boxes_ + coordIdx * boxes_) * ceilHW_;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        const DataCopyExtParams extParams{1, static_cast<uint32_t>(hw_ * sizeof(float)), 0, 0, 0};
        const int32_t hwCount = static_cast<int32_t>(hw_);
        for (uint32_t b = 0; b < boxes_; ++b) {
            LocalTensor<float> inRow = rowInQue_.AllocTensor<float>();
            DataCopyPad(inRow, xGm_[srcCenter + b * hw_], extParams, padParams);
            rowInQue_.EnQue(inRow);
            inRow = rowInQue_.DeQue<float>();

            LocalTensor<float> outRow = outQue_.AllocTensor<float>();
            // HW is not a 32 B multiple, so the 22 float tail cannot be written with an aligned
            // Duplicate at offset HW. Zero the whole aligned row stage first, then write only the
            // HW real values; the 22-float tail keeps its zero.
            Duplicate(outRow, 0.0f, static_cast<int32_t>(ROW_STAGE_FLOATS));
            if (isSigmoid) {
                Sigmoid(outRow, inRow, hw_);
            } else {
                Muls(outRow, inRow, 1.0f, hwCount);
            }
            outQue_.EnQue(outRow);
            rowInQue_.FreeTensor(inRow);

            outRow = outQue_.DeQue<float>();
            DataCopy(coordGm_[dstCenter + b * ceilHW_], outRow, ceilHW_);
            outQue_.FreeTensor(outRow);
        }
    }

    TPipe* pipe_;
    uint32_t boxes_ = 0;
    uint32_t hw_ = 0;
    uint32_t ceilHW_ = 0;
    uint32_t ceilBoxesHw_ = 0;
    uint32_t classes_ = 0;
    uint32_t unitsPerImage_ = 0;
    uint32_t numUnits_ = 0;
    uint32_t padCls_ = 0;
    uint32_t padCoord_ = 0;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> coordGm_;
    GlobalTensor<float> objGm_;
    GlobalTensor<float> clsGm_;
    TQue<TPosition::VECIN, 4> inQue_;
    TQue<TPosition::VECOUT, 4> outQue_;
    TQue<TPosition::VECIN, 4> rowInQue_;
    TBuf<TPosition::VECCALC> onesBuf_;
    LocalTensor<float> ones_;
};

// ==================== Process Entry ====================
// Only the routed geometry/dtype/target takes the DMA path; everything else (float16, other
// geometries, other architectures) runs the pre-existing NsYolo::Process<T, YOLO_MODE_1> path
// unchanged.
template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR coord_data, GM_ADDR obj_prob, GM_ADDR classes_prob,
                               const YoloTilingData* tilingData)
{
#if __NPU_ARCH__ == 3510
    if constexpr (std::is_same<T, float>::value) {
        const uint32_t unitFloats = static_cast<uint32_t>(tilingData->boxes) * static_cast<uint32_t>(tilingData->HW);
        const uint32_t ceilHW = static_cast<uint32_t>(tilingData->ceilHW);
        const uint32_t padCls = static_cast<uint32_t>(tilingData->ceilBoxesHw) - unitFloats;
        const uint32_t padCoord = ceilHW - static_cast<uint32_t>(tilingData->HW);
        if (unitFloats == UNIT_FLOATS && ceilHW <= ROW_STAGE_FLOATS && padCls >= PAD_MIN && padCls <= PAD_MAX &&
            padCoord >= PAD_MIN && padCoord <= PAD_MAX) {
            TPipe pipe;
            KernelChannelPermute op(&pipe);
            op.Init(x, coord_data, obj_prob, classes_prob, tilingData);
            op.Process();
            return;
        }
    }
#endif
    NsYolo::Process<T, YOLO_MODE_1>(x, coord_data, obj_prob, classes_prob, tilingData);
}

} // namespace NsYoloChannelPermute

#endif // YOLO_SIMD_CHANNEL_PERMUTE_H_
