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
 * \file points_in_polygons.h
 * \brief PointsInPolygons AscendC kernel (ray casting / PNPOLY)
 */

#pragma once
#include "kernel_operator.h"
#include "points_in_polygons_tiling_data.h"
#include "points_in_polygons_tiling_key.h"

// A 1 ULP quotient error can turn a strictly inside point into a boundary
// point. Both coordinate-range paths require correctly rounded FP32 division.
constexpr AscendC::Reg::DivSpecificMode kRayCastNonFtzDivMode = {AscendC::Reg::MaskMergeMode::ZEROING, false,
                                                                 AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
constexpr AscendC::Reg::DivSpecificMode kRayCastNormalDivMode = {AscendC::Reg::MaskMergeMode::ZEROING, false,
                                                                 AscendC::DivAlgo::PRECISION_0ULP_FTZ_TRUE};

constexpr uint32_t VL_F32 = 64; // 256B / 4B = 64 elements per vector register

// Nonzero coordinates in [2^-24, 2^24] have differences in [2^-48, 2^25].
// The multiply-before-divide numerator stays in [2^-96, 2^50], and its
// nonzero quotient stays in [2^-121, 2^98], so FTZ division is safe here.
// Inspect bits so subnormal coordinates and signed zero are handled correctly.
template <typename T>
__simd_vf__ inline void CheckCoordinateRangeVF(__ubuf__ T* points, __ubuf__ T* polygons, __ubuf__ uint32_t* flag,
                                               uint32_t pointStride, uint32_t curN, uint32_t polygonStride,
                                               uint32_t curM)
{
    using namespace AscendC;
    using namespace AscendC::Reg;
    constexpr uint32_t kFloatAbsMask = 0x7fffffffU;
    constexpr uint32_t kFastDivMinAbsCoordinateBits = 0x33800000U; // 2^-24
    constexpr uint32_t kFastDivMaxAbsCoordinateBits = 0x4b800000U; // 2^24
    constexpr uint16_t kPointCoordinateRows = 2;
    constexpr uint16_t kPolygonCoordinateRows = 8;
    RegTensor<uint32_t> bits, absBits, signMask, zero, one, dirty, laneDirty;
    MaskReg mask, small, nonzero, large;
    MaskReg full = CreateMask<uint32_t, MaskPattern::ALL>();
    Duplicate(signMask, kFloatAbsMask);
    Duplicate(zero, 0U);
    Duplicate(one, 1U);
    Duplicate(dirty, 0U);
    // N-vec polygon rows can have an unaligned stride (for example M=9).
    // Scan the contiguous 8*M values from the aligned allocation in that case.
    bool contiguousPolygons = polygonStride == curM;
    uint16_t rows = contiguousPolygons ? kPointCoordinateRows + 1 : kPointCoordinateRows + kPolygonCoordinateRows;
    for (uint16_t row = 0; row < rows; ++row) {
        __ubuf__ T* addr = row < kPointCoordinateRows ? points + row * pointStride :
                                                        polygons + (row - kPointCoordinateRows) * polygonStride;
        uint32_t count = row < kPointCoordinateRows ? curN :
                                                      (contiguousPolygons ? kPolygonCoordinateRows * curM : curM);
        uint32_t remain = count;
        uint16_t loops = static_cast<uint16_t>((count + VL_F32 - 1U) / VL_F32);
        for (uint16_t j = 0; j < loops; ++j) {
            mask = UpdateMask<uint32_t>(remain);
            LoadAlign<uint32_t, DataCopyMode::DATA_BLOCK_COPY>(
                bits, reinterpret_cast<__ubuf__ uint32_t*>(addr) + j * VL_F32, 1, mask);
            And(absBits, bits, signMask, mask);
            Compares<uint32_t, CMPMODE::LT>(small, absBits, kFastDivMinAbsCoordinateBits, mask);
            Compares<uint32_t, CMPMODE::NE>(nonzero, absBits, 0U, mask);
            MaskAnd(small, small, nonzero, mask);
            Compares<uint32_t, CMPMODE::GT>(large, absBits, kFastDivMaxAbsCoordinateBits, mask);
            MaskOr(large, large, small, mask);
            Select(laneDirty, one, zero, large);
            Or(dirty, dirty, laneDirty, full);
        }
    }
    Reduce<ReduceType::MAX>(dirty, dirty, full);
    StoreAlign<uint32_t, StoreDist::DIST_FIRST_ELEMENT_B32>(flag, dirty, full);
}

// NORMAL 分支 VF 计算：对每个点用 Regbase 寄存器级 API（VL=64 分块）执行 4 边射线法，
// 中间量全程在 RegTensor<float> 内融合不落 UB，仅最终 output 经 StoreAlign 写回 B2。
template <typename T, bool preserveSubnormals>
__simd_vf__ inline void RayCastComputeVF(__ubuf__ T* pxAddr, __ubuf__ T* pyAddr, __ubuf__ T* polyBase,
                                         __ubuf__ T* outBase, uint32_t tileM, uint32_t curN, uint32_t curM,
                                         uint16_t vfLoopPerRow)
{
    AscendC::Reg::RegTensor<float> vPy, vPx;
    AscendC::Reg::RegTensor<float> vYsK, vYsPrev;
    AscendC::Reg::RegTensor<float> vXsK, vXsPrev;
    AscendC::Reg::RegTensor<float> vPyMinusYsK, vDy, vDx, vT, vXint;
    AscendC::Reg::RegTensor<float> vCross, vCount, vHalf, vFloor, vTwoFloor, vTmp, vOut;
    AscendC::Reg::RegTensor<float> vZero, vOne;
    AscendC::Reg::MaskReg mask, active, cmpMask1, cmpMask2, cmpCond, cmpEq, cmpXint, cmpOdd, cmpPart;

    AscendC::Reg::Duplicate(vZero, 0.0f);
    AscendC::Reg::Duplicate(vOne, 1.0f);

    // B2 输出行 stride = alignUp(curM,8)，匹配 DoCopyOut 的 DataCopyPad srcStride
    uint32_t cmAligned = (curM + 7U) & ~7U;

    for (uint16_t row = 0; row < (uint16_t)curN; row++) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vPy, pyAddr + (uint32_t)row);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vPx, pxAddr + (uint32_t)row);

        uint32_t rowRemain = curM;
        for (uint16_t j = 0; j < vfLoopPerRow; j++) {
            mask = AscendC::Reg::UpdateMask<float>(rowRemain);
            uint32_t mOff = (uint32_t)j * VL_F32;

            AscendC::Reg::Duplicate(vCount, 0.0f);
            AscendC::Reg::MaskOr(active, mask, mask, mask);

            for (uint16_t k = 0; k < 4; k++) { // MMCV: edges (V0,V3), (V1,V0), (V2,V1), (V3,V2)
                __ubuf__ T* xsKRow = polyBase + (uint32_t)(2 * k) * tileM;
                __ubuf__ T* ysKRow = polyBase + (uint32_t)(2 * k + 1) * tileM;
                __ubuf__ T* xsPrevRow = polyBase + (uint32_t)(2 * ((k + 3) % 4)) * tileM;
                __ubuf__ T* ysPrevRow = polyBase + (uint32_t)(2 * ((k + 3) % 4) + 1) * tileM;
                // Predicate the load itself: tileM may be smaller than one full vector.
                AscendC::Reg::LoadAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(vYsK, ysKRow + mOff, 1,
                                                                                            mask);
                AscendC::Reg::LoadAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(vYsPrev, ysPrevRow + mOff,
                                                                                            1, mask);
                AscendC::Reg::LoadAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(vXsK, xsKRow + mOff, 1,
                                                                                            mask);
                AscendC::Reg::LoadAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(vXsPrev, xsPrevRow + mOff,
                                                                                            1, mask);

                // A vertex hit stops this pair without clearing earlier crossings.
                AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(cmpMask1, vXsK, vPx, mask);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(cmpMask2, vYsK, vPy, mask);
                AscendC::Reg::MaskAnd(cmpEq, cmpMask1, cmpMask2, mask);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(cmpMask1, vXsPrev, vPx, mask);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(cmpMask2, vYsPrev, vPy, mask);
                AscendC::Reg::MaskAnd(cmpMask1, cmpMask1, cmpMask2, mask);
                AscendC::Reg::MaskOr(cmpEq, cmpEq, cmpMask1, mask);
                AscendC::Reg::MaskAnd(cmpEq, cmpEq, active, mask);
                AscendC::Reg::MaskXor(active, active, cmpEq, mask);

                // Only min(sy,ty) < py <= max(sy,ty) can contribute a crossing.
                AscendC::Reg::Compare<float, AscendC::CMPMODE::LT>(cmpMask1, vYsK, vPy, mask);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::GE>(cmpMask2, vYsPrev, vPy, mask);
                AscendC::Reg::MaskAnd(cmpCond, cmpMask1, cmpMask2, mask);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::GE>(cmpMask1, vYsK, vPy, mask);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::LT>(cmpMask2, vYsPrev, vPy, mask);
                AscendC::Reg::MaskAnd(cmpPart, cmpMask1, cmpMask2, mask);
                AscendC::Reg::MaskOr(cmpCond, cmpCond, cmpPart, mask);
                AscendC::Reg::MaskAnd(cmpCond, cmpCond, active, mask);

                // Match MMCV's FP32 evaluation order: sx + ((py-sy)*(tx-sx))/(ty-sy).
                AscendC::Reg::Sub<float>(vPyMinusYsK, vPy, vYsK, cmpCond);
                AscendC::Reg::Sub<float>(vDx, vXsPrev, vXsK, cmpCond);
                AscendC::Reg::Sub<float>(vDy, vYsPrev, vYsK, cmpCond);
                AscendC::Reg::Mul<float>(vT, vPyMinusYsK, vDx, cmpCond);
                if constexpr (preserveSubnormals) {
                    AscendC::Reg::Div<float, &kRayCastNonFtzDivMode>(vT, vT, vDy, cmpCond);
                } else {
                    AscendC::Reg::Div<float, &kRayCastNormalDivMode>(vT, vT, vDy, cmpCond);
                }
                AscendC::Reg::Add<float>(vXint, vT, vXsK, cmpCond);

                // An exact intersection hit also preserves the current count and stops.
                AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(cmpEq, vXint, vPx, cmpCond);
                AscendC::Reg::MaskAnd(cmpEq, cmpEq, cmpCond, mask);
                AscendC::Reg::MaskXor(active, active, cmpEq, mask);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::GT>(cmpXint, vXint, vPx, cmpCond);
                AscendC::Reg::MaskAnd(cmpXint, cmpXint, cmpCond, mask);
                AscendC::Reg::Select<float>(vCross, vOne, vZero, cmpXint);
                AscendC::Reg::Add<float>(vCount, vCount, vCross, mask);
            }

            AscendC::Reg::Muls<float>(vHalf, vCount, 0.5f, mask);
            AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_FLOOR, AscendC::Reg::MaskMergeMode::ZEROING>(
                vFloor, vHalf, mask);
            AscendC::Reg::Muls<float>(vTwoFloor, vFloor, 2.0f, mask);
            AscendC::Reg::Sub<float>(vTmp, vCount, vTwoFloor, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::NE>(cmpOdd, vTmp, 0.0f, mask);
            AscendC::Reg::Select<float>(vOut, vOne, vZero, cmpOdd);

            __ubuf__ T* outRow = outBase + (uint32_t)row * cmAligned + mOff;
            AscendC::Reg::StoreAlign(outRow, vOut, mask);
        }
    }
}

// N-vec 分支 VF 计算：广播单个多边形边坐标为标量，VF 沿 N 轴处理 64 个点
template <typename T, bool preserveSubnormals>
__simd_vf__ inline void RayCastComputeVF_N(__ubuf__ T* pxAddr, __ubuf__ T* pyAddr, __ubuf__ T* polyBase,
                                           __ubuf__ T* outBase, uint32_t curNVec, uint32_t M, uint32_t mIdx)
{
    AscendC::Reg::RegTensor<float> vPx, vPy;
    AscendC::Reg::RegTensor<float> vSxK, vSyK, vTxPrev, vTyPrev;
    AscendC::Reg::RegTensor<float> vPyMinusSyK, vDy, vDx, vT, vXint;
    AscendC::Reg::RegTensor<float> vCross, vCount, vHalf, vFloor, vTwoFloor, vTmp, vOut;
    AscendC::Reg::RegTensor<float> vZero, vOne;
    AscendC::Reg::MaskReg mask, active, cmpMask1, cmpMask2, cmpCond, cmpEq, cmpXint, cmpOdd, cmpPart;

    AscendC::Reg::Duplicate(vZero, 0.0f);
    AscendC::Reg::Duplicate(vOne, 1.0f);

    uint32_t rowRemain = curNVec;
    uint16_t vfLoops = (uint16_t)((curNVec + VL_F32 - 1U) / VL_F32);

    for (uint16_t j = 0; j < vfLoops; j++) {
        mask = AscendC::Reg::UpdateMask<float>(rowRemain);
        uint32_t mOff = (uint32_t)j * VL_F32;

        AscendC::Reg::LoadAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(vPx, pxAddr + mOff, 1, mask);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(vPy, pyAddr + mOff, 1, mask);

        AscendC::Reg::Duplicate(vCount, 0.0f);
        AscendC::Reg::MaskOr(active, mask, mask, mask);

        for (uint16_t k = 0; k < 4; k++) {
            uint32_t kPrev = (uint32_t)((k + 3) % 4);
            // (8, M) row-major: addr = row*M + col=mIdx
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                vSyK, polyBase + (uint32_t)(2 * k + 1) * M + mIdx);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vTyPrev,
                                                                                 polyBase + (2 * kPrev + 1) * M + mIdx);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                vSxK, polyBase + (uint32_t)(2 * k) * M + mIdx);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vTxPrev,
                                                                                 polyBase + (2 * kPrev) * M + mIdx);

            // A vertex hit stops this pair without clearing earlier crossings.
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(cmpMask1, vSxK, vPx, mask);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(cmpMask2, vSyK, vPy, mask);
            AscendC::Reg::MaskAnd(cmpEq, cmpMask1, cmpMask2, mask);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(cmpMask1, vTxPrev, vPx, mask);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(cmpMask2, vTyPrev, vPy, mask);
            AscendC::Reg::MaskAnd(cmpMask1, cmpMask1, cmpMask2, mask);
            AscendC::Reg::MaskOr(cmpEq, cmpEq, cmpMask1, mask);
            AscendC::Reg::MaskAnd(cmpEq, cmpEq, active, mask);
            AscendC::Reg::MaskXor(active, active, cmpEq, mask);

            // Only min(sy,ty) < py <= max(sy,ty) can contribute a crossing.
            AscendC::Reg::Compare<float, AscendC::CMPMODE::LT>(cmpMask1, vSyK, vPy, mask);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::GE>(cmpMask2, vTyPrev, vPy, mask);
            AscendC::Reg::MaskAnd(cmpCond, cmpMask1, cmpMask2, mask);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::GE>(cmpMask1, vSyK, vPy, mask);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::LT>(cmpMask2, vTyPrev, vPy, mask);
            AscendC::Reg::MaskAnd(cmpPart, cmpMask1, cmpMask2, mask);
            AscendC::Reg::MaskOr(cmpCond, cmpCond, cmpPart, mask);
            AscendC::Reg::MaskAnd(cmpCond, cmpCond, active, mask);

            // Match MMCV's FP32 evaluation order: sx + ((py-sy)*(tx-sx))/(ty-sy).
            AscendC::Reg::Sub<float>(vPyMinusSyK, vPy, vSyK, cmpCond);
            AscendC::Reg::Sub<float>(vDx, vTxPrev, vSxK, cmpCond);
            AscendC::Reg::Sub<float>(vDy, vTyPrev, vSyK, cmpCond);
            AscendC::Reg::Mul<float>(vT, vPyMinusSyK, vDx, cmpCond);
            if constexpr (preserveSubnormals) {
                AscendC::Reg::Div<float, &kRayCastNonFtzDivMode>(vT, vT, vDy, cmpCond);
            } else {
                AscendC::Reg::Div<float, &kRayCastNormalDivMode>(vT, vT, vDy, cmpCond);
            }
            AscendC::Reg::Add<float>(vXint, vT, vSxK, cmpCond);

            // An exact intersection hit also preserves the current count and stops.
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(cmpEq, vXint, vPx, cmpCond);
            AscendC::Reg::MaskAnd(cmpEq, cmpEq, cmpCond, mask);
            AscendC::Reg::MaskXor(active, active, cmpEq, mask);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::GT>(cmpXint, vXint, vPx, cmpCond);
            AscendC::Reg::MaskAnd(cmpXint, cmpXint, cmpCond, mask);
            AscendC::Reg::Select<float>(vCross, vOne, vZero, cmpXint);
            AscendC::Reg::Add<float>(vCount, vCount, vCross, mask);
        }

        AscendC::Reg::Muls<float>(vHalf, vCount, 0.5f, mask);
        AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_FLOOR, AscendC::Reg::MaskMergeMode::ZEROING>(
            vFloor, vHalf, mask);
        AscendC::Reg::Muls<float>(vTwoFloor, vFloor, 2.0f, mask);
        AscendC::Reg::Sub<float>(vTmp, vCount, vTwoFloor, mask);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(cmpOdd, vTmp, vZero, mask);
        AscendC::Reg::Select<float>(vOut, vOne, vZero, cmpOdd);

        AscendC::Reg::StoreAlign(outBase + mOff, vOut, mask);
    }
}

template <typename T, int KEY>
class PointsInPolygonsKernel {
public:
    __aicore__ inline void Init(GM_ADDR points, GM_ADDR polygons, GM_ADDR output, const PointsInPolygonsTilingData* td)
    {
        td_ = td;
        N_ = td->outN;
        M_ = td->outM;
        tileN_ = td->tileN;
        tileM_ = td->tileM;
        tileNVec_ = td->tileNVec;
        numTilesM_ = (int64_t)td->numTilesM;

        gmIn_[0].SetGlobalBuffer((__gm__ T*)points);
        gmIn_[1].SetGlobalBuffer((__gm__ T*)polygons);
        gmOut_[0].SetGlobalBuffer((__gm__ T*)output);

        if constexpr (KEY == POINTS_IN_POLYGONS_KEY_EMPTY) {
            return;
        }

        if constexpr (KEY == POINTS_IN_POLYGONS_KEY_N_VEC) {
            slotFloats_[0] = 2 * tileNVec_;  // B0: pointsT (2, tileNVec)
            slotFloats_[1] = 8 * M_;         // B1: polygons (8, M)
            slotFloats_[2] = M_ * tileNVec_; // B2: output [M, tileNVec]
            slotFloats_[3] = 1;
            slotFloats_[4] = 1;
            slotFloats_[5] = 1;
            for (int i = 0; i < kPhysNodes; i++) {
                pipe_.InitBuffer(buf_[i], (uint32_t)(slotFloats_[i] * 2 * sizeof(T)));
            }
            pipe_.InitBuffer(maskBuf_, 8);

            // points NDDMA transpose: (tileNVec, 2) → (2, tileNVec)
            pointsNddma_.loopSrcStride[0] = (uint64_t)kPointDim;
            pointsNddma_.loopSrcStride[1] = 1;
            pointsNddma_.loopDstStride[0] = 1;
            pointsNddma_.loopDstStride[1] = (uint32_t)tileNVec_;
            pointsNddma_.loopSize[0] = (uint32_t)tileNVec_;
            pointsNddma_.loopSize[1] = (uint32_t)kPointDim;

            // polygons: GM (8, M) → UB (8, M), strided slice copy, no transpose
            polygonsNddma_.loopSrcStride[0] = 1;
            polygonsNddma_.loopSrcStride[1] = (uint64_t)M_;
            polygonsNddma_.loopDstStride[0] = 1;
            polygonsNddma_.loopDstStride[1] = (uint32_t)M_;
            polygonsNddma_.loopSize[0] = (uint32_t)M_;
            polygonsNddma_.loopSize[1] = (uint32_t)kPolyVertices;
            return;
        }

        // KEY == NORMAL: 6 TBuf ×2 ping-pong
        constexpr int64_t kCmpAlignElems = 64; // 256B / sizeof(float32)
        slotFloats_[0] = 2 * tileN_;
        slotFloats_[1] = 8 * tileM_;
        slotFloats_[2] = tileN_ * tileM_;
        int64_t tmpSlot = (tileM_ > kCmpAlignElems) ? tileM_ : kCmpAlignElems;
        slotFloats_[3] = tmpSlot;
        slotFloats_[4] = tmpSlot;
        slotFloats_[5] = tmpSlot;
        for (int i = 0; i < kPhysNodes; i++) {
            pipe_.InitBuffer(buf_[i], (uint32_t)(slotFloats_[i] * 2 * sizeof(T)));
        }
        pipe_.InitBuffer(maskBuf_, 8);

        // points NDDMA transpose: (tileN, 2) → (2, tileN)
        pointsNddma_.loopSrcStride[0] = (uint64_t)kPointDim;
        pointsNddma_.loopSrcStride[1] = 1;
        pointsNddma_.loopDstStride[0] = 1;
        pointsNddma_.loopDstStride[1] = (uint32_t)tileN_;
        pointsNddma_.loopSize[0] = (uint32_t)tileN_;
        pointsNddma_.loopSize[1] = (uint32_t)kPointDim;

        // polygons: GM (8, M) → UB (8, tileM), strided slice copy, no transpose
        polygonsNddma_.loopSrcStride[0] = 1;
        polygonsNddma_.loopSrcStride[1] = (uint64_t)M_;
        polygonsNddma_.loopDstStride[0] = 1;
        polygonsNddma_.loopDstStride[1] = (uint32_t)tileM_;
        polygonsNddma_.loopSize[0] = (uint32_t)tileM_;
        polygonsNddma_.loopSize[1] = (uint32_t)kPolyVertices;
    }

    __aicore__ inline void Process()
    {
        if constexpr (KEY == POINTS_IN_POLYGONS_KEY_EMPTY) {
            return;
        }
        if constexpr (KEY == POINTS_IN_POLYGONS_KEY_N_VEC) {
            ProcessNVec();
            return;
        }
        uint64_t start = AscendC::GetBlockIdx() * td_->perCoreCount;
        uint64_t end = (start + td_->perCoreCount > td_->totalTiles) ? td_->totalTiles : start + td_->perCoreCount;
        for (uint64_t r = start; r < end; r++) {
            int64_t round = (int64_t)(r - start);
            int64_t nIdx, mIdx, curN, curM;
            CalcTileCoord((int64_t)r, nIdx, mIdx, curN, curM);

            if (round > 1) {
                // ping-pong slot reuse: round R reuses slot (R&1) last written by round R-2
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((uint8_t)(round & 1));
            }

            DoCopyIn(nIdx, mIdx, curN, curM, round);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((uint8_t)(round & 1));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>((uint8_t)(round & 1));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((uint8_t)(round & 1));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>((uint8_t)(round & 1));

            Compute(curN, curM, round);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((uint8_t)(round & 1));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((uint8_t)(round & 1));

            DoCopyOut(nIdx, mIdx, curN, curM, round);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((uint8_t)(round & 1));
        }
    }

private:
    static constexpr int64_t kPhysNodes = 6;
    static constexpr int64_t kMaxInputSlots = 2;
    static constexpr int64_t kMaxOutputSlots = 1;
    static constexpr int64_t kNdDmaDim = 2;
    static constexpr int64_t kPointDim = 2;     // points: [N, 2]
    static constexpr int64_t kPolyVertices = 8; // polygons: [8, M]

    const PointsInPolygonsTilingData* td_ = nullptr;
    int64_t tileN_ = 0;
    int64_t tileM_ = 0;
    int64_t tileNVec_ = 0;
    int64_t N_ = 0;
    int64_t M_ = 0;
    int64_t numTilesM_ = 0;

    AscendC::GlobalTensor<T> gmIn_[kMaxInputSlots];
    AscendC::GlobalTensor<T> gmOut_[kMaxOutputSlots];

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    int64_t slotFloats_[kPhysNodes] = {0};

    AscendC::NdDmaLoopInfo<kNdDmaDim> pointsNddma_{};
    AscendC::NdDmaLoopInfo<kNdDmaDim> polygonsNddma_{};

    __aicore__ inline AscendC::LocalTensor<T> Buf(int64_t i, int64_t round)
    {
        int64_t off = (round & 1) * slotFloats_[i];
        return buf_[i].template Get<T>()[(uint32_t)off];
    }

    __aicore__ inline void CalcTileCoord(int64_t tileIdx, int64_t& nIdx, int64_t& mIdx, int64_t& curN, int64_t& curM)
    {
        nIdx = tileIdx / numTilesM_;
        mIdx = tileIdx % numTilesM_;
        curN = (nIdx + 1) * tileN_ > N_ ? (N_ - nIdx * tileN_) : tileN_;
        if (curN < 1) {
            curN = 1;
        }
        curM = (mIdx + 1) * tileM_ > M_ ? (M_ - mIdx * tileM_) : tileM_;
        if (curM < 1) {
            curM = 1;
        }
    }

    __aicore__ inline void DoCopyIn(int64_t nIdx, int64_t mIdx, int64_t curN, int64_t curM, int64_t round)
    {
        AscendC::LocalTensor<T> B0 = Buf(0, round);
        AscendC::LocalTensor<T> B1 = Buf(1, round);

        // points: (curN, 2) → (2, curN) → B0
        AscendC::NdDmaParams<T, kNdDmaDim> pointsParams{};
        pointsParams.loopInfo = pointsNddma_;
        pointsParams.loopInfo.loopSize[0] = (uint32_t)curN;
        pointsParams.constantValue = (T)0.0f;
        int64_t pointsGmOff = nIdx * tileN_ * kPointDim;
        AscendC::DataCopy<T, kNdDmaDim>(B0, gmIn_[0][pointsGmOff], pointsParams);

        // polygons: GM (8, M) slice → UB (8, tileM)
        AscendC::NdDmaParams<T, kNdDmaDim> polygonsParams{};
        polygonsParams.loopInfo = polygonsNddma_;
        polygonsParams.loopInfo.loopSize[0] = (uint32_t)curM;
        polygonsParams.constantValue = (T)0.0f;
        int64_t polygonsGmOff = mIdx * tileM_;
        AscendC::DataCopy<T, kNdDmaDim>(B1, gmIn_[1][polygonsGmOff], polygonsParams);
    }

    __aicore__ inline void Compute(int64_t curN, int64_t curM, int64_t round)
    {
        AscendC::LocalTensor<T> B0 = Buf(0, round);
        AscendC::LocalTensor<T> B1 = Buf(1, round);
        AscendC::LocalTensor<T> B2 = Buf(2, round);

        __ubuf__ T* pxAddr = (__ubuf__ T*)B0.GetPhyAddr();
        __ubuf__ T* pyAddr = pxAddr + (uint32_t)tileN_;
        __ubuf__ T* polyBase = (__ubuf__ T*)B1.GetPhyAddr();
        __ubuf__ T* outBase = (__ubuf__ T*)B2.GetPhyAddr();

        uint16_t vfLoopPerRow = (uint16_t)(((uint32_t)curM + VL_F32 - 1U) / VL_F32);

        if (NeedsSubnormalDivision(pxAddr, polyBase, tileN_, curN, tileM_, curM)) {
            asc_vf_call<RayCastComputeVF<T, true>>(pxAddr, pyAddr, polyBase, outBase, (uint32_t)tileM_, (uint32_t)curN,
                                                   (uint32_t)curM, vfLoopPerRow);
        } else {
            asc_vf_call<RayCastComputeVF<T, false>>(pxAddr, pyAddr, polyBase, outBase, (uint32_t)tileM_, (uint32_t)curN,
                                                    (uint32_t)curM, vfLoopPerRow);
        }
    }

    __aicore__ inline bool NeedsSubnormalDivision(__ubuf__ T* points, __ubuf__ T* polygons, uint32_t pointStride,
                                                  uint32_t curN, uint32_t polygonStride, uint32_t curM)
    {
        auto flag = maskBuf_.Get<uint32_t>();
        asc_vf_call<CheckCoordinateRangeVF<T>>(points, polygons, (__ubuf__ uint32_t*)flag.GetPhyAddr(), pointStride,
                                               curN, polygonStride, curM);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        return flag.GetValue(0) != 0;
    }

    __aicore__ inline void DoCopyOut(int64_t nIdx, int64_t mIdx, int64_t curN, int64_t curM, int64_t round)
    {
        AscendC::LocalTensor<T> B2 = Buf(2, round);
        int64_t outGmOff = nIdx * tileN_ * M_ + mIdx * tileM_;

        int64_t cmAligned = ((curM + 7) >> 3) << 3;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = (uint16_t)curN;
        copyParams.blockLen = (uint32_t)(curM * sizeof(T));
        // srcStride: 32B-block units (3510 UB→GM DataCopyPad convention)
        copyParams.srcStride = (int64_t)((cmAligned - curM) * (int64_t)sizeof(T) / 32);
        copyParams.dstStride = (int64_t)((M_ - curM) * sizeof(T));
        copyParams.rsv = 0;

        AscendC::DataCopyPad<T>(gmOut_[0][outGmOff], B2, copyParams);
    }

    // N-vec branch (KEY=2): vectorize along N axis; tile = tileNVec points × 1 polygon
    __aicore__ inline void ProcessNVec()
    {
        uint64_t start = AscendC::GetBlockIdx() * td_->perCoreCount;
        uint64_t end = (start + td_->perCoreCount > td_->totalTiles) ? td_->totalTiles : start + td_->perCoreCount;
        for (uint64_t r = start; r < end; r++) {
            int64_t nIdx = (int64_t)r;
            int64_t curNVec = (nIdx + 1) * tileNVec_ > N_ ? (N_ - nIdx * tileNVec_) : tileNVec_;
            if (curNVec < 1) {
                curNVec = 1;
            }

            if (r > start) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            }

            DoCopyInNVec(nIdx, curNVec, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

            ComputeNVec(curNVec, 0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

            DoCopyOutStepA(nIdx, curNVec, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);

            AscendC::NdDmaDci();
            DoCopyInTransposeBack(nIdx, curNVec, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

            DoCopyOutStepC(nIdx, curNVec, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
    }

    __aicore__ inline void DoCopyInNVec(int64_t nIdx, int64_t curNVec, int64_t round)
    {
        AscendC::LocalTensor<T> B0 = Buf(0, round);
        AscendC::LocalTensor<T> B1 = Buf(1, round);

        AscendC::NdDmaParams<T, kNdDmaDim> pointsParams{};
        pointsParams.loopInfo = pointsNddma_;
        pointsParams.loopInfo.loopSize[0] = (uint32_t)curNVec;
        pointsParams.constantValue = (T)0.0f;
        int64_t pointsGmOff = nIdx * tileNVec_ * kPointDim;
        AscendC::DataCopy<T, kNdDmaDim>(B0, gmIn_[0][pointsGmOff], pointsParams);

        // polygons: GM (8, M) → UB (8, M), no transpose, GM offset=0
        AscendC::NdDmaParams<T, kNdDmaDim> polygonsParams{};
        polygonsParams.loopInfo = polygonsNddma_;
        polygonsParams.constantValue = (T)0.0f;
        AscendC::DataCopy<T, kNdDmaDim>(B1, gmIn_[1][0], polygonsParams);
    }

    __aicore__ inline void ComputeNVec(int64_t curNVec, int64_t round)
    {
        AscendC::LocalTensor<T> B0 = Buf(0, round);
        AscendC::LocalTensor<T> B1 = Buf(1, round);
        AscendC::LocalTensor<T> B2 = Buf(2, round);

        __ubuf__ T* pxAddr = (__ubuf__ T*)B0.GetPhyAddr();
        __ubuf__ T* pyAddr = pxAddr + (uint32_t)tileNVec_;
        __ubuf__ T* polyBase = (__ubuf__ T*)B1.GetPhyAddr();
        __ubuf__ T* outBase = (__ubuf__ T*)B2.GetPhyAddr();

        bool preserveSubnormals = NeedsSubnormalDivision(pxAddr, polyBase, tileNVec_, curNVec, M_, M_);
        for (int64_t polyIdx = 0; polyIdx < M_; polyIdx++) {
            __ubuf__ T* outPtr = outBase + polyIdx * tileNVec_;
            if (preserveSubnormals) {
                asc_vf_call<RayCastComputeVF_N<T, true>>(pxAddr, pyAddr, polyBase, outPtr, (uint32_t)curNVec,
                                                         (uint32_t)M_, (uint32_t)polyIdx);
            } else {
                asc_vf_call<RayCastComputeVF_N<T, false>>(pxAddr, pyAddr, polyBase, outPtr, (uint32_t)curNVec,
                                                          (uint32_t)M_, (uint32_t)polyIdx);
            }
        }
    }

    // Step A: copy each polygon's results (B2 [M, tileNVec]) to GM [M, curNVec]
    __aicore__ inline void DoCopyOutStepA(int64_t nIdx, int64_t curNVec, int64_t round)
    {
        AscendC::LocalTensor<T> B2 = Buf(2, round);
        int64_t outGmOff = nIdx * tileNVec_ * M_;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = (uint32_t)(curNVec * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        for (int64_t polyIdx = 0; polyIdx < M_; polyIdx++) {
            auto bLocal = B2[(uint32_t)(polyIdx * tileNVec_)];
            AscendC::DataCopyPad<T>(gmOut_[0][outGmOff + polyIdx * curNVec], bLocal, copyParams);
        }
    }

    // Step B: NDDMA transpose GM [M, curNVec] → UB [curNVec, M] (call after NdDmaDci)
    __aicore__ inline void DoCopyInTransposeBack(int64_t nIdx, int64_t curNVec, int64_t round)
    {
        AscendC::LocalTensor<T> B2 = Buf(2, round);
        int64_t outGmOff = nIdx * tileNVec_ * M_;

        AscendC::NdDmaLoopInfo<kNdDmaDim> loopInfo{};
        loopInfo.loopSrcStride[0] = static_cast<uint64_t>(curNVec);
        loopInfo.loopDstStride[0] = 1;
        loopInfo.loopSize[0] = (uint32_t)M_;
        loopInfo.loopSrcStride[1] = 1;
        loopInfo.loopDstStride[1] = (uint32_t)M_;
        loopInfo.loopSize[1] = (uint32_t)curNVec;

        AscendC::NdDmaParams<T, kNdDmaDim> params{loopInfo, 0};
        AscendC::DataCopy<T, kNdDmaDim>(B2, gmOut_[0][outGmOff], params);
    }

    // Step C: copy UB [curNVec, M] to GM output (final)
    __aicore__ inline void DoCopyOutStepC(int64_t nIdx, int64_t curNVec, int64_t round)
    {
        AscendC::LocalTensor<T> B2 = Buf(2, round);
        int64_t outGmOff = nIdx * tileNVec_ * M_;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = (uint32_t)(curNVec * M_ * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPad<T>(gmOut_[0][outGmOff], B2, copyParams);
    }
};
