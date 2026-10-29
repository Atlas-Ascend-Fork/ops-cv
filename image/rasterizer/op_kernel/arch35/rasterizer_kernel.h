/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RASTERIZER_KERNEL_H
#define RASTERIZER_KERNEL_H

#include <cstdint>
#include <limits>
#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_functions.h"
#include "rasterizer_soft_float64.h"
#include "rasterizer_tiling_struct.h"

namespace RasterizerKernel {

constexpr uint32_t kVectorBytes = 256U;
constexpr uint32_t kFloatLanes = kVectorBytes / sizeof(float);
constexpr uint32_t kFacePacketBytes = 2U * kVectorBytes;
constexpr uint32_t kFaceValidityScratchBytes = 2U * kVectorBytes;
// The largest fused SIMD VF in PixelDirect currently needs 5888 bytes of UB-backed
// stack (verified with Bisheng --cce-res-usage on CANN 9.2.0). Reserve one
// additional vector quantum so TPipe buffers cannot overlap the VF stack.
constexpr uint32_t kPixelDirectMaxVfStackBytes = 5888U;
constexpr uint32_t kPixelDirectVfStackGuardBytes = kVectorBytes;
constexpr uint32_t kFaceCoordSlotFloats = 32U / sizeof(float);
constexpr uint32_t kFaceCoordBase = kFaceCoordSlotFloats;
constexpr uint32_t kFaceVertexStride = 4U * kFaceCoordSlotFloats;
constexpr uint32_t kPackPixels = 16U;

constexpr AscendC::Reg::CastTrait kF32ToI32Trunc = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT,
                                                    AscendC::Reg::MaskMergeMode::ZEROING,
                                                    AscendC::RoundMode::CAST_TRUNC};
constexpr AscendC::Reg::CastTrait kI32ToF32 = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT,
                                               AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
constexpr AscendC::Reg::DivSpecificMode kHighPrecisionDiv = {AscendC::Reg::MaskMergeMode::ZEROING, true};

__simd_vf__ inline void InitWinnerVF(__ubuf__ float* bestQ, __ubuf__ int32_t* bestFace, __ubuf__ float* bestB0,
                                     __ubuf__ float* bestB1, __ubuf__ float* bestB2, uint32_t tileLen)
{
    AscendC::Reg::RegTensor<float> qReg, zeroReg;
    AscendC::Reg::RegTensor<int32_t> faceReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remain = tileLen;
    const uint16_t loops = static_cast<uint16_t>((tileLen + kFloatLanes - 1U) / kFloatLanes);
    AscendC::Reg::Duplicate(qReg, std::numeric_limits<float>::infinity());
    AscendC::Reg::Duplicate(faceReg, INT32_MAX);
    AscendC::Reg::Duplicate(zeroReg, 0.0f);
    for (uint16_t i = 0; i < loops; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(remain);
        const uint32_t off = static_cast<uint32_t>(i) * kFloatLanes;
        AscendC::Reg::StoreAlign(bestQ + off, qReg, mask);
        AscendC::Reg::StoreAlign(bestFace + off, faceReg, mask);
        AscendC::Reg::StoreAlign(bestB0 + off, zeroReg, mask);
        AscendC::Reg::StoreAlign(bestB1 + off, zeroReg, mask);
        AscendC::Reg::StoreAlign(bestB2 + off, zeroReg, mask);
    }
}

// Branch-0 compares the quantized depth as an integer so values above 2^24 do not
// collapse to the same float key. The barycentric planes are scratch until the winner
// is recomputed, so initialize them together with the integer key and face sentinel.
__simd_vf__ inline void InitWinnerIntVF(__ubuf__ int32_t* bestQ, __ubuf__ int32_t* bestFace, __ubuf__ float* bestB0,
                                        __ubuf__ float* bestB1, __ubuf__ float* bestB2, uint32_t valid)
{
    using namespace AscendC::Reg;
    RegTensor<int32_t> maxReg;
    RegTensor<float> zeroReg;
    MaskReg active;
    Duplicate(maxReg, INT32_MAX);
    Duplicate(zeroReg, 0.0f);
    active = UpdateMask<float>(valid);
    StoreAlign(bestQ, maxReg, active);
    StoreAlign(bestFace, maxReg, active);
    StoreAlign(bestB0, zeroReg, active);
    StoreAlign(bestB1, zeroReg, active);
    StoreAlign(bestB2, zeroReg, active);
}

// Double-single primitives preserve float32 inputs with approximately 48 bits of
// significand. Branch-0 uses them for coverage/depth selection and both branches use
// them to recompute the winning perspective-correct barycentric coordinates.
__simd_callee__ inline void DsAdd(AscendC::Reg::RegTensor<float>& outHi, AscendC::Reg::RegTensor<float>& outLo,
                                  AscendC::Reg::RegTensor<float>& aHi, AscendC::Reg::RegTensor<float>& aLo,
                                  AscendC::Reg::RegTensor<float>& bHi, AscendC::Reg::RegTensor<float>& bLo,
                                  AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    RegTensor<float> sum, virtualB, err, tmp0, tmp1;
    Add(sum, aHi, bHi, mask);
    Sub(virtualB, sum, aHi, mask);
    Sub(tmp0, sum, virtualB, mask);
    Sub(tmp0, aHi, tmp0, mask);
    Sub(tmp1, bHi, virtualB, mask);
    Add(err, tmp0, tmp1, mask);
    Add(err, err, aLo, mask);
    Add(err, err, bLo, mask);
    Add(outHi, sum, err, mask);
    Sub(tmp0, outHi, sum, mask);
    Sub(outLo, err, tmp0, mask);
}

__simd_callee__ inline void DsSub(AscendC::Reg::RegTensor<float>& outHi, AscendC::Reg::RegTensor<float>& outLo,
                                  AscendC::Reg::RegTensor<float>& aHi, AscendC::Reg::RegTensor<float>& aLo,
                                  AscendC::Reg::RegTensor<float>& bHi, AscendC::Reg::RegTensor<float>& bLo,
                                  AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    RegTensor<float> negHi, negLo;
    Muls(negHi, bHi, -1.0f, mask);
    Muls(negLo, bLo, -1.0f, mask);
    DsAdd(outHi, outLo, aHi, aLo, negHi, negLo, mask);
}

__simd_callee__ inline void DsMul(AscendC::Reg::RegTensor<float>& outHi, AscendC::Reg::RegTensor<float>& outLo,
                                  AscendC::Reg::RegTensor<float>& aHi, AscendC::Reg::RegTensor<float>& aLo,
                                  AscendC::Reg::RegTensor<float>& bHi, AscendC::Reg::RegTensor<float>& bLo,
                                  AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    RegTensor<float> product, err, tmp, zero;
    Duplicate(zero, 0.0f);
    Mul(product, aHi, bHi, mask);
    Muls(err, product, -1.0f, mask);
    MulAddDst(err, aHi, bHi, mask);
    Mul(tmp, aHi, bLo, mask);
    Add(err, err, tmp, mask);
    Mul(tmp, aLo, bHi, mask);
    Add(err, err, tmp, mask);
    DsAdd(outHi, outLo, product, zero, err, zero, mask);
}

__simd_callee__ inline void DsDiv(AscendC::Reg::RegTensor<float>& outHi, AscendC::Reg::RegTensor<float>& outLo,
                                  AscendC::Reg::RegTensor<float>& aHi, AscendC::Reg::RegTensor<float>& aLo,
                                  AscendC::Reg::RegTensor<float>& bHi, AscendC::Reg::RegTensor<float>& bLo,
                                  AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    RegTensor<float> q0, q1, zero, productHi, productLo, remHi, remLo, rem;
    Duplicate(zero, 0.0f);
    Div<float, &kHighPrecisionDiv>(q0, aHi, bHi, mask);
    DsMul(productHi, productLo, bHi, bLo, q0, zero, mask);
    DsSub(remHi, remLo, aHi, aLo, productHi, productLo, mask);
    Add(rem, remHi, remLo, mask);
    Div<float, &kHighPrecisionDiv>(q1, rem, bHi, mask);
    DsAdd(outHi, outLo, q0, zero, q1, zero, mask);
}

__simd_callee__ inline void DsInUnitInterval(AscendC::Reg::MaskReg& out, AscendC::Reg::RegTensor<float>& hi,
                                             AscendC::Reg::RegTensor<float>& lo, AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    MaskReg primary, tie, secondary, lower, upper;

    Compares<float, AscendC::CMPMODE::GT>(primary, hi, 0.0f, mask);
    Compares<float, AscendC::CMPMODE::EQ>(tie, hi, 0.0f, mask);
    Compares<float, AscendC::CMPMODE::GE>(secondary, lo, 0.0f, mask);
    And(tie, tie, secondary, mask);
    Or(lower, primary, tie, mask);

    Compares<float, AscendC::CMPMODE::LT>(primary, hi, 1.0f, mask);
    Compares<float, AscendC::CMPMODE::EQ>(tie, hi, 1.0f, mask);
    Compares<float, AscendC::CMPMODE::LE>(secondary, lo, 0.0f, mask);
    And(tie, tie, secondary, mask);
    Or(upper, primary, tie, mask);
    And(out, lower, upper, mask);
}

// Reproduce Golden's per-vertex float64 projection order with double-single values.
// point=3 denotes the exact pixel sample; point 0..2 denotes a packet vertex.
__simd_callee__ inline void ProjectPointCoordDs(
    AscendC::Reg::RegTensor<float>& outHi, AscendC::Reg::RegTensor<float>& outLo, __ubuf__ float* packet,
    uint32_t point, uint32_t axis, AscendC::Reg::RegTensor<float>& pixelX, AscendC::Reg::RegTensor<float>& pixelY,
    AscendC::Reg::RegTensor<float>& extentX, AscendC::Reg::RegTensor<float>& extentY,
    AscendC::Reg::RegTensor<float>& zero, AscendC::Reg::RegTensor<float>& half, AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    if (point == 3U) {
        if (axis == 0U) {
            Adds(outHi, pixelX, 0.0f, mask);
        } else {
            Adds(outHi, pixelY, 0.0f, mask);
        }
        Duplicate(outLo, 0.0f);
        return;
    }
    RegTensor<float> coord, clipW, extent;
    const uint32_t row = kFaceCoordBase + point * kFaceVertexStride;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coord, packet + row + axis * kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(clipW, packet + row + 3U * kFaceCoordSlotFloats);
    if (axis == 0U) {
        Adds(extent, extentX, 0.0f, mask);
    } else {
        Adds(extent, extentY, 0.0f, mask);
    }
    DsDiv(outHi, outLo, coord, zero, clipW, zero, mask);
    DsMul(outHi, outLo, outHi, outLo, half, zero, mask);
    DsAdd(outHi, outLo, outHi, outLo, half, zero, mask);
    DsMul(outHi, outLo, outHi, outLo, extent, zero, mask);
    DsAdd(outHi, outLo, outHi, outLo, half, zero, mask);
}

// Compute the projected-screen zero-area predicate with Branch-0's double-single
// projection and determinant order. This determinant is the negation of the
// authority Golden's _tri_area(A,B,C), so its sign is reversed but its exact-zero
// predicate is equivalent. No epsilon or input-specific special case is introduced;
// Branch-1 proceeds only when the shared Key-0-equivalent predicate is non-zero.
__simd_vf__ inline void ComputeProjectedFaceValidityVF(__ubuf__ float* packet, __ubuf__ int32_t* faceValid,
                                                       float widthMinusOneF, float heightMinusOneF)
{
    using namespace AscendC::Reg;
    RegTensor<float> zero, half, extentX, extentY;
    RegTensor<float> aHi, aLo, edgeHi, edgeLo, otherHi, otherLo;
    RegTensor<int32_t> validI;
    MaskReg active, tempNonZero, nonDegenerate;
    __ubuf__ float* productScratch = reinterpret_cast<__ubuf__ float*>(faceValid);
    constexpr uint32_t kScratchPlaneFloats = kVectorBytes / sizeof(float);

    uint32_t activeLanes = 1U;
    active = UpdateMask<float>(activeLanes);
    Duplicate(zero, 0.0f);
    Duplicate(half, 0.5f);
    Duplicate(extentX, widthMinusOneF);
    Duplicate(extentY, heightMinusOneF);

    // First product: (Cx - Ax) * (By - Ay). Save its double-single pair
    // temporarily in the face-validity UB buffer, allowing the same six vector
    // registers to be reused for the second product without splitting this VF.
    ProjectPointCoordDs(aHi, aLo, packet, 0U, 0U, zero, zero, extentX, extentY, zero, half, active);
    ProjectPointCoordDs(edgeHi, edgeLo, packet, 2U, 0U, zero, zero, extentX, extentY, zero, half, active);
    DsSub(edgeHi, edgeLo, edgeHi, edgeLo, aHi, aLo, active);
    ProjectPointCoordDs(aHi, aLo, packet, 0U, 1U, zero, zero, extentX, extentY, zero, half, active);
    ProjectPointCoordDs(otherHi, otherLo, packet, 1U, 1U, zero, zero, extentX, extentY, zero, half, active);
    DsSub(otherHi, otherLo, otherHi, otherLo, aHi, aLo, active);
    DsMul(edgeHi, edgeLo, edgeHi, edgeLo, otherHi, otherLo, active);
    StoreAlign(productScratch, edgeHi, active);
    StoreAlign(productScratch + kScratchPlaneFloats, edgeLo, active);

    // Second product: (Bx - Ax) * (Cy - Ay), using the identical projection
    // and operation order as the Golden-equivalent determinant.
    ProjectPointCoordDs(aHi, aLo, packet, 0U, 0U, zero, zero, extentX, extentY, zero, half, active);
    ProjectPointCoordDs(edgeHi, edgeLo, packet, 1U, 0U, zero, zero, extentX, extentY, zero, half, active);
    DsSub(edgeHi, edgeLo, edgeHi, edgeLo, aHi, aLo, active);
    ProjectPointCoordDs(aHi, aLo, packet, 0U, 1U, zero, zero, extentX, extentY, zero, half, active);
    ProjectPointCoordDs(otherHi, otherLo, packet, 2U, 1U, zero, zero, extentX, extentY, zero, half, active);
    DsSub(otherHi, otherLo, otherHi, otherLo, aHi, aLo, active);
    DsMul(edgeHi, edgeLo, edgeHi, edgeLo, otherHi, otherLo, active);

    LoadAlign<float, LoadDist::DIST_BRC_B32>(aHi, productScratch);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(aLo, productScratch + kScratchPlaneFloats);
    DsSub(aHi, aLo, aHi, aLo, edgeHi, edgeLo, active);
    Compares<float, AscendC::CMPMODE::NE>(tempNonZero, aHi, 0.0f, active);
    Compares<float, AscendC::CMPMODE::NE>(nonDegenerate, aLo, 0.0f, active);
    Or(nonDegenerate, tempNonZero, nonDegenerate, active);
    Duplicate(validI, 0);
    Adds<int32_t>(validI, validI, 1, nonDegenerate);
    StoreAlign(faceValid, validI, active);
}
// The complete per-face vector chain. Each of the 12 vertex coordinates occupies an
// independent 32-byte UB slot so every DIST_BRC_B32 source is naturally aligned.
__simd_vf__ inline void EvaluateFaceTile(__ubuf__ float* packet, __ubuf__ float* bestQ, __ubuf__ int32_t* bestFace,
                                         __ubuf__ float* bestB0, __ubuf__ float* bestB1, __ubuf__ float* bestB2,
                                         uint32_t valid, float tileStartF, float widthF, float widthMinusOneF,
                                         float heightMinusOneF, int32_t faceIndex)
{
    using namespace AscendC::Reg;
    RegTensor<float> x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2;
    RegTensor<float> hx0, hy0, hx1, hy1, hx2, hy2;
    RegTensor<float> one, zero, n, px, py, widthReg, t0, t1, area;
    RegTensor<float> alpha, beta, gamma, q, tmp, bestQReg;
    RegTensor<float> b0, b1, b2, baryDenom, bestBReg;
    RegTensor<int32_t> intTmp, faceReg, bestFaceReg;
    MaskReg allMask, active, nonDeg, covered, m0, m1, m2, win, tie;

    allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    Duplicate(one, 1.0f);
    Duplicate(zero, 0.0f);
    Duplicate(widthReg, widthF);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(x0, packet + kFaceCoordBase);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(y0, packet + kFaceCoordBase + kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(z0, packet + kFaceCoordBase + 2U * kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(w0, packet + kFaceCoordBase + 3U * kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(x1, packet + kFaceCoordBase + kFaceVertexStride);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(y1, packet + kFaceCoordBase + kFaceVertexStride + kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(z1,
                                             packet + kFaceCoordBase + kFaceVertexStride + 2U * kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(w1,
                                             packet + kFaceCoordBase + kFaceVertexStride + 3U * kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(x2, packet + kFaceCoordBase + 2U * kFaceVertexStride);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(y2,
                                             packet + kFaceCoordBase + 2U * kFaceVertexStride + kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(
        z2, packet + kFaceCoordBase + 2U * kFaceVertexStride + 2U * kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(
        w2, packet + kFaceCoordBase + 2U * kFaceVertexStride + 3U * kFaceCoordSlotFloats);

    uint32_t remain = valid;
    Arange(n, tileStartF);
    const uint16_t loops = static_cast<uint16_t>((valid + kFloatLanes - 1U) / kFloatLanes);
    for (uint16_t loop = 0; loop < loops; ++loop) {
        const uint32_t off = static_cast<uint32_t>(loop) * kFloatLanes;
        active = UpdateMask<float>(remain);
        Div<float, &kHighPrecisionDiv>(py, n, widthReg, active);
        Cast<int32_t, float, kF32ToI32Trunc>(intTmp, py, active);
        Cast<float, int32_t, kI32ToF32>(py, intTmp, active);
        Muls(px, py, widthF, active);
        Sub(px, n, px, active);

        // Evaluate edge functions directly in homogeneous clip coordinates. This is
        // algebraically identical to the Golden viewport/perspective path but avoids
        // early x/w and y/w divisions at coverage and barycentric boundaries.
        Muls(px, px, 2.0f, active);
        Adds(px, px, -widthMinusOneF, active);
        Muls(py, py, 2.0f, active);
        Adds(py, py, -heightMinusOneF, active);

        // Fuse each cancellation-prone ax - bw expression.  One multiplicand is
        // still rounded when it becomes the addend, but the second product and
        // subtraction are rounded only once instead of as separate Mul/Sub ops.
        Muls(hx0, x0, widthMinusOneF, active);
        Muls(tmp, w0, -1.0f, active);
        MulAddDst(hx0, px, tmp, active);
        Muls(hy0, y0, heightMinusOneF, active);
        MulAddDst(hy0, py, tmp, active);
        Muls(hx1, x1, widthMinusOneF, active);
        Muls(tmp, w1, -1.0f, active);
        MulAddDst(hx1, px, tmp, active);
        Muls(hy1, y1, heightMinusOneF, active);
        MulAddDst(hy1, py, tmp, active);
        Muls(hx2, x2, widthMinusOneF, active);
        Muls(tmp, w2, -1.0f, active);
        MulAddDst(hx2, px, tmp, active);
        Muls(hy2, y2, heightMinusOneF, active);
        MulAddDst(hy2, py, tmp, active);

        // Use fused determinants so near-edge barycentric numerators do not lose
        // both product rounding errors before their cancellation.
        Mul(alpha, hx1, hy2, active);
        Muls(alpha, alpha, -1.0f, active);
        MulAddDst(alpha, hx2, hy1, active);
        Mul(beta, hx2, hy0, active);
        Muls(beta, beta, -1.0f, active);
        MulAddDst(beta, hx0, hy2, active);
        Mul(gamma, hx0, hy1, active);
        Muls(gamma, gamma, -1.0f, active);
        MulAddDst(gamma, hx1, hy0, active);

        // The projected edge signs are c0*w1*w2, c1*w0*w2 and c2*w0*w1.
        Mul(q, alpha, w1, active);
        Mul(q, q, w2, active);
        Mul(t0, beta, w0, active);
        Mul(t0, t0, w2, active);
        Mul(t1, gamma, w0, active);
        Mul(t1, t1, w1, active);
        Compares<float, AscendC::CMPMODE::GE>(covered, q, 0.0f, active);
        Compares<float, AscendC::CMPMODE::GE>(m0, t0, 0.0f, active);
        And(covered, covered, m0, active);
        Compares<float, AscendC::CMPMODE::GE>(m0, t1, 0.0f, active);
        And(covered, covered, m0, active);
        Compares<float, AscendC::CMPMODE::LE>(m1, q, 0.0f, active);
        Compares<float, AscendC::CMPMODE::LE>(m0, t0, 0.0f, active);
        And(m1, m1, m0, active);
        Compares<float, AscendC::CMPMODE::LE>(m0, t1, 0.0f, active);
        And(m1, m1, m0, active);
        Or(covered, covered, m1, active);

        // D is the non-zero projected triangle denominator; Z/D is NDC depth.
        Mul(area, alpha, w0, active);
        MulAddDst(area, beta, w1, active);
        MulAddDst(area, gamma, w2, active);
        Compares<float, AscendC::CMPMODE::NE>(nonDeg, area, 0.0f, active);
        And(covered, covered, nonDeg, active);
        Select<float>(tmp, area, one, nonDeg);

        Mul(q, alpha, z0, active);
        MulAddDst(q, beta, z1, active);
        MulAddDst(q, gamma, z2, active);
        Div<float, &kHighPrecisionDiv>(q, q, tmp, active);
        Muls(q, q, 0.49999f, active);
        Adds(q, q, 0.5f, active);
        Muls(q, q, 262144.0f, active);
        Abs(tmp, q, active);
        Compares<float, AscendC::CMPMODE::LT>(m0, tmp, 16777216.0f, active);
        Select<float>(tmp, q, zero, m0);
        Cast<int32_t, float, kF32ToI32Trunc>(intTmp, tmp, active);
        Cast<float, int32_t, kI32ToF32>(tmp, intTmp, active);
        Select<float>(q, tmp, q, m0);

        LoadAlign(bestQReg, bestQ + off);
        LoadAlign(bestFaceReg, bestFace + off);
        Compare<float, AscendC::CMPMODE::LT>(m0, q, bestQReg, active);
        Compare<float, AscendC::CMPMODE::EQ>(m1, q, bestQReg, active);
        Duplicate(faceReg, faceIndex);
        Compare<int32_t, AscendC::CMPMODE::LT>(m2, faceReg, bestFaceReg, active);
        And(tie, m1, m2, active);
        Or(win, m0, tie, active);
        And(win, win, covered, active);

        Add(baryDenom, alpha, beta, active);
        Add(baryDenom, baryDenom, gamma, active);
        Select<float>(tmp, baryDenom, one, win);
        Div<float, &kHighPrecisionDiv>(b0, alpha, tmp, active);
        Div<float, &kHighPrecisionDiv>(b1, beta, tmp, active);
        Div<float, &kHighPrecisionDiv>(b2, gamma, tmp, active);

        Select<float>(bestQReg, q, bestQReg, win);
        Select<int32_t>(bestFaceReg, faceReg, bestFaceReg, win);
        StoreAlign(bestQ + off, bestQReg, active);
        StoreAlign(bestFace + off, bestFaceReg, active);
        LoadAlign(bestBReg, bestB0 + off);
        Select<float>(bestBReg, b0, bestBReg, win);
        StoreAlign(bestB0 + off, bestBReg, active);
        LoadAlign(bestBReg, bestB1 + off);
        Select<float>(bestBReg, b1, bestBReg, win);
        StoreAlign(bestB1 + off, bestBReg, active);
        LoadAlign(bestBReg, bestB2 + off);
        Select<float>(bestBReg, b2, bestBReg, win);
        StoreAlign(bestB2 + off, bestBReg, active);
    }
}

__simd_vf__ inline void InitBaryLoVF(__ubuf__ float* bestB0Lo, __ubuf__ float* bestB1Lo, __ubuf__ float* bestB2Lo,
                                     uint32_t valid)
{
    using namespace AscendC::Reg;
    RegTensor<float> zero;
    MaskReg active;
    Duplicate(zero, 0.0f);
    active = UpdateMask<float>(valid);
    StoreAlign(bestB0Lo, zero, active);
    StoreAlign(bestB1Lo, zero, active);
    StoreAlign(bestB2Lo, zero, active);
}

// Project a vertex with Golden's explicit float32 operation order. The first-pass
// formula mirrors _screen_transform; point 3 denotes the exact pixel sample.
__simd_callee__ inline void ProjectScreenCoordF32(AscendC::Reg::RegTensor<float>& out, __ubuf__ float* packet,
                                                  uint32_t point, uint32_t axis, AscendC::Reg::RegTensor<float>& pixelX,
                                                  AscendC::Reg::RegTensor<float>& pixelY, float widthMinusOneF,
                                                  float heightMinusOneF, AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    if (point == 3U) {
        if (axis == 0U) {
            Adds(out, pixelX, 0.0f, mask);
        } else {
            Adds(out, pixelY, 0.0f, mask);
        }
        return;
    }
    RegTensor<float> coord, clipW;
    const uint32_t row = kFaceCoordBase + point * kFaceVertexStride;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coord, packet + row + axis * kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(clipW, packet + row + 3U * kFaceCoordSlotFloats);
    Div<float, &kHighPrecisionDiv>(out, coord, clipW, mask);
    Muls(out, out, 0.5f, mask);
    Adds(out, out, 0.5f, mask);
    if (axis == 0U) {
        Muls(out, out, widthMinusOneF, mask);
    } else {
        Muls(out, out, heightMinusOneF, mask);
    }
    Adds(out, out, 0.5f, mask);
}

// Winner recomputation uses Golden's _persp_x/_persp_y expression order rather
// than an algebraically equivalent homogeneous/double-single determinant.
__simd_callee__ inline void ProjectWinnerCoordF32(AscendC::Reg::RegTensor<float>& out, __ubuf__ float* packet,
                                                  uint32_t point, uint32_t axis, AscendC::Reg::RegTensor<float>& pixelX,
                                                  AscendC::Reg::RegTensor<float>& pixelY, float widthF,
                                                  float widthMinusOneF, float heightMinusOneF,
                                                  AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    if (point == 3U) {
        if (axis == 0U) {
            Adds(out, pixelX, 0.0f, mask);
        } else {
            Adds(out, pixelY, 0.0f, mask);
        }
        return;
    }
    RegTensor<float> coord, clipW;
    const uint32_t row = kFaceCoordBase + point * kFaceVertexStride;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coord, packet + row + axis * kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(clipW, packet + row + 3U * kFaceCoordSlotFloats);
    Div<float, &kHighPrecisionDiv>(out, coord, clipW, mask);
    if (axis == 0U) {
        Muls(out, out, 0.5f * widthMinusOneF, mask);
        Adds(out, out, 0.5f * widthF, mask);
    } else {
        Muls(out, out, 0.5f * heightMinusOneF, mask);
        Adds(out, out, 0.5f * (heightMinusOneF + 1.0f), mask);
    }
}

__simd_callee__ inline void TriAreaF32(AscendC::Reg::RegTensor<float>& out, AscendC::Reg::RegTensor<float>& ax,
                                       AscendC::Reg::RegTensor<float>& ay, AscendC::Reg::RegTensor<float>& bx,
                                       AscendC::Reg::RegTensor<float>& by, AscendC::Reg::RegTensor<float>& cx,
                                       AscendC::Reg::RegTensor<float>& cy, AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    RegTensor<float> dx0, dy0, dx1, dy1, product0, product1;
    Sub(dx0, bx, ax, mask);
    Sub(dy0, cy, ay, mask);
    Mul(product0, dx0, dy0, mask);
    Sub(dx1, by, ay, mask);
    Sub(dy1, cx, ax, mask);
    Mul(product1, dx1, dy1, mask);
    Sub(out, product0, product1, mask);
}

// Recompute one perspective-correct winner component in Golden FP32 order.
// Component 0/1/2 returns corrected alpha/beta/gamma respectively. outLo for
// component 0 carries the direct projected area for Golden's final area check.
__simd_vf__ inline void RecomputeBaryNumeratorVF(__ubuf__ float* packet, __ubuf__ int32_t* bestFace,
                                                 __ubuf__ float* outHi, __ubuf__ float* outLo, uint32_t valid,
                                                 float tileStartF, float widthF, float widthMinusOneF,
                                                 float heightMinusOneF, int32_t faceIndex, uint32_t component,
                                                 uint32_t selectedOnly)
{
    using namespace AscendC::Reg;
    RegTensor<float> n, pixelX, pixelY, widthReg;
    RegTensor<float> p0x, p0y, p1x, p1y, p2x, p2y;
    RegTensor<float> area, safeArea, numerator, value, other, one, zero, clipW;
    RegTensor<float> absArea;
    RegTensor<int32_t> intTmp, winnerFace, faceReg;
    MaskReg active, selected, nonDeg;

    active = UpdateMask<float>(valid);
    selected = active;
    if (selectedOnly != 0U) {
        LoadAlign(winnerFace, bestFace);
        Duplicate(faceReg, faceIndex);
        Compare<int32_t, AscendC::CMPMODE::EQ>(selected, winnerFace, faceReg, active);
    }

    Duplicate(widthReg, widthF);
    Duplicate(one, 1.0f);
    Duplicate(zero, 0.0f);
    Arange(n, tileStartF);
    Div<float, &kHighPrecisionDiv>(pixelY, n, widthReg, active);
    Cast<int32_t, float, kF32ToI32Trunc>(intTmp, pixelY, active);
    Cast<float, int32_t, kI32ToF32>(pixelY, intTmp, active);
    Muls(pixelX, pixelY, widthF, active);
    Sub(pixelX, n, pixelX, active);
    Adds(pixelX, pixelX, 0.5f, active);
    Adds(pixelY, pixelY, 0.5f, active);

    ProjectWinnerCoordF32(p0x, packet, 0U, 0U, pixelX, pixelY, widthF, widthMinusOneF, heightMinusOneF, selected);
    ProjectWinnerCoordF32(p0y, packet, 0U, 1U, pixelX, pixelY, widthF, widthMinusOneF, heightMinusOneF, selected);
    ProjectWinnerCoordF32(p1x, packet, 1U, 0U, pixelX, pixelY, widthF, widthMinusOneF, heightMinusOneF, selected);
    ProjectWinnerCoordF32(p1y, packet, 1U, 1U, pixelX, pixelY, widthF, widthMinusOneF, heightMinusOneF, selected);
    ProjectWinnerCoordF32(p2x, packet, 2U, 0U, pixelX, pixelY, widthF, widthMinusOneF, heightMinusOneF, selected);
    ProjectWinnerCoordF32(p2y, packet, 2U, 1U, pixelX, pixelY, widthF, widthMinusOneF, heightMinusOneF, selected);
    TriAreaF32(area, p0x, p0y, p1x, p1y, p2x, p2y, selected);
    Abs(absArea, area, selected);
    Compares<float, AscendC::CMPMODE::GT>(nonDeg, absArea, 0.0f, selected);
    Select<float>(safeArea, area, one, nonDeg);

    if (component == 1U) {
        TriAreaF32(numerator, p0x, p0y, pixelX, pixelY, p2x, p2y, selected);
        Div<float, &kHighPrecisionDiv>(value, numerator, safeArea, selected);
    } else if (component == 2U) {
        TriAreaF32(numerator, p0x, p0y, p1x, p1y, pixelX, pixelY, selected);
        Div<float, &kHighPrecisionDiv>(value, numerator, safeArea, selected);
    } else {
        TriAreaF32(numerator, p0x, p0y, pixelX, pixelY, p2x, p2y, selected);
        Div<float, &kHighPrecisionDiv>(value, numerator, safeArea, selected);
        TriAreaF32(numerator, p0x, p0y, p1x, p1y, pixelX, pixelY, selected);
        Div<float, &kHighPrecisionDiv>(other, numerator, safeArea, selected);
        Sub(value, one, value, selected);
        Sub(value, value, other, selected);
    }

    const uint32_t row = kFaceCoordBase + component * kFaceVertexStride;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(clipW, packet + row + 3U * kFaceCoordSlotFloats);
    Div<float, &kHighPrecisionDiv>(value, value, clipW, selected);
    StoreAlign(outHi, value, selected);
    if (component == 0U) {
        StoreAlign(outLo, area, selected);
    } else {
        StoreAlign(outLo, zero, selected);
    }
}

// Compute one projected determinant edge and spill its double-single pair to UB.
// edgeKind 0 computes the B-A edge on the phase axis; edgeKind 1 computes
// the C-A edge on the opposite axis. Keeping both projections in one VF avoids
// duplicating the high-pressure projection register frame on Ascend 950.
__simd_vf__ inline void PrepareProjectedBaryEdgeVF(__ubuf__ float* packet, __ubuf__ float* scratchHi,
                                                   __ubuf__ float* scratchLo, uint32_t valid, float tileStartF,
                                                   float widthF, float widthMinusOneF, float heightMinusOneF,
                                                   uint32_t component, uint32_t phase, uint32_t edgeKind)
{
    using namespace AscendC::Reg;
    RegTensor<float> n, pixelCoord, widthReg, zero, half, extent;
    RegTensor<float> edgeHi, edgeLo;
    RegTensor<int32_t> intTmp;
    MaskReg active;

    uint32_t point = 1U;
    uint32_t axis = (phase == 0U) ? 0U : 1U;
    if (edgeKind != 0U) {
        point = 2U;
        axis = (phase == 0U) ? 1U : 0U;
    }
    if ((edgeKind == 0U && component == 1U) || (edgeKind != 0U && component == 2U)) {
        point = 3U;
    }

    active = UpdateMask<float>(valid);
    Duplicate(zero, 0.0f);
    Duplicate(half, 0.5f);
    Duplicate(widthReg, widthF);
    Arange(n, tileStartF);
    Div<float, &kHighPrecisionDiv>(pixelCoord, n, widthReg, active);
    Cast<int32_t, float, kF32ToI32Trunc>(intTmp, pixelCoord, active);
    Cast<float, int32_t, kI32ToF32>(pixelCoord, intTmp, active);
    if (axis == 0U) {
        Muls(pixelCoord, pixelCoord, widthF, active);
        Sub(pixelCoord, n, pixelCoord, active);
        Adds(pixelCoord, pixelCoord, 0.5f, active);
        Duplicate(extent, widthMinusOneF);
    } else {
        Adds(pixelCoord, pixelCoord, 0.5f, active);
        Duplicate(extent, heightMinusOneF);
    }

    ProjectPointCoordDs(n, widthReg, packet, 0U, axis, pixelCoord, pixelCoord, extent, extent, zero, half, active);
    ProjectPointCoordDs(edgeHi, edgeLo, packet, point, axis, pixelCoord, pixelCoord, extent, extent, zero, half,
                        active);
    DsSub(edgeHi, edgeLo, edgeHi, edgeLo, n, widthReg, active);
    StoreAlign(scratchHi, edgeHi, active);
    StoreAlign(scratchLo, edgeLo, active);
}

// Multiply the two spilled edges and preserve Golden's determinant order:
// phase 0 stores the first product; phase 1 subtracts the second product.
__simd_vf__ inline void CombineProjectedBaryProductVF(__ubuf__ float* edgeAHiBuf, __ubuf__ float* edgeALoBuf,
                                                      __ubuf__ float* edgeBHiBuf, __ubuf__ float* edgeBLoBuf,
                                                      __ubuf__ float* outHi, __ubuf__ float* outLo, uint32_t valid,
                                                      uint32_t phase)
{
    using namespace AscendC::Reg;
    RegTensor<float> edgeAHi, edgeALo, edgeBHi, edgeBLo;
    MaskReg active;

    active = UpdateMask<float>(valid);
    LoadAlign(edgeAHi, edgeAHiBuf);
    LoadAlign(edgeALo, edgeALoBuf);
    LoadAlign(edgeBHi, edgeBHiBuf);
    LoadAlign(edgeBLo, edgeBLoBuf);
    DsMul(edgeAHi, edgeALo, edgeAHi, edgeALo, edgeBHi, edgeBLo, active);
    if (phase == 0U) {
        StoreAlign(outHi, edgeAHi, active);
        StoreAlign(outLo, edgeALo, active);
    } else {
        LoadAlign(edgeBHi, outHi);
        LoadAlign(edgeBLo, outLo);
        DsSub(edgeBHi, edgeBLo, edgeBHi, edgeBLo, edgeAHi, edgeALo, active);
        StoreAlign(outHi, edgeBHi, active);
        StoreAlign(outLo, edgeBLo, active);
    }
}

// Convert the direct area and beta/gamma numerators to Golden alpha/beta/gamma,
// then evaluate the explicit [0, 1] coverage predicate with both double-single parts.
__simd_vf__ inline void ComputeProjectedCoverageVF(__ubuf__ int32_t* candidateValid, __ubuf__ float* areaBuf,
                                                   __ubuf__ float* betaBuf, __ubuf__ float* gammaBuf,
                                                   __ubuf__ float* areaLoBuf, __ubuf__ float* betaLoBuf,
                                                   __ubuf__ float* gammaLoBuf, uint32_t valid)
{
    using namespace AscendC::Reg;
    RegTensor<float> areaHi, areaLo, betaHi, betaLo, gammaHi, gammaLo;
    RegTensor<float> safeAreaHi, safeAreaLo, alphaHi, alphaLo, one, zero;
    RegTensor<int32_t> validInt, oneI;
    MaskReg active, highNonZero, lowNonZero, nonDeg, covered, cmp;

    active = UpdateMask<float>(valid);
    Duplicate(one, 1.0f);
    Duplicate(zero, 0.0f);
    LoadAlign(areaHi, areaBuf);
    LoadAlign(areaLo, areaLoBuf);
    LoadAlign(betaHi, betaBuf);
    LoadAlign(betaLo, betaLoBuf);
    LoadAlign(gammaHi, gammaBuf);
    LoadAlign(gammaLo, gammaLoBuf);

    Compares<float, AscendC::CMPMODE::NE>(highNonZero, areaHi, 0.0f, active);
    Compares<float, AscendC::CMPMODE::NE>(lowNonZero, areaLo, 0.0f, active);
    Or(nonDeg, highNonZero, lowNonZero, active);
    Select<float>(safeAreaHi, areaHi, one, nonDeg);
    Select<float>(safeAreaLo, areaLo, zero, nonDeg);

    DsDiv(betaHi, betaLo, betaHi, betaLo, safeAreaHi, safeAreaLo, active);
    DsDiv(gammaHi, gammaLo, gammaHi, gammaLo, safeAreaHi, safeAreaLo, active);
    DsSub(alphaHi, alphaLo, one, zero, betaHi, betaLo, active);
    DsSub(alphaHi, alphaLo, alphaHi, alphaLo, gammaHi, gammaLo, active);

    DsInUnitInterval(covered, alphaHi, alphaLo, active);
    DsInUnitInterval(cmp, betaHi, betaLo, active);
    And(covered, covered, cmp, active);
    DsInUnitInterval(cmp, gammaHi, gammaLo, active);
    And(covered, covered, cmp, active);
    And(covered, covered, nonDeg, active);

    StoreAlign(areaBuf, alphaHi, active);
    StoreAlign(areaLoBuf, alphaLo, active);
    StoreAlign(betaBuf, betaHi, active);
    StoreAlign(betaLoBuf, betaLo, active);
    StoreAlign(gammaBuf, gammaHi, active);
    StoreAlign(gammaLoBuf, gammaLo, active);
    Duplicate(validInt, 0);
    Duplicate(oneI, 1);
    Select<int32_t>(validInt, oneI, validInt, covered);
    StoreAlign(candidateValid, validInt, active);
}

// Convert a packet vertex to the exact binary64 screen coordinate used by the
// Promote Golden. The x and y expressions intentionally keep their distinct
// evaluation order from rasterizer_golden.py.
__simt_callee__ __attribute__((noinline)) inline uint64_t ProjectScreenXF64(float coordinate, float clipW,
                                                                            uint32_t extent)
{
    const uint64_t coordinateBits = SoftFloat64::Float32ToDoubleBits(__float_as_uint(coordinate));
    const uint64_t clipWBits = SoftFloat64::Float32ToDoubleBits(__float_as_uint(clipW));
    uint64_t projected = SoftFloat64::Div(coordinateBits, clipWBits);
    projected = SoftFloat64::Mul(projected, SoftFloat64::kHalfBits);
    projected = SoftFloat64::Add(projected, SoftFloat64::kHalfBits);
    projected = SoftFloat64::Mul(projected, SoftFloat64::UInt32ToDoubleBits(extent));
    return SoftFloat64::Add(projected, SoftFloat64::kHalfBits);
}

__simt_callee__ __attribute__((noinline)) inline uint64_t ProjectScreenYF64(float coordinate, float clipW,
                                                                            uint32_t extent)
{
    const uint64_t coordinateBits = SoftFloat64::Float32ToDoubleBits(__float_as_uint(coordinate));
    const uint64_t clipWBits = SoftFloat64::Float32ToDoubleBits(__float_as_uint(clipW));
    uint64_t projected = SoftFloat64::Mul(SoftFloat64::kHalfBits, coordinateBits);
    projected = SoftFloat64::Div(projected, clipWBits);
    projected = SoftFloat64::Add(SoftFloat64::kHalfBits, projected);
    projected = SoftFloat64::Mul(projected, SoftFloat64::UInt32ToDoubleBits(extent));
    return SoftFloat64::Add(projected, SoftFloat64::kHalfBits);
}

__simt_callee__ inline bool IsCoverageBoundaryUncertain(float highPart)
{
    // This threshold only selects the exact slow path. It never changes the
    // inside/outside result directly. Double-single has about 48 significant
    // bits, so 1e-5 leaves ample margin for cancellation-amplified edge cases.
    constexpr float kBoundaryThreshold = 1.0e-5f;
    return (highPart >= -kBoundaryThreshold && highPart <= kBoundaryThreshold) ||
           (highPart >= 1.0f - kBoundaryThreshold && highPart <= 1.0f + kBoundaryThreshold);
}

// Refine only numerically ambiguous edge pixels with software IEEE-754
// binary64. Each SIMT VF deliberately contains only one small part of the
// calculation. This keeps the Bisheng frontend and VF stack away from the
// very large call graph produced when projection, determinants and division
// are compiled together. Four existing scratch planes are reused; no UB is
// added and the normal double-single path remains unchanged.
constexpr uint32_t kProjectedX0Word = 0U;
constexpr uint32_t kProjectedY0Word = 2U;
constexpr uint32_t kProjectedX1Word = 4U;
constexpr uint32_t kProjectedY1Word = 6U;
constexpr uint32_t kProjectedX2Word = 8U;
constexpr uint32_t kProjectedY2Word = 10U;
constexpr uint32_t kProjectedAreaWord = 12U;
constexpr uint32_t kProjectedRefineFlagWord = 14U;

__simt_callee__ inline uint64_t LoadF64Words(__ubuf__ uint32_t* words, uint32_t wordOffset)
{
    return static_cast<uint64_t>(words[wordOffset]) | (static_cast<uint64_t>(words[wordOffset + 1U]) << 32U);
}

__simt_callee__ inline void StoreF64Words(__ubuf__ uint32_t* words, uint32_t wordOffset, uint64_t value)
{
    words[wordOffset] = static_cast<uint32_t>(value);
    words[wordOffset + 1U] = static_cast<uint32_t>(value >> 32U);
}

__simt_vf__ __aicore__ __launch_bounds__(64) inline void DetectProjectedCoverageRefineF64Simt(
    __ubuf__ float* alphaHi, __ubuf__ float* betaHi, __ubuf__ float* gammaHi, __ubuf__ uint32_t* commonWords,
    uint32_t valid)
{
    if (threadIdx.x != 0U) {
        return;
    }
    bool needsRefine = false;
    for (uint32_t lane = 0U; lane < valid; ++lane) {
        if (IsCoverageBoundaryUncertain(alphaHi[lane]) || IsCoverageBoundaryUncertain(betaHi[lane]) ||
            IsCoverageBoundaryUncertain(gammaHi[lane])) {
            needsRefine = true;
            break;
        }
    }
    commonWords[kProjectedRefineFlagWord] = needsRefine ? 1U : 0U;
    commonWords[kProjectedAreaWord] = 0U;
    commonWords[kProjectedAreaWord + 1U] = 0U;
}

__simt_vf__ __aicore__ __launch_bounds__(64) inline void PrepareProjectedVertexXF64Simt(
    __ubuf__ float* packet, __ubuf__ uint32_t* commonWords, uint32_t vertex, uint32_t extent, uint32_t outputWord)
{
    if (threadIdx.x == 0U && commonWords[kProjectedRefineFlagWord] != 0U) {
        const uint32_t row = kFaceCoordBase + vertex * kFaceVertexStride;
        const uint32_t wOffset = 3U * kFaceCoordSlotFloats;
        const uint64_t projected = ProjectScreenXF64(packet[row], packet[row + wOffset], extent);
        StoreF64Words(commonWords, outputWord, projected);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(64) inline void PrepareProjectedVertexYF64Simt(
    __ubuf__ float* packet, __ubuf__ uint32_t* commonWords, uint32_t vertex, uint32_t extent, uint32_t outputWord)
{
    if (threadIdx.x == 0U && commonWords[kProjectedRefineFlagWord] != 0U) {
        const uint32_t row = kFaceCoordBase + vertex * kFaceVertexStride;
        const uint32_t wOffset = 3U * kFaceCoordSlotFloats;
        const uint64_t projected = ProjectScreenYF64(packet[row + kFaceCoordSlotFloats], packet[row + wOffset], extent);
        StoreF64Words(commonWords, outputWord, projected);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(64) inline void PrepareProjectedAreaF64Simt(__ubuf__ uint32_t* commonWords)
{
    if (threadIdx.x == 0U && commonWords[kProjectedRefineFlagWord] != 0U) {
        const uint64_t area = SoftFloat64::TriangleArea(
            LoadF64Words(commonWords, kProjectedX0Word), LoadF64Words(commonWords, kProjectedY0Word),
            LoadF64Words(commonWords, kProjectedX1Word), LoadF64Words(commonWords, kProjectedY1Word),
            LoadF64Words(commonWords, kProjectedX2Word), LoadF64Words(commonWords, kProjectedY2Word));
        StoreF64Words(commonWords, kProjectedAreaWord, area);
    }
}

// candidateValid is scratch only for lanes selected for exact refinement. The
// untouched lanes retain their double-single coverage decision.
__simt_vf__ __aicore__ __launch_bounds__(64) inline void PrepareProjectedBetaNumeratorF64Simt(
    __ubuf__ float* alphaHi, __ubuf__ float* betaHi, __ubuf__ float* gammaHi, __ubuf__ uint32_t* commonWords,
    __ubuf__ uint32_t* numeratorLowBits, __ubuf__ uint32_t* numeratorHighBits, uint32_t valid, uint64_t tileStart,
    uint32_t width)
{
    for (uint32_t lane = threadIdx.x; lane < valid; lane += blockDim.x) {
        if (!IsCoverageBoundaryUncertain(alphaHi[lane]) && !IsCoverageBoundaryUncertain(betaHi[lane]) &&
            !IsCoverageBoundaryUncertain(gammaHi[lane])) {
            continue;
        }
        uint64_t numerator = 0U;
        if (!SoftFloat64::IsZero(LoadF64Words(commonWords, kProjectedAreaWord))) {
            const uint64_t pixelIndex = tileStart + lane;
            const uint32_t pixelYInt = static_cast<uint32_t>(pixelIndex / width);
            const uint32_t pixelXInt = static_cast<uint32_t>(pixelIndex - static_cast<uint64_t>(pixelYInt) * width);
            const uint64_t pixelX = SoftFloat64::Add(SoftFloat64::UInt32ToDoubleBits(pixelXInt),
                                                     SoftFloat64::kHalfBits);
            const uint64_t pixelY = SoftFloat64::Add(SoftFloat64::UInt32ToDoubleBits(pixelYInt),
                                                     SoftFloat64::kHalfBits);
            numerator = SoftFloat64::TriangleArea(
                LoadF64Words(commonWords, kProjectedX0Word), LoadF64Words(commonWords, kProjectedY0Word), pixelX,
                pixelY, LoadF64Words(commonWords, kProjectedX2Word), LoadF64Words(commonWords, kProjectedY2Word));
        }
        numeratorLowBits[lane] = static_cast<uint32_t>(numerator);
        numeratorHighBits[lane] = static_cast<uint32_t>(numerator >> 32U);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(64) inline void FinishProjectedBetaF64Simt(
    __ubuf__ float* alphaHi, __ubuf__ float* betaHi, __ubuf__ float* gammaHi, __ubuf__ uint32_t* commonWords,
    __ubuf__ uint32_t* numeratorLowBits, __ubuf__ uint32_t* numeratorHighBits, __ubuf__ uint32_t* betaLowBits,
    __ubuf__ uint32_t* betaHighBits, uint32_t valid)
{
    for (uint32_t lane = threadIdx.x; lane < valid; lane += blockDim.x) {
        if (!IsCoverageBoundaryUncertain(alphaHi[lane]) && !IsCoverageBoundaryUncertain(betaHi[lane]) &&
            !IsCoverageBoundaryUncertain(gammaHi[lane])) {
            continue;
        }
        const uint64_t area = LoadF64Words(commonWords, kProjectedAreaWord);
        uint64_t beta = 0U;
        if (!SoftFloat64::IsZero(area)) {
            const uint64_t numerator = static_cast<uint64_t>(numeratorLowBits[lane]) |
                                       (static_cast<uint64_t>(numeratorHighBits[lane]) << 32U);
            beta = SoftFloat64::Div(numerator, area);
        }
        betaLowBits[lane] = static_cast<uint32_t>(beta);
        betaHighBits[lane] = static_cast<uint32_t>(beta >> 32U);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(64) inline void PrepareProjectedGammaNumeratorF64Simt(
    __ubuf__ float* alphaHi, __ubuf__ float* betaHi, __ubuf__ float* gammaHi, __ubuf__ uint32_t* commonWords,
    __ubuf__ uint32_t* numeratorLowBits, __ubuf__ uint32_t* numeratorHighBits, uint32_t valid, uint64_t tileStart,
    uint32_t width)
{
    for (uint32_t lane = threadIdx.x; lane < valid; lane += blockDim.x) {
        if (!IsCoverageBoundaryUncertain(alphaHi[lane]) && !IsCoverageBoundaryUncertain(betaHi[lane]) &&
            !IsCoverageBoundaryUncertain(gammaHi[lane])) {
            continue;
        }
        uint64_t numerator = 0U;
        if (!SoftFloat64::IsZero(LoadF64Words(commonWords, kProjectedAreaWord))) {
            const uint64_t pixelIndex = tileStart + lane;
            const uint32_t pixelYInt = static_cast<uint32_t>(pixelIndex / width);
            const uint32_t pixelXInt = static_cast<uint32_t>(pixelIndex - static_cast<uint64_t>(pixelYInt) * width);
            const uint64_t pixelX = SoftFloat64::Add(SoftFloat64::UInt32ToDoubleBits(pixelXInt),
                                                     SoftFloat64::kHalfBits);
            const uint64_t pixelY = SoftFloat64::Add(SoftFloat64::UInt32ToDoubleBits(pixelYInt),
                                                     SoftFloat64::kHalfBits);
            numerator = SoftFloat64::TriangleArea(LoadF64Words(commonWords, kProjectedX0Word),
                                                  LoadF64Words(commonWords, kProjectedY0Word),
                                                  LoadF64Words(commonWords, kProjectedX1Word),
                                                  LoadF64Words(commonWords, kProjectedY1Word), pixelX, pixelY);
        }
        numeratorLowBits[lane] = static_cast<uint32_t>(numerator);
        numeratorHighBits[lane] = static_cast<uint32_t>(numerator >> 32U);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(64) inline void FinishProjectedCoverageF64Simt(
    __ubuf__ uint32_t* candidateBits, __ubuf__ float* alphaHi, __ubuf__ float* betaHi, __ubuf__ float* gammaHi,
    __ubuf__ uint32_t* commonWords, __ubuf__ uint32_t* numeratorHighBits, __ubuf__ uint32_t* betaLowBits,
    __ubuf__ uint32_t* betaHighBits, uint32_t valid)
{
    for (uint32_t lane = threadIdx.x; lane < valid; lane += blockDim.x) {
        if (!IsCoverageBoundaryUncertain(alphaHi[lane]) && !IsCoverageBoundaryUncertain(betaHi[lane]) &&
            !IsCoverageBoundaryUncertain(gammaHi[lane])) {
            continue;
        }
        const uint64_t area = LoadF64Words(commonWords, kProjectedAreaWord);
        if (SoftFloat64::IsZero(area)) {
            candidateBits[lane] = 0U;
            continue;
        }
        const uint64_t beta = static_cast<uint64_t>(betaLowBits[lane]) |
                              (static_cast<uint64_t>(betaHighBits[lane]) << 32U);
        const uint64_t gammaNumerator = static_cast<uint64_t>(candidateBits[lane]) |
                                        (static_cast<uint64_t>(numeratorHighBits[lane]) << 32U);
        const uint64_t gamma = SoftFloat64::Div(gammaNumerator, area);
        uint64_t alpha = SoftFloat64::Sub(SoftFloat64::kOneBits, beta);
        alpha = SoftFloat64::Sub(alpha, gamma);
        const bool inside = SoftFloat64::InUnitInterval(alpha) && SoftFloat64::InUnitInterval(beta) &&
                            SoftFloat64::InUnitInterval(gamma);
        candidateBits[lane] = inside ? 1U : 0U;
    }
}

__simd_callee__ inline void ProjectDepthDs(AscendC::Reg::RegTensor<float>& outHi, AscendC::Reg::RegTensor<float>& outLo,
                                           __ubuf__ float* packet, uint32_t vertex,
                                           AscendC::Reg::RegTensor<float>& zero, AscendC::Reg::RegTensor<float>& scale,
                                           AscendC::Reg::RegTensor<float>& half, AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    RegTensor<float> coord, clipW;
    const uint32_t row = kFaceCoordBase + vertex * kFaceVertexStride;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coord, packet + row + 2U * kFaceCoordSlotFloats);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(clipW, packet + row + 3U * kFaceCoordSlotFloats);
    DsDiv(outHi, outLo, coord, zero, clipW, zero, mask);
    DsMul(outHi, outLo, outHi, outLo, scale, zero, mask);
    DsAdd(outHi, outLo, outHi, outLo, half, zero, mask);
}

// Compute depth in Golden order: ((alpha*z0) + (beta*z1)) + (gamma*z2),
// preserving both double-single parts throughout the weighted sum.
__simd_vf__ inline void ComputeProjectedDepthFromCoverageVF(__ubuf__ float* packet, __ubuf__ float* candidateDepthHi,
                                                            __ubuf__ float* candidateDepthLo,
                                                            __ubuf__ int32_t* candidateValid, __ubuf__ float* alphaBuf,
                                                            __ubuf__ float* betaBuf, __ubuf__ float* gammaBuf,
                                                            __ubuf__ float* alphaLoBuf, __ubuf__ float* betaLoBuf,
                                                            __ubuf__ float* gammaLoBuf, uint32_t valid)
{
    using namespace AscendC::Reg;
    RegTensor<float> weightHi, weightLo, valueHi, valueLo, depthHi, depthLo;
    RegTensor<float> zero, scale, half;
    RegTensor<int32_t> validInt;
    MaskReg active, covered;

    active = UpdateMask<float>(valid);
    LoadAlign(validInt, candidateValid);
    Compares<int32_t, AscendC::CMPMODE::NE>(covered, validInt, 0, active);
    Duplicate(zero, 0.0f);
    Duplicate(scale, 0.49999f);
    Duplicate(half, 0.5f);
    Duplicate(depthHi, 0.0f);
    Duplicate(depthLo, 0.0f);

    LoadAlign(weightHi, alphaBuf);
    LoadAlign(weightLo, alphaLoBuf);
    ProjectDepthDs(valueHi, valueLo, packet, 0U, zero, scale, half, covered);
    DsMul(depthHi, depthLo, weightHi, weightLo, valueHi, valueLo, covered);

    LoadAlign(weightHi, betaBuf);
    LoadAlign(weightLo, betaLoBuf);
    ProjectDepthDs(valueHi, valueLo, packet, 1U, zero, scale, half, covered);
    DsMul(valueHi, valueLo, weightHi, weightLo, valueHi, valueLo, covered);
    DsAdd(depthHi, depthLo, depthHi, depthLo, valueHi, valueLo, covered);

    LoadAlign(weightHi, gammaBuf);
    LoadAlign(weightLo, gammaLoBuf);
    ProjectDepthDs(valueHi, valueLo, packet, 2U, zero, scale, half, covered);
    DsMul(valueHi, valueLo, weightHi, weightLo, valueHi, valueLo, covered);
    DsAdd(depthHi, depthLo, depthHi, depthLo, valueHi, valueLo, covered);

    StoreAlign(candidateDepthHi, depthHi, active);
    StoreAlign(candidateDepthLo, depthLo, active);
}

// Split projected-depth quantization into two VF calls so the exact double-single
// correction stays below Ascend 950's 6144-byte VF stack limit. The candidate
// barycentric planes are temporary until the winning face is recomputed, so they
// safely carry the double-single remainder between the two calls.
__simd_vf__ inline void PrepareProjectedDepthQuantizeVF(__ubuf__ int32_t* candidateQ, __ubuf__ int32_t* candidateValid,
                                                        __ubuf__ float* candidateDepthHi,
                                                        __ubuf__ float* candidateDepthLo,
                                                        __ubuf__ float* remainderHiBuf, __ubuf__ float* remainderLoBuf,
                                                        uint32_t valid)
{
    using namespace AscendC::Reg;
    RegTensor<float> scaledHi, scaledLo, baseF, remHi, remLo, zero;
    RegTensor<int32_t> validInt, qBase;
    MaskReg active, covered;

    active = UpdateMask<float>(valid);
    LoadAlign(validInt, candidateValid);
    Compares<int32_t, AscendC::CMPMODE::NE>(covered, validInt, 0, active);
    LoadAlign(scaledHi, candidateDepthHi);
    LoadAlign(scaledLo, candidateDepthLo);
    Muls(scaledHi, scaledHi, 262144.0f, covered);
    Muls(scaledLo, scaledLo, 262144.0f, covered);

    Duplicate(qBase, 0);
    Cast<int32_t, float, kF32ToI32Trunc>(qBase, scaledHi, covered);
    Cast<float, int32_t, kI32ToF32>(baseF, qBase, covered);
    Duplicate(zero, 0.0f);
    DsSub(remHi, remLo, scaledHi, scaledLo, baseF, zero, covered);

    StoreAlign(candidateQ, qBase, active);
    StoreAlign(remainderHiBuf, remHi, active);
    StoreAlign(remainderLoBuf, remLo, active);
}

__simd_vf__ inline void FinishProjectedDepthQuantizeVF(__ubuf__ int32_t* candidateQ, __ubuf__ int32_t* candidateValid,
                                                       __ubuf__ float* remainderHiBuf, __ubuf__ float* remainderLoBuf,
                                                       uint32_t valid)
{
    using namespace AscendC::Reg;
    RegTensor<float> remHi, remLo;
    RegTensor<int32_t> validInt, qBase, qInt, qAdjusted;
    MaskReg active, covered, primary, tie, secondary, tmp, adjust, baseSign;

    active = UpdateMask<float>(valid);
    LoadAlign(validInt, candidateValid);
    Compares<int32_t, AscendC::CMPMODE::NE>(covered, validInt, 0, active);
    LoadAlign(qBase, candidateQ);
    LoadAlign(remHi, remainderHiBuf);
    LoadAlign(remLo, remainderLoBuf);
    Adds<int32_t>(qInt, qBase, 0, covered);

    // Positive correction: the remainder crossed +1, or a negative high part
    // must move one integer back toward zero because its exact remainder is positive.
    Compares<float, AscendC::CMPMODE::GT>(primary, remHi, 1.0f, covered);
    Compares<float, AscendC::CMPMODE::EQ>(tie, remHi, 1.0f, covered);
    Compares<float, AscendC::CMPMODE::GE>(secondary, remLo, 0.0f, covered);
    And(tmp, tie, secondary, covered);
    Or(adjust, primary, tmp, covered);

    Compares<float, AscendC::CMPMODE::GT>(primary, remHi, 0.0f, covered);
    Compares<float, AscendC::CMPMODE::EQ>(tie, remHi, 0.0f, covered);
    Compares<float, AscendC::CMPMODE::GT>(secondary, remLo, 0.0f, covered);
    And(tmp, tie, secondary, covered);
    Or(primary, primary, tmp, covered);
    Compares<int32_t, AscendC::CMPMODE::LT>(baseSign, qBase, 0, covered);
    And(tmp, baseSign, primary, covered);
    Or(adjust, adjust, tmp, covered);
    Adds<int32_t>(qAdjusted, qBase, 1, covered);
    Select<int32_t>(qInt, qAdjusted, qInt, adjust);

    // Negative correction. Reuse every predicate and temporary register from the
    // positive path so the VF frame stays below the 6144-byte hardware limit.
    Compares<float, AscendC::CMPMODE::LT>(primary, remHi, -1.0f, covered);
    Compares<float, AscendC::CMPMODE::EQ>(tie, remHi, -1.0f, covered);
    Compares<float, AscendC::CMPMODE::LE>(secondary, remLo, 0.0f, covered);
    And(tmp, tie, secondary, covered);
    Or(adjust, primary, tmp, covered);

    Compares<float, AscendC::CMPMODE::LT>(primary, remHi, 0.0f, covered);
    Compares<float, AscendC::CMPMODE::EQ>(tie, remHi, 0.0f, covered);
    Compares<float, AscendC::CMPMODE::LT>(secondary, remLo, 0.0f, covered);
    And(tmp, tie, secondary, covered);
    Or(primary, primary, tmp, covered);
    Compares<int32_t, AscendC::CMPMODE::GT>(baseSign, qBase, 0, covered);
    And(tmp, baseSign, primary, covered);
    Or(adjust, adjust, tmp, covered);
    Adds<int32_t>(qAdjusted, qBase, -1, covered);
    Select<int32_t>(qInt, qAdjusted, qInt, adjust);
    StoreAlign(candidateQ, qInt, active);
}

__simd_vf__ inline void SelectProjectedCandidateVF(__ubuf__ int32_t* candidateQ, __ubuf__ int32_t* candidateValid,
                                                   __ubuf__ int32_t* bestQ, __ubuf__ int32_t* bestFace, uint32_t valid,
                                                   int32_t faceIndex)
{
    using namespace AscendC::Reg;
    RegTensor<int32_t> qInt, validInt, bestQReg, faceReg, bestFaceReg;
    MaskReg active, covered, lessQ, equalQ, lessFace, tie, win;

    active = UpdateMask<float>(valid);
    LoadAlign(qInt, candidateQ);
    LoadAlign(validInt, candidateValid);
    Compares<int32_t, AscendC::CMPMODE::NE>(covered, validInt, 0, active);
    LoadAlign(bestQReg, bestQ);
    LoadAlign(bestFaceReg, bestFace);
    Compare<int32_t, AscendC::CMPMODE::LT>(lessQ, qInt, bestQReg, covered);
    Compare<int32_t, AscendC::CMPMODE::EQ>(equalQ, qInt, bestQReg, covered);
    Duplicate(faceReg, faceIndex);
    Compare<int32_t, AscendC::CMPMODE::LT>(lessFace, faceReg, bestFaceReg, covered);
    And(tie, equalQ, lessFace, covered);
    Or(win, lessQ, tie, covered);
    And(win, win, covered, active);
    Select<int32_t>(bestQReg, qInt, bestQReg, win);
    Select<int32_t>(bestFaceReg, faceReg, bestFaceReg, win);
    StoreAlign(bestQ, bestQReg, active);
    StoreAlign(bestFace, bestFaceReg, active);
}

__simd_vf__ inline void NormalizeBaryVF(__ubuf__ int32_t* bestFace, __ubuf__ float* bestB0, __ubuf__ float* bestB1,
                                        __ubuf__ float* bestB2, __ubuf__ float* bestB0Lo, __ubuf__ float* bestB1Lo,
                                        __ubuf__ float* bestB2Lo, uint32_t valid)
{
    using namespace AscendC::Reg;
    RegTensor<float> a, b, c, area, absArea, sum, safeSum, one;
    RegTensor<float> normalized, outA, outB, outC, invalid;
    RegTensor<int32_t> face;
    MaskReg active, foreground, areaValid, validWinner, sumNonZero;

    active = UpdateMask<float>(valid);
    LoadAlign(face, bestFace);
    Compares<int32_t, AscendC::CMPMODE::NE>(foreground, face, INT32_MAX, active);
    LoadAlign(area, bestB0Lo);
    Abs(absArea, area, active);
    Compares<float, AscendC::CMPMODE::GT>(areaValid, absArea, 0.0f, active);
    And(validWinner, foreground, areaValid, active);

    LoadAlign(a, bestB0);
    LoadAlign(b, bestB1);
    LoadAlign(c, bestB2);
    Add(sum, a, b, active);
    Add(sum, sum, c, active);
    Compares<float, AscendC::CMPMODE::NE>(sumNonZero, sum, 0.0f, active);
    Duplicate(one, 1.0f);
    Select<float>(safeSum, sum, one, sumNonZero);
    Duplicate(invalid, -1.0f);
    Adds(outA, invalid, 0.0f, active);
    Adds(outB, invalid, 0.0f, active);
    Adds(outC, invalid, 0.0f, active);

    Div<float, &kHighPrecisionDiv>(normalized, a, safeSum, validWinner);
    Select<float>(outA, normalized, outA, validWinner);
    Div<float, &kHighPrecisionDiv>(normalized, b, safeSum, validWinner);
    Select<float>(outB, normalized, outB, validWinner);
    Div<float, &kHighPrecisionDiv>(normalized, c, safeSum, validWinner);
    Select<float>(outC, normalized, outC, validWinner);
    StoreAlign(bestB0, outA, active);
    StoreAlign(bestB1, outB, active);
    StoreAlign(bestB2, outC, active);
}

__simd_vf__ inline void EncodeWinnerVF(__ubuf__ int32_t* bestFace, __ubuf__ float* bestB0, __ubuf__ float* bestB1,
                                       __ubuf__ float* bestB2, uint32_t valid)
{
    using namespace AscendC::Reg;
    RegTensor<int32_t> face, encoded, zeroI;
    RegTensor<float> b, zeroF;
    MaskReg active, foreground;
    Duplicate(zeroI, 0);
    Duplicate(zeroF, 0.0f);
    uint32_t remain = valid;
    const uint16_t loops = static_cast<uint16_t>((valid + kFloatLanes - 1U) / kFloatLanes);
    for (uint16_t i = 0; i < loops; ++i) {
        const uint32_t off = static_cast<uint32_t>(i) * kFloatLanes;
        active = UpdateMask<float>(remain);
        LoadAlign(face, bestFace + off);
        Compares<int32_t, AscendC::CMPMODE::NE>(foreground, face, INT32_MAX, active);
        Adds(encoded, face, 1, active);
        Select<int32_t>(encoded, encoded, zeroI, foreground);
        StoreAlign(bestFace + off, encoded, active);
        LoadAlign(b, bestB0 + off);
        Select<float>(b, b, zeroF, foreground);
        StoreAlign(bestB0 + off, b, active);
        LoadAlign(b, bestB1 + off);
        Select<float>(b, b, zeroF, foreground);
        StoreAlign(bestB1 + off, b, active);
        LoadAlign(b, bestB2 + off);
        Select<float>(b, b, zeroF, foreground);
        StoreAlign(bestB2 + off, b, active);
    }
}

__simd_vf__ inline void MergePartialVF(__ubuf__ float* bestQ, __ubuf__ int32_t* bestFace, __ubuf__ float* bestB0,
                                       __ubuf__ float* bestB1, __ubuf__ float* bestB2, __ubuf__ float* srcQ,
                                       __ubuf__ int32_t* srcFace, __ubuf__ float* srcB0, __ubuf__ float* srcB1,
                                       __ubuf__ float* srcB2, uint32_t valid)
{
    using namespace AscendC::Reg;
    RegTensor<float> currentQ, sourceQ, currentB, sourceB;
    RegTensor<int32_t> currentFace, sourceFace;
    MaskReg active, lessQ, equalQ, lessFace, tie, win;
    uint32_t remain = valid;
    const uint16_t loops = static_cast<uint16_t>((valid + kFloatLanes - 1U) / kFloatLanes);
    for (uint16_t i = 0; i < loops; ++i) {
        active = AscendC::Reg::UpdateMask<float>(remain);
        const uint32_t off = static_cast<uint32_t>(i) * kFloatLanes;
        LoadAlign(currentQ, bestQ + off);
        LoadAlign(sourceQ, srcQ + off);
        Compare<float, AscendC::CMPMODE::LT>(lessQ, sourceQ, currentQ, active);
        Compare<float, AscendC::CMPMODE::EQ>(equalQ, sourceQ, currentQ, active);
        LoadAlign(currentFace, bestFace + off);
        LoadAlign(sourceFace, srcFace + off);
        Compare<int32_t, AscendC::CMPMODE::LT>(lessFace, sourceFace, currentFace, active);
        And(tie, equalQ, lessFace, active);
        Or(win, lessQ, tie, active);

        Select<float>(currentQ, sourceQ, currentQ, win);
        Select<int32_t>(currentFace, sourceFace, currentFace, win);
        StoreAlign(bestQ + off, currentQ, active);
        StoreAlign(bestFace + off, currentFace, active);
        LoadAlign(currentB, bestB0 + off);
        LoadAlign(sourceB, srcB0 + off);
        Select<float>(currentB, sourceB, currentB, win);
        StoreAlign(bestB0 + off, currentB, active);
        LoadAlign(currentB, bestB1 + off);
        LoadAlign(sourceB, srcB1 + off);
        Select<float>(currentB, sourceB, currentB, win);
        StoreAlign(bestB1 + off, currentB, active);
        LoadAlign(currentB, bestB2 + off);
        LoadAlign(sourceB, srcB2 + off);
        Select<float>(currentB, sourceB, currentB, win);
        StoreAlign(bestB2 + off, currentB, active);
    }
}

class PixelDirect {
public:
    __aicore__ inline void Init(GM_ADDR v, GM_ADDR f, GM_ADDR findices, GM_ADDR barycentric,
                                const RasterizerTilingData* tilingData)
    {
        tiling_ = tilingData;
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v));
        fGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(f));
        findicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(findices));
        barycentricGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(barycentric));
        // Host Branch-0 budgets five winner planes after reserving faceQueue_. PixelDirect
        // expands that area to thirteen planes and a 256-byte software-binary64 common
        // buffer. SIMD VF register spills use an UB-backed stack as well: Bisheng reports
        // a 5888-byte maximum for the fused projected-barycentric VF. Keep the common
        // buffer, the measured stack and one vector-quantum guard outside the plane budget.
        const uint32_t hostTileQuanta = static_cast<uint32_t>(tiling_->pixelTileLen) / kFloatLanes;
        const uint32_t totalBudgetQuanta = hostTileQuanta * 5U;
        constexpr uint32_t commonQuanta = 1U;
        constexpr uint32_t stackAndGuardQuanta = (kPixelDirectMaxVfStackBytes + kPixelDirectVfStackGuardBytes +
                                                  kVectorBytes - 1U) /
                                                 kVectorBytes;
        constexpr uint32_t reservedQuanta = commonQuanta + stackAndGuardQuanta;
        const uint32_t planeBudgetQuanta = totalBudgetQuanta > reservedQuanta ? totalBudgetQuanta - reservedQuanta : 0U;
        const uint32_t dsTileQuanta = planeBudgetQuanta / 13U;
        tileLen_ = kFloatLanes;
        const uint32_t planeBytes = tileLen_ * sizeof(float);
        pipe_.InitBuffer(faceQueue_, 1, kFacePacketBytes);
        pipe_.InitBuffer(bestQBuf_, planeBytes);
        pipe_.InitBuffer(bestFaceBuf_, planeBytes);
        pipe_.InitBuffer(bestB0Buf_, planeBytes);
        pipe_.InitBuffer(bestB1Buf_, planeBytes);
        pipe_.InitBuffer(bestB2Buf_, planeBytes);
        pipe_.InitBuffer(bestB0LoBuf_, planeBytes);
        pipe_.InitBuffer(bestB1LoBuf_, planeBytes);
        pipe_.InitBuffer(bestB2LoBuf_, planeBytes);
        pipe_.InitBuffer(coverageBuf_, planeBytes);
        pipe_.InitBuffer(baryScratchHiBuf_, planeBytes);
        pipe_.InitBuffer(baryScratchLoBuf_, planeBytes);
        pipe_.InitBuffer(baryOtherScratchHiBuf_, planeBytes);
        pipe_.InitBuffer(baryOtherScratchLoBuf_, planeBytes);
        pipe_.InitBuffer(projectedF64CommonBuf_, kVectorBytes);
    }

    __aicore__ inline void Process()
    {
        const uint64_t blockStart = static_cast<uint64_t>(AscendC::GetBlockIdx()) * tiling_->pixelBlockLen;
        if (blockStart >= tiling_->totalPixels) {
            return;
        }
        uint64_t blockCount = tiling_->totalPixels - blockStart;
        if (blockCount > tiling_->pixelBlockLen) {
            blockCount = tiling_->pixelBlockLen;
        }
        for (uint64_t done = 0; done < blockCount; done += tileLen_) {
            uint32_t valid = static_cast<uint32_t>(blockCount - done);
            if (valid > tileLen_) {
                valid = tileLen_;
            }
            ProcessTile(blockStart + done, valid);
        }
    }

private:
    __aicore__ inline void CopyInFace(uint32_t faceIndex)
    {
        AscendC::LocalTensor<int32_t> packet = faceQueue_.AllocTensor<int32_t>();
        AscendC::DataCopyExtParams faceCopy{1, static_cast<uint32_t>(3U * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> facePad{false, 0, 0, 0};
        AscendC::DataCopyPad(packet, fGm_[static_cast<uint64_t>(faceIndex) * 3U], faceCopy, facePad);
        AscendC::LocalTensor<float> packetF = packet.ReinterpretCast<float>();
        AscendC::DataCopyExtParams coordCopy{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> coordPad{false, 0, 0, 0.0f};
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        for (uint32_t vertex = 0; vertex < 3U; ++vertex) {
            const int32_t vertexIndex = packet.GetValue(vertex);
            for (uint32_t axis = 0; axis < 4U; ++axis) {
                const uint32_t packetOffset = kFaceCoordBase + vertex * kFaceVertexStride + axis * kFaceCoordSlotFloats;
                const uint64_t vertexOffset = static_cast<uint64_t>(vertexIndex) * 4U + axis;
                AscendC::DataCopyPad(packetF[packetOffset], vGm_[vertexOffset], coordCopy, coordPad);
            }
        }
        faceQueue_.EnQue(packet);
    }

    __aicore__ inline void CopyOut(uint64_t tileStart, uint32_t valid)
    {
        AscendC::LocalTensor<int32_t> face = bestFaceBuf_.Get<int32_t>();
        AscendC::LocalTensor<float> b0 = bestB0Buf_.Get<float>();
        AscendC::LocalTensor<float> b1 = bestB1Buf_.Get<float>();
        AscendC::LocalTensor<float> b2 = bestB2Buf_.Get<float>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams faceCopy{1, static_cast<uint32_t>(valid * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(findicesGm_[tileStart], face, faceCopy);

        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        AscendC::LocalTensor<int32_t> stagingI = faceQueue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<float> staging = stagingI.ReinterpretCast<float>();
        for (uint32_t base = 0; base < valid; base += kPackPixels) {
            uint32_t count = valid - base;
            if (count > kPackPixels) {
                count = kPackPixels;
            }
            for (uint32_t i = 0; i < count; ++i) {
                staging.SetValue(3U * i, b0.GetValue(base + i));
                staging.SetValue(3U * i + 1U, b1.GetValue(base + i));
                staging.SetValue(3U * i + 2U, b2.GetValue(base + i));
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::DataCopyExtParams baryCopy{1, static_cast<uint32_t>(count * 3U * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(barycentricGm_[3U * (tileStart + base)], staging, baryCopy);
            // DataCopyPad is asynchronous on MTE3. Wait before the scalar pipe
            // repacks or releases the shared staging buffer.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        }
        faceQueue_.FreeTensor(stagingI);
    }

    __aicore__ inline void ProcessTile(uint64_t tileStart, uint32_t valid)
    {
        AscendC::LocalTensor<int32_t> bestQ = bestQBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> bestFace = bestFaceBuf_.Get<int32_t>();
        AscendC::LocalTensor<float> bestB0 = bestB0Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB1 = bestB1Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB2 = bestB2Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB0Lo = bestB0LoBuf_.Get<float>();
        AscendC::LocalTensor<float> bestB1Lo = bestB1LoBuf_.Get<float>();
        AscendC::LocalTensor<float> bestB2Lo = bestB2LoBuf_.Get<float>();
        AscendC::LocalTensor<int32_t> coverage = coverageBuf_.Get<int32_t>();
        AscendC::LocalTensor<float> baryScratchHi = baryScratchHiBuf_.Get<float>();
        AscendC::LocalTensor<float> baryScratchLo = baryScratchLoBuf_.Get<float>();
        AscendC::LocalTensor<float> baryOtherScratchHi = baryOtherScratchHiBuf_.Get<float>();
        AscendC::LocalTensor<float> baryOtherScratchLo = baryOtherScratchLoBuf_.Get<float>();
        AscendC::LocalTensor<uint32_t> projectedF64Common = projectedF64CommonBuf_.Get<uint32_t>();
        for (uint32_t base = 0; base < valid; base += kFloatLanes) {
            uint32_t chunk = valid - base;
            if (chunk > kFloatLanes) {
                chunk = kFloatLanes;
            }
            asc_vf_call<InitWinnerIntVF>(
                (__ubuf__ int32_t*)bestQ[base].GetPhyAddr(), (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(),
                (__ubuf__ float*)bestB0[base].GetPhyAddr(), (__ubuf__ float*)bestB1[base].GetPhyAddr(),
                (__ubuf__ float*)bestB2[base].GetPhyAddr(), chunk);
            asc_vf_call<InitBaryLoVF>((__ubuf__ float*)bestB0Lo[base].GetPhyAddr(),
                                      (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(),
                                      (__ubuf__ float*)bestB2Lo[base].GetPhyAddr(), chunk);
        }
        for (uint32_t faceIndex = 0; faceIndex < tiling_->numFaces; ++faceIndex) {
            CopyInFace(faceIndex);
            AscendC::LocalTensor<int32_t> packet = faceQueue_.DeQue<int32_t>();
            for (uint32_t base = 0; base < valid; base += kFloatLanes) {
                uint32_t chunk = valid - base;
                if (chunk > kFloatLanes) {
                    chunk = kFloatLanes;
                }
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 0U, 0U, 0U);
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 0U, 0U, 1U);
                asc_vf_call<CombineProjectedBaryProductVF>((__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB0Lo[base].GetPhyAddr(), chunk, 0U);
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 0U, 1U, 0U);
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 0U, 1U, 1U);
                asc_vf_call<CombineProjectedBaryProductVF>((__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB0Lo[base].GetPhyAddr(), chunk, 1U);
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 1U, 0U, 0U);
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 1U, 0U, 1U);
                asc_vf_call<CombineProjectedBaryProductVF>((__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB1[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(), chunk, 0U);
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 1U, 1U, 0U);
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 1U, 1U, 1U);
                asc_vf_call<CombineProjectedBaryProductVF>((__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB1[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(), chunk, 1U);
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 2U, 0U, 0U);
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 2U, 0U, 1U);
                asc_vf_call<CombineProjectedBaryProductVF>((__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB2[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB2Lo[base].GetPhyAddr(), chunk, 0U);
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 2U, 1U, 0U);
                asc_vf_call<PrepareProjectedBaryEdgeVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                    (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(), chunk, static_cast<float>(tileStart + base),
                    static_cast<float>(tiling_->width), static_cast<float>(tiling_->width - 1U),
                    static_cast<float>(tiling_->height - 1U), 2U, 1U, 1U);
                asc_vf_call<CombineProjectedBaryProductVF>((__ubuf__ float*)baryScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchHi[base].GetPhyAddr(),
                                                           (__ubuf__ float*)baryOtherScratchLo[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB2[base].GetPhyAddr(),
                                                           (__ubuf__ float*)bestB2Lo[base].GetPhyAddr(), chunk, 1U);
                asc_vf_call<ComputeProjectedCoverageVF>(
                    (__ubuf__ int32_t*)coverage[base].GetPhyAddr(), (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB2[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB0Lo[base].GetPhyAddr(), (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB2Lo[base].GetPhyAddr(), chunk);
                asc_vf_call<DetectProjectedCoverageRefineF64Simt>(
                    dim3(kFloatLanes), (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB2[base].GetPhyAddr(),
                    (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr(), chunk);
                asc_vf_call<PrepareProjectedVertexXF64Simt>(dim3(kFloatLanes), (__ubuf__ float*)packet.GetPhyAddr(),
                                                            (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr(), 0U,
                                                            tiling_->width - 1U, kProjectedX0Word);
                asc_vf_call<PrepareProjectedVertexYF64Simt>(dim3(kFloatLanes), (__ubuf__ float*)packet.GetPhyAddr(),
                                                            (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr(), 0U,
                                                            tiling_->height - 1U, kProjectedY0Word);
                asc_vf_call<PrepareProjectedVertexXF64Simt>(dim3(kFloatLanes), (__ubuf__ float*)packet.GetPhyAddr(),
                                                            (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr(), 1U,
                                                            tiling_->width - 1U, kProjectedX1Word);
                asc_vf_call<PrepareProjectedVertexYF64Simt>(dim3(kFloatLanes), (__ubuf__ float*)packet.GetPhyAddr(),
                                                            (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr(), 1U,
                                                            tiling_->height - 1U, kProjectedY1Word);
                asc_vf_call<PrepareProjectedVertexXF64Simt>(dim3(kFloatLanes), (__ubuf__ float*)packet.GetPhyAddr(),
                                                            (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr(), 2U,
                                                            tiling_->width - 1U, kProjectedX2Word);
                asc_vf_call<PrepareProjectedVertexYF64Simt>(dim3(kFloatLanes), (__ubuf__ float*)packet.GetPhyAddr(),
                                                            (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr(), 2U,
                                                            tiling_->height - 1U, kProjectedY2Word);
                asc_vf_call<PrepareProjectedAreaF64Simt>(dim3(kFloatLanes),
                                                         (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr());
                asc_vf_call<PrepareProjectedBetaNumeratorF64Simt>(
                    dim3(kFloatLanes), (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB2[base].GetPhyAddr(),
                    (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr(),
                    (__ubuf__ uint32_t*)coverage[base].GetPhyAddr(),
                    (__ubuf__ uint32_t*)baryScratchLo[base].GetPhyAddr(), chunk, tileStart + base, tiling_->width);
                asc_vf_call<FinishProjectedBetaF64Simt>(
                    dim3(kFloatLanes), (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB2[base].GetPhyAddr(),
                    (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr(),
                    (__ubuf__ uint32_t*)coverage[base].GetPhyAddr(),
                    (__ubuf__ uint32_t*)baryScratchLo[base].GetPhyAddr(),
                    (__ubuf__ uint32_t*)baryOtherScratchHi[base].GetPhyAddr(),
                    (__ubuf__ uint32_t*)baryOtherScratchLo[base].GetPhyAddr(), chunk);
                asc_vf_call<PrepareProjectedGammaNumeratorF64Simt>(
                    dim3(kFloatLanes), (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB2[base].GetPhyAddr(),
                    (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr(),
                    (__ubuf__ uint32_t*)coverage[base].GetPhyAddr(),
                    (__ubuf__ uint32_t*)baryScratchLo[base].GetPhyAddr(), chunk, tileStart + base, tiling_->width);
                asc_vf_call<FinishProjectedCoverageF64Simt>(
                    dim3(kFloatLanes), (__ubuf__ uint32_t*)coverage[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB0[base].GetPhyAddr(), (__ubuf__ float*)bestB1[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB2[base].GetPhyAddr(), (__ubuf__ uint32_t*)projectedF64Common.GetPhyAddr(),
                    (__ubuf__ uint32_t*)baryScratchLo[base].GetPhyAddr(),
                    (__ubuf__ uint32_t*)baryOtherScratchHi[base].GetPhyAddr(),
                    (__ubuf__ uint32_t*)baryOtherScratchLo[base].GetPhyAddr(), chunk);
                asc_vf_call<ComputeProjectedDepthFromCoverageVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB0Lo[base].GetPhyAddr(), (__ubuf__ int32_t*)coverage[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB0[base].GetPhyAddr(), (__ubuf__ float*)bestB1[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB2[base].GetPhyAddr(), (__ubuf__ float*)bestB0Lo[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(), (__ubuf__ float*)bestB2Lo[base].GetPhyAddr(), chunk);
                asc_vf_call<PrepareProjectedDepthQuantizeVF>(
                    (__ubuf__ int32_t*)bestB0[base].GetPhyAddr(), (__ubuf__ int32_t*)coverage[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB0[base].GetPhyAddr(), (__ubuf__ float*)bestB0Lo[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(), chunk);
                asc_vf_call<FinishProjectedDepthQuantizeVF>(
                    (__ubuf__ int32_t*)bestB0[base].GetPhyAddr(), (__ubuf__ int32_t*)coverage[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(), chunk);
                asc_vf_call<SelectProjectedCandidateVF>(
                    (__ubuf__ int32_t*)bestB0[base].GetPhyAddr(), (__ubuf__ int32_t*)coverage[base].GetPhyAddr(),
                    (__ubuf__ int32_t*)bestQ[base].GetPhyAddr(), (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(), chunk,
                    static_cast<int32_t>(faceIndex));
            }
            faceQueue_.FreeTensor(packet);
        }
        for (uint32_t faceIndex = 0; faceIndex < tiling_->numFaces; ++faceIndex) {
            CopyInFace(faceIndex);
            AscendC::LocalTensor<int32_t> packet = faceQueue_.DeQue<int32_t>();
            for (uint32_t base = 0; base < valid; base += kFloatLanes) {
                uint32_t chunk = valid - base;
                if (chunk > kFloatLanes) {
                    chunk = kFloatLanes;
                }
                asc_vf_call<RecomputeBaryNumeratorVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB0[base].GetPhyAddr(), (__ubuf__ float*)bestB0Lo[base].GetPhyAddr(), chunk,
                    static_cast<float>(tileStart + base), static_cast<float>(tiling_->width),
                    static_cast<float>(tiling_->width - 1U), static_cast<float>(tiling_->height - 1U),
                    static_cast<int32_t>(faceIndex), 0U, 1U);
                asc_vf_call<RecomputeBaryNumeratorVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(), chunk,
                    static_cast<float>(tileStart + base), static_cast<float>(tiling_->width),
                    static_cast<float>(tiling_->width - 1U), static_cast<float>(tiling_->height - 1U),
                    static_cast<int32_t>(faceIndex), 1U, 1U);
                asc_vf_call<RecomputeBaryNumeratorVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB2[base].GetPhyAddr(), (__ubuf__ float*)bestB2Lo[base].GetPhyAddr(), chunk,
                    static_cast<float>(tileStart + base), static_cast<float>(tiling_->width),
                    static_cast<float>(tiling_->width - 1U), static_cast<float>(tiling_->height - 1U),
                    static_cast<int32_t>(faceIndex), 2U, 1U);
            }
            faceQueue_.FreeTensor(packet);
        }
        for (uint32_t base = 0; base < valid; base += kFloatLanes) {
            uint32_t chunk = valid - base;
            if (chunk > kFloatLanes) {
                chunk = kFloatLanes;
            }
            asc_vf_call<NormalizeBaryVF>(
                (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(), (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB2[base].GetPhyAddr(),
                (__ubuf__ float*)bestB0Lo[base].GetPhyAddr(), (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(),
                (__ubuf__ float*)bestB2Lo[base].GetPhyAddr(), chunk);
        }
        for (uint32_t base = 0; base < valid; base += kFloatLanes) {
            uint32_t chunk = valid - base;
            if (chunk > kFloatLanes) {
                chunk = kFloatLanes;
            }
            asc_vf_call<EncodeWinnerVF>(
                (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(), (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB2[base].GetPhyAddr(), chunk);
        }
        CopyOut(tileStart, valid);
    }

    AscendC::TPipe pipe_;
    const RasterizerTilingData* tiling_ = nullptr;
    AscendC::GlobalTensor<float> vGm_;
    AscendC::GlobalTensor<int32_t> fGm_;
    AscendC::GlobalTensor<int32_t> findicesGm_;
    AscendC::GlobalTensor<float> barycentricGm_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> faceQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestQBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestFaceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestB0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestB1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestB2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestB0LoBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestB1LoBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestB2LoBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> coverageBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> baryScratchHiBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> baryScratchLoBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> baryOtherScratchHiBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> baryOtherScratchLoBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> projectedF64CommonBuf_;
    uint32_t tileLen_ = 0;
};

class FaceGroup {
public:
    __aicore__ inline void Init(GM_ADDR v, GM_ADDR f, GM_ADDR findices, GM_ADDR barycentric, GM_ADDR workspace,
                                const RasterizerTilingData* tilingData)
    {
        initialized_ = false;
        if (tilingData == nullptr || workspace == nullptr) {
            return;
        }
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        if (userWorkspace == nullptr) {
            return;
        }
        tiling_ = tilingData;
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v));
        fGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(f));
        findicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(findices));
        barycentricGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(barycentric));
        workspaceF_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWorkspace));
        workspaceI_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkspace));
        tileLen_ = static_cast<uint32_t>(tiling_->pixelTileLen);
        const uint32_t planeBytes = tileLen_ * sizeof(float);
        pipe_.InitBuffer(faceQueue_, 1, kFacePacketBytes);
        pipe_.InitBuffer(bestQBuf_, planeBytes);
        pipe_.InitBuffer(bestFaceBuf_, planeBytes);
        pipe_.InitBuffer(bestB0Buf_, planeBytes);
        pipe_.InitBuffer(bestB1Buf_, planeBytes);
        pipe_.InitBuffer(bestB2Buf_, planeBytes);
        pipe_.InitBuffer(srcQBuf_, planeBytes);
        pipe_.InitBuffer(srcFaceBuf_, planeBytes);
        pipe_.InitBuffer(srcB0Buf_, planeBytes);
        pipe_.InitBuffer(srcB1Buf_, planeBytes);
        pipe_.InitBuffer(srcB2Buf_, planeBytes);
        pipe_.InitBuffer(faceValidityBuf_, kFaceValidityScratchBytes);
        initialized_ = true;
    }

    __aicore__ inline void Process()
    {
        if (!initialized_) {
            return;
        }
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t quotient = tiling_->numFaces / tiling_->activeCoreNum;
        const uint32_t remainder = tiling_->numFaces % tiling_->activeCoreNum;
        const uint32_t faceStart = blockIdx * quotient + (blockIdx < remainder ? blockIdx : remainder);
        const uint32_t faceCount = quotient + static_cast<uint32_t>(blockIdx < remainder);

        for (uint64_t tileStart = 0; tileStart < tiling_->totalPixels; tileStart += tileLen_) {
            uint32_t valid = static_cast<uint32_t>(tiling_->totalPixels - tileStart);
            if (valid > tileLen_) {
                valid = tileLen_;
            }
            ProcessPhase1Tile(tileStart, valid, faceStart, faceCount);
        }
        WriteWorkspacePadding();

        AscendC::SyncAll();
        if (blockIdx >= tiling_->mergeCoreNum) {
            return;
        }

        const uint64_t mergeStart = static_cast<uint64_t>(blockIdx) * tiling_->pixelBlockLen;
        if (mergeStart >= tiling_->totalPixels) {
            return;
        }
        uint64_t mergeValid = tiling_->totalPixels - mergeStart;
        if (mergeValid > tiling_->pixelBlockLen) {
            mergeValid = tiling_->pixelBlockLen;
        }
        for (uint64_t done = 0; done < mergeValid; done += tileLen_) {
            uint32_t valid = static_cast<uint32_t>(mergeValid - done);
            if (valid > tileLen_) {
                valid = tileLen_;
            }
            ProcessMergeTile(mergeStart + done, valid);
        }
    }

private:
    __aicore__ inline void CopyInFace(uint32_t faceIndex)
    {
        AscendC::LocalTensor<int32_t> packet = faceQueue_.AllocTensor<int32_t>();
        AscendC::DataCopyExtParams faceCopy{1, static_cast<uint32_t>(3U * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> facePad{false, 0, 0, 0};
        AscendC::DataCopyPad(packet, fGm_[static_cast<uint64_t>(faceIndex) * 3U], faceCopy, facePad);
        AscendC::LocalTensor<float> packetF = packet.ReinterpretCast<float>();
        AscendC::DataCopyExtParams coordCopy{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> coordPad{false, 0, 0, 0.0f};
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        for (uint32_t vertex = 0; vertex < 3U; ++vertex) {
            const int32_t vertexIndex = packet.GetValue(vertex);
            for (uint32_t axis = 0; axis < 4U; ++axis) {
                const uint32_t packetOffset = kFaceCoordBase + vertex * kFaceVertexStride + axis * kFaceCoordSlotFloats;
                const uint64_t vertexOffset = static_cast<uint64_t>(vertexIndex) * 4U + axis;
                AscendC::DataCopyPad(packetF[packetOffset], vGm_[vertexOffset], coordCopy, coordPad);
            }
        }
        faceQueue_.EnQue(packet);
    }

    __aicore__ inline void InitCurrent(uint32_t valid)
    {
        AscendC::LocalTensor<float> bestQ = bestQBuf_.Get<float>();
        AscendC::LocalTensor<int32_t> bestFace = bestFaceBuf_.Get<int32_t>();
        AscendC::LocalTensor<float> bestB0 = bestB0Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB1 = bestB1Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB2 = bestB2Buf_.Get<float>();
        for (uint32_t base = 0; base < valid; base += kFloatLanes) {
            uint32_t chunk = valid - base;
            if (chunk > kFloatLanes) {
                chunk = kFloatLanes;
            }
            asc_vf_call<InitWinnerVF>(
                (__ubuf__ float*)bestQ[base].GetPhyAddr(), (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(),
                (__ubuf__ float*)bestB0[base].GetPhyAddr(), (__ubuf__ float*)bestB1[base].GetPhyAddr(),
                (__ubuf__ float*)bestB2[base].GetPhyAddr(), chunk);
        }
    }

    __aicore__ inline uint64_t CoreBase(uint32_t core) const
    {
        return static_cast<uint64_t>(core) * (tiling_->workspaceCoreStrideBytes / sizeof(float));
    }

    __aicore__ inline void CopyPartialOut(uint64_t tileStart, uint32_t valid)
    {
        const uint64_t plane = tiling_->workspacePixelStride;
        const uint64_t coreBase = CoreBase(AscendC::GetBlockIdx());
        AscendC::LocalTensor<float> bestQ = bestQBuf_.Get<float>();
        AscendC::LocalTensor<int32_t> bestFace = bestFaceBuf_.Get<int32_t>();
        AscendC::LocalTensor<float> bestB0 = bestB0Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB1 = bestB1Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB2 = bestB2Buf_.Get<float>();
        const AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0};
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyPad(workspaceF_[coreBase + tileStart], bestQ, copy);
        AscendC::DataCopyPad(workspaceI_[coreBase + plane + tileStart], bestFace, copy);
        AscendC::DataCopyPad(workspaceF_[coreBase + 2U * plane + tileStart], bestB0, copy);
        AscendC::DataCopyPad(workspaceF_[coreBase + 3U * plane + tileStart], bestB1, copy);
        AscendC::DataCopyPad(workspaceF_[coreBase + 4U * plane + tileStart], bestB2, copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void ProcessPhase1Tile(uint64_t tileStart, uint32_t valid, uint32_t faceStart, uint32_t faceCount)
    {
        InitCurrent(valid);
        AscendC::LocalTensor<float> bestQ = bestQBuf_.Get<float>();
        AscendC::LocalTensor<int32_t> bestFace = bestFaceBuf_.Get<int32_t>();
        AscendC::LocalTensor<float> bestB0 = bestB0Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB1 = bestB1Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB2 = bestB2Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB0Lo = srcB0Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB1Lo = srcB1Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB2Lo = srcB2Buf_.Get<float>();
        AscendC::LocalTensor<int32_t> faceValid = faceValidityBuf_.Get<int32_t>();
        for (uint32_t base = 0; base < valid; base += kFloatLanes) {
            uint32_t chunk = valid - base;
            if (chunk > kFloatLanes) {
                chunk = kFloatLanes;
            }
            asc_vf_call<InitBaryLoVF>((__ubuf__ float*)bestB0Lo[base].GetPhyAddr(),
                                      (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(),
                                      (__ubuf__ float*)bestB2Lo[base].GetPhyAddr(), chunk);
        }
        for (uint32_t faceIndex = faceStart; faceIndex < faceStart + faceCount; ++faceIndex) {
            CopyInFace(faceIndex);
            AscendC::LocalTensor<int32_t> packet = faceQueue_.DeQue<int32_t>();
            asc_vf_call<ComputeProjectedFaceValidityVF>(
                (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ int32_t*)faceValid.GetPhyAddr(),
                static_cast<float>(tiling_->width - 1U), static_cast<float>(tiling_->height - 1U));
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            if (faceValid.GetValue(0) == 0) {
                faceQueue_.FreeTensor(packet);
                continue;
            }
            for (uint32_t base = 0; base < valid; base += kFloatLanes) {
                uint32_t chunk = valid - base;
                if (chunk > kFloatLanes) {
                    chunk = kFloatLanes;
                }
                asc_vf_call<EvaluateFaceTile>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ float*)bestQ[base].GetPhyAddr(),
                    (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(), (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB2[base].GetPhyAddr(), chunk,
                    static_cast<float>(tileStart + base), static_cast<float>(tiling_->width),
                    static_cast<float>(tiling_->width - 1U), static_cast<float>(tiling_->height - 1U),
                    static_cast<int32_t>(faceIndex));
            }
            faceQueue_.FreeTensor(packet);
        }
        for (uint32_t faceIndex = faceStart; faceIndex < faceStart + faceCount; ++faceIndex) {
            CopyInFace(faceIndex);
            AscendC::LocalTensor<int32_t> packet = faceQueue_.DeQue<int32_t>();
            asc_vf_call<ComputeProjectedFaceValidityVF>(
                (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ int32_t*)faceValid.GetPhyAddr(),
                static_cast<float>(tiling_->width - 1U), static_cast<float>(tiling_->height - 1U));
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            if (faceValid.GetValue(0) == 0) {
                faceQueue_.FreeTensor(packet);
                continue;
            }
            for (uint32_t base = 0; base < valid; base += kFloatLanes) {
                uint32_t chunk = valid - base;
                if (chunk > kFloatLanes) {
                    chunk = kFloatLanes;
                }
                asc_vf_call<RecomputeBaryNumeratorVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB0[base].GetPhyAddr(), (__ubuf__ float*)bestB0Lo[base].GetPhyAddr(), chunk,
                    static_cast<float>(tileStart + base), static_cast<float>(tiling_->width),
                    static_cast<float>(tiling_->width - 1U), static_cast<float>(tiling_->height - 1U),
                    static_cast<int32_t>(faceIndex), 0U, 1U);
                asc_vf_call<RecomputeBaryNumeratorVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(), chunk,
                    static_cast<float>(tileStart + base), static_cast<float>(tiling_->width),
                    static_cast<float>(tiling_->width - 1U), static_cast<float>(tiling_->height - 1U),
                    static_cast<int32_t>(faceIndex), 1U, 1U);
                asc_vf_call<RecomputeBaryNumeratorVF>(
                    (__ubuf__ float*)packet.GetPhyAddr(), (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(),
                    (__ubuf__ float*)bestB2[base].GetPhyAddr(), (__ubuf__ float*)bestB2Lo[base].GetPhyAddr(), chunk,
                    static_cast<float>(tileStart + base), static_cast<float>(tiling_->width),
                    static_cast<float>(tiling_->width - 1U), static_cast<float>(tiling_->height - 1U),
                    static_cast<int32_t>(faceIndex), 2U, 1U);
            }
            faceQueue_.FreeTensor(packet);
        }
        for (uint32_t base = 0; base < valid; base += kFloatLanes) {
            uint32_t chunk = valid - base;
            if (chunk > kFloatLanes) {
                chunk = kFloatLanes;
            }
            asc_vf_call<NormalizeBaryVF>(
                (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(), (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB2[base].GetPhyAddr(),
                (__ubuf__ float*)bestB0Lo[base].GetPhyAddr(), (__ubuf__ float*)bestB1Lo[base].GetPhyAddr(),
                (__ubuf__ float*)bestB2Lo[base].GetPhyAddr(), chunk);
        }
        CopyPartialOut(tileStart, valid);
    }

    __aicore__ inline void WriteWorkspacePadding()
    {
        const uint64_t padding = tiling_->workspacePixelStride - tiling_->totalPixels;
        if (padding == 0U) {
            return;
        }
        const uint32_t valid = static_cast<uint32_t>(padding);
        InitCurrent(valid);
        CopyPartialOut(tiling_->totalPixels, valid);
    }

    __aicore__ inline void CopySourceIn(uint32_t sourceCore, uint64_t tileStart, uint32_t valid)
    {
        const uint64_t plane = tiling_->workspacePixelStride;
        const uint64_t coreBase = CoreBase(sourceCore);
        AscendC::LocalTensor<float> srcQ = srcQBuf_.Get<float>();
        AscendC::LocalTensor<int32_t> srcFace = srcFaceBuf_.Get<int32_t>();
        AscendC::LocalTensor<float> srcB0 = srcB0Buf_.Get<float>();
        AscendC::LocalTensor<float> srcB1 = srcB1Buf_.Get<float>();
        AscendC::LocalTensor<float> srcB2 = srcB2Buf_.Get<float>();
        const AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<float> floatPad{false, 0, 0, 0.0f};
        const AscendC::DataCopyPadExtParams<int32_t> intPad{false, 0, 0, 0};
        AscendC::DataCopyPad(srcQ, workspaceF_[coreBase + tileStart], copy, floatPad);
        AscendC::DataCopyPad(srcFace, workspaceI_[coreBase + plane + tileStart], copy, intPad);
        AscendC::DataCopyPad(srcB0, workspaceF_[coreBase + 2U * plane + tileStart], copy, floatPad);
        AscendC::DataCopyPad(srcB1, workspaceF_[coreBase + 3U * plane + tileStart], copy, floatPad);
        AscendC::DataCopyPad(srcB2, workspaceF_[coreBase + 4U * plane + tileStart], copy, floatPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    }

    __aicore__ inline void MergeSource(uint32_t valid)
    {
        AscendC::LocalTensor<float> bestQ = bestQBuf_.Get<float>();
        AscendC::LocalTensor<int32_t> bestFace = bestFaceBuf_.Get<int32_t>();
        AscendC::LocalTensor<float> bestB0 = bestB0Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB1 = bestB1Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB2 = bestB2Buf_.Get<float>();
        AscendC::LocalTensor<float> srcQ = srcQBuf_.Get<float>();
        AscendC::LocalTensor<int32_t> srcFace = srcFaceBuf_.Get<int32_t>();
        AscendC::LocalTensor<float> srcB0 = srcB0Buf_.Get<float>();
        AscendC::LocalTensor<float> srcB1 = srcB1Buf_.Get<float>();
        AscendC::LocalTensor<float> srcB2 = srcB2Buf_.Get<float>();
        for (uint32_t base = 0; base < valid; base += kFloatLanes) {
            uint32_t chunk = valid - base;
            if (chunk > kFloatLanes) {
                chunk = kFloatLanes;
            }
            asc_vf_call<MergePartialVF>(
                (__ubuf__ float*)bestQ[base].GetPhyAddr(), (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(),
                (__ubuf__ float*)bestB0[base].GetPhyAddr(), (__ubuf__ float*)bestB1[base].GetPhyAddr(),
                (__ubuf__ float*)bestB2[base].GetPhyAddr(), (__ubuf__ float*)srcQ[base].GetPhyAddr(),
                (__ubuf__ int32_t*)srcFace[base].GetPhyAddr(), (__ubuf__ float*)srcB0[base].GetPhyAddr(),
                (__ubuf__ float*)srcB1[base].GetPhyAddr(), (__ubuf__ float*)srcB2[base].GetPhyAddr(), chunk);
        }
    }

    __aicore__ inline void EncodeCurrent(uint32_t valid)
    {
        AscendC::LocalTensor<int32_t> bestFace = bestFaceBuf_.Get<int32_t>();
        AscendC::LocalTensor<float> bestB0 = bestB0Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB1 = bestB1Buf_.Get<float>();
        AscendC::LocalTensor<float> bestB2 = bestB2Buf_.Get<float>();
        for (uint32_t base = 0; base < valid; base += kFloatLanes) {
            uint32_t chunk = valid - base;
            if (chunk > kFloatLanes) {
                chunk = kFloatLanes;
            }
            asc_vf_call<EncodeWinnerVF>(
                (__ubuf__ int32_t*)bestFace[base].GetPhyAddr(), (__ubuf__ float*)bestB0[base].GetPhyAddr(),
                (__ubuf__ float*)bestB1[base].GetPhyAddr(), (__ubuf__ float*)bestB2[base].GetPhyAddr(), chunk);
        }
    }

    __aicore__ inline void CopyOut(uint64_t tileStart, uint32_t valid)
    {
        AscendC::LocalTensor<int32_t> face = bestFaceBuf_.Get<int32_t>();
        AscendC::LocalTensor<float> b0 = bestB0Buf_.Get<float>();
        AscendC::LocalTensor<float> b1 = bestB1Buf_.Get<float>();
        AscendC::LocalTensor<float> b2 = bestB2Buf_.Get<float>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const AscendC::DataCopyExtParams faceCopy{1, static_cast<uint32_t>(valid * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(findicesGm_[tileStart], face, faceCopy);

        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        AscendC::LocalTensor<int32_t> stagingI = faceQueue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<float> staging = stagingI.ReinterpretCast<float>();
        for (uint32_t base = 0; base < valid; base += kPackPixels) {
            uint32_t count = valid - base;
            if (count > kPackPixels) {
                count = kPackPixels;
            }
            for (uint32_t i = 0; i < count; ++i) {
                staging.SetValue(3U * i, b0.GetValue(base + i));
                staging.SetValue(3U * i + 1U, b1.GetValue(base + i));
                staging.SetValue(3U * i + 2U, b2.GetValue(base + i));
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            const AscendC::DataCopyExtParams baryCopy{1, static_cast<uint32_t>(count * 3U * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(barycentricGm_[3U * (tileStart + base)], staging, baryCopy);
            // DataCopyPad is asynchronous on MTE3. Wait before the scalar pipe
            // repacks or releases the shared staging buffer.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        }
        faceQueue_.FreeTensor(stagingI);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void ProcessMergeTile(uint64_t tileStart, uint32_t valid)
    {
        InitCurrent(valid);
        for (uint32_t sourceCore = 0; sourceCore < tiling_->activeCoreNum; ++sourceCore) {
            if (sourceCore != 0U) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            }
            CopySourceIn(sourceCore, tileStart, valid);
            MergeSource(valid);
        }
        EncodeCurrent(valid);
        CopyOut(tileStart, valid);
    }

    AscendC::TPipe pipe_;
    const RasterizerTilingData* tiling_ = nullptr;
    bool initialized_ = false;
    AscendC::GlobalTensor<float> vGm_;
    AscendC::GlobalTensor<int32_t> fGm_;
    AscendC::GlobalTensor<int32_t> findicesGm_;
    AscendC::GlobalTensor<float> barycentricGm_;
    AscendC::GlobalTensor<float> workspaceF_;
    AscendC::GlobalTensor<int32_t> workspaceI_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> faceQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestQBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestFaceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestB0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestB1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bestB2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> srcQBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> srcFaceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> srcB0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> srcB1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> srcB2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> faceValidityBuf_;
    uint32_t tileLen_ = 0;
};

} // namespace RasterizerKernel

#endif // RASTERIZER_KERNEL_H
