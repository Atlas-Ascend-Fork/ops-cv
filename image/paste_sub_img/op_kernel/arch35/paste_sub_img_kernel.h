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
 * \file paste_sub_img_kernel.h
 * \brief Kernel implementation for paste_sub_img operator on arch35
 */
#ifndef PASTE_SUB_IMG_KERNEL_H_
#define PASTE_SUB_IMG_KERNEL_H_

#include "kernel_operator.h"
#include "paste_sub_img_tiling_data.h"
#include "paste_sub_img_tiling_key.h"

constexpr int64_t kAxisCount = 3;

static constexpr AscendC::Reg::CastTrait kCastU8ToHalf = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_NONE};

static constexpr AscendC::Reg::CastTrait kCastHalfToU8Sat = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                             AscendC::Reg::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::CAST_RINT};

__simd_vf__ inline void SaturatedAddU8VF(__ubuf__ uint8_t* dst, __ubuf__ uint8_t* src1, __ubuf__ uint8_t* src2,
                                         int64_t count)
{
    constexpr uint32_t VL_H = AscendC::GetVecLen() / sizeof(half);
    uint16_t repeatTime = static_cast<uint16_t>((static_cast<uint64_t>(count) + VL_H - 1) / VL_H);
    uint32_t remaining = static_cast<uint32_t>(count);

    AscendC::Reg::RegTensor<uint8_t> u8Reg1, u8Reg2, u8OutReg;
    AscendC::Reg::RegTensor<half> f16Reg1, f16Reg2;
    AscendC::Reg::MaskReg mask;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL_H);
        mask = AscendC::Reg::UpdateMask<half>(remaining);

        AscendC::Reg::LoadAlign<uint8_t, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(u8Reg1, src1 + off);
        AscendC::Reg::LoadAlign<uint8_t, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(u8Reg2, src2 + off);

        AscendC::Reg::Cast<half, uint8_t, kCastU8ToHalf>(f16Reg1, u8Reg1, mask);
        AscendC::Reg::Cast<half, uint8_t, kCastU8ToHalf>(f16Reg2, u8Reg2, mask);

        AscendC::Reg::Add<half>(f16Reg1, f16Reg1, f16Reg2, mask);

        AscendC::Reg::Cast<uint8_t, half, kCastHalfToU8Sat>(u8OutReg, f16Reg1, mask);

        AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::StoreDist::DIST_PACK_B16>(dst + off, u8OutReg, mask);
    }
}

struct AxisLayout {
    int64_t outStart;
    int64_t length;
    int64_t inStride;
    int64_t outStride;
};

struct Layout {
    AxisLayout axes[kAxisCount];
    int64_t inOffset;
    int64_t outOffset;
};

template <typename T, int KEY>
class PasteSubImgKernel {
    AscendC::TPipe pipe_;
    const PasteSubImgTilingData* td_;
    AscendC::GlobalTensor<T> gmPatch_;
    AscendC::GlobalTensor<T> gmCombine_;
    AscendC::GlobalTensor<T> gmCombineOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufPatch_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufCombine_;
    Layout layout_;
    int64_t perBufElems_;

public:
    __aicore__ inline void Init(GM_ADDR patch_img, GM_ADDR combine_img, GM_ADDR combine_img_out,
                                const PasteSubImgTilingData* td);
    __aicore__ inline void Process();
};

template <typename T, int KEY>
__aicore__ inline void PasteSubImgKernel<T, KEY>::Init(GM_ADDR patch_img, GM_ADDR combine_img, GM_ADDR combine_img_out,
                                                       const PasteSubImgTilingData* td)
{
    td_ = td;
    gmPatch_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(patch_img));
    gmCombine_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(combine_img));
    gmCombineOut_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(combine_img_out));

    pipe_.InitBuffer(bufPatch_, td_->bufferSize * 2);
    pipe_.InitBuffer(bufCombine_, td_->bufferSize * 2);

    perBufElems_ = td_->bufferSize / static_cast<int64_t>(sizeof(T));

    layout_.axes[0].inStride = td_->patchStrideH;
    layout_.axes[0].outStride = td_->combineStrideH;
    layout_.axes[1].inStride = td_->patchStrideW;
    layout_.axes[1].outStride = td_->combineStrideW;
    layout_.axes[2].inStride = 1;
    layout_.axes[2].outStride = 1;
}

template <typename T, int KEY>
__aicore__ inline void PasteSubImgKernel<T, KEY>::Process()
{
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t start = blockIdx * static_cast<int64_t>(td_->perCoreCount);
    int64_t end = start + static_cast<int64_t>(td_->perCoreCount);
    if (end > static_cast<int64_t>(td_->totalCount)) {
        end = static_cast<int64_t>(td_->totalCount);
    }

    auto calcLen = [](int64_t dimSize, int64_t ubF, int64_t oIdx) -> int64_t {
        int64_t remain = dimSize - oIdx * ubF;
        return (remain >= ubF) ? ubF : remain;
    };

    auto ceilDiv = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

    for (int64_t tileIdx = start; tileIdx < end; ++tileIdx) {
        int64_t ubOff = (tileIdx & 1) * perBufElems_;
        int64_t ubFactor = static_cast<int64_t>(td_->ubFactor);

        if constexpr (KEY == 1) {
            // split-C: tileIdx = ((h * activeW) + w) * cChunkCount + c_o
            int64_t cChunkCount = ceilDiv(td_->activeC, ubFactor);
            int64_t c_o = tileIdx % cChunkCount;
            int64_t wIdx = (tileIdx / cChunkCount) % td_->activeW;
            int64_t hIdx = tileIdx / (cChunkCount * td_->activeW);
            layout_.axes[0].outStart = hIdx;
            layout_.axes[0].length = 1;
            layout_.axes[1].outStart = wIdx;
            layout_.axes[1].length = 1;
            layout_.axes[2].outStart = c_o * ubFactor;
            layout_.axes[2].length = calcLen(td_->activeC, ubFactor, c_o);
        } else if constexpr (KEY == 2) {
            // split-W: tileIdx = h * wChunkCount + w_o
            int64_t wChunkCount = ceilDiv(td_->activeW, ubFactor);
            int64_t w_o = tileIdx % wChunkCount;
            int64_t hIdx = tileIdx / wChunkCount;
            layout_.axes[0].outStart = hIdx;
            layout_.axes[0].length = 1;
            layout_.axes[1].outStart = w_o * ubFactor;
            layout_.axes[1].length = calcLen(td_->activeW, ubFactor, w_o);
            layout_.axes[2].outStart = 0;
            layout_.axes[2].length = td_->activeC;
        } else {
            // split-H (KEY == 3): tileIdx = h_o
            int64_t h_o = tileIdx;
            layout_.axes[0].outStart = h_o * ubFactor;
            layout_.axes[0].length = calcLen(td_->activeH, ubFactor, h_o);
            layout_.axes[1].outStart = 0;
            layout_.axes[1].length = td_->activeW;
            layout_.axes[2].outStart = 0;
            layout_.axes[2].length = td_->activeC;
        }

        layout_.inOffset = td_->patchBaseOffset + layout_.axes[0].outStart * td_->patchStrideH +
                           layout_.axes[1].outStart * td_->patchStrideW + layout_.axes[2].outStart;
        layout_.outOffset = td_->combineBaseOffset + layout_.axes[0].outStart * td_->combineStrideH +
                            layout_.axes[1].outStart * td_->combineStrideW + layout_.axes[2].outStart;

        if (tileIdx > (start + 1)) {
            if (tileIdx & 1) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::DataCopyPadExtParams<T> padParams;
        if constexpr (KEY == 3) {
            // split-H：逐行搬运（每行 = activeW × activeC 连续元素）
            int64_t curH = layout_.axes[0].length;
            int64_t rowElems = layout_.axes[1].length * layout_.axes[2].length;
            uint32_t rowBytes = static_cast<uint32_t>(rowElems * static_cast<int64_t>(sizeof(T)));
            int64_t rowAlignedBytes = static_cast<int64_t>(AscendC::AlignUp(rowBytes, AscendC::ONE_BLK_SIZE));
            int64_t rowUbElems = rowAlignedBytes / static_cast<int64_t>(sizeof(T));

            auto b0Base = bufPatch_.Get<T>(ubOff);
            auto b1Base = bufCombine_.Get<T>(ubOff);

            for (int64_t h = 0; h < curH; ++h) {
                uint32_t rowOff = static_cast<uint32_t>(h * rowUbElems);
                int64_t patchGmOff = layout_.inOffset + h * td_->patchStrideH;
                int64_t combineGmOff = layout_.outOffset + h * td_->combineStrideH;

                auto b0 = b0Base[rowOff];
                auto b1 = b1Base[rowOff];

                AscendC::DataCopyExtParams copyPatch;
                copyPatch.blockCount = 1;
                copyPatch.blockLen = rowBytes;
                copyPatch.srcStride = 0;
                copyPatch.dstStride = 0;
                AscendC::DataCopyPad(b0, gmPatch_[patchGmOff], copyPatch, padParams);

                AscendC::DataCopyExtParams copyCombine;
                copyCombine.blockCount = 1;
                copyCombine.blockLen = rowBytes;
                copyCombine.srcStride = 0;
                copyCombine.dstStride = 0;
                AscendC::DataCopyPad(b1, gmCombine_[combineGmOff], copyCombine, padParams);
            }
        } else {
            // split-C (KEY == 1) / split-W (KEY == 2)：单块连续搬运
            int64_t elemCount = layout_.axes[0].length * layout_.axes[1].length * layout_.axes[2].length;
            uint32_t blockLen = static_cast<uint32_t>(elemCount * static_cast<int64_t>(sizeof(T)));
            auto b0 = bufPatch_.Get<T>(ubOff);
            auto b1 = bufCombine_.Get<T>(ubOff);

            AscendC::DataCopyExtParams copyPatch;
            copyPatch.blockCount = 1;
            copyPatch.blockLen = blockLen;
            copyPatch.srcStride = 0;
            copyPatch.dstStride = 0;
            AscendC::DataCopyPad(b0, gmPatch_[layout_.inOffset], copyPatch, padParams);

            AscendC::DataCopyExtParams copyCombine;
            copyCombine.blockCount = 1;
            copyCombine.blockLen = blockLen;
            copyCombine.srcStride = 0;
            copyCombine.dstStride = 0;
            AscendC::DataCopyPad(b1, gmCombine_[layout_.outOffset], copyCombine, padParams);
        }

        if (tileIdx & 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        if constexpr (KEY == 3) {
            int64_t curH = layout_.axes[0].length;
            int64_t rowElems = layout_.axes[1].length * layout_.axes[2].length;
            uint32_t rowBytes = static_cast<uint32_t>(rowElems * static_cast<int64_t>(sizeof(T)));
            int64_t rowAlignedBytes = static_cast<int64_t>(AscendC::AlignUp(rowBytes, AscendC::ONE_BLK_SIZE));
            int64_t rowUbElems = rowAlignedBytes / static_cast<int64_t>(sizeof(T));
            auto b0Base = bufPatch_.Get<T>(ubOff);
            auto b1Base = bufCombine_.Get<T>(ubOff);
            for (int64_t h = 0; h < curH; ++h) {
                uint32_t rowOff = static_cast<uint32_t>(h * rowUbElems);
                if constexpr (std::is_same_v<T, uint8_t>) {
                    asc_vf_call<SaturatedAddU8VF>((__ubuf__ uint8_t*)b1Base.GetPhyAddr(rowOff),
                                                  (__ubuf__ uint8_t*)b0Base.GetPhyAddr(rowOff),
                                                  (__ubuf__ uint8_t*)b1Base.GetPhyAddr(rowOff), rowElems);
                } else {
                    auto b0Row = b0Base[rowOff];
                    auto b1Row = b1Base[rowOff];
                    AscendC::Add(b1Row, b1Row, b0Row, static_cast<int32_t>(rowElems));
                }
            }
        } else {
            auto b0Patch = bufPatch_.Get<T>(ubOff);
            auto b1Combine = bufCombine_.Get<T>(ubOff);
            int64_t elemCount = layout_.axes[0].length * layout_.axes[1].length * layout_.axes[2].length;
            if constexpr (std::is_same_v<T, uint8_t>) {
                asc_vf_call<SaturatedAddU8VF>((__ubuf__ uint8_t*)b1Combine.GetPhyAddr(),
                                              (__ubuf__ uint8_t*)b0Patch.GetPhyAddr(),
                                              (__ubuf__ uint8_t*)b1Combine.GetPhyAddr(), elemCount);
            } else {
                AscendC::Add(b1Combine, b1Combine, b0Patch, static_cast<int32_t>(elemCount));
            }
        }

        if (tileIdx & 1) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        }

        if constexpr (KEY == 3) {
            int64_t curH = layout_.axes[0].length;
            int64_t rowElems = layout_.axes[1].length * layout_.axes[2].length;
            uint32_t rowBytes = static_cast<uint32_t>(rowElems * static_cast<int64_t>(sizeof(T)));
            int64_t rowAlignedBytes = static_cast<int64_t>(AscendC::AlignUp(rowBytes, AscendC::ONE_BLK_SIZE));
            int64_t rowUbElems = rowAlignedBytes / static_cast<int64_t>(sizeof(T));

            auto b1Base = bufCombine_.Get<T>(ubOff);
            for (int64_t h = 0; h < curH; ++h) {
                uint32_t rowOff = static_cast<uint32_t>(h * rowUbElems);
                int64_t combineGmOff = layout_.outOffset + h * td_->combineStrideH;
                auto b1 = b1Base[rowOff];

                AscendC::DataCopyExtParams out;
                out.blockCount = 1;
                out.blockLen = rowBytes;
                out.srcStride = 0;
                out.dstStride = 0;
                AscendC::DataCopyPad(gmCombineOut_[combineGmOff], b1, out);
            }
        } else {
            int64_t elemCount = layout_.axes[0].length * layout_.axes[1].length * layout_.axes[2].length;
            uint32_t blockLen = static_cast<uint32_t>(elemCount * static_cast<int64_t>(sizeof(T)));
            auto b1 = bufCombine_.Get<T>(ubOff);

            AscendC::DataCopyExtParams out;
            out.blockCount = 1;
            out.blockLen = blockLen;
            out.srcStride = 0;
            out.dstStride = 0;
            AscendC::DataCopyPad(gmCombineOut_[layout_.outOffset], b1, out);
        }

        if (tileIdx & 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}

#endif // PASTE_SUB_IMG_KERNEL_H_
