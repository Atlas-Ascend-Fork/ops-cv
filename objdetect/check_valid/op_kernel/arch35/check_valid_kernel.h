/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECK_VALID_KERNEL_H
#define CHECK_VALID_KERNEL_H

#include "kernel_operator.h"
#include "check_valid_tiling_struct.h"

constexpr int64_t kPhysNodes = 3;
constexpr int32_t kBboxCols = 4;
constexpr uint32_t kMetasElems = 3;

// Cast traits for fp16鈫攆p32 conversions inside VF (dav_3510)
constexpr AscendC::Reg::CastTrait kCastB16ToB32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                   AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait kCastB32ToB16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                   AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <typename T>
__simd_vf__ inline void CheckValidVF(__ubuf__ T* dst, __ubuf__ T* src, float img_width_x, float img_height_y,
                                     uint32_t totalElems, uint16_t repeatTime, uint32_t tileN)
{
    constexpr uint32_t VL_F = AscendC::GetVecLen() / sizeof(float); // 64

    float imgW = img_width_x;
    float imgH = img_height_y;

    __ubuf__ T* x0Ptr = src;
    __ubuf__ T* y0Ptr = src + tileN;
    __ubuf__ T* x1Ptr = src + 2 * tileN;
    __ubuf__ T* y1Ptr = src + 3 * tileN;

    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL_F);
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::RegTensor<float> x0Reg, y0Reg, x1Reg, y1Reg;

        if constexpr (std::is_same_v<T, half>) {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                (AscendC::Reg::RegTensor<half>&)x0Reg, x0Ptr + off);
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                (AscendC::Reg::RegTensor<half>&)y0Reg, y0Ptr + off);
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                (AscendC::Reg::RegTensor<half>&)x1Reg, x1Ptr + off);
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                (AscendC::Reg::RegTensor<half>&)y1Reg, y1Ptr + off);
            AscendC::Reg::Cast<float, half, kCastB16ToB32>(x0Reg, (AscendC::Reg::RegTensor<half>&)x0Reg, mask);
            AscendC::Reg::Cast<float, half, kCastB16ToB32>(y0Reg, (AscendC::Reg::RegTensor<half>&)y0Reg, mask);
            AscendC::Reg::Cast<float, half, kCastB16ToB32>(x1Reg, (AscendC::Reg::RegTensor<half>&)x1Reg, mask);
            AscendC::Reg::Cast<float, half, kCastB16ToB32>(y1Reg, (AscendC::Reg::RegTensor<half>&)y1Reg, mask);
        } else {
            AscendC::Reg::LoadAlign<float>(x0Reg, x0Ptr + off);
            AscendC::Reg::LoadAlign<float>(y0Reg, y0Ptr + off);
            AscendC::Reg::LoadAlign<float>(x1Reg, x1Ptr + off);
            AscendC::Reg::LoadAlign<float>(y1Reg, y1Ptr + off);
        }

        AscendC::Reg::RegTensor<float> zeroReg;
        AscendC::Reg::Duplicate<float>(zeroReg, 0.0f);

        AscendC::Reg::MaskReg c1;
        AscendC::Reg::Compare<float, AscendC::CMPMODE::GE>(c1, x0Reg, zeroReg, mask);
        AscendC::Reg::MaskReg c2;
        AscendC::Reg::Compare<float, AscendC::CMPMODE::GE>(c2, y0Reg, zeroReg, mask);
        AscendC::Reg::MaskReg m12;
        AscendC::Reg::And(m12, c1, c2, mask);

        AscendC::Reg::RegTensor<float> imgWReg;
        AscendC::Reg::Duplicate<float>(imgWReg, imgW);
        AscendC::Reg::MaskReg c3;
        AscendC::Reg::Compare<float, AscendC::CMPMODE::LE>(c3, x1Reg, imgWReg, mask);

        AscendC::Reg::RegTensor<float> imgHReg;
        AscendC::Reg::Duplicate<float>(imgHReg, imgH);
        AscendC::Reg::MaskReg c4;
        AscendC::Reg::Compare<float, AscendC::CMPMODE::LE>(c4, y1Reg, imgHReg, mask);

        AscendC::Reg::MaskReg m34;
        AscendC::Reg::And(m34, c3, c4, mask);
        AscendC::Reg::MaskReg valid;
        AscendC::Reg::And(valid, m12, m34, mask);

        AscendC::Reg::RegTensor<float> oneReg, zeroValReg;
        AscendC::Reg::Duplicate<float>(oneReg, 1.0f);
        AscendC::Reg::Duplicate<float>(zeroValReg, 0.0f);
        AscendC::Reg::RegTensor<float> resultReg;
        AscendC::Reg::Select<float>(resultReg, oneReg, zeroValReg, valid);

        if constexpr (std::is_same_v<T, half>) {
            AscendC::Reg::Cast<half, float, kCastB32ToB16>((AscendC::Reg::RegTensor<half>&)resultReg, resultReg, mask);
            AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                dst + off, (AscendC::Reg::RegTensor<half>&)resultReg, mask);
        } else {
            AscendC::Reg::StoreAlign<float>(dst + off, resultReg, mask);
        }
    }
}

template <typename T, bool ISEMPTY>
class CheckValidKernel {
public:
    __aicore__ inline CheckValidKernel() {}

    __aicore__ inline void Init(GM_ADDR bbox_tensor, GM_ADDR img_metas, GM_ADDR valid_tensor, GM_ADDR tiling,
                                const CheckValidTilingData* td)
    {
        td_ = td;
        gmBbox_.SetGlobalBuffer((__gm__ T*)bbox_tensor);
        gmOut_.SetGlobalBuffer((__gm__ int8_t*)valid_tensor);
        gmMetas_.SetGlobalBuffer((__gm__ T*)img_metas);
        if constexpr (ISEMPTY) {
            return;
        }
        for (int64_t i = 0; i < kPhysNodes; i++) {
            pipe_.InitBuffer(buf_[i], static_cast<uint32_t>(td->per_buf_bytes));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (ISEMPTY) {
            return;
        }

        float Hf, Wf, rf;
        if constexpr (std::is_same_v<T, half>) {
            Hf = (float)gmMetas_.GetValue(0);
            Wf = (float)gmMetas_.GetValue(1);
            rf = (float)gmMetas_.GetValue(2);
        } else {
            Hf = gmMetas_.GetValue(0);
            Wf = gmMetas_.GetValue(1);
            rf = gmMetas_.GetValue(2);
        }
        volatile float scaledW = Wf * rf;
        volatile float scaledH = Hf * rf;
        imgWf_ = scaledW - 1.0f;
        imgHf_ = scaledH - 1.0f;

        const int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t start, end;
        GetCoreRange(blockIdx, td_->num_cores, td_->tiles_main, td_->cores_tail, start, end);

        for (int64_t tileIdx = start; tileIdx < end; ++tileIdx) {
            bool isTail = (tileIdx == td_->num_tiles - 1);
            int64_t rowCount = isTail ? td_->tile_n_tail : td_->tile_n;
            int64_t rowBegin = tileIdx * td_->tile_n;
            constexpr uint8_t eid = 0;

            CopyInTile(rowBegin, rowCount);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eid);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eid);
            ComputeTile(rowCount);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eid);
            // B0 is also FP32 conversion scratch; release it before the next input DMA.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eid);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eid);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eid);
            CopyOutTile(rowBegin, rowCount);
            // B2 has one physical allocation; finish reading it before the next vector write.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eid);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eid);
        }
    }

private:
    __aicore__ inline void GetCoreRange(int64_t blockIdx, int64_t numCores, int64_t tilesMain, int64_t coresTail,
                                        int64_t& start, int64_t& end) const
    {
        int64_t tilesThisCore = tilesMain + ((blockIdx < coresTail) ? 1 : 0);
        if (blockIdx < coresTail) {
            start = blockIdx * (tilesMain + 1);
        } else {
            start = coresTail * (tilesMain + 1) + (blockIdx - coresTail) * tilesMain;
        }
        end = start + tilesThisCore;
    }

    __aicore__ inline void CopyInTile(int64_t rowBegin, int64_t rowCount)
    {
        AscendC::DataCopyExtParams params(static_cast<uint16_t>(rowCount), static_cast<uint32_t>(sizeof(T)),
                                          static_cast<int64_t>((kBboxCols - 1) * sizeof(T)), 0, 0);
        AscendC::DataCopyPadExtParams<T> pad(false, 0, 0, 0);
        const uint32_t cap = static_cast<uint32_t>(td_->tile_n);
        for (int32_t c = 0; c < kBboxCols; ++c) {
            const uint64_t bboxOffset = static_cast<uint64_t>(rowBegin) * kBboxCols + static_cast<uint64_t>(c);
            AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(buf_[0].Get<T>()[static_cast<uint32_t>(c) * cap],
                                                                   gmBbox_[bboxOffset], params, pad);
        }
    }

    __aicore__ inline void ComputeTile(int64_t rowCount)
    {
        constexpr uint32_t VL_F = AscendC::GetVecLen() / sizeof(float); // 64
        uint32_t totalElems = static_cast<uint32_t>(rowCount);
        uint16_t repeatTime = static_cast<uint16_t>((totalElems + VL_F - 1) / VL_F);
        uint32_t tileN = static_cast<uint32_t>(td_->tile_n);

        float imgWf = imgWf_;
        float imgHf = imgHf_;

        __ubuf__ T* b0Ptr = (__ubuf__ T*)buf_[0].Get<T>().GetPhyAddr();
        __ubuf__ T* b1Ptr = (__ubuf__ T*)buf_[1].Get<T>().GetPhyAddr();

        asc_vf_call<CheckValidVF<T>>(b1Ptr, b0Ptr, imgWf, imgHf, totalElems, repeatTime, tileN);

        int32_t count = static_cast<int32_t>(rowCount);
        if constexpr (std::is_same_v<T, half>) {
            AscendC::Cast<int8_t, half>(buf_[2].Get<int8_t>(), buf_[1].Get<half>(), AscendC::RoundMode::CAST_RINT,
                                        count);
        } else {
            AscendC::Cast<half, float>(buf_[0].Get<half>(), buf_[1].Get<float>(), AscendC::RoundMode::CAST_RINT, count);
            AscendC::Cast<int8_t, half>(buf_[2].Get<int8_t>(), buf_[0].Get<half>(), AscendC::RoundMode::CAST_RINT,
                                        count);
        }
    }

    __aicore__ inline void CopyOutTile(int64_t rowBegin, int64_t rowCount)
    {
        AscendC::DataCopyExtParams params(1, static_cast<uint32_t>(rowCount * sizeof(int8_t)), 0, 0, 0);
        AscendC::DataCopyPad<int8_t, AscendC::PaddingMode::Compact>(gmOut_[static_cast<uint64_t>(rowBegin)],
                                                                    buf_[2].Get<int8_t>(), params);
    }

    AscendC::TPipe pipe_;
    const CheckValidTilingData* td_;
    AscendC::GlobalTensor<T> gmBbox_;
    AscendC::GlobalTensor<T> gmMetas_;
    AscendC::GlobalTensor<int8_t> gmOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];
    float imgWf_ = 0.0f;
    float imgHf_ = 0.0f;
};

#endif // CHECK_VALID_KERNEL_H
