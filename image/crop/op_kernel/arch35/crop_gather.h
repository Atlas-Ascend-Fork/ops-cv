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
 * \file crop_gather.h
 * \brief Sparse gather template for repeated outer blocks with narrow inner runs.
 *
 * MTE2 batch-loads each block's contiguous span, the vector unit compacts it with
 * Reg::Gather, and MTE3 writes the result contiguously (double buffered).
 * Enabled by tiling when the inner effective dims form a compact box with a
 * contiguous span and the remaining outer dims collapse into one.
 */

#ifndef CROP_GATHER_H_
#define CROP_GATHER_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "crop_tiling_data.h"

namespace NsCrop {
using namespace AscendC;

constexpr int32_t GATHER_BUFFER_NUM = 2;
constexpr uint32_t GATHER_VREG_BYTES = 256U;

// Vector index: output index j -> block id + in-box coords -> input span offset.
template <typename T, int32_t DIMS>
__aicore__ inline void GatherBoxN(const CropTilingData* tilingData, __ubuf__ T* inAddr, __ubuf__ T* outAddr,
                                  uint32_t totalElems)
{
    using RT = int32_t;
    using IT = uint32_t;
    constexpr uint16_t VL = static_cast<uint16_t>(GATHER_VREG_BYTES / sizeof(T));
    uint16_t loopCnt = static_cast<uint16_t>((totalElems + VL - 1) / VL);
    RT box = static_cast<RT>(tilingData->gBoxElems);
    RT ubBlockStride = static_cast<RT>(tilingData->gUbBlockStride);
    RT div0 = static_cast<RT>(tilingData->gOutSize[1]);
    RT div1 = static_cast<RT>(tilingData->gOutSize[2]);
    if constexpr (DIMS == 3) {
        div0 *= static_cast<RT>(tilingData->gOutSize[2]);
    }
    RT st0 = static_cast<RT>(tilingData->gUbStride[0]);
    RT st1 = static_cast<RT>(tilingData->gUbStride[1]);
    RT stLast = static_cast<RT>((DIMS == 3) ? tilingData->gUbStride[2] : tilingData->gUbStride[1]);

    __VEC_SCOPE__
    {
        Reg::RegTensor<RT> idxReg;
        Reg::RegTensor<RT> blkReg;
        Reg::RegTensor<RT> remReg;
        Reg::RegTensor<RT> coordReg;
        Reg::RegTensor<RT> tmpReg;
        Reg::RegTensor<T> dataReg;
        Reg::MaskReg mask;

        for (uint16_t i = 0; i < loopCnt; i++) {
            uint32_t off = static_cast<uint32_t>(i) * VL;
            uint32_t cnt = totalElems - off;
            cnt = (cnt < VL) ? cnt : VL;
            mask = Reg::UpdateMask<RT>(cnt);
            Reg::Arange(idxReg, static_cast<RT>(off));
            // blk = j / box, rem = j % box
            Reg::Duplicate(tmpReg, box, mask);
            Reg::Div(blkReg, idxReg, tmpReg, mask);
            Reg::Muls(tmpReg, blkReg, box, mask);
            Reg::Sub(remReg, idxReg, tmpReg, mask);
            // idx = blk*span + sum(coord_d * stride_d), unrolled over DIMS
            Reg::Muls(idxReg, blkReg, ubBlockStride, mask);
            if constexpr (DIMS >= 2) {
                Reg::Duplicate(tmpReg, div0, mask);
                Reg::Div(coordReg, remReg, tmpReg, mask);
                Reg::Muls(tmpReg, coordReg, div0, mask);
                Reg::Sub(remReg, remReg, tmpReg, mask);
                Reg::Muls(tmpReg, coordReg, st0, mask);
                Reg::Add(idxReg, idxReg, tmpReg, mask);
            }
            if constexpr (DIMS == 3) {
                Reg::Duplicate(tmpReg, div1, mask);
                Reg::Div(coordReg, remReg, tmpReg, mask);
                Reg::Muls(tmpReg, coordReg, div1, mask);
                Reg::Sub(remReg, remReg, tmpReg, mask);
                Reg::Muls(tmpReg, coordReg, st1, mask);
                Reg::Add(idxReg, idxReg, tmpReg, mask);
            }
            Reg::Muls(tmpReg, remReg, stLast, mask);
            Reg::Add(idxReg, idxReg, tmpReg, mask);

            Reg::Gather(dataReg, inAddr, (Reg::RegTensor<IT>&)idxReg, mask);
            Reg::StoreAlign(outAddr + off, dataReg, mask);
        }
    }
}

template <typename T>
__aicore__ inline void CropGatherProcess(const CropTilingData* tilingData, GM_ADDR x, GM_ADDR y)
{
    if (tilingData->totalIters <= 0 || tilingData->simdCoreNum == 0) {
        return;
    }
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(tilingData->simdCoreNum)) {
        return;
    }

    int64_t coreNum = static_cast<int64_t>(tilingData->simdCoreNum);
    int64_t iters = tilingData->totalIters;
    int64_t perCore = iters / coreNum;
    int64_t rem = iters % coreNum;
    int64_t itBegin = perCore * blockIdx + (blockIdx < rem ? blockIdx : rem);
    int64_t itCount = perCore + (blockIdx < rem ? 1 : 0);
    if (itCount <= 0) {
        return;
    }

    TPipe pipe;
    TQue<TPosition::VECIN, GATHER_BUFFER_NUM> inQue;
    TQue<TPosition::VECOUT, GATHER_BUFFER_NUM> outQue;
    pipe.InitBuffer(inQue, GATHER_BUFFER_NUM, tilingData->gBufInBytes);
    pipe.InitBuffer(outQue, GATHER_BUFFER_NUM, tilingData->gBufOutBytes);

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;
    inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + tilingData->baseOffset);
    outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    int64_t blocksPerIter = tilingData->gBlocksPerIter;
    int64_t span = tilingData->gSpanElems;
    int64_t box = tilingData->gBoxElems;
    int64_t blockStride = tilingData->gBlockSrcStride;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (int64_t k = 0; k < itCount; k++) {
        int64_t iter = itBegin + k;
        int64_t blkStart = iter * blocksPerIter;
        int64_t curBlocks = tilingData->gOuterBlocks - blkStart;
        curBlocks = (curBlocks < blocksPerIter) ? curBlocks : blocksPerIter;
        if (curBlocks <= 0) {
            break;
        }

        LocalTensor<T> inLocal = inQue.AllocTensor<T>();
        DataCopyExtParams inParams;
        inParams.blockCount = static_cast<uint16_t>(curBlocks);
        inParams.blockLen = static_cast<uint32_t>(span * sizeof(T));
        inParams.srcStride = (blockStride - span) * static_cast<int64_t>(sizeof(T));
        inParams.dstStride = 0;
        DataCopyPad<T, PaddingMode::Compact>(inLocal, inputGM[blkStart * blockStride], inParams, padParams);
        inQue.EnQue(inLocal);
        inLocal = inQue.DeQue<T>();

        LocalTensor<T> outLocal = outQue.AllocTensor<T>();
        __ubuf__ T* inAddr = reinterpret_cast<__ubuf__ T*>(inLocal.GetPhyAddr());
        __ubuf__ T* outAddr = reinterpret_cast<__ubuf__ T*>(outLocal.GetPhyAddr());
        uint32_t totalElems = static_cast<uint32_t>(curBlocks * box);
        if constexpr (sizeof(T) == 4) {
            if (tilingData->gDimNum == 3) {
                GatherBoxN<T, 3>(tilingData, inAddr, outAddr, totalElems);
            } else if (tilingData->gDimNum == 2) {
                GatherBoxN<T, 2>(tilingData, inAddr, outAddr, totalElems);
            }
        }
        outQue.EnQue(outLocal);
        outLocal = outQue.DeQue<T>();

        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(curBlocks * box * sizeof(T));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(outputGM[blkStart * box], outLocal, outParams);

        outQue.FreeTensor(outLocal);
        inQue.FreeTensor(inLocal);
    }
}
} // namespace NsCrop
#endif // CROP_GATHER_H_
