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
 * \file crop_tiling_data.h
 * \brief Tiling data struct for crop operator
 */

#ifndef CROP_TILING_DATA_H_
#define CROP_TILING_DATA_H_

#include <cstdint>

// Max supported rank (capacity of xStrides/yStrides)
static constexpr int32_t MAX_NDIM = 8;
// Max dims per NDDMA copy (device loopInfo limit)
static constexpr int32_t CROP_SS_NDDMA_DIMS = 5;

// Tiling data for the NDDMA row-partition path
struct CropSsBaseTilingData {
    int64_t blkFactor = 0;
    int64_t blkTailFactor = 0;
    int64_t ubInLoopSteps = 0;
    int64_t ubOutLoopSteps = 0;
    int32_t ubSize = 0;
    int32_t ubFactor = 0;
    int32_t ubTailFactor = 0;
    int32_t ubTailTailFactor = 0;
    int16_t realCoreNum = 0;
    int16_t inputDims = 0;
    int16_t blkIndex = 0;
    int16_t ubIndex = 0;
    int64_t outputShape[MAX_NDIM] = {};
    int64_t begin[MAX_NDIM] = {};      // Start coord of each effective dim (row units)
    int64_t strides[MAX_NDIM] = {};    // Row stride of each effective dim (always 1)
    int64_t inputSteps[MAX_NDIM] = {}; // Input span between adjacent effective dims
    int64_t rowsOffsetSteps[MAX_NDIM] = {};
};

struct CropSsNddmaTilingData {
    CropSsBaseTilingData base;
    int64_t nddmaTotalNum = 1;
    int64_t nddmaLoopSize[CROP_SS_NDDMA_DIMS] = {};
    int64_t nddmaLoopSrcStride[CROP_SS_NDDMA_DIMS] = {};
    int64_t nddmaLoopDstStride[CROP_SS_NDDMA_DIMS] = {};
};
// Max dims per NDDMA copy (hardware loopInfo limit)
static constexpr int32_t SIMD_MAX_DIMS = 5;
// Max outer loop dims on the SIMD path
static constexpr int32_t SIMD_MAX_OUTER_DIMS = 8;

struct CropTilingData {
    int64_t totalElements = 0;
    int32_t rank = 0;
    int64_t xStrides[MAX_NDIM] = {};
    int64_t yStrides[MAX_NDIM] = {};
    int64_t mainBlockFactor = 0;
    int64_t tailBlockFactor = 0;
    // Precomputed offsets; the kernel shifts the input pointer
    int64_t baseOffset = 0;
    int64_t xTotalElements = 0;

    // SIMD (multi-dim NDDMA copy) path
    uint32_t useSimd = 0;
    uint32_t simdCoreNum = 0;
    uint32_t simdUbBufferBytes = 0;
    // One NDDMA copy dims (inner -> outer): size in elements, srcStride in elements, compact dstStride.
    uint32_t copyDimNum = 0;
    uint32_t copySize[SIMD_MAX_DIMS] = {};
    uint64_t copySrcStride[SIMD_MAX_DIMS] = {};
    uint32_t copyDstStride[SIMD_MAX_DIMS] = {};
    // Outer loop dims (outermost -> innermost) with element strides.
    uint32_t outerDimNum = 0;
    int64_t outerSize[SIMD_MAX_OUTER_DIMS] = {};
    int64_t outerSrcStride[SIMD_MAX_OUTER_DIMS] = {};
    // Elements per iteration (first chunk size when chunked)
    int64_t baseElems = 0;
    // >0: copy dim chunked by UB budget; chunkDim is the chunked dim (0=innermost), chunkElems per chunk.
    uint32_t chunkDim = 0;
    int64_t chunkElems = 0;
    // Total length of the chunked dim, used for per-outer-row output offsets
    int64_t chunkDimSize = 0;
    int64_t numChunks = 1;
    int64_t totalIters = 0;
    // >0: row partition mode (contiguous row range per core, chunked by chunkElems)
    int64_t rowTotal = 0;
    // >0: the last chunk of each row is a tail with tailElems elements
    int64_t tailElems = 0;

    // Sparse gather path (vector Reg::Gather compaction)
    uint32_t useGather = 0;
    uint32_t gDimNum = 0;
    int64_t gOuterBlocks = 0;    // Number of outer blocks
    int64_t gBlockSrcStride = 0; // Input stride between blocks (elements)
    int64_t gSpanElems = 0;      // Elements loaded per block (full span)
    int64_t gBoxElems = 0;       // Output elements per block
    int64_t gBlocksPerIter = 0;  // Blocks per iteration
    int64_t gOutSize[4] = {};    // Box dims (outer -> inner)
    int64_t gUbStride[4] = {};   // Box dim input strides within a span (elements)
    int64_t gUbBlockStride = 0;  // Elements per block span
    uint32_t gBufInBytes = 0;
    uint32_t gBufOutBytes = 0;

    // NDDMA row-partition path
    uint32_t useSsNddma = 0;
    CropSsNddmaTilingData ssNddma;
};

#endif // CROP_TILING_DATA_H_
