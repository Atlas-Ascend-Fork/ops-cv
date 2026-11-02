/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flood_fill.h
 * \brief FloodFill Ascend C kernel (span scanline fill).
 *
 *   1. Prove whole-image reachability from dtype bounds or extrema.
 *   2. Otherwise initialize visited state. A row's sole interval is exact;
 *      a second interval materializes both into the mask/workspace bitmap.
 *   3. Consume an append-order table of 8B {row, left, right, pad}
 *      spans, probing both neighboring rows and extending accepted
 *      pixels horizontally. Table overflow uses a full-image closure.
 *   4. Materialize deferred rows for mask output (or before overflow closure).
 *      Sort completed spans and stream copied/filled segments to y; overflow
 *      uses the complete visited bitmap instead of the truncated table.
 *   Large images parallelize the independent extrema, initialization,
 *   mask materialization and output stages; core zero retains the exact
 *   span search. Cache publication precedes each all-core barrier.
 *   Tolerance is always evaluated on ORIGINAL x values, matching
 *   cv::floodFill semantics (NEIGHBOR: ref = original neighbor value;
 *   FIXED_RANGE: ref = original seed value).
 */

#ifndef FLOOD_FILL_H_
#define FLOOD_FILL_H_

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "flood_fill_tiling_data.h"
#include "flood_fill_tiling_key.h"

// Diagnostic builds only. Production builds compile out all timestamps
// and workspace writes. Enable explicitly before rebuilding for profiling.
#ifndef FLOOD_FILL_PROFILE
#define FLOOD_FILL_PROFILE 0
#endif

namespace NsFloodFill {

using namespace AscendC;

// Span stack entry (8B): a filled row span [left, right] on row y.
struct FillSpan {
    uint16_t y;
    uint16_t left;
    uint16_t right;
    uint16_t pad; // Bits 0/1: the already-filled parent covers the full upper/lower probe.
};

template <typename T, bool HAS_MASK_INPUT = false>
class KernelFloodFill {
public:
    __aicore__ inline KernelFloodFill() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR maskIn, GM_ADDR y, GM_ADDR maskOut, GM_ADDR workspace,
                                const FloodFillTilingData* tilingData, TPipe* pipe)
    {
        height = tilingData->height;
        width = tilingData->width;

        seedRow = tilingData->seedRow;
        seedCol = tilingData->seedCol;

        loDiffI = tilingData->loDiffI;
        upDiffI = tilingData->upDiffI;
        loDiffF = tilingData->loDiffF;
        upDiffF = tilingData->upDiffF;

        connectivity = tilingData->connectivity;
        diffMode = tilingData->diffMode;

        // Zero-tolerance edges connect equal original values. Their
        // transitive closure is therefore the seed's fixed-range region.
        // Keep subtraction-based checks for non-finite float seeds:
        // inf - inf and NaN differences still reject every neighbor.
        if constexpr (std::is_same<T, float>::value) {
            if (loDiffF == 0.0f && upDiffF == 0.0f) {
                diffMode = FLOODFILL_DIFF_MODE_FIXED_RANGE;
            }
        } else if (loDiffI == 0 && upDiffI == 0) {
            diffMode = FLOODFILL_DIFF_MODE_FIXED_RANGE;
        }

        if constexpr (std::is_same<T, float>::value) {
            newValueT = tilingData->newValueF;
        } else {
            newValueT = static_cast<T>(tilingData->newValueI);
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

        pipePtr = pipe;
#if FLOOD_FILL_PROFILE
        profileWorkspace = workspace;
#endif

        coreId = GetBlockIdx();
        coreCount = GetBlockNum();
        sharedBase = workspace;
        sharedControl.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(workspace));
        workspace += FLOOD_FILL_CONTROL_BYTES;

        // A 2112B control area (header plus 32 cache-line slots) precedes
        // the ordinary user workspace. The sys
        // area is accounted for on host side only (same convention as
        // upsample_bicubic2d_grad). No offset here.
        // Layout: [ visited bitmap (H*W, only when mask absent) |
        //           row flags (Align32(H)) | sole intervals (Align32(4*H)) | span stack ]
        const uint64_t totalPixels = static_cast<uint64_t>(height) * width;
        // Two optional immutable comparison planes, each with 64B padding.
        // The host reserves the identical layout for large UINT8/FLOAT32.
        if constexpr (!std::is_same<T, int32_t>::value) {
            if (totalPixels >= 262144U && width >= 512U) {
                edgePlaneBytes = ((totalPixels + 7U) / 8U + 63U) / 64U * 64U + 64U;
                edgeForwardGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(workspace));
                const bool shared = diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE ||
                                    (std::is_same<T, uint8_t>::value ? loDiffI == upDiffI : loDiffF == upDiffF);
                edgeReverseGm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ uint64_t*>(workspace + (shared ? 0U : edgePlaneBytes)));
                // In undirected subcases both read planes alias the first.
                // The unused second plane has at least 64 bytes per row
                // because width >= 512. Packed row records need only 8B;
                // work partitioning gives each core complete cache lines.
                rowSeedInfo.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(workspace + edgePlaneBytes));
                workspace += 2U * edgePlaneBytes;
            }
        }
        const uint64_t visitedAligned = (totalPixels + 31U) / 32U * 32U;
        const uint64_t rowFlagsAligned = (static_cast<uint64_t>(height) + 31U) / 32U * 32U;

        if constexpr (HAS_MASK_INPUT) {
            inputMaskGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(maskIn));
            inputMaskWords = reinterpret_cast<__gm__ uint64_t*>(maskIn);
            hasMaskOutput = maskOut != nullptr;
            if (hasMaskOutput)
                maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(maskOut));
        }
        // Input barriers are not visited pixels. Keep reachability in the
        // reserved workspace even when the output mask is bound.
        if (!HAS_MASK_INPUT && maskOut != nullptr) {
            maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(maskOut));
            hasMask = true;
            visitedWordBase = reinterpret_cast<__gm__ uint64_t*>(maskOut);
            rowFlagsBase = reinterpret_cast<__gm__ uint8_t*>(workspace);
        } else {
            visitedGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(workspace));
            visitedWordBase = reinterpret_cast<__gm__ uint64_t*>(workspace);
            rowFlagsBase = reinterpret_cast<__gm__ uint8_t*>(workspace) + visitedAligned;
        }

        rowIntervalsBase = reinterpret_cast<__gm__ uint32_t*>(rowFlagsBase + rowFlagsAligned);
        const uint64_t rowIntervalsAligned = (static_cast<uint64_t>(height) * 4U + 31U) / 32U * 32U;
        spanTable = reinterpret_cast<__gm__ FillSpan*>(rowFlagsBase + rowFlagsAligned + rowIntervalsAligned);

        const uint64_t maxDim = height > width ? static_cast<uint64_t>(height) : static_cast<uint64_t>(width);
        stackCapacity = static_cast<uint32_t>(2U * maxDim + 256U);

        // Degenerate images: skip all global-memory access.
        if (totalPixels == 0U) {
            return;
        }

        pipePtr->InitBuffer(copyQue, 2U, 16384U);
        pipePtr->InitBuffer(fillQue, 2U, 16384U);
        // UB scratch for the whole-image homogeneity scan (see
        // WholeImageIsOneRegion): two staging views for the uint8 cast
        // chain, two accumulators and two check copies.
        pipePtr->InitBuffer(scanStageHalf, 12288U);
        pipePtr->InitBuffer(scanStageFloat, 8192U);
        pipePtr->InitBuffer(scanAccMax, 16384U);
        pipePtr->InitBuffer(scanAccMin, 16384U);
        pipePtr->InitBuffer(scanCheckA, 8192U);
        pipePtr->InitBuffer(scanCheckB, 8192U);

        seedValue = xGm.GetValue(seedRow * width + seedCol);
        if constexpr (std::is_same<T, float>::value) {
            seedEquality = loDiffF == 0.0f && upDiffF == 0.0f && seedValue >= -3.402823466e+38F &&
                           seedValue <= 3.402823466e+38F;
        } else {
            seedEquality = loDiffI == 0 && upDiffI == 0;
        }
    }

    __aicore__ inline void Process()
    {
        ProcessImage();
        if constexpr (HAS_MASK_INPUT) {
            if (height != 0U && width != 0U && hasMaskOutput) {
                if (coreCount > 1U)
                    ShareBarrier();
                MergeInputMask();
            }
        }
    }

private:
    __aicore__ inline void FillLocalMaskRange(LocalTensor<uint8_t>& local, uint32_t first, uint32_t last)
    {
        while (first < last && first % 32U != 0U)
            local.SetValue(first++, 1U);
        const uint32_t bulk = (last - first) / 32U * 32U;
        if (bulk != 0U) {
            const TEventID before = pipePtr->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(before);
            WaitFlag<HardEvent::S_V>(before);
            LocalTensor<uint16_t> packed = local.template ReinterpretCast<uint16_t>();
            Duplicate(packed[first / 2U], uint16_t(257U), static_cast<int32_t>(bulk / 2U));
            const TEventID after = pipePtr->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(after);
            WaitFlag<HardEvent::V_S>(after);
            first += bulk;
        }
        while (first < last)
            local.SetValue(first++, 1U);
    }

    // Search only marks new reachability. Merge it with the immutable input
    // labels after search/output have finished. Each core owns complete 64B
    // lines, including mask rows that straddle a partition or DMA chunk.
    __aicore__ __attribute__((noinline)) void MergeInputMask()
    {
        const uint64_t maskWidth = width + 2U;
        const uint64_t bytes = static_cast<uint64_t>(height + 2U) * maskWidth;
        if (!overflowed && filledPixelCount == static_cast<uint64_t>(height) * width) {
            ParallelFillBytes(maskGm, bytes, 1U);
            return;
        }
        uint64_t first, last;
        Partition(bytes, 64U, first, last);
        uint32_t lo = 0U, hi = spanCount;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2U;
            if (static_cast<uint64_t>(MaskIndex(spanTable[mid].y, spanTable[mid].right)) + 1U <= first)
                lo = mid + 1U;
            else
                hi = mid;
        }
        uint32_t next = lo;
        for (uint64_t offset = first; offset < last; offset += 16384U) {
            const uint32_t count = static_cast<uint32_t>(last - offset < 16384U ? last - offset : 16384U);
            const uint64_t end = offset + count;
            LocalTensor<uint8_t> local = copyQue.AllocTensor<uint8_t>();
            DataCopyExtParams params{1, count, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padding{false, 0, 0, 0};
            DataCopyPad(local, inputMaskGm[offset], params, padding);
            const TEventID loaded = pipePtr->FetchEventID(HardEvent::MTE2_S);
            SetFlag<HardEvent::MTE2_S>(loaded);
            WaitFlag<HardEvent::MTE2_S>(loaded);
            if (overflowed) {
                for (uint32_t i = 0U; i < count; ++i) {
                    const uint32_t row = static_cast<uint32_t>((offset + i) / maskWidth);
                    const uint32_t col = static_cast<uint32_t>((offset + i) % maskWidth);
                    if (row > 0U && row <= height && col > 0U && col <= width && IsVisited(row - 1U, col - 1U))
                        local.SetValue(i, 1U);
                }
            } else {
                while (next < spanCount) {
                    const uint64_t begin = MaskIndex(spanTable[next].y, spanTable[next].left);
                    const uint64_t finish = static_cast<uint64_t>(MaskIndex(spanTable[next].y, spanTable[next].right)) +
                                            1U;
                    if (begin >= end)
                        break;
                    if (finish > offset) {
                        FillLocalMaskRange(local, static_cast<uint32_t>((begin > offset ? begin : offset) - offset),
                                           static_cast<uint32_t>((finish < end ? finish : end) - offset));
                    }
                    if (finish > end)
                        break;
                    ++next;
                }
            }
            for (uint64_t row = offset / maskWidth; row <= (end - 1U) / maskWidth; ++row) {
                const uint64_t begin = row * maskWidth, finish = begin + maskWidth;
                if (row == 0U || row == height + 1U) {
                    FillLocalMaskRange(local, static_cast<uint32_t>((begin > offset ? begin : offset) - offset),
                                       static_cast<uint32_t>((finish < end ? finish : end) - offset));
                } else {
                    if (begin >= offset && begin < end)
                        local.SetValue(begin - offset, 1U);
                    if (finish - 1U >= offset && finish - 1U < end)
                        local.SetValue(finish - 1U - offset, 1U);
                }
            }
            const TEventID patched = pipePtr->FetchEventID(HardEvent::S_MTE3);
            SetFlag<HardEvent::S_MTE3>(patched);
            WaitFlag<HardEvent::S_MTE3>(patched);
            DataCopyPad(maskGm[offset], local, params);
            const TEventID done = pipePtr->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(done);
            WaitFlag<HardEvent::MTE3_MTE2>(done);
            copyQue.FreeTensor(local);
        }
    }

    __aicore__ inline void ProcessImage()
    {
        if (height == 0 || width == 0) {
            return;
        }

        if (coreCount > 1U) {
            ProcessParallel();
            return;
        }

        // Whole-image pre-check: when the extrema of x prove that every
        // pixel is reachable from the seed, the span scan is skipped
        // altogether and y is written directly.
#if FLOOD_FILL_PROFILE
        const int64_t t0 = ProfileClock();
#endif
        const bool wholeRegion = !HAS_MASK_INPUT && (ToleranceCoversDomain() || WholeImageIsOneRegion());
#if FLOOD_FILL_PROFILE
        const int64_t t1 = ProfileClock();
#endif
        if (wholeRegion) {
            FillWholeImage();
#if FLOOD_FILL_PROFILE
            ProfileFinish(t0, t1, t1, t1, ProfileClock(), true);
#endif
            return;
        }

        InitVisited();
#if FLOOD_FILL_PROFILE
        const int64_t t2 = ProfileClock();
#endif
        RunSpanFill();
#if FLOOD_FILL_PROFILE
        const int64_t t3 = ProfileClock();
#endif

        const uint64_t totalPixels = static_cast<uint64_t>(height) * width;
        if (filledPixelCount == totalPixels && !overflowed) {
            // Fast path: every pixel is filled — skip the x→UB
            // copy and directly write newValue to y.
            CopyNewValue();
        } else {
            CopyImageSmart();
        }
#if FLOOD_FILL_PROFILE
        ProfileFinish(t0, t1, t2, t3, ProfileClock(), false);
#endif
    }

private:
    // All participating cores enter every barrier. Batch scheduling and
    // MIX_AIV_1_0 are required by the host and entrypoint respectively.
    // Flush scalar producers before the hardware barrier; readers start
    // without stale cache lines. DMA operations are drained by PIPE_ALL.
    __aicore__ inline void ShareBarrier()
    {
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE>(sharedControl);
        SyncAll();
    }

    __aicore__ inline void Partition(uint64_t total, uint32_t alignment, uint64_t& first, uint64_t& last) const
    {
        const uint64_t units = (total + alignment - 1U) / alignment;
        first = units * coreId / coreCount * alignment;
        last = units * (coreId + 1U) / coreCount * alignment;
        if (first > total)
            first = total;
        if (last > total)
            last = total;
    }

    __aicore__ inline void ParallelFillBytes(GlobalTensor<uint8_t>& dst, uint64_t total, uint8_t value)
    {
        uint64_t first, last;
        Partition(total, 64U, first, last);
        if (first == last)
            return;
        GlobalTensor<uint8_t> part = dst[first];
        FillGlobalBytes(part, last - first, value);
    }

    __aicore__ inline void ParallelMaskBorder()
    {
        const uint64_t maskWidth = width + 2U;
        const uint64_t bytes = static_cast<uint64_t>(height + 2U) * maskWidth;
        uint64_t first, last;
        Partition(bytes, 64U, first, last);
        if (first == last)
            return;
        // Top/bottom rows use DMA before any scalar writes. Each store
        // remains in this core's byte range, including a split mask row.
        for (uint32_t edge = 0U; edge < 2U; ++edge) {
            const uint64_t begin = edge == 0U ? 0U : bytes - maskWidth;
            const uint64_t end = begin + maskWidth;
            const uint64_t lo = begin > first ? begin : first;
            const uint64_t hi = end < last ? end : last;
            if (lo < hi) {
                GlobalTensor<uint8_t> part = maskGm[lo];
                FillGlobalBytes(part, hi - lo, 1U);
            }
        }
        // Cache-line-aligned ownership prevents adjacent cores from
        // dirtying different bytes of the same scalar cache line.
        const uint64_t firstRow = first / maskWidth;
        const uint64_t lastRow = (last - 1U) / maskWidth;
        for (uint64_t row = firstRow; row <= lastRow; ++row) {
            const uint64_t left = row * maskWidth;
            const uint64_t right = left + maskWidth - 1U;
            if (left >= first && left < last)
                maskGm.SetValue(left, 1U);
            if (right >= first && right < last)
                maskGm.SetValue(right, 1U);
        }
    }

    // Core zero has flushed the completed search and sorted spans.
    // The normal output path reads no mask scalars while these disjoint
    // DMA stores run. Overflow was materialized before its closure scan.
    __aicore__ __attribute__((noinline)) void ParallelMaterializeMask()
    {
#if FLOOD_FILL_PROFILE
        const int64_t start = GetSystemCycle();
#endif
        uint64_t first, last;
        Partition(static_cast<uint64_t>(height + 2U) * (width + 2U), 64U, first, last);
        uint32_t lo = 0U, hi = spanCount;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2U;
            const uint64_t end = static_cast<uint64_t>(MaskIndex(spanTable[mid].y, spanTable[mid].right)) + 1U;
            if (end <= first)
                lo = mid + 1U;
            else
                hi = mid;
        }
        if (lo == spanCount || MaskIndex(spanTable[lo].y, spanTable[lo].left) >= last)
            return;
        LocalTensor<uint8_t> local = fillQue.AllocTensor<uint8_t>();
        Duplicate(local.template ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0x0101U), 8192);
        fillQue.EnQue(local);
        LocalTensor<uint8_t> ones = fillQue.DeQue<uint8_t>();
        for (uint32_t i = lo; i < spanCount; ++i) {
            const uint32_t row = spanTable[i].y;
            const uint64_t begin = MaskIndex(row, spanTable[i].left);
            if (begin >= last)
                break;
            if (rowFlagsBase[row] != 1U)
                continue;
            const uint64_t end = static_cast<uint64_t>(MaskIndex(row, spanTable[i].right)) + 1U;
            const uint64_t from = begin > first ? begin : first;
            const uint64_t to = end < last ? end : last;
            for (uint64_t offset = from; offset < to; offset += 16384U) {
                const uint32_t count = static_cast<uint32_t>(to - offset < 16384U ? to - offset : 16384U);
                DataCopyExtParams params{1, count, 0, 0, 0};
                DataCopyPad(maskGm[offset], ones, params);
            }
        }
        const TEventID done = pipePtr->FetchEventID(HardEvent::MTE3_S);
        SetFlag<HardEvent::MTE3_S>(done);
        WaitFlag<HardEvent::MTE3_S>(done);
        fillQue.FreeTensor(ones);
#if FLOOD_FILL_PROFILE
        profileDetails[6] += GetSystemCycle() - start;
#endif
    }

    // Immutable predicates on ORIGINAL x. Directed edges use separate
    // planes for asymmetric tolerances; a fixed-range plane stores pixels.
    // Each producer owns 2048-pixel/256-byte aligned output ranges.
    __aicore__ __attribute__((noinline)) void PrecomputeComparisonBits(uint64_t rangeFirst, uint64_t rangeLast)
    {
        if constexpr (!std::is_same<T, int32_t>::value) {
            using CompareT = typename std::conditional<std::is_same<T, uint8_t>::value, half, float>::type;
            const uint64_t total = static_cast<uint64_t>(height) * width;
            const bool fixed = diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE;
            const bool twoDirections = !fixed &&
                                       (std::is_same<T, uint8_t>::value ? loDiffI != upDiffI : loDiffF != upDiffF);
            uint64_t first, last;
            Partition(rangeLast - rangeFirst, 2048U, first, last);
            first += rangeFirst;
            last += rangeFirst;
            uint64_t offset = first;
            for (; offset + 2048U <= last; offset += 2048U) {
                LocalTensor<T> local = copyQue.AllocTensor<T>();
                DataCopyExtParams load{1, 2048U * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padding{false, 0, 0, 0};
                DataCopyPad(local, xGm[offset], load, padding);
                copyQue.EnQue(local);
                LocalTensor<T> current = copyQue.DeQue<T>();
                LocalTensor<CompareT> values;
                if constexpr (std::is_same<T, uint8_t>::value) {
                    values = scanStageHalf.Get<half>()[512U];
                    Cast(values, current, RoundMode::CAST_NONE, 2048U);
                } else {
                    values = current;
                }
                LocalTensor<T> next;
                LocalTensor<CompareT> neighbors = scanAccMax.Get<CompareT>();
                if (!fixed) {
                    LocalTensor<T> nextLocal = copyQue.AllocTensor<T>();
                    const bool imageEnd = offset + 2048U == total;
                    DataCopyExtParams nextLoad{1, (imageEnd ? 2047U : 2048U) * static_cast<uint32_t>(sizeof(T)), 0, 0,
                                               0};
                    DataCopyPadExtParams<T> nextPadding{imageEnd, 0, static_cast<uint8_t>(imageEnd ? 1U : 0U), 0};
                    DataCopyPad(nextLocal, xGm[offset + 1U], nextLoad, nextPadding);
                    copyQue.EnQue(nextLocal);
                    next = copyQue.DeQue<T>();
                    if constexpr (std::is_same<T, uint8_t>::value) {
                        Cast(neighbors, next, RoundMode::CAST_NONE, 2048U);
                    } else {
                        neighbors = next;
                    }
                }
                for (uint32_t direction = 0U; direction < (twoDirections ? 2U : 1U); ++direction) {
                    LocalTensor<CompareT> delta = scanAccMin.Get<CompareT>();
                    PipeBarrier<PIPE_V>();
                    if (seedEquality) {
                        CompareScalar(scanCheckA.Get<uint8_t>(), values, ComparisonSeed<CompareT>(), CMPMODE::EQ,
                                      2048U);
                    } else {
                        if (fixed)
                            Adds(delta, values, ComparisonSeed<CompareT, true>(), 2048U);
                        else if (direction == 1U)
                            Sub(delta, values, neighbors, 2048U);
                        else
                            Sub(delta, neighbors, values, 2048U);
                        CompareToleranceBits(delta, 2048U);
                    }
                    const TEventID ready = pipePtr->FetchEventID(HardEvent::V_MTE3);
                    SetFlag<HardEvent::V_MTE3>(ready);
                    WaitFlag<HardEvent::V_MTE3>(ready);
                    DataCopyExtParams store{1, 256U, 0, 0, 0};
                    if (direction == 0U)
                        DataCopyPad(edgeForwardGm[offset / 64U], scanCheckA.Get<uint64_t>(), store);
                    else
                        DataCopyPad(edgeReverseGm[offset / 64U], scanCheckA.Get<uint64_t>(), store);
                    // scanCheckA and delta may be overwritten next iteration.
                    const TEventID consumed = pipePtr->FetchEventID(HardEvent::MTE3_V);
                    SetFlag<HardEvent::MTE3_V>(consumed);
                    WaitFlag<HardEvent::MTE3_V>(consumed);
                }
                if (!fixed)
                    copyQue.FreeTensor(next);
                copyQue.FreeTensor(current);
            }
            // Only the image's last partial 2048-pixel block takes this
            // exact scalar tail. It begins at a fresh cache-line boundary.
            for (; offset < last; offset += 64U) {
                uint64_t forward = 0U, reverse = 0U;
                for (uint32_t bit = 0U; bit < 64U && offset + bit < last; ++bit) {
                    const uint64_t index = offset + bit;
                    const T a = xGm.GetValue(index);
                    if (fixed) {
                        const bool ok = seedEquality ? a == seedValue : WithinTolerance(a, seedValue);
                        if (ok)
                            forward |= 1ULL << bit;
                    } else if (index + 1U < total) {
                        const T b = xGm.GetValue(index + 1U);
                        if (WithinTolerance(b, a))
                            forward |= 1ULL << bit;
                        if (twoDirections && WithinTolerance(a, b))
                            reverse |= 1ULL << bit;
                    }
                }
                edgeForwardGm.SetValue(offset / 64U, forward);
                if (twoDirections)
                    edgeReverseGm.SetValue(offset / 64U, reverse);
            }
            if (coreId + 1U == coreCount && rangeLast == total) {
                // Readers can combine adjacent words at an unaligned row.
                // Padding is allocated and initialized, never borrowed from metadata.
                for (uint64_t word = (total + 63U) / 64U; word < edgePlaneBytes / 8U; ++word) {
                    edgeForwardGm.SetValue(word, 0U);
                    if (twoDirections)
                        edgeReverseGm.SetValue(word, 0U);
                }
            }
        }
    }

    __aicore__ inline bool RowComparisonReady(uint32_t row) const
    {
        return edgeBitsReady && row >= edgeFirstRow && row < edgeLastRow;
    }

    __aicore__ inline void RequestComparisonRow(uint32_t row)
    {
        if (row < edgeRequestFirst)
            edgeRequestFirst = row;
        if (row + 1U > edgeRequestLast)
            edgeRequestLast = row + 1U;
    }

    __aicore__ inline void PrecomputeRowInterval(uint32_t firstRow, uint32_t lastRow)
    {
        if (firstRow >= lastRow)
            return;
        const uint64_t total = static_cast<uint64_t>(height) * width;
        const uint64_t first = static_cast<uint64_t>(firstRow) * width / 2048U * 2048U;
        uint64_t last = (static_cast<uint64_t>(lastRow) * width + 2047U) / 2048U * 2048U;
        if (last > total)
            last = total;
#if FLOOD_FILL_PROFILE
        profilePredicatePixels += last - first;
#endif
        PrecomputeComparisonBits(first, last);
    }

    __aicore__ __attribute__((noinline)) void PrecomputeRequestedWindow()
    {
        const uint32_t nextFirst = static_cast<uint32_t>(sharedControl.GetValue(6U));
        const uint32_t nextLast = static_cast<uint32_t>(sharedControl.GetValue(7U));
        if (!edgeBitsReady) {
            PrecomputeRowInterval(nextFirst, nextLast);
        } else {
            // Preserve earlier immutable bits. Only newly exposed strips
            // are generated; aligned boundary overlap has identical values.
            PrecomputeRowInterval(nextFirst, edgeFirstRow);
            PrecomputeRowInterval(edgeLastRow, nextLast);
        }
        edgeFirstRow = nextFirst;
        edgeLastRow = nextLast;
        edgeRequestFirst = 0xffffffffU;
        edgeRequestLast = 0U;
    }

    __aicore__ inline uint32_t ReadComparisonBits(uint32_t row, uint32_t first, uint32_t count, bool reverse)
    {
        const bool fixed = diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE;
        GlobalTensor<uint64_t>& bits = reverse ? edgeReverseGm : edgeForwardGm;
        const uint64_t begin = static_cast<uint64_t>(row) * width + first;
        const uint32_t shift = static_cast<uint32_t>(begin & 63U);
        uint64_t carry = 0U;
        if (shift != 0U) {
            carry = bits.GetValue(begin / 64U + (reverse ? count / 64U : 0U));
        }
        for (uint32_t word = 0U; word < count / 64U; ++word) {
            const uint32_t i = reverse ? count / 64U - 1U - word : word;
            const uint64_t position = begin / 64U + i;
            uint64_t valid;
            if (shift == 0U) {
                valid = bits.GetValue(position);
            } else {
                const uint64_t next = bits.GetValue(position + (reverse ? 0U : 1U));
                valid = reverse ? (next >> shift) | (carry << (64U - shift)) :
                                  (carry >> shift) | (next << (64U - shift));
                carry = next;
            }
            // Horizontal wrap edges between image rows are outside every
            // requested interval. The accepted endpoint's last edge is ignored.
            const uint64_t needed = !fixed && i + 1U == count / 64U ? 0x7fffffffffffffffULL : 0xffffffffffffffffULL;
            const uint64_t failed = (~valid) & needed;
            if (failed == 0U)
                continue;
            uint32_t bit = reverse ? 63U : 0U;
            if (reverse) {
                while ((failed & (1ULL << bit)) == 0U)
                    --bit;
                return count - 1U - (i * 64U + bit);
            }
            while ((failed & (1ULL << bit)) == 0U)
                ++bit;
            return i * 64U + bit + (fixed ? 0U : 1U);
        }
        return count;
    }

    __aicore__ inline uint32_t FindFixedComparisonCandidate(uint32_t row, uint32_t first, uint32_t count,
                                                            uint32_t searchFirst)
    {
        const uint64_t begin = static_cast<uint64_t>(row) * width + first;
        const uint32_t shift = static_cast<uint32_t>(begin & 63U);
        const uint32_t skipped = searchFirst - first;
        for (uint32_t word = skipped / 64U; word < count / 64U; ++word) {
            const uint64_t position = begin / 64U + word;
            uint64_t valid = edgeForwardGm.GetValue(position) >> shift;
            if (shift != 0U)
                valid |= edgeForwardGm.GetValue(position + 1U) << (64U - shift);
            if (word == skipped / 64U)
                valid &= 0xffffffffffffffffULL << (skipped % 64U);
            if (valid != 0U)
                return first + word * 64U + static_cast<uint32_t>(GetSFFValue<1>(valid));
        }
        return first + count;
    }

    __aicore__ inline uint32_t ReadRowSeedBits(uint32_t row, uint32_t first, uint32_t count, bool reverse)
    {
        const bool fixed = diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE;
        GlobalTensor<uint64_t>& bits = reverse ? edgeReverseGm : edgeForwardGm;
        const uint64_t begin = static_cast<uint64_t>(row) * width + first;
        const uint32_t shift = static_cast<uint32_t>(begin & 63U);
        const uint32_t words = (count + 63U) / 64U;
        const uint32_t lastBits = count - (words - 1U) * 64U;
        uint64_t lastNeeded = lastBits == 64U ? 0xffffffffffffffffULL : (1ULL << lastBits) - 1U;
        if (!fixed)
            lastNeeded >>= 1U;
        for (uint32_t word = 0U; word < words; ++word) {
            const uint32_t i = reverse ? words - 1U - word : word;
            const uint64_t position = begin / 64U + i;
            uint64_t valid = bits.GetValue(position) >> shift;
            if (shift != 0U)
                valid |= bits.GetValue(position + 1U) << (64U - shift);
            // Horizontal wrap edges between image rows are outside every
            // requested interval. The accepted endpoint's last edge is ignored.
            const uint64_t needed = i + 1U == words ? lastNeeded : 0xffffffffffffffffULL;
            const uint64_t failed = (~valid) & needed;
            if (failed == 0U)
                continue;
            uint32_t bit = reverse ? (i + 1U == words ? lastBits - 1U : 63U) : 0U;
            if (reverse) {
                while ((failed & (1ULL << bit)) == 0U)
                    --bit;
                return count - 1U - (i * 64U + bit);
            }
            while ((failed & (1ULL << bit)) == 0U)
                ++bit;
            return i * 64U + bit + (fixed ? 0U : 1U);
        }
        return count;
    }

    // Construct a reachable subset in parallel, then let the original
    // queue compute its complete closure. Only undirected subcases use
    // this path; a missing certified edge can discard a speculative row,
    // but cannot lose output because the exact queue still runs.
    __aicore__ inline bool CanTryNarrowRowSeeds()
    {
        if (rowSeedsAttempted || diffMode != FLOODFILL_DIFF_MODE_FIXED_RANGE || spanReadPtr < 256U ||
            deferredRows < 256U || (spanReadPtr & 127U) != 0U ||
            filledPixelCount < static_cast<uint64_t>(spanCount) * 32U ||
            filledPixelCount < static_cast<uint64_t>(height) * width / 1024U)
            return false;
        // Work prediction only: distant original pixels suggest that
        // enough rows remain to pay for full predicate generation. The
        // complete row graph and exact queue still determine reachability.
        const uint32_t distance = spanReadPtr * 2U;
        if (seedRow >= distance && WithinTolerance(xGm.GetValue((seedRow - distance) * width + seedCol), seedValue))
            return true;
        return distance < height - seedRow &&
               WithinTolerance(xGm.GetValue((seedRow + distance) * width + seedCol), seedValue);
    }

    __aicore__ inline bool CanBuildRowSeeds()
    {
        if constexpr (std::is_same<T, int32_t>::value)
            return false;
        const bool undirected = diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE ||
                                (std::is_same<T, uint8_t>::value ? loDiffI == upDiffI : loDiffF == upDiffF);
        return !rowSeedsAttempted && undirected && edgePlaneBytes != 0U &&
               (filledPixelCount >= static_cast<uint64_t>(spanCount) * (width / 4U) || CanTryNarrowRowSeeds());
    }

    __aicore__ inline bool SeedColumnQualified(uint32_t row)
    {
        if (diffMode != FLOODFILL_DIFF_MODE_FIXED_RANGE)
            return true;
        const uint64_t index = static_cast<uint64_t>(row) * width + seedCol;
        return (edgeForwardGm.GetValue(index / 64U) & (1ULL << (index & 63U))) != 0U;
    }

    // Metadata retains its existing 32B-aligned ABI. Assign physical
    // cache lines, then intersect their byte range with the logical array.
    __aicore__ inline void PartitionMetadata(uint64_t address, uint32_t count, uint32_t elementBytes, uint64_t& first,
                                             uint64_t& last) const
    {
        const uint32_t offset = static_cast<uint32_t>(address & 63U);
        Partition(static_cast<uint64_t>(count) * elementBytes + offset, 64U, first, last);
        first = first > offset ? (first - offset) / elementBytes : 0U;
        last = last > offset ? (last - offset) / elementBytes : 0U;
    }

    __aicore__ inline uint32_t FindQualifiedRowAnchor(uint32_t row, uint32_t first, uint32_t last)
    {
        if (first == last)
            return width;
        const uint64_t rowBase = static_cast<uint64_t>(row) * width;
        const uint64_t begin = rowBase + first, end = rowBase + last;
        for (uint64_t word = begin / 64U; word <= (end - 1U) / 64U; ++word) {
            uint64_t valid = edgeForwardGm.GetValue(word);
            if (word == begin / 64U)
                valid &= 0xffffffffffffffffULL << (begin & 63U);
            if (word == (end - 1U) / 64U && (end & 63U) != 0U)
                valid &= (1ULL << (end & 63U)) - 1U;
            if (valid != 0U)
                return static_cast<uint32_t>(word * 64U + GetSFFValue<1>(valid) - rowBase);
        }
        return width;
    }

    __aicore__ inline bool NeighborGapHasNoEdge(uint32_t row, uint32_t first, uint32_t last, uint32_t sourceRow,
                                                uint32_t sourceLeft, uint32_t sourceRight)
    {
        // A work bound only: decline the proof, then run the exact queue.
        if (last - first > 16U)
            return false;
        const uint32_t diagonal = connectivity == 8U ? 1U : 0U;
        for (uint32_t col = first; col < last; ++col) {
            uint32_t from = col > diagonal ? col - diagonal : 0U;
            uint32_t to = col + diagonal;
            if (from < sourceLeft)
                from = sourceLeft;
            if (to > sourceRight)
                to = sourceRight;
            const T value = xGm.GetValue(row * width + col);
            for (uint32_t other = from; other <= to; ++other) {
                if (WithinTolerance(value, xGm.GetValue(sourceRow * width + other)))
                    return false;
            }
        }
        return true;
    }

    __aicore__ inline bool RowProbeAlreadyCovered(uint32_t row, uint32_t first, uint32_t last, uint32_t interval,
                                                  uint32_t sourceRow, uint32_t sourceLeft, uint32_t sourceRight)
    {
        const uint32_t left = interval & 0xffffU, right = interval >> 16U;
        if (left <= first && right >= last)
            return true;
        if (diffMode != FLOODFILL_DIFF_MODE_FIXED_RANGE) {
            if (first < left && !NeighborGapHasNoEdge(row, first, left < last + 1U ? left : last + 1U, sourceRow,
                                                      sourceLeft, sourceRight))
                return false;
            if (last > right && !NeighborGapHasNoEdge(row, first > right + 1U ? first : right + 1U, last + 1U,
                                                      sourceRow, sourceLeft, sourceRight))
                return false;
            return true;
        }
        // Only the interval is already visited. Outside it, prove absence
        // of qualifying PIXELS; a horizontal edge plane cannot do this.
        if (first < left) {
            const uint32_t end = left < last + 1U ? left : last + 1U;
            if (FindQualifiedRowAnchor(row, first, end) != width)
                return false;
        }
        if (last > right) {
            const uint32_t begin = first > right + 1U ? first : right + 1U;
            if (FindQualifiedRowAnchor(row, begin, last + 1U) != width)
                return false;
        }
        return true;
    }

    __aicore__ __attribute__((noinline)) void BuildParallelRowSeeds()
    {
        uint64_t first, last;
        // Each core owns complete 64B lines (eight 8B records).
        Partition(height, 8U, first, last);
        uint32_t localLow = 0U, localHigh = height;
        for (uint32_t row = static_cast<uint32_t>(first); row < last; ++row) {
            uint32_t anchor = seedCol;
            bool valid = SeedColumnQualified(row);
            if (!valid && row != seedRow) {
                anchor = FindQualifiedRowAnchor(row, seedCol, width);
                if (anchor == width)
                    anchor = FindQualifiedRowAnchor(row, 0U, seedCol);
                valid = anchor != width;
            }
            uint32_t left = seedCol, right = seedCol;
            if (valid) {
                left = anchor + 1U - ReadRowSeedBits(row, 0U, anchor + 1U, true);
                right = anchor + ReadRowSeedBits(row, anchor, width - anchor, false) - 1U;
            }
            const uint64_t packed = static_cast<uint64_t>(left | (right << 16U)) | (valid ? 0x100000000ULL : 0ULL);
            bool linked = false;
            if (valid && row > 0U) {
                if (diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE) {
                    linked = SeedColumnQualified(row - 1U);
                } else {
                    linked = WithinTolerance(xGm.GetValue(row * width + seedCol),
                                             xGm.GetValue((row - 1U) * width + seedCol));
                }
            }
            rowSeedInfo.SetValue(row, packed | (linked ? 0x200000000ULL : 0ULL));
            if (diffMode != FLOODFILL_DIFF_MODE_FIXED_RANGE && !linked) {
                if (row <= seedRow)
                    localLow = row;
                else if (row < localHigh)
                    localHigh = row;
            }
        }
        if (diffMode != FLOODFILL_DIFF_MODE_FIXED_RANGE) {
            sharedControl.SetValue(8U + 8U * coreId, localLow);
            sharedControl.SetValue(9U + 8U * coreId, localHigh);
        }
        ShareBarrier();
        if (diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE) {
            // Endpoints and qualification flags are now immutable. Each
            // core changes only link flags in its owned full cache lines.
            const uint32_t diagonal = connectivity == 8U ? 1U : 0U;
            for (uint32_t row = static_cast<uint32_t>(first); row < last; ++row) {
                const uint64_t current = rowSeedInfo.GetValue(row);
                bool linked = false;
                if (row > 0U && (current & 0x100000000ULL) != 0U) {
                    const uint64_t previous = rowSeedInfo.GetValue(row - 1U);
                    if ((previous & 0x100000000ULL) != 0U) {
                        const uint32_t left = static_cast<uint32_t>(current) & 0xffffU;
                        const uint32_t right = static_cast<uint32_t>(current) >> 16U;
                        const uint32_t otherLeft = static_cast<uint32_t>(previous) & 0xffffU;
                        const uint32_t otherRight = static_cast<uint32_t>(previous) >> 16U;
                        linked = left <= otherRight + diagonal && otherLeft <= right + diagonal;
                    }
                }
                rowSeedInfo.SetValue(row, (current & ~0x200000000ULL) | (linked ? 0x200000000ULL : 0ULL));
                if (!linked) {
                    if (row <= seedRow)
                        localLow = row;
                    else if (row < localHigh)
                        localHigh = row;
                }
            }
            sharedControl.SetValue(8U + 8U * coreId, localLow);
            sharedControl.SetValue(9U + 8U * coreId, localHigh);
            ShareBarrier();
        }
        if (coreId == 0U) {
            uint32_t low = 0U, high = height;
            const bool rootValid = (rowSeedInfo.GetValue(seedRow) & 0x100000000ULL) != 0U;
            for (uint32_t core = 0U; core < coreCount; ++core) {
                const uint32_t before = static_cast<uint32_t>(sharedControl.GetValue(8U + 8U * core));
                const uint32_t after = static_cast<uint32_t>(sharedControl.GetValue(9U + 8U * core));
                if (before > low)
                    low = before;
                if (after < high)
                    high = after;
            }
            // Work-only gate. On rejection all original search state is
            // still intact; only immutable predicate coverage has grown.
            const bool choose = rootValid && high - low >= 128U && high - low >= 2U * spanCount;
            sharedControl.SetValue(0U, choose ? 1U : 0U);
            sharedControl.SetValue(6U, low);
            sharedControl.SetValue(7U, high);
        }
        ShareBarrier();
        if (sharedControl.GetValue(0U) == 0U)
            return;

        // Every initial candidate belongs to the certified neighboring-row
        // chain and is horizontally connected to its anchor. Discard all
        // old observation state before adopting this reachable subset.
        if (coreId == 0U && visitedOnesReady) {
            fillQue.FreeTensor(visitedOnes);
            visitedOnesReady = false;
        }
        if (hasMask) {
            ParallelFillBytes(maskGm, static_cast<uint64_t>(height + 2U) * (width + 2U), 0U);
            ParallelMaskBorder();
        } else {
            ParallelFillBytes(visitedGm, static_cast<uint64_t>(height) * width, 0U);
        }
        ShareBarrier();
        const uint32_t low = static_cast<uint32_t>(sharedControl.GetValue(6U));
        const uint32_t high = static_cast<uint32_t>(sharedControl.GetValue(7U));
        PartitionMetadata(reinterpret_cast<uint64_t>(rowFlagsBase), height, 1U, first, last);
        for (uint32_t row = static_cast<uint32_t>(first); row < last; ++row) {
            rowFlagsBase[row] = row >= low && row < high ? 1U : 0U;
        }
        // Adjacent arrays may share a physical cache line, and their owners
        // can differ. Complete and invalidate one array before the next.
        ShareBarrier();
        PartitionMetadata(reinterpret_cast<uint64_t>(rowIntervalsBase), height, 4U, first, last);
        if (first < low)
            first = low;
        if (last > high)
            last = high;
        for (uint32_t row = static_cast<uint32_t>(first); row < last; ++row) {
            rowIntervalsBase[row] = static_cast<uint32_t>(rowSeedInfo.GetValue(row));
        }
        ShareBarrier();
        PartitionMetadata(reinterpret_cast<uint64_t>(spanTable), high - low, 8U, first, last);
        uint64_t localPixels = 0U;
        uint64_t localPendingRows = 0U;
        const uint32_t diagonal = connectivity == 8U ? 1U : 0U;
        __gm__ SpanWord* words = reinterpret_cast<__gm__ SpanWord*>(spanTable);
        for (uint32_t index = static_cast<uint32_t>(first); index < last; ++index) {
            const uint32_t row = low + index;
            const uint32_t interval = static_cast<uint32_t>(rowSeedInfo.GetValue(row));
            const uint32_t left = interval & 0xffffU;
            const uint32_t right = interval >> 16U;
            const uint32_t probeLeft = left > diagonal ? left - diagonal : 0U;
            const uint32_t probeRight = right + diagonal < width ? right + diagonal : width - 1U;
            uint16_t skipped = 0U;
            if (row > low) {
                const uint32_t other = static_cast<uint32_t>(rowSeedInfo.GetValue(row - 1U));
                if (RowProbeAlreadyCovered(row - 1U, probeLeft, probeRight, other, row, left, right))
                    skipped |= 1U;
            } else if (row > 0U && diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE &&
                       FindQualifiedRowAnchor(row - 1U, probeLeft, probeRight + 1U) == width) {
                skipped |= 1U;
            }
            if (row + 1U < high) {
                const uint32_t other = static_cast<uint32_t>(rowSeedInfo.GetValue(row + 1U));
                if (RowProbeAlreadyCovered(row + 1U, probeLeft, probeRight, other, row, left, right))
                    skipped |= 2U;
            } else if (row + 1U < height && diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE &&
                       FindQualifiedRowAnchor(row + 1U, probeLeft, probeRight + 1U) == width) {
                skipped |= 2U;
            }
            words[index] = static_cast<uint64_t>(row) | (static_cast<uint64_t>(interval) << 16U) |
                           (static_cast<uint64_t>(skipped) << 48U);
            localPixels += right - left + 1U;
            if ((row > 0U && (skipped & 1U) == 0U) || (row + 1U < height && (skipped & 2U) == 0U))
                ++localPendingRows;
        }
        sharedControl.SetValue(8U + 8U * coreId, localPixels);
        sharedControl.SetValue(9U + 8U * coreId, localPendingRows);
        ShareBarrier();
        if (coreId == 0U) {
            filledPixelCount = 0U;
            uint64_t pendingRows = 0U;
            for (uint32_t core = 0U; core < coreCount; ++core) {
                filledPixelCount += sharedControl.GetValue(8U + 8U * core);
                pendingRows += sharedControl.GetValue(9U + 8U * core);
            }
            spanCount = high - low;
#if FLOOD_FILL_PROFILE
            profileRowSeedRows = spanCount;
            profileRowSeedPixels = filledPixelCount;
#endif
            // All candidates are already reachable. Zero pending probes
            // certifies their complete closure, including 8-connectivity.
            spanReadPtr = pendingRows == 0U ? spanCount : 0U;
            rowSeedSortedCount = spanCount;
            deferredRows = spanCount;
            lazyVisited = true;
            spanSearchStarted = true;
            spanSearchPaused = false;
            overflowed = false;
            rowProofCooldown = 0U;
            rowProofFailures = 0U;
            LocalTensor<uint32_t> cache = scanCheckB.Get<uint32_t>()[1024U];
            for (uint32_t slot = 0U; slot < kVisitedCacheSlots; ++slot)
                cache.SetValue(slot * 3U, 0xffffffffU);
        }
        ShareBarrier();
    }

    __aicore__ inline void PublishParallelSearch()
    {
        sharedControl.SetValue(5U, spanSearchPaused ? 1U : 0U);
        if (spanSearchPaused) {
            sharedControl.SetValue(0U, CanBuildRowSeeds() ? 1U : 0U);
            uint32_t first = edgeRequestFirst;
            uint32_t last = edgeRequestLast;
            if (edgeBitsReady) {
                if (edgeFirstRow < first)
                    first = edgeFirstRow;
                if (edgeLastRow > last)
                    last = edgeLastRow;
            }
            first = first > 256U ? first - 256U : 0U;
            last = last + 256U < height ? last + 256U : height;
            sharedControl.SetValue(6U, first);
            sharedControl.SetValue(7U, last);
            return;
        }
#if FLOOD_FILL_PROFILE
        profileSearchEnd = ProfileClock();
        const int64_t sortStart = GetSystemCycle();
#endif
        const bool parallelMerge = !overflowed && rowSeedSortedCount != 0U && spanCount > rowSeedSortedCount &&
                                   static_cast<uint64_t>(spanCount) * sizeof(uint64_t) <= edgePlaneBytes;
        if (parallelMerge) {
            HeapSortSpans(reinterpret_cast<__gm__ SpanWord*>(spanTable) + rowSeedSortedCount,
                          spanCount - rowSeedSortedCount);
        } else if (!overflowed) {
            SortSpansForWriteback();
        }
#if FLOOD_FILL_PROFILE
        profileDetails[0] += GetSystemCycle() - sortStart;
#endif
        // Slot 0 describes row-building only while slot 5 says paused.
        // At completion it instead publishes the final merge operation.
        sharedControl.SetValue(0U, parallelMerge ? 1U : 0U);
        if (parallelMerge)
            sharedControl.SetValue(6U, rowSeedSortedCount);
        sharedControl.SetValue(1U, spanCount);
        sharedControl.SetValue(2U, filledPixelCount);
        sharedControl.SetValue(3U, overflowed ? 1U : 0U);
        sharedControl.SetValue(4U, deferredRows);
    }

    __aicore__ __attribute__((noinline)) void ProcessParallel()
    {
#if FLOOD_FILL_PROFILE
        const int64_t t0 = ProfileClock();
#endif
        const uint64_t total = static_cast<uint64_t>(height) * width;
        if constexpr (HAS_MASK_INPUT) {
            if (coreId == 0U)
                sharedControl.SetValue(0U, 0U);
        } else if (ToleranceCoversDomain()) {
            if (coreId == 0U)
                sharedControl.SetValue(0U, 1U);
        } else {
            uint64_t scanFirst, scanLast;
            Partition(total, kScanChunkElems, scanFirst, scanLast);
            // Each participant owns a separate 64B cache line, so scalar
            // stores cannot write back another core's partially updated slot.
            const bool proven = WholeImageIsOneRegion(scanFirst, scanLast, true);
            sharedControl.SetValue(8U + 8U * coreId, proven ? 1U : 0U);
            ShareBarrier();
            if (coreId == 0U)
                sharedControl.SetValue(0U, ReduceShardExtrema() ? 1U : 0U);
        }
        ShareBarrier();
        const bool wholeRegion = sharedControl.GetValue(0U) != 0U;
#if FLOOD_FILL_PROFILE
        const int64_t t1 = ProfileClock();
#endif
        uint64_t first, last;
        Partition(total, 64U / sizeof(T), first, last);
        if (wholeRegion) {
            if (hasMask)
                ParallelFillBytes(maskGm, static_cast<uint64_t>(height + 2U) * (width + 2U), 1U);
            if (first < last)
                CopyNewValue(first, last);
#if FLOOD_FILL_PROFILE
            ShareBarrier();
            if (coreId == 0U)
                ProfileFinish(t0, t1, t1, t1, ProfileClock(), true);
#endif
            return;
        }
        if (hasMask)
            ParallelFillBytes(maskGm, static_cast<uint64_t>(height + 2U) * (width + 2U), 0U);
        else
            ParallelFillBytes(visitedGm, total, 0U);
        if (coreId == 0U) {
            GlobalTensor<uint8_t> rows;
            rows.SetGlobalBuffer(rowFlagsBase);
            ClearGlobalBytes(rows, height);
        }
        if (hasMask)
            ParallelMaskBorder();
        ShareBarrier();
#if FLOOD_FILL_PROFILE
        const int64_t t2 = ProfileClock();
#endif
        if (coreId == 0U) {
            RunSpanFill(!HAS_MASK_INPUT && edgePlaneBytes != 0U);
            PublishParallelSearch();
        }
        ShareBarrier();
        while (sharedControl.GetValue(5U) != 0U) {
#if FLOOD_FILL_PROFILE
            const int64_t predicateStart = ProfileClock();
#endif
            const bool buildRows = sharedControl.GetValue(0U) != 0U;
            if (buildRows) {
                rowSeedsAttempted = true;
#if FLOOD_FILL_PROFILE
                profilePredicatePixels += total;
#endif
                PrecomputeComparisonBits(0U, total);
                edgeFirstRow = 0U;
                edgeLastRow = height;
                edgeRequestFirst = 0xffffffffU;
                edgeRequestLast = 0U;
            } else {
                PrecomputeRequestedWindow();
            }
            ShareBarrier();
#if FLOOD_FILL_PROFILE
            profilePredicateCycles += ProfileClock() - predicateStart;
            ++profilePredicatePasses;
#endif
            edgeBitsReady = true;
            if (buildRows) {
#if FLOOD_FILL_PROFILE
                const int64_t rowSeedStart = ProfileClock();
#endif
                BuildParallelRowSeeds();
#if FLOOD_FILL_PROFILE
                profileRowSeedCycles += ProfileClock() - rowSeedStart;
#endif
            }
            if (coreId == 0U) {
                RunSpanFill(true);
                PublishParallelSearch();
            }
            ShareBarrier();
        }
        spanCount = static_cast<uint32_t>(sharedControl.GetValue(1U));
        filledPixelCount = sharedControl.GetValue(2U);
        overflowed = sharedControl.GetValue(3U) != 0U;
        if (sharedControl.GetValue(0U) != 0U) {
#if FLOOD_FILL_PROFILE
            const int64_t mergeStart = GetSystemCycle();
#endif
            ParallelMergeSpans(static_cast<uint32_t>(sharedControl.GetValue(6U)));
            ShareBarrier();
            // The unused comparison plane now holds the complete sorted
            // output table. All readers switch only after its publication.
            spanTable = reinterpret_cast<__gm__ FillSpan*>(sharedBase + FLOOD_FILL_CONTROL_BYTES + edgePlaneBytes);
#if FLOOD_FILL_PROFILE
            if (coreId == 0U)
                profileDetails[0] += GetSystemCycle() - mergeStart;
#endif
        }
        // The overflow search materialized every row before closure.
        if (overflowed)
            lazyVisited = false;
        if (hasMask && !overflowed && sharedControl.GetValue(4U) != 0U) {
            ParallelMaterializeMask();
        }
        if (first < last) {
            if (filledPixelCount == total && !overflowed)
                CopyNewValue(first, last);
            else
                CopyImageSmart(true, first, last);
        }
#if FLOOD_FILL_PROFILE
        ShareBarrier();
        if (coreId == 0U)
            ProfileFinish(t0, t1, t2, profileSearchEnd, ProfileClock(), false);
#endif
    }

    template <typename Scalar>
    __aicore__ inline bool FinishExtrema(Scalar low, Scalar high, bool publish)
    {
        const bool proven = ExtremaProveOneRegion(low, high);
        if (publish && proven) {
            if constexpr (std::is_same<T, float>::value) {
                GlobalTensor<float> bounds;
                bounds.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sharedBase));
                bounds.SetValue(16U + 16U * coreId + 2U, low);
                bounds.SetValue(16U + 16U * coreId + 3U, high);
            } else {
                GlobalTensor<int64_t> bounds;
                bounds.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(sharedBase));
                bounds.SetValue(8U + 8U * coreId + 1U, low);
                bounds.SetValue(8U + 8U * coreId + 2U, high);
            }
        }
        return proven;
    }

    __aicore__ inline bool ReduceShardExtrema()
    {
        for (uint32_t core = 0U; core < coreCount; ++core) {
            if (sharedControl.GetValue(8U + 8U * core) == 0U)
                return false;
        }
        // Local success alone is insufficient: two homogeneous shards
        // can have very different values. Recheck the GLOBAL min/max.
        if constexpr (std::is_same<T, float>::value) {
            GlobalTensor<float> bounds;
            bounds.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sharedBase));
            float low = bounds.GetValue(18U), high = bounds.GetValue(19U);
            for (uint32_t core = 1U; core < coreCount; ++core) {
                const float a = bounds.GetValue(18U + 16U * core);
                const float b = bounds.GetValue(19U + 16U * core);
                if (a < low)
                    low = a;
                if (b > high)
                    high = b;
            }
            return ExtremaProveOneRegion(low, high);
        } else {
            GlobalTensor<int64_t> bounds;
            bounds.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(sharedBase));
            int64_t low = bounds.GetValue(9U), high = bounds.GetValue(10U);
            for (uint32_t core = 1U; core < coreCount; ++core) {
                const int64_t a = bounds.GetValue(9U + 8U * core);
                const int64_t b = bounds.GetValue(10U + 8U * core);
                if (a < low)
                    low = a;
                if (b > high)
                    high = b;
            }
            return ExtremaProveOneRegion(low, high);
        }
    }

    __aicore__ inline bool ToleranceCoversDomain() const
    {
        if constexpr (std::is_same<T, uint8_t>::value) {
            if (diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE) {
                return loDiffI >= static_cast<int64_t>(seedValue) && upDiffI >= 255 - static_cast<int64_t>(seedValue);
            }
            return loDiffI >= 255 && upDiffI >= 255;
        }
        return false;
    }

#if FLOOD_FILL_PROFILE
    __aicore__ inline int64_t ProfileClock()
    {
        PipeBarrier<PIPE_ALL>();
        return GetSystemCycle();
    }

    __aicore__ inline void ProfileFinish(int64_t t0, int64_t t1, int64_t t2, int64_t t3, int64_t t4, bool fastPath)
    {
        // Search and output have completed. Reuse the first 288 workspace
        // bytes for diagnostics; the next invocation reinitializes them.
        GlobalTensor<uint64_t> record;
        record.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(profileWorkspace));
        record.SetValue(0, 0x464650524f463131ULL); // FFPROF11: 36 uint64 fields.
        record.SetValue(1, static_cast<uint64_t>(t1 - t0));
        record.SetValue(2, static_cast<uint64_t>(t2 - t1));
        record.SetValue(3, static_cast<uint64_t>(t3 - t2));
        record.SetValue(4, static_cast<uint64_t>(t4 - t3));
        record.SetValue(5, static_cast<uint64_t>(t4 - t0));
        record.SetValue(6, spanCount);
        record.SetValue(7, overflowed ? 1U : 0U);
        record.SetValue(8, fastPath ? 1U : 0U);
        record.SetValue(9, profileProofCalls);
        record.SetValue(10, profileProofFailures);
        record.SetValue(11, profileScalarSteps);
        record.SetValue(12, profileLongRevisitSpans);
        record.SetValue(13, profileLongUntouchedSpans);
        record.SetValue(14, profileLongCooledSpans);
        record.SetValue(15, profileLongRevisitPixels);
        record.SetValue(16, profileVisitedCacheHits);
        record.SetValue(17, profileVisitedCacheMisses);
        record.SetValue(18, profileVisitedMissPixels);
        for (uint32_t i = 0U; i < 9U; ++i)
            record.SetValue(19U + i, profileDetails[i]);
        record.SetValue(28U, profilePredicateCycles);
        record.SetValue(29U, profilePredicatePasses);
        record.SetValue(30U, profilePredicatePixels);
        record.SetValue(31U, profileCachedProofs);
        record.SetValue(32U, profileRowSeedCycles);
        record.SetValue(33U, rowSeedsAttempted ? 1U : 0U);
        record.SetValue(34U, profileRowSeedRows);
        record.SetValue(35U, profileRowSeedPixels);
    }
    // Raw timestamps inside already synchronized sections, diagnostic only.
    // sort, load, patch, store, horizontal proof, vertical filter, visited,
    // output chunks, unchanged output chunks. Instrumentation adds overhead.
    uint64_t profileDetails[9] = {};
    GM_ADDR profileWorkspace = nullptr;
    int64_t profileSearchEnd = 0;
    uint64_t profilePredicateCycles = 0U;
    uint64_t profilePredicatePasses = 0U;
    uint64_t profilePredicatePixels = 0U;
    uint64_t profileCachedProofs = 0U;
    uint64_t profileRowSeedCycles = 0U;
    uint64_t profileRowSeedRows = 0U;
    uint64_t profileRowSeedPixels = 0U;
    uint64_t profileProofCalls = 0U;
    uint64_t profileProofFailures = 0U;
    uint64_t profileScalarSteps = 0U;
    uint64_t profileLongRevisitSpans = 0U;
    uint64_t profileLongUntouchedSpans = 0U;
    uint64_t profileLongCooledSpans = 0U;
    uint64_t profileLongRevisitPixels = 0U;
    uint64_t profileVisitedCacheHits = 0U;
    uint64_t profileVisitedCacheMisses = 0U;
    uint64_t profileVisitedMissPixels = 0U;
#endif
    // ------------------------------------------------------------------
    // Whole-image homogeneity pre-check (drives the full-fill fast path).
    // ------------------------------------------------------------------
    // Every pixel is reachable from the seed when the global extrema of x
    // satisfy (lo/up = loDiff/upDiff, v = seed value):
    //   FIXED_RANGE: min(x) >= v - lo   &&   max(x) <= v + up
    //   NEIGHBOR   : max(x) - min(x) <= min(lo, up); every adjacent pair
    //                is then inside the tolerance window, so the fill
    //                spreads over the whole connected image.
    // Both tests are sufficient, never necessary: a false answer only
    // skips the fast path, it cannot change the fill result.
    // The scan is vectorized (Max/Min on UB) and aborts as soon as the
    // running extrema break the criterion, so a heterogeneous image
    // costs one chunk (a few KB) instead of a full pass over x.
    //
    // Power-of-two chunk for pairwise reduction. The final 32-byte block
    // is reduced in scalar code: vector operands must remain 32B aligned.
    // calCount uses counter mode; repeatTime=1 is NOT a one-repeat limit.
    static constexpr uint32_t kScanChunkElems = 2048U;

    // Criterion on the extrema, float images.
    __aicore__ inline bool ExtremaProveOneRegion(float valueLow, float valueHigh) const
    {
        if (diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE) {
            // Match WithinTolerance's subtraction order, including float
            // rounding near the tolerance boundary.
            return WithinTolerance(valueLow, seedValue) && WithinTolerance(valueHigh, seedValue);
        }
        const float width = loDiffF < upDiffF ? loDiffF : upDiffF;
        return (valueHigh - valueLow) <= width;
    }

    // Criterion on the extrema, integer images (exact int64 math).
    __aicore__ inline bool ExtremaProveOneRegion(int64_t valueLow, int64_t valueHigh) const
    {
        const int64_t diffLow = static_cast<int64_t>(loDiffI);
        const int64_t diffHigh = static_cast<int64_t>(upDiffI);
        const int64_t seed = static_cast<int64_t>(seedValue);

        if (diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE) {
            // Window of WithinTolerance(x, seed): [seed - lo, seed + up].
            return valueHigh <= seed + diffHigh && valueLow >= seed - diffLow;
        }
        return (valueHigh - valueLow) <= (diffLow < diffHigh ? diffLow : diffHigh);
    }

    // Reduce to one aligned block, never to unaligned t[4]/t[2]/t[1].
    // len must be a power of two; dependent vector instructions need a
    // barrier before reusing the result as the next instruction's input.
    template <typename AccT>
    __aicore__ inline static void PairReduceMax(LocalTensor<AccT> t, uint32_t len)
    {
        constexpr uint32_t blockElems = 32U / sizeof(AccT);
        while (len > blockElems) {
            len >>= 1U;
            PipeBarrier<PIPE_V>();
            Max(t, t, t[len], static_cast<int32_t>(len));
        }
    }

    template <typename AccT>
    __aicore__ inline static void PairReduceMin(LocalTensor<AccT> t, uint32_t len)
    {
        constexpr uint32_t blockElems = 32U / sizeof(AccT);
        while (len > blockElems) {
            len >>= 1U;
            PipeBarrier<PIPE_V>();
            Min(t, t, t[len], static_cast<int32_t>(len));
        }
    }

    // Reduces the accumulators to scalar extrema (destructive).
    template <typename AccT>
    __aicore__ inline void ReduceAccumulators(LocalTensor<AccT> accMax, LocalTensor<AccT> accMin, AccT& valueLow,
                                              AccT& valueHigh, uint32_t length = kScanChunkElems)
    {
        PairReduceMax(accMax, length);
        PairReduceMin(accMin, length);

        // VECTOR -> S: the reductions above wrote UB with vector ops.
        const TEventID eventId = pipePtr->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);

        valueLow = accMin.GetValue(0);
        valueHigh = accMax.GetValue(0);
        for (uint32_t i = 1U; i < 32U / sizeof(AccT); ++i) {
            const AccT low = accMin.GetValue(i);
            const AccT high = accMax.GetValue(i);
            if (low < valueLow) {
                valueLow = low;
            }
            if (high > valueHigh) {
                valueHigh = high;
            }
        }

        // A later running check overwrites the same scratch buffers.
        const TEventID scalarEvent = pipePtr->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarEvent);
        WaitFlag<HardEvent::S_V>(scalarEvent);
    }

    // Evaluates the criterion on extrema of the accumulation type.
    template <typename AccT>
    __aicore__ inline bool ExtremaSatisfyCriterion(AccT valueLow, AccT valueHigh) const
    {
        if constexpr (std::is_same<T, float>::value) {
            return ExtremaProveOneRegion(valueLow, valueHigh);
        } else {
            // uint8 accumulates in float (exact for 0..255), int32 in
            // int32; both convert to int64 without loss.
            return ExtremaProveOneRegion(static_cast<int64_t>(valueLow), static_cast<int64_t>(valueHigh));
        }
    }

    // Evaluates the criterion on the accumulators without destroying
    // them (the copies are reduced, the scan can continue on success).
    template <typename AccT>
    __aicore__ inline bool RunningExtremaInTolerance(LocalTensor<AccT> accMax, LocalTensor<AccT> accMin)
    {
        LocalTensor<AccT> scratchMax = scanCheckA.Get<AccT>();
        LocalTensor<AccT> scratchMin = scanCheckB.Get<AccT>();

        // Copy the accumulators with an identity Max/Min (element-wise
        // self-op) so their contents survive the destructive reduction.
        PipeBarrier<PIPE_V>();
        Max(scratchMax, accMax, accMax, static_cast<int32_t>(kScanChunkElems));
        Min(scratchMin, accMin, accMin, static_cast<int32_t>(kScanChunkElems));

        AccT valueLow = static_cast<AccT>(0);
        AccT valueHigh = static_cast<AccT>(0);
        ReduceAccumulators(scratchMax, scratchMin, valueLow, valueHigh);
        return ExtremaSatisfyCriterion(valueLow, valueHigh);
    }

    // Images smaller than one chunk: scalar GM scan over all pixels.
    __aicore__ inline bool SmallImageIsOneRegion(uint64_t totalPixels, uint64_t first = 0U, bool publish = false)
    {
        if constexpr (std::is_same<T, float>::value) {
            float valueLow = xGm.GetValue(first);
            if (valueLow != valueLow)
                return false;
            float valueHigh = valueLow;
            for (uint64_t i = first + 1U; i < totalPixels; ++i) {
                const float value = xGm.GetValue(i);
                if (value != value)
                    return false;
                if (value < valueLow) {
                    valueLow = value;
                }
                if (value > valueHigh) {
                    valueHigh = value;
                }
            }
            return FinishExtrema(valueLow, valueHigh, publish);
        } else {
            int64_t valueLow = static_cast<int64_t>(xGm.GetValue(first));
            int64_t valueHigh = valueLow;
            for (uint64_t i = first + 1U; i < totalPixels; ++i) {
                const int64_t value = static_cast<int64_t>(xGm.GetValue(i));
                if (value < valueLow) {
                    valueLow = value;
                }
                if (value > valueHigh) {
                    valueHigh = value;
                }
            }
            return FinishExtrema(valueLow, valueHigh, publish);
        }
    }

    __aicore__ inline bool WholeImageIsOneRegion(uint64_t first = 0U, uint64_t last = 0U, bool publish = false)
    {
        const uint64_t totalPixels = last == 0U ? static_cast<uint64_t>(height) * width : last;

        if (totalPixels - first < static_cast<uint64_t>(kScanChunkElems)) {
            return SmallImageIsOneRegion(totalPixels, first, publish);
        }

        // Accumulation type: Max/Min support half/float/int16/int32 but
        // not uint8, so uint8 accumulates in float through the exact
        // cast chain uint8 -> half -> float.
        constexpr bool kIsInt32 = std::is_same<T, int32_t>::value;
        using AccT = typename std::conditional<kIsInt32, int32_t, float>::type;

        LocalTensor<AccT> accMax = scanAccMax.Get<AccT>();
        LocalTensor<AccT> accMin = scanAccMin.Get<AccT>();

        if constexpr (std::is_same<T, float>::value) {
            // Extrema may discard NaNs. Accumulate self-equality bits
            // separately and reject the shortcut if any real lane was
            // NaN. This cast scratch is otherwise unused for FLOAT32.
            Duplicate(scanStageHalf.Get<uint16_t>(), static_cast<uint16_t>(0xffffU),
                      static_cast<int32_t>(kScanChunkElems / 16U));
        }

        // Neutral start: chunk 0 (always full width here) overwrites
        // every lane, so only the trailing lanes of the last partial
        // chunk can stay at these values - and those lanes keep the
        // values of the earlier full chunks, i.e. real pixels.
        if constexpr (kIsInt32) {
            Duplicate(accMax, static_cast<int32_t>(-2147483647 - 1), static_cast<int32_t>(kScanChunkElems));
            Duplicate(accMin, static_cast<int32_t>(2147483647), static_cast<int32_t>(kScanChunkElems));
        } else {
            Duplicate(accMax, -3.402823466e+38F, static_cast<int32_t>(kScanChunkElems));
            Duplicate(accMin, 3.402823466e+38F, static_cast<int32_t>(kScanChunkElems));
        }

        // GM->UB moves whole 32B blocks only (plain DataCopy, the same
        // path CopyImageSmart uses for x); the trailing sub-block pixels
        // are folded in scalar from GM after the loop.
        constexpr uint32_t blockElems = 32U / static_cast<uint32_t>(sizeof(T));
        const uint64_t vectorPixels = totalPixels / blockElems * blockElems;

        uint64_t px = first;
        uint32_t chunkIndex = 0U;
        // Running-extrema checks after chunk 1, 2, 4, 8, ...: dense at
        // the start (a non-homogeneous image is rejected after a few KB)
        // and sparse later on (a homogeneous image pays ~log2(chunks)
        // checks in total).
        uint32_t nextCheck = 1U;

        while (px < vectorPixels) {
            uint32_t elems = static_cast<uint32_t>(vectorPixels - px);
            if (elems > kScanChunkElems) {
                elems = kScanChunkElems;
            }

            LocalTensor<T> inLocal = copyQue.AllocTensor<T>();
            DataCopy(inLocal, xGm[px], elems);
            copyQue.EnQue(inLocal);
            LocalTensor<T> chunk = copyQue.DeQue<T>();

            if constexpr (std::is_same<T, uint8_t>::value) {
                // uint8 -> half -> float, both hops exact for 0..255.
                LocalTensor<half> halfChunk = scanStageHalf.Get<half>();
                LocalTensor<float> floatChunk = scanStageFloat.Get<float>();
                Cast(halfChunk, chunk, RoundMode::CAST_NONE, elems);
                PipeBarrier<PIPE_V>();
                Cast(floatChunk, halfChunk, RoundMode::CAST_NONE, elems);
                PipeBarrier<PIPE_V>();
                Max(accMax, accMax, floatChunk, static_cast<int32_t>(elems));
                Min(accMin, accMin, floatChunk, static_cast<int32_t>(elems));
            } else {
                PipeBarrier<PIPE_V>();
                if constexpr (std::is_same<T, float>::value) {
                    // Compare requires a multiple of 256 source bytes.
                    // Up to 63 remaining pixels are checked below from
                    // GM, so no uninitialized padding participates.
                    const uint32_t nanElems = elems / 64U * 64U;
                    if (nanElems != 0U) {
                        Compare(scanStageHalf.Get<uint8_t>()[256U], chunk, chunk, CMPMODE::EQ, nanElems);
                        PipeBarrier<PIPE_V>();
                        And(scanStageHalf.Get<uint16_t>(), scanStageHalf.Get<uint16_t>(),
                            scanStageHalf.Get<uint16_t>()[128U], static_cast<int32_t>(nanElems / 16U));
                    }
                }
                Max(accMax, accMax, chunk, static_cast<int32_t>(elems));
                Min(accMin, accMin, chunk, static_cast<int32_t>(elems));
            }

            // Protect staging buffers/accumulators before the next chunk.
            PipeBarrier<PIPE_V>();
            copyQue.FreeTensor(chunk);
            px += elems;
            ++chunkIndex;

            // The running extrema cover a subset of the image, so their
            // extrema are a subset of the global ones: failing here is
            // final, passing is not.
            if (chunkIndex == nextCheck) {
                nextCheck <<= 1U;
                if (!RunningExtremaInTolerance(accMax, accMin)) {
                    return false;
                }
            }
        }

        // Global extrema: the vector accumulators merged with the
        // scalar tail (fewer than one 32B block, so at most 31 pixels).
        AccT valueLow = static_cast<AccT>(0);
        AccT valueHigh = static_cast<AccT>(0);
        ReduceAccumulators(accMax, accMin, valueLow, valueHigh);

        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<uint64_t> valid = scanStageHalf.Get<uint64_t>();
            for (uint32_t i = 0U; i < kScanChunkElems / 64U; ++i) {
                if (valid.GetValue(i) != 0xffffffffffffffffULL)
                    return false;
            }
            float low = valueLow;
            float high = valueHigh;
            for (uint64_t i = totalPixels / 64U * 64U; i < totalPixels; ++i) {
                const float value = xGm.GetValue(i);
                if (value != value)
                    return false;
                if (value < low) {
                    low = value;
                }
                if (value > high) {
                    high = value;
                }
            }
            return FinishExtrema(low, high, publish);
        } else {
            int64_t low = static_cast<int64_t>(valueLow);
            int64_t high = static_cast<int64_t>(valueHigh);
            for (uint64_t i = px; i < totalPixels; ++i) {
                const int64_t value = static_cast<int64_t>(xGm.GetValue(i));
                if (value < low) {
                    low = value;
                }
                if (value > high) {
                    high = value;
                }
            }
            return FinishExtrema(low, high, publish);
        }
    }

    // Whole-image fill: y is newValue for every pixel, and in mask mode
    // the mask is 1 everywhere (every interior pixel is filled and the
    // border sentinel is 1 as well).
    __aicore__ inline void FillWholeImage()
    {
        if (hasMask) {
            FillGlobalBytes(maskGm, static_cast<uint64_t>(height + 2U) * (width + 2U), 1U);
        }

        CopyNewValue();
    }

    // Fast path for full-fill: every pixel is filled, so we skip
    // the x→UB copy and directly write newValue to y with Duplicate.
    __aicore__ inline void CopyNewValue(uint64_t first = 0U, uint64_t last = 0U)
    {
        const uint64_t totalPixels = last == 0U ? static_cast<uint64_t>(height) * width : last;
        if (totalPixels <= first)
            return;
        constexpr uint32_t chunkElems = 16384U / sizeof(T);
        const uint32_t fillElems = static_cast<uint32_t>(totalPixels - first < chunkElems ? totalPixels - first :
                                                                                            chunkElems);
        LocalTensor<T> local = fillQue.AllocTensor<T>();
        if constexpr (std::is_same<T, uint8_t>::value) {
            const uint16_t packedValue = static_cast<uint16_t>(static_cast<uint16_t>(newValueT) * 257U);
            Duplicate(local.template ReinterpretCast<uint16_t>(), packedValue,
                      static_cast<int32_t>((fillElems + 1U) / 2U));
        } else {
            Duplicate(local, newValueT, static_cast<int32_t>(fillElems));
        }
        fillQue.EnQue(local);
        LocalTensor<T> tile = fillQue.DeQue<T>();
        for (uint64_t px = first; px < totalPixels; px += chunkElems) {
            const uint32_t elems = static_cast<uint32_t>(totalPixels - px < chunkElems ? totalPixels - px : chunkElems);
            DataCopyExtParams params{1, elems * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm[px], tile, params);
        }
        // The immutable tile is retained until every store has consumed it.
        const TEventID done = pipePtr->FetchEventID(HardEvent::MTE3_S);
        SetFlag<HardEvent::MTE3_S>(done);
        WaitFlag<HardEvent::MTE3_S>(done);
        fillQue.FreeTensor(tile);
    }

    // Stream the sorted spans through two existing 16KB input slots.
    // Retain both allocations until the last store completes: VECIN alone
    // does not protect a buffer still being read by MTE3. Per-slot events
    // order the previous store before its next load. Unchanged chunks only
    // need MTE2->MTE3, so neither scalar nor vector execution waits on DMA.
    __aicore__ inline void CopyImageSmart(bool prepared = false, uint64_t first = 0U, uint64_t last = 0U)
    {
#if FLOOD_FILL_PROFILE
        const int64_t sortStart = GetSystemCycle();
#endif
        if (!prepared && !overflowed)
            SortSpansForWriteback();
#if FLOOD_FILL_PROFILE
        profileDetails[0] += GetSystemCycle() - sortStart;
#endif
        if constexpr (std::is_same<T, float>::value) {
            // Direct spans save input traffic, but short DMA segments lose
            // to the chunked path. Keep large, sufficiently broad regions.
            if (!prepared && !overflowed && static_cast<uint64_t>(height) * width >= 1048576U &&
                filledPixelCount >= static_cast<uint64_t>(spanCount) * 1024U) {
                CopySpanSegments();
                return;
            }
        }
        uint32_t nextSpan = 0U;
        const uint64_t totalPixels = last == 0U ? static_cast<uint64_t>(height) * width : last;
        if (prepared && !overflowed) {
            uint32_t lo = 0U, hi = spanCount;
            while (lo < hi) {
                const uint32_t mid = lo + (hi - lo) / 2U;
                const uint64_t end = static_cast<uint64_t>(spanTable[mid].y) * width + spanTable[mid].right + 1U;
                if (end <= first)
                    lo = mid + 1U;
                else
                    hi = mid;
            }
            nextSpan = lo;
        }
        constexpr uint32_t elemBytes = static_cast<uint32_t>(sizeof(T));
        constexpr uint32_t chunkElems = 16384U / elemBytes;
        LocalTensor<T> tiles[2] = {copyQue.AllocTensor<T>(), copyQue.AllocTensor<T>()};
        TEventID reuse[2] = {pipePtr->AllocEventID<HardEvent::MTE3_MTE2>(),
                             pipePtr->AllocEventID<HardEvent::MTE3_MTE2>()};
        TEventID ready[2] = {pipePtr->AllocEventID<HardEvent::MTE2_MTE3>(),
                             pipePtr->AllocEventID<HardEvent::MTE2_MTE3>()};
        bool pending[2] = {false, false};
        uint32_t slot = 0U;
        for (uint64_t px = first; px < totalPixels; px += chunkElems, slot ^= 1U) {
            const uint32_t elems = static_cast<uint32_t>(totalPixels - px < chunkElems ? totalPixels - px : chunkElems);
            const bool unchanged = !overflowed &&
                                   (nextSpan == spanCount ||
                                    static_cast<uint64_t>(spanTable[nextSpan].y) * width + spanTable[nextSpan].left >=
                                        px + elems);
#if FLOOD_FILL_PROFILE
            const int64_t loadStart = GetSystemCycle();
            ++profileDetails[7];
            if (unchanged)
                ++profileDetails[8];
#endif
            if (pending[slot])
                WaitFlag<HardEvent::MTE3_MTE2>(reuse[slot]);
            LocalTensor<T>& local = tiles[slot];
            // Exact byte length handles the last chunk without scalar GM
            // reads, preserving every original FLOAT32 payload bit.
            DataCopyExtParams params{1, elems * elemBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padding{false, 0, 0, 0};
            DataCopyPad(local, xGm[px], params, padding);
            if (!unchanged) {
                const TEventID loaded = pipePtr->FetchEventID(HardEvent::MTE2_S);
                SetFlag<HardEvent::MTE2_S>(loaded);
                WaitFlag<HardEvent::MTE2_S>(loaded);
            }
#if FLOOD_FILL_PROFILE
            const int64_t patchStart = GetSystemCycle();
            profileDetails[1] += patchStart - loadStart;
#endif
            if (!unchanged)
                PatchChunkWithSpans(local, px, elems, nextSpan);
#if FLOOD_FILL_PROFILE
            const int64_t storeStart = GetSystemCycle();
            profileDetails[2] += storeStart - patchStart;
#endif
            if (unchanged) {
                SetFlag<HardEvent::MTE2_MTE3>(ready[slot]);
                WaitFlag<HardEvent::MTE2_MTE3>(ready[slot]);
            } else {
                const TEventID patched = pipePtr->FetchEventID(HardEvent::S_MTE3);
                SetFlag<HardEvent::S_MTE3>(patched);
                WaitFlag<HardEvent::S_MTE3>(patched);
            }
            DataCopyPad(yGm[px], local, params);
            SetFlag<HardEvent::MTE3_MTE2>(reuse[slot]);
            pending[slot] = true;
#if FLOOD_FILL_PROFILE
            profileDetails[3] += GetSystemCycle() - storeStart;
#endif
        }
        const TEventID drained = pipePtr->FetchEventID(HardEvent::MTE3_S);
        SetFlag<HardEvent::MTE3_S>(drained);
        WaitFlag<HardEvent::MTE3_S>(drained);
        for (uint32_t i = 0U; i < 2U; ++i) {
            if (pending[i])
                WaitFlag<HardEvent::MTE3_MTE2>(reuse[i]);
            copyQue.FreeTensor(tiles[i]);
            pipePtr->ReleaseEventID<HardEvent::MTE3_MTE2>(reuse[i]);
            pipePtr->ReleaseEventID<HardEvent::MTE2_MTE3>(ready[i]);
        }
    }

    // Sorted filled spans and unfilled gaps partition y. Never load x for
    // a filled segment; an immutable VECOUT tile supplies its new value.
    // Gap copies retain two slots and per-slot DMA lifetime dependencies.
    __aicore__ inline void CopySpanSegments()
    {
        constexpr uint32_t chunkElems = 16384U / sizeof(T);
        const uint64_t totalPixels = static_cast<uint64_t>(height) * width;
        LocalTensor<T> tiles[2] = {copyQue.AllocTensor<T>(), copyQue.AllocTensor<T>()};
        TEventID reuse[2] = {pipePtr->AllocEventID<HardEvent::MTE3_MTE2>(),
                             pipePtr->AllocEventID<HardEvent::MTE3_MTE2>()};
        TEventID ready[2] = {pipePtr->AllocEventID<HardEvent::MTE2_MTE3>(),
                             pipePtr->AllocEventID<HardEvent::MTE2_MTE3>()};
        bool pending[2] = {false, false};
        LocalTensor<T> values = fillQue.AllocTensor<T>();
        if constexpr (std::is_same<T, uint8_t>::value) {
            const uint16_t packedValue = static_cast<uint16_t>(static_cast<uint16_t>(newValueT) * 257U);
            Duplicate(values.template ReinterpretCast<uint16_t>(), packedValue, 8192);
        } else {
            Duplicate(values, newValueT, static_cast<int32_t>(chunkElems));
        }
        fillQue.EnQue(values);
        LocalTensor<T> fill = fillQue.DeQue<T>();
        uint32_t slot = 0U;
        uint64_t px = 0U;
        for (uint32_t i = 0U; i <= spanCount; ++i) {
            const uint64_t rowBase = i == spanCount ? totalPixels : static_cast<uint64_t>(spanTable[i].y) * width;
            const uint64_t first = i == spanCount ? totalPixels : rowBase + spanTable[i].left;
            const uint64_t end = i == spanCount ? totalPixels : rowBase + spanTable[i].right + 1U;
            while (px < first) {
                const uint32_t elems = static_cast<uint32_t>(first - px < chunkElems ? first - px : chunkElems);
                if (pending[slot])
                    WaitFlag<HardEvent::MTE3_MTE2>(reuse[slot]);
                DataCopyExtParams params{1, elems * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padding{false, 0, 0, 0};
                DataCopyPad(tiles[slot], xGm[px], params, padding);
                SetFlag<HardEvent::MTE2_MTE3>(ready[slot]);
                WaitFlag<HardEvent::MTE2_MTE3>(ready[slot]);
                DataCopyPad(yGm[px], tiles[slot], params);
                SetFlag<HardEvent::MTE3_MTE2>(reuse[slot]);
                pending[slot] = true;
                slot ^= 1U;
                px += elems;
#if FLOOD_FILL_PROFILE
                ++profileDetails[7];
                ++profileDetails[8];
#endif
            }
            while (px < end) {
                const uint32_t elems = static_cast<uint32_t>(end - px < chunkElems ? end - px : chunkElems);
                DataCopyExtParams params{1, elems * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                DataCopyPad(yGm[px], fill, params);
                px += elems;
#if FLOOD_FILL_PROFILE
                ++profileDetails[7];
#endif
            }
        }
        const TEventID drained = pipePtr->FetchEventID(HardEvent::MTE3_S);
        SetFlag<HardEvent::MTE3_S>(drained);
        WaitFlag<HardEvent::MTE3_S>(drained);
        for (uint32_t i = 0U; i < 2U; ++i) {
            if (pending[i])
                WaitFlag<HardEvent::MTE3_MTE2>(reuse[i]);
            copyQue.FreeTensor(tiles[i]);
            pipePtr->ReleaseEventID<HardEvent::MTE3_MTE2>(reuse[i]);
            pipePtr->ReleaseEventID<HardEvent::MTE2_MTE3>(ready[i]);
        }
        fillQue.FreeTensor(fill);
    }

    // The table base is 32B aligned and each entry is exactly 8B. Use an
    // alias-capable word view after search completes; retain all four uint16
    // fields, including padding. Fewer GM accesses than swapping six fields.
    typedef uint64_t SpanWord __attribute__((may_alias));

    __aicore__ inline uint32_t PackedSpanKey(uint64_t span) const
    {
        return static_cast<uint32_t>((span & 0xffffULL) << 16U) | static_cast<uint32_t>((span >> 16U) & 0xffffULL);
    }

    // Hold the displaced entry in a register and move one child into each
    // hole. Heap construction and extraction both reuse this sift routine.
    __aicore__ inline void SiftSpans(__gm__ SpanWord* words, uint32_t root, uint32_t count, uint64_t value)
    {
        const uint32_t key = PackedSpanKey(value);
        uint32_t child = 2U * root + 1U;
        while (child < count) {
            uint64_t selected = words[child];
            uint32_t selectedKey = PackedSpanKey(selected);
            if (child + 1U < count) {
                const uint64_t sibling = words[child + 1U];
                const uint32_t siblingKey = PackedSpanKey(sibling);
                if (selectedKey < siblingKey) {
                    ++child;
                    selected = sibling;
                    selectedKey = siblingKey;
                }
            }
            if (key >= selectedKey)
                break;
            words[root] = selected;
            root = child;
            child = 2U * root + 1U;
        }
        words[root] = value;
    }

    __aicore__ inline void HeapSortSpans(__gm__ SpanWord* words, uint32_t count)
    {
        if (count < 2U)
            return;
        for (uint32_t root = count / 2U; root > 0U; --root) {
            SiftSpans(words, root - 1U, count, words[root - 1U]);
        }
        for (uint32_t end = count - 1U; end > 0U; --end) {
            const uint64_t value = words[end];
            words[end] = words[0];
            SiftSpans(words, 0U, end, value);
        }
    }

    // Both input runs are immutable and sorted by (row,left). Merge-path
    // co-ranking locates each disjoint 64B-aligned output partition. Equal
    // keys take the prefix first, matching the serial backwards merge.
    __aicore__ __attribute__((noinline)) void ParallelMergeSpans(uint32_t prefix)
    {
        uint64_t first, last;
        Partition(spanCount, 8U, first, last);
        if (first == last)
            return;
        __gm__ SpanWord* words = reinterpret_cast<__gm__ SpanWord*>(spanTable);
        const uint32_t tailCount = spanCount - prefix;
        const uint32_t diagonal = static_cast<uint32_t>(first);
        uint32_t low = diagonal > tailCount ? diagonal - tailCount : 0U;
        uint32_t high = diagonal < prefix ? diagonal : prefix;
        uint32_t a = low, b = diagonal - low;
        while (low <= high) {
            a = low + (high - low) / 2U;
            b = diagonal - a;
            if (a > 0U && b < tailCount && PackedSpanKey(words[a - 1U]) > PackedSpanKey(words[prefix + b])) {
                high = a - 1U;
            } else if (b > 0U && a < prefix && PackedSpanKey(words[prefix + b - 1U]) >= PackedSpanKey(words[a])) {
                low = a + 1U;
            } else {
                break;
            }
        }
        for (uint32_t output = diagonal; output < last; ++output) {
            uint64_t value;
            if (b == tailCount || (a < prefix && PackedSpanKey(words[a]) <= PackedSpanKey(words[prefix + b]))) {
                value = words[a++];
            } else {
                value = words[prefix + b++];
            }
            rowSeedInfo.SetValue(output, value);
        }
    }

    // Dense row coverage: counting the row key avoids a global comparison
    // sort. Two existing 16KB slots hold the row ends and write cursors, so
    // heights above 4096 and sparse tables retain the bounded heap fallback.
    __aicore__ inline void BucketSpansByRow(__gm__ SpanWord* words)
    {
        // Search used both slots for vector reads. Scalar index arrays must
        // wait for those reads and for any pending queue reuse dependencies.
        PipeBarrier<PIPE_ALL>();
        LocalTensor<uint32_t> ends = copyQue.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> next = copyQue.AllocTensor<uint32_t>();
        for (uint32_t row = 0U; row < height; ++row)
            ends.SetValue(row, 0U);
        for (uint32_t i = 0U; i < spanCount; ++i) {
            const uint32_t row = static_cast<uint32_t>(words[i] & 0xffffULL);
            ends.SetValue(row, ends.GetValue(row) + 1U);
        }
        uint32_t sum = 0U;
        for (uint32_t row = 0U; row < height; ++row) {
            const uint32_t count = ends.GetValue(row);
            next.SetValue(row, sum);
            sum += count;
            ends.SetValue(row, sum);
        }
        for (uint32_t row = 0U; row < height; ++row) {
            const uint32_t end = ends.GetValue(row);
            uint32_t index = next.GetValue(row);
            while (index < end) {
                uint64_t value = words[index];
                uint32_t target = static_cast<uint32_t>(value & 0xffffULL);
                while (target != row) {
                    const uint32_t dest = next.GetValue(target);
                    next.SetValue(target, dest + 1U);
                    const uint64_t displaced = words[dest];
                    words[dest] = value;
                    value = displaced;
                    target = static_cast<uint32_t>(value & 0xffffULL);
                }
                words[index] = value;
                next.SetValue(row, ++index);
            }
        }
        uint32_t first = 0U;
        for (uint32_t row = 0U; row < height; ++row) {
            const uint32_t end = ends.GetValue(row);
            HeapSortSpans(words + first, end - first);
            first = end;
        }
        // Finish scalar UB stores before the output DMA reuses these slots.
        const TEventID indexesDone = pipePtr->FetchEventID(HardEvent::S_MTE2);
        SetFlag<HardEvent::S_MTE2>(indexesDone);
        WaitFlag<HardEvent::S_MTE2>(indexesDone);
        copyQue.FreeTensor(next);
        copyQue.FreeTensor(ends);
    }

    // All output paths observe ascending (row, left), independent of the
    // discovery order, repeated rows, or the sorting algorithm selected.
    __aicore__ inline void SortSpansForWriteback()
    {
        static_assert(sizeof(FillSpan) == sizeof(uint64_t), "packed span size");
        if (spanCount < 2U)
            return;
        // The adopted table contains one interval per ascending row.
        // Queue processing only appends entries; it never moves this prefix.
        if (rowSeedSortedCount != 0U && spanCount == rowSeedSortedCount)
            return;
        __gm__ SpanWord* words = reinterpret_cast<__gm__ SpanWord*>(spanTable);
        const uint32_t extra = spanCount - rowSeedSortedCount;
        if (rowSeedSortedCount != 0U && static_cast<uint64_t>(extra) * sizeof(uint64_t) <= edgePlaneBytes) {
            HeapSortSpans(words + rowSeedSortedCount, extra);
            // All row-seed reads finished before closure. Only undirected
            // modes can set rowSeedSortedCount, so this second plane is no
            // longer needed by either comparison direction.
            for (uint32_t i = 0U; i < extra; ++i)
                rowSeedInfo.SetValue(i, words[rowSeedSortedCount + i]);
            uint32_t a = rowSeedSortedCount, b = extra, output = spanCount;
            while (b > 0U) {
                const uint64_t tail = rowSeedInfo.GetValue(b - 1U);
                if (a > 0U && PackedSpanKey(words[a - 1U]) > PackedSpanKey(tail)) {
                    words[--output] = words[--a];
                } else {
                    words[--output] = tail;
                    --b;
                }
            }
            return;
        }
        if (height >= 16U && height <= 4096U && spanCount >= 128U && spanCount >= height / 2U) {
            BucketSpansByRow(words);
        } else {
            HeapSortSpans(words, spanCount);
        }
    }

    // Keep partial 32B blocks scalar; vector operands start at an aligned
    // UB address. Explicit S->V->S handshakes retain the existing S->MTE3
    // writeback ordering even though this buffer belongs to VECIN.
    __aicore__ inline void FillLocalSpan(LocalTensor<T>& local, uint32_t begin, uint32_t end)
    {
        constexpr uint32_t blockElems = 32U / sizeof(T);
        uint32_t index = begin;
        while (index < end && index % blockElems != 0U) {
            local.SetValue(index++, newValueT);
        }
        const uint32_t bulk = (end - index) / blockElems * blockElems;
        if (bulk > 0U) {
            const TEventID before = pipePtr->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(before);
            WaitFlag<HardEvent::S_V>(before);
            if constexpr (std::is_same<T, uint8_t>::value) {
                LocalTensor<uint16_t> packed = local.template ReinterpretCast<uint16_t>();
                const uint16_t value = static_cast<uint16_t>(newValueT) * 257U;
                Duplicate(packed[index / 2U], value, static_cast<int32_t>(bulk / 2U));
            } else {
                Duplicate(local[index], newValueT, static_cast<int32_t>(bulk));
            }
            const TEventID after = pipePtr->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(after);
            WaitFlag<HardEvent::V_S>(after);
            index += bulk;
        }
        while (index < end)
            local.SetValue(index++, newValueT);
    }

    // Consume sorted intervals once. A span crossing a chunk boundary
    // remains at nextSpan for the next chunk; all other spans advance.
    __aicore__ inline void PatchChunkWithSpans(LocalTensor<T>& xLocal, uint64_t px, uint32_t elems, uint32_t& nextSpan)
    {
        if (overflowed) {
            // FallbackSweep can visit pixels whose spans were not stored.
            // The visited map, not the truncated table, is authoritative.
            for (uint32_t i = 0U; i < elems; ++i) {
                const uint64_t pixel = px + i;
                if (IsVisited(static_cast<uint32_t>(pixel / width), static_cast<uint32_t>(pixel % width)))
                    xLocal.SetValue(i, newValueT);
            }
            return;
        }

        const uint64_t chunkEnd = px + elems;
        while (nextSpan < spanCount) {
            const uint64_t rowBase = static_cast<uint64_t>(spanTable[nextSpan].y) * width;
            const uint64_t segStart = rowBase + spanTable[nextSpan].left;
            const uint64_t segEnd = rowBase + spanTable[nextSpan].right + 1U;
            if (segStart >= chunkEnd)
                break;
            if (segEnd > px) {
                const uint64_t lo = segStart > px ? segStart : px;
                const uint64_t hi = segEnd < chunkEnd ? segEnd : chunkEnd;
                FillLocalSpan(xLocal, static_cast<uint32_t>(lo - px), static_cast<uint32_t>(hi - px));
            }
            if (segEnd > chunkEnd)
                break;
            ++nextSpan;
        }
    }

    // Vectorized zero fill of a byte-addressable GM region (see
    // FillGlobalBytes).
    __aicore__ inline void ClearGlobalBytes(GlobalTensor<uint8_t>& dst, uint64_t bytes)
    {
        FillGlobalBytes(dst, bytes, 0U);
    }

    // Vectorized byte fill, including the non-32B-aligned tail. A scalar
    // tail store can write back a stale cache line over the DMA prefix.
    // Use VECOUT for V→MTE3 synchronization and exact-length DataCopyPad.
    __aicore__ inline void FillGlobalBytes(GlobalTensor<uint8_t>& dst, uint64_t bytes, uint8_t value)
    {
        if (bytes == 0U)
            return;
        constexpr uint32_t chunkBytes = 16384U;
        const uint32_t fillBytes = static_cast<uint32_t>(bytes < chunkBytes ? bytes : chunkBytes);
        LocalTensor<uint8_t> local = fillQue.AllocTensor<uint8_t>();
        const uint16_t packedValue = static_cast<uint16_t>(static_cast<uint16_t>(value) * 257U);
        Duplicate(local.template ReinterpretCast<uint16_t>(), packedValue, static_cast<int32_t>((fillBytes + 1U) / 2U));
        fillQue.EnQue(local);
        LocalTensor<uint8_t> tile = fillQue.DeQue<uint8_t>();
        for (uint64_t offset = 0U; offset < bytes; offset += chunkBytes) {
            const uint32_t elems = static_cast<uint32_t>(bytes - offset < chunkBytes ? bytes - offset : chunkBytes);
            DataCopyExtParams params{1, elems, 0, 0, 0};
            DataCopyPad(dst[offset], tile, params);
        }
        // Order all zeroing before subsequent scalar border/visited stores.
        const TEventID done = pipePtr->FetchEventID(HardEvent::MTE3_S);
        SetFlag<HardEvent::MTE3_S>(done);
        WaitFlag<HardEvent::MTE3_S>(done);
        fillQue.FreeTensor(tile);
    }

    // mask mode: interior 0 + border 1 (sentinel).
    // bitmap mode: all zero.
    // row flags: zeroed.
    __aicore__ inline void InitVisited(bool alreadyCleared = false)
    {
        if (hasMask) {
            const uint32_t maskWidth = width + 2U;
            const uint32_t maskHeight = height + 2U;

            // Interior clear (the border is patched afterwards).
            if (!alreadyCleared)
                ClearGlobalBytes(maskGm, static_cast<uint64_t>(maskHeight) * maskWidth);

            // Border sentinels (2*H + 2*W scalar writes, negligible
            // next to the H*W vectorized clear above).
            for (uint32_t column = 0U; column < maskWidth; ++column) {
                maskGm.SetValue(column, 1U);
                maskGm.SetValue(static_cast<uint64_t>(height + 1U) * maskWidth + column, 1U);
            }

            for (uint32_t row = 0U; row < maskHeight; ++row) {
                const uint64_t rowBase = static_cast<uint64_t>(row) * maskWidth;
                maskGm.SetValue(rowBase, 1U);
                maskGm.SetValue(rowBase + maskWidth - 1U, 1U);
            }
        } else {
            const uint64_t totalPixels = static_cast<uint64_t>(height) * width;
            if (!alreadyCleared)
                ClearGlobalBytes(visitedGm, totalPixels);
        }

        GlobalTensor<uint8_t> rowFlagsGm;
        rowFlagsGm.SetGlobalBuffer(rowFlagsBase);
        if (!alreadyCleared)
            ClearGlobalBytes(rowFlagsGm, static_cast<uint64_t>(height));
    }

    // Visited accessors: mask mode maps (row, col) -> mask[row+1][col+1].
    __aicore__ inline uint32_t MaskIndex(uint32_t row, uint32_t col) const
    {
        const uint32_t maskWidth = width + 2U;
        return (row + 1U) * maskWidth + (col + 1U);
    }

    __aicore__ inline bool IsVisited(uint32_t row, uint32_t col) const
    {
        if (lazyVisited) {
            const uint8_t state = rowFlagsBase[row];
            if (state == 0U)
                return false;
            if (state == 1U) {
                const uint32_t interval = rowIntervalsBase[row];
                return col >= (interval & 0xffffU) && col <= (interval >> 16U);
            }
        }
        if (hasMask) {
            return maskGm.GetValue(MaskIndex(row, col)) != 0U;
        }
        return visitedGm.GetValue(row * width + col) != 0U;
    }

    __aicore__ inline bool IsBlocked(uint32_t row, uint32_t col) const
    {
        if constexpr (HAS_MASK_INPUT)
            return inputMaskGm.GetValue(MaskIndex(row, col)) != 0U;
        return false;
    }

    __aicore__ inline void MarkVisited(uint32_t row, uint32_t col)
    {
        if (hasMask) {
            maskGm.SetValue(MaskIndex(row, col), 1U);
        } else {
            visitedGm.SetValue(row * width + col, 1U);
        }
    }

    // Long spans use DMA after explicitly cleaning/invalidating every
    // overlapping scalar cache line, including both boundary lines.
    // Short spans retain packed scalar stores to avoid DMA setup cost.
    // A sole interval is an exact visited representation, including
    // intervals ending at column 65534. A second disjoint interval first
    // materializes the old one, then uses the ordinary bitmap path.
    __aicore__ inline void MarkVisitedSpan(uint32_t row, uint32_t left, uint32_t right)
    {
        if (lazyVisited) {
            const uint8_t state = rowFlagsBase[row];
            if (state == 0U) {
                rowIntervalsBase[row] = left | (right << 16U);
                rowFlagsBase[row] = 1U;
                ++deferredRows;
                return;
            }
            if (state == 1U) {
                const uint32_t old = rowIntervalsBase[row];
                WriteVisitedSpan(row, old & 0xffffU, old >> 16U);
                --deferredRows;
            }
        }
        WriteVisitedSpan(row, left, right);
    }

    __aicore__ inline void WriteVisitedSpan(uint32_t row, uint32_t left, uint32_t right)
    {
        uint32_t col = left;
        uint32_t index = hasMask ? MaskIndex(row, left) : row * width + left;
        const uint32_t length = right - left + 1U;
        if (length >= 256U) {
            GlobalTensor<uint8_t> visited;
            visited.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(visitedWordBase));
            const uint64_t baseAddress = reinterpret_cast<uint64_t>(visitedWordBase);
            uint32_t offset = index;
            while (offset < index + length) {
                DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE>(visited[offset]);
                offset += 64U - static_cast<uint32_t>((baseAddress + offset) & 63U);
            }
            // Complete scalar writes/cache maintenance before DMA can
            // modify the same lines. No scalar visited access occurs
            // until the copy has finished its MTE3->S handshake.
            const TEventID before = pipePtr->FetchEventID(HardEvent::S_MTE3);
            SetFlag<HardEvent::S_MTE3>(before);
            WaitFlag<HardEvent::S_MTE3>(before);
            GlobalTensor<uint8_t> span;
            span.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(visitedWordBase) + index);
            if (!visitedOnesReady) {
                // Borrow one existing VECOUT slot for the duration of the
                // search. Its contents stay immutable across all spans.
                LocalTensor<uint8_t> ones = fillQue.AllocTensor<uint8_t>();
                Duplicate(ones.template ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0x0101U), 8192);
                fillQue.EnQue(ones);
                visitedOnes = fillQue.DeQue<uint8_t>();
                visitedOnesReady = true;
            }
            for (uint32_t done = 0U; done < length; done += 16384U) {
                const uint32_t bytes = length - done < 16384U ? length - done : 16384U;
                DataCopyExtParams params{1, bytes, 0, 0, 0};
                DataCopyPad(span[done], visitedOnes, params);
            }
            const TEventID complete = pipePtr->FetchEventID(HardEvent::MTE3_S);
            SetFlag<HardEvent::MTE3_S>(complete);
            WaitFlag<HardEvent::MTE3_S>(complete);
            rowFlagsBase[row] = rowFlagsBase[row] == 0U ? 1U : 2U;
            return;
        }
        while (col <= right && (index & 7U) != 0U) {
            MarkVisited(row, col++);
            ++index;
        }
        while (col + 7U <= right) {
            visitedWordBase[index / 8U] = 0x0101010101010101ULL;
            index += 8U;
            col += 8U;
        }
        while (col <= right)
            MarkVisited(row, col++);
        rowFlagsBase[row] = rowFlagsBase[row] == 0U ? 1U : 2U;
    }

    // Flush the remaining sole intervals together. No scalar bitmap
    // access overlaps these DMA stores; metadata is read-only throughout.
    // Overflow calls this before disabling lazy reads and running closure.
    __aicore__ __attribute__((noinline)) void MaterializeVisitedRows(bool allRows = false)
    {
        if (deferredRows == 0U)
            return;
#if FLOOD_FILL_PROFILE
        const int64_t start = GetSystemCycle();
#endif
        GlobalTensor<uint8_t> visited;
        visited.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(visitedWordBase));
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE>(visited);
        const TEventID before = pipePtr->FetchEventID(HardEvent::S_MTE3);
        SetFlag<HardEvent::S_MTE3>(before);
        WaitFlag<HardEvent::S_MTE3>(before);
        if (!visitedOnesReady) {
            LocalTensor<uint8_t> ones = fillQue.AllocTensor<uint8_t>();
            Duplicate(ones.template ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0x0101U), 8192);
            fillQue.EnQue(ones);
            visitedOnes = fillQue.DeQue<uint8_t>();
            visitedOnesReady = true;
        }
        // Normal output visits only queued spans, so a tiny fill does
        // not acquire an O(height) scan. Overflow includes dropped spans
        // and therefore must inspect every row's exact metadata.
        const uint32_t iterations = allRows ? height : spanCount;
        for (uint32_t i = 0U; i < iterations; ++i) {
            const uint32_t row = allRows ? i : spanTable[i].y;
            if (rowFlagsBase[row] != 1U)
                continue;
            const uint32_t interval = rowIntervalsBase[row];
            const uint32_t left = interval & 0xffffU;
            const uint32_t length = (interval >> 16U) - left + 1U;
            const uint32_t index = hasMask ? MaskIndex(row, left) : row * width + left;
            for (uint32_t offset = 0U; offset < length; offset += 16384U) {
                const uint32_t bytes = length - offset < 16384U ? length - offset : 16384U;
                DataCopyExtParams params{1, bytes, 0, 0, 0};
                DataCopyPad(visited[index + offset], visitedOnes, params);
            }
        }
        const TEventID done = pipePtr->FetchEventID(HardEvent::MTE3_S);
        SetFlag<HardEvent::MTE3_S>(done);
        WaitFlag<HardEvent::MTE3_S>(done);
        // A metadata read can share the last bitmap cache line in the
        // mask-absent layout. Invalidate it before fallback reads bitmap.
        DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE>(visited);
#if FLOOD_FILL_PROFILE
        profileDetails[6] += GetSystemCycle() - start;
#endif
    }

    // Per-dtype tolerance test: candidate value against reference value.
    __aicore__ inline bool WithinTolerance(T candidate, T reference) const
    {
        if constexpr (std::is_same<T, float>::value) {
            const float delta = candidate - reference;
            return delta >= -loDiffF && delta <= upDiffF;
        } else {
            const int64_t delta = static_cast<int64_t>(candidate) - static_cast<int64_t>(reference);
            return delta >= -static_cast<int64_t>(loDiffI) && delta <= static_cast<int64_t>(upDiffI);
        }
    }

    // Extension has a known visited horizontal neighbor. Try that
    // original value first; on failure other visited neighbors can
    // still make the candidate reachable, so retain the complete test.
    __aicore__ inline bool AcceptHorizontal(uint32_t row, uint32_t col, T& reference)
    {
        if (IsBlocked(row, col))
            return false;
#if FLOOD_FILL_PROFILE
        ++profileScalarSteps;
#endif
        const T value = xGm.GetValue(row * width + col);
        const bool accepted = diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE ?
                                  WithinTolerance(value, seedValue) :
                                  (WithinTolerance(value, reference) || BfsAccept(row, col));
        if (accepted)
            reference = value;
        return accepted;
    }

    // Device scalar arithmetic uses float/int32, never scalar half ops.
    template <typename CompareT, bool negate = false>
    __aicore__ inline CompareT ComparisonSeed() const
    {
        if constexpr (std::is_same<T, uint8_t>::value) {
            const float value = static_cast<float>(static_cast<int32_t>(seedValue));
            return static_cast<CompareT>(negate ? -value : value);
        } else {
            return negate ? -seedValue : seedValue;
        }
    }

    template <typename CompareT>
    __aicore__ inline void CompareToleranceBits(LocalTensor<CompareT> delta, uint32_t count)
    {
        PipeBarrier<PIPE_V>();
        const CompareT lower = static_cast<CompareT>(std::is_same<T, uint8_t>::value ? -static_cast<float>(loDiffI) :
                                                                                       -loDiffF);
        const CompareT upper = static_cast<CompareT>(std::is_same<T, uint8_t>::value ? static_cast<float>(upDiffI) :
                                                                                       upDiffF);
        if (std::is_same<T, uint8_t>::value ? loDiffI == upDiffI : loDiffF == upDiffF) {
            Abs(delta, delta, count);
            PipeBarrier<PIPE_V>();
            CompareScalar(scanCheckA.Get<uint8_t>(), delta, upper, CMPMODE::LE, count);
        } else {
            CompareScalar(scanCheckA.Get<uint8_t>(), delta, lower, CMPMODE::GE, count);
            CompareScalar(scanCheckB.Get<uint8_t>(), delta, upper, CMPMODE::LE, count);
            PipeBarrier<PIPE_V>();
            And(scanCheckA.Get<uint16_t>(), scanCheckA.Get<uint16_t>(), scanCheckB.Get<uint16_t>(),
                static_cast<int32_t>(count / 16U));
        }
    }

    // Count the reachable prefix/suffix from an accepted endpoint by
    // testing directed edges (NEIGHBOR) or seed eligibility (FIXED_RANGE).
    // INT32 retains exact integer extrema and scalar fallback.
    __aicore__ inline uint32_t ProveRowEdges(LocalTensor<T> chunk, uint32_t row, uint32_t first, uint32_t count,
                                             bool reverse)
    {
        // uint8 pixels, differences [-255,255], and host-quantized
        // tolerances are exact in half. FLOAT32 keeps float subtraction.
        using CompareT = typename std::conditional<std::is_same<T, uint8_t>::value, half, float>::type;
        // Two aligned UB arrays represent adjacent ORIGINAL pixels.
        // Padding initializes the unused final lane; its comparison bit
        // is ignored. Every other directed horizontal edge must pass.
        const bool fixedRange = diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE;
        LocalTensor<T> next;
        LocalTensor<CompareT> currentValues;
        LocalTensor<CompareT> nextValues = scanAccMax.Get<CompareT>();
        if constexpr (std::is_same<T, uint8_t>::value) {
            currentValues = scanStageHalf.Get<half>()[512U];
        } else {
            currentValues = chunk;
        }
        if (!fixedRange) {
            LocalTensor<T> nextLocal = copyQue.AllocTensor<T>();
            DataCopyExtParams params{1, (count - 1U) * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padding{true, 0, 1, 0};
            DataCopyPad(nextLocal, xGm[static_cast<uint64_t>(row) * width + first + 1U], params, padding);
            copyQue.EnQue(nextLocal);
            next = copyQue.DeQue<T>();
            if constexpr (std::is_same<T, uint8_t>::value) {
                Cast(nextValues, next, RoundMode::CAST_NONE, count);
            } else {
                nextValues = next;
            }
        }
        LocalTensor<CompareT> delta = scanAccMin.Get<CompareT>();
        PipeBarrier<PIPE_V>();
        if (seedEquality) {
            // Equality is equivalent to zero subtraction tolerance only
            // for a finite seed. Infinities and NaNs keep the original
            // subtraction path, including its unordered comparisons.
            CompareScalar(scanCheckA.Get<uint8_t>(), currentValues, ComparisonSeed<CompareT>(), CMPMODE::EQ, count);
        } else {
            if (fixedRange)
                Adds(delta, currentValues, ComparisonSeed<CompareT, true>(), count);
            else if (reverse)
                Sub(delta, currentValues, nextValues, count);
            else
                Sub(delta, nextValues, currentValues, count);
            CompareToleranceBits(delta, count);
        }
        const TEventID vectorEvent = pipePtr->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(vectorEvent);
        WaitFlag<HardEvent::V_S>(vectorEvent);
        // Preserve the reachable prefix instead of discarding the entire
        // block when one later edge fails. Reverse extension consumes the
        // suffix, in the opposite directed order. A failed edge still goes
        // through AcceptHorizontal/BfsAccept in the caller.
        uint32_t accepted = count;
        LocalTensor<uint64_t> valid = scanCheckA.Get<uint64_t>();
        for (uint32_t word = 0U; word < count / 64U; ++word) {
            const uint32_t i = reverse ? count / 64U - 1U - word : word;
            const uint64_t needed = !fixedRange && i + 1U == count / 64U ? 0x7fffffffffffffffULL :
                                                                           0xffffffffffffffffULL;
            const uint64_t failed = (~valid.GetValue(i)) & needed;
            if (failed == 0U)
                continue;
            uint32_t bit = reverse ? 63U : 0U;
            if (reverse) {
                while ((failed & (1ULL << bit)) == 0U)
                    --bit;
                accepted = count - 1U - (i * 64U + bit);
            } else {
                while ((failed & (1ULL << bit)) == 0U)
                    ++bit;
                accepted = i * 64U + bit + (fixedRange ? 0U : 1U);
            }
            break;
        }
        const TEventID scalarEvent = pipePtr->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarEvent);
        WaitFlag<HardEvent::S_V>(scalarEvent);
        if (!fixedRange)
            copyQue.FreeTensor(next);
        return accepted;
    }

    // The block includes an accepted endpoint and lies in an interval known
    // unvisited before this span. Previously accepted span pixels may overlap
    // the block. Return the proven pixel count from its directed endpoint;
    // INT32 uses exact extrema. Keep this
    // vector path out of the scalar extension loop's inlined code.
    __aicore__ __attribute__((noinline)) uint32_t UnblockedVectorPrefix(uint32_t row, uint32_t first, uint32_t count,
                                                                        bool reverse)
    {
        LocalTensor<uint8_t> local = copyQue.AllocTensor<uint8_t>();
        DataCopyExtParams params{1, count, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padding{false, 0, 0, 0};
        DataCopyPad(local, inputMaskGm[MaskIndex(row, first)], params, padding);
        copyQue.EnQue(local);
        LocalTensor<uint8_t> input = copyQue.DeQue<uint8_t>();
        LocalTensor<half> values = scanStageHalf.Get<half>()[512U];
        const uint32_t rounded = (count + 127U) / 128U * 128U;
        Duplicate(values, half(0.0f), rounded);
        PipeBarrier<PIPE_V>();
        Cast(values, input, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        CompareScalar(scanCheckA.Get<uint8_t>(), values, half(0.0f), CMPMODE::EQ, rounded);
        const TEventID ready = pipePtr->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(ready);
        WaitFlag<HardEvent::V_S>(ready);
        LocalTensor<uint64_t> valid = scanCheckA.Get<uint64_t>();
        uint32_t accepted = count;
        const uint32_t words = (count + 63U) / 64U;
        for (uint32_t word = 0U; word < words; ++word) {
            const uint32_t i = reverse ? words - 1U - word : word;
            const uint32_t lanes = count - i * 64U < 64U ? count - i * 64U : 64U;
            const uint64_t needed = lanes == 64U ? 0xffffffffffffffffULL : (1ULL << lanes) - 1U;
            const uint64_t failed = (~valid.GetValue(i)) & needed;
            if (failed == 0U)
                continue;
            uint32_t bit = reverse ? lanes - 1U : 0U;
            if (reverse) {
                while ((failed & (1ULL << bit)) == 0U)
                    --bit;
                accepted = count - 1U - (i * 64U + bit);
            } else {
                while ((failed & (1ULL << bit)) == 0U)
                    ++bit;
                accepted = i * 64U + bit;
            }
            break;
        }
        const TEventID consumed = pipePtr->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(consumed);
        WaitFlag<HardEvent::S_V>(consumed);
        copyQue.FreeTensor(input);
        return accepted;
    }

    __aicore__ inline uint32_t UnblockedPrefix(uint32_t row, uint32_t first, uint32_t count, bool reverse)
    {
        if (count >= 256U)
            return UnblockedVectorPrefix(row, first, count, reverse);
        const uint32_t begin = MaskIndex(row, first);
        const uint32_t end = begin + count;
        if (reverse) {
            uint32_t pos = end;
            while (pos > begin && (pos & 7U) != 0U) {
                if (inputMaskGm.GetValue(pos - 1U) != 0U)
                    return end - pos;
                --pos;
            }
            while (pos >= begin + 8U && inputMaskWords[pos / 8U - 1U] == 0U)
                pos -= 8U;
            while (pos > begin && inputMaskGm.GetValue(pos - 1U) == 0U)
                --pos;
            return end - pos;
        }
        uint32_t pos = begin;
        while (pos < end && (pos & 7U) != 0U) {
            if (inputMaskGm.GetValue(pos) != 0U)
                return pos - begin;
            ++pos;
        }
        while (pos + 8U <= end && inputMaskWords[pos / 8U] == 0U)
            pos += 8U;
        while (pos < end && inputMaskGm.GetValue(pos) == 0U)
            ++pos;
        return pos - begin;
    }

    __aicore__ __attribute__((noinline)) uint32_t ProveRowBlock(uint32_t row, uint32_t first, uint32_t count,
                                                                bool reverse)
    {
#if FLOOD_FILL_PROFILE
        const int64_t proofStart = GetSystemCycle();
        ++profileProofCalls;
#endif
        uint32_t accepted = 1U;
        if (RowComparisonReady(row)) {
#if FLOOD_FILL_PROFILE
            ++profileCachedProofs;
#endif
            accepted = ReadComparisonBits(row, first, count, reverse);
        } else {
            if (edgeBitsReady)
                RequestComparisonRow(row);
            using AccT = typename std::conditional<std::is_same<T, int32_t>::value, int32_t, float>::type;
            LocalTensor<T> local = copyQue.AllocTensor<T>();
            DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padding{false, 0, 0, 0};
            DataCopyPad(local, xGm[static_cast<uint64_t>(row) * width + first], params, padding);
            copyQue.EnQue(local);
            LocalTensor<T> chunk = copyQue.DeQue<T>();
            if constexpr (std::is_same<T, uint8_t>::value) {
                // Keep the same aligned staging view as the whole-image
                // implementation; the span cache lives in scanCheckB's tail.
                LocalTensor<half> halfChunk = scanStageHalf.Get<half>()[512U];
                Cast(halfChunk, chunk, RoundMode::CAST_NONE, count);
            }
            if constexpr (std::is_same<T, int32_t>::value) {
                LocalTensor<AccT> accMax = scanAccMax.Get<AccT>();
                LocalTensor<AccT> accMin = scanAccMin.Get<AccT>();
                Max(accMax, chunk, chunk, static_cast<int32_t>(count));
                Min(accMin, chunk, chunk, static_cast<int32_t>(count));
                AccT low = 0;
                AccT high = 0;
                ReduceAccumulators(accMax, accMin, low, high, count);
                accepted = ExtremaSatisfyCriterion(low, high) ? count : 1U;
            } else {
                accepted = ProveRowEdges(chunk, row, first, count, reverse);
            }
            copyQue.FreeTensor(chunk);
        }
#if FLOOD_FILL_PROFILE
        if (accepted != count)
            ++profileProofFailures;
#endif
        if (accepted == count || accepted >= 64U) {
            rowProofFailures = 0U;
        } else if (++rowProofFailures >= 4U) {
            // Connected regions can fail both sufficient tests (INT32
            // drift or paths through adjacent rows). Bound speculation;
            // after 32 known unvisited intervals, probe again in new areas.
            rowProofCooldown = 32U;
            rowProofFailures = 0U;
        }
#if FLOOD_FILL_PROFILE
        profileDetails[4] += GetSystemCycle() - proofStart;
#endif
        if constexpr (HAS_MASK_INPUT) {
            // A value proof only establishes a directed prefix from an
            // accepted endpoint. Intersect exactly that prefix with the
            // zero-mask run; no value shortcut may jump over an obstacle.
            if (accepted != 0U)
                accepted = UnblockedPrefix(row, reverse ? first + count - accepted : first, accepted, reverse);
        }
        return accepted;
    }

    uint32_t rowProofFailures = 0U;
    uint32_t rowProofCooldown = 0U;

    // Keep packed half comparison outputs on complete 128-lane repeats.
    // Float comparisons and integer extrema use 64-element granularity.
    static constexpr uint32_t kRowBlockGranularity = std::is_same<T, uint8_t>::value ? 128U : 64U;

    __aicore__ inline uint32_t RowBlockLength(uint32_t available) const
    {
        if constexpr (!std::is_same<T, int32_t>::value) {
            return available >= 4096U ? 4096U : available / kRowBlockGranularity * kRowBlockGranularity;
        }
        uint32_t count = 64U;
        while (count < 4096U && count * 2U <= available)
            count *= 2U;
        return count;
    }

    // BFS-complete acceptance test (exact closure of the CPU reference):
    // pixel (row, col) is accepted iff any already-visited neighbor
    // (4- or 8-connectivity) has an ORIGINAL value within tolerance.
    //   NEIGHBOR:    |x[row][col] - x[nr][nc]| within tolerance
    //   FIXED_RANGE: |x[row][col] - seedValue| within tolerance
    __aicore__ inline bool BfsAccept(uint32_t row, uint32_t col, bool requireNeighbor = false)
    {
        if (IsBlocked(row, col))
            return false;
        if (diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE) {
            if (!WithinTolerance(xGm.GetValue(row * width + col), seedValue))
                return false;
            // Scanline callers already establish adjacency. A full-image
            // overflow sweep must explicitly require a visited neighbor.
            if (!requireNeighbor)
                return true;
        }

        static const int32_t deltas4[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        static const int32_t deltas8[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

        const auto* deltas = connectivity == 8U ? deltas8 : deltas4;
        const uint32_t deltaCount = connectivity == 8U ? 8U : 4U;

        const T value = xGm.GetValue(row * width + col);

        for (uint32_t d = 0U; d < deltaCount; ++d) {
            const int32_t nr = static_cast<int32_t>(row) + deltas[d][0];
            const int32_t nc = static_cast<int32_t>(col) + deltas[d][1];

            if (nr < 0 || nr >= static_cast<int32_t>(height) || nc < 0 || nc >= static_cast<int32_t>(width)) {
                continue;
            }

            if (!IsVisited(static_cast<uint32_t>(nr), static_cast<uint32_t>(nc))) {
                continue;
            }

            if (diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE ||
                WithinTolerance(value, xGm.GetValue(static_cast<uint32_t>(nr) * width + static_cast<uint32_t>(nc)))) {
                return true;
            }
        }

        return false;
    }

    // A queued span is already visited. Test only edges from this span;
    // every other visited span will probe its own outgoing edges. Together
    // with horizontal extension this processes every reachable edge, without
    // repeatedly consulting the bitmap for up to eight unrelated neighbors.
    // Dropped spans on overflow are covered by the unchanged full BFS sweep.
    __aicore__ inline bool AcceptFromSpan(uint32_t row, uint32_t col, uint32_t parentRow, uint32_t parentLeft,
                                          uint32_t parentRight)
    {
        if (IsBlocked(row, col))
            return false;
        const T candidate = xGm.GetValue(row * width + col);
        if (diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE) {
            return WithinTolerance(candidate, seedValue);
        }
        if (col >= parentLeft && col <= parentRight &&
            WithinTolerance(candidate, xGm.GetValue(parentRow * width + col)))
            return true;
        if (connectivity == 8U) {
            if (col > parentLeft && col - 1U <= parentRight &&
                WithinTolerance(candidate, xGm.GetValue(parentRow * width + col - 1U)))
                return true;
            if (col + 1U >= parentLeft && col < parentRight &&
                WithinTolerance(candidate, xGm.GetValue(parentRow * width + col + 1U)))
                return true;
        }
        return false;
    }

    // Locate the first eligible vertical edge in a block (4-connectivity).
    // This reads ORIGINAL x only: it neither snapshots the visited bitmap nor
    // caches decisions across span mutations. Return end when no edge passes.
    // All count lanes are real pixels. CompareScalar's 256B count alignment
    // requires multiples of 128 for uint8->half, 64 for float; at most 4096.
    // Ignore bits before searchFirst when overlapping a processed prefix.
    __aicore__ __attribute__((noinline)) uint32_t FindVerticalCandidate(uint32_t row, uint32_t parentRow,
                                                                        uint32_t first, uint32_t count,
                                                                        uint32_t searchFirst)
    {
#if FLOOD_FILL_PROFILE
        const int64_t verticalStart = GetSystemCycle();
#endif
        if (RowComparisonReady(row) && diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE) {
            const uint32_t found = FindFixedComparisonCandidate(row, first, count, searchFirst);
#if FLOOD_FILL_PROFILE
            profileDetails[5] += GetSystemCycle() - verticalStart;
#endif
            return found;
        }
        if (edgeBitsReady && diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE)
            RequestComparisonRow(row);
        using CompareT = typename std::conditional<std::is_same<T, uint8_t>::value, half, float>::type;
        LocalTensor<T> inputLocal = copyQue.AllocTensor<T>();
        DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        DataCopyPad(inputLocal, xGm[static_cast<uint64_t>(row) * width + first], params, padding);
        copyQue.EnQue(inputLocal);
        LocalTensor<T> input = copyQue.DeQue<T>();
        LocalTensor<CompareT> candidate;
        LocalTensor<CompareT> reference = scanAccMax.Get<CompareT>();
        if constexpr (std::is_same<T, uint8_t>::value) {
            candidate = scanStageHalf.Get<half>()[512U];
            Cast(candidate, input, RoundMode::CAST_NONE, count);
        } else {
            candidate = input;
        }

        LocalTensor<T> parent;
        const bool fixedRange = diffMode == FLOODFILL_DIFF_MODE_FIXED_RANGE;
        if (!fixedRange) {
            LocalTensor<T> parentLocal = copyQue.AllocTensor<T>();
            DataCopyPad(parentLocal, xGm[static_cast<uint64_t>(parentRow) * width + first], params, padding);
            copyQue.EnQue(parentLocal);
            parent = copyQue.DeQue<T>();
            if constexpr (std::is_same<T, uint8_t>::value) {
                Cast(reference, parent, RoundMode::CAST_NONE, count);
            } else {
                reference = parent;
            }
        }
        LocalTensor<CompareT> delta = scanAccMin.Get<CompareT>();
        PipeBarrier<PIPE_V>();
        if (seedEquality) {
            CompareScalar(scanCheckA.Get<uint8_t>(), candidate, ComparisonSeed<CompareT>(), CMPMODE::EQ, count);
        } else {
            if (fixedRange)
                Adds(delta, candidate, ComparisonSeed<CompareT, true>(), count);
            else
                Sub(delta, candidate, reference, count);
            CompareToleranceBits(delta, count);
        }
        const TEventID vectorEvent = pipePtr->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(vectorEvent);
        WaitFlag<HardEvent::V_S>(vectorEvent);
        uint32_t found = first + count;
        LocalTensor<uint64_t> valid = scanCheckA.Get<uint64_t>();
        const uint32_t skipped = searchFirst - first;
        for (uint32_t word = skipped / 64U; word < count / 64U; ++word) {
            uint64_t bits = valid.GetValue(word);
            if (word == skipped / 64U)
                bits &= 0xffffffffffffffffULL << (skipped % 64U);
            if (bits == 0U)
                continue;
            const uint32_t bit = static_cast<uint32_t>(GetSFFValue<1>(bits));
            found = first + word * 64U + bit;
            break;
        }
        const TEventID scalarEvent = pipePtr->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarEvent);
        WaitFlag<HardEvent::S_V>(scalarEvent);
        if (!fixedRange)
            copyQue.FreeTensor(parent);
        copyQue.FreeTensor(input);
#if FLOOD_FILL_PROFILE
        profileDetails[5] += GetSystemCycle() - verticalStart;
#endif
        return found;
    }

    // Record a span in the GM span table. Capacity is
    // 2*max(H,W)+256; on overflow the entry is dropped and the fill
    // falls back to a full sweep closure (FallbackSweep), which
    // guarantees correctness on adversarial span-explosive patterns.
    __aicore__ inline void PushSpan(uint32_t row, uint32_t left, uint32_t right, uint16_t skippedRows = 0U)
    {
        if (spanCount >= stackCapacity) {
            overflowed = true;
            return;
        }

        spanTable[spanCount].y = static_cast<uint16_t>(row);
        spanTable[spanCount].left = static_cast<uint16_t>(left);
        spanTable[spanCount].right = static_cast<uint16_t>(right);
        spanTable[spanCount].pad = skippedRows;
        filledPixelCount += static_cast<uint64_t>(right) - left + 1U;
        ++spanCount;
    }

    static constexpr uint32_t kVisitedCacheSlots = 256U;

    // A cache of intervals proven visited. It is only a shortcut:
    // misses still use the authoritative GM bitmap. Never merge across
    // an unvisited gap. Reuse pre-check scratch after its vector work
    // has completed; all accesses here are scalar UB operations.
    __aicore__ inline void RememberVisitedSpan(uint32_t row, uint32_t left, uint32_t right)
    {
        LocalTensor<uint32_t> cache = scanCheckB.Get<uint32_t>()[1024U];
        const uint32_t slot = (row & (kVisitedCacheSlots - 1U)) * 3U;
        if (cache.GetValue(slot) == row) {
            const uint32_t oldLeft = cache.GetValue(slot + 1U);
            const uint32_t oldRight = cache.GetValue(slot + 2U);
            if (left <= oldRight + 1U && oldLeft <= right + 1U) {
                if (oldLeft < left)
                    left = oldLeft;
                if (oldRight > right)
                    right = oldRight;
            }
        }
        cache.SetValue(slot, row);
        cache.SetValue(slot + 1U, left);
        cache.SetValue(slot + 2U, right);
    }

    __aicore__ inline uint32_t SkipVisitedSpan(uint32_t row, uint32_t col)
    {
        const uint8_t rowState = lazyVisited ? rowFlagsBase[row] : 0U;
        if (rowState == 1U) {
            return (rowIntervalsBase[row] >> 16U) + 1U;
        }
        LocalTensor<uint32_t> cache = scanCheckB.Get<uint32_t>()[1024U];
        const uint32_t slot = (row & (kVisitedCacheSlots - 1U)) * 3U;
        if (cache.GetValue(slot) == row && col >= cache.GetValue(slot + 1U) && col <= cache.GetValue(slot + 2U)) {
#if FLOOD_FILL_PROFILE
            ++profileVisitedCacheHits;
#endif
            return cache.GetValue(slot + 2U) + 1U;
        }
#if FLOOD_FILL_PROFILE
        ++profileVisitedCacheMisses;
#endif
        if (rowState == 2U) {
            // Additional spans materialize, but never invalidate, the
            // original committed interval. It remains a proven visited
            // subset; do not infer that pixels outside it are unvisited.
            const uint32_t interval = rowIntervalsBase[row];
            const uint32_t left = interval & 0xffffU;
            const uint32_t right = interval >> 16U;
            if (col >= left && col <= right) {
#if FLOOD_FILL_PROFILE
                profileVisitedMissPixels += right + 1U - col;
#endif
                return right + 1U;
            }
        }
        // Cache collisions or multiple intervals must not turn a long
        // visited run into one scalar probe per pixel. Every committed
        // visited byte is exactly 1; stop at the first zero or row end.
        // Do not cache this suffix as a sole span: its left edge is unknown.
        uint32_t next = col + 1U;
        uint32_t index = hasMask ? MaskIndex(row, next) : row * width + next;
        while (next < width && (index & 7U) != 0U) {
            if (!IsVisited(row, next)) {
#if FLOOD_FILL_PROFILE
                profileVisitedMissPixels += next - col;
#endif
                return next;
            }
            ++next;
            ++index;
        }
        while (next + 7U < width && visitedWordBase[index / 8U] == 0x0101010101010101ULL) {
            next += 8U;
            index += 8U;
        }
        while (next < width && IsVisited(row, next))
            ++next;
#if FLOOD_FILL_PROFILE
        profileVisitedMissPixels += next - col;
#endif
        return next;
    }

    // Fill a maximal in-row span starting at column col (must already
    // be accepted by the caller). Only marks visited: y is written
    // after the fill by WriteRow (single MTE3 write path per pixel).
    __aicore__ inline void FillAndPushSpan(uint32_t row, uint32_t col, uint32_t& outLeft, uint32_t& outRight,
                                           uint32_t parentLeft = 0U, uint32_t parentRight = 0U, int32_t toParent = 0)
    {
        uint32_t left = col;
        uint32_t right = col;

        // State 0 is untouched, 1 has exactly one committed span, and 2
        // means two or more. When the sole span is still in our cache,
        // the intervals outside it are known unvisited. Never infer this
        // from a cache entry alone: it may omit other disjoint spans.
        const uint8_t rowState = rowFlagsBase[row];
        const bool emptyRow = rowState == 0U;
        bool clearRange = emptyRow;
        uint32_t rangeLeft = 0U;
        uint32_t rangeRight = width - 1U;
        if (rowState == 1U) {
            LocalTensor<uint32_t> cache = scanCheckB.Get<uint32_t>()[1024U];
            const uint32_t slot = (row & (kVisitedCacheSlots - 1U)) * 3U;
            if (cache.GetValue(slot) == row) {
                const uint32_t oldLeft = cache.GetValue(slot + 1U);
                const uint32_t oldRight = cache.GetValue(slot + 2U);
                if (col < oldLeft) {
                    rangeRight = oldLeft - 1U;
                    clearRange = true;
                } else if (col > oldRight) {
                    rangeLeft = oldRight + 1U;
                    clearRange = true;
                }
            }
        }
        const uint32_t rangeWidth = rangeRight - rangeLeft + 1U;
        const bool tryProof = clearRange && rowProofCooldown == 0U;
        if (clearRange && rowProofCooldown != 0U)
            --rowProofCooldown;
        constexpr uint32_t prefixSteps = std::is_same<T, int32_t>::value ? 16U : 4U;
        T leftValue = xGm.GetValue(row * width + col);
        T rightValue = leftValue;

        bool leftBlocked = false;
        bool rightBlocked = false;
        if (tryProof) {
            uint32_t steps = 0U;
            for (; steps < prefixSteps && left > rangeLeft; ++steps) {
                if (!AcceptHorizontal(row, left - 1U, leftValue)) {
                    leftBlocked = true;
                    break;
                }
                --left;
            }
            if (steps == prefixSteps && !leftBlocked) {
                bool first = true;
                while (left > rangeLeft) {
                    const bool initial = first;
                    uint32_t blockEnd = first ? col : left;
                    first = false;
                    uint32_t count;
                    if constexpr (std::is_same<T, int32_t>::value) {
                        if (blockEnd + 1U - rangeLeft < 64U)
                            break;
                        count = RowBlockLength(blockEnd + 1U - rangeLeft);
                    } else {
                        if (rangeWidth < kRowBlockGranularity)
                            break;
                        const uint32_t available = blockEnd + 1U - rangeLeft;
                        count = available >= 4096U ? 4096U :
                                                     (available + kRowBlockGranularity - 1U) / kRowBlockGranularity *
                                                         kRowBlockGranularity;
                        if (count > rangeWidth)
                            count = RowBlockLength(available);
                        if (count < kRowBlockGranularity)
                            break;
                        if (blockEnd + 1U - rangeLeft < count)
                            blockEnd = rangeLeft + count - 1U;
                        // The reverse chain's endpoint must already have
                        // been accepted; overlap never crosses this span.
                        if (blockEnd > right) {
                            count = RowBlockLength(available);
                            if (count < kRowBlockGranularity)
                                break;
                            blockEnd = rangeLeft + available - 1U;
                        }
                    }
                    const uint32_t previousLeft = left;
                    const uint32_t accepted = ProveRowBlock(row, blockEnd + 1U - count, count, true);
                    if (accepted > 0U && blockEnd + 1U - accepted < left) {
                        left = blockEnd + 1U - accepted;
                        leftValue = xGm.GetValue(row * width + left);
                    }
                    // The scalar prefix may have used a path through an
                    // adjacent row. If its internal edge blocks this first
                    // proof, retry from the already accepted boundary once.
                    if (accepted != count && (!initial || left < previousLeft))
                        break;
                }
            }
        }
        // Failed sufficient proofs enter the original scalar loop once.
        // There is no per-pixel speculation branch on long gradients.
        if (!leftBlocked) {
            while (left > rangeLeft && (clearRange || !IsVisited(row, left - 1U))) {
                if (!AcceptHorizontal(row, left - 1U, leftValue))
                    break;
                --left;
            }
        }

        if (tryProof) {
            uint32_t steps = 0U;
            for (; steps < prefixSteps && right < rangeRight; ++steps) {
                if (!AcceptHorizontal(row, right + 1U, rightValue)) {
                    rightBlocked = true;
                    break;
                }
                ++right;
            }
            if (steps == prefixSteps && !rightBlocked) {
                bool first = true;
                while (right < rangeRight) {
                    const bool initial = first;
                    uint32_t blockStart = first ? col : right;
                    first = false;
                    uint32_t count;
                    if constexpr (std::is_same<T, int32_t>::value) {
                        if (rangeRight + 1U - blockStart < 64U)
                            break;
                        count = RowBlockLength(rangeRight + 1U - blockStart);
                    } else {
                        if (rangeWidth < kRowBlockGranularity)
                            break;
                        const uint32_t available = rangeRight + 1U - blockStart;
                        count = available >= 4096U ? 4096U :
                                                     (available + kRowBlockGranularity - 1U) / kRowBlockGranularity *
                                                         kRowBlockGranularity;
                        if (count > rangeWidth)
                            count = RowBlockLength(available);
                        if (count < kRowBlockGranularity)
                            break;
                        if (blockStart + count > rangeRight + 1U)
                            blockStart = rangeRight + 1U - count;
                        // Includes an accepted endpoint and every real edge
                        // to the new boundary, with no out-of-row padding.
                        if (blockStart < left) {
                            count = RowBlockLength(available);
                            if (count < kRowBlockGranularity)
                                break;
                            blockStart = rangeRight + 1U - available;
                        }
                    }
                    const uint32_t previousRight = right;
                    const uint32_t accepted = ProveRowBlock(row, blockStart, count, false);
                    if (accepted > 0U && blockStart + accepted - 1U > right) {
                        right = blockStart + accepted - 1U;
                        rightValue = xGm.GetValue(row * width + right);
                    }
                    if (accepted != count && (!initial || right > previousRight))
                        break;
                }
            }
        }
        if (!rightBlocked) {
            while (right < rangeRight && (clearRange || !IsVisited(row, right + 1U))) {
                if (!AcceptHorizontal(row, right + 1U, rightValue))
                    break;
                ++right;
            }
        }

        // During extension the accepted horizontal neighbor is held in
        // a register. If its tolerance test fails, its absent visited
        // bit cannot change BfsAccept's answer. All other neighboring
        // rows are already committed. Commit this span before probing
        // any other row, preserving the original reachability closure.
#if FLOOD_FILL_PROFILE
        if (right - left + 1U >= 256U) {
            if (!emptyRow) {
                ++profileLongRevisitSpans;
                profileLongRevisitPixels += right - left + 1U;
            } else {
                ++profileLongUntouchedSpans;
                if (!tryProof)
                    ++profileLongCooledSpans;
            }
        }
#endif
#if FLOOD_FILL_PROFILE
        const int64_t visitedStart = GetSystemCycle();
#endif
        MarkVisitedSpan(row, left, right);
#if FLOOD_FILL_PROFILE
        profileDetails[6] += GetSystemCycle() - visitedStart;
#endif
        RememberVisitedSpan(row, left, right);
        // Every parent pixel in this probe interval is already visited.
        // This is a coverage proof, independent of tolerance/dtype and of
        // whether other spans are later added to either row.
        const uint32_t diagonal = connectivity == 8U ? 1U : 0U;
        const uint32_t probeLeft = left > diagonal ? left - diagonal : 0U;
        const uint32_t probeRight = right + diagonal < width ? right + diagonal : width - 1U;
        const uint16_t skippedRows = toParent != 0 && parentLeft <= probeLeft && parentRight >= probeRight ?
                                         (toParent < 0 ? 1U : 2U) :
                                         0U;
        PushSpan(row, left, right, skippedRows);

        outLeft = left;
        outRight = right;
    }

    // Core span-stack scanline fill. Every popped span probes both
    // adjacent rows over [L - span8, R + span8]; each unvisited and
    // pixel reachable directly from that span becomes a maximal in-row span. This
    // visits every pixel a constant number of times -> O(filled area),
    // independent of total image size (OpenCV span-fill closure).
    __aicore__ __attribute__((noinline)) void RunSpanFill(bool allowPause = false)
    {
        if (!spanSearchStarted) {
            if (IsBlocked(seedRow, seedCol)) {
                spanSearchStarted = true;
                return;
            }
            // The whole-image reduction has finished. Later CompareScalar
            // results use only the first 512B of scanCheckB; reserve bytes
            // [4096,7168) for 256 scalar interval entries. UB size is unchanged.
            LocalTensor<uint32_t> spanCache = scanCheckB.Get<uint32_t>()[1024U];
            for (uint32_t slot = 0U; slot < kVisitedCacheSlots; ++slot)
                spanCache.SetValue(slot * 3U, 0xffffffffU);
            // The seed pixel itself has no visited neighbor: it is always
            // accepted (reference-free).
            uint32_t seedLeft = 0U;
            uint32_t seedRight = 0U;
            FillAndPushSpan(seedRow, seedCol, seedLeft, seedRight);

            spanSearchStarted = true;
        }
        const uint8_t span8 = connectivity == 8U ? 1U : 0U;

        // Span table with read pointer (new spans appended behind).
        spanSearchPaused = false;

        while (spanReadPtr < spanCount) {
            // Same 64-bit layout used by parallel adoption and sorting.
            const SpanWord entry = reinterpret_cast<__gm__ SpanWord*>(spanTable)[spanReadPtr++];
            const uint16_t skippedRows = static_cast<uint16_t>(entry >> 48U);
            // Both bits can be set only by the row-seed closure proof,
            // after full immutable predicate coverage has been published.
            if ((skippedRows & 3U) == 3U)
                continue;
            const uint32_t row = static_cast<uint32_t>(entry & 0xffffU);
            const uint32_t L = static_cast<uint32_t>((entry >> 16U) & 0xffffU);
            const uint32_t R = static_cast<uint32_t>((entry >> 32U) & 0xffffU);

            for (int32_t dy = -1; dy <= 1; dy += 2) {
                if ((skippedRows & (dy < 0 ? 1U : 2U)) != 0U)
                    continue;
                const int32_t probeRow = static_cast<int32_t>(row) + dy;

                if (probeRow < 0 || probeRow >= static_cast<int32_t>(height)) {
                    continue;
                }

                const uint32_t prow = static_cast<uint32_t>(probeRow);

                int32_t col = static_cast<int32_t>(L) - span8;
                if (col < 0) {
                    col = 0;
                }

                int32_t colEnd = static_cast<int32_t>(R) + span8;
                if (colEnd >= static_cast<int32_t>(width)) {
                    colEnd = static_cast<int32_t>(width) - 1;
                }

                uint32_t rejected = 0U;
                while (col <= colEnd) {
                    if constexpr (!std::is_same<T, int32_t>::value) {
                        if (connectivity == 4U && rejected >= 8U && colEnd - col + 1 >= 16 &&
                            R - L + 1U >= kRowBlockGranularity) {
                            const uint32_t searchFirst = static_cast<uint32_t>(col);
                            const uint32_t available = static_cast<uint32_t>(colEnd - col + 1);
                            uint32_t count = available >= 4096U ? 4096U :
                                                                  (available + kRowBlockGranularity - 1U) /
                                                                      kRowBlockGranularity * kRowBlockGranularity;
                            if (count > R - L + 1U)
                                count = RowBlockLength(R - L + 1U);
                            // Overlap an already processed prefix at the end
                            // of a parent span. All loads stay inside [L,R],
                            // and those old decision bits are explicitly masked.
                            const uint32_t first = count > available ? static_cast<uint32_t>(colEnd) + 1U - count :
                                                                       searchFirst;
                            col = static_cast<int32_t>(FindVerticalCandidate(prow, row, first, count, searchFirst));
                            if (static_cast<uint32_t>(col) == first + count)
                                continue;
                            rejected = 0U;
                        }
                    }
                    if (IsVisited(prow, col)) {
                        col = static_cast<int32_t>(SkipVisitedSpan(prow, static_cast<uint32_t>(col)));
                        continue;
                    }

                    if (AcceptFromSpan(prow, static_cast<uint32_t>(col), row, L, R)) {
                        uint32_t outLeft = 0U;
                        uint32_t outRight = 0U;
                        FillAndPushSpan(prow, static_cast<uint32_t>(col), outLeft, outRight, L, R, -dy);
                        col = static_cast<int32_t>(outRight) + 1;
                        rejected = 0U;
                    } else {
                        ++col;
                        ++rejected;
                    }
                }
            }
            // A cost-only decision: pause at a complete queue-entry boundary.
            // Tiny regions finish first; narrow work needs its separate cost gate.
            if (allowPause && !overflowed && (!edgeBitsReady || edgeRequestFirst != 0xffffffffU) &&
                spanReadPtr >= 128U && spanReadPtr < spanCount &&
                ((filledPixelCount >= static_cast<uint64_t>(spanCount) * (width < 1024U ? width / 4U : 256U) &&
                  filledPixelCount >= static_cast<uint64_t>(height) * width / 128U) ||
                 CanTryNarrowRowSeeds())) {
                if (!edgeBitsReady) {
                    RequestComparisonRow(spanTable[spanReadPtr].y);
                    RequestComparisonRow(spanTable[spanCount - 1U].y);
                }
                spanSearchPaused = true;
                return;
            }
        }

        if (overflowed) {
            MaterializeVisitedRows(true);
            lazyVisited = false;
            FallbackSweep();
        } else if (hasMask && coreCount == 1U) {
            MaterializeVisitedRows();
        }
        if (visitedOnesReady)
            fillQue.FreeTensor(visitedOnes);
    }

    // Stack-overflow fallback: full-image sweeps to the fixpoint (no
    // span pushes). Only reachable on adversarial span-explosive
    // inputs; completes the BFS closure unconditionally.
    __aicore__ inline void FallbackSweep()
    {
        bool changed = true;

        while (changed) {
            changed = false;

            for (uint32_t row = 0U; row < height; ++row) {
                for (uint32_t col = 0U; col < width; ++col) {
                    if (!IsVisited(row, col) && BfsAccept(row, col, true)) {
                        MarkVisited(row, col);
                        changed = true;
                    }
                }
            }
        }
    }

private:
    TPipe* pipePtr = nullptr;
    uint32_t coreId = 0U;
    uint32_t coreCount = 1U;
    GlobalTensor<uint64_t> sharedControl;
    __gm__ uint8_t* sharedBase = nullptr;
    TQue<QuePosition::VECIN, 2U> copyQue;
    TQue<QuePosition::VECOUT, 2U> fillQue;
    LocalTensor<uint8_t> visitedOnes;
    bool visitedOnesReady = false;

    // UB scratch of the whole-image homogeneity scan (WholeImageIsOneRegion).
    TBuf<TPosition::VECCALC> scanStageHalf;
    TBuf<TPosition::VECCALC> scanStageFloat;
    TBuf<TPosition::VECCALC> scanAccMax;
    TBuf<TPosition::VECCALC> scanAccMin;
    TBuf<TPosition::VECCALC> scanCheckA;
    TBuf<TPosition::VECCALC> scanCheckB;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<uint8_t> visitedGm;
    GlobalTensor<uint8_t> maskGm;
    GlobalTensor<uint8_t> inputMaskGm;
    __gm__ uint64_t* inputMaskWords = nullptr;
    bool hasMaskOutput = false;
    __gm__ uint64_t* visitedWordBase = nullptr;
    bool hasMask = false;

    __gm__ uint8_t* rowFlagsBase = nullptr;
    __gm__ uint32_t* rowIntervalsBase = nullptr;
    GlobalTensor<uint64_t> edgeForwardGm;
    GlobalTensor<uint64_t> edgeReverseGm;
    uint64_t edgePlaneBytes = 0U;
    bool edgeBitsReady = false;
    GlobalTensor<uint64_t> rowSeedInfo;
    bool rowSeedsAttempted = false;
    uint32_t rowSeedSortedCount = 0U;
    uint32_t edgeFirstRow = 0U;
    uint32_t edgeLastRow = 0U;
    uint32_t edgeRequestFirst = 0xffffffffU;
    uint32_t edgeRequestLast = 0U;
    bool spanSearchStarted = false;
    bool spanSearchPaused = false;
    uint32_t spanReadPtr = 0U;
    uint32_t deferredRows = 0U;
    bool lazyVisited = true;
    __gm__ FillSpan* spanTable = nullptr;
    uint32_t spanCount = 0;
    uint64_t filledPixelCount = 0;
    uint32_t stackCapacity = 0;
    bool overflowed = false;

    uint32_t height = 0;
    uint32_t width = 0;

    uint32_t seedRow = 0;
    uint32_t seedCol = 0;
    T seedValue{};

    T newValueT{};
    int32_t loDiffI = 0;
    int32_t upDiffI = 0;
    float loDiffF = 0.0F;
    float upDiffF = 0.0F;

    uint32_t connectivity = 4;
    uint32_t diffMode = FLOODFILL_DIFF_MODE_NEIGHBOR;
    bool seedEquality = false;
};

} // namespace NsFloodFill

#endif // FLOOD_FILL_H_
