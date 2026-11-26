/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_KERNEL_H_
#define BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_KERNEL_H_

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "batch_multi_class_non_max_suppression_tiling_data.h"

namespace BatchMultiClassNonMaxSuppressionOp {
using namespace AscendC;

constexpr int64_t kBoxCoordinateCount = 4;
constexpr int64_t kPaddedCountStride = 8;
constexpr int64_t kUnpaddedCountStride = 1;
constexpr uint32_t kGatherThreadNum32 = 1024;
constexpr uint32_t kGatherThreadNum64 = 512;
constexpr uint32_t kMergeThreadNum = 256;
constexpr uint32_t kMergeHeapCapacity = 512;
constexpr uint32_t kClassPositionBits = 10;
constexpr uint32_t kClassPositionMask = (1U << kClassPositionBits) - 1U;
constexpr float kNoCandidate = -(__builtin_inff());
constexpr float kMinPositive = 1.0e-12F;

template <bool Use32Bit>
struct GatherIndexType {
    using type = uint64_t;
};

template <>
struct GatherIndexType<true> {
    using type = uint32_t;
};

template <bool Use32Bit>
using GatherIndex = typename GatherIndexType<Use32Bit>::type;

// Fused inputs use scores [B, C, N] and boxes [B, q, 4, N].
// Use the 950 SIMT unit to compact one (batch, class) task into five contiguous FP32 arrays.
// Subsequent score reduction and IoU work is entirely vectorized on UB tiles.
template <typename T, bool Use32Bit>
__simt_vf__ __aicore__
__launch_bounds__(Use32Bit ? kGatherThreadNum32 : kGatherThreadNum64) inline void GatherClassInput(
    const __gm__ T* boxesYMin, const __gm__ T* boxesXMin, const __gm__ T* boxesYMax, const __gm__ T* boxesXMax,
    const __gm__ T* scores, const __gm__ int32_t* numValidBoxes, __gm__ float* stageScores, __gm__ float* stageYMin,
    __gm__ float* stageXMin, __gm__ float* stageYMax, __gm__ float* stageXMax, GatherIndex<Use32Bit> batchIndex,
    GatherIndex<Use32Bit> classIndex, GatherIndex<Use32Bit> boxesNum, GatherIndex<Use32Bit> classesNum,
    bool hasNumValidBoxes)
{
    using IndexT = GatherIndex<Use32Bit>;
    IndexT validBoxes = boxesNum;
    if (hasNumValidBoxes) {
        const int32_t validBoxesRaw = numValidBoxes[batchIndex];
        validBoxes = validBoxesRaw <= 0 ?
                         0 :
                         (static_cast<IndexT>(validBoxesRaw) > boxesNum ? boxesNum :
                                                                          static_cast<IndexT>(validBoxesRaw));
    }
    for (IndexT boxIndex = static_cast<IndexT>(threadIdx.x); boxIndex < boxesNum;
         boxIndex += static_cast<IndexT>(blockDim.x)) {
        const IndexT scoreOffset = (batchIndex * classesNum + classIndex) * boxesNum + boxIndex;
        stageScores[boxIndex] = boxIndex < validBoxes ? static_cast<float>(scores[scoreOffset]) : kNoCandidate;
        // Each [B, q, 4, N] field is contiguous. Their bases are computed
        // by AIV before the SIMT launch, so this path only needs a per-box
        // offset. Non-transpose inputs use the persistent gather below.
        stageYMin[boxIndex] = static_cast<float>(boxesYMin[boxIndex]);
        stageXMin[boxIndex] = static_cast<float>(boxesXMin[boxIndex]);
        stageYMax[boxIndex] = static_cast<float>(boxesYMax[boxIndex]);
        stageXMax[boxIndex] = static_cast<float>(boxesXMax[boxIndex]);
    }
}

// Keep one asynchronous VF task alive per AIV core for the complete class
// wave.  Re-launching a short VF task for every class exhausts the 950 VF
// task queue after several waves.  Each task writes to an independent staging
// slice, which the AIV NMS stage subsequently consumes.
template <typename T, bool Use32Bit>
__simt_vf__ __aicore__
__launch_bounds__(Use32Bit ? kGatherThreadNum32 : kGatherThreadNum64) inline void GatherAllClassInputs(
    const __gm__ T* boxes, const __gm__ T* scores, const __gm__ int32_t* numValidBoxes, __gm__ float* scratch,
    GatherIndex<Use32Bit> taskStart, GatherIndex<Use32Bit> taskStride, GatherIndex<Use32Bit> taskCount,
    GatherIndex<Use32Bit> boxesNum, GatherIndex<Use32Bit> classesNum, GatherIndex<Use32Bit> boxClassesNum,
    uint64_t scratchFieldStride, bool hasNumValidBoxes)
{
    using IndexT = GatherIndex<Use32Bit>;
    const uint64_t scratchFieldElements = scratchFieldStride / sizeof(float);
    for (IndexT taskIndex = taskStart; taskIndex < taskCount; taskIndex += taskStride) {
        const IndexT batchIndex = taskIndex / classesNum;
        const IndexT classIndex = taskIndex % classesNum;
        IndexT validBoxes = boxesNum;
        if (hasNumValidBoxes) {
            const int32_t validBoxesRaw = numValidBoxes[batchIndex];
            validBoxes = validBoxesRaw <= 0 ?
                             0 :
                             (static_cast<IndexT>(validBoxesRaw) > boxesNum ? boxesNum :
                                                                              static_cast<IndexT>(validBoxesRaw));
        }
        const uint64_t taskScratchOffset = static_cast<uint64_t>(taskIndex) * scratchFieldElements * 5;
        __gm__ float* const stageScores = scratch + taskScratchOffset;
        __gm__ float* const stageYMin = stageScores + scratchFieldElements;
        __gm__ float* const stageXMin = stageYMin + scratchFieldElements;
        __gm__ float* const stageYMax = stageXMin + scratchFieldElements;
        __gm__ float* const stageXMax = stageYMax + scratchFieldElements;
        const IndexT boxClass = boxClassesNum == 1 ? 0 : classIndex;
        for (IndexT boxIndex = static_cast<IndexT>(threadIdx.x); boxIndex < boxesNum;
             boxIndex += static_cast<IndexT>(blockDim.x)) {
            const IndexT scoreOffset = (batchIndex * boxesNum + boxIndex) * classesNum + classIndex;
            const IndexT boxOffset = ((batchIndex * boxesNum + boxIndex) * boxClassesNum + boxClass) * 4;
            stageScores[boxIndex] = boxIndex < validBoxes ? static_cast<float>(scores[scoreOffset]) : kNoCandidate;
            stageYMin[boxIndex] = static_cast<float>(boxes[boxOffset]);
            stageXMin[boxIndex] = static_cast<float>(boxes[boxOffset + 1]);
            stageYMax[boxIndex] = static_cast<float>(boxes[boxOffset + 2]);
            stageXMax[boxIndex] = static_cast<float>(boxes[boxOffset + 3]);
        }
    }
}

// Turn flattened class-result positions into the irregular box/class outputs.
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(kMergeThreadNum) inline void GatherMergedOutput(
    const __gm__ float* classBoxes, const __gm__ float* mergeScores, const __gm__ int32_t* mergeIndices,
    __gm__ T* nmsedBoxes, __gm__ T* nmsedScores, __gm__ T* nmsedClasses, __gm__ int32_t* nmsedNum, uint64_t batchIndex,
    uint64_t classesNum, uint64_t maxSizePerClass, uint64_t maxTotalSize, bool transposeBox)
{
    constexpr uint64_t kXMinCoordinate = 1;
    constexpr uint64_t kYMaxCoordinate = 2;
    constexpr uint64_t kXMaxCoordinate = 3;
    const uint64_t countOffset = batchIndex * (transposeBox ? kPaddedCountStride : kUnpaddedCountStride);
    const uint64_t validOutputCount = static_cast<uint64_t>(nmsedNum[countOffset]);
    if (transposeBox && threadIdx.x < kPaddedCountStride - kUnpaddedCountStride) {
        nmsedNum[countOffset + threadIdx.x + 1] = 0;
    }
    for (uint64_t outputIndex = static_cast<uint64_t>(threadIdx.x); outputIndex < maxTotalSize;
         outputIndex += static_cast<uint64_t>(blockDim.x)) {
        const uint64_t outputOffset = batchIndex * maxTotalSize + outputIndex;
        if (outputIndex < validOutputCount) {
            const float score = mergeScores[outputOffset];
            const uint64_t flatIndex = static_cast<uint64_t>(mergeIndices[outputOffset]);
            const uint64_t classIndex = flatIndex / maxSizePerClass;
            const uint64_t classPosition = flatIndex % maxSizePerClass;
            const uint64_t candidateOffset = (batchIndex * classesNum + classIndex) * maxSizePerClass + classPosition;
            const uint64_t boxOffset = candidateOffset * kBoxCoordinateCount;
            const uint64_t outputBoxOffset = transposeBox ?
                                                 batchIndex * maxTotalSize * kBoxCoordinateCount + outputIndex :
                                                 outputOffset * kBoxCoordinateCount;
            const uint64_t coordinateStride = transposeBox ? maxTotalSize : 1;
            nmsedBoxes[outputBoxOffset] = static_cast<T>(classBoxes[boxOffset]);
            nmsedBoxes[outputBoxOffset + kXMinCoordinate * coordinateStride] = static_cast<T>(
                classBoxes[boxOffset + kXMinCoordinate]);
            nmsedBoxes[outputBoxOffset + kYMaxCoordinate * coordinateStride] = static_cast<T>(
                classBoxes[boxOffset + kYMaxCoordinate]);
            nmsedBoxes[outputBoxOffset + kXMaxCoordinate * coordinateStride] = static_cast<T>(
                classBoxes[boxOffset + kXMaxCoordinate]);
            nmsedScores[outputOffset] = static_cast<T>(score);
            nmsedClasses[outputOffset] = static_cast<T>(classIndex);
        } else {
            const uint64_t outputBoxOffset = transposeBox ?
                                                 batchIndex * maxTotalSize * kBoxCoordinateCount + outputIndex :
                                                 outputOffset * kBoxCoordinateCount;
            const uint64_t coordinateStride = transposeBox ? maxTotalSize : 1;
            nmsedBoxes[outputBoxOffset] = static_cast<T>(0);
            nmsedBoxes[outputBoxOffset + kXMinCoordinate * coordinateStride] = static_cast<T>(0);
            nmsedBoxes[outputBoxOffset + kYMaxCoordinate * coordinateStride] = static_cast<T>(0);
            nmsedBoxes[outputBoxOffset + kXMaxCoordinate * coordinateStride] = static_cast<T>(0);
            nmsedScores[outputOffset] = static_cast<T>(0);
            nmsedClasses[outputOffset] = static_cast<T>(0);
        }
    }
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool HeapEntryPrecedes(float lhsScore,
                                                                                        int32_t lhsIndex,
                                                                                        float rhsScore,
                                                                                        int32_t rhsIndex)
{
    return lhsScore > rhsScore || (lhsScore == rhsScore && lhsIndex < rhsIndex);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SiftClassHeap(__ubuf__ float* heapScores,
                                                                                    __ubuf__ int32_t* heapIndices,
                                                                                    uint64_t heapSize, uint64_t root)
{
    while (true) {
        const uint64_t left = root * 2 + 1;
        if (left >= heapSize) {
            return;
        }
        uint64_t best = root;
        if (HeapEntryPrecedes(heapScores[left], heapIndices[left], heapScores[best], heapIndices[best])) {
            best = left;
        }
        const uint64_t right = left + 1;
        if (right < heapSize &&
            HeapEntryPrecedes(heapScores[right], heapIndices[right], heapScores[best], heapIndices[best])) {
            best = right;
        }
        if (best == root) {
            return;
        }
        const float score = heapScores[root];
        heapScores[root] = heapScores[best];
        heapScores[best] = score;
        const int32_t index = heapIndices[root];
        heapIndices[root] = heapIndices[best];
        heapIndices[best] = index;
        root = best;
    }
}

// ProcessClass already leaves every class sorted. Merge their heads without
// relying on TopK's value/index association.
__simt_vf__ __aicore__ __launch_bounds__(kMergeThreadNum) inline void MergeClassOutput(
    const __gm__ float* classScores, __gm__ float* classPositions, __gm__ float* mergeScores,
    __gm__ int32_t* mergeIndices, __gm__ int32_t* nmsedNum, __ubuf__ float* heapScores, __ubuf__ int32_t* heapIndices,
    uint64_t batchIndex, uint64_t classesNum, uint64_t maxSizePerClass, uint64_t maxTotalSize, bool transposeBox)
{
    const uint64_t classBase = batchIndex * classesNum;
    const uint64_t outputBase = batchIndex * maxTotalSize;
    if (classesNum <= kMergeHeapCapacity) {
        for (uint64_t classIndex = static_cast<uint64_t>(threadIdx.x); classIndex < classesNum;
             classIndex += static_cast<uint64_t>(blockDim.x)) {
            heapIndices[classIndex] = static_cast<int32_t>(classIndex << kClassPositionBits);
            heapScores[classIndex] = classScores[(classBase + classIndex) * maxSizePerClass];
        }
        asc_syncthreads();
        if (threadIdx.x != 0) {
            return;
        }
        for (int64_t root = static_cast<int64_t>(classesNum / 2) - 1; root >= 0; --root) {
            SiftClassHeap(heapScores, heapIndices, classesNum, static_cast<uint64_t>(root));
        }
        uint64_t validOutputCount = 0;
        for (; validOutputCount < maxTotalSize && heapScores[0] != kNoCandidate; ++validOutputCount) {
            const uint32_t packedIndex = static_cast<uint32_t>(heapIndices[0]);
            const uint64_t classIndex = packedIndex >> kClassPositionBits;
            const uint64_t classPosition = packedIndex & kClassPositionMask;
            mergeScores[outputBase + validOutputCount] = heapScores[0];
            mergeIndices[outputBase + validOutputCount] = static_cast<int32_t>(classIndex * maxSizePerClass +
                                                                               classPosition);
            const uint64_t nextPosition = classPosition + 1;
            if (nextPosition < maxSizePerClass) {
                heapIndices[0] += 1;
                heapScores[0] = classScores[(classBase + classIndex) * maxSizePerClass + nextPosition];
            } else {
                heapScores[0] = kNoCandidate;
            }
            SiftClassHeap(heapScores, heapIndices, classesNum, 0);
        }
        nmsedNum[batchIndex * (transposeBox ? kPaddedCountStride : kUnpaddedCountStride)] = static_cast<int32_t>(
            validOutputCount);
        return;
    }

    // The fixed UB heap covers normal class counts. For larger shapes, reuse
    // the no-longer-needed count array as per-class cursors.
    if (threadIdx.x == 0) {
        for (uint64_t classIndex = 0; classIndex < classesNum; ++classIndex) {
            classPositions[classBase + classIndex] = 0.0F;
        }
        uint64_t validOutputCount = 0;
        for (; validOutputCount < maxTotalSize; ++validOutputCount) {
            int32_t bestClass = -1;
            float bestScore = kNoCandidate;
            for (uint64_t classIndex = 0; classIndex < classesNum; ++classIndex) {
                const uint64_t position = static_cast<uint64_t>(classPositions[classBase + classIndex]);
                const float score = position < maxSizePerClass ?
                                        classScores[(classBase + classIndex) * maxSizePerClass + position] :
                                        kNoCandidate;
                if (bestClass < 0 || score > bestScore ||
                    (score == bestScore && classIndex < static_cast<uint64_t>(bestClass))) {
                    bestClass = static_cast<int32_t>(classIndex);
                    bestScore = score;
                }
            }
            if (bestScore == kNoCandidate) {
                break;
            }
            const uint64_t position = static_cast<uint64_t>(classPositions[classBase + bestClass]);
            mergeScores[outputBase + validOutputCount] = bestScore;
            mergeIndices[outputBase + validOutputCount] = bestClass * static_cast<int32_t>(maxSizePerClass) +
                                                          static_cast<int32_t>(position);
            classPositions[classBase + bestClass] = static_cast<float>(position + 1);
        }
        nmsedNum[batchIndex * (transposeBox ? kPaddedCountStride : kUnpaddedCountStride)] = static_cast<int32_t>(
            validOutputCount);
    }
}

template <typename T>
class BatchMultiClassNonMaxSuppressionKernel {
public:
    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR scores, GM_ADDR clipWindow, GM_ADDR numValidBoxes,
                                GM_ADDR nmsedBoxes, GM_ADDR nmsedScores, GM_ADDR nmsedClasses, GM_ADDR nmsedNum,
                                GM_ADDR workspace, const BatchMultiClassNonMaxSuppressionTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitWorkspace();
    __aicore__ inline void InitTileBuffers();
    __aicore__ inline void GatherClass(int64_t taskIndex);
    __aicore__ inline bool FilterStageBoxes(int64_t batchIndex);
    __aicore__ inline bool FindBestCandidate(int64_t& bestIndex, float& bestScore);
    __aicore__ inline void SuppressBySelectedBox(float yMin, float xMin, float yMax, float xMax);
    __aicore__ inline void ProcessClass(int64_t taskIndex);
    __aicore__ inline void PadClassScores(int64_t taskIndex, int64_t selectedCount);
    __aicore__ inline void StoreClassCount(int64_t taskIndex, int64_t selectedCount);
    __aicore__ inline void MergeBatch(int64_t batchIndex);
    __aicore__ inline void LoadTile(int64_t offset, int64_t count);
    __aicore__ inline void StoreTile(int64_t offset, int64_t count);
    __aicore__ inline void LoadScores(int64_t offset, int64_t count);
    __aicore__ inline void StoreScores(int64_t offset, int64_t count);
    __aicore__ inline void CopyIn(LocalTensor<float>& dst, const GlobalTensor<float>& src, int64_t offset,
                                  int64_t count);
    __aicore__ inline void CopyOut(GlobalTensor<float>& dst, int64_t offset, LocalTensor<float>& src, int64_t count);
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> scoreBuffer_;
    TBuf<QuePosition::VECCALC> yMinBuffer_;
    TBuf<QuePosition::VECCALC> xMinBuffer_;
    TBuf<QuePosition::VECCALC> yMaxBuffer_;
    TBuf<QuePosition::VECCALC> xMaxBuffer_;
    TBuf<QuePosition::VECCALC> temp0Buffer_;
    TBuf<QuePosition::VECCALC> temp1Buffer_;
    TBuf<QuePosition::VECCALC> temp2Buffer_;
    TBuf<QuePosition::VECCALC> temp3Buffer_;
    TBuf<QuePosition::VECCALC> reduceWorkBuffer_;
    TBuf<QuePosition::VECCALC> reduceOutputBuffer_;
    TBuf<QuePosition::VECCALC> compareMaskBuffer_;
    TBuf<QuePosition::VECCALC> mergeInputScoresBuffer_;
    TBuf<QuePosition::VECCALC> mergeInputIndicesBuffer_;

    LocalTensor<float> scoreLocal_;
    LocalTensor<float> yMinLocal_;
    LocalTensor<float> xMinLocal_;
    LocalTensor<float> yMaxLocal_;
    LocalTensor<float> xMaxLocal_;
    LocalTensor<float> temp0Local_;
    LocalTensor<float> temp1Local_;
    LocalTensor<float> temp2Local_;
    LocalTensor<float> temp3Local_;
    LocalTensor<float> reduceWorkLocal_;
    LocalTensor<float> reduceOutputLocal_;
    LocalTensor<uint8_t> compareMaskLocal_;
    LocalTensor<float> mergeInputScoresLocal_;
    LocalTensor<int32_t> mergeInputIndicesLocal_;

    GlobalTensor<T> clipWindowGm_;
    GlobalTensor<T> nmsedBoxesGm_;
    GlobalTensor<T> nmsedScoresGm_;
    GlobalTensor<T> nmsedClassesGm_;
    GlobalTensor<int32_t> nmsedNumGm_;
    GlobalTensor<float> stageScoresGm_;
    GlobalTensor<float> stageYMinGm_;
    GlobalTensor<float> stageXMinGm_;
    GlobalTensor<float> stageYMaxGm_;
    GlobalTensor<float> stageXMaxGm_;
    GlobalTensor<float> classBoxesGm_;
    GlobalTensor<float> classScoresGm_;
    GlobalTensor<float> classCountsGm_;
    GlobalTensor<float> mergeScoresGm_;
    GlobalTensor<int32_t> mergeIndicesGm_;

    GM_ADDR boxesAddr_{nullptr};
    GM_ADDR scoresAddr_{nullptr};
    GM_ADDR numValidBoxesAddr_{nullptr};
    GM_ADDR userWorkspace_{nullptr};
    const BatchMultiClassNonMaxSuppressionTilingData* tilingData_{nullptr};
    int64_t cachedClipBatch_{-1};
    float cachedClipYMin_{0.0F};
    float cachedClipXMin_{0.0F};
    float cachedClipYMax_{0.0F};
    float cachedClipXMax_{0.0F};
    bool workspaceReady_{false};
};

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::Init(
    GM_ADDR boxes, GM_ADDR scores, GM_ADDR clipWindow, GM_ADDR numValidBoxes, GM_ADDR nmsedBoxes, GM_ADDR nmsedScores,
    GM_ADDR nmsedClasses, GM_ADDR nmsedNum, GM_ADDR workspace,
    const BatchMultiClassNonMaxSuppressionTilingData* tilingData)
{
    tilingData_ = tilingData;
    boxesAddr_ = boxes;
    scoresAddr_ = scores;
    numValidBoxesAddr_ = numValidBoxes;
    workspaceReady_ = workspace != nullptr;
    if (!workspaceReady_) {
        return;
    }
    userWorkspace_ = GetUserWorkspace(workspace);
    workspaceReady_ = userWorkspace_ != nullptr;
    if (!workspaceReady_) {
        return;
    }

    if (tilingData_->hasClipWindow != 0U) {
        clipWindowGm_.SetGlobalBuffer((__gm__ T*)clipWindow, tilingData_->batch * 4);
    }
    const int64_t resultElements = tilingData_->batch * tilingData_->maxTotalSize;
    nmsedBoxesGm_.SetGlobalBuffer((__gm__ T*)nmsedBoxes, resultElements * 4);
    nmsedScoresGm_.SetGlobalBuffer((__gm__ T*)nmsedScores, resultElements);
    nmsedClassesGm_.SetGlobalBuffer((__gm__ T*)nmsedClasses, resultElements);
    nmsedNumGm_.SetGlobalBuffer(
        (__gm__ int32_t*)nmsedNum,
        tilingData_->batch * (tilingData_->transposeBox != 0U ? kPaddedCountStride : kUnpaddedCountStride));
    InitWorkspace();
    InitTileBuffers();
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::InitWorkspace()
{
    const int64_t taskCount = tilingData_->batch * tilingData_->classesNum;
    const int64_t classResultCount = taskCount * tilingData_->maxSizePerClass;
    const int64_t classBoxesElements = classResultCount * 4;
    classBoxesGm_.SetGlobalBuffer((__gm__ float*)(userWorkspace_ + tilingData_->classBoxesOffset), classBoxesElements);
    classScoresGm_.SetGlobalBuffer((__gm__ float*)(userWorkspace_ + tilingData_->classScoresOffset), classResultCount);
    classCountsGm_.SetGlobalBuffer((__gm__ float*)(userWorkspace_ + tilingData_->classCountsOffset), taskCount);
    const int64_t mergeResultCount = tilingData_->batch * tilingData_->maxTotalSize;
    mergeScoresGm_.SetGlobalBuffer((__gm__ float*)(userWorkspace_ + tilingData_->mergeScoresOffset), mergeResultCount);
    mergeIndicesGm_.SetGlobalBuffer((__gm__ int32_t*)(userWorkspace_ + tilingData_->mergeIndicesOffset),
                                    mergeResultCount);
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::InitTileBuffers()
{
    const int64_t tileSize = tilingData_->tileSize;
    const int64_t floatBytes = tileSize * static_cast<int64_t>(sizeof(float));
    pipe_.InitBuffer(mergeInputScoresBuffer_, kMergeHeapCapacity * static_cast<int64_t>(sizeof(float)));
    pipe_.InitBuffer(mergeInputIndicesBuffer_, kMergeHeapCapacity * static_cast<int64_t>(sizeof(int32_t)));
    pipe_.InitBuffer(scoreBuffer_, floatBytes);
    pipe_.InitBuffer(yMinBuffer_, floatBytes);
    pipe_.InitBuffer(xMinBuffer_, floatBytes);
    pipe_.InitBuffer(yMaxBuffer_, floatBytes);
    pipe_.InitBuffer(xMaxBuffer_, floatBytes);
    pipe_.InitBuffer(temp0Buffer_, floatBytes);
    pipe_.InitBuffer(temp1Buffer_, floatBytes);
    pipe_.InitBuffer(temp2Buffer_, floatBytes);
    pipe_.InitBuffer(temp3Buffer_, floatBytes);
    pipe_.InitBuffer(reduceWorkBuffer_, tilingData_->reduceBufferSize * static_cast<int64_t>(sizeof(float)));
    pipe_.InitBuffer(reduceOutputBuffer_, 64);
    pipe_.InitBuffer(compareMaskBuffer_, tileSize * static_cast<int64_t>(sizeof(uint8_t)));

    scoreLocal_ = scoreBuffer_.Get<float>();
    yMinLocal_ = yMinBuffer_.Get<float>();
    xMinLocal_ = xMinBuffer_.Get<float>();
    yMaxLocal_ = yMaxBuffer_.Get<float>();
    xMaxLocal_ = xMaxBuffer_.Get<float>();
    temp0Local_ = temp0Buffer_.Get<float>();
    temp1Local_ = temp1Buffer_.Get<float>();
    temp2Local_ = temp2Buffer_.Get<float>();
    temp3Local_ = temp3Buffer_.Get<float>();
    reduceWorkLocal_ = reduceWorkBuffer_.Get<float>();
    reduceOutputLocal_ = reduceOutputBuffer_.Get<float>();
    compareMaskLocal_ = compareMaskBuffer_.Get<uint8_t>();
    mergeInputScoresLocal_ = mergeInputScoresBuffer_.Get<float>();
    mergeInputIndicesLocal_ = mergeInputIndicesBuffer_.Get<int32_t>();
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::CopyIn(LocalTensor<float>& dst,
                                                                         const GlobalTensor<float>& src, int64_t offset,
                                                                         int64_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src[offset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::CopyOut(GlobalTensor<float>& dst, int64_t offset,
                                                                          LocalTensor<float>& src, int64_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
    DataCopyPad(dst[offset], src, copyParams);
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::LoadTile(int64_t offset, int64_t count)
{
    CopyIn(scoreLocal_, stageScoresGm_, offset, count);
    CopyIn(yMinLocal_, stageYMinGm_, offset, count);
    CopyIn(xMinLocal_, stageXMinGm_, offset, count);
    CopyIn(yMaxLocal_, stageYMaxGm_, offset, count);
    CopyIn(xMaxLocal_, stageXMaxGm_, offset, count);
    const event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::StoreTile(int64_t offset, int64_t count)
{
    PipeBarrier<PIPE_ALL>();
    CopyOut(stageScoresGm_, offset, scoreLocal_, count);
    CopyOut(stageYMinGm_, offset, yMinLocal_, count);
    CopyOut(stageXMinGm_, offset, xMinLocal_, count);
    CopyOut(stageYMaxGm_, offset, yMaxLocal_, count);
    CopyOut(stageXMaxGm_, offset, xMaxLocal_, count);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::LoadScores(int64_t offset, int64_t count)
{
    CopyIn(scoreLocal_, stageScoresGm_, offset, count);
    const event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::StoreScores(int64_t offset, int64_t count)
{
    PipeBarrier<PIPE_ALL>();
    CopyOut(stageScoresGm_, offset, scoreLocal_, count);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::GatherClass(int64_t taskIndex)
{
    const int64_t batchIndex = taskIndex / tilingData_->classesNum;
    const int64_t classIndex = taskIndex % tilingData_->classesNum;
    const int64_t scratchOffset = taskIndex * tilingData_->scratchBytesPerCore;
    GM_ADDR scratchBase = userWorkspace_ + scratchOffset;
    const int64_t scratchElements = tilingData_->boxesNum;
    stageScoresGm_.SetGlobalBuffer((__gm__ float*)scratchBase, scratchElements);
    scratchBase += tilingData_->scratchFieldStride;
    stageYMinGm_.SetGlobalBuffer((__gm__ float*)scratchBase, scratchElements);
    scratchBase += tilingData_->scratchFieldStride;
    stageXMinGm_.SetGlobalBuffer((__gm__ float*)scratchBase, scratchElements);
    scratchBase += tilingData_->scratchFieldStride;
    stageYMaxGm_.SetGlobalBuffer((__gm__ float*)scratchBase, scratchElements);
    scratchBase += tilingData_->scratchFieldStride;
    stageXMaxGm_.SetGlobalBuffer((__gm__ float*)scratchBase, scratchElements);

    if (tilingData_->transposeBox == 0U) {
        return;
    }

    const int64_t boxClass = tilingData_->boxClassesNum == 1 ? 0 : classIndex;
    const int64_t transposeBoxBase = ((batchIndex * tilingData_->boxClassesNum + boxClass) * 4) * tilingData_->boxesNum;
    const __gm__ T* transposeYMin = (__gm__ T*)boxesAddr_ + transposeBoxBase;
    const __gm__ T* transposeXMin = transposeYMin + tilingData_->boxesNum;
    const __gm__ T* transposeYMax = transposeXMin + tilingData_->boxesNum;
    const __gm__ T* transposeXMax = transposeYMax + tilingData_->boxesNum;

    if (tilingData_->use32Index != 0U) {
        asc_vf_call<GatherClassInput<T, true>>(
            dim3{kGatherThreadNum32}, transposeYMin, transposeXMin, transposeYMax, transposeXMax,
            (__gm__ T*)scoresAddr_, (__gm__ int32_t*)numValidBoxesAddr_, (__gm__ float*)stageScoresGm_.GetPhyAddr(),
            (__gm__ float*)stageYMinGm_.GetPhyAddr(), (__gm__ float*)stageXMinGm_.GetPhyAddr(),
            (__gm__ float*)stageYMaxGm_.GetPhyAddr(), (__gm__ float*)stageXMaxGm_.GetPhyAddr(),
            static_cast<uint32_t>(batchIndex), static_cast<uint32_t>(classIndex),
            static_cast<uint32_t>(tilingData_->boxesNum), static_cast<uint32_t>(tilingData_->classesNum),
            tilingData_->hasNumValidBoxes != 0U);
    } else {
        asc_vf_call<GatherClassInput<T, false>>(
            dim3{kGatherThreadNum64}, transposeYMin, transposeXMin, transposeYMax, transposeXMax,
            (__gm__ T*)scoresAddr_, (__gm__ int32_t*)numValidBoxesAddr_, (__gm__ float*)stageScoresGm_.GetPhyAddr(),
            (__gm__ float*)stageYMinGm_.GetPhyAddr(), (__gm__ float*)stageXMinGm_.GetPhyAddr(),
            (__gm__ float*)stageYMaxGm_.GetPhyAddr(), (__gm__ float*)stageXMaxGm_.GetPhyAddr(),
            static_cast<uint64_t>(batchIndex), static_cast<uint64_t>(classIndex),
            static_cast<uint64_t>(tilingData_->boxesNum), static_cast<uint64_t>(tilingData_->classesNum),
            tilingData_->hasNumValidBoxes != 0U);
    }
}

template <typename T>
__aicore__ inline bool BatchMultiClassNonMaxSuppressionKernel<T>::FilterStageBoxes(int64_t batchIndex)
{
    float clipYMin = 0.0F;
    float clipXMin = 0.0F;
    float clipYMax = 0.0F;
    float clipXMax = 0.0F;
    if (tilingData_->hasClipWindow != 0U) {
        // A core commonly handles several classes from the same batch in
        // adjacent task waves.  Keep these four scalar parameters in core
        // state so only the first class performs scalar GM reads.
        if (cachedClipBatch_ != batchIndex) {
            const int64_t clipOffset = batchIndex * 4;
            cachedClipYMin_ = static_cast<float>(clipWindowGm_.GetValue(clipOffset));
            cachedClipXMin_ = static_cast<float>(clipWindowGm_.GetValue(clipOffset + 1));
            cachedClipYMax_ = static_cast<float>(clipWindowGm_.GetValue(clipOffset + 2));
            cachedClipXMax_ = static_cast<float>(clipWindowGm_.GetValue(clipOffset + 3));
            cachedClipBatch_ = batchIndex;
        }
        clipYMin = cachedClipYMin_;
        clipXMin = cachedClipXMin_;
        clipYMax = cachedClipYMax_;
        clipXMax = cachedClipXMax_;
        if (tilingData_->changeCoordinateFrame != 0U && (clipYMax <= clipYMin || clipXMax <= clipXMin)) {
            return false;
        }
    }

    for (int64_t offset = 0; offset < tilingData_->boxesNum; offset += tilingData_->tileSize) {
        const int64_t count = (tilingData_->boxesNum - offset) < tilingData_->tileSize ?
                                  (tilingData_->boxesNum - offset) :
                                  tilingData_->tileSize;
        LoadTile(offset, count);
        if (tilingData_->hasClipWindow != 0U) {
            Maxs(yMinLocal_, yMinLocal_, clipYMin, count);
            Maxs(xMinLocal_, xMinLocal_, clipXMin, count);
            Mins(yMaxLocal_, yMaxLocal_, clipYMax, count);
            Mins(xMaxLocal_, xMaxLocal_, clipXMax, count);
            if (tilingData_->changeCoordinateFrame != 0U) {
                Adds(yMinLocal_, yMinLocal_, -clipYMin, count);
                Adds(yMaxLocal_, yMaxLocal_, -clipYMin, count);
                Adds(xMinLocal_, xMinLocal_, -clipXMin, count);
                Adds(xMaxLocal_, xMaxLocal_, -clipXMin, count);
                Muls(yMinLocal_, yMinLocal_, 1.0F / (clipYMax - clipYMin), count);
                Muls(yMaxLocal_, yMaxLocal_, 1.0F / (clipYMax - clipYMin), count);
                Muls(xMinLocal_, xMinLocal_, 1.0F / (clipXMax - clipXMin), count);
                Muls(xMaxLocal_, xMaxLocal_, 1.0F / (clipXMax - clipXMin), count);
            }
        }
        CompareScalar(compareMaskLocal_, scoreLocal_, tilingData_->scoreThreshold, CMPMODE::GT, count);
        Select(scoreLocal_, compareMaskLocal_, scoreLocal_, kNoCandidate, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        Compare(compareMaskLocal_, yMaxLocal_, yMinLocal_, CMPMODE::GT, count);
        Select(scoreLocal_, compareMaskLocal_, scoreLocal_, kNoCandidate, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        Compare(compareMaskLocal_, xMaxLocal_, xMinLocal_, CMPMODE::GT, count);
        Select(scoreLocal_, compareMaskLocal_, scoreLocal_, kNoCandidate, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        StoreTile(offset, count);
    }
    return true;
}

template <typename T>
__aicore__ inline bool BatchMultiClassNonMaxSuppressionKernel<T>::FindBestCandidate(int64_t& bestIndex,
                                                                                    float& bestScore)
{
    bestIndex = -1;
    bestScore = kNoCandidate;
    for (int64_t offset = 0; offset < tilingData_->boxesNum; offset += tilingData_->tileSize) {
        const int64_t count = (tilingData_->boxesNum - offset) < tilingData_->tileSize ?
                                  (tilingData_->boxesNum - offset) :
                                  tilingData_->tileSize;
        LoadScores(offset, count);
        ReduceMax<float>(reduceOutputLocal_, scoreLocal_, reduceWorkLocal_, static_cast<int32_t>(count), true);
        PipeBarrier<PIPE_V>();
        const event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        const float tileScore = reduceOutputLocal_.GetValue(0);
        // ReduceMax's index result is invalid for this Ascend 950 kernel.
        // Its value result is reliable: reduce matching negative indices with
        // it, so the maximum gives the first matching index on a tied score.
        CompareScalar(compareMaskLocal_, scoreLocal_, tileScore, CMPMODE::EQ, count);
        ArithProgression<float>(temp0Local_, 0.0F, -1.0F, static_cast<int32_t>(count));
        Select(temp0Local_, compareMaskLocal_, temp0Local_, -static_cast<float>(count),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        // The mask/select chain and the following reduction share the V
        // pipeline.  Make the data dependency explicit on 950 before
        // consuming the selected negative indices.
        PipeBarrier<PIPE_V>();
        ReduceMax<float>(reduceOutputLocal_, temp0Local_, reduceWorkLocal_, static_cast<int32_t>(count), false);
        PipeBarrier<PIPE_V>();
        const event_t eventIndexVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIndexVS);
        WaitFlag<HardEvent::V_S>(eventIndexVS);
        const int64_t tileIndex = -static_cast<int64_t>(reduceOutputLocal_.GetValue(0));
        const int64_t candidateIndex = offset + tileIndex;
        if (tileScore > bestScore || (tileScore == bestScore && (bestIndex < 0 || candidateIndex < bestIndex))) {
            bestScore = tileScore;
            bestIndex = candidateIndex;
        }
    }
    return bestIndex >= 0 && bestScore > tilingData_->scoreThreshold;
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::SuppressBySelectedBox(float selectedYMin,
                                                                                        float selectedXMin,
                                                                                        float selectedYMax,
                                                                                        float selectedXMax)
{
    const float selectedArea = (selectedYMax - selectedYMin) * (selectedXMax - selectedXMin);
    for (int64_t offset = 0; offset < tilingData_->boxesNum; offset += tilingData_->tileSize) {
        const int64_t count = (tilingData_->boxesNum - offset) < tilingData_->tileSize ?
                                  (tilingData_->boxesNum - offset) :
                                  tilingData_->tileSize;
        LoadTile(offset, count);
        Maxs(temp0Local_, yMinLocal_, selectedYMin, count);
        Mins(temp1Local_, yMaxLocal_, selectedYMax, count);
        Sub(temp0Local_, temp1Local_, temp0Local_, count);
        Maxs(temp0Local_, temp0Local_, 0.0F, count);
        Maxs(temp1Local_, xMinLocal_, selectedXMin, count);
        Mins(temp2Local_, xMaxLocal_, selectedXMax, count);
        Sub(temp1Local_, temp2Local_, temp1Local_, count);
        Maxs(temp1Local_, temp1Local_, 0.0F, count);
        Mul(temp2Local_, temp0Local_, temp1Local_, count);
        Sub(temp0Local_, yMaxLocal_, yMinLocal_, count);
        Sub(temp1Local_, xMaxLocal_, xMinLocal_, count);
        Mul(temp3Local_, temp0Local_, temp1Local_, count);
        Adds(temp3Local_, temp3Local_, selectedArea, count);
        Sub(temp3Local_, temp3Local_, temp2Local_, count);
        Maxs(temp3Local_, temp3Local_, kMinPositive, count);
        Div(temp2Local_, temp2Local_, temp3Local_, count);
        CompareScalar(compareMaskLocal_, temp2Local_, tilingData_->iouThreshold, CMPMODE::GT, count);
        Duplicate(temp3Local_, kNoCandidate, count);
        Select(scoreLocal_, compareMaskLocal_, temp3Local_, scoreLocal_, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        StoreScores(offset, count);
    }
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::ProcessClass(int64_t taskIndex)
{
    const int64_t classResultBase = taskIndex * tilingData_->maxSizePerClass;
    const int64_t batchIndex = taskIndex / tilingData_->classesNum;
    if (!FilterStageBoxes(batchIndex)) {
        PadClassScores(taskIndex, 0);
        StoreClassCount(taskIndex, 0);
        return;
    }
    int64_t selectedCount = 0;
    while (selectedCount < tilingData_->maxSizePerClass) {
        int64_t bestIndex = -1;
        float bestScore = kNoCandidate;
        if (!FindBestCandidate(bestIndex, bestScore)) {
            break;
        }
        const float yMin = stageYMinGm_.GetValue(bestIndex);
        const float xMin = stageXMinGm_.GetValue(bestIndex);
        const float yMax = stageYMaxGm_.GetValue(bestIndex);
        const float xMax = stageXMaxGm_.GetValue(bestIndex);
        const int64_t resultOffset = classResultBase + selectedCount;
        temp0Local_.SetValue(0, yMin);
        temp0Local_.SetValue(1, xMin);
        temp0Local_.SetValue(2, yMax);
        temp0Local_.SetValue(3, xMax);
        PipeBarrier<PIPE_ALL>();
        CopyOut(classBoxesGm_, resultOffset * 4, temp0Local_, 4);
        PipeBarrier<PIPE_ALL>();
        // Scalar GM writes are not visible to the MTE2 class-score merge on
        // Ascend 950.  Materialize the selected score through the vector
        // pipeline and publish it with MTE3, as PadClassScores does for the
        // sentinel tail of this class.
        Duplicate(temp0Local_, bestScore, 1);
        const event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        CopyOut(classScoresGm_, resultOffset, temp0Local_, 1);
        PipeBarrier<PIPE_ALL>();
        // Publish removal of the selected candidate through MTE3.  A scalar
        // GM store is invisible to the subsequent MTE2 score reload on 950,
        // which otherwise selects this same candidate repeatedly.
        Duplicate(temp0Local_, kNoCandidate, 1);
        const event_t invalidateVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(invalidateVToMte3);
        WaitFlag<HardEvent::V_MTE3>(invalidateVToMte3);
        CopyOut(stageScoresGm_, bestIndex, temp0Local_, 1);
        PipeBarrier<PIPE_ALL>();
        SuppressBySelectedBox(yMin, xMin, yMax, xMax);
        ++selectedCount;
    }
    PadClassScores(taskIndex, selectedCount);
    StoreClassCount(taskIndex, selectedCount);
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::PadClassScores(int64_t taskIndex,
                                                                                 int64_t selectedCount)
{
    const int64_t classResultBase = taskIndex * tilingData_->maxSizePerClass;
    for (int64_t offset = selectedCount; offset < tilingData_->maxSizePerClass; offset += tilingData_->tileSize) {
        const int64_t count = (tilingData_->maxSizePerClass - offset) < tilingData_->tileSize ?
                                  (tilingData_->maxSizePerClass - offset) :
                                  tilingData_->tileSize;
        Duplicate(scoreLocal_, kNoCandidate, count);
        PipeBarrier<PIPE_ALL>();
        CopyOut(classScoresGm_, classResultBase + offset, scoreLocal_, count);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::StoreClassCount(int64_t taskIndex,
                                                                                  int64_t selectedCount)
{
    // Publish the count through the vector/MTE3 path used for class scores.
    // A scalar LocalTensor write is not visible to a later DMA consumer on
    // Ascend 950, which would make the merge use stale workspace content.
    Duplicate(scoreLocal_, static_cast<float>(selectedCount), 1);
    const event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
    CopyOut(classCountsGm_, taskIndex, scoreLocal_, 1);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::MergeBatch(int64_t batchIndex)
{
    asc_vf_call<MergeClassOutput>(
        dim3{kMergeThreadNum}, (__gm__ float*)classScoresGm_.GetPhyAddr(), (__gm__ float*)classCountsGm_.GetPhyAddr(),
        (__gm__ float*)mergeScoresGm_.GetPhyAddr(), (__gm__ int32_t*)mergeIndicesGm_.GetPhyAddr(),
        (__gm__ int32_t*)nmsedNumGm_.GetPhyAddr(),
        reinterpret_cast<__ubuf__ float*>(mergeInputScoresLocal_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(mergeInputIndicesLocal_.GetPhyAddr()), static_cast<uint64_t>(batchIndex),
        static_cast<uint64_t>(tilingData_->classesNum), static_cast<uint64_t>(tilingData_->maxSizePerClass),
        static_cast<uint64_t>(tilingData_->maxTotalSize), tilingData_->transposeBox != 0U);

    // MergeClassOutput publishes GM results from the SIMT VF task. Ensure the
    // scalar side observes those writes before launching the consumer VF task.
    const event_t mergeDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(mergeDone);
    WaitFlag<HardEvent::V_S>(mergeDone);

    asc_vf_call<GatherMergedOutput<T>>(
        dim3{kMergeThreadNum}, (__gm__ float*)classBoxesGm_.GetPhyAddr(), (__gm__ float*)mergeScoresGm_.GetPhyAddr(),
        (__gm__ int32_t*)mergeIndicesGm_.GetPhyAddr(), (__gm__ T*)nmsedBoxesGm_.GetPhyAddr(),
        (__gm__ T*)nmsedScoresGm_.GetPhyAddr(), (__gm__ T*)nmsedClassesGm_.GetPhyAddr(),
        (__gm__ int32_t*)nmsedNumGm_.GetPhyAddr(), static_cast<uint64_t>(batchIndex),
        static_cast<uint64_t>(tilingData_->classesNum), static_cast<uint64_t>(tilingData_->maxSizePerClass),
        static_cast<uint64_t>(tilingData_->maxTotalSize), tilingData_->transposeBox != 0U);
}

template <typename T>
__aicore__ inline void BatchMultiClassNonMaxSuppressionKernel<T>::Process()
{
    if (!workspaceReady_ || static_cast<int64_t>(GetBlockIdx()) >= tilingData_->usedCoreNum) {
        return;
    }
    const int64_t taskCount = tilingData_->batch * tilingData_->classesNum;
    if (tilingData_->transposeBox == 0U) {
        if (tilingData_->use32Index != 0U) {
            asc_vf_call<GatherAllClassInputs<T, true>>(
                dim3{kGatherThreadNum32}, (__gm__ T*)boxesAddr_, (__gm__ T*)scoresAddr_,
                (__gm__ int32_t*)numValidBoxesAddr_, (__gm__ float*)userWorkspace_,
                static_cast<uint32_t>(GetBlockIdx()), static_cast<uint32_t>(tilingData_->usedCoreNum),
                static_cast<uint32_t>(taskCount), static_cast<uint32_t>(tilingData_->boxesNum),
                static_cast<uint32_t>(tilingData_->classesNum), static_cast<uint32_t>(tilingData_->boxClassesNum),
                tilingData_->scratchFieldStride, tilingData_->hasNumValidBoxes != 0U);
        } else {
            asc_vf_call<GatherAllClassInputs<T, false>>(
                dim3{kGatherThreadNum64}, (__gm__ T*)boxesAddr_, (__gm__ T*)scoresAddr_,
                (__gm__ int32_t*)numValidBoxesAddr_, (__gm__ float*)userWorkspace_,
                static_cast<uint64_t>(GetBlockIdx()), static_cast<uint64_t>(tilingData_->usedCoreNum),
                static_cast<uint64_t>(taskCount), static_cast<uint64_t>(tilingData_->boxesNum),
                static_cast<uint64_t>(tilingData_->classesNum), static_cast<uint64_t>(tilingData_->boxClassesNum),
                tilingData_->scratchFieldStride, tilingData_->hasNumValidBoxes != 0U);
        }
        // All cores have launched their persistent gather task before any
        // class enters the AIV NMS stage.
        SyncAll();
    }
    for (int64_t taskBase = 0; taskBase < taskCount; taskBase += tilingData_->usedCoreNum) {
        const int64_t taskIndex = taskBase + static_cast<int64_t>(GetBlockIdx());
        const bool hasTask = taskIndex < taskCount;
        if (hasTask) {
            GatherClass(taskIndex);
        }
        // asc_vf_call is synchronized with the AIV pipeline at the task-wave
        // boundary.  All blocks enter the barrier, including the last partial
        // wave, so this remains valid when batch * class is not core-aligned.
        SyncAll();
        if (hasTask) {
            ProcessClass(taskIndex);
        }
        SyncAll();
    }
    for (int64_t batchBase = 0; batchBase < tilingData_->batch; batchBase += tilingData_->usedCoreNum) {
        const int64_t batchIndex = batchBase + static_cast<int64_t>(GetBlockIdx());
        if (batchIndex < tilingData_->batch) {
            MergeBatch(batchIndex);
        }
        // Keep the AIV->SIMT hand-off collective.  All blocks enter this
        // barrier, including inactive tail blocks, before the next batch wave.
        SyncAll();
    }
}
} // namespace BatchMultiClassNonMaxSuppressionOp

#endif // BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_KERNEL_H_
