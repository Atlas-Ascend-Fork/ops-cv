/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_nearest_exact2d_grad_kernel.h
 * \brief
 */

#ifndef UPSAMPLE_NEAREST_EXACT2D_GRAD
#define UPSAMPLE_NEAREST_EXACT2D_GRAD

#include "upsample_nearest_exact2d_grad_kernel_base.h"

namespace UpSampleNearestExact2dGrad {

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::Init(GM_ADDR input, GM_ADDR output, bool isExact,
                                                             GM_ADDR workspace,
                                                             UpsampleNearestExact2dGradTilingData* tilingData)
{
    blockIdx = GetBlockIdx() / 2;
    exactMode = isExact;
    inTensorsPtr = input;
    outTensorsPtr = output;
    ParseTilingData(tilingData);
    isExpandH = FloatEqual(scale_h, 1.0);
    isExpandW = FloatEqual(scale_w, 1.0);

    int64_t radioSize = getMax(radio_matrix_size, radio_matrix_size_h);
    int64_t interpsize = getMax(max_interp_size_h, max_interp_size_w);
    middleSize = input_shapes[2] * output_shapes[3];
    cubeSize = output_shapes[2] * output_shapes[3];
    pipe.InitBuffer(centerQueue, (slide_size * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(upQueue, (slide_size * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(downQueue, (slide_size * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(radioQueue, NO_BUFFER_NUM, (radioSize * sizeof(float) + 31) / 32 * 32);
    pipe.InitBuffer(radioCastQueue, NO_BUFFER_NUM, (radioSize * sizeof(T) + 31) / 32 * 32);

    intermediateTensorGm.SetGlobalBuffer((__gm__ T*)workspace);
    inTensorsGM.SetGlobalBuffer((__gm__ T*)inTensorsPtr);
    outTensorsGM.SetGlobalBuffer((__gm__ T*)outTensorsPtr);
};

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::wDirectionExpansion()
{
    if (blockIdx < need_core_num_w) {
        // 存放中心点映射的位置和grad_output切块对应的
        LocalTensor<float> centerTensor = centerQueue.Get<float>();
        LocalTensor<float> downTensor = downQueue.Get<float>();
        LocalTensor<float> upTensor = upQueue.Get<float>();
        // 计算滑块映射范围

        if (slideStart_w < slideEnd_w) {
            for (int64_t index = slideStart_w; index < slideEnd_w; index += slide_size) {
                int64_t length = Min(slide_size, slideEnd_w - index);
                calculateIntermediateTensorX(centerTensor, downTensor, upTensor, index, index + length);

                slidelen = length;
                calculateRadioTensorW(centerTensor, downTensor, upTensor, index, length);
                copyRadioTensorToGm();
                calculateWidthExtension(index, 0, 0);
            }
        }
        if (tailSlideStart_w < tailSlideEnd_w) {
            for (int64_t index = tailSlideStart_w; index < tailSlideEnd_w; index += slide_size) {
                int64_t length = Min(slide_size, tailSlideEnd_w - index);
                calculateIntermediateTensorX(centerTensor, downTensor, upTensor, index, index + length);

                slidelen = length;
                calculateRadioTensorW(centerTensor, downTensor, upTensor, index, length);
                copyRadioTensorToGm();
                calculateWidthExtension(index, tailRowStart_w, tailRowEnd_w);
            }
        }
        centerQueue.FreeTensor(centerTensor);
        downQueue.FreeTensor(downTensor);
        upQueue.FreeTensor(upTensor);
    }
};

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::hDirectionExpansion()
{
    if (blockIdx < need_core_num_h) {
        LocalTensor<float> centerTensor = centerQueue.Get<float>();
        LocalTensor<float> downTensor = downQueue.Get<float>();
        LocalTensor<float> upTensor = upQueue.Get<float>();
        if (slideStart_h < slideEnd_h) {
            for (int64_t index = slideStart_h; index < slideEnd_h; index += slide_size) {
                int64_t length = Min(slide_size, slideEnd_h - index);
                calculateIntermediateTensorY(centerTensor, downTensor, upTensor, index, index + length);

                slidelen_h = length;
                calculateRadioTensorH(centerTensor, downTensor, upTensor, index, length);
                copyRadioTensorToGm();
                calculateHeightExtension(index, 0, 0);
            }
        }

        if (tailSlideStart_h < tailSlideEnd_h) {
            for (int64_t index = tailSlideStart_h; index < tailSlideEnd_h; index += slide_size) {
                int64_t length = Min(slide_size, tailSlideEnd_h - index);
                calculateIntermediateTensorY(centerTensor, downTensor, upTensor, index, index + length);

                slidelen_h = length;
                calculateRadioTensorH(centerTensor, downTensor, upTensor, index, length);
                copyRadioTensorToGm();
                calculateHeightExtension(index, tailBatchStart_h, tailBatchEnd_h);
            }
        }
        centerQueue.FreeTensor(centerTensor);
        downQueue.FreeTensor(downTensor);
        upQueue.FreeTensor(upTensor);
    }
};

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::Process()
{
    if (GetSubBlockIdx() == 1) {
        SyncAll();
        return;
    }
    if (!FloatEqual(scale_w, 1.0)) {
        wDirectionExpansion();
    }

    SyncAll();

    if ((!FloatEqual(scale_h, 1.0)) || (FloatEqual(scale_w, 1.0))) {
        hDirectionExpansion();
    }
}

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::calculateIntermediateTensorX(LocalTensor<float> centerTensor,
                                                                                     LocalTensor<float> downTensor,
                                                                                     LocalTensor<float> upTensor,
                                                                                     int64_t slideStart_w,
                                                                                     int64_t slideEnd_w)
{
    instart_w = slideStart_w;

    int64_t length = static_cast<int64_t>(centerTensor.GetSize());
    int64_t wSize = input_shapes[3];
    // 先计算影响范围和中心点对应的位置，对象为输入矩阵中所有的列
    ArithProgression(centerTensor, static_cast<float>(instart_w), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
    // 计算center下标映射下界
    Adds(downTensor, centerTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();
    Muls(downTensor, downTensor, scale_w, length);
    PipeBarrier<PIPE_V>();
    if (exactMode) {
        Adds(downTensor, downTensor, -(float)0.5, length);
        PipeBarrier<PIPE_V>();
    }
    Ceil(centerTensor, downTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(downTensor, centerTensor, static_cast<float>(wSize), length);
    // 计算center下标映射上界
    ArithProgression(upTensor, static_cast<float>(instart_w + 1), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
    Muls(upTensor, upTensor, scale_w, length);
    PipeBarrier<PIPE_V>();
    if (exactMode) {
        Adds(upTensor, upTensor, -(float)0.5, length);
        PipeBarrier<PIPE_V>();
    }
    Ceil(centerTensor, upTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(upTensor, centerTensor, static_cast<float>(wSize), length);
}

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::calculateIntermediateTensorY(LocalTensor<float> centerTensor,
                                                                                     LocalTensor<float> downTensor,
                                                                                     LocalTensor<float> upTensor,
                                                                                     int64_t slideStart_h,
                                                                                     int64_t slideEnd_h)
{
    instart_h = slideStart_h;
    int64_t length = static_cast<int64_t>(centerTensor.GetSize());
    int64_t hSize = input_shapes[2];
    ArithProgression(centerTensor, static_cast<float>(instart_h), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
    // 计算center下标映射下界
    Adds(downTensor, centerTensor, (float)0.0, length);
    PipeBarrier<PIPE_V>();
    Muls(downTensor, downTensor, scale_h, length);
    PipeBarrier<PIPE_V>();
    if (exactMode) {
        Adds(downTensor, downTensor, -(float)0.5, length);
        PipeBarrier<PIPE_V>();
    }
    Ceil(centerTensor, downTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(downTensor, centerTensor, static_cast<float>(hSize), length);
    // 计算center下标映射上界
    ArithProgression(upTensor, static_cast<float>(instart_h + 1), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
    Muls(upTensor, upTensor, scale_h, length);
    PipeBarrier<PIPE_V>();
    if (exactMode) {
        Adds(upTensor, upTensor, -(float)0.5, length);
        PipeBarrier<PIPE_V>();
    }
    Ceil(centerTensor, upTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(upTensor, centerTensor, static_cast<float>(hSize), length);
}

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::calculateRadioTensorW(LocalTensor<float> centerTensor,
                                                                              LocalTensor<float> downTensor,
                                                                              LocalTensor<float> upTensor,
                                                                              int64_t index, int64_t length)
{
    instartIndex = downTensor.GetValue(0);
    inendIndex = upTensor.GetValue(length - 1);
    singleCoreK = inendIndex - instartIndex;

    LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
    // 初始化为0
    Duplicate(radioTensor, float(0.0), radioTensor.GetSize());

    // 计算影响该块的原始矩阵点的下标
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    for (int i = 0; i < length; i++) {
        int64_t downIndex = downTensor.GetValue(i) - instartIndex;
        int64_t upIndex = upTensor.GetValue(i) - instartIndex;
        for (int j = downIndex; j < upIndex; j++) {
            int64_t radioIndex = j * length + i;
            radioTensor.SetValue(radioIndex, (float)1.0);
        }
    }

    if (dataType != 2) {
        LocalTensor<T> radioCastTensorW = radioCastQueue.AllocTensor<T>();
        Cast(radioCastTensorW, radioTensor, RoundMode::CAST_RINT, radioTensor.GetSize());
        radioCastQueue.EnQue(radioCastTensorW);
        radioQueue.FreeTensor(radioTensor);
    } else {
        radioQueue.EnQue(radioTensor);
    }
}

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::calculateRadioTensorH(LocalTensor<float> centerTensor_h,
                                                                              LocalTensor<float> downTensor,
                                                                              LocalTensor<float> upTensor,
                                                                              int64_t index, int64_t length)
{
    instartIndex = downTensor.GetValue(0);
    inendIndex = upTensor.GetValue(length - 1);
    singleCoreK_h = static_cast<int32_t>(inendIndex - instartIndex);

    LocalTensor<float> radioTensor_h = radioQueue.AllocTensor<float>();
    Duplicate(radioTensor_h, float(0.0), radioTensor_h.GetSize());

    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    for (int i = 0; i < length; i++) {
        int64_t downIndex = downTensor.GetValue(i) - instartIndex;
        int64_t upIndex = upTensor.GetValue(i) - instartIndex;
        for (int j = downIndex; j < upIndex; j++) {
            int64_t radioIndex = static_cast<int64_t>(i) * static_cast<int64_t>(singleCoreK_h) +
                                 static_cast<int64_t>(j);
            radioTensor_h.SetValue(radioIndex, (float)1.0);
        }
    }

    if (dataType != 2) {
        LocalTensor<T> radioCastTensor_h = radioCastQueue.AllocTensor<T>();
        Cast(radioCastTensor_h, radioTensor_h, RoundMode::CAST_RINT, radioTensor_h.GetSize());
        radioCastQueue.EnQue(radioCastTensor_h);
        radioQueue.FreeTensor(radioTensor_h);
    }

    else {
        radioQueue.EnQue(radioTensor_h);
    }
}

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::copyRadioTensorToGm()
{
    // 槽位步长与 host tiling 同口径：max(w, h)，host 侧已按 32 字节对齐
    int64_t radioSize = getMax(radio_matrix_size, radio_matrix_size_h);
    workSpaceRadioOffset = intermediate_matrix_size + radioSize * blockIdx;

    // 写入长度不超过实际缓冲大小，避免尾部取整越过槽位边界
    if (dataType == 2) {
        LocalTensor<T> radioTensor = radioQueue.DeQue<T>();
        DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioTensor, radioTensor.GetSize());
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        radioQueue.FreeTensor(radioTensor);
    } else {
        LocalTensor<T> radioCastTensor = radioCastQueue.DeQue<T>();
        DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioCastTensor, radioCastTensor.GetSize());
        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        radioCastQueue.FreeTensor(radioCastTensor);
    }
}

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::calculateWidthExtension(int64_t tensorCIndex, int64_t rowStart,
                                                                                int64_t rowEnd)
{
    int64_t singleCoreM = matmulTiling_w->singleCoreM;
    int64_t singleCoreN = matmulTiling_w->singleCoreN;
    singleCoreN = slidelen;

    // 尾块batch分批处理
    if (rowEnd != 0) {
        singleCoreM = rowEnd - rowStart;
    }
    if (singleCoreK == 0) {
        singleCoreK++;
    }
    if (singleCoreN == 0) {
        singleCoreN++;
    }
    matmulW.SetOrgShape(singleCoreM, singleCoreN, input_shapes[3], singleCoreK, output_shapes[3]);
    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
    if (tensorCIndex + slide_size > output_shapes[3] - 1) {
        matmulW.SetTail(singleCoreM, output_shapes[3] - tensorCIndex, singleCoreK);
    }
    if (instartIndex >= input_shapes[3]) {
        instartIndex = 0;
    }
    int64_t xIndex = instartIndex + rowStart * input_shapes[3];
    int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * output_shapes[3];

    matmulW.SetTensorA(inTensorsGM[xIndex], false);
    matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);
    int64_t radioSize = getMax(radio_matrix_size, radio_matrix_size_h);
    if (isExpandH) {
        matmulW.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
    } else {
        matmulW.IterateAll(intermediateTensorGm[tensorCIndexWithOffset], false);
    }
    matmulW.End();
}

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::calculateHeightExtension(int64_t tensorCIndex,
                                                                                 int64_t batchStart, int64_t batchEnd)
{
    int64_t singleCoreM = matmulTiling_h->singleCoreM;
    int64_t singleCoreN = matmulTiling_h->singleCoreN;
    // 尾块batch分批处理
    if (singleCoreK_h == 0) {
        singleCoreK_h++;
    }
    if (batchEnd == 0) {
        batchEnd = input_shapes[0] * input_shapes[1];
    }
    singleCoreN = output_shapes[3];

    if (tensorCIndex + slide_size > output_shapes[2]) {
        singleCoreM = output_shapes[2] - tensorCIndex;
    }

    matmulH.SetOrgShape(singleCoreM, output_shapes[3], singleCoreK_h, output_shapes[2], output_shapes[3]);
    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreK_h);

    if (tensorCIndex + slide_size > output_shapes[2] - 1) {
        matmulH.SetTail(output_shapes[2] - tensorCIndex, singleCoreN, singleCoreK_h);
    }
    if (instartIndex >= input_shapes[2]) {
        instartIndex = 0;
    }
    int64_t xIndex = instartIndex * output_shapes[3] + batchStart * middleSize;
    int64_t tensorCIdxWithOffset = tensorCIndex * output_shapes[3] + batchStart * cubeSize;
    int64_t radioSize = getMax(radio_matrix_size, radio_matrix_size_h);
    matmulH.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], false);
    for (int64_t i = batchStart; i < batchEnd; i++) {
        // 系数矩阵起始位置
        if (FloatEqual(scale_w, 1.0)) {
            matmulH.SetTensorB(inTensorsGM[xIndex], false);
        } else {
            matmulH.SetTensorB(intermediateTensorGm[xIndex], false);
        }
        matmulH.IterateAll(outTensorsGM[tensorCIdxWithOffset], false);
        xIndex += middleSize;
        tensorCIdxWithOffset += cubeSize;
        matmulH.End();
    }
}

template <typename T>
__aicore__ inline void UpSampleNearestExact2dGradND<T>::ParseTilingData(
    UpsampleNearestExact2dGradTilingData* tilingData)
{
    slide_size = tilingData->slide_size;
    scale_h = tilingData->scale_h;
    scale_w = tilingData->scale_w;
    invscale_h = tilingData->invscale_h;
    invscale_w = tilingData->invscale_w;

    support_h = tilingData->support_h;
    support_w = tilingData->support_w;
    max_interp_size_h = tilingData->max_interp_size_h;
    max_interp_size_w = tilingData->max_interp_size_w;

    need_core_num_w = tilingData->need_core_num_w;
    need_core_num_h = tilingData->need_core_num_h;

    for (int8_t i = 0; i < 4; i++) {
        input_shapes[i] = tilingData->input_shapes[i];
    }
    for (int8_t i = 0; i < 4; i++) {
        output_shapes[i] = tilingData->output_shapes[i];
    }

    intermediate_matrix_size = tilingData->intermediate_matrix_size;
    radio_matrix_size = tilingData->radio_matrix_size;
    radio_matrix_size_h = tilingData->radio_matrix_size_h;

    slideStart_w = tilingData->slideStartList_w[blockIdx];
    slideEnd_w = tilingData->slideEndList_w[blockIdx];
    tailSlideStart_w = tilingData->tailSlideStartList_w[blockIdx];
    tailRowStart_w = tilingData->tailRowStartList_w[blockIdx];
    tailRowEnd_w = tilingData->tailRowEndList_w[blockIdx];
    tailSlideEnd_w = tilingData->tailSlideEndList_w[blockIdx];

    slideStart_h = tilingData->slideStartList_h[blockIdx];
    slideEnd_h = tilingData->slideEndList_h[blockIdx];
    tailSlideStart_h = tilingData->tailSlideStartList_h[blockIdx];
    tailSlideEnd_h = tilingData->tailSlideEndList_h[blockIdx];
    tailRowStart_h = tilingData->tailRowStartList_h[blockIdx];
    tailRowEnd_h = tilingData->tailRowEndList_h[blockIdx];
    tailBatchStart_h = tilingData->tailBatchStartListH[blockIdx];
    tailBatchEnd_h = tilingData->tailBatchEndListH[blockIdx];
    dataType = tilingData->dataType;

    matmulTiling_w = &tilingData->matmulTiling_w;
    matmulTiling_h = &tilingData->matmulTiling_h;
}

} // namespace UpSampleNearestExact2dGrad

#endif
