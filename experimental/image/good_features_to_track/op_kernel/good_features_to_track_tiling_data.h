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
 * \file good_features_to_track_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __GOOD_FEATURES_TO_TRACK_TILING_DATA_H__
#define __GOOD_FEATURES_TO_TRACK_TILING_DATA_H__

#include <cstdint>
// 响应计算按像素均匀分发到所有 AIV 核；角点选取由 0 号核串行完成（保证贪婪 NMS 确定性）
struct GoodFeaturesToTrackTilingData {
    uint32_t batch;
    uint32_t coreNum;
    uint32_t height;
    uint32_t width;
    uint32_t imageElements;
    uint32_t outputCapacity;
    int32_t blockSize;
    float qualityLevel;
    float minDistance;
    float harrisK;
    uint32_t useHarris;
    uint32_t hasMask;
};
#endif
