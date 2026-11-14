/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file crop_and_resize_tiling_data.h
 * \brief Tiling data struct for crop_and_resize operator
 */

#ifndef CROP_AND_RESIZE_TILING_DATA_H_
#define CROP_AND_RESIZE_TILING_DATA_H_

#include <cstdint>

struct CropAndResizeTilingData {
    int64_t totalPositions;   // 总位置数 = num_boxes * crop_height * crop_width
    int64_t batch;            // N — x.shape[0]，用于 box_index 范围检查
    int64_t imageHeight;      // H — x.shape[1]，用于坐标映射
    int64_t imageWidth;       // W — x.shape[2]，用于坐标映射
    int64_t depth;            // C — x.shape[3]，通道循环上界
    int64_t cropHeight;       // crop_size[0]，位置解码 + 坐标映射
    int64_t cropWidth;        // crop_size[1]，位置解码 + 坐标映射
    int64_t numBoxes;         // boxes.shape[0]，tiling 约束检查用（不传给 VF）
    float extrapolationValue; // ATTR extrapolation_value，越界输出值
};

#endif // CROP_AND_RESIZE_TILING_DATA_H_
