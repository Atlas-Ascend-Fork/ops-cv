/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file paste_sub_img_tiling_data.h
 * \brief Tiling data struct for paste_sub_img operator
 */
#ifndef PASTE_SUB_IMG_TILING_DATA_H_
#define PASTE_SUB_IMG_TILING_DATA_H_

#include <cstdint>

static constexpr int64_t PASTE_SUB_IMG_AXIS_COUNT = 3;

struct PasteSubImgTilingData {
    uint8_t rank;
    int64_t inShape[PASTE_SUB_IMG_AXIS_COUNT];
    int64_t outShape[PASTE_SUB_IMG_AXIS_COUNT];
    uint64_t totalCount;
    uint64_t perCoreCount;
    uint8_t ubAxis;
    uint32_t ubFactor;
    uint32_t bufferSize;
    int64_t patchBaseOffset;
    int64_t combineBaseOffset;
    int64_t patchStrideH;
    int64_t patchStrideW;
    int64_t combineStrideH;
    int64_t combineStrideW;
    int64_t activeH;
    int64_t activeW;
    int64_t activeC;
    uint8_t dtypeBytes;
};

#endif // PASTE_SUB_IMG_TILING_DATA_H_
