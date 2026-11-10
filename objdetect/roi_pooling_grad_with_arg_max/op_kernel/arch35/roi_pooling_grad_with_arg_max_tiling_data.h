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
 * \file roi_pooling_grad_with_arg_max_tiling_data.h
 * \brief roi_pooling_grad_with_arg_max_tiling_data.h
 */

#ifndef _ROI_POOLING_GRAD_WITH_ARG_MAX_REGBASE_TILING_DATA_H_
#define _ROI_POOLING_GRAD_WITH_ARG_MAX_REGBASE_TILING_DATA_H_

#include <string>

struct RoiPoolingGradWithArgMaxRegBaseTilingData {
    int64_t yTotalCoreNum;
    int64_t yDataPerCore;
    int64_t yDataTailCore;
    int64_t totalLength;
    int64_t yTotalLength;
    int64_t pooledH;
    int64_t pooledW;
    int64_t height;
    int64_t width;
    int64_t poolChannel;
    int64_t useCoreNum;
    int64_t batch;

    std::string toString() const
    {
        return "yTotalCoreNum = " + std::to_string(yTotalCoreNum) + ", yDataPerCore = " + std::to_string(yDataPerCore) +
               ", yDataTailCore = " + std::to_string(yDataTailCore) + ", totalLength = " + std::to_string(totalLength) +
               ", yTotalLength = " + std::to_string(yTotalLength) + ", pooledH = " + std::to_string(pooledH) +
               ", pooledW = " + std::to_string(pooledW) + ", height = " + std::to_string(height) +
               ", width = " + std::to_string(width) + ", poolChannel = " + std::to_string(poolChannel) +
               ", useCoreNum = " + std::to_string(useCoreNum) + ", batch = " + std::to_string(batch);
    }
};
#endif
