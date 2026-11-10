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
 * \file roi_pooling_grad_with_arg_max_tiling_arch35.h
 * \brief
 */

#ifndef ROI_POOLING_GRAD_WITH_ARG_MAX_TILING_H
#define ROI_POOLING_GRAD_WITH_ARG_MAX_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/roi_pooling_grad_with_arg_max_tiling_data.h"

namespace optiling {

struct RoiPoolingGradWithArgMaxCompileInfo {};

class RoiPoolingGradWithArgMaxTiling {
public:
    explicit RoiPoolingGradWithArgMaxTiling(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus DoTiling();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus InitTilingData();
    ge::graphStatus GetInputTensorInfo();
    ge::graphStatus SetAttrParams();
    uint64_t GetTilingKey();
    bool GetDataTypeKey(ge::DataType dataType);

    void SetTilingKey();
    void SetTilingData();

private:
    RoiPoolingGradWithArgMaxRegBaseTilingData* tilingData_{nullptr};
    gert::TilingContext* context_ = nullptr;

    uint64_t dataTypeTilingKey_ = 0;
    uint64_t tilingKey_{0};
    int64_t useCoreNum_ = 1;
    ge::DataType gardInDType_ = ge::DT_UNDEFINED;

    int64_t totalCoreNum_ = 1;
    int64_t yTotalCoreNum_ = 1;
    int64_t yDataPerCore_ = 0;
    int64_t yDataTailCore_ = 0;
    int64_t totalLength_ = 1;
    int64_t yTotalLength_ = 1;
    int64_t pooledH_ = 1;
    int64_t pooledW_ = 1;
    int64_t spatialScaleH_ = 1;
    int64_t spatialScaleW_ = 1;
    int64_t poolChannel_ = 1;
    int64_t height_ = 1;
    int64_t width_ = 1;
    int64_t batch_ = 1;
    int64_t usrWorkspaceSize_ = 1;
};
} // namespace optiling
#endif
