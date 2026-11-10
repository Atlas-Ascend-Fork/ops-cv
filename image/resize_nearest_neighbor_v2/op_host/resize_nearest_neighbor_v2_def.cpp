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
 * \file resize_nearest_neighbor_v2.cpp
 * \brief resize_nearest_neighbor_v2
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> valueDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                                                        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};

static const std::vector<ge::DataType> sizeDataType = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                       ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};

static const std::vector<ge::Format> resizeNearestNeighborV2Format = {
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
};

static const std::vector<ge::Format> sizeFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ResizeNearestNeighborV2 : public OpDef {
public:
    explicit ResizeNearestNeighborV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(valueDataType)
            .Format(resizeNearestNeighborV2Format)
            .UnknownShapeFormat(resizeNearestNeighborV2Format);
        this->Input("size")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(sizeDataType)
            .Format(sizeFormat)
            .UnknownShapeFormat(sizeFormat);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(valueDataType)
            .Format(resizeNearestNeighborV2Format)
            .UnknownShapeFormat(resizeNearestNeighborV2Format);

        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);
        this->Attr("half_pixel_centers").AttrType(OPTIONAL).Bool(false);
        this->Attr("scales").AttrType(OPTIONAL).ListFloat({0.0f, 0.0f});

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "resize_nearest_neighbor_v2_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);
        this->AICore().AddConfig("ascend960dt", aicoreConfig);
        this->AICore().AddConfig("ascend960pr", aicoreConfig);
        this->AICore().AddConfig("mc62", aicoreConfig);
        this->AICore().AddConfig("ascend5162", aicoreConfig);
    }
};

OP_ADD(ResizeNearestNeighborV2);
} // namespace ops
