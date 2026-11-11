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
 * \file resize_bilinear_v2_def.cpp
 * \brief resize_bilinear_v2_def
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> valueDataTypeX = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,
                                                         ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                                         ge::DT_BF16,    ge::DT_BF16};
static const std::vector<ge::DataType> valueDataTypeY = {ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_BF16,
                                                         ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                                                         ge::DT_BF16,    ge::DT_FLOAT};

static const std::vector<ge::DataType> sizeDataType = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                       ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                       ge::DT_INT32, ge::DT_INT32};

static const std::vector<ge::Format> resizeBilinearV2Format = {
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC};

static const std::vector<ge::Format> sizeFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND};

class ResizeBilinearV2 : public OpDef {
public:
    explicit ResizeBilinearV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(valueDataTypeX)
            .Format(resizeBilinearV2Format)
            .UnknownShapeFormat(resizeBilinearV2Format);
        this->Input("size")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(sizeDataType)
            .Format(sizeFormat)
            .UnknownShapeFormat(sizeFormat);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(valueDataTypeY)
            .Format(resizeBilinearV2Format)
            .UnknownShapeFormat(resizeBilinearV2Format);

        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);
        this->Attr("half_pixel_centers").AttrType(OPTIONAL).Bool(false);
        this->Attr("dtype").AttrType(OPTIONAL).Int(ge::DT_FLOAT);
        this->Attr("scales").AttrType(OPTIONAL).ListFloat({0.0f, 0.0f});

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "resize_bilinear_v2_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);
        this->AICore().AddConfig("ascend960dt", aicoreConfig);
        this->AICore().AddConfig("ascend960pr", aicoreConfig);
        this->AICore().AddConfig("mc62", aicoreConfig);

        OpAICoreConfig config5162 = Get5162CoreConfig();
        this->AICore().AddConfig("ascend5162", config5162);
    }

private:
    OpAICoreConfig Get5162CoreConfig() const
    {
        OpAICoreConfig config5162;
        config5162.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "resize_bilinear_v2_apt");
        config5162.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC});
        config5162.Input("size")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config5162.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC});
        return config5162;
    }
};

OP_ADD(ResizeBilinearV2);
} // namespace ops
