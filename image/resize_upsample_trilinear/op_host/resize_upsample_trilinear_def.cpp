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
 * \file resize_upsample_trilinear.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> xDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                                                 ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
static const std::vector<ge::Format> xFormat = {ge::FORMAT_ND,    ge::FORMAT_ND,    ge::FORMAT_ND,
                                                ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW};

class ResizeUpsampleTrilinear : public OpDef {
public:
    explicit ResizeUpsampleTrilinear(const char* name) : OpDef(name)
    {
        // input
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // output
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("output_size").AttrType(OPTIONAL).ListInt();
        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);
        this->Attr("scales_d").AttrType(OPTIONAL).Float();
        this->Attr("scales_h").AttrType(OPTIONAL).Float();
        this->Attr("scales_w").AttrType(OPTIONAL).Float();

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig regbaseConfig;
        regbaseConfig.Input("input").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xFormat);
        regbaseConfig.Output("output").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xFormat);
        regbaseConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "resize_upsample_trilinear_apt");
        this->AICore().AddConfig("ascend950", regbaseConfig);
        this->AICore().AddConfig("ascend350", regbaseConfig);
        this->AICore().AddConfig("ascend960dt", regbaseConfig);
        this->AICore().AddConfig("ascend960pr", regbaseConfig);

        OpAICoreConfig config310p;
        config310p.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        config310p.Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        config310p.Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().AddConfig("ascend310p", config310p);
    }
};

OP_ADD(ResizeUpsampleTrilinear);
} // namespace ops
