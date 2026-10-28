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
 * \file upsample_nearest_exact3d_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> xDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
static const std::vector<ge::Format> xFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class UpsampleNearestExact3d : public OpDef {
public:
    explicit UpsampleNearestExact3d(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xFormat);

        this->Output("y").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xFormat);

        this->Attr("output_size").AttrType(REQUIRED).ListInt();
        this->Attr("scale_d").AttrType(OPTIONAL).Float(0.0);
        this->Attr("scale_h").AttrType(OPTIONAL).Float(0.0);
        this->Attr("scale_w").AttrType(OPTIONAL).Float(0.0);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig regbaseConfig;
        regbaseConfig.Input("x").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xFormat);
        regbaseConfig.Output("y").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xFormat);
        regbaseConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "upsample_nearest_exact3d_apt");
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
        config310p.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        config310p.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().AddConfig("ascend310p", config310p);
        this->AICore().AddConfig("kirinx90", config310p);
        this->AICore().AddConfig("kirin9030", config310p);
    }
};

OP_ADD(UpsampleNearestExact3d);
} // namespace ops
