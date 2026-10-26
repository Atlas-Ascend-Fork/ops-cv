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
 * \file grid_sampler2_d_def.cpp
 * \brief Operator definition for grid_sampler2_d operator
 */

#include "register/op_def_registry.h"

namespace ops {
class GridSampler2D : public OpDef {
public:
    explicit GridSampler2D(const char* name) : OpDef(name)
    {
        // x: (N, C, H_in, W_in), float16/float32
        // Format must be NCHW to match CANN built-in op_proto tiling requirement
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW, ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW, ge::FORMAT_NCHW})
            .AutoContiguous();
        // grid: (N, H_out, W_out, 2), dtype with x
        this->Input("grid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // y: (N, C, H_out, W_out), dtype with x
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW, ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW, ge::FORMAT_NCHW})
            .AutoContiguous();

        // Attributes (all OPTIONAL with defaults, matching REG_OP ATTR)
        this->Attr("interpolation_mode").AttrType(OPTIONAL).String("bilinear");
        this->Attr("padding_mode").AttrType(OPTIONAL).String("zeros");
        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "grid_sampler2_d_apt")
            .ExtendCfgInfo("opInterface.value", "grid_sampler2_d");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);
    }
};
OP_ADD(GridSampler2D);
} // namespace ops
