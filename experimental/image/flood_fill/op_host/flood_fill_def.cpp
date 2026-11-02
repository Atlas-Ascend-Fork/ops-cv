/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flood_fill_def.cpp
 * \brief FloodFill operator definition.
 */
#include "register/op_def_registry.h"

namespace ops {
class FloodFill : public OpDef {
public:
    explicit FloodFill(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Nonzero interior pixels are barriers. Input labels are preserved
        // in mask_out; the two tensors use independent storage.
        this->Input("mask_in")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // OpenCV-style fill mask: [H+2, W+2], border set to 1,
        // interior 1 where newly filled, existing input labels otherwise.
        // NOTE: dtype list length must match x/y (one entry per dtype
        // template); mask_out is always uint8 so the value repeats.
        this->Output("mask_out")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("seed").AttrType(REQUIRED).ListInt();
        this->Attr("new_value").AttrType(REQUIRED).ListFloat();
        this->Attr("lo_diff").AttrType(REQUIRED).ListFloat();
        this->Attr("up_diff").AttrType(REQUIRED).ListFloat();
        this->Attr("connectivity").AttrType(OPTIONAL).Int(4);
        this->Attr("diff_mode").AttrType(OPTIONAL).String("NEIGHBOR");

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(FloodFill);
} // namespace ops
