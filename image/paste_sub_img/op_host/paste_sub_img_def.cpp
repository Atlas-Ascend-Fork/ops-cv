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
 * \file paste_sub_img_def.cpp
 * \brief Operator definition for paste_sub_img operator
 */
#include "register/op_def_registry.h"

namespace ops {

class PasteSubImg : public OpDef {
public:
    explicit PasteSubImg(const char* name) : OpDef(name)
    {
        this->Input("patch_img")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("patch_coord")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("core_area_coord")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("combine_img")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("combine_img")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("scale").AttrType(REQUIRED).Float();

        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "paste_sub_img_apt");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
        this->AICore().AddConfig("ascend350", aiCoreConfig);
    }
};
OP_ADD(PasteSubImg);
} // namespace ops
