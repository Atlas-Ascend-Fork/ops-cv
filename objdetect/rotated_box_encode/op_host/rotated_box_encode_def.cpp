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
 * \file rotated_box_encode_def.cpp
 * \brief RotatedBoxEncode operator definition (OpDef registration)
 */

#include "register/op_def_registry.h"

namespace ops {

class RotatedBoxEncode : public OpDef {
public:
    explicit RotatedBoxEncode(const char* name) : OpDef(name)
    {
        // —— Input 声明：ParamType + DataType（按 §2.2 组合列展开，2 种：fp16 / fp32）+ Format + UnknownShapeFormat ——
        this->Input("anchor_box")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gt_box")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // —— Output 声明：同 Input 结构（dtype / format / 数组长度一致）——
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // —— Attr 声明 ——
        this->Attr("weight").AttrType(OPTIONAL).ListFloat(
            {1.0, 1.0, 1.0, 1.0, 1.0}); // 5 通道编码权重，缺省 [1,1,1,1,1]

        // —— AICore 配置（字段值取自信息库 aic-ascend950-ops-info.json）——
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "rotated_box_encode_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);
    }
};

OP_ADD(RotatedBoxEncode);

} // namespace ops
