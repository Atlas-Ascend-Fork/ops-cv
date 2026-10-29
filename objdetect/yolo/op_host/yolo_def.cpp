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
 * \file yolo_def.cpp
 * \brief Operator definition for yolo operator
 */

#include "register/op_def_registry.h"

namespace ops {
class Yolo : public OpDef {
public:
    explicit Yolo(const char* name) : OpDef(name)
    {
        // Input x: (N, boxes*(coords+1+classes), H, W), float16/float32
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Output coord_data: (N, boxes*coords, H*W)
        this->Output("coord_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Output obj_prob: (N, boxes*H*W)
        this->Output("obj_prob")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Output classes_prob: (N, classes, boxes*H*W)
        this->Output("classes_prob")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Attributes (matching REG_OP order)
        this->Attr("boxes").AttrType(OPTIONAL).Int(3);
        this->Attr("coords").AttrType(OPTIONAL).Int(4);
        this->Attr("classes").AttrType(OPTIONAL).Int(80);
        this->Attr("yolo_version").AttrType(OPTIONAL).String("V3");
        this->Attr("softmax").AttrType(OPTIONAL).Bool(false);
        this->Attr("background").AttrType(OPTIONAL).Bool(false);
        this->Attr("softmaxtree").AttrType(OPTIONAL).Bool(false);

        // AICore configuration
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "yolo_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);
    }
};
OP_ADD(Yolo);
} // namespace ops
