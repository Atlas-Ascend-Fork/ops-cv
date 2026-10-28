/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_align_rotated_def.cpp
 * \brief RoiAlignRotated 算子定义
 */

#include "register/op_def_registry.h"

namespace ops {
class RoiAlignRotated : public OpDef {
public:
    explicit RoiAlignRotated(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous(); // 仅支持连续布局

        this->Input("rois")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("pooled_h").AttrType(REQUIRED).Int();
        this->Attr("pooled_w").AttrType(REQUIRED).Int();
        this->Attr("spatial_scale").AttrType(REQUIRED).Float();
        this->Attr("sampling_ratio").AttrType(OPTIONAL).Int(0);
        this->Attr("aligned").AttrType(OPTIONAL).Bool(true);
        this->Attr("clockwise").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicore95Config;
        aicore95Config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "roi_align_rotated_apt");
        this->AICore().AddConfig("ascend950", aicore95Config);
        this->AICore().AddConfig("ascend350", aicore95Config);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p");
        this->AICore().AddConfig("kirinx90");
        this->AICore().AddConfig("kirin9030");
    }
};
OP_ADD(RoiAlignRotated);
} // namespace ops
