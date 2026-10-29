/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file gamma_correction_def.cpp
 *  \brief GammaCorrection operator prototype.
 */
#include <register/op_def_registry.h>

namespace ops {
class GammaCorrection : public OpDef {
public:
    explicit GammaCorrection(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_UINT8, ge::DT_INT8, ge::DT_UINT16, ge::DT_INT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_UINT8, ge::DT_INT8, ge::DT_UINT16, ge::DT_INT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("gamma").AttrType(REQUIRED).Float();
        // correction_mode: 0 = pure pow(x, gamma) (backward compatible);
        //                 1 = normalize to [0,1], pow, denormalize, clamp to [0, max_value];
        //                 2 = sRGB encoding curve (linear -> sRGB), gamma ignored.
        this->Attr("correction_mode").AttrType(OPTIONAL).Int(0);
        // max_value: value-range upper bound. When omitted (<=0) it is derived from
        // the input dtype: floats -> 1.0; uint8/int8 -> 255.0; uint16/int16 -> 65535.0.
        this->Attr("max_value").AttrType(OPTIONAL).Float(0.0F);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "gamma_correction");
        this->AICore().AddConfig("ascend910b", config);
    }
};

OP_ADD(GammaCorrection);
} // namespace ops
