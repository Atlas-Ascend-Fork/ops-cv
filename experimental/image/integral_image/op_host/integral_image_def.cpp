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
 * \file integral_image_def.cpp
 * \brief IntegralImage 算子定义
 */

#include "register/op_def_registry.h"

namespace ops {

namespace {
// Four position-matched signatures: u8->i32, fp16->fp32, fp32->fp32, u8->fp32.
// Every tensor uses the same ND format list for these four signatures.
const std::vector<ge::Format> kIntegralImageNdFormats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
} // namespace

class IntegralImage : public OpDef {
public:
    explicit IntegralImage(const char* name) : OpDef(name)
    {
        // 输入 image：ND，(H,W) 或 (H,W,C) HWC 布局，支持 u8 / fp16 / fp32。
        this->Input("image")
            .DataType({ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_UINT8})
            .ParamType(REQUIRED)
            .Format(kIntegralImageNdFormats)
            .UnknownShapeFormat(kIntegralImageNdFormats)
            .AutoContiguous();

        // 必选输出 sat：积分图本身，u8 输入对应 int32（或用 sdepth 指定 fp32），
        // fp16 / fp32 输入对应 fp32。
        this->Output("sat")
            .DataType({ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .ParamType(REQUIRED)
            .Format(kIntegralImageNdFormats)
            .UnknownShapeFormat(kIntegralImageNdFormats)
            .AutoContiguous();

        // sqsum: 平方和积分图，FLOAT32 累加并输出；可选输出缺省时跳过计算。
        this->Output("sqsum")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(kIntegralImageNdFormats)
            .UnknownShapeFormat(kIntegralImageNdFormats)
            .AutoContiguous();

        // tilted: 向上按 45° 扩展的三角区域累加，输出类型与 sat 一致。
        // 通过两组对角前缀计算；末行、末列为零，可选输出缺省时跳过计算。
        this->Output("tilted")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(kIntegralImageNdFormats)
            .UnknownShapeFormat(kIntegralImageNdFormats)
            .AutoContiguous();

        // sdepth: -1 自动（u8->int32，fp16/fp32->float32）；
        // 0=int32（仅 uint8 输入）；1=float32。
        this->Attr("sdepth").AttrType(OPTIONAL).Int(-1);

        // 910B 配置：编译期静态化、放开动态 rank/shape，格式固定为 ND，
        // kernel 入口文件名为 integral_image。各开关相互独立，逐个设置。
        OpAICoreConfig aicoreConfig;
        aicoreConfig.ExtendCfgInfo("opFile.value", "integral_image");
        aicoreConfig.PrecisionReduceFlag(true);
        aicoreConfig.NeedCheckSupportFlag(false);
        aicoreConfig.DynamicShapeSupportFlag(true);
        aicoreConfig.DynamicRankSupportFlag(true);
        aicoreConfig.DynamicFormatFlag(false);
        aicoreConfig.DynamicCompileStaticFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};
OP_ADD(IntegralImage);

} // namespace ops
