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
 * \file resize_linear_grad_def.cpp
 * \brief resize_linear_grad_def
 */
#include "op_common/op_host/util/op_arch_helper.h"
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> inputDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};

static const std::vector<ge::Format> inputFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ResizeLinearGrad : public OpDef {
public:
    explicit ResizeLinearGrad(const char* name) : OpDef(name)
    {
        this->Input("grads")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(inputFormat)
            .UnknownShapeFormat(inputFormat);
        this->Input("original_image")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(inputFormat)
            .UnknownShapeFormat(inputFormat);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(inputFormat)
            .UnknownShapeFormat(inputFormat);

        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);
        this->Attr("scale").AttrType(OPTIONAL).Float(0.0f);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "resize_linear_grad_apt");
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, aicoreConfig);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9201, aicoreConfig);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9202, aicoreConfig);
    }
};

OP_ADD(ResizeLinearGrad);
} // namespace ops
