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
 * \file resize_linear_def.cpp
 * \brief resize_linear_def
 */
#include "op_common/op_host/util/op_arch_helper.h"
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> valueDataTypeX = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};

static const std::vector<ge::DataType> sizeDataType = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};

static const std::vector<ge::Format> resizeLinearFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::Format> sizeFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ResizeLinear : public OpDef {
public:
    explicit ResizeLinear(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(valueDataTypeX)
            .Format(resizeLinearFormat)
            .UnknownShapeFormat(resizeLinearFormat);
        this->Input("size")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(sizeDataType)
            .Format(sizeFormat)
            .UnknownShapeFormat(sizeFormat);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(valueDataTypeX)
            .Format(resizeLinearFormat)
            .UnknownShapeFormat(resizeLinearFormat);

        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);
        this->Attr("scale").AttrType(OPTIONAL).Float(0.0f);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "resize_linear_apt");
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, aicoreConfig);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9201, aicoreConfig);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9202, aicoreConfig);
    }
};

OP_ADD(ResizeLinear);
} // namespace ops
