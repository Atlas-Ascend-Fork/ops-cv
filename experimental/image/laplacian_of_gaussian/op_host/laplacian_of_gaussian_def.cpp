/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "register/op_def_registry.h"
namespace ops {
class LaplacianOfGaussian : public OpDef {
public:
    explicit LaplacianOfGaussian(const char* name) : OpDef(name)
    {
        Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous(); // kernel 按 GM 连续布局访问
        Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous(); // 输出由 kernel 按连续布局写入
        Attr("sigma").AttrType(REQUIRED).Float();
        Attr("radius").AttrType(OPTIONAL).Int(-1);
        Attr("truncate").AttrType(OPTIONAL).Float(4.f);
        Attr("border_type").AttrType(OPTIONAL).String("REFLECT");
        AICore().AddConfig("ascend910b");
    }
};
OP_ADD(LaplacianOfGaussian);
} // namespace ops
