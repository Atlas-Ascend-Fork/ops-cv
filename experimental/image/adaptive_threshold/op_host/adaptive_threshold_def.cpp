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
 * \file adaptive_threshold_def.cpp
 * \brief AdaptiveThreshold operator registration.
 */
#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class AdaptiveThreshold : public OpDef {
public:
    explicit AdaptiveThreshold(const char* name) : OpDef(name)
    {
        ConfigureImageTensor(this->Input("x")).AutoContiguous();
        ConfigureImageTensor(this->Output("y"));

        this->Attr("max_value").AttrType(REQUIRED).Float();
        this->Attr("adaptive_method").AttrType(REQUIRED).String();
        this->Attr("threshold_type").AttrType(REQUIRED).String();
        this->Attr("block_size").AttrType(REQUIRED).Int();
        this->Attr("C").AttrType(REQUIRED).Float();

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }

private:
    static OpParamDef& ConfigureImageTensor(OpParamDef& tensor)
    {
        // Input and output share the same dtype/format combinations.
        const std::vector<ge::DataType> imageTypes = {ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT};
        const std::vector<ge::Format> imageFormats(imageTypes.size(), ge::FORMAT_ND);
        return tensor.ParamType(REQUIRED).DataType(imageTypes).Format(imageFormats).UnknownShapeFormat(imageFormats);
    }
};

OP_ADD(AdaptiveThreshold);
} // namespace ops
