/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <register/op_def_registry.h>
#include <vector>

namespace ops {

class HOGDescriptor : public OpDef {
public:
    explicit HOGDescriptor(const char* name) : OpDef(name)
    {
        RegisterTensorParameters();
        RegisterAttributes();
        RegisterAscend910B();
    }

private:
    template <typename Param>
    static void ConfigureRequiredNd(Param& param, const std::vector<ge::DataType>& types)
    {
        const std::vector<ge::Format> formats(types.size(), ge::FORMAT_ND);
        param.ParamType(REQUIRED);
        param.DataType(types);
        param.Format(formats);
        param.UnknownShapeFormat(formats);
    }

    void RegisterTensorParameters()
    {
        auto& image = this->Input("x");
        ConfigureRequiredNd(image, {ge::DT_FLOAT16, ge::DT_FLOAT});
        image.AutoContiguous();

        auto& descriptor = this->Output("descriptor");
        ConfigureRequiredNd(descriptor, std::vector<ge::DataType>(2, ge::DT_FLOAT));
        descriptor.AutoContiguous();
    }

    void RegisterAttributes()
    {
        this->Attr("pixels_per_cell").AttrType(REQUIRED).ListInt();
        this->Attr("cells_per_block").AttrType(REQUIRED).ListInt();
        this->Attr("block_stride").AttrType(REQUIRED).ListInt();
        this->Attr("orientations").AttrType(REQUIRED).Int();
        this->Attr("block_norm").AttrType(REQUIRED).String();
        this->Attr("l2_hys_threshold").AttrType(REQUIRED).Float();
        this->Attr("gamma_correction").AttrType(REQUIRED).Bool();
        this->Attr("signed_gradient").AttrType(REQUIRED).Bool();
        this->Attr("derivative_aperture").AttrType(REQUIRED).Int();
        this->Attr("eps").AttrType(REQUIRED).Float();
    }

    void RegisterAscend910B()
    {
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true);
        config.DynamicFormatFlag(false);
        config.DynamicRankSupportFlag(false);
        config.DynamicShapeSupportFlag(true);
        config.NeedCheckSupportFlag(false);
        config.PrecisionReduceFlag(true);
        config.ExtendCfgInfo("opFile.value", "hog_descriptor");
        this->AICore().AddConfig("ascend910b", config);
    }
};

OP_ADD(HOGDescriptor);

} // namespace ops
