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
 * \file col2_im_v2_def.cpp
 * \brief Col2ImV2 op def
 */
#include "register/op_def_registry.h"
#include "op_common/op_host/util/op_arch_helper.h"

namespace ops {
class Col2ImV2 : public OpDef {
public:
    explicit Col2ImV2(const char* name) : OpDef(name)
    {
        // dtype 组合（共 2 组，位置一一对应）：
        //   组1: x=fp32, output_size=int32, kernel_size=int32, y=fp32
        //   组2: x=fp16, output_size=int32, kernel_size=int32, y=fp16
        this->Input("x")                                        // 输入x定义
            .ParamType(REQUIRED)                                // 必选输入
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})           // 支持数据类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})             // 支持format格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}) // 未确定大小shape对应format格式
            .AutoContiguous();                                  // 内存自动连续化
        this->Input("output_size")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED) // 值依赖输入
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kernel_size")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED) // 值依赖输入
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y") // 输出y定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dilation").AttrType(REQUIRED).ListInt();
        this->Attr("padding").AttrType(REQUIRED).ListInt();
        this->Attr("stride").AttrType(REQUIRED).ListInt();

        OpAICoreConfig aicoreConfig;
        aicoreConfig
            .DynamicCompileStaticFlag(true) // 激活 DTYPE_X 宏
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "col2_im_v2_apt"); // 这里指定的值会对应到kernel入口文件名.cpp
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, aicoreConfig);
    }
};
OP_ADD(Col2ImV2); // 添加算子信息库
} // namespace ops
