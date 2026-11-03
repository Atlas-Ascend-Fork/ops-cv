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
 * \file gaussian_filter_3_d_def.cpp
 * \brief GaussianFilter3D算子原型定义
 * 对5D体数据 x[N,C,D,H,W] 沿 D/H/W 三个空间轴执行三维高斯滤波（可分离实现），
 * 语义对齐 scipy.ndimage.gaussian_filter（truncate 固定 4.0）。
 */
#include "register/op_def_registry.h"

namespace ops {
namespace {
// 910B 上支持的 dtype 组合，与 kernel 侧编译的算子变体顺序保持一致
const std::vector<ge::DataType> kGaussianDtypes{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
// 算子只处理 ND 布局，且不做格式推导
const std::vector<ge::Format> kGaussianFormats{ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

// 输入与输出均为同 dtype、同布局的连续 ND 张量，用同一套描述
void DescribeNdTensor(OpParamDef& tensor)
{
    tensor.ParamType(REQUIRED);
    tensor.DataType(kGaussianDtypes);
    tensor.Format(kGaussianFormats);
    tensor.UnknownShapeFormat(kGaussianFormats);
    tensor.AutoContiguous();
}
} // namespace

class GaussianFilter3D : public OpDef {
public:
    explicit GaussianFilter3D(const char* name) : OpDef(name)
    {
        DescribeNdTensor(this->Input("x"));
        DescribeNdTensor(this->Output("y"));

        this->Attr("sigma").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("mode").AttrType(OPTIONAL).String("REFLECT");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true);
        aicoreConfig.DynamicRankSupportFlag(true);
        aicoreConfig.DynamicShapeSupportFlag(true);
        aicoreConfig.PrecisionReduceFlag(false);
        // 统一kernel文件/入口命名，规避OpType尾数字在构建与运行时的snake转换不一致
        aicoreConfig.ExtendCfgInfo("opFile.value", "gaussian_filter_3_d");
        aicoreConfig.ExtendCfgInfo("opInterface.value", "gaussian_filter_3_d");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(GaussianFilter3D);
} // namespace ops
