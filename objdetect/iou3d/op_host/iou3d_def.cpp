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
 * \file iou3d_def.cpp
 * \brief Iou3D 算子定义，声明输入输出和算子配置
 *
 * 3D 旋转框 IoU：BEV 旋转矩形交集面积 × Z 轴重叠高度 / 并集体积。
 *   inputs : bboxes  [B, 7, N]  (float32)   —— 预测框（7-DoF: x,y,z,w,h,d,theta）
 *            gtboxes [B, 7, K]  (float32)   —— 真值框
 *   output : iou     [B, N, K]  (float32)
 * 目标芯片：Ascend950PR / Ascend950DT（arch35 / DAV_3510）。
 */
#include "register/op_def_registry.h"
#include "op_common/op_host/util/op_arch_helper.h"

namespace ops {
class Iou3D : public OpDef {
public:
    explicit Iou3D(const char* name) : OpDef(name)
    {
        this->Input("bboxes") // 预测框 [B, 7, N]
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gtboxes") // 真值框 [B, 7, K]
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("iou") // 输出 [B, N, K]
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // Iou3D 的 [B, 7, N/K] 是固定 ND 逻辑布局，禁止 GE 将 NCHW 等格式
        // 归一化后再进入算子，确保非法 format 在图构建阶段被拒绝。
        this->FormatMatchMode(FormatCheckOption::STRICT);

        // 目标平台：仅 Ascend950（arch35 / DAV_3510）
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "iou3d");
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, aiCoreConfig);
    }
};
OP_ADD(Iou3D); // 添加算子信息库
} // namespace ops
