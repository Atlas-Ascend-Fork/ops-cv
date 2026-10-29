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
 * \file good_features_to_track_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class GoodFeaturesToTrack : public OpDef {
public:
    explicit GoodFeaturesToTrack(const char* name) : OpDef(name)
    {
        // 输入：单通道灰度图 [N, H, W]
        this->Input("image")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        // 输入（可选）：ROI 掩码 [N, H, W]，与 image 同尺寸，非 0 有效
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        // 属性：角点检测参数（对齐 OpenCV goodFeaturesToTrack）
        this->Attr("max_corners").AttrType(OPTIONAL).Int(1000);
        this->Attr("quality_level").AttrType(OPTIONAL).Float(0.01);
        this->Attr("min_distance").AttrType(OPTIONAL).Float(1.0);
        this->Attr("block_size").AttrType(OPTIONAL).Int(3);
        this->Attr("use_harris_detector").AttrType(OPTIONAL).Bool(false);
        this->Attr("k").AttrType(OPTIONAL).Float(0.04);

        // 输出：角点坐标 [N, max_corners, 2]，固定上界 + 计数
        this->Output("corners")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        // 输出：每图实际角点数 [N]
        this->Output("num_corners")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicShapeSupportFlag(true)
            .DynamicRankSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "good_features_to_track");

        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};
OP_ADD(GoodFeaturesToTrack); // 添加算子信息库
} // namespace ops
