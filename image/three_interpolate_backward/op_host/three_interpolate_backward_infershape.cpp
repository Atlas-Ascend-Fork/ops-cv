/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file three_interpolate_backward.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"
#include "log/log.h"

using namespace ge;

namespace {
const uint32_t INDEX_INPUT_GRAD_X = 0u;
const uint32_t INDEX_OUTPUT_GRAD_Y = 0u;
const int32_t UNKNOW_DIM = -1;
const size_t DIM_NUM_ND = 3;  // 950(arch35) ND 路径：(B, C, N)
const size_t DIM_NUM_5HD = 5; // 910b/910_93 5HD 路径：(B, C1, N, 1, C0)
enum class DIM : size_t { DIM_0, DIM_1, DIM_2, DIM_3, DIM_4, DIM_5 };
} // namespace

namespace ops {
static graphStatus InferShape4ThreeInterpolateBackward(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter InferShapeThreeInterpolateBackward");
    OP_LOGI(context, "Enter InferShape4ThreeInterpolateBackward");

    const gert::Shape* grad_x_shape = context->GetInputShape(INDEX_INPUT_GRAD_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_x_shape);

    gert::Shape* grad_y_shape = context->GetOutputShape(INDEX_OUTPUT_GRAD_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_y_shape);

    if (Ops::Base::IsUnknownRank(*grad_x_shape)) {
        OP_LOGI(context, "input is UnknownRank, set output as UnknownRank.");
        Ops::Base::SetUnknownRank(*grad_y_shape);
        return GRAPH_SUCCESS;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // 950(arch35) ND 路径：grad_x (B, C, N) -> grad_y (B, C, M)
    if (grad_x_shape->GetDimNum() == DIM_NUM_ND) {
        // def.cpp 中 Attr("m").Int() 存储为 int64_t，此处按 int64_t 读取
        // （修正参考版 GetAttrPointer<uint32_t> 的类型不匹配问题）
        auto attr_m = attrs->GetAttrPointer<int64_t>(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, attr_m);
        // 校验属性 m > 0（与 tiling 侧双重拦截）
        OP_CHECK_IF(*attr_m <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "m", std::to_string(*attr_m).c_str(),
                                                          "attr m must be > 0"),
                    return GRAPH_FAILED);
        // 校验 idx/weight shape：dimNum==3、idx shape == weight shape、idx == (B, N, 3)
        // （各输入独立判定：unknown rank 输入跳过自身校验；unknown dim 仅校验已知维度）
        const gert::Shape* idx_shape = context->GetInputShape(static_cast<size_t>(DIM::DIM_1));
        OP_CHECK_NULL_WITH_CONTEXT(context, idx_shape);
        const gert::Shape* weight_shape = context->GetInputShape(static_cast<size_t>(DIM::DIM_2));
        OP_CHECK_NULL_WITH_CONTEXT(context, weight_shape);
        const bool idxKnownRank = !Ops::Base::IsUnknownRank(*idx_shape);
        const bool weightKnownRank = !Ops::Base::IsUnknownRank(*weight_shape);
        if (idxKnownRank) {
            OP_CHECK_IF(idx_shape->GetDimNum() != DIM_NUM_ND,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "idx",
                                                     std::to_string(idx_shape->GetDimNum()).c_str(), "3"),
                        return GRAPH_FAILED);
            int64_t dimB = grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_0));
            int64_t dimN = grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_2));
            int64_t idxDim0 = idx_shape->GetDim(static_cast<size_t>(DIM::DIM_0));
            int64_t idxDim1 = idx_shape->GetDim(static_cast<size_t>(DIM::DIM_1));
            int64_t idxDim2 = idx_shape->GetDim(static_cast<size_t>(DIM::DIM_2));
            OP_CHECK_IF(
                (idxDim0 != UNKNOW_DIM && dimB != UNKNOW_DIM && idxDim0 != dimB) ||
                    (idxDim1 != UNKNOW_DIM && dimN != UNKNOW_DIM && idxDim1 != dimN) ||
                    (idxDim2 != UNKNOW_DIM && idxDim2 != 3),
                OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "idx", Ops::Base::ToString(*idx_shape).c_str(),
                                          ("[" + std::to_string(dimB) + "," + std::to_string(dimN) + ",3]").c_str()),
                return GRAPH_FAILED);
        }
        if (weightKnownRank) {
            OP_CHECK_IF(weight_shape->GetDimNum() != DIM_NUM_ND,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "weight",
                                                     std::to_string(weight_shape->GetDimNum()).c_str(), "3"),
                        return GRAPH_FAILED);
        }
        if (idxKnownRank && weightKnownRank) {
            for (size_t i = 0; i < DIM_NUM_ND; i++) {
                int64_t idxDim = idx_shape->GetDim(i);
                int64_t weightDim = weight_shape->GetDim(i);
                OP_CHECK_IF(idxDim != UNKNOW_DIM && weightDim != UNKNOW_DIM && idxDim != weightDim,
                            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                                context->GetNodeName(), "idx, weight",
                                (Ops::Base::ToString(*idx_shape) + " vs " + Ops::Base::ToString(*weight_shape)).c_str(),
                                "idx shape must equal weight shape"),
                            return GRAPH_FAILED);
            }
        }
        grad_y_shape->SetDimNum(DIM_NUM_ND);
        grad_y_shape->SetDim(static_cast<size_t>(DIM::DIM_0), grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_0)));
        grad_y_shape->SetDim(static_cast<size_t>(DIM::DIM_1), grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_1)));
        grad_y_shape->SetDim(static_cast<size_t>(DIM::DIM_2), *attr_m);
        OP_LOGI(context, "Infershape ND B:%ld C:%ld M:%ld.", grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_0)),
                grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_1)),
                grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_2)));
        return GRAPH_SUCCESS;
    }

    // grad_x rank 校验：仅支持 ND(3D) 与 5HD(5D) 两种形状。其余 rank 若继续进入 5HD
    // 分支，会读取不存在的 DIM_4 并构造 5D 输出，故在此显式拒绝（与 ND 分支同等约束）。
    OP_CHECK_IF(grad_x_shape->GetDimNum() != DIM_NUM_5HD,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "grad_x",
                                             std::to_string(grad_x_shape->GetDimNum()).c_str(), "3 or 5"),
                return GRAPH_FAILED);

    // 910b/910_93 5HD 路径：grad_x (B, C1, N, 1, C0) -> grad_y (B, C1, M, 1, C0)
    // 属性 m 在 def.cpp 注册为 Int（int64_t），必须按 int64_t 读取；以 uint32_t 读取会截断
    // 低 32 位（m=-1 变 4294967295），绕过 m>0 校验并污染 grad_y 的 DIM_2。
    auto attr_m = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_m);
    OP_CHECK_IF(*attr_m <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "m", std::to_string(*attr_m).c_str(),
                                                      "attr m must be > 0"),
                return GRAPH_FAILED);
    int64_t ms = *attr_m;
    int64_t bs = grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_0));
    int64_t c1 = grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_1));
    int64_t c0 = grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_4));

    // idx/weight 形状校验（与 ND 分支同等约束）：dimNum==3、idx shape == weight shape、
    // idx == (B, N, 3)，其中 N 取 grad_x 的 DIM_2。
    const gert::Shape* idx_shape_5hd = context->GetInputShape(static_cast<size_t>(DIM::DIM_1));
    OP_CHECK_NULL_WITH_CONTEXT(context, idx_shape_5hd);
    const gert::Shape* weight_shape_5hd = context->GetInputShape(static_cast<size_t>(DIM::DIM_2));
    OP_CHECK_NULL_WITH_CONTEXT(context, weight_shape_5hd);
    const bool idxKnownRank5hd = !Ops::Base::IsUnknownRank(*idx_shape_5hd);
    const bool weightKnownRank5hd = !Ops::Base::IsUnknownRank(*weight_shape_5hd);
    if (idxKnownRank5hd) {
        OP_CHECK_IF(idx_shape_5hd->GetDimNum() != DIM_NUM_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "idx",
                                                 std::to_string(idx_shape_5hd->GetDimNum()).c_str(), "3"),
                    return GRAPH_FAILED);
        int64_t dimB = grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_0));
        int64_t dimN = grad_x_shape->GetDim(static_cast<size_t>(DIM::DIM_2));
        int64_t idxDim0 = idx_shape_5hd->GetDim(static_cast<size_t>(DIM::DIM_0));
        int64_t idxDim1 = idx_shape_5hd->GetDim(static_cast<size_t>(DIM::DIM_1));
        int64_t idxDim2 = idx_shape_5hd->GetDim(static_cast<size_t>(DIM::DIM_2));
        OP_CHECK_IF(
            (idxDim0 != UNKNOW_DIM && dimB != UNKNOW_DIM && idxDim0 != dimB) ||
                (idxDim1 != UNKNOW_DIM && dimN != UNKNOW_DIM && idxDim1 != dimN) ||
                (idxDim2 != UNKNOW_DIM && idxDim2 != 3),
            OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "idx", Ops::Base::ToString(*idx_shape_5hd).c_str(),
                                      ("[" + std::to_string(dimB) + "," + std::to_string(dimN) + ",3]").c_str()),
            return GRAPH_FAILED);
    }
    if (weightKnownRank5hd) {
        OP_CHECK_IF(weight_shape_5hd->GetDimNum() != DIM_NUM_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "weight",
                                                 std::to_string(weight_shape_5hd->GetDimNum()).c_str(), "3"),
                    return GRAPH_FAILED);
    }
    if (idxKnownRank5hd && weightKnownRank5hd) {
        for (size_t i = 0; i < DIM_NUM_ND; i++) {
            int64_t idxDim = idx_shape_5hd->GetDim(i);
            int64_t weightDim = weight_shape_5hd->GetDim(i);
            OP_CHECK_IF(
                idxDim != UNKNOW_DIM && weightDim != UNKNOW_DIM && idxDim != weightDim,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "idx, weight",
                    (Ops::Base::ToString(*idx_shape_5hd) + " vs " + Ops::Base::ToString(*weight_shape_5hd)).c_str(),
                    "idx shape must equal weight shape"),
                return GRAPH_FAILED);
        }
    }

    grad_y_shape->SetDimNum(DIM_NUM_5HD);
    grad_y_shape->SetDim(static_cast<size_t>(DIM::DIM_0), bs);
    grad_y_shape->SetDim(static_cast<size_t>(DIM::DIM_1), c1);
    grad_y_shape->SetDim(static_cast<size_t>(DIM::DIM_2), ms);
    grad_y_shape->SetDim(static_cast<size_t>(DIM::DIM_3), 1);
    grad_y_shape->SetDim(static_cast<size_t>(DIM::DIM_4), c0);

    OP_LOGI(
        context, "Infershape N:%ld C1:%ld H:%ld W:%ld C0:%ld.", grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_0)),
        grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_1)), grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_2)),
        grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_3)), grad_y_shape->GetDim(static_cast<size_t>(DIM::DIM_4)));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ThreeInterpolateBackward(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4ThreeInterpolateBackward");
    const ge::DataType input_grad_x_dtype = context->GetInputDataType(INDEX_INPUT_GRAD_X);
    context->SetOutputDataType(INDEX_OUTPUT_GRAD_Y, input_grad_x_dtype);
    OP_LOGD(context, "End to do InferDataType4ThreeInterpolateBackward");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ThreeInterpolateBackward)
    .InferShape(InferShape4ThreeInterpolateBackward)
    .InferDataType(InferDataType4ThreeInterpolateBackward);
} // namespace ops
